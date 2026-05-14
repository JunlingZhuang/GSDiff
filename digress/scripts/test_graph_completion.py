import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
SRC_DIR = PROJECT_ROOT / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from diffusion.absorbing_utils import is_absorbing_transition  # noqa: E402
from diffusion_model_absorbing import AbsorbingDenoisingDiffusion  # noqa: E402
from test_graph_generation import (  # noqa: E402
    build_cfg,
    build_model_kwargs,
    compute_summary,
    distribution_metrics,
    graph_to_record,
    label_at,
    numeric_delta,
    resolve_checkpoint,
    sample_to_graph,
    spec_from_cfg,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Config-driven partial graph completion test for absorbing DiGress checkpoints."
    )
    parser.add_argument(
        "config",
        help="Experiment config name or path, e.g. 'msd_wall_absorbing_v2'.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Optional Hydra overrides, e.g. "
            "completion.num_samples=64 completion.known_ratio=0.5."
        ),
    )
    return parser.parse_args()


def cfg_get(section, key, default):
    if section is None:
        return default
    return section.get(key, default)


def resolve_completion_checkpoint(cfg, completion_cfg):
    checkpoint = cfg_get(completion_cfg, "checkpoint", None) or cfg.test.checkpoint
    if checkpoint:
        path = Path(checkpoint)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return resolve_checkpoint(cfg)


def data_to_dense_indices(data):
    n = int(data.x.size(0))
    x_idx = torch.argmax(data.x, dim=-1).long()
    e_idx = torch.zeros((n, n), dtype=torch.long)
    if data.edge_index.numel() > 0:
        edge_types = torch.argmax(data.edge_attr, dim=-1).long()
        for edge_id in range(data.edge_index.size(1)):
            src = int(data.edge_index[0, edge_id])
            dst = int(data.edge_index[1, edge_id])
            e_idx[src, dst] = int(edge_types[edge_id])
    return x_idx, e_idx


def choose_known_nodes(n, known_ratio, rng):
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    k = int(round(float(known_ratio) * n))
    k = max(1, min(n - 1, k))
    return np.sort(rng.choice(n, size=k, replace=False))


def make_completion_tensors(x_idx, e_idx, known_nodes):
    n = int(x_idx.numel())
    known = torch.zeros(n, dtype=torch.bool)
    known[torch.as_tensor(known_nodes, dtype=torch.long)] = True

    anchor_e = known.unsqueeze(0) & known.unsqueeze(1)
    return {
        "X_idx": x_idx.unsqueeze(0),
        "E_idx": e_idx.unsqueeze(0),
        "node_mask": torch.ones((1, n), dtype=torch.bool),
        "anchor_X": known.unsqueeze(0),
        "anchor_E": anchor_e.unsqueeze(0),
    }


def partial_graph_from_dense(x_idx, e_idx, known_nodes, spec):
    known_set = {int(node) for node in known_nodes}
    graph = nx.Graph()
    for node in sorted(known_set):
        attr = int(x_idx[node])
        graph.add_node(node, attr=attr, room_type=label_at(spec.node_labels, attr))

    for i in sorted(known_set):
        for j in sorted(known_set):
            if i >= j:
                continue
            edge_type = int(e_idx[i, j])
            if edge_type > 0:
                graph.add_edge(
                    i,
                    j,
                    edge_type=edge_type,
                    edge_label=label_at(spec.edge_labels, edge_type),
                )
    return graph


def tensor_accuracy(pred, target, mask):
    total = int(mask.sum().item())
    if total == 0:
        return None, 0, 0
    correct = int((pred[mask] == target[mask]).sum().item())
    return correct / total, correct, total


def evaluate_completion(completed_sample, x_true, e_true, known_nodes):
    x_pred, e_pred = completed_sample
    n = int(x_true.numel())
    known = torch.zeros(n, dtype=torch.bool)
    known[torch.as_tensor(known_nodes, dtype=torch.long)] = True

    unknown_nodes = ~known
    node_acc, node_correct, node_total = tensor_accuracy(x_pred.long(), x_true.long(), unknown_nodes)

    upper = torch.triu(torch.ones((n, n), dtype=torch.bool), diagonal=1)
    anchor_edges = known.unsqueeze(0) & known.unsqueeze(1)
    masked_edges = upper & (~anchor_edges)
    edge_acc, edge_correct, edge_total = tensor_accuracy(e_pred.long(), e_true.long(), masked_edges)

    present_masked_edges = masked_edges & (e_true.long() > 0)
    edge_present_acc, edge_present_correct, edge_present_total = tensor_accuracy(
        e_pred.long(),
        e_true.long(),
        present_masked_edges,
    )

    return {
        "node_unknown_accuracy": node_acc,
        "node_unknown_correct": node_correct,
        "node_unknown_total": node_total,
        "edge_masked_accuracy_all": edge_acc,
        "edge_masked_correct": edge_correct,
        "edge_masked_total": edge_total,
        "edge_masked_accuracy_present": edge_present_acc,
        "edge_masked_present_correct": edge_present_correct,
        "edge_masked_present_total": edge_present_total,
    }


def draw_graph_with_positions(ax, graph, title, spec, pos):
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if graph.number_of_nodes() == 0:
        return

    graph_pos = {node: pos[node] for node in graph.nodes() if node in pos}
    node_colors = [
        spec.node_colors[graph.nodes[node].get("attr", 0) % len(spec.node_colors)]
        for node in graph.nodes()
    ]
    labels = {
        node: label_at(spec.node_labels, graph.nodes[node].get("attr", 0))
        for node in graph.nodes()
    }
    edge_colors = [
        spec.edge_colors.get(graph.edges[edge].get("edge_type", 1), "#64748b")
        for edge in graph.edges()
    ]
    nx.draw_networkx_nodes(
        graph,
        graph_pos,
        ax=ax,
        node_color=node_colors,
        node_size=520,
        edgecolors="black",
        linewidths=0.8,
    )
    nx.draw_networkx_edges(graph, graph_pos, ax=ax, edge_color=edge_colors, width=1.4)
    nx.draw_networkx_labels(graph, graph_pos, labels=labels, ax=ax, font_size=6)


def save_completion_grid(cases, path, max_cases, spec):
    cases = cases[:max_cases]
    if not cases:
        return
    fig, axes = plt.subplots(len(cases), 3, figsize=(12.0, 3.6 * len(cases)))
    axes = np.array(axes).reshape(len(cases), 3)

    for row, case in enumerate(cases):
        reference = case["reference_graph"]
        completed = case["completed_graph"]
        partial = case["partial_graph"]
        pos = nx.spring_layout(reference, seed=0, k=0.9, iterations=100)
        prefix = (
            f"case {case['index']} | known {case['num_known_nodes']}/{case['num_nodes']} "
            f"({case['known_ratio']:.2f})"
        )
        draw_graph_with_positions(axes[row, 0], partial, f"{prefix}\npartial input", spec, pos)
        draw_graph_with_positions(axes[row, 1], completed, "model completion", spec, pos)
        draw_graph_with_positions(axes[row, 2], reference, "reference graph", spec, pos)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def aggregate_case_metrics(case_metrics):
    totals = {
        "node_unknown_correct": 0,
        "node_unknown_total": 0,
        "edge_masked_correct": 0,
        "edge_masked_total": 0,
        "edge_masked_present_correct": 0,
        "edge_masked_present_total": 0,
    }
    for metrics in case_metrics:
        for key in totals:
            totals[key] += int(metrics[key])

    return {
        "node_unknown_accuracy": (
            totals["node_unknown_correct"] / totals["node_unknown_total"]
            if totals["node_unknown_total"] else None
        ),
        "edge_masked_accuracy_all": (
            totals["edge_masked_correct"] / totals["edge_masked_total"]
            if totals["edge_masked_total"] else None
        ),
        "edge_masked_accuracy_present": (
            totals["edge_masked_present_correct"] / totals["edge_masked_present_total"]
            if totals["edge_masked_present_total"] else None
        ),
        **totals,
    }


def main():
    args = parse_args()
    cfg = build_cfg(args.config, args.overrides)
    cfg.general.wandb = "disabled"
    completion_cfg = cfg.get("completion", {})
    cfg.train.batch_size = int(cfg_get(completion_cfg, "batch_size", cfg.test.batch_size))

    if not is_absorbing_transition(cfg):
        raise ValueError("Graph completion currently requires model.transition='absorbing'.")

    spec = spec_from_cfg(cfg)
    ckpt_path = resolve_completion_checkpoint(cfg, completion_cfg)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir_value = cfg_get(completion_cfg, "out_dir", None)
    out_dir = Path(out_dir_value) if out_dir_value else ckpt_path.parent.parent.parent / "completion_samples"
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs = build_model_kwargs(cfg, spec)
    datamodule = model_kwargs["dataset_infos"].datamodule
    model = AbsorbingDenoisingDiffusion.load_from_checkpoint(str(ckpt_path), **model_kwargs)
    # Lightning restores training-time hparams. Re-apply sampling config so
    # completion tests use the active experiment overrides, not stale ckpt cfg.
    model.cfg.model.edge_none_logit_bias = cfg.model.get("edge_none_logit_bias", 0.0)
    model.cfg.model.maskgit_steps = cfg.model.get("maskgit_steps", model.cfg.model.get("maskgit_steps", 16))
    model.cfg.general.wandb = "disabled"
    model.visualization_tools = None
    model.eval()

    device = cfg_get(completion_cfg, "device", cfg.test.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    num_samples = int(cfg_get(completion_cfg, "num_samples", 32))
    known_ratio = float(cfg_get(completion_cfg, "known_ratio", 0.5))
    seed = int(cfg_get(completion_cfg, "seed", 0))
    rng = np.random.default_rng(seed)

    cases = []
    case_metrics = []
    completed_graphs = []
    reference_graphs = []

    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    for batch in datamodule.test_dataloader():
        for data in batch.to_data_list():
            if len(cases) >= num_samples:
                break
            x_true, e_true = data_to_dense_indices(data)
            known_nodes = choose_known_nodes(int(x_true.numel()), known_ratio, rng)
            tensors = make_completion_tensors(x_true, e_true, known_nodes)
            completed_sample = model.complete_batch(**tensors)[0]
            completed_graph = sample_to_graph(completed_sample, spec)
            reference_graph = sample_to_graph([x_true, e_true], spec)
            partial_graph = partial_graph_from_dense(x_true, e_true, known_nodes, spec)
            metrics = evaluate_completion(completed_sample, x_true, e_true, known_nodes)

            index = len(cases)
            case_metrics.append(metrics)
            completed_graphs.append(completed_graph)
            reference_graphs.append(reference_graph)
            cases.append({
                "index": index,
                "known_ratio": known_ratio,
                "num_nodes": int(x_true.numel()),
                "num_known_nodes": int(len(known_nodes)),
                "known_nodes": [int(node) for node in known_nodes.tolist()],
                "metrics": metrics,
                "partial_graph": partial_graph,
                "completed_graph": completed_graph,
                "reference_graph": reference_graph,
            })
        if len(cases) >= num_samples:
            break
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start_time

    aggregate_metrics = aggregate_case_metrics(case_metrics)
    completed_summary = compute_summary(completed_graphs, spec)
    reference_summary = compute_summary(reference_graphs, spec)
    completed_distribution = distribution_metrics(completed_graphs, spec)
    reference_distribution = distribution_metrics(reference_graphs, spec)
    metrics = {
        "dataset": cfg.dataset.name,
        "checkpoint": str(ckpt_path),
        "completion": {
            "num_samples": len(cases),
            "known_ratio": known_ratio,
            "seed": seed,
            "fixed_target_node_count": True,
        },
        "inference": {
            "device": device,
            "elapsed_seconds": elapsed_seconds,
            "seconds_per_sample": elapsed_seconds / len(cases) if cases else 0.0,
            "samples_per_second": len(cases) / elapsed_seconds if elapsed_seconds > 0 else 0.0,
            "maskgit_steps": int(model.cfg.model.get("maskgit_steps", 16)),
        },
        "completion_accuracy": aggregate_metrics,
        "completed_summary": completed_summary,
        "reference_summary": reference_summary,
        "summary_delta": numeric_delta(completed_summary, reference_summary),
        "completed_distribution": completed_distribution,
        "reference_distribution": reference_distribution,
        "distribution_delta": numeric_delta(completed_distribution, reference_distribution),
    }

    serializable_cases = []
    for case in cases:
        serializable_cases.append({
            "index": case["index"],
            "known_ratio": case["known_ratio"],
            "num_nodes": case["num_nodes"],
            "num_known_nodes": case["num_known_nodes"],
            "known_nodes": case["known_nodes"],
            "metrics": case["metrics"],
            "partial": graph_to_record(case["partial_graph"]),
            "completed": graph_to_record(case["completed_graph"]),
            "reference": graph_to_record(case["reference_graph"]),
        })

    with (out_dir / "completion_samples.json").open("w") as f:
        json.dump(serializable_cases, f, indent=2)
    with (out_dir / "completion_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    save_completion_grid(
        cases=cases,
        path=out_dir / "completion_grid.png",
        max_cases=int(cfg_get(completion_cfg, "grid_samples", 12)),
        spec=spec,
    )

    print(f"Saved {len(cases)} completion cases to {out_dir}")
    print(
        "Completion accuracy: "
        f"node_unknown={aggregate_metrics['node_unknown_accuracy']}, "
        f"edge_all={aggregate_metrics['edge_masked_accuracy_all']}, "
        f"edge_present={aggregate_metrics['edge_masked_accuracy_present']}"
    )
    print(
        "Completed vs reference: "
        f"avg_nodes {completed_summary['avg_nodes']:.2f} vs {reference_summary['avg_nodes']:.2f}, "
        f"avg_edges {completed_summary['avg_edges']:.2f} vs {reference_summary['avg_edges']:.2f}, "
        f"connected {completed_summary['connected_frac']:.3f} vs {reference_summary['connected_frac']:.3f}"
    )
    print(f"Detailed metrics: {out_dir / 'completion_metrics.json'}")


if __name__ == "__main__":
    main()
