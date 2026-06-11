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
from test_graph_completion import (  # noqa: E402
    class_counts,
    cfg_get,
    data_to_dense_indices,
    distribution_similarity,
    draw_graph_with_positions,
    f1_from_counts,
    graph_to_record,
    label_at,
    precision_recall_f1,
    resolve_completion_checkpoint,
)
from test_graph_generation import (  # noqa: E402
    build_cfg,
    build_model_kwargs,
    compute_summary,
    distribution_metrics,
    numeric_delta,
    sample_to_graph,
    spec_from_cfg,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Config-driven next-node completion test for absorbing DiGress checkpoints."
    )
    parser.add_argument(
        "config",
        help="Experiment config name or path, e.g. 'msd_wall_absorbing_completion'.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Optional Hydra overrides, e.g. "
            "next_node.num_samples=64 next_node.checkpoint=outputs/.../best.ckpt."
        ),
    )
    return parser.parse_args()


def choose_target_node(n, rng):
    if n <= 0:
        return None
    return int(rng.integers(0, n))


def make_next_node_tensors(x_idx, e_idx, target_node):
    n = int(x_idx.numel())
    known = torch.ones(n, dtype=torch.bool)
    known[target_node] = False
    anchor_e = known.unsqueeze(0) & known.unsqueeze(1)
    return {
        "X_idx": x_idx.unsqueeze(0),
        "E_idx": e_idx.unsqueeze(0),
        "node_mask": torch.ones((1, n), dtype=torch.bool),
        "anchor_X": known.unsqueeze(0),
        "anchor_E": anchor_e.unsqueeze(0),
    }


def graph_policy_condition(model, x_idx, target_node):
    if getattr(model.dataset_info, "graph_condition_dim", 0) == 0:
        return None
    n = int(x_idx.numel())
    node_mask = torch.ones((1, n), dtype=torch.bool)
    known = torch.ones((1, n), dtype=torch.bool)
    known[0, target_node] = False
    target = torch.zeros((1, n), dtype=torch.bool)
    target[0, target_node] = True
    return model._build_graph_condition(
        node_mask=node_mask.to(model.device),
        known_nodes=known.to(model.device),
        target_nodes=target.to(model.device),
        X_idx=x_idx.unsqueeze(0).to(model.device),
    )


def next_node_input_graph(x_idx, e_idx, target_node, spec):
    graph = nx.Graph()
    n = int(x_idx.numel())
    for node in range(n):
        if node == target_node:
            graph.add_node(node, attr=0, room_type="[MASK]", masked=True)
        else:
            attr = int(x_idx[node])
            graph.add_node(node, attr=attr, room_type=label_at(spec.node_labels, attr), masked=False)

    for i in range(n):
        if i == target_node:
            continue
        for j in range(i + 1, n):
            if j == target_node:
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


def draw_next_node_input(ax, graph, title, spec, pos):
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if graph.number_of_nodes() == 0:
        return

    graph_pos = {node: pos[node] for node in graph.nodes() if node in pos}
    node_colors = []
    labels = {}
    for node in graph.nodes():
        if graph.nodes[node].get("masked", False):
            node_colors.append("#e5e7eb")
            labels[node] = "[MASK]"
        else:
            attr = graph.nodes[node].get("attr", 0)
            node_colors.append(spec.node_colors[attr % len(spec.node_colors)])
            labels[node] = label_at(spec.node_labels, attr)
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
        linewidths=1.2,
    )
    nx.draw_networkx_edges(graph, graph_pos, ax=ax, edge_color=edge_colors, width=1.4)
    nx.draw_networkx_labels(graph, graph_pos, labels=labels, ax=ax, font_size=6)


def evaluate_next_node(completed_sample, x_true, e_true, target_node):
    x_pred, e_pred = completed_sample
    n = int(x_true.numel())
    known = torch.ones(n, dtype=torch.bool)
    known[target_node] = False

    pred_target_type = int(x_pred[target_node])
    true_target_type = int(x_true[target_node])
    target_node_correct = int(pred_target_type == true_target_type)

    pred_edges = e_pred[target_node, known].long()
    true_edges = e_true[target_node, known].long()
    edge_total = int(true_edges.numel())
    edge_correct = int((pred_edges == true_edges).sum().item())

    pred_present = pred_edges > 0
    true_present = true_edges > 0
    present_total = int(true_present.sum().item())
    present_correct = int(((pred_edges == true_edges) & true_present).sum().item())
    tp = int((pred_present & true_present).sum().item())
    fp = int((pred_present & (~true_present)).sum().item())
    fn = int(((~pred_present) & true_present).sum().item())
    predicted_degree = int(pred_present.sum().item())
    reference_degree = int(true_present.sum().item())

    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    typed_tp = int(((pred_edges == true_edges) & true_present).sum().item())
    typed_fp = int((pred_present & (pred_edges != true_edges)).sum().item())
    typed_fn = int((true_present & (pred_edges != true_edges)).sum().item())
    typed_precision, typed_recall, typed_f1 = precision_recall_f1(typed_tp, typed_fp, typed_fn)

    target_mask = torch.zeros(n, dtype=torch.bool)
    target_mask[target_node] = True
    edge_classes = [int(value) for value in torch.unique(true_edges[true_present]).detach().cpu().tolist()]

    return {
        "target_node_correct": target_node_correct,
        "target_node_total": 1,
        "target_edge_accuracy_all": edge_correct / edge_total if edge_total else None,
        "target_edge_correct": edge_correct,
        "target_edge_total": edge_total,
        "target_edge_accuracy_present": present_correct / present_total if present_total else None,
        "target_edge_present_correct": present_correct,
        "target_edge_present_total": present_total,
        "connection_tp": tp,
        "connection_fp": fp,
        "connection_fn": fn,
        "connection_precision": precision,
        "connection_recall": recall,
        "connection_f1": f1,
        "typed_edge_tp": typed_tp,
        "typed_edge_fp": typed_fp,
        "typed_edge_fn": typed_fn,
        "typed_edge_precision": typed_precision,
        "typed_edge_recall": typed_recall,
        "typed_edge_f1": typed_f1,
        "target_degree_error": predicted_degree - reference_degree,
        "target_degree_abs_error": abs(predicted_degree - reference_degree),
        "normalized_edit_proxy": (
            int(pred_target_type != true_target_type) + (edge_total - edge_correct)
        ) / (1 + edge_total) if edge_total else None,
        "target_node_type_counts": class_counts(
            x_pred.long(),
            x_true.long(),
            target_mask,
            [true_target_type],
        ),
        "target_edge_type_present_counts": class_counts(
            pred_edges,
            true_edges,
            true_present,
            edge_classes,
        ),
        "predicted_degree": predicted_degree,
        "reference_degree": reference_degree,
        "predicted_connected_to_known": predicted_degree > 0,
        "reference_connected_to_known": reference_degree > 0,
        "pred_target_type": pred_target_type,
        "true_target_type": true_target_type,
    }


def aggregate_case_metrics(case_metrics):
    totals = {
        "target_node_correct": 0,
        "target_node_total": 0,
        "target_edge_correct": 0,
        "target_edge_total": 0,
        "target_edge_present_correct": 0,
        "target_edge_present_total": 0,
        "connection_tp": 0,
        "connection_fp": 0,
        "connection_fn": 0,
        "typed_edge_tp": 0,
        "typed_edge_fp": 0,
        "typed_edge_fn": 0,
        "predicted_connected_count": 0,
        "reference_connected_count": 0,
    }
    predicted_degrees = []
    reference_degrees = []
    target_degree_errors = []
    target_degree_abs_errors = []
    normalized_edit_proxy = []
    target_node_type_counts = {}
    target_edge_type_present_counts = {}
    for metrics in case_metrics:
        for key in [
            "target_node_correct",
            "target_node_total",
            "target_edge_correct",
            "target_edge_total",
            "target_edge_present_correct",
            "target_edge_present_total",
            "connection_tp",
            "connection_fp",
            "connection_fn",
            "typed_edge_tp",
            "typed_edge_fp",
            "typed_edge_fn",
        ]:
            totals[key] += int(metrics[key])
        totals["predicted_connected_count"] += int(metrics["predicted_connected_to_known"])
        totals["reference_connected_count"] += int(metrics["reference_connected_to_known"])
        predicted_degrees.append(metrics["predicted_degree"])
        reference_degrees.append(metrics["reference_degree"])
        target_degree_errors.append(metrics["target_degree_error"])
        target_degree_abs_errors.append(metrics["target_degree_abs_error"])
        if metrics["normalized_edit_proxy"] is not None:
            normalized_edit_proxy.append(metrics["normalized_edit_proxy"])
        for target, source in [
            (target_node_type_counts, metrics["target_node_type_counts"]),
            (target_edge_type_present_counts, metrics["target_edge_type_present_counts"]),
        ]:
            for cls, values in source.items():
                if cls not in target:
                    target[cls] = {"tp": 0, "fp": 0, "fn": 0, "support": 0}
                for count_key in ["tp", "fp", "fn", "support"]:
                    target[cls][count_key] += int(values[count_key])

    precision, recall, f1 = precision_recall_f1(
        totals["connection_tp"],
        totals["connection_fp"],
        totals["connection_fn"],
    )
    typed_precision, typed_recall, typed_f1 = precision_recall_f1(
        totals["typed_edge_tp"],
        totals["typed_edge_fp"],
        totals["typed_edge_fn"],
    )
    num_cases = max(1, len(case_metrics))
    return {
        "target_node_accuracy": totals["target_node_correct"] / totals["target_node_total"]
        if totals["target_node_total"] else None,
        "target_edge_accuracy_all": totals["target_edge_correct"] / totals["target_edge_total"]
        if totals["target_edge_total"] else None,
        "target_edge_accuracy_present": (
            totals["target_edge_present_correct"] / totals["target_edge_present_total"]
            if totals["target_edge_present_total"] else None
        ),
        "connection_precision": precision,
        "connection_recall": recall,
        "connection_f1": f1,
        "typed_edge_precision": typed_precision,
        "typed_edge_recall": typed_recall,
        "typed_edge_f1": typed_f1,
        "predicted_connected_to_known_frac": totals["predicted_connected_count"] / num_cases,
        "reference_connected_to_known_frac": totals["reference_connected_count"] / num_cases,
        "avg_predicted_target_degree": float(np.mean(predicted_degrees)) if predicted_degrees else 0.0,
        "avg_reference_target_degree": float(np.mean(reference_degrees)) if reference_degrees else 0.0,
        "mean_target_degree_error": float(np.mean(target_degree_errors)) if target_degree_errors else None,
        "mean_target_degree_abs_error": float(np.mean(target_degree_abs_errors))
        if target_degree_abs_errors else None,
        "mean_normalized_edit_proxy": float(np.mean(normalized_edit_proxy))
        if normalized_edit_proxy else None,
        "target_node_type_f1": f1_from_counts(target_node_type_counts),
        "target_edge_type_present_f1": f1_from_counts(target_edge_type_present_counts),
        **totals,
    }


def save_next_node_grid(cases, path, max_cases, spec):
    cases = cases[:max_cases]
    if not cases:
        return
    fig, axes = plt.subplots(len(cases), 3, figsize=(12.0, 3.6 * len(cases)))
    axes = np.array(axes).reshape(len(cases), 3)
    for row, case in enumerate(cases):
        reference = case["reference_graph"]
        completed = case["completed_graph"]
        input_graph = case["input_graph"]
        pos = nx.spring_layout(reference, seed=0, k=0.9, iterations=100)
        prefix = (
            f"case {case['index']} | target {case['target_node']} | "
            f"true={case['metrics']['true_target_type']} pred={case['metrics']['pred_target_type']}"
        )
        draw_next_node_input(axes[row, 0], input_graph, f"{prefix}\ninput graph", spec, pos)
        draw_graph_with_positions(axes[row, 1], completed, "model next-node completion", spec, pos)
        draw_graph_with_positions(axes[row, 2], reference, "reference graph", spec, pos)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = build_cfg(args.config, args.overrides)
    cfg.general.wandb = "disabled"
    next_cfg = cfg.get("next_node", cfg.get("completion", {}))
    cfg.train.batch_size = int(cfg_get(next_cfg, "batch_size", cfg.test.batch_size))

    if not is_absorbing_transition(cfg):
        raise ValueError("Next-node completion currently requires model.transition='absorbing'.")

    spec = spec_from_cfg(cfg)
    ckpt_path = resolve_completion_checkpoint(cfg, next_cfg)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir_value = cfg_get(next_cfg, "out_dir", None)
    out_dir = Path(out_dir_value) if out_dir_value else ckpt_path.parent.parent.parent / "next_node_test"
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs = build_model_kwargs(cfg, spec)
    datamodule = model_kwargs["dataset_infos"].datamodule
    model = AbsorbingDenoisingDiffusion.load_from_checkpoint(str(ckpt_path), **model_kwargs)
    model.cfg.model.edge_none_logit_bias = cfg.model.get("edge_none_logit_bias", 0.0)
    model.cfg.model.maskgit_steps = cfg.model.get("maskgit_steps", model.cfg.model.get("maskgit_steps", 16))
    model.cfg.general.wandb = "disabled"
    model.visualization_tools = None
    model.eval()

    device = cfg_get(next_cfg, "device", cfg.test.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    num_samples = int(cfg_get(next_cfg, "num_samples", 32))
    seed = int(cfg_get(next_cfg, "seed", 0))
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
            if int(x_true.numel()) <= 1:
                continue
            target_node = choose_target_node(int(x_true.numel()), rng)
            tensors = make_next_node_tensors(x_true, e_true, target_node)
            completed_sample = model.complete_batch(
                **tensors,
                graph_condition=graph_policy_condition(model, x_true, target_node),
            )[0]
            completed_graph = sample_to_graph(completed_sample, spec)
            reference_graph = sample_to_graph([x_true, e_true], spec)
            input_graph = next_node_input_graph(x_true, e_true, target_node, spec)
            metrics = evaluate_next_node(completed_sample, x_true, e_true, target_node)

            index = len(cases)
            case_metrics.append(metrics)
            completed_graphs.append(completed_graph)
            reference_graphs.append(reference_graph)
            cases.append({
                "index": index,
                "target_node": int(target_node),
                "num_nodes": int(x_true.numel()),
                "metrics": metrics,
                "input_graph": input_graph,
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
        "next_node": {
            "num_samples": len(cases),
            "seed": seed,
        },
        "inference": {
            "device": device,
            "elapsed_seconds": elapsed_seconds,
            "seconds_per_sample": elapsed_seconds / len(cases) if cases else 0.0,
            "samples_per_second": len(cases) / elapsed_seconds if elapsed_seconds > 0 else 0.0,
            "maskgit_steps": int(model.cfg.model.get("maskgit_steps", 16)),
        },
        "next_node_accuracy": aggregate_metrics,
        "completed_summary": completed_summary,
        "reference_summary": reference_summary,
        "summary_delta": numeric_delta(completed_summary, reference_summary),
        "completed_distribution": completed_distribution,
        "reference_distribution": reference_distribution,
        "distribution_delta": numeric_delta(completed_distribution, reference_distribution),
        "distribution_similarity": distribution_similarity(completed_distribution, reference_distribution),
    }

    serializable_cases = []
    for case in cases:
        serializable_cases.append({
            "index": case["index"],
            "target_node": case["target_node"],
            "num_nodes": case["num_nodes"],
            "metrics": case["metrics"],
            "input": graph_to_record(case["input_graph"]),
            "completed": graph_to_record(case["completed_graph"]),
            "reference": graph_to_record(case["reference_graph"]),
        })

    with (out_dir / "next_node_samples.json").open("w") as f:
        json.dump(serializable_cases, f, indent=2)
    with (out_dir / "next_node_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    save_next_node_grid(
        cases=cases,
        path=out_dir / "next_node_grid.png",
        max_cases=int(cfg_get(next_cfg, "grid_samples", 12)),
        spec=spec,
    )

    print(f"Saved {len(cases)} next-node cases to {out_dir}")
    print(
        "Next-node accuracy: "
        f"node={aggregate_metrics['target_node_accuracy']}, "
        f"edge_all={aggregate_metrics['target_edge_accuracy_all']}, "
        f"edge_present={aggregate_metrics['target_edge_accuracy_present']}, "
        f"conn_f1={aggregate_metrics['connection_f1']}"
    )
    print(
        "Target connectivity: "
        f"pred {aggregate_metrics['predicted_connected_to_known_frac']:.3f} vs "
        f"ref {aggregate_metrics['reference_connected_to_known_frac']:.3f}; "
        f"degree {aggregate_metrics['avg_predicted_target_degree']:.2f} vs "
        f"{aggregate_metrics['avg_reference_target_degree']:.2f}"
    )
    print(f"Detailed metrics: {out_dir / 'next_node_metrics.json'}")


if __name__ == "__main__":
    main()
