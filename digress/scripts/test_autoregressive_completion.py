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
    aggregate_case_metrics,
    cfg_get,
    data_to_dense_indices,
    distribution_similarity,
    draw_graph_with_positions,
    evaluate_completion,
    graph_to_record,
    partial_graph_from_dense,
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
        description="Autoregressive full-completion test using an absorbing next-node checkpoint."
    )
    parser.add_argument(
        "config",
        help="Experiment config name or path, e.g. 'msd_wall_absorbing_next_node'.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Optional Hydra overrides, e.g. "
            "autoregressive.checkpoint=outputs/.../best.ckpt autoregressive.known_ratio=0.5."
        ),
    )
    return parser.parse_args()


def choose_known_nodes(n, known_ratio, rng):
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    k = int(round(float(known_ratio) * n))
    k = max(1, min(n - 1, k))
    return np.sort(rng.choice(n, size=k, replace=False))


def make_step_tensors(x_known, e_known):
    """Add one masked target node after the current known graph."""
    n_known = int(x_known.numel())
    n_total = n_known + 1
    x_idx = torch.zeros((1, n_total), dtype=torch.long)
    x_idx[0, :n_known] = x_known.long()

    e_idx = torch.zeros((1, n_total, n_total), dtype=torch.long)
    e_idx[0, :n_known, :n_known] = e_known.long()

    node_mask = torch.ones((1, n_total), dtype=torch.bool)
    anchor_x = torch.ones((1, n_total), dtype=torch.bool)
    anchor_x[0, n_known] = False

    anchor_e = torch.zeros((1, n_total, n_total), dtype=torch.bool)
    anchor_e[0, :n_known, :n_known] = True
    diagonal = torch.eye(n_total, dtype=torch.bool).unsqueeze(0)
    anchor_e[diagonal] = True

    return {
        "X_idx": x_idx,
        "E_idx": e_idx,
        "node_mask": node_mask,
        "anchor_X": anchor_x,
        "anchor_E": anchor_e,
    }


def build_step_condition(model, x_true, known_reference_nodes):
    if getattr(model.dataset_info, "graph_condition_dim", 0) == 0:
        return None
    n = int(x_true.numel())
    known = torch.zeros((1, n), dtype=torch.bool)
    known[0, torch.as_tensor(known_reference_nodes, dtype=torch.long)] = True
    node_mask = torch.ones((1, n), dtype=torch.bool)
    return model._build_graph_condition(
        node_mask=node_mask.to(model.device),
        known_nodes=known.to(model.device),
        target_nodes=(node_mask & (~known)).to(model.device),
        X_idx=x_true.unsqueeze(0).to(model.device),
    )


def append_next_node(model, x_known, e_known, graph_condition=None):
    sample = model.complete_batch(
        **make_step_tensors(x_known, e_known),
        graph_condition=graph_condition,
    )[0]
    x_completed, e_completed = sample
    return x_completed.long(), e_completed.long()


def autoregressive_complete(model, x_true, e_true, known_nodes, order):
    """Complete to the reference node count by repeatedly adding one node.

    This evaluator uses the reference hidden-node order only to make metrics
    well-defined: the kth generated node is compared with the kth hidden GT
    node. At app time this order would be user-selected or sampled/reranked.
    """
    known_nodes = [int(node) for node in known_nodes]
    order = [int(node) for node in order]

    x_current = x_true[known_nodes].clone().long()
    e_current = e_true[known_nodes][:, known_nodes].clone().long()
    generated_to_reference = list(known_nodes)

    for ref_node in order:
        graph_condition = build_step_condition(model, x_true, generated_to_reference)
        x_next, e_next = append_next_node(model, x_current, e_current, graph_condition=graph_condition)
        x_current = x_next
        e_current = e_next
        generated_to_reference.append(ref_node)

    n = int(x_true.numel())
    x_aligned = torch.zeros(n, dtype=torch.long)
    e_aligned = torch.zeros((n, n), dtype=torch.long)

    for generated_idx, reference_idx in enumerate(generated_to_reference):
        x_aligned[reference_idx] = x_current[generated_idx]
    for i_gen, i_ref in enumerate(generated_to_reference):
        for j_gen, j_ref in enumerate(generated_to_reference):
            e_aligned[i_ref, j_ref] = e_current[i_gen, j_gen]
    return [x_aligned, e_aligned]


def save_grid(cases, path, max_cases, spec):
    cases = cases[:max_cases]
    if not cases:
        return
    fig, axes = plt.subplots(len(cases), 3, figsize=(12.0, 3.6 * len(cases)))
    axes = np.array(axes).reshape(len(cases), 3)
    for row, case in enumerate(cases):
        reference = case["reference_graph"]
        pos = nx.spring_layout(reference, seed=0, k=0.9, iterations=100)
        title = (
            f"case {case['index']} | known {case['num_known_nodes']}/{case['num_nodes']} "
            f"({case['known_ratio']:.2f})"
        )
        draw_graph_with_positions(axes[row, 0], case["partial_graph"], f"{title}\npartial input", spec, pos)
        draw_graph_with_positions(axes[row, 1], case["completed_graph"], "autoregressive completion", spec, pos)
        draw_graph_with_positions(axes[row, 2], reference, "reference graph", spec, pos)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = build_cfg(args.config, args.overrides)
    cfg.general.wandb = "disabled"
    ar_cfg = cfg.get("autoregressive", cfg.get("next_node", cfg.get("completion", {})))
    cfg.train.batch_size = int(cfg_get(ar_cfg, "batch_size", cfg.test.batch_size))

    if not is_absorbing_transition(cfg):
        raise ValueError("Autoregressive completion requires model.transition='absorbing'.")

    spec = spec_from_cfg(cfg)
    ckpt_path = resolve_completion_checkpoint(cfg, ar_cfg)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir_value = cfg_get(ar_cfg, "out_dir", None)
    out_dir = Path(out_dir_value) if out_dir_value else ckpt_path.parent.parent.parent / "autoregressive_completion"
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

    device = cfg_get(ar_cfg, "device", cfg.test.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    num_samples = int(cfg_get(ar_cfg, "num_samples", 32))
    known_ratio = float(cfg_get(ar_cfg, "known_ratio", 0.5))
    seed = int(cfg_get(ar_cfg, "seed", 0))
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
            n = int(x_true.numel())
            known_nodes = choose_known_nodes(n, known_ratio, rng)
            hidden_nodes = [node for node in range(n) if node not in set(known_nodes.tolist())]
            if not hidden_nodes:
                continue
            hidden_order = list(rng.permutation(hidden_nodes))

            completed_sample = autoregressive_complete(model, x_true, e_true, known_nodes, hidden_order)
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
                "num_nodes": n,
                "num_known_nodes": int(len(known_nodes)),
                "known_nodes": [int(node) for node in known_nodes.tolist()],
                "hidden_order": [int(node) for node in hidden_order],
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
        "autoregressive_completion": {
            "num_samples": len(cases),
            "known_ratio": known_ratio,
            "seed": seed,
            "fixed_target_node_count": True,
            "order": "random_hidden_reference_order_for_evaluation",
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
        "distribution_similarity": distribution_similarity(completed_distribution, reference_distribution),
    }

    serializable_cases = []
    for case in cases:
        serializable_cases.append({
            "index": case["index"],
            "known_ratio": case["known_ratio"],
            "num_nodes": case["num_nodes"],
            "num_known_nodes": case["num_known_nodes"],
            "known_nodes": case["known_nodes"],
            "hidden_order": case["hidden_order"],
            "metrics": case["metrics"],
            "partial": graph_to_record(case["partial_graph"]),
            "completed": graph_to_record(case["completed_graph"]),
            "reference": graph_to_record(case["reference_graph"]),
        })

    with (out_dir / "autoregressive_completion_samples.json").open("w") as f:
        json.dump(serializable_cases, f, indent=2)
    with (out_dir / "autoregressive_completion_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    save_grid(
        cases=cases,
        path=out_dir / "autoregressive_completion_grid.png",
        max_cases=int(cfg_get(ar_cfg, "grid_samples", 12)),
        spec=spec,
    )

    print(f"Saved {len(cases)} autoregressive completion cases to {out_dir}")
    print(
        "Completion accuracy: "
        f"node_unknown={aggregate_metrics['node_unknown_accuracy']}, "
        f"edge_all={aggregate_metrics['edge_masked_accuracy_all']}, "
        f"edge_present={aggregate_metrics['edge_masked_accuracy_present']}, "
        f"presence_f1={aggregate_metrics['edge_presence_f1']}, "
        f"typed_f1={aggregate_metrics['typed_edge_f1']}"
    )
    print(
        "Completed vs reference: "
        f"avg_nodes {completed_summary['avg_nodes']:.2f} vs {reference_summary['avg_nodes']:.2f}, "
        f"avg_edges {completed_summary['avg_edges']:.2f} vs {reference_summary['avg_edges']:.2f}, "
        f"connected {completed_summary['connected_frac']:.3f} vs {reference_summary['connected_frac']:.3f}"
    )
    print(f"Detailed metrics: {out_dir / 'autoregressive_completion_metrics.json'}")


if __name__ == "__main__":
    main()
