import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
from graph_completion_eval import (  # noqa: E402
    aggregate_case_metrics,
    cfg_get,
    choose_known_nodes,
    class_counts,
    completion_graph_condition,
    data_to_dense_indices,
    distribution_similarity,
    draw_graph_with_positions,
    evaluate_completion,
    f1_from_counts,
    graph_to_record,
    label_at,
    make_completion_tensors,
    partial_graph_from_dense,
    precision_recall_f1,
    repair_connectivity_sample,
    repeat_completion_tensors,
    repeat_graph_condition,
    resolve_completion_checkpoint,
    save_completion_grid,
    select_reranked_completion,
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
    repair_connectivity = bool(cfg_get(completion_cfg, "repair_connectivity", False))
    repair_edge_type = int(cfg_get(completion_cfg, "repair_edge_type", 1))
    rerank_candidates = int(cfg_get(completion_cfg, "rerank_candidates", 1))
    rerank_cfg = cfg_get(completion_cfg, "rerank", {})
    condition_mode = str(cfg_get(completion_cfg, "condition_mode", "oracle")).lower()
    if condition_mode not in {"oracle", "app"}:
        raise ValueError("completion.condition_mode must be 'oracle' or 'app'")
    rng = np.random.default_rng(seed)

    cases = []
    case_metrics = []
    raw_case_metrics = []
    completed_graphs = []
    raw_graphs = []
    reference_graphs = []
    repaired_case_metrics = []
    repaired_graphs = []

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
            graph_condition = completion_graph_condition(
                model,
                x_true,
                known_nodes,
                e_idx=e_true,
                condition_mode=condition_mode,
            )
            batch_tensors = tensors
            batch_condition = graph_condition
            if rerank_candidates > 1:
                batch_tensors = repeat_completion_tensors(tensors, rerank_candidates)
                batch_condition = repeat_graph_condition(graph_condition, rerank_candidates)
            candidate_samples = model.complete_batch(
                **batch_tensors,
                graph_condition=batch_condition,
            )
            raw_sample = candidate_samples[0]
            rerank_info = None
            if rerank_candidates > 1:
                completed_sample, rerank_info = select_reranked_completion(candidate_samples, rerank_cfg)
            else:
                completed_sample = raw_sample
            raw_graph = sample_to_graph(raw_sample, spec)
            completed_graph = sample_to_graph(completed_sample, spec)
            reference_graph = sample_to_graph([x_true, e_true], spec)
            partial_graph = partial_graph_from_dense(x_true, e_true, known_nodes, spec)
            raw_metrics = evaluate_completion(raw_sample, x_true, e_true, known_nodes)
            metrics = evaluate_completion(completed_sample, x_true, e_true, known_nodes)
            repaired_graph = None
            repaired_metrics = None
            if repair_connectivity:
                repaired_sample = repair_connectivity_sample(
                    completed_sample,
                    known_nodes=known_nodes,
                    edge_type=repair_edge_type,
                )
                repaired_graph = sample_to_graph(repaired_sample, spec)
                repaired_metrics = evaluate_completion(repaired_sample, x_true, e_true, known_nodes)

            index = len(cases)
            case_metrics.append(metrics)
            raw_case_metrics.append(raw_metrics)
            completed_graphs.append(completed_graph)
            raw_graphs.append(raw_graph)
            reference_graphs.append(reference_graph)
            if repair_connectivity:
                repaired_case_metrics.append(repaired_metrics)
                repaired_graphs.append(repaired_graph)
            cases.append({
                "index": index,
                "known_ratio": known_ratio,
                "num_nodes": int(x_true.numel()),
                "num_known_nodes": int(len(known_nodes)),
                "known_nodes": [int(node) for node in known_nodes.tolist()],
                "raw_metrics": raw_metrics,
                "metrics": metrics,
                "repaired_metrics": repaired_metrics,
                "rerank": rerank_info,
                "partial_graph": partial_graph,
                "raw_graph": raw_graph,
                "completed_graph": completed_graph,
                "repaired_graph": repaired_graph,
                "reference_graph": reference_graph,
            })
        if len(cases) >= num_samples:
            break
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start_time

    aggregate_metrics = aggregate_case_metrics(case_metrics)
    raw_aggregate_metrics = aggregate_case_metrics(raw_case_metrics)
    completed_summary = compute_summary(completed_graphs, spec)
    raw_summary = compute_summary(raw_graphs, spec)
    reference_summary = compute_summary(reference_graphs, spec)
    completed_distribution = distribution_metrics(completed_graphs, spec)
    raw_distribution = distribution_metrics(raw_graphs, spec)
    reference_distribution = distribution_metrics(reference_graphs, spec)
    repaired_summary = compute_summary(repaired_graphs, spec) if repair_connectivity else None
    repaired_distribution = distribution_metrics(repaired_graphs, spec) if repair_connectivity else None
    metrics = {
        "dataset": cfg.dataset.name,
        "checkpoint": str(ckpt_path),
        "completion": {
            "num_samples": len(cases),
            "known_ratio": known_ratio,
            "seed": seed,
            "fixed_target_node_count": True,
            "repair_connectivity": repair_connectivity,
            "repair_edge_type": repair_edge_type if repair_connectivity else None,
            "rerank_candidates": rerank_candidates,
            "rerank": dict(rerank_cfg) if rerank_candidates > 1 else None,
            "condition_mode": condition_mode,
        },
        "inference": {
            "device": device,
            "elapsed_seconds": elapsed_seconds,
            "seconds_per_sample": elapsed_seconds / len(cases) if cases else 0.0,
            "samples_per_second": len(cases) / elapsed_seconds if elapsed_seconds > 0 else 0.0,
            "maskgit_steps": int(model.cfg.model.get("maskgit_steps", 16)),
        },
        "raw_completion_accuracy": raw_aggregate_metrics,
        "completion_accuracy": aggregate_metrics,
        "raw_summary": raw_summary,
        "completed_summary": completed_summary,
        "reference_summary": reference_summary,
        "raw_summary_delta": numeric_delta(raw_summary, reference_summary),
        "summary_delta": numeric_delta(completed_summary, reference_summary),
        "raw_distribution": raw_distribution,
        "completed_distribution": completed_distribution,
        "reference_distribution": reference_distribution,
        "raw_distribution_delta": numeric_delta(raw_distribution, reference_distribution),
        "distribution_delta": numeric_delta(completed_distribution, reference_distribution),
        "raw_distribution_similarity": distribution_similarity(raw_distribution, reference_distribution),
        "distribution_similarity": distribution_similarity(completed_distribution, reference_distribution),
    }
    if repair_connectivity:
        metrics.update({
            "repaired_completion_accuracy": aggregate_case_metrics(repaired_case_metrics),
            "repaired_summary": repaired_summary,
            "repaired_summary_delta": numeric_delta(repaired_summary, reference_summary),
            "repaired_distribution": repaired_distribution,
            "repaired_distribution_delta": numeric_delta(repaired_distribution, reference_distribution),
            "repaired_distribution_similarity": distribution_similarity(
                repaired_distribution,
                reference_distribution,
            ),
        })

    serializable_cases = []
    for case in cases:
        serializable_cases.append({
            "index": case["index"],
            "known_ratio": case["known_ratio"],
            "num_nodes": case["num_nodes"],
            "num_known_nodes": case["num_known_nodes"],
            "known_nodes": case["known_nodes"],
            "raw_metrics": case["raw_metrics"],
            "metrics": case["metrics"],
            "repaired_metrics": case["repaired_metrics"],
            "rerank": case["rerank"],
            "partial": graph_to_record(case["partial_graph"]),
            "raw": graph_to_record(case["raw_graph"]),
            "completed": graph_to_record(case["completed_graph"]),
            "repaired": graph_to_record(case["repaired_graph"]) if case["repaired_graph"] is not None else None,
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
    if rerank_candidates > 1:
        print(
            "Raw completion: "
            f"node_unknown={raw_aggregate_metrics['node_unknown_accuracy']}, "
            f"edge_present={raw_aggregate_metrics['edge_masked_accuracy_present']}, "
            f"presence_f1={raw_aggregate_metrics['edge_presence_f1']}, "
            f"avg_edges {raw_summary['avg_edges']:.2f} vs {reference_summary['avg_edges']:.2f}, "
            f"connected {raw_summary['connected_frac']:.3f} vs {reference_summary['connected_frac']:.3f}"
        )
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
    if repair_connectivity:
        repaired_metrics = metrics["repaired_completion_accuracy"]
        print(
            "Repaired completion: "
            f"node_unknown={repaired_metrics['node_unknown_accuracy']}, "
            f"edge_present={repaired_metrics['edge_masked_accuracy_present']}, "
            f"presence_f1={repaired_metrics['edge_presence_f1']}, "
            f"avg_edges {repaired_summary['avg_edges']:.2f} vs {reference_summary['avg_edges']:.2f}, "
            f"connected {repaired_summary['connected_frac']:.3f} vs {reference_summary['connected_frac']:.3f}"
        )
    print(f"Detailed metrics: {out_dir / 'completion_metrics.json'}")


if __name__ == "__main__":
    main()
