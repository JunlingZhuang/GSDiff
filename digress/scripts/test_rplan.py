import argparse
import json
import os
import sys
import time
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.rplan_utils import RPlanVisualization  # noqa: E402
from datasets.rplan_dataset import RPlanDataModule, RPlanDatasetInfos  # noqa: E402
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures  # noqa: E402
from diffusion_model_discrete import DiscreteDenoisingDiffusion  # noqa: E402
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete  # noqa: E402


ROOM_TYPES = ["living", "bedroom", "bathroom", "kitchen", "balcony", "storage"]
EDGE_TYPES = ["none", "wall", "door"]
ROOM_TYPE_COLORS = ["#47b39c", "#8da0cb", "#bdbdbd", "#f1c27d", "#a6d854", "#ffd92f"]
EDGE_TYPE_COLORS = {1: "#64748b", 2: "#c2410c"}


def parse_args():
    parser = argparse.ArgumentParser(description="Sample and evaluate an RPLAN DiGress checkpoint.")
    parser.add_argument(
        "config",
        help="Experiment config name or path, e.g. 'rplan' or 'configs/experiment/rplan.yaml'.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional Hydra overrides, e.g. test.num_samples=128 test.checkpoint=outputs/.../best.ckpt.",
    )
    return parser.parse_args()


def normalize_experiment_config(config_arg):
    config_path = Path(config_arg)
    if config_path.suffix in {".yaml", ".yml"}:
        return config_path.name
    return f"{config_arg}.yaml"


def infer_dataset_name(config_arg):
    stem = Path(config_arg).stem
    if stem.endswith(".yaml") or stem.endswith(".yml"):
        stem = Path(stem).stem
    return stem


def build_cfg(config_arg, overrides):
    experiment_config = normalize_experiment_config(config_arg)
    dataset_name = infer_dataset_name(experiment_config)

    has_dataset_override = any(
        override.startswith("dataset=") or override.startswith("dataset.name=")
        for override in overrides
    )
    base_overrides = []
    if not has_dataset_override:
        base_overrides.append(f"dataset={dataset_name}")
    base_overrides.append(f"+experiment={experiment_config}")

    with hydra.initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base="1.3"):
        return hydra.compose(
            config_name="config",
            overrides=[*base_overrides, *overrides],
        )


def build_model_kwargs(cfg):
    datamodule = RPlanDataModule(cfg)
    dataset_infos = RPlanDatasetInfos(datamodule, cfg.dataset)
    extra_features = (
        ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        if cfg.model.type == "discrete" and cfg.model.extra_features is not None
        else DummyExtraFeatures()
    )
    domain_features = DummyExtraFeatures()
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
    )
    sampling_metrics = None
    visualization_tools = None
    return {
        "dataset_infos": dataset_infos,
        "train_metrics": TrainAbstractMetricsDiscrete(),
        "sampling_metrics": sampling_metrics,
        "visualization_tools": visualization_tools,
        "extra_features": extra_features,
        "domain_features": domain_features,
    }


def sample_to_graph(sample):
    node_types, edge_types = sample
    node_types = node_types.detach().cpu().long().tolist()
    edge_types = edge_types.detach().cpu().long()

    graph = nx.Graph()
    for idx, node_type in enumerate(node_types):
        graph.add_node(idx, attr=int(node_type), room_type=ROOM_TYPES[int(node_type)])

    for i in range(edge_types.shape[0]):
        for j in range(i + 1, edge_types.shape[1]):
            edge_type = int(edge_types[i, j])
            if edge_type > 0:
                graph.add_edge(i, j, edge_type=edge_type, edge_label=EDGE_TYPES[edge_type])
    return graph


def graph_to_record(graph):
    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "nodes": [
            {
                "id": int(node),
                "attr": int(data["attr"]),
                "room_type": data["room_type"],
            }
            for node, data in graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": int(src),
                "target": int(dst),
                "edge_type": int(data["edge_type"]),
                "edge_label": data["edge_label"],
            }
            for src, dst, data in graph.edges(data=True)
        ],
    }


def compute_metrics(graphs):
    valid_graphs = [graph for graph in graphs if graph.number_of_nodes() > 0]
    connected = [nx.is_connected(graph) for graph in valid_graphs]
    living_top_degree = []
    living_top_eigen = []

    for graph in valid_graphs:
        living_nodes = [node for node, data in graph.nodes(data=True) if data.get("attr") == 0]
        if not living_nodes:
            living_top_degree.append(False)
            living_top_eigen.append(False)
            continue

        degrees = dict(graph.degree())
        max_degree = max(degrees.values()) if degrees else 0
        living_top_degree.append(any(degrees[node] == max_degree for node in living_nodes))

        if graph.number_of_edges() == 0:
            living_top_eigen.append(False)
            continue
        try:
            centrality = nx.eigenvector_centrality_numpy(graph)
            max_eigen = max(centrality.values())
            living_top_eigen.append(any(np.isclose(centrality[node], max_eigen) for node in living_nodes))
        except Exception:
            living_top_eigen.append(False)

    return {
        "num_samples": len(graphs),
        "avg_nodes": float(np.mean([graph.number_of_nodes() for graph in graphs])) if graphs else 0.0,
        "avg_edges": float(np.mean([graph.number_of_edges() for graph in graphs])) if graphs else 0.0,
        "connected_frac": float(np.mean(connected)) if connected else 0.0,
        "has_living_frac": float(np.mean([
            any(data.get("attr") == 0 for _, data in graph.nodes(data=True))
            for graph in graphs
        ])) if graphs else 0.0,
        "living_top_degree_frac": float(np.mean(living_top_degree)) if living_top_degree else 0.0,
        "living_top_eigen_frac": float(np.mean(living_top_eigen)) if living_top_eigen else 0.0,
    }


def loader_to_graphs(loader):
    graphs = []
    for batch in loader:
        for data in batch.to_data_list():
            graph = nx.Graph()
            node_types = torch.argmax(data.x, dim=-1).tolist()
            for idx, node_type in enumerate(node_types):
                graph.add_node(idx, attr=int(node_type), room_type=ROOM_TYPES[int(node_type)])

            edge_index = data.edge_index.t().tolist()
            edge_types = torch.argmax(data.edge_attr, dim=-1).tolist()
            for (src, dst), edge_type in zip(edge_index, edge_types):
                if src < dst and edge_type > 0:
                    graph.add_edge(int(src), int(dst), edge_type=int(edge_type), edge_label=EDGE_TYPES[int(edge_type)])
            graphs.append(graph)
    return graphs


def histogram(values):
    counts = {}
    for value in values:
        key = str(int(value))
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    return {key: count / total for key, count in sorted(counts.items(), key=lambda item: int(item[0]))} if total else {}


def distribution_metrics(graphs):
    node_type_counts = {name: 0 for name in ROOM_TYPES}
    edge_type_counts = {name: 0 for name in EDGE_TYPES}
    graph_sizes = []
    edge_counts = []
    degrees = []

    for graph in graphs:
        graph_sizes.append(graph.number_of_nodes())
        edge_counts.append(graph.number_of_edges())
        degrees.extend([degree for _, degree in graph.degree()])
        for _, data in graph.nodes(data=True):
            node_type_counts[ROOM_TYPES[int(data["attr"])]] += 1
        for _, _, data in graph.edges(data=True):
            edge_type_counts[EDGE_TYPES[int(data["edge_type"])]] += 1

        possible_edges = graph.number_of_nodes() * (graph.number_of_nodes() - 1) // 2
        edge_type_counts["none"] += max(0, possible_edges - graph.number_of_edges())

    total_nodes = sum(node_type_counts.values())
    total_edge_slots = sum(edge_type_counts.values())
    return {
        "graph_size_hist": histogram(graph_sizes),
        "edge_count_hist": histogram(edge_counts),
        "degree_hist": histogram(degrees),
        "node_type_dist": {
            key: value / total_nodes for key, value in node_type_counts.items()
        } if total_nodes else node_type_counts,
        "edge_type_dist": {
            key: value / total_edge_slots for key, value in edge_type_counts.items()
        } if total_edge_slots else edge_type_counts,
    }


def flatten_numeric(prefix, value):
    items = {}
    if isinstance(value, dict):
        for key, child in value.items():
            items.update(flatten_numeric(f"{prefix}.{key}" if prefix else str(key), child))
    elif isinstance(value, (int, float)):
        items[prefix] = float(value)
    return items


def numeric_delta(generated, reference):
    gen_flat = flatten_numeric("", generated)
    ref_flat = flatten_numeric("", reference)
    keys = sorted(set(gen_flat) | set(ref_flat))
    return {
        key: gen_flat.get(key, 0.0) - ref_flat.get(key, 0.0)
        for key in keys
    }


def key_dataset_stats(graphs):
    summary = compute_metrics(graphs)
    distribution = distribution_metrics(graphs)
    return {
        "num_samples": summary["num_samples"],
        "avg_nodes": summary["avg_nodes"],
        "avg_edges": summary["avg_edges"],
        "connected_frac": summary["connected_frac"],
        "node_type_dist": distribution["node_type_dist"],
        "edge_type_dist": distribution["edge_type_dist"],
    }


def reference_dataset_metrics(datamodule):
    all_graphs = []
    split_key_stats = {}
    split_loaders = {
        "train": datamodule.train_dataloader,
        "val": datamodule.val_dataloader,
        "test": datamodule.test_dataloader,
    }
    for split, loader_fn in split_loaders.items():
        split_graphs = loader_to_graphs(loader_fn())
        all_graphs.extend(split_graphs)
        split_key_stats[split] = key_dataset_stats(split_graphs)

    return {
        "summary": compute_metrics(all_graphs),
        "distribution": distribution_metrics(all_graphs),
        "split_key_stats": split_key_stats,
    }


def draw_graph_on_axis(ax, graph, title):
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if graph.number_of_nodes() == 0:
        return

    pos = nx.spring_layout(graph, seed=0, k=0.9, iterations=80)
    node_colors = [
        ROOM_TYPE_COLORS[graph.nodes[node].get("attr", 0) % len(ROOM_TYPE_COLORS)]
        for node in graph.nodes()
    ]
    labels = {
        node: ROOM_TYPES[graph.nodes[node].get("attr", 0) % len(ROOM_TYPES)]
        for node in graph.nodes()
    }
    edge_colors = [
        EDGE_TYPE_COLORS.get(graph.edges[edge].get("edge_type", 1), "#64748b")
        for edge in graph.edges()
    ]
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=650,
        edgecolors="black",
        linewidths=0.8,
    )
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=edge_colors, width=1.5)
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=6)


def save_graph_grid(graphs, path, max_graphs, cols):
    graphs = graphs[:max_graphs]
    if not graphs:
        return
    cols = max(1, int(cols))
    rows = int(np.ceil(len(graphs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.6 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for idx, graph in enumerate(graphs):
        draw_graph_on_axis(ax=axes[idx], graph=graph, title=f"Sample {idx}")

    from matplotlib.lines import Line2D

    room_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
               markeredgecolor="black", markersize=8, label=name)
        for name, color in zip(ROOM_TYPES, ROOM_TYPE_COLORS)
    ]
    edge_handles = [
        Line2D([0], [0], color=EDGE_TYPE_COLORS[1], lw=2, label="wall"),
        Line2D([0], [0], color=EDGE_TYPE_COLORS[2], lw=2, label="door"),
    ]
    fig.legend(
        handles=room_handles + edge_handles,
        loc="lower center",
        ncol=8,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = build_cfg(args.config, args.overrides)
    cfg.general.wandb = "disabled"
    test_cfg = cfg.test
    cfg.train.batch_size = int(test_cfg.batch_size)

    ckpt_path = Path(test_cfg.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(test_cfg.out_dir) if test_cfg.out_dir else ckpt_path.parent.parent.parent / "test_samples"
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs = build_model_kwargs(cfg)
    datamodule = model_kwargs["dataset_infos"].datamodule
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(str(ckpt_path), **model_kwargs)
    model.cfg.general.wandb = "disabled"
    model.visualization_tools = None
    model.eval()
    device = test_cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    samples = []
    remaining = int(test_cfg.num_samples)
    batch_id = 0
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    while remaining > 0:
        batch = min(int(test_cfg.batch_size), remaining)
        samples.extend(
            model.sample_batch(
                batch_id=batch_id,
                batch_size=batch,
                keep_chain=0,
                number_chain_steps=model.number_chain_steps,
                save_final=0,
                num_nodes=test_cfg.num_nodes,
            )
        )
        batch_id += batch
        remaining -= batch
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start_time

    graphs = [sample_to_graph(sample) for sample in samples]
    records = [graph_to_record(graph) for graph in graphs]
    generated_summary = compute_metrics(graphs)
    generated_distribution = distribution_metrics(graphs)
    reference_metrics = reference_dataset_metrics(datamodule)
    metrics = {
        "checkpoint": str(ckpt_path),
        "inference": {
            "device": device,
            "num_samples": len(graphs),
            "batch_size": int(test_cfg.batch_size),
            "diffusion_steps": int(cfg.model.diffusion_steps),
            "elapsed_seconds": elapsed_seconds,
            "seconds_per_sample": elapsed_seconds / len(graphs) if graphs else 0.0,
            "samples_per_second": len(graphs) / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        },
        "generated_summary": generated_summary,
        "dataset_summary": reference_metrics["summary"],
        "summary_delta": numeric_delta(generated_summary, reference_metrics["summary"]),
        "generated_distribution": generated_distribution,
        "dataset_distribution": reference_metrics["distribution"],
        "distribution_delta": numeric_delta(generated_distribution, reference_metrics["distribution"]),
        "dataset_split_key_stats": reference_metrics["split_key_stats"],
    }

    with (out_dir / "samples.json").open("w") as f:
        json.dump(records, f, indent=2)
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    save_graph_grid(
        graphs=graphs,
        path=out_dir / "samples_grid.png",
        max_graphs=int(test_cfg.grid_samples),
        cols=int(test_cfg.grid_cols),
    )

    print(f"Saved {len(graphs)} samples to {out_dir}")
    print(
        "Inference: "
        f"{metrics['inference']['elapsed_seconds']:.2f}s total, "
        f"{metrics['inference']['seconds_per_sample']:.2f}s/sample, "
        f"{metrics['inference']['samples_per_second']:.2f} samples/s"
    )
    print(
        "Generated vs dataset: "
        f"avg_nodes {generated_summary['avg_nodes']:.2f} vs {reference_metrics['summary']['avg_nodes']:.2f}, "
        f"avg_edges {generated_summary['avg_edges']:.2f} vs {reference_metrics['summary']['avg_edges']:.2f}, "
        f"connected {generated_summary['connected_frac']:.3f} vs {reference_metrics['summary']['connected_frac']:.3f}"
    )
    print("Dataset split check:")
    for split, stats in reference_metrics["split_key_stats"].items():
        print(
            f"  {split}: "
            f"n={stats['num_samples']}, "
            f"avg_nodes={stats['avg_nodes']:.2f}, "
            f"avg_edges={stats['avg_edges']:.2f}, "
            f"connected={stats['connected_frac']:.3f}"
        )
    print(f"Detailed metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
