import argparse
import json
import sys
import time
from dataclasses import dataclass
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

from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures  # noqa: E402
from diffusion.absorbing_utils import is_absorbing_transition, prepare_absorbing_dataset_infos  # noqa: E402
from diffusion_model_absorbing import AbsorbingDenoisingDiffusion  # noqa: E402
from diffusion_model_discrete import DiscreteDenoisingDiffusion  # noqa: E402
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    datamodule_cls: object
    dataset_infos_cls: object
    node_labels: list[str]
    edge_labels: list[str]
    node_colors: list[str]
    edge_colors: dict[int, str]
    central_node_attr: int | None = None
    central_node_name: str | None = None


def get_dataset_spec(dataset_name):
    if dataset_name == "rplan":
        from datasets.rplan_dataset import RPlanDataModule, RPlanDatasetInfos

        return DatasetSpec(
            name="rplan",
            datamodule_cls=RPlanDataModule,
            dataset_infos_cls=RPlanDatasetInfos,
            node_labels=["living", "bedroom", "bathroom", "kitchen", "balcony", "storage"],
            edge_labels=["none", "wall", "door"],
            node_colors=["#47b39c", "#8da0cb", "#bdbdbd", "#f1c27d", "#a6d854", "#ffd92f"],
            edge_colors={1: "#64748b", 2: "#c2410c"},
            central_node_attr=0,
            central_node_name="living",
        )
    if dataset_name == "msd":
        from datasets.msd_dataset import MSDDataModule, MSDDatasetInfos, MSD_EDGE_DECODER, MSD_NODE_DECODER

        return DatasetSpec(
            name="msd",
            datamodule_cls=MSDDataModule,
            dataset_infos_cls=MSDDatasetInfos,
            node_labels=list(MSD_NODE_DECODER),
            edge_labels=list(MSD_EDGE_DECODER),
            node_colors=["#8da0cb", "#47b39c", "#f1c27d", "#fdae6b", "#fdd0a2", "#72246c", "#ffd92f", "#bdbdbd", "#a6d854"],
            edge_colors={1: "#64748b", 2: "#c2410c", 3: "#d97706"},
        )
    raise NotImplementedError(
        f"Unsupported graph test dataset '{dataset_name}'. Add it to get_dataset_spec()."
    )


def spec_from_cfg(cfg):
    spec = get_dataset_spec(cfg.dataset.name)
    node_labels = list(cfg.dataset.get("node_decoder", spec.node_labels))
    edge_labels = list(cfg.dataset.get("edge_decoder", spec.edge_labels))

    edge_color_by_name = {
        "wall": "#334155",
        "passage": "#64748b",
        "door": "#c2410c",
        "entrance": "#d97706",
        "none": "#cbd5e1",
    }
    edge_colors = {
        idx: edge_color_by_name.get(label, "#64748b")
        for idx, label in enumerate(edge_labels)
        if idx > 0
    }
    return DatasetSpec(
        name=spec.name,
        datamodule_cls=spec.datamodule_cls,
        dataset_infos_cls=spec.dataset_infos_cls,
        node_labels=node_labels,
        edge_labels=edge_labels,
        node_colors=spec.node_colors,
        edge_colors=edge_colors,
        central_node_attr=spec.central_node_attr,
        central_node_name=spec.central_node_name,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Config-driven DiGress graph generation test for supported graph datasets."
    )
    parser.add_argument(
        "config",
        help="Experiment config name or path, e.g. 'rplan', 'msd', or 'configs/experiment/msd.yaml'.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional Hydra overrides, e.g. test.num_samples=64 test.checkpoint=outputs/.../best.ckpt.",
    )
    return parser.parse_args()


def normalize_experiment_config(config_arg):
    config_path = Path(config_arg)
    if config_path.suffix in {".yaml", ".yml"}:
        return config_path.name
    return f"{config_arg}.yaml"


def infer_dataset_name(config_arg):
    stem = Path(normalize_experiment_config(config_arg)).stem
    if stem.startswith("msd_wall"):
        return "msd_wall"
    return stem


def build_cfg(config_arg, overrides):
    experiment_config = normalize_experiment_config(config_arg)
    dataset_name = infer_dataset_name(config_arg)
    has_dataset_override = any(
        override.startswith("dataset=") or override.startswith("dataset.name=")
        for override in overrides
    )

    base_overrides = []
    if not has_dataset_override:
        base_overrides.append(f"dataset={dataset_name}")
    base_overrides.append(f"+experiment={experiment_config}")

    with hydra.initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base="1.3"):
        return hydra.compose(config_name="config", overrides=[*base_overrides, *overrides])


def build_model_kwargs(cfg, spec):
    datamodule = spec.datamodule_cls(cfg)
    dataset_infos = spec.dataset_infos_cls(datamodule, cfg.dataset)
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
    if is_absorbing_transition(cfg):
        prepare_absorbing_dataset_infos(dataset_infos)
    return {
        "dataset_infos": dataset_infos,
        "train_metrics": TrainAbstractMetricsDiscrete(),
        "sampling_metrics": None,
        "visualization_tools": None,
        "extra_features": extra_features,
        "domain_features": domain_features,
    }


def resolve_checkpoint(cfg):
    checkpoint = cfg.test.checkpoint
    if checkpoint:
        path = Path(checkpoint)
        return path if path.is_absolute() else PROJECT_ROOT / path

    names = []
    for value in [cfg.general.name, cfg.dataset.name]:
        if value not in names:
            names.append(value)

    for name in names:
        candidates = sorted(
            (PROJECT_ROOT / "outputs").glob(f"**/checkpoints/{name}/best*.ckpt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
        candidates = sorted(
            (PROJECT_ROOT / "outputs").glob(f"**/checkpoints/{name}/last.ckpt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    raise FileNotFoundError("No checkpoint found. Set test.checkpoint=... explicitly.")


def label_at(labels, index):
    return labels[index] if 0 <= index < len(labels) else f"class_{index}"


def sample_to_graph(sample, spec):
    node_types, edge_types = sample
    node_types = node_types.detach().cpu().long().tolist()
    edge_types = edge_types.detach().cpu().long()

    graph = nx.Graph()
    for idx, node_type in enumerate(node_types):
        node_type = int(node_type)
        graph.add_node(idx, attr=node_type, room_type=label_at(spec.node_labels, node_type))

    for i in range(edge_types.shape[0]):
        for j in range(i + 1, edge_types.shape[1]):
            edge_type = int(edge_types[i, j])
            if edge_type > 0:
                graph.add_edge(
                    i,
                    j,
                    edge_type=edge_type,
                    edge_label=label_at(spec.edge_labels, edge_type),
                )
    return graph


def loader_to_graphs(loader, spec):
    graphs = []
    for batch in loader:
        for data in batch.to_data_list():
            graph = nx.Graph()
            node_types = torch.argmax(data.x, dim=-1).tolist()
            for idx, node_type in enumerate(node_types):
                node_type = int(node_type)
                graph.add_node(idx, attr=node_type, room_type=label_at(spec.node_labels, node_type))

            edge_index = data.edge_index.t().tolist()
            edge_types = torch.argmax(data.edge_attr, dim=-1).tolist()
            for (src, dst), edge_type in zip(edge_index, edge_types):
                edge_type = int(edge_type)
                if src < dst and edge_type > 0:
                    graph.add_edge(
                        int(src),
                        int(dst),
                        edge_type=edge_type,
                        edge_label=label_at(spec.edge_labels, edge_type),
                    )
            graphs.append(graph)
    return graphs


def graph_to_record(graph):
    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "nodes": [
            {"id": int(node), "attr": int(data["attr"]), "room_type": data["room_type"]}
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


def compute_summary(graphs, spec):
    valid_graphs = [graph for graph in graphs if graph.number_of_nodes() > 0]
    connected = [nx.is_connected(graph) for graph in valid_graphs]
    summary = {
        "num_samples": len(graphs),
        "avg_nodes": float(np.mean([graph.number_of_nodes() for graph in graphs])) if graphs else 0.0,
        "avg_edges": float(np.mean([graph.number_of_edges() for graph in graphs])) if graphs else 0.0,
        "connected_frac": float(np.mean(connected)) if connected else 0.0,
    }

    if spec.central_node_attr is not None:
        central_attr = spec.central_node_attr
        central_name = spec.central_node_name or f"node_{central_attr}"
        top_degree = []
        top_eigen = []
        for graph in valid_graphs:
            central_nodes = [
                node for node, data in graph.nodes(data=True)
                if data.get("attr") == central_attr
            ]
            if not central_nodes:
                top_degree.append(False)
                top_eigen.append(False)
                continue

            degrees = dict(graph.degree())
            max_degree = max(degrees.values()) if degrees else 0
            top_degree.append(any(degrees[node] == max_degree for node in central_nodes))

            if graph.number_of_edges() == 0:
                top_eigen.append(False)
                continue
            try:
                centrality = nx.eigenvector_centrality_numpy(graph)
                max_eigen = max(centrality.values())
                top_eigen.append(any(np.isclose(centrality[node], max_eigen) for node in central_nodes))
            except Exception:
                top_eigen.append(False)

        summary[f"has_{central_name}_frac"] = float(np.mean([
            any(data.get("attr") == central_attr for _, data in graph.nodes(data=True))
            for graph in graphs
        ])) if graphs else 0.0
        summary[f"{central_name}_top_degree_frac"] = float(np.mean(top_degree)) if top_degree else 0.0
        summary[f"{central_name}_top_eigen_frac"] = float(np.mean(top_eigen)) if top_eigen else 0.0

    return summary


def histogram(values):
    counts = {}
    for value in values:
        key = str(int(value))
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    if not total:
        return {}
    return {
        key: count / total
        for key, count in sorted(counts.items(), key=lambda item: int(item[0]))
    }


def distribution_metrics(graphs, spec):
    node_type_counts = {name: 0 for name in spec.node_labels}
    edge_type_counts = {name: 0 for name in spec.edge_labels}
    graph_sizes = []
    edge_counts = []
    degrees = []

    for graph in graphs:
        graph_sizes.append(graph.number_of_nodes())
        edge_counts.append(graph.number_of_edges())
        degrees.extend([degree for _, degree in graph.degree()])
        for _, data in graph.nodes(data=True):
            node_type_counts[label_at(spec.node_labels, int(data["attr"]))] += 1
        for _, _, data in graph.edges(data=True):
            edge_type_counts[label_at(spec.edge_labels, int(data["edge_type"]))] += 1

        possible_edges = graph.number_of_nodes() * (graph.number_of_nodes() - 1) // 2
        if "none" in edge_type_counts:
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
    return {
        key: gen_flat.get(key, 0.0) - ref_flat.get(key, 0.0)
        for key in sorted(set(gen_flat) | set(ref_flat))
    }


def key_dataset_stats(graphs, spec):
    summary = compute_summary(graphs, spec)
    distribution = distribution_metrics(graphs, spec)
    return {
        "num_samples": summary["num_samples"],
        "avg_nodes": summary["avg_nodes"],
        "avg_edges": summary["avg_edges"],
        "connected_frac": summary["connected_frac"],
        "node_type_dist": distribution["node_type_dist"],
        "edge_type_dist": distribution["edge_type_dist"],
    }


def reference_dataset_metrics(datamodule, spec):
    all_graphs = []
    split_key_stats = {}
    for split, loader_fn in {
        "train": datamodule.train_dataloader,
        "val": datamodule.val_dataloader,
        "test": datamodule.test_dataloader,
    }.items():
        split_graphs = loader_to_graphs(loader_fn(), spec)
        all_graphs.extend(split_graphs)
        split_key_stats[split] = key_dataset_stats(split_graphs, spec)

    return {
        "summary": compute_summary(all_graphs, spec),
        "distribution": distribution_metrics(all_graphs, spec),
        "split_key_stats": split_key_stats,
    }


def draw_graph_on_axis(ax, graph, title, spec):
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if graph.number_of_nodes() == 0:
        return

    pos = nx.spring_layout(graph, seed=0, k=0.9, iterations=100)
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
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=600,
        edgecolors="black",
        linewidths=0.8,
    )
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=edge_colors, width=1.4)
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=6)


def save_graph_grid(graphs, path, max_graphs, cols, spec):
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
        draw_graph_on_axis(axes[idx], graph, f"Sample {idx}", spec)

    from matplotlib.lines import Line2D

    node_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
               markeredgecolor="black", markersize=8, label=name)
        for name, color in zip(spec.node_labels, spec.node_colors)
    ]
    edge_handles = [
        Line2D([0], [0], color=color, lw=2, label=label_at(spec.edge_labels, edge_type))
        for edge_type, color in sorted(spec.edge_colors.items())
    ]
    fig.legend(
        handles=node_handles + edge_handles,
        loc="lower center",
        ncol=min(8, len(node_handles) + len(edge_handles)),
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

    spec = spec_from_cfg(cfg)
    ckpt_path = resolve_checkpoint(cfg)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(test_cfg.out_dir) if test_cfg.out_dir else ckpt_path.parent.parent.parent / "test_samples"
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs = build_model_kwargs(cfg, spec)
    datamodule = model_kwargs["dataset_infos"].datamodule
    model_cls = AbsorbingDenoisingDiffusion if is_absorbing_transition(cfg) else DiscreteDenoisingDiffusion
    model = model_cls.load_from_checkpoint(str(ckpt_path), **model_kwargs)
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

    graphs = [sample_to_graph(sample, spec) for sample in samples]
    generated_summary = compute_summary(graphs, spec)
    generated_distribution = distribution_metrics(graphs, spec)
    reference_metrics = reference_dataset_metrics(datamodule, spec)
    metrics = {
        "dataset": cfg.dataset.name,
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
        json.dump([graph_to_record(graph) for graph in graphs], f, indent=2)
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    save_graph_grid(
        graphs=graphs,
        path=out_dir / "samples_grid.png",
        max_graphs=int(test_cfg.grid_samples),
        cols=int(test_cfg.grid_cols),
        spec=spec,
    )

    print(f"Saved {len(graphs)} {cfg.dataset.name} samples to {out_dir}")
    print(
        "Inference: "
        f"{elapsed_seconds:.2f}s total, "
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
