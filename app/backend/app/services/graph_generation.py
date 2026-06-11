"""DiGress graph sampling service.

Supports multiple datasets (rplan, msd_wall) via per-dataset spec dispatch.
Each dataset has its own:
  - Hydra config name (dataset + experiment)
  - DiGress datamodule / dataset_infos pair (rplan vs msd shape are different)
  - Room-type and edge-type vocabularies, surfaced to the frontend per-sample

Room types and edge types are read directly from `cfg.dataset.node_decoder` /
`cfg.dataset.edge_decoder` at model load time, so the backend stays in sync
when the digress config files change.
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.config import DEVICE, PROJECT_ROOT
from app.services.model_status import model_status


DIGRESS_ROOT = Path(PROJECT_ROOT) / "digress"
DIGRESS_SRC = DIGRESS_ROOT / "src"


def _module_path(module: Any) -> Path | None:
    paths = getattr(module, "__path__", None)
    if paths:
        try:
            return Path(next(iter(paths))).resolve()
        except StopIteration:
            return None
    file = getattr(module, "__file__", None)
    return Path(file).resolve().parent if file else None


def _clear_conflicting_src_package(desired_src_dir: Path) -> None:
    """Drop another project's top-level `src` package before importing DiGress."""
    existing_src = sys.modules.get("src")
    existing_path = _module_path(existing_src) if existing_src is not None else None
    desired_src_dir = desired_src_dir.resolve()
    if existing_path is None or existing_path == desired_src_dir:
        return
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            sys.modules.pop(name, None)


# Frontend dataset id → Hydra config selectors + which datamodule family to
# import. The "datamodule" key matches the dataset_config["name"] dispatch key
# inside digress/src/main.py — both `msd` and `msd_wall` share the MSD
# datamodule because their dataset config name is "msd".
# Graph model key -> Hydra config selectors + runtime behavior. Dataset ids
# stay stable for the frontend, while model keys select a concrete checkpoint
# and task-specific sampler.
GRAPH_MODEL_CONFIGS = {
    "rplan": {
        "dataset": "rplan",
        "experiment": "rplan.yaml",
        "datamodule": "rplan",
        "task": "sample",
    },
    "msd_wall": {
        "dataset": "msd_wall",
        "experiment": "msd_wall.yaml",
        "datamodule": "msd",
        "task": "sample",
    },
    "msd_wall_next_node": {
        "dataset": "msd_wall",
        "experiment": "msd_wall_absorbing_next_node.yaml",
        "datamodule": "msd",
        "task": "next_node",
        "checkpoint": "outputs/2026-05-18/13-38-27-msd_wall_absorbing_next_node/checkpoints/msd_wall_absorbing_next_node/best.ckpt",
        # Calibrated after sweep: reduces over-connection while preserving
        # the best observed next-node connection F1.
        "edge_none_logit_bias": 0.4,
    },
    "msd_wall_full_completion_v2": {
        "dataset": "msd_wall",
        "experiment": "msd_wall_full_completion_v2.yaml",
        "datamodule": "msd",
        "task": "completion",
        "checkpoint": "outputs/2026-05-19/18-58-03-msd_wall_full_completion_v2/checkpoints/msd_wall_full_completion_v2/best.ckpt",
        # Evaluation sweep showed 0.2 best matches MSD edge density while
        # reranking/repair handles app-facing connectedness.
        "edge_none_logit_bias": 0.2,
        "completion_rerank_candidates": 8,
        "completion_repair_connectivity": True,
        "completion_repair_edge_type": 1,
        "completion_rerank": {
            "expected_degree": 4.2,
            "connected_weight": 4.0,
            "edge_count_weight": 1.0,
            "isolated_weight": 0.25,
        },
        # Used only when the app has no target inventory. The order follows
        # msd_wall.yaml node_decoder and is derived from the MSD preprocessing
        # stats recorded in the experiment notes.
        "room_type_prior": [0.2666, 0.0808, 0.1024, 0.0032, 0.1525, 0.0602, 0.0449, 0.1721, 0.1174],
        "default_target_num_nodes": 30,
    },
}

DATASET_CONFIGS = {
    key: value for key, value in GRAPH_MODEL_CONFIGS.items() if value["task"] == "sample"
}


def _ensure_digress_import_path() -> None:
    _clear_conflicting_src_package(DIGRESS_SRC)
    for path in [DIGRESS_SRC, DIGRESS_ROOT]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_checkpoint(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = DIGRESS_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"DiGress checkpoint not found: {path}")
    return path


def _datamodule_factory(name: str):
    """Defer the heavy DiGress imports until first use of a given dataset."""
    if name == "rplan":
        from datasets.rplan_dataset import RPlanDataModule, RPlanDatasetInfos
        return RPlanDataModule, RPlanDatasetInfos
    if name == "msd":
        from datasets.msd_dataset import MSDDataModule, MSDDatasetInfos
        return MSDDataModule, MSDDatasetInfos
    raise ValueError(f"Unknown DiGress datamodule family: {name!r}")


def _sample_to_record(
    sample: tuple[torch.Tensor, torch.Tensor],
    room_types: list[str],
    edge_types: list[str],
) -> dict[str, Any]:
    node_types, edge_type_tensor = sample
    node_type_values = node_types.detach().cpu().long().tolist()
    edge_type_tensor = edge_type_tensor.detach().cpu().long()
    n_nodes = len(node_type_values)

    adjacency = [[0 for _ in range(n_nodes)] for _ in range(n_nodes)]
    edge_type_matrix = [[0 for _ in range(n_nodes)] for _ in range(n_nodes)]
    edges = []

    def edge_label(edge_type: int) -> str:
        return edge_types[edge_type] if 0 <= edge_type < len(edge_types) else f"class_{edge_type}"

    def room_label(node_type: int) -> str:
        return room_types[node_type] if 0 <= node_type < len(room_types) else f"class_{node_type}"

    for i in range(edge_type_tensor.shape[0]):
        for j in range(i + 1, edge_type_tensor.shape[1]):
            edge_type = int(edge_type_tensor[i, j])
            if edge_type <= 0:
                continue
            adjacency[i][j] = 1
            adjacency[j][i] = 1
            edge_type_matrix[i][j] = edge_type
            edge_type_matrix[j][i] = edge_type
            edges.append({
                "source": i,
                "target": j,
                "edge_type": edge_type,
                "edge_label": edge_label(edge_type),
            })

    return {
        "num_nodes": n_nodes,
        "num_edges": len(edges),
        "rooms": [int(value) for value in node_type_values],
        "adjacency": adjacency,
        "edge_types": edge_type_matrix,
        "nodes": [
            {
                "id": idx,
                "attr": int(node_type),
                "room_type": room_label(int(node_type)),
            }
            for idx, node_type in enumerate(node_type_values)
        ],
        "edges": edges,
    }


@dataclass
class LoadedGraphModel:
    cfg: Any
    model: Any
    checkpoint: Path
    room_types: list[str]
    edge_types: list[str]


class DigressGraphGenerator:
    def __init__(self) -> None:
        self._models: dict[str, LoadedGraphModel] = {}
        for model_key, cfg in GRAPH_MODEL_CONFIGS.items():
            model_status.register(
                f"digress_{model_key}",
                group="DiGress",
                label=f"{cfg['dataset'].upper()} {cfg['task']}",
                metadata={
                    "dataset": cfg["dataset"],
                    "experiment": cfg["experiment"],
                    "task": cfg["task"],
                },
            )

    def supported_datasets(self) -> list[str]:
        return sorted(DATASET_CONFIGS.keys())

    def supported_models(self) -> list[str]:
        return sorted(GRAPH_MODEL_CONFIGS.keys())

    def max_num_nodes(self, dataset: str) -> int:
        """Per-dataset upper bound for the optional num_nodes parameter."""
        if dataset not in DATASET_CONFIGS:
            return 16
        return 64 if dataset == "msd_wall" else 16

    def generate(
        self,
        *,
        dataset: str,
        model: str | None = None,
        num_samples: int,
        num_nodes: int | None,
        seed: int | None,
    ) -> dict[str, Any]:
        model_key = model or dataset
        if model_key not in GRAPH_MODEL_CONFIGS:
            supported = ", ".join(self.supported_models())
            raise ValueError(f"Unsupported graph model '{model_key}'. Supported: {supported}")
        spec = GRAPH_MODEL_CONFIGS[model_key]
        if spec["dataset"] != dataset:
            raise ValueError(f"Graph model '{model_key}' is for dataset '{spec['dataset']}', not '{dataset}'")
        if spec["task"] != "sample":
            raise ValueError(f"Graph model '{model_key}' does not support unconditional sampling")
        if dataset not in DATASET_CONFIGS:
            supported = ", ".join(self.supported_datasets())
            raise ValueError(f"Unsupported graph dataset '{dataset}'. Supported: {supported}")

        loaded = self._load(model_key)
        _set_seed(seed)

        device = DEVICE
        loaded.model.to(device)
        loaded.model.eval()

        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            samples = loaded.model.sample_batch(
                batch_id=0,
                batch_size=num_samples,
                keep_chain=0,
                number_chain_steps=loaded.model.number_chain_steps,
                save_final=0,
                num_nodes=num_nodes,
            )

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - start_time

        graphs = [
            _sample_to_record(sample, loaded.room_types, loaded.edge_types)
            for sample in samples
        ]
        return {
            "dataset": dataset,
            "checkpoint": str(loaded.checkpoint),
            "inference_seconds": elapsed_seconds,
            "graphs": graphs,
            "room_types": loaded.room_types,
            "edge_types": loaded.edge_types,
        }

    def complete_next_node(
        self,
        *,
        dataset: str,
        model: str,
        graph: dict[str, Any],
        num_candidates: int,
        seed: int | None,
    ) -> dict[str, Any]:
        if model not in GRAPH_MODEL_CONFIGS:
            supported = ", ".join(self.supported_models())
            raise ValueError(f"Unsupported graph model '{model}'. Supported: {supported}")
        spec = GRAPH_MODEL_CONFIGS[model]
        if spec["dataset"] != dataset:
            raise ValueError(f"Graph model '{model}' is for dataset '{spec['dataset']}', not '{dataset}'")
        if spec["task"] != "next_node":
            raise ValueError(f"Graph model '{model}' does not support next-node completion")

        loaded = self._load(model)
        if not hasattr(loaded.model, "complete_batch"):
            raise ValueError(f"Graph model '{model}' does not expose complete_batch")

        _set_seed(seed)
        device = DEVICE
        loaded.model.to(device)
        loaded.model.eval()

        tensors = self._build_next_node_tensors(graph, loaded, num_candidates)

        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            samples = loaded.model.complete_batch(**tensors)

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - start_time

        graphs = [
            _sample_to_record(sample, loaded.room_types, loaded.edge_types)
            for sample in samples
        ]
        return {
            "dataset": dataset,
            "checkpoint": str(loaded.checkpoint),
            "inference_seconds": elapsed_seconds,
            "graphs": graphs,
            "room_types": loaded.room_types,
            "edge_types": loaded.edge_types,
        }

    def complete_graph(
        self,
        *,
        dataset: str,
        model: str,
        graph: dict[str, Any],
        target_num_nodes: int | None,
        num_candidates: int | None,
        seed: int | None,
    ) -> dict[str, Any]:
        if model not in GRAPH_MODEL_CONFIGS:
            supported = ", ".join(self.supported_models())
            raise ValueError(f"Unsupported graph model '{model}'. Supported: {supported}")
        spec = GRAPH_MODEL_CONFIGS[model]
        if spec["dataset"] != dataset:
            raise ValueError(f"Graph model '{model}' is for dataset '{spec['dataset']}', not '{dataset}'")
        if spec["task"] != "completion":
            raise ValueError(f"Graph model '{model}' does not support full graph completion")

        loaded = self._load(model)
        if not hasattr(loaded.model, "complete_batch"):
            raise ValueError(f"Graph model '{model}' does not expose complete_batch")

        _set_seed(seed)
        device = DEVICE
        loaded.model.to(device)
        loaded.model.eval()

        candidates = int(num_candidates or spec.get("completion_rerank_candidates", 1))
        candidates = max(1, min(candidates, 8))
        tensors = self._build_full_completion_tensors(
            graph=graph,
            loaded=loaded,
            model_key=model,
            batch_size=candidates,
            target_num_nodes=target_num_nodes,
        )

        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            samples = loaded.model.complete_batch(**tensors)

        selected = self._select_completion_candidate(samples, model)
        if spec.get("completion_repair_connectivity", False):
            selected = self._repair_connectivity(
                selected,
                repair_edge_type=int(spec.get("completion_repair_edge_type", 1)),
            )

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - start_time

        return {
            "dataset": dataset,
            "checkpoint": str(loaded.checkpoint),
            "inference_seconds": elapsed_seconds,
            "graphs": [_sample_to_record(selected, loaded.room_types, loaded.edge_types)],
            "room_types": loaded.room_types,
            "edge_types": loaded.edge_types,
        }

    def _build_next_node_tensors(
        self,
        graph: dict[str, Any],
        loaded: LoadedGraphModel,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        nodes = list(graph.get("nodes") or [])
        if not nodes:
            raise ValueError("next-node completion needs at least one existing node")
        if len(nodes) >= 64:
            raise ValueError("next-node completion supports fewer than 64 existing nodes")

        # Preserve frontend node ids but pack them into dense tensor indices.
        id_to_index = {int(node.get("id", idx)): idx for idx, node in enumerate(nodes)}
        n_known = len(nodes)
        n_total = n_known + 1
        target = n_known

        X_idx = torch.zeros((batch_size, n_total), dtype=torch.long)
        for idx, node in enumerate(nodes):
            attr = int(node.get("attr", 0))
            X_idx[:, idx] = max(0, min(attr, len(loaded.room_types) - 1))

        E_idx = torch.zeros((batch_size, n_total, n_total), dtype=torch.long)
        for edge in graph.get("edges") or []:
            source = id_to_index.get(int(edge.get("source", -1)))
            target_idx = id_to_index.get(int(edge.get("target", -1)))
            if source is None or target_idx is None or source == target_idx:
                continue
            edge_type = int(edge.get("edge_type", 0))
            edge_type = max(0, min(edge_type, len(loaded.edge_types) - 1))
            E_idx[:, source, target_idx] = edge_type
            E_idx[:, target_idx, source] = edge_type

        node_mask = torch.ones((batch_size, n_total), dtype=torch.bool)
        anchor_X = torch.ones((batch_size, n_total), dtype=torch.bool)
        anchor_X[:, target] = False

        # All known-known slots, including known non-edges, are conditioning
        # anchors. Only the new node type and target-to-known edges are sampled.
        anchor_E = torch.zeros((batch_size, n_total, n_total), dtype=torch.bool)
        anchor_E[:, :n_known, :n_known] = True
        diagonal = torch.eye(n_total, dtype=torch.bool).unsqueeze(0)
        anchor_E[diagonal.expand(batch_size, -1, -1)] = True

        return {
            "X_idx": X_idx,
            "E_idx": E_idx,
            "node_mask": node_mask,
            "anchor_X": anchor_X,
            "anchor_E": anchor_E,
        }

    def _build_full_completion_tensors(
        self,
        *,
        graph: dict[str, Any],
        loaded: LoadedGraphModel,
        model_key: str,
        batch_size: int,
        target_num_nodes: int | None,
    ) -> dict[str, torch.Tensor]:
        nodes = list(graph.get("nodes") or [])
        if not nodes:
            raise ValueError("full graph completion needs at least one existing node")

        spec = GRAPH_MODEL_CONFIGS[model_key]
        max_nodes = int(getattr(loaded.model.dataset_info, "max_n_nodes", 64))
        requested_total = int(target_num_nodes or spec.get("default_target_num_nodes", 30))
        n_known = len(nodes)
        n_total = max(n_known + 1, min(requested_total, max_nodes))
        if n_known >= max_nodes:
            raise ValueError(f"full graph completion supports fewer than {max_nodes} existing nodes")

        id_to_index = {int(node.get("id", idx)): idx for idx, node in enumerate(nodes)}

        X_idx = torch.zeros((batch_size, n_total), dtype=torch.long)
        for idx, node in enumerate(nodes):
            attr = int(node.get("attr", 0))
            X_idx[:, idx] = max(0, min(attr, len(loaded.room_types) - 1))

        E_idx = torch.zeros((batch_size, n_total, n_total), dtype=torch.long)
        for edge in graph.get("edges") or []:
            source = id_to_index.get(int(edge.get("source", -1)))
            target = id_to_index.get(int(edge.get("target", -1)))
            if source is None or target is None or source == target:
                continue
            edge_type = int(edge.get("edge_type", 0))
            edge_type = max(0, min(edge_type, len(loaded.edge_types) - 1))
            E_idx[:, source, target] = edge_type
            E_idx[:, target, source] = edge_type

        node_mask = torch.ones((batch_size, n_total), dtype=torch.bool)
        anchor_X = torch.zeros((batch_size, n_total), dtype=torch.bool)
        anchor_X[:, :n_known] = True

        # Known-known edges include explicit non-edges. All slots involving
        # missing nodes are sampled by the full-completion checkpoint.
        anchor_E = torch.zeros((batch_size, n_total, n_total), dtype=torch.bool)
        anchor_E[:, :n_known, :n_known] = True
        diagonal = torch.eye(n_total, dtype=torch.bool).unsqueeze(0)
        anchor_E[diagonal.expand(batch_size, -1, -1)] = True

        graph_condition = self._build_app_completion_condition(
            loaded=loaded,
            model_key=model_key,
            X_idx=X_idx,
            node_mask=node_mask,
            known_count=n_known,
        )

        return {
            "X_idx": X_idx,
            "E_idx": E_idx,
            "node_mask": node_mask,
            "anchor_X": anchor_X,
            "anchor_E": anchor_E,
            "graph_condition": graph_condition,
        }

    def _build_app_completion_condition(
        self,
        *,
        loaded: LoadedGraphModel,
        model_key: str,
        X_idx: torch.Tensor,
        node_mask: torch.Tensor,
        known_count: int,
    ) -> torch.Tensor | None:
        dim = int(getattr(loaded.model.dataset_info, "graph_condition_dim", 0))
        if dim == 0:
            return None

        spec = GRAPH_MODEL_CONFIGS[model_key]
        bs, n_total = X_idx.shape
        max_nodes = max(1.0, float(getattr(loaded.model.dataset_info, "max_n_nodes", n_total)))
        remaining = max(0, n_total - known_count)
        condition = torch.zeros(bs, dim, dtype=torch.float32)
        condition[:, 0] = float(known_count) / max_nodes
        condition[:, 1] = float(n_total) / max_nodes
        condition[:, 2] = float(remaining) / max_nodes

        if dim <= 3:
            return condition

        prior = torch.tensor(spec.get("room_type_prior") or [], dtype=torch.float32)
        num_room_types = min(dim - 3, len(loaded.room_types), int(getattr(loaded.model, "base_Xdim_output", dim - 3)))
        if prior.numel() < num_room_types or float(prior.sum()) <= 0:
            prior = torch.ones(num_room_types, dtype=torch.float32) / max(1, num_room_types)
        else:
            prior = prior[:num_room_types]
            prior = prior / prior.sum().clamp(min=1e-6)

        known_counts = torch.zeros(num_room_types, dtype=torch.float32)
        known_values = X_idx[0, :known_count].long().clamp(min=0, max=num_room_types - 1)
        if known_values.numel() > 0:
            known_counts.scatter_add_(0, known_values, torch.ones_like(known_values, dtype=torch.float32))

        target_counts = prior * float(n_total)
        remaining_counts = torch.clamp(target_counts - known_counts, min=0.0)
        remaining_sum = remaining_counts.sum().clamp(min=1e-6)
        remaining_counts = remaining_counts / remaining_sum * float(remaining)
        condition[:, 3:3 + num_room_types] = remaining_counts.unsqueeze(0) / max(1.0, float(n_total))
        return condition

    def _select_completion_candidate(
        self,
        samples: list[tuple[torch.Tensor, torch.Tensor]],
        model_key: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(samples) <= 1:
            return samples[0]
        rerank_cfg = GRAPH_MODEL_CONFIGS[model_key].get("completion_rerank", {})
        best_score = None
        best_sample = samples[0]
        for sample in samples:
            score = self._completion_score(sample, rerank_cfg)
            if best_score is None or score > best_score:
                best_score = score
                best_sample = sample
        return best_sample

    def _completion_score(self, sample: tuple[torch.Tensor, torch.Tensor], rerank_cfg: dict[str, Any]) -> float:
        _, edge_types = sample
        E = edge_types.detach().cpu().long()
        n = int(E.shape[0])
        if n <= 1:
            return 0.0
        adjacency = E > 0
        adjacency.fill_diagonal_(False)
        degrees = adjacency.sum(dim=1).float()
        edge_count = float(adjacency.triu(1).sum().item())
        expected_degree = float(rerank_cfg.get("expected_degree", 4.2))
        expected_edges = max(1.0, expected_degree * n / 2.0)
        isolated_fraction = float((degrees == 0).float().mean().item())
        connected = 1.0 if self._is_connected(adjacency) else 0.0
        return (
            float(rerank_cfg.get("connected_weight", 4.0)) * connected
            - float(rerank_cfg.get("edge_count_weight", 1.0)) * abs(edge_count - expected_edges) / expected_edges
            - float(rerank_cfg.get("isolated_weight", 0.25)) * isolated_fraction
        )

    def _is_connected(self, adjacency: torch.Tensor) -> bool:
        n = int(adjacency.shape[0])
        if n <= 1:
            return True
        seen = {0}
        stack = [0]
        while stack:
            cur = stack.pop()
            neighbors = torch.nonzero(adjacency[cur], as_tuple=False).flatten().tolist()
            for nxt in neighbors:
                if int(nxt) not in seen:
                    seen.add(int(nxt))
                    stack.append(int(nxt))
        return len(seen) == n

    def _repair_connectivity(
        self,
        sample: tuple[torch.Tensor, torch.Tensor],
        repair_edge_type: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_types, edge_types = sample
        E = edge_types.detach().cpu().long().clone()
        n = int(E.shape[0])
        if n <= 1:
            return node_types, E

        adjacency = E > 0
        adjacency.fill_diagonal_(False)
        components: list[list[int]] = []
        unseen = set(range(n))
        while unseen:
            start = unseen.pop()
            comp = [start]
            stack = [start]
            while stack:
                cur = stack.pop()
                for nxt in torch.nonzero(adjacency[cur], as_tuple=False).flatten().tolist():
                    nxt = int(nxt)
                    if nxt in unseen:
                        unseen.remove(nxt)
                        comp.append(nxt)
                        stack.append(nxt)
            components.append(comp)

        if len(components) <= 1:
            return node_types, E

        edge_type = max(1, repair_edge_type)
        anchor = components[0][0]
        for comp in components[1:]:
            target = comp[0]
            E[anchor, target] = edge_type
            E[target, anchor] = edge_type
        return node_types, E

    def _load(self, model_key: str) -> LoadedGraphModel:
        cached = self._models.get(model_key)
        if cached is not None:
            return cached

        status_key = f"digress_{model_key}"
        model_status.loading(status_key)
        _ensure_digress_import_path()

        try:
            import hydra
            from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
            from diffusion.absorbing_utils import is_absorbing_transition, prepare_absorbing_dataset_infos
            from diffusion_model_absorbing import AbsorbingDenoisingDiffusion
            from diffusion_model_discrete import DiscreteDenoisingDiffusion
            from metrics.abstract_metrics import TrainAbstractMetricsDiscrete

            spec = GRAPH_MODEL_CONFIGS[model_key]
            overrides = [
                f"dataset={spec['dataset']}",
                f"+experiment={spec['experiment']}",
            ]
            with hydra.initialize_config_dir(config_dir=str(DIGRESS_ROOT / "configs"), version_base="1.3"):
                cfg = hydra.compose(config_name="config", overrides=overrides)

            cfg.general.wandb = "disabled"

            datamodule_cls, dataset_infos_cls = _datamodule_factory(spec["datamodule"])
            datamodule = datamodule_cls(cfg)
            dataset_infos = dataset_infos_cls(datamodule, cfg.dataset)

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
                prepare_absorbing_dataset_infos(dataset_infos, cfg)

            checkpoint = _resolve_checkpoint(spec.get("checkpoint") or cfg.test.checkpoint)
            model_cls = AbsorbingDenoisingDiffusion if is_absorbing_transition(cfg) else DiscreteDenoisingDiffusion
            model = model_cls.load_from_checkpoint(
                str(checkpoint),
                dataset_infos=dataset_infos,
                train_metrics=TrainAbstractMetricsDiscrete(),
                sampling_metrics=None,
                visualization_tools=None,
                extra_features=extra_features,
                domain_features=domain_features,
            )
            if is_absorbing_transition(cfg):
                model.cfg.model.edge_none_logit_bias = spec.get(
                    "edge_none_logit_bias",
                    cfg.model.get("edge_none_logit_bias", 0.0),
                )
                model.cfg.model.maskgit_steps = cfg.model.get(
                    "maskgit_steps",
                    model.cfg.model.get("maskgit_steps", 16),
                )
            model.cfg.general.wandb = "disabled"
            model.visualization_tools = None
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad = False

            room_types = list(cfg.dataset.get("node_decoder") or getattr(dataset_infos, "node_decoder", []))
            edge_types = list(cfg.dataset.get("edge_decoder") or getattr(dataset_infos, "edge_decoder", []))

            loaded = LoadedGraphModel(
                cfg=cfg,
                model=model,
                checkpoint=checkpoint,
                room_types=room_types,
                edge_types=edge_types,
            )
            self._models[model_key] = loaded
            model_status.loaded(status_key)
            return loaded
        except Exception as exc:
            model_status.error(status_key, exc)
            raise


graph_generator = DigressGraphGenerator()
