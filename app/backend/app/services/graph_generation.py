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


# Frontend dataset id → Hydra config selectors + which datamodule family to
# import. The "datamodule" key matches the dataset_config["name"] dispatch key
# inside digress/src/main.py — both `msd` and `msd_wall` share the MSD
# datamodule because their dataset config name is "msd".
DATASET_CONFIGS = {
    "rplan": {
        "dataset": "rplan",
        "experiment": "rplan.yaml",
        "datamodule": "rplan",
    },
    "msd_wall": {
        "dataset": "msd_wall",
        "experiment": "msd_wall.yaml",
        "datamodule": "msd",
    },
}


def _ensure_digress_import_path() -> None:
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
        for dataset, cfg in DATASET_CONFIGS.items():
            model_status.register(
                f"digress_{dataset}",
                group="DiGress",
                label=f"{dataset.upper()} graph generator",
                metadata={"dataset": dataset, "experiment": cfg["experiment"]},
            )

    def supported_datasets(self) -> list[str]:
        return sorted(DATASET_CONFIGS.keys())

    def max_num_nodes(self, dataset: str) -> int:
        """Per-dataset upper bound for the optional num_nodes parameter."""
        if dataset not in DATASET_CONFIGS:
            return 16
        return 64 if dataset == "msd_wall" else 16

    def generate(
        self,
        *,
        dataset: str,
        num_samples: int,
        num_nodes: int | None,
        seed: int | None,
    ) -> dict[str, Any]:
        if dataset not in DATASET_CONFIGS:
            supported = ", ".join(self.supported_datasets())
            raise ValueError(f"Unsupported graph dataset '{dataset}'. Supported: {supported}")

        loaded = self._load(dataset)
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

    def _load(self, dataset: str) -> LoadedGraphModel:
        cached = self._models.get(dataset)
        if cached is not None:
            return cached

        status_key = f"digress_{dataset}"
        model_status.loading(status_key)
        _ensure_digress_import_path()

        try:
            import hydra
            from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
            from diffusion.absorbing_utils import is_absorbing_transition, prepare_absorbing_dataset_infos
            from diffusion_model_absorbing import AbsorbingDenoisingDiffusion
            from diffusion_model_discrete import DiscreteDenoisingDiffusion
            from metrics.abstract_metrics import TrainAbstractMetricsDiscrete

            spec = DATASET_CONFIGS[dataset]
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
                prepare_absorbing_dataset_infos(dataset_infos)

            checkpoint = _resolve_checkpoint(cfg.test.checkpoint)
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
            self._models[dataset] = loaded
            model_status.loaded(status_key)
            return loaded
        except Exception as exc:
            model_status.error(status_key, exc)
            raise


graph_generator = DigressGraphGenerator()
