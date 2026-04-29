"""DiGress graph sampling service.

The API keeps the generated graph format aligned with the topology-conditioned
GSDiff endpoint: `rooms` is the node type list and `adjacency` is a binary room
adjacency matrix.
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

ROOM_TYPES = ["living", "bedroom", "bathroom", "kitchen", "balcony", "storage"]
EDGE_TYPES = ["none", "wall", "door"]

DATASET_CONFIGS = {
    "rplan": {
        "dataset": "rplan",
        "experiment": "rplan.yaml",
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


def _sample_to_record(sample: tuple[torch.Tensor, torch.Tensor]) -> dict[str, Any]:
    node_types, edge_types = sample
    node_type_values = node_types.detach().cpu().long().tolist()
    edge_type_tensor = edge_types.detach().cpu().long()
    n_nodes = len(node_type_values)

    adjacency = [[0 for _ in range(n_nodes)] for _ in range(n_nodes)]
    edge_type_matrix = [[0 for _ in range(n_nodes)] for _ in range(n_nodes)]
    edges = []

    for i in range(edge_type_tensor.shape[0]):
        for j in range(i + 1, edge_type_tensor.shape[1]):
            edge_type = int(edge_type_tensor[i, j])
            if edge_type <= 0:
                continue
            adjacency[i][j] = 1
            adjacency[j][i] = 1
            edge_type_matrix[i][j] = edge_type
            edge_type_matrix[j][i] = edge_type
            edges.append(
                {
                    "source": i,
                    "target": j,
                    "edge_type": edge_type,
                    "edge_label": EDGE_TYPES[edge_type],
                }
            )

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
                "room_type": ROOM_TYPES[int(node_type)],
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

        graphs = [_sample_to_record(sample) for sample in samples]
        return {
            "dataset": dataset,
            "checkpoint": str(loaded.checkpoint),
            "inference_seconds": elapsed_seconds,
            "graphs": graphs,
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
            from datasets.rplan_dataset import RPlanDataModule, RPlanDatasetInfos
            from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
            from diffusion_model_discrete import DiscreteDenoisingDiffusion
            from metrics.abstract_metrics import TrainAbstractMetricsDiscrete

            dataset_cfg = DATASET_CONFIGS[dataset]
            overrides = [
                f"dataset={dataset_cfg['dataset']}",
                f"+experiment={dataset_cfg['experiment']}",
            ]
            with hydra.initialize_config_dir(config_dir=str(DIGRESS_ROOT / "configs"), version_base="1.3"):
                cfg = hydra.compose(config_name="config", overrides=overrides)

            cfg.general.wandb = "disabled"

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

            checkpoint = _resolve_checkpoint(cfg.test.checkpoint)
            model = DiscreteDenoisingDiffusion.load_from_checkpoint(
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

            loaded = LoadedGraphModel(cfg=cfg, model=model, checkpoint=checkpoint)
            self._models[dataset] = loaded
            model_status.loaded(status_key)
            return loaded
        except Exception as exc:
            model_status.error(status_key, exc)
            raise


graph_generator = DigressGraphGenerator()
