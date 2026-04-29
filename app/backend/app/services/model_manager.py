"""Singleton model manager -- loads all GSDiff models at startup."""

import os
import sys

import torch

from app.config import DEVICE, MODEL_PATHS, PROJECT_ROOT
from app.services.model_status import model_status

# Ensure gsdiff package is importable
for _p in [PROJECT_ROOT, os.path.join(PROJECT_ROOT, "gsdiff"), os.path.join(PROJECT_ROOT, "datasets")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


class ModelManager:
    def __init__(self):
        self.device = DEVICE
        self.unconst_node = None
        self.unconst_edge = None
        self.topo_encoder = None
        self.topo_node = None
        self.topo_edge = None
        self.boun_encoder = None
        self.boun_node = None
        self.boun_edge = None
        self._model_specs = {
            "unconst_node": ("GSDiff", "Unconstrained node"),
            "unconst_edge": ("GSDiff", "Unconstrained edge"),
            "topo_encoder": ("GSDiff", "Topology encoder"),
            "topo_node": ("GSDiff", "Topology node"),
            "topo_edge": ("GSDiff", "Topology edge"),
            "boun_encoder": ("GSDiff", "Boundary encoder"),
            "boun_node": ("GSDiff", "Boundary node"),
            "boun_edge": ("GSDiff", "Boundary edge"),
        }
        for key, (group, label) in self._model_specs.items():
            model_status.register(
                key,
                group=group,
                label=label,
                metadata={"path": MODEL_PATHS[key]},
            )

    def load_all(self):
        from gsdiff.house_nn1 import HeterHouseModel
        from gsdiff.house_nn2 import EdgeModel
        from gsdiff.bubble_diagram_57_9 import TopoGraphModel
        from gsdiff.heterhouse_80_106_2 import TopoHeterHouseModel
        from gsdiff.heterhouse_56_31 import TopoEdgeModel
        from gsdiff.boundary_78_10 import BoundaryModel
        from gsdiff.heterhouse_81_106_3 import BoundHeterHouseModel
        from gsdiff.heterhouse_56_32 import BoundEdgeModel

        self.unconst_node = self._load("unconst_node", HeterHouseModel, MODEL_PATHS["unconst_node"])
        self.unconst_edge = self._load("unconst_edge", EdgeModel, MODEL_PATHS["unconst_edge"])
        self.topo_encoder = self._load("topo_encoder", TopoGraphModel, MODEL_PATHS["topo_encoder"])
        self.topo_node = self._load("topo_node", TopoHeterHouseModel, MODEL_PATHS["topo_node"])
        self.topo_edge = self._load("topo_edge", TopoEdgeModel, MODEL_PATHS["topo_edge"])
        self.boun_encoder = self._load("boun_encoder", BoundaryModel, MODEL_PATHS["boun_encoder"])
        self.boun_node = self._load("boun_node", BoundHeterHouseModel, MODEL_PATHS["boun_node"])
        self.boun_edge = self._load("boun_edge", BoundEdgeModel, MODEL_PATHS["boun_edge"])

    def _load(self, key: str, model_cls, weight_path: str):
        print(f"Loading {weight_path}...")
        model_status.loading(key)
        try:
            model = model_cls().to(self.device)
            model.load_state_dict(torch.load(weight_path, map_location=self.device))
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            model_status.loaded(key)
            return model
        except Exception as exc:
            model_status.error(key, exc)
            raise


manager = ModelManager()
