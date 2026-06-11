"""Full-graph -> full-graph retrieval via cosine similarity on WL features."""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import numpy as np

from ..features.base import FeatureExtractor
from ..index.faiss_index import FlatIPIndex
from .base import Retriever


def _l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


class CosineRetriever(Retriever):
    """Symmetric retrieval: cosine between L2-normalized feature vectors."""

    def __init__(self, extractor: FeatureExtractor):
        self.extractor = extractor
        self.index: FlatIPIndex | None = None
        self.n_items: int = 0

    def fit(self, features: np.ndarray) -> "CosineRetriever":
        feats = _l2_normalize(features.astype(np.float32))
        self.index = FlatIPIndex(feats.shape[1])
        self.index.add(feats)
        self.n_items = feats.shape[0]
        return self

    def query(self, query_graph: nx.Graph, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = self.extractor.transform([query_graph]).astype(np.float32)
        q = _l2_normalize(q)
        scores, idx = self.index.search(q, k)
        return scores[0], idx[0]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.index.save(path / "index.npy")
        with (path / "meta.pkl").open("wb") as f:
            pickle.dump({"n_items": self.n_items, "dim": self.index.dim}, f)

    def load(self, path: str | Path) -> "CosineRetriever":
        path = Path(path)
        with (path / "meta.pkl").open("rb") as f:
            meta = pickle.load(f)
        self.index = FlatIPIndex.load(path / "index.npy", dim=meta["dim"])
        self.n_items = meta["n_items"]
        return self
