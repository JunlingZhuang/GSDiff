"""Partial-graph -> full-graph retrieval via histogram intersection (mode A).

score(q, g) = sum_i min(q_feat[i], g_feat[i]) / sum_i q_feat[i]

Asymmetric: the numerator caps each WL-label contribution at the query's
count, so the score measures "what fraction of q's labels are already present
in g" rather than overall similarity. Fast (one vectorized pass over the
corpus) and robust to query noise but blind to which query node matches
which candidate node.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np

from ..features.base import FeatureExtractor
from .base import Retriever


class ContainmentRetriever(Retriever):
    def __init__(self, extractor: FeatureExtractor):
        self.extractor = extractor
        self.db: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "ContainmentRetriever":
        self.db = features.astype(np.float32)
        return self

    def query(self, query_graph: nx.Graph, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = self.extractor.transform([query_graph])[0].astype(np.float32)
        q_sum = float(q.sum())
        if q_sum <= 0.0:
            scores = np.zeros(self.db.shape[0], dtype=np.float32)
        else:
            inter = np.minimum(self.db, q[None, :]).sum(axis=1)
            scores = inter / q_sum
        order = np.argsort(-scores)[: min(k, scores.shape[0])]
        return scores[order], order

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "db.npy", self.db)

    def load(self, path: str | Path) -> "ContainmentRetriever":
        self.db = np.load(Path(path) / "db.npy")
        return self
