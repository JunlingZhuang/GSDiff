"""Partial-graph -> full-graph retrieval via greedy node-level alignment (mode B).

For each query node v_q with WL signature S(v_q), find the candidate-graph
node v_c maximizing Jaccard(S(v_q), S(v_c)). The candidate-graph score is the
mean of those best matches across query nodes.

Stronger constraint than histogram intersection because every query node must
locally resemble some node in the candidate, but ~10x slower and sensitive to
noisy query nodes (one bad node drags the average down).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import numpy as np

from ..features.base import FeatureExtractor
from .base import Retriever


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


class NodeMatchingRetriever(Retriever):
    def __init__(self, extractor: FeatureExtractor):
        self.extractor = extractor
        self.db_signatures: list[list[set]] = []

    def fit(self, graphs: list[nx.Graph]) -> "NodeMatchingRetriever":
        self.db_signatures = [self.extractor.node_signatures(g) for g in graphs]
        return self

    def query(self, query_graph: nx.Graph, k: int) -> tuple[np.ndarray, np.ndarray]:
        q_sigs = self.extractor.node_signatures(query_graph)
        n = len(self.db_signatures)
        if not q_sigs or n == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)
        scores = np.zeros(n, dtype=np.float32)
        for i, db_sigs in enumerate(self.db_signatures):
            if not db_sigs:
                continue
            scores[i] = float(
                np.mean([max(jaccard(qs, ds) for ds in db_sigs) for qs in q_sigs])
            )
        order = np.argsort(-scores)[: min(k, n)]
        return scores[order], order

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "signatures.pkl").open("wb") as f:
            pickle.dump(self.db_signatures, f)

    def load(self, path: str | Path) -> "NodeMatchingRetriever":
        with (Path(path) / "signatures.pkl").open("rb") as f:
            self.db_signatures = pickle.load(f)
        return self
