"""Weisfeiler-Lehman subtree kernel for attributed graphs.

Each graph is encoded as a histogram over WL refinement labels collected at
iterations 0..n_iter. Edge labels participate in the refinement, so the
encoding is sensitive to typed connectivity (MSD's passage / door / entrance /
wall).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Hashable

import networkx as nx
import numpy as np

from .base import FeatureExtractor


WLLabel = tuple


class WLFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        n_iter: int = 2,
        node_attr: str = "room_type",
        edge_attr: str | None = "connectivity",
    ):
        self.n_iter = int(n_iter)
        self.node_attr = node_attr
        self.edge_attr = edge_attr
        self.vocab: dict[WLLabel, int] = {}

    # --- WL refinement ---------------------------------------------------

    def _init_label(self, graph: nx.Graph, node: Hashable) -> WLLabel:
        return ("v", graph.nodes[node].get(self.node_attr, -1))

    def _refine(
        self, graph: nx.Graph, labels: dict[Hashable, WLLabel]
    ) -> dict[Hashable, WLLabel]:
        new_labels: dict[Hashable, WLLabel] = {}
        for v in graph.nodes():
            msg: list[tuple] = []
            for u in graph.neighbors(v):
                edge_label: Hashable = -1
                if self.edge_attr is not None:
                    edge_label = graph.edges[v, u].get(self.edge_attr, -1)
                msg.append((edge_label, labels[u]))
            msg.sort()
            new_labels[v] = (labels[v], tuple(msg))
        return new_labels

    def _iter_labels(self, graph: nx.Graph):
        """Yield (iteration, node, label) over all nodes and refinement rounds."""
        labels = {v: self._init_label(graph, v) for v in graph.nodes()}
        for v in graph.nodes():
            yield 0, v, (0,) + labels[v]
        for it in range(1, self.n_iter + 1):
            labels = self._refine(graph, labels)
            for v in graph.nodes():
                yield it, v, (it,) + labels[v]

    # --- FeatureExtractor API --------------------------------------------

    def fit(self, graphs: list[nx.Graph]) -> "WLFeatureExtractor":
        for g in graphs:
            for _, _, lbl in self._iter_labels(g):
                if lbl not in self.vocab:
                    self.vocab[lbl] = len(self.vocab)
        return self

    def transform(self, graphs: list[nx.Graph]) -> np.ndarray:
        n = len(graphs)
        d = len(self.vocab)
        out = np.zeros((n, d), dtype=np.float32)
        for i, g in enumerate(graphs):
            for _, _, lbl in self._iter_labels(g):
                col = self.vocab.get(lbl)
                if col is not None:
                    out[i, col] += 1.0
        return out

    def transform_one(self, graph: nx.Graph) -> np.ndarray:
        return self.transform([graph])[0]

    def node_signatures(self, graph: nx.Graph) -> list[set]:
        per_node: dict[Hashable, set] = {v: set() for v in graph.nodes()}
        for _, v, lbl in self._iter_labels(graph):
            per_node[v].add(lbl)
        return [per_node[v] for v in graph.nodes()]

    # --- Persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "n_iter": self.n_iter,
                    "node_attr": self.node_attr,
                    "edge_attr": self.edge_attr,
                    "vocab": self.vocab,
                },
                f,
            )

    def load(self, path: str | Path) -> "WLFeatureExtractor":
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        self.n_iter = int(state["n_iter"])
        self.node_attr = state["node_attr"]
        self.edge_attr = state["edge_attr"]
        self.vocab = dict(state["vocab"])
        return self

    @property
    def dim(self) -> int:
        return len(self.vocab)
