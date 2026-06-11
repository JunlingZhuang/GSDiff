"""Abstract base class for graph feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import networkx as nx
import numpy as np


class FeatureExtractor(ABC):
    """Map a graph to a fixed-length feature vector.

    Subclasses implement `fit`, `transform`, `save`, `load`. Extractors that
    also support node-level matching should override `node_signatures` to
    return per-node descriptor sets used by NodeMatchingRetriever.
    """

    @abstractmethod
    def fit(self, graphs: list[nx.Graph]) -> "FeatureExtractor":
        """Learn vocabulary / normalization from a corpus."""

    @abstractmethod
    def transform(self, graphs: list[nx.Graph]) -> np.ndarray:
        """Encode each graph as a row in an (N, D) feature matrix."""

    def fit_transform(self, graphs: list[nx.Graph]) -> np.ndarray:
        self.fit(graphs)
        return self.transform(graphs)

    def node_signatures(self, graph: nx.Graph) -> list[set]:
        raise NotImplementedError(
            f"{type(self).__name__} does not expose per-node signatures"
        )

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @abstractmethod
    def load(self, path: str | Path) -> "FeatureExtractor": ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Feature dimensionality (valid after `fit`)."""
