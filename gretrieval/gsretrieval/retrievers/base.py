"""Abstract Retriever interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import networkx as nx
import numpy as np


class Retriever(ABC):
    """Index a corpus of graphs and answer top-k queries.

    Each retriever chooses what to consume at `fit` time (feature matrix vs.
    raw graphs) and what to return: a (scores, indices) pair sorted by
    descending score. Both arrays have length min(k, corpus_size).
    """

    @abstractmethod
    def query(self, query_graph: nx.Graph, k: int) -> tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @abstractmethod
    def load(self, path: str | Path) -> "Retriever": ...
