"""Two-stage retrieval: containment recall followed by node-matching rerank.

Containment finds the top `recall_k` candidates over the full corpus in
vectorized time; node-matching then reranks just those candidates. Final
latency is dominated by the (small, bounded) rerank step.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np

from .base import Retriever
from .containment import ContainmentRetriever
from .node_matching import NodeMatchingRetriever, jaccard


class TwoStageRetriever(Retriever):
    def __init__(
        self,
        recall: ContainmentRetriever,
        rerank: NodeMatchingRetriever,
        recall_k: int = 50,
    ):
        self.recall = recall
        self.rerank = rerank
        self.recall_k = int(recall_k)

    def query(self, query_graph: nx.Graph, k: int) -> tuple[np.ndarray, np.ndarray]:
        _, candidates = self.recall.query(query_graph, self.recall_k)
        q_sigs = self.rerank.extractor.node_signatures(query_graph)
        if not q_sigs or candidates.size == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)
        scores = np.zeros(candidates.size, dtype=np.float32)
        for j, idx in enumerate(candidates):
            db_sigs = self.rerank.db_signatures[int(idx)]
            if not db_sigs:
                continue
            scores[j] = float(
                np.mean([max(jaccard(qs, ds) for ds in db_sigs) for qs in q_sigs])
            )
        order = np.argsort(-scores)[: min(k, candidates.size)]
        return scores[order], candidates[order]

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("Compose existing retrievers; persist them individually.")

    def load(self, path: str | Path) -> "TwoStageRetriever":
        raise NotImplementedError("Load the underlying retrievers and pass them to __init__.")
