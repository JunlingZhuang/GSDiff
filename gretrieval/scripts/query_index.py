"""Query built retrieval indices and print top-k matches per mode.

For a quick demo this script picks a graph from the corpus by index, optionally
masks out a fraction of its nodes to simulate a partial-graph query, and runs
each configured retriever on the result.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gsretrieval.datasets.msd_loader import load_msd_graphs  # noqa: E402
from gsretrieval.features.wl_kernel import WLFeatureExtractor  # noqa: E402
from gsretrieval.retrievers.containment import ContainmentRetriever  # noqa: E402
from gsretrieval.retrievers.cosine import CosineRetriever  # noqa: E402
from gsretrieval.retrievers.node_matching import NodeMatchingRetriever  # noqa: E402
from gsretrieval.retrievers.two_stage import TwoStageRetriever  # noqa: E402


MODES = ("cosine", "containment", "node_matching", "two_stage")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query MSD retrieval indices.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "retrieval" / "msd_wl.yaml"),
        help="Same YAML used at build time.",
    )
    parser.add_argument("--index-dir", default=None, help="Override config output_dir.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument(
        "--query-idx",
        type=int,
        default=0,
        help="Pick the query graph from the corpus by list index.",
    )
    parser.add_argument(
        "--mask-frac",
        type=float,
        default=0.0,
        help="Fraction of nodes to drop from the chosen graph to simulate a partial query.",
    )
    parser.add_argument("--rerank-recall-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_under_root(value: str | Path, root: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p).resolve()


def partial_graph(graph: nx.Graph, mask_frac: float, rng: np.random.Generator) -> nx.Graph:
    if mask_frac <= 0.0:
        return graph
    n = graph.number_of_nodes()
    n_keep = max(1, int(round(n * (1.0 - mask_frac))))
    nodes = list(graph.nodes())
    keep = rng.choice(nodes, size=n_keep, replace=False).tolist()
    return graph.subgraph(keep).copy()


def print_topk(label: str, scores: np.ndarray, indices: np.ndarray, elapsed: float) -> None:
    print(f"\n[{label}]  elapsed={elapsed * 1000:.0f} ms")
    for rank, (s, i) in enumerate(zip(scores, indices), start=1):
        print(f"  #{rank:2d}  idx={int(i):5d}  score={float(s):.4f}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    index_dir = resolve_under_root(args.index_dir or cfg["output_dir"], PROJECT_ROOT)
    graphs_path = resolve_under_root(cfg["graphs_path"], PROJECT_ROOT)

    extractor = WLFeatureExtractor().load(index_dir / "extractor.pkl")
    filter_cfg = cfg.get("filter", {}) or {}
    graphs = load_msd_graphs(
        graphs_path,
        min_nodes=int(filter_cfg.get("min_nodes", 2)),
        max_nodes=filter_cfg.get("max_nodes"),
    )

    rng = np.random.default_rng(args.seed)
    base_graph = graphs[args.query_idx]
    query_graph = partial_graph(base_graph, args.mask_frac, rng)
    print(
        f"[query] base idx={args.query_idx}: "
        f"|V|={base_graph.number_of_nodes()}, |E|={base_graph.number_of_edges()}  "
        f"-> partial |V|={query_graph.number_of_nodes()}, |E|={query_graph.number_of_edges()}"
    )

    modes = list(MODES) if args.mode == "all" else [args.mode]

    if "cosine" in modes:
        r = CosineRetriever(extractor).load(index_dir / "cosine")
        t0 = time.time()
        scores, idx = r.query(query_graph, args.k)
        print_topk("cosine (full -> full)", scores, idx, time.time() - t0)

    need_containment = "containment" in modes or "two_stage" in modes
    need_node = "node_matching" in modes or "two_stage" in modes

    r_c: ContainmentRetriever | None = None
    r_n: NodeMatchingRetriever | None = None
    if need_containment:
        r_c = ContainmentRetriever(extractor).load(index_dir / "containment")
    if need_node:
        r_n = NodeMatchingRetriever(extractor).load(index_dir / "node_matching")

    if "containment" in modes:
        t0 = time.time()
        scores, idx = r_c.query(query_graph, args.k)
        print_topk("containment (mode A, partial -> full)", scores, idx, time.time() - t0)

    if "node_matching" in modes:
        t0 = time.time()
        scores, idx = r_n.query(query_graph, args.k)
        print_topk("node_matching (mode B, partial -> full)", scores, idx, time.time() - t0)

    if "two_stage" in modes:
        r2 = TwoStageRetriever(r_c, r_n, recall_k=args.rerank_recall_k)
        t0 = time.time()
        scores, idx = r2.query(query_graph, args.k)
        print_topk(
            f"two_stage (recall@{args.rerank_recall_k} -> rerank)",
            scores,
            idx,
            time.time() - t0,
        )


if __name__ == "__main__":
    main()
