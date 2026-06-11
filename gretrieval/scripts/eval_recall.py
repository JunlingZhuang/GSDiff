"""Self-recall evaluation for partial -> full graph retrieval.

For each (mask_frac, retriever) combination, sample N graphs from the corpus,
drop a fraction of their nodes to form a partial query, then check whether the
source graph is ranked in the retriever's top-k.

Metrics
  R@k:  fraction of queries where the source graph appears in the top-k result
  MRR:  mean reciprocal rank of the source graph (0 when outside top-k_max)

Caveat: self-retrieval is a sanity check, not a ground-truth measure of
plausibility. Two corpus graphs may be genuinely interchangeable matches for a
given partial query; this script does not detect that.
"""

from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-recall eval for MSD retrieval.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "retrieval" / "msd_wl.yaml"),
    )
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument(
        "--mask-fracs",
        nargs="+",
        type=float,
        default=[0.0, 0.3, 0.5, 0.7],
    )
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--rerank-recall-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-query-nodes",
        type=int,
        default=2,
        help="Skip queries that fall below this many nodes after masking.",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional path to dump per-row results as JSON.",
    )
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


def rank_of(indices: np.ndarray, target: int) -> int | None:
    hits = np.where(indices == target)[0]
    if hits.size == 0:
        return None
    return int(hits[0]) + 1


def evaluate_mode(retriever, queries, ground_truths, k_max):
    ranks: list[int | None] = []
    t0 = time.time()
    for query, gt in zip(queries, ground_truths):
        _, idx = retriever.query(query, k_max)
        ranks.append(rank_of(idx, gt))
    elapsed = time.time() - t0
    return ranks, elapsed


def summarize(ranks: list[int | None], ks: list[int]) -> dict[str, float]:
    n = len(ranks)
    out: dict[str, float] = {}
    for k in ks:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        out[f"R@{k}"] = hits / n if n else 0.0
    out["MRR"] = float(np.mean([1.0 / r if r is not None else 0.0 for r in ranks])) if n else 0.0
    return out


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
    n_corpus = len(graphs)
    num_queries = min(args.num_queries, n_corpus)
    print(
        f"[eval] corpus={n_corpus}  num_queries={num_queries}  "
        f"mask_fracs={args.mask_fracs}  ks={args.ks}  seed={args.seed}"
    )

    sampler = np.random.default_rng(args.seed)
    query_indices = sampler.choice(n_corpus, size=num_queries, replace=False)

    r_cos = CosineRetriever(extractor).load(index_dir / "cosine")
    r_con = ContainmentRetriever(extractor).load(index_dir / "containment")
    r_nod = NodeMatchingRetriever(extractor).load(index_dir / "node_matching")
    r_two = TwoStageRetriever(r_con, r_nod, recall_k=args.rerank_recall_k)
    retrievers = [
        ("cosine", r_cos),
        ("containment", r_con),
        ("node_matching", r_nod),
        ("two_stage", r_two),
    ]
    k_max = max(args.ks)

    rows: list[dict] = []
    for mask in args.mask_fracs:
        queries: list[nx.Graph] = []
        gts: list[int] = []
        for gt_idx in query_indices:
            g = graphs[int(gt_idx)]
            mask_rng = np.random.default_rng(args.seed + int(gt_idx))
            q = partial_graph(g, mask, mask_rng)
            if q.number_of_nodes() < args.min_query_nodes:
                continue
            queries.append(q)
            gts.append(int(gt_idx))
        print(f"\n=== mask_frac={mask:.2f}  n_queries={len(queries)} ===")
        for name, retr in retrievers:
            ranks, elapsed = evaluate_mode(retr, queries, gts, k_max)
            metrics = summarize(ranks, args.ks)
            metric_str = "  ".join(f"{kn}={kv:.3f}" for kn, kv in metrics.items())
            per_q = elapsed * 1000 / max(len(queries), 1)
            print(f"  {name:14s}  {metric_str}  [{elapsed:.1f}s, {per_q:.0f}ms/q]")
            rows.append(
                {
                    "mask": float(mask),
                    "mode": name,
                    "n": len(queries),
                    **metrics,
                    "elapsed_s": elapsed,
                    "ms_per_query": per_q,
                }
            )

    print("\n[summary]")
    cols = ["mask", "mode", "n", *(f"R@{k}" for k in args.ks), "MRR", "ms_per_query"]
    widths = {"mode": 14, "n": 5, "mask": 6}
    header_cells = []
    for c in cols:
        w = widths.get(c, 8)
        header_cells.append(f"{c:>{w}s}")
    print("  " + "  ".join(header_cells))
    for row in rows:
        cells = []
        for c in cols:
            w = widths.get(c, 8)
            val = row[c]
            if isinstance(val, float):
                cells.append(f"{val:>{w}.3f}")
            else:
                cells.append(f"{val:>{w}}")
        print("  " + "  ".join(cells))

    if args.out_json:
        out_path = resolve_under_root(args.out_json, PROJECT_ROOT)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"config": str(args.config), "rows": rows}, f, indent=2)
        print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
