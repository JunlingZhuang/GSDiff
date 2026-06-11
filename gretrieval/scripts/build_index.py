"""Build WL retrieval indices over a corpus of MSD NetworkX graphs.

Loads `graphs.p`, fits the configured feature extractor and each requested
retriever, and writes everything under `output_dir/` so `query_index.py` can
answer queries without re-extracting features.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

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


FEATURE_REGISTRY = {
    "wl": WLFeatureExtractor,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build WL retrieval indices for MSD.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "retrieval" / "msd_wl.yaml"),
        help="Path to retrieval YAML config.",
    )
    parser.add_argument("--graphs-path", default=None, help="Override config graphs_path.")
    parser.add_argument("--output-dir", default=None, help="Override config output_dir.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_under_root(value: str | Path, root: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p).resolve()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    graphs_path = resolve_under_root(args.graphs_path or cfg["graphs_path"], PROJECT_ROOT)
    output_dir = resolve_under_root(args.output_dir or cfg["output_dir"], PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    filter_cfg = cfg.get("filter", {}) or {}
    feat_cfg = dict(cfg["features"])
    feat_name = feat_cfg.pop("name", "wl")
    retr_names = set(cfg.get("retrievers", ["cosine", "containment", "node_matching"]))

    if feat_name not in FEATURE_REGISTRY:
        raise ValueError(
            f"Unknown feature extractor '{feat_name}'. Available: {list(FEATURE_REGISTRY)}"
        )

    print(f"[load] {graphs_path}")
    t0 = time.time()
    graphs = load_msd_graphs(
        graphs_path,
        min_nodes=int(filter_cfg.get("min_nodes", 2)),
        max_nodes=filter_cfg.get("max_nodes"),
    )
    print(f"[load] {len(graphs)} graphs in {time.time() - t0:.1f}s")

    print(f"[fit] {feat_name}({feat_cfg})")
    t0 = time.time()
    extractor = FEATURE_REGISTRY[feat_name](**feat_cfg)
    features = extractor.fit_transform(graphs)
    print(
        f"[fit] vocab={extractor.dim}, features={features.shape}, "
        f"dense_MB={features.nbytes / 1e6:.1f} in {time.time() - t0:.1f}s"
    )

    extractor.save(output_dir / "extractor.pkl")
    np.save(output_dir / "features.npy", features)
    with (output_dir / "corpus_meta.pkl").open("wb") as f:
        pickle.dump({"n_graphs": len(graphs), "graphs_path": str(graphs_path)}, f)

    if "cosine" in retr_names:
        t0 = time.time()
        CosineRetriever(extractor).fit(features).save(output_dir / "cosine")
        print(f"[fit] cosine in {time.time() - t0:.1f}s")

    if "containment" in retr_names:
        t0 = time.time()
        ContainmentRetriever(extractor).fit(features).save(output_dir / "containment")
        print(f"[fit] containment in {time.time() - t0:.1f}s")

    if "node_matching" in retr_names or "two_stage" in retr_names:
        t0 = time.time()
        NodeMatchingRetriever(extractor).fit(graphs).save(output_dir / "node_matching")
        print(f"[fit] node_matching in {time.time() - t0:.1f}s")

    print(f"[done] indices written to {output_dir}")


if __name__ == "__main__":
    main()
