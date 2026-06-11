"""MSD graph retrieval service.

Wraps the `gretrieval` package: loads the WL extractor, the four retrievers,
and the raw MSD corpus so we can serve top-k partial-graph queries from the
frontend. The corpus and the extractor are loaded lazily on first request.

Each retrieved corpus graph carries `geometry` polygons per node (from
`digress/scripts/prepare_msd_graphs.py`), so we also rasterize a small
floorplan PNG and surface it to the frontend as a base64 data URI. Rendering
mirrors `draw_floor_shapes` in prepare_msd_graphs.py for visual consistency
with the existing MSD vis pipeline.
"""

from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # must be set before any pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

from app.config import PROJECT_ROOT  # noqa: E402
from app.services.model_status import model_status  # noqa: E402


GRETRIEVAL_ROOT = Path(PROJECT_ROOT) / "gretrieval"
# Index built from digress/data/msd_wall/graphs.p so the corpus includes wall
# edges (in addition to door / passage / entrance).
GRETRIEVAL_DATA_DIR = GRETRIEVAL_ROOT / "data" / "msd_wall" / "wl_n3"

# Edge-type table mirrors `app/lib/constants.ts` EDGE_TYPES_MSD. The retriever
# corpus uses connectivity *strings*, so we round-trip via these lookups.
MSD_EDGE_DECODER = ["none", "wall", "passage", "door", "entrance"]
MSD_NODE_DECODER = [
    "Bedroom",
    "Livingroom",
    "Kitchen",
    "Dining",
    "Corridor",
    "Stairs",
    "Storeroom",
    "Bathroom",
    "Balcony",
]


SUPPORTED_MODES = ("cosine", "containment", "node_matching", "two_stage")


# Hex colors mirror digress/scripts/prepare_msd_graphs.py NODE_COLORS so the
# retrieval thumbnails match the existing MSD vis pipeline.
MSD_NODE_COLORS_HEX: tuple[str, ...] = (
    "#8da0cb",  # 0 Bedroom
    "#47b39c",  # 1 Livingroom
    "#f1c27d",  # 2 Kitchen
    "#fdae6b",  # 3 Dining
    "#fdd0a2",  # 4 Corridor
    "#72246c",  # 5 Stairs
    "#ffd92f",  # 6 Storeroom
    "#bdbdbd",  # 7 Bathroom
    "#a6d854",  # 8 Balcony
)

# Edge styles match prepare_msd_graphs.py EDGE_COLORS / linestyle conventions.
MSD_EDGE_STYLE: dict[str, tuple[str, str]] = {
    "wall":     ("#334155", ":"),
    "passage":  ("#64748b", "--"),
    "door":     ("#c2410c", "-"),
    "entrance": ("#d97706", "-."),
}


def _node_position(graph: nx.Graph, node) -> tuple[float, float] | None:
    """Return (x, y) for a node, preferring `centroid` over polygon mean."""
    cen = graph.nodes[node].get("centroid")
    if cen is not None:
        try:
            return float(cen[0]), float(cen[1])
        except (TypeError, IndexError):
            pass
    geom = graph.nodes[node].get("geometry")
    if geom and len(geom) > 0:
        xs = [float(p[0]) for p in geom]
        ys = [float(p[1]) for p in geom]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return None


def _render_msd_floorplan_png(graph: nx.Graph, *, size_px: int = 320, dpi: int = 80) -> str:
    """Render a single-panel floor-shapes view as a base64 PNG data URI.

    This is the `draw_floor_shapes` panel from prepare_msd_graphs.py distilled
    to one tight matplotlib figure (no title, no axes), sized for a results
    card thumbnail. Returns an empty white PNG if the graph has no usable
    geometry.
    """
    inches = max(size_px / dpi, 1.0)
    fig, ax = plt.subplots(figsize=(inches, inches), dpi=dpi)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    xs: list[float] = []
    ys: list[float] = []
    drew_any = False
    for node in graph.nodes():
        geom = graph.nodes[node].get("geometry")
        if not geom or len(geom) < 3:
            continue
        room_type = int(graph.nodes[node].get("room_type", 0))
        color = MSD_NODE_COLORS_HEX[room_type % len(MSD_NODE_COLORS_HEX)]
        ax.add_patch(
            MplPolygon(
                geom,
                closed=True,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.95,
            )
        )
        for x, y in geom:
            xs.append(float(x))
            ys.append(float(y))
        drew_any = True

    if drew_any and xs and ys:
        # Square aspect with a small margin so polygons don't kiss the edge.
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        if span <= 0:
            span = 1.0
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        half = span / 2.0 * 1.08
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)

    buf = io.BytesIO()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _render_msd_bubble_png(graph: nx.Graph, *, size_px: int = 320, dpi: int = 80) -> str:
    """Render bubble-graph view: room-colored nodes + typed edges, no force sim.

    Positions come from each node's `centroid` field, so the bubble graph lays
    out on the same world coordinates as the floorplan. The crop is tight to
    the node cloud + a small margin so there's no wasted white space.
    """
    inches = max(size_px / dpi, 1.0)
    fig, ax = plt.subplots(figsize=(inches, inches), dpi=dpi)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    positions: dict = {}
    xs: list[float] = []
    ys: list[float] = []
    for node in graph.nodes():
        pt = _node_position(graph, node)
        if pt is None:
            continue
        positions[node] = pt
        xs.append(pt[0])
        ys.append(pt[1])

    if not positions:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    span = max(max(xs) - min(xs), max(ys) - min(ys))
    if span <= 0:
        span = 1.0
    radius = span * 0.035  # match prepare_msd_graphs.py draw_bubble

    for u, v, data in graph.edges(data=True):
        if u not in positions or v not in positions:
            continue
        sx, sy = positions[u]
        tx, ty = positions[v]
        conn = data.get("connectivity", "passage")
        color, style = MSD_EDGE_STYLE.get(conn, ("#64748b", "-"))
        ax.plot([sx, tx], [sy, ty], color=color, linewidth=1.6, linestyle=style, alpha=0.9, zorder=2)

    for node, (x, y) in positions.items():
        rt = int(graph.nodes[node].get("room_type", 0))
        color = MSD_NODE_COLORS_HEX[rt % len(MSD_NODE_COLORS_HEX)]
        ax.add_patch(
            plt.Circle((x, y), radius, facecolor=color, edgecolor="black", linewidth=0.8, alpha=0.95, zorder=3)
        )

    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    half = span / 2.0 * 1.12  # small margin so node circles don't clip
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)

    buf = io.BytesIO()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _ensure_import_path() -> None:
    p = str(GRETRIEVAL_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def _edge_int_to_str(edge_type: int) -> str:
    if 0 <= edge_type < len(MSD_EDGE_DECODER):
        return MSD_EDGE_DECODER[edge_type]
    return "none"


def _edge_str_to_int(label: str) -> int:
    try:
        return MSD_EDGE_DECODER.index(label)
    except ValueError:
        return 0


def _graph_to_record(graph: nx.Graph) -> dict[str, Any]:
    """Convert a corpus NetworkX graph to the GeneratedGraph JSON shape."""
    nodes = list(graph.nodes())
    id_map = {nid: i for i, nid in enumerate(nodes)}
    n = len(nodes)

    rooms = [int(graph.nodes[v].get("room_type", 0)) for v in nodes]
    adjacency = [[0] * n for _ in range(n)]
    edge_type_matrix = [[0] * n for _ in range(n)]
    edges: list[dict[str, Any]] = []

    for u, v, data in graph.edges(data=True):
        i, j = id_map[u], id_map[v]
        conn = data.get("connectivity", "none")
        edge_int = _edge_str_to_int(conn)
        adjacency[i][j] = 1
        adjacency[j][i] = 1
        edge_type_matrix[i][j] = edge_int
        edge_type_matrix[j][i] = edge_int
        edges.append(
            {
                "source": i,
                "target": j,
                "edge_type": edge_int,
                "edge_label": conn,
            }
        )

    return {
        "num_nodes": n,
        "num_edges": len(edges),
        "rooms": rooms,
        "adjacency": adjacency,
        "edge_types": edge_type_matrix,
        "nodes": [
            {
                "id": i,
                "attr": rooms[i],
                "room_type": MSD_NODE_DECODER[rooms[i]] if rooms[i] < len(MSD_NODE_DECODER) else f"class_{rooms[i]}",
            }
            for i in range(n)
        ],
        "edges": edges,
    }


def _query_to_nx(rooms: list[int], edges_in: list[dict[str, Any]]) -> nx.Graph:
    g = nx.Graph()
    for i, room in enumerate(rooms):
        g.add_node(i, room_type=int(room))
    for e in edges_in:
        s, t = int(e["source"]), int(e["target"])
        if s == t or s >= len(rooms) or t >= len(rooms):
            continue
        if g.has_edge(s, t):
            continue
        g.add_edge(s, t, connectivity=_edge_int_to_str(int(e.get("edge_type", 0))))
    return g


class MSDGraphRetriever:
    """Lazily-loaded MSD retrieval backend."""

    STATUS_KEY = "gretrieval_msd"

    def __init__(self) -> None:
        self._extractor: Any = None
        self._retrievers: dict[str, Any] = {}
        self._corpus: list[nx.Graph] = []
        model_status.register(
            self.STATUS_KEY,
            group="Retrieval",
            label="MSD graph retriever (WL + FAISS)",
            metadata={"index": str(GRETRIEVAL_DATA_DIR)},
        )

    def supported_modes(self) -> list[str]:
        return list(SUPPORTED_MODES)

    def is_loaded(self) -> bool:
        return self._extractor is not None and bool(self._retrievers)

    def retrieve(
        self,
        *,
        rooms: list[int],
        edges_in: list[dict[str, Any]],
        mode: str,
        k: int,
        recall_k: int = 50,
    ) -> dict[str, Any]:
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported retrieval mode '{mode}'. Supported: {list(SUPPORTED_MODES)}")
        self._ensure_loaded()

        query_graph = _query_to_nx(rooms, edges_in)
        if query_graph.number_of_nodes() == 0:
            raise ValueError("Query graph must have at least one node.")

        retriever = self._retrievers[mode]
        if mode == "two_stage":
            retriever.recall_k = int(recall_k)

        start = time.perf_counter()
        scores, indices = retriever.query(query_graph, int(k))
        elapsed = time.perf_counter() - start

        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            graph = self._corpus[int(idx)]
            results.append(
                {
                    "idx": int(idx),
                    "score": float(score),
                    "graph": _graph_to_record(graph),
                    "floorplan_image": _render_msd_floorplan_png(graph),
                    "bubble_image": _render_msd_bubble_png(graph),
                }
            )

        return {
            "mode": mode,
            "k": int(k),
            "retrieval_seconds": elapsed,
            "room_types": list(MSD_NODE_DECODER),
            "edge_types": list(MSD_EDGE_DECODER),
            "results": results,
        }

    def _ensure_loaded(self) -> None:
        if self.is_loaded():
            return
        model_status.loading(self.STATUS_KEY)
        try:
            _ensure_import_path()
            from gsretrieval.datasets.msd_loader import load_msd_graphs  # noqa: E402
            from gsretrieval.features.wl_kernel import WLFeatureExtractor  # noqa: E402
            from gsretrieval.retrievers.containment import ContainmentRetriever  # noqa: E402
            from gsretrieval.retrievers.cosine import CosineRetriever  # noqa: E402
            from gsretrieval.retrievers.node_matching import NodeMatchingRetriever  # noqa: E402
            from gsretrieval.retrievers.two_stage import TwoStageRetriever  # noqa: E402

            if not GRETRIEVAL_DATA_DIR.exists():
                raise FileNotFoundError(
                    f"Retrieval index not found at {GRETRIEVAL_DATA_DIR}. "
                    f"Run `python gretrieval/scripts/build_index.py` first."
                )

            extractor = WLFeatureExtractor().load(GRETRIEVAL_DATA_DIR / "extractor.pkl")
            r_cos = CosineRetriever(extractor).load(GRETRIEVAL_DATA_DIR / "cosine")
            r_con = ContainmentRetriever(extractor).load(GRETRIEVAL_DATA_DIR / "containment")
            r_nod = NodeMatchingRetriever(extractor).load(GRETRIEVAL_DATA_DIR / "node_matching")
            r_two = TwoStageRetriever(r_con, r_nod, recall_k=50)

            # Corpus path is recorded in build_index.py's corpus_meta.pkl, but we
            # also need the raw graphs to surface as results, so reload via the
            # same loader (cheap: ~6s, single time).
            import pickle

            with (GRETRIEVAL_DATA_DIR / "corpus_meta.pkl").open("rb") as f:
                corpus_meta = pickle.load(f)
            corpus_path = Path(corpus_meta["graphs_path"])
            graphs = load_msd_graphs(corpus_path, min_nodes=2, max_nodes=64)

            self._extractor = extractor
            self._retrievers = {
                "cosine": r_cos,
                "containment": r_con,
                "node_matching": r_nod,
                "two_stage": r_two,
            }
            self._corpus = graphs
            model_status.loaded(self.STATUS_KEY)
        except Exception as exc:
            model_status.error(self.STATUS_KEY, exc)
            raise


msd_retriever = MSDGraphRetriever()
