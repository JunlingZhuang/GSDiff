# -*- coding: utf-8 -*-
"""Colour-block floor-plan image -> Plan JSON.

Each legend colour is an instance seed: one cleaned connected component of a room
colour is one room. Seeds then grow toward each other, but only up to the wall
**centre line** (a fill bounded by half the wall thickness, NOT a whole-footprint
flood). So adjacent rooms end up sharing a boundary exactly on the wall mid-line
while the black wall band stays as the real divider between them — walls are
boundaries, not gaps to be erased. Each room's polygon is traced directly from its
own region and forced to right angles.

This replaces the earlier whole-footprint nearest-seed flood, which assigned every
pixel inside the building to its nearest room: that ate the walls, let a room
balloon across a missing neighbour, and sprouted long medial-axis triangles where
three rooms met. Capping the fill at half a wall removes all three.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from hfagent.schema.palette import ROOM_RGB
from hfagent.schema.plan import Plan, Room
from hfagent.tools.raster_geometry import clean_ring, snap_axes

MAX_COLOR_DIST = 90.0


def _to_rgb_array(image) -> np.ndarray:
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("expected RGB image")
    return arr


def _classify(arr: np.ndarray) -> np.ndarray:
    """Return label ids into ROOM_RGB order; -1 means wall/bg/noise."""
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3).astype(np.float32)
    legend = np.array(list(ROOM_RGB.values()), dtype=np.float32)
    d = np.linalg.norm(flat[:, None, :] - legend[None, :, :], axis=2)
    labels = d.argmin(axis=1)
    labels[d.min(axis=1) > MAX_COLOR_DIST] = -1
    return labels.reshape(h, w)


def _clean_type_mask(mask: np.ndarray, min_room_px: int) -> np.ndarray:
    """Despeckle, reconnect across thin text strokes, drop sub-room blobs.

    Holes left by labels / door arcs are NOT filled here — the centre-line fill
    reclaims them later, which also avoids ever painting over an enclosed room of a
    different colour.
    """
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, comp = cv2.connectedComponents(mask)
    out = np.zeros_like(mask)
    for ci in range(1, n):
        cmask = comp == ci
        if int(cmask.sum()) >= min_room_px:
            out[cmask] = 1
    return out


def _estimate_wall_px(arr: np.ndarray, labels: np.ndarray) -> int:
    """Typical wall thickness in px from the dark (non-room) band widths."""
    gray = arr.astype(np.float32).mean(axis=2)
    dark = ((labels < 0) & (gray < 150.0)).astype(np.uint8)
    if not dark.any():
        return max(3, round(arr.shape[1] * 0.006))

    dist = cv2.distanceTransform(dark, cv2.DIST_L2, 3)
    vals = dist[dist > 0]
    if vals.size == 0:
        return max(3, round(arr.shape[1] * 0.006))
    wall_px = int(round(float(np.percentile(vals, 92)) * 2.0))
    return max(3, min(wall_px, max(6, round(arr.shape[1] * 0.035))))


def _grow_to_centerline(inst: np.ndarray, half_px: int) -> np.ndarray:
    """Bounded fill: give each non-room pixel within ``half_px`` of a seed to the
    nearest seed, so neighbours meet on the wall mid-line and label/door holes are
    reclaimed by their surrounding room. The distance cap is what keeps walls intact
    and stops the medial-axis triangles a whole-footprint flood produced.
    """
    from scipy import ndimage

    bg = inst == 0
    dist, idx = ndimage.distance_transform_edt(bg, return_distances=True, return_indices=True)
    nearest = inst[tuple(idx)]
    out = inst.copy()
    grow = bg & (dist <= half_px) & (nearest > 0)
    out[grow] = nearest[grow]
    return out


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, comp = cv2.connectedComponents(mask.astype(np.uint8))
    if n <= 1:
        return mask.astype(np.uint8)
    best = max(range(1, n), key=lambda ci: int((comp == ci).sum()))
    return (comp == best).astype(np.uint8)


def _region_corners(mask: np.ndarray, eps: float) -> np.ndarray | None:
    """Trace a region's outer boundary at full resolution down to its corners.

    RETR_EXTERNAL follows concavities (an L/U room with a corner toilet bitten out
    of it stays L/U-shaped) while ignoring interior holes, so residual label pixels
    never punch a fake hole into the polygon. ``approxPolyDP`` reduces the staircased
    raster boundary to its few real corners; right-angling happens later, after the
    plan-wide axis snap has pulled each corner's slightly chamfered coordinates onto
    a shared line.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2).astype(np.float64)
    return approx if len(approx) >= 3 else None


def _rectilinearize(poly: np.ndarray) -> np.ndarray:
    """Force right angles on an already axis-snapped ring.

    After the plan-wide snap each corner sits on a shared x/y line, so any edge that
    is still diagonal is a sub-pixel chamfer whose endpoints differ on both axes.
    Split it into an axis-aligned L through a corner built from existing snapped
    coordinates; because the residual diagonal is tiny, either corner is safe.
    """
    points: list[tuple[float, float]] = []
    for i in range(len(poly)):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % len(poly)]
        points.append((float(ax), float(ay)))
        if abs(ax - bx) > 1e-9 and abs(ay - by) > 1e-9:
            points.append((float(bx), float(ay)))
    return np.array(points, dtype=np.float64)


def cv_parse(
    image,
    px_per_mm: float | None = None,
    min_room_px: int = 400,
    wall_px: int | None = None,
) -> Plan:
    """Parse a flat colour-block plan into a Plan of orthogonal room polygons.

    ``wall_px`` overrides the auto-estimated wall thickness; it sets both the
    centre-line fill cap (half a wall) and the polygon simplification tolerance.
    """
    arr = _to_rgb_array(image)
    labels = _classify(arr)
    types = list(ROOM_RGB.keys())
    wall = int(wall_px) if wall_px is not None else _estimate_wall_px(arr, labels)

    inst = np.zeros(labels.shape, np.int32)
    id_type: dict[int, str] = {}
    for ti, t in enumerate(types):
        mask = (labels == ti).astype(np.uint8)
        if not mask.any():
            continue
        mask = _clean_type_mask(mask, min_room_px)
        n, comp = cv2.connectedComponents(mask)
        for ci in range(1, n):
            rid = len(id_type) + 1
            inst[comp == ci] = rid
            id_type[rid] = t

    if id_type:
        inst = _grow_to_centerline(inst, half_px=max(1, wall // 2 + 1))

    raw: list[tuple[str, np.ndarray]] = []
    eps = max(2.0, wall * 0.5)
    for rid, t in id_type.items():
        cmask = _largest_component(inst == rid)
        if int(cmask.sum()) < min_room_px:
            continue
        corners = _region_corners(cmask, eps)
        if corners is not None:
            raw.append((t, corners))

    # Snap chamfered corners onto plan-wide x/y lines first, THEN right-angle them:
    # snapping turns each near-axis edge truly axis-aligned, so rectilinearising can
    # no longer pick a wrong-side corner and self-intersect.
    if raw:
        snap_axes([p for _, p in raw], tolerance=max(2.0, wall * 0.6))

    rooms: list[Room] = []
    scale = 1.0 / px_per_mm if px_per_mm else 1.0
    for t, poly in raw:
        poly = clean_ring(_rectilinearize(poly))
        if len(poly) < 3:
            continue
        pts = [(float(x) * scale, float(y) * scale) for x, y in poly]
        rooms.append(Room(id=f"r{len(rooms) + 1}", type=t, polygon=pts))

    return Plan(units="mm" if px_per_mm else "px", rooms=rooms)
