# -*- coding: utf-8 -*-
"""Tool ② cv_parse: colour-block floorplan image -> Plan JSON.

I/O contract:
    cv_parse(image, px_per_mm=None, min_room_px=400) -> Plan
      image: PIL.Image | np.ndarray(H,W,3 RGB) | path str
      px_per_mm: known render scale -> output in mm; None -> output units "px".

Pipeline (deterministic, no ML): nearest-legend-colour classification per
pixel -> per-type masks -> morphological clean -> connected components ->
contour -> Douglas-Peucker simplify -> Manhattan snap (dominant-axis
orthogonalisation) -> Plan.

The colour tolerance is generous (nearest-colour, capped distance) because the
same parser must handle both our exact renders and Gemini's slightly drifted
colours.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from hfagent.schema.palette import ROOM_RGB
from hfagent.schema.plan import Plan, Room

MAX_COLOR_DIST = 90.0  # pixels farther than this from every legend colour = wall/bg/noise


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
    """(H,W) int label map: index into ROOM_RGB order, -1 = none."""
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3).astype(np.float32)
    legend = np.array(list(ROOM_RGB.values()), dtype=np.float32)  # (K,3)
    # (N,K) distances — fine at Phase-0 image sizes (~1-4 MP)
    d = np.linalg.norm(flat[:, None, :] - legend[None, :, :], axis=2)
    labels = d.argmin(axis=1)
    labels[d.min(axis=1) > MAX_COLOR_DIST] = -1
    return labels.reshape(h, w)


def _grid_contour(cmask: np.ndarray, grid: int) -> np.ndarray | None:
    """Trace the component boundary on a coarse grid — the resulting polygon is
    rectilinear (Manhattan) BY CONSTRUCTION, instead of hoping approxPolyDP
    output is near-orthogonal."""
    h, w = cmask.shape
    small = cv2.resize(cmask.astype(np.float32), (max(1, w // grid), max(1, h // grid)),
                       interpolation=cv2.INTER_AREA)
    small = (small > 0.45).astype(np.uint8)
    small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # kill 1-cell notches
    contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(cnt) < 3:
        return None
    # rectilinearise any residual diagonal step (insert the corner point)
    pts: list[tuple[float, float]] = []
    n = len(cnt)
    for i in range(n):
        a, b = cnt[i], cnt[(i + 1) % n]
        pts.append((a[0], a[1]))
        if a[0] != b[0] and a[1] != b[1]:
            pts.append((b[0], a[1]))
    return (np.array(pts) + 0.5) * grid


def _snap_axes(polys: list[np.ndarray], tol: float) -> None:
    """Cluster all x (and y) coordinates across ALL rooms and snap each cluster
    to its mean — shared walls become exactly collinear plan-wide."""
    for axis in (0, 1):
        vals = np.sort(np.unique(np.concatenate([p[:, axis] for p in polys])))
        mapping: dict[float, float] = {}
        group: list[float] = [float(vals[0])]
        for v in list(vals[1:]) + [float("inf")]:
            if v - group[-1] <= tol:
                group.append(float(v))
            else:
                m = float(np.mean(group))
                mapping.update({g: m for g in group})
                group = [float(v)]
        for p in polys:
            p[:, axis] = [mapping[float(v)] for v in p[:, axis]]


def _clean_ring(pts: np.ndarray) -> np.ndarray:
    """Drop duplicate and collinear vertices left over after snapping."""
    out: list[np.ndarray] = []
    n = len(pts)
    for i in range(n):
        b = pts[i]
        if out and np.allclose(b, out[-1]):
            continue
        out.append(b)
    pts = np.array(out)
    n = len(pts)
    keep = []
    for i in range(n):
        a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(cross) > 1e-6:
            keep.append(b)
    return np.array(keep) if len(keep) >= 3 else pts


def cv_parse(
    image,
    px_per_mm: float | None = None,
    min_room_px: int = 400,
    grid_px: int = 8,
    expand_px: int = 6,
) -> Plan:
    arr = _to_rgb_array(image)
    labels = _classify(arr)
    types = list(ROOM_RGB.keys())
    kernel = np.ones((5, 5), np.uint8)
    expand_k = np.ones((2 * expand_px + 1, 2 * expand_px + 1), np.uint8)

    raw: list[tuple[str, np.ndarray]] = []
    for ti, t in enumerate(types):
        mask = (labels == ti).astype(np.uint8)
        if not mask.any():
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        n, comp = cv2.connectedComponents(mask)
        for ci in range(1, n):
            cmask = (comp == ci).astype(np.uint8)
            if int(cmask.sum()) < min_room_px:
                continue
            # grow back the half wall thickness eaten by the black stroke, so
            # adjacent rooms meet at the wall centreline instead of leaving gaps
            cmask = cv2.dilate(cmask, expand_k)
            poly = _grid_contour(cmask, grid_px)
            if poly is None:
                continue
            raw.append((t, poly))

    if raw:
        _snap_axes([p for _, p in raw], tol=grid_px * 1.6)

    rooms: list[Room] = []
    scale = 1.0 / px_per_mm if px_per_mm else 1.0
    for t, poly in raw:
        poly = _clean_ring(poly)
        if len(poly) < 3:
            continue
        pts = [(float(x) * scale, float(y) * scale) for x, y in poly]
        rooms.append(Room(id=f"r{len(rooms) + 1}", type=t, polygon=pts))

    return Plan(units="mm" if px_per_mm else "px", rooms=rooms)
