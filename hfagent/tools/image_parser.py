# -*- coding: utf-8 -*-
"""Colour-block floor-plan image -> Plan JSON.

The parser treats coloured room pixels as instance seeds, builds a global
room/wall footprint, then assigns every footprint pixel to the nearest room
seed. Shared boundaries are therefore decided once in a single label map instead
of by independently expanding each room polygon.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from hfagent.schema.palette import ROOM_RGB
from hfagent.schema.plan import Plan, Room

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


def _fill_small_holes(mask: np.ndarray, max_hole_px: int) -> np.ndarray:
    """Fill label/text/door-arc holes inside each same-colour component."""
    if not mask.any():
        return mask.astype(np.uint8)

    out = mask.astype(np.uint8).copy()
    n, comp = cv2.connectedComponents(out)
    for ci in range(1, n):
        ys, xs = np.where(comp == ci)
        if xs.size == 0:
            continue
        x0, x1 = max(0, xs.min() - 1), min(mask.shape[1], xs.max() + 2)
        y0, y1 = max(0, ys.min() - 1), min(mask.shape[0], ys.max() + 2)
        region = (comp[y0:y1, x0:x1] == ci).astype(np.uint8)
        if region.shape[0] < 3 or region.shape[1] < 3:
            continue

        bg = (region == 0).astype(np.uint8)
        exterior = bg.copy()
        ff_mask = np.zeros((bg.shape[0] + 2, bg.shape[1] + 2), np.uint8)
        for x in range(bg.shape[1]):
            if exterior[0, x]:
                cv2.floodFill(exterior, ff_mask, (x, 0), 2)
            if exterior[-1, x]:
                cv2.floodFill(exterior, ff_mask, (x, bg.shape[0] - 1), 2)
        for y in range(bg.shape[0]):
            if exterior[y, 0]:
                cv2.floodFill(exterior, ff_mask, (0, y), 2)
            if exterior[y, -1]:
                cv2.floodFill(exterior, ff_mask, (bg.shape[1] - 1, y), 2)

        holes = bg.astype(bool) & (exterior != 2)
        hn, hcomp = cv2.connectedComponents(holes.astype(np.uint8))
        for hi in range(1, hn):
            hmask = hcomp == hi
            if int(hmask.sum()) <= max_hole_px:
                region[hmask] = 1
        out[y0:y1, x0:x1][region.astype(bool)] = 1
    return out


def _clean_type_mask(mask: np.ndarray, min_room_px: int, hole_px: int) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = _fill_small_holes(mask, hole_px)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, comp = cv2.connectedComponents(mask)
    out = np.zeros_like(mask)
    for ci in range(1, n):
        cmask = comp == ci
        if int(cmask.sum()) >= min_room_px:
            out[cmask] = 1
    return out


def _grid_contour(cmask: np.ndarray, grid: int) -> np.ndarray | None:
    """Trace the component boundary on a coarse occupancy grid."""
    h, w = cmask.shape
    small = cv2.resize(
        cmask.astype(np.float32),
        (max(1, w // grid), max(1, h // grid)),
        interpolation=cv2.INTER_AREA,
    )
    small = (small > 0.10).astype(np.uint8)
    small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(cnt) < 3:
        return None
    pts: list[tuple[float, float]] = []
    for i in range(len(cnt)):
        a, b = cnt[i], cnt[(i + 1) % len(cnt)]
        pts.append((a[0], a[1]))
        if a[0] != b[0] and a[1] != b[1]:
            pts.append((b[0], a[1]))
    return (np.array(pts) + 0.5) * grid


def _snap_axes(polys: list[np.ndarray], tol: float) -> None:
    """Snap nearby x/y coordinates plan-wide without chained clusters."""
    if not polys:
        return
    for axis in (0, 1):
        vals = np.sort(np.unique(np.concatenate([p[:, axis] for p in polys])))
        mapping: dict[float, float] = {}
        group: list[float] = [float(vals[0])]
        for v in list(vals[1:]) + [float("inf")]:
            if v - group[0] <= tol:
                group.append(float(v))
            else:
                mean = float(np.mean(group))
                mapping.update({g: mean for g in group})
                group = [float(v)]
        for p in polys:
            p[:, axis] = [mapping[float(v)] for v in p[:, axis]]


def _clean_ring(pts: np.ndarray) -> np.ndarray:
    out: list[np.ndarray] = []
    for b in pts:
        if out and np.allclose(b, out[-1]):
            continue
        out.append(b)
    pts = np.array(out)
    keep = []
    for i in range(len(pts)):
        a, b, c = pts[(i - 1) % len(pts)], pts[i], pts[(i + 1) % len(pts)]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(cross) > 1e-6:
            keep.append(b)
    return np.array(keep) if len(keep) >= 3 else pts


def _estimate_wall_px(arr: np.ndarray, labels: np.ndarray) -> int:
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


def _fill_enclosed_background(mask: np.ndarray) -> np.ndarray:
    bg = (mask == 0).astype(np.uint8)
    exterior = bg.copy()
    ff_mask = np.zeros((bg.shape[0] + 2, bg.shape[1] + 2), np.uint8)
    for x in range(bg.shape[1]):
        if exterior[0, x]:
            cv2.floodFill(exterior, ff_mask, (x, 0), 2)
        if exterior[-1, x]:
            cv2.floodFill(exterior, ff_mask, (x, bg.shape[0] - 1), 2)
    for y in range(bg.shape[0]):
        if exterior[y, 0]:
            cv2.floodFill(exterior, ff_mask, (0, y), 2)
        if exterior[y, -1]:
            cv2.floodFill(exterior, ff_mask, (bg.shape[1] - 1, y), 2)
    holes = bg.astype(bool) & (exterior != 2)
    out = mask.astype(np.uint8).copy()
    out[holes] = 1
    return out


def _building_footprint(inst: np.ndarray, arr: np.ndarray, labels: np.ndarray, wall_px: int) -> np.ndarray:
    gray = arr.astype(np.float32).mean(axis=2)
    dark = (labels < 0) & (gray < 150.0)
    occupied = ((inst > 0) | dark).astype(np.uint8)

    k = max(3, int(round(wall_px * 2.0)) | 1)
    kernel = np.ones((k, k), np.uint8)
    closed = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, kernel)
    return _fill_enclosed_background(closed).astype(bool)


def _propagate_instances(inst: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    bg = inst == 0
    _, idx = ndimage.distance_transform_edt(bg, return_indices=True)
    nearest = inst[tuple(idx)]
    out = inst.copy()
    grow = footprint & bg & (nearest > 0)
    out[grow] = nearest[grow]
    return out


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, comp = cv2.connectedComponents(mask.astype(np.uint8))
    if n <= 1:
        return mask.astype(np.uint8)
    best = max(range(1, n), key=lambda ci: int((comp == ci).sum()))
    return (comp == best).astype(np.uint8)


def cv_parse(
    image,
    px_per_mm: float | None = None,
    min_room_px: int = 400,
    grid_px: int = 8,
    gap_fill_px: int | None = None,
) -> Plan:
    arr = _to_rgb_array(image)
    labels = _classify(arr)
    types = list(ROOM_RGB.keys())
    hole_px = max(min_room_px * 8, round(arr.shape[0] * arr.shape[1] * 0.001))

    inst = np.zeros(labels.shape, np.int32)
    id_type: dict[int, str] = {}
    for ti, t in enumerate(types):
        mask = (labels == ti).astype(np.uint8)
        if not mask.any():
            continue
        mask = _clean_type_mask(mask, min_room_px, hole_px)
        n, comp = cv2.connectedComponents(mask)
        for ci in range(1, n):
            cmask = comp == ci
            rid = len(id_type) + 1
            inst[cmask] = rid
            id_type[rid] = t

    if id_type:
        wall_px = int(gap_fill_px) if gap_fill_px is not None else _estimate_wall_px(arr, labels)
        footprint = _building_footprint(inst, arr, labels, wall_px)
        inst = _propagate_instances(inst, footprint)

    raw: list[tuple[str, np.ndarray]] = []
    for rid, t in id_type.items():
        cmask = _largest_component(inst == rid)
        if int(cmask.sum()) < min_room_px:
            continue
        poly = _grid_contour(cmask, grid_px)
        if poly is not None:
            raw.append((t, poly))

    if raw:
        _snap_axes([p for _, p in raw], tol=grid_px * 2.5)

    rooms: list[Room] = []
    scale = 1.0 / px_per_mm if px_per_mm else 1.0
    for t, poly in raw:
        poly = _clean_ring(poly)
        if len(poly) < 3:
            continue
        pts = [(float(x) * scale, float(y) * scale) for x, y in poly]
        rooms.append(Room(id=f"r{len(rooms) + 1}", type=t, polygon=pts))

    return Plan(units="mm" if px_per_mm else "px", rooms=rooms)
