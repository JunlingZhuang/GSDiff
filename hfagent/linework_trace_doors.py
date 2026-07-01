# -*- coding: utf-8 -*-
"""Offline prototype: faithful wall trace + door-arc-driven gap bridging (linework mode).

EXPERIMENTAL / PARKED. This module is NOT wired into the hfagent pipeline and makes
ZERO network/Gemini calls. It re-architects linework wall+door extraction away from
the gap-classification heuristic in ``tools/wall_graph.py`` toward:

1. WALLS - trace exactly the walls that exist. Faithful set is
   ``_merge_overlaps(_cluster_axes(_raw_segments(...)))`` with NO gap bridging, plus a
   dedicated short-stub pass so legitimately-short walls (door jambs, T-junction piers)
   that ``_raw_segments`` drops on its ``min_length`` floor are kept. Nothing is invented.
2. DOORS - a gap between two collinear traced walls becomes a wall ONLY if a real door
   swing arc (quarter circle at door scale, centred on one jamb, radius ~= the opening
   width) actually straddles that gap. The gap localises the arc search; an angular
   quarter-turn coverage test on the non-wall residual confirms a genuine arc and rejects
   stray ink / a neighbouring door's arc. Confirmed -> bridge the gap AND record a Door.
   No arc -> leave the gap open (real passage / corridor connection).

Run offline:

    python -m hfagent.linework_trace_doors

Writes all artifacts to ``hfagent/out/eval/<YYYYmmdd-HHMMSS>-linework-trace-doors/``.

It reuses the public helpers in ``tools/wall_graph.py`` without modifying them, so the
existing ``reconstruct_wall_graph`` / ``parse_linework`` callers behave identically.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from hfagent.tools.wall_graph import (
    WallSegment,
    _cluster_axes,
    _directional_masks,
    _estimate_wall_width,
    _merge_overlaps,
    _raw_segments,
    _snap_junctions,
)

SRC = (
    r"D:\Github\GSDiff\hfagent\out\eval\20260629-154509"
    r"\hospital-tower-floor\gemini_r1.real.png"
)

# ---- tunable parameters (kept in one place so they can be swept) -------------------
PARAMS = dict(
    # short-stub trace: keep wall-thick components down to this length (in wall_width)
    stub_min_len_factor=1.0,      # min length of a kept short stub, * wall_width
    stub_min_cross_factor=0.60,   # min cross thickness, * wall_width (== _raw_segments)
    stub_min_density=0.45,        # min fill density of the component bbox (== _raw_segments)
    # gap -> door search  (door-scale: a swing arc fits a single-leaf opening)
    door_gap_lo_factor=1.6,       # opening must be wider than this * wall_width
    door_gap_hi_factor=10.0,      # ... and narrower than this * wall_width (else passage)
    radius_scales=(0.9, 1.0, 1.1),  # door fills the opening: r ~= gap width (tight band)
    band_factor=0.16,             # annulus half-width = max(0.4*ww, band_factor * r)
    ang_tol_deg=14.0,             # angular slack outside the [0,90] quarter
    ang_bins=18,                  # bins across the 90 deg quarter (5 deg each)
    coverage_thresh=0.60,         # min fraction of quarter-turn bins occupied
    inlier_ratio_thresh=0.45,     # min fraction of quadrant ink lying ON the annulus
    max_empty_run=4,              # reject arcs with a > this consecutive-empty-bin gap
    min_pixels_factor=1.2,        # min residual pixels supporting the arc = factor * ww
    # gap ink trim / jamb-stub recovery (hidden corner stubs inside a gap)
    gap_ink_frac=0.50,            # cross-band ink fraction that counts as wall ink
    stub_recover_factor=0.5,      # min recovered ink run, * wall_width
    # postprocess corner snap (L-corner pinholes; ink-gated, never crosses an opening)
    corner_snap_reach=2.5,        # max free-end extension onto a perpendicular, * ww
    corner_ink_cover=0.80,        # min fraction of ink-covered positions on the extension
    # metrics
    room_max_area_frac=0.02,      # polygons above this image fraction are corridor-scale
)


@dataclass
class Door:
    orientation: str
    axis: float
    gap_start: float
    gap_end: float
    hinge: tuple[float, float]
    swing: tuple[float, float]   # unit perpendicular pointing into the swing room
    radius: float
    coverage: float
    pixels: int
    inlier_ratio: float = 0.0

    @property
    def center(self) -> tuple[float, float]:
        mid = (self.gap_start + self.gap_end) / 2.0
        return (mid, self.axis) if self.orientation == "horizontal" else (self.axis, mid)


# ---- faithful wall trace ----------------------------------------------------------

def _thick_ink(dark: np.ndarray, wall_width: int) -> np.ndarray:
    """Wall-thick ink only: opening by an ellipse drops thin text/arc strokes (~1-2px)."""
    k = max(3, wall_width - 2)
    return cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def _short_stub_segments(
    dark: np.ndarray,
    hm: np.ndarray,
    vm: np.ndarray,
    wall_axes: dict[str, list[float]],
    wall_width: int,
) -> list[WallSegment]:
    """Recover wall-thick stubs (door jambs, T-junction piers) the long trace drops.

    ``_directional_masks`` opens with a >=15px kernel, so it erases any wall piece
    shorter than that - exactly the door jambs (the short wall between an opening and a
    corner) and the short piers at T-junctions. We re-open ``dark`` with a *short*
    directional kernel, intersect with the wall-thick mask (kills thin text/arc ink),
    drop what the long trace already owns, and keep only short runs that line up with an
    existing wall axis of the same orientation. That alignment test rejects door leaves
    (a leaf sits inside the opening, perpendicular to the wall, at an axis no wall shares),
    so jambs are *traced*, never invented.
    """
    thick = _thick_ink(dark, wall_width)
    stub_k = max(3, round(wall_width * 0.9))
    raw_floor = max(8, wall_width * 2)
    stub_floor = max(2, round(wall_width * PARAMS["stub_min_len_factor"]))
    min_cross = max(2, math.ceil(wall_width * PARAMS["stub_min_cross_factor"]))
    align_tol = wall_width * 1.5

    h_short = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (stub_k, 1))) & thick
    v_short = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, stub_k))) & thick
    hm_d = cv2.dilate(hm, np.ones((3, 3), np.uint8))
    vm_d = cv2.dilate(vm, np.ones((3, 3), np.uint8))
    h_new = ((h_short > 0) & (hm_d == 0)).astype(np.uint8)
    v_new = ((v_short > 0) & (vm_d == 0)).astype(np.uint8)

    def collect(mask, orientation, axes):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        out = []
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            length = width if orientation == "horizontal" else height
            cross = height if orientation == "horizontal" else width
            density = area / max(1, width * height)
            if not (stub_floor <= length < raw_floor) or cross < min_cross or density < PARAMS["stub_min_density"]:
                continue
            ys, xs = np.where(labels == index)
            axis = float(np.median(ys)) if orientation == "horizontal" else float(np.median(xs))
            if not any(abs(axis - a) <= align_tol for a in axes):
                continue
            if orientation == "horizontal":
                out.append(WallSegment(orientation, axis, float(x), float(x + width - 1), float(cross)))
            else:
                out.append(WallSegment(orientation, axis, float(y), float(y + height - 1), float(cross)))
        return out

    return collect(h_new, "horizontal", wall_axes["horizontal"]) + collect(v_new, "vertical", wall_axes["vertical"])


def trace_walls(gray: np.ndarray):
    """Faithful directional-morphology wall trace with a short-stub recovery pass."""
    _, dark = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hm, vm = _directional_masks(dark)
    ww = _estimate_wall_width(hm, vm, gray.shape)

    raw = _raw_segments(hm, "horizontal", ww) + _raw_segments(vm, "vertical", ww)
    count_raw = len(raw)
    wall_axes = {
        "horizontal": [s.axis for s in raw if s.orientation == "horizontal"],
        "vertical": [s.axis for s in raw if s.orientation == "vertical"],
    }
    stubs = _short_stub_segments(dark, hm, vm, wall_axes, ww)
    count_stub = len(stubs)

    def finish(segs):
        clustered = _cluster_axes(segs, tolerance=max(1.0, ww * 0.5))
        return _merge_overlaps(clustered, tolerance=max(1.0, ww * 0.5))

    walls_no_stub = finish(list(raw))
    walls = finish(raw + stubs)
    wall_pixels = ((hm > 0) | (vm > 0)).astype(np.uint8)
    return walls, walls_no_stub, ww, dark, wall_pixels, count_raw, count_stub


# ---- gaps -------------------------------------------------------------------------

def find_gaps(segments: list[WallSegment], wall_width: int):
    """Collinear gaps between consecutive walls sharing an axis.

    ``real_start``/``real_end`` mark gap ends that are genuine traced wall ends (door
    jambs). Pier-splitting and terminal gaps introduce non-real ends; a door may only
    be confirmed on a gap with at least one real jamb (see ``detect_doors``).
    """
    by_axis: dict[tuple[str, float], list[WallSegment]] = {}
    for s in segments:
        by_axis.setdefault((s.orientation, round(s.axis, 1)), []).append(s)
    gaps = []
    for (orientation, axis), items in by_axis.items():
        items.sort(key=lambda s: s.start)
        for a, b in zip(items, items[1:]):
            gap = b.start - a.end
            if gap > wall_width * 0.75:
                gaps.append(dict(orientation=orientation, axis=float(axis),
                                 start=float(a.end), end=float(b.start), width=float(gap),
                                 real_start=True, real_end=True))
    return gaps


def add_terminal_gaps(segments: list[WallSegment], wall_width: int):
    """Gaps between a collinear run's terminal wall END and a crossing perpendicular wall.

    A door drawn right next to a room corner has its far jamb as a short stub merged into
    the perpendicular wall's ink, so the trace has NO collinear segment on that side and
    ``find_gaps`` never sees the opening (e.g. exam-room doors at block corners). For the
    outermost end of every collinear run, the span to the nearest perpendicular wall that
    crosses the axis line (within door range) is a candidate opening; any hidden corner
    stub inside it is recovered from ink by ``trim_gap_ink`` before the arc test.
    """
    by_axis: dict[tuple[str, float], list[WallSegment]] = {}
    for s in segments:
        by_axis.setdefault((s.orientation, round(s.axis, 1)), []).append(s)
    horizontal = [s for s in segments if s.orientation == "horizontal"]
    vertical = [s for s in segments if s.orientation == "vertical"]
    reach = wall_width * 1.5
    lo_w = wall_width * 0.75
    hi_w = wall_width * PARAMS["door_gap_hi_factor"]
    out = []
    for (orientation, axis), items in by_axis.items():
        items.sort(key=lambda s: s.start)
        crossers = vertical if orientation == "horizontal" else horizontal
        for p, sign in ((items[0].start, -1), (items[-1].end, +1)):
            best = None
            for c in crossers:
                if not (c.start - reach <= axis <= c.end + reach):
                    continue
                d = (c.axis - p) * sign
                if lo_w <= d <= hi_w and (best is None or d < best[1]):
                    best = (c.axis, d)
            if best is None:
                continue
            g0, g1 = (p, best[0]) if sign > 0 else (best[0], p)
            out.append(dict(orientation=orientation, axis=float(axis),
                            start=float(g0), end=float(g1), width=float(g1 - g0),
                            real_start=(sign > 0), real_end=(sign < 0)))
    return out


def split_gaps_at_piers(gaps, segments, wall_width):
    """Split each collinear gap wherever a perpendicular wall (a real pier) crosses it.

    A wide opening that hosts two single-leaf doors back-to-back has the room partition
    reaching the wall as a pier in the middle. ``find_gaps`` cannot see it (the pier is on
    a perpendicular axis), so the gap reads as one wide opening and the single-arc test
    fails. Cutting the gap at every crossing perpendicular wall turns it into door-scale
    sub-gaps - and leaves a genuinely wide passage (piers only at its ends) wide open.
    """
    horizontal = [s for s in segments if s.orientation == "horizontal"]
    vertical = [s for s in segments if s.orientation == "vertical"]
    reach = wall_width * 1.5
    half = wall_width / 2.0
    out = []
    for g in gaps:
        crossers = vertical if g["orientation"] == "horizontal" else horizontal
        cuts = sorted(
            c.axis for c in crossers
            if g["start"] < c.axis < g["end"] and (c.start - reach) <= g["axis"] <= (c.end + reach)
        )
        bounds = [g["start"]] + cuts + [g["end"]]
        for i in range(len(bounds) - 1):
            first, last = i == 0, i + 1 == len(bounds) - 1
            lo = bounds[i] + (0.0 if first else half)          # trim pier half-thickness
            hi = bounds[i + 1] - (0.0 if last else half)
            if hi - lo > wall_width * 0.75:
                out.append(dict(orientation=g["orientation"], axis=g["axis"],
                                start=float(lo), end=float(hi), width=float(hi - lo),
                                real_start=bool(g.get("real_start", True)) and first,
                                real_end=bool(g.get("real_end", True)) and last))
    return out


# ---- gap ink trim (hidden jamb-stub recovery) ---------------------------------------

def trim_gap_ink(gap, dark: np.ndarray, wall_width: int):
    """Shrink a gap past contiguous wall-thick ink at either end; recover that ink.

    The directional trace drops jamb stubs that merge into a perpendicular wall's body
    (a 13px-thick wall survives the short directional opening, so the stub's component
    exceeds the stub-length ceiling). Those stubs are still plain wall ink in ``dark``:
    walk inward from each gap end while the cross-band ink fraction stays wall-like and
    emit the consumed run as a traced ``WallSegment`` (nothing invented - it is ink).
    Returns ``(trimmed_gap_or_None, recovered_stubs)``; ``None`` when no opening remains
    (the whole gap was ink, i.e. a wall piece the trace missed outright).
    """
    h, w = dark.shape
    half = wall_width // 2 + 1
    axis = int(round(gap["axis"]))
    lo = int(math.floor(gap["start"]))
    hi = int(math.ceil(gap["end"]))
    if gap["orientation"] == "horizontal":
        b0, b1 = max(0, axis - half), min(h, axis + half + 1)
        lo, hi = max(0, lo), min(w, hi)
        band = dark[b0:b1, lo:hi]
        frac = band.mean(axis=0) if band.size else np.zeros(0)
    else:
        b0, b1 = max(0, axis - half), min(w, axis + half + 1)
        lo, hi = max(0, lo), min(h, hi)
        band = dark[lo:hi, b0:b1]
        frac = band.mean(axis=1) if band.size else np.zeros(0)
    n = int(frac.shape[0])
    if n == 0:
        return None, []
    is_wall = frac >= PARAMS["gap_ink_frac"]
    left = 0
    while left < n and is_wall[left]:
        left += 1
    right = 0
    while right < n - left and is_wall[n - 1 - right]:
        right += 1

    min_run = wall_width * PARAMS["stub_recover_factor"]
    stubs = []
    if left >= min_run:
        stubs.append(WallSegment(gap["orientation"], gap["axis"],
                                 float(lo), float(lo + left), float(wall_width)))
    if right >= min_run:
        stubs.append(WallSegment(gap["orientation"], gap["axis"],
                                 float(hi - right), float(hi), float(wall_width)))
    new_start, new_end = float(lo + left), float(hi - right)
    if new_end - new_start <= wall_width * 0.75:
        return None, stubs
    trimmed = dict(gap, start=new_start, end=new_end, width=new_end - new_start)
    return trimmed, stubs


# ---- door-arc detector ------------------------------------------------------------

def _residual_mask(dark, wall_pixels):
    return ((dark > 0) & (wall_pixels == 0)).astype(np.uint8)


def _arc_score(pts_a, pts_p, r, band, ang_tol, ang_bins):
    """Quarter-turn arc quality of residual points around a hinge.

    ``pts_a`` = along-wall component (0 -> opposite jamb), ``pts_p`` = into-swing-room
    component, both relative to the hinge. A genuine door swing is a thin curve that
    (a) covers most of the [0,90] deg quarter (``coverage``), (b) is *continuous* - no
    long empty angular run (``max_empty_run``), and (c) is radially concentrated: the
    bulk of the residual ink in the quadrant lies ON the annulus, not scattered through
    it (``inlier_ratio``). The latter two reject big-radius sweeps fitted to text / a
    neighbouring door's arc, which cover angles but are radially diffuse and patchy.

    Returns ``(coverage, inlier_pixels, inlier_ratio, max_empty_run)``.
    """
    d = np.hypot(pts_a, pts_p)
    ang_all = np.degrees(np.arctan2(pts_p, pts_a))
    in_quad = (ang_all >= -ang_tol) & (ang_all <= 90.0 + ang_tol)
    near = in_quad & (d >= 0.45 * r) & (d <= 1.55 * r)
    quad_n = int(near.sum())
    if quad_n == 0:
        return 0.0, 0, 0.0, ang_bins
    in_band = near & (np.abs(d - r) <= band)
    inliers = int(in_band.sum())
    if inliers == 0:
        return 0.0, 0, 0.0, ang_bins
    inlier_ratio = inliers / quad_n
    ang = np.clip(ang_all[in_band], 0.0, 90.0 - 1e-6)
    bins = np.unique(np.floor(ang / (90.0 / ang_bins)).astype(int))
    coverage = len(bins) / ang_bins
    # longest run of consecutive empty bins across the quarter
    occupied = np.zeros(ang_bins, dtype=bool)
    occupied[bins] = True
    max_empty = run = 0
    for b in occupied:
        run = 0 if b else run + 1
        max_empty = max(max_empty, run)
    return coverage, inliers, inlier_ratio, max_empty


def detect_door_at_gap(resid_pts, gap, wall_width):
    """Return the best Door for a gap, or None. Tests both jambs x both swing sides."""
    w = gap["width"]
    if not (wall_width * PARAMS["door_gap_lo_factor"] <= w <= wall_width * PARAMS["door_gap_hi_factor"]):
        return None
    min_pixels = max(6, round(wall_width * PARAMS["min_pixels_factor"]))
    band_floor = 0.4 * wall_width
    orientation, axis = gap["orientation"], gap["axis"]
    e0, e1 = gap["start"], gap["end"]
    if orientation == "horizontal":
        jambs = [((e0, axis), np.array([1.0, 0.0])), ((e1, axis), np.array([-1.0, 0.0]))]
        perp_dirs = [np.array([0.0, -1.0]), np.array([0.0, 1.0])]
    else:
        jambs = [((axis, e0), np.array([0.0, 1.0])), ((axis, e1), np.array([0.0, -1.0]))]
        perp_dirs = [np.array([-1.0, 0.0]), np.array([1.0, 0.0])]

    r_hi = w * max(PARAMS["radius_scales"])
    # window around the gap to limit candidate residual points
    if orientation == "horizontal":
        gx0, gx1 = min(e0, e1), max(e0, e1)
        win = ((resid_pts[:, 0] >= gx0 - r_hi - 4) & (resid_pts[:, 0] <= gx1 + r_hi + 4)
               & (np.abs(resid_pts[:, 1] - axis) <= r_hi + 4))
    else:
        gy0, gy1 = min(e0, e1), max(e0, e1)
        win = ((resid_pts[:, 1] >= gy0 - r_hi - 4) & (resid_pts[:, 1] <= gy1 + r_hi + 4)
               & (np.abs(resid_pts[:, 0] - axis) <= r_hi + 4))
    local = resid_pts[win]
    if local.shape[0] < min_pixels:
        return None

    best = None  # (score, cov, ratio, empty, npix, hinge, perp, r)
    for (hinge, along_unit) in jambs:
        rel = local - np.array(hinge)
        a_all = rel @ along_unit
        for perp_unit in perp_dirs:
            p_all = rel @ perp_unit
            for scale in PARAMS["radius_scales"]:
                r = w * scale
                band = max(band_floor, PARAMS["band_factor"] * r)
                cov, npix, ratio, empty = _arc_score(a_all, p_all, r, band,
                                                     PARAMS["ang_tol_deg"], PARAMS["ang_bins"])
                if npix < min_pixels:
                    continue
                # a real swing must clear every threshold; rank survivors by cov*ratio
                if (cov < PARAMS["coverage_thresh"] or ratio < PARAMS["inlier_ratio_thresh"]
                        or empty > PARAMS["max_empty_run"]):
                    continue
                score = cov * ratio
                if best is None or score > best[0]:
                    best = (score, cov, ratio, empty, npix, hinge, perp_unit, r)
    if best is None:
        return None
    _, cov, ratio, empty, npix, hinge, perp_unit, r = best
    return Door(orientation, axis, e0, e1, (float(hinge[0]), float(hinge[1])),
                (float(perp_unit[0]), float(perp_unit[1])), float(r), float(cov), int(npix),
                float(ratio))


def detect_doors(segments, dark, wall_pixels, wall_width):
    """Return ``(raw_gaps, sub_gaps, doors, jamb_stubs)``.

    Doors are confirmed on pier-split sub-gaps of collinear + terminal gaps. A sub-gap
    whose BOTH ends are pier cuts has no traced wall jamb at all - it is a rounding
    coincidence of far-apart walls sharing an axis (the source of phantom doors bridged
    in mid-room on a neighbouring door's arc ink) - and is rejected outright. Each
    surviving gap is first ink-trimmed so hidden corner/jamb stubs become walls and the
    arc test sees the true opening width.
    """
    resid = _residual_mask(dark, wall_pixels)
    ys, xs = np.where(resid > 0)
    resid_pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    raw_gaps = find_gaps(segments, wall_width) + add_terminal_gaps(segments, wall_width)
    sub_gaps = split_gaps_at_piers(raw_gaps, segments, wall_width)
    doors, jamb_stubs = [], []
    for gap in sub_gaps:
        if not (gap.get("real_start", True) or gap.get("real_end", True)):
            continue
        trimmed, stubs = trim_gap_ink(gap, dark, wall_width)
        jamb_stubs.extend(stubs)
        if trimmed is None:
            continue
        door = detect_door_at_gap(resid_pts, trimmed, wall_width)
        if door is not None:
            doors.append(door)
    return raw_gaps, sub_gaps, _dedupe_doors(doors, wall_width), jamb_stubs


def _dedupe_doors(doors: list[Door], wall_width: int) -> list[Door]:
    """Collapse twin detections of ONE physical opening.

    Axis rounding can leave the same opening on two nearly-identical collinear axes
    (e.g. x=957 and x=963); both then confirm on the same arc ink and bridge as two
    parallel segments a few px apart (rendered as a double line). Two doors within
    ~1 wall thickness in axis whose gap spans overlap are one door - keep the better
    arc (coverage * inlier_ratio).
    """
    kept: list[Door] = []
    for d in sorted(doors, key=lambda d: d.coverage * d.inlier_ratio, reverse=True):
        dup = False
        for k in kept:
            if k.orientation != d.orientation or abs(k.axis - d.axis) > wall_width * 1.2:
                continue
            lo, hi = max(k.gap_start, d.gap_start), min(k.gap_end, d.gap_end)
            if hi - lo > 0.5 * min(k.gap_end - k.gap_start, d.gap_end - d.gap_start):
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept


# ---- bridging at confirmed doors --------------------------------------------------

def bridge_at_doors(segments, doors, wall_width, extra=()):
    """Walls + a wall segment filling each confirmed door gap (continuous at doors only).

    Non-door gaps get no fill, so they stay open. Adding the bridge as its own segment and
    re-merging keeps the faithful walls untouched while closing exactly the door openings.
    ``extra`` carries ink-recovered jamb stubs from ``trim_gap_ink`` (traced, not invented).
    """
    bridges = [WallSegment(d.orientation, d.axis, d.gap_start, d.gap_end, float(wall_width))
               for d in doors]
    clustered = _cluster_axes(list(segments) + list(extra) + bridges,
                              tolerance=max(1.0, wall_width * 0.5))
    merged = _merge_overlaps(clustered, tolerance=max(1.0, wall_width * 0.5))
    return merged, len(bridges)


# ---- post-processing (after door bridging) ----------------------------------------

def _junction_axes(seg: WallSegment, others: list[WallSegment], tol: float) -> list[float]:
    """Axes of perpendicular walls that actually cross/touch ``seg`` (within tol)."""
    axes = []
    for o in others:
        if o.orientation == seg.orientation:
            continue
        if seg.start - tol <= o.axis <= seg.end + tol and o.start - tol <= seg.axis <= o.end + tol:
            axes.append(float(o.axis))
    return sorted(axes)


def _endpoint_touched(x: float, y: float, s: WallSegment, segs: list[WallSegment], eps: float) -> bool:
    """True if point (x, y) lies on some OTHER segment's line (within eps)."""
    for o in segs:
        if o is s:
            continue
        if o.orientation == "horizontal":
            if abs(y - o.axis) <= eps and o.start - eps <= x <= o.end + eps:
                return True
        else:
            if abs(x - o.axis) <= eps and o.start - eps <= y <= o.end + eps:
                return True
    return False


def _ink_covered(dark: np.ndarray, orientation: str, axis: float, a: float, b: float,
                 wall_width: int) -> bool:
    """True when the axis-aligned span [a, b] is continuously covered by wall ink."""
    if b - a < 1.0:
        return True
    h, w = dark.shape
    half = wall_width // 2 + 1
    ax = int(round(axis))
    lo, hi = int(math.floor(a)), int(math.ceil(b))
    if orientation == "horizontal":
        band = dark[max(0, ax - half):min(h, ax + half + 1), max(0, lo):min(w, hi)]
        frac = band.mean(axis=0) if band.size else np.zeros(0)
    else:
        band = dark[max(0, lo):min(h, hi), max(0, ax - half):min(w, ax + half + 1)]
        frac = band.mean(axis=1) if band.size else np.zeros(0)
    if frac.shape[0] == 0:
        return False
    return float((frac >= PARAMS["gap_ink_frac"]).mean()) >= PARAMS["corner_ink_cover"]


def _absorb_jamb_posts(segs: list[WallSegment], wall_width: int) -> tuple[list[WallSegment], int]:
    """Drop short door-frame posts drawn parallel against a longer wall.

    The source draws a small frame post (~1 wall_width long, thinner than a wall) at some
    jambs, offset ~1 wall_width from the main wall line. It gets traced as a tiny separate
    segment, pier-splitting then cuts gaps at ITS axis, and door bridges/junction snaps
    stop on it - a topological island that blocks the L-corner snap while connecting to
    nothing (parallel lines never intersect). A post = short (<= 2 ww) segment whose span
    is covered by a longer PARALLEL wall within [0.6, 1.5] ww axis offset; absorbing it
    lets the bridge end reach the real wall via the ink-gated corner snap.
    """
    tol = wall_width * 1.2
    max_len = wall_width * 2.0
    off_lo, off_hi = wall_width * 0.6, wall_width * 1.5
    keep, dropped = [], 0
    for s in segs:
        absorbed = False
        if (s.end - s.start) <= max_len:
            for o in segs:
                if o is s or o.orientation != s.orientation:
                    continue
                if (o.end - o.start) <= max_len:
                    continue
                off = abs(o.axis - s.axis)
                if off_lo <= off <= off_hi and o.start - tol <= s.start and s.end <= o.end + tol:
                    absorbed = True
                    break
        if absorbed:
            dropped += 1
        else:
            keep.append(s)
    return keep, dropped


def _merge_parallel_faces(segs: list[WallSegment], wall_width: int) -> tuple[list[WallSegment], int]:
    """Merge a wall's re-traced FACE line into the wall proper.

    The trace sometimes emits ONE physical wall as two nearly-parallel segments —
    the centre line plus a face line offset by up to ~1 wall thickness (e.g. axes
    6–12px apart at ww=11). ``_cluster_axes`` (tol 0.5*ww) cannot merge them and
    the covered-span whisker rule misses the part dangling past the host's end.
    Any two same-orientation segments within ``1.3*ww`` in axis whose spans
    overlap by at least ``0.5*ww`` are two tracings of one wall — real parallel
    partitions are at least a room apart, never 1 wall thickness. Keep the longer
    segment's axis and take the union span (the dangling extension was traced
    from real ink on the same wall). Iterates longest-first so triples collapse.
    """
    off_max = wall_width * 1.3
    min_overlap = wall_width * 0.5
    ordered = sorted(segs, key=lambda s: s.end - s.start, reverse=True)
    out: list[WallSegment] = []
    merged = 0
    for s in ordered:
        host = None
        for o in out:
            if o.orientation != s.orientation:
                continue
            d = abs(o.axis - s.axis)
            if d < 1e-6 or d > off_max:
                continue
            if min(s.end, o.end) - max(s.start, o.start) >= min_overlap:
                host = o
                break
        if host is not None:
            host.start = min(host.start, s.start)
            host.end = max(host.end, s.end)
            host.thickness = max(host.thickness, s.thickness)
            merged += 1
        else:
            out.append(WallSegment(s.orientation, s.axis, s.start, s.end, s.thickness))
    return out, merged


def postprocess_walls(segments: list[WallSegment], wall_width: int,
                      dark: np.ndarray | None = None) -> tuple[list[WallSegment], dict]:
    """Clean the bridged wall graph so polygonize sees closed rooms and no whiskers.

    Deterministic steps (geometry + source-ink checks only, nothing invented):
    0. ABSORB - drop parallel door-frame posts (_absorb_jamb_posts) so wall/bridge
                ends are free to snap onto the REAL wall instead of an island.
    1. SNAP   - extend/shrink endpoints onto crossing perpendicular walls
                (_snap_junctions). Closes the corner pinholes - a jamb ending a few
                px short of the main wall - through which a room leaks into the
                corridor during polygonize.
    2. CORNER - a still-FREE endpoint is extended onto the farthest perpendicular
                wall line within ``corner_snap_reach * ww``, but ONLY when the
                extension path is continuously covered by wall ink in the source
                (solid L-corner ink; a real opening is white and is never sealed).
                Closes the corner holes _snap_junctions cannot see because neither
                wall's span reaches the other's axis (both stop short at an
                L-corner, e.g. where a door bridge ends at a jamb-post pier).
    3. TRIM   - an endpoint overhanging its outermost perpendicular junction by
                less than ``2 * wall_width`` is cut back to that junction (the small
                perpendicular ticks poking out of a main wall).
    4. DROP   - segments shorter than ``2.5 * wall_width`` that still have a free
                end after snapping (dangling whiskers) are removed; they can never
                close a ring, they are pure visual noise.
    """
    tol = wall_width * 1.2
    overhang = wall_width * 2.0
    segs = [WallSegment(s.orientation, s.axis, s.start, s.end, s.thickness) for s in segments]
    segs, posts_absorbed = _absorb_jamb_posts(segs, wall_width)
    segs, faces_merged = _merge_parallel_faces(segs, wall_width)
    segs = _snap_junctions(segs, tolerance=tol)

    corner_snapped = 0
    if dark is not None:
        reach = wall_width * PARAMS["corner_snap_reach"]
        for _ in range(2):
            moved = False
            for s in segs:
                for which in (0, 1):
                    p = s.start if which == 0 else s.end
                    x, y = (p, s.axis) if s.orientation == "horizontal" else (s.axis, p)
                    if _endpoint_touched(x, y, s, segs, eps=1.0):
                        continue
                    best = None
                    for o in segs:
                        if o.orientation == s.orientation:
                            continue
                        if not (o.start - tol <= s.axis <= o.end + tol):
                            continue
                        d = p - o.axis if which == 0 else o.axis - p
                        if not (0.0 < d <= reach):
                            continue
                        a, b = (o.axis, p) if which == 0 else (p, o.axis)
                        if not _ink_covered(dark, s.orientation, s.axis, a, b, wall_width):
                            continue
                        if best is None or d > best[1]:      # farthest ink-backed wall line
                            best = (o.axis, d)
                    if best is not None:
                        if which == 0:
                            s.start = best[0]
                        else:
                            s.end = best[0]
                        corner_snapped += 1
                        moved = True
            if not moved:
                break

    trimmed = 0
    for s in segs:
        axes = _junction_axes(s, segs, tol)
        if not axes:
            continue
        lo, hi = axes[0], axes[-1]
        if s.start < lo and (lo - s.start) <= overhang:
            s.start = lo
            trimmed += 1
        if s.end > hi and (s.end - hi) <= overhang:
            s.end = hi
            trimmed += 1

    def has_free_end(s: WallSegment, all_segs: list[WallSegment]) -> bool:
        pts = ([(s.start, s.axis), (s.end, s.axis)] if s.orientation == "horizontal"
               else [(s.axis, s.start), (s.axis, s.end)])
        for (x, y) in pts:
            touched = False
            for o in all_segs:
                if o is s:
                    continue
                if o.orientation == "horizontal":
                    if abs(y - o.axis) <= tol and o.start - tol <= x <= o.end + tol:
                        touched = True
                        break
                else:
                    if abs(x - o.axis) <= tol and o.start - tol <= y <= o.end + tol:
                        touched = True
                        break
            if not touched:
                return True
        return False

    def is_face_line(s: WallSegment, all_segs: list[WallSegment]) -> bool:
        """A dangling re-trace of a thick wall's FACE: its whole span is covered by a
        longer parallel wall within ~1 wall thickness. Real partitions are never that
        close to another parallel wall, so this only removes redundant ink."""
        for o in all_segs:
            if o is s or o.orientation != s.orientation:
                continue
            if (o.end - o.start) <= (s.end - s.start):
                continue
            if abs(o.axis - s.axis) <= wall_width * 1.3 \
                    and o.start - tol <= s.start and s.end <= o.end + tol:
                return True
        return False

    keep, dropped, face_lines = [], 0, 0
    for s in segs:
        if has_free_end(s, segs):
            if (s.end - s.start) <= wall_width * 2.5:
                dropped += 1
                continue
            if is_face_line(s, segs):
                face_lines += 1
                continue
        keep.append(s)
    stats = dict(snapped_tol=tol, posts_absorbed=posts_absorbed, faces_merged=faces_merged,
                 corners_snapped=corner_snapped,
                 endpoints_trimmed=trimmed, whiskers_dropped=dropped, face_lines_dropped=face_lines,
                 segments_in=len(segments), segments_out=len(keep))
    return keep, stats


def polygonize_rooms(segments: list[WallSegment], image_shape) -> list:
    """Room polygons from the wall graph (closed rings only; open edges vanish)."""
    from shapely.ops import polygonize, unary_union

    net = unary_union([s.line() for s in segments])
    min_area = image_shape[0] * image_shape[1] * 0.0004
    return [p for p in polygonize(net) if p.area >= min_area]


# ---- rendering --------------------------------------------------------------------

def _draw_walls(canvas, segments, color, thickness):
    for s in segments:
        if s.orientation == "horizontal":
            p0 = (round(s.start), round(s.axis))
            p1 = (round(s.end), round(s.axis))
        else:
            p0 = (round(s.axis), round(s.start))
            p1 = (round(s.axis), round(s.end))
        cv2.line(canvas, p0, p1, color, thickness)


def _draw_door(canvas, door: Door, color):
    hinge = np.array(door.hinge)
    # opposite jamb = the gap endpoint that is not the hinge; along points hinge->opposite
    if door.orientation == "horizontal":
        opposite = np.array([door.gap_end if abs(hinge[0] - door.gap_start) < abs(hinge[0] - door.gap_end)
                             else door.gap_start, door.axis])
    else:
        opposite = np.array([door.axis, door.gap_end if abs(hinge[1] - door.gap_start) < abs(hinge[1] - door.gap_end)
                             else door.gap_start])
    along_vec = opposite - hinge
    norm = np.hypot(*along_vec) or 1.0
    along = along_vec / norm
    perp = np.array(door.swing)
    # arc from along (0) to perp (90)
    pts = []
    for t in np.linspace(0, math.pi / 2, 24):
        v = math.cos(t) * along + math.sin(t) * perp
        pts.append(hinge + door.radius * v)
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts], False, color, 1, cv2.LINE_AA)
    # leaf
    leaf_end = hinge + door.radius * perp
    cv2.line(canvas, tuple(hinge.astype(int)), tuple(leaf_end.astype(int)), color, 1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(hinge.astype(int)), 3, color, -1)


METHOD_NOTES = (
    "Faithful trace = _merge_overlaps(_cluster_axes(_raw_segments + short_stub_pass)); "
    "NO size-threshold gap bridging. Collinear gaps + terminal gaps (wall end -> crossing "
    "perpendicular wall, for doors drawn against a room corner) are split at perpendicular "
    "wall crossings (real piers) into door-scale sub-gaps; sub-gaps whose both ends are "
    "pier cuts (no traced jamb) are rejected - kills phantom doors on axis-rounding "
    "coincidences. Each gap is ink-trimmed first: contiguous wall-thick ink at a gap end "
    "(jamb/corner stubs the directional trace merged into a perpendicular wall) is "
    "recovered as traced wall and removed from the opening. A gap becomes a Door only if a "
    "genuine quarter-circle swing arc straddles it: angular coverage >= {cov} over {bins} "
    "bins, radial inlier-ratio >= {ratio}, max empty angular run <= {empty}, on an annulus "
    "centred at a jamb with radius ~= opening width. Walls bridge ONLY at confirmed doors. "
    "postprocess_walls: junction snap, ink-gated L-corner snap (free end -> perpendicular "
    "wall line within {corner}*ww, only over solid source ink), overhang trim, whisker "
    "drop. Short-stub min length = {stub}*wall_width. All thresholds are wall_width-"
    "relative, so the same params handle 1K (~6px) and 2K (~12px)."
)


def process(src_path: str, out_dir: str) -> dict:
    """Trace walls + detect doors for one image; write the 4 artifacts; return metrics."""
    gray = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    src_bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if gray is None:
        raise SystemExit(f"could not read source: {src_path}")
    os.makedirs(out_dir, exist_ok=True)

    walls, walls_no_stub, ww, dark, wall_pixels, count_raw, count_stub = trace_walls(gray)
    raw_gaps, sub_gaps, doors, jamb_stubs = detect_doors(walls, dark, wall_pixels, ww)
    bridged, bridges = bridge_at_doors(walls, doors, ww, extra=jamb_stubs)

    line_w = max(2, round(ww * 0.4))
    # walls overlay (faithful, red) - verify no spurious corridor/toilet walls
    walls_overlay = src_bgr.copy()
    _draw_walls(walls_overlay, walls, (0, 0, 255), line_w)
    cv2.imwrite(os.path.join(out_dir, "walls_overlay.png"), walls_overlay)

    # doors overlay (faithful walls grey + detected swing arcs orange)
    doors_overlay = src_bgr.copy()
    _draw_walls(doors_overlay, walls, (150, 150, 150), line_w)
    for d in doors:
        _draw_door(doors_overlay, d, (0, 140, 255))
    cv2.imwrite(os.path.join(out_dir, "doors_overlay.png"), doors_overlay)

    # recon: walls bridged ONLY at confirmed doors (black), doors shown as blue markers
    recon = np.full((*gray.shape, 3), 255, np.uint8)
    _draw_walls(recon, bridged, (0, 0, 0), max(2, ww // 2))
    for d in doors:
        cv2.circle(recon, tuple(map(round, d.center)), max(3, ww), (255, 60, 60), 1)
        _draw_door(recon, d, (255, 60, 60))
    cv2.imwrite(os.path.join(out_dir, "recon.png"), recon)

    # recon_post: snap pinholes closed + trim/drop whiskers, then re-render + rooms
    post, post_stats = postprocess_walls(bridged, ww, dark)
    recon_post = np.full((*gray.shape, 3), 255, np.uint8)
    _draw_walls(recon_post, post, (0, 0, 0), max(2, ww // 2))
    for d in doors:
        cv2.circle(recon_post, tuple(map(round, d.center)), max(3, ww), (255, 60, 60), 1)
        _draw_door(recon_post, d, (255, 60, 60))
    cv2.imwrite(os.path.join(out_dir, "recon_post.png"), recon_post)

    rooms = polygonize_rooms(post, gray.shape)
    rooms_img = np.full((*gray.shape, 3), 255, np.uint8)
    for p in rooms:
        ext = np.array(p.exterior.coords, np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(rooms_img, [ext], (200, 200, 200))
    _draw_walls(rooms_img, post, (0, 0, 0), max(2, ww // 2))
    cv2.imwrite(os.path.join(out_dir, "rooms_uniform.png"), rooms_img)

    room_max_area = gray.shape[0] * gray.shape[1] * PARAMS["room_max_area_frac"]
    true_room_closure = sum(1 for p in rooms if p.area <= room_max_area)
    metrics = dict(
        source=src_path,
        image_shape=[int(gray.shape[1]), int(gray.shape[0])],
        wall_width=int(ww),
        faithful_wall_count=len(walls),
        faithful_wall_count_no_stub=len(walls_no_stub),
        raw_segment_count=count_raw,
        short_stub_count=count_stub,
        gap_count=len(raw_gaps),
        sub_gap_count=len(sub_gaps),
        jamb_stub_count=len(jamb_stubs),
        doors_detected=len(doors),
        bridges_made=bridges,
        postprocess=post_stats,
        rooms_closed=len(rooms),
        true_room_closure=true_room_closure,
        params=PARAMS,
        method_notes=METHOD_NOTES.format(
            cov=PARAMS["coverage_thresh"], bins=PARAMS["ang_bins"],
            ratio=PARAMS["inlier_ratio_thresh"], empty=PARAMS["max_empty_run"],
            stub=PARAMS["stub_min_len_factor"], corner=PARAMS["corner_snap_reach"],
        ),
    )
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    _save_verification_crops(out_dir, walls_overlay, doors_overlay)
    _save_leak_diag(out_dir, src_bgr, recon_post, rooms_img)
    return metrics


def _save_verification_crops(out_dir, walls_overlay, doors_overlay):
    """Zoomed crops (resolution-independent regions) for visual precision/recall checks."""
    crops = os.path.join(out_dir, "crops")
    os.makedirs(crops, exist_ok=True)
    h, w = walls_overlay.shape[:2]
    regions = {  # fractions of the image: a patient-room+toilet+jamb wing, and the central cross
        "wing": (0.03, 0.30, 0.00, 0.42),
        "cross": (0.28, 0.74, 0.31, 0.53),
    }
    for name, (fy0, fy1, fx0, fx1) in regions.items():
        y0, y1, x0, x1 = round(fy0 * h), round(fy1 * h), round(fx0 * w), round(fx1 * w)
        for src, tag in ((walls_overlay, "walls"), (doors_overlay, "doors")):
            crop = cv2.resize(src[y0:y1, x0:x1], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(crops, f"{tag}_{name}.png"), crop)


# Historic leak regions (image fractions x0,y0,x1,y1) - each held an unclosed room in an
# earlier iteration; the diag crops let one eyeball source vs recon_post vs rooms per run.
DIAG_REGIONS = {
    "A_exam2_left": (0.109, 0.280, 0.283, 0.495),
    "B_exam5_toilets": (0.632, 0.208, 0.770, 0.495),
    "C_nurse4_office7": (0.640, 0.514, 0.887, 0.801),
    "D_office1_bl": (0.109, 0.658, 0.211, 0.801),
    "E_toilet4_top": (0.298, 0.208, 0.367, 0.352),
    "F_bottom_left": (0.051, 0.807, 0.171, 1.000),
}


def _save_leak_diag(out_dir, src_bgr, recon_post, rooms_img):
    """Side-by-side source | recon_post | rooms crops for the historic leak regions."""
    diag = os.path.join(out_dir, "diag")
    os.makedirs(diag, exist_ok=True)
    h, w = src_bgr.shape[:2]
    for name, (fx0, fy0, fx1, fy1) in DIAG_REGIONS.items():
        x0, y0, x1, y1 = round(fx0 * w), round(fy0 * h), round(fx1 * w), round(fy1 * h)
        panels = []
        for img in (src_bgr, recon_post, rooms_img):
            crop = cv2.copyMakeBorder(img[y0:y1, x0:x1], 2, 2, 2, 2,
                                      cv2.BORDER_CONSTANT, value=(0, 200, 0))
            panels.append(crop)
        row = np.hstack(panels)
        scale = min(2.0, 2400.0 / row.shape[1])
        row = cv2.resize(row, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(diag, f"{name}.png"), row)


# 2K validation source (same plan, double resolution) - confirms wall_width-relativity.
SRC_2K = (
    r"D:\Github\GSDiff\hfagent\out\eval\20260629-195311-2k-linework"
    r"\hospital-tower-floor.real.png"
)


def main():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(r"D:\Github\GSDiff\hfagent\out\eval", f"{ts}-linework-trace-doors")
    # 2K only — small features (toilet doors, jambs) resolve far better at 2K.
    summary = {"2k": process(SRC_2K, out_dir)}
    brief = {res: {k: m[k] for k in ("wall_width", "faithful_wall_count",
                                     "short_stub_count", "jamb_stub_count",
                                     "doors_detected", "bridges_made",
                                     "rooms_closed", "true_room_closure")}
             for res, m in summary.items()}
    print(json.dumps(brief, indent=2))
    print("out_dir", out_dir)
    return out_dir, summary


if __name__ == "__main__":
    main()
