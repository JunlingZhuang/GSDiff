# -*- coding: utf-8 -*-
"""Faithful wall trace + arc-confirmed door detection: the linework structure engine.

``trace_linework(png_bytes)`` turns a realistic line-plan drawing into a geometry-only
``Plan`` (untyped rooms, px units) plus a ``RoomGraph`` whose door edges name the two
plan rooms flanking each detected door. No OCR, no room typing, no model calls:

1. WALLS  — trace exactly the walls that exist: ``_merge_overlaps(_cluster_axes(
   _raw_segments(...)))`` with NO gap bridging, plus a short-stub pass so legitimately
   short walls (door jambs, T-junction piers) below the ``_raw_segments`` length floor
   are kept. Nothing is invented. Acceptance is THICKNESS-AGNOSTIC: a run is a wall
   from a small absolute cross floor upward (clears arc/leaf strokes) as long as its
   ink anchors to the image-spanning wall network (drops floating label text), because
   image models draw thickness inconsistently — fat perimeter bands over thin
   partitions in the same drawing. Each run keeps its own measured thickness.
2. DOORS  — a wall gap becomes a door ONLY if a real quarter-circle swing arc (radius
   ~= the opening width for a single leaf, ~= half of it for a double door, ~a third
   for a small leaf in an oversized mouth, centred on one jamb) straddles it.
   Candidate gaps come from four sources: collinear gaps + terminal gaps (wall end ->
   crossing perpendicular wall), both split at crossing perpendicular walls (real
   piers) into door-scale sub-gaps, SLOT gaps (white slots cut into a band whose wall
   line never breaks — the door-as-window-symbol style), and JOGGED gaps (the two
   jambs sit on slightly offset axes because the wall thickness changes across the
   opening; their thickness bands still overlap). A sub-gap with no traced jamb at
   either end is rejected outright (kills phantom doors on axis-rounding
   coincidences), and a wide gap flanked only by tiny wall fragments is a face-line
   artifact, never a door. Past the single-leaf width cap only the regimes a drawn
   door can physically produce confirm (undersized leaf slightly past it, true
   half-gap double door further out). Each gap is ink-trimmed
   and ink-split first so hidden jamb/corner stubs and untraced ornament piers become
   traced wall and the arc test sees the true opening(s). The arc test demands
   angular quarter-turn coverage, radial inlier concentration and continuity, and arc
   ink is exclusive to one door, so stray ink, label text and neighbouring arcs fail.
3. BRIDGE — walls are made continuous ONLY across confirmed doors (a jogged door also
   gets its tiny perpendicular connectors) and across WINDOW breaks — band breaks
   whose cross band keeps continuous wall-anchored ink (hollow window faces/sills):
   drawn glazing is wall for room topology. Every other gap stays open (real passage
   / corridor connection); nothing is bridged without ink or arc evidence.
4. POST   — ``postprocess_walls``: absorb parallel door-frame posts, merge re-traced
   wall faces, junction snap, ink-gated L-corner snap (never seals a real opening),
   overhang trim, whisker drop. Then shapely polygonizes the closed rooms.

``wall_width`` — the scale anchor for door/gap/stub/postprocess factors — is the MEDIAN
of the accepted runs' own cross thickness (robust to fat exteriors), never an
acceptance gate; wall ACCEPTANCE floors are absolute fractions of the image dimension,
so the same parameters handle 1K renders, 2K renders and mixed-thickness drawings.
A 3x3 solidify close fuses hairline-split wall faces before extraction.

Fixed artifacts (returned as PNG bytes, written by the pipeline into the work dir):
    walls_overlay.png    faithful traced walls in red on the source drawing
    doors_overlay.png    faithful walls grey + every detected swing arc/leaf/hinge
    recon.png            bridged walls (pre-postprocess) + door arc markers
    recon_post.png       post-processed wall graph + door arc markers
    rooms_colorful.png   each closed room filled with a distinct deterministic colour,
                         walls black on top, doors as white gaps

Offline debug harness (no pipeline, no model calls):

    python -m hfagent.tools.linework_tracer <real_plan.png> [--out DIR]
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from shapely.geometry import Point, Polygon as ShapelyPolygon
from shapely.ops import polygonize, unary_union

from hfagent.schema.plan import Plan, Room
from hfagent.schema.roomgraph import Door as DoorEdge, RoomGraph, RoomNode
from hfagent.tools.text_mask import detect_label_quads
from hfagent.tools.wall_graph import (
    WallSegment,
    _cluster_axes,
    _directional_masks,
    _estimate_stroke_thickness,
    _ink_anchor,
    _is_anchored,
    _median_wall_width,
    _merge_overlaps,
    _raw_segments,
    _snap_junctions,
    _solidify,
)

# ---- tunable parameters (kept in one place so they can be swept) -------------------
PARAMS = dict(
    # wall acceptance: ABSOLUTE floors, never wall_width-relative — thickness varies
    # wildly within one drawing (thin partitions under fat perimeter bands)
    min_cross_frac=0.0033,        # thickness floor, * min image dim (clears arc/leaf strokes)
    anchor_reach_frac=0.10,       # min ink-component reach, * min image dim (drops label text)
    anchor_area_frac=0.10,        # min ink area vs the largest component (drops merged words)
    max_symbol_density=0.18,      # max bbox ink fill for detached door-symbol components
    # short-stub trace: keep wall runs down to this length (in wall_width)
    stub_min_len_factor=1.0,      # min length of a kept short stub, * wall_width
    stub_min_density=0.45,        # min fill density of the component bbox (== _raw_segments)
    # text-bar drop: a run whose PRE-solidify ink breaks repeatedly is a label word,
    # not a wall (a drawn wall is one continuous stroke; letters break at every gap)
    text_bar_min_breaks=3,        # min empty runs across the span to read as text
    text_bar_max_coverage=0.90,   # ... and raw ink must cover no more than this
    # gap -> door search  (door-scale: a swing arc fits a single-leaf opening)
    door_gap_lo_factor=1.6,       # opening must be wider than this * wall_width
    door_gap_hi_factor=10.0,      # ... and narrower than this * wall_width (else passage)
    # past the single-leaf cap a door can still confirm, but only on regimes a
    # drawn door can physically produce out there: an undersized leaf slightly
    # past the cap, or a true double door (two mirrored half-gap leaves) further
    undersized_gap_hi_factor=11.0,
    double_gap_hi_factor=14.0,
    # swing radius as a fraction of the opening: ~1.0 = single leaf fills the opening,
    # ~0.5 = double door (two mirrored leaves), 0.7-0.8 = undersized leaf in a wide
    # mouth; 0.3-0.35 = a small drawn door in an OVERSIZED mouth - allowed only just
    # past the single-leaf cap, where no in-cap door competes for the same arc ink
    # (at door scale a small-radius regime hijacks neighbouring doors' arcs)
    radius_scales=(0.3, 0.35, 0.45, 0.5, 0.55, 0.7, 0.8, 0.9, 1.0, 1.1),
    # jogged collinear gaps: both walls end short of each other with a small axis
    # jog (their thickness bands still overlap - one drawn wall line)
    jog_axis_frac=1.2,            # max axis offset, * wall_width
    # window-band break fill: a gap whose cross band keeps CONTINUOUS ink presence
    # (hollow window faces/sills) is drawn wall, not an opening
    window_presence=0.85,         # min fraction of gap columns with any band ink
    window_fill_hi_factor=20.0,   # max filled break, * wall_width
    band_factor=0.16,             # annulus half-width = max(0.4*ww, band_factor * r)
    ang_tol_deg=14.0,             # angular slack outside the [0,90] quarter
    ang_bins=18,                  # bins across the 90 deg quarter (5 deg each)
    coverage_thresh=0.60,         # min fraction of quarter-turn bins occupied
    inlier_ratio_thresh=0.45,     # min fraction of quadrant ink lying ON the annulus
    max_empty_run=4,              # reject arcs with a > this consecutive-empty-bin gap
    arc_claim_max_overlap=0.75,   # drop a door whose arc ink is mostly already claimed
    min_pixels_factor=1.2,        # min residual pixels supporting the arc = factor * ww
    # gap ink trim / jamb-stub recovery (hidden corner stubs inside a gap)
    gap_ink_frac=0.50,            # cross-band ink fraction that counts as wall ink
    stub_recover_factor=0.5,      # min recovered ink run, * wall_width
    # postprocess corner snap (L-corner pinholes; ink-gated, never crosses an opening)
    corner_snap_reach=2.5,        # max free-end extension onto a perpendicular, * ww
    corner_ink_cover=0.80,        # min fraction of ink-covered positions on the extension
    # stub thickness must match its aligned wall's (a jamb IS that wall; a door leaf
    # drawn as a solid bar is thinner than the wall whose axis it happens to align with)
    stub_thickness_match=(0.55, 1.8),
    # diagnostics: polygons above this image fraction are corridor-scale, not room-scale
    room_max_area_frac=0.02,
)

# door-side probe distance from the wall axis, * wall_width (clears the wall band)
_PROBE_OFFSET_FACTOR = 1.5


@dataclass
class TracedDoor:
    """A physical door confirmed by its swing arc (all coordinates in px)."""
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
    # true wall axes at the gap ends when they JOG off the door axis (see
    # ``add_jogged_gaps``); the bridge adds perpendicular connectors for them
    start_axis: float | None = None
    end_axis: float | None = None

    @property
    def center(self) -> tuple[float, float]:
        mid = (self.gap_start + self.gap_end) / 2.0
        return (mid, self.axis) if self.orientation == "horizontal" else (self.axis, mid)


@dataclass
class LineworkTrace:
    """Result of ``trace_linework``: geometry-only plan + doors + fixed artifacts."""
    plan: Plan                      # untyped rooms (type "unknown", ids r1..rN, px units)
    room_graph: RoomGraph           # door edges between flanking room ids / "exterior"
    diagnostics: dict               # trace metrics (wall_width, counts, postprocess stats)
    artifacts: dict[str, bytes]     # fixed artifact filename -> PNG bytes


# ---- faithful wall trace ----------------------------------------------------------

def _short_stub_segments(
    dark: np.ndarray,
    hm: np.ndarray,
    vm: np.ndarray,
    wall_axes: dict[str, list[tuple[float, float]]],
    wall_width: int,
    min_cross: int,
    anchor: tuple[np.ndarray, np.ndarray],
) -> list[WallSegment]:
    """Recover short wall stubs (door jambs, T-junction piers) the long trace drops.

    ``_directional_masks`` opens with a long kernel, so it erases any wall piece
    shorter than that - exactly the door jambs (the short wall between an opening and a
    corner) and the short piers at T-junctions. We re-open ``dark`` with a *short*
    directional kernel, drop what the long trace already owns, and keep only short runs
    that (a) clear the same absolute thickness floor as the long trace (kills arc/leaf
    strokes), (b) are anchored to the image-spanning wall network (kills label text),
    and (c) line up with a traced wall axis of the same orientation AND match that
    wall's own thickness. A jamb is a piece of the wall it aligns with, so their
    thickness agrees; a door LEAF drawn as a solid bar (some styles do) hangs at
    whatever axis a nearby partition happens to share but is leaf-thin relative to it,
    so the thickness match rejects it - jambs are *traced*, never invented.
    """
    stub_k = max(3, round(wall_width * 0.9))
    raw_floor = max(8, wall_width * 2)
    stub_floor = max(2, round(wall_width * PARAMS["stub_min_len_factor"]))
    align_tol = wall_width * 1.5
    match_lo, match_hi = PARAMS["stub_thickness_match"]

    h_short = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (stub_k, 1)))
    v_short = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, stub_k)))
    hm_d = cv2.dilate(hm, np.ones((3, 3), np.uint8))
    vm_d = cv2.dilate(vm, np.ones((3, 3), np.uint8))
    h_new = ((h_short > 0) & (hm_d == 0)).astype(np.uint8)
    v_new = ((v_short > 0) & (vm_d == 0)).astype(np.uint8)

    def collect(mask, orientation, axes):
        count, comp_labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        out = []
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            length = width if orientation == "horizontal" else height
            cross = height if orientation == "horizontal" else width
            density = area / max(1, width * height)
            if not (stub_floor <= length < raw_floor) or cross < min_cross or density < PARAMS["stub_min_density"]:
                continue
            ys, xs = np.where(comp_labels == index)
            axis = float(np.median(ys)) if orientation == "horizontal" else float(np.median(xs))
            if not any(abs(axis - a) <= align_tol and match_lo <= cross / t <= match_hi
                       for a, t in axes):
                continue
            if orientation == "horizontal":
                stub = WallSegment(orientation, axis, float(x), float(x + width - 1), float(cross))
            else:
                stub = WallSegment(orientation, axis, float(y), float(y + height - 1), float(cross))
            if _is_anchored(stub, anchor):
                out.append(stub)
        return out

    return collect(h_new, "horizontal", wall_axes["horizontal"]) + collect(v_new, "vertical", wall_axes["vertical"])


def _drop_leaf_bars(segments: list[WallSegment], wall_width: int) -> list[WallSegment]:
    """Drop door LEAVES traced as walls (styles that draw the open leaf as a solid bar).

    A leaf bar is door-length, hangs off its hinge wall into open room space, and is
    leaf-THIN — a drawn leaf is a stroke line, never a full wall band, so a run at
    band thickness (>= 0.8 wall_width) is a wall no matter how it hangs (a short
    partition whose far end stops at a door opening and whose continuation is
    jogged off-axis otherwise reads exactly like a leaf and vanishes). For thin
    runs, geometry decides: no collinear sibling of similar thickness (a jamb piece
    is part of an interrupted wall LINE; a leaf sits on its own axis) and no
    perpendicular run at the far end (a real pier terminates on walls at both ends;
    a leaf tip floats). Only runs failing BOTH tests are dropped, so door-pierced
    walls and short piers survive.
    """
    max_len = wall_width * PARAMS["door_gap_hi_factor"]
    band_min = 0.8 * wall_width
    sibling_tol = max(1.0, wall_width * 0.5)
    touch_tol = wall_width * 1.2
    match_lo, match_hi = PARAMS["stub_thickness_match"]

    def has_collinear_sibling(s: WallSegment) -> bool:
        return any(
            o is not s and o.orientation == s.orientation
            and abs(o.axis - s.axis) <= sibling_tol
            and match_lo <= s.thickness / max(1.0, o.thickness) <= match_hi
            for o in segments
        )

    def both_ends_on_walls(s: WallSegment) -> bool:
        for p in (s.start, s.end):
            x, y = (p, s.axis) if s.orientation == "horizontal" else (s.axis, p)
            if not any(
                o.orientation != s.orientation
                and abs((x if s.orientation == "horizontal" else y) - o.axis) <= touch_tol
                and o.start - touch_tol <= (y if s.orientation == "horizontal" else x) <= o.end + touch_tol
                for o in segments
            ):
                return False
        return True

    return [s for s in segments
            if (s.end - s.start) > max_len or s.thickness >= band_min
            or has_collinear_sibling(s) or both_ends_on_walls(s)]


def _drop_text_bars(segments: list[WallSegment], raw_ink: np.ndarray,
                    siblings: list[WallSegment] | None = None) -> list[WallSegment]:
    """Drop runs that are LABEL TEXT welded into bars.

    A drawn wall is one continuous stroke; a label word is letters separated by
    white, which ``_solidify``'s closing welds into a bar long enough to trace as
    wall. Worst on thin-wall drawings, where the text stroke matches the wall
    thickness (defeating the cross floor) and a label squeezed into a small room
    touches the wall network (defeating anchoring). On the PRE-solidify ink a wall
    covers its span continuously while a text bar breaks at every inter-letter gap:
    repeated breaks + low raw coverage is text. A text-like fragment that HUGS a
    solid run on the same axis is NOT dropped: a label whose glyphs overlap a wall
    contaminates the wall's edge re-trace into a broken signature, but that
    fragment is still wall (it carries real jamb geometry — dropping it loses real
    doors); a label floating on its own axis has no solid sibling and dies. (A
    per-column ink-thickness uniformity test for words WELDED into one solid blob
    was tried on top and regresses: on raw pre-cluster runs the metric is too
    noisy — door leaves and junction ink fluctuate a real wall's columns just like
    letters do.) Image-scale runs are exempt — a hollow double-line exterior band
    with window slots legitimately breaks many times, and no label is a quarter of
    the image long.
    """
    height, width = raw_ink.shape

    def band_stats(s: WallSegment) -> tuple[float, int]:
        half = max(1, round(s.thickness / 2))
        axis = int(round(s.axis))
        lo, hi = int(round(s.start)), int(round(s.end)) + 1
        if s.orientation == "horizontal":
            band = raw_ink[max(0, axis - half):axis + half + 1, max(0, lo):min(width, hi)]
            presence = band.any(axis=0) if band.size else np.zeros(0, bool)
        else:
            band = raw_ink[max(0, lo):min(height, hi), max(0, axis - half):axis + half + 1]
            presence = band.any(axis=1) if band.size else np.zeros(0, bool)
        if presence.size == 0:
            return 0.0, 0
        breaks = int(np.count_nonzero(np.diff(presence.astype(np.int8)) == -1)
                     + (not presence[0]))
        return float(presence.mean()), breaks

    pool = list(segments) + [s for s in (siblings or []) if s not in segments]
    stats = {id(s): band_stats(s) for s in pool}

    def has_solid_sibling(s: WallSegment) -> bool:
        tol = max(2.0, s.thickness)
        for o in pool:
            if o is s or o.orientation != s.orientation or abs(o.axis - s.axis) > tol:
                continue
            if min(o.end, s.end) - max(o.start, s.start) <= 0:
                continue
            if stats[id(o)][0] >= 0.95:
                return True
        return False

    kept = []
    for s in segments:
        span = width if s.orientation == "horizontal" else height
        if (s.end - s.start) >= 0.25 * span:
            kept.append(s)
            continue
        coverage, breaks = stats[id(s)]
        if (coverage > 0.0 and breaks >= PARAMS["text_bar_min_breaks"]
                and coverage <= PARAMS["text_bar_max_coverage"]
                and not has_solid_sibling(s)):
            continue
        kept.append(s)
    return kept


def trace_walls(gray: np.ndarray):
    """Faithful directional-morphology wall trace with a short-stub recovery pass.

    Acceptance is thickness-agnostic (see ``_raw_segments``): image models vary wall
    thickness wildly across AND within drawings (a 6-room clinic gets ~55 px perimeter
    bands over ~15 px partitions where an 80-room tower draws uniform ~11 px), so runs
    are accepted from a small absolute floor upward and ``wall_width`` — the scale
    anchor for door/gap/postprocess factors — is the MEDIAN of the accepted runs' own
    cross thickness, never an acceptance gate.

    Returns ``(walls, wall_width, dark, wall_pixels, raw_count, stub_count, anchor)``
    — ``anchor`` is the ink component labels/reach pair the anchoring gates share.
    """
    _, raw_ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = _solidify(raw_ink)
    # the directional opening kernels must exceed the THICKEST band (else perpendicular
    # bands leak into both masks), so they follow the measured heaviest stroke.
    stroke = _estimate_stroke_thickness(dark)
    hm, vm = _directional_masks(dark, min_len=max(15, round(stroke * 1.6)))

    min_cross = max(3, round(min(gray.shape) * PARAMS["min_cross_frac"]))
    min_reach = max(64.0, min(gray.shape) * PARAMS["anchor_reach_frac"])
    anchor = _ink_anchor(dark, min_reach, PARAMS["anchor_area_frac"],
                         PARAMS["max_symbol_density"])
    raw = _raw_segments(hm, "horizontal", min_cross) + _raw_segments(vm, "vertical", min_cross)
    raw = [s for s in raw if _is_anchored(s, anchor)]
    raw = _drop_text_bars(raw, raw_ink)
    ww = _median_wall_width(raw, gray.shape)
    raw = _drop_leaf_bars(raw, ww)

    wall_axes = {
        "horizontal": [(s.axis, max(1.0, s.thickness)) for s in raw if s.orientation == "horizontal"],
        "vertical": [(s.axis, max(1.0, s.thickness)) for s in raw if s.orientation == "vertical"],
    }
    stubs = _drop_text_bars(
        _short_stub_segments(dark, hm, vm, wall_axes, ww, min_cross, anchor),
        raw_ink, siblings=raw)

    clustered = _cluster_axes(raw + stubs, tolerance=max(1.0, ww * 0.5))
    walls = _merge_overlaps(clustered, tolerance=max(1.0, ww * 0.5))
    wall_pixels = ((hm > 0) | (vm > 0)).astype(np.uint8)
    return walls, ww, dark, wall_pixels, len(raw), len(stubs), anchor


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


def add_jogged_gaps(segments: list[WallSegment], wall_width: int):
    """Gaps between two collinear-INTENT walls whose axes jog slightly apart.

    A door drawn where the wall thickness changes (a fat exterior band meeting a
    thinner return, a wall fused with a window strip) leaves the two jambs on
    axes ~0.5-1 wall thickness apart: ``_cluster_axes`` keeps them in separate
    axis groups, so ``find_gaps`` never pairs them and the opening has NO
    candidate at all. Two same-orientation walls facing each other end-to-start
    across a door-scale span are one interrupted wall line when their thickness
    BANDS still overlap (genuinely parallel walls have clear space between
    bands); the gap is emitted on the longer wall's axis with both jambs real,
    and ``start_axis``/``end_axis`` carry each side's true axis so a confirmed
    door's bridge can add the tiny perpendicular connectors at the jog.
    """
    off_max = wall_width * PARAMS["jog_axis_frac"]
    lo = wall_width * 0.75
    hi = wall_width * PARAMS["double_gap_hi_factor"]
    out = []
    for a in segments:
        for b in segments:
            if a is b or a.orientation != b.orientation or a.end >= b.start:
                continue
            gap = b.start - a.end
            if not (lo <= gap <= hi):
                continue
            off = abs(a.axis - b.axis)
            if off < 1e-6 or off > off_max:
                continue
            if off >= (a.thickness + b.thickness) / 2.0:   # bands must overlap
                continue
            # a third wall already spanning the gap near either axis means this
            # is not an interrupted line (e.g. a parallel face re-trace)
            if any(o is not a and o is not b and o.orientation == a.orientation
                   and (abs(o.axis - a.axis) <= off_max or abs(o.axis - b.axis) <= off_max)
                   and o.start < b.start and o.end > a.end
                   for o in segments):
                continue
            axis = a.axis if (a.end - a.start) >= (b.end - b.start) else b.axis
            out.append(dict(orientation=a.orientation, axis=float(axis),
                            start=float(a.end), end=float(b.start), width=float(gap),
                            real_start=True, real_end=True,
                            start_axis=float(a.axis), end_axis=float(b.axis)))
    return out


def find_slot_gaps(segments, dark: np.ndarray, wall_width: int):
    """Door-slot styles: gaps cut INTO a wall band whose traced line never breaks.

    Some drawings keep every wall line continuous and mark a door as a white slot
    inside the band (exactly like their window symbol) with the swing arc alongside —
    the collinear/terminal gap finders see no break at all. Scan each traced wall's
    band for interior low-ink runs at door scale and emit them as gap candidates with
    real jambs (the band continues on both sides). Only a swing arc confirms them, so
    window slots stay walls. Only segments of real BAND thickness are scanned: a
    door-leaf line traced as a wall must not fabricate slot gaps under the door's own
    arc (which would out-claim the true opening's candidate). The band floor scales
    with ``wall_width``, not the image — thin-wall drawings (ww ~5 px) draw their
    whole wall net below any resolution-based floor, and their half-depth door slots
    (a lintel line keeps the band's top edge continuous) live in exactly those
    segments.
    """
    height, width = dark.shape
    half = wall_width // 2 + 1
    lo_w = wall_width * PARAMS["door_gap_lo_factor"]
    hi_w = wall_width * PARAMS["door_gap_hi_factor"]
    min_band = max(4.0, PARAMS["stub_thickness_match"][0] * wall_width)
    gaps = []
    for s in segments:
        if s.thickness < min_band:
            continue
        axis = int(round(s.axis))
        lo, hi = int(math.floor(s.start)), int(math.ceil(s.end))
        if s.orientation == "horizontal":
            band = dark[max(0, axis - half):min(height, axis + half + 1), max(0, lo):min(width, hi)]
            frac = band.mean(axis=0) if band.size else np.zeros(0)
        else:
            band = dark[max(0, lo):min(height, hi), max(0, axis - half):min(width, axis + half + 1)]
            frac = band.mean(axis=1) if band.size else np.zeros(0)
        n = int(frac.shape[0])
        if n == 0:
            continue
        is_open = frac < PARAMS["gap_ink_frac"]
        position = 0
        while position < n:
            if is_open[position]:
                begin = position
                while position < n and is_open[position]:
                    position += 1
                if begin > 0 and position < n and lo_w <= (position - begin) <= hi_w:
                    gaps.append(dict(orientation=s.orientation, axis=float(s.axis),
                                     start=float(lo + begin), end=float(lo + position),
                                     width=float(position - begin),
                                     real_start=True, real_end=True, slot=True))
            else:
                position += 1
    return gaps


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

def trim_gap_ink(gap, dark: np.ndarray, wall_width: int,
                 anchor: tuple[np.ndarray, np.ndarray] | None = None):
    """Shrink/split a gap on contiguous wall-thick ink in its band; recover that ink.

    The directional trace drops wall pieces that are not axis-aligned runs: jamb stubs
    merged into a perpendicular wall's body, and ORNAMENT PIERS (diagonal diamond
    hinge posts some styles draw mid-opening). Both are still plain wall ink in
    ``dark``: scan the gap band for runs where the cross-band ink fraction stays
    wall-like. Runs at the gap ends shrink the gap; a long-enough interior run that is
    ANCHORED to the wall network is an untraced pier that SPLITS the gap into
    door-scale pieces (label text floating on the gap line is not anchored and never
    splits). Every consumed run is emitted as a traced ``WallSegment`` (nothing
    invented - it is ink). Returns ``(gap_pieces, recovered_stubs)``; no pieces when
    no opening remains.
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
        return [], []
    is_wall = frac >= PARAMS["gap_ink_frac"]

    # contiguous wall-ink runs across the band, as [start, end) offsets
    runs, position = [], 0
    while position < n:
        if is_wall[position]:
            begin = position
            while position < n and is_wall[position]:
                position += 1
            runs.append((begin, position))
        else:
            position += 1

    min_run = wall_width * PARAMS["stub_recover_factor"]
    stubs, cuts = [], []
    for begin, end in runs:
        run_len = end - begin
        at_edge = begin == 0 or end == n
        stub = WallSegment(gap["orientation"], gap["axis"],
                           float(lo + begin), float(lo + end), float(wall_width))
        if at_edge:
            cuts.append((begin, end))          # end ink always trims the opening
            if run_len >= min_run:
                stubs.append(stub)             # ... and long enough to keep as wall
            continue
        # interior ink = candidate untraced pier; only long wall-network ink splits
        if run_len >= min_run and anchor is not None and _is_anchored(stub, anchor):
            stubs.append(stub)
            cuts.append((begin, end))

    # ink-free pieces between consumed runs (and the gap ends)
    bounds, cursor = [], 0
    for begin, end in cuts:
        bounds.append((cursor, begin))
        cursor = end
    bounds.append((cursor, n))
    pieces = []
    for begin, end in bounds:
        if (end - begin) <= wall_width * 0.75:
            continue
        pieces.append(dict(gap, start=float(lo + begin), end=float(lo + end),
                           width=float(end - begin),
                           real_start=bool(gap.get("real_start", True)) if begin == 0 else True,
                           real_end=bool(gap.get("real_end", True)) if end == n else True))
    return pieces, stubs


# ---- door-arc detector ------------------------------------------------------------

def _residual_mask(dark, wall_pixels, anchor):
    """Non-wall ink that may support a swing arc — no label text.

    Door symbols usually touch the wall network (anchored); some styles draw them
    fully detached, in which case they are still sparse thin curves (symbol_like).
    Label text is neither — dense floating glyphs — and is excluded so room labels
    cannot score as arc evidence (a floating word inside a big axis-coincidence
    gap can otherwise outscore real doors).
    """
    labels, anchored, symbol_like = anchor
    resid = ((dark > 0) & (wall_pixels == 0)).astype(np.uint8)
    resid[~(anchored | symbol_like)[labels]] = 0
    return resid


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
    """Return ``(best TracedDoor, supporting resid_pts indices)`` for a gap, or None.

    Tests both jambs x both swing sides x every ``radius_scales`` regime: single leaf
    (r ~= gap), double door (two mirrored leaves, each r ~= gap/2 — either arc alone
    confirms it, so wide waiting/corridor double doors bridge like any opening) and
    an undersized leaf in a wide mouth (r ~= 0.7-0.8 gap). The supporting indices
    are the winning arc's annulus inliers — ``detect_doors`` uses them to make arc
    ink EXCLUSIVE to one door, killing phantom gaps scored on a neighbouring door's
    arc.
    """
    w = gap["width"]
    if not (wall_width * PARAMS["door_gap_lo_factor"] <= w <= wall_width * PARAMS["double_gap_hi_factor"]):
        return None
    # past the single-leaf cap only the regimes a drawn door can physically
    # produce confirm: an undersized leaf slightly past it, a true double door
    # (two mirrored half-gap leaves) further out
    wide = w > wall_width * PARAMS["door_gap_hi_factor"]
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

    radius_scales = PARAMS["radius_scales"]
    r_hi = w * max(radius_scales)
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
    win_idx = np.where(win)[0]

    best = None  # (score, cov, ratio, empty, npix, hinge, along, perp, r)
    for (hinge, along_unit) in jambs:
        rel = local - np.array(hinge)
        a_all = rel @ along_unit
        for perp_unit in perp_dirs:
            p_all = rel @ perp_unit
            for scale in radius_scales:
                if wide:
                    if not (0.45 <= scale <= 0.55    # true double door
                            or (scale < 0.45
                                and w <= wall_width * PARAMS["undersized_gap_hi_factor"])):
                        continue
                elif scale < 0.45:
                    continue    # small-radius regimes hijack in-cap doors' arcs
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
                    best = (score, cov, ratio, empty, npix, hinge, along_unit, perp_unit, r)
    if best is None:
        return None
    _, cov, ratio, empty, npix, hinge, along_unit, perp_unit, r = best
    # winning arc's annulus inliers, as global resid_pts indices (its exclusive ink)
    rel = local - np.array(hinge)
    a_all, p_all = rel @ along_unit, rel @ perp_unit
    d = np.hypot(a_all, p_all)
    ang = np.degrees(np.arctan2(p_all, a_all))
    band = max(band_floor, PARAMS["band_factor"] * r)
    in_band = ((ang >= -PARAMS["ang_tol_deg"]) & (ang <= 90.0 + PARAMS["ang_tol_deg"])
               & (d >= 0.45 * r) & (d <= 1.55 * r) & (np.abs(d - r) <= band))
    door = TracedDoor(orientation, axis, e0, e1, (float(hinge[0]), float(hinge[1])),
                      (float(perp_unit[0]), float(perp_unit[1])), float(r), float(cov), int(npix),
                      float(ratio),
                      start_axis=gap.get("start_axis"), end_axis=gap.get("end_axis"))
    return door, win_idx[in_band]


def window_break_fills(gaps, doors, dark, anchor, wall_width):
    """Fill wall breaks that are drawn WINDOWS, not openings (local, ink-gated).

    Hollow window symbols interrupt the solid band: the trace splits the wall
    there and the room behind bleeds out through a "gap" that is actually
    glazing. A drawn opening (door/passage) is WHITE across the band; a window
    keeps continuous ink presence along the whole break (its thin face/sill
    lines, wall-network anchored - label text floating near the line is not).
    Such a break is drawn wall for room topology: emit a fill segment (plus jog
    connectors when the gap carries offset end axes). A gap already confirmed
    as a door is never filled.
    """
    labels, anchored, _symbol = anchor
    wall_ink = ((dark > 0) & anchored[labels]).astype(np.uint8)
    h, w = dark.shape
    half = wall_width // 2 + 1
    hi = wall_width * PARAMS["window_fill_hi_factor"]
    fills: list[WallSegment] = []
    for gap in gaps:
        if not (wall_width * 0.75 <= gap["width"] <= hi):
            continue
        if any(d.orientation == gap["orientation"] and abs(d.axis - gap["axis"]) <= wall_width
               and min(d.gap_end, gap["end"]) - max(d.gap_start, gap["start"]) > 0
               for d in doors):
            continue
        axis = int(round(gap["axis"]))
        lo_i, hi_i = int(math.floor(gap["start"])), int(math.ceil(gap["end"]))
        if gap["orientation"] == "horizontal":
            band = wall_ink[max(0, axis - half):min(h, axis + half + 1),
                            max(0, lo_i):min(w, hi_i)]
            presence = (band > 0).any(axis=0) if band.size else np.zeros(0, bool)
        else:
            band = wall_ink[max(0, lo_i):min(h, hi_i),
                            max(0, axis - half):min(w, axis + half + 1)]
            presence = (band > 0).any(axis=1) if band.size else np.zeros(0, bool)
        if presence.size == 0 or float(presence.mean()) < PARAMS["window_presence"]:
            continue
        fills.append(WallSegment(gap["orientation"], gap["axis"],
                                 gap["start"], gap["end"], float(wall_width)))
        perp = "vertical" if gap["orientation"] == "horizontal" else "horizontal"
        for pos_key, axis_key in (("start", "start_axis"), ("end", "end_axis")):
            other = gap.get(axis_key)
            if other is not None and abs(other - gap["axis"]) > 0.5:
                fills.append(WallSegment(perp, gap[pos_key],
                                         min(other, gap["axis"]), max(other, gap["axis"]),
                                         float(wall_width)))
    return fills


def detect_doors(segments, dark, wall_pixels, wall_width, anchor):
    """Return ``(raw_gaps, sub_gaps, doors, jamb_stubs, window_fills)``.

    Doors are confirmed on pier-split sub-gaps of collinear + terminal gaps. A sub-gap
    whose BOTH ends are pier cuts has no traced wall jamb at all - it is a rounding
    coincidence of far-apart walls sharing an axis (the source of phantom doors bridged
    in mid-room on a neighbouring door's arc ink) - and is rejected outright. Each
    surviving gap is then ink-trimmed AND ink-split (``trim_gap_ink``) so hidden
    corner/jamb stubs and untraced ornament piers become walls and the arc test sees
    the true opening(s).

    Arc ink is EXCLUSIVE to one door: each residual point supports only the first
    door that claims it, so an axis-rounding gap whose annulus merely grazes a
    NEIGHBOUR's swing arc (its own opening has no arc) finds its ink already claimed
    and dies. Real doors own disjoint arcs. ONE physical arc can read as a door in
    two different walls (an L-corner break in one wall vs a slot in the perpendicular
    band), so the claim order decides which reading bridges: BREAK/terminal/jogged
    candidates outrank ALL slot candidates, because only a break's bridge closes
    topology — a slot's band is already continuous, so letting the slot win leaves
    the real opening unbridged and the room leaks around the corner. (Ranking by arc
    fit instead was tried and regresses: a corner slot reading is biased canonical —
    its white run is the arc's own footprint, so it out-fits a real wide-mouth door
    drawn with an undersized leaf.) Within a class, arc quality ranks.
    """

    def _claim_rank(candidate):
        door, _support, is_slot = candidate
        return (is_slot, -door.coverage * door.inlier_ratio)

    resid = _residual_mask(dark, wall_pixels, anchor)
    ys, xs = np.where(resid > 0)
    resid_pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    collinear = find_gaps(segments, wall_width)
    jogged = add_jogged_gaps(segments, wall_width)
    raw_gaps = collinear + add_terminal_gaps(segments, wall_width)
    sub_gaps = (split_gaps_at_piers(raw_gaps, segments, wall_width)
                + find_slot_gaps(segments, dark, wall_width) + jogged)

    def lacks_jamb_support(piece) -> bool:
        """A WIDE 'gap' flanked only by tiny wall fragments is a FACE-LINE
        artifact (a thick wall's re-traced face broken into stubs on a nearby
        axis), not a doorway - a real wide opening has a substantial wall on at
        least one side. Applies only past the single-leaf cap (the regimes this
        change opened up), so every door-scale gap keeps the plain real-jamb
        rule (an in-band slot gap counts as supported by its own host segment)."""
        if piece["width"] <= wall_width * PARAMS["door_gap_hi_factor"]:
            return False
        tol = wall_width * 1.5
        support = wall_width * 2.0
        for s in segments:
            if s.orientation != piece["orientation"] or abs(s.axis - piece["axis"]) > 1.0:
                continue
            if s.start <= piece["start"] + 1.0 and s.end >= piece["end"] - 1.0:
                return False                       # slot inside one host segment
            if (abs(s.end - piece["start"]) <= tol or abs(s.start - piece["end"]) <= tol) \
                    and (s.end - s.start) >= support:
                return False                       # a real flanking wall
        return True

    candidates, jamb_stubs = [], []
    for gap in sub_gaps:
        if not (gap.get("real_start", True) or gap.get("real_end", True)):
            continue
        pieces, stubs = trim_gap_ink(gap, dark, wall_width, anchor)
        jamb_stubs.extend(stubs)
        for piece in pieces:
            if lacks_jamb_support(piece):
                continue
            detected = detect_door_at_gap(resid_pts, piece, wall_width)
            if detected is not None:
                candidates.append((*detected, bool(piece.get("slot"))))

    claimed = np.zeros(resid_pts.shape[0], dtype=bool)
    doors = []
    for door, support, _is_slot in sorted(candidates, key=_claim_rank):
        if support.size and float(claimed[support].mean()) > PARAMS["arc_claim_max_overlap"]:
            continue
        claimed[support] = True
        doors.append(door)
    doors = _dedupe_doors(doors, wall_width)
    fills = window_break_fills(collinear + jogged, doors, dark, anchor, wall_width)
    return raw_gaps, sub_gaps, doors, jamb_stubs, fills


def _dedupe_doors(doors: list[TracedDoor], wall_width: int) -> list[TracedDoor]:
    """Collapse twin detections of ONE physical opening.

    Axis rounding can leave the same opening on two nearly-identical collinear axes
    (e.g. x=957 and x=963); both then confirm on the same arc ink and bridge as two
    parallel segments a few px apart (rendered as a double line). Two doors within
    ~1 wall thickness in axis whose gap spans overlap are one door - keep the better
    arc (coverage * inlier_ratio).
    """
    kept: list[TracedDoor] = []
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
    ``extra`` carries ink-recovered jamb stubs from ``trim_gap_ink`` and window-break
    fills (traced/ink-gated, not invented). A door whose jambs sit on jogged axes
    (``start_axis``/``end_axis``) also gets the tiny perpendicular connectors, so the
    bridge meets both walls instead of ending a jog away from one.
    """
    bridges = []
    for d in doors:
        bridges.append(WallSegment(d.orientation, d.axis, d.gap_start, d.gap_end,
                                   float(wall_width)))
        perp = "vertical" if d.orientation == "horizontal" else "horizontal"
        for pos, other in ((d.gap_start, d.start_axis), (d.gap_end, d.end_axis)):
            if other is not None and abs(other - d.axis) > 0.5:
                bridges.append(WallSegment(perp, pos, min(other, d.axis),
                                           max(other, d.axis), float(wall_width)))
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
    3. PIER   - a gap between two COLLINEAR wall ends within the same reach whose
                span is continuously covered by wall ink is filled: a double-door
                pier drawn FATTER than the wall band escapes both the long trace
                (too short) and the stub pass (thickness mismatch), leaving a
                solid-ink hole between the two bridged door openings. A drawn
                opening is white there and is never filled.
    4. TRIM   - an endpoint overhanging its outermost perpendicular junction by
                less than ``2 * wall_width`` is cut back to that junction (the small
                perpendicular ticks poking out of a main wall).
    5. DROP   - segments shorter than ``2.5 * wall_width`` that still have a free
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

    piers_filled = 0
    if dark is not None:
        reach = wall_width * PARAMS["corner_snap_reach"]
        by_axis: dict[tuple[str, float], list[WallSegment]] = {}
        for s in segs:
            key = (s.orientation, round(s.axis / max(1.0, wall_width * 0.5)))
            by_axis.setdefault(key, []).append(s)
        for items in by_axis.values():
            items.sort(key=lambda s: s.start)
            for a, b in zip(items, items[1:]):
                gap = b.start - a.end
                if not (0.0 < gap <= reach):
                    continue
                # gap-trim tolerance leaves a white sliver at the wall ends; the
                # ink test judges the gap's CORE (a drawn opening stays all white)
                pad = min(gap * 0.25, wall_width * 0.5)
                if _ink_covered(dark, a.orientation, a.axis, a.end + pad,
                                b.start - pad, wall_width):
                    a.end = b.start
                    piers_filled += 1

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
                 corners_snapped=corner_snapped, piers_filled=piers_filled,
                 endpoints_trimmed=trimmed, whiskers_dropped=dropped, face_lines_dropped=face_lines,
                 segments_in=len(segments), segments_out=len(keep))
    return keep, stats


def polygonize_rooms(segments: list[WallSegment], image_shape) -> list:
    """Room polygons from the wall graph (closed rings only; open edges vanish),
    ordered top-to-bottom then left-to-right so room ids are stable per image."""
    net = unary_union([s.line() for s in segments])
    min_area = image_shape[0] * image_shape[1] * 0.0004
    rooms = [p for p in polygonize(net) if p.area >= min_area]
    rooms.sort(key=lambda p: (p.centroid.y, p.centroid.x))
    return rooms


def hide_label_residue(segments: list[WallSegment], polygons, wall_width: int,
                       text_quads=()) -> list[WallSegment]:
    """Hide label-text residue from the RENDERED walls (topology already fixed).

    Welded label words that survive every ink-level gate render as short black
    dashes inside rooms. They are recognisable exactly here — AFTER
    ``polygonize_rooms`` — because "inside a room" and "on a room boundary" now
    have meanings. Three convictions, all render-only (ink-level text removal
    was refuted three ways: text ink is load-bearing for the trace; this runs
    after the trace is done and can change nothing but the drawing):

    1. a short segment fully inside one room polygon, clear of its boundary by
       a wall width, closes nothing and carries no door — decoration. Fragments
       come in CLUSTERS (a label line breaks into chunks), so a fragment with a
       fellow nearby is text, and so is a tiny lone one — while a single longer
       solid bar (a drawn counter/fixture) stays;
    2. with OCR text quads available (``text_mask.detect_label_quads``), a
       segment lying mostly inside a text region but NOT along any room
       boundary ring is text welded to a wall — every real wall carries a ring
       edge, a label stub never does;
    3. a boundary-welded stub on the same text line as fragments already
       hidden (a real door-jamb stub never shares its axis with label residue).
    """
    max_len = wall_width * PARAMS["door_gap_hi_factor"]
    margin = float(wall_width)
    lone_max = 4.0 * wall_width
    neighbour_gap = 2.5 * wall_width

    rooms = [room if room.is_valid else room.buffer(0) for room in polygons]
    interior = []
    for i, s in enumerate(segments):
        if (s.end - s.start) > max_len or (s.end - s.start) <= 0:
            continue
        line = s.line()
        for room in rooms:
            if not room.is_empty and room.contains(line):
                clearance = room.boundary.distance(line)
                if clearance == clearance and clearance >= margin:   # NaN-safe
                    interior.append(i)
                break

    def bounds(s: WallSegment):
        half = max(1.0, s.thickness / 2.0)
        if s.orientation == "horizontal":
            return s.start, s.axis - half, s.end, s.axis + half
        return s.axis - half, s.start, s.axis + half, s.end

    def near(a: WallSegment, b: WallSegment) -> bool:
        ax0, ay0, ax1, ay1 = bounds(a)
        bx0, by0, bx1, by1 = bounds(b)
        dx = max(0.0, max(ax0, bx0) - min(ax1, bx1))
        dy = max(0.0, max(ay0, by0) - min(ay1, by1))
        return max(dx, dy) <= neighbour_gap

    hidden = set()
    for i in interior:
        s = segments[i]
        clustered = any(j != i and near(s, segments[j]) for j in interior)
        if clustered or (s.end - s.start) <= lone_max:
            hidden.add(i)

    # conviction 2: mostly inside an OCR text region AND not along any ring
    if len(text_quads) and rooms:
        text_zone = unary_union([ShapelyPolygon(q.tolist()).buffer(0)
                                 for q in text_quads])
        ring_zone = unary_union([room.boundary for room in rooms
                                 if not room.is_empty]).buffer(max(1.5, 0.3 * wall_width))
        for i, s in enumerate(segments):
            if i in hidden or not (0 < (s.end - s.start) <= 3.0 * max_len):
                continue
            line = s.line()
            if line.intersection(text_zone).length < 0.7 * line.length:
                continue
            if line.intersection(ring_zone).length >= 0.5 * line.length:
                continue    # carries a room-boundary edge: real wall
            hidden.add(i)

    # conviction 3: a label's leading word often welds onto the partition beside
    # it and traces as a short T-stub — boundary-touching, so the interior test
    # spares it. It convicts itself by lying ON THE SAME TEXT LINE as fragments
    # already hidden: a real door-jamb stub never shares its axis with mid-room
    # label residue.
    if hidden:
        for i, s in enumerate(segments):
            if i in hidden or (s.end - s.start) > max_len:
                continue
            on_text_line = any(
                segments[j].orientation == s.orientation
                and abs(segments[j].axis - s.axis) <= wall_width
                and (max(s.start, segments[j].start)
                     - min(s.end, segments[j].end)) <= 6.0 * wall_width
                for j in hidden
            )
            if on_text_line:
                hidden.add(i)
    return [s for i, s in enumerate(segments) if i not in hidden]


# ---- plan + room-graph assembly ----------------------------------------------------

def _rooms_from_polygons(polygons) -> list[Room]:
    """Geometry-only rooms: untyped (type "unknown"), ids r1..rN, px coordinates."""
    rooms = []
    for polygon in polygons:
        coords = [(float(x), float(y)) for x, y in list(polygon.exterior.coords)[:-1]]
        if len(coords) < 3:
            continue
        rooms.append(Room(id=f"r{len(rooms) + 1}", type="unknown", polygon=coords))
    return rooms


def _door_edges(doors: list[TracedDoor], polygons, rooms: list[Room],
                wall_width: int) -> list[DoorEdge]:
    """Connect each detected door to the two room polygons flanking its opening.

    Probes one point on each side of the door centre along the wall's perpendicular
    (just past the wall band); the room polygon covering a probe names that side,
    "exterior" when no room does. Duplicate pairs collapse to one edge.
    """
    offset = max(2.0, wall_width * _PROBE_OFFSET_FACTOR)
    edges: list[DoorEdge] = []
    seen: set[frozenset] = set()
    for door in doors:
        cx, cy = door.center
        if door.orientation == "horizontal":
            probes = (Point(cx, cy - offset), Point(cx, cy + offset))
        else:
            probes = (Point(cx - offset, cy), Point(cx + offset, cy))
        sides = []
        for probe in probes:
            room_id = next(
                (rooms[i].id for i, polygon in enumerate(polygons) if polygon.covers(probe)),
                "exterior",
            )
            sides.append(room_id)
        room_a, room_b = sides
        if room_a == room_b:      # both open ground, or a door inside one region
            continue
        key = frozenset((room_a, room_b))
        if key in seen:
            continue
        seen.add(key)
        edges.append(DoorEdge(room_a=room_a, room_b=room_b))
    return edges


# ---- fixed artifacts ---------------------------------------------------------------

def _png_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def _draw_walls(canvas, segments, color, thickness):
    for s in segments:
        if s.orientation == "horizontal":
            p0 = (round(s.start), round(s.axis))
            p1 = (round(s.end), round(s.axis))
        else:
            p0 = (round(s.axis), round(s.start))
            p1 = (round(s.axis), round(s.end))
        cv2.line(canvas, p0, p1, color, thickness)


def _draw_door_symbol(canvas, door: TracedDoor, color):
    """Swing arc + leaf + hinge dot at the door's detected geometry."""
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
    pts = []
    for t in np.linspace(0, math.pi / 2, 24):    # arc from along (0 deg) to perp (90 deg)
        v = math.cos(t) * along + math.sin(t) * perp
        pts.append(hinge + door.radius * v)
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts], False, color, 1, cv2.LINE_AA)
    leaf_end = hinge + door.radius * perp
    cv2.line(canvas, tuple(hinge.astype(int)), tuple(leaf_end.astype(int)), color, 1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(hinge.astype(int)), 3, color, -1)


def _room_color(index: int) -> tuple[int, int, int]:
    """Deterministic distinct BGR colour for room ``index`` (golden-angle hue walk),
    so re-runs of the same plan colour the same regions comparably."""
    hue = (index * 0.618033988749895) % 1.0
    hsv = np.uint8([[[round(hue * 179), 150, 235]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def _render_artifacts(src_bgr, shape, walls, bridged, post, doors, polygons,
                      wall_width) -> dict[str, bytes]:
    """Render the five fixed linework artifacts (see module docstring)."""
    overlay_w = max(2, round(wall_width * 0.4))
    recon_w = max(2, wall_width // 2)

    walls_overlay = src_bgr.copy()
    _draw_walls(walls_overlay, walls, (0, 0, 255), overlay_w)

    doors_overlay = src_bgr.copy()
    _draw_walls(doors_overlay, walls, (150, 150, 150), overlay_w)
    for d in doors:
        _draw_door_symbol(doors_overlay, d, (0, 140, 255))

    def recon_image(segments):
        canvas = np.full((*shape, 3), 255, np.uint8)
        _draw_walls(canvas, segments, (0, 0, 0), recon_w)
        for d in doors:
            cv2.circle(canvas, tuple(map(round, d.center)), max(3, wall_width), (255, 60, 60), 1)
            _draw_door_symbol(canvas, d, (255, 60, 60))
        return canvas

    rooms_colorful = np.full((*shape, 3), 255, np.uint8)
    for index, polygon in enumerate(polygons):
        ring = np.array(polygon.exterior.coords, np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(rooms_colorful, [ring], _room_color(index))
    _draw_walls(rooms_colorful, post, (0, 0, 0), recon_w)
    for d in doors:                              # doors read as white gaps in the walls
        gap = WallSegment(d.orientation, d.axis, d.gap_start, d.gap_end, float(wall_width))
        _draw_walls(rooms_colorful, [gap], (255, 255, 255), recon_w + 2)

    return {
        "walls_overlay.png": _png_bytes(walls_overlay),
        "doors_overlay.png": _png_bytes(doors_overlay),
        "recon.png": _png_bytes(recon_image(bridged)),
        "recon_post.png": _png_bytes(recon_image(post)),
        "rooms_colorful.png": _png_bytes(rooms_colorful),
    }


# ---- public entry -------------------------------------------------------------------

def trace_linework(png: bytes) -> LineworkTrace:
    """Trace a realistic line-plan drawing into rooms + doors, geometry only.

    Returns a :class:`LineworkTrace` with an untyped px-unit :class:`Plan`, a
    :class:`RoomGraph` whose door edges name the plan rooms flanking each detected
    door (or "exterior"), trace diagnostics, and the five fixed artifact PNGs.
    """
    src_bgr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    if src_bgr is None:
        raise ValueError("trace_linework: could not decode the plan image")
    gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)

    walls, wall_width, dark, wall_pixels, raw_count, stub_count, anchor = trace_walls(gray)
    raw_gaps, sub_gaps, doors, jamb_stubs, window_fills = detect_doors(
        walls, dark, wall_pixels, wall_width, anchor)
    bridged, bridge_count = bridge_at_doors(walls, doors, wall_width,
                                            extra=jamb_stubs + window_fills)
    post, post_stats = postprocess_walls(bridged, wall_width, dark)
    polygons = polygonize_rooms(post, gray.shape)
    shown = hide_label_residue(post, polygons, wall_width,
                               text_quads=detect_label_quads(gray))

    rooms = _rooms_from_polygons(polygons)
    edges = _door_edges(doors, polygons, rooms, wall_width)
    plan = Plan(units="px", rooms=rooms)
    room_graph = RoomGraph(
        rooms=[RoomNode(id=room.id, type=room.type) for room in rooms],
        doors=edges,
    )

    room_max_area = gray.shape[0] * gray.shape[1] * PARAMS["room_max_area_frac"]
    diagnostics = dict(
        image_shape=[int(gray.shape[1]), int(gray.shape[0])],
        wall_width=int(wall_width),
        raw_segments=raw_count,
        short_stubs=stub_count,
        walls_traced=len(walls),
        gaps=len(raw_gaps),
        sub_gaps=len(sub_gaps),
        jamb_stubs=len(jamb_stubs),
        window_fills=len(window_fills),
        doors_detected=len(doors),
        bridges_made=bridge_count,
        postprocess=post_stats,
        rooms_closed=len(rooms),
        room_scale_rooms=sum(1 for p in polygons if p.area <= room_max_area),
        label_fragments_hidden=len(post) - len(shown),
        door_edges=dict(
            interior=sum(1 for e in edges if "exterior" not in (e.room_a, e.room_b)),
            exterior=sum(1 for e in edges if "exterior" in (e.room_a, e.room_b)),
        ),
    )
    artifacts = _render_artifacts(src_bgr, gray.shape, walls, bridged, shown, doors,
                                  polygons, wall_width)
    return LineworkTrace(plan=plan, room_graph=room_graph,
                         diagnostics=diagnostics, artifacts=artifacts)


def main() -> None:
    """Thin offline debug harness: trace one PNG, write the artifacts, print metrics."""
    import argparse
    import json
    import time
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Trace a linework floor plan offline (no model calls).")
    ap.add_argument("image", help="path to a realistic line-plan PNG")
    ap.add_argument("--out", default=None, help="output dir (default: hfagent/out/eval/<ts>-linework-trace)")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[1] / "out" / "eval"
        / f"{time.strftime('%Y%m%d-%H%M%S')}-linework-trace"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = trace_linework(Path(args.image).read_bytes())
    for filename, payload in trace.artifacts.items():
        (out_dir / filename).write_bytes(payload)
    (out_dir / "trace.json").write_text(json.dumps(trace.diagnostics, indent=2), encoding="utf-8")
    brief = {k: trace.diagnostics[k] for k in
             ("wall_width", "walls_traced", "doors_detected", "rooms_closed", "door_edges")}
    print(json.dumps(brief, indent=2))
    print(f"out: {out_dir}")


if __name__ == "__main__":
    main()
