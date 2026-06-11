# -*- coding: utf-8 -*-
"""Plan-vs-plan comparison metrics for Phase 0-A acceptance.

compare_plans(truth, parsed) greedily matches rooms of the same type by IoU
(shapely) and reports: room count match, type accuracy, mean/min IoU.
"""
from __future__ import annotations

from shapely.geometry import Polygon

from hfagent.schema.plan import Plan


def _poly(room) -> Polygon:
    p = Polygon(room.polygon)
    return p if p.is_valid else p.buffer(0)


def compare_plans(truth: Plan, parsed: Plan) -> dict:
    t_rooms = [(r, _poly(r)) for r in truth.rooms]
    p_rooms = [(r, _poly(r)) for r in parsed.rooms]

    pairs = []
    for ti, (tr, tp) in enumerate(t_rooms):
        for pi, (pr, pp) in enumerate(p_rooms):
            inter = tp.intersection(pp).area
            union = tp.union(pp).area
            if union > 0 and inter > 0:
                pairs.append((inter / union, ti, pi))
    pairs.sort(reverse=True)

    used_t, used_p, matches = set(), set(), []
    for iou, ti, pi in pairs:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        matches.append((iou, ti, pi))

    type_ok = sum(1 for _, ti, pi in matches if t_rooms[ti][0].type == p_rooms[pi][0].type)
    ious = [iou for iou, _, _ in matches]
    return {
        "truth_rooms": len(t_rooms),
        "parsed_rooms": len(p_rooms),
        "matched": len(matches),
        "room_count_match": len(t_rooms) == len(p_rooms),
        "type_accuracy": type_ok / len(matches) if matches else 0.0,
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "min_iou": min(ious) if ious else 0.0,
        "unmatched_truth": len(t_rooms) - len(matches),
        "unmatched_parsed": len(p_rooms) - len(matches),
    }
