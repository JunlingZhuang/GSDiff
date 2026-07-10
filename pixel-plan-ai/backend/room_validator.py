from __future__ import annotations

from collections import Counter
from math import hypot
from typing import Any

# Deterministic schematic validator for a single-patient ICU room. Mirrors the floor
# validator's check shape ({category, label, pass}) so service.failed_check_keys and
# service.validation_delta operate on it unchanged, and produces a comparable weighted
# 0-100 score. It is a planning study, not a code-compliance determination.

# Weighted category buckets sum to 100, mirroring validator.py's component-score style.
# The five hard categories (count, overlap, clearance, access, anchor) carry 80 of the 100;
# room_service treats a failure in any of them as a blocker regardless of score.
CATEGORY_WEIGHTS: dict[str, int] = {
    "room": 5,
    "count": 20,
    "overlap": 15,
    "anchor": 10,
    "clearance": 20,
    "access": 15,
    "equipment": 10,
    "zoning": 5,
}

TOL = 1e-6
ANCHOR_TOL = 0.25  # flush-to-wall tolerance (ft)

# Facing convention (shared with room_prompt.py, the 2D symbols in
# web/public/assets/2d, and the GLB builders in scripts/generate-3d-assets.mjs):
# at rotation_deg 0 an asset's FRONT is its SOUTH edge. A wall-anchored asset must
# rotate so its front turns INTO the room, giving this deterministic wall->rotation
# map. Both renderers rotate counter-clockwise in the room's y-up frame, so a
# front-south asset at these rotations shows its front inward in 2D and 3D alike.
# On E/W walls the rotation is 90/270, which swaps the placed footprint.
WALL_FACING_ROTATION: dict[str, int] = {"N": 0, "S": 180, "E": 270, "W": 90}


def asset_rect(asset: dict[str, Any]) -> tuple[float, float, float, float]:
    return (asset["x_ft"], asset["y_ft"], asset["w_ft"], asset["d_ft"])


def rects_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float], tol: float = TOL) -> bool:
    ix = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    iy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return ix > tol and iy > tol


def rect_within_room(rect: tuple[float, float, float, float], width: float, depth: float, tol: float = 0.01) -> bool:
    return rect[0] >= -tol and rect[1] >= -tol and rect[0] + rect[2] <= width + tol and rect[1] + rect[3] <= depth + tol


def dist_point_rect(px: float, py: float, rect: tuple[float, float, float, float]) -> float:
    cx = min(max(px, rect[0]), rect[0] + rect[2])
    cy = min(max(py, rect[1]), rect[1] + rect[3])
    return hypot(px - cx, py - cy)


def rect_clear_of(
    rect: tuple[float, float, float, float],
    assets: list[dict[str, Any]],
    exclude_ids: set[str],
    obstruction_anchors: set[str],
) -> bool:
    """True when no asset with an obstructing anchor overlaps rect (ceiling never obstructs)."""
    for asset in assets:
        if asset["id"] in exclude_ids or asset["anchor"] not in obstruction_anchors:
            continue
        if rects_overlap(rect, asset_rect(asset)):
            return False
    return True


def bed_frame(bed: dict[str, Any], width: float, depth: float, foot_min: float) -> dict[str, Any]:
    """Derive the bed's head/foot orientation, clearance rectangles and long axis.

    The head is the short side facing the nearest wall; the long axis runs head-to-foot.
    """
    bx, by, bw, bd = asset_rect(bed)
    if bd >= bw:  # long axis vertical (head/foot on N/S)
        south_gap, north_gap = by, depth - (by + bd)
        if north_gap <= south_gap:
            headwall, head_y, foot_y, head_gap = "N", by + bd, by, north_gap
        else:
            headwall, head_y, foot_y, head_gap = "S", by, by + bd, south_gap
        axis_pos = bx + bw / 2.0
        center_y = by + bd / 2.0
        head_center = (axis_pos, head_y)
        head_half = (bx, by, bw, center_y - by) if headwall == "S" else (bx, center_y, bw, (by + bd) - center_y)
        foot_rect = (bx, by - foot_min, bw, foot_min) if headwall == "N" else (bx, by + bd, bw, foot_min)
        long_sides = {"E": (bx + bw, by, None, bd), "W": (bx, by, None, bd)}  # depth filled per query
        return {
            "axis": "vertical",
            "axis_pos": axis_pos,
            "headwall": headwall,
            "headwall_length": width,
            "head_center": head_center,
            "head_half": head_half,
            "head_gap": head_gap,
            "foot_rect": foot_rect,
            "long_sides": long_sides,
            "near_edge": (bx, by) if headwall == "N" else (bx, by + bd),
        }
    # long axis horizontal (head/foot on E/W)
    west_gap, east_gap = bx, width - (bx + bw)
    if east_gap <= west_gap:
        headwall, head_x, foot_x, head_gap = "E", bx + bw, bx, east_gap
    else:
        headwall, head_x, foot_x, head_gap = "W", bx, bx + bw, west_gap
    axis_pos = by + bd / 2.0
    center_x = bx + bw / 2.0
    head_center = (head_x, axis_pos)
    head_half = (bx, by, center_x - bx, bd) if headwall == "W" else (center_x, by, (bx + bw) - center_x, bd)
    foot_rect = (bx - foot_min, by, foot_min, bd) if headwall == "E" else (bx + bw, by, foot_min, bd)
    long_sides = {"N": (bx, by + bd, bw, None), "S": (bx, by, bw, None)}
    return {
        "axis": "horizontal",
        "axis_pos": axis_pos,
        "headwall": headwall,
        "headwall_length": depth,
        "head_center": head_center,
        "head_half": head_half,
        "head_gap": head_gap,
        "foot_rect": foot_rect,
        "long_sides": long_sides,
        "near_edge": (bx, by) if headwall == "E" else (bx + bw, by),
    }


def side_clearance_rect(frame: dict[str, Any], side: str, depth: float) -> tuple[float, float, float, float]:
    base = frame["long_sides"][side]
    if frame["axis"] == "vertical":  # sides E/W, clearance grows in x
        x, y, _, bd = base
        return (x, y, depth, bd) if side == "E" else (x - depth, y, depth, bd)
    x, y, bw, _ = base  # sides N/S, clearance grows in y
    return (x, y, bw, depth) if side == "N" else (x, y - depth, bw, depth)


def bed_side_of_point(px: float, py: float, frame: dict[str, Any]) -> int:
    delta = (px - frame["axis_pos"]) if frame["axis"] == "vertical" else (py - frame["axis_pos"])
    if delta > TOL:
        return 1
    if delta < -TOL:
        return -1
    return 0


def door_hinge(door: dict[str, Any], width: float, depth: float) -> tuple[float, float]:
    wall, offset, dw = door["wall"], door["offset_ft"], door["width_ft"]
    low = offset - dw / 2.0
    if wall == "S":
        return (low, 0.0)
    if wall == "N":
        return (low, depth)
    if wall == "W":
        return (0.0, low)
    return (width, low)


def access_path_rect(door: dict[str, Any], bed: dict[str, Any], width: float, depth: float, path_width: float) -> tuple[float, float, float, float]:
    """A path_width-wide corridor from the door opening straight to the bed's near edge."""
    wall, offset = door["wall"], door["offset_ft"]
    bx, by, bw, bd = asset_rect(bed)
    half = path_width / 2.0
    if wall == "S":
        return (offset - half, 0.0, path_width, by)
    if wall == "N":
        return (offset - half, by + bd, path_width, depth - (by + bd))
    if wall == "W":
        return (0.0, offset - half, bx, path_width)
    return (bx + bw, offset - half, width - (bx + bw), path_width)


def validate_room(
    plan: dict[str, Any],
    rules: dict[str, Any],
    catalog: dict[str, Any],
    counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    room = plan["room"]
    assets: list[dict[str, Any]] = plan["assets"]
    width = float(room["width_ft"])
    depth = float(room["depth_ft"])
    door = room["door"]

    checks: list[dict[str, Any]] = []
    issues: list[str] = []

    def record(category: str, label: str, passed: bool, issue: str | None = None) -> bool:
        checks.append({"category": category, "label": label, "pass": bool(passed)})
        if not passed and issue:
            issues.append(issue)
        return bool(passed)

    room_rules = rules.get("room", {})
    clearances = rules.get("bed_clearances_ft", {})
    door_rules = rules.get("door", {})
    foot_min = float(clearances.get("foot_min", 5.0))
    transfer_min = float(clearances.get("transfer_side_min", 5.0))
    other_min = float(clearances.get("other_side_min", 4.0))
    head_min = float(clearances.get("head_to_wall_min", 1.0))
    head_max = float(clearances.get("head_to_wall_max", 1.5))
    path_width = float(door_rules.get("path_to_bed_min_width_ft", 4.0))

    catalog_by_type = {str(entry["type"]): entry for entry in catalog.get("assets", [])}
    # Requested per-type counts: the catalog default (count_required) unless the caller
    # overrides it. Type-specific checks below gate on these so a 0-count type is skipped
    # entirely rather than recorded as a failure, and a multi-count type checks every instance.
    required_counts = {asset_type: int(entry.get("count_required", 1)) for asset_type, entry in catalog_by_type.items()}
    if counts:
        required_counts.update({str(key): int(value) for key, value in counts.items()})
    beds = [asset for asset in assets if asset["type"] == "icu_bed"]
    frame = bed_frame(beds[0], width, depth, foot_min) if beds else None

    # --- room ------------------------------------------------------------------
    area = width * depth
    min_area = float(room_rules.get("min_clear_area_sf", 200))
    record(
        "room",
        f"Clear floor area {area:.0f} sf (min {min_area:.0f})",
        area + TOL >= min_area,
        f"Room clear area {area:.0f} sf is below the {min_area:.0f} sf minimum.",
    )
    min_headwall = float(room_rules.get("min_headwall_width_ft", 13))
    if frame is not None:
        headwall_len = frame["headwall_length"]
        record(
            "room",
            f"Headwall {frame['headwall']} length {headwall_len:.2f} ft (min {min_headwall:.0f})",
            headwall_len + TOL >= min_headwall,
            f"Headwall ({frame['headwall']}) is {headwall_len:.2f} ft; the {min_headwall:.0f} ft minimum is not met.",
        )
    else:
        record("room", f"Headwall length (min {min_headwall:.0f})", False, "No ICU bed placed; headwall cannot be evaluated.")

    # --- count -----------------------------------------------------------------
    actual_counts = Counter(asset["type"] for asset in assets)
    for asset_type in catalog_by_type:
        required = int(required_counts.get(asset_type, 0))
        actual = actual_counts.get(asset_type, 0)
        record(
            "count",
            f"{asset_type}: {actual}/{required}",
            actual == required,
            f"Expected {required} {asset_type}, found {actual}.",
        )

    # --- overlap ---------------------------------------------------------------
    static_assets = [asset for asset in assets if asset["anchor"] in {"floor", "wall"}]
    static_overlaps = 0
    for i in range(len(static_assets)):
        for j in range(i + 1, len(static_assets)):
            if rects_overlap(asset_rect(static_assets[i]), asset_rect(static_assets[j])):
                static_overlaps += 1
                issues.append(f"{static_assets[i]['id']} overlaps {static_assets[j]['id']}.")
    record("overlap", f"Static assets non-overlapping ({static_overlaps} conflicts)", static_overlaps == 0)

    mobile_assets = [asset for asset in assets if asset["anchor"] == "mobile"]
    bed_ids = {bed["id"] for bed in beds}
    mobile_conflicts = 0
    for mobile in mobile_assets:
        for other in assets:
            if other["id"] == mobile["id"] or other["anchor"] == "ceiling":
                continue
            if other["id"] in bed_ids:
                continue  # side-clearance overlap with the bed is not a collision
            if rects_overlap(asset_rect(mobile), asset_rect(other)):
                mobile_conflicts += 1
                issues.append(f"Mobile asset {mobile['id']} overlaps {other['id']}.")
    if frame is not None:
        for mobile in mobile_assets:
            if rects_overlap(asset_rect(mobile), frame["foot_rect"]):
                mobile_conflicts += 1
                issues.append(f"Mobile asset {mobile['id']} sits in the bed foot clearance.")
    record("overlap", f"Mobile assets clear of collisions and foot zone ({mobile_conflicts} conflicts)", mobile_conflicts == 0)

    # --- anchor ----------------------------------------------------------------
    wall_assets = [asset for asset in assets if asset["anchor"] == "wall"]
    unflush: list[str] = []
    for asset in wall_assets:
        wall = asset["wall"]
        x0, y0, x1, y1 = asset["x_ft"], asset["y_ft"], asset["x_ft"] + asset["w_ft"], asset["y_ft"] + asset["d_ft"]
        gap = {
            "N": abs(depth - y1),
            "S": abs(y0),
            "E": abs(width - x1),
            "W": abs(x0),
        }.get(wall)
        if gap is None or gap > ANCHOR_TOL:
            unflush.append(asset["id"])
    if unflush:
        issues.append("Wall assets not flush against their wall: " + ", ".join(sorted(unflush)) + ".")
    record("anchor", f"Wall assets flush to their wall ({len(unflush)} loose)", not unflush)

    # Every wall-anchored asset must SHOW ITS FRONT to the room. By the facing
    # convention (front = south edge at rotation 0) that fixes rotation_deg to a
    # deterministic value per wall (WALL_FACING_ROTATION); a casework/monitor/sink
    # at any other rotation faces the wall (the observed "casework doors into the
    # wall" defect). Ceiling, mobile, and floor assets have no wall and are exempt.
    misfacing: list[str] = []
    for asset in wall_assets:
        expected = WALL_FACING_ROTATION.get(asset["wall"])
        if expected is None or int(asset.get("rotation_deg", 0)) != expected:
            misfacing.append(asset["id"])
    if misfacing:
        issues.append(
            "Wall assets not facing the room (rotation_deg must be "
            + ", ".join(f"{wall}->{deg}" for wall, deg in WALL_FACING_ROTATION.items())
            + "): " + ", ".join(sorted(misfacing)) + "."
        )
    record("anchor", f"Wall assets face the room ({len(misfacing)} misfacing)", not misfacing)

    # --- clearance -------------------------------------------------------------
    if frame is not None:
        head_gap = frame["head_gap"]
        head_ok = head_min - 0.01 <= head_gap <= head_max + 0.01
        record(
            "clearance",
            f"Bed head-to-wall gap {head_gap:.2f} ft (target {head_min:.1f}-{head_max:.1f})",
            head_ok,
            f"Bed head-to-wall gap is {head_gap:.2f} ft; target {head_min:.1f}-{head_max:.1f} ft.",
        )
        foot_rect = frame["foot_rect"]
        foot_ok = rect_within_room(foot_rect, width, depth) and rect_clear_of(foot_rect, assets, bed_ids, {"floor", "wall", "mobile"})
        record(
            "clearance",
            f"Bed foot clearance {foot_min:.1f} ft clear",
            foot_ok,
            f"Bed foot needs {foot_min:.1f} ft clear of walls and assets.",
        )
        sides = list(frame["long_sides"].keys())

        def side_ok(side: str, depth_ft: float) -> bool:
            rect = side_clearance_rect(frame, side, depth_ft)
            return rect_within_room(rect, width, depth) and rect_clear_of(rect, assets, bed_ids, {"floor", "wall"})

        assign_a = side_ok(sides[0], transfer_min) and side_ok(sides[1], other_min)
        assign_b = side_ok(sides[1], transfer_min) and side_ok(sides[0], other_min)
        record(
            "clearance",
            f"Bed transfer/other side clearances ({transfer_min:.0f} ft / {other_min:.0f} ft)",
            assign_a or assign_b,
            f"Bed long sides need {transfer_min:.0f} ft on the transfer side and {other_min:.0f} ft on the other.",
        )
    else:
        record("clearance", "Bed head-to-wall gap", False)
        record("clearance", "Bed foot clearance", False)
        record("clearance", "Bed transfer/other side clearances", False)

    # --- access ----------------------------------------------------------------
    door_width = float(door["width_ft"])
    min_door = float(door_rules.get("min_clear_width_ft", 5.0))
    record(
        "access",
        f"Door clear width {door_width:.2f} ft (min {min_door:.1f})",
        door_width + 0.01 >= min_door,
        f"Door clear width {door_width:.2f} ft is below the {min_door:.1f} ft minimum.",
    )
    if frame is not None:
        path_rect = access_path_rect(door, beds[0], width, depth, path_width)
        path_ok = (
            rect_within_room(path_rect, width, depth)
            and path_rect[2] > TOL
            and path_rect[3] > TOL
            and rect_clear_of(path_rect, assets, bed_ids, {"floor", "wall"})
            and rects_overlap(path_rect, frame["foot_rect"])
        )
        record(
            "access",
            f"Clear {path_width:.0f} ft path from door to bed",
            path_ok,
            f"A {path_width:.0f} ft clear path from the door to the bed foot is blocked.",
        )
    else:
        record("access", "Clear path from door to bed", False)

    sinks = [asset for asset in assets if asset["type"] == "handwash_sink"]
    requested_sinks = int(required_counts.get("handwash_sink", 0))
    max_sink = float(rules.get("sink", {}).get("max_distance_from_door_ft", 8.0))
    door_center = door_center_point(door, width, depth)
    if requested_sinks >= 1:
        if sinks:
            sink_ok = True
            for sink in sinks:
                sink_center = asset_center(sink)
                distance = hypot(sink_center[0] - door_center[0], sink_center[1] - door_center[1])
                if distance > max_sink + 0.01:
                    sink_ok = False
                    issues.append(f"Handwash sink {sink['id']} is {distance:.2f} ft from the door; the maximum is {max_sink:.0f} ft.")
            record("access", f"Handwash sink(s) within {max_sink:.0f} ft of the door", sink_ok)
        else:
            record("access", "Handwash sink near door", False, "No handwash sink placed.")
    # requested_sinks == 0: no handwash sink in this program, so skip the sink-near-door check.

    # --- equipment -------------------------------------------------------------
    # Every equipment check gates on the REQUESTED count: a 0-count type records no check
    # (graceful skip), a 1-count boom drops the opposite-sides requirement, and multi-count
    # monitors/iv poles require EACH placed instance to satisfy its distance check.
    boom_rules = rules.get("booms", {})
    booms = [asset for asset in assets if asset["type"] == "ceiling_boom"]
    requested_booms = int(required_counts.get("ceiling_boom", 0))
    boom_mount_max = float(boom_rules.get("mount_max_distance_from_bed_head_ft", 3.0))
    coverage_radius = float(boom_rules.get("coverage_radius_ft", 5.0))
    if requested_booms >= 1:
        if frame is not None and booms:
            head_half = frame["head_half"]
            centers = [asset_center(boom) for boom in booms]
            near = all(dist_point_rect(cx, cy, head_half) <= boom_mount_max + 0.01 for cx, cy in centers)
            covers = all(dist_point_rect(cx, cy, head_half) <= coverage_radius + 0.01 for cx, cy in centers)
            boom_ok = near and covers
            if requested_booms >= 2:
                # Two-boom layouts must flank the head on opposite sides of the long axis.
                boom_ok = boom_ok and len(centers) >= 2 and bed_side_of_point(*centers[0], frame) * bed_side_of_point(*centers[1], frame) < 0
            side_note = " on opposite sides" if requested_booms >= 2 else ""
            record(
                "equipment",
                f"{requested_booms} boom(s) flank the bed head",
                boom_ok,
                f"Booms must reach and cover the bed head{side_note}.",
            )
        else:
            record("equipment", f"{requested_booms} boom(s) flank the bed head", False, "Booms are missing or the bed is unplaced.")
    # requested_booms == 0: no boom checks recorded.

    monitors = [asset for asset in assets if asset["type"] == "patient_monitor"]
    requested_monitors = int(required_counts.get("patient_monitor", 0))
    monitor_max = float(rules.get("monitor", {}).get("max_distance_from_bed_head_ft", 6.0))
    equipment_side = 0
    if requested_monitors >= 1:
        if frame is not None and monitors:
            head_center = frame["head_center"]
            monitor_ok = True
            for monitor in monitors:
                center = asset_center(monitor)
                side = bed_side_of_point(*center, frame)
                distance = hypot(center[0] - head_center[0], center[1] - head_center[1])
                if not (distance <= monitor_max + 0.01 and side != 0):
                    monitor_ok = False
            # The first monitor defines the equipment side used by the visitor-chair zoning check.
            equipment_side = bed_side_of_point(*asset_center(monitors[0]), frame)
            record(
                "equipment",
                f"Monitor(s) within {monitor_max:.0f} ft of the bed head on the equipment side",
                monitor_ok,
                f"Each patient monitor must be within {monitor_max:.0f} ft of the bed head on a defined equipment side.",
            )
        else:
            record("equipment", "Monitor near bed head on equipment side", False, "Patient monitor missing or bed unplaced.")
    # requested_monitors == 0: no monitor check; equipment_side stays undefined.

    iv_poles = [asset for asset in assets if asset["type"] == "iv_pole"]
    requested_iv = int(required_counts.get("iv_pole", 0))
    iv_max = float(rules.get("iv_pole", {}).get("max_distance_from_bed_head_ft", 3.0))
    if requested_iv >= 1:
        if frame is not None and iv_poles:
            head_half = frame["head_half"]
            iv_ok = all(dist_point_rect(*asset_center(iv), head_half) <= iv_max + 0.01 for iv in iv_poles)
            record(
                "equipment",
                f"IV pole(s) within {iv_max:.0f} ft of the bed head",
                iv_ok,
                f"Each IV pole must be within {iv_max:.0f} ft of the bed head.",
            )
        else:
            record("equipment", "IV pole near bed head", False, "IV pole missing or bed unplaced.")
    # requested_iv == 0: no IV pole check recorded.

    # --- zoning ----------------------------------------------------------------
    # EACH placed visitor chair must sit opposite the equipment side; the check is skipped
    # when no chair is requested or no monitor defines an equipment side to compare against.
    chairs = [asset for asset in assets if asset["type"] == "visitor_chair"]
    requested_chairs = int(required_counts.get("visitor_chair", 0))
    if requested_chairs >= 1 and frame is not None and equipment_side != 0:
        chair_sides = [bed_side_of_point(*asset_center(chair), frame) for chair in chairs]
        chair_ok = bool(chair_sides) and all(side != 0 and side == -equipment_side for side in chair_sides)
        record(
            "zoning",
            "Visitor chair(s) on the side opposite the equipment zone",
            chair_ok,
            "Every visitor chair must sit on the opposite side of the bed axis from the equipment side.",
        )
    # requested_chairs == 0 or equipment side undefined: no visitor-side check recorded.

    caseworks = [asset for asset in assets if asset["type"] == "casework"]
    requested_casework = int(required_counts.get("casework", 0))
    if requested_casework >= 1 and frame is not None:
        if caseworks:
            hinge = door_hinge(door, width, depth)
            casework_ok = True
            for casework in caseworks:
                rect = asset_rect(casework)
                not_headwall = casework["wall"] != frame["headwall"]
                clear_of_swing = dist_point_rect(hinge[0], hinge[1], rect) >= door_width - 0.01
                if not (not_headwall and clear_of_swing):
                    casework_ok = False
            record(
                "zoning",
                "Casework off the headwall and clear of the door swing",
                casework_ok,
                "Every casework run must stay off the headwall and clear of the door swing.",
            )
        else:
            record("zoning", "Casework placement", False, "Casework missing.")
    # requested_casework == 0: no casework zoning check recorded.

    # --- score -----------------------------------------------------------------
    per_category: dict[str, list[bool]] = {}
    for check in checks:
        per_category.setdefault(check["category"], []).append(check["pass"])
    total_weight = 0.0
    earned = 0.0
    for category, results in per_category.items():
        weight = CATEGORY_WEIGHTS.get(category, 0)
        if not weight or not results:
            continue
        total_weight += weight
        earned += weight * (sum(1 for value in results if value) / len(results))
    score = round(100 * earned / total_weight) if total_weight else 0

    return {
        "score": score,
        "checks": checks,
        "issues": list(dict.fromkeys(issues))[:100],
        "summary": {
            "assets_placed": len(assets),
            "checks_passed": sum(1 for check in checks if check["pass"]),
            "checks_total": len(checks),
            "rule_profile": rules.get("profile", "icu-room-schematic-v1"),
        },
    }


def asset_center(asset: dict[str, Any]) -> tuple[float, float]:
    return (asset["x_ft"] + asset["w_ft"] / 2.0, asset["y_ft"] + asset["d_ft"] / 2.0)


def door_center_point(door: dict[str, Any], width: float, depth: float) -> tuple[float, float]:
    wall, offset = door["wall"], door["offset_ft"]
    if wall == "S":
        return (offset, 0.0)
    if wall == "N":
        return (offset, depth)
    if wall == "W":
        return (0.0, offset)
    return (width, offset)
