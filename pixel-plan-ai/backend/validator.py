from __future__ import annotations

from collections import Counter
from typing import Any

from constants import area_for
from healthcare_rules import load_healthcare_rules, room_rule, uses_healthcare_rules


FEET_PER_METER = 3.280839895
SQUARE_FEET_PER_SQUARE_METER = 10.763910417


def pair_key(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((first, second)))


def has_non_rectangular_footprint(footprint: list[int], width: int, height: int) -> bool:
    row_widths = [
        sum(bool(footprint[y * width + x]) for x in range(width))
        for y in range(height)
    ]
    column_heights = [
        sum(bool(footprint[y * width + x]) for y in range(height))
        for x in range(width)
    ]
    active_row_widths = {value for value in row_widths if value > 0}
    active_column_heights = {value for value in column_heights if value > 0}
    return len(active_row_widths) > 1 or len(active_column_heights) > 1


def bounds_contains(outer: dict[str, int], inner: dict[str, int]) -> bool:
    outer_right = outer["x"] + outer["width"]
    outer_bottom = outer["y"] + outer["height"]
    inner_right = inner["x"] + inner["width"]
    inner_bottom = inner["y"] + inner["height"]
    contains = (
        outer["x"] <= inner["x"]
        and outer["y"] <= inner["y"]
        and outer_right >= inner_right
        and outer_bottom >= inner_bottom
    )
    strictly_larger = (
        outer["x"] < inner["x"]
        or outer["y"] < inner["y"]
        or outer_right > inner_right
        or outer_bottom > inner_bottom
    )
    return contains and strictly_larger


def is_corner_inset(outer: dict[str, int], inner: dict[str, int]) -> bool:
    if not bounds_contains(outer, inner):
        return False
    touches_horizontal_edge = inner["x"] == outer["x"] or inner["x"] + inner["width"] == outer["x"] + outer["width"]
    touches_vertical_edge = inner["y"] == outer["y"] or inner["y"] + inner["height"] == outer["y"] + outer["height"]
    return touches_horizontal_edge and touches_vertical_edge


def largest_owned_rectangle(
    plan: dict[str, Any],
    room_index: int,
    bounds: dict[str, int],
) -> tuple[int, int]:
    """Return width and height of the largest rectangular clear zone in a room mask."""

    histogram = [0] * bounds["width"]
    best_area = 0
    best_width = 0
    best_height = 0
    for y in range(bounds["y"], bounds["y"] + bounds["height"]):
        for local_x in range(bounds["width"]):
            x = bounds["x"] + local_x
            histogram[local_x] = histogram[local_x] + 1 if plan["cells"][y * plan["width"] + x] == room_index else 0
        stack: list[tuple[int, int]] = []
        for local_x in range(bounds["width"] + 1):
            current_height = histogram[local_x] if local_x < bounds["width"] else 0
            start = local_x
            while stack and stack[-1][1] > current_height:
                rectangle_start, rectangle_height = stack.pop()
                rectangle_width = local_x - rectangle_start
                rectangle_area = rectangle_width * rectangle_height
                if rectangle_area > best_area:
                    best_area = rectangle_area
                    best_width = rectangle_width
                    best_height = rectangle_height
                start = rectangle_start
            if current_height and (not stack or stack[-1][1] < current_height):
                stack.append((start, current_height))
    return best_width, best_height


def validate_plan(plan: dict[str, Any], program: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    checks: list[dict[str, Any]] = []
    expected_counts = {room["type"]: room["count"] for room in program["rooms"]}
    actual_counts = Counter(room["type"] for room in plan["rooms"])
    count_hits = 0
    count_total = 0
    for room_type, expected in expected_counts.items():
        actual = actual_counts[room_type]
        count_hits += min(expected, actual)
        count_total += expected
        passed = actual == expected
        checks.append({"category": "count", "label": f"{room_type}: {actual}/{expected}", "pass": passed})
        if not passed:
            issues.append(f"Expected {expected} {room_type} spaces, generated {actual}.")

    observed_pairs: set[tuple[str, str]] = set()
    spatial_neighbor_indexes: list[set[int]] = [set() for _ in plan["rooms"]]
    for y in range(plan["height"]):
        for x in range(plan["width"]):
            here = plan["cells"][y * plan["width"] + x]
            if here < 0:
                continue
            if x + 1 < plan["width"]:
                right = plan["cells"][y * plan["width"] + x + 1]
                if right >= 0 and right != here:
                    observed_pairs.add(pair_key(plan["rooms"][here]["type"], plan["rooms"][right]["type"]))
                    spatial_neighbor_indexes[here].add(right)
                    spatial_neighbor_indexes[right].add(here)
            if y + 1 < plan["height"]:
                below = plan["cells"][(y + 1) * plan["width"] + x]
                if below >= 0 and below != here:
                    observed_pairs.add(pair_key(plan["rooms"][here]["type"], plan["rooms"][below]["type"]))
                    spatial_neighbor_indexes[here].add(below)
                    spatial_neighbor_indexes[below].add(here)

    adjacency_hits = 0
    for first, second in program["adjacency"]:
        passed = pair_key(first, second) in observed_pairs
        adjacency_hits += int(passed)
        checks.append({"category": "adjacency", "label": f"{first} <-> {second}", "pass": passed})
        if not passed:
            issues.append(f"Required adjacency is missing: {first} <-> {second}.")

    healthcare_profile = uses_healthcare_rules(program)
    profile = load_healthcare_rules() if healthcare_profile else None
    targets = {room["type"]: area_for(room) for room in program["rooms"]}
    cell_area = plan["meters_per_cell"] ** 2
    cell_area_ft2 = cell_area * SQUARE_FEET_PER_SQUARE_METER
    cell_dimension_ft = plan["meters_per_cell"] * FEET_PER_METER
    area_rows: list[dict[str, Any]] = []
    area_quality = 0.0
    area_room_count = 0
    area_compliant = 0
    occupiable = [room for room in plan["rooms"] if room["type"] not in {"corridor", "circulation"}]

    for room in occupiable:
        if room["type"] not in targets:
            continue
        target = targets[room["type"]]
        actual = room["pixel_count"] * cell_area
        error = abs(actual - target) / target
        rule = room_rule(room["type"]) if healthcare_profile else None
        tolerance = float(rule.get("area_tolerance_fraction", 0.35)) if rule else 0.35
        minimum_area_ft2 = float(rule.get("minimum_area_ft2", 0.0)) if rule else 0.0
        actual_ft2 = actual * SQUARE_FEET_PER_SQUARE_METER
        minimum_ok = actual_ft2 + cell_area_ft2 * 0.5 >= minimum_area_ft2
        area_pass = error <= tolerance and minimum_ok
        area_compliant += int(area_pass)
        area_quality += max(0.0, 1.0 - error)
        area_room_count += 1
        area_rows.append(
            {
                "id": room["id"],
                "type": room["type"],
                "target_m2": round(target, 1),
                "actual_m2": round(actual, 1),
                "target_ft2": round(target * SQUARE_FEET_PER_SQUARE_METER),
                "actual_ft2": round(actual_ft2),
                "target_cells": round(target / cell_area, 1),
                "actual_cells": room["pixel_count"],
                "error_percent": round(error * 100),
                "tolerance_percent": round(tolerance * 100),
                "pass": area_pass,
            }
        )
        if not area_pass:
            issues.append(
                f"{room['id']} area is {actual_ft2:.0f} sf; target {target * SQUARE_FEET_PER_SQUARE_METER:.0f} sf "
                f"with ±{round(tolerance * 100)}% tolerance and {minimum_area_ft2:.0f} sf profile minimum."
            )
    if area_room_count:
        checks.append(
            {
                "category": "area",
                "label": f"U.S. room-area range: {area_compliant}/{area_room_count}",
                "pass": area_compliant == area_room_count,
            }
        )

    room_by_id = {room["id"]: room for room in plan["rooms"]}
    connected_types: dict[str, set[str]] = {room["id"]: set() for room in plan["rooms"]}
    connected_room_ids: dict[str, set[str]] = {room["id"]: set() for room in plan["rooms"]}
    door_rooms: set[str] = set()
    for door in plan["doors"]:
        from_room = door["from_room"]
        to_room = door["to_room"]
        door_rooms.add(from_room)
        if to_room:
            door_rooms.add(to_room)
            connected_types[from_room].add(room_by_id[to_room]["type"])
            connected_types[to_room].add(room_by_id[from_room]["type"])
            connected_room_ids[from_room].add(to_room)
            connected_room_ids[to_room].add(from_room)
        else:
            connected_types[from_room].add("outside")

    rooms_with_doors = sum(room["id"] in door_rooms for room in occupiable)
    room_indexes = {room["id"]: index for index, room in enumerate(plan["rooms"])}
    for room in occupiable:
        if room["id"] not in door_rooms:
            issues.append(f"{room['id']} has no door.")
    checks.append(
        {
            "category": "doors",
            "label": f"Rooms with doors: {rooms_with_doors}/{len(occupiable)}",
            "pass": rooms_with_doors == len(occupiable),
        }
    )
    has_entrance = any(door["to_room"] is None for door in plan["doors"])
    checks.append({"category": "doors", "label": "Main entrance", "pass": has_entrance})
    if not has_entrance:
        issues.append("No exterior entrance door was generated.")

    access_compliant = 0
    for room in occupiable:
        rule = room_rule(room["type"]) if healthcare_profile else None
        if not rule:
            access_compliant += int(room["id"] in door_rooms)
            continue
        actual_connections = connected_types.get(room["id"], set())
        allowed_connections = set(rule.get("allowed_door_connections", []))
        allowed_access = bool(actual_connections & allowed_connections)
        corridor_access = bool(actual_connections & {"corridor", "circulation"})
        direct_required = bool(rule.get("requires_direct_corridor_access"))
        access_pass = allowed_access and (corridor_access or not direct_required)
        access_compliant += int(access_pass)
        if not access_pass:
            requirement = "direct corridor access" if direct_required else f"access from {sorted(allowed_connections)}"
            issues.append(f"{room['id']} requires {requirement}; connected to {sorted(actual_connections)}.")
    if occupiable:
        checks.append(
            {
                "category": "access",
                "label": f"Functional room access: {access_compliant}/{len(occupiable)}",
                "pass": access_compliant == len(occupiable),
            }
        )

    proportion_compliant = 0
    for room in occupiable:
        rule = room_rule(room["type"]) if healthcare_profile else None
        if not rule:
            proportion_compliant += 1
            continue
        bounds = room["bounds"]
        clear_width_cells = bounds["width"]
        clear_height_cells = bounds["height"]
        compactness = room["pixel_count"] / max(1, bounds["width"] * bounds["height"])
        en_suite_rule = rule.get("en_suite_module") if room["type"] == "patient_room" else None
        if isinstance(en_suite_rule, dict) and en_suite_rule.get("validate_combined_module_compactness"):
            inset_toilets = [
                room_by_id[connected_id]
                for connected_id in connected_room_ids.get(room["id"], set())
                if room_by_id[connected_id]["type"] == en_suite_rule.get("allowed_inset_type")
                and bounds_contains(bounds, room_by_id[connected_id]["bounds"])
            ]
            if inset_toilets:
                clear_width_cells, clear_height_cells = largest_owned_rectangle(
                    plan,
                    room_indexes[room["id"]],
                    bounds,
                )
                module_pixels = room["pixel_count"] + sum(inset["pixel_count"] for inset in inset_toilets)
                compactness = module_pixels / max(1, bounds["width"] * bounds["height"])
        width_ft = clear_width_cells * cell_dimension_ft
        height_ft = clear_height_cells * cell_dimension_ft
        minimum_dimension_ft = min(width_ft, height_ft)
        aspect_ratio = max(width_ft, height_ft) / max(minimum_dimension_ft, 1e-9)
        required_minimum = float(rule["minimum_clear_dimension_ft"])
        maximum_aspect = float(rule["hard_max_aspect_ratio"])
        minimum_compactness = float(rule["minimum_compactness"])
        dimension_ok = minimum_dimension_ft + cell_dimension_ft * 0.5 >= required_minimum
        aspect_ok = aspect_ratio <= maximum_aspect + 1e-9
        compactness_ok = compactness + 1e-9 >= minimum_compactness
        proportion_pass = dimension_ok and aspect_ok and compactness_ok
        proportion_compliant += int(proportion_pass)
        if not proportion_pass:
            issues.append(
                f"{room['id']} proportion fails U.S. schematic range: {width_ft:.1f} ft x {height_ft:.1f} ft, "
                f"aspect {aspect_ratio:.2f} (max {maximum_aspect:.2f}), compactness {compactness:.2f} "
                f"(min {minimum_compactness:.2f}), minimum dimension {required_minimum:.1f} ft."
            )
    if occupiable:
        checks.append(
            {
                "category": "proportion",
                "label": f"Reasonable room proportions: {proportion_compliant}/{len(occupiable)}",
                "pass": proportion_compliant == len(occupiable),
            }
        )

    building_type = program["building_type"].lower()
    expected_en_suite_count = 0
    private_en_suite_count = 0
    inboard_en_suite_count = 0
    if healthcare_profile and ("inpatient" in building_type or "ward" in building_type):
        expected_en_suite_count = min(expected_counts.get("patient_room", 0), expected_counts.get("toilet", 0))
        if expected_en_suite_count:
            patient_rooms = [room for room in occupiable if room["type"] == "patient_room"]
            for patient_room in patient_rooms:
                connected_toilets = [
                    room_by_id[connected_id]
                    for connected_id in connected_room_ids.get(patient_room["id"], set())
                    if room_by_id[connected_id]["type"] == "toilet"
                ]
                private_toilets = [
                    toilet
                    for toilet in connected_toilets
                    if sum(
                        room_by_id[neighbor_id]["type"] == "patient_room"
                        for neighbor_id in connected_room_ids.get(toilet["id"], set())
                    ) == 1
                ]
                if len(private_toilets) == 1:
                    private_en_suite_count += 1
                    toilet_index = room_indexes[private_toilets[0]["id"]]
                    if any(
                        plan["rooms"][neighbor_index]["type"] in {"corridor", "circulation"}
                        for neighbor_index in spatial_neighbor_indexes[toilet_index]
                    ):
                        inboard_en_suite_count += 1
                else:
                    issues.append(f"{patient_room['id']} does not have one private directly connected en-suite toilet.")
            checks.append(
                {
                    "category": "zoning",
                    "label": f"Private en-suite patient modules: {private_en_suite_count}/{expected_en_suite_count}",
                    "pass": private_en_suite_count == expected_en_suite_count,
                }
            )
            checks.append(
                {
                    "category": "configuration",
                    "label": f"Preferred inboard en-suite placement: {inboard_en_suite_count}/{expected_en_suite_count}",
                    "pass": inboard_en_suite_count == expected_en_suite_count,
                }
            )

    allowed_containment = {
        tuple(pair)
        for pair in (profile.get("allowed_module_containment", []) if profile else [])
        if isinstance(pair, list) and len(pair) == 2
    }
    containment_violations: list[str] = []
    for outer in occupiable:
        for inner in occupiable:
            if outer["id"] == inner["id"] or not bounds_contains(outer["bounds"], inner["bounds"]):
                continue
            if (outer["type"], inner["type"]) in allowed_containment:
                continue
            containment_violations.append(f"{inner['id']} is embedded in the bounding module of {outer['id']}.")
    containment_violations = list(dict.fromkeys(containment_violations))
    checks.append(
        {
            "category": "zoning",
            "label": "No improper room nesting",
            "pass": not containment_violations,
        }
    )
    issues.extend(containment_violations)

    footprint = plan.get("footprint") or [1] * len(plan["cells"])
    footprint_cells = sum(bool(cell) for cell in footprint)
    covered = sum(cell >= 0 and bool(footprint[index]) for index, cell in enumerate(plan["cells"]))
    coverage = covered / footprint_cells if footprint_cells else 0.0
    checks.append({"category": "coverage", "label": f"Footprint coverage: {round(coverage * 100)}%", "pass": coverage >= 0.85})

    count_score = count_hits / count_total * 20 if count_total else 20
    adjacency_score = adjacency_hits / len(program["adjacency"]) * 20 if program["adjacency"] else 20
    area_score = area_quality / area_room_count * 20 if area_room_count else 20
    general_door_ratio = rooms_with_doors / len(occupiable) if occupiable else 1.0
    access_ratio = access_compliant / len(occupiable) if occupiable else 1.0
    door_score = general_door_ratio * 8 + access_ratio * 7 + (5 if has_entrance else 0)
    proportion_ratio = proportion_compliant / len(occupiable) if occupiable else 1.0
    geometry_score = proportion_ratio * 7 + (3 if not containment_violations else 0)

    is_tower = "tower" in program["building_type"].lower()
    if is_tower:
        non_rectangular = has_non_rectangular_footprint(footprint, plan["width"], plan["height"])
        corridor_bounds = [room["bounds"] for room in plan["rooms"] if room["type"] in {"corridor", "circulation"}]
        has_vertical = any(bounds["height"] >= bounds["width"] * 1.4 for bounds in corridor_bounds)
        has_horizontal = any(bounds["width"] >= bounds["height"] * 1.4 for bounds in corridor_bounds)
        patient_indexes = {index for index, room in enumerate(plan["rooms"]) if room["type"] == "patient_room"}
        perimeter_patients: set[int] = set()
        for y in range(plan["height"]):
            for x in range(plan["width"]):
                offset = y * plan["width"] + x
                room_index = plan["cells"][offset]
                if room_index not in patient_indexes:
                    continue
                neighbors = ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
                if any(
                    nx < 0
                    or nx >= plan["width"]
                    or ny < 0
                    or ny >= plan["height"]
                    or not footprint[ny * plan["width"] + nx]
                    for nx, ny in neighbors
                ):
                    perimeter_patients.add(room_index)
        perimeter_ratio = len(perimeter_patients) / len(patient_indexes) if patient_indexes else 1.0
        checks.extend(
            [
                {"category": "massing", "label": "Explicit non-rectangular tower footprint", "pass": non_rectangular},
                {"category": "circulation", "label": "Multi-axis corridor organization", "pass": has_vertical and has_horizontal},
                {"category": "perimeter", "label": f"Perimeter patient rooms: {round(perimeter_ratio * 100)}%", "pass": perimeter_ratio >= 0.7},
            ]
        )
        if not non_rectangular:
            issues.append("Tower plan uses a full rectangular canvas instead of an explicit winged footprint.")
        if not (has_vertical and has_horizontal):
            issues.append("Tower circulation does not include both horizontal and vertical corridor axes.")
        if perimeter_ratio < 0.7:
            issues.append("Too few patient rooms reach the building perimeter.")
        coverage_score = min(1.0, coverage / 0.9) * 4
        coverage_score += 2 if non_rectangular else 0
        coverage_score += 2 if has_vertical and has_horizontal else 0
        coverage_score += min(1.0, perimeter_ratio / 0.7) * 2
    else:
        coverage_score = min(1.0, coverage / 0.9) * 10

    score = max(0, min(100, round(count_score + adjacency_score + area_score + door_score + geometry_score + coverage_score)))
    return {
        "score": score,
        "checks": checks,
        "issues": list(dict.fromkeys(issues))[:100],
        "areas": area_rows,
        "summary": {
            "expected_rooms": count_total,
            "generated_rooms": len(plan["rooms"]),
            "doors": len(plan["doors"]),
            "rooms_with_doors": rooms_with_doors,
            "required_adjacencies": len(program["adjacency"]),
            "satisfied_adjacencies": adjacency_hits,
            "coverage_percent": round(coverage * 100),
            "area_compliant_rooms": area_compliant,
            "area_compliance_percent": round(area_compliant / area_room_count * 100) if area_room_count else 100,
            "proportion_compliant_rooms": proportion_compliant,
            "access_compliant_rooms": access_compliant,
            "zoning_violations": len(containment_violations),
            "private_en_suite_modules": private_en_suite_count,
            "inboard_en_suite_modules": inboard_en_suite_count,
            "rule_profile": profile["profile"] if profile else "generic-schematic",
        },
    }
