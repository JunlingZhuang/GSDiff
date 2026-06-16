# -*- coding: utf-8 -*-
"""Tool ③ render_plan: Plan JSON -> deterministic colour-block PNG.

I/O contract:
    render_plan(plan, px_per_mm=0.05, wall_px=8, margin_px=40) -> PIL.Image (RGB)

Uses exactly the shared palette (schema/palette.py): room fills from ROOM_RGB,
black outlines, white background — i.e. the same visual language we ask Gemini
for, which is what makes render->parse a meaningful round trip.
"""
from __future__ import annotations

from PIL import Image, ImageDraw

from hfagent.schema.palette import BACKGROUND_RGB, ROOM_RGB, WALL_RGB
from hfagent.schema.plan import Plan

DEFAULT_PX_PER_MM = 0.05  # 20 mm per pixel -> a 20 m building is 1000 px


def render_plan(
    plan: Plan,
    px_per_mm: float = DEFAULT_PX_PER_MM,
    wall_px: int = 8,
    margin_px: int = 40,
) -> Image.Image:
    minx, miny, maxx, maxy = plan.bounds()
    w = int((maxx - minx) * px_per_mm) + 2 * margin_px
    h = int((maxy - miny) * px_per_mm) + 2 * margin_px
    img = Image.new("RGB", (max(w, 1), max(h, 1)), BACKGROUND_RGB)
    draw = ImageDraw.Draw(img)

    def to_px(pt: tuple[float, float]) -> tuple[float, float]:
        return ((pt[0] - minx) * px_per_mm + margin_px, (pt[1] - miny) * px_per_mm + margin_px)

    for room in plan.rooms:
        pts = [to_px(p) for p in room.polygon]
        draw.polygon(pts, fill=ROOM_RGB.get(room.type, (128, 128, 128)))
    # outlines second so shared edges read as walls
    for room in plan.rooms:
        pts = [to_px(p) for p in room.polygon]
        draw.line(pts + [pts[0]], fill=WALL_RGB, width=wall_px, joint="curve")
    # doors last: a white gap across the party wall at the hung position
    walls = {w.id: w for w in plan.walls}
    for door in plan.doors:
        w = walls.get(door.wall_id)
        if not w:
            continue
        (x0, y0), (x1, y1) = w.start, w.end
        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 or 1.0
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        mx = x0 + (x1 - x0) * door.position
        my = y0 + (y1 - y0) * door.position
        half = door.width / 2
        a = to_px((mx - ux * half, my - uy * half))
        b = to_px((mx + ux * half, my + uy * half))
        draw.line([a, b], fill=(255, 255, 255), width=wall_px + 2)
    return img


def save_render(plan: Plan, path: str, **kw) -> None:
    render_plan(plan, **kw).save(path)
