"""Render cycle polygons to a PIL Image."""

from PIL import Image, ImageDraw

from app.config import ROOM_COLORS, RESOLUTION


def render_floorplan(simple_cycles, simple_cycles_semantics, resolution=RESOLUTION):
    """Return a resolution x resolution RGB PIL Image of the floorplan."""
    img = Image.new("RGB", (resolution, resolution), (255, 255, 255))
    if not simple_cycles:
        return img

    draw = ImageDraw.Draw(img)

    for poly_i, polygon in enumerate(simple_cycles):
        coords = [(p[0], p[1]) for p in polygon]
        sem = simple_cycles_semantics[poly_i]
        colour = ROOM_COLORS.get(sem, (200, 200, 200))
        draw.polygon(coords, fill=colour, outline=None)

    for polygon in simple_cycles:
        for pt_i, point in enumerate(polygon):
            if pt_i < len(polygon) - 1:
                p1 = (point[0], point[1])
                p2 = (polygon[pt_i + 1][0], polygon[pt_i + 1][1])
                draw.rectangle((p1[0] - 3, p1[1] - 3, p1[0] + 3, p1[1] + 3), fill=(150, 150, 150))
                draw.line((p1[0], p1[1], p2[0], p2[1]), fill=(150, 150, 150), width=7)

    return img
