from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image"

# Drawing rules ported from hfagent's linework prompt (hfagent/tools/
# floor_plan_generator.py `_PLAN_RULES`, commit 3f1a3fa). Text duplication is
# deliberate: the two projects stay decoupled and only share prompt wording.
PLAN_DRAWING_RULES = """VIEW
- Strictly TOP-DOWN orthographic plan view (looking straight down, bird's-eye).
- NO perspective, NO 3D, NO isometric/axonometric, NO camera tilt or vanishing points.
- Do NOT rotate or skew the building. Its outer walls run purely horizontal and vertical, axis-aligned to the image edges.

WALLS
- Draw every wall as a SOLID BLACK FILLED band (poché) — continuous black fill between and around rooms, NOT a thin single outline and NOT a hairline.
- ONE fixed, uniform thickness for EVERY wall — exterior walls, interior partitions and party walls are ALL the same thickness. Do NOT make exterior or load-bearing walls thicker than the others.
- Keep walls THIN and uniform: a thin filled black band about 12 px wide on the ~1400 px-wide output (≈1% of the image width) — never a hairline, never thicker than ~20 px.
- Only right angles: no diagonal, curved or angled walls. Every room is fully enclosed.

LAYOUT
- Every room is rectangular (or a clean L-shape) with orthogonal corners.
- Rooms completely TILE the building footprint — no gaps, no overlaps, no leftover background slivers inside the outer walls.
- Organise rooms along a clear corridor/circulation spine; size rooms roughly by the areas given (the corridor is a long thin band).
- One single connected building footprint, not scattered blocks.

DOORS (required — they are read back from this drawing)
- Every door is the standard architect's door symbol: a thin BLACK quarter-circle swing arc sweeping across the wall opening, together with its straight door-leaf line from the hinge. The quarter-circle arc is what makes an opening a door — an opening in a wall without its quarter-circle arc must not appear anywhere in the drawing.
- The door-leaf line is a thin plain GREY line — never a black-filled bar; the swing ARC itself stays a thin black line.
- Draw the full arc at every door, including the smallest toilet and storage rooms.
- Every enclosed room has at least one door to a corridor / circulation space so it is reachable.
- Put doors only where two spaces should connect.

LABELS
- One label per room, centred, small plain black text, the EXACT instance name from the program below.
- No other text anywhere.
- Draw ONLY the room instances listed below — nothing extra. Every enclosed space in the drawing
  must carry exactly one label from the list; do not add unlabeled closets, voids, shafts, or
  leftover enclosed pockets. If geometry does not fill the massing, enlarge listed rooms or the
  corridor instead of inventing space.

EXCLUDE — keep it a clean SCHEMATIC plan, NOT a construction / working drawing. Do NOT draw any of:
- furniture, fixtures, equipment, sanitary ware, beds, sinks, desks;
- windows of any kind;
- decorative door hardware or annotations (door frames, thresholds, hinge marks, door tags or numbers);
- dimension lines or strings, grid / column lines, section / elevation / detail markers;
- hatching, fill patterns, textures, gradients or shadows (walls stay flat solid black);
- schedules, legends, keynotes, callouts, scale bars, north arrows, title blocks.
- Background outside the building is pure white and empty."""

VARIANT_HINTS = (
    "Layout variant A: choose the massing and corridor arrangement you consider most functional.",
    "Layout variant B: explore a DIFFERENT massing than the obvious one — e.g. a loop or L/T-shaped circulation spine.",
    "Layout variant C: explore yet another arrangement — vary the wing count, corridor topology or room banding.",
)


def room_instance_names(program: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for room in program.get("rooms", []):
        count = int(room.get("count", 1))
        room_type = str(room.get("type", "room"))
        if count == 1:
            names.append(room_type)
        else:
            names.extend(f"{room_type}_{index}" for index in range(1, count + 1))
    return names


def build_plan_image_prompt(program: dict[str, Any], variant_hint: str) -> str:
    rooms_block = "\n".join(f"- {name}" for name in room_instance_names(program))
    adjacency_block = "\n".join(
        f"- {pair[0]} <-> {pair[1]}" for pair in program.get("adjacency", [])
    ) or "- (none specified)"
    return f"""Draw a schematic architectural floor plan of a {program.get('building_type', 'building')} as a clean black-and-white line drawing.

{PLAN_DRAWING_RULES}

ROOMS (draw and label every instance exactly once):
{rooms_block}

REQUIRED ADJACENCIES (these pairs share a door):
{adjacency_block}

{variant_hint}"""


def request_plan_image(api_key: str, model: str, prompt: str, timeout_seconds: float) -> dict[str, str]:
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            # 2K since 2026-07-09: 1K candidates looked soft in the fullscreen lightbox;
            # b64 payloads stay well under the 8 MB reference-image / 10 MB request caps.
            "imageConfig": {"aspectRatio": "16:9", "imageSize": "2K"},
        },
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    attempts = 3
    last_failure = "Image model returned no image."
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            try:
                message = json.loads(body).get("error", {}).get("message", body)
            except json.JSONDecodeError:
                message = body
            raise ValueError(f"Image request failed: {message}") from error
        except urllib.error.URLError as error:
            raise ValueError(f"Image network request failed: {error.reason}") from error
        candidates = response_payload.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        for part in parts:
            inline = part.get("inlineData") or part.get("inline_data") or {}
            data = inline.get("data")
            if data:
                mime = str(inline.get("mimeType") or inline.get("mime_type") or "image/png")
                base64.b64decode(data, validate=True)   # reject malformed payloads early
                return {"mime": mime, "data": data}
        last_failure = "".join(str(part.get("text", "")) for part in parts)[:200] or last_failure
    raise ValueError(f"Image model returned no image after {attempts} attempts: {last_failure}")


def generate_plan_images(api_key: str, program: dict[str, Any], count: int = 3) -> list[dict[str, str]]:
    """``count`` candidate plan drawings, generated concurrently."""
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")
    model = os.environ.get("GEMINI_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)
    timeout_seconds = max(30.0, float(os.environ.get("GEMINI_TIMEOUT_SECONDS", "300")))
    count = max(1, min(4, int(count)))
    prompts = [build_plan_image_prompt(program, VARIANT_HINTS[index % len(VARIANT_HINTS)])
               for index in range(count)]
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(request_plan_image, api_key, model, prompt, timeout_seconds)
                   for prompt in prompts]
        return [future.result() for future in futures]
