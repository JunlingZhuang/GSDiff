# -*- coding: utf-8 -*-
"""Thin Gemini client wrapper — the ONLY place that talks to the model API.

Backward/forward compatibility contract:
  * No model id is hardcoded anywhere else in hfagent. Text and image models
    are resolved here, overridable via env (HFAGENT_TEXT_MODEL /
    HFAGENT_IMAGE_MODEL), otherwise auto-discovered from `models.list()` by a
    preference order — when Google ships a newer family, this picks it up
    without code changes.
  * Auth works in both modes with the same code: Gemini API key (env GEMINI /
    GEMINI_API_KEY / GOOGLE_API_KEY) or Vertex (GOOGLE_GENAI_USE_VERTEXAI=true
    + GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION). The google-genai SDK uses
    the identical generateContent surface for both.
  * Text, structured-JSON and image generation all go through the same chat
    `generate_content` API, so a future agent loop can interleave them in one
    conversation (multi-turn image editing for the correction loop).
"""
from __future__ import annotations

import base64
import binascii
import os
import re
from pathlib import Path

from dotenv import load_dotenv

# image magic bytes -> mime type
_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF8", "image/gif"),
    (b"RIFF", "image/webp"),  # RIFF....WEBP
)


def _sniff_mime(raw: bytes) -> str | None:
    for magic, mime in _MAGIC:
        if raw.startswith(magic):
            return mime
    return None


def decode_image_bytes(data) -> bytes:
    """Return real image bytes from a model response part.

    Some image models (e.g. gemini-3-pro-image) put base64-encoded text in
    inline_data.data instead of raw bytes; decode it. Already-binary images are
    returned unchanged.
    """
    raw = data.encode("ascii", "ignore") if isinstance(data, str) else bytes(data)
    if _sniff_mime(raw):
        return raw  # already a real image
    try:
        decoded = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        return raw  # not base64 — hand back as-is, caller validates
    return decoded if _sniff_mime(decoded) else raw

# Newest-first preference per capability. Plain substring match against the
# available model list; first hit wins.
TEXT_MODEL_PREFERENCE = [
    "gemini-3.1-pro",
    "gemini-3-pro-preview",
    "gemini-3-pro",
    "gemini-3-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]
IMAGE_MODEL_PREFERENCE = [
    "gemini-3.1-pro-image",
    "gemini-3-pro-image",
    "gemini-3-flash-image",
    "gemini-2.5-flash-image",
]
# Pin the generated image size so the px-based wall/parse geometry is deterministic.
# Gemini sizes are tiers (1K/2K/4K), not arbitrary pixels: 1K @ 16:9 = 1376x768,
# 2K @ 16:9 = 2752x1536. The tier is config-driven (config.json: image_size); the
# resolved value lives on the client instance (see GeminiClient.__init__).
DEFAULT_IMAGE_ASPECT = "16:9"
DEFAULT_IMAGE_SIZE = "1K"
# image models we must NOT pick for text and vice versa
_IMAGE_MARKER = re.compile(r"image|imagen")

# Gemini image output aspect tokens (docs) -> width/height ratio.
_SUPPORTED_ASPECTS = {
    "1:1": 1.0, "3:2": 1.5, "2:3": 2 / 3, "3:4": 0.75, "4:3": 4 / 3,
    "4:5": 0.8, "5:4": 1.25, "9:16": 9 / 16, "16:9": 16 / 9, "21:9": 21 / 9,
}


def _nearest_aspect_token(width: int, height: int) -> str | None:
    if not width or not height:
        return None
    ratio = width / height
    return min(_SUPPORTED_ASPECTS, key=lambda token: abs(_SUPPORTED_ASPECTS[token] - ratio))


def _first_image_aspect(contents) -> str | None:
    """Aspect token matching the FIRST input image in a multimodal contents list.

    Image models ignore — or reshuffle the layout of — a reference image when the
    requested output aspect conflicts with it (a documented behaviour: a tall input
    forced to 16:9 comes back resized/ignored). So when conditioning on an input
    image (a boundary outline, or a previous-round image in the correction loop), the
    output aspect must FOLLOW that image rather than a fixed default. Returns None for
    pure text-to-image (no input image), where the configured default applies.
    """
    if not isinstance(contents, list):
        return None
    for c in contents:
        if isinstance(c, (bytes, bytearray)):
            try:
                import io
                from PIL import Image
                width, height = Image.open(io.BytesIO(bytes(c))).size
                return _nearest_aspect_token(width, height)
            except Exception:
                return None
    return None


def _load_env() -> None:
    # repo root .env.local first (user keeps GEMINI= there), then hfagent/.env
    here = Path(__file__).resolve()
    load_dotenv(here.parents[1] / ".env.local", override=False)
    load_dotenv(here.parent / ".env", override=False)


def _api_key() -> str | None:
    for name in ("GEMINI", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.environ.get(name)
        if v:
            return v
    return None


class GeminiClient:
    def __init__(
        self,
        text_model: str | None = None,
        image_model: str | None = None,
        image_size: str | None = None,
        image_aspect: str | None = None,
    ):
        _load_env()
        from google import genai  # imported lazily so unit tests need no SDK

        if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true"):
            self.client = genai.Client(
                vertexai=True,
                project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
            )
        else:
            key = _api_key()
            if not key:
                raise RuntimeError(
                    "No Gemini credentials: set GEMINI / GEMINI_API_KEY in .env.local, "
                    "or GOOGLE_GENAI_USE_VERTEXAI=true with GOOGLE_CLOUD_PROJECT."
                )
            self.client = genai.Client(api_key=key)

        self._available: list[str] | None = None
        self.text_model = text_model or os.environ.get("HFAGENT_TEXT_MODEL") or self._resolve(
            TEXT_MODEL_PREFERENCE, want_image=False
        )
        self.image_model = image_model or os.environ.get("HFAGENT_IMAGE_MODEL") or self._resolve(
            IMAGE_MODEL_PREFERENCE, want_image=True
        )
        # image output tier: explicit arg (config.json) wins, then env, then default
        self.image_size = image_size or os.environ.get("HFAGENT_IMAGE_SIZE") or DEFAULT_IMAGE_SIZE
        self.image_aspect = image_aspect or os.environ.get("HFAGENT_IMAGE_ASPECT") or DEFAULT_IMAGE_ASPECT

    # ── model discovery ─────────────────────────────────────────────────────
    def available_models(self) -> list[str]:
        if self._available is None:
            names = []
            for m in self.client.models.list():
                n = getattr(m, "name", "") or ""
                names.append(n.removeprefix("models/").removeprefix("publishers/google/models/"))
            self._available = names
        return self._available

    def _resolve(self, preference: list[str], want_image: bool) -> str:
        try:
            models = self.available_models()
        except Exception:
            # list() unavailable (some Vertex setups): fall back to first preference
            return preference[0]
        for pref in preference:
            hits = [m for m in models if pref in m and bool(_IMAGE_MARKER.search(m)) == want_image]
            if hits:
                # prefer non-dated/shortest alias (e.g. gemini-3-pro-image over ...-preview-11-2025)
                return sorted(hits, key=len)[0]
        return preference[0]

    # ── generation (all via the same generateContent chat surface) ──────────
    def generate_text(self, prompt: str, model: str | None = None) -> str:
        resp = self.client.models.generate_content(model=model or self.text_model, contents=prompt)
        return resp.text or ""

    @staticmethod
    def _to_parts(contents):
        """Wrap raw image bytes as Parts with a sniffed mime type; pass strings through."""
        from google.genai import types

        if not isinstance(contents, list):
            return contents
        out = []
        for c in contents:
            if isinstance(c, (bytes, bytearray)):
                mime = _sniff_mime(bytes(c)) or "image/png"
                out.append(types.Part.from_bytes(data=bytes(c), mime_type=mime))
            else:
                out.append(c)
        return out

    def generate_json(self, contents, schema: dict | None = None, model: str | None = None) -> str:
        """Returns JSON string. `contents` may be a str prompt or a list mixing str and image bytes."""
        from google.genai import types

        cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            **({"response_json_schema": schema} if schema else {}),
        )
        resp = self.client.models.generate_content(
            model=model or self.text_model, contents=self._to_parts(contents), config=cfg
        )
        return resp.text or ""

    def generate_image(self, contents, model: str | None = None, attempts: int = 3) -> bytes:
        """Returns PNG/JPEG bytes of the first image part in the response.

        `contents` may be a prompt string or a list mixing str and raw image
        bytes — the multi-turn editing path of the correction loop sends
        [previous_image_bytes, feedback_text]. Handles models that return the
        image as base64 text (see decode_image_bytes).

        Image models intermittently return only a text part (no image), especially
        on multi-turn edits; the same request usually succeeds on retry, so we retry
        up to `attempts` times before giving up.
        """
        from google.genai import types

        # When conditioning on an input image (boundary outline / previous-round image),
        # the output aspect must follow that image — forcing a conflicting aspect makes
        # the model ignore or reshuffle the reference. Pure text-to-image uses the default.
        aspect = _first_image_aspect(contents) or self.image_aspect
        cfg = types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=aspect, image_size=self.image_size),
        )
        parts = self._to_parts(contents)
        target = model or self.image_model
        last_text = ""
        for _ in range(max(1, attempts)):
            resp = self.client.models.generate_content(model=target, contents=parts, config=cfg)
            for cand in resp.candidates or []:
                for part in (cand.content.parts or []) if cand.content else []:
                    data = getattr(part, "inline_data", None)
                    if data and data.data:
                        return decode_image_bytes(data.data)
            last_text = (resp.text or "").strip().replace("\n", " ")[:200]
        raise RuntimeError(
            f"model {target} returned no image part after {max(1, attempts)} attempts"
            + (f" (last text: {last_text!r})" if last_text else "")
        )
