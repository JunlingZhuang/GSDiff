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

import os
import re
from pathlib import Path

from dotenv import load_dotenv

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
# image models we must NOT pick for text and vice versa
_IMAGE_MARKER = re.compile(r"image|imagen")


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
    def __init__(self, text_model: str | None = None, image_model: str | None = None):
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

    def generate_json(self, prompt: str, schema: dict | None = None, model: str | None = None) -> str:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            **({"response_json_schema": schema} if schema else {}),
        )
        resp = self.client.models.generate_content(
            model=model or self.text_model, contents=prompt, config=cfg
        )
        return resp.text or ""

    def generate_image(self, contents, model: str | None = None) -> bytes:
        """Returns PNG/JPEG bytes of the first image part in the response.

        `contents` may be a prompt string or a list mixing str and raw image
        bytes — the multi-turn editing path of the correction loop sends
        [previous_image_bytes, feedback_text].
        """
        from google.genai import types

        if isinstance(contents, list):
            contents = [
                types.Part.from_bytes(data=c, mime_type="image/png") if isinstance(c, bytes) else c
                for c in contents
            ]
        cfg = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
        resp = self.client.models.generate_content(
            model=model or self.image_model, contents=contents, config=cfg
        )
        for cand in resp.candidates or []:
            for part in (cand.content.parts or []) if cand.content else []:
                data = getattr(part, "inline_data", None)
                if data and data.data:
                    return data.data
        raise RuntimeError(f"model {model or self.image_model} returned no image part")
