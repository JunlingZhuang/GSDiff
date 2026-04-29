"""Shared in-memory model loading status registry."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import torch

from app.config import DEVICE

ModelState = Literal["not_loaded", "loading", "loaded", "error"]


@dataclass
class ModelStatus:
    key: str
    group: str
    label: str
    state: ModelState = "not_loaded"
    device: str | None = None
    started_at: float | None = None
    loaded_at: float | None = None
    load_seconds: float | None = None
    error: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class ModelStatusRegistry:
    def __init__(self) -> None:
        self._items: dict[str, ModelStatus] = {}

    def register(self, key: str, *, group: str, label: str, metadata: dict[str, str] | None = None) -> None:
        if key in self._items:
            if metadata:
                self._items[key].metadata.update(metadata)
            return
        self._items[key] = ModelStatus(
            key=key,
            group=group,
            label=label,
            metadata=metadata or {},
        )

    def loading(self, key: str) -> None:
        item = self._items[key]
        item.state = "loading"
        item.device = DEVICE
        item.started_at = time.time()
        item.loaded_at = None
        item.load_seconds = None
        item.error = None

    def loaded(self, key: str) -> None:
        item = self._items[key]
        item.state = "loaded"
        item.device = DEVICE
        item.loaded_at = time.time()
        if item.started_at is not None:
            item.load_seconds = item.loaded_at - item.started_at
        item.error = None

    def error(self, key: str, error: Exception) -> None:
        item = self._items[key]
        item.state = "error"
        item.device = DEVICE
        item.loaded_at = time.time()
        if item.started_at is not None:
            item.load_seconds = item.loaded_at - item.started_at
        item.error = str(error)

    def snapshot(self) -> dict:
        gpu = None
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            gpu = {
                "name": torch.cuda.get_device_name(device_index),
                "allocated_mb": round(torch.cuda.memory_allocated(device_index) / 1024 / 1024, 2),
                "reserved_mb": round(torch.cuda.memory_reserved(device_index) / 1024 / 1024, 2),
            }

        return {
            "device": DEVICE,
            "gpu": gpu,
            "models": [
                {
                    "key": item.key,
                    "group": item.group,
                    "label": item.label,
                    "state": item.state,
                    "device": item.device,
                    "load_seconds": item.load_seconds,
                    "error": item.error,
                    "metadata": item.metadata,
                }
                for item in sorted(self._items.values(), key=lambda value: (value.group, value.label))
            ],
        }


model_status = ModelStatusRegistry()
