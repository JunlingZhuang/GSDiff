# -*- coding: utf-8 -*-
"""Label-text REGION detection for the linework tracer's render-side cleanup.

Detection only (PP-OCRv3 DB, ``data/models/text_detection_en_ppocrv3.onnx``):
nothing is read, and the trace itself never sees the quads — erasing or
whitening label ink BEFORE tracing was refuted by forensics (in label-dense
drawings text ink is load-bearing: it extends divider runs, vouches for short
walls, forms door jambs and supplies arc evidence; removal loses real doors
and splits real rooms). The quads are consumed only AFTER ``polygonize_rooms``
by ``linework_tracer.hide_label_residue``, where hiding a segment can change
nothing but the rendered drawing. The model file is optional — without it
detection returns no quads and rendering keeps every segment.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "models" / "text_detection_en_ppocrv3.onnx"

# PP-OCRv3 DB preprocessing, per opencv_zoo's ppocr_det.py. CPU only: inference
# stays deterministic and a fresh net per call keeps the module thread-safe.
_BINARY_THRESHOLD = 0.3
_POLYGON_THRESHOLD = 0.5
_UNCLIP_RATIO = 1.5   # tight quads — padding swallows door symbols beside labels
_MAX_CANDIDATES = 400
_INPUT_MEAN = (123.675, 116.28, 103.53)
_INPUT_SCALE = 1.0 / 255.0 / 0.226
# labels in 1K-era drawings are ~7 px tall — below the detector's comfort zone;
# detecting on a 2x upscale recovers them (2K drawings run at native size)
_UPSCALE_BELOW = 1024


def _detect_at_scale(bgr: np.ndarray, scale: float) -> list[np.ndarray]:
    height, width = bgr.shape[:2]
    net_w = max(32, int(round(width * scale / 32)) * 32)
    net_h = max(32, int(round(height * scale / 32)) * 32)
    detector = cv2.dnn_TextDetectionModel_DB(cv2.dnn.readNet(str(MODEL_PATH)))
    detector.setBinaryThreshold(_BINARY_THRESHOLD)
    detector.setPolygonThreshold(_POLYGON_THRESHOLD)
    detector.setUnclipRatio(_UNCLIP_RATIO)
    detector.setMaxCandidates(_MAX_CANDIDATES)
    detector.setInputParams(_INPUT_SCALE, (net_w, net_h), _INPUT_MEAN, False)
    quads, _confidences = detector.detect(cv2.resize(bgr, (net_w, net_h)))
    sx, sy = width / net_w, height / net_h
    out = []
    for quad in quads:
        pts = np.asarray(quad, dtype=np.float64)
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        out.append(pts.astype(np.int32))
    return out


def detect_label_quads(gray: np.ndarray) -> list[np.ndarray]:
    """Text-region quads (int32, image coordinates); [] when no model is present.

    Detection runs at TWO scales: the base scale tuned for normal label sizes
    (2x for 1K-era drawings whose ~7 px text is below the detector's comfort
    zone) and additionally at half of it — some drawings letter their rooms in
    GIANT fonts that the detector overlooks at full resolution but reads fine
    once shrunk. Near-duplicate quads across scales are merged (larger wins).
    """
    if not MODEL_PATH.exists():
        return []
    height, width = gray.shape[:2]
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray
    base = 2.0 if min(height, width) < _UPSCALE_BELOW else 1.0
    precise = _detect_at_scale(bgr, base)
    kept: list[np.ndarray] = list(precise)
    # half-scale quads only fill true blind spots: keep one when it overlaps NO
    # precise quad at all (giant fonts detect only when shrunk; everything the
    # base scale saw stays authoritative)
    for quad in _detect_at_scale(bgr, base * 0.5):
        qx0, qy0 = float(quad[:, 0].min()), float(quad[:, 1].min())
        qx1, qy1 = float(quad[:, 0].max()), float(quad[:, 1].max())
        overlaps = any(
            qx0 < float(k[:, 0].max()) and qx1 > float(k[:, 0].min())
            and qy0 < float(k[:, 1].max()) and qy1 > float(k[:, 1].min())
            for k in precise
        )
        if not overlaps:
            kept.append(quad)
    return kept
