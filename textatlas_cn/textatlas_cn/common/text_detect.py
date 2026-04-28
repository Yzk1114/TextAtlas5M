"""Wrapper for fillable-region detection used by StyledTextSynth-CN.

Mirrors the paper's pipeline:
1. YOLOv11 detects the largest fillable region per topic group.
2. Optional RT-DETR fine-tuned for `packing_box` / `booklet`.
3. SAM2 refines rectangle to irregular quadrilateral starting from bbox center.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .schema import BBox


@dataclass
class DetectorConfig:
    yolo_weights: str = "data/models/yolo11l_textregion.pt"
    rtdetr_weights: str | None = None
    sam2_checkpoint: str | None = None
    sam2_config: str | None = None


class TextRegionDetector:
    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self._yolo = None
        self._rtdetr = None
        self._sam2 = None

    def _ensure_yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO  # type: ignore
            self._yolo = YOLO(self.cfg.yolo_weights)
        return self._yolo

    def _ensure_sam2(self):
        if self._sam2 is None and self.cfg.sam2_checkpoint:
            try:
                from sam2.build_sam import build_sam2  # type: ignore
                from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
                model = build_sam2(self.cfg.sam2_config, self.cfg.sam2_checkpoint)
                self._sam2 = SAM2ImagePredictor(model)
            except Exception:
                self._sam2 = None
        return self._sam2

    # ------------------------------------------------------------------
    def detect(self, image: Image.Image, group: str) -> BBox | None:
        """Return one fillable region (refined to a quadrilateral when SAM2 is available)."""
        yolo = self._ensure_yolo()
        arr = np.array(image.convert("RGB"))
        result = yolo.predict(arr, verbose=False)[0]
        if not len(result.boxes):
            return None
        # Largest area box (matches paper §B.4.1).
        boxes = result.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = int(np.argmax(areas))
        x0, y0, x1, y1 = map(float, boxes[idx])
        rect = BBox.from_xyxy(x0, y0, x1, y1, label=group)

        sam2 = self._ensure_sam2()
        if sam2 is None:
            return rect
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        sam2.set_image(arr)
        masks, _, _ = sam2.predict(point_coords=np.array([[cx, cy]]), point_labels=np.array([1]))
        mask = masks[0].astype(np.uint8) * 255
        return _mask_to_quad(mask, rect)


def _mask_to_quad(mask: np.ndarray, fallback: BBox) -> BBox:
    import cv2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fallback
    cnt = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    if len(approx) >= 4:
        rect = cv2.minAreaRect(approx)
        box = cv2.boxPoints(rect)
        return BBox(points=[(float(x), float(y)) for x, y in box], label=fallback.label)
    return fallback
