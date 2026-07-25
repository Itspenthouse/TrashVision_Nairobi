from __future__ import annotations

import time
from io import BytesIO

from PIL import Image

from .config import settings
from .schemas import Detection, WasteClass


_model = None


def get_model():
    """Load the configured custom YOLO checkpoint once."""
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO(settings.model_weights)
    return _model


def warm_up() -> None:
    get_model()


def _raw_yolo_to_detections(yolo_result) -> list[Detection]:
    detections: list[Detection] = []
    for box in yolo_result.boxes:
        raw_name = yolo_result.names[int(box.cls[0])]
        try:
            class_name = WasteClass(raw_name)
        except ValueError:
            continue

        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
        detections.append(
            Detection(
                class_name=class_name,
                confidence=float(box.conf[0]),
                bbox=[x1, y1, x2, y2],
            )
        )
    return detections


def run_inference(image_bytes: bytes) -> tuple[list[Detection], int, int, float]:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    model = get_model()

    start = time.perf_counter()
    results = model.predict(
        image,
        conf=settings.confidence_threshold,
        verbose=False,
    )
    inference_ms = (time.perf_counter() - start) * 1000
    return (
        _raw_yolo_to_detections(results[0]),
        width,
        height,
        round(inference_ms, 2),
    )
