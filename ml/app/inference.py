"""
inference.py — the ONLY file that actually touches the AI model.

Everything else in your service is plain Python. By hiding YOLO behind one
clean function, `run_inference(image) -> list[Detection]`, the rest of the code
(scoring, API) never has to know which model you're using. The SDD calls this
"one ML inference module hidden behind a service interface." Swapping the
pretrained model for your future custom one changes ONLY this file's config.

KEY CONCEPT — object detection:
  A detection model looks at an image and returns a list of boxes. Each box has
  a class ("banana"), a confidence (0..1), and pixel coordinates. That's it.
  YOLO ("You Only Look Once") is a family of fast detection models. We use the
  smallest, "yolov8n" (n = nano), because it runs quickly on a CPU.

KEY CONCEPT — why a "proxy" today:
  The pretrained model was trained on COCO, a dataset of 80 everyday objects. It
  has never seen the label "organic_waste". But it CAN see bananas, apples,
  pizza, etc. — which, in a market waste pile, ARE organic waste. So in demo
  mode we translate COCO food classes into `organic_waste`. This is an honest
  stand-in, clearly versioned as "...coco-proxy...", until your trained model
  (which outputs the 3 real classes directly) replaces it.
"""

from __future__ import annotations

import time
from io import BytesIO

from PIL import Image

from .config import settings
from .schemas import Detection, WasteClass

# ---------------------------------------------------------------------------
# COCO -> TrashVision proxy mapping (used only when settings.use_coco_proxy).
#
# HONESTY NOTE: the pretrained model has NO reliable signal for drain_blockage
# or standing_water — those classes need your custom-trained model. So in demo
# mode only `organic_waste` is proxied, from COCO's food classes. This is
# exactly the kind of limitation the SDD wants you to state out loud, and it is
# *why* your data-collection work matters.
# ---------------------------------------------------------------------------
COCO_TO_TRASH: dict[str, WasteClass] = {
    "banana": WasteClass.organic_waste,
    "apple": WasteClass.organic_waste,
    "orange": WasteClass.organic_waste,
    "sandwich": WasteClass.organic_waste,
    "broccoli": WasteClass.organic_waste,
    "carrot": WasteClass.organic_waste,
    "hot dog": WasteClass.organic_waste,
    "pizza": WasteClass.organic_waste,
    "donut": WasteClass.organic_waste,
    "cake": WasteClass.organic_waste,
}

# The model is expensive to load, so we load it ONCE and reuse it (a "singleton").
# This is the SDD's "model warm-up" idea: the first request is slow, the rest fast.
_model = None


def get_model():
    """Load the YOLO model on first use, then reuse the cached instance."""
    global _model
    if _model is None:
        # Imported here (not at top) so the rest of the app — and the scoring
        # unit tests — don't require the heavy `ultralytics`/`torch` install.
        from ultralytics import YOLO

        _model = YOLO(settings.model_weights)
    return _model


def warm_up() -> None:
    """
    Force the model to load NOW (e.g. at server startup) so the first real user
    doesn't pay the loading cost. Called from the FastAPI startup hook.
    """
    get_model()


def _raw_yolo_to_detections(yolo_result) -> list[Detection]:
    """
    Translate ONE ultralytics result object into our clean Detection list.

    `yolo_result.boxes` gives us, per detected object:
      - .cls  : the class index (an int)
      - .conf : the confidence (0..1)
      - .xyxy : the box as [x1, y1, x2, y2] pixels
    and `yolo_result.names` maps a class index -> its string name.
    """
    detections: list[Detection] = []
    names = yolo_result.names  # e.g. {0: 'person', 46: 'banana', ...}

    for box in yolo_result.boxes:
        raw_name = names[int(box.cls[0])]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])

        if settings.use_coco_proxy:
            # Demo mode: keep only COCO classes we can honestly map.
            mapped = COCO_TO_TRASH.get(raw_name)
            if mapped is None:
                continue
            class_name = mapped
        else:
            # Custom-model mode: the model's own labels ARE our 3 classes.
            try:
                class_name = WasteClass(raw_name)
            except ValueError:
                continue  # ignore any class that isn't one of our three

        detections.append(
            Detection(class_name=class_name, confidence=confidence, bbox=[x1, y1, x2, y2])
        )

    return detections


def run_inference(image_bytes: bytes) -> tuple[list[Detection], int, int, float]:
    """
    THE MAIN ENTRY POINT of this module.

    Input : raw bytes of a JPG/PNG/WebP image.
    Output: (detections, image_width, image_height, inference_ms)

    The API layer calls this, then passes the detections straight into
    score_detections(). Clean separation: this file detects, scoring.py judges.
    """
    # Decode bytes -> a real image, and force RGB (drops alpha/grayscale quirks).
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    model = get_model()

    start = time.perf_counter()
    # verbose=False keeps YOLO from printing to the console on every request.
    results = model.predict(image, conf=settings.confidence_threshold, verbose=False)
    inference_ms = (time.perf_counter() - start) * 1000.0

    # predict() returns a list (one entry per image); we sent one image.
    detections = _raw_yolo_to_detections(results[0])
    return detections, width, height, round(inference_ms, 2)
