"""
integration_example.py — how the BACKEND calls your ML lane.

Hand this file to Patricia. It shows BOTH supported integration styles so she
can pick whichever fits the backend. Neither requires her to understand YOLO —
she just sends an image and reads back the JSON contract (see app/schemas.py).

Run the HTTP example (with the service running on :8001) via:
    python integration_example.py path/to/image.jpg
"""

from __future__ import annotations

import sys


# ===========================================================================
# STYLE A — HTTP call (recommended: keeps ML as a separate service)
# The backend does NOT import any ML code. It just POSTs the image file.
# ===========================================================================
def predict_over_http(image_path: str, ml_service_url: str = "http://127.0.0.1:8001"):
    import httpx  # or `requests` — same idea

    with open(image_path, "rb") as f:
        files = {"image": (image_path, f, "image/jpeg")}
        response = httpx.post(f"{ml_service_url}/predict", files=files, timeout=30.0)

    response.raise_for_status()  # raises on 4xx/5xx so failures are visible
    return response.json()       # <-- a PredictionResult as a plain dict


# ===========================================================================
# STYLE B — direct Python import (if the ML code runs inside the same process)
# The backend imports two functions and never makes a network call.
# ===========================================================================
def predict_in_process(image_path: str):
    from app.config import settings
    from app.inference import run_inference
    from app.scoring import score_detections

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    detections, width, height, inference_ms = run_inference(image_bytes)
    result = score_detections(
        detections=detections,
        image_width=width,
        image_height=height,
        model_version=settings.model_version,
        inference_ms=inference_ms,
    )
    return result.model_dump()  # same dict shape as the HTTP response


# ===========================================================================
# What the backend gets back (either style) — map these onto the DB columns
# from the SDD `predictions` table:
#
#   result["model_version"]   -> predictions.model_version
#   result["detections"]      -> predictions.classes_json
#   result["max_confidence"]  -> predictions.max_confidence
#   result["severity_score"]  -> predictions.severity_score
#   result["risk_proxy"]      -> predictions.risk_proxy
#   result["priority"]        -> predictions.priority
#   result["explanation"]     -> predictions.explanation
#   result["inference_ms"]    -> predictions.inference_ms
#   result["needs_review"]    -> drives report status: True => 'needs_review',
#                                False => 'analyzed'
# ===========================================================================


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python integration_example.py <image_path>")
        raise SystemExit(1)

    path = sys.argv[1]
    print("STYLE B (in-process) result:")
    import json

    print(json.dumps(predict_in_process(path), indent=2))
    # To test STYLE A, start the server first, then uncomment:
    # print(json.dumps(predict_over_http(path), indent=2))
