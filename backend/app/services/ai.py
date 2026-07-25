from time import perf_counter

from app.models import Detection, Prediction
from app.services.scoring import calculate_score


# This identifies the current AI implementation.
# Your AI teammate can change this when they plug in a real model.
MODEL_VERSION = "demo-heuristic-v1"


# Temporary AI function used until the real model is connected.
# It keeps the backend/frontend contract stable for your team.
def analyze_image(filename: str, content_type: str, note: str = "") -> Prediction:
    # Track rough inference time so the API already returns inference_ms.
    started_at = perf_counter()

    # The demo version looks for keywords in filename/note.
    # Real AI will inspect image bytes or an image URL instead.
    text = f"{filename} {content_type} {note}".lower()
    detections: list[Detection] = []

    # Add a drain blockage detection if text hints at drains/flooding.
    if any(term in text for term in ("drain", "block", "blocked", "flood")):
        detections.append(Detection(label="drain_blockage", confidence=0.72))

    # Add a standing water detection if text hints at stagnant water.
    if any(term in text for term in ("water", "standing", "stagnant")):
        detections.append(Detection(label="standing_water", confidence=0.68))

    # Add organic waste detection if text hints at trash or market waste.
    if any(term in text for term in ("waste", "trash", "garbage", "organic", "market")):
        detections.append(Detection(label="organic_waste", confidence=0.64))

    # If nothing obvious is found, return a low-confidence default.
    if not detections:
        detections.append(Detection(label="organic_waste", confidence=0.42))

    # max_confidence is used to decide analyzed vs needs_review.
    max_confidence = max(detection.confidence for detection in detections)

    # Convert detections into severity score, risk proxy, priority, and explanation.
    severity_score, risk_proxy, priority, explanation = calculate_score(detections)

    # Calculate how long this analysis took in milliseconds.
    inference_ms = max(1, round((perf_counter() - started_at) * 1000))

    # Return the prediction in the same structure the real model should use later.
    return Prediction(
        model_version=MODEL_VERSION,
        classes=detections,
        max_confidence=max_confidence,
        severity_score=severity_score,
        risk_proxy=risk_proxy,
        priority=priority,
        explanation=explanation,
        inference_ms=inference_ms,
    )
