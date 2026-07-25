"""
scoring.py — turn raw detections into an explainable 0..100 severity score.

This is 100% YOUR logic, Michael. There is no AI here — just deterministic math.
"Deterministic" means: same detections in -> exact same score out, every time.
That is a feature. Judges can trust it, and you can unit-test it.

The pipeline for one image:

    detections + image size
        -> per-class "evidence"  (a 0..1 signal for each of the 3 classes)
        -> severity_score  (0..100, the SDD weighted formula)
        -> risk_proxy      (0..100, a VISIBLE-condition flood proxy, not a forecast)
        -> priority        (low / medium / high / urgent)
        -> explanation     (a plain-English sentence)
        -> needs_review    (True when the model is too unsure to trust)

The two formulas come straight from the SDD section 4.3:

    severity   = 0.45*waste + 0.25*blockage + 0.20*standing_water + 0.10*confidence
    risk_proxy = 0.55*blockage + 0.35*standing_water + 0.10*severity
"""

from __future__ import annotations

from .schemas import Detection, PredictionResult, Priority, WasteClass

# ---------------------------------------------------------------------------
# Tunable constants. Keeping them named + in one place (instead of "magic
# numbers" sprinkled in the code) is a habit that will save you every time you
# want to explain or adjust behaviour.
# ---------------------------------------------------------------------------

# How we blend the two signals that make up a class's "evidence":
#   evidence = CONF_WEIGHT * (how confident) + COVERAGE_WEIGHT * (how much area)
CONF_WEIGHT = 0.6
COVERAGE_WEIGHT = 0.4

# If a class's boxes cover this fraction of the image, we treat coverage as
# "maxed out" (1.0). 0.35 = boxes over ~a third of the photo is already a lot.
COVERAGE_SATURATION = 0.35

# Below this confidence, or with zero detections, we don't trust the result and
# route the report to a human (SDD: "send ambiguous reports to needs_review").
REVIEW_CONFIDENCE_THRESHOLD = 0.35

# Priority bands (SDD 4.3). (low_inclusive, high_inclusive, label).
PRIORITY_BANDS = [
    (0, 29, Priority.low),
    (30, 54, Priority.medium),
    (55, 79, Priority.high),
    (80, 100, Priority.urgent),
]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Force a number to stay inside [low, high]. Guards against weird inputs."""
    return max(low, min(high, value))


def _box_area(bbox: list[float]) -> float:
    """Area of a [x1, y1, x2, y2] box in pixels. Never negative."""
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def compute_class_evidence(
    detections: list[Detection],
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    """
    Collapse a list of detections into ONE 0..1 number per class.

    For each class we combine two intuitions:
      - CONFIDENCE: the model's single most confident box for that class.
      - COVERAGE:   how much of the image that class's boxes take up. A drain
                    buried under a huge pile of waste should score higher than a
                    single crisp of litter, even if both are detected at 0.9.

    Returns e.g. {"organic_waste": 0.71, "drain_blockage": 0.0, "standing_water": 0.12}
    """
    image_area = max(1.0, float(image_width) * float(image_height))  # avoid /0

    evidence: dict[str, float] = {}
    for cls in WasteClass:  # always report all 3 classes, even if 0
        cls_dets = [d for d in detections if d.class_name == cls]

        if not cls_dets:
            evidence[cls.value] = 0.0
            continue

        max_conf = max(d.confidence for d in cls_dets)

        total_area = sum(_box_area(d.bbox) for d in cls_dets)
        area_fraction = total_area / image_area
        # Scale so that COVERAGE_SATURATION of the image counts as full coverage.
        coverage = _clamp(area_fraction / COVERAGE_SATURATION)

        evidence[cls.value] = _clamp(
            CONF_WEIGHT * max_conf + COVERAGE_WEIGHT * coverage
        )

    return evidence


def _priority_for(score: int) -> Priority:
    """Map a 0..100 score to its label using the SDD bands."""
    for low, high, label in PRIORITY_BANDS:
        if low <= score <= high:
            return label
    return Priority.low  # unreachable, but a safe default


def _build_explanation(
    evidence: dict[str, float],
    severity: int,
    risk_proxy: int,
    max_confidence: float,
    needs_review: bool,
) -> str:
    """
    Produce a short, honest, human sentence. This is what a coordinator reads,
    so it must never *overclaim*. Notice the wording is 'AI suggests', not
    'there IS' — that is a deliberate responsible-AI choice from the SDD.
    """
    # Rank the classes by how much evidence each contributed.
    ranked = sorted(evidence.items(), key=lambda kv: kv[1], reverse=True)
    top_class, top_val = ranked[0]

    if max_confidence == 0.0:
        return (
            "The AI found nothing it recognises in this image. "
            "Severity defaults to 0 and the report is sent for human review."
        )

    nice = top_class.replace("_", " ")
    parts = [
        f"AI suggests mainly {nice} "
        f"(evidence {top_val:.2f}). "
        f"Severity {severity}/100, flood-risk proxy {risk_proxy}/100."
    ]
    if needs_review:
        parts.append(
            "Confidence is low, so this is flagged for human review rather "
            "than treated as certain."
        )
    return " ".join(parts)


def score_detections(
    detections: list[Detection],
    image_width: int,
    image_height: int,
    model_version: str,
    inference_ms: float | None = None,
) -> PredictionResult:
    """
    THE MAIN ENTRY POINT of this module.

    Give it the model's detections + the image size, and it returns a complete,
    validated PredictionResult (the contract object the backend stores).
    """
    evidence = compute_class_evidence(detections, image_width, image_height)
    max_confidence = max((d.confidence for d in detections), default=0.0)

    # --- The SDD severity formula (all terms are 0..1, result scaled to 0..100).
    severity_float = (
        0.45 * evidence[WasteClass.organic_waste.value]
        + 0.25 * evidence[WasteClass.drain_blockage.value]
        + 0.20 * evidence[WasteClass.standing_water.value]
        + 0.10 * max_confidence
    )
    severity = round(_clamp(severity_float) * 100)

    # --- The SDD flood-risk proxy. Note it reuses `severity_float` (0..1) for
    #     its last term so both formulas stay on the same 0..1 scale.
    risk_float = (
        0.55 * evidence[WasteClass.drain_blockage.value]
        + 0.35 * evidence[WasteClass.standing_water.value]
        + 0.10 * severity_float
    )
    risk_proxy = round(_clamp(risk_float) * 100)

    priority = _priority_for(severity)

    # Flag for a human when the model is unsure or saw nothing.
    needs_review = (
        len(detections) == 0 or max_confidence < REVIEW_CONFIDENCE_THRESHOLD
    )

    explanation = _build_explanation(
        evidence, severity, risk_proxy, max_confidence, needs_review
    )

    return PredictionResult(
        model_version=model_version,
        detections=detections,
        evidence=evidence,
        max_confidence=round(max_confidence, 4),
        severity_score=severity,
        risk_proxy=risk_proxy,
        priority=priority,
        explanation=explanation,
        needs_review=needs_review,
        inference_ms=inference_ms,
    )
