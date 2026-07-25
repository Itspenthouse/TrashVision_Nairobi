"""
test_scoring.py — proof that the scoring math behaves.

Run these with:  pytest -v

Each `test_...` function is one claim about our code. pytest runs them and an
`assert` that fails turns the test red. This is how you catch the day you
accidentally change a weight and break production behavior. The SDD explicitly requires
testing "scoring boundaries" — that's exactly what we do here.
"""

from app.schemas import Detection, Priority, WasteClass
from app.scoring import (
    REVIEW_CONFIDENCE_THRESHOLD,
    compute_class_evidence,
    score_detections,
)

MODEL = "test-model"
# A 1000x1000 image = 1,000,000 px. Easy round numbers for reasoning about area.
W, H = 1000, 1000


def make_det(cls: WasteClass, conf: float, box=(0, 0, 100, 100)) -> Detection:
    """Tiny helper so each test reads clearly."""
    return Detection(class_name=cls, confidence=conf, bbox=list(box))


# --- Empty input --------------------------------------------------------------
def test_no_detections_is_zero_and_needs_review():
    result = score_detections([], W, H, MODEL)
    assert result.severity_score == 0
    assert result.risk_proxy == 0
    assert result.priority == Priority.low
    assert result.needs_review is True  # saw nothing -> a human should look
    assert result.max_confidence == 0.0


# --- Evidence building --------------------------------------------------------
def test_evidence_always_has_all_three_classes():
    ev = compute_class_evidence([make_det(WasteClass.organic_waste, 0.9)], W, H)
    assert set(ev.keys()) == {"organic_waste", "drain_blockage", "standing_water"}
    assert ev["drain_blockage"] == 0.0  # none detected


def test_bigger_box_means_more_evidence():
    """Same confidence, bigger area -> higher evidence (coverage term works)."""
    small = compute_class_evidence(
        [make_det(WasteClass.organic_waste, 0.8, (0, 0, 50, 50))], W, H
    )["organic_waste"]
    big = compute_class_evidence(
        [make_det(WasteClass.organic_waste, 0.8, (0, 0, 600, 600))], W, H
    )["organic_waste"]
    assert big > small


# --- The severity formula weights --------------------------------------------
def test_organic_waste_weighted_higher_than_water():
    """
    Same confidence + same box, organic_waste (0.45 weight) must yield a higher
    severity than standing_water (0.20 weight). This locks in the SDD formula.
    """
    box = (0, 0, 300, 300)
    waste = score_detections(
        [make_det(WasteClass.organic_waste, 0.9, box)], W, H, MODEL
    ).severity_score
    water = score_detections(
        [make_det(WasteClass.standing_water, 0.9, box)], W, H, MODEL
    ).severity_score
    assert waste > water


def test_score_stays_within_bounds():
    """Even with many huge, max-confidence boxes, score never exceeds 100."""
    dets = [make_det(c, 1.0, (0, 0, 1000, 1000)) for c in WasteClass]
    result = score_detections(dets, W, H, MODEL)
    assert 0 <= result.severity_score <= 100
    assert 0 <= result.risk_proxy <= 100


# --- Priority bands (the boundaries the SDD calls out) ------------------------
def test_priority_bands_match_sdd():
    """Directly test the mapping 0-29 low / 30-54 med / 55-79 high / 80-100 urgent."""
    from app.scoring import _priority_for

    assert _priority_for(0) == Priority.low
    assert _priority_for(29) == Priority.low
    assert _priority_for(30) == Priority.medium
    assert _priority_for(54) == Priority.medium
    assert _priority_for(55) == Priority.high
    assert _priority_for(79) == Priority.high
    assert _priority_for(80) == Priority.urgent
    assert _priority_for(100) == Priority.urgent


# --- Review flag --------------------------------------------------------------
def test_low_confidence_triggers_review():
    low = REVIEW_CONFIDENCE_THRESHOLD - 0.05
    result = score_detections([make_det(WasteClass.organic_waste, low)], W, H, MODEL)
    assert result.needs_review is True


def test_high_confidence_does_not_trigger_review():
    result = score_detections(
        [make_det(WasteClass.organic_waste, 0.95, (0, 0, 400, 400))], W, H, MODEL
    )
    assert result.needs_review is False


# --- Determinism (the whole point) -------------------------------------------
def test_same_input_same_output():
    dets = [make_det(WasteClass.drain_blockage, 0.7, (10, 10, 200, 200))]
    a = score_detections(dets, W, H, MODEL)
    b = score_detections(dets, W, H, MODEL)
    assert a.severity_score == b.severity_score
    assert a.risk_proxy == b.risk_proxy
    assert a.explanation == b.explanation
