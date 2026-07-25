"""
test_service.py — ML smoke test + API test.

This is heavier than test_scoring.py: it actually loads the YOLO model and hits
the real /predict endpoint. On the FIRST run it downloads yolov8n.pt (~6 MB), so
it needs internet once. It's skipped automatically if ultralytics isn't
installed, so the pure-Python scoring tests still run anywhere.

What it proves (SDD "ML smoke" + "API" test levels):
  - the model loads,
  - /predict accepts an image and returns HTTP 200,
  - the response matches our PredictionResult contract exactly,
  - /health works and reports the model as loaded.
"""

from io import BytesIO

import pytest

# Skip this whole file gracefully if the ML stack isn't installed yet.
ultralytics = pytest.importorskip("ultralytics")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.schemas import PredictionResult  # noqa: E402
from app.service import app  # noqa: E402

client = TestClient(app)


def _fake_image_bytes(color=(120, 90, 60), size=(320, 320)) -> bytes:
    """Make a plain JPEG in memory so we don't need a real photo to smoke-test."""
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_version" in body


def test_predict_returns_valid_contract():
    files = {"image": ("test.jpg", _fake_image_bytes(), "image/jpeg")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 200

    # The strongest possible check: re-validate the JSON against our schema.
    # If any field is missing or the wrong type, this raises and the test fails.
    result = PredictionResult.model_validate(resp.json())

    assert 0 <= result.severity_score <= 100
    assert 0 <= result.risk_proxy <= 100
    # A blank synthetic image has no recognisable objects, so the model should
    # find nothing -> the report is honestly flagged for human review.
    assert result.needs_review is True


def test_predict_rejects_non_image():
    files = {"image": ("notes.txt", b"hello", "text/plain")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 415
    assert resp.json()["code"] == "unsupported_media_type"
