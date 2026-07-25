from fastapi import HTTPException

from app.routes.reports import _validate_location


def test_accepts_valid_nairobi_coordinates():
    _validate_location(-1.2864, 36.8318)


def test_rejects_invalid_latitude():
    try:
        _validate_location(91, 36.8318)
    except HTTPException as exc:
        assert exc.status_code == 400
        assert exc.detail["field"] == "latitude"
    else:
        raise AssertionError("Expected invalid latitude to be rejected")


def test_rejects_invalid_longitude():
    try:
        _validate_location(-1.2864, 181)
    except HTTPException as exc:
        assert exc.status_code == 400
        assert exc.detail["field"] == "longitude"
    else:
        raise AssertionError("Expected invalid longitude to be rejected")
