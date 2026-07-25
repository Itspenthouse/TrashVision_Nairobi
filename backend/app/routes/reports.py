from datetime import UTC, datetime
from typing import Any

# FastAPI tools used for routes, file uploads, forms, errors, and query params.
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status

# Backend helpers for Supabase, response models, AI analysis, and image storage.
from app.databases import get_supabase
from app.models import Prediction, Report, ReportListResponse, ReviewRequest
from app.services.ai import analyze_image, analyze_image_url
from app.services.storage import build_image_key, upload_report_image, validate_image


# All routes in this file start with /api/v1 and are grouped under "reports".
router = APIRouter(prefix="/api/v1", tags=["reports"])


# Creates a standard database error response for the frontend.
def _database_error(message: str = "Database operation failed.") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail={"code": "database_error", "message": message},
    )


# Creates a standard storage error response when image upload fails.
def _storage_error(message: str = "Image upload failed.") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail={"code": "storage_error", "message": message, "field": "image"},
    )


# Checks that latitude and longitude are valid map coordinates.
def _validate_location(latitude: float, longitude: float) -> None:
    if not -90 <= latitude <= 90:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "invalid_latitude",
                "message": "Latitude must be between -90 and 90.",
                "field": "latitude",
            },
        )
    if not -180 <= longitude <= 180:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "invalid_longitude",
                "message": "Longitude must be between -180 and 180.",
                "field": "longitude",
            },
        )


# Converts a Supabase prediction row into a Pydantic Prediction object.
def _prediction_from_row(row: dict[str, Any] | None) -> Prediction | None:
    if not row:
        return None
    classes = row.get("classes_json") or row.get("classes") or []
    return Prediction(
        model_version=row["model_version"],
        classes=classes,
        max_confidence=row["max_confidence"],
        severity_score=row["severity_score"],
        risk_proxy=row["risk_proxy"],
        priority=row["priority"],
        explanation=row["explanation"],
        inference_ms=row["inference_ms"],
    )


# Picks the newest prediction attached to a report.
def _latest_prediction(row: dict[str, Any]) -> Prediction | None:
    predictions = row.get("predictions")
    if isinstance(predictions, list) and predictions:
        latest = sorted(
            predictions,
            key=lambda prediction: prediction.get("created_at") or "",
        )[-1]
        return _prediction_from_row(latest)
    if isinstance(predictions, dict):
        return _prediction_from_row(predictions)
    return None


# Converts a Supabase report row into the API response format.
def _report_from_row(row: dict[str, Any]) -> Report:
    return Report(
        id=row["id"],
        market=row["market"],
        latitude=row["latitude"],
        longitude=row["longitude"],
        note=row.get("note"),
        image_url=row.get("image_url"),
        status=row["status"],
        created_at=row.get("created_at"),
        reviewed_at=row.get("reviewed_at"),
        prediction=_latest_prediction(row),
    )


# Builds the dictionary that gets inserted into the predictions table.
def _prediction_payload(report_id: Any, prediction: Prediction) -> dict[str, Any]:
    return {
        "report_id": report_id,
        "model_version": prediction.model_version,
        "classes_json": [
            detection.model_dump() for detection in prediction.classes
        ],
        "max_confidence": prediction.max_confidence,
        "severity_score": prediction.severity_score,
        "risk_proxy": prediction.risk_proxy,
        "priority": prediction.priority,
        "explanation": prediction.explanation,
        "inference_ms": prediction.inference_ms,
    }


# Fetches one report by ID, including its linked prediction rows.
def _fetch_report(report_id: str) -> dict[str, Any]:
    try:
        response = (
            get_supabase()
            .table("reports")
            .select("*, predictions(*)")
            .eq("id", report_id)
            .single()
            .execute()
        )
    except Exception as exc:
        raise _database_error("Could not fetch report.") from exc

    if not response.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": "Report was not found."},
        )
    return response.data


# Creates a new report from a multipart form upload.
# Flow: validate fields -> upload image -> save report -> run AI stub -> save prediction.
@router.post("/reports", response_model=Report, status_code=status.HTTP_201_CREATED)
async def upload_report(
    market: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    note: str = Form(""),
    image: UploadFile = File(...),
):
    # Clean text inputs so blank spaces do not count as real values.
    market = market.strip()
    note = note.strip()

    # Market name is required for dashboard filtering and map labels.
    if not market:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "required",
                "message": "Market is required.",
                "field": "market",
            },
        )
    if len(market) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "too_long",
                "message": "Market must be 100 characters or fewer.",
                "field": "market",
            },
        )
    if len(note) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "too_long",
                "message": "Note must be 1000 characters or fewer.",
                "field": "note",
            },
        )

    # Validate GPS/manual coordinates before saving anything.
    _validate_location(latitude, longitude)

    # Read and validate the uploaded image before sending it to storage.
    image_content = await validate_image(image)

    # Build a safe unique storage path and upload the file to Supabase Storage.
    image_key = build_image_key(image.filename or "report", image.content_type or "")
    try:
        image_url = upload_report_image(image_key, image_content, image.content_type or "")
    except Exception as exc:
        raise _storage_error() from exc

    # This is the metadata saved into the reports table.
    report_payload = {
        "market": market,
        "latitude": latitude,
        "longitude": longitude,
        "note": note,
        "image_url": image_url,
        "status": "pending",
    }

    # Create the report row first with pending status.
    try:
        report_response = (
            get_supabase().table("reports").insert(report_payload).execute()
        )
    except Exception as exc:
        raise _database_error("Could not create report.") from exc

    # Run the AI/scoring placeholder after the report exists.
    report_row = report_response.data[0]
    try:
        prediction = await analyze_image(
            image_content,
            image.filename or image_key,
            image.content_type or "image/jpeg",
        )
    except Exception as exc:
        get_supabase().table("reports").update({"status": "failed"}).eq(
            "id", report_row["id"]
        ).execute()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "inference_error",
                "message": "The image analysis service could not process this report.",
            },
        ) from exc

    # Low-confidence predictions are sent to human review instead of marked analyzed.
    final_status = "needs_review" if prediction.max_confidence < 0.5 else "analyzed"

    # Save prediction results and update the report status.
    try:
        get_supabase().table("predictions").insert(
            _prediction_payload(report_row["id"], prediction)
        ).execute()
        update_response = (
            get_supabase()
            .table("reports")
            .update({"status": final_status})
            .eq("id", report_row["id"])
            .execute()
        )
    except Exception as exc:
        # If prediction saving fails, mark the report as failed so it can be retried.
        get_supabase().table("reports").update({"status": "failed"}).eq(
            "id", report_row["id"]
        ).execute()
        raise _database_error("Could not save prediction.") from exc

    # Return the final report response, including the prediction.
    updated_report = update_response.data[0] if update_response.data else report_row
    updated_report["status"] = final_status
    updated_report["predictions"] = [_prediction_payload(report_row["id"], prediction)]
    return _report_from_row(updated_report)


# Lists reports for the dashboard, with optional filters.
@router.get("/reports", response_model=ReportListResponse)
def list_reports(
    status_filter: str | None = Query(default=None, alias="status"),
    priority: str | None = None,
    market: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
):
    try:
        # Start a Supabase query that fetches reports with their predictions.
        query = (
            get_supabase()
            .table("reports")
            .select("*, predictions(*)")
            .order("created_at", desc=True)
            .limit(limit)
        )

        # Apply filters only when the frontend sends them.
        if status_filter:
            query = query.eq("status", status_filter)
        if market:
            query = query.ilike("market", f"%{market}%")
        response = query.execute()
    except Exception as exc:
        raise _database_error("Could not list reports.") from exc

    reports = [_report_from_row(row) for row in response.data or []]

    # Priority lives inside prediction rows, so it is filtered after fetch.
    if priority:
        reports = [
            report
            for report in reports
            if report.prediction and report.prediction.priority == priority
        ]
    return ReportListResponse(reports=reports)


# Returns one report detail page by report ID.
@router.get("/reports/{report_id}", response_model=Report)
def get_report(report_id: str):
    return _report_from_row(_fetch_report(report_id))


# Re-runs analysis for a report that failed or needs a fresh prediction.
@router.post("/reports/{report_id}/retry", response_model=Report)
async def retry_report(report_id: str):
    report_row = _fetch_report(report_id)

    image_url = report_row.get("image_url")
    if not image_url:
        raise _storage_error("The stored report image is unavailable.")
    try:
        prediction = await analyze_image_url(image_url)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "inference_error",
                "message": "The image analysis service could not process this report.",
            },
        ) from exc
    final_status = "needs_review" if prediction.max_confidence < 0.5 else "analyzed"

    try:
        # Store the new prediction and reset reviewed_at because this is a fresh analysis.
        get_supabase().table("predictions").insert(
            _prediction_payload(report_row["id"], prediction)
        ).execute()
        response = (
            get_supabase()
            .table("reports")
            .update({"status": final_status, "reviewed_at": None})
            .eq("id", report_row["id"])
            .execute()
        )
    except Exception as exc:
        raise _database_error("Could not retry analysis.") from exc

    updated_report = response.data[0] if response.data else report_row
    updated_report["predictions"] = [_prediction_payload(report_row["id"], prediction)]
    return _report_from_row(updated_report)


# Marks a report as reviewed by a human coordinator.
@router.post("/reports/{report_id}/review", response_model=Report)
def review_report(report_id: str, review: ReviewRequest):
    report_row = _fetch_report(report_id)

    # Store an ISO timestamp so Supabase can save it in reviewed_at.
    reviewed_at = datetime.now(UTC).isoformat()

    try:
        # Update the main report status to reviewed.
        response = (
            get_supabase()
            .table("reports")
            .update({"status": "reviewed", "reviewed_at": reviewed_at})
            .eq("id", report_id)
            .execute()
        )
    except Exception as exc:
        raise _database_error("Could not mark report as reviewed.") from exc

    try:
        # Save an optional audit event; if this fails, the review still succeeds.
        get_supabase().table("review_events").insert(
            {
                "report_id": report_id,
                "old_status": report_row["status"],
                "new_status": "reviewed",
                "reviewer_alias": review.reviewer_alias,
            }
        ).execute()
    except Exception:
        pass

    updated_report = response.data[0] if response.data else report_row
    updated_report["predictions"] = report_row.get("predictions")
    return _report_from_row(updated_report)
