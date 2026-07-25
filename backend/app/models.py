from datetime import datetime
from typing import Any, Literal

# Pydantic models define and validate the shape of API data.
from pydantic import BaseModel, Field


# Allowed report statuses in the report lifecycle.
ReportStatus = Literal["pending", "analyzed", "needs_review", "reviewed", "failed"]

# Allowed priority labels shown to the dashboard.
Priority = Literal["low", "medium", "high", "urgent"]


# One AI-detected class and its confidence score.
class Detection(BaseModel):
    label: Literal["organic_waste", "drain_blockage", "standing_water"]
    confidence: float = Field(ge=0, le=1)


# AI/scoring result saved in the predictions table and returned to frontend.
class Prediction(BaseModel):
    model_version: str
    classes: list[Detection]
    max_confidence: float = Field(ge=0, le=1)
    severity_score: int = Field(ge=0, le=100)
    risk_proxy: int = Field(ge=0, le=100)
    priority: Priority
    explanation: str
    inference_ms: int


# Main report response returned by create, list, detail, retry, and review endpoints.
class Report(BaseModel):
    id: Any
    market: str
    latitude: float
    longitude: float
    note: str | None = None
    image_url: str | None = None
    status: ReportStatus
    created_at: datetime | None = None
    reviewed_at: datetime | None = None
    prediction: Prediction | None = None


# Wrapper response for the report list endpoint.
class ReportListResponse(BaseModel):
    reports: list[Report]


# Body accepted by the review endpoint.
class ReviewRequest(BaseModel):
    reviewer_alias: str | None = None
