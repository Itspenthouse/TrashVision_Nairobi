"""
schemas.py — THE CONTRACT.

Every piece of data that flows in or out of your ML service is described here
using Pydantic models. A Pydantic model is just a Python class that also:
  1. validates data (rejects a confidence of 5.0, a negative box, etc.), and
  2. converts cleanly to/from JSON (what the backend actually sends over HTTP).

Think of this file as the "menu" your teammates order from. If they know these
shapes, they can build against your service before it even works.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Fixed vocabularies (Enums). Using an Enum instead of a plain string means a
# typo like "organic_wast" is caught immediately instead of silently breaking.
# ---------------------------------------------------------------------------
class WasteClass(str, Enum):
    """The three things TrashVision looks for. Nothing else is a valid class."""
    organic_waste = "organic_waste"
    drain_blockage = "drain_blockage"
    standing_water = "standing_water"


class Priority(str, Enum):
    """Human-facing urgency label derived from the severity score."""
    low = "low"
    medium = "medium"
    high = "high"
    urgent = "urgent"


# ---------------------------------------------------------------------------
# One detected object in an image.
# ---------------------------------------------------------------------------
class Detection(BaseModel):
    """A single box the model drew, e.g. 'organic_waste, 0.82 confident, here'."""
    class_name: WasteClass
    confidence: float = Field(ge=0.0, le=1.0, description="0..1, how sure the model is")
    # Bounding box in pixels: top-left (x1,y1) to bottom-right (x2,y2).
    bbox: list[float] = Field(min_length=4, max_length=4, description="[x1, y1, x2, y2]")


# ---------------------------------------------------------------------------
# The full result your scoring produces for one image. This is the object the
# backend stores in its `predictions` table (see the SDD data model).
# ---------------------------------------------------------------------------
class PredictionResult(BaseModel):
    model_version: str = Field(description="Which model produced this, for auditability")
    detections: list[Detection] = Field(default_factory=list)

    # Per-class "evidence" in 0..1 — the intermediate signals the score is built
    # from. Exposing these is what makes the score *explainable* rather than a
    # magic number.
    evidence: dict[str, float] = Field(default_factory=dict)
    max_confidence: float = Field(ge=0.0, le=1.0, default=0.0)

    severity_score: int = Field(ge=0, le=100)
    risk_proxy: int = Field(ge=0, le=100)
    priority: Priority
    explanation: str

    # A hint for the backend's status lifecycle. True when the model is unsure
    # and a human should look (SDD "responsible AI" rule).
    needs_review: bool = False

    inference_ms: Optional[float] = Field(default=None, description="model time only")
