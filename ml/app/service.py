"""
service.py — the HTTP wrapper around your ML lane.

This turns your Python functions into a web service the backend can call. Two
endpoints, matching the SDD:
    GET  /health   -> is the service alive and is the model loaded?
    POST /predict  -> send an image, get back a full PredictionResult

Run it with:   uvicorn app.service:app --reload --port 8001
Then open:     http://127.0.0.1:8001/docs   (auto-generated, clickable API docs!)

NOTE ON DECOUPLING: notice this file imports run_inference and score_detections
but contains almost no logic itself. That's on purpose. The backend could ALSO
`from app.inference import run_inference` and skip HTTP entirely — that's why you
get "both" integration styles for free.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from .config import settings
from .inference import run_inference, warm_up
from .schemas import PredictionResult
from .scoring import score_detections

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the server starts. We warm up the model here so the first
    real request is fast (SDD "model warm-up"). Everything before `yield` is
    startup; anything after would be shutdown cleanup.
    """
    try:
        warm_up()
        print(f"[startup] model '{settings.model_version}' ready.")
    except Exception as exc:  # don't crash the server if weights are missing
        print(f"[startup] WARNING: model failed to load: {exc}")
    yield


app = FastAPI(title="TrashVision ML Service", version=settings.model_version, lifespan=lifespan)


def _error(status: int, code: str, message: str, field: str | None = None):
    """
    Consistent error shape for the whole service (SDD error rule): always
    include code, message, optional field, and a request_id — and NEVER leak a
    stack trace to the caller.
    """
    return JSONResponse(
        status_code=status,
        content={
            "code": code,
            "message": message,
            "field": field,
            "request_id": str(uuid.uuid4()),
        },
    )


@app.get("/health")
def health():
    """Cheap liveness check. Reports whether the model is loaded yet."""
    from .inference import _model  # None until first load / warm-up

    return {
        "status": "ok",
        "model_version": settings.model_version,
        "model_loaded": _model is not None,
    }


@app.post("/predict", response_model=PredictionResult)
async def predict(image: UploadFile = File(...)):
    """
    The heart of the service:
        upload image -> run_inference (YOLO) -> score_detections -> JSON result
    """
    # 1) Validate the file type BEFORE doing any expensive work.
    if image.content_type not in ALLOWED_TYPES:
        return _error(
            415, "unsupported_media_type",
            f"Only {', '.join(sorted(ALLOWED_TYPES))} are accepted.", field="image",
        )

    # 2) Read bytes and enforce the size limit.
    data = await image.read()
    if len(data) > settings.max_image_mb * 1024 * 1024:
        return _error(
            413, "file_too_large",
            f"Image exceeds the {settings.max_image_mb} MB limit.", field="image",
        )
    if not data:
        return _error(400, "empty_file", "The uploaded image was empty.", field="image")

    # 3) Run the model. If decoding/inference fails, return a clean error — this
    #    is the SDD "failed" path (the report can later be retried).
    try:
        detections, width, height, inference_ms = run_inference(data)
    except Exception:
        return _error(
            500, "inference_failed",
            "The model could not process this image. The report can be retried.",
        )

    # 4) Turn detections into the explainable score. Pure, deterministic, tested.
    result = score_detections(
        detections=detections,
        image_width=width,
        image_height=height,
        model_version=settings.model_version,
        inference_ms=inference_ms,
    )
    return result
