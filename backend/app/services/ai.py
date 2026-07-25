import os

import httpx

from app.models import Detection, Prediction


ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://127.0.0.1:8001").rstrip("/")
ML_TIMEOUT_SECONDS = float(os.getenv("ML_TIMEOUT_SECONDS", "60"))


def _prediction_from_ml(payload: dict) -> Prediction:
    detections = [
        Detection(label=item["class_name"], confidence=item["confidence"])
        for item in payload.get("detections", [])
    ]
    return Prediction(
        model_version=payload["model_version"],
        classes=detections,
        max_confidence=payload["max_confidence"],
        severity_score=payload["severity_score"],
        risk_proxy=payload["risk_proxy"],
        priority=payload["priority"],
        explanation=payload["explanation"],
        inference_ms=round(payload["inference_ms"]),
    )


async def analyze_image(
    image_content: bytes,
    filename: str,
    content_type: str,
) -> Prediction:
    async with httpx.AsyncClient(timeout=ML_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{ML_SERVICE_URL}/predict",
            files={"image": (filename, image_content, content_type)},
        )
        response.raise_for_status()
        return _prediction_from_ml(response.json())


async def analyze_image_url(image_url: str) -> Prediction:
    async with httpx.AsyncClient(timeout=ML_TIMEOUT_SECONDS) as client:
        image_response = await client.get(image_url)
        image_response.raise_for_status()
        content_type = image_response.headers.get("content-type", "image/jpeg")

    filename = image_url.rsplit("/", 1)[-1].split("?", 1)[0] or "report.jpg"
    return await analyze_image(image_response.content, filename, content_type)
