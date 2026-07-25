from app.models import Detection


# Converts AI detections into a deterministic cleanup-priority score.
def calculate_score(detections: list[Detection]) -> tuple[int, int, str, str]:
    # If there are no detections, return a low score with a clear explanation.
    if not detections:
        return 15, 10, "low", "No clear waste or drainage issue was detected."

    # Drainage-related issues get higher weight because they can affect flooding.
    weights = {
        "organic_waste": 35,
        "drain_blockage": 70,
        "standing_water": 60,
    }

    # Use the strongest weighted detection as the main signal.
    max_weighted_score = max(
        weights[detection.label] * detection.confidence for detection in detections
    )

    # Multiple detections increase severity, but the combined signal is capped.
    combined_signal = min(sum(detection.confidence for detection in detections), 1.5)
    severity_score = min(100, round(max_weighted_score + combined_signal * 20))

    # Convert the numeric score into a dashboard-friendly priority label.
    if severity_score >= 80:
        priority = "urgent"
    elif severity_score >= 60:
        priority = "high"
    elif severity_score >= 35:
        priority = "medium"
    else:
        priority = "low"

    # Build a readable explanation for the detail page.
    labels = ", ".join(detection.label for detection in detections)

    # risk_proxy is an integer because your Supabase column is int4.
    risk_proxy = 75 if any(
        detection.label in {"drain_blockage", "standing_water"}
        for detection in detections
    ) else 25

    # This explanation helps users understand that the score is not magic.
    explanation = (
        f"Score is based on detected classes ({labels}), confidence, and drainage risk."
    )

    # Return all scoring outputs used by the predictions table.
    return severity_score, risk_proxy, priority, explanation
