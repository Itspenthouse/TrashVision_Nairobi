"""
config.py — one place for every setting.

Why not just hard-code "yolov8n.pt" wherever we need it? Because settings change
per machine and per environment (your laptop vs the deployed server). Reading
them from environment variables / a .env file means you change behaviour WITHOUT
editing code and WITHOUT committing secrets. This is the SDD's ".env locally,
.env.example for names only" rule.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path to the ml/ folder, so paths work no matter where you run from.
ML_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Pydantic-settings automatically overrides any field below from an env var
    # of the SAME name (e.g. MODEL_WEIGHTS=... in .env). Case-insensitive.
    model_config = SettingsConfigDict(
        env_file=ML_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Our fields start with "model_" (model_weights, model_version). Pydantic
        # reserves that prefix by default and warns; this tells it that's fine.
        protected_namespaces=(),
    )

    # --- Model ---
    # "yolov8n.pt" = the smallest ("n" for nano) pretrained YOLOv8. Downloaded
    # once on first run. Later, point this at your fine-tuned weights, e.g.
    # models/trashvision_best.pt — and nothing else in the code changes.
    model_weights: str = "models/trashvision_best.pt"

    # A human-readable id stored with every prediction. CHANGE THIS when you
    # swap models, so every result is traceable to the model that made it.
    model_version: str = "trashvision-custom-v1"

    # Detections below this confidence are thrown away before scoring.
    confidence_threshold: float = 0.25

    max_image_mb: int = 10  # reject uploads bigger than this


settings = Settings()
