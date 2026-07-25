"""
predict_cli.py — run your whole pipeline on a local image, no server needed.

This is your fastest feedback loop while learning: point it at any photo and see
the detections + score printed as JSON. It also (optionally) saves an annotated
copy with boxes drawn, and can build the demo "seed" file the SDD asks for.

Examples (run from the ml/ folder with the venv active):
    python predict_cli.py demo/images/pile.jpg
    python predict_cli.py demo/images/pile.jpg --save-annotated
    python predict_cli.py demo/images/*.jpg --seed demo/seed_predictions.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from app.config import settings
from app.inference import run_inference
from app.scoring import score_detections


def predict_one(path: Path) -> dict:
    data = path.read_bytes()
    detections, w, h, ms = run_inference(data)
    result = score_detections(
        detections=detections,
        image_width=w,
        image_height=h,
        model_version=settings.model_version,
        inference_ms=ms,
    )
    # .model_dump() turns the Pydantic object into a plain dict/JSON.
    return {"image": path.name, **result.model_dump()}


def save_annotated(path: Path) -> None:
    """Draw YOLO's boxes on a copy so you can SEE what the model saw."""
    from ultralytics import YOLO

    model = YOLO(settings.model_weights)
    results = model.predict(str(path), conf=settings.confidence_threshold, verbose=False)
    out = path.with_name(path.stem + "_annotated.jpg")
    results[0].save(filename=str(out))  # ultralytics draws + writes the file
    print(f"  annotated image -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TrashVision inference + scoring.")
    parser.add_argument("images", nargs="+", help="image path(s) or glob(s)")
    parser.add_argument("--save-annotated", action="store_true", help="write *_annotated.jpg")
    parser.add_argument("--seed", metavar="FILE", help="write all results to a JSON seed file")
    args = parser.parse_args()

    # Expand any globs the shell didn't (Windows often doesn't).
    paths: list[Path] = []
    for pattern in args.images:
        paths.extend(Path(p) for p in glob.glob(pattern))
    if not paths:
        print("No matching images found.")
        return

    all_results = []
    for path in paths:
        print(f"\n### {path.name}")
        result = predict_one(path)
        all_results.append(result)
        print(json.dumps(result, indent=2))
        if args.save_annotated:
            save_annotated(path)

    if args.seed:
        Path(args.seed).write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\nSeed data for {len(all_results)} image(s) -> {args.seed}")


if __name__ == "__main__":
    main()
