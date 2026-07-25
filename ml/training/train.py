"""
train.py — fine-tune YOLO on YOUR 3 TrashVision classes.

You run this LATER, once you've collected and labeled ~150-300 images (see the
SDD dataset plan). Today it's here so the path is real and documented.

WHAT "FINE-TUNING" MEANS (the single most important ML idea for you):
  Training a detector from scratch needs millions of images. Instead we take a
  model that ALREADY learned to see edges, shapes, and textures from COCO, and
  we nudge it to recognise our 3 new classes using a small dataset. Reusing that
  prior knowledge is called TRANSFER LEARNING. It's why 200 images can be enough
  for an initial baseline instead of 200,000.

RUN IT (from the ml/ folder, with the venv active):
    python training/train.py

Outputs land in training/runs/detect/trashvision/weights/:
    best.pt   <- the checkpoint with the best VALIDATION score (use THIS one)
    last.pt   <- the final epoch (usually ignore; it can be overfit)

Then point the service at it by setting in .env:
    MODEL_WEIGHTS=training/runs/detect/trashvision/weights/best.pt
    MODEL_VERSION=trashvision-yolov8n-v1
"""

from pathlib import Path

import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset"
RUNTIME_DATA_YAML = HERE / ".data.resolved.yaml"


def main() -> None:
    RUNTIME_DATA_YAML.write_text(
        yaml.safe_dump(
            {
                "path": DATASET.as_posix(),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {
                    0: "organic_waste",
                    1: "drain_blockage",
                    2: "standing_water",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    # Start from the small pretrained model (transfer learning).
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(RUNTIME_DATA_YAML),
        # --- The knobs you'll actually tune ---
        epochs=25,          # practical baseline; early stopping still applies.
        imgsz=640,          # images are resized to 640x640 for training.
        batch=8,            # images processed at once. Lower if you run out of RAM.
        patience=15,        # stop early if validation hasn't improved in 15 epochs.
        seed=42,            # fixes randomness so runs are REPRODUCIBLE (SDD rule).
        # --- Bookkeeping so every run is traceable ---
        project=str(HERE / "runs" / "detect"),
        name="trashvision",
        exist_ok=True,
        # CPU-only machine (no GPU) -> device="cpu". Training will be slow-ish;
        # that's expected. A small dataset + nano model keeps it manageable.
        device=None,        # automatically uses a GPU when one is available.
    )

    # Evaluate the best checkpoint on the held-out VALIDATION set and print the
    # headline metric, mAP (mean Average Precision) — higher is better, max 1.0.
    metrics = model.val()
    print("\n=== Validation metrics ===")
    print(f"mAP50    : {metrics.box.map50:.3f}   (IoU>=0.50, the friendly metric)")
    print(f"mAP50-95 : {metrics.box.map:.3f}   (stricter average, the honest one)")
    print("\nBest weights saved under runs/detect/trashvision/weights/best.pt")


if __name__ == "__main__":
    main()
