# TrashVision ML Service

This service runs a custom YOLO checkpoint against uploaded image pixels. It
does not use filename heuristics or COCO class proxies.

## Prepare real data

TACO is a public, MIT-licensed dataset of litter in real environments. The
converter keeps only its genuine `Food waste` annotations:

```powershell
python training/prepare_taco.py
```

For a quick pipeline check, use `--limit 20`. This is still real data, but is
not enough for a production-quality model.

Add locally collected and annotated `drain_blockage` and `standing_water`
images under `training/dataset`. Follow `training/ANNOTATION_GUIDE.md`.

## Train

```powershell
python training/train.py
Copy-Item training/runs/detect/trashvision/weights/best.pt models/trashvision_best.pt
```

The service expects `models/trashvision_best.pt` by default. Configure another
checkpoint using `MODEL_WEIGHTS`.

## Run

```powershell
uvicorn app.service:app --reload --port 8001
```

- `GET /health`: model and service status.
- `POST /predict`: multipart image inference.

## Test

```powershell
pytest
```
