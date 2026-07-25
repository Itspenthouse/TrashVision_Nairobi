# TrashVision Nairobi

TrashVision is a real report-to-review system for visible waste, blocked drains,
and standing water in Nairobi markets. It contains:

- `index.html`: mobile-first reporting, dashboard, map, and review interface.
- `backend/`: FastAPI lifecycle API backed by Supabase.
- `ml/`: custom YOLO inference service and reproducible training pipeline.

There is no seeded data, randomized inference, or offline simulation. The web
client shows an explicit service error when the API is unavailable.

## Architecture

1. The browser sends a photo, market, coordinates, and optional note.
2. The backend validates and stores the original image and report.
3. The backend sends the actual image bytes to the ML `/predict` endpoint.
4. The ML service runs the configured custom checkpoint and returns detections.
5. The backend stores the prediction and exposes it to the dashboard and map.

## Local setup

Create separate Python environments for the backend and ML service:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

```powershell
cd ml
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.service:app --reload --port 8001
```

Serve the frontend from the repository root:

```powershell
python -m http.server 8765
```

Open `http://127.0.0.1:8765`. The frontend API defaults to
`http://127.0.0.1:8000`. Override it with `window.TRASHVISION_API_URL` before
the application script or the `trashvision_api_url` local-storage value.

## Required environment

Backend:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE`
- `BUCKET_NAME`
- `FRONTEND_ORIGINS`
- `ML_SERVICE_URL`

ML:

- `MODEL_WEIGHTS`
- `MODEL_VERSION`
- `CONFIDENCE_THRESHOLD`
- `MAX_IMAGE_MB`

## Real image training

The included TACO preparation script downloads the official annotations and
source images, then converts genuine `Food waste` boxes into YOLO
`organic_waste` labels:

```powershell
cd ml
python training/prepare_taco.py
python training/train.py
Copy-Item training/runs/detect/trashvision/weights/best.pt models/trashvision_best.pt
```

TACO does not label `drain_blockage` or `standing_water`. Add consented,
locally relevant images for those classes using
`ml/training/ANNOTATION_GUIDE.md`. Do not infer those labels from filenames or
reuse unrelated litter labels.

## Verification

```powershell
cd backend
pytest

cd ..\ml
pytest
```

See `ml/model_card.md` for model scope and current dataset limitations.
