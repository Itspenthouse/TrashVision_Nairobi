# TrashVision — ML Service (Michael's lane)

The AI/ML + scoring part of TrashVision Nairobi. It takes an image and returns
an explainable severity score. Built to be called by the backend either over
HTTP (`POST /predict`) or by importing the Python functions directly.

## What's in here

```
ml/
├─ app/
│  ├─ schemas.py     # THE CONTRACT — data shapes shared with the backend
│  ├─ scoring.py     # deterministic 0–100 severity formula (pure Python)
│  ├─ inference.py   # the ONLY file that touches the YOLO model
│  ├─ service.py     # FastAPI app: /predict and /health
│  └─ config.py      # settings, read from .env
├─ training/
│  ├─ data.yaml      # dataset config for fine-tuning later
│  └─ train.py       # transfer-learning script for your 3 classes
├─ tests/            # pytest unit tests (scoring boundaries)
├─ demo/images/      # put ~5 approved demo photos here
├─ predict_cli.py    # run the pipeline on a local image, no server needed
├─ model_card.md     # honest "nutrition label" for the model
└─ requirements.txt
```

## The flow, in one line

`image bytes → inference.run_inference() [YOLO] → scoring.score_detections() [math] → PredictionResult (JSON)`

## Setup (Windows, Python 3.13)

From this `ml/` folder:

```bash
# 1) create + activate the virtual environment (already created if you followed along)
py -3.13 -m venv .venv
.venv\Scripts\activate

# 2) install dependencies
pip install -r requirements.txt

# 3) copy the env template
copy .env.example .env
```

> The first prediction auto-downloads `yolov8n.pt` (~6 MB). One-time.

## Run it

**Unit tests (no model needed — proves the scoring math):**
```bash
pytest
```

**Try a prediction on a local image (no server):**
```bash
python predict_cli.py demo/images/your_photo.jpg --save-annotated
```

**Start the API service:**
```bash
uvicorn app.service:app --reload --port 8001
```
Then open **http://127.0.0.1:8001/docs** — interactive, clickable API docs. You
can upload an image to `/predict` right in the browser.

## The API (what the backend calls)

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | is the service up? is the model loaded? |
| POST | `/predict` | multipart image upload → full `PredictionResult` |

Example `PredictionResult`:
```json
{
  "model_version": "pretrained-yolov8n-coco-proxy-v0",
  "detections": [{"class_name": "organic_waste", "confidence": 0.82, "bbox": [12, 30, 210, 240]}],
  "evidence": {"organic_waste": 0.71, "drain_blockage": 0.0, "standing_water": 0.0},
  "max_confidence": 0.82,
  "severity_score": 40,
  "risk_proxy": 4,
  "priority": "medium",
  "explanation": "AI suggests mainly organic waste (evidence 0.71). Severity 40/100 ...",
  "needs_review": false,
  "inference_ms": 88.5
}
```

## Going from demo model → real model (later)

1. Collect + label ~150–300 images (see `training/data.yaml` for the format).
2. `python training/train.py`
3. In `.env`, set `MODEL_WEIGHTS` to the new `best.pt`, bump `MODEL_VERSION`,
   and set `USE_COCO_PROXY=false`.
4. Re-run the tests and `/predict`. Nothing else changes. That's the payoff of
   keeping the model behind one clean interface.

See `model_card.md` for the honest limitations to present in the demo.
```
