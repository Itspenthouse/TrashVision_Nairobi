# TrashVision Backend

FastAPI backend for report upload, Supabase persistence, image storage, AI suggestion stubs, severity scoring, and review workflow.

## Local Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Supabase Setup

1. Create a Supabase project.
2. Open the Supabase SQL editor.
3. Run `schema.sql`.
4. Put your Supabase values in `.env`.

Required environment variables:

```env
SUPABASE_URL=
SUPABASE_SERVICE_ROLE=
BUCKET_NAME=reports
FRONTEND_ORIGINS=http://localhost:3000
MAX_IMAGE_BYTES=5242880
```

## Render

This repo includes `render.yaml`. Render start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```
