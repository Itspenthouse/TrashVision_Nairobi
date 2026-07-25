from fastapi import APIRouter


# APIRouter groups related endpoints before they are added to the main app.
router = APIRouter()


# Health check endpoint used by developers and deployment platforms like Render.
@router.get("/health")
def health():
    return {"status": "healthy"}
