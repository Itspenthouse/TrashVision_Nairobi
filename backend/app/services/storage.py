import os
from pathlib import Path
from uuid import uuid4

# FastAPI upload and error helpers.
from fastapi import HTTPException, UploadFile, status

from app.databases import get_supabase


# Accepted image MIME types and their file extensions.
ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

# Maximum image size in bytes; default is 5MB.
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))


# Reads and validates an uploaded image.
async def validate_image(image: UploadFile) -> bytes:
    # Reject unsupported file types before reading or storing them.
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "invalid_file_type",
                "message": "Only JPG, PNG, and WebP images are allowed.",
                "field": "image",
            },
        )

    # Read the uploaded file into memory.
    content = await image.read()

    # Reject empty uploads.
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "empty_file",
                "message": "Uploaded image is empty.",
                "field": "image",
            },
        )

    # Reject images larger than the configured limit.
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "file_too_large",
                "message": "Image must be 5MB or smaller.",
                "field": "image",
            },
        )

    return content


# Builds a unique, safe storage path for each uploaded image.
def build_image_key(filename: str, content_type: str) -> str:
    extension = ALLOWED_IMAGE_TYPES[content_type]
    safe_stem = Path(filename or "report").stem.replace(" ", "-")[:40] or "report"
    return f"reports/{uuid4()}-{safe_stem}{extension}"


# Uploads the image bytes to Supabase Storage and returns a public URL.
def upload_report_image(image_key: str, content: bytes, content_type: str) -> str:
    # BUCKET_NAME defaults to reports, matching your Supabase storage bucket.
    bucket_name = os.getenv("BUCKET_NAME", "reports")

    # Select the bucket from the Supabase client.
    bucket = get_supabase().storage.from_(bucket_name)

    # Upload the image using the generated key.
    bucket.upload(
        image_key,
        content,
        {"content-type": content_type, "upsert": "false"},
    )

    # Return the image URL that can be stored in the reports table.
    return bucket.get_public_url(image_key)
