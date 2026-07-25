import os
from uuid import uuid4

# FastAPI is the main web framework used to create the API app.
from fastapi import FastAPI, HTTPException

# RequestValidationError lets us customize errors caused by bad request data.
from fastapi.exceptions import RequestValidationError

# CORS allows the frontend app to call this backend from another domain.
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse

# These routers contain endpoint groups from separate files.
from app.routes.health import router as health_router
from app.routes.reports import router as reports_router


# This creates the FastAPI application instance.
app = FastAPI()

# FRONTEND_ORIGINS controls which frontend URLs can call this backend.
# In local development it can be "*"; in deployment it should be your frontend URL.
frontend_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "*").split(",")
    if origin.strip()
]

# This middleware enables browser requests from your frontend to the backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=frontend_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# This middleware adds a unique request ID to every request and response.
# It helps trace errors when debugging locally or on Render logs.
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


# This handler converts FastAPI validation errors into frontend-friendly JSON.
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(part) for part in error.get("loc", []) if part != "body")
    return JSONResponse(
        status_code=422,
        content={
            "code": "validation_error",
            "message": error.get("msg", "Request validation failed."),
            "field": field or None,
            "request_id": request.state.request_id,
        },
    )


# This handler makes intentional API errors return a consistent shape.
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        content = dict(exc.detail)
    else:
        content = {"code": "http_error", "message": str(exc.detail)}
    content["request_id"] = request.state.request_id
    return JSONResponse(status_code=exc.status_code, content=content)


# This handler prevents unexpected Python errors from leaking stack traces.
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "code": "server_error",
            "message": "Unexpected server error.",
            "request_id": request.state.request_id,
        },
    )


# Root endpoint used as a simple "API is alive" message.
@app.get("/")
def root():
    return {"message": "TrashVision API"}


# Register the separate route groups on the main app.
app.include_router(health_router)
app.include_router(reports_router)
