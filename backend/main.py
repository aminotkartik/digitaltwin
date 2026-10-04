"""
VitalSync — FastAPI application entry point.

Run:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000

The same process serves the JSON API under ``/api`` and the static clinical
dashboard from ``frontend/`` at ``/``, so the prototype needs exactly one
command and no separate web server.  The browser only ever talks to this origin,
which keeps the Groq key and every other secret server-side.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.routes import api_router
from backend.services.patient_service import PatientNotFoundError, get_repository
from backend.services.simulation_service import SessionNotFoundError
from backend.settings import FRONTEND_DIR, settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the twins once so the first dashboard render is not paying for it."""
    started = time.time()
    state: Dict[str, Any] = {"patients": [], "model": "unavailable", "groq": "disabled"}
    try:
        repository = get_repository()
        state["patients"] = repository.ids()
    except Exception as exc:  # pragma: no cover
        print(f"[vitalsync] patient repository failed to build: {exc}")
    try:
        from backend.models.predictor import get_predictor

        predictor = get_predictor()
        state["model"] = f"{predictor.model_id} ({predictor.estimator_name})"
    except Exception as exc:
        state["model"] = f"unavailable — {exc}. Run: python model/train.py"
    if settings.groq_enabled:
        state["groq"] = f"enabled ({settings.groq_model})"
    else:
        state["groq"] = "disabled — deterministic interpretation fallback active"

    app.state.startup = state
    print("=" * 78)
    print(f"  {settings.app_name} — {settings.app_tagline}")
    print(f"  environment : {settings.environment}")
    print(f"  patients    : {', '.join(state['patients']) or 'none'}")
    print(f"  model       : {state['model']}")
    print(f"  groq        : {state['groq']}")
    print(f"  demo clock  : stream ends {settings.demo_stream_end_clock}, now {settings.demo_now_clock}, "
          f"simulation starts {settings.demo_simulation_start_clock}")
    print(f"  ready in    : {time.time() - started:.1f}s")
    print(f"  dashboard   : http://0.0.0.0:{settings.port}/")
    print(f"  api docs    : http://0.0.0.0:{settings.port}/docs")
    print(f"  {settings.disclaimer}")
    print("=" * 78)
    yield


app = FastAPI(
    title=f"{settings.app_name} API",
    version=settings.app_version,
    description=(
        "Clinical Digital Twin API for predicting a significant blood glucose spike two hours in advance.\n\n"
        "**All data is synthetic.** Prototype for research and demonstration — not intended for diagnosis "
        "or medical decision-making."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    # The prototype is same-origin in normal use; these origins exist so the
    # dashboard can also be opened from a file:// page or a sandbox preview host.
    allow_origin_regex=r"(http|https)://(localhost|127\.0\.0\.1|0\.0\.0\.0|[a-z0-9\-]+\.e2b\.app)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


# ---------------------------------------------------------------------------
# Error handling — the API returns structured, actionable errors rather than
# stack traces, so the UI can always explain what went wrong.
# ---------------------------------------------------------------------------
@app.exception_handler(PatientNotFoundError)
async def patient_not_found_handler(request: Request, exc: PatientNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error": "patient_not_found",
            "patient_id": exc.patient_id,
            "available": get_repository().ids(),
            "path": request.url.path,
        },
    )


@app.exception_handler(SessionNotFoundError)
async def session_not_found_handler(request: Request, exc: SessionNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error": "session_not_found",
            "session_id": str(exc),
            "hint": "Start a simulation first: POST /api/simulation/start",
            "path": request.url.path,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "detail": json_safe(exc.errors()), "path": request.url.path},
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    # Never leak internals, but always tell the operator what to look at.
    print(f"[vitalsync] unhandled error on {request.url.path}: {exc.__class__.__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "type": exc.__class__.__name__,
            "path": request.url.path,
            "hint": "See the server console for detail.",
        },
    )


def json_safe(errors: Any) -> Any:
    """Make pydantic validation errors JSON-serialisable."""
    cleaned = []
    for error in errors:
        item = dict(error)
        item.pop("ctx", None)
        item["loc"] = [str(part) for part in item.get("loc", [])]
        cleaned.append(item)
    return cleaned


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
INDEX_FILE = FRONTEND_DIR / "index.html"


@app.get("/", include_in_schema=False)
def index() -> Any:
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE)
    return JSONResponse(
        status_code=200,
        content={
            "app": settings.app_name,
            "status": "api-only",
            "message": "frontend/index.html not found — the API is running.",
            "docs": "/docs",
            "health": "/api/health",
        },
    )


if FRONTEND_DIR.exists():
    # Mounted last so /api and / always take precedence over static files.
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
