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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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

# Rendered only while the single-page app has not been built yet. It is a plain
# service-status page, deliberately not a mock-up of the clinical dashboard.
_API_STATUS_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__APP__ — API running</title>
<style>
  :root { color-scheme: light; }
  body { margin: 0; background: #f7f6f3; color: #23272b;
         font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  main { max-width: 760px; margin: 0 auto; padding: 56px 24px 40px; }
  h1 { font-size: 21px; margin: 0 0 4px; letter-spacing: .01em; }
  p.sub { margin: 0 0 28px; color: #5b6470; }
  .card { background: #fff; border: 1px solid #e3e1dc; border-radius: 8px; padding: 20px 22px;
          box-shadow: 0 1px 2px rgba(20,25,30,.04); margin-bottom: 18px; }
  .status { display: inline-flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 600;
            color: #2f6f4f; background: #f0f7f2; border: 1px solid #d6e8dc; border-radius: 999px;
            padding: 3px 11px; }
  .status::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: #4a9d70; }
  h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .08em; color: #6b7280;
       margin: 0 0 12px; font-weight: 600; }
  ul { list-style: none; margin: 0; padding: 0; }
  li { border-bottom: 1px solid #f0efeb; }
  li:last-child { border-bottom: 0; }
  a { color: #245e7a; text-decoration: none; display: flex; justify-content: space-between;
      gap: 16px; padding: 9px 0; }
  a:hover { color: #16455c; text-decoration: underline; }
  a span { color: #8a9099; font-size: 13px; white-space: nowrap; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
  .note { color: #5b6470; font-size: 13.5px; margin: 0; }
  .disclaimer { color: #7a818b; font-size: 12.5px; border-top: 1px solid #e3e1dc; padding-top: 14px; }
</style>
</head>
<body>
<main>
  <h1>__APP__</h1>
  <p class="sub">__TAGLINE__ &middot; version __VERSION__</p>

  <div class="card">
    <span class="status">API running</span>
    <p class="note" style="margin-top:14px">
      The backend is live. <strong>The clinical dashboard (single-page app) has not been built yet</strong>,
      so this page is a service index rather than the product interface. Every endpoint below returns
      real data computed from the synthetic patient streams and the trained model.
    </p>
  </div>

  <div class="card">
    <h2>Interactive documentation</h2>
    <ul>
      <li><a href="/docs"><code>/docs</code><span>Swagger UI — try every endpoint</span></a></li>
      <li><a href="/redoc"><code>/redoc</code><span>ReDoc</span></a></li>
      <li><a href="/openapi.json"><code>/openapi.json</code><span>OpenAPI schema</span></a></li>
    </ul>
  </div>

  <div class="card">
    <h2>Patient &amp; digital twin</h2>
    <ul>
      <li><a href="/api/patients"><code>/api/patients</code><span>cohort index</span></a></li>
      <li><a href="/api/patients/DT-1047"><code>/api/patients/DT-1047</code><span>record + header</span></a></li>
      <li><a href="/api/patients/DT-1047/ehr"><code>/api/patients/DT-1047/ehr</code><span>static clinical history</span></a></li>
      <li><a href="/api/patients/DT-1047/twin"><code>/api/patients/DT-1047/twin</code><span>twin state + radar</span></a></li>
      <li><a href="/api/patients/DT-1047/twin/fusion"><code>/api/patients/DT-1047/twin/fusion</code><span>data fusion panel</span></a></li>
      <li><a href="/api/patients/DT-1047/twin/baseline"><code>/api/patients/DT-1047/twin/baseline</code><span>compared with baseline</span></a></li>
      <li><a href="/api/patients/DT-1047/sensors"><code>/api/patients/DT-1047/sensors</code><span>sensor cards</span></a></li>
    </ul>
  </div>

  <div class="card">
    <h2>Prediction &amp; explainability</h2>
    <ul>
      <li><a href="/api/patients/DT-1047/prediction"><code>/api/patients/DT-1047/prediction</code><span>2-hour risk card</span></a></li>
      <li><a href="/api/patients/DT-1047/prediction/explain"><code>&hellip;/prediction/explain</code><span>why is the risk high?</span></a></li>
      <li><a href="/api/patients/DT-1047/prediction/trajectory"><code>&hellip;/prediction/trajectory</code><span>forecast fan</span></a></li>
      <li><a href="/api/patients/DT-1047/charts"><code>/api/patients/DT-1047/charts</code><span>glucose chart payload</span></a></li>
      <li><a href="/api/patients/DT-1047/timeline"><code>/api/patients/DT-1047/timeline</code><span>clinical timeline</span></a></li>
      <li><a href="/api/model/metrics"><code>/api/model/metrics</code><span>validation metrics</span></a></li>
      <li><a href="/api/model/importance"><code>/api/model/importance</code><span>feature importance</span></a></li>
      <li><a href="/api/model/calibration"><code>/api/model/calibration</code><span>calibration curve</span></a></li>
      <li><a href="/api/model/limitations"><code>/api/model/limitations</code><span>limitations &amp; ethics</span></a></li>
    </ul>
  </div>

  <div class="card">
    <h2>Simulation, interpretation &amp; platform</h2>
    <ul>
      <li><a href="/api/simulation/snapshot?patient_id=DT-1047"><code>/api/simulation/snapshot</code><span>full dashboard payload</span></a></li>
      <li><a href="/api/insights/status"><code>/api/insights/status</code><span>Groq layer state</span></a></li>
      <li><a href="/api/insights/context?patient_id=DT-1047"><code>/api/insights/context</code><span>prompt context sent to Groq</span></a></li>
      <li><a href="/api/datasources"><code>/api/datasources</code><span>data provenance</span></a></li>
      <li><a href="/api/architecture"><code>/api/architecture</code><span>architecture diagram data</span></a></li>
      <li><a href="/api/profile"><code>/api/profile</code><span>about / team</span></a></li>
      <li><a href="/api/health"><code>/api/health</code><span>health check</span></a></li>
    </ul>
  </div>

  <p class="disclaimer">__DISCLAIMER__ &nbsp;&middot;&nbsp; __CLASSIFICATION__</p>
</main>
</body>
</html>
"""


@app.get("/", include_in_schema=False)
def index() -> Any:
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE)
    page = (
        _API_STATUS_PAGE.replace("__APP__", settings.app_name)
        .replace("__TAGLINE__", settings.app_tagline)
        .replace("__VERSION__", settings.app_version)
        .replace("__DISCLAIMER__", settings.disclaimer)
        .replace("__CLASSIFICATION__", settings.data_classification)
    )
    return HTMLResponse(content=page, status_code=200)


if FRONTEND_DIR.exists():
    # Mounted last so /api and / always take precedence over static files.
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
