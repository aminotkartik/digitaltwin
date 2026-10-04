"""
VitalSync — system routes: health, runtime configuration, data provenance and
the machine-readable architecture description used by both the Architecture
view and the generated PDF.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter

from backend.routes.deps import model_status
from backend.services import feature_engineering as fe
from backend.services.groq_service import status as groq_status
from backend.services.patient_service import get_repository
from backend.services.synthetic_data import STREAM_COLUMNS
from backend.settings import DATA_DIR, REPO_ROOT, SENSOR_DIR, settings

router = APIRouter(tags=["system"])

STARTED_AT = datetime.utcnow().isoformat(timespec="seconds") + "Z"


@router.get("/health")
def health() -> Dict[str, Any]:
    """Liveness plus a compact readiness summary for the status bar."""
    repository_ok = True
    patients: List[str] = []
    try:
        repository = get_repository()
        patients = repository.ids()
    except Exception:
        repository_ok = False

    model = model_status()
    groq = groq_status()
    ready = repository_ok and bool(patients) and model.get("trained", False)
    return {
        "status": "ok" if ready else "degraded",
        "ready": ready,
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "started_at": STARTED_AT,
        "server_time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "demo_clock": {
            "stream_end": settings.demo_stream_end_clock,
            "now": settings.demo_now_clock,
            "simulation_start": settings.demo_simulation_start_clock,
        },
        "components": {
            "patient_repository": {"ok": repository_ok, "patients": patients},
            "model": model,
            "groq": {"enabled": groq["enabled"], "model": groq["model"], "reason": groq["reason"]},
            "synthetic_generator": {"ok": True, "module": "backend/services/synthetic_data.py"},
        },
        "disclaimer": settings.disclaimer,
    }


@router.get("/config")
def runtime_config() -> Dict[str, Any]:
    """
    Non-secret runtime configuration for the UI.

    Deliberately excludes anything credential-shaped: the Groq key never leaves
    the server process.
    """
    return {
        "app": {"name": settings.app_name, "version": settings.app_version, "environment": settings.environment},
        "event_definition": {
            "high_threshold_mgdl": settings.glucose_high_threshold_mgdl,
            "rise_delta_mgdl": settings.glucose_rise_delta_mgdl,
            "low_threshold_mgdl": settings.glucose_low_threshold_mgdl,
            "horizon_minutes": settings.forecast_horizon_minutes,
            "description": (
                f"Glucose reaches ≥ {int(settings.glucose_high_threshold_mgdl)} mg/dL for at least 10 consecutive "
                f"samples within the next {settings.forecast_horizon_minutes} minutes, and that peak is at least "
                f"{int(settings.glucose_rise_delta_mgdl)} mg/dL above the value at the prediction instant."
            ),
            "threshold_reference": "ATTD international consensus on time in range (upper bound 180 mg/dL)",
        },
        "risk_bands": [
            {"key": key, "label": label, "range": value}
            for (key, value), label in zip(settings.risk_bands.items(), ("LOW", "MODERATE", "HIGH"))
        ],
        "sampling": {
            "interval_minutes": settings.sampling_interval_minutes,
            "stream_hours": settings.stream_hours,
            "baseline_onboarding_hours": settings.baseline_onboarding_hours,
            "horizon_steps": settings.horizon_steps,
        },
        "demo": {
            "patient_id": settings.demo_patient_id,
            "now_clock": settings.demo_now_clock,
            "simulation_start_clock": settings.demo_simulation_start_clock,
            "stream_end_clock": settings.demo_stream_end_clock,
        },
        "model": model_status(),
        "groq": groq_status(),
        "preferred_model": settings.preferred_model,
        "feature_count": len(fe.FEATURE_NAMES),
        "disclaimer": settings.disclaimer,
    }


@router.get("/datasources")
def data_sources() -> Dict[str, Any]:
    """Everything the Data Sources / provenance page needs to cite its inputs."""
    sources: List[Dict[str, Any]] = []

    for column, device, kind in (
        ("glucose_mgdl", "Continuous glucose monitor (interstitial fluid)", "DYNAMIC"),
        ("heart_rate_bpm", "Wrist wearable — photoplethysmography", "DYNAMIC"),
        ("hrv_rmssd_ms", "Wrist wearable — inter-beat intervals (RMSSD)", "DYNAMIC"),
        ("steps_5min", "Wrist wearable — accelerometer step counts", "DYNAMIC"),
        ("activity_met", "Wrist wearable — derived metabolic equivalents", "DYNAMIC"),
        ("spo2_pct", "Wrist wearable — reflectance pulse oximetry", "DYNAMIC"),
        ("sleep_stage", "Wrist wearable — hypnogram (wake/light/deep/REM)", "DYNAMIC"),
    ):
        sources.append(
            {
                "id": column,
                "name": column.replace("_", " ").title(),
                "kind": kind,
                "fusion_layer": "Live physiological stream",
                "device": device,
                "sampling": f"every {settings.sampling_interval_minutes} minutes",
                "precision": _precision_for(column),
                "unit": _unit_for(column),
                "origin": "Generated by backend/services/synthetic_data.py",
                "stored_at": f"backend/data/sensor_data/<patient_id>_signals.csv",
                "real_patient_data": False,
            }
        )

    for name, description, path in (
        ("Demographics", "Age, sex, BMI, waist circumference", "backend/data/patients.json → ehr.demographics"),
        ("Diagnoses (ICD-10)", "Type 2 diabetes, hypertension, dyslipidaemia, obesity, CKD stage, prediabetes", "backend/data/patients.json → ehr.diagnoses"),
        ("Medication history", "Current and discontinued glucose-lowering therapy with dates and doses", "backend/data/patients.json → ehr.medications"),
        ("Laboratory results", "HbA1c (with trend), fasting glucose, lipids, eGFR, urine ACR", "backend/data/patients.json → ehr.lab_results"),
        ("Vitals history", "Clinic blood pressure and weight readings", "backend/data/patients.json → ehr.vitals_history"),
        ("Family history", "First-degree relatives with diabetes or cardiovascular disease", "backend/data/patients.json → ehr.family_history"),
        ("Risk factors", "Documented lifestyle and clinical risk factors", "backend/data/patients.json → ehr.risk_factors"),
        ("Previous CGM report", "Prior 14-day ambulatory glucose profile: mean, CV, time in range, episodes", "backend/data/patients.json → ehr.previous_glucose_instability"),
    ):
        sources.append(
            {
                "id": name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", ""),
                "name": name,
                "kind": "STATIC",
                "fusion_layer": "Historical patient record",
                "device": "Longitudinal EHR extract",
                "sampling": "point-in-time at the moment of prediction",
                "precision": "as recorded",
                "unit": "",
                "description": description,
                "origin": "Authored synthetic record; CGM report fields recomputed from generated history",
                "stored_at": path,
                "real_patient_data": False,
            }
        )

    files = _artifact_files()
    return {
        "classification": "SYNTHETIC — NOT REAL PATIENT DATA",
        "statement": (
            "Every value served by this API originates from a deterministic simulator committed to this repository. "
            "No real patient data, no protected health information and no third-party clinical dataset is used."
        ),
        "fusion_model": {
            "static_inputs": sum(1 for s in sources if s["kind"] == "STATIC"),
            "dynamic_inputs": sum(1 for s in sources if s["kind"] == "DYNAMIC"),
            "features": len(fe.FEATURE_NAMES),
            "feature_groups": [
                {"key": key, "label": label, "count": sum(1 for s in fe.FEATURE_SPECS if s.group == key)}
                for key, label in fe.FEATURE_GROUPS.items()
            ],
            "description": (
                "Static record fields and live signal features are concatenated into a single fixed-order vector. "
                "Personal baselines are estimated only from an onboarding window that strictly precedes every scored "
                "sample, so no future information can reach the model."
            ),
        },
        "sources": sources,
        "stream_columns": STREAM_COLUMNS,
        "files": files,
        "lineage": [
            {"step": 1, "action": "Author synthetic patient records", "artifact": "backend/data/patients.json"},
            {"step": 2, "action": "Generate multi-day physiology streams deterministically", "artifact": "backend/data/sensor_data/*_signals.csv"},
            {"step": 3, "action": "Persist provenance metadata per stream", "artifact": "backend/data/sensor_data/*_metadata.json"},
            {"step": 4, "action": "Generate the training cohort (96 patients × 6 days)", "artifact": "datasets/synthetic/cohort_features.csv.gz"},
            {"step": 5, "action": "Train, validate and export model artifacts", "artifact": "model/artifacts/*"},
            {"step": 6, "action": "Serve fused twin state, predictions and explanations", "artifact": "backend/routes/*"},
        ],
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def _precision_for(column: str) -> str:
    return {
        "glucose_mgdl": "integer mg/dL",
        "heart_rate_bpm": "integer bpm",
        "hrv_rmssd_ms": "integer ms",
        "steps_5min": "integer steps per interval",
        "activity_met": "0.1 MET",
        "spo2_pct": "integer %",
        "sleep_stage": "categorical",
    }.get(column, "")


def _unit_for(column: str) -> str:
    return {
        "glucose_mgdl": "mg/dL",
        "heart_rate_bpm": "bpm",
        "hrv_rmssd_ms": "ms",
        "steps_5min": "steps",
        "activity_met": "MET",
        "spo2_pct": "%",
        "sleep_stage": "",
    }.get(column, "")


def _artifact_files() -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    for path in sorted(SENSOR_DIR.glob("*")):
        stat = path.stat()
        files.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "bytes": stat.st_size,
                "sha256_16": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
                "modified": datetime.utcfromtimestamp(stat.st_mtime).isoformat(timespec="seconds") + "Z",
            }
        )
    for relative in ("backend/data/patients.json", "datasets/synthetic/dataset_metadata.json"):
        path = REPO_ROOT / relative
        if path.exists():
            stat = path.stat()
            files.append(
                {
                    "path": relative,
                    "bytes": stat.st_size,
                    "sha256_16": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
                    "modified": datetime.utcfromtimestamp(stat.st_mtime).isoformat(timespec="seconds") + "Z",
                }
            )
    return files


@router.get("/architecture")
def architecture() -> Dict[str, Any]:
    """Machine-readable architecture used by the SVG view and the PDF export."""
    return {
        "title": f"{settings.app_name} — System Architecture",
        "subtitle": "Clinical Digital Twin for 2-Hour Blood Glucose Spike Prediction",
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "layers": [
            {
                "id": "sources",
                "label": "1 · Data Sources",
                "colour": "static",
                "nodes": [
                    {"id": "ehr", "label": "Longitudinal EHR", "detail": "demographics · diagnoses (ICD-10) · medications · HbA1c & labs · BP · BMI · family history · prior CGM report"},
                    {"id": "cgm", "label": "Continuous Glucose Monitor", "detail": f"interstitial glucose every {settings.sampling_interval_minutes} min"},
                    {"id": "wearable", "label": "Wrist Wearable", "detail": "heart rate · HRV (RMSSD) · steps · activity MET · SpO₂ · hypnogram"},
                    {"id": "app", "label": "Patient App Logs", "detail": "meal carbohydrate entries with timestamps"},
                ],
            },
            {
                "id": "ingest",
                "label": "2 · Ingestion & Normalisation",
                "colour": "process",
                "nodes": [
                    {"id": "ingest_api", "label": "Signal Ingestion", "detail": "timestamp alignment to a common grid, unit normalisation, plausibility bounds"},
                    {"id": "quality", "label": "Data Quality Gate", "detail": "freshness per source, gap detection, completeness score"},
                    {"id": "baseline", "label": "Personal Baseline Estimator", "detail": f"{settings.baseline_onboarding_hours} h onboarding window strictly before every scored sample"},
                ],
            },
            {
                "id": "fusion",
                "label": "3 · Feature Fusion (Digital Twin Core)",
                "colour": "fusion",
                "nodes": [
                    {"id": "features", "label": f"Fused Feature Vector — {len(fe.FEATURE_NAMES)} inputs", "detail": " · ".join(f"{label} ({sum(1 for s in fe.FEATURE_SPECS if s.group == key)})" for key, label in fe.FEATURE_GROUPS.items())},
                    {"id": "state", "label": "Twin State Engine", "detail": "5 normalised domains: metabolic stability · glucose stability · cardiovascular state · recovery · activity"},
                ],
            },
            {
                "id": "ml",
                "label": "4 · Prediction & Explainability",
                "colour": "model",
                "nodes": [
                    {"id": "model", "label": "Risk Model", "detail": "gradient boosting (primary) + logistic regression (transparent comparison), patient-level splits"},
                    {"id": "explain", "label": "Explainer", "detail": "TreeSHAP local attribution · permutation importance · logistic coefficients"},
                    {"id": "trajectory", "label": "Trajectory Estimator", "detail": "risk-conditional empirical distribution + personal damped-trend projection"},
                    {"id": "confidence", "label": "Confidence Estimator", "detail": "signal completeness · model agreement · measured calibration error"},
                ],
            },
            {
                "id": "interpret",
                "label": "5 · Clinical Interpretation",
                "colour": "llm",
                "nodes": [
                    {"id": "context", "label": "Structured Context Builder", "detail": "assembles only measured and computed values"},
                    {"id": "groq", "label": "Groq (Llama 3.3 70B)", "detail": "natural-language clinical summary · never produces a probability"},
                    {"id": "fallback", "label": "Deterministic Fallback", "detail": "template interpretation when the key is absent or the call fails"},
                ],
            },
            {
                "id": "presentation",
                "label": "6 · Clinical Presentation",
                "colour": "ui",
                "nodes": [
                    {"id": "api", "label": "FastAPI REST Layer", "detail": "patients · twin · prediction · timeline · insights · simulation · model · profile · provenance"},
                    {"id": "ui", "label": "Vanilla JS Dashboard", "detail": "Chart.js (vendored) + inline SVG · desktop-first responsive layout"},
                    {"id": "sim", "label": "Simulation Controller", "detail": "play · pause · reset · +15 min · +30 min · +1 h"},
                ],
            },
        ],
        "flows": [
            {"from": "ehr", "to": "baseline", "label": "static context"},
            {"from": "cgm", "to": "ingest_api", "label": "5-min glucose"},
            {"from": "wearable", "to": "ingest_api", "label": "HR · HRV · steps · SpO₂ · sleep"},
            {"from": "app", "to": "ingest_api", "label": "meal logs"},
            {"from": "ingest_api", "to": "quality", "label": "aligned samples"},
            {"from": "quality", "to": "baseline", "label": "clean signal"},
            {"from": "baseline", "to": "features", "label": "personal reference values"},
            {"from": "quality", "to": "features", "label": "live window"},
            {"from": "features", "to": "state", "label": "fused vector"},
            {"from": "features", "to": "model", "label": "fused vector"},
            {"from": "model", "to": "explain", "label": "probability"},
            {"from": "model", "to": "trajectory", "label": "probability"},
            {"from": "model", "to": "confidence", "label": "probability"},
            {"from": "state", "to": "context", "label": "twin state"},
            {"from": "explain", "to": "context", "label": "attributions"},
            {"from": "confidence", "to": "context", "label": "confidence"},
            {"from": "context", "to": "groq", "label": "structured JSON"},
            {"from": "groq", "to": "api", "label": "clinical prose"},
            {"from": "context", "to": "fallback", "label": "if unavailable"},
            {"from": "fallback", "to": "api", "label": "template prose"},
            {"from": "explain", "to": "api", "label": "attributions"},
            {"from": "trajectory", "to": "api", "label": "forecast fan"},
            {"from": "state", "to": "api", "label": "domains"},
            {"from": "api", "to": "ui", "label": "JSON"},
            {"from": "sim", "to": "api", "label": "replay clock"},
        ],
        "guardrails": [
            {"title": "No leakage", "detail": "Personal baselines come from an onboarding window that strictly precedes every scored sample; features are backward-looking only."},
            {"title": "No LLM in the prediction path", "detail": "The probability is produced solely by the trained statistical model; Groq only phrases observations about it."},
            {"title": "No fabricated metrics", "detail": "Every reported number is computed from the committed synthetic cohort at training time."},
            {"title": "Graceful degradation", "detail": "If Groq is unavailable the deterministic template path answers; if model artifacts are missing a labelled heuristic keeps the UI alive."},
            {"title": "Secrets stay server-side", "detail": "GROQ_API_KEY is read from .env by the backend only and is never serialised into any response."},
            {"title": "Synthetic data only", "detail": "No real patient data at any point; every screen states this."},
        ],
        "directories": [
            {"path": "frontend/", "purpose": "HTML shell, CSS design system and ES-module JavaScript (api, charts, twin, simulation, prediction, insights, pages)"},
            {"path": "backend/", "purpose": "FastAPI application: routes, services, model wrapper, settings, synthetic data"},
            {"path": "model/", "purpose": "training, evaluation, feature-engineering adapter and exported artifacts"},
            {"path": "datasets/synthetic/", "purpose": "cohort generator, generated cohort, patient profiles and dataset metadata"},
            {"path": "docs/", "purpose": "generated architecture and presentation PDFs plus the PDF generator"},
            {"path": "tests/", "purpose": "pytest suite covering the simulator, features, labelling, API and fallbacks"},
            {"path": "scripts/", "purpose": "one-command setup, run, train and PDF generation helpers"},
        ],
    }
