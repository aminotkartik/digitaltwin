"""
VitalSync — Groq clinical interpretation routes.

POST /api/insights builds the structured clinical context from the twin and asks
Groq to phrase it.  The endpoint also returns the context that was sent, so a
reviewer can verify that the language model only ever saw measured values.

If Groq is not configured or the call fails, the deterministic template path
answers instead and says so — the presentation never breaks because of the LLM.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Query
from pydantic import BaseModel, Field

from backend.routes.deps import get_record, resolve_index, safe_predictor
from backend.services import groq_service, insights_service, prediction_service, twin_service
from backend.services.groq_service import ACTION_LABELS
from backend.settings import settings

router = APIRouter(prefix="/insights", tags=["insights"])


class InsightRequest(BaseModel):
    patient_id: Optional[str] = Field(None, description="defaults to the demo patient")
    action: str = Field("summarize_changes", description="summarize_changes | explain_risk | review_pattern | clinical_brief")
    at: Optional[str] = Field(None, description="ISO timestamp or clock time such as 10:45")
    index: Optional[int] = None
    session_id: Optional[str] = None
    trend_hours: float = Field(3.0, ge=0.5, le=24.0)
    include_context: bool = Field(True, description="echo the structured context that was sent to the model")


@router.get("/status")
def insight_status() -> Dict[str, Any]:
    """Is the interpretation layer live, and what will happen if it is not?"""
    status = groq_service.status()
    status.update(
        {
            "fallback_behaviour": (
                "Clinical interpretation is generated from deterministic templates built from the same structured "
                "data. Predictions, twin state and explanations are unaffected because they never depend on Groq."
            ),
            "ui_message_when_disabled": "Groq interpretation layer disabled — deterministic prototype mode active.",
            "ui_message_on_failure": "Clinical interpretation unavailable — showing deterministic summary.",
            "key_exposure": "GROQ_API_KEY is read from the server environment only and is never sent to the browser.",
        }
    )
    return status


@router.post("")
async def create_insight(payload: InsightRequest = Body(...)) -> Dict[str, Any]:
    patient_id = payload.patient_id or settings.demo_patient_id
    record = get_record(patient_id)
    position = resolve_index(record, index=payload.index, at=payload.at, session_id=payload.session_id)
    predictor = safe_predictor()

    prediction = prediction_service.build_prediction(record, position, predictor, explain=True, reveal_outcome=False)
    baseline_comparison = twin_service.build_baseline_comparison(record, position)
    twin_state = twin_service.build_twin_state(record, position)
    context = insights_service.build_context(
        record, position, prediction, baseline_comparison, twin_state, trend_hours=payload.trend_hours
    )

    result = await groq_service.generate_insight(context, payload.action)
    response: Dict[str, Any] = {
        "patient_id": patient_id,
        "as_of": record.timestamp_at(position).isoformat(),
        "action": payload.action,
        "action_label": ACTION_LABELS.get(payload.action, payload.action),
        "source": result.source,
        "ok": result.ok,
        "error": result.error,
        "latency_ms": result.latency_ms,
        "groq_enabled": settings.groq_enabled,
        "status_message": (
            None
            if result.source == "groq"
            else (
                "Groq interpretation layer disabled — deterministic prototype mode active."
                if not settings.groq_enabled
                else "Clinical interpretation unavailable — showing deterministic summary."
            )
        ),
        "insight": result.payload,
        "prediction_probability": prediction["probability"],
        "risk_band": prediction["risk"]["label"],
        "provenance": {
            "probability_source": "trained statistical model (never the language model)",
            "context_source": "computed twin state, sensor window and personal baseline",
            "prompt_constraints": [
                "Use only the provided structured patient data.",
                "Do not invent measurements, diagnoses, medications, symptoms, test results, or events.",
                "No treatment instructions; considerations for review only.",
                "Structured JSON output with a fixed key set.",
            ],
        },
    }
    if payload.include_context:
        response["context_sent"] = context
    return response


@router.get("/context")
def insight_context(
    patient_id: Optional[str] = Query(None),
    at: Optional[str] = None,
    index: Optional[int] = None,
    session_id: Optional[str] = None,
    trend_hours: float = Query(3.0, ge=0.5, le=24.0),
) -> Dict[str, Any]:
    """Inspect exactly what the interpretation layer would receive."""
    patient_id = patient_id or settings.demo_patient_id
    record = get_record(patient_id)
    position = resolve_index(record, index=index, at=at, session_id=session_id)
    predictor = safe_predictor()
    prediction = prediction_service.build_prediction(record, position, predictor, explain=True)
    baseline_comparison = twin_service.build_baseline_comparison(record, position)
    twin_state = twin_service.build_twin_state(record, position)
    return {
        "patient_id": patient_id,
        "as_of": record.timestamp_at(position).isoformat(),
        "context": insights_service.build_context(
            record, position, prediction, baseline_comparison, twin_state, trend_hours=trend_hours
        ),
        "system_prompt": groq_service.SYSTEM_PROMPT,
        "actions": [{"key": key, "label": label, "instruction": groq_service.ACTION_INSTRUCTIONS[key]} for key, label in ACTION_LABELS.items()],
    }
