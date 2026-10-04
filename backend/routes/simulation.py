"""
VitalSync — simulation (replay) routes.

Controls: start, pause, resume, reset, step (+15 / +30 / +60 minutes), jump and
state.  Every step returns a complete dashboard payload so the UI can render the
whole clinical picture from a single response while replaying.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.routes.deps import get_record, resolve_index, safe_predictor
from backend.services.dashboard_service import build_dashboard
from backend.services.patient_service import PatientNotFoundError
from backend.services.simulation_service import SessionNotFoundError, get_simulation_service
from backend.settings import settings

router = APIRouter(prefix="/simulation", tags=["simulation"])


class StartRequest(BaseModel):
    patient_id: Optional[str] = None
    from_clock: Optional[str] = Field(None, description='clock time to start from, e.g. "09:00"')
    speed_minutes_per_tick: int = Field(5, ge=1, le=120)
    tick_interval_ms: int = Field(900, ge=150, le=5000)
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    session_id: Optional[str] = None
    minutes: Optional[int] = Field(None, ge=1, le=240, description="defaults to the session tick size")
    include_dashboard: bool = True


class SessionRequest(BaseModel):
    session_id: str


class SpeedRequest(BaseModel):
    session_id: str
    speed_minutes_per_tick: int = Field(5, ge=1, le=120)
    tick_interval_ms: Optional[int] = Field(None, ge=150, le=5000)


class JumpRequest(BaseModel):
    session_id: str
    minutes: int = Field(..., description="signed offset in simulated minutes")


def _service():
    return get_simulation_service()


def _dashboard_for(session, include: bool = True) -> Optional[Dict[str, Any]]:
    if not include:
        return None
    record = get_record(session.patient_id)
    return build_dashboard(
        record,
        session.index,
        predictor=safe_predictor(),
        session=_service().public_state(session),
        reveal_outcome=True,
    )


@router.post("/start")
def start_simulation(payload: StartRequest) -> Dict[str, Any]:
    try:
        session = _service().start(
            patient_id=payload.patient_id,
            from_clock=payload.from_clock,
            speed_minutes_per_tick=payload.speed_minutes_per_tick,
            tick_interval_ms=payload.tick_interval_ms,
            session_id=payload.session_id,
        )
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"error": "patient_not_found", "patient_id": exc.patient_id})
    return {
        "session": _service().public_state(session),
        "dashboard": _dashboard_for(session),
        "controls": _controls(session),
    }


@router.post("/step")
def step_simulation(payload: StepRequest) -> Dict[str, Any]:
    try:
        result = _service().step(payload.session_id, payload.minutes)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "session_id": payload.session_id})
    session = _service().get(result["session"]["session_id"])
    return {
        **result,
        "dashboard": _dashboard_for(session, payload.include_dashboard),
        "controls": _controls(session),
    }


@router.post("/pause")
def pause_simulation(payload: SessionRequest) -> Dict[str, Any]:
    session = _require(payload.session_id)
    session = _service().stop(session.session_id)
    return {"session": _service().public_state(session), "controls": _controls(session)}


@router.post("/resume")
def resume_simulation(payload: SessionRequest) -> Dict[str, Any]:
    session = _require(payload.session_id)
    session = _service().resume(session.session_id)
    return {"session": _service().public_state(session), "controls": _controls(session)}


@router.post("/reset")
def reset_simulation(payload: SessionRequest) -> Dict[str, Any]:
    session = _require(payload.session_id)
    session = _service().reset(session.session_id)
    return {
        "session": _service().public_state(session),
        "dashboard": _dashboard_for(session),
        "controls": _controls(session),
    }


@router.post("/jump")
def jump_simulation(payload: JumpRequest) -> Dict[str, Any]:
    """Signed jump through the recorded stream (used by +15 min / +30 min / +1 h)."""
    session = _require(payload.session_id)
    service = _service()
    steps = payload.minutes
    if steps > 0:
        result = service.step(session.session_id, steps)
        session = service.get(result["session"]["session_id"])
    else:
        record = get_record(session.patient_id)
        move = int(round(-steps / settings.sampling_interval_minutes))
        with service._lock:
            session.index = record.clip_index(session.index - move)
            session.minutes_advanced = max(0, session.minutes_advanced + steps)
            session.updated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return {
        "session": service._public_state(session),
        "dashboard": _dashboard_for(session),
        "controls": _controls(session),
    }


@router.post("/speed")
def set_speed(payload: SpeedRequest) -> Dict[str, Any]:
    session = _service().set_speed(payload.session_id, payload.speed_minutes_per_tick, payload.tick_interval_ms)
    return {"session": _service().public_state(session), "controls": _controls(session)}


@router.get("/state")
def simulation_state(session_id: str) -> Dict[str, Any]:
    session = _require(session_id)
    service = _service()
    return {"session": service._public_state(session), "controls": _controls(session)}


@router.get("/sessions")
def list_sessions() -> Dict[str, Any]:
    return {"sessions": _service().list_sessions(), "ttl_seconds": 3600}


@router.get("/snapshot")
def snapshot(
    session_id: Optional[str] = Query(None),
    patient_id: Optional[str] = Query(None),
    at: Optional[str] = Query(None),
    index: Optional[int] = Query(None),
) -> Dict[str, Any]:
    """Full dashboard payload for one instant — with or without a session."""
    if session_id:
        session = _require(session_id)
        record = get_record(session.patient_id)
        return {
            "session": _service().public_state(session),
            "dashboard": build_dashboard(
                record, session.index, predictor=safe_predictor(),
                session=_service().public_state(session), reveal_outcome=True,
            ),
            "controls": _controls(session),
        }
    target = patient_id or settings.demo_patient_id
    record = get_record(target)
    position = resolve_index(record, index=index, at=at)
    return {
        "session": None,
        "dashboard": build_dashboard(record, position, predictor=safe_predictor()),
        "controls": _controls(None),
    }


def _require(session_id: str):
    try:
        return _service().get(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail={"error": "session_not_found", "session_id": session_id})


def _controls(session) -> Dict[str, Any]:
    """Button state for the replay control bar."""
    quick = [
        {"key": "step_15", "label": "+15 min", "minutes": 15},
        {"key": "step_30", "label": "+30 min", "minutes": 30},
        {"key": "step_60", "label": "+1 h", "minutes": 60},
    ]
    return {
        "actions": [
            {"key": "start", "label": "Start simulation", "endpoint": "POST /api/simulation/start"},
            {"key": "pause", "label": "Pause", "endpoint": "POST /api/simulation/pause", "enabled": bool(session and session.running)},
            {"key": "resume", "label": "Play", "endpoint": "POST /api/simulation/resume", "enabled": bool(session and not session.running)},
            {"key": "reset", "label": "Reset", "endpoint": "POST /api/simulation/reset", "enabled": bool(session)},
        ] + quick,
        "speed_options": [
            {"minutes_per_tick": 5, "label": "×1 (5 min / tick)"},
            {"minutes_per_tick": 15, "label": "×3 (15 min / tick)"},
            {"minutes_per_tick": 30, "label": "×6 (30 min / tick)"},
            {"minutes_per_tick": 60, "label": "×12 (1 h / tick)"},
        ],
        "current_speed_minutes_per_tick": session.speed_minutes_per_tick if session else 5,
        "tick_interval_ms": session.tick_interval_ms if session else 900,
        "note": "The replay advances through a recorded synthetic day; no value is generated on the fly.",
    }
