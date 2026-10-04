"""
VitalSync — clinical timeline routes.

Every entry is detected from the recorded signal and the model's own output
history; nothing is authored in the interface.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Query

from backend.routes.deps import get_record, resolve_index
from backend.services import timeline_service

router = APIRouter(prefix="/patients/{patient_id}", tags=["timeline"])


@router.get("/timeline")
def timeline(
    patient_id: str,
    hours: float = Query(8.0, ge=0.5, le=72.0),
    at: Optional[str] = None,
    index: Optional[int] = None,
    session_id: Optional[str] = None,
    category: Optional[str] = Query(None, description="meal | sleep | activity | glucose | cardio | model | alert | twin | outcome"),
    severity: Optional[str] = Query(None, description="info | good | watch | warning | critical"),
) -> Dict[str, Any]:
    record = get_record(patient_id)
    position = resolve_index(record, index=index, at=at, session_id=session_id)
    payload = timeline_service.build_timeline(record, position, hours=hours)
    events = payload["events"]
    if category:
        events = [e for e in events if e["category"] == category]
    if severity:
        events = [e for e in events if e["severity"] == severity]
    payload["events"] = events
    payload["event_count"] = len(events)
    payload["filters"] = {"category": category, "severity": severity}
    return payload


@router.get("/timeline/pattern")
def pattern(
    patient_id: str,
    hours: float = Query(24.0, ge=1.0, le=72.0),
    at: Optional[str] = None,
    index: Optional[int] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregated description of the recent window (also feeds the Groq context)."""
    record = get_record(patient_id)
    position = resolve_index(record, index=index, at=at, session_id=session_id)
    summary = timeline_service.build_pattern_summary(record, position, hours=hours)
    summary["as_of"] = record.timestamp_at(position).isoformat()
    summary["patient_id"] = record.patient_id
    summary["timeline"] = timeline_service.build_timeline(record, position, hours=min(hours, 12.0))["events"][-20:]
    return summary


@router.get("/timeline/day")
def day_view(
    patient_id: str,
    at: Optional[str] = None,
    index: Optional[int] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Full scenario-day view: sleep, meals, activity blocks and the risk trace."""
    record = get_record(patient_id)
    position = resolve_index(record, index=index, at=at, session_id=session_id)
    day = record.timestamp_at(position).normalize()
    frame = record.frame
    mask = frame["timestamp"].dt.normalize() == day
    day_frame = frame[mask]
    if day_frame.empty:
        return {"patient_id": record.patient_id, "date": day.isoformat(), "samples": 0}

    start_index = record.index_at(day_frame["timestamp"].iloc[0])
    end_index = record.index_at(day_frame["timestamp"].iloc[-1])
    stride = 3
    indices = list(range(start_index, end_index + 1, stride))

    day_key = day.strftime("%Y-%m-%d")
    previous_key = (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    # the night that ends on this day started on the previous calendar day
    nights: List[Dict[str, Any]] = [n for n in record.stream.sleep_nights if n["night_of"] in (day_key, previous_key)]
    meals = [m for m in record.stream.meals if _on_day(m.get("datetime"), day)]
    blocks = [b for b in record.stream.activity_blocks if _block_on_day(record, b, day)]

    return {
        "patient_id": record.patient_id,
        "date": day.strftime("%Y-%m-%d"),
        "display_date": day.strftime("%a %d %b %Y"),
        "series": {
            "timestamps": [record.timestamp_at(i).strftime("%H:%M") for i in indices],
            "glucose_mgdl": [int(frame["glucose_mgdl"].iloc[i]) for i in indices],
            "risk": [round(float(record.risk_primary[i]), 4) for i in indices],
            "heart_rate_bpm": [int(frame["heart_rate_bpm"].iloc[i]) for i in indices],
            "hrv_rmssd_ms": [int(frame["hrv_rmssd_ms"].iloc[i]) for i in indices],
            "steps_5min": [int(frame["steps_5min"].iloc[i]) for i in indices],
            "sleep_stage": [str(frame["sleep_stage"].iloc[i]) for i in indices],
        },
        "sleep_nights": nights,
        "meals": [
            {"clock": m["clock"], "label": m["label"], "carbs_g": m["carbs_g"], "logged_in_app": m.get("logged_in_app", True)}
            for m in meals
        ],
        "activity_blocks": blocks,
        "summary": {
            "glucose_mean": round(float(day_frame["glucose_mgdl"].mean()), 1),
            "glucose_max": int(day_frame["glucose_mgdl"].max()),
            "glucose_min": int(day_frame["glucose_mgdl"].min()),
            "time_above_180_pct": round(float((day_frame["glucose_mgdl"] >= 180).mean() * 100), 1),
            "steps": int(day_frame["steps_5min"].sum()),
            "risk_mean": round(float(record.risk_primary[start_index : end_index + 1].mean()), 4),
            "risk_max": round(float(record.risk_primary[start_index : end_index + 1].max()), 4),
        },
        "current_index": position,
        "current_clock": record.timestamp_at(position).strftime("%H:%M"),
    }


def _on_day(timestamp: Any, day: pd.Timestamp) -> bool:
    try:
        return pd.Timestamp(timestamp).normalize() == day
    except Exception:
        return False


def _block_on_day(record, block: Dict[str, Any], day: pd.Timestamp) -> bool:
    try:
        block_day = record.stream.scenario_day + pd.Timedelta(days=int(block.get("day_offset", 0)))
        return block_day.normalize() == day
    except Exception:
        return False
