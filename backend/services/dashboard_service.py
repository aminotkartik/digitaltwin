"""
VitalSync — dashboard assembly.

One call returns everything the clinical dashboard renders at a single instant:
patient header, twin status, risk card, sensor cards, chart series, baseline
comparison, twin domains, timeline and alerts.  Both the dashboard route and the
simulation step route use it, so the replay and the static view can never
disagree.

The payload is assembled from cached per-patient arrays (risk series, features)
and only the attribution is computed on the fly, which keeps a replay tick well
inside a browser frame budget.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.services import prediction_service, timeline_service, twin_service
from backend.services.patient_service import PatientRecord
from backend.services.prediction_service import most_recent_night, risk_band
from backend.settings import settings


def build_dashboard(
    record: PatientRecord,
    index: int,
    predictor=None,
    session: Optional[Dict[str, Any]] = None,
    history_hours: float = 6.0,
    explain: bool = True,
    reveal_outcome: bool = False,
    timeline_hours: float = 6.0,
) -> Dict[str, Any]:
    index = record.clip_index(index)
    anchor = record.timestamp_at(index)
    probability = float(record.risk_primary[index])
    profile = record.profile
    ehr = profile.get("ehr", {})

    prediction = prediction_service.build_prediction(
        record, index, predictor, explain=explain, reveal_outcome=reveal_outcome
    )
    prediction["already_above_threshold"] = _above_threshold(record, index)
    twin = twin_service.build_twin_state(record, index)
    comparison = twin_service.build_baseline_comparison(record, index)
    cards = twin_service.build_sensor_cards(record, index)
    timeline = timeline_service.build_timeline(record, index, hours=timeline_hours)
    chart = _chart_series(record, index, history_hours, prediction["trajectory"])

    return {
        "generated_at": anchor.isoformat(),
        "server_time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "patient": {
            "patient_id": record.patient_id,
            "mrn": profile.get("mrn"),
            "name": record.name,
            "age": profile.get("age"),
            "sex": profile.get("sex"),
            "primary_condition": profile.get("primary_condition"),
            "conditions": [d["name"] for d in ehr.get("diagnoses", []) if d.get("status") == "active"],
            "care_setting": profile.get("care_setting"),
            "bmi": ehr.get("demographics", {}).get("bmi"),
            "hba1c_pct": _lab(ehr, "HbA1c"),
            "blood_pressure": _latest_bp(ehr),
            "medications": [
                f"{m['name']} {m['dose']} {m['frequency']}"
                for m in ehr.get("medications", [])
                if str(m.get("status", "")).lower() == "current"
            ],
            "data_classification": "SYNTHETIC — NOT REAL PATIENT DATA",
            "is_demo_patient": bool(profile.get("is_demo_patient", False)),
        },
        "twin_status": {
            "label": "Digital Twin Status",
            "value": twin["synchronisation"]["status"],
            "last_updated": anchor.strftime("%H:%M:%S"),
            "last_sample": anchor.isoformat(),
            "next_update_in_seconds": settings.sampling_interval_minutes * 60,
            "sources_online": _sources_online(record, index),
            "feature_count": twin["synchronisation"].get("feature_count"),
        },
        "clock": {
            "timestamp": anchor.isoformat(),
            "clock": anchor.strftime("%H:%M:%S"),
            "date": anchor.strftime("%a %d %b %Y"),
            "index": index,
            "phase": "history" if index < record.now_index else ("live" if index == record.now_index else "future-replay"),
            "is_default_now": index == record.now_index,
            "stream_start": record.stream.start.isoformat(),
            "stream_end": record.stream.end.isoformat(),
            "now_index": record.now_index,
            "last_index": record.last_index,
        },
        "prediction": prediction,
        "twin": twin,
        "baseline_comparison": comparison,
        "sensor_cards": cards,
        "chart": chart,
        "fusion": twin_service.build_fusion(record, index),
        "timeline": timeline,
        "alerts": _alerts(record, index, timeline),
        "sleep": {
            "last_night": most_recent_night(record, index),
            "nights": record.stream.sleep_nights,
        },
        "session": session,
        "disclaimer": settings.disclaimer,
    }


def _above_threshold(record: PatientRecord, index: int) -> Dict[str, Any]:
    current = float(record.frame["glucose_mgdl"].iloc[index])
    threshold = float(settings.glucose_high_threshold_mgdl)
    if current < threshold:
        return {"above_threshold": False, "current_mgdl": round(current, 1), "threshold_mgdl": threshold}
    step = settings.sampling_interval_minutes
    start = max(0, index - int(180 / step))
    window = record.frame["glucose_mgdl"].iloc[start : index + 1].astype(float).to_numpy()
    run = 0
    for value in window[::-1]:
        if value < threshold:
            break
        run += 1
    return {
        "above_threshold": True,
        "current_mgdl": round(current, 1),
        "threshold_mgdl": threshold,
        "minutes_above_threshold": int(run * step),
        "note": (
            f"Glucose is already {current - threshold:.0f} mg/dL above the {int(threshold)} mg/dL threshold and has "
            f"been for roughly {run * step} minutes. The forecast target is a further rise of at least "
            f"{int(settings.glucose_rise_delta_mgdl)} mg/dL; sustained hyperglycaemia is reported separately."
        ),
    }


def _chart_series(record: PatientRecord, index: int, history_hours: float, trajectory: Dict[str, Any]) -> Dict[str, Any]:
    history = record.history(index, history_hours)
    stride = max(1, int(5 / settings.sampling_interval_minutes))
    sampled = history.iloc[::stride]
    start_index = max(0, index - len(history) + 1)
    risk = record.risk_primary[start_index : index + 1][::stride]
    baseline = record.baseline
    return {
        "past": {
            "timestamps": [t.isoformat() for t in sampled["timestamp"]],
            "clocks": [t.strftime("%H:%M") for t in sampled["timestamp"]],
            "glucose_mgdl": [int(v) for v in sampled["glucose_mgdl"]],
            "risk": [round(float(v), 4) for v in risk],
            "heart_rate_bpm": [int(v) for v in sampled["heart_rate_bpm"]],
            "hrv_rmssd_ms": [int(v) for v in sampled["hrv_rmssd_ms"]],
            "steps_5min": [int(v) for v in sampled["steps_5min"]],
        },
        "forecast": trajectory,
        "bands": {
            "glucose_high_mgdl": settings.glucose_high_threshold_mgdl,
            "glucose_low_mgdl": settings.glucose_low_threshold_mgdl,
            "personal_baseline_band": [baseline.get("glucose_p10"), baseline.get("glucose_p90")],
            "personal_median": baseline.get("glucose_median"),
            "risk_moderate": settings.risk_moderate_min,
            "risk_high": settings.risk_high_min,
        },
        "regions": {
            "past_hours": history_hours,
            "current_index": len(sampled) - 1,
            "forecast_minutes": settings.forecast_horizon_minutes,
        },
    }


def _sources_online(record: PatientRecord, index: int) -> List[Dict[str, Any]]:
    quality = prediction_service.assess_data_quality(record, index)
    return [
        {"key": check["key"], "source": check["source"], "online": check["present"], "age_minutes": check["age_minutes"], "required": check["required"]}
        for check in quality["checks"]
    ]


def _alerts(record: PatientRecord, index: int, timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Most recent actionable alerts, newest first."""
    alerts: List[Dict[str, Any]] = []
    for event in reversed(timeline["events"]):
        if event["severity"] not in ("warning", "critical"):
            continue
        alerts.append(
            {
                "id": f"{event['timestamp']}|{event['title']}",
                "timestamp": event["timestamp"],
                "clock": event["clock"],
                "level": event["severity"],
                "title": event["title"],
                "detail": event["detail"],
                "category": event["category"],
                "category_label": event["category_label"],
                "risk_at_time": round(float(record.risk_primary[event["index"]]), 4),
            }
        )
        if len(alerts) >= 12:
            break
    probability = float(record.risk_primary[index])
    if probability >= settings.risk_high_min:
        alerts.insert(
            0,
            {
                "id": f"current-risk-high-{record.timestamp_at(index).isoformat()}",
                "timestamp": record.timestamp_at(index).isoformat(),
                "clock": record.timestamp_at(index).strftime("%H:%M"),
                "level": "critical",
                "title": f"Predicted 2-hour risk HIGH ({probability * 100:.0f}%)",
                "detail": (
                    f"Model forecasts a {probability * 100:.0f}% probability of a significant glucose elevation "
                    f"within {settings.forecast_horizon_minutes // 60} hours."
                ),
                "category": "alert",
                "category_label": "Alert",
                "risk_at_time": round(probability, 4),
            },
        )
    return alerts[:14]


def _lab(ehr: Dict[str, Any], test: str) -> Optional[float]:
    for row in ehr.get("lab_results", []) or []:
        if str(row.get("test", "")).lower() == test.lower():
            return row.get("value")
    return None


def _latest_bp(ehr: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    history = (ehr.get("vitals_history", {}) or {}).get("blood_pressure") or []
    return history[0] if history else None
