"""
VitalSync — patient routes: roster, full record, EHR view, raw sensor stream and
the chart-ready series used by Live Monitoring.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

from backend.routes.deps import get_record, resolve_index, safe_predictor
from backend.services import feature_engineering as fe
from backend.services import prediction_service, twin_service
from backend.services.patient_service import get_repository
from backend.services.prediction_service import most_recent_night, risk_band
from backend.settings import settings

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("")
def list_patients() -> Dict[str, Any]:
    repository = get_repository()
    roster = repository.roster_metadata()
    patients = repository.all_summaries()
    for patient in patients:
        record = repository.get(patient["patient_id"])
        profile = record.profile
        patient["scenario"] = {
            "label": profile.get("scenario", {}).get("label"),
            "summary": profile.get("scenario", {}).get("summary"),
        }
        patient["hba1c_pct"] = _lab(profile, "HbA1c")
        patient["fasting_glucose_mgdl"] = _lab(profile, "Fasting plasma glucose")
        patient["medications"] = [m["name"] for m in profile.get("ehr", {}).get("medications", []) if str(m.get("status", "")).lower() == "current"]
        index = record.now_index
        patient["twin"] = {
            "composite_index": twin_service.build_twin_state(record, index)["composite_index"],
            "risk": round(float(record.risk_primary[index]), 4),
            "band": risk_band(float(record.risk_primary[index]))["label"],
        }
    return {
        "classification": roster.get("data_classification", "SYNTHETIC — NOT REAL PATIENT DATA"),
        "count": len(patients),
        "demo_patient_id": settings.demo_patient_id,
        "units": roster.get("units", {}),
        "patients": patients,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def _lab(profile: Dict[str, Any], test: str) -> Optional[float]:
    for row in profile.get("ehr", {}).get("lab_results", []) or []:
        if str(row.get("test", "")).lower() == test.lower():
            return row.get("value")
    return None


@router.get("/{patient_id}")
def patient_detail(
    patient_id: str,
    at: Optional[str] = Query(None, description="ISO timestamp or clock time, e.g. 10:45"),
    index: Optional[int] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Header data + a compact clinical snapshot at one instant."""
    record = get_record(patient_id)
    position = resolve_index(record, index=index, at=at, session_id=session_id)
    profile = record.profile
    ehr = profile.get("ehr", {})
    frame = record.frame
    latest = frame.iloc[position]
    probability = float(record.risk_primary[position])
    twin = twin_service.build_twin_state(record, position)
    night = most_recent_night(record, position)

    return {
        "header": {
            "patient_id": record.patient_id,
            "mrn": profile.get("mrn"),
            "name": record.name,
            "age": profile.get("age"),
            "sex": profile.get("sex"),
            "primary_condition": profile.get("primary_condition"),
            "conditions": [d["name"] for d in ehr.get("diagnoses", []) if d.get("status") == "active"],
            "care_setting": profile.get("care_setting"),
            "is_demo_patient": bool(profile.get("is_demo_patient", False)),
            "bmi": ehr.get("demographics", {}).get("bmi"),
            "hba1c_pct": _lab(profile, "HbA1c"),
            "blood_pressure": _latest_bp(ehr),
            "medications": [
                {"name": m["name"], "dose": m["dose"], "frequency": m["frequency"]}
                for m in ehr.get("medications", [])
                if str(m.get("status", "")).lower() == "current"
            ],
            "data_classification": "SYNTHETIC — NOT REAL PATIENT DATA",
        },
        "twin_status": twin["synchronisation"],
        "as_of": {
            "timestamp": record.timestamp_at(position).isoformat(),
            "clock": record.timestamp_at(position).strftime("%H:%M:%S"),
            "date": record.timestamp_at(position).strftime("%a %d %b %Y"),
            "index": position,
            "is_default_now": position == record.now_index,
            "phase": "history" if position < record.now_index else ("live" if position == record.now_index else "future-replay"),
        },
        "snapshot": {
            "glucose_mgdl": int(latest["glucose_mgdl"]),
            "heart_rate_bpm": int(latest["heart_rate_bpm"]),
            "hrv_rmssd_ms": int(latest["hrv_rmssd_ms"]),
            "spo2_pct": int(latest["spo2_pct"]),
            "steps_today": int(record.feature_row(position)["steps_today"]),
            "activity_met": round(float(latest["activity_met"]), 1),
            "sleep_stage": str(latest["sleep_stage"]),
            "last_sleep_night": night,
        },
        "risk": {
            "probability": round(probability, 4),
            **risk_band(probability),
        },
        "twin": {
            "composite_index": twin["composite_index"],
            "composite_status": twin["composite_status"],
            "domains": twin["domains"],
        },
        "scenario": profile.get("scenario", {}),
        "stream": {
            "start": record.stream.start.isoformat(),
            "end": record.stream.end.isoformat(),
            "scenario_day": record.stream.scenario_day.isoformat(),
            "samples": int(len(frame)),
            "interval_minutes": settings.sampling_interval_minutes,
        },
    }


def _latest_bp(ehr: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    history = (ehr.get("vitals_history", {}) or {}).get("blood_pressure") or []
    return history[0] if history else None


@router.get("/{patient_id}/ehr")
def patient_ehr(patient_id: str) -> Dict[str, Any]:
    """Full longitudinal record plus the static features derived from it."""
    record = get_record(patient_id)
    profile = record.profile
    ehr = profile.get("ehr", {})
    return {
        "patient_id": record.patient_id,
        "name": record.name,
        "classification": "SYNTHETIC — NOT REAL PATIENT DATA",
        "record": ehr,
        "derived_static_features": {k: v for k, v in record.ehr_static.items()},
        "feature_mapping": [
            {"field": s.name, "unit": s.unit, "description": s.description}
            for s in fe.FEATURE_SPECS
            if s.group == "historical"
        ],
        "note": (
            "The derived block is what the model actually consumes: each record field is mapped to a numeric "
            "feature with a documented unit. Derived fields (for example eGFR from CKD stage) are labelled as such."
        ),
    }


@router.get("/{patient_id}/sensors")
def patient_sensors(
    patient_id: str,
    hours: float = Query(6.0, ge=0.5, le=72.0),
    at: Optional[str] = None,
    index: Optional[int] = None,
    session_id: Optional[str] = None,
    stride_minutes: int = Query(5, ge=1, le=60),
) -> Dict[str, Any]:
    """Raw and derived sensor window for the Live Monitoring charts."""
    record = get_record(patient_id)
    position = resolve_index(record, index=index, at=at, session_id=session_id)
    window = record.history(position, hours)
    stride = max(1, int(stride_minutes / settings.sampling_interval_minutes))
    sampled = window.iloc[::stride]
    start_index = max(0, position - len(window) + 1)
    risk = record.risk_primary[start_index : position + 1][::stride]
    risk_secondary = record.risk_secondary[start_index : position + 1][::stride]

    return {
        "patient_id": record.patient_id,
        "window": {
            "start": window["timestamp"].iloc[0].isoformat() if len(window) else None,
            "end": record.timestamp_at(position).isoformat(),
            "hours": hours,
            "samples": int(len(sampled)),
            "stride_minutes": stride_minutes,
        },
        "series": {
            "timestamps": [t.isoformat() for t in sampled["timestamp"]],
            "clocks": [t.strftime("%H:%M") for t in sampled["timestamp"]],
            "glucose_mgdl": [int(v) for v in sampled["glucose_mgdl"]],
            "heart_rate_bpm": [int(v) for v in sampled["heart_rate_bpm"]],
            "hrv_rmssd_ms": [int(v) for v in sampled["hrv_rmssd_ms"]],
            "steps_5min": [int(v) for v in sampled["steps_5min"]],
            "activity_met": [round(float(v), 1) for v in sampled["activity_met"]],
            "spo2_pct": [int(v) for v in sampled["spo2_pct"]],
            "sleep_stage": [str(v) for v in sampled["sleep_stage"]],
        },
        "risk_series": {
            "primary": [round(float(v), 4) for v in risk],
            "secondary": [round(float(v), 4) for v in risk_secondary],
        },
        "thresholds": {
            "glucose_high_mgdl": settings.glucose_high_threshold_mgdl,
            "glucose_low_mgdl": settings.glucose_low_threshold_mgdl,
            "risk_moderate": settings.risk_moderate_min,
            "risk_high": settings.risk_high_min,
        },
        "personal_baseline": {
            "glucose_median": record.baseline.get("glucose_median"),
            "glucose_p10": record.baseline.get("glucose_p10"),
            "glucose_p90": record.baseline.get("glucose_p90"),
            "glucose_median_daytime": record.baseline.get("glucose_median_daytime"),
            "hr_resting_median": record.baseline.get("hr_resting_median"),
            "hrv_daytime_median": record.baseline.get("hrv_daytime_median"),
            "daily_steps_median": record.baseline.get("daily_steps_median"),
            "sleep_duration_median_h": record.baseline.get("sleep_duration_median_h"),
        },
        "context": {
            "meals": [
                {
                    "timestamp": m["datetime"],
                    "clock": m["clock"],
                    "label": m["label"],
                    "carbs_g": m["carbs_g"],
                    "logged_in_app": m.get("logged_in_app", True),
                }
                for m in record.stream.meals
                if pd.Timestamp(m["datetime"]) >= window["timestamp"].iloc[0]
                and pd.Timestamp(m["datetime"]) <= record.timestamp_at(position)
            ],
            "sleep_nights": [
                {
                    "night_of": n["night_of"],
                    "bed_time": n["bed_time"],
                    "wake_time": n["wake_time"],
                    "sleep_duration_h": n["sleep_duration_h"],
                    "efficiency": n["efficiency"],
                    "awakenings": n["awakenings"],
                    "note": n.get("note", ""),
                }
                for n in record.stream.sleep_nights
            ],
        },
        "cards": twin_service.build_sensor_cards(record, position),
        "data_quality": prediction_service.assess_data_quality(record, position),
    }


@router.get("/{patient_id}/charts")
def patient_charts(
    patient_id: str,
    at: Optional[str] = None,
    index: Optional[int] = None,
    session_id: Optional[str] = None,
    past_hours: float = Query(4.0, ge=0.5, le=24.0),
    forecast_minutes: int = Query(120, ge=15, le=240),
) -> Dict[str, Any]:
    """
    Everything the main glucose chart needs in one call: the past trace, the
    current instant, the forecast fan, the personal baseline band and the
    clinical threshold.
    """
    record = get_record(patient_id)
    position = resolve_index(record, index=index, at=at, session_id=session_id)
    predictor = safe_predictor()
    past = record.history(position, past_hours)
    stride = max(1, int(5 / settings.sampling_interval_minutes))
    sampled = past.iloc[::stride]
    probability = float(record.risk_primary[position])
    prediction = prediction_service.build_prediction(record, position, predictor, explain=False)
    trajectory = prediction["trajectory"]

    baseline = record.baseline
    return {
        "patient_id": record.patient_id,
        "as_of": record.timestamp_at(position).isoformat(),
        "regions": {
            "past": {"label": "PAST", "start": sampled["timestamp"].iloc[0].isoformat() if len(sampled) else None,
                     "end": record.timestamp_at(position).isoformat()},
            "current": {"label": "CURRENT", "timestamp": record.timestamp_at(position).isoformat(),
                        "glucose_mgdl": int(record.frame["glucose_mgdl"].iloc[position])},
            "forecast": {
                "label": "FORECAST",
                "start": record.timestamp_at(position).isoformat(),
                "end": (record.timestamp_at(position) + pd.Timedelta(minutes=forecast_minutes)).isoformat(),
                "horizon_minutes": forecast_minutes,
            },
        },
        "past": {
            "timestamps": [t.isoformat() for t in sampled["timestamp"]],
            "clocks": [t.strftime("%H:%M") for t in sampled["timestamp"]],
            "glucose_mgdl": [int(v) for v in sampled["glucose_mgdl"]],
            "risk": [round(float(record.risk_primary[record.index_at(t)]), 4) for t in sampled["timestamp"]],
        },
        "forecast": {
            "basis": trajectory["primary_basis"],
            "points": [p for p in trajectory["points"] if p["minutes_ahead"] <= forecast_minutes],
            "trend_continuation": [
                {"minutes_ahead": p["minutes_ahead"], "timestamp": p["timestamp"], "p50": p["p50"]}
                for p in trajectory["trend_continuation"]["points"]
                if p["minutes_ahead"] <= forecast_minutes
            ],
            "conditional": trajectory.get("conditional"),
            "reading_note": trajectory["reading_note"],
        },
        "bands": {
            "threshold_mgdl": settings.glucose_high_threshold_mgdl,
            "low_threshold_mgdl": settings.glucose_low_threshold_mgdl,
            "personal_baseline_band": [baseline.get("glucose_p10"), baseline.get("glucose_p90")],
            "personal_median": baseline.get("glucose_median"),
            "baseline_window": {"start": baseline.get("window_start"), "end": baseline.get("window_end")},
        },
        "probability": round(probability, 4),
        "risk": risk_band(probability),
        "annotations": _chart_annotations(record, position, past_hours),
    }


def _chart_annotations(record, position: int, past_hours: float) -> List[Dict[str, Any]]:
    """Meals and sleep boundaries inside the visible window, for chart markers."""
    start = record.timestamp_at(position) - pd.Timedelta(hours=past_hours)
    end = record.timestamp_at(position) + pd.Timedelta(minutes=settings.forecast_horizon_minutes)
    annotations: List[Dict[str, Any]] = []
    for meal in record.stream.meals:
        ts = pd.Timestamp(meal["datetime"])
        if start <= ts <= end:
            annotations.append(
                {
                    "type": "meal",
                    "timestamp": ts.isoformat(),
                    "clock": ts.strftime("%H:%M"),
                    "label": meal["label"],
                    "carbs_g": meal["carbs_g"],
                    "logged_in_app": meal.get("logged_in_app", True),
                    "in_future": ts > record.timestamp_at(position),
                }
            )
    for night in record.stream.sleep_nights:
        for key, kind in (("bed_time", "bedtime"), ("wake_time", "wake")):
            try:
                base = pd.Timestamp(night["night_of"]) + (pd.Timedelta(days=1) if kind == "wake" else pd.Timedelta(0))
                hh, mm = str(night[key]).split(":")[:2]
                ts = base + pd.Timedelta(hours=int(hh), minutes=int(mm))
            except Exception:
                continue
            if start <= ts <= end:
                annotations.append(
                    {
                        "type": kind,
                        "timestamp": ts.isoformat(),
                        "clock": ts.strftime("%H:%M"),
                        "label": f"{kind.title()} {night[key]}",
                        "in_future": ts > record.timestamp_at(position),
                    }
                )
    annotations.sort(key=lambda a: a["timestamp"])
    return annotations
