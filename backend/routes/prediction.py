"""
VitalSync — prediction routes.

``GET /api/patients/{id}/prediction`` is the primary 2-hour risk call.  The
probability always comes from the trained statistical model; the language model
is never involved here (see ``insights.py``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

from backend.routes.deps import get_record, resolve_index, safe_predictor
from backend.services import feature_engineering as fe
from backend.services import prediction_service
from backend.services.prediction_service import observed_outcome, risk_band
from backend.settings import settings

router = APIRouter(prefix="/patients/{patient_id}", tags=["prediction"])


@router.get("/prediction")
def prediction(
    patient_id: str,
    at: Optional[str] = None,
    index: Optional[int] = None,
    session_id: Optional[str] = None,
    explain: bool = Query(True, description="include per-feature attribution"),
    reveal_outcome: bool = Query(False, description="include what actually happened next (replay only)"),
) -> Dict[str, Any]:
    record = get_record(patient_id)
    position = resolve_index(record, index=index, at=at, session_id=session_id)
    predictor = safe_predictor()
    payload = prediction_service.build_prediction(
        record, position, predictor, explain=explain, reveal_outcome=reveal_outcome
    )
    payload["already_above_threshold"] = _above_threshold_context(record, position)
    return payload


def _above_threshold_context(record, position: int) -> Dict[str, Any]:
    """
    The forecast target is a *significant rise*, not merely being high.  When the
    patient is already above range the card says so explicitly, so a low forecast
    probability is never mistaken for "glucose is normal".
    """
    current = float(record.frame["glucose_mgdl"].iloc[position])
    threshold = float(settings.glucose_high_threshold_mgdl)
    if current < threshold:
        return {"above_threshold": False, "current_mgdl": round(current, 1), "threshold_mgdl": threshold}
    step = settings.sampling_interval_minutes
    start = max(0, position - int(180 / step))
    window = record.frame["glucose_mgdl"].iloc[start : position + 1].astype(float)
    above = (window >= threshold).to_numpy()
    run = 0
    for value in above[::-1]:
        if not value:
            break
        run += 1
    return {
        "above_threshold": True,
        "current_mgdl": round(current, 1),
        "threshold_mgdl": threshold,
        "minutes_above_threshold": int(run * step),
        "note": (
            f"Glucose is already {current - threshold:.0f} mg/dL above the {int(threshold)} mg/dL threshold and has been "
            f"for about {run * step} minutes. This model forecasts a further significant rise of at least "
            f"{int(settings.glucose_rise_delta_mgdl)} mg/dL; sustained hyperglycaemia is reported separately in the "
            "glucose panel and the clinical timeline."
        ),
    }


@router.get("/prediction/explain")
def explain_prediction(
    patient_id: str,
    at: Optional[str] = None,
    index: Optional[int] = None,
    session_id: Optional[str] = None,
    top_k: int = Query(10, ge=1, le=40),
    reference: str = Query("training_median", description="training_median | personal_baseline | zeros"),
) -> Dict[str, Any]:
    """Full local attribution for one prediction."""
    record = get_record(patient_id)
    position = resolve_index(record, index=index, at=at, session_id=session_id)
    predictor = safe_predictor()
    features = record.feature_row(position)
    if predictor is None:
        return {
            "error": "model_not_trained",
            "hint": "Run: python model/train.py",
            "features": features,
        }

    overrides: Optional[Dict[str, float]] = None
    if reference == "personal_baseline":
        overrides = _personal_reference(record, features)
    elif reference == "zeros":
        overrides = {name: 0.0 for name in fe.FEATURE_NAMES}

    attribution = predictor.explain(features, top_k=top_k, reference=overrides)
    return {
        "patient_id": record.patient_id,
        "as_of": record.timestamp_at(position).isoformat(),
        "index": position,
        "reference_mode": reference,
        "reference_description": {
            "training_median": "each feature is replaced by the median value seen during training",
            "personal_baseline": "each feature is replaced by this patient's own baseline value where one exists",
            "zeros": "each feature is replaced by zero (diagnostic only)",
        }[reference],
        "probability": round(float(record.risk_primary[position]), 4),
        "risk": risk_band(float(record.risk_primary[position])),
        "attribution": attribution,
        "feature_values": features,
        "feature_catalogue": [
            {"name": s.name, "group": s.group, "group_label": fe.FEATURE_GROUPS.get(s.group, s.group), "unit": s.unit, "description": s.description}
            for s in fe.FEATURE_SPECS
        ],
    }


def _personal_reference(record, features: Dict[str, float]) -> Dict[str, float]:
    """
    Reference case for a "compared with this patient's own baseline" attribution:
    personal medians where the baseline knows them, training medians otherwise.
    """
    baseline = record.baseline
    mapping = {
        "glucose_current": baseline.get("glucose_median_daytime"),
        "glucose_mean_30": baseline.get("glucose_median_daytime"),
        "glucose_mean_60": baseline.get("glucose_median_daytime"),
        "glucose_mean_12h": baseline.get("glucose_mean"),
        "glucose_mean_24h": baseline.get("glucose_mean"),
        "glucose_cv_24h": baseline.get("glucose_cv_pct"),
        "hr_current": baseline.get("hr_resting_median"),
        "hr_mean_60": baseline.get("hr_resting_median"),
        "hr_mean_180": baseline.get("hr_resting_median"),
        "hrv_current": baseline.get("hrv_daytime_median"),
        "hrv_mean_180": baseline.get("hrv_daytime_median"),
        "hrv_overnight_mean": baseline.get("overnight_hrv_median"),
        "sleep_duration_h": baseline.get("sleep_duration_median_h"),
        "sleep_efficiency": baseline.get("sleep_efficiency_median"),
        "steps_today": baseline.get("daily_steps_median"),
    }
    return {key: float(value) for key, value in mapping.items() if value is not None}


@router.get("/prediction/trajectory")
def trajectory(
    patient_id: str,
    at: Optional[str] = None,
    index: Optional[int] = None,
    session_id: Optional[str] = None,
    horizon_minutes: int = Query(120, ge=15, le=240),
) -> Dict[str, Any]:
    """Forecast fan (risk-conditional) and the personal trend-continuation line."""
    record = get_record(patient_id)
    position = resolve_index(record, index=index, at=at, session_id=session_id)
    predictor = safe_predictor()
    probability = float(record.risk_primary[position])
    anchor = record.timestamp_at(position)
    conditional = prediction_service.conditional_trajectory(probability, predictor, horizon_minutes, anchor)
    projection = prediction_service.get_projector(record).project(position, horizon_minutes=horizon_minutes)
    glucose_now = float(record.frame["glucose_mgdl"].iloc[position])

    points: List[Dict[str, Any]] = []
    if conditional:
        for point in conditional["delta_points"]:
            points.append(
                {
                    "minutes_ahead": point["minutes_ahead"],
                    "timestamp": point["timestamp"],
                    "p10": round(glucose_now + point["delta_q10"], 1),
                    "p50": round(glucose_now + point["delta_q50"], 1),
                    "p90": round(glucose_now + point["delta_q90"], 1),
                }
            )
    return {
        "patient_id": record.patient_id,
        "anchor_timestamp": anchor.isoformat(),
        "anchor_glucose_mgdl": round(glucose_now, 1),
        "probability": round(probability, 4),
        "risk": risk_band(probability),
        "horizon_minutes": horizon_minutes,
        "threshold_mgdl": settings.glucose_high_threshold_mgdl,
        "conditional": conditional,
        "points": points,
        "trend_continuation": projection,
        "outcome": observed_outcome(record, position),
        "reading_note": (
            "The band is the observed distribution of glucose change in held-out patients with a similar predicted "
            "risk. The dashed line is this patient's own trend and post-prandial response extrapolated forward."
        ),
    }


@router.get("/prediction/history")
def prediction_history(
    patient_id: str,
    hours: float = Query(12.0, ge=0.5, le=72.0),
    at: Optional[str] = None,
    index: Optional[int] = None,
    session_id: Optional[str] = None,
    stride_minutes: int = Query(15, ge=5, le=60),
) -> Dict[str, Any]:
    """Risk probability alongside glucose over a window — the trend sparkline."""
    record = get_record(patient_id)
    position = resolve_index(record, index=index, at=at, session_id=session_id)
    stride = max(1, int(stride_minutes / settings.sampling_interval_minutes))
    start = max(0, position - int(hours * 60 / settings.sampling_interval_minutes))
    timestamps: List[str] = []
    risks: List[float] = []
    secondary: List[float] = []
    glucose: List[int] = []
    bands: List[str] = []
    for i in range(start, position + 1, stride):
        timestamps.append(record.timestamp_at(i).isoformat())
        value = float(record.risk_primary[i])
        risks.append(round(value, 4))
        secondary.append(round(float(record.risk_secondary[i]), 4))
        glucose.append(int(record.frame["glucose_mgdl"].iloc[i]))
        bands.append(risk_band(value)["band"])
    return {
        "patient_id": record.patient_id,
        "window_hours": hours,
        "stride_minutes": stride_minutes,
        "timestamps": timestamps,
        "risk_primary": risks,
        "risk_secondary": secondary,
        "glucose_mgdl": glucose,
        "bands": bands,
        "thresholds": {
            "moderate": settings.risk_moderate_min,
            "high": settings.risk_high_min,
            "decision": float(safe_predictor().threshold) if safe_predictor() else settings.risk_moderate_min,
        },
        "transitions": _band_transitions(timestamps, risks, bands),
    }


def _band_transitions(timestamps: List[str], risks: List[float], bands: List[str]) -> List[Dict[str, Any]]:
    transitions: List[Dict[str, Any]] = []
    for i in range(1, len(bands)):
        if bands[i] != bands[i - 1]:
            transitions.append(
                {
                    "timestamp": timestamps[i],
                    "from": bands[i - 1],
                    "to": bands[i],
                    "risk": risks[i],
                    "previous_risk": risks[i - 1],
                    "escalation": _rank(bands[i]) > _rank(bands[i - 1]),
                }
            )
    return transitions


def _rank(band: str) -> int:
    return {"low": 0, "moderate": 1, "high": 2}.get(band, 0)
