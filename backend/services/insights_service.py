"""
VitalSync — clinical context assembly for the interpretation layer.

Builds the *structured* payload handed to Groq.  Everything in it is measured or
computed by the pipeline: the language model receives facts and returns prose, so
it has nothing to invent.  The same payload is exposed to the UI (via
``POST /api/insights``) so a reviewer can inspect exactly what was sent.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.settings import settings
from backend.services import feature_engineering as fe
from backend.services import timeline_service
from backend.services.patient_service import PatientRecord
from backend.services.prediction_service import assess_data_quality, risk_band


def build_context(
    record: PatientRecord,
    index: int,
    prediction: Dict[str, Any],
    baseline_comparison: Dict[str, Any],
    twin_state: Dict[str, Any],
    trend_hours: float = 3.0,
) -> Dict[str, Any]:
    index = record.clip_index(index)
    profile = record.profile
    ehr = profile.get("ehr", {})
    frame = record.frame
    features = record.features
    anchor = record.timestamp_at(index)

    history = record.history(index, trend_hours)
    start_index = max(0, record.index_at(anchor - pd.Timedelta(hours=trend_hours)))
    risk_series = record.risk_primary[start_index : index + 1]
    glucose_series = frame["glucose_mgdl"].iloc[start_index : index + 1].astype(float)

    def lab(name: str) -> Optional[float]:
        for row in ehr.get("lab_results", []) or []:
            if str(row.get("test", "")).lower() == name.lower():
                return row.get("value")
        return None

    latest_bp = (ehr.get("vitals_history", {}) or {}).get("blood_pressure") or [{}]
    instability = ehr.get("previous_glucose_instability", {}) or {}
    f = record.feature_row(index)

    patient = {
        "patient_id": record.patient_id,
        "mrn": profile.get("mrn"),
        "name": record.name,
        "age": profile.get("age"),
        "sex": profile.get("sex"),
        "primary_condition": profile.get("primary_condition"),
        "care_setting": profile.get("care_setting"),
        "data_classification": "SYNTHETIC DEMO RECORD — no real patient data",
    }

    historical = {
        "bmi": ehr.get("demographics", {}).get("bmi"),
        "waist_cm": ehr.get("demographics", {}).get("waist_cm"),
        "diabetes_duration_years": ehr.get("diabetes", {}).get("duration_years"),
        "diabetes_status": ehr.get("diabetes", {}).get("status"),
        "hba1c_pct": lab("HbA1c"),
        "hba1c_trend": [t.get("value") for t in _lab_trend(ehr, "HbA1c")],
        "fasting_glucose_mgdl": lab("Fasting plasma glucose"),
        "blood_pressure": f"{latest_bp[0].get('systolic')}/{latest_bp[0].get('diastolic')} mmHg",
        "egfr": lab("eGFR"),
        "urine_acr": lab("Urine albumin/creatinine ratio"),
        "triglycerides": lab("Triglycerides"),
        "hdl": lab("HDL cholesterol"),
        "active_medications": [
            {"name": m["name"], "dose": m["dose"], "frequency": m["frequency"], "class": m.get("class"), "since": m.get("started")}
            for m in ehr.get("medications", [])
            if str(m.get("status", "")).lower() in ("current", "active")
        ],
        "discontinued_medications": [
            {"name": m["name"], "stopped": m.get("stopped"), "reason": m.get("reason")}
            for m in ehr.get("medications", [])
            if str(m.get("status", "")).lower() == "discontinued"
        ],
        "diagnoses": [{"code": d["code"], "name": d["name"], "since": d["diagnosed"], "status": d["status"]} for d in ehr.get("diagnoses", [])],
        "family_history": [f"{fh['relation']}: {fh['condition']} (onset {fh.get('age_at_onset')})" for fh in ehr.get("family_history", [])],
        "documented_risk_factors": ehr.get("risk_factors", []),
        "previous_cgm_report": {
            "period": instability.get("source"),
            "mean_glucose_mgdl": instability.get("mean_glucose_mgdl"),
            "cv_pct": instability.get("coefficient_of_variation_pct"),
            "time_in_range_pct": instability.get("time_in_range_70_180_pct"),
            "time_above_180_pct": instability.get("time_above_180_pct"),
            "hyperglycaemic_episodes": instability.get("hyperglycaemic_episodes_14d"),
        },
        "lifestyle": ehr.get("lifestyle", {}),
    }

    latest_sensor = {
        "as_of": anchor.isoformat(),
        "clock": anchor.strftime("%H:%M:%S"),
        "cgm_interval_minutes": settings.sampling_interval_minutes,
        "glucose_mgdl": int(frame["glucose_mgdl"].iloc[index]),
        "glucose_15min_mean_mgdl": round(float(f.get("glucose_current", 0)), 1),
        "glucose_slope_30min_mgdl_per_h": round(float(f.get("glucose_slope_30", 0)) * 60, 1),
        "glucose_slope_60min_mgdl_per_h": round(float(f.get("glucose_slope_60", 0)) * 60, 1),
        "glucose_mean_3h_mgdl": round(float(f.get("glucose_mean_180", 0)), 1),
        "glucose_cv_3h_pct": round(float(f.get("glucose_cv_180", 0)), 1),
        "glucose_time_above_180_24h_pct": round(float(f.get("glucose_above_180_24h_pct", 0)), 1),
        "heart_rate_bpm": int(frame["heart_rate_bpm"].iloc[index]),
        "heart_rate_60min_mean_bpm": round(float(f.get("hr_mean_60", 0)), 1),
        "hrv_rmssd_30min_mean_ms": round(float(f.get("hrv_current", 0)), 1),
        "hrv_overnight_mean_ms": round(float(f.get("hrv_overnight_mean", 0)), 1),
        "spo2_pct": int(frame["spo2_pct"].iloc[index]),
        "steps_last_3h": int(f.get("steps_180", 0)),
        "steps_today": int(f.get("steps_today", 0)),
        "activity_met_60min": round(float(f.get("activity_met_60", 0)), 2),
        "sedentary_minutes_continuous": int(f.get("sedentary_minutes_continuous", 0)),
        "sleep_last_night": {
            "duration_h": round(float(f.get("sleep_duration_h", 0)), 1),
            "efficiency": round(float(f.get("sleep_efficiency", 0)), 2),
            "deep_fraction": round(float(f.get("sleep_deep_fraction", 0)), 2),
            "awakenings": int(f.get("sleep_awakenings", 0)),
            "hours_since_wake": round(float(f.get("hours_since_wake", 0)), 1),
            "note": _last_night_note(record, index),
        },
        "meal_context": {
            "minutes_since_last_logged_meal": int(f.get("minutes_since_last_meal", 0)),
            "last_meal_carbs_g": int(f.get("last_meal_carbs_g", 0)),
            "carbs_last_24h_g": int(f.get("carbs_last_24h", 0)),
            "minutes_to_next_habitual_meal": int(f.get("minutes_to_next_habitual_meal", 0)),
            "habitual_meal_times": [
                {"meal": m["name"], "clock": _clock_from_hours(m["hour"]), "usual_carbs_g": m["carbs_g"]}
                for m in (record.baseline.get("habitual_meals") or [])
            ],
            "meals_logged_today": [
                {"clock": m["clock"], "label": m["label"], "carbs_g": m["carbs_g"], "logged_in_app": m.get("logged_in_app", True)}
                for m in record.stream.meals
                if pd.Timestamp(m["datetime"]).date() == anchor.date() and pd.Timestamp(m["datetime"]) <= anchor
            ],
            "meal_log_present": bool(f.get("minutes_since_last_meal", 600) < 600),
        },
    }

    recent_trends = {
        "window_hours": trend_hours,
        "window_start": record.timestamp_at(start_index).isoformat(),
        "window_end": anchor.isoformat(),
        "glucose_start_mgdl": int(glucose_series.iloc[0]) if len(glucose_series) else None,
        "glucose_end_mgdl": int(glucose_series.iloc[-1]) if len(glucose_series) else None,
        "glucose_min_mgdl": int(glucose_series.min()) if len(glucose_series) else None,
        "glucose_max_mgdl": int(glucose_series.max()) if len(glucose_series) else None,
        "glucose_direction": _direction(float(glucose_series.iloc[-1] - glucose_series.iloc[0]), 8.0) if len(glucose_series) else "stable",
        "glucose_net_change_mgdl": int(glucose_series.iloc[-1] - glucose_series.iloc[0]) if len(glucose_series) else None,
        "risk_start": round(float(risk_series[0]), 4) if len(risk_series) else None,
        "risk_end": round(float(risk_series[-1]), 4) if len(risk_series) else None,
        "risk_max": round(float(np.max(risk_series)), 4) if len(risk_series) else None,
        "risk_direction": _direction(float(risk_series[-1] - risk_series[0]), 0.03) if len(risk_series) else "stable",
        "risk_change_60min": prediction.get("risk_trend", {}).get("change_60min"),
        "heart_rate_change_bpm": round(float(f.get("hr_delta_60", 0)), 1),
        "hrv_change_vs_baseline_pct": round(float(f.get("hrv_dev_personal_pct", 0)), 1),
        "activity_change_vs_usual_pct": round(float(f.get("activity_dev_personal_pct", 0)), 1),
        "steps_last_3h": int(f.get("steps_180", 0)),
        "notable_timeline_events": [
            {"clock": e["clock"], "category": e["category_label"], "title": e["title"], "severity": e["severity"]}
            for e in timeline_service.build_timeline(record, index, hours=trend_hours)["events"][-12:]
        ],
    }

    attribution = prediction.get("attribution") or {}
    prediction_block = {
        "target": prediction.get("event_definition", {}).get("description"),
        "probability": prediction.get("probability"),
        "band": (prediction.get("risk") or {}).get("label"),
        "horizon_minutes": prediction.get("forecast_horizon_minutes"),
        "generated_at": prediction.get("generated_at"),
        "model_id": (prediction.get("model") or {}).get("model_id"),
        "estimator": (prediction.get("model") or {}).get("estimator"),
        "decision_threshold": prediction.get("decision_threshold"),
        "risk_trend_direction": prediction.get("risk_trend", {}).get("direction"),
        "risk_change_60min": prediction.get("risk_trend", {}).get("change_60min"),
        "risk_change_180min": prediction.get("risk_trend", {}).get("change_180min"),
        "confidence": (prediction.get("confidence") or {}).get("score"),
        "confidence_components": (prediction.get("confidence") or {}).get("components"),
        "secondary_model_probability": prediction.get("secondary_probability"),
        "top_contributors": [
            {
                "feature": c["feature"],
                "label": c["label"],
                "group_label": c["group_label"],
                "value": c["value"],
                "reference_value": c["reference_value"],
                "contribution": c["contribution"],
                "direction": c["direction"],
            }
            for c in (attribution.get("top_contributors") or [])[:8]
        ],
        "contributions_by_group": attribution.get("by_group"),
        "attribution_method": attribution.get("method"),
        "projected_trajectory": {
            "method": (prediction.get("trajectory") or {}).get("method"),
            "projected_peak_p50_mgdl": (prediction.get("trajectory") or {}).get("projected_peak_p50"),
            "projected_peak_p90_mgdl": (prediction.get("trajectory") or {}).get("projected_peak_p90"),
            "personal_postprandial_response": (prediction.get("trajectory") or {}).get("personal_postprandial_response"),
        },
        "note": "The probability was produced by the trained statistical model, not by a language model.",
    }

    risk_factors: List[Dict[str, Any]] = []
    for contributor in (attribution.get("top_contributors") or [])[:6]:
        risk_factors.append(
            {
                "label": _contributor_sentence(contributor),
                "feature": contributor["feature"],
                "stream": "historical" if contributor["group"] == "historical" else "live",
                "contribution": contributor["contribution"],
                "value": contributor["value"],
                "reference_value": contributor["reference_value"],
            }
        )

    return {
        "patient": patient,
        "historical_data": historical,
        "latest_sensor_data": latest_sensor,
        "recent_trends": recent_trends,
        "prediction": prediction_block,
        "risk_factors": risk_factors,
        "baseline_comparison": {
            "statement": baseline_comparison.get("statement"),
            "baseline_window": baseline_comparison.get("baseline_window"),
            "metrics": [
                {k: v for k, v in row.items() if k != "personal_baseline_source"}
                for row in baseline_comparison.get("metrics", [])
            ],
        },
        "digital_twin_state": {
            "composite_index": twin_state.get("composite_index"),
            "domains": [
                {"label": d["label"], "score": d["score"], "status": d["word"], "change_vs_24h": twin_state.get("change_vs_24h", {}).get(d["key"])}
                for d in twin_state.get("domains", {}).values()
            ],
            "drivers": twin_state.get("drivers"),
        },
        "data_quality": assess_data_quality(record, index),
        "prototype_constraints": [
            "Synthetic demonstration data only; no real patient data is involved.",
            "Prototype for research and demonstration. Not intended for diagnosis or medical decision-making.",
            "Not clinically validated and not a medical device.",
        ],
    }


def _lab_trend(ehr: Dict[str, Any], test_name: str) -> List[Dict[str, Any]]:
    for row in ehr.get("lab_results", []) or []:
        if str(row.get("test", "")).lower() == test_name.lower():
            return row.get("trend", []) or []
    return []


def _last_night_note(record: PatientRecord, index: int) -> Optional[str]:
    from backend.services.prediction_service import most_recent_night

    night = most_recent_night(record, index)
    return (night or {}).get("note") or None


def _clock_from_hours(hours: float) -> str:
    hh = int(hours)
    mm = int(round((hours - hh) * 60))
    if mm == 60:
        hh, mm = hh + 1, 0
    return f"{hh % 24:02d}:{mm:02d}"


def _direction(delta: float, tolerance: float) -> str:
    if delta > tolerance:
        return "rising"
    if delta < -tolerance:
        return "falling"
    return "stable"


def _contributor_sentence(contributor: Dict[str, Any]) -> str:
    label = contributor.get("label", contributor.get("feature"))
    value = contributor.get("value")
    reference = contributor.get("reference_value")
    unit = contributor.get("unit", "")
    direction = "increases" if contributor.get("direction") == "increases_risk" else "reduces"
    magnitude = abs(float(contributor.get("contribution", 0.0))) * 100.0
    return (
        f"{label} {direction} predicted risk by {magnitude:.1f} percentage points "
        f"(patient value {value}{(' ' + unit) if unit else ''}, reference {reference}{(' ' + unit) if unit else ''})."
    )
