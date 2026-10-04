"""
VitalSync — Digital Twin state engine.

Fuses the two required data streams into an interpretable patient state:

    historical EHR  ─┐
                     ├─► feature vector ─► domain scores ─► twin state
    live physiology ─┘

Five domains are reported, each on a 0-100 scale where **higher is better**:

    Metabolic Stability   long-horizon glycaemic control (HbA1c, prior CGM report,
                          12-hour mean, 24-hour variability)
    Glucose Stability     short-horizon glucose behaviour (level vs personal norm,
                          3-hour variability, slope, time above range)
    Cardiovascular State  heart rate and HRV relative to personal baseline, SpO₂
    Recovery              last night's sleep against the patient's own norm
    Activity              movement today against the patient's own time-of-day profile

Every score is a deterministic, documented function of the underlying synthetic
signals — there are no random or hand-tuned "dashboard numbers".  Each domain
also returns the sub-terms it was built from, so a reviewer can recompute any
value by hand.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.settings import settings
from backend.services import feature_engineering as fe
from backend.services.patient_service import PatientRecord

DOMAIN_WEIGHTS = {
    "metabolic_stability": 0.25,
    "glucose_stability": 0.25,
    "cardiovascular_state": 0.20,
    "recovery": 0.15,
    "activity": 0.15,
}

DOMAIN_LABELS = {
    "metabolic_stability": "Metabolic Stability",
    "glucose_stability": "Glucose Stability",
    "cardiovascular_state": "Cardiovascular State",
    "recovery": "Recovery",
    "activity": "Activity",
}


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return lo
    return float(np.clip(value, lo, hi))


def _status(score: float) -> Dict[str, str]:
    if score >= 75:
        return {"status": "stable", "word": "Stable", "tone": "good"}
    if score >= 55:
        return {"status": "fair", "word": "Fair", "tone": "neutral"}
    if score >= 35:
        return {"status": "watch", "word": "Guarded", "tone": "warning"}
    return {"status": "unstable", "word": "Unstable", "tone": "critical"}


# ---------------------------------------------------------------------------
# Domain scorers — each returns (score, contributors)
# ---------------------------------------------------------------------------
def _score_glucose_stability(f: Dict[str, float], baseline: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    cv_ratio = f.get("glucose_cv_dev_personal", 1.0)
    deviation_pct = f.get("glucose_dev_personal_pct", 0.0)
    above_24h = f.get("glucose_above_180_24h_pct", 0.0)
    slope = abs(f.get("glucose_slope_30", 0.0))

    score_cv = _clip(100.0 - 55.0 * max(cv_ratio - 1.0, 0.0))
    score_level = _clip(100.0 - 1.8 * max(deviation_pct, 0.0) - 0.6 * max(-deviation_pct - 15.0, 0.0))
    score_above = _clip(100.0 - 1.7 * above_24h)
    score_slope = _clip(100.0 - 240.0 * slope)

    score = 0.30 * score_cv + 0.30 * score_level + 0.22 * score_above + 0.18 * score_slope
    contributors = [
        {"term": "Glucose vs personal median", "input": round(deviation_pct, 1), "unit": "%", "score": round(score_level, 1), "weight": 0.30},
        {"term": "3-hour variability vs personal norm", "input": round(cv_ratio, 2), "unit": "ratio", "score": round(score_cv, 1), "weight": 0.30},
        {"term": "Time above 180 mg/dL (24 h)", "input": round(above_24h, 1), "unit": "%", "score": round(score_above, 1), "weight": 0.22},
        {"term": "30-minute glucose slope", "input": round(f.get("glucose_slope_30", 0.0), 3), "unit": "mg/dL/min", "score": round(score_slope, 1), "weight": 0.18},
    ]
    return _clip(score), contributors


def _score_metabolic_stability(f: Dict[str, float], baseline: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    hba1c = f.get("hba1c_pct", 6.0)
    prior_tir = f.get("prior_time_in_range_pct", 70.0)
    mean_12h = f.get("glucose_mean_12h", 140.0)
    personal_median = float(baseline.get("glucose_median", mean_12h))
    mean_dev = 100.0 * (mean_12h - personal_median) / max(personal_median, 1e-6)
    cv24 = f.get("glucose_cv_24h", 18.0)
    prior_cv = f.get("prior_cv_pct", cv24) or cv24
    cv_ratio = cv24 / max(prior_cv, 1e-6)

    score_hba1c = _clip(100.0 - 22.0 * max(hba1c - 5.7, 0.0))
    score_tir = _clip(prior_tir)
    score_mean = _clip(100.0 - 1.5 * max(mean_dev, 0.0))
    score_cv = _clip(100.0 - 60.0 * max(cv_ratio - 1.0, 0.0))

    score = 0.34 * score_hba1c + 0.26 * score_tir + 0.20 * score_mean + 0.20 * score_cv
    contributors = [
        {"term": "Most recent HbA1c", "input": round(hba1c, 1), "unit": "%", "score": round(score_hba1c, 1), "weight": 0.34},
        {"term": "Prior CGM time in range", "input": round(prior_tir, 1), "unit": "%", "score": round(score_tir, 1), "weight": 0.26},
        {"term": "12-hour mean vs personal median", "input": round(mean_dev, 1), "unit": "%", "score": round(score_mean, 1), "weight": 0.20},
        {"term": "24-hour variability vs prior report", "input": round(cv_ratio, 2), "unit": "ratio", "score": round(score_cv, 1), "weight": 0.20},
    ]
    return _clip(score), contributors


def _score_cardiovascular(f: Dict[str, float], baseline: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    hr_dev = f.get("hr_dev_personal", 0.0)
    hrv_dev = f.get("hrv_dev_personal_pct", 0.0)
    hrv_night_dev = f.get("hrv_overnight_dev_personal_pct", 0.0)
    spo2 = f.get("spo2_current", 97.0)

    score_hr = _clip(100.0 - 5.0 * max(hr_dev, 0.0) - 2.5 * max(-hr_dev - 8.0, 0.0))
    score_hrv = _clip(100.0 + 1.7 * hrv_dev)
    score_hrv_night = _clip(100.0 + 1.2 * hrv_night_dev)
    score_spo2 = _clip(100.0 - 9.0 * max(95.0 - spo2, 0.0))

    score = 0.30 * score_hr + 0.32 * score_hrv + 0.23 * score_hrv_night + 0.15 * score_spo2
    contributors = [
        {"term": "Heart rate vs personal resting median", "input": round(hr_dev, 1), "unit": "bpm", "score": round(score_hr, 1), "weight": 0.30},
        {"term": "HRV vs personal daytime baseline", "input": round(hrv_dev, 1), "unit": "%", "score": round(score_hrv, 1), "weight": 0.32},
        {"term": "Overnight HRV vs personal baseline", "input": round(hrv_night_dev, 1), "unit": "%", "score": round(score_hrv_night, 1), "weight": 0.23},
        {"term": "SpO₂", "input": round(spo2, 0), "unit": "%", "score": round(score_spo2, 1), "weight": 0.15},
    ]
    return _clip(score), contributors


def _score_recovery(f: Dict[str, float], baseline: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    duration = f.get("sleep_duration_h", 6.5)
    baseline_duration = float(baseline.get("sleep_duration_median_h", duration))
    efficiency = f.get("sleep_efficiency", 0.85)
    quality = f.get("sleep_quality_index", 0.7)
    awakenings = f.get("sleep_awakenings", 1.0)
    hrv_night_dev = f.get("hrv_overnight_dev_personal_pct", 0.0)

    ratio = duration / max(baseline_duration, 1e-6)
    score_duration = _clip(100.0 - 75.0 * max(1.0 - ratio, 0.0))
    score_efficiency = _clip(100.0 * efficiency / 0.92)
    score_quality = _clip(100.0 * quality)
    score_awakenings = _clip(100.0 - 11.0 * awakenings)
    score_hrv = _clip(100.0 + 1.2 * hrv_night_dev)

    score = 0.30 * score_duration + 0.18 * score_efficiency + 0.20 * score_quality + 0.14 * score_awakenings + 0.18 * score_hrv
    contributors = [
        {"term": "Sleep duration vs personal baseline", "input": round(duration - baseline_duration, 1), "unit": "h", "score": round(score_duration, 1), "weight": 0.30},
        {"term": "Sleep efficiency", "input": round(efficiency, 2), "unit": "fraction", "score": round(score_efficiency, 1), "weight": 0.18},
        {"term": "Composite sleep quality index", "input": round(quality, 2), "unit": "0-1", "score": round(score_quality, 1), "weight": 0.20},
        {"term": "Wake episodes", "input": round(awakenings, 0), "unit": "count", "score": round(score_awakenings, 1), "weight": 0.14},
        {"term": "Overnight HRV vs personal baseline", "input": round(hrv_night_dev, 1), "unit": "%", "score": round(score_hrv, 1), "weight": 0.18},
    ]
    return _clip(score), contributors


def _score_activity(f: Dict[str, float], baseline: Dict[str, Any], record: PatientRecord, index: int) -> Tuple[float, List[Dict[str, Any]]]:
    steps_today = f.get("steps_today", 0.0)
    daily_median = float(baseline.get("daily_steps_median", max(steps_today, 1.0)))
    expected_fraction = _expected_activity_fraction(record, index)
    expected_steps = daily_median * expected_fraction
    ratio = steps_today / max(expected_steps, 40.0)
    deviation = f.get("activity_dev_personal_pct", 0.0)
    sedentary = f.get("sedentary_minutes_continuous", 0.0)
    met = f.get("activity_met_60", 1.2)

    score_volume = _clip(100.0 * ratio)
    score_window = _clip(100.0 + 0.85 * deviation)
    score_sedentary = _clip(100.0 - 0.24 * sedentary)
    score_intensity = _clip(100.0 * (met - 0.9) / 1.1)

    score = 0.34 * score_volume + 0.28 * score_window + 0.26 * score_sedentary + 0.12 * score_intensity
    contributors = [
        {"term": "Steps today vs expected by this time", "input": round(steps_today - expected_steps, 0), "unit": "steps", "score": round(score_volume, 1), "weight": 0.34},
        {"term": "Last 3 h vs usual for this clock window", "input": round(deviation, 1), "unit": "%", "score": round(score_window, 1), "weight": 0.28},
        {"term": "Uninterrupted sedentary time", "input": round(sedentary, 0), "unit": "min", "score": round(score_sedentary, 1), "weight": 0.26},
        {"term": "Mean MET over the last hour", "input": round(met, 1), "unit": "MET", "score": round(score_intensity, 1), "weight": 0.12},
    ]
    return _clip(score), contributors


def _expected_activity_fraction(record: PatientRecord, index: int) -> float:
    """
    Fraction of the patient's usual daily step volume that should already have
    happened by this clock time, taken from their own hourly activity profile.
    """
    profile = record.baseline.get("steps_by_hour") or {}
    if not profile:
        hour = record.timestamp_at(index).hour + record.timestamp_at(index).minute / 60.0
        return float(np.clip((hour - 6.0) / 16.0, 0.02, 1.0))
    hours = sorted(int(h) for h in profile.keys())
    weights = np.array([float(profile[h]) for h in hours]) * 12.0  # 12 five-minute intervals per hour
    total = float(weights.sum())
    if total <= 0:
        return 0.5
    now = record.timestamp_at(index)
    current_hour = now.hour
    elapsed_index = [i for i, h in enumerate(hours) if h < current_hour]
    partial = float(weights[[i for i, h in enumerate(hours) if h == current_hour][0]]) * (now.minute / 60.0) if current_hour in hours else 0.0
    done = float(weights[elapsed_index].sum()) if elapsed_index else 0.0
    return float(np.clip((done + partial) / total, 0.01, 1.0))


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------
def domain_scores(record: PatientRecord, index: int) -> Dict[str, Dict[str, Any]]:
    f = record.feature_row(index)
    baseline = record.baseline
    raw = {
        "metabolic_stability": _score_metabolic_stability(f, baseline),
        "glucose_stability": _score_glucose_stability(f, baseline),
        "cardiovascular_state": _score_cardiovascular(f, baseline),
        "recovery": _score_recovery(f, baseline),
        "activity": _score_activity(f, baseline, record, index),
    }
    out: Dict[str, Dict[str, Any]] = {}
    for key, (score, contributors) in raw.items():
        state = _status(score)
        out[key] = {
            "key": key,
            "label": DOMAIN_LABELS[key],
            "score": round(score, 1),
            "normalised": round(score / 100.0, 3),
            "status": state["status"],
            "word": state["word"],
            "tone": state["tone"],
            "weight": DOMAIN_WEIGHTS[key],
            "contributors": contributors,
        }
    return out


def build_twin_state(record: PatientRecord, index: int) -> Dict[str, Any]:
    index = record.clip_index(index)
    domains = domain_scores(record, index)
    composite = sum(domains[k]["score"] * DOMAIN_WEIGHTS[k] for k in domains)

    day_steps_index = int(24 * 60 / settings.sampling_interval_minutes)
    previous = domain_scores(record, max(0, index - day_steps_index)) if index > 12 else None
    deltas = {}
    if previous:
        for key in domains:
            deltas[key] = round(domains[key]["score"] - previous[key]["score"], 1)

    f = record.feature_row(index)
    anchor = record.timestamp_at(index)
    return {
        "patient_id": record.patient_id,
        "as_of": anchor.isoformat(),
        "as_of_clock": anchor.strftime("%H:%M:%S"),
        "composite_index": round(composite, 1),
        "composite_status": _status(composite),
        "domains": domains,
        "radar": [
            {"label": DOMAIN_LABELS[k], "value": round(domains[k]["score"], 1), "key": k}
            for k in ("metabolic_stability", "glucose_stability", "cardiovascular_state", "recovery", "activity")
        ],
        "change_vs_24h": deltas,
        "synchronisation": {
            "status": "SYNCHRONIZED",
            "last_sample": anchor.isoformat(),
            "stream_start": record.stream.start.isoformat(),
            "stream_end": record.stream.end.isoformat(),
            "samples_available": int(len(record.frame)),
            "sampling_interval_minutes": settings.sampling_interval_minutes,
            "baseline_window": {
                "start": record.baseline.get("window_start"),
                "end": record.baseline.get("window_end"),
                "hours": record.baseline.get("window_hours"),
                "samples": record.baseline.get("samples"),
            },
            "feature_count": len(fe.FEATURE_NAMES),
            "twin_built_at": record.built_at,
        },
        "drivers": _twin_drivers(record, index, domains, f),
    }


def _twin_drivers(record: PatientRecord, index: int, domains: Dict[str, Any], f: Dict[str, float]) -> List[Dict[str, Any]]:
    """Plain-language statements about what is currently pulling the state down."""
    drivers: List[Dict[str, Any]] = []
    if domains["activity"]["score"] < 60:
        drivers.append(
            {
                "domain": "Activity",
                "statement": f"Only {int(f.get('steps_today', 0)):,} steps so far today against a personal median of "
                f"{int(record.baseline.get('daily_steps_median', 0)):,}; {int(f.get('sedentary_minutes_continuous', 0))} continuous sedentary minutes.",
                "severity": "warning",
            }
        )
    if domains["recovery"]["score"] < 60:
        drivers.append(
            {
                "domain": "Recovery",
                "statement": f"Last night {f.get('sleep_duration_h', 0):.1f} h of sleep "
                f"({abs(f.get('sleep_deficit_h', 0)):.1f} h below personal baseline) at "
                f"{f.get('sleep_efficiency', 0) * 100:.0f}% efficiency.",
                "severity": "warning",
            }
        )
    if f.get("hrv_dev_personal_pct", 0) < -8:
        drivers.append(
            {
                "domain": "Cardiovascular State",
                "statement": f"HRV {f.get('hrv_current', 0):.0f} ms is {abs(f.get('hrv_dev_personal_pct', 0)):.0f}% below this patient's daytime baseline.",
                "severity": "warning",
            }
        )
    if f.get("glucose_cv_dev_personal", 1) > 1.15:
        drivers.append(
            {
                "domain": "Glucose Stability",
                "statement": f"3-hour glucose variability is {f.get('glucose_cv_dev_personal', 1):.2f}× the patient's own norm.",
                "severity": "warning",
            }
        )
    if f.get("glucose_slope_30", 0) > 0.15:
        drivers.append(
            {
                "domain": "Glucose Stability",
                "statement": f"Glucose rising at {f.get('glucose_slope_30', 0) * 60:.0f} mg/dL per hour over the last 30 minutes.",
                "severity": "critical" if f.get("glucose_slope_30", 0) > 0.35 else "warning",
            }
        )
    if not drivers:
        drivers.append(
            {
                "domain": "Overall",
                "statement": "All fused domains are within this patient's usual range; no dominant driver at this instant.",
                "severity": "good",
            }
        )
    return drivers


def build_baseline_comparison(record: PatientRecord, index: int) -> Dict[str, Any]:
    """
    "Compared with your baseline" — the personalisation panel.

    Every comparison is against a statistic estimated from *this patient's* own
    onboarding window, never against a population threshold.
    """
    index = record.clip_index(index)
    f = record.feature_row(index)
    baseline = record.baseline
    frame = record.frame
    now = record.timestamp_at(index)

    def window_mean(column: str, minutes: float) -> float:
        start = now - pd.Timedelta(minutes=minutes)
        block = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= now)]
        return float(block[column].astype(float).mean()) if not block.empty else float(frame[column].iloc[index])

    glucose_now = window_mean("glucose_mgdl", 15)
    glucose_baseline = float(baseline.get("glucose_median_daytime", 0))
    hrv_now = window_mean("hrv_rmssd_ms", 30)
    hrv_baseline = float(baseline.get("hrv_daytime_median", 0))
    hr_now = window_mean("heart_rate_bpm", 30)
    hr_baseline = float(baseline.get("hr_resting_median", 0))
    steps_3h = float(f.get("steps_180", 0))
    steps_expected = float(_expected_window_steps(record, index, 180))
    sleep_last = float(f.get("sleep_duration_h", 0))
    sleep_baseline = float(baseline.get("sleep_duration_median_h", 0))
    overnight_hrv = float(f.get("hrv_overnight_mean", 0) or 0)
    overnight_baseline = float(baseline.get("overnight_hrv_median", 0) or 0)

    rows = [
        {
            "key": "glucose",
            "metric": "Glucose (15-min mean)",
            "current": round(glucose_now, 0),
            "baseline": round(glucose_baseline, 0),
            "unit": "mg/dL",
            "delta_absolute": round(glucose_now - glucose_baseline, 1),
            "delta_pct": _pct(glucose_now, glucose_baseline),
            "personal_baseline_source": f"median of daytime CGM values across the {baseline.get('window_hours')} h onboarding window",
            "direction": "adverse",
        },
        {
            "key": "hrv",
            "metric": "HRV (RMSSD, 30-min mean)",
            "current": round(hrv_now, 0),
            "baseline": round(hrv_baseline, 0),
            "unit": "ms",
            "delta_absolute": round(hrv_now - hrv_baseline, 1),
            "delta_pct": _pct(hrv_now, hrv_baseline),
            "personal_baseline_source": "median of daytime RMSSD in the onboarding window",
            "direction": "beneficial",
        },
        {
            "key": "overnight_hrv",
            "metric": "Overnight HRV (last night)",
            "current": round(overnight_hrv, 0),
            "baseline": round(overnight_baseline, 0),
            "unit": "ms",
            "delta_absolute": round(overnight_hrv - overnight_baseline, 1),
            "delta_pct": _pct(overnight_hrv, overnight_baseline),
            "personal_baseline_source": "median overnight RMSSD across recorded nights",
            "direction": "beneficial",
        },
        {
            "key": "heart_rate",
            "metric": "Heart rate (30-min mean)",
            "current": round(hr_now, 0),
            "baseline": round(hr_baseline, 0),
            "unit": "bpm",
            "delta_absolute": round(hr_now - hr_baseline, 1),
            "delta_pct": _pct(hr_now, hr_baseline),
            "personal_baseline_source": "median waking heart rate in the onboarding window",
            "direction": "adverse",
        },
        {
            "key": "activity",
            "metric": "Steps (last 3 h vs usual for this window)",
            "current": round(steps_3h, 0),
            "baseline": round(steps_expected, 0),
            "unit": "steps",
            "delta_absolute": round(steps_3h - steps_expected, 0),
            "delta_pct": _pct(steps_3h, steps_expected),
            "personal_baseline_source": "this patient's own median steps for the same clock hours",
            "direction": "beneficial",
        },
        {
            "key": "sleep",
            "metric": "Sleep duration (last night)",
            "current": round(sleep_last, 1),
            "baseline": round(sleep_baseline, 1),
            "unit": "h",
            "delta_absolute": round(sleep_last - sleep_baseline, 1),
            "delta_pct": _pct(sleep_last, sleep_baseline),
            "personal_baseline_source": "median sleep duration across recorded nights",
            "direction": "beneficial",
        },
        {
            "key": "variability",
            "metric": "Glucose variability (3 h CV vs personal norm)",
            "current": round(float(f.get("glucose_cv_180", 0)), 1),
            "baseline": round(float(baseline.get("glucose_cv_pct", 0)), 1),
            "unit": "%",
            "delta_absolute": round(float(f.get("glucose_cv_180", 0)) - float(baseline.get("glucose_cv_pct", 0)), 1),
            "delta_pct": _pct(float(f.get("glucose_cv_180", 0)), float(baseline.get("glucose_cv_pct", 1))),
            "personal_baseline_source": "coefficient of variation across the whole onboarding window",
            "direction": "adverse",
        },
    ]
    for row in rows:
        row["interpretation"] = _interpret(row)
        row["flag"] = _flag(row)
    return {
        "as_of": now.isoformat(),
        "baseline_window": {
            "start": baseline.get("window_start"),
            "end": baseline.get("window_end"),
            "hours": baseline.get("window_hours"),
            "samples": baseline.get("samples"),
        },
        "statement": (
            "Comparisons are made against this patient's own history, not population reference ranges. "
            "A value can be normal in absolute terms and still abnormal for this individual."
        ),
        "metrics": rows,
    }


def _expected_window_steps(record: PatientRecord, index: int, minutes: float) -> float:
    """
    Steps this patient usually takes in the same clock window, from their own
    hourly profile.  ``steps_by_hour`` holds a median *per sampling interval*,
    so each hour contributes that median times the intervals in an hour.
    """
    profile = record.baseline.get("steps_by_hour") or {}
    now = record.timestamp_at(index)
    daily_median = float(record.baseline.get("daily_steps_median", 4000))
    if not profile:
        return daily_median * minutes / (24 * 60)
    intervals_per_hour = 60.0 / settings.sampling_interval_minutes
    total = 0.0
    cursor = now - pd.Timedelta(minutes=minutes)
    while cursor <= now:
        total += float(profile.get(cursor.hour, 0.0)) * intervals_per_hour
        cursor = cursor + pd.Timedelta(minutes=60)
    return float(total)


def _pct(current: float, baseline: float) -> Optional[float]:
    if baseline is None or abs(baseline) < 1e-6:
        return None
    return round(100.0 * (current - baseline) / abs(baseline), 1)


def _interpret(row: Dict[str, Any]) -> str:
    pct = row.get("delta_pct")
    if pct is None:
        return "No personal baseline available for this metric."
    adverse = row["direction"] == "adverse"
    if abs(pct) < 5:
        return "Within this patient's usual range."
    higher = pct > 0
    worse = higher if adverse else not higher
    magnitude = "markedly" if abs(pct) >= 20 else ("notably" if abs(pct) >= 10 else "slightly")
    direction_word = "above" if higher else "below"
    return f"{magnitude.capitalize()} {direction_word} personal baseline — {'adverse direction' if worse else 'favourable direction'}."


def _flag(row: Dict[str, Any]) -> str:
    pct = row.get("delta_pct")
    if pct is None:
        return "unknown"
    adverse = row["direction"] == "adverse"
    worse = pct > 0 if adverse else pct < 0
    if not worse:
        return "good" if abs(pct) >= 5 else "stable"
    if abs(pct) >= 20:
        return "critical"
    if abs(pct) >= 10:
        return "warning"
    return "watch"


def build_sensor_cards(record: PatientRecord, index: int) -> List[Dict[str, Any]]:
    """Compact live-signal cards with sparkline, baseline and status."""
    index = record.clip_index(index)
    frame = record.frame
    now = record.timestamp_at(index)
    baseline = record.baseline
    f = record.feature_row(index)

    def stats(column: str, minutes: float) -> Tuple[float, List[float], List[str]]:
        start = now - pd.Timedelta(minutes=minutes)
        block = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= now)]
        if block.empty:
            block = frame.iloc[: index + 1].tail(6)
        values = block[column].astype(float).to_numpy()
        times = [t.isoformat() for t in block["timestamp"]]
        return float(np.mean(values[-3:])) if len(values) >= 3 else float(values[-1]), [float(v) for v in values], times

    def sparkline(column: str, hours: float = 3.0, stride_minutes: int = 15) -> Dict[str, Any]:
        start = now - pd.Timedelta(hours=hours)
        block = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= now)]
        stride = max(1, int(stride_minutes / settings.sampling_interval_minutes))
        block = block.iloc[::stride]
        return {
            "labels": [t.strftime("%H:%M") for t in block["timestamp"]],
            "values": [round(float(v), 1) for v in block[column].astype(float)],
            "timestamps": [t.isoformat() for t in block["timestamp"]],
        }

    cards: List[Dict[str, Any]] = []

    glucose_current, _, _ = stats("glucose_mgdl", 15)
    glucose_baseline = float(baseline.get("glucose_median_daytime", 0))
    cards.append(
        {
            "key": "glucose",
            "title": "Glucose (CGM)",
            "current": round(glucose_current, 0),
            "unit": "mg/dL",
            "baseline": round(glucose_baseline, 0),
            "baseline_label": "personal daytime median",
            "change_pct": _pct(glucose_current, glucose_baseline),
            "reference_range": [70, 180],
            "status": "critical" if glucose_current >= 200 else ("warning" if glucose_current >= 180 else ("watch" if glucose_current < 70 else "good")),
            "status_word": "Above range" if glucose_current >= 180 else ("In range" if glucose_current >= 70 else "Below range"),
            "trend": _trend_arrow(f.get("glucose_rate_of_change_15", 0.0)),
            "extra": {"slope_30min": round(f.get("glucose_slope_30", 0.0) * 60, 1), "slope_unit": "mg/dL/h"},
            "sparkline": sparkline("glucose_mgdl"),
            "source": "Continuous glucose monitor, 5-minute sampling",
        }
    )

    hr_current, _, _ = stats("heart_rate_bpm", 15)
    hr_baseline = float(baseline.get("hr_resting_median", 0))
    cards.append(
        {
            "key": "heart_rate",
            "title": "Heart Rate",
            "current": round(hr_current, 0),
            "unit": "bpm",
            "baseline": round(hr_baseline, 0),
            "baseline_label": "personal waking median",
            "change_pct": _pct(hr_current, hr_baseline),
            "reference_range": [55, 95],
            "status": "warning" if hr_current > 100 else ("good" if hr_current <= hr_baseline + 8 else "watch"),
            "status_word": "Elevated" if hr_current > hr_baseline + 8 else "Usual",
            "trend": _trend_arrow(hr_current - hr_baseline),
            "extra": {"mean_60min": round(f.get("hr_mean_60", 0), 0)},
            "sparkline": sparkline("heart_rate_bpm"),
            "source": "Wrist wearable (PPG)",
        }
    )

    hrv_current, _, _ = stats("hrv_rmssd_ms", 30)
    hrv_baseline = float(baseline.get("hrv_daytime_median", 0))
    cards.append(
        {
            "key": "hrv",
            "title": "HRV (RMSSD)",
            "current": round(hrv_current, 0),
            "unit": "ms",
            "baseline": round(hrv_baseline, 0),
            "baseline_label": "personal daytime median",
            "change_pct": _pct(hrv_current, hrv_baseline),
            "reference_range": [None, None],
            "status": "warning" if _pct(hrv_current, hrv_baseline) is not None and _pct(hrv_current, hrv_baseline) <= -12 else "good",
            "status_word": "Below baseline" if hrv_current < hrv_baseline - 3 else "Usual",
            "trend": _trend_arrow(-(hrv_current - hrv_baseline)),
            "extra": {"overnight_mean": round(f.get("hrv_overnight_mean", 0), 0)},
            "sparkline": sparkline("hrv_rmssd_ms"),
            "source": "Wrist wearable (PPG, inter-beat intervals)",
        }
    )

    steps_3h = float(f.get("steps_180", 0))
    steps_expected = _expected_window_steps(record, index, 180)
    cards.append(
        {
            "key": "steps",
            "title": "Steps",
            "current": round(steps_3h, 0),
            "unit": "steps / 3 h",
            "baseline": round(steps_expected, 0),
            "baseline_label": "usual for this clock window",
            "change_pct": _pct(steps_3h, steps_expected),
            "reference_range": [None, None],
            "status": "warning" if _pct(steps_3h, steps_expected) is not None and _pct(steps_3h, steps_expected) <= -30 else "good",
            "status_word": "Below usual" if steps_3h < steps_expected * 0.7 else "Usual",
            "trend": _trend_arrow(steps_3h - steps_expected),
            "extra": {"today": round(f.get("steps_today", 0), 0), "daily_median": round(float(baseline.get("daily_steps_median", 0)), 0)},
            "sparkline": sparkline("steps_5min"),
            "source": "Wrist wearable (accelerometer)",
        }
    )

    sleep_last = float(f.get("sleep_duration_h", 0))
    sleep_baseline = float(baseline.get("sleep_duration_median_h", 0))
    cards.append(
        {
            "key": "sleep",
            "title": "Sleep",
            "current": round(sleep_last, 1),
            "unit": "h last night",
            "baseline": round(sleep_baseline, 1),
            "baseline_label": "personal median duration",
            "change_pct": _pct(sleep_last, sleep_baseline),
            "reference_range": [7.0, 9.0],
            "status": "warning" if sleep_last < sleep_baseline - 0.8 else "good",
            "status_word": "Short" if sleep_last < sleep_baseline - 0.8 else "Usual",
            "trend": _trend_arrow(sleep_last - sleep_baseline),
            "extra": {
                "efficiency": round(float(f.get("sleep_efficiency", 0)) * 100, 0),
                "awakenings": round(float(f.get("sleep_awakenings", 0)), 0),
            },
            "sparkline": None,
            "source": "Wrist wearable (hypnogram)",
        }
    )

    met = float(f.get("activity_met_60", 1.0))
    cards.append(
        {
            "key": "activity",
            "title": "Activity Level",
            "current": round(met, 1),
            "unit": "MET (60 min)",
            "baseline": 1.6,
            "baseline_label": "typical daytime sitting/standing",
            "change_pct": _pct(met, 1.6),
            "reference_range": [1.5, 3.0],
            "status": "warning" if met < 1.3 else "good",
            "status_word": _activity_word(met),
            "trend": _trend_arrow(met - 1.6),
            "extra": {"sedentary_minutes": round(float(f.get("sedentary_minutes_continuous", 0)), 0)},
            "sparkline": sparkline("activity_met"),
            "source": "Wrist wearable (accelerometer + heart rate)",
        }
    )

    spo2_current, _, _ = stats("spo2_pct", 30)
    cards.append(
        {
            "key": "spo2",
            "title": "SpO₂",
            "current": round(spo2_current, 0),
            "unit": "%",
            "baseline": round(float(record.frame["spo2_pct"].astype(float).median()), 0),
            "baseline_label": "personal median",
            "change_pct": _pct(spo2_current, float(record.frame["spo2_pct"].astype(float).median())),
            "reference_range": [95, 100],
            "status": "warning" if spo2_current < 94 else "good",
            "status_word": "Normal" if spo2_current >= 95 else "Low",
            "trend": _trend_arrow(0),
            "extra": {},
            "sparkline": sparkline("spo2_pct"),
            "source": "Wrist wearable (reflectance oximetry)",
        }
    )

    for card in cards:
        card["precision_note"] = "Values are reported at device precision."
    return cards


def _activity_word(met: float) -> str:
    if met < 1.3:
        return "Sedentary"
    if met < 1.8:
        return "Light"
    if met < 3.0:
        return "Moderate"
    return "Vigorous"


def _trend_arrow(delta: float) -> str:
    if delta > 1.5:
        return "rising"
    if delta < -1.5:
        return "falling"
    return "flat"


def build_fusion(record: PatientRecord, index: int) -> Dict[str, Any]:
    """The DATA FUSION panel: historical record | live stream | fused state."""
    index = record.clip_index(index)
    profile = record.profile
    ehr = profile.get("ehr", {})
    f = record.feature_row(index)
    domains = domain_scores(record, index)
    now = record.timestamp_at(index)

    historical = {
        "title": "Historical Patient Record",
        "subtitle": "Longitudinal EHR — static at the moment of prediction",
        "groups": [
            {
                "label": "Demographics",
                "items": [
                    {"k": "Age", "v": f"{ehr.get('demographics', {}).get('age', profile.get('age'))} years"},
                    {"k": "Sex", "v": profile.get("sex")},
                    {"k": "BMI", "v": f"{ehr.get('demographics', {}).get('bmi')} kg/m²"},
                    {"k": "Waist", "v": f"{ehr.get('demographics', {}).get('waist_cm')} cm"},
                ],
            },
            {
                "label": "Diagnoses",
                "items": [{"k": d["name"], "v": d["diagnosed"]} for d in ehr.get("diagnoses", [])[:5]],
            },
            {
                "label": "Medication history",
                "items": [
                    {"k": m["name"], "v": f"{m['dose']} {m['frequency']} — {m['status']}"}
                    for m in ehr.get("medications", [])[:6]
                ],
            },
            {
                "label": "Laboratory",
                "items": [
                    {"k": lab["test"], "v": f"{lab['value']} {lab['unit']}"}
                    for lab in ehr.get("lab_results", [])[:6]
                ],
            },
            {
                "label": "Historical risk",
                "items": [
                    {"k": "Prior CGM time in range", "v": f"{ehr.get('previous_glucose_instability', {}).get('time_in_range_70_180_pct')}%"},
                    {"k": "Prior glucose CV", "v": f"{ehr.get('previous_glucose_instability', {}).get('coefficient_of_variation_pct')}%"},
                    {"k": "Hyperglycaemic episodes (14 d)", "v": ehr.get("previous_glucose_instability", {}).get("hyperglycaemic_episodes_14d")},
                    {"k": "Family history", "v": ", ".join(sorted({fh["condition"] for fh in ehr.get("family_history", [])})) or "none recorded"},
                ],
            },
        ],
        "feature_block": {
            "count": sum(1 for s in fe.FEATURE_SPECS if s.group == "historical"),
            "examples": ["hba1c_pct", "bmi", "on_metformin", "prior_time_in_range_pct", "family_history_diabetes"],
        },
    }

    live = {
        "title": "Live Physiological Stream",
        "subtitle": f"Continuous signals up to {now.strftime('%H:%M')} — updated every {settings.sampling_interval_minutes} minutes",
        "groups": [
            {
                "label": "Continuous glucose",
                "items": [
                    {"k": "Current (15-min mean)", "v": f"{round(f.get('glucose_current', 0))} mg/dL"},
                    {"k": "30-min slope", "v": f"{round(f.get('glucose_slope_30', 0) * 60, 1)} mg/dL/h"},
                    {"k": "3-hour CV", "v": f"{round(f.get('glucose_cv_180', 0), 1)}%"},
                    {"k": "Time above 180 (24 h)", "v": f"{round(f.get('glucose_above_180_24h_pct', 0), 1)}%"},
                ],
            },
            {
                "label": "Cardiovascular",
                "items": [
                    {"k": "Heart rate (15-min mean)", "v": f"{round(f.get('hr_current', 0))} bpm"},
                    {"k": "HRV RMSSD (30-min mean)", "v": f"{round(f.get('hrv_current', 0))} ms"},
                    {"k": "Overnight HRV", "v": f"{round(f.get('hrv_overnight_mean', 0))} ms"},
                    {"k": "SpO₂", "v": f"{round(f.get('spo2_current', 0))}%"},
                ],
            },
            {
                "label": "Activity",
                "items": [
                    {"k": "Steps (3 h)", "v": f"{round(f.get('steps_180', 0))}"},
                    {"k": "Steps today", "v": f"{round(f.get('steps_today', 0))}"},
                    {"k": "Sedentary run", "v": f"{round(f.get('sedentary_minutes_continuous', 0))} min"},
                    {"k": "MET (60 min)", "v": f"{round(f.get('activity_met_60', 0), 1)}"},
                ],
            },
            {
                "label": "Sleep",
                "items": [
                    {"k": "Duration last night", "v": f"{round(f.get('sleep_duration_h', 0), 1)} h"},
                    {"k": "Efficiency", "v": f"{round(f.get('sleep_efficiency', 0) * 100)}%"},
                    {"k": "Awakenings", "v": f"{round(f.get('sleep_awakenings', 0))}"},
                    {"k": "Hours since wake", "v": f"{round(f.get('hours_since_wake', 0), 1)} h"},
                ],
            },
            {
                "label": "Meal context",
                "items": [
                    {"k": "Minutes since last logged meal", "v": f"{round(f.get('minutes_since_last_meal', 0))}"},
                    {"k": "Last meal carbohydrate", "v": f"{round(f.get('last_meal_carbs_g', 0))} g"},
                    {"k": "Carbohydrate last 24 h", "v": f"{round(f.get('carbs_last_24h', 0))} g"},
                    {"k": "Minutes to next habitual meal", "v": f"{round(f.get('minutes_to_next_habitual_meal', 0))}"},
                ],
            },
        ],
        "feature_block": {
            "count": sum(1 for s in fe.FEATURE_SPECS if s.group != "historical"),
            "examples": ["glucose_slope_30", "hrv_dev_personal_pct", "sedentary_hours_ewma", "sleep_deficit_h", "minutes_to_next_habitual_meal"],
        },
    }

    fused = {
        "title": "Digital Twin State",
        "subtitle": "Normalised fusion of both streams (0-100, higher is better)",
        "domains": [
            {
                "key": key,
                "label": DOMAIN_LABELS[key],
                "value": domains[key]["score"],
                "normalised": domains[key]["normalised"],
                "status": domains[key]["status"],
                "word": domains[key]["word"],
                "tone": domains[key]["tone"],
                "weight": domains[key]["weight"],
                "contributing_terms": domains[key]["contributors"],
            }
            for key in DOMAIN_LABELS
        ],
        "composite": round(sum(domains[k]["score"] * DOMAIN_WEIGHTS[k] for k in domains), 1),
        "fusion_note": (
            "Domain scores are deterministic functions of the fused feature vector: the historical block sets the "
            "reference (HbA1c, prior CGM report, personal medians) and the live block sets the deviation from it."
        ),
    }
    return {"as_of": now.isoformat(), "historical": historical, "live": live, "fused": fused}
