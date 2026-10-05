"""
VitalSync — feature engineering.

This module converts a raw multi-signal stream plus the patient's longitudinal
record into the numeric feature vector consumed by the risk model.  It is the
canonical implementation: ``model/feature_engineering.py`` (training) and
``backend/services/prediction_service.py`` (inference) both call into it, so a
feature can never drift between training and serving.

Three ideas drive the design
---------------------------
1. **Fusion.**  Static EHR attributes and dynamic wearable/CGM signals are
   concatenated into one vector, so the model learns interactions such as
   "high HbA1c + sedentary morning" rather than treating them separately.

2. **Personal baselines.**  Every dynamic signal is also expressed as a
   deviation from *this patient's own* baseline, estimated from an onboarding
   window that strictly precedes the scored sample.  A heart rate of 80 bpm is
   unremarkable for one patient and a marked elevation for another.

3. **No future information.**  Features use only data available at the
   prediction instant.  Meal logs count only meals already recorded; sleep
   features describe the most recent *completed* night; the personal baseline
   comes from the onboarding window.  Circadian encodings are included as a
   meal-timing proxy because population-level post-meal risk really is
   time-of-day dependent — but no future meal is ever revealed to the model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature catalogue (also drives the "Model inputs" UI panel)
# ---------------------------------------------------------------------------
FEATURE_GROUPS: Dict[str, str] = {
    "historical": "Historical / EHR (static)",
    "glucose": "Dynamic — continuous glucose",
    "cardio": "Dynamic — heart rate & HRV",
    "activity": "Dynamic — activity & steps",
    "sleep": "Dynamic — sleep",
    "meal": "Dynamic — meal context",
    "baseline": "Personal-baseline deviation",
    "temporal": "Circadian / calendar",
}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    unit: str
    description: str


FEATURE_SPECS: List[FeatureSpec] = [
    # ---- historical / EHR -------------------------------------------------
    FeatureSpec("age", "historical", "years", "Patient age"),
    FeatureSpec("sex_male", "historical", "0/1", "Sex assigned at birth (1 = male)"),
    FeatureSpec("bmi", "historical", "kg/m²", "Body mass index"),
    FeatureSpec("waist_cm", "historical", "cm", "Waist circumference (central adiposity)"),
    FeatureSpec("diabetes_duration_years", "historical", "years", "Years since diabetes/prediabetes diagnosis"),
    FeatureSpec("hba1c_pct", "historical", "%", "Most recent HbA1c"),
    FeatureSpec("fasting_glucose_mgdl", "historical", "mg/dL", "Most recent fasting plasma glucose"),
    FeatureSpec("systolic_bp", "historical", "mmHg", "Most recent systolic blood pressure"),
    FeatureSpec("diastolic_bp", "historical", "mmHg", "Most recent diastolic blood pressure"),
    FeatureSpec("egfr", "historical", "mL/min/1.73m²", "Estimated glomerular filtration rate"),
    FeatureSpec("triglycerides", "historical", "mg/dL", "Fasting triglycerides"),
    FeatureSpec("hdl_cholesterol", "historical", "mg/dL", "HDL cholesterol"),
    FeatureSpec("on_metformin", "historical", "0/1", "Currently prescribed metformin"),
    FeatureSpec("on_sglt2", "historical", "0/1", "Currently prescribed SGLT2 inhibitor"),
    FeatureSpec("on_glp1", "historical", "0/1", "Currently prescribed GLP-1 receptor agonist"),
    FeatureSpec("on_sulfonylurea", "historical", "0/1", "Currently prescribed sulfonylurea"),
    FeatureSpec("on_insulin", "historical", "0/1", "Currently prescribed insulin"),
    FeatureSpec("n_glucose_meds", "historical", "count", "Number of active glucose-lowering agents"),
    FeatureSpec("medication_adherence", "historical", "0-1", "Mean recorded adherence across active medications"),
    FeatureSpec("family_history_diabetes", "historical", "0/1", "First-degree relative with diabetes"),
    FeatureSpec("prior_time_in_range_pct", "historical", "%", "Time in range 70-180 mg/dL on the previous CGM report"),
    FeatureSpec("prior_cv_pct", "historical", "%", "Coefficient of variation on the previous CGM report"),
    FeatureSpec("prior_time_above_180_pct", "historical", "%", "Time above 180 mg/dL on the previous CGM report"),
    FeatureSpec("prior_hyper_episodes", "historical", "count", "Hyperglycaemic episodes in the previous 14-day report"),
    FeatureSpec("hypoglycaemia_episodes_12m", "historical", "count", "Reported hypoglycaemic episodes in 12 months"),
    # ---- dynamic: glucose -------------------------------------------------
    FeatureSpec("glucose_current", "glucose", "mg/dL", "Latest CGM value (mean of last 10 min)"),
    FeatureSpec("glucose_slope_15", "glucose", "mg/dL/min", "15-minute linear glucose slope"),
    FeatureSpec("glucose_slope_30", "glucose", "mg/dL/min", "30-minute linear glucose slope"),
    FeatureSpec("glucose_slope_60", "glucose", "mg/dL/min", "60-minute linear glucose slope"),
    FeatureSpec("glucose_acceleration", "glucose", "mg/dL/min²", "Short-window slope minus long-window slope (curvature)"),
    FeatureSpec("glucose_mean_30", "glucose", "mg/dL", "Rolling 30-minute mean"),
    FeatureSpec("glucose_mean_60", "glucose", "mg/dL", "Rolling 60-minute mean"),
    FeatureSpec("glucose_mean_180", "glucose", "mg/dL", "Rolling 3-hour mean"),
    FeatureSpec("glucose_mean_12h", "glucose", "mg/dL", "Rolling 12-hour mean"),
    FeatureSpec("glucose_std_60", "glucose", "mg/dL", "Rolling 60-minute standard deviation"),
    FeatureSpec("glucose_std_180", "glucose", "mg/dL", "Rolling 3-hour standard deviation"),
    FeatureSpec("glucose_cv_180", "glucose", "%", "Rolling 3-hour coefficient of variation"),
    FeatureSpec("glucose_cv_24h", "glucose", "%", "24-hour coefficient of variation"),
    FeatureSpec("glucose_delta_60", "glucose", "mg/dL", "Change versus 60 minutes ago"),
    FeatureSpec("glucose_delta_180", "glucose", "mg/dL", "Change versus 3 hours ago"),
    FeatureSpec("glucose_min_60", "glucose", "mg/dL", "Rolling 60-minute minimum"),
    FeatureSpec("glucose_max_60", "glucose", "mg/dL", "Rolling 60-minute maximum"),
    FeatureSpec("glucose_max_180", "glucose", "mg/dL", "Rolling 3-hour maximum"),
    FeatureSpec("glucose_range_180", "glucose", "mg/dL", "Rolling 3-hour peak-to-trough range"),
    FeatureSpec("glucose_above_180_24h_pct", "glucose", "%", "Share of the last 24 h spent above 180 mg/dL"),
    FeatureSpec("glucose_rate_of_change_15", "glucose", "mg/dL/15min", "Simple 15-minute difference (device-style arrow)"),
    # ---- dynamic: cardiovascular -----------------------------------------
    FeatureSpec("hr_current", "cardio", "bpm", "Heart rate, mean of last 15 min"),
    FeatureSpec("hr_mean_60", "cardio", "bpm", "Rolling 60-minute mean heart rate"),
    FeatureSpec("hr_delta_60", "cardio", "bpm", "Heart-rate trend versus 60 min ago"),
    FeatureSpec("hrv_current", "cardio", "ms", "RMSSD, mean of last 30 min"),
    FeatureSpec("hrv_mean_180", "cardio", "ms", "Rolling 3-hour mean RMSSD"),
    FeatureSpec("hrv_overnight_mean", "cardio", "ms", "Mean RMSSD during the most recent completed night"),
    FeatureSpec("spo2_current", "cardio", "%", "Latest peripheral oxygen saturation"),
    # ---- dynamic: activity ------------------------------------------------
    FeatureSpec("steps_30", "activity", "steps", "Steps in the last 30 minutes"),
    FeatureSpec("steps_60", "activity", "steps", "Steps in the last 60 minutes"),
    FeatureSpec("steps_180", "activity", "steps", "Steps in the last 3 hours"),
    FeatureSpec("steps_today", "activity", "steps", "Cumulative steps since waking today"),
    FeatureSpec("activity_met_60", "activity", "MET", "Mean metabolic equivalent over 60 min"),
    FeatureSpec("activity_change_pct", "activity", "%", "Activity in the last 60 min versus the preceding 60 min"),
    FeatureSpec("sedentary_minutes_continuous", "activity", "min", "Uninterrupted time below the movement threshold"),
    FeatureSpec("sedentary_hours_ewma", "activity", "h", "Exponentially-weighted recent sedentary hours"),
    # ---- dynamic: sleep ---------------------------------------------------
    FeatureSpec("sleep_duration_h", "sleep", "h", "Sleep duration of the most recent completed night"),
    FeatureSpec("sleep_efficiency", "sleep", "0-1", "Sleep efficiency of the most recent night"),
    FeatureSpec("sleep_deep_fraction", "sleep", "0-1", "Deep-sleep share of the most recent night"),
    FeatureSpec("sleep_awakenings", "sleep", "count", "Wake episodes during the most recent night"),
    FeatureSpec("hours_since_wake", "sleep", "h", "Hours since the patient woke up"),
    FeatureSpec("sleep_quality_index", "sleep", "0-1", "Composite of duration, efficiency and fragmentation"),
    # ---- dynamic: meal context --------------------------------------------
    FeatureSpec("minutes_since_last_meal", "meal", "min", "Time since the most recent logged meal (capped at 600)"),
    FeatureSpec("last_meal_carbs_g", "meal", "g", "Carbohydrate load of the most recent logged meal"),
    FeatureSpec("carbs_last_24h", "meal", "g", "Total logged carbohydrate in the last 24 h"),
    FeatureSpec("is_postprandial", "meal", "0/1", "Within 180 min of a logged meal"),
    FeatureSpec("expected_meal_proximity", "meal", "0-1", "Proximity to this patient's own habitual meal times, learned from the onboarding window and weighted by their usual carbohydrate load"),
    FeatureSpec("minutes_to_next_habitual_meal", "meal", "min", "Minutes until the next habitual meal time for this patient (capped at 480)"),
    # ---- personal baseline deviations --------------------------------------
    FeatureSpec("glucose_dev_personal_pct", "baseline", "%", "Current glucose versus personal median"),
    FeatureSpec("glucose_dev_clock_pct", "baseline", "%", "Current glucose versus personal median for this clock hour"),
    FeatureSpec("glucose_cv_dev_personal", "baseline", "ratio", "Recent glucose variability versus personal norm"),
    FeatureSpec("hr_dev_personal", "baseline", "bpm", "Heart rate minus personal daytime resting median"),
    FeatureSpec("hrv_dev_personal_pct", "baseline", "%", "HRV versus personal daytime baseline"),
    FeatureSpec("hrv_overnight_dev_personal_pct", "baseline", "%", "Overnight HRV versus personal overnight baseline"),
    FeatureSpec("activity_dev_personal_pct", "baseline", "%", "Steps in this clock window versus the patient's usual"),
    FeatureSpec("sleep_deficit_h", "baseline", "h", "Personal sleep baseline minus last night's duration"),
    # ---- circadian / calendar ----------------------------------------------
    FeatureSpec("hour_sin", "temporal", "-", "Cyclical encoding of time of day (sin)"),
    FeatureSpec("hour_cos", "temporal", "-", "Cyclical encoding of time of day (cos)"),
    FeatureSpec("is_weekend", "temporal", "0/1", "Saturday or Sunday"),
]

FEATURE_NAMES: List[str] = [f.name for f in FEATURE_SPECS]

# Sensible neutral values used only if an entire column is unobservable
# (e.g. no completed sleep night inside the onboarding window).
_FALLBACK_VALUES: Dict[str, float] = {
    "sleep_duration_h": 6.5,
    "sleep_efficiency": 0.85,
    "sleep_deep_fraction": 0.17,
    "sleep_quality_index": 0.6,
    "hrv_overnight_mean": 45.0,
    "hours_since_wake": 8.0,
    "minutes_since_last_meal": 600.0,
    "sedentary_minutes_continuous": 0.0,
    "is_weekend": 0.0,
    "sex_male": 0.0,
    "medication_adherence": 0.8,
}
FEATURE_NAME_SET = set(FEATURE_NAMES)


def spec_for(name: str) -> Optional[FeatureSpec]:
    for spec in FEATURE_SPECS:
        if spec.name == name:
            return spec
    return None


def _ns_index(values: Any) -> np.ndarray:
    """
    Timestamps as int64 **nanoseconds**.

    pandas >= 3 infers ``datetime64[us]`` for many inputs, and ``.astype(int64)``
    then returns microseconds.  Comparing those against ``Timestamp.value``
    (nanoseconds) silently corrupts every ``searchsorted`` in this module, so all
    conversions go through this helper.
    """
    index = pd.DatetimeIndex(pd.to_datetime(np.asarray(values)))
    return index.as_unit("ns").asi8


def _ns_value(timestamp: Any) -> int:
    return int(pd.Timestamp(timestamp).as_unit("ns").value)


# ---------------------------------------------------------------------------
# EHR normalisation
# ---------------------------------------------------------------------------
def normalise_ehr(ehr: Dict[str, Any]) -> Dict[str, float]:
    """Flatten the nested EHR record into the static feature block."""
    demographics = ehr.get("demographics", {})
    diabetes = ehr.get("diabetes", {})
    meds = ehr.get("medications", []) or []
    labs = ehr.get("lab_results", []) or []
    vitals = ehr.get("vitals_history", {}) or {}
    instability = ehr.get("previous_glucose_instability", {}) or {}
    family = ehr.get("family_history", []) or []

    def lab(name: str, default: float = np.nan) -> float:
        for row in labs:
            if str(row.get("test", "")).strip().lower() == name.lower():
                try:
                    return float(row.get("value"))
                except (TypeError, ValueError):
                    return default
        return default

    bp_readings = vitals.get("blood_pressure") or []
    systolic = float(bp_readings[0]["systolic"]) if bp_readings else float(demographics.get("systolic_bp", np.nan))
    diastolic = float(bp_readings[0]["diastolic"]) if bp_readings else float(demographics.get("diastolic_bp", np.nan))

    active = [m for m in meds if str(m.get("status", "")).lower() in ("current", "active")]

    def has_med(*keywords: str) -> float:
        for med in active:
            blob = f"{med.get('name','')} {med.get('class','')}".lower()
            if any(k.lower() in blob for k in keywords):
                return 1.0
        return 0.0

    adherence_values = [float(m["adherence"]) for m in active if isinstance(m.get("adherence"), (int, float))]
    family_diabetes = 1.0 if any("diabet" in str(f.get("condition", "")).lower() for f in family) else 0.0

    sex = str(demographics.get("sex", ehr.get("sex", ""))).lower()
    if not sex:
        sex = str(ehr.get("sex", "")).lower()

    return {
        "age": _f(demographics.get("age", ehr.get("age"))),
        "sex_male": 1.0 if sex.startswith("m") else 0.0,
        "bmi": _f(demographics.get("bmi")),
        "waist_cm": _f(demographics.get("waist_cm")),
        "diabetes_duration_years": _f(diabetes.get("duration_years")),
        "hba1c_pct": _f(lab("HbA1c")),
        "fasting_glucose_mgdl": _f(lab("Fasting plasma glucose")),
        "systolic_bp": _f(systolic),
        "diastolic_bp": _f(diastolic),
        "egfr": _f(lab("eGFR")),
        "triglycerides": _f(lab("Triglycerides")),
        "hdl_cholesterol": _f(lab("HDL cholesterol")),
        "on_metformin": has_med("metformin", "biguanide"),
        "on_sglt2": has_med("sglt2", "empagliflozin", "dapagliflozin"),
        "on_glp1": has_med("glp-1", "glp1", "semaglutide", "dulaglutide"),
        "on_sulfonylurea": has_med("sulfonylurea", "glimepiride", "gliclazide"),
        "on_insulin": has_med("insulin"),
        "n_glucose_meds": float(len([m for m in active if str(m.get("class", "")).lower() in _GLUCOSE_CLASSES])),
        "medication_adherence": float(np.mean(adherence_values)) if adherence_values else 0.8,
        "family_history_diabetes": family_diabetes,
        "prior_time_in_range_pct": _f(instability.get("time_in_range_70_180_pct")),
        "prior_cv_pct": _f(instability.get("coefficient_of_variation_pct")),
        "prior_time_above_180_pct": _f(instability.get("time_above_180_pct")),
        "prior_hyper_episodes": _f(instability.get("hyperglycaemic_episodes_14d")),
        "hypoglycaemia_episodes_12m": _f(diabetes.get("hypoglycaemia_episodes_12m")),
    }


_GLUCOSE_CLASSES = {
    "biguanide",
    "sglt2 inhibitor",
    "glp-1 receptor agonist",
    "glp-1",
    "sulfonylurea",
    "basal insulin",
    "prandial insulin",
    "insulin",
    "dpp-4 inhibitor",
    "thiazolidinedione",
    "meglitinide",
}


def _f(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Personal baseline (learned from an onboarding window)
# ---------------------------------------------------------------------------
def compute_personal_baseline(
    frame: pd.DataFrame,
    sleep_nights: Sequence[Dict[str, Any]],
    ehr_static: Optional[Dict[str, float]] = None,
    meals: Optional[Sequence[Dict[str, Any]]] = None,
    min_hours: float = 12.0,
) -> Dict[str, Any]:
    """
    Estimate *this patient's* normal ranges from an onboarding window.

    The window handed to this function must strictly precede the samples being
    scored — that constraint is what keeps personalisation from leaking the
    label into the features.
    """
    if frame.empty:
        return {}

    g = frame["glucose_mgdl"].astype(float)
    hr = frame["heart_rate_bpm"].astype(float)
    hrv = frame["hrv_rmssd_ms"].astype(float)
    steps = frame["steps_5min"].astype(float)
    asleep = frame["sleep_stage"].astype(str).str.len() > 0

    awake_mask = ~asleep
    awake = frame.loc[awake_mask]
    clock_hour = awake["timestamp"].dt.hour

    baseline: Dict[str, Any] = {
        "window_start": frame["timestamp"].iloc[0].isoformat(),
        "window_end": frame["timestamp"].iloc[-1].isoformat(),
        "window_hours": round(float(len(frame)) * _interval_minutes(frame) / 60.0, 1),
        "samples": int(len(frame)),
        "glucose_median": float(np.nanmedian(g)),
        "glucose_mean": float(np.nanmean(g)),
        "glucose_p10": float(np.nanpercentile(g, 10)),
        "glucose_p90": float(np.nanpercentile(g, 90)),
        "glucose_std": float(np.nanstd(g)),
        "glucose_cv_pct": float(100.0 * np.nanstd(g) / max(np.nanmean(g), 1e-6)),
        "glucose_median_daytime": float(np.nanmedian(g[awake_mask])) if awake_mask.any() else float(np.nanmedian(g)),
        "hr_resting_median": float(np.nanmedian(hr[awake_mask])) if awake_mask.any() else float(np.nanmedian(hr)),
        "hr_sleep_median": float(np.nanmedian(hr[asleep])) if asleep.any() else float(np.nanmedian(hr)),
        "hrv_daytime_median": float(np.nanmedian(hrv[awake_mask])) if awake_mask.any() else float(np.nanmedian(hrv)),
        "daily_steps_median": float(np.nanmedian(_daily_totals(steps, frame))),
        "sleep_duration_median_h": float(np.nanmedian([s["sleep_duration_h"] for s in sleep_nights])) if sleep_nights else 6.5,
        "sleep_efficiency_median": float(np.nanmedian([s["efficiency"] for s in sleep_nights])) if sleep_nights else 0.85,
        "overnight_hrv_median": float(np.nanmedian([s["mean_hrv_ms"] for s in sleep_nights])) if sleep_nights else float(np.nanmedian(hrv)),
        "time_in_range_pct": float(100.0 * np.mean((g >= 70) & (g <= 180))),
        "time_above_180_pct": float(100.0 * np.mean(g > 180)),
    }

    # habitual meal times, learned only from meals logged inside the window
    baseline["habitual_meals"] = compute_habitual_meals(meals or [], frame)

    # per-clock-hour profiles — lets us ask "is this normal *for 11 a.m.*?"
    baseline["glucose_by_hour"] = _hour_profile(awake["glucose_mgdl"].astype(float), clock_hour)
    baseline["steps_by_hour"] = _hour_profile(steps[awake_mask], clock_hour)
    baseline["hrv_by_hour"] = _hour_profile(hrv[awake_mask], clock_hour)

    if ehr_static:
        for key in ("hba1c_pct", "bmi", "age"):
            if key in ehr_static and not np.isnan(ehr_static[key]):
                baseline.setdefault(key, ehr_static[key])

    baseline["min_hours"] = min_hours
    return baseline


def _interval_minutes(frame: pd.DataFrame) -> float:
    if len(frame) < 2:
        return 5.0
    delta = (frame["timestamp"].iloc[1] - frame["timestamp"].iloc[0]).total_seconds() / 60.0
    return float(delta) if delta > 0 else 5.0


def _daily_totals(steps: pd.Series, frame: pd.DataFrame) -> np.ndarray:
    totals = steps.groupby(frame["timestamp"].dt.date).sum()
    # drop partial days at the edges
    return totals.values[1:-1] if len(totals) > 2 else totals.values


MEAL_BUCKETS = (("breakfast", 4.0, 11.0), ("lunch", 11.0, 16.0), ("dinner", 16.0, 27.0))


def compute_habitual_meals(
    meals: Sequence[Dict[str, Any]], frame: pd.DataFrame
) -> List[Dict[str, float]]:
    """
    Learn *this patient's* habitual meal clock from the onboarding window.

    Meals are bucketed into breakfast / lunch / dinner (times after midnight are
    treated as late dinners by adding 24 h), and each bucket collapses to a
    median clock hour plus a median carbohydrate load.  The result is what makes
    ``expected_meal_proximity`` a personalised feature rather than a generic
    "is it lunchtime for the population" clock.
    """
    if len(frame) == 0:
        return []
    start = frame["timestamp"].iloc[0]
    end = frame["timestamp"].iloc[-1]
    buckets: Dict[str, List[Tuple[float, float]]] = {name: [] for name, _, _ in MEAL_BUCKETS}
    for meal in meals:
        if not meal.get("logged_in_app", True):
            continue
        when = meal.get("datetime") or meal.get("timestamp")
        if when is None:
            continue
        when = pd.Timestamp(when)
        if not (start <= when <= end):
            continue
        hour = when.hour + when.minute / 60.0
        carbs = float(meal.get("carbs_g", 50) or 50)
        if hour < 4.0:
            hour += 24.0
        for name, lo, hi in MEAL_BUCKETS:
            if lo <= hour < hi:
                buckets[name].append((hour, carbs))
                break

    habitual: List[Dict[str, float]] = []
    for name, lo, hi in MEAL_BUCKETS:
        entries = buckets[name]
        if not entries:
            continue
        hours = np.array([e[0] for e in entries])
        carbs = np.array([e[1] for e in entries])
        hour = float(np.median(hours))
        if hour >= 24.0:
            hour -= 24.0
        habitual.append(
            {
                "name": name,
                "hour": round(hour, 2),
                "carbs_g": round(float(np.median(carbs)), 1),
                "weight": round(float(np.clip(np.median(carbs) / 70.0, 0.45, 1.5)), 3),
                "width_h": round(float(np.clip(0.55 + 0.012 * np.median(carbs), 0.9, 2.0)), 2),
                "observations": int(len(entries)),
            }
        )
    return habitual


def _hour_profile(values: pd.Series, hours: pd.Series) -> Dict[int, float]:
    profile: Dict[int, float] = {}
    if len(values) == 0:
        return profile
    grouped = pd.DataFrame({"v": values.to_numpy(), "h": hours.to_numpy()}).groupby("h")["v"].median()
    for hour, value in grouped.items():
        profile[int(hour)] = float(value)
    return profile


def _lookup(profile: Dict[int, float], hours: np.ndarray, fallback: float) -> np.ndarray:
    if not profile:
        return np.full(len(hours), fallback, dtype=float)
    keys = np.array(sorted(profile.keys()))
    vals = np.array([profile[int(k)] for k in keys])
    idx = np.searchsorted(keys, hours, side="left")
    idx = np.clip(idx, 0, len(keys) - 1)
    out = vals[idx]
    return np.where(np.isnan(out), fallback, out)


# ---------------------------------------------------------------------------
# Vectorised feature construction
# ---------------------------------------------------------------------------
def build_feature_frame(
    frame: pd.DataFrame,
    meals: Sequence[Dict[str, Any]],
    sleep_nights: Sequence[Dict[str, Any]],
    ehr_static: Dict[str, float],
    baseline: Dict[str, Any],
    interval_minutes: Optional[float] = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix for every row of ``frame``.

    All rolling statistics are right-aligned and inclusive of the current row
    only, so row *i* never sees row *i+1*.
    """
    df = frame.reset_index(drop=True).copy()
    n = len(df)
    step_min = float(interval_minutes or _interval_minutes(df))
    ts = df["timestamp"]

    def win(minutes: float) -> int:
        return max(1, int(round(minutes / step_min)))

    g = df["glucose_mgdl"].astype(float)
    hr = df["heart_rate_bpm"].astype(float)
    hrv = df["hrv_rmssd_ms"].astype(float)
    steps = df["steps_5min"].astype(float)
    met = df["activity_met"].astype(float)
    spo2 = df["spo2_pct"].astype(float)
    asleep = df["sleep_stage"].astype(str).str.len() > 0

    feats: Dict[str, np.ndarray] = {}

    # ---------------- static block ----------------
    for key, value in ehr_static.items():
        if key in FEATURE_NAME_SET:
            feats[key] = np.full(n, float(value) if value is not None and not pd.isna(value) else np.nan)

    # ---------------- glucose ----------------
    feats["glucose_current"] = g.rolling(win(10), min_periods=1).mean().to_numpy()
    for minutes in (15, 30, 60):
        feats[f"glucose_slope_{minutes}"] = _rolling_slope(g.to_numpy(), win(minutes), step_min)
    feats["glucose_acceleration"] = feats["glucose_slope_15"] - feats["glucose_slope_60"]
    for minutes in (30, 60, 180, 720):
        key = {30: "glucose_mean_30", 60: "glucose_mean_60", 180: "glucose_mean_180", 720: "glucose_mean_12h"}[minutes]
        feats[key] = g.rolling(win(minutes), min_periods=1).mean().to_numpy()
    feats["glucose_std_60"] = g.rolling(win(60), min_periods=2).std().to_numpy()
    feats["glucose_std_180"] = g.rolling(win(180), min_periods=2).std().to_numpy()
    feats["glucose_cv_180"] = 100.0 * feats["glucose_std_180"] / np.maximum(feats["glucose_mean_180"], 1e-6)
    mean_24h = g.rolling(win(1440), min_periods=win(180)).mean().to_numpy()
    std_24h = g.rolling(win(1440), min_periods=win(180)).std().to_numpy()
    feats["glucose_cv_24h"] = 100.0 * std_24h / np.maximum(mean_24h, 1e-6)
    feats["glucose_delta_60"] = (g - g.shift(win(60))).to_numpy()
    feats["glucose_delta_180"] = (g - g.shift(win(180))).to_numpy()
    feats["glucose_min_60"] = g.rolling(win(60), min_periods=1).min().to_numpy()
    feats["glucose_max_60"] = g.rolling(win(60), min_periods=1).max().to_numpy()
    feats["glucose_max_180"] = g.rolling(win(180), min_periods=1).max().to_numpy()
    feats["glucose_range_180"] = feats["glucose_max_180"] - g.rolling(win(180), min_periods=1).min().to_numpy()
    above = (g > 180.0).astype(float)
    feats["glucose_above_180_24h_pct"] = 100.0 * above.rolling(win(1440), min_periods=win(60)).mean().to_numpy()
    feats["glucose_rate_of_change_15"] = (g - g.shift(win(15))).to_numpy()

    # ---------------- cardiovascular ----------------
    feats["hr_current"] = hr.rolling(win(15), min_periods=1).mean().to_numpy()
    feats["hr_mean_60"] = hr.rolling(win(60), min_periods=1).mean().to_numpy()
    feats["hr_delta_60"] = feats["hr_mean_60"] - hr.shift(win(60)).to_numpy()
    feats["hrv_current"] = hrv.rolling(win(30), min_periods=1).mean().to_numpy()
    feats["hrv_mean_180"] = hrv.rolling(win(180), min_periods=1).mean().to_numpy()
    feats["spo2_current"] = spo2.rolling(win(30), min_periods=1).mean().to_numpy()

    # ---------------- activity ----------------
    feats["steps_30"] = steps.rolling(win(30), min_periods=1).sum().to_numpy()
    feats["steps_60"] = steps.rolling(win(60), min_periods=1).sum().to_numpy()
    feats["steps_180"] = steps.rolling(win(180), min_periods=1).sum().to_numpy()
    day_key = ts.dt.date
    feats["steps_today"] = steps.groupby(day_key).cumsum().to_numpy()
    feats["activity_met_60"] = met.rolling(win(60), min_periods=1).mean().to_numpy()
    prev_60 = steps.shift(win(60)).rolling(win(60), min_periods=1).sum().to_numpy()
    feats["activity_change_pct"] = 100.0 * (feats["steps_60"] - prev_60) / np.maximum(prev_60, 12.0)
    feats["sedentary_minutes_continuous"] = _minutes_since_active(steps.to_numpy(), asleep.to_numpy(), step_min, threshold=15.0)
    feats["sedentary_hours_ewma"] = _sedentary_ewma(steps.to_numpy(), asleep.to_numpy(), step_min)

    # ---------------- sleep ----------------
    sleep_feats = _sleep_features(ts.to_numpy(), sleep_nights, baseline, n)
    feats.update(sleep_feats)

    # ---------------- meal context ----------------
    meal_feats = _meal_features(ts.to_numpy(), meals, n, baseline)
    feats.update(meal_feats)

    # ---------------- personal-baseline deviations ----------------
    glucose_median = float(baseline.get("glucose_median_daytime", np.nanmedian(g)))
    glucose_cv_personal = float(baseline.get("glucose_cv_pct", 100.0 * np.nanstd(g) / max(np.nanmean(g), 1e-6)))
    hr_resting = float(baseline.get("hr_resting_median", np.nanmedian(hr)))
    hrv_daytime = float(baseline.get("hrv_daytime_median", np.nanmedian(hrv)))
    hrv_overnight = float(baseline.get("overnight_hrv_median", hrv_daytime))

    feats["glucose_dev_personal_pct"] = 100.0 * (feats["glucose_current"] - glucose_median) / max(glucose_median, 1e-6)

    hours = ts.dt.hour.to_numpy()
    clock_profile = _lookup(baseline.get("glucose_by_hour", {}), hours, glucose_median)
    feats["glucose_dev_clock_pct"] = 100.0 * (feats["glucose_current"] - clock_profile) / np.maximum(clock_profile, 1e-6)
    feats["glucose_cv_dev_personal"] = feats["glucose_cv_180"] / max(glucose_cv_personal, 1e-6)
    feats["hr_dev_personal"] = feats["hr_mean_60"] - hr_resting
    feats["hrv_dev_personal_pct"] = 100.0 * (feats["hrv_current"] - hrv_daytime) / max(hrv_daytime, 1e-6)
    feats["hrv_overnight_dev_personal_pct"] = 100.0 * (feats["hrv_overnight_mean"] - hrv_overnight) / max(hrv_overnight, 1e-6)

    # `steps_by_hour` is a median per sampling interval, so the expected number
    # of steps in a 3-hour window is that median times the number of intervals
    # in 3 hours.  The denominator floor is proportional to the patient's own
    # daily volume so that a quiet hour cannot produce a 1000% deviation.
    # `steps_by_hour` is a median per sampling interval for each clock hour, so
    # the expected number of steps in a window is the sum of the per-interval
    # expectations across that window (not one hour's rate scaled up).
    expected_rate = _lookup(baseline.get("steps_by_hour", {}), hours, 12.0)
    steps_expected_180 = pd.Series(expected_rate).rolling(win(180), min_periods=win(180)).sum().to_numpy()
    daily_median = float(baseline.get("daily_steps_median", 4000.0))
    floor = np.maximum(steps_expected_180, 0.10 * daily_median * (3.0 / 24.0))
    feats["activity_dev_personal_pct"] = np.clip(
        100.0 * (feats["steps_180"] - steps_expected_180) / np.maximum(floor, 10.0), -100.0, 300.0
    )
    sleep_baseline_h = float(baseline.get("sleep_duration_median_h", 6.5))
    feats["sleep_deficit_h"] = sleep_baseline_h - feats["sleep_duration_h"]

    # ---------------- circadian ----------------
    frac = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy() / 60.0
    feats["hour_sin"] = np.sin(2 * np.pi * frac / 24.0)
    feats["hour_cos"] = np.cos(2 * np.pi * frac / 24.0)
    feats["is_weekend"] = np.isin(ts.dt.weekday.to_numpy(), [5, 6]).astype(float)

    out = pd.DataFrame(feats, index=df.index)
    out.insert(0, "timestamp", ts)

    # guarantee a complete, ordered column set
    for name in FEATURE_NAMES:
        if name not in out.columns:
            out[name] = np.nan
    out = out[["timestamp"] + FEATURE_NAMES]

    # ---- cleaning -------------------------------------------------------
    # Non-finite values are replaced with the most recent valid observation and
    # then, for the unavoidable warm-up rows at the head of the stream, with the
    # first valid value.  (Those head rows sit inside the onboarding window and
    # are never scored, so the back-fill cannot leak label information.)
    out = out.replace([np.inf, -np.inf], np.nan)
    out[FEATURE_NAMES] = out[FEATURE_NAMES].ffill().bfill()
    for name in FEATURE_NAMES:
        column = out[name].to_numpy(dtype=float)
        missing = np.isnan(column)
        if missing.any():
            finite = column[~missing]
            fill = float(np.median(finite)) if finite.size else _FALLBACK_VALUES.get(name, 0.0)
            out[name] = np.where(missing, fill, column)
    return out


def _rolling_slope(values: np.ndarray, window: int, step_min: float) -> np.ndarray:
    """
    Ordinary-least-squares slope (mg/dL per minute) over a rolling window.

    OLS is used instead of a simple endpoint difference because CGM noise makes
    two-point slopes very unstable at 5-minute sampling.
    """
    n = len(values)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    window = max(2, min(window, n))
    x = np.arange(window, dtype=float) * step_min
    x = x - x.mean()
    denom = float((x**2).sum())
    if denom <= 0:
        return np.zeros(n)
    # sliding dot product via cumulative sums is overkill; use stride tricks
    padded = np.concatenate([np.full(window - 1, np.nan), values])
    for i in range(window - 1, n):
        y = padded[i - window + 1 : i + 1]
        if np.isnan(y).any():
            continue
        out[i] = float((x * (y - y.mean())).sum() / denom)
    out[: window - 1] = out[window - 1] if n >= window else 0.0
    return out


def _minutes_since_active(steps: np.ndarray, asleep: np.ndarray, step_min: float, threshold: float = 15.0) -> np.ndarray:
    """Minutes of continuous low movement (resets on any active interval)."""
    out = np.zeros(len(steps))
    acc = 0.0
    for i in range(len(steps)):
        active = (steps[i] >= threshold) or asleep[i]
        acc = 0.0 if active else acc + step_min
        out[i] = min(acc, 600.0)
    return out


def _sedentary_ewma(steps: np.ndarray, asleep: np.ndarray, step_min: float, decay_hours: float = 2.5) -> np.ndarray:
    """Exponentially-weighted recent sedentary hours (matches the twin state)."""
    decay = np.exp(-(step_min / 60.0) / decay_hours)
    out = np.zeros(len(steps))
    acc = 0.0
    for i in range(len(steps)):
        sedentary = 1.0 if (steps[i] < 12.0 and not asleep[i]) else 0.0
        acc = acc * decay + sedentary * (step_min / 60.0)
        out[i] = min(acc, 3.0)
    return out


def _sleep_features(timestamps: np.ndarray, sleep_nights: Sequence[Dict[str, Any]], baseline: Dict[str, Any], n: int) -> Dict[str, np.ndarray]:
    """Attach the most recent *completed* night to every timestamp."""
    out = {
        "sleep_duration_h": np.full(n, np.nan),
        "sleep_efficiency": np.full(n, np.nan),
        "sleep_deep_fraction": np.full(n, np.nan),
        "sleep_awakenings": np.full(n, np.nan),
        "hours_since_wake": np.full(n, np.nan),
        "sleep_quality_index": np.full(n, np.nan),
        "hrv_overnight_mean": np.full(n, np.nan),
    }
    nights = []
    for s in sleep_nights:
        wake = s.get("wake_dt")
        if wake is None:
            # reconstruct from the stored clock fields + night_of date
            wake = pd.Timestamp(f"{s['night_of']} {s.get('wake_time','06:00')}") + pd.Timedelta(days=1)
        nights.append((pd.Timestamp(wake), s))
    nights.sort(key=lambda item: item[0])
    if not nights:
        return out

    wake_times = np.array([_ns_value(w) for w, _ in nights], dtype="int64")
    ts_values = _ns_index(timestamps)
    idx = np.searchsorted(wake_times, ts_values, side="right") - 1

    dur_baseline = float(baseline.get("sleep_duration_median_h", 6.5))
    eff_baseline = float(baseline.get("sleep_efficiency_median", 0.85))

    for i, pos in enumerate(idx):
        if pos < 0:
            continue
        night = nights[pos][1]
        duration = float(night.get("sleep_duration_h", dur_baseline))
        efficiency = float(night.get("efficiency", eff_baseline))
        deep = float(night.get("deep_fraction", 0.17))
        awakenings = float(night.get("awakenings", 1))
        out["sleep_duration_h"][i] = duration
        out["sleep_efficiency"][i] = efficiency
        out["sleep_deep_fraction"][i] = deep
        out["sleep_awakenings"][i] = awakenings
        out["hrv_overnight_mean"][i] = float(night.get("mean_hrv_ms", np.nan))
        hours_since = (ts_values[i] - nights[pos][0].value) / 3_600_000_000_000.0
        out["hours_since_wake"][i] = float(np.clip(hours_since, 0.0, 24.0))
        # composite: duration, efficiency, deep sleep and fragmentation
        quality = (
            0.40 * np.clip(duration / max(dur_baseline, 1e-6), 0, 1.2) / 1.2
            + 0.30 * np.clip(efficiency / 0.95, 0, 1.0)
            + 0.20 * np.clip(deep / 0.22, 0, 1.0)
            + 0.10 * np.clip(1.0 - awakenings / 6.0, 0, 1.0)
        )
        out["sleep_quality_index"][i] = float(np.clip(quality, 0.0, 1.0))
    return out


def _meal_features(
    timestamps: np.ndarray, meals: Sequence[Dict[str, Any]], n: int, baseline: Optional[Dict[str, Any]] = None
) -> Dict[str, np.ndarray]:
    """
    Meal context available *at* each timestamp.

    Only meals already logged are visible.  ``expected_meal_proximity`` is a
    circadian proxy for habitual meal timing: it rises shortly before the
    population-typical breakfast/lunch/dinner windows, which is legitimate
    prior information, whereas the actual future meal is not.
    """
    out = {
        "minutes_since_last_meal": np.full(n, 600.0),
        "last_meal_carbs_g": np.zeros(n),
        "carbs_last_24h": np.zeros(n),
        "is_postprandial": np.zeros(n),
        "expected_meal_proximity": np.zeros(n),
    }
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    ts_values = _ns_index(timestamps)

    logged = []
    for m in meals:
        when = m.get("datetime") or m.get("timestamp")
        if when is None:
            continue
        logged.append((pd.Timestamp(when), float(m.get("carbs_g", 0.0)), bool(m.get("logged_in_app", True))))
    logged.sort(key=lambda item: item[0])

    if logged:
        meal_times = np.array([_ns_value(m[0]) for m in logged], dtype="int64")
        meal_carbs = np.array([m[1] for m in logged], dtype=float)
        meal_logged = np.array([m[2] for m in logged], dtype=bool)
        pos = np.searchsorted(meal_times, ts_values, side="right") - 1
        valid = pos >= 0
        p = np.clip(pos, 0, len(logged) - 1)
        minutes = (ts_values - meal_times[p]) / 60_000_000_000.0
        out["minutes_since_last_meal"] = np.where(valid & meal_logged[p], np.clip(minutes, 0, 600), 600.0)
        out["last_meal_carbs_g"] = np.where(valid & meal_logged[p], meal_carbs[p], 0.0)
        out["is_postprandial"] = np.where(valid & meal_logged[p] & (minutes <= 180), 1.0, 0.0)

        # carbohydrate load over the trailing 24 h
        carbs_24 = np.zeros(n)
        day_ns = 24 * 3_600_000_000_000
        for mt, carbs, was_logged in logged:
            if not was_logged:
                continue
            mt_ns = _ns_value(mt)
            sel = (ts_values >= mt_ns) & (ts_values < mt_ns + day_ns)
            carbs_24[sel] += carbs
        out["carbs_last_24h"] = carbs_24

    # Personalised meal clock: gaussian bumps centred on the patient's own
    # habitual meal times, weighted by their usual carbohydrate load.  If the
    # onboarding window contained no logged meals we fall back to population
    # defaults so the feature is never missing.
    frac = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    habitual = (baseline or {}).get("habitual_meals") or []
    if habitual:
        centres = [(float(m["hour"]), float(m["width_h"]), float(m["weight"])) for m in habitual]
    else:
        centres = [(8.0, 1.4, 1.0), (13.0, 1.6, 1.0), (20.5, 1.5, 0.9)]
    proximity = np.zeros(n)
    minutes_to_next = np.full(n, 480.0)
    for centre, width, weight in centres:
        d = np.minimum(np.abs(frac - centre), 24.0 - np.abs(frac - centre))
        proximity += weight * np.exp(-(d**2) / (2 * width**2))
        ahead = (centre - frac) % 24.0                     # hours until this meal
        minutes_to_next = np.minimum(minutes_to_next, ahead * 60.0)
    out["expected_meal_proximity"] = np.clip(proximity, 0.0, 2.0)
    out["minutes_to_next_habitual_meal"] = np.clip(minutes_to_next, 0.0, 480.0)
    return out


# ---------------------------------------------------------------------------
# Single-instant feature vector (used by the live prediction path)
# ---------------------------------------------------------------------------
def build_features_at(
    frame: pd.DataFrame,
    meals: Sequence[Dict[str, Any]],
    sleep_nights: Sequence[Dict[str, Any]],
    ehr_static: Dict[str, float],
    baseline: Dict[str, Any],
    index: int,
    interval_minutes: Optional[float] = None,
) -> Dict[str, float]:
    """Return the feature vector for one instant (``index`` into ``frame``)."""
    window = frame.iloc[: index + 1]
    matrix = build_feature_frame(window, meals, sleep_nights, ehr_static, baseline, interval_minutes)
    row = matrix.iloc[-1]
    return {name: float(row[name]) for name in FEATURE_NAMES}
