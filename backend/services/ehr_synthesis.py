"""
VitalSync — synthetic EHR synthesis.

Two jobs:

1. ``cgm_report_from_stream`` computes the standard ATTD/ADA glycaemic
   metrics (mean glucose, CV, time in/above/below range, estimated GMI) from a
   generated glucose trace.  The demo patients' "previous CGM report" fields in
   ``patients.json`` are produced this way, so the historical record and the
   live signal can never contradict each other.

2. ``ehr_from_physiology`` derives a plausible longitudinal record (HbA1c,
   fasting glucose, BMI, blood pressure, medications, prior instability) from a
   set of generator parameters.  The training cohort uses it, which is what
   makes the *static* half of the feature vector genuinely informative instead
   of random noise — HbA1c really does track the mean glucose of the stream it
   belongs to.

Small amounts of measurement noise are added on top so the model cannot recover
the generator parameters exactly; a real HbA1c is not a deterministic function
of two weeks of CGM data.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

HIGH_THRESHOLD = 180.0
LOW_THRESHOLD = 70.0


def cgm_report_from_stream(
    frame: pd.DataFrame,
    high_threshold: float = HIGH_THRESHOLD,
    low_threshold: float = LOW_THRESHOLD,
    episode_gap_minutes: int = 30,
) -> Dict[str, Any]:
    """Standard ambulatory glucose profile metrics for a generated trace."""
    g = frame["glucose_mgdl"].astype(float).to_numpy()
    if g.size == 0:
        return {}
    interval_min = 5.0
    if len(frame) > 1:
        delta = (frame["timestamp"].iloc[1] - frame["timestamp"].iloc[0]).total_seconds() / 60.0
        interval_min = float(delta) if delta > 0 else 5.0

    mean = float(np.mean(g))
    sd = float(np.std(g, ddof=1)) if len(g) > 1 else 0.0
    above = g > high_threshold
    below = g < low_threshold

    # count contiguous excursions above threshold, merging gaps shorter than
    # `episode_gap_minutes` (a standard way of counting discrete events)
    gap_samples = max(1, int(episode_gap_minutes / interval_min))
    episodes = 0
    run_gap = gap_samples + 1
    for value in above:
        if value:
            if run_gap > gap_samples:
                episodes += 1
            run_gap = 0
        else:
            run_gap += 1

    return {
        "mean_glucose_mgdl": round(mean, 0),
        "standard_deviation_mgdl": round(sd, 1),
        "coefficient_of_variation_pct": round(100.0 * sd / mean, 1) if mean > 0 else 0.0,
        "time_in_range_70_180_pct": round(100.0 * float(np.mean((g >= low_threshold) & (g <= high_threshold))), 1),
        "time_above_180_pct": round(100.0 * float(np.mean(above)), 1),
        "time_below_70_pct": round(100.0 * float(np.mean(below)), 1),
        "hyperglycaemic_episodes": episodes,
        "estimated_gmi_pct": round((mean + 46.7) / 28.7, 1),
        "samples": int(len(g)),
        "days_covered": round(len(g) * interval_min / (24 * 60), 1),
    }


def ehr_from_physiology(
    physiology: Dict[str, Any],
    rng: np.random.Generator,
    report: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Build the static (historical) feature block for a synthetic cohort member.

    Values are anchored to the generator parameters and then perturbed, so the
    record is *correlated* with the signal it describes without being a copy of
    it.
    """
    fasting = float(physiology.get("fasting_glucose_mgdl", 118.0))
    sensitivity = float(physiology.get("insulin_sensitivity", 0.75))
    steps = float(physiology.get("daily_steps_baseline", 5600))
    sleep_h = float(physiology.get("sleep_duration_baseline_h", 6.6))

    # metabolic severity: lower sensitivity + higher fasting glucose = worse
    severity = float(np.clip((fasting - 90.0) / 70.0 + (1.0 - sensitivity) * 0.9, 0.0, 2.2))

    if report:
        mean_glucose = float(report.get("mean_glucose_mgdl", fasting + 15))
    else:
        mean_glucose = fasting + 12.0 + 16.0 * severity

    hba1c = float(np.clip((mean_glucose + 46.7) / 28.7 + rng.normal(0.0, 0.22), 4.9, 12.4))
    age = float(np.clip(rng.normal(54 - 6 * sensitivity + 4 * severity, 8.0), 26, 84))
    bmi = float(np.clip(rng.normal(24.6 + 3.6 * severity + (0.0 if steps > 8000 else 1.1), 2.6), 18.2, 44.0))
    waist = float(np.clip(bmi * 3.15 + rng.normal(6.0, 3.2), 62, 132))
    systolic = float(np.clip(rng.normal(118 + 9.5 * severity + 0.13 * (age - 50), 8.5), 96, 188))
    diastolic = float(np.clip(systolic * 0.55 + rng.normal(8.0, 4.0), 56, 110))
    egfr = float(np.clip(rng.normal(104 - 8.5 * severity - 0.35 * max(age - 45, 0), 9.0), 26, 125))
    triglycerides = float(np.clip(rng.normal(112 + 42 * severity + 0.25 * (bmi - 25) * 6, 34), 48, 420))
    hdl = float(np.clip(rng.normal(52 - 6.5 * severity + 0.22 * (steps / 1000.0), 7.5), 22, 92))
    duration = float(np.clip(rng.normal(1.4 + 4.2 * severity, 2.4), 0.0, 26.0))

    # treatment intensity follows severity
    on_metformin = 1.0 if severity > 0.22 or rng.random() < 0.25 else 0.0
    on_sglt2 = 1.0 if severity > 0.85 and rng.random() < 0.72 else 0.0
    on_glp1 = 1.0 if severity > 1.05 and rng.random() < 0.55 else 0.0
    on_sulfonylurea = 1.0 if severity > 0.6 and rng.random() < 0.4 else 0.0
    on_insulin = 1.0 if severity > 1.35 and rng.random() < 0.68 else 0.0
    n_meds = on_metformin + on_sglt2 + on_glp1 + on_sulfonylurea + on_insulin

    cv = float(report.get("coefficient_of_variation_pct", 18.0)) if report else float(np.clip(rng.normal(17 + 6 * severity, 4.0), 8, 46))
    tir = float(report.get("time_in_range_70_180_pct", 80.0)) if report else float(np.clip(rng.normal(92 - 22 * severity, 8.0), 18, 100))
    tar = float(report.get("time_above_180_pct", 15.0)) if report else float(np.clip(100 - tir - rng.uniform(0, 3), 0, 82))

    return {
        "age": round(age, 0),
        "sex_male": float(rng.random() < 0.54),
        "bmi": round(bmi, 1),
        "waist_cm": round(waist, 0),
        "diabetes_duration_years": round(duration, 1),
        "hba1c_pct": round(hba1c, 1),
        "fasting_glucose_mgdl": round(float(np.clip(fasting + rng.normal(0, 6), 78, 240)), 0),
        "systolic_bp": round(systolic, 0),
        "diastolic_bp": round(diastolic, 0),
        "egfr": round(egfr, 0),
        "triglycerides": round(triglycerides, 0),
        "hdl_cholesterol": round(hdl, 0),
        "on_metformin": on_metformin,
        "on_sglt2": on_sglt2,
        "on_glp1": on_glp1,
        "on_sulfonylurea": on_sulfonylurea,
        "on_insulin": on_insulin,
        "n_glucose_meds": float(n_meds),
        "medication_adherence": round(float(np.clip(rng.normal(0.86 - 0.06 * severity, 0.09), 0.35, 1.0)), 2),
        "family_history_diabetes": float(rng.random() < (0.34 + 0.16 * severity)),
        "prior_time_in_range_pct": round(tir, 1),
        "prior_cv_pct": round(cv, 1),
        "prior_time_above_180_pct": round(tar, 1),
        "prior_hyper_episodes": round(float(np.clip(rng.normal(2 + 7 * severity, 2.4), 0, 34)), 0),
        "hypoglycaemia_episodes_12m": round(float(np.clip(rng.normal(0.4 + 1.1 * on_insulin + 0.5 * on_sulfonylurea, 0.8), 0, 9)), 0),
    }


def sample_physiology(rng: np.random.Generator, index: int) -> Dict[str, Any]:
    """Draw a coherent physiology profile for a synthetic cohort member."""
    # The target population is people with impaired glucose regulation who are
    # wearing a CGM, so severity is centred above the healthy range.
    severity = float(np.clip(rng.beta(3.2, 2.0), 0.05, 1.0))       # 0 = healthy, 1 = dysregulated
    fitness = float(np.clip(rng.beta(2.4, 2.6), 0.0, 1.0))         # 0 = sedentary, 1 = very active

    fasting = float(np.clip(rng.normal(96 + 58 * severity, 8), 84, 190))
    sensitivity = float(np.clip(1.02 - 0.58 * severity + 0.14 * fitness + rng.normal(0, 0.05), 0.34, 1.18))
    steps = int(np.clip(rng.normal(2900 + 8200 * fitness - 900 * severity, 1500), 700, 18000))
    sleep_h = float(np.clip(rng.normal(6.9 - 1.15 * severity + 0.35 * fitness, 0.7), 4.1, 9.3))
    sleep_eff = float(np.clip(rng.normal(0.885 - 0.11 * severity, 0.045), 0.60, 0.96))
    resting_hr = float(np.clip(rng.normal(74 - 13 * fitness + 7 * severity, 6), 48, 98))
    hrv = float(np.clip(rng.normal(63 - 26 * severity + 13 * fitness - 0.16 * max(resting_hr - 62, 0), 9), 12, 96))

    return {
        "seed": int(10_000_000 + index * 7919 + int(rng.integers(0, 7919))),
        "fasting_glucose_mgdl": round(fasting, 1),
        "glucose_floor_mgdl": round(float(np.clip(fasting - 42 - 8 * severity, 58, 120)), 1),
        "insulin_sensitivity": round(sensitivity, 3),
        "cgm_noise_mgdl": round(float(np.clip(rng.normal(3.1 + 1.9 * severity, 0.7), 1.6, 7.4)), 2),
        "carb_response_mgdl_per_g": round(float(np.clip(rng.normal(0.46, 0.04), 0.32, 0.60)), 3),
        "med_hepatic_output_factor": round(float(np.clip(0.99 - 0.06 * severity, 0.88, 1.0)), 3),
        "med_postprandial_factor": round(float(np.clip(1.0 - 0.13 * severity, 0.80, 1.0)), 3),
        "resting_hr_bpm": round(resting_hr, 1),
        "hrv_baseline_ms": round(hrv, 1),
        "daily_steps_baseline": steps,
        "sleep_duration_baseline_h": round(sleep_h, 2),
        "sleep_efficiency_baseline": round(sleep_eff, 3),
        "sleep_deep_fraction_baseline": round(float(np.clip(rng.normal(0.19 - 0.06 * severity, 0.035), 0.05, 0.27)), 3),
        "spo2_baseline_pct": round(float(np.clip(rng.normal(96.8 - 0.7 * severity, 0.7), 93.5, 99.0)), 1),
        "weekend_activity_factor": round(float(np.clip(rng.normal(0.92, 0.16), 0.55, 1.45)), 2),
        "_severity": round(severity, 3),
        "_fitness": round(fitness, 3),
    }


def sample_scenario(rng: np.random.Generator, physiology: Dict[str, Any], days: int) -> Dict[str, Any]:
    """
    Build a randomised-but-plausible multi-day scenario (meals, sleep nights,
    activity blocks).  Roughly one day in six carries an deliberate
    "deterioration" pattern — short sleep, a sedentary block and a heavier
    lunch — which is what creates positive events for the model to learn.
    """
    meals: list[Dict[str, Any]] = []
    sleep_nights: list[Dict[str, Any]] = []
    activity_blocks: list[Dict[str, Any]] = []

    habitual_breakfast = round(float(rng.uniform(7.2, 9.6)) * 4) / 4
    habitual_lunch = round(float(rng.uniform(12.2, 14.4)) * 4) / 4
    habitual_dinner = round(float(rng.uniform(19.2, 21.8)) * 4) / 4
    carb_scale = float(rng.uniform(0.78, 1.28))

    for offset in range(-(days - 1), 1):
        deterioration = rng.random() < (0.34 if offset == 0 else 0.22)

        base_carbs = {"b": 54, "l": 88, "d": 78}
        if deterioration:
            base_carbs = {"b": 64, "l": 108, "d": 90}
        # Real life is irregular: meals are sometimes skipped entirely and
        # often shift by up to ~40 minutes.  Without this the model could learn
        # a patient's meal clock perfectly, which would overstate how much a
        # meal-timing prior is worth in deployment.
        skipped = None
        if rng.random() < 0.16:
            skipped = rng.choice(["b", "l", "d"], p=[0.5, 0.2, 0.3])
        for key, clock in (("b", habitual_breakfast), ("l", habitual_lunch), ("d", habitual_dinner)):
            if key == skipped:
                continue
            carbs = float(np.clip(rng.normal(base_carbs[key] * carb_scale, 9), 18, 145))
            shift = float(rng.normal(0, 0.30)) if rng.random() < 0.35 else float(rng.normal(0, 0.12))
            meals.append(
                {
                    "day_offset": offset,
                    "clock": _hours_to_clock(clock + shift),
                    "carbs_g": int(round(carbs)),
                    "label": {"b": "Breakfast", "l": "Lunch", "d": "Dinner"}[key],
                    "logged": bool(rng.random() > 0.10),
                }
            )
        # snacks
        for _ in range(int(rng.integers(0, 3))):
            meals.append(
                {
                    "day_offset": offset,
                    "clock": _hours_to_clock(float(rng.uniform(10.0, 22.0))),
                    "carbs_g": int(rng.integers(12, 42)),
                    "label": "Snack",
                    "logged": bool(rng.random() > 0.35),
                }
            )

        if deterioration:
            duration = float(np.clip(rng.normal(4.9, 0.55), 3.6, 6.4))
            efficiency = float(np.clip(rng.normal(0.74, 0.05), 0.58, 0.86))
        else:
            duration = float(np.clip(rng.normal(physiology["sleep_duration_baseline_h"], 0.5), 4.0, 9.4))
            efficiency = float(np.clip(rng.normal(physiology["sleep_efficiency_baseline"], 0.045), 0.60, 0.96))
        time_in_bed = duration / efficiency
        wake_h = float(np.clip(rng.normal(6.4, 0.7), 4.2, 9.5))
        sleep_nights.append(
            {
                "day_offset": offset,
                "bed": _hours_to_clock((wake_h - time_in_bed) % 24.0),
                "wake": _hours_to_clock(wake_h),
                "efficiency": round(efficiency, 2),
                "deep_fraction": round(float(np.clip(rng.normal(physiology["sleep_deep_fraction_baseline"], 0.03), 0.05, 0.28)), 2),
                "awakenings": int(np.clip(round(rng.normal(4.2 if deterioration else 1.8, 1.1)), 0, 8)),
                "note": "Short, fragmented sleep" if deterioration else "",
            }
        )

        if deterioration and rng.random() < 0.85:
            start = float(rng.uniform(9.0, 11.5))
            activity_blocks.append(
                {
                    "day_offset": offset,
                    "start": _hours_to_clock(start),
                    "end": _hours_to_clock(start + float(rng.uniform(1.6, 3.2))),
                    "intensity": "sedentary",
                    "steps_per_min": float(rng.uniform(0.6, 2.2)),
                    "label": "Prolonged sitting",
                }
            )
        if rng.random() < 0.5:
            start = float(rng.uniform(16.5, 19.5))
            activity_blocks.append(
                {
                    "day_offset": offset,
                    "start": _hours_to_clock(start),
                    "end": _hours_to_clock(start + float(rng.uniform(0.4, 1.1))),
                    "intensity": "moderate",
                    "steps_per_min": float(rng.uniform(55, 105)),
                    "label": "Planned walk / exercise",
                }
            )

    meals.sort(key=lambda m: (m["day_offset"], m["clock"]))
    return {"meals": meals, "sleep_nights": sleep_nights, "activity_blocks": activity_blocks}


def _hours_to_clock(h: float) -> str:
    h = float(h) % 24.0
    hh = int(h)
    mm = int(round((h - hh) * 60))
    if mm == 60:
        hh, mm = hh + 1, 0
    return f"{hh % 24:02d}:{mm:02d}"
