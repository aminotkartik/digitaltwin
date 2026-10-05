"""
VitalSync — deterministic synthetic physiology generator.

This module is the single source of truth for *all* physiological time-series
in the prototype:

  * the demo patients streamed into the clinician dashboard, and
  * the multi-patient training cohort used to fit the risk model.

Using one generator for both means the model is trained on exactly the same
signal dynamics it is asked to score at inference time, and the whole dataset
can be reproduced from a seed.

Design notes (clinical plausibility)
-----------------------------------
Glucose is modelled as

    G(t) = fasting_set_point
         + circadian_drive(t)                 (dawn phenomenon, evening dip)
         + sum_meals postprandial_curve(t)    (gamma-shaped absorption)
         + sedentary_drift(t)                 (hepatic glucose output)
         + AR(1) sensor/physiological noise

Postprandial amplitude scales with carbohydrate load and inversely with the
patient's insulin sensitivity, is blunted by glucose-lowering medication, and
is amplified after a night of short / fragmented sleep — the mechanism that
drives the scripted deterioration in the demo scenario.

All outputs are rounded to the precision a real device would report
(CGM: whole mg/dL, HR: whole bpm, HRV: whole ms, SpO2: whole %, steps: whole).
No feature is left with artificial floating point precision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 24 * 60

# Columns emitted for every generated stream (device-native precision).
STREAM_COLUMNS = [
    "timestamp",
    "glucose_mgdl",
    "heart_rate_bpm",
    "hrv_rmssd_ms",
    "steps_5min",
    "activity_met",
    "spo2_pct",
    "sleep_stage",
]

SLEEP_STAGES = ("awake", "light", "deep", "rem")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _clock_to_hours(clock: str) -> float:
    """'13:45' -> 13.75 (decimal hours since midnight)."""
    hh, mm = clock.split(":")[:2]
    return int(hh) + int(mm) / 60.0


def _ar1(rng: np.random.Generator, n: int, sd: float, rho: float = 0.86) -> np.ndarray:
    """Stationary AR(1) noise — CGM error is autocorrelated, not white."""
    if sd <= 0 or n <= 0:
        return np.zeros(max(n, 0))
    innovations = rng.normal(0.0, sd * np.sqrt(1.0 - rho**2), size=n)
    out = np.empty(n)
    acc = rng.normal(0.0, sd)
    for i in range(n):
        acc = rho * acc + innovations[i]
        out[i] = acc
    return out


def _gamma_postprandial(hours_since: np.ndarray, amplitude: float, t_peak: float, shape: float = 3.0) -> np.ndarray:
    """
    Gamma-shaped postprandial excursion.

    Peaks at ``t_peak`` hours after the meal with height ``amplitude`` and
    decays back to baseline over ~3.5-4 h, which matches observed CGM
    postprandial morphology far better than a symmetric Gaussian.
    """
    x = np.clip(hours_since, 0.0, None)
    curve = (x / t_peak) ** shape * np.exp(shape * (1.0 - x / t_peak))
    curve[hours_since <= 0] = 0.0
    return amplitude * curve


# ---------------------------------------------------------------------------
# Default physiology / scenario templates
# ---------------------------------------------------------------------------
DEFAULT_PHYSIOLOGY: Dict[str, Any] = {
    "seed": 1000,
    "fasting_glucose_mgdl": 118.0,
    "glucose_floor_mgdl": 78.0,
    "insulin_sensitivity": 0.75,          # 0.40 = resistant, 1.10 = sensitive
    "cgm_noise_mgdl": 3.2,
    "carb_response_mgdl_per_g": 0.42,     # at insulin_sensitivity = 1.0
    "med_hepatic_output_factor": 0.94,    # metformin-like fasting reduction
    "med_postprandial_factor": 0.95,      # GLP-1 / SGLT2-like blunting
    "resting_hr_bpm": 70.0,
    "hrv_baseline_ms": 46.0,
    "daily_steps_baseline": 5600,
    "sleep_duration_baseline_h": 6.6,
    "sleep_efficiency_baseline": 0.84,
    "sleep_deep_fraction_baseline": 0.17,
    "spo2_baseline_pct": 96.6,
    "weekend_activity_factor": 0.82,
    # day-to-day (between-day) variability multiplier: how erratic this
    # patient's glycaemia is from one day to the next
    "variability_scale": 1.0,
}

DEFAULT_MEALS: List[Dict[str, Any]] = [
    {"clock": "07:45", "carbs_g": 52, "label": "Breakfast"},
    {"clock": "13:00", "carbs_g": 74, "label": "Lunch"},
    {"clock": "20:15", "carbs_g": 66, "label": "Dinner"},
]

DEFAULT_SLEEP: Dict[str, Any] = {
    "bed": "23:15",
    "wake": "06:00",
    "efficiency": None,        # None -> use baseline + jitter
    "deep_fraction": None,
    "awakenings": None,
}

# Hourly relative activity template (0-23). Scaled to the patient's step target.
HOURLY_ACTIVITY_TEMPLATE = np.array(
    [
        0.00, 0.00, 0.00, 0.00, 0.00, 0.02,   # 00-05  asleep
        0.35, 0.95, 1.35, 0.85, 0.55, 0.75,   # 06-11  wake, commute, morning
        0.90, 0.45, 0.60, 0.45, 0.55, 0.80,   # 12-17  lunch walk, afternoon
        1.05, 0.85, 0.55, 0.35, 0.20, 0.08,   # 18-23  evening, wind down
    ]
)


@dataclass
class GeneratedStream:
    """Container for a generated physiological stream plus its provenance."""

    patient_id: str
    frame: pd.DataFrame
    meals: List[Dict[str, Any]]
    sleep_nights: List[Dict[str, Any]]
    activity_blocks: List[Dict[str, Any]]
    params: Dict[str, Any]
    scenario_day: date
    interval_minutes: int

    @property
    def start(self) -> pd.Timestamp:
        return self.frame["timestamp"].iloc[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.frame["timestamp"].iloc[-1]

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "interval_minutes": self.interval_minutes,
            "scenario_day": self.scenario_day.isoformat(),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "rows": int(len(self.frame)),
            "meals": self.meals,
            "sleep_nights": self.sleep_nights,
            "activity_blocks": self.activity_blocks,
            "generator_params": self.params,
        }


# ---------------------------------------------------------------------------
# Scenario resolution
# ---------------------------------------------------------------------------
def _resolve_day_plan(
    day_offset: int,
    scenario: Dict[str, Any],
    physiology: Dict[str, Any],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Merge scripted scenario entries for one day with sensible defaults.

    Anything not explicitly scripted is randomised deterministically from the
    patient seed, so ordinary days look ordinary and only the scripted
    scenario day carries the engineered deterioration.
    """
    scripted_meals = [m for m in scenario.get("meals", []) if int(m.get("day_offset", 0)) == day_offset]
    scripted_sleep = [s for s in scenario.get("sleep_nights", []) if int(s.get("day_offset", 0)) == day_offset]
    scripted_blocks = [b for b in scenario.get("activity_blocks", []) if int(b.get("day_offset", 0)) == day_offset]

    weekday = (scenario.get("_weekday", 2) + day_offset) % 7
    is_weekend = weekday in (5, 6)

    if scripted_meals:
        meals = [
            {
                "clock": m["clock"],
                "carbs_g": float(m.get("carbs_g", 60)),
                "label": m.get("label", "Meal"),
                "day_offset": day_offset,
                "logged": bool(m.get("logged", True)),
            }
            for m in sorted(scripted_meals, key=lambda x: _clock_to_hours(x["clock"]))
        ]
    else:
        scale = float(rng.uniform(0.88, 1.14))
        meals = []
        for template in DEFAULT_MEALS:
            carbs = template["carbs_g"] * scale * float(rng.uniform(0.9, 1.1))
            meals.append(
                {
                    "clock": template["clock"],
                    "carbs_g": round(carbs),
                    "label": template["label"],
                    "day_offset": day_offset,
                    "logged": bool(rng.random() > 0.12),  # patients miss some logs
                }
            )
        # occasional evening snack
        if rng.random() < 0.35:
            meals.append(
                {
                    "clock": "17:30",
                    "carbs_g": int(rng.integers(14, 30)),
                    "label": "Snack",
                    "day_offset": day_offset,
                    "logged": bool(rng.random() > 0.3),
                }
            )

    # Sleep semantics: `bed`/`wake` bound the time *in bed*; sleep efficiency
    # determines how much of that time was actually spent asleep.
    if scripted_sleep:
        s = scripted_sleep[0]
        bed_h = _clock_to_hours(s["bed"])
        wake_h = _clock_to_hours(s["wake"])
        time_in_bed = (wake_h - bed_h) % 24.0
        efficiency = float(s.get("efficiency", physiology["sleep_efficiency_baseline"]))
        night = {
            "day_offset": day_offset,
            "bed": s["bed"],
            "wake": s["wake"],
            "time_in_bed_h": round(time_in_bed, 2),
            "duration_h": round(time_in_bed * efficiency, 2),
            "efficiency": round(efficiency, 2),
            "deep_fraction": float(s.get("deep_fraction", physiology["sleep_deep_fraction_baseline"])),
            "rem_fraction": float(s.get("rem_fraction", 0.21)),
            "awakenings": int(s.get("awakenings", 1)),
            "note": s.get("note", ""),
        }
    else:
        dur = float(np.clip(rng.normal(physiology["sleep_duration_baseline_h"], 0.42), 4.2, 9.2))
        eff = float(np.clip(rng.normal(physiology["sleep_efficiency_baseline"], 0.045), 0.62, 0.96))
        time_in_bed = dur / eff
        wake_h = 6.0 + float(rng.normal(0, 0.35))
        bed_h = (wake_h - time_in_bed) % 24.0
        night = {
            "day_offset": day_offset,
            "bed": _hours_to_clock(bed_h),
            "wake": _hours_to_clock(wake_h),
            "time_in_bed_h": round(time_in_bed, 2),
            "duration_h": round(dur, 2),
            "efficiency": round(eff, 2),
            "deep_fraction": round(float(np.clip(rng.normal(physiology["sleep_deep_fraction_baseline"], 0.03), 0.06, 0.28)), 2),
            "rem_fraction": round(float(np.clip(rng.normal(0.21, 0.03), 0.10, 0.28)), 2),
            "awakenings": int(np.clip(round(rng.normal(1.8, 0.9)), 0, 6)),
            "note": "",
        }

    return {
        "day_offset": day_offset,
        "meals": meals,
        "sleep": night,
        "activity_blocks": scripted_blocks,
        "is_weekend": is_weekend,
        "day_activity_factor": float(rng.uniform(0.8, 1.2)) * (physiology["weekend_activity_factor"] if is_weekend else 1.0),
    }


def _hours_to_clock(h: float) -> str:
    h = h % 24.0
    hh = int(h)
    mm = int(round((h - hh) * 60))
    if mm == 60:
        hh, mm = hh + 1, 0
    return f"{hh % 24:02d}:{mm:02d}"


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
def generate_stream(
    patient_id: str,
    physiology: Optional[Dict[str, Any]] = None,
    scenario: Optional[Dict[str, Any]] = None,
    end_time: Optional[datetime] = None,
    hours: int = 72,
    interval_minutes: int = 5,
) -> GeneratedStream:
    """
    Build a multi-day, multi-signal physiological stream for one patient.

    Parameters
    ----------
    patient_id : stable identifier (used for nothing but metadata)
    physiology : per-patient generator parameters (see DEFAULT_PHYSIOLOGY)
    scenario   : scripted meals / sleep nights / activity blocks by day_offset,
                 where day_offset 0 is the calendar day of ``end_time``.
    end_time   : wall-clock instant the stream ends at (the "now" of the twin).
    hours      : how much history to generate.
    interval_minutes : sampling cadence (5 min = typical CGM/wearable rate).
    """
    phys = {**DEFAULT_PHYSIOLOGY, **(physiology or {})}
    scen = dict(scenario or {})
    rng = np.random.default_rng(int(phys["seed"]))

    if end_time is None:
        end_time = datetime.now().replace(second=0, microsecond=0)
    end_time = end_time.replace(second=0, microsecond=0)
    # align the end of the stream to the sampling grid
    minute = (end_time.minute // interval_minutes) * interval_minutes
    end_time = end_time.replace(minute=minute)

    n = int(hours * 60 / interval_minutes) + 1
    timestamps = pd.date_range(end=pd.Timestamp(end_time), periods=n, freq=f"{interval_minutes}min")
    scenario_day = end_time.date()
    scen.setdefault("_weekday", scenario_day.weekday())

    ts = timestamps.to_numpy().astype("datetime64[m]").astype(float)  # minutes since epoch
    minutes_since_start = (ts - ts[0])
    hours_since_start = minutes_since_start / 60.0
    clock_hours = (timestamps.hour + timestamps.minute / 60.0).to_numpy()
    day_offsets = np.array([(t.date() - scenario_day).days for t in timestamps])

    # ---- resolve per-day plans -----------------------------------------
    plans: Dict[int, Dict[str, Any]] = {}
    for off in range(int(day_offsets.min()), int(day_offsets.max()) + 1):
        plans[off] = _resolve_day_plan(off, scen, phys, rng)

    # ---- between-day variability ----------------------------------------
    # Real ambulatory glucose is not a repeating pattern: the fasting set point
    # moves from day to day, meal responses vary with timing/stress/hydration,
    # and occasional stress or minor-illness episodes raise glucose for hours.
    # These three terms are what give the trace a realistic coefficient of
    # variation instead of an implausibly smooth curve.
    vscale = float(phys.get("variability_scale", 1.0))
    day_offsets_unique = sorted(plans.keys())

    # Days that are explicitly scripted (the demo scenario) keep a factor of
    # 1.0 so the engineered clinical narrative is reproduced exactly; every
    # other day drifts, which is what creates realistic between-day variation.
    def _scripted(off: int) -> bool:
        return any(int(m.get("day_offset", -999)) == off for m in scen.get("meals", []))

    set_point_factor = {
        off: 1.0 if _scripted(off) else float(np.clip(rng.normal(1.0, 0.095 * vscale), 0.80, 1.22))
        for off in day_offsets_unique
    }
    amplitude_factor = {
        off: 1.0 if _scripted(off) else float(np.clip(rng.normal(1.0, 0.26 * vscale), 0.55, 1.75))
        for off in day_offsets_unique
    }
    stress_profile = np.zeros(n)
    for off in day_offsets_unique:
        if _scripted(off):
            continue
        if rng.random() < min(0.42, 0.16 * vscale):
            start_h = float(rng.uniform(8.5, 19.0))
            duration_h = float(rng.uniform(2.5, 7.0))
            magnitude = float(rng.uniform(9.0, 28.0) * vscale)
            centre = start_h + duration_h / 2.0
            spread = duration_h / 2.6
            day_mask = day_offsets == off
            bump = magnitude * np.exp(-((clock_hours - centre) ** 2) / (2 * spread**2))
            stress_profile[day_mask] += bump[day_mask]
    set_point_by_sample = np.array([set_point_factor[int(d)] for d in day_offsets])
    amplitude_by_day = amplitude_factor

    # ---- absolute timestamps for meals / sleep --------------------------
    meals_abs: List[Dict[str, Any]] = []
    for off, plan in sorted(plans.items()):
        base_day = datetime.combine(scenario_day + timedelta(days=off), time(0, 0))
        for m in plan["meals"]:
            mt = base_day + timedelta(hours=_clock_to_hours(m["clock"]))
            meals_abs.append({**m, "datetime": mt})
    meals_abs.sort(key=lambda m: m["datetime"])

    # sleep of "night N" starts on day N-1 evening and ends on day N morning
    sleep_abs: List[Dict[str, Any]] = []
    for off, plan in sorted(plans.items()):
        s = plan["sleep"]
        wake_day = datetime.combine(scenario_day + timedelta(days=off), time(0, 0))
        wake_dt = wake_day + timedelta(hours=_clock_to_hours(s["wake"]))
        bed_dt = wake_dt - timedelta(hours=max(s["duration_h"] / max(s["efficiency"], 0.4), 0.5))
        sleep_abs.append({**s, "bed_dt": bed_dt, "wake_dt": wake_dt, "target_h": s["duration_h"]})
    sleep_abs.sort(key=lambda s: s["bed_dt"])

    # ---- sleep stages ----------------------------------------------------
    sleep_stage = np.array([""] * n, dtype=object)
    asleep_mask = np.zeros(n, dtype=bool)
    for s in sleep_abs:
        idx = np.where((timestamps >= pd.Timestamp(s["bed_dt"])) & (timestamps < pd.Timestamp(s["wake_dt"])))[0]
        if len(idx) == 0:
            continue
        asleep_mask[idx] = True
        sleep_stage[idx] = _build_sleep_stages(len(idx), s, rng)

    # ---- steps / activity ------------------------------------------------
    steps = _build_steps(
        n=n,
        clock_hours=clock_hours,
        day_offsets=day_offsets,
        plans=plans,
        asleep_mask=asleep_mask,
        daily_target=float(phys["daily_steps_baseline"]),
        interval_minutes=interval_minutes,
        rng=rng,
    )

    # ---- glucose ---------------------------------------------------------
    fasting = float(phys["fasting_glucose_mgdl"]) * float(phys["med_hepatic_output_factor"])

    circadian = (
        3.4 * np.sin(2 * np.pi * (clock_hours - 4.2) / 24.0)      # dawn rise
        + 1.8 * np.sin(2 * np.pi * (clock_hours - 9.0) / 12.0)     # afternoon wobble
    )

    # Cumulative sedentary exposure: exponentially-weighted count of recent
    # sedentary hours.  Sustained inactivity raises hepatic glucose output and
    # acutely impairs postprandial glucose disposal (well described in the
    # "breaking up sitting time" literature), so it feeds both the drift term
    # and the amplitude of the next meal excursion.
    sedentary = (steps < 12).astype(float) * (~asleep_mask).astype(float)
    sed_hours = _running_integral(
        sedentary * (interval_minutes / 60.0), decay_hours=2.5, interval_hours=interval_minutes / 60.0
    )
    sed_hours = np.clip(sed_hours, 0.0, 3.0)
    drift = 5.0 * sed_hours

    glucose = np.full(n, fasting, dtype=float) * set_point_by_sample + circadian + drift + stress_profile

    # Sleep debt -> acute insulin resistance.  One night of short, fragmented
    # sleep reduces glucose disposal the following day; the factor is applied
    # per sample across the night itself and the whole of the following day.
    quality = np.ones(n, dtype=float)
    night_quality: Dict[int, float] = {}
    for s_night in sleep_abs:
        deficit = float(phys["sleep_duration_baseline_h"]) - s_night["duration_h"]
        eff_deficit = float(phys["sleep_efficiency_baseline"]) - s_night["efficiency"]
        amp = float(np.clip(1.0 + 0.045 * max(deficit, 0.0) + 0.30 * max(eff_deficit, 0.0), 1.0, 1.30))
        night_quality[s_night["day_offset"]] = amp
        sel = np.asarray(
            (timestamps >= pd.Timestamp(s_night["bed_dt"]))
            & (timestamps < pd.Timestamp(s_night["wake_dt"]) + pd.Timedelta(hours=18))
        )
        quality[sel] = np.maximum(quality[sel], amp)

    for meal in meals_abs:
        m_ts = pd.Timestamp(meal["datetime"])
        hours_since = (timestamps - m_ts).to_numpy().astype("timedelta64[s]").astype(float) / 3600.0
        if np.nanmin(hours_since) > 6.0:
            continue
        # acute inactivity at the time of the meal amplifies the excursion
        meal_idx = int(np.abs((timestamps - m_ts).total_seconds()).argmin())
        day_factor = float(quality[min(meal_idx, n - 1)])
        inactivity_factor = 1.0 + 0.06 * float(sed_hours[min(meal_idx, n - 1)])
        amplitude = (
            meal["carbs_g"]
            * float(phys["carb_response_mgdl_per_g"])
            / max(float(phys["insulin_sensitivity"]), 0.25)
            * float(phys["med_postprandial_factor"])
            * day_factor
            * inactivity_factor
            * amplitude_by_day.get(meal["day_offset"], 1.0)
        )
        t_peak = 1.02 + 0.30 * (meal["carbs_g"] / 85.0)
        glucose += _gamma_postprandial(hours_since, amplitude, t_peak)

    # CGM error is proportional to the reading (MARD-like behaviour) and
    # autocorrelated, so noise = fixed floor + percentage of the current value.
    noise_sd = float(phys["cgm_noise_mgdl"]) + 0.022 * np.clip(glucose, 60, None)
    glucose = glucose + _ar1(rng, n, float(np.mean(noise_sd)), rho=0.85) * (noise_sd / max(float(np.mean(noise_sd)), 1e-6))
    glucose = np.clip(glucose, float(phys["glucose_floor_mgdl"]), 420.0)
    # nocturnal glucose settles towards a lower set point
    glucose[asleep_mask] = glucose[asleep_mask] * 0.955 + 2.0

    # ---- heart rate ------------------------------------------------------
    activity_bpm = np.clip(steps * 0.095, 0, 58)
    postprandial_hr = np.zeros(n)
    for meal in meals_abs:
        m_ts = pd.Timestamp(meal["datetime"])
        h_since = (timestamps - m_ts).to_numpy().astype("timedelta64[s]").astype(float) / 3600.0
        postprandial_hr += 4.2 * np.exp(-((h_since - 0.7) ** 2) / 1.1) * (np.abs(h_since) < 3.0)
    hr_circadian = 3.1 * np.sin(2 * np.pi * (clock_hours - 8.0) / 24.0)
    metabolic_load = 0.055 * np.clip(glucose - 120.0, 0, None)
    hr = (
        float(phys["resting_hr_bpm"])
        + activity_bpm
        + postprandial_hr
        + hr_circadian
        + metabolic_load
        + _ar1(rng, n, 2.1, rho=0.80)
    )
    hr[asleep_mask] = float(phys["resting_hr_bpm"]) - 9.0 + 0.03 * activity_bpm[asleep_mask] + _ar1(rng, int(asleep_mask.sum()), 1.7)
    hr = np.clip(hr, 44, 178)

    # ---- HRV (RMSSD) -----------------------------------------------------
    # `hrv_baseline_ms` is the patient's typical daytime resting RMSSD; sleep
    # raises parasympathetic tone, so nocturnal values sit above it.
    night_gain = 1.15
    hrv = np.where(asleep_mask, float(phys["hrv_baseline_ms"]) * night_gain, float(phys["hrv_baseline_ms"]))
    hrv = hrv / quality ** 1.25
    hrv -= 0.30 * (hr - float(phys["resting_hr_bpm"]))
    hrv -= 0.12 * np.clip(glucose - 150.0, 0, None)
    hrv += 1.9 * np.sin(2 * np.pi * (clock_hours - 3.0) / 24.0)
    hrv += _ar1(rng, n, 3.1, rho=0.83)
    hrv = np.clip(hrv, 7, 145)

    # ---- SpO2 ------------------------------------------------------------
    spo2 = float(phys["spo2_baseline_pct"]) + 0.35 * np.sin(2 * np.pi * (clock_hours - 10.0) / 24.0)
    spo2 -= 0.7 * asleep_mask
    spo2 += _ar1(rng, n, 0.42, rho=0.75)
    spo2 = np.clip(spo2, 88, 100)

    # ---- activity METs ---------------------------------------------------
    met = 1.0 + steps / 62.0
    met[asleep_mask] = 0.92
    met += _ar1(rng, n, 0.11, rho=0.7)
    met = np.clip(met, 0.8, 11.5)

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "glucose_mgdl": np.rint(glucose).astype(int),
            "heart_rate_bpm": np.rint(hr).astype(int),
            "hrv_rmssd_ms": np.rint(hrv).astype(int),
            "steps_5min": steps.astype(int),
            "activity_met": np.round(met, 1),
            "spo2_pct": np.rint(spo2).astype(int),
            "sleep_stage": sleep_stage,
        }
    )

    # ---- nightly sleep summaries -----------------------------------------
    sleep_nights_out: List[Dict[str, Any]] = []
    for s in sleep_abs:
        mask = (frame["timestamp"] >= pd.Timestamp(s["bed_dt"])) & (frame["timestamp"] < pd.Timestamp(s["wake_dt"]))
        block = frame.loc[mask]
        if block.empty:
            continue
        asleep_rows = int((block["sleep_stage"] != "awake").sum())
        total_rows = int(len(block))
        stage_seq = block["sleep_stage"].tolist()
        # an "awakening" is a *transition* into wakefulness after sleep onset,
        # not every awake sample (ignore the sleep-onset latency period)
        onset = next((i for i, st in enumerate(stage_seq) if st != "awake"), 0)
        awakenings = sum(
            1 for i in range(max(onset, 1), len(stage_seq)) if stage_seq[i] == "awake" and stage_seq[i - 1] != "awake"
        )
        sleep_nights_out.append(
            {
                "night_of": s["bed_dt"].date().isoformat(),
                "bed_time": s["bed_dt"].strftime("%H:%M"),
                "wake_time": s["wake_dt"].strftime("%H:%M"),
                "time_in_bed_h": round(total_rows * interval_minutes / 60.0, 1),
                "sleep_duration_h": round(asleep_rows * interval_minutes / 60.0, 1),
                "efficiency": round(asleep_rows / max(total_rows, 1), 2),
                "deep_fraction": round(float((block["sleep_stage"] == "deep").sum() / max(asleep_rows, 1)), 2),
                "rem_fraction": round(float((block["sleep_stage"] == "rem").sum() / max(asleep_rows, 1)), 2),
                "awakenings": awakenings,
                "mean_hr_bpm": int(round(float(block["heart_rate_bpm"].mean()))),
                "mean_hrv_ms": int(round(float(block["hrv_rmssd_ms"].mean()))),
                "mean_glucose_mgdl": int(round(float(block["glucose_mgdl"].mean()))),
                "note": s.get("note", ""),
            }
        )

    meals_out = [
        {
            "datetime": m["datetime"].isoformat(),
            "clock": m["clock"],
            "day_offset": m["day_offset"],
            "label": m["label"],
            "carbs_g": m["carbs_g"],
            "logged_in_app": m["logged"],
        }
        for m in meals_abs
        if timestamps[0] <= pd.Timestamp(m["datetime"]) <= timestamps[-1]
    ]

    return GeneratedStream(
        patient_id=patient_id,
        frame=frame,
        meals=meals_out,
        sleep_nights=sleep_nights_out,
        activity_blocks=[
            {
                "day_offset": int(b.get("day_offset", 0)),
                "start": b["start"],
                "end": b["end"],
                "intensity": b.get("intensity", "sedentary"),
                "label": b.get("label", ""),
            }
            for b in scen.get("activity_blocks", [])
        ],
        params={k: v for k, v in phys.items()},
        scenario_day=scenario_day,
        interval_minutes=interval_minutes,
    )


def _running_integral(values: np.ndarray, decay_hours: float, interval_hours: float) -> np.ndarray:
    """
    Exponentially decaying accumulation of ``values`` over time.

    ``values`` are per-interval hour contributions, so the accumulator
    saturates near ``decay_hours`` — i.e. it behaves like an
    exponentially-weighted count of recent sedentary hours.
    """
    decay = np.exp(-interval_hours / max(decay_hours, 1e-6))
    out = np.empty_like(values, dtype=float)
    acc = 0.0
    for i, v in enumerate(values):
        acc = acc * decay + v
        out[i] = acc
    return out


def _build_steps(
    n: int,
    clock_hours: np.ndarray,
    day_offsets: np.ndarray,
    plans: Dict[int, Dict[str, Any]],
    asleep_mask: np.ndarray,
    daily_target: float,
    interval_minutes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Distribute a daily step target across the day, honouring scripted blocks."""
    template_hour = np.floor(clock_hours).astype(int) % 24
    weights = HOURLY_ACTIVITY_TEMPLATE[template_hour].astype(float)

    steps = np.zeros(n)
    for off, plan in plans.items():
        mask = day_offsets == off
        count = int(mask.sum())
        if count == 0:
            continue
        w = weights[mask]
        total_w = float(w.sum())
        if total_w <= 0:
            continue
        # Distribute this day's step target proportionally to the hourly
        # activity template, then add bursty per-interval variation.
        target = daily_target * plan["day_activity_factor"]
        share = target * (w / total_w)
        steps[mask] = share * rng.uniform(0.55, 1.55, size=count)

    # scripted activity blocks override the generic template
    for off, plan in plans.items():
        for block in plan.get("activity_blocks", []):
            start_clock = _clock_to_hours(block["start"])
            end_clock = _clock_to_hours(block["end"])
            spm = float(block.get("steps_per_min", 2.0))
            sel = (day_offsets == off) & (clock_hours >= start_clock) & (clock_hours < end_clock) & (~asleep_mask)
            if sel.any():
                steps[sel] = spm * interval_minutes * rng.uniform(0.6, 1.4, size=int(sel.sum()))

    steps[asleep_mask] = rng.integers(0, 2, size=int(asleep_mask.sum()))
    steps = np.clip(steps, 0, None)
    # bursty behaviour: some intervals have zero steps even when awake
    awake = ~asleep_mask
    zero_burst = rng.random(n) < np.where(awake, 0.18, 0.0)
    steps[zero_burst] = 0.0
    return np.rint(steps)


def _build_sleep_stages(n: int, night: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """
    Synthesise a hypnogram for one night.

    Structure follows normal sleep architecture: deep (N3) sleep is
    front-loaded into the first cycles, REM lengthens towards morning, and
    wake after sleep onset occurs as a small number of discrete episodes whose
    total duration reproduces the recorded sleep efficiency.
    """
    efficiency = float(np.clip(night.get("efficiency", 0.85), 0.50, 0.98))
    deep_fraction = float(np.clip(night.get("deep_fraction", 0.17), 0.04, 0.30))
    rem_fraction = float(np.clip(night.get("rem_fraction", 0.21), 0.08, 0.30))
    awakenings = int(np.clip(night.get("awakenings", 1), 0, 8))

    stages = np.array(["light"] * n, dtype=object)
    if n <= 0:
        return stages

    asleep_budget = int(round(n * efficiency))
    deep_budget = int(round(asleep_budget * deep_fraction))
    rem_budget = int(round(asleep_budget * rem_fraction))

    # sleep onset latency (a few minutes awake at the start of the night)
    onset = int(np.clip(round(n * rng.uniform(0.02, 0.06)), 0, max(n - 1, 0)))
    stages[:onset] = "awake"

    # --- deep sleep: front-loaded, in blocks inside the first cycles --------
    cycles = max(3, int(round(n / 18.0)))          # ~90 min cycles at 5-min sampling
    cycle_len = max(1, n // cycles)
    placed_deep = 0
    for c in range(cycles):
        if placed_deep >= deep_budget:
            break
        weight = max(0.0, 1.0 - 0.85 * (c / max(cycles - 1, 1)))   # deep declines across night
        quota = int(round(deep_budget * weight / max(cycles / 2.0, 1)))
        quota = min(quota, deep_budget - placed_deep, cycle_len)
        if quota <= 0:
            continue
        start_idx = c * cycle_len + onset + 1
        end_idx = min(n, start_idx + quota)
        stages[start_idx:end_idx] = "deep"
        placed_deep += end_idx - start_idx

    # --- REM: back-loaded ---------------------------------------------------
    placed_rem = 0
    for c in range(cycles - 1, -1, -1):
        if placed_rem >= rem_budget:
            break
        weight = max(0.0, (c + 1) / cycles)
        quota = int(round(rem_budget * weight / max(cycles / 2.0, 1)))
        quota = min(quota, rem_budget - placed_rem, cycle_len)
        if quota <= 0:
            continue
        start_idx = min(n - 1, (c + 1) * cycle_len - quota)
        end_idx = start_idx + quota
        block = stages[start_idx:end_idx]
        free = block != "deep"
        take = min(int(free.sum()), quota)
        if take > 0:
            idxs = np.where(free)[0][:take]
            stages[start_idx + idxs] = "rem"
            placed_rem += take

    # --- wake after sleep onset: discrete episodes --------------------------
    total_wake_target = max(0, n - asleep_budget)
    waso_target = max(0, total_wake_target - onset)          # wake after sleep onset
    n_episodes = awakenings if waso_target > 0 else 0
    if n_episodes > 0:
        positions = np.linspace(int(n * 0.20), int(n * 0.92), n_episodes).astype(int)
        per_episode = max(1, int(np.ceil(waso_target / n_episodes)))
        placed = 0
        for pos in positions:
            if placed >= waso_target:
                break
            end = min(n, pos + per_episode)
            stages[pos:end] = "awake"
            placed += end - pos

    # --- reconcile with the recorded efficiency -----------------------------
    # Grow or shrink wake time so that (asleep samples / total samples) matches
    # the efficiency reported by the wearable for this night.
    deficit = total_wake_target - int((stages == "awake").sum())
    if deficit > 0:
        wake_idx = np.where(stages == "awake")[0]
        i = 0
        while deficit > 0 and len(wake_idx) > 0:
            idx = int(wake_idx[i % len(wake_idx)])
            # extend the episode forwards into adjacent sleep
            while deficit > 0 and idx + 1 < n and stages[idx + 1] != "awake":
                idx += 1
                stages[idx] = "awake"
                deficit -= 1
            i += 1
            if i > 6 * max(len(wake_idx), 1):
                break
    elif deficit < 0:
        wake_idx = np.where(stages == "awake")[0]
        for idx in wake_idx[::-1][: -deficit]:
            if idx < onset:
                continue
            stages[idx] = "light"
            deficit += 1
    return stages


# ---------------------------------------------------------------------------
# Event labelling (shared by training, evaluation and the live demo)
# ---------------------------------------------------------------------------
def sustained_high_mask(
    glucose: np.ndarray, threshold: float = 180.0, min_consecutive: int = 2
) -> np.ndarray:
    """
    True where glucose is at/above ``threshold`` for ``min_consecutive``
    consecutive samples.

    Requiring two consecutive readings (10 minutes at 5-minute sampling) is the
    standard way of ignoring single-sample sensor artefacts while preserving the
    true peak magnitude — unlike a moving average, which would blunt the very
    excursion we are trying to detect.
    """
    g = np.asarray(glucose, dtype=float)
    above = g >= threshold
    mask = above.copy()
    for k in range(1, max(1, min_consecutive)):
        shifted = np.zeros_like(above)
        if k < len(above):
            shifted[: len(above) - k] = above[k:]
        mask &= shifted
    return mask


def label_events_fast(
    glucose: np.ndarray,
    horizon_steps: int,
    high_threshold: float = 180.0,
    rise_delta: float = 25.0,
    min_consecutive: int = 2,
) -> np.ndarray:
    """
    Vectorised label construction.

    Label 1 at index *i* means: within the next ``horizon_steps`` samples the
    glucose trace reaches ``high_threshold`` mg/dL for at least
    ``min_consecutive`` consecutive samples **and** that peak is at least
    ``rise_delta`` mg/dL above the value at *i*.

    Rows without a complete future window are marked -1 and must be dropped
    before training or evaluation.
    """
    g = np.asarray(glucose, dtype=float)
    n = len(g)
    out = np.full(n, -1, dtype=int)
    if n <= horizon_steps:
        return out
    sustained = sustained_high_mask(g, high_threshold, min_consecutive)
    sentinel = np.where(sustained, g, -1e9)
    # NOTE the ordering: rolling(...).max() first, then shift(-horizon).
    # That yields future_peak[i] = max(sentinel[i+1 ... i+horizon]) — a strictly
    # forward-looking window.  Shifting before rolling would silently include
    # past samples and leak the current value into the label.
    future_peak = (
        pd.Series(sentinel)
        .rolling(window=horizon_steps, min_periods=horizon_steps)
        .max()
        .shift(-horizon_steps)
        .to_numpy()
    )
    valid = ~np.isnan(future_peak)
    out[valid] = (
        (future_peak[valid] >= high_threshold) & ((future_peak[valid] - g[valid]) >= rise_delta)
    ).astype(int)
    return out


def label_events(
    glucose: Sequence[float],
    horizon_steps: int,
    high_threshold: float = 180.0,
    rise_delta: float = 25.0,
    min_consecutive: int = 2,
) -> np.ndarray:
    """Reference (loop) implementation, kept for unit testing the fast path."""
    g = np.asarray(glucose, dtype=float)
    n = len(g)
    sustained = sustained_high_mask(g, high_threshold, min_consecutive)
    out = np.full(n, -1, dtype=int)
    if n <= horizon_steps:
        return out
    for i in range(n - horizon_steps):
        future = g[i + 1 : i + 1 + horizon_steps]
        future_flags = sustained[i + 1 : i + 1 + horizon_steps]
        if not future_flags.any():
            out[i] = 0
            continue
        peak = float(np.max(future[future_flags]))
        out[i] = int(peak >= high_threshold and (peak - g[i]) >= rise_delta)
    return out
