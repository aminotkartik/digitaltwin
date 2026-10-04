"""
VitalSync — prediction service.

Turns a materialised twin into the object a clinician actually reads:

  * a probability that a significant glucose elevation occurs within the next
    two hours (from the trained model, never from the LLM),
  * a risk band and an honest, decomposed confidence figure,
  * an attribution panel explaining *why* the risk has that value,
  * a projected glucose trajectory with uncertainty bands calibrated on the
    patient's own history,
  * a data-quality assessment, and
  * — only once the replay has advanced far enough — the observed outcome.

The trajectory projection is a separate, deliberately simple statistical model
(damped-trend extrapolation plus this patient's own post-prandial response).  It
is *not* the classifier: the classifier answers "how likely is an event", the
projector answers "what shape might the trace take".  Keeping them apart avoids
presenting a visual extrapolation as if it were the model's probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backend.settings import settings
from backend.services import feature_engineering as fe
from backend.services.patient_service import PatientRecord

HORIZON_STEPS = settings.horizon_steps
# The fan must be tight close to the anchor and widen with the horizon, so the
# grid starts at one sampling interval rather than at 15 minutes.
TAU_GRID = [5, 10, 15, 30, 45, 60, 75, 90, 105, 120]


# ---------------------------------------------------------------------------
# Banding
# ---------------------------------------------------------------------------
def risk_band(probability: float) -> Dict[str, Any]:
    """Map a probability onto the LOW / MODERATE / HIGH clinical bands."""
    p = float(probability)
    if p >= settings.risk_high_min:
        band, label = "high", "HIGH"
    elif p >= settings.risk_moderate_min:
        band, label = "moderate", "MODERATE"
    else:
        band, label = "low", "LOW"
    bands = []
    for key, name in (("low", "LOW"), ("moderate", "MODERATE"), ("high", "HIGH")):
        lo, hi = settings.risk_bands[key]
        bands.append({"key": key, "label": name, "range": [round(lo, 2), round(hi, 2)], "active": key == band})
    return {
        "band": band,
        "label": label,
        "probability": round(p, 4),
        "percentage": round(p * 100, 1),
        "bands": bands,
        "above_decision_threshold": bool(p >= settings.risk_moderate_min),
    }


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------
def assess_data_quality(record: PatientRecord, index: int) -> Dict[str, Any]:
    """
    How much trustworthy information is behind this prediction?

    Reported per source so the UI can show a completeness figure instead of
    silently scoring on stale or absent signals.
    """
    frame = record.frame
    index = record.clip_index(index)
    now = record.timestamp_at(index)
    interval = pd.Timedelta(minutes=settings.sampling_interval_minutes)
    recent = frame.iloc[max(0, index - 11) : index + 1]

    def freshness(column: str, tolerance_minutes: float) -> Tuple[bool, Optional[float]]:
        series = recent[column].dropna()
        if series.empty:
            return False, None
        last_ts = recent.loc[series.index[-1], "timestamp"]
        age = (now - last_ts).total_seconds() / 60.0
        return age <= tolerance_minutes, round(age, 1)

    checks: List[Dict[str, Any]] = []
    cgm_ok, cgm_age = freshness("glucose_mgdl", 15)
    checks.append({"source": "Continuous glucose monitor", "key": "cgm", "present": bool(cgm_ok), "age_minutes": cgm_age, "required": True})
    hr_ok, hr_age = freshness("heart_rate_bpm", 30)
    checks.append({"source": "Heart rate (wearable)", "key": "hr", "present": bool(hr_ok), "age_minutes": hr_age, "required": True})
    hrv_ok, hrv_age = freshness("hrv_rmssd_ms", 45)
    checks.append({"source": "HRV (wearable)", "key": "hrv", "present": bool(hrv_ok), "age_minutes": hrv_age, "required": True})
    steps_ok, steps_age = freshness("steps_5min", 45)
    checks.append({"source": "Activity / steps", "key": "activity", "present": bool(steps_ok), "age_minutes": steps_age, "required": True})
    spo2_ok, spo2_age = freshness("spo2_pct", 60)
    checks.append({"source": "SpO₂", "key": "spo2", "present": bool(spo2_ok), "age_minutes": spo2_age, "required": False})

    todays_meals = [m for m in record.stream.meals if pd.Timestamp(m["datetime"]).date() == now.date()]
    meal_logged = any(m.get("logged_in_app", True) and pd.Timestamp(m["datetime"]) <= now for m in todays_meals)
    checks.append({"source": "Meal log", "key": "meal_log", "present": bool(meal_logged), "age_minutes": None, "required": False})

    last_night = most_recent_night(record, index)
    checks.append({"source": "Sleep session", "key": "sleep", "present": last_night is not None, "age_minutes": None, "required": False})

    required = [c for c in checks if c["required"]]
    required_ok = sum(1 for c in required if c["present"])
    optional = [c for c in checks if not c["required"]]
    optional_ok = sum(1 for c in optional if c["present"])
    completeness = (0.75 * required_ok / max(len(required), 1)) + (0.25 * optional_ok / max(len(optional), 1))

    issues = [c["source"] + (" unavailable" if not c["present"] else f" stale ({c['age_minutes']} min old)")
              for c in checks if not c["present"] or (c["age_minutes"] is not None and c["age_minutes"] > 20)]

    # implausible values are surfaced rather than silently used
    latest = frame.iloc[index]
    implausible: List[str] = []
    if not (39 <= float(latest["glucose_mgdl"]) <= 500):
        implausible.append("glucose outside physiological range")
    if not (30 <= float(latest["heart_rate_bpm"]) <= 220):
        implausible.append("heart rate outside physiological range")
    if not (70 <= float(latest["spo2_pct"]) <= 100):
        implausible.append("SpO₂ outside physiological range")

    return {
        "completeness": round(float(completeness), 3),
        "checks": checks,
        "issues": issues + implausible,
        "implausible_values": implausible,
        "onboarding_complete": bool(index * settings.sampling_interval_minutes / 60.0 >= settings.baseline_onboarding_hours),
        "baseline_window_hours": record.baseline.get("window_hours"),
        "baseline_samples": record.baseline.get("samples"),
    }


def most_recent_night(record: PatientRecord, index: int) -> Optional[Dict[str, Any]]:
    """Most recent completed sleep session at or before ``index``."""
    now = record.timestamp_at(index)
    candidates = []
    for night in record.stream.sleep_nights:
        wake = _night_wake_timestamp(record, night)
        if wake is not None and wake <= now:
            candidates.append((wake, night))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _night_wake_timestamp(record: PatientRecord, night: Dict[str, Any]) -> Optional[pd.Timestamp]:
    wake_dt = night.get("wake_dt")
    if wake_dt is not None:
        return pd.Timestamp(wake_dt)
    try:
        base = pd.Timestamp(night["night_of"]) + pd.Timedelta(days=1)
        return base + pd.Timedelta(hours=_hours_from_clock(night.get("wake_time", "06:00")))
    except Exception:
        return None


def _hours_from_clock(clock: str) -> float:
    hh, mm = str(clock).split(":")[:2]
    return int(hh) + int(mm) / 60.0


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------
def compute_confidence(
    probability: float,
    secondary_probability: float,
    quality: Dict[str, Any],
    predictor_info: Optional[Dict[str, Any]] = None,
    calibration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    A decomposed, deliberately conservative confidence figure.

    Three components, all measurable:

      * ``data_completeness``  — how much of the required signal is present and fresh
      * ``model_agreement``    — 1 - |p_primary - p_secondary| normalised; two
        independently trained models disagreeing is evidence the input is
        ambiguous
      * ``calibration_trust``  — 1 - the empirical calibration error in the
        probability band this prediction falls into, measured on held-out
        patients during training

    This is *not* a statement of medical certainty and the UI says so.
    """
    completeness = float(quality.get("completeness", 0.0))
    disagreement = abs(float(probability) - float(secondary_probability))
    agreement = float(np.clip(1.0 - disagreement / 0.35, 0.0, 1.0))

    calibration_error = _calibration_error_at(probability, calibration)
    calibration_trust = float(np.clip(1.0 - calibration_error / 0.25, 0.0, 1.0))

    score = 100.0 * (0.40 * completeness + 0.35 * agreement + 0.25 * calibration_trust)
    return {
        "score": round(float(score), 1),
        "components": {
            "data_completeness": round(completeness, 3),
            "model_agreement": round(agreement, 3),
            "calibration_trust": round(calibration_trust, 3),
        },
        "weights": {"data_completeness": 0.40, "model_agreement": 0.35, "calibration_trust": 0.25},
        "model_disagreement": round(disagreement, 4),
        "calibration_error_in_band": round(calibration_error, 4),
        "interpretation": (
            "Composite of signal completeness, agreement between two independently trained models, "
            "and measured calibration error in this probability band. It expresses how much the "
            "prediction can be trusted as a number — not medical certainty."
        ),
    }


def _calibration_error_at(probability: float, calibration: Optional[Dict[str, Any]]) -> float:
    if not calibration:
        return 0.05
    curve = (calibration.get("test") or calibration.get("curve") or {})
    grid = curve.get("curve", curve) if isinstance(curve, dict) else {}
    predicted = grid.get("mean_predicted") or []
    observed = grid.get("observed_frequency") or []
    pairs = [(p, o) for p, o in zip(predicted, observed) if p is not None and o is not None]
    if not pairs:
        return 0.05
    pairs.sort(key=lambda item: item[0])
    xs = np.array([p[0] for p in pairs])
    errs = np.array([abs(p[1] - p[0]) for p in pairs])
    return float(np.interp(probability, xs, errs))


# ---------------------------------------------------------------------------
# Trajectory projection
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryProjector:
    """Damped-trend + personal post-prandial projection with an empirical fan."""

    record: PatientRecord
    damping_minutes: float = 75.0
    rise_per_gram: float = 0.5
    peak_time_h: float = 1.25
    shape: float = 2.0
    fan: Dict[int, Dict[str, float]] = None  # type: ignore[assignment]
    postprandial_observations: int = 0

    def __post_init__(self) -> None:
        self._fit_postprandial()
        self.fan = self._calibrate_fan()

    # -- personal post-prandial response ---------------------------------
    def _fit_postprandial(self) -> None:
        frame = self.record.frame
        cutoff = frame["timestamp"].iloc[0] + pd.Timedelta(hours=float(settings.baseline_onboarding_hours))
        window = frame[frame["timestamp"] < cutoff]
        if window.empty:
            return
        rises: List[float] = []
        peaks: List[float] = []
        per_gram: List[float] = []
        early_ratios: List[float] = []
        for meal in self.record.stream.meals:
            meal_ts = pd.Timestamp(meal["datetime"])
            if meal_ts >= cutoff or not meal.get("logged_in_app", True):
                continue
            pre = window[(window["timestamp"] >= meal_ts - pd.Timedelta(minutes=30)) & (window["timestamp"] < meal_ts)]
            post = window[(window["timestamp"] >= meal_ts) & (window["timestamp"] <= meal_ts + pd.Timedelta(minutes=180))]
            if pre.empty or post.empty or len(post) < 6:
                continue
            base = float(pre["glucose_mgdl"].mean())
            peak_series = post["glucose_mgdl"].astype(float)
            peak = float(peak_series.max())
            peak_idx = int(peak_series.values.argmax())
            peak_rise = peak - base
            rises.append(peak_rise)
            peaks.append(peak_idx * settings.sampling_interval_minutes / 60.0)
            carbs = float(meal.get("carbs_g") or 0)
            if carbs > 10:
                per_gram.append(peak_rise / carbs)
            # how much of the peak rise has already appeared 30 minutes in —
            # this is what sets the steepness of the projected absorption curve
            at_30 = post[post["timestamp"] <= meal_ts + pd.Timedelta(minutes=30)]
            if peak_rise > 12 and not at_30.empty:
                early_ratios.append(float(np.clip((float(at_30["glucose_mgdl"].iloc[-1]) - base) / peak_rise, 0.05, 0.98)))
        if per_gram:
            self.rise_per_gram = float(np.clip(np.median(per_gram), 0.20, 1.40))
            self.postprandial_observations = len(per_gram)
        if peaks:
            self.peak_time_h = float(np.clip(np.median(peaks), 0.6, 2.6))
        if early_ratios:
            # invert the gamma shape at 30 minutes to recover the steepness
            ratio = float(np.median(early_ratios))
            x = 0.5 / self.peak_time_h
            denominator = (1.0 - x) + np.log(max(x, 1e-3))
            if abs(denominator) > 1e-3:
                self.shape = float(np.clip(np.log(ratio) / denominator, 1.0, 6.0))

    # -- empirical uncertainty fan ---------------------------------------
    def _calibrate_fan(self) -> Dict[int, Dict[str, float]]:
        """
        Back-test the projector inside the onboarding window and keep the 10th,
        50th and 90th percentiles of its error at each horizon.  The fan drawn
        on the chart is therefore this patient's own historical forecast error,
        not an arbitrary band.
        """
        frame = self.record.frame
        cutoff_index = int(settings.baseline_onboarding_hours * 60 / settings.sampling_interval_minutes)
        cutoff_index = min(cutoff_index, len(frame) - HORIZON_STEPS - 2)
        fan: Dict[int, Dict[str, float]] = {}
        if cutoff_index < 24:
            return {tau: {"q10": -18.0, "q50": 0.0, "q90": 18.0} for tau in TAU_GRID}

        glucose = frame["glucose_mgdl"].astype(float).to_numpy()
        slopes = self.record.features["glucose_slope_60"].to_numpy(dtype=float)
        indices = np.arange(12, cutoff_index, 3)
        meal_matrix = self._meal_increment_matrix(indices)

        for column, tau in enumerate(TAU_GRID):
            steps = int(tau / settings.sampling_interval_minutes)
            valid = indices + steps < len(glucose)
            idx = indices[valid]
            if len(idx) < 8:
                fan[tau] = {"q10": -18.0, "q50": 0.0, "q90": 18.0}
                continue
            damped = self.damping_minutes * (1.0 - np.exp(-tau / self.damping_minutes))
            projection = glucose[idx] + slopes[idx] * damped + meal_matrix[valid, column]
            actual = glucose[idx + steps]
            errors = actual - projection
            fan[tau] = {
                "q10": round(float(np.percentile(errors, 10)), 2),
                "q50": round(float(np.percentile(errors, 50)), 2),
                "q90": round(float(np.percentile(errors, 90)), 2),
                "rmse": round(float(np.sqrt(np.mean(errors**2))), 2),
                "n": int(len(errors)),
            }
        # Enforce monotone widening: an uncertainty band must never be tighter at
        # a longer horizon than at a shorter one.
        for key in ("q10", "q90"):
            running = fan[sorted(fan.keys())[0]][key]
            for tau in sorted(fan.keys()):
                if key == "q10":
                    running = min(running, fan[tau][key])
                else:
                    running = max(running, fan[tau][key])
                fan[tau][key] = round(running, 2)
        return fan

    def _habitual_meals(self) -> List[Dict[str, float]]:
        habitual = self.record.baseline.get("habitual_meals") or []
        return [m for m in habitual if isinstance(m, dict)]

    def _meal_increment(self, at: pd.Timestamp, tau_minutes: float) -> float:
        """
        Expected *change* in post-prandial glucose between ``at`` and ``at + tau``.

        The current reading already contains whatever a previous meal contributed,
        so the increment is the difference of the gamma response at the two ends,
        not the response at the far end.  Adding the full response would
        double-count meals that have already been absorbed.
        """
        total = 0.0
        for meal in self._habitual_meals():
            centre = float(meal["hour"])
            now_hours = at.hour + at.minute / 60.0 + at.second / 3600.0
            target_hours = now_hours + tau_minutes / 60.0
            # nearest occurrence of the habitual meal time (may be "yesterday")
            offset = centre - now_hours
            while offset < -12.0:
                offset += 24.0
            while offset > 12.0:
                offset -= 24.0
            since_now = -offset                     # hours since the meal (negative = still to come)
            since_target = since_now + tau_minutes / 60.0
            amplitude = float(meal.get("carbs_g", 60)) * self.rise_per_gram
            total += amplitude * (float(self._gamma(since_target)) - float(self._gamma(since_now)))
        return float(total)

    def _meal_increment_matrix(self, indices: np.ndarray) -> np.ndarray:
        """Vectorised version of :meth:`_meal_increment` for the back-test."""
        timestamps = self.record.frame["timestamp"].iloc[indices]
        hours = timestamps.dt.hour.to_numpy() + timestamps.dt.minute.to_numpy() / 60.0
        habitual = self._habitual_meals()
        out = np.zeros((len(indices), len(TAU_GRID)))
        if not habitual:
            return out
        for column, tau in enumerate(TAU_GRID):
            for meal in habitual:
                centre = float(meal["hour"])
                # hours since the nearest occurrence of this habitual meal time
                offset = centre - hours
                offset = np.where(offset < -12.0, offset + 24.0, offset)
                offset = np.where(offset > 12.0, offset - 24.0, offset)
                since_now = -offset
                since_target = since_now + tau / 60.0
                amplitude = float(meal.get("carbs_g", 60)) * self.rise_per_gram
                out[:, column] += amplitude * (self._gamma(since_target) - self._gamma(since_now))
        return out

    def _gamma(self, hours_since) -> Any:
        """
        Normalised post-prandial response curve (peaks at 1.0 at ``peak_time_h``).

        Negative arguments mean "the meal has not happened yet" and must give
        exactly zero.  The clip is applied before the power so numpy never sees a
        negative base with a fractional exponent (which would return NaN).
        """
        raw = np.asarray(hours_since, dtype=float)
        x = np.maximum(raw, 0.0)
        tp = max(self.peak_time_h, 0.2)
        value = (x / tp) ** self.shape * np.exp(self.shape * (1.0 - x / tp))
        return np.where(raw > 0, value, 0.0)

    # -- public ----------------------------------------------------------
    def project(self, index: int, horizon_minutes: Optional[int] = None, step_minutes: Optional[int] = None) -> Dict[str, Any]:
        horizon = int(horizon_minutes or settings.forecast_horizon_minutes)
        step = int(step_minutes or settings.sampling_interval_minutes)
        index = self.record.clip_index(index)
        row = self.record.feature_row(index)
        glucose_now = float(self.record.frame["glucose_mgdl"].iloc[index])
        slope = float(row["glucose_slope_60"])
        anchor = self.record.timestamp_at(index)

        points: List[Dict[str, Any]] = []
        taus = list(range(step, horizon + 1, step))
        for tau in taus:
            damped = self.damping_minutes * (1.0 - np.exp(-tau / self.damping_minutes))
            meal_term = self._meal_increment(anchor, tau)
            centre = glucose_now + slope * damped + meal_term
            band = self._band_for(tau)
            points.append(
                {
                    "minutes_ahead": tau,
                    "timestamp": (anchor + pd.Timedelta(minutes=tau)).isoformat(),
                    "p10": round(centre + band["q10"], 1),
                    "p50": round(centre + band["q50"], 1),
                    "p90": round(centre + band["q90"], 1),
                }
            )
        p50 = np.array([p["p50"] for p in points])
        p90 = np.array([p["p90"] for p in points])
        return {
            "method": (
                "damped-trend extrapolation of the 60-minute glucose slope plus this patient's own "
                "post-prandial response to their habitual meal times"
            ),
            "anchor_timestamp": anchor.isoformat(),
            "anchor_glucose_mgdl": round(glucose_now, 1),
            "slope_mgdl_per_min": round(slope, 4),
            "damping_minutes": self.damping_minutes,
            "personal_postprandial_response": {
                "rise_per_gram_carbohydrate": round(self.rise_per_gram, 3),
                "time_to_peak_h": round(self.peak_time_h, 2),
                "absorption_shape": round(self.shape, 2),
                "observations": self.postprandial_observations,
                "source": "fitted to meals logged inside this patient's onboarding window",
            },
            "habitual_meals": self._habitual_meals(),
            "points": points,
            "projected_peak_p50": round(float(p50.max()), 1) if len(p50) else round(glucose_now, 1),
            "projected_peak_p90": round(float(p90.max()), 1) if len(p90) else round(glucose_now, 1),
            "probability_band_crosses_threshold": bool(p90.max() >= settings.glucose_high_threshold_mgdl) if len(p90) else False,
            "uncertainty_source": "10th/50th/90th percentile of this projector's own back-test error inside the onboarding window",
        }

    def _band_for(self, tau: int) -> Dict[str, float]:
        if not self.fan:
            return {"q10": -18.0, "q50": 0.0, "q90": 18.0}
        keys = sorted(self.fan.keys())
        if tau <= keys[0]:
            return self.fan[keys[0]]
        if tau >= keys[-1]:
            return self.fan[keys[-1]]
        upper = next(k for k in keys if k >= tau)
        lower = keys[keys.index(upper) - 1]
        weight = (tau - lower) / max(upper - lower, 1)
        lo, hi = self.fan[lower], self.fan[upper]
        return {q: lo[q] + weight * (hi[q] - lo[q]) for q in ("q10", "q50", "q90")}


_PROJECTORS: Dict[str, TrajectoryProjector] = {}


def get_projector(record: PatientRecord) -> TrajectoryProjector:
    key = f"{record.patient_id}:{record.built_at}"
    if key not in _PROJECTORS:
        _PROJECTORS.clear()
        _PROJECTORS[key] = TrajectoryProjector(record)
    return _PROJECTORS[key]


# ---------------------------------------------------------------------------
# Risk-conditional trajectory (estimated from the training cohort)
# ---------------------------------------------------------------------------
def conditional_trajectory(
    probability: float, predictor, horizon_minutes: Optional[int] = None, anchor: Optional[pd.Timestamp] = None
) -> Optional[Dict[str, Any]]:
    """
    What does glucose *usually do next* at this level of predicted risk?

    The answer is read from an empirical distribution estimated on held-out
    patients during training: for each band of predicted probability we stored
    the 10th / 50th / 90th percentile of the realised glucose change at every
    horizon up to two hours.  Interpolating that distribution at the current
    probability gives a forecast fan that reflects the model's own risk level
    instead of a hand-drawn extrapolation.
    """
    if predictor is None:
        return None
    data = predictor.bundle.get("conditional_trajectory") or {}
    bins = [b for b in data.get("bins", []) if b.get("sufficient") and b.get("delta")]
    if not bins:
        return None
    horizon = int(horizon_minutes or settings.forecast_horizon_minutes)

    centres = np.array([float(np.mean(b["risk_range"])) for b in bins])
    order = np.argsort(centres)
    centres = centres[order]
    bins = [bins[i] for i in order]

    matched_index = int(np.argmin(np.abs(centres - probability)))
    taus = sorted({int(t) for b in bins for t in b["delta"].keys()})
    taus = [t for t in taus if t <= horizon]

    points: List[Dict[str, Any]] = []
    for tau in taus:
        xs, q10s, q50s, q90s = [], [], [], []
        for centre, b in zip(centres, bins):
            entry = b["delta"].get(str(tau))
            if not entry:
                continue
            xs.append(centre)
            q10s.append(entry["q10"])
            q50s.append(entry["q50"])
            q90s.append(entry["q90"])
        if not xs:
            continue
        points.append(
            {
                "minutes_ahead": tau,
                "timestamp": (anchor + pd.Timedelta(minutes=tau)).isoformat() if anchor is not None else None,
                "delta_q10": round(float(np.interp(probability, xs, q10s)), 1),
                "delta_q50": round(float(np.interp(probability, xs, q50s)), 1),
                "delta_q90": round(float(np.interp(probability, xs, q90s)), 1),
            }
        )
    matched = bins[matched_index]
    return {
        "source": data.get("source"),
        "note": data.get("note"),
        "matched_risk_band": matched["risk_range"],
        "matched_samples": matched["n_samples"],
        "matched_observed_event_rate": matched.get("observed_event_rate"),
        "matched_mean_predicted_risk": matched.get("mean_predicted_risk"),
        "probability": round(float(probability), 4),
        "interpolation": "linear between band centres of the empirical distribution",
        "delta_points": points,
    }


def _build_trajectory(record: PatientRecord, index: int, probability: float, predictor=None) -> Dict[str, Any]:
    """
    Two clearly separated forecasts:

      * ``conditional``       — the cohort-estimated distribution of glucose
        change given this level of predicted risk (the band drawn on the chart),
      * ``trend_continuation`` — this patient's own damped-trend + post-prandial
        extrapolation, shown as a reference line.

    Neither is the classifier's probability; both are derived from data.
    """
    anchor = record.timestamp_at(index)
    glucose_now = float(record.frame["glucose_mgdl"].iloc[index])
    projection = get_projector(record).project(index)
    conditional = conditional_trajectory(probability, predictor, anchor=anchor)

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
                    "basis": "risk-conditional",
                }
            )
    if not points:
        points = [dict(point, basis="trend-continuation") for point in projection["points"]]

    p50 = np.array([p["p50"] for p in points])
    p90 = np.array([p["p90"] for p in points])
    return {
        "anchor_timestamp": anchor.isoformat(),
        "anchor_glucose_mgdl": round(glucose_now, 1),
        "threshold_mgdl": settings.glucose_high_threshold_mgdl,
        "horizon_minutes": settings.forecast_horizon_minutes,
        "points": points,
        "primary_basis": "risk-conditional" if conditional else "trend-continuation",
        "conditional": conditional,
        "trend_continuation": projection,
        "projected_peak_p50": round(float(p50.max()), 1) if len(p50) else round(glucose_now, 1),
        "projected_peak_p90": round(float(p90.max()), 1) if len(p90) else round(glucose_now, 1),
        "probability_band_crosses_threshold": bool(p90.max() >= settings.glucose_high_threshold_mgdl) if len(p90) else False,
        "reading_note": (
            "The shaded band is the observed distribution of glucose change in held-out patients with a similar "
            "predicted risk. The dashed reference line is this patient's own trend and post-prandial response "
            "extrapolated forward. Neither line is the model probability."
        ),
    }


# ---------------------------------------------------------------------------
# Observed outcome (only revealed once the replay has moved past it)
# ---------------------------------------------------------------------------
def observed_outcome(record: PatientRecord, index: int) -> Dict[str, Any]:
    """What actually happened in the following two hours of the recorded stream."""
    index = record.clip_index(index)
    end = min(index + HORIZON_STEPS, len(record.frame) - 1)
    available_steps = end - index
    if available_steps <= 0:
        return {"available": False, "reason": "the replay has not reached this point yet"}
    future = record.frame["glucose_mgdl"].iloc[index + 1 : end + 1].astype(float)
    current = float(record.frame["glucose_mgdl"].iloc[index])
    peak = float(future.max())
    peak_offset = int(future.values.argmax()) + 1
    label = int(record.labels[index]) if record.labels[index] >= 0 else None
    crossing = future[future >= settings.glucose_high_threshold_mgdl]
    return {
        "available": available_steps >= HORIZON_STEPS,
        "partial": available_steps < HORIZON_STEPS,
        "horizon_minutes": settings.forecast_horizon_minutes,
        "minutes_observed": int(available_steps * settings.sampling_interval_minutes),
        "current_glucose_mgdl": round(current, 1),
        "peak_glucose_mgdl": round(peak, 1),
        "peak_in_minutes": int(peak_offset * settings.sampling_interval_minutes),
        "rise_mgdl": round(peak - current, 1),
        "crossed_threshold": bool(len(crossing) > 0),
        "minutes_to_threshold": (
            int((crossing.index[0] - (index + 1)) * settings.sampling_interval_minutes) if len(crossing) else None
        ),
        "event_occurred": label,
        "timestamp_peak": record.timestamp_at(index + peak_offset).isoformat(),
    }


# ---------------------------------------------------------------------------
# Full prediction payload
# ---------------------------------------------------------------------------
def build_prediction(
    record: PatientRecord,
    index: int,
    predictor=None,
    explain: bool = True,
    reveal_outcome: bool = False,
) -> Dict[str, Any]:
    """Assemble everything the Prediction view needs for one instant."""
    index = record.clip_index(index)
    features = record.feature_row(index)
    probability = float(record.risk_primary[index])
    secondary = float(record.risk_secondary[index])
    anchor = record.timestamp_at(index)

    quality = assess_data_quality(record, index)
    calibration = None
    predictor_info = None
    attribution = None
    if predictor is not None:
        try:
            predictor_info = predictor.info()
            calibration = predictor.metrics().get("test", {}).get("calibration")
        except Exception:
            predictor_info = None
        if explain:
            try:
                attribution = predictor.explain(features, top_k=8)
            except Exception as exc:  # pragma: no cover
                attribution = {"error": str(exc)}

    confidence = compute_confidence(probability, secondary, quality, predictor_info, calibration)
    band = risk_band(probability)

    # risk history for the trend sparkline / "why did it change" panel
    window_start = max(0, index - int(180 / settings.sampling_interval_minutes))
    risk_history = [
        {
            "timestamp": record.timestamp_at(i).isoformat(),
            "risk": round(float(record.risk_primary[i]), 4),
        }
        for i in range(window_start, index + 1, max(1, int(15 / settings.sampling_interval_minutes)))
    ]
    risk_change_60 = float(probability - record.risk_primary[max(0, index - int(60 / settings.sampling_interval_minutes))])
    risk_change_180 = float(probability - record.risk_primary[max(0, index - int(180 / settings.sampling_interval_minutes))])

    payload: Dict[str, Any] = {
        "patient_id": record.patient_id,
        "generated_at": anchor.isoformat(),
        "generated_at_clock": anchor.strftime("%H:%M:%S"),
        "forecast_horizon_minutes": settings.forecast_horizon_minutes,
        "horizon_end": (anchor + pd.Timedelta(minutes=settings.forecast_horizon_minutes)).isoformat(),
        "probability": round(probability, 4),
        "risk": band,
        "decision_threshold": float(predictor.threshold) if predictor is not None else settings.risk_moderate_min,
        "confidence": confidence,
        "secondary_probability": round(secondary, 4),
        "risk_trend": {
            "history": risk_history,
            "change_60min": round(risk_change_60, 4),
            "change_180min": round(risk_change_180, 4),
            "direction": "rising" if risk_change_60 > 0.02 else ("falling" if risk_change_60 < -0.02 else "stable"),
        },
        "event_definition": {
            "description": (
                f"Glucose reaches ≥ {int(settings.glucose_high_threshold_mgdl)} mg/dL for at least 10 consecutive "
                f"minutes within the next {settings.forecast_horizon_minutes} minutes, at least "
                f"{int(settings.glucose_rise_delta_mgdl)} mg/dL above the current value."
            ),
            "high_threshold_mgdl": settings.glucose_high_threshold_mgdl,
            "rise_delta_mgdl": settings.glucose_rise_delta_mgdl,
            "horizon_minutes": settings.forecast_horizon_minutes,
        },
        "data_quality": quality,
        "model": predictor_info
        or {
            "model_id": "unavailable",
            "estimator": "heuristic fallback (model artifacts not trained)",
            "disclaimer": "Run python model/train.py to fit the real model.",
        },
        "attribution": attribution,
        "trajectory": _build_trajectory(record, index, probability, predictor),
    }
    if reveal_outcome:
        payload["observed_outcome"] = observed_outcome(record, index)
    return payload
