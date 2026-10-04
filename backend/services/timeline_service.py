"""
VitalSync — clinical timeline construction.

Every entry in the Clinical Timeline view is *derived from the recorded signal*,
never hardcoded in the UI.  The service walks the stream up to the current twin
instant and emits events of three kinds:

  record events    meals logged, sleep periods, activity blocks
  signal events    threshold crossings, sustained trends, baseline deviations
  model events     risk-band transitions, alert issuance, twin state updates

Detections are debounced (a state must persist for a minimum duration before it
becomes an event, and cannot re-fire inside a refractory period) so the timeline
reads like a clinical log rather than a trace of sensor noise.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.settings import settings
from backend.services import feature_engineering as fe
from backend.services.patient_service import PatientRecord

CATEGORY_LABELS = {
    "meal": "Nutrition",
    "sleep": "Sleep",
    "activity": "Activity",
    "glucose": "Glucose",
    "cardio": "Cardiovascular",
    "model": "Model",
    "alert": "Alert",
    "twin": "Digital Twin",
    "outcome": "Outcome",
}

SEVERITY_ORDER = {"info": 0, "good": 1, "watch": 2, "warning": 3, "critical": 4}


def _event(
    record: PatientRecord,
    index: int,
    category: str,
    title: str,
    detail: str,
    severity: str = "info",
    source: str = "sensor",
    metrics: Optional[Dict[str, Any]] = None,
    timestamp: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    ts = timestamp if timestamp is not None else record.timestamp_at(index)
    return {
        "timestamp": ts.isoformat(),
        "clock": ts.strftime("%H:%M"),
        "date": ts.strftime("%a %d %b"),
        "category": category,
        "category_label": CATEGORY_LABELS.get(category, category.title()),
        "title": title,
        "detail": detail,
        "severity": severity,
        "severity_rank": SEVERITY_ORDER.get(severity, 0),
        "source": source,
        "metrics": metrics or {},
        "index": int(index),
    }


def build_timeline(
    record: PatientRecord,
    index: int,
    hours: float = 8.0,
    include_model_events: bool = True,
) -> Dict[str, Any]:
    """Build the clinical timeline up to (and including) ``index``."""
    index = record.clip_index(index)
    frame = record.frame
    anchor = record.timestamp_at(index)
    start = anchor - pd.Timedelta(hours=hours)
    start_index = max(0, record.index_at(start))

    events: List[Dict[str, Any]] = []
    events.extend(_record_events(record, start, anchor))
    events.extend(_signal_events(record, start_index, index, start))
    if include_model_events:
        events.extend(_model_events(record, start_index, index, start))

    events.sort(key=lambda e: (e["timestamp"], -e["severity_rank"], e["category"]))
    for position, event in enumerate(events):
        event["sequence"] = position + 1

    counts: Dict[str, int] = {}
    for event in events:
        counts[event["category"]] = counts.get(event["category"], 0) + 1

    highest = max((SEVERITY_ORDER.get(e["severity"], 0) for e in events), default=0)
    return {
        "patient_id": record.patient_id,
        "as_of": anchor.isoformat(),
        "window": {"start": start.isoformat(), "end": anchor.isoformat(), "hours": hours},
        "events": events,
        "event_count": len(events),
        "counts_by_category": counts,
        "highest_severity": _severity_name(highest),
        "note": (
            "Events are detected from the recorded signal and the model's own output history. "
            "Nothing in this timeline is authored in the interface."
        ),
    }


def _severity_name(rank: int) -> str:
    for name, value in SEVERITY_ORDER.items():
        if value == rank:
            return name
    return "info"


# ---------------------------------------------------------------------------
# Record-derived events
# ---------------------------------------------------------------------------
def _record_events(record: PatientRecord, start: pd.Timestamp, end: pd.Timestamp) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    for meal in record.stream.meals:
        ts = pd.Timestamp(meal["datetime"])
        if not (start <= ts <= end):
            continue
        logged = meal.get("logged_in_app", True)
        events.append(
            _event(
                record,
                record.index_at(ts),
                "meal",
                meal.get("label", "Meal"),
                (
                    f"{meal.get('carbs_g', 0)} g carbohydrate logged in the patient app at {ts.strftime('%H:%M')}."
                    if logged
                    else f"Meal detected in the record but not logged by the patient (~{meal.get('carbs_g', 0)} g carbohydrate)."
                ),
                severity="info" if logged else "watch",
                source="record",
                metrics={"carbs_g": meal.get("carbs_g"), "logged": logged, "label": meal.get("label")},
                timestamp=ts,
            )
        )

    for night in record.stream.sleep_nights:
        bed = _night_timestamp(record, night, "bed_time", "night_of")
        wake = _night_timestamp(record, night, "wake_time", "night_of", next_day=True)
        if wake is None:
            continue
        if start <= wake <= end:
            events.append(
                _event(
                    record,
                    record.index_at(wake),
                    "sleep",
                    f"Woke after {night['sleep_duration_h']} h of sleep",
                    (
                        f"In bed {night['bed_time']}–{night['wake_time']} "
                        f"({night['time_in_bed_h']} h), sleep efficiency {int(night['efficiency'] * 100)}%, "
                        f"{night['awakenings']} wake episodes, deep sleep {int(night['deep_fraction'] * 100)}%."
                    )
                    + (f" Note: {night['note']}" if night.get("note") else ""),
                    severity="warning" if night["sleep_duration_h"] < float(record.baseline.get("sleep_duration_median_h", 6.5)) - 0.8 else "info",
                    source="record",
                    metrics={
                        "duration_h": night["sleep_duration_h"],
                        "efficiency": night["efficiency"],
                        "awakenings": night["awakenings"],
                        "deep_fraction": night["deep_fraction"],
                        "mean_hrv_ms": night.get("mean_hrv_ms"),
                    },
                    timestamp=wake,
                )
            )
        if bed is not None and start <= bed <= end:
            events.append(
                _event(
                    record,
                    record.index_at(bed),
                    "sleep",
                    "Bedtime",
                    f"Lights out at {night['bed_time']}; wearable recorded {night['time_in_bed_h']} h in bed.",
                    severity="info",
                    source="record",
                    metrics={"bed_time": night["bed_time"]},
                    timestamp=bed,
                )
            )

    for block in record.stream.activity_blocks:
        block_ts = _activity_block_timestamp(record, block)
        if block_ts is None or not (start <= block_ts <= end):
            continue
        intensity = block.get("intensity", "activity")
        label = block.get("label") or intensity.title()
        events.append(
            _event(
                record,
                record.index_at(block_ts),
                "activity",
                label,
                f"{intensity.title()} activity block from {block['start']} to {block['end']}.",
                severity="warning" if intensity == "sedentary" else "good",
                source="record",
                metrics={"intensity": intensity, "start": block["start"], "end": block["end"]},
                timestamp=block_ts,
            )
        )
    return events


def _night_timestamp(
    record: PatientRecord, night: Dict[str, Any], key: str, date_key: str, next_day: bool = False
) -> Optional[pd.Timestamp]:
    try:
        base = pd.Timestamp(night[date_key]) + (pd.Timedelta(days=1) if next_day else pd.Timedelta(0))
        hh, mm = str(night[key]).split(":")[:2]
        return base + pd.Timedelta(hours=int(hh), minutes=int(mm))
    except Exception:
        return None


def _activity_block_timestamp(record: PatientRecord, block: Dict[str, Any]) -> Optional[pd.Timestamp]:
    try:
        day = record.stream.scenario_day + pd.Timedelta(days=int(block.get("day_offset", 0)))
        hh, mm = str(block["start"]).split(":")[:2]
        return pd.Timestamp(day) + pd.Timedelta(hours=int(hh), minutes=int(mm))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Signal-derived events
# ---------------------------------------------------------------------------
def _signal_events(record: PatientRecord, start_index: int, index: int, start: pd.Timestamp) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    frame = record.frame
    features = record.features
    glucose = frame["glucose_mgdl"].astype(float)
    hrv = features["hrv_current"].astype(float)
    slope30 = features["glucose_slope_30"].astype(float) * 60.0          # mg/dL per hour
    sedentary = features["sedentary_minutes_continuous"].astype(float)
    steps_60 = features["steps_60"].astype(float)

    hrv_baseline = float(record.baseline.get("hrv_daytime_median", 45.0))
    threshold = float(settings.glucose_high_threshold_mgdl)
    step = settings.sampling_interval_minutes
    debounce = max(2, int(15 / step))         # 15 minutes of persistence
    refractory = max(6, int(45 / step))       # no repeat inside 45 minutes

    def run_detector(condition: np.ndarray) -> List[int]:
        """Indices where a condition becomes true and stays true for `debounce`."""
        fires: List[int] = []
        last = -10**9
        run = 0
        for i in range(start_index, index + 1):
            if condition[i]:
                run += 1
                if run == debounce and i - last > refractory:
                    fires.append(i)
                    last = i
            else:
                run = 0
        return fires

    # glucose above range
    above = (glucose >= threshold).to_numpy()
    for i in run_detector(above):
        peak_window = glucose.iloc[i : min(i + int(60 / step), index + 1)]
        events.append(
            _event(
                record, i, "glucose",
                f"Glucose sustained above {int(threshold)} mg/dL",
                f"CGM held at or above {int(threshold)} mg/dL for {int(debounce * step)} minutes "
                f"(current {int(glucose.iloc[i])} mg/dL). Threshold-based observation, not a forecast.",
                severity="warning" if glucose.iloc[i] < 250 else "critical",
                source="sensor",
                metrics={"glucose_mgdl": int(glucose.iloc[i]), "threshold_mgdl": threshold,
                         "max_next_60min": int(peak_window.max()) if len(peak_window) else None},
            )
        )

    # return to range
    below = ~above
    for i in run_detector(below):
        if i > start_index and above[i - debounce]:
            events.append(
                _event(
                    record, i, "glucose",
                    "Glucose back within target range",
                    f"CGM below {int(threshold)} mg/dL for {int(debounce * step)} minutes ({int(glucose.iloc[i])} mg/dL).",
                    severity="good", source="sensor",
                    metrics={"glucose_mgdl": int(glucose.iloc[i])},
                )
            )

    # rising trend
    rising = (slope30 >= 12.0).to_numpy()
    for i in run_detector(rising):
        events.append(
            _event(
                record, i, "glucose",
                "Glucose begins rising",
                f"30-minute slope sustained at +{slope30[i]:.0f} mg/dL per hour; "
                f"glucose {int(glucose.iloc[i])} mg/dL and climbing.",
                severity="watch", source="sensor",
                metrics={"glucose_mgdl": int(glucose.iloc[i]), "slope_mgdl_per_h": round(float(slope30[i]), 1)},
            )
        )

    # HRV below personal baseline
    hrv_low = (hrv <= hrv_baseline * 0.90).to_numpy()
    for i in run_detector(hrv_low):
        deviation = 100.0 * (hrv[i] - hrv_baseline) / max(hrv_baseline, 1e-6)
        events.append(
            _event(
                record, i, "cardio",
                "HRV below personal baseline",
                f"30-minute mean RMSSD {hrv[i]:.0f} ms versus a personal daytime baseline of "
                f"{hrv_baseline:.0f} ms ({deviation:+.0f}%), sustained for {int(debounce * step)} minutes.",
                severity="watch", source="sensor",
                metrics={"hrv_ms": round(float(hrv[i]), 0), "baseline_ms": round(hrv_baseline, 0), "deviation_pct": round(deviation, 1)},
            )
        )

    # prolonged inactivity
    sedentary_long = (sedentary >= 90.0).to_numpy()
    for i in run_detector(sedentary_long):
        events.append(
            _event(
                record, i, "activity",
                f"Prolonged inactivity ({int(sedentary[i])} min)",
                f"No meaningful movement for {int(sedentary[i])} minutes; "
                f"{int(steps_60[i])} steps in the last hour.",
                severity="warning", source="sensor",
                metrics={"sedentary_minutes": int(sedentary[i]), "steps_60min": int(steps_60[i])},
            )
        )

    # observed outcome: the forecast target actually happening
    for i in run_detector(above):
        future_end = min(i + settings.horizon_steps, len(glucose) - 1)
        if future_end - i >= settings.horizon_steps:
            window = glucose.iloc[i : future_end + 1]
            peak_index = int(window.values.argmax())
            if record.labels[i] == 1:
                peak_ts = record.timestamp_at(i + peak_index)
                events.append(
                    _event(
                        record, i + peak_index, "outcome",
                        f"Significant elevation confirmed ({int(window.max())} mg/dL)",
                        f"The event the model was forecasting occurred at {peak_ts.strftime('%H:%M')}: "
                        f"glucose reached {int(window.max())} mg/dL, "
                        f"{int(window.max() - glucose.iloc[i])} mg/dL above the value at {record.timestamp_at(i).strftime('%H:%M')}.",
                        severity="critical", source="sensor",
                        metrics={"peak_mgdl": int(window.max()), "rise_mgdl": int(window.max() - glucose.iloc[i])},
                        timestamp=peak_ts,
                    )
                )
    return events


# ---------------------------------------------------------------------------
# Model-derived events
# ---------------------------------------------------------------------------
def _model_events(record: PatientRecord, start_index: int, index: int, start: pd.Timestamp) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    risk = record.risk_primary
    bands = [_band(risk[i]) for i in range(len(risk))]
    step = settings.sampling_interval_minutes
    refractory = max(3, int(20 / step))
    last_fire = -10**9

    for i in range(max(start_index, 1), index + 1):
        previous = bands[i - 1]
        current = bands[i]
        if previous == current or i - last_fire < refractory:
            continue
        last_fire = i
        rank = {"low": 0, "moderate": 1, "high": 2}
        escalated = rank[current] > rank[previous]
        if escalated and current == "high":
            title = f"2-hour glucose risk raised to HIGH ({risk[i] * 100:.0f}%)"
            detail = (
                f"Model output moved from {risk[i - 1] * 100:.0f}% to {risk[i] * 100:.0f}% "
                f"({previous} → {current}). Forecast horizon {settings.forecast_horizon_minutes} minutes."
            )
            severity = "critical"
        elif escalated:
            title = f"2-hour glucose risk rising ({risk[i] * 100:.0f}%, {current.upper()})"
            detail = f"Model output moved from {risk[i - 1] * 100:.0f}% to {risk[i] * 100:.0f}%."
            severity = "warning"
        else:
            title = f"2-hour glucose risk easing ({risk[i] * 100:.0f}%, {current.upper()})"
            detail = f"Model output moved from {risk[i - 1] * 100:.0f}% to {risk[i] * 100:.0f}%."
            severity = "good"
        events.append(
            _event(record, i, "model", title, detail, severity=severity, source="model",
                   metrics={"risk": round(float(risk[i]), 4), "previous_risk": round(float(risk[i - 1]), 4),
                            "band": current, "previous_band": previous})
        )

    # alert issuance: crossing the model's operating threshold upward
    threshold = float(settings.risk_moderate_min)
    last_alert = -10**9
    for i in range(max(start_index, 1), index + 1):
        if risk[i] >= threshold > risk[i - 1] and i - last_alert > max(6, int(60 / step)):
            last_alert = i
            events.append(
                _event(
                    record, i, "alert",
                    "Emerging risk detected",
                    f"Predicted probability of a significant glucose elevation within "
                    f"{settings.forecast_horizon_minutes // 60} h crossed the review threshold "
                    f"({risk[i] * 100:.0f}% ≥ {threshold * 100:.0f}%).",
                    severity="warning", source="model",
                    metrics={"risk": round(float(risk[i]), 4), "threshold": threshold},
                )
            )

    # twin synchronisation heartbeat (most recent update only)
    events.append(
        _event(
            record, index, "twin",
            "Digital Twin updated",
            f"Twin state recomputed at {record.timestamp_at(index).strftime('%H:%M:%S')} from "
            f"{len(fe.FEATURE_NAMES)} fused features "
            f"({sum(1 for s in fe.FEATURE_SPECS if s.group == 'historical')} historical, "
            f"{sum(1 for s in fe.FEATURE_SPECS if s.group != 'historical')} dynamic).",
            severity="info", source="twin",
            metrics={"feature_count": len(fe.FEATURE_NAMES), "risk": round(float(risk[index]), 4)},
        )
    )
    return events


def _band(probability: float) -> str:
    if probability >= settings.risk_high_min:
        return "high"
    if probability >= settings.risk_moderate_min:
        return "moderate"
    return "low"


def build_pattern_summary(record: PatientRecord, index: int, hours: float = 24.0) -> Dict[str, Any]:
    """
    Aggregate description of the last `hours`, used by the Groq
    "Review recent pattern" action and the overview panels.
    """
    index = record.clip_index(index)
    history = record.history(index, hours)
    if history.empty:
        return {"hours": hours, "samples": 0}
    glucose = history["glucose_mgdl"].astype(float)
    risk = record.risk_primary[record.index_at(history["timestamp"].iloc[0]) : index + 1]
    meals = [m for m in record.stream.meals if pd.Timestamp(m["datetime"]) >= history["timestamp"].iloc[0]]
    meals = [m for m in meals if pd.Timestamp(m["datetime"]) <= record.timestamp_at(index)]
    nights = [n for n in record.stream.sleep_nights if n.get("night_of")]
    last_night = nights[-1] if nights else None

    above = float((glucose > settings.glucose_high_threshold_mgdl).mean() * 100.0)
    return {
        "hours": hours,
        "samples": int(len(history)),
        "glucose": {
            "mean_mgdl": round(float(glucose.mean()), 1),
            "min_mgdl": int(glucose.min()),
            "max_mgdl": int(glucose.max()),
            "std_mgdl": round(float(glucose.std()), 1),
            "cv_pct": round(float(100.0 * glucose.std() / max(glucose.mean(), 1e-6)), 1),
            "time_above_180_pct": round(above, 1),
            "time_in_range_pct": round(float(((glucose >= 70) & (glucose <= 180)).mean() * 100.0), 1),
            "excursions_above_180": int((glucose.diff() > 0).sum() * 0 + _count_excursions(glucose.to_numpy())),
        },
        "risk": {
            "mean": round(float(np.mean(risk)), 4) if len(risk) else None,
            "max": round(float(np.max(risk)), 4) if len(risk) else None,
            "latest": round(float(risk[-1]), 4) if len(risk) else None,
            "share_of_time_high_pct": round(float((risk >= settings.risk_high_min).mean() * 100.0), 1) if len(risk) else None,
        },
        "activity": {
            "steps": int(history["steps_5min"].sum()),
            "mean_met": round(float(history["activity_met"].mean()), 2),
            "longest_sedentary_minutes": int(record.features["sedentary_minutes_continuous"].iloc[max(0, index - len(history)) : index + 1].max()),
        },
        "cardio": {
            "mean_hr": round(float(history["heart_rate_bpm"].mean()), 1),
            "mean_hrv_ms": round(float(history["hrv_rmssd_ms"].mean()), 1),
            "min_spo2": int(history["spo2_pct"].min()),
        },
        "meals_logged": [
            {"clock": m["clock"], "label": m["label"], "carbs_g": m["carbs_g"], "logged_in_app": m.get("logged_in_app", True)}
            for m in meals
        ],
        "last_sleep_night": last_night,
    }


def _count_excursions(glucose: np.ndarray, threshold: Optional[float] = None) -> int:
    threshold = threshold if threshold is not None else settings.glucose_high_threshold_mgdl
    above = glucose >= threshold
    count = 0
    run = 0
    for value in above:
        if value:
            run += 1
        else:
            if run >= 2:
                count += 1
            run = 0
    if run >= 2:
        count += 1
    return count
