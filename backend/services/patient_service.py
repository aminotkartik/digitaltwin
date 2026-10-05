"""
VitalSync — patient repository and twin materialisation.

On start-up this service:

  1. loads the synthetic patient roster (``backend/data/patients.json``),
  2. regenerates each patient's multi-day sensor stream with the deterministic
     physiology simulator and persists it to ``backend/data/sensor_data/`` so the
     exact bytes behind the demo are inspectable and citable,
  3. estimates the personal baseline from the onboarding window,
  4. builds the fused feature matrix and scores every instant with both trained
     models, so the dashboard can show a continuous risk history rather than a
     single point estimate.

Everything is cached in memory and rebuilt on demand — there is no hidden state
and no database, which keeps the prototype auditable.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.settings import SENSOR_DIR, settings
from backend.services import feature_engineering as fe
from backend.services.synthetic_data import GeneratedStream, generate_stream, label_events_fast


class PatientNotFoundError(KeyError):
    """Raised for an unknown patient identifier."""

    def __init__(self, patient_id: str):
        super().__init__(patient_id)
        self.patient_id = patient_id


PATIENTS_FILE = Path(__file__).resolve().parents[1] / "data" / "patients.json"


def _clock(clock: str) -> time:
    hh, mm = clock.split(":")[:2]
    return time(int(hh), int(mm))


@dataclass
class PatientRecord:
    """A materialised digital twin: history + signal + features + risk."""

    patient_id: str
    profile: Dict[str, Any]
    stream: GeneratedStream
    baseline: Dict[str, Any]
    features: pd.DataFrame
    ehr_static: Dict[str, float]
    risk_primary: np.ndarray
    risk_secondary: np.ndarray
    labels: np.ndarray
    now_index: int
    simulation_start_index: int
    last_index: int
    built_at: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")

    # ------------------------------------------------------------- helpers
    @property
    def frame(self) -> pd.DataFrame:
        return self.stream.frame

    @property
    def name(self) -> str:
        return str(self.profile.get("name", self.patient_id))

    @property
    def timestamps(self) -> pd.Series:
        return self.frame["timestamp"]

    def index_at(self, when: pd.Timestamp) -> int:
        """Nearest sample index at or before ``when``."""
        positions = np.where(self.timestamps.to_numpy() <= np.datetime64(when))[0]
        if len(positions) == 0:
            return 0
        return int(positions[-1])

    def timestamp_at(self, index: int) -> pd.Timestamp:
        index = int(np.clip(index, 0, len(self.frame) - 1))
        return self.timestamps.iloc[index]

    def clip_index(self, index: int) -> int:
        return int(np.clip(index, 0, len(self.frame) - 1))

    def feature_row(self, index: int) -> Dict[str, float]:
        index = self.clip_index(index)
        row = self.features.iloc[index]
        return {name: float(row[name]) for name in fe.FEATURE_NAMES}

    def sensor_snapshot(self, index: int) -> Dict[str, Any]:
        index = self.clip_index(index)
        return self.frame.iloc[index].to_dict()

    def history(self, index: int, hours: float) -> pd.DataFrame:
        index = self.clip_index(index)
        start = self.timestamp_at(index) - pd.Timedelta(hours=hours)
        window = self.frame.iloc[: index + 1]
        return window[window["timestamp"] >= start]

    def to_summary(self) -> Dict[str, Any]:
        ehr = self.profile.get("ehr", {})
        demographics = ehr.get("demographics", {})
        latest_bp = (ehr.get("vitals_history", {}).get("blood_pressure") or [{}])[0]
        return {
            "patient_id": self.patient_id,
            "name": self.name,
            "age": self.profile.get("age"),
            "sex": self.profile.get("sex"),
            "primary_condition": self.profile.get("primary_condition"),
            "bmi": demographics.get("bmi"),
            "is_demo_patient": bool(self.profile.get("is_demo_patient", False)),
            "latest_systolic": latest_bp.get("systolic"),
            "latest_diastolic": latest_bp.get("diastolic"),
            "current_risk": round(float(self.risk_primary[self.now_index]), 4),
            "current_glucose": int(self.frame["glucose_mgdl"].iloc[self.now_index]),
            "now": self.timestamp_at(self.now_index).isoformat(),
            "stream_start": self.stream.start.isoformat(),
            "stream_end": self.stream.end.isoformat(),
            "samples": int(len(self.frame)),
        }


class PatientRepository:
    """Loads and materialises every patient's digital twin."""

    def __init__(self, patients_file: Path = PATIENTS_FILE, rebuild_files: bool = True) -> None:
        self.patients_file = Path(patients_file)
        self._records: Dict[str, PatientRecord] = {}
        self._order: List[str] = []
        self._lock = threading.RLock()
        self._roster: Dict[str, Any] = {}
        self.rebuild_files = rebuild_files
        self.reload()

    # ------------------------------------------------------------------ load
    def reload(self) -> None:
        raw = json.loads(self.patients_file.read_text())
        self._roster = {k: v for k, v in raw.items() if k != "patients"}
        records: Dict[str, PatientRecord] = {}
        order: List[str] = []
        for profile in raw["patients"]:
            record = self._materialise(profile)
            records[record.patient_id] = record
            order.append(record.patient_id)
        with self._lock:
            self._records = records
            self._order = order

    def _materialise(self, profile: Dict[str, Any]) -> PatientRecord:
        patient_id = str(profile["patient_id"])
        # Anchored to settings.scenario_date (see DEMO_ANCHOR_DATE) so the demo
        # day — and therefore the committed provenance files — never drift.
        end_time = datetime.combine(settings.scenario_date, _clock(settings.demo_stream_end_clock))
        stream = generate_stream(
            patient_id=patient_id,
            physiology=profile.get("physiology", {}),
            scenario=profile.get("scenario", {}),
            end_time=end_time,
            hours=int(settings.stream_hours),
            interval_minutes=int(settings.sampling_interval_minutes),
        )
        frame = stream.frame
        ehr_static = fe.normalise_ehr(profile.get("ehr", {}))

        cutoff = frame["timestamp"].iloc[0] + pd.Timedelta(hours=float(settings.baseline_onboarding_hours))
        baseline_frame = frame[frame["timestamp"] < cutoff]
        baseline = fe.compute_personal_baseline(
            baseline_frame, stream.sleep_nights, ehr_static, meals=stream.meals
        )
        features = fe.build_feature_frame(
            frame, stream.meals, stream.sleep_nights, ehr_static, baseline, settings.sampling_interval_minutes
        )

        risk_primary, risk_secondary = self._score(features)
        labels = label_events_fast(
            frame["glucose_mgdl"].to_numpy(dtype=float),
            horizon_steps=settings.horizon_steps,
            high_threshold=settings.glucose_high_threshold_mgdl,
            rise_delta=settings.glucose_rise_delta_mgdl,
        )

        now_ts = pd.Timestamp(datetime.combine(end_time.date(), _clock(settings.demo_now_clock)))
        sim_start_ts = pd.Timestamp(
            datetime.combine(end_time.date(), _clock(settings.demo_simulation_start_clock))
        )
        record = PatientRecord(
            patient_id=patient_id,
            profile=profile,
            stream=stream,
            baseline=baseline,
            features=features,
            ehr_static=ehr_static,
            risk_primary=risk_primary,
            risk_secondary=risk_secondary,
            labels=labels,
            now_index=_nearest_index(frame["timestamp"], now_ts),
            simulation_start_index=_nearest_index(frame["timestamp"], sim_start_ts),
            last_index=len(frame) - 1,
        )
        if self.rebuild_files:
            self._persist(record)
        return record

    def _score(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Score every instant with both trained models (falls back to a
        transparent heuristic if artifacts are missing, so the UI never dies)."""
        try:
            from backend.models.predictor import get_predictor

            predictor = get_predictor()
            matrix = features[fe.FEATURE_NAMES]
            primary = predictor.predict_proba(matrix)
            secondary = predictor.predict_proba_secondary(matrix)
            return np.asarray(primary, dtype=float), np.asarray(secondary, dtype=float)
        except Exception as exc:  # pragma: no cover - degraded mode
            print(f"[patient_service] model artifacts unavailable ({exc}); using heuristic scorer")
            heuristic = _heuristic_risk(features)
            return heuristic, heuristic

    def _persist(self, record: PatientRecord) -> None:
        """Write the generated signal + its provenance next to the code."""
        try:
            SENSOR_DIR.mkdir(parents=True, exist_ok=True)
            csv_path = SENSOR_DIR / f"{record.patient_id}_signals.csv"
            record.frame.to_csv(csv_path, index=False)
            metadata = record.stream.to_metadata()
            metadata.update(
                {
                    "classification": "SYNTHETIC — generated, not measured",
                    "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "generator": "backend/services/synthetic_data.py",
                    "scenario_day": record.stream.scenario_day.isoformat(),
                    "sampling_interval_minutes": settings.sampling_interval_minutes,
                    "signals": [
                        {"column": "glucose_mgdl", "device": "Continuous glucose monitor", "precision": "integer mg/dL"},
                        {"column": "heart_rate_bpm", "device": "Wrist wearable (PPG)", "precision": "integer bpm"},
                        {"column": "hrv_rmssd_ms", "device": "Wrist wearable (PPG)", "precision": "integer ms"},
                        {"column": "steps_5min", "device": "Wrist wearable (accelerometer)", "precision": "integer steps"},
                        {"column": "activity_met", "device": "Wrist wearable", "precision": "0.1 MET"},
                        {"column": "spo2_pct", "device": "Wrist wearable (SpO₂)", "precision": "integer %"},
                        {"column": "sleep_stage", "device": "Wrist wearable (hypnogram)", "precision": "categorical"},
                    ],
                    "contains_real_patient_data": False,
                    "personal_baseline": {
                        k: v for k, v in record.baseline.items() if not isinstance(v, (dict, list))
                    },
                    "habitual_meals": record.baseline.get("habitual_meals", []),
                }
            )
            metadata_path = SENSOR_DIR / f"{record.patient_id}_metadata.json"
            # The generator is deterministic, so a restart produces byte-identical
            # provenance apart from `generated_at`.  Skip the write in that case to
            # keep these committed files (and therefore `git status`) clean.
            if metadata_path.exists():
                try:
                    previous = json.loads(metadata_path.read_text())
                except json.JSONDecodeError:
                    previous = None
                if previous is not None:
                    drift = lambda payload: {k: v for k, v in payload.items() if k != "generated_at"}
                    if drift(previous) == drift(metadata):
                        return
            metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
        except OSError as exc:  # pragma: no cover
            print(f"[patient_service] could not persist sensor data: {exc}")

    # ----------------------------------------------------------------- access
    def ids(self) -> List[str]:
        with self._lock:
            return list(self._order)

    def all_summaries(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._records[pid].to_summary() for pid in self._order]

    def get(self, patient_id: str) -> PatientRecord:
        with self._lock:
            record = self._records.get(patient_id)
        if record is None:
            raise PatientNotFoundError(patient_id)
        return record

    def exists(self, patient_id: str) -> bool:
        with self._lock:
            return patient_id in self._records

    def roster_metadata(self) -> Dict[str, Any]:
        return dict(self._roster)


def _nearest_index(timestamps: pd.Series, target: pd.Timestamp) -> int:
    values = timestamps.to_numpy()
    target_np = np.datetime64(target)
    before = np.where(values <= target_np)[0]
    if len(before) == 0:
        return 0
    return int(before[-1])


def _heuristic_risk(features: pd.DataFrame) -> np.ndarray:
    """
    Transparent fallback scorer used only when no trained artifact is present.

    It is a fixed logistic combination of a handful of clinically obvious
    signals so that the dashboard still behaves sensibly on a machine where
    ``model/train.py`` has not been run.  It is clearly labelled as such in the
    API response and is never presented as the trained model.
    """
    z = (
        0.9 * np.clip(features["glucose_current"] - 150.0, -60, 120) / 40.0
        + 1.4 * np.clip(features["glucose_slope_30"], -1.0, 1.5)
        + 0.8 * np.clip(features["expected_meal_proximity"], 0, 1.5)
        + 0.5 * np.clip(features["sedentary_hours_ewma"], 0, 3) / 3.0
        + 0.5 * np.clip(features["sleep_deficit_h"], -1, 3) / 2.0
        + 0.6 * np.clip(features["hba1c_pct"] - 6.5, -1.5, 3.5) / 2.0
        + 0.4 * np.clip(features["prior_time_above_180_pct"], 0, 60) / 30.0
        - 0.4 * np.clip(features["activity_dev_personal_pct"], -80, 80) / 60.0
        - 0.3 * np.clip(features["hrv_dev_personal_pct"], -40, 40) / 30.0
        - 2.2
    )
    return 1.0 / (1.0 + np.exp(-np.clip(z, -8, 8)))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_REPOSITORY: Optional[PatientRepository] = None
_REPO_LOCK = threading.Lock()


def get_repository(rebuild: bool = False) -> PatientRepository:
    global _REPOSITORY
    with _REPO_LOCK:
        if _REPOSITORY is None or rebuild:
            _REPOSITORY = PatientRepository()
        return _REPOSITORY
