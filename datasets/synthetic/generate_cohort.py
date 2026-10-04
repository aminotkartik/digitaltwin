"""
VitalSync — synthetic training cohort generator.

Builds a reproducible multi-patient dataset for the 2-hour glucose risk model.

For each synthetic patient we:
  1. draw a physiology profile (``ehr_synthesis.sample_physiology``),
  2. generate ``days`` of multi-signal history with the same generator that
     drives the live demo patients,
  3. derive a plausible longitudinal EHR record from that physiology
     (HbA1c really does track the mean glucose of the stream),
  4. estimate the patient's personal baseline from the first 48 h only
     (an "onboarding window" that strictly precedes every scored sample),
  5. score every 15-minute position after the onboarding window: build the
     fused feature vector and label the next 2 h.

Output files (written to ``datasets/synthetic/``):
  cohort_features.csv.gz   feature matrix + label + keys
  cohort_patients.json     per-patient physiology and derived EHR record
  dataset_metadata.json    provenance: sizes, seeds, versions, distributions

Reproduce with:
    python datasets/synthetic/generate_cohort.py --patients 64 --days 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services import feature_engineering as fe  # noqa: E402
from backend.services import ehr_synthesis  # noqa: E402
from backend.services.synthetic_data import generate_stream, label_events_fast  # noqa: E402

DATASET_DIR = REPO_ROOT / "datasets" / "synthetic"
FEATURES_FILE = DATASET_DIR / "cohort_features.csv.gz"
PATIENTS_FILE = DATASET_DIR / "cohort_patients.json"
METADATA_FILE = DATASET_DIR / "dataset_metadata.json"

FORWARD_DELTA_MINUTES = (15, 30, 45, 60, 75, 90, 105, 120)

DEFAULTS = {
    "n_patients": 96,
    "days": 6,                 # 2 days onboarding + 4 scored days
    "onboarding_hours": 48,
    "stride_minutes": 15,
    "interval_minutes": 5,
    "horizon_minutes": 120,
    "high_threshold_mgdl": 180.0,
    "rise_delta_mgdl": 25.0,
    "cohort_seed": 20260117,
    "anchor_date": "2026-09-20",   # fixed so the dataset is bit-reproducible
}


def build_cohort(
    n_patients: int = DEFAULTS["n_patients"],
    days: int = DEFAULTS["days"],
    onboarding_hours: float = DEFAULTS["onboarding_hours"],
    stride_minutes: int = DEFAULTS["stride_minutes"],
    interval_minutes: int = DEFAULTS["interval_minutes"],
    horizon_minutes: int = DEFAULTS["horizon_minutes"],
    high_threshold: float = DEFAULTS["high_threshold_mgdl"],
    rise_delta: float = DEFAULTS["rise_delta_mgdl"],
    cohort_seed: int = DEFAULTS["cohort_seed"],
    anchor_date: str = DEFAULTS["anchor_date"],
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    """Generate the cohort. Returns (feature frame, patient records, stats)."""
    rng = np.random.default_rng(cohort_seed)
    horizon_steps = int(horizon_minutes / interval_minutes)
    end_time = datetime.fromisoformat(f"{anchor_date}T18:00:00")

    rows: List[pd.DataFrame] = []
    patient_records: List[Dict[str, Any]] = []
    per_patient_events: List[float] = []
    t_start = time.time()

    for i in range(n_patients):
        patient_id = f"SYN-{i + 1:04d}"
        physiology = ehr_synthesis.sample_physiology(rng, i)
        scenario = ehr_synthesis.sample_scenario(rng, physiology, days)

        stream = generate_stream(
            patient_id=patient_id,
            physiology=physiology,
            scenario=scenario,
            end_time=end_time,
            hours=int(days * 24),
            interval_minutes=interval_minutes,
        )
        frame = stream.frame

        # Derive the longitudinal record from the generated signal so the static
        # features are informative rather than decorative noise.
        report = ehr_synthesis.cgm_report_from_stream(frame.iloc[: int(onboarding_hours * 60 / interval_minutes)])
        ehr_static = ehr_synthesis.ehr_from_physiology(physiology, rng, report)

        # Personal baseline: onboarding window ONLY (strictly in the past).
        cutoff = frame["timestamp"].iloc[0] + pd.Timedelta(hours=onboarding_hours)
        baseline_frame = frame[frame["timestamp"] < cutoff]
        baseline = fe.compute_personal_baseline(
            baseline_frame, stream.sleep_nights, ehr_static, meals=stream.meals
        )

        matrix = fe.build_feature_frame(
            frame, stream.meals, stream.sleep_nights, ehr_static, baseline, interval_minutes
        )

        # Realised forward deltas.  These are FUTURE information and are never
        # used as model inputs (they are not in FEATURE_NAMES); they exist only so
        # the training script can estimate "what does glucose usually do next,
        # conditional on the predicted risk level" — the trajectory band drawn
        # behind the forecast on the dashboard.
        glucose_series = frame["glucose_mgdl"].astype(float)
        for tau in FORWARD_DELTA_MINUTES:
            steps = int(tau / interval_minutes)
            matrix[f"future_delta_{tau}"] = (glucose_series.shift(-steps) - glucose_series).to_numpy()

        labels = label_events_fast(
            frame["glucose_mgdl"].to_numpy(dtype=float),
            horizon_steps=horizon_steps,
            high_threshold=high_threshold,
            rise_delta=rise_delta,
        )
        matrix["label"] = labels

        scored = matrix[(matrix["timestamp"] >= cutoff) & (matrix["label"] >= 0)].copy()
        stride = max(1, int(stride_minutes / interval_minutes))
        scored = scored.iloc[::stride]
        scored.insert(0, "patient_id", patient_id)
        rows.append(scored)

        event_rate = float(scored["label"].mean()) if len(scored) else 0.0
        per_patient_events.append(event_rate)
        patient_records.append(
            {
                "patient_id": patient_id,
                "physiology": {k: v for k, v in physiology.items() if not k.startswith("_")},
                "severity_index": physiology.get("_severity"),
                "fitness_index": physiology.get("_fitness"),
                "derived_ehr": ehr_static,
                "personal_baseline": {
                    k: v for k, v in baseline.items() if not isinstance(v, (dict,))
                },
                "habitual_meals": baseline.get("habitual_meals", []),
                "cgm_report_onboarding": report,
                "scored_samples": int(len(scored)),
                "event_rate": round(event_rate, 4),
            }
        )
        if verbose and (i + 1) % 8 == 0:
            print(f"  ... {i + 1}/{n_patients} patients ({time.time() - t_start:.1f}s)", flush=True)

    cohort = pd.concat(rows, ignore_index=True)
    cohort["timestamp"] = pd.to_datetime(cohort["timestamp"])

    stats = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "generator": "datasets/synthetic/generate_cohort.py",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "n_patients": int(n_patients),
        "days_per_patient": int(days),
        "onboarding_hours": onboarding_hours,
        "sampling_interval_minutes": interval_minutes,
        "scoring_stride_minutes": stride_minutes,
        "forecast_horizon_minutes": horizon_minutes,
        "event_definition": {
            "description": (
                "Within the next {h} minutes the 15-min-filtered CGM trace reaches "
                ">= {t} mg/dL AND rises >= {d} mg/dL above the value at the prediction instant."
            ).format(h=horizon_minutes, t=int(high_threshold), d=int(rise_delta)),
            "high_threshold_mgdl": high_threshold,
            "rise_delta_mgdl": rise_delta,
            "labelling_filter": "centred 3-sample (15 min) moving average",
            "reference": "ATTD international consensus on time in range (180 mg/dL upper bound)",
        },
        "rows": int(len(cohort)),
        "features": len(fe.FEATURE_NAMES),
        "feature_names": fe.FEATURE_NAMES,
        "positive_events": int(cohort["label"].sum()),
        "positive_rate": round(float(cohort["label"].mean()), 4),
        "per_patient_event_rate": {
            "min": round(float(np.min(per_patient_events)), 4),
            "median": round(float(np.median(per_patient_events)), 4),
            "max": round(float(np.max(per_patient_events)), 4),
        },
        "cohort_seed": cohort_seed,
        "anchor_date": anchor_date,
        "generation_seconds": round(time.time() - t_start, 1),
        "licence": "CC-BY-4.0 (synthetic data generated by this repository)",
        "contains_real_patient_data": False,
    }
    return cohort, patient_records, stats


def write_cohort(cohort: pd.DataFrame, patients: List[Dict[str, Any]], stats: Dict[str, Any]) -> Dict[str, str]:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(FEATURES_FILE, index=False, compression="gzip")
    PATIENTS_FILE.write_text(json.dumps({"patients": patients}, indent=1))
    digest = hashlib.sha256(FEATURES_FILE.read_bytes()).hexdigest()[:16]
    stats["sha256_16"] = digest
    stats["files"] = {
        "features": FEATURES_FILE.name,
        "patients": PATIENTS_FILE.name,
        "metadata": METADATA_FILE.name,
    }
    METADATA_FILE.write_text(json.dumps(stats, indent=2))
    return {"features": str(FEATURES_FILE), "patients": str(PATIENTS_FILE), "metadata": str(METADATA_FILE)}


def load_cohort() -> Optional[pd.DataFrame]:
    if not FEATURES_FILE.exists():
        return None
    df = pd.read_csv(FEATURES_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the VitalSync synthetic training cohort")
    parser.add_argument("--patients", type=int, default=DEFAULTS["n_patients"])
    parser.add_argument("--days", type=int, default=DEFAULTS["days"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["cohort_seed"])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    print(f"Generating synthetic cohort: {args.patients} patients x {args.days} days ...")
    cohort, patients, stats = build_cohort(
        n_patients=args.patients, days=args.days, cohort_seed=args.seed, verbose=not args.quiet
    )
    paths = write_cohort(cohort, patients, stats)
    print(f"  rows            : {stats['rows']:,}")
    print(f"  features        : {stats['features']}")
    print(f"  positive rate   : {stats['positive_rate']:.3f}")
    print(f"  seconds         : {stats['generation_seconds']}")
    for key, path in paths.items():
        print(f"  {key:15s}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
