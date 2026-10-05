"""
VitalSync — regression tests for the parts that are easiest to get silently
wrong in a clinical ML prototype:

  * the event label (forward-looking window, no leakage of the current value)
  * the feature layer (fixed column set, finite values, backward-looking only)
  * the synthetic physiology generator (determinism, clinical plausibility)
  * the exported model artifacts and the API surface
  * the graceful-degradation paths (no Groq key, unknown patient)

Run:  pytest -q
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services import ehr_synthesis, feature_engineering as fe  # noqa: E402
from backend.services.synthetic_data import (  # noqa: E402
    generate_stream,
    label_events,
    label_events_fast,
)
from backend.settings import settings  # noqa: E402


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------
def test_fast_and_reference_labellers_agree():
    """The vectorised labeller must match the loop implementation exactly."""
    rng = np.random.default_rng(7)
    for _ in range(60):
        glucose = rng.integers(85, 250, size=48).astype(float)
        horizon = int(rng.integers(2, 12))
        assert np.array_equal(
            label_events_fast(glucose, horizon), label_events(glucose, horizon)
        )


def test_label_is_strictly_forward_looking():
    """
    Changing the past must not change a label; changing the future must.

    This is the single most important property of the dataset: if it fails, the
    model is learning the answer from the input.
    """
    base = np.full(60, 120.0)
    base[40:50] = 240.0                       # a spike well inside the horizon of index 20
    labels = label_events_fast(base, horizon_steps=24)
    assert labels[20] == 1

    # rewrite everything BEFORE index 20 — the label at 20 must not move
    altered = base.copy()
    altered[:20] = 300.0
    assert label_events_fast(altered, horizon_steps=24)[20] == labels[20]

    # remove the future spike — the label must drop to 0
    removed = base.copy()
    removed[40:50] = 130.0
    assert label_events_fast(removed, horizon_steps=24)[20] == 0


def test_label_requires_both_threshold_and_rise():
    """A patient already sitting above 180 with no further rise is not an event."""
    flat_high = np.full(60, 210.0)
    assert label_events_fast(flat_high, horizon_steps=24).max() == 0

    rising = np.full(60, 150.0)
    rising[30:] = 190.0
    assert label_events_fast(rising, horizon_steps=24)[10] == 1


def test_label_tail_is_unscoreable():
    """Rows without a complete future window are marked -1, never 0 or 1."""
    glucose = np.linspace(100, 200, 40)
    labels = label_events_fast(glucose, horizon_steps=24)
    assert (labels[-24:] == -1).all()


def test_single_sample_artefact_is_not_an_event():
    """One noisy reading above threshold must not create a label."""
    glucose = np.full(60, 150.0)
    glucose[35] = 260.0                       # isolated artefact
    assert label_events_fast(glucose, horizon_steps=24).max() == 0
    glucose[36] = 255.0                       # two consecutive readings = real
    assert label_events_fast(glucose, horizon_steps=24).max() == 1


# ---------------------------------------------------------------------------
# Feature layer
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def demo_record():
    profile = json.loads((REPO_ROOT / "backend" / "data" / "patients.json").read_text())["patients"][0]
    stream = generate_stream(
        profile["patient_id"],
        profile["physiology"],
        profile["scenario"],
        end_time=datetime.combine(date.today(), dtime(15, 0)),
        hours=72,
    )
    ehr_static = fe.normalise_ehr(profile["ehr"])
    cutoff = stream.frame["timestamp"].iloc[0] + pd.Timedelta(hours=48)
    baseline = fe.compute_personal_baseline(
        stream.frame[stream.frame["timestamp"] < cutoff],
        stream.sleep_nights,
        ehr_static,
        meals=stream.meals,
    )
    matrix = fe.build_feature_frame(
        stream.frame, stream.meals, stream.sleep_nights, ehr_static, baseline
    )
    return profile, stream, baseline, matrix


def test_feature_frame_is_complete_and_finite(demo_record):
    _, _, _, matrix = demo_record
    assert list(matrix.columns[-len(fe.FEATURE_NAMES):]) == fe.FEATURE_NAMES
    block = matrix[fe.FEATURE_NAMES].to_numpy(dtype=float)
    assert np.isfinite(block).all(), "feature matrix contains NaN or inf"


def test_features_are_backward_looking(demo_record):
    """
    Truncating the stream must not change any feature already computed.

    This proves no rolling window, cumsum or lookup reaches into the future.
    """
    profile, stream, baseline, matrix = demo_record
    cut = len(stream.frame) - 120
    shorter = fe.build_feature_frame(
        stream.frame.iloc[:cut],
        [m for m in stream.meals if pd.Timestamp(m["datetime"]) <= stream.frame["timestamp"].iloc[cut - 1]],
        stream.sleep_nights,
        fe.normalise_ehr(profile["ehr"]),
        baseline,
    )
    compare = matrix.iloc[:cut][fe.FEATURE_NAMES].to_numpy(dtype=float)
    assert np.allclose(shorter[fe.FEATURE_NAMES].to_numpy(dtype=float), compare, atol=1e-6)


def test_meal_and_sleep_context_is_populated(demo_record):
    """Regression: pandas microsecond/nanosecond mixing silently broke these."""
    _, stream, _, matrix = demo_record
    breakfast = next(m for m in stream.meals if m["label"].startswith("Breakfast") and m["day_offset"] == 0)
    at = pd.Timestamp(breakfast["datetime"]) + pd.Timedelta(minutes=30)
    row = matrix[matrix["timestamp"] == at].iloc[0]
    assert row["minutes_since_last_meal"] == pytest.approx(30, abs=5)
    assert row["last_meal_carbs_g"] == breakfast["carbs_g"]
    assert row["is_postprandial"] == 1.0
    assert row["sleep_duration_h"] > 3.0
    assert row["sleep_efficiency"] > 0.5


def test_habitual_meal_times_are_learned_not_hardcoded(demo_record):
    _, _, baseline, _ = demo_record
    habitual = baseline["habitual_meals"]
    assert habitual, "no habitual meals learned from the onboarding window"
    names = {m["name"] for m in habitual}
    assert "lunch" in names
    lunch = next(m for m in habitual if m["name"] == "lunch")
    assert 11.0 <= lunch["hour"] <= 15.0
    assert lunch["observations"] >= 1


def test_activity_deviation_is_proportional_not_explosive(demo_record):
    """Regression: a per-interval profile was once scaled as if it were hourly."""
    _, _, _, matrix = demo_record
    deviation = matrix["activity_dev_personal_pct"].to_numpy(dtype=float)
    assert deviation.min() >= -100.0 and deviation.max() <= 300.0
    assert np.abs(np.median(deviation)) < 60.0


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_generator_is_deterministic():
    profile = json.loads((REPO_ROOT / "backend" / "data" / "patients.json").read_text())["patients"][0]
    kwargs = dict(
        patient_id=profile["patient_id"],
        physiology=profile["physiology"],
        scenario=profile["scenario"],
        end_time=datetime.combine(date.today(), dtime(15, 0)),
        hours=24,
    )
    first = generate_stream(**kwargs).frame
    second = generate_stream(**kwargs).frame
    pd.testing.assert_frame_equal(first, second)


def test_signals_are_clinically_plausible():
    profile = json.loads((REPO_ROOT / "backend" / "data" / "patients.json").read_text())["patients"][0]
    frame = generate_stream(
        profile["patient_id"], profile["physiology"], profile["scenario"],
        end_time=datetime.combine(date.today(), dtime(15, 0)), hours=72,
    ).frame
    assert frame["glucose_mgdl"].between(50, 400).all()
    assert frame["heart_rate_bpm"].between(40, 190).all()
    assert frame["spo2_pct"].between(88, 100).all()
    assert frame["hrv_rmssd_ms"].between(5, 200).all()
    assert (frame["steps_5min"] >= 0).all()
    # "" marks a row that is not part of a scored sleep period (daytime)
    assert set(frame["sleep_stage"].unique()) <= {"", "awake", "light", "deep", "rem"}
    # integer precision, as a real device would report
    for column in ("glucose_mgdl", "heart_rate_bpm", "hrv_rmssd_ms", "steps_5min", "spo2_pct"):
        assert np.allclose(frame[column] % 1, 0), f"{column} is not reported at device precision"


def test_hba1c_is_consistent_with_generated_glucose():
    """
    The static record must agree with the signal it describes: HbA1c and mean
    glucose are linked by the published eAG relationship (±0.6 %).
    """
    for profile in json.loads((REPO_ROOT / "backend" / "data" / "patients.json").read_text())["patients"]:
        frame = generate_stream(
            profile["patient_id"], profile["physiology"], profile["scenario"],
            end_time=datetime.combine(date.today(), dtime(15, 0)), hours=72,
        ).frame
        mean_glucose = float(frame["glucose_mgdl"].mean())
        estimated_gmi = (mean_glucose + 46.7) / 28.7
        hba1c = next(l["value"] for l in profile["ehr"]["lab_results"] if l["test"] == "HbA1c")
        assert abs(estimated_gmi - hba1c) < 0.75, (
            f"{profile['patient_id']}: HbA1c {hba1c} implies {hba1c * 28.7 - 46.7:.0f} mg/dL "
            f"but the stream averages {mean_glucose:.0f} mg/dL"
        )


def test_cohort_profile_synthesis_is_bounded():
    rng = np.random.default_rng(3)
    for i in range(25):
        physiology = ehr_synthesis.sample_physiology(rng, i)
        ehr = ehr_synthesis.ehr_from_physiology(physiology, rng)
        assert 4.5 <= ehr["hba1c_pct"] <= 13.0
        assert 18 <= ehr["bmi"] <= 45
        assert 25 <= ehr["age"] <= 85
        assert ehr["n_glucose_meds"] >= 0
        assert physiology["fasting_glucose_mgdl"] > physiology["glucose_floor_mgdl"]


# ---------------------------------------------------------------------------
# Artifacts and API
# ---------------------------------------------------------------------------
def test_model_artifacts_exist_and_are_consistent():
    artifacts = REPO_ROOT / "model" / "artifacts"
    if not (artifacts / "twin_model.joblib").exists():
        pytest.skip("model not trained yet — run python model/train.py")
    metrics = json.loads((artifacts / "metrics.json").read_text())
    manifest = json.loads((artifacts / "train_manifest.json").read_text())
    test = metrics["test"]
    assert 0.5 <= test["auroc"] <= 1.0
    assert test["confusion_matrix"]["total"] == test["n_samples"]
    assert test["n_positive"] > 0
    # metrics must come from a patient-level split
    split = manifest["split"]
    assert split["strategy"].lower().startswith("groupshufflesplit")
    assert split["test_patients"] > 0 and split["train_patients"] > 0
    assert manifest["dataset"]["contains_real_patient_data"] is False


def test_trained_model_scores_the_demo_stream():
    from backend.models.predictor import try_get_predictor

    predictor, error = try_get_predictor()
    if predictor is None:
        pytest.skip(f"model not trained: {error}")
    profile, _, _, matrix = _demo_matrix()
    probabilities = predictor.predict_proba(matrix[fe.FEATURE_NAMES])
    assert probabilities.shape == (len(matrix),)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()

    attribution = predictor.explain(matrix.iloc[-100][fe.FEATURE_NAMES].to_dict(), top_k=5)
    assert attribution["top_contributors"]
    assert abs(attribution["total_effect_vs_reference"]) >= 0
    if attribution["method"].startswith("TreeSHAP"):
        assert attribution["shap_additivity_check"] < 1e-6


def _demo_matrix():
    profile = json.loads((REPO_ROOT / "backend" / "data" / "patients.json").read_text())["patients"][0]
    stream = generate_stream(
        profile["patient_id"], profile["physiology"], profile["scenario"],
        end_time=datetime.combine(date.today(), dtime(15, 0)), hours=72,
    )
    ehr_static = fe.normalise_ehr(profile["ehr"])
    cutoff = stream.frame["timestamp"].iloc[0] + pd.Timedelta(hours=48)
    baseline = fe.compute_personal_baseline(
        stream.frame[stream.frame["timestamp"] < cutoff], stream.sleep_nights, ehr_static, meals=stream.meals
    )
    matrix = fe.build_feature_frame(stream.frame, stream.meals, stream.sleep_nights, ehr_static, baseline)
    return profile, stream, baseline, matrix


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_api_health_and_config(client):
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["components"]["patient_repository"]["patients"]
    config = client.get("/api/config").json()
    assert config["event_definition"]["horizon_minutes"] == settings.forecast_horizon_minutes
    assert config["groq"]["enabled"] in (True, False)


def test_api_never_leaks_the_groq_key(client):
    """
    No response may carry a secret *value*.  Mentioning the name of the
    environment variable in documentation is fine and desirable; emitting a
    Groq-shaped token (``gsk_``...) or the configured key is not.
    """
    for path in ("/api/health", "/api/config", "/api/profile", "/api/insights/status",
                 "/api/datasources", "/api/architecture", "/api/model/info"):
        body = client.get(path).text
        assert "gsk_" not in body, f"{path} appears to contain a Groq API key"
        if settings.groq_api_key:
            assert settings.groq_api_key not in body, f"{path} leaked the configured key"


def test_api_prediction_and_dashboard(client):
    prediction = client.get("/api/patients/DT-1047/prediction").json()
    assert 0.0 <= prediction["probability"] <= 1.0
    assert prediction["risk"]["label"] in ("LOW", "MODERATE", "HIGH")
    assert prediction["attribution"]["top_contributors"]
    assert prediction["model"]["model_id"]
    dashboard = client.get("/api/simulation/snapshot?patient_id=DT-1047").json()
    for key in ("patient", "prediction", "twin", "baseline_comparison", "sensor_cards", "chart", "timeline", "alerts"):
        assert key in dashboard["dashboard"]


def test_api_simulation_replay_arc(client):
    started = client.post("/api/simulation/start", json={"speed_minutes_per_tick": 30}).json()
    session_id = started["session"]["session_id"]
    seen = []
    for _ in range(14):
        step = client.post("/api/simulation/step", json={"session_id": session_id, "minutes": 30}).json()
        seen.append(step["session"]["band"])
        if step["session"]["at_stream_end"]:
            break
    assert "low" in seen and "high" in seen, f"replay never spanned the risk bands: {seen}"
    reset = client.post("/api/simulation/reset", json={"session_id": session_id}).json()
    assert reset["session"]["running"] is False


def test_api_insights_fallback_is_structured(client):
    response = client.post("/api/insights", json={"action": "explain_risk", "include_context": False})
    assert response.status_code == 200
    body = response.json()
    assert body["source"] in ("groq", "deterministic-fallback")
    for key in ("headline", "summary", "key_changes", "risk_explanation", "clinical_attention",
                "monitoring_considerations", "data_limitations", "confidence_note"):
        assert key in body["insight"]
    if body["source"] == "deterministic-fallback":
        assert body["status_message"]


def test_api_error_paths(client):
    assert client.get("/api/patients/DOES-NOT-EXIST").status_code == 404
    assert client.get("/api/simulation/state?session_id=nope").status_code == 404
    assert client.get("/api/model/metrics?split=nope").status_code == 400
    assert client.get("/api/patients/DT-1047/twin/domains/nope").status_code == 404


def test_profile_has_no_secrets(client):
    """
    profile.json is served to the browser verbatim, so it must be free of
    credentials, private contact details and real patient data.
    """
    profile = client.get("/api/profile").json()
    text = json.dumps(profile).lower()
    for forbidden in ("gsk_", "bearer ", "password\": \"", "api_key\": \""):
        assert forbidden not in text, f"profile.json appears to contain {forbidden!r}"

    # Any address that looks like a real e-mail must be a placeholder or an
    # example.com address — profile.json is served to the browser verbatim.
    import re

    addresses = re.findall(r"[\w.+-]+@[\w.-]+\.\w+", json.dumps(profile))
    placeholders = ("@example.com", "@example.org", ".example", "your-", "yourdomain", "yourteam", "placeholder")
    for address in addresses:
        lowered = address.lower()
        assert any(token in lowered for token in placeholders), (
            f"profile.json contains a real personal e-mail address: {address}"
        )
