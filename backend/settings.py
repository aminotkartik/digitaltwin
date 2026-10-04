"""
VitalSync — application settings.

All runtime configuration is read from environment variables (optionally via a
`.env` file at the repository root).  Nothing secret is ever hardcoded, and no
value defined here is exposed to the browser: the frontend only receives what
the API routes explicitly serialise.

Clinical thresholds used by the prediction target are defined here so that the
event definition stays in one place and can be reviewed by a clinician.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = REPO_ROOT / "frontend"
DATA_DIR = BACKEND_DIR / "data"
SENSOR_DIR = DATA_DIR / "sensor_data"
CONFIG_DIR = BACKEND_DIR / "config"
ARTIFACT_DIR = REPO_ROOT / "model" / "artifacts"
DATASET_DIR = REPO_ROOT / "datasets" / "synthetic"

# `.env` lives at the repository root so a single file configures the project.
ENV_FILE = REPO_ROOT / ".env"


class Settings(BaseSettings):
    """Typed application settings (env driven)."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- Product ----------------------------------------------------------
    app_name: str = "VitalSync"
    app_subtitle: str = "Metabolic Digital Twin Platform"
    app_tagline: str = "Metabolic Digital Twin Platform"
    app_version: str = "1.0.0"
    environment: str = "prototype"  # prototype | staging | production
    api_prefix: str = "/api"
    # Shown on every screen and in every API payload that carries a prediction.
    disclaimer: str = (
        "Prototype for research and demonstration. Not intended for diagnosis or medical decision-making."
    )
    data_classification: str = "SYNTHETIC — NOT REAL PATIENT DATA"

    # -- Server -----------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

    # -- Groq (LLM interpretation layer, NOT the prediction engine) -------
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_timeout_seconds: float = 25.0
    groq_max_tokens: int = 900
    groq_temperature: float = 0.2
    # Force-disable the LLM layer even when a key is present (demo/offline mode)
    groq_force_disabled: bool = False

    # -- Clinical event definition ----------------------------------------
    # Target event: a *significant* glucose elevation inside the forecast
    # horizon.  180 mg/dL is the hyperglycaemic threshold used by the ATTD
    # international consensus on glucose metrics; the additional rise delta
    # keeps the label focused on clinically meaningful excursions rather than
    # a patient who is simply sitting just above threshold.
    glucose_high_threshold_mgdl: float = 180.0
    glucose_rise_delta_mgdl: float = 25.0
    glucose_low_threshold_mgdl: float = 70.0
    forecast_horizon_minutes: int = 120

    # -- Risk banding ------------------------------------------------------
    # Risk bands shown to the clinician.  These are display bands, distinct from
    # the model's operating threshold (chosen by max-F1 on validation patients).
    risk_moderate_min: float = 0.35
    risk_high_min: float = 0.70

    # -- Signal / data -----------------------------------------------------
    sampling_interval_minutes: int = 5
    stream_hours: int = 72
    demo_patient_id: str = "DT-1047"
    # Personal baselines are estimated from an onboarding window at the start of
    # the stored history; only samples after it are ever scored.
    baseline_onboarding_hours: int = 48

    # -- Demo clock --------------------------------------------------------
    # The twin replays a scripted scenario day, so the "current" instant is a
    # fixed clock time on today's date rather than the wall clock.  This keeps
    # the clinical narrative identical whenever the prototype is demonstrated.
    demo_stream_end_clock: str = "15:00"     # last sample held by the twin
    demo_now_clock: str = "10:45"            # where the dashboard opens
    demo_simulation_start_clock: str = "09:30"  # where "Start simulation" begins

    # -- Model selection ---------------------------------------------------
    # "auto" picks the strongest available estimator (XGBoost > HistGradient
    # Boosting > Logistic Regression).  Set explicitly to compare models.
    preferred_model: str = "auto"
    # The always-available explainable baseline, used for model-disagreement
    # based confidence reporting.
    baseline_model: str = "logistic"

    # -- Feature pipeline --------------------------------------------------
    slope_windows_minutes: List[int] = Field(default_factory=lambda: [15, 30, 60])
    rolling_windows_minutes: List[int] = Field(default_factory=lambda: [30, 60, 180])

    @field_validator("cors_origins", "slope_windows_minutes", "rolling_windows_minutes", mode="before")
    @classmethod
    def _split_csv(cls, value):
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    # -- Derived helpers ---------------------------------------------------
    @property
    def groq_enabled(self) -> bool:
        """Groq is used only when a key is configured and not force-disabled."""
        return bool(self.groq_api_key.strip()) and not self.groq_force_disabled

    @property
    def steps_per_interval(self) -> int:
        return max(1, int(self.sampling_interval_minutes))

    @property
    def horizon_steps(self) -> int:
        return int(self.forecast_horizon_minutes / self.sampling_interval_minutes)

    @property
    def risk_bands(self) -> dict:
        return {
            "low": (0.0, self.risk_moderate_min),
            "moderate": (self.risk_moderate_min, self.risk_high_min),
            "high": (self.risk_high_min, 1.0),
        }


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


settings = get_settings()
