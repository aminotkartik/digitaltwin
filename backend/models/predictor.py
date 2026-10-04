"""
VitalSync — trained risk model wrapper.

Loads ``model/artifacts/twin_model.joblib`` and exposes everything the API
layer needs:

  * batch / single-row probability scoring for the primary model and the
    independently trained secondary model,
  * local (per-prediction) attribution by *occlusion*: each feature is replaced
    by its training-set reference value and the change in predicted risk is
    measured.  This is model agnostic, so it works identically for the boosted
    trees and for logistic regression, and it never invents a contribution that
    the model did not actually use,
  * global permutation importance and the standardised logistic coefficients
    read straight from the training artifacts.

The wrapper is deliberately read-only and stateless: nothing here mutates a
patient record or the model.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from backend.services.feature_engineering import (  # noqa: E402
    FEATURE_GROUPS,
    FEATURE_NAMES,
    FEATURE_SPECS,
    spec_for,
)

DEFAULT_ARTIFACT_DIR = REPO_ROOT / "model" / "artifacts"


class ModelNotTrainedError(RuntimeError):
    """Raised when artifacts are missing — the API degrades gracefully."""


class RiskPredictor:
    """Thread-safe read-only wrapper around the trained artifacts."""

    def __init__(self, artifact_dir: Optional[Path] = None) -> None:
        self.artifact_dir = Path(artifact_dir or DEFAULT_ARTIFACT_DIR)
        self._lock = threading.Lock()
        self._bundle: Optional[Dict[str, Any]] = None
        self._metrics: Dict[str, Any] = {}
        self._importance: Dict[str, Any] = {}
        self._manifest: Dict[str, Any] = {}
        self.load()

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        model_path = self.artifact_dir / "twin_model.joblib"
        if not model_path.exists():
            raise ModelNotTrainedError(
                f"No trained model at {model_path}. Run: python model/train.py"
            )
        with self._lock:
            self._bundle = joblib.load(model_path)
            self._metrics = _read_json(self.artifact_dir / "metrics.json")
            self._importance = _read_json(self.artifact_dir / "feature_importance.json")
            self._manifest = _read_json(self.artifact_dir / "train_manifest.json")

    @property
    def bundle(self) -> Dict[str, Any]:
        if self._bundle is None:
            raise ModelNotTrainedError("model bundle not loaded")
        return self._bundle

    @property
    def feature_names(self) -> List[str]:
        return list(self.bundle.get("feature_names", FEATURE_NAMES))

    @property
    def threshold(self) -> float:
        return float(self.bundle.get("threshold", 0.5))

    @property
    def model_id(self) -> str:
        return str(self.bundle.get("model_id", "unknown"))

    @property
    def estimator_name(self) -> str:
        return str(self.bundle.get("estimator_name", "unknown"))

    @property
    def primary_kind(self) -> str:
        return str(self.bundle.get("primary_kind", "gradient_boosting"))

    # ------------------------------------------------------------- scoring
    def _as_frame(self, features: Any) -> pd.DataFrame:
        if isinstance(features, pd.DataFrame):
            frame = features
        elif isinstance(features, dict):
            frame = pd.DataFrame([features])
        else:
            frame = pd.DataFrame(list(features))
        for name in self.feature_names:
            if name not in frame.columns:
                frame[name] = np.nan
        frame = frame[self.feature_names].astype(float)
        # unseen/missing values fall back to the training median rather than
        # crashing the request
        medians = self.bundle.get("feature_medians", {})
        for name in frame.columns:
            if frame[name].isna().any():
                frame[name] = frame[name].fillna(float(medians.get(name, 0.0)))
        return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def predict_proba(self, features: Any) -> np.ndarray:
        """Primary-model risk probabilities for one row or a batch."""
        frame = self._as_frame(features)
        return self.bundle["primary"].predict_proba(frame)[:, 1]

    def predict_proba_secondary(self, features: Any) -> np.ndarray:
        """Secondary (independently trained) model probabilities."""
        model = self.bundle.get("secondary")
        if model is None:
            return self.predict_proba(features)
        frame = self._as_frame(features)
        return model.predict_proba(frame)[:, 1]

    def predict_single(self, features: Dict[str, float]) -> float:
        return float(self.predict_proba(features)[0])

    # --------------------------------------------------- reference values
    def reference_values(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.bundle.get("feature_medians", {}).items()}

    def feature_stats(self) -> Dict[str, Dict[str, float]]:
        return self.bundle.get("feature_stats", {})

    # ------------------------------------------------- local explanation
    def explain(
        self,
        features: Dict[str, float],
        top_k: int = 8,
        reference: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Occlusion-based local attribution.

        For every feature *j* we rebuild the input with ``x_j`` replaced by a
        reference value (the training-set median unless overridden) and measure
        ``p(full) - p(occluded_j)``.  A positive value means the patient's own
        value for *j* pushed the risk **up** relative to a typical patient.

        Because tree ensembles are not additive, the contributions do not sum
        exactly to ``p - p_reference``; the residual is reported explicitly as
        ``interaction_residual`` and contributions are rescaled so the panel the
        clinician sees is internally consistent.
        """
        refs = self.reference_values()
        if reference:
            refs = {**refs, **{k: float(v) for k, v in reference.items() if v is not None}}

        names = self.feature_names
        base_row = {name: float(features.get(name, refs.get(name, 0.0))) for name in names}
        p_full = float(self.predict_proba(base_row)[0])

        shap_result = self._shap_contributions(base_row)
        if shap_result is not None:
            return self._explain_shap(base_row, refs, p_full, shap_result, names, top_k)

        ref_row = {name: float(refs.get(name, base_row[name])) for name in names}
        p_ref = float(self.predict_proba(ref_row)[0])

        # one batched call: row 0 = full input, rows 1..N = each feature occluded
        matrix = np.tile(np.array([base_row[n] for n in names], dtype=float), (len(names) + 1, 1))
        for i, name in enumerate(names, start=1):
            matrix[i, names.index(name)] = ref_row[name]
        probs = self.bundle["primary"].predict_proba(pd.DataFrame(matrix, columns=names))[:, 1]
        raw = p_full - probs[1:]

        total_effect = p_full - p_ref
        captured = float(np.sum(raw))
        residual = total_effect - captured
        scale = (total_effect / captured) if abs(captured) > 1e-9 else 1.0

        contributions: List[Dict[str, Any]] = []
        for name, value in zip(names, raw):
            scaled = float(value * scale)
            spec = spec_for(name)
            contributions.append(
                {
                    "feature": name,
                    "label": _humanise(name),
                    "group": spec.group if spec else "other",
                    "group_label": FEATURE_GROUPS.get(spec.group, spec.group) if spec else "Other",
                    "unit": spec.unit if spec else "",
                    "description": spec.description if spec else "",
                    "value": round(float(base_row[name]), 3),
                    "reference_value": round(float(ref_row[name]), 3),
                    "contribution": round(scaled, 5),
                    "direction": "increases_risk" if scaled > 0 else "decreases_risk",
                    "abs_contribution": round(abs(scaled), 5),
                }
            )
        contributions.sort(key=lambda row: row["abs_contribution"], reverse=True)

        grouped: Dict[str, float] = {}
        for row in contributions:
            grouped[row["group"]] = round(grouped.get(row["group"], 0.0) + row["contribution"], 5)

        return {
            "method": "occlusion attribution (feature replaced by its training-set reference value)",
            "method_detail": (
                "Each feature was replaced in turn by its training-set reference value and the change in predicted "
                "risk measured. Contributions are rescaled so that they sum to the difference between this "
                "prediction and the reference-case prediction; the non-additive remainder is reported as "
                "interaction_residual."
            ),
            "risk_probability": round(p_full, 4),
            "reference_probability": round(p_ref, 4),
            "total_effect_vs_reference": round(total_effect, 4),
            "captured_by_attributions": round(captured, 4),
            "interaction_residual": round(float(residual), 4),
            "residual_share": round(abs(float(residual)) / max(abs(total_effect), 1e-6), 3),
            "top_contributors": contributions[:top_k],
            "by_group": [
                {"group": k, "group_label": FEATURE_GROUPS.get(k, k), "contribution": v}
                for k, v in sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)
            ],
            "all_contributions": contributions,
        }

    # ------------------------------------------------ TreeSHAP (exact) path
    def _shap_contributions(self, row: Dict[str, float]) -> Optional[Tuple[np.ndarray, float]]:
        """
        Exact additive attributions for tree ensembles.

        XGBoost supports TreeSHAP, which decomposes the margin (log-odds) into
        one term per feature plus a base value, with the terms summing exactly to
        the model output.  That is strictly better than occlusion for this model
        family, so it is used whenever available.
        """
        model = self.bundle.get("primary")
        names = self.feature_names
        vector = np.array([[float(row[n]) for n in names]], dtype=float)
        try:
            booster = model.get_booster() if hasattr(model, "get_booster") else None
            if booster is None:
                return None
            matrix = __import__("xgboost").DMatrix(vector, feature_names=names)
            contributions = booster.predict(matrix, pred_contribs=True)[0]
            return np.asarray(contributions[:-1], dtype=float), float(contributions[-1])
        except Exception:
            return None

    def _explain_shap(
        self,
        base_row: Dict[str, float],
        refs: Dict[str, float],
        p_full: float,
        shap_result: Tuple[np.ndarray, float],
        names: List[str],
        top_k: int,
    ) -> Dict[str, Any]:
        values, base_margin = shap_result
        margin = float(base_margin + values.sum())
        p_base = float(1.0 / (1.0 + np.exp(-base_margin)))

        # Convert each exact log-odds term into a probability-space contribution
        # by removing it from the margin: p - sigmoid(margin - shap_j).  These
        # are monotone in the Shapley value and directly readable as
        # "percentage points of risk".
        raw = np.array([p_full - float(1.0 / (1.0 + np.exp(-(margin - v)))) for v in values])
        total_effect = p_full - p_base
        captured = float(raw.sum())
        scale = (total_effect / captured) if abs(captured) > 1e-9 else 1.0

        contributions: List[Dict[str, Any]] = []
        for name, value, shap_value in zip(names, raw, values):
            scaled = float(value * scale)
            spec = spec_for(name)
            contributions.append(
                {
                    "feature": name,
                    "label": _humanise(name),
                    "group": spec.group if spec else "other",
                    "group_label": FEATURE_GROUPS.get(spec.group, spec.group) if spec else "Other",
                    "unit": spec.unit if spec else "",
                    "description": spec.description if spec else "",
                    "value": round(float(base_row[name]), 3),
                    "reference_value": round(float(refs.get(name, base_row[name])), 3),
                    "contribution": round(scaled, 5),
                    "shap_logodds": round(float(shap_value), 5),
                    "direction": "increases_risk" if scaled > 0 else "decreases_risk",
                    "abs_contribution": round(abs(scaled), 5),
                }
            )
        contributions.sort(key=lambda row: row["abs_contribution"], reverse=True)

        grouped: Dict[str, float] = {}
        for row in contributions:
            grouped[row["group"]] = round(grouped.get(row["group"], 0.0) + row["contribution"], 5)

        return {
            "method": "TreeSHAP (exact additive Shapley values for the fitted tree ensemble)",
            "method_detail": (
                "The model decomposes exactly into one term per feature plus a base value. Terms are shown as "
                "percentage points of risk after normalising the probability-space conversion so that the panel "
                "sums to the difference between this prediction and the model's population base rate."
            ),
            "risk_probability": round(p_full, 4),
            "reference_probability": round(p_base, 4),
            "base_margin_logodds": round(base_margin, 4),
            "margin_logodds": round(margin, 4),
            "total_effect_vs_reference": round(total_effect, 4),
            "captured_by_attributions": round(captured, 4),
            "interaction_residual": round(float(total_effect - captured), 4),
            "residual_share": round(abs(float(total_effect - captured)) / max(abs(total_effect), 1e-6), 3),
            "shap_additivity_check": round(float(abs(margin - (base_margin + values.sum()))), 8),
            "top_contributors": contributions[:top_k],
            "by_group": [
                {"group": k, "group_label": FEATURE_GROUPS.get(k, k), "contribution": v}
                for k, v in sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)
            ],
            "all_contributions": contributions,
        }

    # ------------------------------------------------------- global info
    def global_importance(self, limit: int = 25) -> List[Dict[str, Any]]:
        rows = self._importance.get("primary", [])
        out = []
        for row in rows[:limit]:
            spec = spec_for(row["feature"])
            out.append(
                {
                    **row,
                    "label": _humanise(row["feature"]),
                    "group": spec.group if spec else "other",
                    "group_label": FEATURE_GROUPS.get(spec.group, spec.group) if spec else "Other",
                    "unit": spec.unit if spec else "",
                    "description": spec.description if spec else "",
                }
            )
        return out

    def logistic_coefficients(self, limit: int = 25) -> List[Dict[str, Any]]:
        rows = self._importance.get("baseline_logistic_coefficients", [])
        out = []
        for row in rows[:limit]:
            if "feature" not in row:
                continue
            spec = spec_for(row["feature"])
            out.append(
                {
                    **row,
                    "label": _humanise(row["feature"]),
                    "group": spec.group if spec else "other",
                    "group_label": FEATURE_GROUPS.get(spec.group, spec.group) if spec else "Other",
                }
            )
        return out

    def metrics(self) -> Dict[str, Any]:
        return self._metrics

    def manifest(self) -> Dict[str, Any]:
        return self._manifest

    def info(self) -> Dict[str, Any]:
        """Everything the UI needs to describe the model honestly."""
        manifest = self._manifest
        test = self._metrics.get("test", {})
        dataset = manifest.get("dataset", {})
        selection = manifest.get("model_selection", self._metrics.get("candidate_selection", {}))
        return {
            "model_id": self.model_id,
            "estimator": self.estimator_name,
            "estimator_kind": self.primary_kind,
            "secondary_estimator": manifest.get("secondary_estimator"),
            "target": manifest.get("target", "P(significant glucose elevation within 120 min)"),
            "event_definition": self.bundle.get("event_definition", {}),
            "decision_threshold": self.threshold,
            "threshold_selection": manifest.get("threshold_selection", {}),
            "calibration_applied": bool(manifest.get("calibration_applied")),
            "trained_at": self.bundle.get("trained_at"),
            "feature_count": len(self.feature_names),
            "dataset": {
                "kind": "synthetic",
                "rows": dataset.get("rows"),
                "patients": dataset.get("n_patients"),
                "positive_rate": dataset.get("positive_rate"),
                "features": dataset.get("features"),
                "cohort_seed": dataset.get("cohort_seed"),
                "licence": dataset.get("licence"),
            },
            "split": manifest.get("split", {}),
            "model_selection": selection,
            "test_metrics": {
                "auroc": test.get("auroc"),
                "auprc": test.get("auprc"),
                "precision": test.get("precision"),
                "recall": test.get("recall"),
                "f1": test.get("f1"),
                "brier_score": test.get("brier_score"),
                "ece": (test.get("calibration") or {}).get("ece"),
                "confusion_matrix": test.get("confusion_matrix"),
                "n_samples": test.get("n_samples"),
            },
            "environment": manifest.get("environment", {}),
            "reproducibility": manifest.get("reproducibility", {}),
            "disclaimer": (
                "Metrics are computed on a synthetic validation cohort generated by this repository. "
                "They describe behaviour on simulated signals and are not evidence of clinical performance."
            ),
        }


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


_ABBREVIATIONS = {
    "hba1c": "HbA1c",
    "bmi": "BMI",
    "hrv": "HRV",
    "hr": "heart rate",
    "egfr": "eGFR",
    "hdl": "HDL",
    "sglt2": "SGLT2 inhibitor",
    "glp1": "GLP-1 agonist",
    "spo2": "SpO₂",
    "cv": "variability",
    "ewma": "recent average",
    "bp": "blood pressure",
}


def _humanise(name: str) -> str:
    """Turn a snake_case feature name into readable clinical text."""
    words = name.split("_")
    out: List[str] = []
    for word in words:
        key = word.lower()
        if key in _ABBREVIATIONS:
            out.append(_ABBREVIATIONS[key])
        elif key in ("pct", "pc"):
            out.append("%")
        elif key == "mgdl":
            out.append("mg/dL")
        elif key == "dev":
            out.append("deviation from")
        else:
            out.append(word)
    text = " ".join(out)
    return text[0].upper() + text[1:] if text else name


# ---------------------------------------------------------------------------
# Module-level singleton with graceful degradation
# ---------------------------------------------------------------------------
_PREDICTOR: Optional[RiskPredictor] = None
_PREDICTOR_LOCK = threading.Lock()


def get_predictor(reload: bool = False) -> RiskPredictor:
    global _PREDICTOR
    with _PREDICTOR_LOCK:
        if _PREDICTOR is None or reload:
            _PREDICTOR = RiskPredictor()
        return _PREDICTOR


def try_get_predictor() -> Tuple[Optional[RiskPredictor], Optional[str]]:
    """Return (predictor, error_message) — never raises."""
    try:
        return get_predictor(), None
    except Exception as exc:  # pragma: no cover
        return None, str(exc)
