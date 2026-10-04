"""
VitalSync — model evaluation utilities.

Every number shown on the "Model validation" page of the UI is produced here,
from the actual synthetic cohort that the model was trained on.  Nothing is
hand-written or aspirational: if the cohort changes, the reported metrics
change with it.

Run standalone to re-report metrics from saved artifacts:
    python model/evaluate.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ARTIFACT_DIR = REPO_ROOT / "model" / "artifacts"

from backend.settings import settings as _settings  # noqa: E402

# Display bands must match the API exactly, so they are read from settings.
BANDS = {
    "low": (0.0, _settings.risk_moderate_min),
    "moderate": (_settings.risk_moderate_min, _settings.risk_high_min),
    "high": (_settings.risk_high_min, 1.0),
}


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": tp + tn + fp + fn}


def classification_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Threshold-independent and threshold-dependent performance measures."""
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    youden_threshold = float(thresholds[int(np.argmax(j))]) if len(thresholds) else threshold

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    specificity = cm["tn"] / max(cm["tn"] + cm["fp"], 1)
    npv = cm["tn"] / max(cm["tn"] + cm["fn"], 1)

    metrics: Dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "positive_rate": round(float(y_true.mean()), 4),
        "decision_threshold": round(float(threshold), 4),
        "auroc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "auprc": round(float(average_precision_score(y_true, y_prob)), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "sensitivity": round(recall, 4),
        "specificity": round(float(specificity), 4),
        "ppv": round(precision, 4),
        "npv": round(float(npv), 4),
        "accuracy": round(float((y_pred == y_true).mean()), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "log_loss": round(float(log_loss(y_true, y_prob, labels=[0, 1])), 4),
        "youden_optimal_threshold": round(youden_threshold, 4),
        "confusion_matrix": cm,
        "predicted_positive_rate": round(float(y_pred.mean()), 4),
        "roc_curve": {
            "fpr": [round(float(v), 4) for v in fpr[:: max(1, len(fpr) // 120)]],
            "tpr": [round(float(v), 4) for v in tpr[:: max(1, len(tpr) // 120)]],
        },
        "risk_band_distribution": _band_distribution(y_prob),
    }
    metrics["calibration"] = calibration_summary(y_true, y_prob)
    return metrics


def _band_distribution(y_prob: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for band, (lo, hi) in BANDS.items():
        mask = (y_prob >= lo) & (y_prob < hi if band != "high" else y_prob <= hi)
        out[band] = {"count": int(mask.sum()), "share": round(float(mask.mean()), 4)}
    return out


def calibration_curve(
    y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10
) -> Dict[str, List[Any]]:
    """Empirical reliability diagram (equal-width probability bins)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    predicted: List[float] = []
    observed: List[float] = []
    counts: List[int] = []
    centres: List[float] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        counts.append(int(mask.sum()))
        centres.append(round(float((lo + hi) / 2), 4))
        if mask.sum() == 0:
            predicted.append(None)
            observed.append(None)
        else:
            predicted.append(round(float(y_prob[mask].mean()), 4))
            observed.append(round(float(y_true[mask].mean()), 4))
    return {"bin_centres": centres, "mean_predicted": predicted, "observed_frequency": observed, "counts": counts}


def calibration_summary(y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10) -> Dict[str, Any]:
    curve = calibration_curve(y_true, y_prob, n_bins)
    obs = np.array([v for v in curve["observed_frequency"] if v is not None])
    pred = np.array([v for v in curve["mean_predicted"] if v is not None])
    counts = np.array([c for c, o in zip(curve["counts"], curve["observed_frequency"]) if o is not None], dtype=float)
    if obs.size == 0:
        return {"ece": None, "mce": None, "curve": curve}
    weights = counts / max(counts.sum(), 1)
    ece = float(np.sum(weights * np.abs(obs - pred)))
    mce = float(np.max(np.abs(obs - pred)))
    return {
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "interpretation": (
            "Expected calibration error: the average absolute gap between predicted risk and "
            "observed event frequency across deciles of predicted risk."
        ),
        "curve": curve,
    }


def lead_time_analysis(
    y_true: Sequence[int], y_prob: Sequence[float], threshold: float
) -> Dict[str, Any]:
    """
    How early does the model raise a flag before an event actually occurs?

    Operates on consecutive runs of samples: a "detection" is the first sample
    of a flagged run whose event window contains a positive label.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    flagged = y_prob >= threshold
    lead_times: List[int] = []
    i = 0
    n = len(y_true)
    while i < n:
        if flagged[i] and y_true[i] == 1:
            # walk forward to the end of this flagged run and count how many
            # consecutive positive-label samples it covers
            j = i
            while j < n and flagged[j]:
                j += 1
            lead_times.append(j - i)
            i = j
        else:
            i += 1
    if not lead_times:
        return {"detected_events": 0, "mean_flagged_run_samples": None, "note": "no flagged runs"}
    return {
        "detected_events": int(len(lead_times)),
        "mean_flagged_run_samples": round(float(np.mean(lead_times)), 2),
        "median_flagged_run_samples": float(np.median(lead_times)),
        "note": (
            "Runs are measured in scored samples (15-minute stride). A run of k samples means the "
            "model kept the alert raised for roughly k*15 minutes of the 2-hour forecast window."
        ),
    }


def evaluate_artifacts(artifact_dir: Path = ARTIFACT_DIR) -> Dict[str, Any]:
    """Re-load saved artifacts and re-report the test-set metrics."""
    import joblib

    bundle = joblib.load(artifact_dir / "twin_model.joblib")
    metrics_path = artifact_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics not found at {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    metrics["model_id"] = bundle.get("model_id")
    metrics["estimator"] = bundle.get("estimator_name")
    return metrics


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Report saved VitalSync model metrics")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args(argv)
    metrics = evaluate_artifacts(args.artifacts)
    test = metrics.get("test", {})
    print(f"Model            : {metrics.get('model_id')} ({metrics.get('estimator')})")
    print(f"Test samples     : {test.get('n_samples')}  positives {test.get('n_positive')} ({test.get('positive_rate')})")
    print(f"AUROC            : {test.get('auroc')}")
    print(f"AUPRC            : {test.get('auprc')}")
    print(f"Precision/Recall : {test.get('precision')} / {test.get('recall')}")
    print(f"F1               : {test.get('f1')}")
    print(f"Brier            : {test.get('brier_score')}")
    print(f"ECE              : {(test.get('calibration') or {}).get('ece')}")
    print(f"Confusion matrix : {test.get('confusion_matrix')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
