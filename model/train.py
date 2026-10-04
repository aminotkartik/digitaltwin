"""
VitalSync — risk model training.

Trains the 2-hour significant-glucose-elevation risk model on the synthetic
cohort and writes every artifact the backend needs at inference time.

Two models are fitted on purpose:

  * **primary**   — gradient boosting (XGBoost when available, otherwise
                    scikit-learn's HistGradientBoosting).  Captures the
                    non-linear interactions between longitudinal EHR state and
                    the current physiological window.
  * **baseline**  — L2 logistic regression on standardised features.  Fully
                    transparent (one coefficient per feature) and used both as
                    a comparison point on the validation page and as the
                    disagreement term in the prediction-confidence estimate.

Methodology notes that matter for a clinical reviewer:

  * Splits are **by patient**, never by row, so no patient appears in both
    train and test (GroupShuffleSplit).
  * The operating threshold is chosen on a validation subset of the training
    patients and then applied unchanged to the held-out test patients.
  * Probability calibration is measured (ECE + reliability diagram) and, if
    poor, corrected with a sigmoid calibrator fitted by cross-validation.
  * Feature importance is **permutation importance on held-out patients**, not
    training-set impurity gain, which is biased toward high-cardinality
    features.

Usage:
    python model/train.py                 # uses the existing cohort
    python model/train.py --regenerate    # rebuild the cohort first
    python model/train.py --quick         # small cohort, fast iteration
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import evaluate as ev  # noqa: E402
from model.feature_engineering import FEATURE_NAMES, feature_catalogue  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "datasets" / "synthetic"))
import generate_cohort  # noqa: E402  (datasets/synthetic/generate_cohort.py)

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.feature_selection import VarianceThreshold  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, log_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ARTIFACT_DIR = REPO_ROOT / "model" / "artifacts"
RANDOM_STATE = 20260117

try:  # XGBoost is optional — the pipeline degrades gracefully without it
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
def build_primary_estimator(seed: int = RANDOM_STATE):
    """Gradient-boosted trees; XGBoost when installed, else sklearn's HistGB."""
    if XGBOOST_AVAILABLE:
        model = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.75,
            min_child_weight=8,
            reg_lambda=2.0,
            reg_alpha=0.2,
            gamma=0.4,
            tree_method="hist",
            device="cpu",
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=4,
        )
        return model, "xgboost-3.x-hist"
    model = HistGradientBoostingClassifier(
        max_iter=450,
        learning_rate=0.06,
        max_depth=None,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.5,
        early_stopping=False,
        random_state=seed,
    )
    return model, "sklearn-histgradientboosting"


def build_baseline_estimator(seed: int = RANDOM_STATE) -> Pipeline:
    """Transparent logistic-regression baseline."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=1e-8)),
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=0.35,
                    solver="lbfgs",
                    max_iter=4000,
                    random_state=seed,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def split_by_patient(
    cohort: pd.DataFrame, test_size: float = 0.22, val_size: float = 0.18, seed: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Two patient-level splits: train/hold-out, then train/validation."""
    groups = cohort["patient_id"].to_numpy()
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(outer.split(cohort, groups=groups))
    train = cohort.iloc[train_idx]
    test = cohort.iloc[test_idx]

    inner = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + 1)
    tr_idx, va_idx = next(inner.split(train, groups=train["patient_id"].to_numpy()))
    return train.iloc[tr_idx], train.iloc[va_idx], test


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """Pick the operating point that maximises F1 on the validation patients."""
    grid = np.arange(0.20, 0.81, 0.01)
    scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in grid]
    best = int(np.argmax(scores))
    return float(grid[best]), {
        "method": "max-F1 on patient-level validation split",
        "grid": [round(float(t), 3) for t in grid[::5]],
        "f1_at_grid": [round(float(s), 4) for s in scores[::5]],
        "selected": round(float(grid[best]), 3),
        "best_f1": round(float(scores[best]), 4),
    }


def train(
    cohort: Optional[pd.DataFrame] = None,
    dataset_stats: Optional[Dict[str, Any]] = None,
    regenerate: bool = False,
    quick: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    if regenerate or cohort is None:
        if cohort is None and not regenerate and generate_cohort.FEATURES_FILE.exists():
            cohort = generate_cohort.load_cohort()
            dataset_stats = json.loads(generate_cohort.METADATA_FILE.read_text())
            if verbose:
                print(f"Loaded existing cohort: {len(cohort):,} rows")
        else:
            n_patients = 18 if quick else generate_cohort.DEFAULTS["n_patients"]
            days = 5 if quick else generate_cohort.DEFAULTS["days"]
            if verbose:
                print(f"Generating cohort ({n_patients} patients x {days} days) ...")
            cohort, _patients, dataset_stats = generate_cohort.build_cohort(
                n_patients=n_patients, days=days, verbose=verbose
            )
            generate_cohort.write_cohort(cohort, _patients, dataset_stats)

    if "label" not in cohort.columns:
        raise ValueError("cohort has no label column")
    cohort = cohort[cohort["label"].isin([0, 1])].copy()

    X_all = cohort[FEATURE_NAMES].astype(float)
    y_all = cohort["label"].astype(int).to_numpy()

    train_df, val_df, test_df = split_by_patient(cohort)
    X_tr, y_tr = train_df[FEATURE_NAMES].astype(float), train_df["label"].astype(int).to_numpy()
    X_va, y_va = val_df[FEATURE_NAMES].astype(float), val_df["label"].astype(int).to_numpy()
    X_te, y_te = test_df[FEATURE_NAMES].astype(float), test_df["label"].astype(int).to_numpy()

    if verbose:
        print(
            f"Split (by patient): train {len(train_df):,} rows / {train_df['patient_id'].nunique()} patients, "
            f"val {len(val_df):,} / {val_df['patient_id'].nunique()}, "
            f"test {len(test_df):,} / {test_df['patient_id'].nunique()}"
        )
        print(f"Positive rate: train {y_tr.mean():.3f}  val {y_va.mean():.3f}  test {y_te.mean():.3f}")

    # ---------------- fit both candidate models ----------------
    boosting, boosting_name = build_primary_estimator()
    logistic = build_baseline_estimator()
    if verbose:
        print(f"Fitting candidates: {boosting_name} and logistic-regression baseline")
    boosting.fit(X_tr, y_tr)
    logistic.fit(X_tr, y_tr)

    candidates = {
        "gradient_boosting": {"name": boosting_name, "model": boosting},
        "logistic_regression": {"name": "sklearn-logistic-regression-L2(C=0.35)", "model": logistic},
    }
    for key, cand in candidates.items():
        prob = cand["model"].predict_proba(X_va)[:, 1]
        cand["val_auroc"] = float(roc_auc_score(y_va, prob))
        cand["val_log_loss"] = float(log_loss(y_va, prob, labels=[0, 1]))
        cand["val_ece"] = ev.calibration_summary(y_va, prob)["ece"] or 0.0

    # ---------------- operating threshold ----------------
    # Chosen on the validation patients (never on the test patients) using the
    # candidate that will end up primary; the same threshold is then applied to
    # both models so their comparison stays fair.
    boosting_val_prob = candidates["gradient_boosting"]["model"].predict_proba(X_va)[:, 1]
    threshold, threshold_info = select_threshold(y_va, boosting_val_prob)
    if verbose:
        print(f"Operating threshold (max-F1 on validation patients): {threshold:.2f}")

    # ---------------- calibration of the boosted model ----------------
    calibrated = False
    boosting_name_final = boosting_name
    if candidates["gradient_boosting"]["val_ece"] > 0.05:
        from sklearn.calibration import CalibratedClassifierCV

        if verbose:
            print(
                f"Boosted model validation ECE {candidates['gradient_boosting']['val_ece']:.3f} > 0.05 "
                "-> fitting sigmoid calibrator"
            )
        calibrator = CalibratedClassifierCV(build_primary_estimator()[0], method="sigmoid", cv=3)
        calibrator.fit(X_tr, y_tr)
        cal_prob = calibrator.predict_proba(X_va)[:, 1]
        new_ece = ev.calibration_summary(y_va, cal_prob)["ece"] or 1.0
        if new_ece < candidates["gradient_boosting"]["val_ece"]:
            candidates["gradient_boosting"]["model"] = calibrator
            candidates["gradient_boosting"]["val_ece"] = new_ece
            candidates["gradient_boosting"]["val_auroc"] = float(roc_auc_score(y_va, cal_prob))
            boosting_name_final = f"{boosting_name}+sigmoid-calibration"
            calibrated = True
            if verbose:
                print(f"Calibrator accepted (ECE {new_ece:.3f})")
    candidates["gradient_boosting"]["name"] = boosting_name_final

    # ---------------- which model goes live? ----------------
    # Rule (fixed before looking at test data): the model with the better
    # validation AUROC wins, unless the two are within 0.005 AUROC of each
    # other, in which case the better-calibrated model wins.  Clinical decision
    # support is only trustworthy if the probability *means* something, so
    # calibration breaks ties.
    gb = candidates["gradient_boosting"]
    lr = candidates["logistic_regression"]
    auroc_gap = abs(gb["val_auroc"] - lr["val_auroc"])
    if auroc_gap > 0.005:
        primary_key = "gradient_boosting" if gb["val_auroc"] > lr["val_auroc"] else "logistic_regression"
        rationale = (
            f"validation AUROC differs by {auroc_gap:.4f} (> 0.005); the higher-scoring model was selected"
        )
    else:
        primary_key = "gradient_boosting" if gb["val_ece"] <= lr["val_ece"] else "logistic_regression"
        rationale = (
            f"validation AUROC is statistically indistinguishable (gap {auroc_gap:.4f} <= 0.005); "
            f"the better-calibrated model was selected "
            f"(ECE {gb['val_ece']:.4f} boosting vs {lr['val_ece']:.4f} logistic)"
        )
    secondary_key = "logistic_regression" if primary_key == "gradient_boosting" else "gradient_boosting"
    primary = candidates[primary_key]["model"]
    secondary = candidates[secondary_key]["model"]
    if verbose:
        print(f"Selected primary model: {primary_key} ({candidates[primary_key]['name']})")
        print(f"  rationale: {rationale}")

    # ---------------- evaluation ----------------
    test_prob = primary.predict_proba(X_te)[:, 1]
    secondary_test_prob = secondary.predict_proba(X_te)[:, 1]
    train_prob = primary.predict_proba(X_tr)[:, 1]
    val_prob = primary.predict_proba(X_va)[:, 1]

    metrics = {
        "train": ev.classification_metrics(y_tr, train_prob, threshold),
        "validation": ev.classification_metrics(y_va, val_prob, threshold),
        "test": ev.classification_metrics(y_te, test_prob, threshold),
        "test_by_model": {
            primary_key: ev.classification_metrics(y_te, test_prob, threshold),
            secondary_key: ev.classification_metrics(y_te, secondary_test_prob, threshold),
        },
        "model_agreement": {
            "mean_absolute_difference": round(float(np.mean(np.abs(test_prob - secondary_test_prob))), 4),
            "p90_absolute_difference": round(float(np.percentile(np.abs(test_prob - secondary_test_prob), 90)), 4),
            "note": (
                "Disagreement between the two independently trained models is used as one component of the "
                "prediction-confidence estimate shown to the clinician."
            ),
        },
        "lead_time": _lead_time(test_df, test_prob, threshold),
        "candidate_selection": {
            "rule": "higher validation AUROC; ties within 0.005 broken by lower validation ECE",
            "rationale": rationale,
            "selected": primary_key,
            "candidates": {
                key: {
                    "name": cand["name"],
                    "validation_auroc": round(cand["val_auroc"], 4),
                    "validation_log_loss": round(cand["val_log_loss"], 4),
                    "validation_ece": round(cand["val_ece"], 4),
                }
                for key, cand in candidates.items()
            },
        },
    }
    if verbose:
        t = metrics["test"]
        print(
            f"TEST  AUROC {t['auroc']:.3f}  AUPRC {t['auprc']:.3f}  F1 {t['f1']:.3f}  "
            f"precision {t['precision']:.3f}  recall {t['recall']:.3f}  Brier {t['brier_score']:.3f}  "
            f"ECE {t['calibration']['ece']:.3f}"
        )
        for key, m in metrics["test_by_model"].items():
            print(f"      {key:22s} AUROC {m['auroc']:.3f}  F1 {m['f1']:.3f}  ECE {m['calibration']['ece']:.3f}")

    # ---------------- risk-conditional trajectory --------------------------
    conditional = _conditional_trajectory(val_df, val_prob)

    # ---------------- permutation importance on held-out patients ----------
    if verbose:
        print("Computing permutation importance on the held-out patients ...")
    sample = test_df.sample(min(len(test_df), 4000), random_state=RANDOM_STATE)
    perm = permutation_importance(
        primary,
        sample[FEATURE_NAMES].astype(float),
        sample["label"].astype(int).to_numpy(),
        scoring="roc_auc",
        n_repeats=6,
        random_state=RANDOM_STATE,
        n_jobs=4,
    )
    importance = sorted(
        (
            {
                "feature": name,
                "importance_mean": round(float(m), 5),
                "importance_std": round(float(s), 5),
            }
            for name, m, s in zip(FEATURE_NAMES, perm.importances_mean, perm.importances_std)
        ),
        key=lambda row: row["importance_mean"],
        reverse=True,
    )

    coefficients = _logistic_coefficients(logistic)

    # ---------------- artifacts ----------------
    model_id = _model_id(primary_key, dataset_stats)
    feature_medians = {name: float(np.nanmedian(X_tr[name].to_numpy())) for name in FEATURE_NAMES}
    feature_stats = {
        name: {
            "median": float(np.nanmedian(X_tr[name].to_numpy())),
            "p10": float(np.nanpercentile(X_tr[name].to_numpy(), 10)),
            "p90": float(np.nanpercentile(X_tr[name].to_numpy(), 90)),
            "mean": float(np.nanmean(X_tr[name].to_numpy())),
            "std": float(np.nanstd(X_tr[name].to_numpy())),
            "min": float(np.nanmin(X_tr[name].to_numpy())),
            "max": float(np.nanmax(X_tr[name].to_numpy())),
        }
        for name in FEATURE_NAMES
    }

    bundle = {
        "model_id": model_id,
        "estimator_name": candidates[primary_key]["name"],
        "primary": primary,
        "primary_kind": primary_key,
        "baseline": logistic,
        "secondary": secondary,
        "secondary_kind": secondary_key,
        "feature_names": FEATURE_NAMES,
        "feature_medians": feature_medians,
        "feature_stats": feature_stats,
        "threshold": threshold,
        "threshold_info": threshold_info,
        "model_selection": metrics["candidate_selection"],
        "conditional_trajectory": conditional,
        "calibrated": calibrated,
        "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "event_definition": (dataset_stats or {}).get("event_definition", {}),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "xgboost_available": XGBOOST_AVAILABLE,
        "sklearn_version": __import__("sklearn").__version__,
        "python_version": platform.python_version(),
    }
    joblib.dump(bundle, ARTIFACT_DIR / "twin_model.joblib", compress=3)

    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (ARTIFACT_DIR / "feature_importance.json").write_text(
        json.dumps(
            {
                "method": "permutation importance (ROC-AUC drop), 6 repeats, held-out patients",
                "n_samples_used": int(len(sample)),
                "primary_model": candidates[primary_key]["name"],
                "primary_kind": primary_key,
                "primary": importance,
                "baseline_logistic_coefficients": coefficients,
            },
            indent=2,
        )
    )
    (ARTIFACT_DIR / "conditional_trajectory.json").write_text(json.dumps(conditional, indent=2))
    (ARTIFACT_DIR / "calibration.json").write_text(
        json.dumps(
            {
                "test": metrics["test"]["calibration"],
                "validation": metrics["validation"]["calibration"],
                "calibrator_applied": calibrated,
                "validation_ece_before_calibration": round(float(gb["val_ece"] or 0.0), 4),
            },
            indent=2,
        )
    )

    manifest = {
        "model_id": model_id,
        "trained_at": bundle["trained_at"],
        "estimator": candidates[primary_key]["name"],
        "estimator_kind": primary_key,
        "secondary_estimator": candidates[secondary_key]["name"],
        "model_selection": metrics["candidate_selection"],
        "target": "P(significant glucose elevation within 120 min)",
        "feature_count": len(FEATURE_NAMES),
        "feature_catalogue": feature_catalogue(),
        "dataset": dataset_stats or {},
        "split": {
            "strategy": "GroupShuffleSplit by patient_id (no patient in two splits)",
            "train_patients": int(train_df["patient_id"].nunique()),
            "validation_patients": int(val_df["patient_id"].nunique()),
            "test_patients": int(test_df["patient_id"].nunique()),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "random_state": RANDOM_STATE,
        },
        "threshold": threshold,
        "threshold_selection": threshold_info,
        "calibration_applied": calibrated,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "xgboost_available": XGBOOST_AVAILABLE,
            "platform": platform.platform(),
        },
        "training_seconds": round(time.time() - started, 1),
        "artifacts": {
            "model": "model/artifacts/twin_model.joblib",
            "metrics": "model/artifacts/metrics.json",
            "feature_importance": "model/artifacts/feature_importance.json",
            "calibration": "model/artifacts/calibration.json",
            "manifest": "model/artifacts/train_manifest.json",
        },
        "reproducibility": {
            "cohort_seed": (dataset_stats or {}).get("cohort_seed"),
            "random_state": RANDOM_STATE,
            "command": "python model/train.py --regenerate",
        },
    }
    (ARTIFACT_DIR / "train_manifest.json").write_text(json.dumps(manifest, indent=2))

    if verbose:
        print(f"Artifacts written to {ARTIFACT_DIR} in {manifest['training_seconds']}s")
    return manifest


CONDITIONAL_BINS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]


def _conditional_trajectory(val_df: pd.DataFrame, val_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    Empirical distribution of realised glucose change, conditional on the
    model's predicted risk.

    For each decile of predicted probability we record the 10th / 50th / 90th
    percentile of ``glucose(t + tau) - glucose(t)`` measured on held-out
    patients.  This is what the dashboard draws as the forecast fan: it is an
    observed conditional distribution, not an extrapolation drawn by hand.
    """
    taus = sorted(int(c.split("_")[-1]) for c in val_df.columns if c.startswith("future_delta_"))
    work = val_df[[f"future_delta_{t}" for t in taus] + (["label"] if "label" in val_df.columns else [])].copy()
    work["_p"] = val_prob
    if work.empty:
        return {"available": False, "reason": "no forward deltas in cohort"}
    # Fixed probability bins (not quantiles): predicted risk is heavily
    # concentrated near zero, and fixed edges keep the bands interpretable.
    work["_bin"] = pd.cut(work["_p"], bins=CONDITIONAL_BINS, include_lowest=True)
    bins: List[Dict[str, Any]] = []
    for interval, group in work.groupby("_bin", observed=False):
        entry: Dict[str, Any] = {
            "risk_range": [round(float(interval.left), 3), round(float(interval.right), 3)],
            "n_samples": int(len(group)),
            "sufficient": bool(len(group) >= 25),
            "observed_event_rate": (
                round(float(group["label"].mean()), 4) if "label" in group and len(group) else None
            ),
            "mean_predicted_risk": round(float(group["_p"].mean()), 4) if len(group) else None,
            "delta": {},
        }
        for tau in taus:
            column = group[f"future_delta_{tau}"].dropna()
            if len(column) < 25:
                continue
            entry["delta"][str(tau)] = {
                "q10": round(float(np.percentile(column, 10)), 1),
                "q50": round(float(np.percentile(column, 50)), 1),
                "q90": round(float(np.percentile(column, 90)), 1),
                "mean": round(float(column.mean()), 1),
                "n": int(len(column)),
            }
        bins.append(entry)
    return {
        "available": bool(bins),
        "source": "validation patients (out-of-sample), realised glucose deltas",
        "binning": "fixed probability edges " + str(CONDITIONAL_BINS),
        "n_bins": len([b for b in bins if b["sufficient"]]),
        "taus_minutes": sorted(taus),
        "bins": bins,
        "note": (
            "Conditional distribution of realised glucose change given the predicted risk level. "
            "Used only to draw the forecast fan; it never influences the predicted probability."
        ),
    }


def _lead_time(test_df: pd.DataFrame, test_prob: np.ndarray, threshold: float, stride_minutes: int = 15) -> Dict[str, Any]:
    """
    How long the alert stays raised before an event, measured per patient.

    Runs are computed inside each patient's own time series so that a run can
    never span two patients.  Because consecutive scored samples have
    overlapping 2-hour forecast windows, a single true event produces a run of
    consecutive positive labels; the length of the flagged run that covers it is
    a proxy for usable warning time.
    """
    probs = pd.Series(test_prob, index=test_df.index)
    run_lengths: List[int] = []
    warning_minutes: List[float] = []
    for _pid, group in test_df.assign(_p=probs).groupby("patient_id", sort=False):
        group = group.sort_values("timestamp")
        flags = (group["_p"].to_numpy() >= threshold).astype(int)
        labels = group["label"].to_numpy().astype(int)
        i, n = 0, len(flags)
        while i < n:
            if flags[i] == 1:
                j = i
                while j < n and flags[j] == 1:
                    j += 1
                if labels[i:j].any():
                    run_lengths.append(j - i)
                    warning_minutes.append((j - i) * stride_minutes)
                i = j
            else:
                i += 1
    if not run_lengths:
        return {"detected_event_runs": 0}
    return {
        "detected_event_runs": int(len(run_lengths)),
        "mean_run_minutes": round(float(np.mean(warning_minutes)), 1),
        "median_run_minutes": float(np.median(warning_minutes)),
        "p90_run_minutes": round(float(np.percentile(warning_minutes, 90)), 1),
        "scoring_stride_minutes": stride_minutes,
        "note": (
            "Scored samples are 15 minutes apart and their 2-hour forecast windows overlap, so one true "
            "event produces a run of consecutive positive labels. The reported run length is the time the "
            "alert stayed raised across that event, not an independent per-event lead time."
        ),
    }


def _logistic_coefficients(pipeline: Pipeline) -> List[Dict[str, Any]]:
    """Standardised logistic-regression coefficients (the explainable baseline)."""
    try:
        scaler = pipeline.named_steps["scaler"]
        logreg = pipeline.named_steps["logreg"]
        variance = pipeline.named_steps.get("variance")
        names = list(FEATURE_NAMES)
        if variance is not None and hasattr(variance, "get_support"):
            names = [n for n, keep in zip(FEATURE_NAMES, variance.get_support()) if keep]
        coefs = logreg.coef_[0]
        rows = [
            {
                "feature": name,
                "standardised_coefficient": round(float(c), 4),
                "odds_ratio": round(float(np.exp(c)), 4),
                "direction": "increases risk" if c > 0 else "decreases risk",
            }
            for name, c in zip(names, coefs)
        ]
        return sorted(rows, key=lambda r: abs(r["standardised_coefficient"]), reverse=True)
    except Exception as exc:  # pragma: no cover
        return [{"error": str(exc)}]


def _model_id(primary_kind: str, dataset_stats: Optional[Dict[str, Any]]) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d")
    seed = (dataset_stats or {}).get("cohort_seed", RANDOM_STATE)
    digest = hashlib.sha256(f"{primary_kind}:{seed}:{stamp}".encode()).hexdigest()[:6]
    family = {"gradient_boosting": "gbm", "logistic_regression": "logreg"}.get(primary_kind, "mdl")
    return f"vitalsync-gluco2h-{family}-{stamp}-{digest}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train the VitalSync 2-hour glucose risk model")
    parser.add_argument("--regenerate", action="store_true", help="rebuild the synthetic cohort first")
    parser.add_argument("--quick", action="store_true", help="small cohort for fast iteration")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    train(regenerate=args.regenerate, quick=args.quick, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
