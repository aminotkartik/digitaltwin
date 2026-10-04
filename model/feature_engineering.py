"""
Training-side feature engineering entry point.

The canonical implementation lives in ``backend/services/feature_engineering.py``
so that training and serving can never diverge (a classic cause of silent
production failure in clinical ML).  This module re-exports it and adds the
couple of helpers that only the training pipeline needs.

Usage:
    from model.feature_engineering import build_feature_frame, FEATURE_NAMES
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backend.services.feature_engineering import (  # noqa: E402,F401
    FEATURE_GROUPS,
    FEATURE_NAMES,
    FEATURE_SPECS,
    build_feature_frame,
    build_features_at,
    compute_personal_baseline,
    normalise_ehr,
    spec_for,
)


def feature_catalogue() -> List[Dict[str, Any]]:
    """Serialisable description of every model input (used by docs and UI)."""
    return [
        {"name": s.name, "group": s.group, "group_label": FEATURE_GROUPS.get(s.group, s.group), "unit": s.unit, "description": s.description}
        for s in FEATURE_SPECS
    ]


def group_feature_names() -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for spec in FEATURE_SPECS:
        grouped.setdefault(spec.group, []).append(spec.name)
    return grouped


def describe_matrix(matrix: pd.DataFrame) -> Dict[str, Any]:
    """Sanity summary of a built feature matrix (used by the training log)."""
    feature_block = matrix[FEATURE_NAMES]
    return {
        "rows": int(len(matrix)),
        "columns": int(feature_block.shape[1]),
        "missing_cells": int(feature_block.isna().sum().sum()),
        "non_finite_cells": int((~np.isfinite(feature_block.to_numpy(dtype=float))).sum()),
        "constant_features": [c for c in FEATURE_NAMES if feature_block[c].nunique(dropna=True) <= 1],
    }
