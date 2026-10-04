"""
VitalSync — digital twin routes: fused state, the DATA FUSION panel, the
"compared with your baseline" panel and the domain history used by the radar
trend.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.routes.deps import get_record, resolve_index
from backend.services import feature_engineering as fe
from backend.services import twin_service
from backend.settings import settings

router = APIRouter(prefix="/patients/{patient_id}", tags=["twin"])


@router.get("/twin")
def twin_state(
    patient_id: str,
    at: Optional[str] = None,
    index: Optional[int] = None,
    session_id: Optional[str] = None,
    include_history_hours: float = Query(6.0, ge=0.0, le=48.0),
) -> Dict[str, Any]:
    record = get_record(patient_id)
    position = resolve_index(record, index=index, at=at, session_id=session_id)
    state = twin_service.build_twin_state(record, position)
    state["history"] = _domain_history(record, position, include_history_hours)
    state["baseline"] = _public_baseline(record)
    state["feature_count"] = len(fe.FEATURE_NAMES)
    return state


def _public_baseline(record) -> Dict[str, Any]:
    """Personal baseline without the large internal lookup tables."""
    skip = {"glucose_by_hour", "hrv_by_hour", "hr_by_hour", "steps_by_hour"}
    return {k: v for k, v in record.baseline.items() if k not in skip}


def _domain_history(record, position: int, hours: float) -> Dict[str, Any]:
    if hours <= 0:
        return {"timestamps": [], "domains": {}}
    stride_samples = max(1, int(15 / settings.sampling_interval_minutes))
    start = max(0, position - int(hours * 60 / settings.sampling_interval_minutes))
    timestamps: List[str] = []
    series: Dict[str, List[float]] = {key: [] for key in twin_service.DOMAIN_LABELS}
    composite: List[float] = []
    for i in range(start, position + 1, stride_samples):
        domains = twin_service.domain_scores(record, i)
        timestamps.append(record.timestamp_at(i).isoformat())
        for key in series:
            series[key].append(round(domains[key]["score"], 1))
        composite.append(
            round(sum(domains[k]["score"] * twin_service.DOMAIN_WEIGHTS[k] for k in domains), 1)
        )
    return {"timestamps": timestamps, "domains": series, "composite": composite, "stride_minutes": 15}


@router.get("/twin/fusion")
def twin_fusion(
    patient_id: str,
    at: Optional[str] = None,
    index: Optional[int] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """The DATA FUSION panel: historical record | live stream | fused state."""
    record = get_record(patient_id)
    position = resolve_index(record, index=index, at=at, session_id=session_id)
    return twin_service.build_fusion(record, position)


@router.get("/twin/baseline")
def twin_baseline(
    patient_id: str,
    at: Optional[str] = None,
    index: Optional[int] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """"Compared with your baseline" — every comparison is patient-specific."""
    record = get_record(patient_id)
    position = resolve_index(record, index=index, at=at, session_id=session_id)
    return twin_service.build_baseline_comparison(record, position)


@router.get("/twin/domains/{domain}")
def twin_domain_detail(
    domain: str,
    patient_id: str,
    at: Optional[str] = None,
    index: Optional[int] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Drill-down for one twin domain, including the formula it is built from."""
    record = get_record(patient_id)
    position = resolve_index(record, index=index, at=at, session_id=session_id)
    domains = twin_service.domain_scores(record, position)
    if domain not in domains:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "unknown_domain",
                "requested": domain,
                "available": list(twin_service.DOMAIN_LABELS.keys()),
            },
        )
    entry = domains[domain]
    return {
        "patient_id": record.patient_id,
        "as_of": record.timestamp_at(position).isoformat(),
        **entry,
        "formula": _DOMAIN_FORMULAS.get(domain, ""),
        "inputs": {
            "feature_names": [c["term"] for c in entry["contributors"]],
            "weights": twin_service.DOMAIN_WEIGHTS,
        },
        "note": (
            "Every domain score is a deterministic function of the fused feature vector. "
            "Sub-terms are returned so any value can be recomputed by hand."
        ),
    }


_DOMAIN_FORMULAS = {
    "glucose_stability": (
        "0.30 · f(glucose vs personal median) + 0.30 · f(3-hour CV vs personal CV) + "
        "0.22 · f(time above 180 mg/dL in 24 h) + 0.18 · f(|30-minute slope|)"
    ),
    "metabolic_stability": (
        "0.34 · f(HbA1c above 5.7%) + 0.26 · prior CGM time in range + "
        "0.20 · f(12-hour mean vs personal median) + 0.20 · f(24-hour CV vs prior report CV)"
    ),
    "cardiovascular_state": (
        "0.30 · f(heart rate above personal resting median) + 0.32 · f(HRV vs personal daytime baseline) + "
        "0.23 · f(overnight HRV vs personal baseline) + 0.15 · f(SpO₂ below 95%)"
    ),
    "recovery": (
        "0.30 · f(sleep duration vs personal median) + 0.18 · f(sleep efficiency / 0.92) + "
        "0.20 · composite sleep quality index + 0.14 · f(wake episodes) + 0.18 · f(overnight HRV deviation)"
    ),
    "activity": (
        "0.34 · f(steps today vs expected by this clock time) + 0.28 · f(last 3 h vs same clock window) + "
        "0.26 · f(uninterrupted sedentary minutes) + 0.12 · f(mean MET over the last hour)"
    ),
}
