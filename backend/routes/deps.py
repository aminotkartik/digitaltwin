"""
Shared route dependencies: patient lookup and "which instant are we looking at".

Every patient-scoped endpoint accepts the same three optional selectors:

  ``index``       explicit sample index in the stored stream
  ``at``          ISO timestamp, or a bare clock time ("11:05") on the scenario day
  ``session_id``  replay the instant a running simulation session is at

If none is given the endpoint uses the twin's default "now"
(``VITALSYNC_DEMO_NOW_CLOCK``).  Centralising this keeps the dashboard, the
replay controls and the Groq context looking at exactly the same instant.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
from fastapi import HTTPException

from backend.models.predictor import ModelNotTrainedError, get_predictor, try_get_predictor
from backend.services.patient_service import PatientNotFoundError, PatientRecord, get_repository
from backend.services.simulation_service import SessionNotFoundError, get_simulation_service


def get_record(patient_id: str) -> PatientRecord:
    try:
        return get_repository().get(patient_id)
    except PatientNotFoundError:
        available = get_repository().ids()
        raise HTTPException(
            status_code=404,
            detail={"error": "patient_not_found", "patient_id": patient_id, "available": available},
        )


def resolve_index(
    record: PatientRecord,
    index: Optional[int] = None,
    at: Optional[str] = None,
    session_id: Optional[str] = None,
) -> int:
    """Resolve the sample index a request is asking about."""
    if session_id:
        try:
            session = get_simulation_service().get(session_id)
            if session.patient_id != record.patient_id:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "session_patient_mismatch",
                        "session_patient": session.patient_id,
                        "requested_patient": record.patient_id,
                    },
                )
            return record.clip_index(session.index)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail={"error": "session_not_found", "session_id": session_id})
    if index is not None:
        return record.clip_index(int(index))
    if at:
        try:
            timestamp = pd.Timestamp(at)
        except ValueError:
            try:
                hh, mm = str(at).split(":")[:2]
                reference = record.timestamp_at(record.now_index).normalize()
                timestamp = reference + pd.Timedelta(hours=int(hh), minutes=int(mm))
            except Exception:
                raise HTTPException(status_code=400, detail={"error": "invalid_at", "value": at})
        return record.clip_index(record.index_at(timestamp))
    return record.now_index


def require_predictor():
    predictor, error = try_get_predictor()
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_trained",
                "message": error,
                "hint": "Run: python model/train.py",
            },
        )
    return predictor


def safe_predictor():
    """Predictor if available, else None — endpoints degrade instead of failing."""
    predictor, _ = try_get_predictor()
    return predictor


def model_status() -> Dict[str, Any]:
    try:
        predictor = get_predictor()
        return {
            "trained": True,
            "model_id": predictor.model_id,
            "estimator": predictor.estimator_name,
            "primary_kind": predictor.primary_kind,
            "threshold": predictor.threshold,
            "feature_count": len(predictor.feature_names),
        }
    except ModelNotTrainedError as exc:
        return {"trained": False, "error": str(exc), "hint": "Run: python model/train.py"}
