"""
VitalSync — API router aggregation.

Every route module owns one concern; this package composes them under ``/api``.
"""
from __future__ import annotations

from fastapi import APIRouter

from backend.routes import (
    insights,
    model,
    patients,
    prediction,
    profile,
    simulation,
    system,
    timeline,
    twin,
)

api_router = APIRouter(prefix="/api")

# system & provenance (no prefix of their own)
api_router.include_router(system.router)
api_router.include_router(profile.router)

# patient-scoped views
api_router.include_router(patients.router)
api_router.include_router(twin.router)
api_router.include_router(prediction.router)
api_router.include_router(timeline.router)

# cross-cutting
api_router.include_router(simulation.router)
api_router.include_router(insights.router)
api_router.include_router(model.router)

__all__ = ["api_router"]
