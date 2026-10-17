"""
VitalSync — Root FastAPI application entrypoint for Vercel and local ASGI servers.

Exports the FastAPI ``app`` instance from ``backend.main`` at a conventional
repository-root location (``main.py``) so Vercel's FastAPI runtime and local
``uvicorn main:app`` / ``uvicorn backend.main:app`` commands resolve identically.
"""
from __future__ import annotations

from backend.main import app

__all__ = ["app"]
