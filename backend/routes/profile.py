"""
VitalSync — team / About profile route.

The About page is driven entirely by ``backend/config/profile.json``; nothing
about the team is hardcoded in the frontend.  Keys beginning with ``_`` are
instructions for whoever edits the file and are passed through untouched so the
placeholder guidance stays visible in the UI until real details are supplied.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from backend.settings import CONFIG_DIR, settings

router = APIRouter(tags=["profile"])

PROFILE_FILE = CONFIG_DIR / "profile.json"
_SECRET_PATTERNS = (
    "api_key",
    "apikey",
    "secret",
    "password",
    "passwd",
    "token",
    "authorization",
    "private_key",
    "access_key",
)


def load_profile() -> Dict[str, Any]:
    if not PROFILE_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail={"error": "profile_missing", "path": str(PROFILE_FILE), "hint": "Restore backend/config/profile.json"},
        )
    try:
        profile = json.loads(PROFILE_FILE.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "profile_invalid_json", "message": str(exc), "path": str(PROFILE_FILE)},
        )
    return _scrub(profile)


def _scrub(value: Any) -> Any:
    """Defence in depth: drop anything that looks credential-shaped."""
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items() if not _looks_secret(k)}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, str):
        return _redact(value)
    return value


def _looks_secret(key: str) -> bool:
    lowered = str(key).lower()
    return any(pattern in lowered for pattern in _SECRET_PATTERNS)


def _redact(text: str) -> str:
    if re.search(r"sk-[A-Za-z0-9]{16,}|gsk_[A-Za-z0-9]{16,}", text):
        return "[redacted — credential-shaped value removed]"
    return text


@router.get("/profile")
def get_profile() -> Dict[str, Any]:
    """Full team / project profile served to the About page."""
    profile = load_profile()
    return {
        "source": "backend/config/profile.json",
        "editable": True,
        "edit_instructions": (
            "Replace every PLACEHOLDER value in backend/config/profile.json with real team details. "
            "Keys beginning with an underscore are instructions and can be deleted. Never add credentials here."
        ),
        "placeholders_remaining": _count_placeholders(profile),
        "served_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "contains_secrets": False,
        **profile,
    }


def _count_placeholders(value: Any) -> int:
    if isinstance(value, str):
        return 1 if "PLACEHOLDER" in value else 0
    if isinstance(value, dict):
        return sum(_count_placeholders(v) for v in value.values())
    if isinstance(value, list):
        return sum(_count_placeholders(v) for v in value)
    return 0


@router.get("/profile/summary")
def profile_summary() -> Dict[str, Any]:
    """Compact version used by the landing screen and the header."""
    profile = load_profile()
    product = profile.get("product", {})
    team = profile.get("team", {})
    return {
        "product": product,
        "team_name": team.get("name"),
        "organisation": team.get("organisation"),
        "member_count": len(team.get("members", []) or []),
        "members": [
            {"name": m.get("name"), "role": m.get("role"), "avatar_initials": m.get("avatar_initials")}
            for m in (team.get("members") or [])
        ],
        "contact": profile.get("contact", {}),
        "clinical_disclaimer": profile.get("clinical_disclaimer", {}),
        "data_statement": profile.get("data_statement", {}),
    }
