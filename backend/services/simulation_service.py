"""
VitalSync — simulation (replay) service.

The prototype streams a recorded synthetic day through the twin so a reviewer can
watch risk evolve and then see the outcome the model was forecasting.  This is a
*replay*, not a random generator: every value shown has already been produced by
the deterministic physiology simulator and stored in the patient's stream, so the
demo is reproducible and the model is scored on exactly the data it would see in
production.

Sessions live in memory, are independent of each other, and expire.  Nothing here
mutates a patient record.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.settings import settings
from backend.services import timeline_service
from backend.services.patient_service import PatientRecord, get_repository


class SessionNotFoundError(KeyError):
    pass


@dataclass
class SimulationSession:
    session_id: str
    patient_id: str
    index: int
    running: bool = False
    speed_minutes_per_tick: int = 5
    tick_interval_ms: int = 900
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")
    steps_taken: int = 0
    minutes_advanced: int = 0
    started_at_index: int = 0
    seen_event_keys: set = field(default_factory=set)
    alert_log: List[Dict[str, Any]] = field(default_factory=list)
    band_history: List[Dict[str, Any]] = field(default_factory=list)
    last_band: str = "low"
    expired: bool = False


SESSION_TTL_SECONDS = 60 * 60
_MAX_SESSIONS = 64


class SimulationService:
    def __init__(self) -> None:
        self._sessions: Dict[str, SimulationSession] = {}
        self._lock = threading.RLock()

    # ----------------------------------------------------------------- CRUD
    def _purge(self) -> None:
        now = time.time()
        for key in list(self._sessions.keys()):
            session = self._sessions[key]
            updated = datetime.fromisoformat(session.updated_at.replace("Z", ""))
            if (datetime.utcnow() - updated).total_seconds() > SESSION_TTL_SECONDS:
                session.expired = True
                self._sessions.pop(key, None)
        while len(self._sessions) > _MAX_SESSIONS:
            oldest = min(self._sessions.values(), key=lambda s: s.updated_at)
            self._sessions.pop(oldest.session_id, None)

    def get(self, session_id: str) -> SimulationSession:
        with self._lock:
            self._purge()
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._purge()
            return [self._public_state(s) for s in self._sessions.values()]

    def start(
        self,
        patient_id: Optional[str] = None,
        from_clock: Optional[str] = None,
        speed_minutes_per_tick: int = 5,
        tick_interval_ms: int = 900,
        session_id: Optional[str] = None,
    ) -> SimulationSession:
        repository = get_repository()
        patient_id = patient_id or settings.demo_patient_id
        record = repository.get(patient_id)          # raises PatientNotFoundError
        start_index = record.simulation_start_index
        if from_clock:
            try:
                hh, mm = str(from_clock).split(":")[:2]
                target = record.timestamp_at(record.now_index).normalize() + pd.Timedelta(hours=int(hh), minutes=int(mm))
                start_index = record.index_at(target)
            except ValueError:
                start_index = record.simulation_start_index

        with self._lock:
            self._purge()
            sid = session_id or f"sim-{uuid.uuid4().hex[:10]}"
            session = SimulationSession(
                session_id=sid,
                patient_id=patient_id,
                index=start_index,
                started_at_index=start_index,
                running=True,
                speed_minutes_per_tick=int(speed_minutes_per_tick),
                tick_interval_ms=int(tick_interval_ms),
                last_band=_band(float(record.risk_primary[start_index])),
            )
            self._sessions[sid] = session
            return session

    def stop(self, session_id: str) -> SimulationSession:
        session = self.get(session_id)
        with self._lock:
            session.running = False
            session.updated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        return session

    def resume(self, session_id: str) -> SimulationSession:
        session = self.get(session_id)
        with self._lock:
            session.running = True
            session.updated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        return session

    def reset(self, session_id: str) -> SimulationSession:
        session = self.get(session_id)
        repository = get_repository()
        record = repository.get(session.patient_id)
        with self._lock:
            session.index = session.started_at_index
            session.running = False
            session.steps_taken = 0
            session.minutes_advanced = 0
            session.seen_event_keys = set()
            session.alert_log = []
            session.band_history = []
            session.last_band = _band(float(record.risk_primary[session.index]))
            session.updated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        return session

    def set_speed(self, session_id: str, speed_minutes_per_tick: int, tick_interval_ms: Optional[int] = None) -> SimulationSession:
        session = self.get(session_id)
        with self._lock:
            session.speed_minutes_per_tick = int(np.clip(speed_minutes_per_tick, 1, 120))
            if tick_interval_ms is not None:
                session.tick_interval_ms = int(np.clip(tick_interval_ms, 150, 5000))
            session.updated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        return session

    # ----------------------------------------------------------------- step
    def step(self, session_id: Optional[str], minutes: Optional[int] = None) -> Dict[str, Any]:
        """Advance the replay and return everything the dashboard needs."""
        if session_id is None:
            session = self.start()
        else:
            session = self.get(session_id)
        record = get_repository().get(session.patient_id)

        advance = int(minutes if minutes is not None else session.speed_minutes_per_tick)
        advance = int(np.clip(advance, 1, 240))
        steps = max(1, int(round(advance / settings.sampling_interval_minutes)))

        with self._lock:
            previous_index = session.index
            session.index = record.clip_index(session.index + steps)
            session.steps_taken += 1
            session.minutes_advanced += int((session.index - previous_index) * settings.sampling_interval_minutes)
            session.updated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            if session.index >= record.last_index:
                session.running = False

        new_events, alerts = self._collect_events(session, record, previous_index)
        return {
            "session": self._public_state(session),
            "record": _record_summary(record, session.index),
            "advanced_minutes": int((session.index - previous_index) * settings.sampling_interval_minutes),
            "at_stream_end": session.index >= record.last_index,
            "new_events": new_events,
            "alerts": alerts,
        }

    def _collect_events(
        self, session: SimulationSession, record: PatientRecord, previous_index: int
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Detect timeline events and alerts across the range just traversed."""
        window_start = max(0, previous_index - 6)
        timeline = timeline_service.build_timeline(record, session.index, hours=1.5)
        new_events: List[Dict[str, Any]] = []
        alerts: List[Dict[str, Any]] = []
        with self._lock:
            for event in timeline["events"]:
                if event["index"] < window_start:
                    continue
                key = f"{event['timestamp']}|{event['title']}"
                if key in session.seen_event_keys:
                    continue
                session.seen_event_keys.add(key)
                new_events.append(event)
                if event["category"] in ("alert", "model", "outcome") or event["severity"] in ("warning", "critical"):
                    alert = {
                        "id": key,
                        "timestamp": event["timestamp"],
                        "clock": event["clock"],
                        "level": "critical" if event["severity"] == "critical" else "warning",
                        "title": event["title"],
                        "detail": event["detail"],
                        "category": event["category"],
                        "risk": round(float(record.risk_primary[session.index]), 4),
                    }
                    session.alert_log.append(alert)
                    alerts.append(alert)

            band = _band(float(record.risk_primary[session.index]))
            if band != session.last_band:
                session.band_history.append(
                    {
                        "timestamp": record.timestamp_at(session.index).isoformat(),
                        "from": session.last_band,
                        "to": band,
                        "risk": round(float(record.risk_primary[session.index]), 4),
                    }
                )
                session.last_band = band
            session.alert_log = session.alert_log[-40:]
            session.band_history = session.band_history[-40:]
        return new_events, alerts

    # ---------------------------------------------------------------- state
    def state(self, session_id: str) -> Dict[str, Any]:
        session = self.get(session_id)
        record = get_repository().get(session.patient_id)
        return {"session": self._public_state(session), "record": _record_summary(record, session.index)}

    def public_state(self, session: SimulationSession) -> Dict[str, Any]:
        """Serialisable view of a session (used by the routes)."""
        return self._public_state(session)

    def _public_state(self, session: SimulationSession) -> Dict[str, Any]:
        record = get_repository().get(session.patient_id)
        timestamp = record.timestamp_at(session.index)
        risk = float(record.risk_primary[session.index])
        phase = "history" if session.index < record.now_index else ("live" if session.index == record.now_index else "future-replay")
        return {
            "session_id": session.session_id,
            "patient_id": session.patient_id,
            "running": session.running,
            "index": session.index,
            "started_at_index": session.started_at_index,
            "last_index": record.last_index,
            "now_index": record.now_index,
            "timestamp": timestamp.isoformat(),
            "clock": timestamp.strftime("%H:%M:%S"),
            "date": timestamp.strftime("%a %d %b %Y"),
            "phase": phase,
            "progress_pct": round(
                100.0 * (session.index - session.started_at_index) / max(record.last_index - session.started_at_index, 1), 1
            ),
            "minutes_advanced": session.minutes_advanced,
            "steps_taken": session.steps_taken,
            "speed_minutes_per_tick": session.speed_minutes_per_tick,
            "tick_interval_ms": session.tick_interval_ms,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "risk": round(risk, 4),
            "band": _band(risk),
            "band_transitions": session.band_history,
            "alert_count": len(session.alert_log),
            "alerts": session.alert_log[-8:],
            "glucose_mgdl": int(record.frame["glucose_mgdl"].iloc[session.index]),
            "at_stream_end": session.index >= record.last_index,
        }


def _band(probability: float) -> str:
    if probability >= settings.risk_high_min:
        return "high"
    if probability >= settings.risk_moderate_min:
        return "moderate"
    return "low"


def _record_summary(record: PatientRecord, index: int) -> Dict[str, Any]:
    return {
        "patient_id": record.patient_id,
        "name": record.name,
        "timestamp": record.timestamp_at(index).isoformat(),
        "glucose_mgdl": int(record.frame["glucose_mgdl"].iloc[index]),
        "risk": round(float(record.risk_primary[index]), 4),
        "samples": int(len(record.frame)),
    }


_SERVICE: Optional[SimulationService] = None
_SERVICE_LOCK = threading.Lock()


def get_simulation_service() -> SimulationService:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = SimulationService()
        return _SERVICE
