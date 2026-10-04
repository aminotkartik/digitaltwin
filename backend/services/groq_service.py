"""
VitalSync — Groq clinical interpretation layer.

Groq is an *explanation* layer.  It never produces a probability, never sees a
raw API key on the client, and never decides what the risk is: the risk comes
from the trained model in ``prediction_service``.  Groq receives that model's
output plus the structured clinical context and writes short, clinician-facing
prose about it.

Robustness contract
-------------------
If the key is absent, the network fails, the response is malformed, or the model
returns something outside the schema, this service returns a **deterministic
template-based interpretation built from the same structured data** and marks it
``source = "deterministic-fallback"``.  The dashboard therefore never breaks
because of the LLM, and the UI can always tell the clinician which path produced
the text they are reading.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from backend.settings import settings

CANONICAL_FIELDS = [
    "headline",
    "summary",
    "key_changes",
    "risk_explanation",
    "clinical_attention",
    "monitoring_considerations",
    "data_limitations",
    "confidence_note",
]

ACTION_LABELS = {
    "summarize_changes": "Summarise Changes",
    "explain_risk": "Explain Risk",
    "review_pattern": "Review Recent Pattern",
    "clinical_brief": "Prepare Clinician Note",
}

ACTION_INSTRUCTIONS = {
    "summarize_changes": (
        "Describe what has changed in this patient's digital twin over the recent window compared with their "
        "own baseline. Focus on direction and magnitude of change, not absolute values."
    ),
    "explain_risk": (
        "Explain, for a physician, why the risk model produced this specific probability. Attribute the risk to the "
        "listed contributors and state clearly which are historical record factors and which are live-signal factors."
    ),
    "review_pattern": (
        "Review the recent pattern of glucose, activity, sleep and cardiovascular signals and describe the recurring "
        "behaviour the twin is showing, including any relationship between them."
    ),
    "clinical_brief": (
        "Prepare a short structured clinical brief a physician could paste into a note. Cover current state, recent "
        "changes, predicted risk, contributing signals, monitoring considerations and data limitations. Do not give "
        "treatment instructions; phrase everything as considerations for review."
    ),
}

SYSTEM_PROMPT = f"""You are a clinical documentation assistant embedded in {settings.app_name}, a research \
prototype digital-twin platform for predictive metabolic monitoring. Your reader is a physician reviewing one \
patient's digital twin.

HARD CONSTRAINTS
1. Use only the provided structured patient data. Do not invent measurements, diagnoses, medications, symptoms, \
test results, thresholds, timelines, or events that are not present in the data.
2. The risk probability is produced by a separate statistical model and is given to you in the "prediction" object. \
Never compute, adjust, or propose your own probability, and never cite a number that is not in the input.
3. Do not give treatment instructions, drug or dose recommendations, or diagnostic conclusions. Use phrasing such as \
"consider reviewing..." or "further clinical assessment may be appropriate...".
4. This is synthetic demonstration data from a prototype that is not clinically validated. Never claim validation, \
regulatory approval, or medical certainty.
5. Be concise and specific: headline under 12 words; summary under 70 words; every list item under 22 words.
6. Write in neutral clinical English. No emoji, no markdown, no headings.
7. Respond with a single JSON object and nothing else, using exactly these keys: {", ".join(CANONICAL_FIELDS)}. \
"key_changes", "clinical_attention", "monitoring_considerations" and "data_limitations" are arrays of short strings; \
the others are single strings. Omit no key."""


@dataclass
class InsightResult:
    payload: Dict[str, Any]
    source: str
    ok: bool
    error: Optional[str] = None
    latency_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.payload,
            "_meta": {
                "source": self.source,
                "generated_by": "groq" if self.source == "groq" else "vitalsync-deterministic-templates",
                "model": settings.groq_model if self.source == "groq" else None,
                "ok": self.ok,
                "error": self.error,
                "latency_ms": self.latency_ms,
                "groq_enabled": settings.groq_enabled,
            },
        }


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
def status() -> Dict[str, Any]:
    """Public, key-free description of the interpretation layer's state."""
    if settings.groq_force_disabled:
        reason = "Groq interpretation layer explicitly disabled (VITALSYNC_GROQ_FORCE_DISABLED)."
    elif not settings.groq_api_key.strip():
        reason = "No GROQ_API_KEY configured — deterministic prototype mode active."
    else:
        reason = "Groq interpretation layer enabled."
    return {
        "enabled": settings.groq_enabled,
        "model": settings.groq_model if settings.groq_enabled else None,
        "reason": reason,
        "fallback_active": not settings.groq_enabled,
        "note": (
            "Groq writes clinician-facing text about a prediction that a separate statistical model produced. "
            "It never generates the risk value."
        ),
        "actions": [{"key": key, "label": label} for key, label in ACTION_LABELS.items()],
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def generate_insight(context: Dict[str, Any], action: str = "summarize_changes") -> InsightResult:
    action = action if action in ACTION_INSTRUCTIONS else "summarize_changes"
    if not settings.groq_enabled:
        return InsightResult(
            payload=deterministic_insight(context, action),
            source="deterministic-fallback",
            ok=True,
            error=None if settings.groq_force_disabled else "GROQ_API_KEY not configured",
            latency_ms=0,
        )
    started = time.time()
    try:
        raw = await _call_groq(context, action)
    except httpx.TimeoutException:
        return _fallback(context, action, "Groq request timed out", started)
    except httpx.HTTPError as exc:
        return _fallback(context, action, f"Groq request failed: {exc.__class__.__name__}", started)
    except Exception as exc:  # pragma: no cover
        return _fallback(context, action, f"Groq request error: {exc}", started)

    parsed, parse_error = _parse_response(raw)
    if parse_error:
        return _fallback(context, action, parse_error, started, raw_excerpt=raw[:400])
    payload = _normalise(parsed, context, action)
    latency = int((time.time() - started) * 1000)
    return InsightResult(payload=payload, source="groq", ok=True, latency_ms=latency)


def _fallback(
    context: Dict[str, Any], action: str, error: str, started: float, raw_excerpt: Optional[str] = None
) -> InsightResult:
    return InsightResult(
        payload=deterministic_insight(context, action),
        source="deterministic-fallback",
        ok=False,
        error=error,
        latency_ms=int((time.time() - started) * 1000),
    )


async def _call_groq(context: Dict[str, Any], action: str) -> str:
    user_message = (
        f"ACTION: {action} — {ACTION_INSTRUCTIONS[action]}\n\n"
        "STRUCTURED PATIENT DATA (the only facts you may use):\n"
        f"{json.dumps(context, default=str, separators=(',', ':'))}\n\n"
        f"Return the JSON object with keys: {', '.join(CANONICAL_FIELDS)}."
    )
    payload = {
        "model": settings.groq_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": settings.groq_temperature,
        "max_tokens": settings.groq_max_tokens,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=settings.groq_timeout_seconds) as client:
        response = await client.post(
            f"{settings.groq_base_url.rstrip('/')}/chat/completions", json=payload, headers=headers
        )
    if response.status_code != 200:
        detail = response.text[:200]
        raise httpx.HTTPError(f"Groq returned HTTP {response.status_code}: {detail}")
    body = response.json()
    choices = body.get("choices") or []
    if not choices:
        raise httpx.HTTPError("Groq response contained no choices")
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise httpx.HTTPError("Groq response contained no content")
    return content


def _parse_response(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0)), None
        except json.JSONDecodeError as exc:
            return None, f"Groq returned malformed JSON ({exc.__class__.__name__})"
    return None, "Groq response was not a JSON object"


def _normalise(parsed: Dict[str, Any], context: Dict[str, Any], action: str) -> Dict[str, Any]:
    """Force the model's output into the canonical schema."""
    out: Dict[str, Any] = {}
    for field in CANONICAL_FIELDS:
        value = parsed.get(field)
        if field in ("key_changes", "clinical_attention", "monitoring_considerations", "data_limitations"):
            out[field] = _string_list(value, limit=6)
        else:
            out[field] = _clean_text(value)
    if not out["headline"]:
        out["headline"] = _default_headline(context, action)
    if not out["summary"]:
        out["summary"] = out["risk_explanation"] or out["headline"]
    out["action"] = action
    out["action_label"] = ACTION_LABELS.get(action, action)
    out["schema_complete"] = all(bool(out[f]) for f in CANONICAL_FIELDS)
    return out


def _clean_text(value: Any, limit: int = 900) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:limit]


def _string_list(value: Any, limit: int = 6) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p.strip(" -•\t") for p in re.split(r"[\n;]|\. ", value) if p.strip()]
    elif isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            if isinstance(item, dict):
                item = item.get("text") or item.get("value") or json.dumps(item)
            parts.append(_clean_text(item, 240))
    else:
        parts = [_clean_text(value)]
    return [p for p in parts if p][:limit]


# ---------------------------------------------------------------------------
# Deterministic fallback (also used as the offline demo mode)
# ---------------------------------------------------------------------------
def deterministic_insight(context: Dict[str, Any], action: str) -> Dict[str, Any]:
    """
    Template interpretation built only from numbers present in ``context``.

    This is what runs when Groq is unavailable.  It is intentionally literal:
    every sentence quotes a value from the structured payload, so it cannot
    hallucinate.
    """
    patient = context.get("patient", {})
    prediction = context.get("prediction", {})
    trends = context.get("recent_trends", {})
    sensors = context.get("latest_sensor_data", {})
    baseline = context.get("baseline_comparison", {})
    risk_factors = context.get("risk_factors", []) or []
    historical = context.get("historical_data", {})

    probability = prediction.get("probability")
    band = (prediction.get("band") or "").upper()
    trend = prediction.get("risk_trend_direction", "stable")
    contributors = prediction.get("top_contributors", []) or []
    confidence = prediction.get("confidence")
    horizon = prediction.get("horizon_minutes", settings.forecast_horizon_minutes)

    pct = f"{probability * 100:.0f}%" if isinstance(probability, (int, float)) else "unavailable"
    name = patient.get("name", "this patient")

    headline = _default_headline(context, action)

    summary_parts = [
        f"{name}'s digital twin currently estimates a {pct} probability ({band} band) of a significant glucose "
        f"elevation within the next {horizon // 60} hour{'s' if horizon // 60 != 1 else ''}.",
    ]
    if trends:
        # Two separate windows, each described with its own endpoints so the
        # sentence can never contradict the numbers it quotes.
        summary_parts.append(
            f"Over the last {_hours_word(trends.get('window_hours', 3))}, glucose moved from "
            f"{trends.get('glucose_start_mgdl', '—')} to {trends.get('glucose_end_mgdl', '—')} mg/dL "
            f"({_worded(trends.get('glucose_direction'))}), and modelled risk moved from "
            f"{_fmt_prob(trends.get('risk_start'))} to {_fmt_prob(trends.get('risk_end'))} "
            f"({_worded(trends.get('risk_direction'))})."
        )
        if prediction.get("risk_change_60min") is not None:
            summary_parts.append(
                f"In the last hour alone the predicted probability changed by "
                f"{prediction['risk_change_60min'] * 100:+.0f} percentage points."
            )
    summary = " ".join(summary_parts)

    key_changes: List[str] = []
    for row in (baseline.get("metrics") or [])[:5]:
        delta = row.get("delta_pct")
        if delta is None:
            continue
        if abs(delta) < 5:
            continue
        key_changes.append(
            f"{row.get('metric')}: {row.get('current')} {row.get('unit')} versus a personal baseline of "
            f"{row.get('baseline')} {row.get('unit')} ({delta:+.0f}%)."
        )
    if not key_changes:
        key_changes.append("No metric deviates from this patient's baseline by more than 5%.")

    if contributors:
        top = contributors[0]
        risk_explanation = (
            f"The strongest single contributor is {top.get('label', top.get('feature'))} "
            f"({top.get('value')} versus a reference of {top.get('reference_value')}), "
            f"shifting predicted risk by {top.get('contribution', 0) * 100:+.1f} percentage points. "
        )
        rest = ", ".join(c.get("label", c.get("feature")) for c in contributors[1:4])
        if rest:
            risk_explanation += f"Further contributors: {rest}. "
        groups = prediction.get("contributions_by_group") or []
        if groups:
            strongest = groups[0]
            risk_explanation += (
                f"By data stream, the {strongest.get('group_label', strongest.get('group'))} block contributes most "
                f"({strongest.get('contribution', 0) * 100:+.1f} points)."
            )
    else:
        risk_explanation = "Attribution details are unavailable for this prediction."

    clinical_attention: List[str] = []
    for row in (baseline.get("metrics") or []):
        if row.get("flag") in ("warning", "critical"):
            clinical_attention.append(
                f"{row.get('metric')} is {abs(row.get('delta_pct') or 0):.0f}% "
                f"{'above' if (row.get('delta_pct') or 0) > 0 else 'below'} this patient's baseline."
            )
    for factor in risk_factors[:4]:
        label = factor.get("label") if isinstance(factor, dict) else str(factor)
        if label:
            clinical_attention.append(str(label))
    if band == "HIGH":
        clinical_attention.insert(
            0, f"Forecast risk is in the HIGH band ({pct}); consider reviewing the next {horizon // 60} h of monitoring."
        )
    if not clinical_attention:
        clinical_attention.append("No deviation currently warrants attention beyond routine monitoring.")

    monitoring = [
        f"Continue {sensors.get('cgm_interval_minutes', settings.sampling_interval_minutes)}-minute CGM sampling "
        f"through the {horizon // 60}-hour forecast window.",
        "Further clinical assessment may be appropriate if the HIGH band persists or glucose exceeds the recorded peak.",
    ]
    if sensors.get("meal_log_present") is False:
        monitoring.append("Meal logging is incomplete; carbohydrate-dependent features are less reliable.")

    limitations = [
        "All data is synthetic and generated for demonstration; no real patient data is involved.",
        "The prototype is not clinically validated and is not a medical device.",
        f"Risk reflects a {horizon // 60}-hour forecast of a significant elevation, not a diagnosis.",
    ]
    if confidence is not None:
        limitations.append(
            f"Prediction confidence {confidence}% reflects signal completeness, model agreement and calibration error."
        )
    if historical.get("diabetes_duration_years") is None:
        limitations.append("Some historical fields are absent from the record supplied to this summary.")

    return {
        "headline": headline,
        "summary": summary,
        "key_changes": key_changes[:6],
        "risk_explanation": risk_explanation,
        "clinical_attention": clinical_attention[:6],
        "monitoring_considerations": monitoring[:5],
        "data_limitations": limitations[:6],
        "confidence_note": (
            "Generated by deterministic templates from the structured payload because the Groq layer was unavailable. "
            "No language model was used and no values were inferred beyond those supplied."
        ),
        "action": action,
        "action_label": ACTION_LABELS.get(action, action),
        "schema_complete": True,
    }


def _hours_word(hours: Any) -> str:
    try:
        value = float(hours)
    except (TypeError, ValueError):
        return "recent window"
    if value == int(value):
        return f"{int(value)} h"
    return f"{value:.1f} h"


def _worded(direction: Any) -> str:
    return {"rising": "rising", "falling": "falling", "stable": "broadly stable"}.get(str(direction), "broadly stable")


def _fmt_prob(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.0f}%"
    return "—"


def _default_headline(context: Dict[str, Any], action: str) -> str:
    prediction = context.get("prediction", {})
    band = (prediction.get("band") or "").upper()
    probability = prediction.get("probability")
    # The headline uses the short (60-minute) direction: that is the change a
    # clinician can still act on.
    direction = prediction.get("risk_trend_direction") or context.get("recent_trends", {}).get("risk_direction") or "stable"
    pct = f"{probability * 100:.0f}%" if isinstance(probability, (int, float)) else "n/a"
    if action == "clinical_brief":
        return f"Clinical brief — {band} 2-hour risk ({pct})"
    if direction == "rising":
        return f"Rising 2-hour glucose risk — {band} ({pct})"
    if direction == "falling":
        return f"Easing 2-hour glucose risk — {band} ({pct})"
    return f"{band} 2-hour glucose risk ({pct})"
