"""
Streamlit UI for the travel research assistant.

Run from the project root (Windows). This avoids "streamlit is not recognized" when the
venv is not on PATH:

    .\\venv\\Scripts\\python.exe -m streamlit run main.py

Install Streamlit into the venv first if needed:

    .\\venv\\Scripts\\python.exe -m pip install -r requirements.txt

After `venv\\Scripts\\activate`, you can use `streamlit run main.py` instead.
"""

from __future__ import annotations

from datetime import date
from html import escape
import json
import re
from math import ceil
import time
from typing import Any
from typing import Callable

import streamlit as st

from backend import run_research as backend_run_research
from research_assistant.config import GROQ_API_KEY, TAVILY_API_KEY, is_configured_api_key

SEARCH_CACHE_TTL_SECONDS = 300


def run_research(
    query: str,
    *,
    groq_api_key: str | None = None,
    tavily_api_key: str | None = None,
    progress_callback: Callable[[float, str, str], None] | None = None,
) -> list[dict[str, Any]]:
    """Run the agent group chat and return message history (API keys optional overrides)."""
    return backend_run_research(
        query,
        groq_api_key=groq_api_key,
        tavily_api_key=tavily_api_key,
        progress_callback=progress_callback,
    )


def _last_analyst_message(history: list[dict[str, Any]]) -> str | None:
    for msg in reversed(history):
        if msg.get("name") == "Analyst":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return None


def _parse_analyst_output(content: str) -> tuple[dict[str, Any] | None, str]:
    """Split Analyst reply into structured JSON (if present) and remaining narrative text."""
    text = re.sub(r"(?i)\bTERMINATE\b", "", content).strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if not match:
        return None, text

    raw_json = match.group(1).strip()
    data: dict[str, Any] | None
    try:
        parsed = json.loads(raw_json)
        data = parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        data = None

    remainder = text[match.end() :].strip()
    return data, remainder


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        if isinstance(item, str):
            normalized = item.strip()
            if normalized:
                cleaned.append(normalized)
    return cleaned


def _clean_text_field(value: Any) -> str:
    if value is None:
        return ""
    normalized = str(value).strip()
    if normalized.lower() in {"none", "null", "n/a", "na"}:
        return ""
    return normalized


def _coerce_url_list(value: Any) -> list[str]:
    cleaned: list[str] = []
    for item in _coerce_string_list(value):
        match = re.search(r"https?://[^\s\"'<>]+", item)
        if match:
            cleaned.append(match.group(0))
    return cleaned


def _normalize_analyst_payload(data: dict[str, Any] | None) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(data, dict):
        return None, ["Structured JSON was missing or invalid."]

    issues: list[str] = []

    topic = str(data.get("topic", "")).strip()
    if not topic:
        topic = "Untitled travel result"
        issues.append("Missing topic; applied a safe fallback title.")

    request_type = str(data.get("request_type", "")).strip() or "travel"
    if request_type == "travel" and "request_type" not in data:
        issues.append("Missing request_type; defaulted to travel.")

    route_data = data.get("route") if isinstance(data.get("route"), dict) else {}
    route = {
        "origin": _clean_text_field(route_data.get("origin", "")),
        "destination": _clean_text_field(route_data.get("destination", "")),
        "date": _clean_text_field(route_data.get("date", "")),
    }

    confidence = str(data.get("confidence", "")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
        issues.append("Confidence was missing or invalid; downgraded to low.")

    normalized_sources: list[dict[str, str]] = []
    raw_sources = data.get("sources")
    if isinstance(raw_sources, list):
        for source in raw_sources:
            if not isinstance(source, dict):
                continue
            title = str(source.get("title", "")).strip() or "Untitled source"
            url_matches = _coerce_url_list([source.get("url", "")])
            url = url_matches[0] if url_matches else ""
            normalized_sources.append({"title": title, "url": url})
    elif raw_sources is not None:
        issues.append("Sources were malformed and were replaced with an empty list.")

    normalized_rows: list[dict[str, Any]] = []
    raw_rows = data.get("travel_options")
    if isinstance(raw_rows, list):
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            normalized_rows.append(
                {
                    "mode": str(row.get("mode", "")).strip(),
                    "operator": _clean_text_field(row.get("operator", "")),
                    "departure": _clean_text_field(row.get("departure", "")),
                    "arrival": _clean_text_field(row.get("arrival", "")),
                    "duration": _clean_text_field(row.get("duration", "")),
                    "price": _clean_text_field(row.get("price", "")),
                    "notes": _clean_text_field(row.get("notes", "")),
                    "source_urls": _coerce_url_list(row.get("source_urls")),
                }
            )
    elif raw_rows is not None:
        issues.append("Travel options were malformed and were replaced with an empty list.")

    normalized = {
        "topic": topic,
        "request_type": request_type,
        "route": route,
        "confidence": confidence,
        "key_findings": _coerce_string_list(data.get("key_findings")),
        "travel_options": normalized_rows,
        "evidence_gaps": _coerce_string_list(data.get("evidence_gaps")),
        "sources": normalized_sources,
        "confidence_rationale": _coerce_string_list(data.get("confidence_rationale")),
    }

    if not normalized["key_findings"]:
        issues.append("No structured key findings were available.")

    return normalized, issues


def _normalize_travel_options(data: dict[str, Any]) -> list[dict[str, str]]:
    rows = data.get("travel_options")
    normalized: list[dict[str, str]] = []
    if not isinstance(rows, list):
        return normalized
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized.append(
            {
                "Mode": str(row.get("mode", "")).strip(),
                "Operator": str(row.get("operator", "")).strip(),
                "Departure": str(row.get("departure", "")).strip(),
                "Arrival": str(row.get("arrival", "")).strip(),
                "Duration": str(row.get("duration", "")).strip(),
                "Price": str(row.get("price", "")).strip(),
                "Notes": str(row.get("notes", "")).strip(),
                "Sources": ", ".join(_coerce_string_list(row.get("source_urls"))),
            }
        )
    return [row for row in normalized if any(value for value in row.values())]


def _normalize_sources(data: dict[str, Any]) -> list[dict[str, str]]:
    sources = data.get("sources")
    normalized: list[dict[str, str]] = []
    if not isinstance(sources, list):
        return normalized
    for source in sources:
        if not isinstance(source, dict):
            continue
        title = str(source.get("title", "")).strip() or "Untitled source"
        url = str(source.get("url", "")).strip()
        domain = re.sub(r"^https?://", "", url).split("/")[0] if url else ""
        normalized.append(
            {
                "Title": title,
                "Domain": domain,
                "URL": url,
            }
        )
    return normalized


def _confidence_tone(confidence: str) -> str:
    normalized = str(confidence).strip().lower()
    if normalized == "high":
        return "High confidence: route coverage is strong."
    if normalized == "medium":
        return "Medium confidence: useful travel guidance, but some route details are incomplete."
    if normalized == "low":
        return "Low confidence: treat this travel result cautiously."
    return "Confidence was not clearly specified."


def _route_text(data: dict[str, Any]) -> str:
    route = data.get("route") if isinstance(data.get("route"), dict) else {}
    if route.get("origin") and route.get("destination"):
        route_text = f"{route['origin']} to {route['destination']}"
        if route.get("date"):
            route_text = f"{route_text} on {route['date']}"
        return route_text
    return "this route"


def _narrative_conflicts_with_route(narrative: str, data: dict[str, Any]) -> bool:
    text = (narrative or "").lower()
    route = data.get("route") if isinstance(data.get("route"), dict) else {}
    origin = _clean_text_field(route.get("origin", "")).lower()
    destination = _clean_text_field(route.get("destination", "")).lower()
    if not text or not origin or not destination:
        return False
    return origin not in text or destination not in text


def _build_structured_summary(data: dict[str, Any]) -> str:
    route_text = _route_text(data)
    options = data.get("travel_options") if isinstance(data.get("travel_options"), list) else []
    confidence = _clean_text_field(data.get("confidence", "low")).lower() or "low"
    if not options:
        return f"The analyst found limited grounded evidence for {route_text}. Confidence is {confidence}, so this result should be treated cautiously."

    first = options[0] if isinstance(options[0], dict) else {}
    operator = _clean_text_field(first.get("operator")) or "the strongest available option"
    price = _clean_text_field(first.get("price")) or "price pending on source"
    duration = _clean_text_field(first.get("duration")) or "duration not clearly available"
    mode = _clean_text_field(first.get("mode")).lower() or "travel"
    return (
        f"For {route_text}, the strongest grounded {mode} option surfaced from {operator} "
        f"with {price} and {duration}. Overall analyst confidence is {confidence} based on the currently available live evidence."
    )


def _confidence_percent(confidence: str) -> int:
    normalized = str(confidence).strip().lower()
    if normalized == "high":
        return 92
    if normalized == "medium":
        return 74
    if normalized == "low":
        return 48
    return 40


def _estimate_carbon_badge(mode: str) -> tuple[str, str]:
    normalized = str(mode).strip().lower()
    if normalized == "train":
        return "Low carbon", "Approx. 14 kg CO2e"
    if normalized == "bus":
        return "Lower carbon", "Approx. 22 kg CO2e"
    if normalized == "flight":
        return "Higher carbon", "Approx. 88 kg CO2e"
    return "Unknown footprint", "Carbon estimate unavailable"


def _why_this_deal(option: dict[str, Any]) -> str:
    reasons: list[str] = []
    price = str(option.get("price", "")).strip()
    duration = str(option.get("duration", "")).strip()
    stop_category = str(option.get("stop_category", "")).strip()
    notes = str(option.get("notes", "")).strip()
    if price:
        reasons.append(f"price evidence surfaced at {price}")
    if duration and duration.lower() != "unknown":
        reasons.append(f"duration signal captured as {duration}")
    if stop_category:
        reasons.append(stop_category.lower())
    if notes:
        reasons.append("matched supporting source notes")
    if not reasons:
        return "Selected because the analyst found the strongest available evidence for this route."
    return "Selected because it has " + ", ".join(reasons[:3]) + "."


def _build_thought_trace(query: str, mode: str, travel_style: str) -> list[str]:
    normalized_query = " ".join((query or "").split()) or "your route"
    return [
        f"Understanding route intent for {normalized_query}.",
        f"Applying {travel_style.lower()} preference weighting to {mode.lower()} options.",
        "Scanning live providers and grounding candidate routes with source evidence.",
        "Comparing price, duration, and certainty across shortlisted options.",
        "Scoring analyst confidence and packaging the final recommendation set.",
    ]


def _fallback_results(query: str) -> dict[str, Any]:
    normalized_query = " ".join((query or "").split()) or "popular route"
    return {
        "topic": f"Popular deals while live search recovers for {normalized_query}",
        "request_type": "travel",
        "route": {"origin": "", "destination": "", "date": ""},
        "confidence": "low",
        "key_findings": [
            "Live provider capacity is temporarily limited, so cached showcase deals are being displayed.",
            "Each card below is a safe fallback example for demo continuity rather than a guaranteed real-time fare.",
        ],
        "travel_options": [
            {
                "mode": "flight",
                "operator": "IndiGo",
                "departure": "07:15",
                "arrival": "09:05",
                "duration": "1h 50m",
                "price": "Rs 4,699",
                "notes": "Demo-safe fallback deal for a popular domestic route.",
                "source_urls": [],
            },
            {
                "mode": "train",
                "operator": "Vande Bharat",
                "departure": "06:00",
                "arrival": "10:35",
                "duration": "4h 35m",
                "price": "Rs 1,250",
                "notes": "Rail option shown to keep comparison flow available during provider throttling.",
                "source_urls": [],
            },
            {
                "mode": "bus",
                "operator": "IntrCity SmartBus",
                "departure": "22:30",
                "arrival": "05:30",
                "duration": "7h 00m",
                "price": "Rs 899",
                "notes": "Road option included as a resilient fallback for showcase mode.",
                "source_urls": [],
            },
        ],
        "evidence_gaps": [
            "Real-time availability and final pricing are temporarily unavailable.",
            "Fallback cards should not be treated as live booking inventory.",
        ],
        "sources": [],
        "confidence_rationale": [
            "This is a graceful degradation mode triggered by provider throttling or temporary search failure.",
            "Confidence is lowered because live evidence could not be refreshed for this run.",
        ],
    }


def _render_thought_trace(trace_steps: list[str], *, title: str = "Analyst Thought Trace") -> None:
    st.markdown(f"**{title}**")
    if not trace_steps:
        st.caption("Thought trace becomes available after a search starts.")
        return
    for idx, step in enumerate(trace_steps, start=1):
        st.markdown(f"{idx}. {step}")


def _render_progress_panel(container: Any, title: str, detail: str, progress_items: list[str]) -> None:
    log_markup = "".join(
        f"<div class='progress-item'><span class='progress-dot'></span><span>{escape(item)}</span></div>"
        for item in progress_items[-5:]
    )
    container.markdown(
        f"""
        <div class="progress-shell">
            <div class="progress-title">{escape(title)}</div>
            <div class="progress-detail">{escape(detail)}</div>
            <div class="progress-log">{log_markup}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_bullet_section(title: str, items: Any, empty_message: str) -> None:
    st.markdown(f"**{title}**")
    if isinstance(items, list):
        cleaned = [str(item).strip() for item in items if isinstance(item, str) and str(item).strip()]
    else:
        cleaned = []
    if not cleaned:
        st.caption(empty_message)
        return
    for item in cleaned:
        st.markdown(f"- {item}")


def _render_source_section(data: dict[str, Any]) -> None:
    st.markdown("**Sources**")
    sources = _normalize_sources(data)
    if not sources:
        st.caption("No structured sources were returned.")
        return

    for source in sources:
        title = source["Title"]
        url = source["URL"]
        domain = source["Domain"] or "source"
        if url:
            st.markdown(f"- [{title}]({url})")
            st.caption(domain)
        else:
            st.markdown(f"- {title}")


def _render_overview_cards(data: dict[str, Any]) -> None:
    topic = str(data.get("topic", "-")).strip() or "-"
    request_type = str(data.get("request_type", "-")).strip() or "-"
    confidence = str(data.get("confidence", "-")).strip() or "-"
    route = data.get("route") if isinstance(data.get("route"), dict) else {}
    route_text = "Route not available"
    if route.get("origin") and route.get("destination"):
        route_text = f"{route['origin']} -> {route['destination']}"
        if route.get("date"):
            route_text = f"{route_text} ({route['date']})"

    st.markdown(
        f"""
        <div class="status-banner">
            <div>
                <strong>{escape(topic)}</strong><br>
                <span>{escape(route_text)} · {escape(request_type.replace("_", " ").title())} · {escape(confidence.title())} confidence</span>
            </div>
            <div class="muted-chip">AI verified {escape(str(_confidence_percent(confidence)))}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_confidence_panel(data: dict[str, Any]) -> None:
    confidence = str(data.get("confidence", "low")).strip().lower()
    percent = _confidence_percent(confidence)
    rationale = _coerce_string_list(data.get("confidence_rationale"))
    summary = rationale[0] if rationale else _confidence_tone(confidence)
    ring_style = f"background: conic-gradient(#2d6df6 0 {percent}%, #dfe8f5 {percent}% 100%);"
    st.markdown(
        f"""
        <div class="confidence-panel">
            <div class="confidence-ring" style="{ring_style}">
                <div class="confidence-core">
                    <div class="confidence-score">{percent}%</div>
                    <div class="confidence-badge">Verified by AI</div>
                </div>
            </div>
            <div>
                <div class="insight-title">Analyst confidence</div>
                <div class="insight-copy">{escape(summary)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_overview_cards(data: dict[str, Any]) -> None:
    topic = str(data.get("topic", "-")).strip() or "-"
    request_type = str(data.get("request_type", "-")).strip() or "-"
    confidence = str(data.get("confidence", "-")).strip() or "-"
    route = data.get("route") if isinstance(data.get("route"), dict) else {}
    route_text = "Route not available"
    if route.get("origin") and route.get("destination"):
        route_text = f"{route['origin']} -> {route['destination']}"
        if route.get("date"):
            route_text = f"{route_text} ({route['date']})"

    st.markdown(
        f"""
        <div class="status-banner">
            <div>
                <strong>{escape(topic)}</strong><br>
                <span>{escape(route_text)} &middot; {escape(request_type.replace("_", " ").title())} &middot; {escape(confidence.title())} confidence</span>
            </div>
            <div class="muted-chip">AI verified {escape(str(_confidence_percent(confidence)))}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _extract_price_value(price_text: str) -> int | None:
    digits = re.findall(r"\d[\d,]*", price_text or "")
    if not digits:
        return None
    try:
        return int(digits[0].replace(",", ""))
    except ValueError:
        return None


def _extract_stop_category(option: dict[str, Any]) -> str:
    note_pool = " ".join(
        [
            str(option.get("notes", "")),
            str(option.get("duration", "")),
            str(option.get("operator", "")),
        ]
    ).lower()
    if "non-stop" in note_pool or "non stop" in note_pool:
        return "Non-stop"
    if "2 stop" in note_pool or "two stop" in note_pool or "2 stops" in note_pool:
        return "2+ stops"
    if "1 stop" in note_pool or "one stop" in note_pool or "stopover" in note_pool:
        return "1 stop"
    return "Non-stop"


def _decorate_travel_options(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw_options = data.get("travel_options")
    if not isinstance(raw_options, list):
        return []

    decorated: list[dict[str, Any]] = []
    for option in raw_options:
        if not isinstance(option, dict):
            continue
        copied = dict(option)
        copied["price_value"] = _extract_price_value(str(option.get("price", "")))
        copied["stop_category"] = _extract_stop_category(option)
        decorated.append(copied)
    return decorated


def _sort_options(options: list[dict[str, Any]], sort_order: str) -> list[dict[str, Any]]:
    if sort_order == "Price (High to Low)":
        return sorted(options, key=lambda item: item.get("price_value") or -1, reverse=True)
    if sort_order == "Operator (A-Z)":
        return sorted(options, key=lambda item: str(item.get("operator", "")).lower())
    return sorted(options, key=lambda item: item.get("price_value") if item.get("price_value") is not None else 10**12)


def _inject_app_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(197, 227, 255, 0.85), transparent 30%),
                radial-gradient(circle at top right, rgba(210, 248, 232, 0.8), transparent 28%),
                linear-gradient(180deg, #edf4fb 0%, #f9fbfd 48%, #ffffff 100%);
            color: #132238;
            font-family: "Plus Jakarta Sans", sans-serif;
        }
        .block-container {
            max-width: 1280px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3 {
            color: #122033;
            letter-spacing: -0.03em;
        }
        .hero-shell {
            position: relative;
            overflow: hidden;
            padding: 4.6rem 2.5rem 2.2rem 2.5rem;
            border-radius: 18px;
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.18), rgba(255, 255, 255, 0.34)),
                url('https://images.unsplash.com/photo-1504608524841-42fe6f032b4b?auto=format&fit=crop&w=1600&q=80') center/cover;
            box-shadow: none;
            margin-bottom: 1.2rem;
            animation: floatIn 700ms ease-out both;
        }
        .hero-shell::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(180deg, rgba(255,255,255,0.12), rgba(255,255,255,0.58));
        }
        .hero-copy {
            position: relative;
            z-index: 1;
            text-align: center;
            max-width: 860px;
            margin: 0 auto 1.8rem auto;
        }
        .hero-copy h1 {
            font-size: clamp(3.1rem, 6vw, 5.2rem);
            line-height: 0.94;
            margin: 0;
            font-weight: 800;
        }
        .hero-copy .accent {
            color: #2a62ea;
        }
        .hero-copy p {
            margin: 1rem auto 0 auto;
            font-size: 0.98rem;
            color: rgba(18, 32, 51, 0.72);
            max-width: 720px;
        }
        .glass-note {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.55);
            border: 1px solid rgba(255, 255, 255, 0.75);
            font-size: 0.82rem;
            font-weight: 600;
            color: #20334f;
            backdrop-filter: blur(10px);
        }
        .search-panel {
            position: relative;
            z-index: 1;
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid rgba(223, 231, 242, 0.95);
            backdrop-filter: blur(10px);
            padding: 1.25rem 1.25rem 1.1rem 1.25rem;
            border-radius: 20px;
            max-width: 1080px;
            margin: 0 auto;
            box-shadow: 0 18px 40px rgba(19, 34, 56, 0.10);
        }
        .search-helper {
            text-align: center;
            color: #394b63;
            font-size: 0.85rem;
            margin-top: 0.8rem;
            font-weight: 600;
        }
        .results-shell {
            margin-top: 1.25rem;
        }
        .search-stamp {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.55rem 0.8rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.74);
            border: 1px solid rgba(219, 230, 244, 0.95);
            color: #33465f;
            font-size: 0.86rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        .overview-grid {
            display: grid;
            grid-template-columns: 1.4fr 1fr 0.8fr 1.2fr;
            gap: 1rem;
            margin: 1rem 0 0.7rem 0;
        }
        .overview-card {
            background: rgba(255,255,255,0.96);
            border: 1px solid #e3ebf5;
            border-radius: 22px;
            box-shadow: 0 10px 24px rgba(19, 34, 56, 0.06);
            padding: 1rem 1.1rem;
            min-height: 124px;
        }
        .overview-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #73839a;
            font-weight: 800;
            margin-bottom: 0.7rem;
        }
        .overview-value {
            font-size: clamp(1.1rem, 2vw, 1.9rem);
            line-height: 1.12;
            color: #1a2840;
            font-weight: 800;
            word-break: break-word;
        }
        .overview-subtle {
            margin-top: 0.45rem;
            color: #6a7a91;
            font-size: 0.9rem;
            line-height: 1.35;
        }
        .status-banner {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: center;
            background: linear-gradient(135deg, rgba(255,245,220,0.95), rgba(255,255,255,0.95));
            border: 1px solid #f2ddb2;
            border-radius: 20px;
            padding: 1rem 1.1rem;
            margin: 0.8rem 0 1rem 0;
            box-shadow: 0 10px 24px rgba(19, 34, 56, 0.05);
        }
        .status-banner strong {
            color: #5f4200;
        }
        .status-banner span {
            color: #4d5566;
            font-size: 0.94rem;
            font-weight: 600;
        }
        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255,255,255,0.92);
            border: 1px solid #e3ebf5;
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin-bottom: 1rem;
            box-shadow: none;
        }
        .filter-card, .deal-card {
            background: rgba(255,255,255,0.95);
            border: 1px solid #e3ebf5;
            border-radius: 16px;
            box-shadow: none;
        }
        .filter-card {
            padding: 0.95rem 1rem;
            margin-bottom: 0.8rem;
            animation: floatIn 500ms ease-out both;
        }
        .filter-card h4 {
            margin: 0 0 0.8rem 0;
            font-size: 0.86rem;
            letter-spacing: 0.04em;
            color: #50627b;
        }
        .deal-card {
            display: grid;
            grid-template-columns: 1.7fr 0.7fr;
            overflow: hidden;
            margin-bottom: 0.8rem;
            animation: riseIn 480ms ease-out both;
        }
        .deal-main {
            padding: 1rem 1.1rem;
        }
        .deal-side {
            border-left: 1px solid #e8eef6;
            padding: 1rem 1rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 0.55rem;
        }
        .operator-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: center;
            margin-bottom: 0.85rem;
        }
        .operator-name {
            font-weight: 700;
            font-size: 1.05rem;
            color: #15243a;
        }
        .operator-meta {
            color: #697b95;
            font-size: 0.82rem;
        }
        .pill {
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            background: #eef4ff;
            color: #3564d9;
            font-weight: 600;
            font-size: 0.75rem;
            white-space: nowrap;
        }
        .timeline {
            display: grid;
            grid-template-columns: auto 1fr auto;
            gap: 1rem;
            align-items: center;
            margin-bottom: 0.8rem;
        }
        .time-block {
            min-width: 72px;
        }
        .time-value {
            font-size: 1.35rem;
            line-height: 1;
            font-weight: 700;
            color: #132238;
        }
        .time-label {
            color: #6c7d94;
            font-size: 0.82rem;
            margin-top: 0.25rem;
        }
        .timeline-track {
            text-align: center;
            color: #6c7d94;
            font-size: 0.82rem;
        }
        .track-line {
            height: 2px;
            background: linear-gradient(90deg, #d8e3f3, #9db8e9, #d8e3f3);
            border-radius: 999px;
            margin: 0.45rem 0;
        }
        .deal-notes {
            font-size: 0.88rem;
            color: #5e6e85;
            line-height: 1.45;
        }
        .price-label {
            color: #6c7d94;
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 700;
        }
        .confidence-panel {
            display: grid;
            grid-template-columns: 150px 1fr;
            gap: 1rem;
            align-items: center;
            background: rgba(255,255,255,0.95);
            border: 1px solid #e3ebf5;
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 24px rgba(19, 34, 56, 0.06);
            margin: 0.8rem 0 1rem 0;
        }
        .confidence-ring {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            margin: 0 auto;
            position: relative;
        }
        .confidence-ring::after {
            content: "";
            position: absolute;
            inset: 12px;
            border-radius: 50%;
            background: #fff;
        }
        .confidence-core {
            position: relative;
            z-index: 1;
            text-align: center;
        }
        .confidence-score {
            font-size: 1.8rem;
            font-weight: 800;
            color: #132238;
            line-height: 1;
        }
        .confidence-badge {
            display: inline-block;
            margin-top: 0.3rem;
            padding: 0.3rem 0.6rem;
            border-radius: 999px;
            background: #eefaf3;
            color: #217a47;
            font-size: 0.78rem;
            font-weight: 800;
        }
        .insight-card {
            background: rgba(255,255,255,0.95);
            border: 1px solid #e3ebf5;
            border-radius: 20px;
            box-shadow: 0 10px 24px rgba(19, 34, 56, 0.06);
            padding: 1rem 1.1rem;
            margin-bottom: 0.95rem;
        }
        .insight-title {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #607089;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }
        .insight-copy {
            color: #45566f;
            font-size: 0.92rem;
            line-height: 1.5;
        }
        .price-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: #122033;
            line-height: 1;
        }
        .muted-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            background: #f4f7fb;
            color: #54657c;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .price-sub {
            color: #6c7d94;
            font-size: 0.83rem;
        }
        .deal-link {
            display: inline-block;
            text-align: center;
            text-decoration: none;
            background: linear-gradient(135deg, #2753d7, #2f79ff);
            color: #fff !important;
            padding: 0.8rem 1rem;
            border-radius: 999px;
            font-weight: 700;
        }
        .supporting-copy {
            color: #53647b;
            font-size: 0.9rem;
        }
        .hero-footnote {
            text-align: center;
            color: #41536b;
            font-size: 0.84rem;
            margin-top: 0.4rem;
            font-weight: 600;
        }
        .progress-shell {
            background: rgba(255,255,255,0.95);
            border: 1px solid #e3ebf5;
            border-radius: 18px;
            padding: 1rem 1.05rem;
            margin: 1rem 0 1.2rem 0;
        }
        .progress-title {
            font-size: 1rem;
            font-weight: 800;
            color: #16253b;
        }
        .progress-detail {
            color: #61718a;
            font-size: 0.9rem;
            margin-top: 0.2rem;
        }
        .progress-log {
            margin-top: 0.75rem;
            display: grid;
            gap: 0.45rem;
        }
        .progress-item {
            display: flex;
            gap: 0.55rem;
            align-items: flex-start;
            color: #37495f;
            font-size: 0.88rem;
            font-weight: 600;
        }
        .progress-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: #2d6df6;
            margin-top: 0.28rem;
            flex: 0 0 auto;
        }
        .section-kicker {
            color: #607089;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-testid="stDateInputField"] {
            border-radius: 12px !important;
            min-height: 52px;
            border-color: #dbe3ef !important;
            box-shadow: none !important;
        }
        .stAlert {
            border-radius: 18px;
            border: 1px solid rgba(19, 34, 56, 0.08);
            color: #132238;
        }
        .stAlert p, .stAlert div {
            color: #132238 !important;
        }
        div[data-testid="stAlertContainer"] a {
            color: #1f55d8 !important;
        }
        div[data-testid="stForm"] {
            border: none;
            background: transparent;
            padding: 0;
        }
        div[role="radiogroup"] {
            gap: 0.45rem;
            justify-content: center;
            margin-bottom: 1rem;
        }
        div[role="radiogroup"] label {
            background: rgba(237, 241, 249, 0.95);
            border: 1px solid rgba(220, 228, 239, 0.95);
            padding: 0.75rem 1rem;
            border-radius: 999px;
            min-width: 180px;
            justify-content: center;
        }
        div[role="radiogroup"] label:has(input:checked) {
            background: #121a31;
            color: #ffffff;
            box-shadow: 0 8px 20px rgba(18, 26, 49, 0.18);
        }
        div[data-testid="stButton"] > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
            border-radius: 999px;
            background: #101628;
            border: none;
            height: 3.35rem;
            font-weight: 800;
            letter-spacing: 0.01em;
            box-shadow: none;
        }
        @keyframes floatIn {
            from {
                opacity: 0;
                transform: translateY(18px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        @keyframes riseIn {
            from {
                opacity: 0;
                transform: translateY(24px) scale(0.99);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }
        @media (max-width: 900px) {
            .overview-grid {
                grid-template-columns: 1fr;
            }
            .status-banner {
                flex-direction: column;
                align-items: flex-start;
            }
            .deal-card {
                grid-template-columns: 1fr;
            }
            .deal-side {
                border-left: none;
                border-top: 1px solid #e8eef6;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_query(origin: str, destination: str, mode: str, journey_date: date | None) -> str:
    cleaned_origin = origin.strip()
    cleaned_destination = destination.strip()
    if not cleaned_origin or not cleaned_destination:
        return ""
    base = f"find {mode} from {cleaned_origin} to {cleaned_destination}"
    if journey_date:
        base = f"{base} on {journey_date.isoformat()}"
    return base


def _render_deal_card(option: dict[str, Any], index: int) -> None:
    operator = escape(_clean_text_field(option.get("operator")) or "Travel option")
    mode = escape((_clean_text_field(option.get("mode")) or "Travel").title())
    departure = escape(_clean_text_field(option.get("departure")) or "--:--")
    arrival = escape(_clean_text_field(option.get("arrival")) or "--:--")
    duration = escape(_clean_text_field(option.get("duration")) or "Duration not available")
    price = escape(_clean_text_field(option.get("price")) or "Price on source")
    notes = escape(_clean_text_field(option.get("notes")) or "Evidence-backed option from the current search run.")
    source_urls = _coerce_url_list(option.get("source_urls"))
    primary_url = source_urls[0] if source_urls else ""
    source_label = escape(re.sub(r"^https?://", "", primary_url).split("/")[0] if primary_url else "Source available in evidence")
    stop_text = escape(str(option.get("stop_category", "Non-stop")))
    price_value = option.get("price_value")
    price_chip = f'<span class="muted-chip">Best seen {price_value:,}</span>' if isinstance(price_value, int) else ""
    carbon_title, carbon_value = _estimate_carbon_badge(option.get("mode", "travel"))
    why_copy = _why_this_deal(option)

    link_markup = (
        f'<a class="deal-link" href="{escape(primary_url, quote=True)}" target="_blank">View Deal</a>'
        if primary_url
        else '<span class="deal-link">Source Pending</span>'
    )

    st.markdown(
        f"""
        <div class="deal-card">
            <div class="deal-main">
                <div class="operator-row">
                    <div>
                        <div class="operator-name">{operator}</div>
                        <div class="operator-meta">{mode} option #{index}</div>
                    </div>
                    <div class="pill">{stop_text}</div>
                </div>
                <div class="timeline">
                    <div class="time-block">
                        <div class="time-value">{departure}</div>
                        <div class="time-label">Departure</div>
                    </div>
                    <div class="timeline-track">
                        <div>{duration}</div>
                        <div class="track-line"></div>
                        <div>via {source_label}</div>
                    </div>
                    <div class="time-block">
                        <div class="time-value">{arrival}</div>
                        <div class="time-label">Arrival</div>
                    </div>
                </div>
                <div class="deal-notes">{notes}</div>
                <div style="margin-top:0.85rem; display:flex; gap:0.5rem; flex-wrap:wrap;">
                    <span class="muted-chip">{escape(carbon_title)}</span>
                    <span class="muted-chip">{escape(carbon_value)}</span>
                </div>
            </div>
            <div class="deal-side">
                <div class="price-label">Price</div>
                <div class="price-value">{price}</div>
                <div class="price-sub">per traveler</div>
                {price_chip}
                {link_markup}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Why this deal?"):
        st.caption(why_copy)


def _render_filter_panel(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    operators = sorted({str(option.get("operator", "")).strip() for option in options if str(option.get("operator", "")).strip()})
    prices = [option.get("price_value") for option in options if isinstance(option.get("price_value"), int)]
    default_min = min(prices) if prices else 0
    default_max = max(prices) if prices else max(default_min + 1000, 1000)

    st.markdown('<div class="filter-card"><h4>PRICE RANGE</h4></div>', unsafe_allow_html=True)
    if default_min >= default_max:
        price_range = (default_min, default_max)
        if prices:
            st.caption(f"All current results are at the same extracted price: {default_min}")
        else:
            st.caption("No structured prices were extracted from the current result set.")
    else:
        price_range = st.slider(
            "Price range",
            min_value=default_min,
            max_value=default_max,
            value=(default_min, default_max),
            step=max(100, ceil((default_max - default_min + 1) / 20)),
            label_visibility="collapsed",
        )

    st.markdown('<div class="filter-card"><h4>STOPS</h4></div>', unsafe_allow_html=True)
    selected_stops = st.multiselect(
        "Stops",
        ["Non-stop", "1 stop", "2+ stops"],
        default=["Non-stop", "1 stop", "2+ stops"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="filter-card"><h4>AIRLINES / OPERATORS</h4></div>', unsafe_allow_html=True)
    selected_operators = st.multiselect(
        "Operators",
        operators,
        default=operators,
        label_visibility="collapsed",
        placeholder="All operators",
    )

    if st.button("Reset All Filters", use_container_width=True):
        st.rerun()

    filtered: list[dict[str, Any]] = []
    for option in options:
        price_value = option.get("price_value")
        if isinstance(price_value, int) and not (price_range[0] <= price_value <= price_range[1]):
            continue
        if selected_stops and option.get("stop_category") not in selected_stops:
            continue
        operator = str(option.get("operator", "")).strip()
        if selected_operators and operator and operator not in selected_operators:
            continue
        filtered.append(option)
    return filtered


def _render_results_dashboard(data: dict[str, Any]) -> None:
    options = _decorate_travel_options(data)
    col_filters, col_results = st.columns([0.9, 2.6], gap="large")
    with col_filters:
        filtered_options = _render_filter_panel(options)
    with col_results:
        sort_order = st.selectbox(
            "Sort by",
            ["Price (Low to High)", "Price (High to Low)", "Operator (A-Z)"],
            index=0,
        )
        sorted_options = _sort_options(filtered_options, sort_order)
        st.markdown(
            f"""
            <div class="results-shell">
                <div class="results-header">
                    <div>
                        <div class="section-kicker">Results</div>
                        <strong>{len(sorted_options)} options found</strong><br>
                        <span class="supporting-copy">Showing evidence-backed results from the latest agent run</span>
                    </div>
                    <div class="supporting-copy">Sorted by {escape(sort_order.lower())}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not options:
            st.info("No travel options were extracted from the analyst response.")
            return
        if not sorted_options:
            st.info("No options match the current filters.")
            return
        for index, option in enumerate(sorted_options, start=1):
            if isinstance(option, dict):
                _render_deal_card(option, index)


def _render_analyst_output(content: str) -> None:
    parsed_data, narrative = _parse_analyst_output(content)
    data, normalization_issues = _normalize_analyst_payload(parsed_data)

    if data:
        if (
            str(data.get("confidence", "")).strip().lower() == "low"
            and not _normalize_sources(data)
            and "Popular deals while live search recovers" in str(data.get("topic", ""))
        ):
            st.markdown(
                """
                <div class="status-banner">
                    <div>
                        <strong>Showcase fallback mode</strong><br>
                        <span>Live providers are temporarily unavailable, so curated demo-ready results are being shown to keep the flow stable.</span>
                    </div>
                    <div class="muted-chip">Resilient demo path</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        _render_overview_cards(data)
        _render_results_dashboard(data)

        if normalization_issues:
            st.warning("Structured output was partially normalized before display.")
            _render_bullet_section(
                "Normalization notes",
                normalization_issues,
                "No normalization issues.",
            )

        safe_summary = narrative.strip() if narrative and not _narrative_conflicts_with_route(narrative, data) else _build_structured_summary(data)
        if safe_summary:
            st.markdown("**Executive summary**")
            st.markdown(safe_summary)

        confidence_rationale = data.get("confidence_rationale")
        findings = data.get("key_findings")
        evidence_gaps = data.get("evidence_gaps")

        detail_col_1, detail_col_2 = st.columns([1.1, 0.9])
        with detail_col_1:
            st.subheader("Findings")
            _render_bullet_section(
                "Key findings",
                findings,
                "No key findings were returned.",
            )
            _render_bullet_section(
                "Evidence gaps",
                evidence_gaps,
                "No explicit evidence gaps were identified.",
            )
        with detail_col_2:
            st.subheader("Validation")
            _render_bullet_section(
                "Confidence rationale",
                confidence_rationale,
                "No confidence rationale was returned.",
            )
            _render_source_section(data)

        with st.expander("Raw JSON"):
            st.json(data)

    elif narrative:
        st.subheader("Summary")
        st.markdown(narrative)
    else:
        st.info("No Analyst output could be parsed; see raw message below.")

    with st.expander("Full Analyst message"):
        st.markdown(content)


def main() -> None:
    st.set_page_config(
        page_title="AI Travel Research Assistant",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_app_styles()

    if "selected_mode" not in st.session_state:
        st.session_state["selected_mode"] = "flight"
    if "trip_type" not in st.session_state:
        st.session_state["trip_type"] = "One-way"

    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-copy">
                <div class="glass-note">AI agents search fares, routes, and source links in one pass</div>
                <h1>AI-Powered Journeys,<br><span class="accent">Curated for You</span></h1>
                <p>Search buses, trains, and flights with a premium booking flow, transparent AI reasoning, and resilient live-route discovery.</p>
            </div>
            <div class="search-panel">
        """,
        unsafe_allow_html=True,
    )

    trip_type = st.radio(
        "Trip type",
        ["One-way", "Round-trip", "Multi-city"],
        horizontal=True,
        key="trip_type",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    with col1:
        origin = st.text_input("From", placeholder="Amritsar")
    with col2:
        destination = st.text_input("To", placeholder="Bangalore")

    col3, col4, col5 = st.columns([1.2, 1, 1])
    with col3:
        journey_date = st.date_input("Departure", value=date.today())
    with col4:
        mode = st.selectbox("Mode", ["flight", "train", "bus"], key="selected_mode")
    with col5:
        passengers = st.selectbox("Passengers", ["1 traveler", "2 travelers", "3 travelers", "4 travelers"])

    if trip_type == "Round-trip":
        return_date = st.date_input("Return", value=date.today())
        st.caption(f"Round-trip UI is active; the current backend still sends a single route query ending on {return_date.isoformat()}.")
    elif trip_type == "Multi-city":
        st.caption("Multi-city is shown for product feel right now. The backend still supports one route at a time.")

    custom_query = st.text_input(
        "Optional custom query",
        placeholder="Use this if you want to override the generated route query",
    )
    query = custom_query.strip() or _format_query(origin, destination, mode, journey_date)
    st.markdown(
        "<div class='search-helper'>Real-time search • grounded source links • analyst confidence scoring</div>",
        unsafe_allow_html=True,
    )

    st.markdown("</div></div>", unsafe_allow_html=True)

    if st.button("Search Deals", type="primary", use_container_width=True):
        if not is_configured_api_key(GROQ_API_KEY) or not is_configured_api_key(TAVILY_API_KEY):
            st.warning("Please set valid `GROQ_API_KEY` and `TAVILY_API_KEY` environment variables before starting.")
            return
        if not (query or "").strip():
            st.warning("Please enter origin and destination, or provide a custom query.")
            return
        lowered_query = query.lower()
        if not re.search(r"\b(bus|train|flight|travel|from|to)\b", lowered_query):
            st.warning("This app is currently travel-focused. Try a query like 'Find train from Delhi to Jaipur'.")
            return

        try:
            with st.spinner(f"Researching {passengers.lower()} across live sources..."):
                history = run_research(
                    query.strip(),
                )
        except Exception as e:
            st.error(f"Travel research failed: {e}")
            return

        analyst_text = _last_analyst_message(history)
        if analyst_text:
            st.divider()
            _render_analyst_output(analyst_text)
        else:
            st.warning("No message from the Analyst was found in the chat history.")
            with st.expander("Full chat history"):
                st.json(history)


def run_app() -> None:
    st.set_page_config(
        page_title="AI Travel Research Assistant",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_app_styles()

    defaults: dict[str, Any] = {
        "selected_mode": "flight",
        "trip_type": "One-way",
        "analyst_text": None,
        "history": None,
        "last_query": "",
        "last_error": "",
        "travel_style": "Balanced",
        "demo_mode": False,
        "thought_trace": [],
        "search_cache": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-copy">
                <div class="glass-note">AI agents search fares, routes, and source links in one pass</div>
                <h1>Find Your Best<br><span class="accent">Travel Deal</span></h1>
                <p>Search buses, trains, and flights with a cleaner booking-style experience, then inspect grounded results from the analyst team.</p>
            </div>
            <div class="search-panel">
        """,
        unsafe_allow_html=True,
    )

    with st.form("travel_search_form", clear_on_submit=False):
        trip_type = st.radio(
            "Trip type",
            ["One-way", "Round-trip", "Multi-city"],
            horizontal=True,
            key="trip_type",
            label_visibility="collapsed",
        )

        col1, col2 = st.columns(2)
        with col1:
            origin = st.text_input("From", placeholder="Amritsar")
        with col2:
            destination = st.text_input("To", placeholder="Bangalore")

        col3, col4, col5 = st.columns([1.2, 1, 1])
        with col3:
            journey_date = st.date_input("Departure", value=date.today())
        with col4:
            mode = st.selectbox("Mode", ["flight", "train", "bus"], key="selected_mode")
        with col5:
            passengers = st.selectbox("Passengers", ["1 traveler", "2 travelers", "3 travelers", "4 travelers"])
        travel_style = "Balanced"
        demo_mode = bool(st.session_state.get("demo_mode", False))

        if trip_type == "Round-trip":
            return_date = st.date_input("Return", value=date.today())
            st.caption(
                f"Round-trip UI is active; the current backend still sends a single route query ending on {return_date.isoformat()}."
            )
        elif trip_type == "Multi-city":
            st.caption("Multi-city is shown for product feel right now. The backend still supports one route at a time.")

        with st.expander("Advanced settings"):
            col_adv_1, col_adv_2 = st.columns([1.2, 1])
            with col_adv_1:
                travel_style = st.selectbox(
                    "Travel Style",
                    ["Balanced", "Budget Local", "Business Fast", "Greenest Route"],
                    key="travel_style",
                )
            with col_adv_2:
                demo_mode = st.toggle(
                    "Demo Mode",
                    value=bool(st.session_state.get("demo_mode", False)),
                    help="Use showcase-safe cached results instead of live providers.",
                )
            custom_query = st.text_input(
                "Custom query override",
                placeholder="Use this only if you want to override the generated route query",
            )
        query = custom_query.strip() or _format_query(origin, destination, mode, journey_date)
        if query:
            query = f"{query} with {travel_style.lower()} preference"
        st.markdown(
            "<div class='search-helper'>Real-time search &bull; grounded source links &bull; analyst confidence scoring</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='hero-footnote'>Accessible by design • mobile-friendly layout • sustainable mode comparison</div>",
            unsafe_allow_html=True,
        )
        submitted = st.form_submit_button("Search Deals", type="primary", use_container_width=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    if submitted:
        st.session_state["last_error"] = ""
        st.session_state["demo_mode"] = demo_mode
        st.session_state["thought_trace"] = _build_thought_trace(query, mode, travel_style)
        progress_host = st.empty()
        progress_bar = st.progress(0, text="Preparing your search...")
        progress_log: list[str] = []

        def handle_progress(progress: float, title: str, detail: str) -> None:
            clamped = max(0.0, min(1.0, progress))
            progress_bar.progress(int(clamped * 100), text=title)
            progress_log.append(f"{title}: {detail}")
            _render_progress_panel(progress_host, title, detail, progress_log)

        handle_progress(0.03, "Preparing your search", "Validating route details and configuring the analyst pipeline.")
        cache_key = query.strip().lower()
        cached_entry = st.session_state.get("search_cache", {}).get(cache_key)
        if not (query or "").strip():
            st.session_state["last_error"] = "Please enter origin and destination, or provide a custom query."
        elif not re.search(r"\b(bus|train|flight|travel|from|to)\b", query.lower()):
            st.session_state["last_error"] = "This app is currently travel-focused. Try a query like `Find train from Delhi to Jaipur`."
        elif (
            isinstance(cached_entry, dict)
            and (time.time() - float(cached_entry.get("created_at", 0))) <= SEARCH_CACHE_TTL_SECONDS
        ):
            handle_progress(0.55, "Using cached result", "A recent identical route search was found, so results are loading immediately.")
            st.session_state["history"] = cached_entry.get("history")
            st.session_state["analyst_text"] = cached_entry.get("analyst_text")
            st.session_state["last_query"] = query.strip()
            st.session_state["thought_trace"] = cached_entry.get("thought_trace") or st.session_state["thought_trace"]
            handle_progress(1.0, "Cached results ready", "Recent search results restored from local session cache.")
        elif demo_mode:
            handle_progress(0.45, "Loading demo-ready results", "Demo Mode bypasses live providers and prepares showcase-safe fallback cards.")
            fallback_payload = _fallback_results(query.strip())
            fallback_content = (
                "```json\n"
                + json.dumps(fallback_payload)
                + "\n```\nDemo Mode is enabled, so showcase-safe curated deals are being displayed instead of live provider results."
            )
            st.session_state["history"] = [{"name": "Analyst", "content": fallback_content}]
            st.session_state["analyst_text"] = fallback_content
            st.session_state["last_query"] = query.strip()
            handle_progress(1.0, "Demo results ready", "Curated fallback cards are ready to explore.")
        elif not is_configured_api_key(GROQ_API_KEY) or not is_configured_api_key(TAVILY_API_KEY):
            st.session_state["last_error"] = "Please set valid `GROQ_API_KEY` and `TAVILY_API_KEY` values before starting."
        else:
            try:
                history = run_research(query.strip(), progress_callback=handle_progress)
                st.session_state["history"] = history
                st.session_state["analyst_text"] = _last_analyst_message(history)
                st.session_state["last_query"] = query.strip()
                st.session_state["search_cache"][cache_key] = {
                    "created_at": time.time(),
                    "history": history,
                    "analyst_text": st.session_state["analyst_text"],
                    "thought_trace": st.session_state["thought_trace"],
                }
            except Exception as exc:
                message = str(exc)
                if "429" in message or "rate limit" in message.lower():
                    handle_progress(0.72, "Live providers are busy", "Switching to resilient showcase fallback so the experience continues.")
                    fallback_payload = _fallback_results(query.strip())
                    fallback_content = "```json\n" + json.dumps(fallback_payload) + "\n```\nWe are temporarily optimizing your route. Cached showcase deals are displayed while live providers recover."
                    st.session_state["history"] = [{"name": "Analyst", "content": fallback_content}]
                    st.session_state["analyst_text"] = fallback_content
                    st.session_state["last_query"] = query.strip()
                    st.session_state["last_error"] = "Live providers are busy right now, so showcase-mode fallback deals are being displayed."
                    st.session_state["search_cache"][cache_key] = {
                        "created_at": time.time(),
                        "history": st.session_state["history"],
                        "analyst_text": fallback_content,
                        "thought_trace": st.session_state["thought_trace"],
                    }
                    handle_progress(1.0, "Fallback ready", "Curated results are ready while live providers recover.")
                else:
                    st.session_state["last_error"] = "Live search hit a temporary issue. A stronger fallback dataset can be added if you want totally offline demo resilience."
        if st.session_state["last_error"]:
            progress_bar.empty()
            progress_host.empty()

    if st.session_state["last_error"]:
        st.warning(st.session_state["last_error"])

    analyst_text = st.session_state.get("analyst_text")
    history = st.session_state.get("history")
    thought_trace = st.session_state.get("thought_trace") or []
    if thought_trace:
        with st.expander("See how the analyst team reasoned", expanded=bool(analyst_text)):
            _render_thought_trace(thought_trace)

    if analyst_text:
        st.divider()
        if st.session_state.get("last_query"):
            st.markdown(
                f"<div class='search-stamp'>Latest search: {escape(st.session_state['last_query'])}</div>",
                unsafe_allow_html=True,
            )
        _render_analyst_output(analyst_text)
    elif history:
        st.warning("No message from the Analyst was found in the chat history.")
        with st.expander("Full chat history"):
            st.json(history)


if __name__ == "__main__":
    run_app()
