from __future__ import annotations

import re
from typing import Any


DIMENSION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "price": ("price", "cost", "fare", "rs", "inr", "$", "usd"),
    "duration": ("duration", "hours", "mins", "journey time"),
    "availability": ("available", "availability", "seats", "booking", "running"),
}

TRAVEL_MODE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "flight": ("flight", "airline", "airport"),
    "train": ("train", "rail", "railway", "irctc"),
    "bus": ("bus", "volvo", "sleeper", "ac bus"),
}

MODE_STOPWORDS = {"find", "from", "to", "on", "the", "a", "an", "and"}


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _tokenize_place(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if token not in MODE_STOPWORDS]


def _contains_all_tokens(text: str, tokens: list[str]) -> bool:
    return all(token in text for token in tokens)


def _contains_any_token(text: str, tokens: list[str]) -> bool:
    return any(token in text for token in tokens)


def _looks_like_international_place(text: str) -> bool:
    normalized = (text or "").strip().lower()
    international_markers = {
        "dubai",
        "abu dhabi",
        "singapore",
        "london",
        "paris",
        "new york",
        "toronto",
        "bangkok",
        "doha",
        "riyadh",
        "kuwait",
        "muscat",
        "sydney",
        "melbourne",
        "usa",
        "uk",
        "uae",
        "canada",
        "australia",
        "germany",
        "france",
    }
    return any(marker in normalized for marker in international_markers)


def _result_matches_mode(text: str, mode: str) -> bool:
    normalized_text = (text or "").lower()
    if mode == "travel":
        return True
    keywords = TRAVEL_MODE_KEYWORDS.get(mode, ())
    if not any(keyword in normalized_text for keyword in keywords):
        return False
    competing_modes = [other_mode for other_mode in TRAVEL_MODE_KEYWORDS if other_mode != mode]
    if mode == "flight":
        return True
    return not any(
        any(keyword in normalized_text for keyword in TRAVEL_MODE_KEYWORDS[other_mode])
        and not any(keyword in normalized_text for keyword in keywords)
        for other_mode in competing_modes
    )


def _result_matches_route(text: str, route: dict[str, str], mode: str) -> bool:
    normalized_text = (text or "").lower()
    origin_tokens = _tokenize_place(route.get("origin", ""))
    destination_tokens = _tokenize_place(route.get("destination", ""))
    if mode == "flight":
        if origin_tokens and destination_tokens:
            if _contains_all_tokens(normalized_text, origin_tokens) and _contains_all_tokens(normalized_text, destination_tokens):
                return True
            if _contains_all_tokens(normalized_text, destination_tokens) and _contains_any_token(normalized_text, origin_tokens):
                return True
            if _contains_all_tokens(normalized_text, origin_tokens) and _contains_any_token(normalized_text, destination_tokens):
                return True
            return False
        if origin_tokens and not _contains_any_token(normalized_text, origin_tokens):
            return False
        if destination_tokens and not _contains_any_token(normalized_text, destination_tokens):
            return False
        return True
    if origin_tokens and not _contains_all_tokens(normalized_text, origin_tokens):
        return False
    if destination_tokens and not _contains_all_tokens(normalized_text, destination_tokens):
        return False
    return True


def _route_is_feasible(route: dict[str, str], mode: str) -> bool:
    if mode not in {"train", "bus"}:
        return True
    return not (
        _looks_like_international_place(route.get("origin", ""))
        or _looks_like_international_place(route.get("destination", ""))
    )


def parse_tavily_text(search_text: str) -> list[dict[str, str]]:
    if not search_text or search_text.startswith("Error:") or search_text.startswith("Tavily request failed"):
        return []
    sections = re.split(r"\n(?=\[\d+\]\s)", search_text.strip())
    results: list[dict[str, str]] = []
    for section in sections:
        section = section.strip()
        if not section.startswith("["):
            continue
        lines = section.splitlines()
        if len(lines) < 2:
            continue
        title = re.sub(r"^\[\d+\]\s*", "", lines[0]).strip()
        url_line = lines[1].strip()
        if not url_line.startswith("URL:"):
            continue
        url = url_line.replace("URL:", "", 1).strip()
        content = "\n".join(lines[2:]).strip()
        results.append({"title": title, "url": url, "content": content})
    return results


def detect_travel_mode(query: str) -> str:
    lowered = query.lower()
    for mode, keywords in TRAVEL_MODE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return mode
    return "travel"


def extract_route(query: str) -> dict[str, str]:
    text = " ".join(query.strip().split())
    match = re.search(r"(?i)\bfrom\s+(?P<origin>.+?)\s+to\s+(?P<destination>.+?)(?:\s+on\s+(?P<date>.+))?$", text)
    if not match:
        return {"origin": "", "destination": "", "date": ""}
    return {
        "origin": (match.groupdict().get("origin") or "").strip(" ?"),
        "destination": (match.groupdict().get("destination") or "").strip(" ?"),
        "date": (match.groupdict().get("date") or "").strip(" ?"),
    }


def _detect_dimensions(text: str) -> list[str]:
    lowered = text.lower()
    dimensions = [name for name, keywords in DIMENSION_KEYWORDS.items() if any(keyword in lowered for keyword in keywords)]
    return dimensions or ["general"]


def _best_sentence_for_dimension(text: str, dimension: str) -> str:
    sentences = _split_sentences(text)
    keywords = DIMENSION_KEYWORDS.get(dimension, ())
    for sentence in sentences:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in keywords):
            return sentence
    return sentences[0] if sentences else text.strip()


def build_extracted_evidence(query: str, search_text: str) -> dict[str, Any]:
    parsed_results = parse_tavily_text(search_text)
    route = extract_route(query)
    mode = detect_travel_mode(query)
    generic_results: list[dict[str, Any]] = []
    for result in parsed_results:
        combined_text = " ".join([result["title"], result["content"]]).strip()
        combined_text_with_url = " ".join([combined_text, result.get("url", "")]).strip()
        if not _route_is_feasible(route, mode):
            continue
        if not _result_matches_mode(combined_text_with_url, mode):
            continue
        if not _result_matches_route(combined_text_with_url, route, mode):
            continue
        generic_results.append({**result, "dimensions": _detect_dimensions(combined_text)})
    return {
        "query": query,
        "route": route,
        "travel_mode": mode,
        "generic_evidence": generic_results,
        "result_count": len(generic_results),
    }


def build_grounded_travel_options(extracted: dict[str, Any]) -> list[dict[str, Any]]:
    results = extracted.get("generic_evidence") or []
    mode = extracted.get("travel_mode", "travel")
    options: list[dict[str, Any]] = []
    for item in results[:6]:
        content = item.get("content", "")
        title = item.get("title", "") or "Unknown operator"
        snippet = " ".join(_split_sentences(content)[:2]).strip()
        options.append(
            {
                "mode": mode,
                "operator": title,
                "departure": "unknown",
                "arrival": "unknown",
                "duration": _best_sentence_for_dimension(content, "duration") if content else "unknown",
                "price": _best_sentence_for_dimension(content, "price") if content else "unknown",
                "notes": snippet or "Limited travel evidence available.",
                "source_urls": [item.get("url", "")] if item.get("url") else [],
            }
        )
    return options


def score_evidence_confidence(
    extracted: dict[str, Any],
    grounded_options: list[dict[str, Any]],
) -> dict[str, Any]:
    result_count = int(extracted.get("result_count") or 0)
    score = 0
    reasons: list[str] = []
    if result_count >= 6:
        score += 2
        reasons.append("Collected a broad set of travel search results.")
    elif result_count >= 3:
        score += 1
        reasons.append("Collected a moderate number of travel search results.")
    else:
        reasons.append("Travel search coverage is limited.")

    if len(grounded_options) >= 3:
        score += 2
        reasons.append("Multiple grounded travel options were identified.")
    elif len(grounded_options) >= 1:
        score += 1
        reasons.append("At least one grounded travel option was identified.")
    else:
        reasons.append("No grounded travel options were identified.")

    missing_fields = 0
    for option in grounded_options:
        if option.get("price", "unknown") == "unknown":
            missing_fields += 1
        if option.get("duration", "unknown") == "unknown":
            missing_fields += 1
    if missing_fields:
        score -= 1
        reasons.append("Some travel options are missing price or duration details.")

    if score >= 4:
        confidence = "high"
    elif score >= 2:
        confidence = "medium"
    else:
        confidence = "low"
    return {"confidence": confidence, "score": score, "reasons": reasons}


def extracted_evidence_to_text(extracted: dict[str, Any]) -> str:
    lines: list[str] = []
    route = extracted.get("route") or {}
    if route.get("origin") and route.get("destination"):
        lines.append(f"Detected route: {route['origin']} -> {route['destination']}")
        if route.get("date"):
            lines.append(f"Detected date: {route['date']}")

    generic = extracted.get("generic_evidence") or []
    if generic:
        lines.append("")
        lines.append("Travel evidence:")
        for item in generic[:6]:
            dims = ", ".join(item.get("dimensions") or ["general"])
            lines.append(f"- [{dims}] {item.get('title', '')}")
            lines.append(f"  URL: {item.get('url', '')}")
            snippet = item.get("content", "").replace("\n", " ").strip()
            if snippet:
                lines.append(f"  Snippet: {snippet[:300]}")

    if not lines:
        return "No deterministic travel evidence could be extracted."
    return "\n".join(lines).strip()


def grounded_travel_options_to_text(options: list[dict[str, Any]]) -> str:
    if not options:
        return "No grounded travel options available."
    lines = ["Grounded travel options:"]
    for option in options:
        urls = ", ".join(option.get("source_urls") or [])
        lines.append(f"- Operator: {option.get('operator', 'Unknown operator')}")
        lines.append(f"  Mode: {option.get('mode', 'travel')}")
        lines.append(f"  Price: {option.get('price', 'unknown')}")
        lines.append(f"  Duration: {option.get('duration', 'unknown')}")
        lines.append(f"  Notes: {option.get('notes', '')}")
        if urls:
            lines.append(f"  Sources: {urls}")
    return "\n".join(lines).strip()


def confidence_summary_to_text(summary: dict[str, Any]) -> str:
    reasons = summary.get("reasons") or []
    lines = [
        f"Deterministic confidence: {summary.get('confidence', 'low')}",
        f"Deterministic score: {summary.get('score', 0)}",
    ]
    if reasons:
        lines.append("Reasons:")
        for reason in reasons:
            lines.append(f"- {reason}")
    return "\n".join(lines).strip()
