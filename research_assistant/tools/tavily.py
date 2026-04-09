from __future__ import annotations

from typing import Any

import requests

from research_assistant.config import TAVILY_API_KEY, TAVILY_SEARCH_URL, is_configured_api_key


def tavily_search(query: str, max_results: int, api_key: str) -> str:
    if not query or not str(query).strip():
        return "Error: empty query."
    if not is_configured_api_key(api_key):
        return "Error: Tavily API key is not configured."

    payload: dict[str, Any] = {
        "api_key": api_key,
        "query": str(query).strip(),
        "max_results": min(max(1, int(max_results)), 10),
        "search_depth": "basic",
        "include_answer": True,
    }
    try:
        resp = requests.post(TAVILY_SEARCH_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return f"Tavily request failed: {e}"
    except ValueError as e:
        return f"Tavily response was not valid JSON: {e}"

    lines: list[str] = []
    if data.get("answer"):
        lines.append(f"Answer (Tavily): {data['answer']}")

    for i, item in enumerate(data.get("results") or [], start=1):
        title = item.get("title") or "(no title)"
        url = item.get("url") or ""
        body = item.get("content") or item.get("snippet") or ""
        lines.append(f"\n[{i}] {title}\nURL: {url}\n{body.strip()}")

    if not lines:
        return "No results returned from Tavily."
    return "\n".join(lines).strip()


def search_web(query: str, max_results: int = 5) -> str:
    """Public web-search helper that uses the default Tavily API key."""
    return tavily_search(query, max_results, TAVILY_API_KEY)
