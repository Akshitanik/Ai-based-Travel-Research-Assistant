"""
Thin compatibility facade for the modular research assistant backend.

Main public API:
- run_research(query, groq_api_key=..., tavily_api_key=...) -> list[dict]
- search_web(query, max_results=5) -> str
"""

from __future__ import annotations

from typing import Callable

from research_assistant.config import GROQ_API_KEY, TAVILY_API_KEY, is_configured_api_key
from research_assistant.runner import run_research_sync
from research_assistant.tools.tavily import search_web


def run_research(
    query: str,
    *,
    groq_api_key: str | None = None,
    tavily_api_key: str | None = None,
    progress_callback: Callable[[float, str, str], None] | None = None,
) -> list[dict[str, str]]:
    gk = (groq_api_key or "").strip() or GROQ_API_KEY
    tk = (tavily_api_key or "").strip() or TAVILY_API_KEY
    if not is_configured_api_key(gk):
        raise ValueError("Groq API key is not configured.")
    if not is_configured_api_key(tk):
        raise ValueError("Tavily API key is not configured.")
    return run_research_sync(
        query,
        groq_api_key=gk,
        tavily_api_key=tk,
        progress_callback=progress_callback,
    )
