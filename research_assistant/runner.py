from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import Future
from threading import Thread
from typing import Callable

from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

from research_assistant.agents import build_analyst, build_user_proxy
from research_assistant.config import (
    GROQ_BASE_URL,
    GROQ_MODEL,
    GROQ_MODEL_INFO,
    MAX_TEAM_MESSAGES,
    SEARCH_RESULTS_PER_QUERY,
)
from research_assistant.extractors import (
    build_extracted_evidence,
    build_grounded_travel_options,
    confidence_summary_to_text,
    detect_travel_mode,
    extract_route,
    extracted_evidence_to_text,
    grounded_travel_options_to_text,
    score_evidence_confidence,
)
from research_assistant.history import task_result_to_history
from research_assistant.tools.tavily import tavily_search

ProgressCallback = Callable[[float, str, str], None]


def create_model_client(groq_api_key: str) -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        model=GROQ_MODEL,
        api_key=groq_api_key,
        base_url=GROQ_BASE_URL,
        model_info=ModelInfo(**GROQ_MODEL_INFO),
        temperature=0.3,
        parallel_tool_calls=False,
        include_name_in_message=False,
    )


def infer_request_type(query: str) -> str:
    text = query.lower()
    if re.search(r"\b(train|flight|bus|ticket|travel|from|to)\b", text):
        return f"{detect_travel_mode(query)}_travel"
    return "unsupported"


def plan_search_queries(query: str, request_type: str) -> list[str]:
    normalized_query = " ".join(query.strip().split())
    route = extract_route(normalized_query)
    mode = detect_travel_mode(normalized_query)
    origin = route.get("origin") or "origin"
    destination = route.get("destination") or "destination"
    date_suffix = f" {route['date']}" if route.get("date") else ""
    planned = [
        normalized_query,
        f"{origin} to {destination} {mode} price duration availability{date_suffix}".strip(),
    ]
    if mode == "flight":
        planned.append(f"{origin} to {destination} flights airline airport fare{date_suffix}".strip())

    seen: set[str] = set()
    deduped: list[str] = []
    for item in planned:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _trim_text(text: str, limit: int) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _emit_progress(
    callback: ProgressCallback | None,
    progress: float,
    title: str,
    detail: str,
) -> None:
    if callback is not None:
        callback(progress, title, detail)


def gather_search_evidence(
    query: str,
    tavily_api_key: str,
    request_type: str,
    *,
    progress_callback: ProgressCallback | None = None,
) -> str:
    planned_queries = plan_search_queries(query, request_type)
    max_results = SEARCH_RESULTS_PER_QUERY + 1 if request_type.startswith("flight") else SEARCH_RESULTS_PER_QUERY
    _emit_progress(
        progress_callback,
        0.34,
        "Searching live providers",
        f"Running {len(planned_queries)} route checks across web sources.",
    )
    with ThreadPoolExecutor(max_workers=min(3, len(planned_queries) or 1)) as pool:
        search_results = list(pool.map(lambda planned_query: tavily_search(planned_query, max_results, tavily_api_key), planned_queries))

    sections = [
        f"Search plan [{idx}]: {planned_query}\n{_trim_text(search_text, 1600)}"
        for idx, (planned_query, search_text) in enumerate(zip(planned_queries, search_results), start=1)
    ]
    _emit_progress(
        progress_callback,
        0.56,
        "Search complete",
        "Live evidence collected. Grounding route details now.",
    )
    return "\n\n".join(sections).strip()


async def run_research_async(
    query: str,
    *,
    groq_api_key: str,
    tavily_api_key: str,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, str]]:
    model_client = create_model_client(groq_api_key)
    _emit_progress(progress_callback, 0.08, "Understanding your route", "Parsing origin, destination, mode, and travel date.")
    request_type = infer_request_type(query)
    if request_type == "unsupported":
        raise ValueError("This assistant is currently travel-focused. Use a bus, train, or flight route query.")
    route = extract_route(query)
    search_plan = plan_search_queries(query, request_type)
    _emit_progress(progress_callback, 0.18, "Planning search strategy", f"Prepared {len(search_plan)} evidence queries for this route.")
    search_evidence = gather_search_evidence(query, tavily_api_key, request_type, progress_callback=progress_callback)
    extracted_evidence = build_extracted_evidence(query, search_evidence)
    _emit_progress(progress_callback, 0.68, "Grounding route evidence", "Extracting prices, durations, and route signals from the search data.")
    extracted_evidence_text = _trim_text(extracted_evidence_to_text(extracted_evidence), 2200)
    grounded_options = build_grounded_travel_options(extracted_evidence)
    grounded_rows_text = _trim_text(grounded_travel_options_to_text(grounded_options), 1800)
    confidence_summary = score_evidence_confidence(extracted_evidence, grounded_options)
    confidence_summary_text = _trim_text(confidence_summary_to_text(confidence_summary), 1200)
    _emit_progress(progress_callback, 0.82, "Synthesizing final result", "Condensing grounded evidence into the final structured recommendation.")
    task_prompt = (
        f"User travel request:\n{query}\n\n"
        f"Detected request type:\n{request_type}\n\n"
        f"Extracted route:\nOrigin: {route.get('origin', '')}\nDestination: {route.get('destination', '')}\nDate: {route.get('date', '') or 'not specified'}\n\n"
        f"Search plan:\n" + "\n".join(f"- {item}" for item in search_plan) + "\n\n"
        f"Deterministic extracted evidence:\n{extracted_evidence_text}\n\n"
        f"Grounded travel options:\n{grounded_rows_text}\n\n"
        f"Deterministic confidence guidance:\n{confidence_summary_text}\n\n"
        f"Web search evidence:\n{_trim_text(search_evidence, 2600)}\n\n"
        "Workflow requirements:\n"
        "1. Return final travel JSON and summary directly from the grounded evidence.\n"
        "2. Use the grounded travel options as the default backbone for travel_options and only extend them with evidence-backed details.\n"
        "3. Use the deterministic confidence guidance when setting final confidence and explain any downgrade or upgrade in confidence_rationale.\n"
        "4. Never invent missing schedules, prices, durations, or availability."
    )

    user_proxy = build_user_proxy(task_prompt)
    analyst = build_analyst(model_client)

    termination = MaxMessageTermination(MAX_TEAM_MESSAGES) | TextMentionTermination("TERMINATE")
    team = SelectorGroupChat(
        [user_proxy, analyst],
        model_client=model_client,
        termination_condition=termination,
        allow_repeated_speaker=False,
    )

    try:
        result = await team.run(task=task_prompt)
        _emit_progress(progress_callback, 1.0, "Results ready", "Structured travel options and confidence notes are ready to display.")
        return task_result_to_history(result)
    finally:
        await model_client.close()


def run_research_sync(
    query: str,
    *,
    groq_api_key: str,
    tavily_api_key: str,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, str]]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            run_research_async(
                query,
                groq_api_key=groq_api_key,
                tavily_api_key=tavily_api_key,
                progress_callback=progress_callback,
            )
        )

    result_future: Future[list[dict[str, str]]] = Future()

    def _run_in_thread() -> None:
        try:
            result_future.set_result(
                asyncio.run(
                    run_research_async(
                        query,
                        groq_api_key=groq_api_key,
                        tavily_api_key=tavily_api_key,
                        progress_callback=progress_callback,
                    )
                )
            )
        except Exception as exc:
            result_future.set_exception(exc)

    worker = Thread(target=_run_in_thread, daemon=True)
    worker.start()
    return result_future.result()
