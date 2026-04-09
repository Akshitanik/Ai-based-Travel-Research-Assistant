from __future__ import annotations

import asyncio
from concurrent.futures import Future
from threading import Thread

from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

from research_assistant.agents import build_analyst, build_researcher, build_user_proxy
from research_assistant.config import (
    GROQ_BASE_URL,
    GROQ_MODEL,
    GROQ_MODEL_INFO,
    MAX_TEAM_MESSAGES,
)
from research_assistant.history import task_result_to_history
from research_assistant.tools.tavily import tavily_search


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


async def run_research_async(
    query: str,
    *,
    groq_api_key: str,
    tavily_api_key: str,
) -> list[dict[str, str]]:
    model_client = create_model_client(groq_api_key)
    search_evidence = tavily_search(query, 5, tavily_api_key)
    task_prompt = (
        f"User research request:\n{query}\n\n"
        f"Web search evidence:\n{search_evidence}\n\n"
        "Researcher: summarize the evidence with the URLs provided.\n"
        "Analyst: produce the required JSON and final summary."
    )

    user_proxy = build_user_proxy(task_prompt)
    researcher = build_researcher(model_client)
    analyst = build_analyst(model_client)

    termination = MaxMessageTermination(MAX_TEAM_MESSAGES) | TextMentionTermination("TERMINATE")
    team = SelectorGroupChat(
        [user_proxy, researcher, analyst],
        model_client=model_client,
        termination_condition=termination,
        allow_repeated_speaker=False,
    )

    try:
        result = await team.run(task=task_prompt)
        return task_result_to_history(result)
    finally:
        await model_client.close()


def run_research_sync(
    query: str,
    *,
    groq_api_key: str,
    tavily_api_key: str,
) -> list[dict[str, str]]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            run_research_async(
                query,
                groq_api_key=groq_api_key,
                tavily_api_key=tavily_api_key,
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
                    )
                )
            )
        except Exception as exc:
            result_future.set_exception(exc)

    worker = Thread(target=_run_in_thread, daemon=True)
    worker.start()
    return result_future.result()
