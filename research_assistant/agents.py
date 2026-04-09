from __future__ import annotations

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


def build_researcher(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        "Researcher",
        model_client,
        description="Summarizes the provided web evidence for the team.",
        system_message=(
            "You are a research assistant. The conversation already includes the user's question "
            "and web evidence gathered from search. Summarize the evidence clearly, point out gaps "
            "or uncertainty, and include URLs that appear in the evidence so the Analyst can cite them. "
            "Do not invent sources or facts that are not supported by the evidence. After sharing findings, "
            "invite the Analyst to synthesize."
        ),
    )


def build_analyst(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        "Analyst",
        model_client,
        description="Produces structured JSON plus a short narrative summary.",
        system_message=(
            "You are a data analyst. Read the conversation and the researcher's evidence. "
            "Respond in two parts:\n"
            "1) A single JSON object (valid JSON only) with keys: "
            "topic (string), key_findings (array of strings), "
            "sources (array of objects with title and url strings), "
            "confidence (string, e.g. high/medium/low).\n"
            "2) A short plain-language summary paragraph after the JSON.\n"
            "Put the JSON inside a markdown fenced code block labeled json. "
            "When you are finished, end your message with TERMINATE on its own line."
        ),
    )


def build_user_proxy(query: str) -> UserProxyAgent:
    rounds = [0]

    def user_input(_prompt: str) -> str:
        rounds[0] += 1
        if rounds[0] == 1:
            return f"My research request:\n{query}"
        return (
            "I have no further input. Researcher: finish gathering evidence if needed. "
            "Analyst: produce the JSON block and summary, then end with TERMINATE."
        )

    return UserProxyAgent(
        "User_Proxy",
        description="End user requesting research.",
        input_func=user_input,
    )
