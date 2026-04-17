from __future__ import annotations

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


def build_researcher(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        "Researcher",
        model_client,
        description="Summarizes travel evidence for the team.",
        system_message=(
            "You are a travel research assistant. The conversation already includes the user's travel query "
            "and web evidence gathered from search. Summarize route options clearly, point out gaps "
            "or uncertainty, and include URLs that appear in the evidence so the Analyst can cite them. "
            "Do not invent operators, prices, schedules, durations, or availability that are not supported by the evidence. "
            "Group findings by travel option when possible. After sharing findings, invite the Reviewer to challenge weak or missing evidence."
        ),
    )


def build_reviewer(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        "Reviewer",
        model_client,
        description="Critiques travel evidence quality and checks for unsupported claims or missing route details.",
        system_message=(
            "You are a travel verification reviewer. Read the user's request, the web evidence, and the Researcher's summary. "
            "Your job is to challenge weak evidence, point out unsupported claims, identify unanswered route details, "
            "and note where prices, schedules, or availability may be stale, incomplete, or contradictory. "
            "Keep your response concise and action-oriented. End with either 'APPROVED FOR SYNTHESIS' or "
            "'NEEDS REVISION' so the Analyst knows whether the evidence is strong enough."
        ),
    )


def build_analyst(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        "Analyst",
        model_client,
        description="Produces final travel analysis with route options, findings, gaps, and summary.",
        system_message=(
            "You are a travel analyst. Read the conversation, the search evidence, the Researcher's summary, "
            "and the Reviewer's critique. Produce the best final answer grounded only in the available evidence. "
            "Respond in two parts:\n"
            "1) A single JSON object (valid JSON only) with keys: "
            "topic (string), request_type (string), route (object with origin, destination, date), "
            "key_findings (array of strings), "
            "travel_options (array of objects with keys mode, operator, departure, arrival, duration, price, notes, source_urls), "
            "evidence_gaps (array of strings), "
            "sources (array of objects with title and url strings), "
            "confidence (string, e.g. high/medium/low), confidence_rationale (array of strings).\n"
            "2) A short plain-language summary paragraph after the JSON.\n"
            "If grounded travel options are provided in the conversation, preserve them unless there is a clear evidence-based reason to adjust them. "
            "The route in the final JSON must match the user's requested origin, destination, and date exactly unless the request itself was ambiguous. "
            "If search evidence is noisy or references a different route, treat that as an evidence gap instead of changing the route. "
            "Use the deterministic confidence guidance in the conversation as a floor and anchor for confidence. "
            "If the evidence is incomplete, say so in evidence_gaps and lower confidence instead of guessing. "
            "Put the JSON inside a markdown fenced code block labeled json. "
            "When you are finished, end your message with TERMINATE on its own line."
        ),
    )


def build_user_proxy(query: str) -> UserProxyAgent:
    rounds = [0]

    def user_input(_prompt: str) -> str:
        rounds[0] += 1
        if rounds[0] == 1:
            return f"My travel research request:\n{query}"
        return (
            "I have no further input. Researcher: summarize the travel evidence. "
            "Reviewer: critique the evidence quality and identify route or pricing gaps. "
            "Analyst: use both to produce the JSON block and summary, then end with TERMINATE."
        )

    return UserProxyAgent(
        "User_Proxy",
        description="End user requesting travel research.",
        input_func=user_input,
    )
