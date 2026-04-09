from __future__ import annotations

from typing import Any

from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage


def message_to_history_row(msg: BaseChatMessage | BaseAgentEvent) -> dict[str, Any]:
    """Map AgentChat messages to the dict shape expected by the Streamlit UI."""
    name = getattr(msg, "source", None) or "unknown"
    if hasattr(msg, "content") and isinstance(getattr(msg, "content", None), str):
        content = getattr(msg, "content")
    elif hasattr(msg, "to_text"):
        content = msg.to_text()
    else:
        content = str(msg)
    return {"name": name, "content": content}


def task_result_to_history(result: TaskResult) -> list[dict[str, Any]]:
    return [message_to_history_row(m) for m in result.messages]

