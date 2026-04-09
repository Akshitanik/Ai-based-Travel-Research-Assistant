from __future__ import annotations

import os

from autogen_core.models import ModelFamily

# --- API keys (replace or set GROQ_API_KEY / TAVILY_API_KEY in the environment) ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "YOUR_TAVILY_API_KEY")

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# Groq OpenAI-compatible endpoint
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Double-check the model ID in Groq docs or the dashboard before running - IDs and availability change.
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MODEL_INFO = {
    "vision": False,
    "function_calling": True,
    "json_output": True,
    "family": ModelFamily.LLAMA_3_3_70B,
    "structured_output": False,
}

# Keep hackathon token/turn usage bounded.
MAX_TEAM_MESSAGES = 8
MAX_TOOL_ITERATIONS = 4

PLACEHOLDER_API_KEYS = {
    "",
    "YOUR_GROQ_API_KEY",
    "YOUR_TAVILY_API_KEY",
}


def is_configured_api_key(value: str | None) -> bool:
    normalized = (value or "").strip()
    return bool(normalized) and normalized not in PLACEHOLDER_API_KEYS
