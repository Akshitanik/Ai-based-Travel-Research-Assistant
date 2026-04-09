"""
Streamlit UI for the multi-agent research assistant.

Run from the project root (Windows). This avoids "streamlit is not recognized" when the
venv is not on PATH:

    .\\venv\\Scripts\\python.exe -m streamlit run main.py

Install Streamlit into the venv first if needed:

    .\\venv\\Scripts\\python.exe -m pip install -r requirements.txt

After `venv\\Scripts\\activate`, you can use `streamlit run main.py` instead.
"""

from __future__ import annotations

import json
import re
from typing import Any

import streamlit as st

from backend import run_research as backend_run_research
from research_assistant.config import GROQ_API_KEY, TAVILY_API_KEY, is_configured_api_key


def run_research(
    query: str,
    *,
    groq_api_key: str | None = None,
    tavily_api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Run the agent group chat and return message history (API keys optional overrides)."""
    return backend_run_research(
        query,
        groq_api_key=groq_api_key,
        tavily_api_key=tavily_api_key,
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


def _render_analyst_output(content: str) -> None:
    data, narrative = _parse_analyst_output(content)

    if data:
        st.subheader("Structured analysis")
        topic = data.get("topic", "-")
        confidence = data.get("confidence", "-")
        st.markdown(f"**Topic:** {topic}  \n**Confidence:** {confidence}")

        findings = data.get("key_findings")
        if isinstance(findings, list) and findings:
            st.markdown("**Key findings**")
            for item in findings:
                if isinstance(item, str) and item.strip():
                    st.markdown(f"- {item.strip()}")

        sources = data.get("sources")
        if isinstance(sources, list) and sources:
            st.markdown("**Sources**")
            for s in sources:
                if isinstance(s, dict):
                    title = s.get("title") or "Link"
                    url = s.get("url") or ""
                    if url:
                        st.markdown(f"- [{title}]({url})")
                    else:
                        st.markdown(f"- {title}")

        with st.expander("Raw JSON"):
            st.json(data)

    if narrative:
        st.subheader("Summary")
        st.markdown(narrative)

    if not data and not narrative:
        st.info("No Analyst output could be parsed; see raw message below.")

    with st.expander("Full Analyst message"):
        st.markdown(content)


def main() -> None:
    st.set_page_config(
        page_title="Multi-Agent Research Assistant",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("AI-Powered Multi-Agent Research Assistant")
    st.caption(" Enter a research query and let the agents do the work!")

    query = st.text_area(
        "Research query",
        height=120,
        placeholder="e.g. Compare iPhone 15 vs S24",
    )

    if st.button("Start Research", type="primary"):
        if not is_configured_api_key(GROQ_API_KEY) or not is_configured_api_key(TAVILY_API_KEY):
            st.warning("Please set valid `GROQ_API_KEY` and `TAVILY_API_KEY` environment variables before starting.")
            return
        if not (query or "").strip():
            st.warning("Please enter a research query.")
            return

        try:
            with st.spinner("Agents are researching..."):
                history = run_research(
                    query.strip(),
                )
        except Exception as e:
            st.error(f"Research run failed: {e}")
            return

        analyst_text = _last_analyst_message(history)
        if analyst_text:
            st.divider()
            _render_analyst_output(analyst_text)
        else:
            st.warning("No message from the Analyst was found in the chat history.")
            with st.expander("Full chat history"):
                st.json(history)


if __name__ == "__main__":
    main()
