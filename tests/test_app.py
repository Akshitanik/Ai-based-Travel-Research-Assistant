import asyncio
import unittest
from unittest.mock import Mock, patch

from main import _parse_analyst_output
from research_assistant.config import is_configured_api_key
from research_assistant.runner import create_model_client, run_research_sync
from research_assistant.tools.tavily import tavily_search


class ParseAnalystOutputTests(unittest.TestCase):
    def test_parse_json_block_and_summary(self) -> None:
        data, summary = _parse_analyst_output(
            '```json\n{"topic":"x","confidence":"high","key_findings":[],"sources":[]}\n```\nSummary\nTERMINATE'
        )
        self.assertEqual(data["topic"], "x")
        self.assertEqual(summary, "Summary")

    def test_preserves_unstructured_text_when_json_missing(self) -> None:
        data, summary = _parse_analyst_output('{"topic":"x"}\nSummary\nTERMINATE')
        self.assertIsNone(data)
        self.assertEqual(summary, '{"topic":"x"}\nSummary')


class ConfigValidationTests(unittest.TestCase):
    def test_placeholder_keys_are_not_treated_as_configured(self) -> None:
        self.assertFalse(is_configured_api_key(""))
        self.assertFalse(is_configured_api_key("YOUR_GROQ_API_KEY"))
        self.assertFalse(is_configured_api_key("YOUR_TAVILY_API_KEY"))
        self.assertTrue(is_configured_api_key("real-key"))


class TavilyTests(unittest.TestCase):
    def test_invalid_json_returns_tool_error(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.side_effect = ValueError("bad json")

        with patch("research_assistant.tools.tavily.requests.post", return_value=response):
            result = tavily_search("query", 5, "key")

        self.assertIn("Tavily response was not valid JSON", result)


class RunnerTests(unittest.TestCase):
    def test_create_model_client_supports_groq_model(self) -> None:
        client = create_model_client("key")
        try:
            self.assertEqual(client.model_info["family"], "llama-3.3-70b")
        finally:
            asyncio.run(client.close())

    def test_sync_wrapper_works_inside_running_event_loop(self) -> None:
        async def exercise() -> list[dict[str, str]]:
            with patch("research_assistant.runner.run_research_async", return_value=[{"name": "Analyst", "content": "ok"}]):
                return run_research_sync("query", groq_api_key="g", tavily_api_key="t")

        result = asyncio.run(exercise())
        self.assertEqual(result, [{"name": "Analyst", "content": "ok"}])


if __name__ == "__main__":
    unittest.main()
