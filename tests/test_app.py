import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from main import (
    _build_structured_summary,
    _clean_text_field,
    _confidence_tone,
    _coerce_string_list,
    _coerce_url_list,
    _narrative_conflicts_with_route,
    _normalize_analyst_payload,
    _normalize_sources,
    _normalize_travel_options,
    _parse_analyst_output,
)
from research_assistant.config import is_configured_api_key
from research_assistant.extractors import (
    build_extracted_evidence,
    build_grounded_travel_options,
    confidence_summary_to_text,
    detect_travel_mode,
    extract_route,
    extracted_evidence_to_text,
    grounded_travel_options_to_text,
    parse_tavily_text,
    score_evidence_confidence,
)
from research_assistant.runner import (
    create_model_client,
    gather_search_evidence,
    infer_request_type,
    plan_search_queries,
    run_research_sync,
)
from research_assistant.tools.tavily import tavily_search


class ParseAnalystOutputTests(unittest.TestCase):
    def test_parse_json_block_and_summary(self) -> None:
        data, summary = _parse_analyst_output(
            '```json\n{"topic":"x","request_type":"bus_travel","route":{"origin":"Delhi","destination":"Jaipur","date":""},"confidence":"high","key_findings":[],"travel_options":[],"evidence_gaps":[],"sources":[]}\n```\nSummary\nTERMINATE'
        )
        self.assertEqual(data["topic"], "x")
        self.assertEqual(summary, "Summary")

    def test_preserves_unstructured_text_when_json_missing(self) -> None:
        data, summary = _parse_analyst_output('{"topic":"x"}\nSummary\nTERMINATE')
        self.assertIsNone(data)
        self.assertEqual(summary, '{"topic":"x"}\nSummary')

    def test_normalize_travel_options(self) -> None:
        rows = _normalize_travel_options(
            {
                "travel_options": [
                    {
                        "mode": "bus",
                        "operator": "RedBus",
                        "departure": "unknown",
                        "arrival": "unknown",
                        "duration": "unknown",
                        "price": "$100",
                        "notes": "Based on listed pricing",
                    }
                ]
            }
        )
        self.assertEqual(
            rows,
            [
                {
                    "Mode": "bus",
                    "Operator": "RedBus",
                    "Departure": "unknown",
                    "Arrival": "unknown",
                    "Duration": "unknown",
                    "Price": "$100",
                    "Notes": "Based on listed pricing",
                    "Sources": "",
                }
            ],
        )

    def test_normalize_sources(self) -> None:
        sources = _normalize_sources(
            {
                "sources": [
                    {
                        "title": "Example Source",
                        "url": "https://example.com/article",
                    }
                ]
            }
        )
        self.assertEqual(
            sources,
            [
                {
                    "Title": "Example Source",
                    "Domain": "example.com",
                    "URL": "https://example.com/article",
                }
            ],
        )

    def test_confidence_tone(self) -> None:
        self.assertIn("High confidence", _confidence_tone("high"))
        self.assertIn("Low confidence", _confidence_tone("low"))

    def test_coerce_string_list(self) -> None:
        self.assertEqual(_coerce_string_list(["a", " ", 1, None, "b"]), ["a", "b"])

    def test_clean_text_field_treats_none_like_empty(self) -> None:
        self.assertEqual(_clean_text_field(None), "")
        self.assertEqual(_clean_text_field("None"), "")
        self.assertEqual(_clean_text_field("null"), "")
        self.assertEqual(_clean_text_field("Air India"), "Air India")

    def test_coerce_url_list_extracts_only_http_urls(self) -> None:
        urls = _coerce_url_list(
            [
                'https://www.ixigo.com/check?x=1"></div>',
                '<a href="https://example.com/deal">deal</a>',
                "not-a-url",
            ]
        )
        self.assertEqual(urls, ["https://www.ixigo.com/check?x=1", "https://example.com/deal"])

    def test_normalize_analyst_payload_with_partial_data(self) -> None:
        normalized, issues = _normalize_analyst_payload(
            {
                "topic": "",
                "confidence": "maybe",
                "key_findings": ["alpha", 1, " ", "beta"],
                "travel_options": "bad-shape",
                "sources": [{"title": "Example", "url": "https://example.com"}],
            }
        )
        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(normalized["topic"], "Untitled travel result")
        self.assertEqual(normalized["confidence"], "low")
        self.assertEqual(normalized["key_findings"], ["alpha", "beta"])
        self.assertEqual(normalized["travel_options"], [])
        self.assertTrue(any("downgraded to low" in issue for issue in issues))

    def test_normalize_analyst_payload_cleans_none_fields_and_urls(self) -> None:
        normalized, issues = _normalize_analyst_payload(
            {
                "topic": "Jaipur to Dubai",
                "request_type": "flight_travel",
                "route": {"origin": "Jaipur", "destination": "Dubai", "date": "2026-04-17"},
                "confidence": "medium",
                "key_findings": ["Flight evidence found."],
                "travel_options": [
                    {
                        "mode": "flight",
                        "operator": "Air India",
                        "departure": None,
                        "arrival": "None",
                        "duration": "1h 50m",
                        "price": None,
                        "notes": "Early morning flight",
                        "source_urls": ['<a href="https://example.com/deal">View</a>'],
                    }
                ],
                "sources": [{"title": "Deal page", "url": 'https://example.com/deal"></div>'}],
            }
        )
        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(normalized["travel_options"][0]["departure"], "")
        self.assertEqual(normalized["travel_options"][0]["arrival"], "")
        self.assertEqual(normalized["travel_options"][0]["price"], "")
        self.assertEqual(normalized["travel_options"][0]["source_urls"], ["https://example.com/deal"])
        self.assertEqual(normalized["sources"][0]["url"], "https://example.com/deal")
        self.assertFalse(issues)

    def test_normalize_analyst_payload_invalid(self) -> None:
        normalized, issues = _normalize_analyst_payload(None)
        self.assertIsNone(normalized)
        self.assertIn("Structured JSON was missing or invalid.", issues)

    def test_narrative_conflict_detection_and_safe_summary(self) -> None:
        data = {
            "route": {"origin": "Jaipur", "destination": "Dubai", "date": "2026-04-17"},
            "confidence": "medium",
            "travel_options": [
                {
                    "mode": "flight",
                    "operator": "IndiGo",
                    "price": "Rs 4,161",
                    "duration": "1.5 to 2.25 hours",
                }
            ],
        }
        self.assertTrue(_narrative_conflicts_with_route("Cheapest flight from Jaipur to Amritsar.", data))
        summary = _build_structured_summary(data)
        self.assertIn("Jaipur to Dubai on 2026-04-17", summary)
        self.assertIn("IndiGo", summary)


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


class ExtractorTests(unittest.TestCase):
    def test_parse_tavily_text(self) -> None:
        parsed = parse_tavily_text(
            "[1] Delhi to Jaipur buses\nURL: https://example.com/bus\nPrice starts at Rs 499\n"
            "\n[2] Jaipur route buses\nURL: https://example.com/bus2\nDuration around 5 hours"
        )
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["title"], "Delhi to Jaipur buses")

    def test_detect_travel_mode(self) -> None:
        self.assertEqual(detect_travel_mode("find flight from delhi to jaipur"), "flight")
        self.assertEqual(detect_travel_mode("find bus from delhi to jaipur"), "bus")

    def test_extract_route(self) -> None:
        self.assertEqual(
            extract_route("find bus from delhi to jaipur"),
            {"origin": "delhi", "destination": "jaipur", "date": ""},
        )

    def test_build_extracted_evidence_for_travel(self) -> None:
        extracted = build_extracted_evidence(
            "find bus from delhi to jaipur",
            "[1] Delhi to Jaipur buses\nURL: https://example.com/bus\nPrice starts at Rs 499\n"
            "\n[2] Jaipur route buses\nURL: https://example.com/bus2\nDuration around 5 hours",
        )
        self.assertEqual(extracted["travel_mode"], "bus")
        self.assertEqual(extracted["route"]["origin"], "delhi")
        self.assertEqual(len(extracted["generic_evidence"]), 1)

    def test_extracted_evidence_to_text_contains_route(self) -> None:
        text = extracted_evidence_to_text(
            build_extracted_evidence(
                "find bus from delhi to jaipur",
                "[1] Delhi to Jaipur buses\nURL: https://example.com/bus\nPrice starts at Rs 499",
            )
        )
        self.assertIn("Detected route: delhi -> jaipur", text)

    def test_build_grounded_travel_options(self) -> None:
        extracted = build_extracted_evidence(
            "find bus from delhi to jaipur",
            "[1] Delhi to Jaipur buses\nURL: https://example.com/bus\nPrice starts at Rs 499. Duration around 5 hours.",
        )
        options = build_grounded_travel_options(extracted)
        self.assertEqual(options[0]["mode"], "bus")
        self.assertIn("Price starts at Rs 499", options[0]["price"])

    def test_build_extracted_evidence_filters_cross_mode_results(self) -> None:
        extracted = build_extracted_evidence(
            "find flight from delhi to dubai",
            "[1] Delhi to Dubai flights\nURL: https://example.com/flight\nAirline fare starts at Rs 12000\n"
            "\n[2] Delhi to Dubai train ideas\nURL: https://example.com/train\nRail journey details here",
        )
        self.assertEqual(extracted["result_count"], 1)
        self.assertEqual(extracted["generic_evidence"][0]["title"], "Delhi to Dubai flights")

    def test_build_extracted_evidence_keeps_flight_aggregator_result(self) -> None:
        extracted = build_extracted_evidence(
            "find flight from jaipur to dubai",
            "[1] Cheap Dubai fares from Jaipur\nURL: https://example.com/jaipur-dubai-flight-deals\n"
            "Compare airline ticket prices, airport departure times, and travel duration.",
        )
        self.assertEqual(extracted["result_count"], 1)
        self.assertEqual(extracted["generic_evidence"][0]["title"], "Cheap Dubai fares from Jaipur")

    def test_build_extracted_evidence_rejects_impossible_train_route(self) -> None:
        extracted = build_extracted_evidence(
            "find train from delhi to dubai",
            "[1] Delhi to Dubai trains\nURL: https://example.com/train\nTrain fare starts at Rs 5000",
        )
        self.assertEqual(extracted["result_count"], 0)
        self.assertEqual(extracted["generic_evidence"], [])

    def test_grounded_travel_options_to_text(self) -> None:
        text = grounded_travel_options_to_text(
            [
                {
                    "mode": "bus",
                    "operator": "RedBus",
                    "duration": "5 hours",
                    "price": "Rs 500",
                    "notes": "Sleeper option found",
                    "source_urls": ["https://example.com/bus"],
                }
            ]
        )
        self.assertIn("Grounded travel options:", text)
        self.assertIn("Operator: RedBus", text)

    def test_score_evidence_confidence_high(self) -> None:
        extracted = {"result_count": 6}
        grounded_rows = [
            {"price": "Rs 100", "duration": "5 hours"},
            {"price": "Rs 120", "duration": "6 hours"},
            {"price": "Rs 130", "duration": "4 hours"},
        ]
        summary = score_evidence_confidence(extracted, grounded_rows)
        self.assertEqual(summary["confidence"], "high")

    def test_score_evidence_confidence_low(self) -> None:
        extracted = {"result_count": 1}
        grounded_rows = [
            {"price": "unknown", "duration": "unknown"},
        ]
        summary = score_evidence_confidence(extracted, grounded_rows)
        self.assertEqual(summary["confidence"], "low")

    def test_confidence_summary_to_text(self) -> None:
        text = confidence_summary_to_text(
            {
                "confidence": "medium",
                "score": 3,
                "reasons": ["Collected a moderate number of travel search results."],
            }
        )
        self.assertIn("Deterministic confidence: medium", text)
        self.assertIn("Reasons:", text)


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

    def test_infer_request_type(self) -> None:
        self.assertEqual(infer_request_type("find flight from delhi to jaipur"), "flight_travel")
        self.assertEqual(infer_request_type("find bus from delhi to jaipur"), "bus_travel")
        self.assertEqual(infer_request_type("research autogen architecture"), "unsupported")

    def test_plan_search_queries_for_travel(self) -> None:
        planned = plan_search_queries("find bus from delhi to jaipur", "bus_travel")
        self.assertEqual(
            planned,
            [
                "find bus from delhi to jaipur",
                "delhi to jaipur bus price duration availability",
            ],
        )

    def test_plan_search_queries_for_flight_adds_extra_coverage(self) -> None:
        planned = plan_search_queries("find flight from jaipur to dubai", "flight_travel")
        self.assertEqual(
            planned,
            [
                "find flight from jaipur to dubai",
                "jaipur to dubai flight price duration availability",
                "jaipur to dubai flights airline airport fare",
            ],
        )

    def test_gather_search_evidence_uses_all_planned_queries(self) -> None:
        with patch("research_assistant.runner.tavily_search", side_effect=lambda q, _m, _k: f"results for {q}") as mocked:
            evidence = gather_search_evidence("find bus from delhi to jaipur", "key", "bus_travel")

        self.assertEqual(mocked.call_count, 2)
        self.assertIn("Search plan [1]: find bus from delhi to jaipur", evidence)
        self.assertIn("results for delhi to jaipur bus price duration availability", evidence)

    def test_gather_search_evidence_gives_flights_more_coverage(self) -> None:
        calls: list[tuple[str, int]] = []

        def fake_search(query: str, max_results: int, _key: str) -> str:
            calls.append((query, max_results))
            return f"results for {query}"

        with patch("research_assistant.runner.tavily_search", side_effect=fake_search):
            evidence = gather_search_evidence("find flight from jaipur to dubai", "key", "flight_travel")

        self.assertEqual(len(calls), 3)
        self.assertTrue(all(max_results == 3 for _, max_results in calls))
        self.assertIn("results for jaipur to dubai flights airline airport fare", evidence)


class IntegrationTests(unittest.TestCase):
    class _FakeClient:
        async def close(self) -> None:
            return None

    def test_end_to_end_travel_route_flow(self) -> None:
        captured: dict[str, str] = {}

        class FakeTeam:
            def __init__(self, participants, model_client, termination_condition, allow_repeated_speaker):
                captured["participant_count"] = str(len(participants))

            async def run(self, task):
                captured["task"] = task
                return SimpleNamespace(
                    messages=[
                        SimpleNamespace(
                            source="Analyst",
                            content=(
                                "```json\n"
                                '{"topic":"Delhi to Jaipur bus options","request_type":"bus_travel","route":{"origin":"Delhi","destination":"Jaipur","date":""},"confidence":"medium",'
                                '"confidence_rationale":["Multiple route sources were found.","Schedule fields remain partial."],'
                                '"key_findings":["Several bus options were found.","Price evidence is available from search results."],'
                                '"travel_options":[{"mode":"bus","operator":"Delhi to Jaipur buses","departure":"unknown","arrival":"unknown","duration":"Duration around 5 hours.","price":"Price starts at Rs 499.","notes":"Grounded from bus listing evidence.","source_urls":["https://example.com/bus"]}],'
                                '"evidence_gaps":["Exact departure times were not consistently available."],'
                                '"sources":[{"title":"Bus listing","url":"https://example.com/bus"}]}\n'
                                "```\n"
                                "Travel summary.\nTERMINATE"
                            ),
                        )
                    ]
                )

        def fake_search(query: str, _max_results: int, _key: str) -> str:
            mapping = {
                "find bus from delhi to jaipur": "[1] Delhi to Jaipur buses\nURL: https://example.com/bus\nPrice starts at Rs 499.",
                "delhi to jaipur bus price duration availability": "[1] Delhi to Jaipur buses\nURL: https://example.com/bus2\nDuration around 5 hours.",
            }
            return mapping[query]

        with patch("research_assistant.runner.create_model_client", return_value=self._FakeClient()), patch(
            "research_assistant.runner.SelectorGroupChat", FakeTeam
        ), patch("research_assistant.runner.tavily_search", side_effect=fake_search):
            history = run_research_sync(
                "find bus from delhi to jaipur",
                groq_api_key="g",
                tavily_api_key="t",
            )

        self.assertEqual(captured["participant_count"], "2")
        self.assertIn("Detected request type:\nbus_travel", captured["task"])
        self.assertIn("Grounded travel options:", captured["task"])
        self.assertIn("Deterministic confidence guidance:", captured["task"])
        self.assertEqual(history[-1]["name"], "Analyst")
        parsed, _ = _parse_analyst_output(history[-1]["content"])
        normalized, issues = _normalize_analyst_payload(parsed)
        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(normalized["request_type"], "bus_travel")
        self.assertEqual(normalized["confidence"], "medium")
        self.assertFalse(issues)

    def test_end_to_end_travel_flow(self) -> None:
        captured: dict[str, str] = {}

        class FakeTeam:
            def __init__(self, participants, model_client, termination_condition, allow_repeated_speaker):
                pass

            async def run(self, task):
                captured["task"] = task
                return SimpleNamespace(
                    messages=[
                        SimpleNamespace(
                            source="Analyst",
                            content=(
                                "```json\n"
                                '{"topic":"Delhi to Jaipur bus options","request_type":"bus_travel","route":{"origin":"Delhi","destination":"Jaipur","date":""},"confidence":"medium",'
                                '"confidence_rationale":["Travel options were found from live search results."],'
                                '"key_findings":["Several bus options were identified."],'
                                '"travel_options":[],"evidence_gaps":["Real-time availability may change quickly."],'
                                '"sources":[{"title":"Bus listing","url":"https://example.com/bus"}]}\n'
                                "```\n"
                                "Travel summary.\nTERMINATE"
                            ),
                        )
                    ]
                )

        with patch("research_assistant.runner.create_model_client", return_value=self._FakeClient()), patch(
            "research_assistant.runner.SelectorGroupChat", FakeTeam
        ), patch(
            "research_assistant.runner.tavily_search",
            return_value="[1] Delhi to Jaipur buses\nURL: https://example.com/bus\nMultiple buses available today.",
        ):
            history = run_research_sync(
                "find bus from delhi to jaipur",
                groq_api_key="g",
                tavily_api_key="t",
            )

        self.assertIn("Detected request type:\nbus_travel", captured["task"])
        self.assertIn("Search plan:\n- find bus from delhi to jaipur", captured["task"])
        parsed, _ = _parse_analyst_output(history[-1]["content"])
        normalized, _ = _normalize_analyst_payload(parsed)
        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(normalized["request_type"], "bus_travel")
        self.assertEqual(normalized["travel_options"], [])

    def test_end_to_end_incomplete_evidence_flow(self) -> None:
        captured: dict[str, str] = {}

        class FakeTeam:
            def __init__(self, participants, model_client, termination_condition, allow_repeated_speaker):
                pass

            async def run(self, task):
                captured["task"] = task
                return SimpleNamespace(
                    messages=[
                        SimpleNamespace(
                            source="Analyst",
                            content=(
                                "```json\n"
                                '{"topic":"","confidence":"unclear","key_findings":["Only limited route evidence was found."],"travel_options":"bad","sources":[]}\n'
                                "```\n"
                                "Incomplete summary.\nTERMINATE"
                            ),
                        )
                    ]
                )

        def sparse_search(query: str, _max_results: int, _key: str) -> str:
            return f"[1] Sparse result\nURL: https://example.com/{query.replace(' ', '-')}\nLimited evidence only."

        with patch("research_assistant.runner.create_model_client", return_value=self._FakeClient()), patch(
            "research_assistant.runner.SelectorGroupChat", FakeTeam
        ), patch("research_assistant.runner.tavily_search", side_effect=sparse_search):
            history = run_research_sync(
                "find bus from delhi to jaipur",
                groq_api_key="g",
                tavily_api_key="t",
            )

        self.assertIn("Deterministic confidence:", captured["task"])
        parsed, _ = _parse_analyst_output(history[-1]["content"])
        normalized, issues = _normalize_analyst_payload(parsed)
        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(normalized["confidence"], "low")
        self.assertEqual(normalized["travel_options"], [])
        self.assertTrue(any("fallback title" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
