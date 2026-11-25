"""KA-18: adapter extraction tổng quát và tương thích legacy."""

from __future__ import annotations

from typing import Any

import pytest

from core.definitions import adapt_legacy_definition
from core.extraction import (
    ExtractionRequest,
    ExtractorRegistry,
    GenericExtractionEngine,
    ParserRegistry,
)


def test_regex_adapter_supports_single_multiple_and_empty_string_values() -> None:
    registry = ExtractorRegistry()
    engine = GenericExtractionEngine(registry)
    definition = {
        "intent": "BRAND_NEW_INTENT",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}, "tags": {"type": "array"}},
        },
        "field_policies": {
            "code": {
                "extraction": {"method": "regex", "pattern": r"code\s*[:=]\s*([\w-]+)"}
            },
            "tags": {
                "extraction": {
                    "method": "regex",
                    "pattern": r"#([\w-]+)",
                    "multiple": True,
                }
            },
            "empty": {
                "extraction": {"method": "regex", "pattern": r"empty=()", "capture_group": 1}
            },
        },
    }

    result = engine.extract("code: X-4 #red #blue empty=", definition)

    assert result.values == {"code": "X-4", "tags": ["red", "blue"], "empty": ""}
    assert result.missing == []
    assert result.issues == []


def test_regex_configuration_errors_are_structured() -> None:
    outcome = ExtractorRegistry().get("regex").extract(
        ExtractionRequest("text", "value", {"pattern": "(", "capture_group": 1}, {}, {})
    )

    assert not outcome.found
    assert outcome.issue is not None
    assert outcome.issue.code == "configuration"


def test_path_adapter_supports_nested_objects_arrays_and_json_pointer_escaping() -> None:
    extractor = ExtractorRegistry().get("path")
    source = {"groups": [{"student/name": "An"}, {"student/name": "Bình"}]}

    outcome = extractor.extract(
        ExtractionRequest(
            "",
            "selected",
            {"path": "/groups/1/student~1name"},
            {"type": "string"},
            {"source": source},
        )
    )

    assert outcome.found and outcome.value == "Bình"


def test_path_adapter_supports_simple_dot_and_index_notation() -> None:
    outcome = ExtractorRegistry().get("path").extract(
        ExtractionRequest(
            "",
            "score",
            {"path": "students[1].scores[0]"},
            {"type": "integer"},
            {"source": {"students": [{}, {"scores": [7]}]}},
        )
    )

    assert outcome.found and outcome.value == 7


def test_manual_adapter_returns_absent_until_value_is_supplied() -> None:
    engine = GenericExtractionEngine()
    definition = {
        "input_schema": {"type": "object", "properties": {"matrix": {"type": "array"}}},
        "field_policies": {
            "matrix": {"extraction": {"method": "manual"}},
        },
    }

    absent = engine.extract("", definition)
    supplied = engine.extract("", definition, {"manual_values": {"matrix": [[1, 2], [3, 4]]}})

    assert absent.missing == ["matrix"]
    assert supplied.values == {"matrix": [[1, 2], [3, 4]]}


def test_parser_is_explicitly_registered_and_never_imported_from_data() -> None:
    parsers = ParserRegistry()
    parsers.register("split-values", lambda text, _policy, _context: text.split(","))
    engine = GenericExtractionEngine(ExtractorRegistry(parser_registry=parsers))
    definition = {
        "input_schema": {"type": "object", "properties": {"values": {"type": "array"}}},
        "field_policies": {
            "values": {"extraction": {"method": "parser", "parser": "split-values"}},
        },
    }

    result = engine.extract("a,b,c", definition)

    assert result.values == {"values": ["a", "b", "c"]}
    with pytest.raises(ValueError, match="đã được đăng ký"):
        parsers.register("split-values", lambda *_args: [])


def test_model_extraction_is_unavailable_without_provider_and_uses_configured_provider() -> None:
    definition = {
        "input_schema": {"type": "object", "properties": {"quantity": {"type": "integer"}}},
        "field_policies": {"quantity": {"extraction": {"method": "model"}}},
    }
    unavailable = GenericExtractionEngine().extract("có 4 vé", definition)

    class TestProvider:
        def extract(self, prompt: str) -> Any:
            assert '"quantity"' in prompt
            return 4

    configured = GenericExtractionEngine(
        ExtractorRegistry(model_provider=TestProvider())
    ).extract("có 4 vé", definition)

    assert unavailable.missing == ["quantity"]
    assert unavailable.issues[0].code == "provider_unavailable"
    assert configured.values == {"quantity": 4}


def test_legacy_adapter_regex_policy_extracts_legacy_field_without_intent_branch() -> None:
    legacy = {
        "intent": "SAMPLE_LEGACY",
        "slots": {
            "package_name": {
                "type": "text",
                "regex": r"(?:gói|package)\s*(?:là|:)?\s*([A-Za-z][A-Za-z0-9_-]*)",
                "required": True,
                "default": None,
            }
        },
        "templates": {"latex": r"\usepackage{<<package_name>>}"},
    }
    adapted = adapt_legacy_definition(legacy)

    result = GenericExtractionEngine().extract("thêm package là amsmath", adapted)

    assert result.values == {"package_name": "amsmath"}
    assert not result.issues
