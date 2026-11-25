"""Regression tests cho Contract v2 và adapter legacy của KA-16."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from core.config import DEFAULT_INTENTS_PATH
from core.definitions import (
    DefinitionLoadError,
    DefinitionLoader,
    LoadedDefinition,
    adapt_legacy_definition,
)
from core.definitions import loader as definition_loader


def test_legacy_adapter_maps_types_missing_policy_and_template_without_mutation() -> None:
    legacy = {
        "intent": "LEGACY_SAMPLE",
        "slots": {
            "count": {"type": "number", "required": True, "default": 0},
            "label": {"type": "text", "required": True, "regex": "(.*)"},
            "factor": {"type": "number", "absent_means": 0},
        },
        "templates": {"latex": r"\frac{<<count>>}{<<label>>}"},
    }
    original = copy.deepcopy(legacy)

    adapted = adapt_legacy_definition(legacy)

    assert legacy == original
    assert adapted["input_schema"]["properties"]["count"]["type"] == "number"
    assert adapted["input_schema"]["properties"]["count"]["default"] == 0
    assert adapted["field_policies"]["count"]["on_missing"] == {
        "action": "use_default",
        "value": 0,
    }
    assert adapted["field_policies"]["factor"]["on_missing"]["value"] == 0
    assert adapted["field_policies"]["label"]["on_missing"]["action"] == "ask_user"
    assert adapted["render"]["templates"]["latex"] == r"\frac{ {{ count }} }{ {{ label }} }"


def test_legacy_adapter_preserves_false_empty_and_null_distinctions() -> None:
    adapted = adapt_legacy_definition(
        {
            "intent": "LEGACY_DEFAULTS",
            "slots": {
                "enabled": {"type": "boolean", "default": False},
                "text": {"type": "text", "default": ""},
                "unset": {"type": "text", "default": None},
            },
            "templates": {"latex": ""},
        }
    )

    assert adapted["input_schema"]["properties"]["enabled"]["default"] is False
    assert adapted["input_schema"]["properties"]["text"]["default"] == ""
    assert "default" not in adapted["input_schema"]["properties"]["unset"]
    assert adapted["field_policies"]["unset"]["on_missing"]["action"] == "omit"


def test_loader_fails_with_actionable_message_if_jsonschema_is_missing() -> None:
    if definition_loader.Draft202012Validator is not None:
        pytest.skip("jsonschema đã được cài; kiểm tra hành vi thiếu package không còn áp dụng")
    with pytest.raises(DefinitionLoadError, match="Thiếu jsonschema"):
        DefinitionLoader().load([])


def _require_jsonschema() -> None:
    if definition_loader.Draft202012Validator is None:
        pytest.skip("Cần cài jsonschema==4.26.0 theo requirements.txt để chạy validation.")


def test_loads_full_bank_as_v2_and_does_not_modify_source_bytes(tmp_path: Path) -> None:
    _require_jsonschema()
    source_bytes = DEFAULT_INTENTS_PATH.read_bytes()
    source_copy = tmp_path / "intents.json"
    source_copy.write_bytes(source_bytes)

    loaded = DefinitionLoader().load(source_copy)

    assert len(loaded) == 189
    assert all(item.source_contract == "v2" for item in loaded)
    assert all("slots" in item.as_legacy_payload() for item in loaded)
    snapshot = loaded[0].as_legacy_payload()
    snapshot["slots"]["sentinel"] = {"type": "text"}
    assert "sentinel" not in loaded[0].as_legacy_payload()["slots"]
    assert source_copy.read_bytes() == source_bytes


def test_nested_invalid_schema_reports_intent_and_json_path() -> None:
    _require_jsonschema()
    definition = json.loads(DEFAULT_INTENTS_PATH.read_text(encoding="utf-8"))[0]
    definition["input_schema"]["properties"]["bad"] = {"type": "not-a-json-schema-type"}

    with pytest.raises(DefinitionLoadError) as error:
        DefinitionLoader().load([definition])

    assert "LATEX_TAI_LIEU_TOI_GIAN.input_schema" in str(error.value)
    assert "properties.bad.type" in str(error.value)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda definition: definition.update(definition_version="3.0"), "version không được hỗ trợ"),
        (lambda definition: definition.pop("render"), "Contract v2 thiếu"),
    ],
)
def test_rejects_unsupported_or_partial_v2_without_legacy_fallback(mutate, message: str) -> None:
    _require_jsonschema()
    definition = json.loads(DEFAULT_INTENTS_PATH.read_text(encoding="utf-8"))[0]
    mutate(definition)

    with pytest.raises(DefinitionLoadError, match=message):
        DefinitionLoader().load([definition])


def test_rejects_duplicate_intent_ids() -> None:
    _require_jsonschema()
    definition = json.loads(DEFAULT_INTENTS_PATH.read_text(encoding="utf-8"))[0]

    with pytest.raises(DefinitionLoadError, match="intent bị trùng"):
        DefinitionLoader().load([definition, definition])


def test_validate_instance_returns_nested_input_path() -> None:
    _require_jsonschema()
    definition = json.loads(DEFAULT_INTENTS_PATH.read_text(encoding="utf-8"))[0]
    definition["input_schema"]["properties"]["payload"] = {
        "type": "object",
        "properties": {"amount": {"type": "number"}},
        "required": ["amount"],
    }
    definition["input_schema"]["additionalProperties"] = True
    loaded = DefinitionLoader().load([definition])[0]

    issues = DefinitionLoader().validate_instance(loaded, {"payload": {"amount": "bad"}})

    assert issues
    assert issues[0].path == ".payload.amount"


def test_v2_is_canonical_while_legacy_snapshot_remains_separate() -> None:
    _require_jsonschema()
    definition = json.loads(DEFAULT_INTENTS_PATH.read_text(encoding="utf-8"))[0]
    definition["legacy_note"] = "preserve me"
    loaded: LoadedDefinition = DefinitionLoader().load([definition])[0]

    assert loaded.definition["definition_version"] == "2.0"
    assert loaded.legacy_definition["legacy_note"] == "preserve me"
    loaded.as_legacy_payload()["legacy_note"] = "changed"
    assert loaded.legacy_definition["legacy_note"] == "preserve me"
