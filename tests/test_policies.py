"""KA-17: test đệ quy JSON Schema, normalizer và missing policies."""

from __future__ import annotations

import copy

import pytest

from core.policies import (
    FieldPolicyEngine,
    GenericSchemaValidator,
    NormalizationError,
    ValueNormalizer,
)


def test_normalizer_handles_scalar_and_nested_array_object_without_mutation() -> None:
    schema = {
        "type": "object",
        "properties": {
            "count": {"type": "integer"},
            "active": {"type": "boolean"},
            "students": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "scores": {"type": "array", "items": {"type": "number"}},
                    },
                },
            },
        },
    }
    payload = {
        "count": "3",
        "active": "false",
        "students": [{"name": "An", "scores": ["9.5", "8"]}],
    }
    original = copy.deepcopy(payload)

    normalized = ValueNormalizer().normalize(payload, schema)

    assert normalized == {
        "count": 3,
        "active": False,
        "students": [{"name": "An", "scores": [9.5, 8.0]}],
    }
    assert payload == original


def test_normalizer_reports_bad_nested_value_with_precise_path() -> None:
    schema = {
        "type": "object",
        "properties": {
            "students": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "scores": {"type": "array", "items": {"type": "integer"}}
                    },
                },
            }
        },
    }

    with pytest.raises(NormalizationError, match=r"\$\.students\[1\]\.scores\[2\]"):
        ValueNormalizer().normalize(
            {"students": [{"scores": [1]}, {"scores": [2, 3, "bad"]}]}, schema
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"x": "2"}, {"x": 2.0}),
        ({"x": "0"}, {"x": 0.0}),
        ({"x": 0}, {"x": 0.0}),
    ],
)
def test_normalizer_preserves_numeric_zero(value, expected) -> None:
    assert ValueNormalizer().normalize(value, {"type": "object", "properties": {"x": {"type": "number"}}}) == expected


@pytest.mark.parametrize("value", [True, None, "NaN", "Infinity", "not-a-number"])
def test_number_normalizer_rejects_non_numbers_and_non_finite_values(value) -> None:
    with pytest.raises(NormalizationError):
        ValueNormalizer().normalize(value, {"type": "number"})


def test_schema_validator_reports_nested_required_and_array_paths() -> None:
    schema = {
        "type": "object",
        "properties": {
            "students": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"scores": {"type": "array", "items": {"type": "integer"}}},
                    "required": ["scores"],
                },
            }
        },
    }

    issues = GenericSchemaValidator().validate({"students": [{"scores": [1]}, {}]}, schema)

    assert [issue.path for issue in issues] == ["$.students[1].scores"]
    assert issues[0].keyword == "required"


def test_invalid_schema_reports_location() -> None:
    issues = GenericSchemaValidator().check_schema(
        {"type": "object", "properties": {"items": {"type": "unknown"}}}
    )

    assert issues
    assert "items" in issues[0].path


@pytest.mark.parametrize(
    ("action", "policy", "schema_default", "expected", "missing", "errors"),
    [
        ("use_value", {"value": 0}, None, 0, False, False),
        ("use_value", {"value": False}, None, False, False, False),
        ("use_value", {"value": None}, None, None, False, False),
        ("use_value", {"value": ""}, None, "", False, False),
        ("use_default", {}, 0, 0, False, False),
        ("use_default", {"value": None}, False, None, False, False),
        ("ask_user", {}, None, None, True, False),
        ("omit", {}, None, None, False, False),
        ("error", {}, None, None, False, True),
    ],
)
def test_missing_actions_preserve_falsy_and_null_values(
    action, policy, schema_default, expected, missing, errors
) -> None:
    on_missing = {"action": action, **policy}
    property_schema = {"type": "string"}
    if schema_default is not None:
        property_schema["default"] = schema_default
    schema = {"type": "object", "properties": {"field": property_schema}}

    result = FieldPolicyEngine().apply(
        {}, schema, {"field": {"required": action == "ask_user", "on_missing": on_missing}}
    )

    assert result.value.get("field") == expected
    assert bool(result.missing) is missing
    assert bool(result.errors) is errors


def test_present_null_zero_false_and_empty_are_not_treated_as_absent() -> None:
    schema = {
        "type": "object",
        "properties": {
            "nullable": {"type": ["string", "null"]},
            "zero": {"type": "integer"},
            "flag": {"type": "boolean"},
            "empty": {"type": "string"},
        },
    }
    policies = {
        name: {"required": True, "on_missing": {"action": "ask_user"}}
        for name in schema["properties"]
    }

    result = FieldPolicyEngine().apply(
        {"nullable": None, "zero": 0, "flag": False, "empty": ""}, schema, policies
    )

    assert result.value == {"nullable": None, "zero": 0, "flag": False, "empty": ""}
    assert result.is_complete


def test_required_omit_conflict_and_value_action_without_value_are_errors() -> None:
    schema = {"type": "object", "required": ["a", "b"], "properties": {}}
    result = FieldPolicyEngine().apply(
        {},
        schema,
        {
            "a": {"required": True, "on_missing": {"action": "omit"}},
            "b": {"required": True, "on_missing": {"action": "use_value"}},
        },
    )

    assert len(result.errors) == 2


def test_unconfigured_required_fields_are_returned_as_missing() -> None:
    result = FieldPolicyEngine().apply(
        {}, {"type": "object", "required": ["missing"], "properties": {}}, {}
    )

    assert result.missing == ["$.missing"]
