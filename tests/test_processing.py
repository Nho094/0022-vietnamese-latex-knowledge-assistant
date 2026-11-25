"""KA-19: JSONata identity, filtering, projection và object/array construction."""

from __future__ import annotations

import copy

import pytest

from core.definitions import DefinitionLoader
from core.processing import JsonataProcessor, TransformEngine, TransformError


def test_jsonata_identity_preserves_json_value_without_mutation() -> None:
    value = {"rows": [{"amount": 3}, {"amount": 5}]}
    original = copy.deepcopy(value)

    result = JsonataProcessor().transform("$", value)

    assert result == value
    assert value == original


def test_all_current_189_definitions_execute_their_declared_transform() -> None:
    definitions = DefinitionLoader().load()
    engine = TransformEngine()

    assert len(definitions) == 189
    for loaded in definitions:
        assert engine.transform(loaded.definition, {"sample": [1, 2]}) == {
            "sample": [1, 2]
        }


def test_jsonata_nested_access() -> None:
    result = JsonataProcessor().transform(
        "student.profile.name", {"student": {"profile": {"name": "An"}}}
    )

    assert result == "An"


def test_jsonata_filter_and_projection_on_dynamic_array() -> None:
    value = {
        "items": [
            {"name": "pencil", "quantity": 1, "price": 2},
            {"name": "book", "quantity": 3, "price": 4},
        ]
    }

    filtered = JsonataProcessor().transform("items[quantity >= 2].name", value)
    projected = JsonataProcessor().transform(
        'items.{"name": name, "total": quantity * price}', value
    )

    assert filtered == "book"
    assert projected == [
        {"name": "pencil", "total": 2},
        {"name": "book", "total": 12},
    ]


def test_jsonata_constructs_nested_object_and_array() -> None:
    value = {"students": [{"name": "An", "scores": [8, 9]}, {"name": "Bình", "scores": [10]}]}

    result = JsonataProcessor().transform(
        "{'count': $count(students), 'students': students.{'name': name, 'best': $max(scores)}}",
        value,
    )

    assert result == {
        "count": 2,
        "students": [{"name": "An", "best": 9}, {"name": "Bình", "best": 10}],
    }


def test_jsonata_engine_reads_contract_and_registry_engine_name() -> None:
    definition = {
        "transform": {"engine": "jsonata", "expression": "{'sum': $sum(values)}"}
    }

    assert TransformEngine().transform(definition, {"values": [1, 2, 3]}) == {"sum": 6}


@pytest.mark.parametrize(
    ("definition", "value", "message"),
    [
        ({"transform": {"engine": "missing", "expression": "$"}}, {}, "processor chưa"),
        ({"transform": {"engine": "jsonata", "expression": "("}}, {}, "jsonata"),
        ({"transform": {"engine": "jsonata", "expression": ""}}, {}, "không rỗng"),
        ({"transform": None}, {}, "phải là object"),
    ],
)
def test_transform_errors_are_clear_and_structured(definition, value, message) -> None:
    with pytest.raises(TransformError, match=message):
        TransformEngine().transform(definition, value)


def test_jsonata_guardrails_must_be_positive_and_are_configurable() -> None:
    processor = JsonataProcessor(timeout_ms=25, stack_limit=20, sequence_limit=100)

    assert processor.timeout_ms == 25
    assert processor.stack_limit == 20
    assert processor.sequence_limit == 100
    with pytest.raises(ValueError, match="số nguyên dương"):
        JsonataProcessor(timeout_ms=0)
