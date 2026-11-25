"""KA-20: nested Jinja rendering, registry và sandbox/security regressions."""

from __future__ import annotations

from typing import Any

import pytest

from core.definitions import DefinitionLoader
from core.rendering import Jinja2Renderer, RenderEngine, RenderError


def test_jinja_supports_conditionals_loops_nested_objects_and_arrays() -> None:
    template = (
        "{% if students %}{% for student in students %}"
        "{{ student.name }}:{% for score in student.scores %}{{ score }}{% if not loop.last %}, {% endif %}{% endfor %}\n"
        "{% endfor %}{% else %}empty{% endif %}"
    )

    output = Jinja2Renderer().render(
        template,
        {
            "students": [
                {"name": "An", "scores": [8, 9]},
                {"name": "Bình", "scores": [10]},
            ]
        },
    )

    assert output == "An:8, 9\nBình:10\n"


def test_jinja_supports_latex_escape_filter_for_untrusted_text() -> None:
    output = Jinja2Renderer().render(
        r"\text{ {{ title | latex_escape }} }", {"title": r"A&B_50%"}
    )

    assert output == r"\text{ A\&B\_50\% }"


def test_renderer_uses_declared_engine_and_renders_multiple_dynamic_formats() -> None:
    definition = {
        "render": {
            "engine": "jinja2",
            "templates": {
                "latex": r"\text{ {{ value | latex_escape }} }",
                "obsidian": "**{{ value }}**",
                "custom-format": "{{ values | join(',') }}",
            },
        }
    }

    outputs = RenderEngine().render(
        definition, {"value": "A&B", "values": [1, 2, 3]}
    )

    assert outputs == {
        "latex": r"\text{ A\&B }",
        "obsidian": "**A&B**",
        "custom-format": "1,2,3",
    }
    assert RenderEngine().render(definition, {"value": "x"}, ["obsidian"]) == {
        "obsidian": "**x**"
    }


def _sample_value(schema: dict[str, Any]) -> Any:
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    value_type = schema.get("type")
    if isinstance(value_type, list):
        value_type = next((kind for kind in value_type if kind != "null"), "null")
    if value_type == "object":
        return {
            key: _sample_value(child)
            for key, child in schema.get("properties", {}).items()
        }
    if value_type == "array":
        items = schema.get("items", {})
        return [_sample_value(items)] if isinstance(items, dict) and items else []
    if value_type in {"integer", "number"}:
        return 1
    if value_type == "boolean":
        return True
    if value_type == "null":
        return None
    return "sample"


def test_all_current_definitions_render_declared_templates_with_schema_shaped_data() -> None:
    loaded = DefinitionLoader().load()
    engine = RenderEngine()

    assert len(loaded) == 189
    for item in loaded:
        definition = item.definition
        value = _sample_value(definition["input_schema"])
        outputs = engine.render(definition, value)
        assert set(outputs) == set(definition["render"]["templates"])


@pytest.mark.parametrize(
    "template",
    [
        "{{ payload.__class__ }}",
        "{{ payload.items() }}",
        "{{ cycler(1, 2) }}",
        '{% include "secret.txt" %}',
        '{% import "secret.txt" as secret %}',
    ],
)
def test_sandbox_blocks_python_attributes_callables_and_filesystem_templates(template: str) -> None:
    with pytest.raises(RenderError):
        Jinja2Renderer().render(template, {"payload": {"key": "value"}})


def test_renderer_rejects_non_json_context_and_undefined_variables() -> None:
    with pytest.raises(RenderError, match="not JSON serializable"):
        Jinja2Renderer().render("{{ value }}", {"value": lambda: "unsafe"})
    with pytest.raises(RenderError, match="undefined"):
        Jinja2Renderer().render("{{ absent }}", {})


def test_latex_target_runs_existing_static_security_check() -> None:
    definition = {"render": {"engine": "jinja2", "templates": {"latex": r"\input{secret}"}}}

    with pytest.raises(RenderError, match="bị cấm"):
        RenderEngine().render(definition, {})


@pytest.mark.parametrize(
    ("definition", "message"),
    [
        ({"render": {"engine": "unknown", "templates": {}}}, "chưa được đăng ký"),
        ({"render": {"engine": "jinja2", "templates": {"latex": 4}}}, "template string"),
        ({"render": []}, "phải là object"),
    ],
)
def test_renderer_reports_contract_errors(definition, message: str) -> None:
    with pytest.raises(RenderError, match=message):
        RenderEngine().render(definition, {})
