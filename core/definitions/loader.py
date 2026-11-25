"""Đọc Contract v2 an toàn và dựng adapter legacy chỉ trong bộ nhớ.

Định nghĩa v2 là nguồn chuẩn cho consumer mới. Các pipeline cũ tiếp tục nhận
snapshot của trường legacy cho đến milestone tích hợp; loader không đồng bộ,
ghi lại hoặc tự sửa file nguồn. Definition legacy-only được chuyển đổi trên
bản sao, còn contract v2 thiếu một phần hoặc sai version sẽ bị từ chối.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import (
    DATA_CONTRACT_MANIFEST_PATH,
    DEFAULT_INTENTS_PATH,
    INTENT_DEFINITION_SCHEMA_PATH,
    SUPPORTED_TARGETS,
)

try:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError
except ImportError:  # pragma: no cover - phụ thuộc runtime được khai báo trong requirements.txt
    Draft202012Validator = None  # type: ignore[assignment,misc]
    SchemaError = ValueError  # type: ignore[misc,assignment]


_V2_FIELDS = frozenset(
    {"definition_version", "input_schema", "field_policies", "transform", "render"}
)
_LEGACY_MARKERS = frozenset({"slots", "templates"})
_LEGACY_TYPE_MAP = {
    "number": "number",
    "integer": "integer",
    "boolean": "boolean",
    "object": "object",
    "array": "array",
}


@dataclass(frozen=True)
class DefinitionIssue:
    """Một lỗi contract có đường dẫn JSON để định vị trường nguồn."""

    intent: str
    path: str
    message: str


class DefinitionLoadError(ValueError):
    """Lỗi đọc hoặc kiểm tra một hay nhiều definition."""

    def __init__(self, issues: Sequence[DefinitionIssue] | str) -> None:
        if isinstance(issues, str):
            self.issues = ()
            message = issues
        else:
            self.issues = tuple(issues)
            message = "\n".join(
                f"{issue.intent or '<bank>'}{issue.path}: {issue.message}"
                for issue in self.issues
            )
        super().__init__(message)


@dataclass(frozen=True)
class LoadedDefinition:
    """Definition canonique et snapshot legacy immuable par convention."""

    intent: str
    definition: dict[str, Any]
    source_contract: str
    legacy_definition: dict[str, Any]

    def as_legacy_payload(self) -> dict[str, Any]:
        """Trả snapshot sâu để caller cũ không sửa dữ liệu loader đang giữ."""
        return copy.deepcopy(self.legacy_definition)


def _require_validator() -> Any:
    if Draft202012Validator is None:
        raise DefinitionLoadError(
            "Thiếu jsonschema; cài các dependency đã pin trong requirements.txt."
        )
    return Draft202012Validator


def _json_path(parts: Sequence[Any]) -> str:
    path = ""
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}"
    return path


def _check_json_schema(schema: Any, *, intent: str, path: str) -> list[DefinitionIssue]:
    validator_type = _require_validator()
    if not isinstance(schema, dict):
        return [DefinitionIssue(intent, path, "phải là một JSON Schema object")]
    try:
        validator_type.check_schema(schema)
    except SchemaError as exc:
        return [
            DefinitionIssue(
                intent,
                path + _json_path(tuple(exc.absolute_path)),
                str(exc.message),
            )
        ]
    return []


def _is_v2(definition: Mapping[str, Any]) -> bool:
    return bool(_V2_FIELDS.intersection(definition))


def _legacy_template(template: str, field_names: Sequence[str]) -> str:
    """Chuyển placeholder slot cũ, giữ nguyên lệnh và ngoặc LaTeX tĩnh."""
    field_set = set(field_names)
    # Bảo toàn cặp {{...}} tĩnh như \graphicspath{{images/}} khỏi cú pháp Jinja2
    protected_raw: list[str] = []

    def _protect_raw(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if inner in field_set or "<<" in inner or ">>" in inner:
            return match.group(0)
        token = f"__KA_RAW_{len(protected_raw)}__"
        protected_raw.append(match.group(0))
        return token

    result = re.sub(r"\{\{([^{}]+)\}\}", _protect_raw, template)
    result = re.sub(r"\{<<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>>", r"{ <<\1>>", result)
    result = re.sub(r"<<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>>\}", r"<<\1>> }", result)
    result = re.sub(r"<<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>>", r"{{ \1 }}", result)
    for name in sorted(field_names, key=len, reverse=True):
        result = result.replace("{" + name + "}", "{{ " + name + " }}")
    # Tránh chuỗi `{{{` / `}}}` khiến Jinja2 hiểu nhầm là dict literal
    while "{{{" in result:
        result = result.replace("{{{", "{ {{")
    while "}}}" in result:
        result = result.replace("}}}", "}} }")
    for idx, original in enumerate(protected_raw):
        result = result.replace(f"__KA_RAW_{idx}__", f"{{% raw %}}{original}{{% endraw %}}")
    return result


def adapt_legacy_definition(legacy: Mapping[str, Any]) -> dict[str, Any]:
    """Tạo Contract v2 từ record legacy mà không mutate record đầu vào.

    Adapter giữ nguyên các trường legacy ở bản sao để consumer cũ tiếp tục
    hoạt động. Nó chỉ ánh xạ schema/policy/render, không suy diễn dữ liệu mặc định.
    """
    if not isinstance(legacy, Mapping):
        raise DefinitionLoadError("Legacy definition phải là object.")
    result: dict[str, Any] = copy.deepcopy(dict(legacy))
    slots = legacy.get("slots", {})
    templates = legacy.get("templates", {})
    if not isinstance(slots, Mapping) or not isinstance(templates, Mapping):
        raise DefinitionLoadError("Legacy slots và templates phải là object.")

    properties: dict[str, Any] = {}
    policies: dict[str, Any] = {}
    required: list[str] = []
    for name, slot in slots.items():
        if not isinstance(name, str) or not isinstance(slot, Mapping):
            raise DefinitionLoadError(f"slots.{name}: phải là object.")
        legacy_type = str(slot.get("type", "text")).lower()
        value_type = _LEGACY_TYPE_MAP.get(legacy_type, "string")
        field_schema: dict[str, Any] = {"type": value_type}
        if legacy_type == "matrix":
            field_schema["format"] = "matrix"
        if isinstance(slot.get("desc"), str):
            field_schema["description"] = slot["desc"]
        # Legacy default null biểu thị chưa có mặc định; các giá trị falsy được giữ.
        if slot.get("default") is not None:
            field_schema["default"] = copy.deepcopy(slot["default"])
        properties[name] = field_schema

        is_required = bool(slot.get("required", False))
        if is_required:
            required.append(name)
        if legacy_type == "matrix":
            extraction: dict[str, Any] = {"method": "parser", "parser": "matrix_content"}
            if isinstance(slot.get("regex"), str):
                extraction["pattern"] = slot["regex"]
        else:
            extraction = {"method": "regex"}
            if isinstance(slot.get("regex"), str):
                extraction.update({"pattern": slot["regex"], "capture_group": 1})
        policy: dict[str, Any] = {
            "value_type": value_type,
            "extraction": extraction,
            "required": is_required,
        }
        if isinstance(slot.get("desc"), str):
            policy["description"] = slot["desc"]
        if "absent_means" in slot:
            policy["on_missing"] = {
                "action": "use_value",
                "value": copy.deepcopy(slot["absent_means"]),
                "source": "legacy.absent_means",
            }
        elif slot.get("default") is not None:
            policy["on_missing"] = {
                "action": "use_default",
                "value": copy.deepcopy(slot["default"]),
            }
        elif is_required:
            policy["on_missing"] = {"action": "ask_user"}
        else:
            policy["on_missing"] = {"action": "omit"}
        constraints = {
            key: copy.deepcopy(slot[key])
            for key in ("not_equal", "min", "max", "enum")
            if key in slot
        }
        if constraints:
            policy["constraints"] = constraints
        if isinstance(slot.get("constraint_error"), str):
            policy.setdefault("constraints", {})["error"] = slot["constraint_error"]
        policies[name] = policy

    render_templates = {
        target: _legacy_template(template, list(slots))
        for target, template in templates.items()
        if target in SUPPORTED_TARGETS and isinstance(template, str) and template.strip()
    }
    result.update(
        {
            "definition_version": "2.0",
            "input_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            "field_policies": policies,
            "transform": {"engine": "jsonata", "expression": "$", "mode": "identity"},
            "render": {
                "engine": "jinja2",
                "context": "normalized_input",
                "templates": render_templates,
            },
        }
    )
    return result


class DefinitionLoader:
    """Load/validate Contract v2 cùng adapter legacy không ghi dữ liệu."""

    def __init__(
        self,
        manifest_path: Path = DATA_CONTRACT_MANIFEST_PATH,
        schema_path: Path = INTENT_DEFINITION_SCHEMA_PATH,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.schema_path = Path(schema_path)

    def _contract_validator(self) -> Any:
        validator_type = _require_validator()
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DefinitionLoadError(f"Không đọc được manifest/schema: {exc}") from exc
        if manifest.get("data_contract_version") != "2.0":
            raise DefinitionLoadError("manifest.data_contract_version không được hỗ trợ")
        if manifest.get("json_schema_draft") != "2020-12":
            raise DefinitionLoadError("manifest.json_schema_draft phải là 2020-12")
        try:
            validator_type.check_schema(schema)
        except SchemaError as exc:
            raise DefinitionLoadError(f"Schema Contract v2 không hợp lệ: {exc.message}") from exc
        return validator_type(schema)

    def load(
        self,
        source: Path | Sequence[Mapping[str, Any]] = DEFAULT_INTENTS_PATH,
    ) -> list[LoadedDefinition]:
        """Nạp file hoặc sequence đã parse, xác thực toàn bank và trả snapshot."""
        validator = self._contract_validator()
        if isinstance(source, (str, Path)):
            try:
                raw = json.loads(Path(source).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise DefinitionLoadError(f"Không đọc được definitions: {exc}") from exc
        else:
            raw = copy.deepcopy(list(source))
        if not isinstance(raw, list):
            raise DefinitionLoadError("Root intents.json phải là array.")

        issues: list[DefinitionIssue] = []
        loaded: list[LoadedDefinition] = []
        seen: set[str] = set()
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                issues.append(DefinitionIssue("", f"[{index}]", "phải là object"))
                continue
            intent = item.get("intent")
            intent_name = intent if isinstance(intent, str) else f"[{index}]"
            if isinstance(intent, str):
                if intent in seen:
                    issues.append(DefinitionIssue(intent, ".intent", "intent bị trùng"))
                seen.add(intent)
            has_v2 = _is_v2(item)
            has_legacy = _LEGACY_MARKERS.issubset(item)
            if has_v2:
                if item.get("definition_version") != "2.0":
                    issues.append(
                        DefinitionIssue(intent_name, ".definition_version", "version không được hỗ trợ")
                    )
                    continue
                if not _V2_FIELDS.issubset(item):
                    missing = sorted(_V2_FIELDS.difference(item))
                    issues.append(
                        DefinitionIssue(intent_name, "", "Contract v2 thiếu: " + ", ".join(missing))
                    )
                    continue
                definition = copy.deepcopy(item)
                source_contract = "v2"
            elif has_legacy:
                try:
                    definition = adapt_legacy_definition(item)
                except DefinitionLoadError as exc:
                    issues.append(DefinitionIssue(intent_name, "", str(exc)))
                    continue
                source_contract = "legacy-adapted"
            else:
                issues.append(
                    DefinitionIssue(intent_name, "", "không phải Contract v2 hoàn chỉnh hay legacy definition")
                )
                continue

            for error in sorted(validator.iter_errors(definition), key=lambda err: list(err.absolute_path)):
                issues.append(
                    DefinitionIssue(
                        intent_name,
                        _json_path(tuple(error.absolute_path)),
                        error.message,
                    )
                )
            issues.extend(
                _check_json_schema(
                    definition.get("input_schema"),
                    intent=intent_name,
                    path=".input_schema",
                )
            )
            legacy_snapshot = copy.deepcopy(item)
            loaded.append(
                LoadedDefinition(intent_name, definition, source_contract, legacy_snapshot)
            )
        if issues:
            raise DefinitionLoadError(issues)
        return loaded

    def validate_instance(
        self, definition: LoadedDefinition | Mapping[str, Any], payload: Any
    ) -> list[DefinitionIssue]:
        """Kiểm tra payload theo input_schema, trả lỗi cùng JSON path."""
        validator_type = _require_validator()
        canonical = definition.definition if isinstance(definition, LoadedDefinition) else definition
        intent = definition.intent if isinstance(definition, LoadedDefinition) else str(canonical.get("intent", ""))
        schema = canonical.get("input_schema")
        try:
            validator_type.check_schema(schema)
        except SchemaError as exc:
            return [DefinitionIssue(intent, ".input_schema" + _json_path(tuple(exc.absolute_path)), exc.message)]
        return [
            DefinitionIssue(intent, _json_path(tuple(error.absolute_path)), error.message)
            for error in validator_type(schema).iter_errors(payload)
        ]
