"""Schema validation, normalization và missing policies theo data.

Module này xử lý JSON values đệ quy theo JSON Schema Draft 2020-12. Nó không
đọc tên intent/slot; các giá trị mặc định và hành vi thiếu chỉ đến từ schema
hoặc field policy. Cách này giữ nguyên slot normalizer legacy trong KA-08 cho
đường pipeline cũ cho tới khi KA-21 chứng minh parity.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from core.slots import normalize_matrix


@dataclass(frozen=True)
class SchemaIssue:
    """Lỗi schema hoặc dữ liệu kèm JSONPath cụ thể."""

    path: str
    message: str
    keyword: str = ""


@dataclass
class PolicyResult:
    """Payload sau policy cùng các trường cần hỏi/lỗi theo đường dẫn."""

    value: dict[str, Any]
    missing: list[str] = field(default_factory=list)
    errors: list[SchemaIssue] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """Chỉ complete khi không cần hỏi thêm và không có lỗi policy."""
        return not self.missing and not self.errors


class NormalizationError(ValueError):
    """Giá trị không thể chuẩn hóa theo kiểu trong schema."""

    def __init__(self, issue: SchemaIssue) -> None:
        self.issue = issue
        super().__init__(f"{issue.path}: {issue.message}")


def _path_child(path: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    if key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{key!r}]"


class GenericSchemaValidator:
    """Validator Draft 2020-12, giữ đường dẫn nested và lỗi rõ ràng."""

    def check_schema(self, schema: Any) -> list[SchemaIssue]:
        """Kiểm tra chính schema trước khi nhận dữ liệu người dùng."""
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            path = "$"
            for key in exc.absolute_path:
                path = _path_child(path, key)
            return [SchemaIssue(path, exc.message, str(exc.validator or "schema"))]
        return []

    def validate(self, value: Any, schema: dict[str, Any]) -> list[SchemaIssue]:
        """Trả mọi lỗi dữ liệu theo thứ tự ổn định, không raise lỗi đầu tiên."""
        schema_issues = self.check_schema(schema)
        if schema_issues:
            return schema_issues
        validator = Draft202012Validator(schema)
        issues: list[SchemaIssue] = []
        errors = sorted(
            validator.iter_errors(value),
            key=lambda error: (tuple(map(str, error.absolute_path)), str(error.validator)),
        )
        for error in errors:
            path = "$"
            for key in error.absolute_path:
                path = _path_child(path, key)
            if error.validator == "required" and isinstance(error.instance, dict):
                absent = [key for key in error.validator_value if key not in error.instance]
                for key in absent:
                    issues.append(
                        SchemaIssue(
                            _path_child(path, key),
                            "trường bắt buộc chưa được cung cấp",
                            "required",
                        )
                    )
                continue
            issues.append(SchemaIssue(path, error.message, str(error.validator or "")))
        return issues


class ValueNormalizer:
    """Chuẩn hóa scalar và container theo schema mà không làm mất giá trị."""

    def normalize(self, value: Any, schema: dict[str, Any], path: str = "$") -> Any:
        """Trả bản sao đã chuẩn hóa; không chèn default hoặc bỏ field."""
        expected = schema.get("type")
        if isinstance(expected, list):
            # Union có nhiều kiểu giữ nguyên dữ liệu; JSON Schema quyết định hợp lệ.
            return self._normalize_union(value, expected, schema, path)
        if expected == "number":
            return self._number(value, path)
        if expected == "integer":
            return self._integer(value, path)
        if expected == "boolean":
            return self._boolean(value, path)
        if expected == "string" and schema.get("format") == "matrix":
            try:
                return normalize_matrix(value)
            except ValueError as exc:
                raise NormalizationError(SchemaIssue(path, str(exc), "format")) from None
        if expected == "object" and isinstance(value, dict):
            result = copy.deepcopy(value)
            properties = schema.get("properties", {})
            for key, child in list(result.items()):
                child_schema = properties.get(key, schema.get("additionalProperties"))
                if isinstance(child_schema, dict):
                    result[key] = self.normalize(
                        child, child_schema, _path_child(path, key)
                    )
            return result
        if expected == "array" and isinstance(value, list):
            result = copy.deepcopy(value)
            prefix_items = schema.get("prefixItems", [])
            item_schema = schema.get("items")
            for index, item in enumerate(result):
                current_schema = (
                    prefix_items[index]
                    if index < len(prefix_items)
                    else item_schema
                )
                if isinstance(current_schema, dict):
                    result[index] = self.normalize(
                        item, current_schema, _path_child(path, index)
                    )
            return result
        return copy.deepcopy(value)

    def _normalize_union(
        self, value: Any, expected: list[str], schema: dict[str, Any], path: str
    ) -> Any:
        if value is None and "null" in expected:
            return None
        if isinstance(value, str) and "string" in expected:
            return value
        if isinstance(value, bool) and "boolean" in expected:
            return value
        if isinstance(value, dict) and "object" in expected:
            return self.normalize(value, {**schema, "type": "object"}, path)
        if isinstance(value, list) and "array" in expected:
            return self.normalize(value, {**schema, "type": "array"}, path)
        if "integer" in expected:
            return self._integer(value, path)
        if "number" in expected:
            return self._number(value, path)
        return copy.deepcopy(value)

    @staticmethod
    def _number(value: Any, path: str) -> int | float:
        if isinstance(value, bool) or value is None:
            raise NormalizationError(
                SchemaIssue(path, "không thể chuẩn hóa thành number", "type")
            )
        try:
            number = Decimal(str(value).strip())
        except (InvalidOperation, ValueError):
            raise NormalizationError(
                SchemaIssue(path, "không thể chuẩn hóa thành number", "type")
            ) from None
        if not number.is_finite():
            raise NormalizationError(
                SchemaIssue(path, "number phải hữu hạn", "type")
            )
        result = float(number)
        if not math.isfinite(result):
            raise NormalizationError(
                SchemaIssue(path, "number vượt miền biểu diễn hữu hạn", "type")
            )
        return result

    @classmethod
    def _integer(cls, value: Any, path: str) -> int:
        if isinstance(value, bool):
            raise NormalizationError(
                SchemaIssue(path, "boolean không được chuyển thành integer", "type")
            )
        try:
            number = Decimal(str(value).strip())
        except (InvalidOperation, ValueError):
            raise NormalizationError(
                SchemaIssue(path, "không thể chuẩn hóa thành integer", "type")
            ) from None
        if not number.is_finite() or number != number.to_integral_value():
            raise NormalizationError(
                SchemaIssue(path, "giá trị không phải integer hữu hạn", "type")
            )
        return int(number)

    @staticmethod
    def _boolean(value: Any, path: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized == "true":
                return True
            if normalized == "false":
                return False
        raise NormalizationError(
            SchemaIssue(path, "boolean chỉ nhận true/false", "type")
        )


class FieldPolicyEngine:
    """Áp dụng missing behavior do definition khai báo trên top-level fields."""

    def apply(
        self,
        value: Any,
        schema: dict[str, Any],
        field_policies: dict[str, Any],
    ) -> PolicyResult:
        """Giải quyết policy thiếu; giữ nguyên null/0/false/chuỗi rỗng đã có."""
        if not isinstance(value, dict):
            return PolicyResult(
                {}, errors=[SchemaIssue("$", "payload phải là object", "type")]
            )
        result = copy.deepcopy(value)
        required = set(schema.get("required", []))
        missing: list[str] = []
        errors: list[SchemaIssue] = []
        for name, policy in field_policies.items():
            if not isinstance(policy, dict):
                errors.append(
                    SchemaIssue(_path_child("$", name), "field policy phải là object", "policy")
                )
                continue
            if policy.get("required") is True:
                required.add(name)
            if name in result:
                continue
            on_missing = policy.get("on_missing", {})
            if not isinstance(on_missing, dict):
                errors.append(
                    SchemaIssue(_path_child("$", name), "on_missing phải là object", "policy")
                )
                continue
            action = on_missing.get("action")
            field_path = _path_child("$", name)
            if action == "use_value":
                if "value" in on_missing:
                    result[name] = copy.deepcopy(on_missing["value"])
                else:
                    errors.append(SchemaIssue(field_path, "use_value cần khai báo value", "policy"))
            elif action == "use_default":
                property_schema = schema.get("properties", {}).get(name, {})
                if "value" in on_missing:
                    result[name] = copy.deepcopy(on_missing["value"])
                elif isinstance(property_schema, dict) and "default" in property_schema:
                    result[name] = copy.deepcopy(property_schema["default"])
                else:
                    errors.append(
                        SchemaIssue(field_path, "use_default không có giá trị mặc định", "policy")
                    )
            elif action == "ask_user":
                missing.append(field_path)
            elif action == "omit":
                if name in required:
                    errors.append(
                        SchemaIssue(field_path, "omit xung đột với trường required", "policy")
                    )
                continue
            elif action == "error":
                errors.append(SchemaIssue(field_path, "thiếu trường theo policy error", "policy"))
            else:
                errors.append(
                    SchemaIssue(field_path, f"missing action không được hỗ trợ: {action!r}", "policy")
                )

        covered = set(field_policies)
        for name in sorted(required.difference(result).difference(covered)):
            missing.append(_path_child("$", name))
        return PolicyResult(result, missing, errors)
