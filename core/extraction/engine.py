"""Extraction adapters tổng quát cho regex, parser, path, manual và model.

Contract chỉ mô tả adapter; nó không làm NLU tự xuất hiện. Vì vậy model/parser
chỉ hoạt động khi ứng dụng đăng ký provider/parser cụ thể. Regex là primitive
nội bộ; legacy `core.slots.extract_slots` vẫn giữ nguyên cho đến KA-21 parity.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ExtractionRequest:
    """Ngữ cảnh đầu vào chung để mọi adapter cùng thực thi một interface."""

    text: str
    field_name: str
    policy: Mapping[str, Any]
    schema: Mapping[str, Any]
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionIssue:
    """Lỗi cấu hình/provider hoặc trích xuất cho một field."""

    field: str
    code: str
    message: str


@dataclass(frozen=True)
class ExtractionOutcome:
    """Kết quả một adapter; found phân biệt absent với null/empty value."""

    found: bool
    value: Any = None
    issue: ExtractionIssue | None = None


@dataclass
class ExtractionResult:
    """Các giá trị được trích xuất cùng field còn thiếu và lỗi."""

    values: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    issues: list[ExtractionIssue] = field(default_factory=list)


class Extractor(Protocol):
    """Protocol cho adapter có thể tái sử dụng qua nhiều field/intent."""

    def extract(self, request: ExtractionRequest) -> ExtractionOutcome:
        """Trích xuất một giá trị hoặc trả absent/issue."""


class ModelProvider(Protocol):
    """Interface provider do ứng dụng cấu hình; không giả định LLM có sẵn."""

    def extract(self, prompt: str) -> Any:
        """Trả một JSON value cho prompt schema-driven."""


from core.slots import extract_matrix_raw, normalize_matrix


_ABSENT = object()


class RegexExtractor:
    """Trích group cấu hình, hỗ trợ match đơn hoặc danh sách động."""

    def extract(self, request: ExtractionRequest) -> ExtractionOutcome:
        config = request.policy
        pattern = config.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            return self._issue(request, "configuration", "regex cần pattern không rỗng")
        try:
            compiled = re.compile(pattern, re.IGNORECASE | re.UNICODE)
        except re.error as exc:
            return self._issue(request, "configuration", f"regex không hợp lệ: {exc}")
        try:
            capture_group = int(config.get("capture_group", 1))
            if capture_group < 0 or capture_group > compiled.groups:
                return self._issue(
                    request,
                    "configuration",
                    f"capture_group {capture_group} không tồn tại trong regex",
                )
        except (TypeError, ValueError):
            return self._issue(request, "configuration", "capture_group phải là số nguyên")

        try:
            if config.get("multiple", False):
                values = [
                    match.group(capture_group)
                    for match in compiled.finditer(request.text)
                    if match.group(capture_group) is not None
                ]
                return ExtractionOutcome(bool(values), values)
            match = compiled.search(request.text)
        except (IndexError, re.error) as exc:
            return self._issue(request, "extraction", str(exc))
        if match is None or match.group(capture_group) is None:
            return ExtractionOutcome(False)
        return ExtractionOutcome(True, match.group(capture_group))

    @staticmethod
    def _issue(request: ExtractionRequest, code: str, message: str) -> ExtractionOutcome:
        return ExtractionOutcome(False, issue=ExtractionIssue(request.field_name, code, message))


ParserFunction = Callable[[str, Mapping[str, Any], Mapping[str, Any]], Any]


def _default_matrix_parser(text: str, policy: Mapping[str, Any], _context: Mapping[str, Any]) -> Any:
    """Parser dựng sẵn cho nội dung ma trận nhiều dòng/nhiều phần tử."""
    raw_pattern = policy.get("pattern")
    matched_raw, rows_hint, cols_hint = extract_matrix_raw(
        text, str(raw_pattern) if isinstance(raw_pattern, str) else None
    )
    if matched_raw is None:
        return _ABSENT
    return normalize_matrix(matched_raw, rows_hint=rows_hint, cols_hint=cols_hint)


class ParserRegistry:
    """Registry parser explicit; không import module tùy ý từ JSON."""

    def __init__(self, *, register_builtins: bool = True) -> None:
        self._parsers: dict[str, ParserFunction] = {}
        if register_builtins:
            self.register("matrix_content", _default_matrix_parser)

    def register(self, name: str, parser: ParserFunction) -> None:
        """Đăng ký parser từ code extension tin cậy của ứng dụng."""
        if not name or name in self._parsers:
            raise ValueError(f"Parser name rỗng hoặc đã được đăng ký: {name!r}")
        self._parsers[name] = parser

    def parse(self, name: str, request: ExtractionRequest) -> ExtractionOutcome:
        parser = self._parsers.get(name)
        if parser is None:
            return ExtractionOutcome(
                False,
                issue=ExtractionIssue(
                    request.field_name,
                    "unavailable",
                    f"parser '{name}' chưa được đăng ký",
                ),
            )
        try:
            parsed = parser(request.text, request.policy, request.context)
            if parsed is _ABSENT:
                return ExtractionOutcome(False)
            return ExtractionOutcome(True, parsed)
        except Exception as exc:  # extension boundary: convert provider exceptions to structured errors
            return ExtractionOutcome(
                False,
                issue=ExtractionIssue(request.field_name, "parser_error", str(exc)),
            )


class ParserExtractor:
    """Chuyển parser policy sang parser đã đăng ký trong registry."""

    def __init__(self, registry: ParserRegistry) -> None:
        self.registry = registry

    def extract(self, request: ExtractionRequest) -> ExtractionOutcome:
        name = request.policy.get("parser")
        if not isinstance(name, str) or not name:
            return ExtractionOutcome(
                False,
                issue=ExtractionIssue(request.field_name, "configuration", "parser policy thiếu tên parser"),
            )
        return self.registry.parse(name, request)


class PathExtractor:
    """Đọc giá trị từ JSON context theo JSON Pointer hoặc đường dẫn đơn giản."""

    def extract(self, request: ExtractionRequest) -> ExtractionOutcome:
        path = request.policy.get("path")
        if not isinstance(path, str) or not path:
            return ExtractionOutcome(
                False,
                issue=ExtractionIssue(request.field_name, "configuration", "path policy thiếu path"),
            )
        source_name = request.policy.get("source", "source")
        source = request.context.get(source_name, request.context)
        try:
            found, value = self._lookup(source, path)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            return ExtractionOutcome(
                False,
                issue=ExtractionIssue(request.field_name, "path_error", str(exc)),
            )
        return ExtractionOutcome(found, value)

    @staticmethod
    def _lookup(source: Any, path: str) -> tuple[bool, Any]:
        if path in {"", "$"}:
            return True, source
        if path.startswith("/"):
            parts: list[str | int] = [
                part.replace("~1", "/").replace("~0", "~")
                for part in path[1:].split("/")
            ]
        else:
            expression = path.removeprefix("$")
            parts = []
            cursor = 0
            token_pattern = re.compile(r"(?:^|\.)([^.[\]]+)|\[(\d+)\]")
            for match in token_pattern.finditer(expression):
                if match.start() != cursor:
                    raise ValueError(f"path syntax không hợp lệ gần {expression[cursor:]!r}")
                parts.append(match.group(1) if match.group(1) is not None else int(match.group(2)))
                cursor = match.end()
            if cursor != len(expression):
                raise ValueError(f"path syntax không hợp lệ gần {expression[cursor:]!r}")
        current = source
        for part in parts:
            if isinstance(current, list):
                index = part if isinstance(part, int) else int(part) if part.isdigit() else None
                if index is None:
                    raise TypeError(f"{part!r} không phải chỉ số array")
                if index >= len(current):
                    return False, None
                current = current[index]
            elif isinstance(current, Mapping):
                key = str(part)
                if key not in current:
                    return False, None
                current = current[key]
            else:
                raise TypeError(f"{part!r} yêu cầu object hoặc array")
        return True, current


class ManualExtractor:
    """Nhận giá trị người dùng đã cung cấp, không mở prompt/CLI riêng."""

    def extract(self, request: ExtractionRequest) -> ExtractionOutcome:
        manual_values = request.context.get("manual_values", {})
        if not isinstance(manual_values, Mapping):
            return ExtractionOutcome(
                False,
                issue=ExtractionIssue(request.field_name, "context_error", "manual_values phải là object"),
            )
        if request.field_name not in manual_values:
            return ExtractionOutcome(False)
        return ExtractionOutcome(True, manual_values[request.field_name])


class ModelExtractor:
    """Adapter provider tùy chọn với prompt sinh từ schema/policy."""

    def __init__(self, provider: ModelProvider | None = None) -> None:
        self.provider = provider

    def extract(self, request: ExtractionRequest) -> ExtractionOutcome:
        if self.provider is None:
            return ExtractionOutcome(
                False,
                issue=ExtractionIssue(
                    request.field_name,
                    "provider_unavailable",
                    "model extraction chưa có provider được cấu hình",
                ),
            )
        prompt = json.dumps(
            {
                "task": "Extract exactly one JSON value for the requested field.",
                "field": request.field_name,
                "schema": request.schema,
                "policy": request.policy,
                "input": request.text,
                "output_requirement": "Return only a JSON value; do not invent missing data.",
            },
            ensure_ascii=False,
        )
        try:
            value = self.provider.extract(prompt)
        except Exception as exc:  # extension boundary
            return ExtractionOutcome(
                False,
                issue=ExtractionIssue(request.field_name, "provider_error", str(exc)),
            )
        return ExtractionOutcome(True, value)


class ExtractorRegistry:
    """Map method names to reusable adapters; data cannot register code."""

    def __init__(
        self,
        parser_registry: ParserRegistry | None = None,
        model_provider: ModelProvider | None = None,
    ) -> None:
        parser_registry = parser_registry or ParserRegistry()
        self._extractors: dict[str, Extractor] = {
            "regex": RegexExtractor(),
            "parser": ParserExtractor(parser_registry),
            "path": PathExtractor(),
            "manual": ManualExtractor(),
            "model": ModelExtractor(model_provider),
        }

    def register(self, method: str, extractor: Extractor, *, replace: bool = False) -> None:
        """Đăng ký adapter do ứng dụng tin cậy cấu hình."""
        if not method or (method in self._extractors and not replace):
            raise ValueError(f"Extraction method rỗng hoặc đã tồn tại: {method!r}")
        self._extractors[method] = extractor

    def get(self, method: str) -> Extractor | None:
        """Lấy adapter đã đăng ký, không import động theo data."""
        return self._extractors.get(method)


class GenericExtractionEngine:
    """Chạy các extractor theo field_policies mà không xét tên intent."""

    def __init__(self, registry: ExtractorRegistry | None = None) -> None:
        self.registry = registry or ExtractorRegistry()

    def extract(
        self,
        text: str,
        definition: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> ExtractionResult:
        policies = definition.get("field_policies", {})
        schema = definition.get("input_schema", {})
        properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
        if not isinstance(policies, Mapping):
            return ExtractionResult(
                issues=[ExtractionIssue("", "definition_error", "field_policies phải là object")]
            )
        result = ExtractionResult()
        request_context = context or {}
        for field_name, policy in policies.items():
            if not isinstance(field_name, str) or not isinstance(policy, Mapping):
                result.issues.append(
                    ExtractionIssue(str(field_name), "definition_error", "field policy không hợp lệ")
                )
                continue
            extraction = policy.get("extraction", {})
            if not isinstance(extraction, Mapping):
                result.issues.append(
                    ExtractionIssue(field_name, "definition_error", "extraction policy phải là object")
                )
                result.missing.append(field_name)
                continue
            method = extraction.get("method")
            extractor = self.registry.get(str(method))
            if extractor is None:
                outcome = ExtractionOutcome(
                    False,
                    issue=ExtractionIssue(
                        field_name, "unavailable", f"extraction method chưa đăng ký: {method!r}"
                    ),
                )
            else:
                field_schema = properties.get(field_name, {})
                request = ExtractionRequest(
                    text=text,
                    field_name=field_name,
                    policy=extraction,
                    schema=field_schema if isinstance(field_schema, Mapping) else {},
                    context=request_context,
                )
                outcome = extractor.extract(request)
            if outcome.found:
                result.values[field_name] = outcome.value
            else:
                result.missing.append(field_name)
            if outcome.issue is not None:
                result.issues.append(outcome.issue)
        return result
