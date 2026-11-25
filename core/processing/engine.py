"""JSONata execution layer với guardrails và registry processor.

JSONata là ngôn ngữ biến đổi dữ liệu, không phải Python; tuyệt đối không dùng
`eval`. Expression là dữ liệu trong definition. Timeout, stack và sequence
limits giảm rủi ro biểu thức đệ quy/tạo dữ liệu quá lớn; template vẫn do renderer
riêng xử lý ở KA-20.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Protocol

import jsonata

from core.config import (
    DEFAULT_JSONATA_SEQUENCE_LIMIT,
    DEFAULT_JSONATA_STACK_LIMIT,
    DEFAULT_JSONATA_TIMEOUT_MS,
)


class TransformError(ValueError):
    """Expression sai, engine không hỗ trợ hoặc runtime transform thất bại."""

    def __init__(self, engine: str, message: str) -> None:
        self.engine = engine
        self.message = message
        super().__init__(f"{engine}: {message}")


class Processor(Protocol):
    """Interface chung cho processor đăng ký theo engine identifier."""

    def transform(self, expression: str, value: Any) -> Any:
        """Biến đổi một JSON value thành JSON value khác."""


class JsonataProcessor:
    """Adapter JSONata có giới hạn tài nguyên cấu hình tập trung."""

    def __init__(
        self,
        timeout_ms: int = DEFAULT_JSONATA_TIMEOUT_MS,
        stack_limit: int = DEFAULT_JSONATA_STACK_LIMIT,
        sequence_limit: int = DEFAULT_JSONATA_SEQUENCE_LIMIT,
    ) -> None:
        if timeout_ms <= 0 or stack_limit <= 0 or sequence_limit <= 0:
            raise ValueError("JSONata limits phải là số nguyên dương")
        self.timeout_ms = timeout_ms
        self.stack_limit = stack_limit
        self.sequence_limit = sequence_limit

    def transform(self, expression: str, value: Any) -> Any:
        """Evaluate JSONata trên bản sao input; trả lỗi có engine rõ ràng."""
        if not isinstance(expression, str) or not expression.strip():
            raise TransformError("jsonata", "expression phải là chuỗi không rỗng")
        try:
            compiled = jsonata.Jsonata(
                expression,
                timeout=self.timeout_ms,
                stack=self.stack_limit,
                sequence=self.sequence_limit,
            )
            return compiled.evaluate(copy.deepcopy(value))
        except Exception as exc:  # JSONata runtime exceptions become API-safe errors
            raise TransformError("jsonata", str(exc)) from exc


class ProcessorRegistry:
    """Engine registry: JSON data selects a registered processor, never code."""

    def __init__(self) -> None:
        self._processors: dict[str, Processor] = {"jsonata": JsonataProcessor()}

    def register(self, engine: str, processor: Processor, *, replace: bool = False) -> None:
        """Đăng ký extension đáng tin của ứng dụng, không load theo import path."""
        if not engine or (engine in self._processors and not replace):
            raise ValueError(f"Engine rỗng hoặc đã được đăng ký: {engine!r}")
        self._processors[engine] = processor

    def get(self, engine: str) -> Processor | None:
        """Lấy processor; engine lạ sẽ bị từ chối bởi TransformEngine."""
        return self._processors.get(engine)


class TransformEngine:
    """Đọc transform contract rồi dispatch tới engine registry."""

    def __init__(self, registry: ProcessorRegistry | None = None) -> None:
        self.registry = registry or ProcessorRegistry()

    def transform(self, definition: Mapping[str, Any], value: Any) -> Any:
        """Thực thi definition.transform.expression với engine khai báo."""
        transform = definition.get("transform")
        if not isinstance(transform, Mapping):
            raise TransformError("unknown", "definition.transform phải là object")
        engine = transform.get("engine")
        expression = transform.get("expression")
        if not isinstance(engine, str) or not isinstance(expression, str):
            raise TransformError("unknown", "transform cần engine và expression dạng string")
        processor = self.registry.get(engine)
        if processor is None:
            raise TransformError(engine, "processor chưa được đăng ký")
        return processor.transform(expression, value)
