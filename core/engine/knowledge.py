"""Điều phối extraction → policy → normalization → validate → transform → render."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from core.config import (
    STATUS_NEED_SLOTS,
    STATUS_OK,
    STATUS_PROCESSING_ERROR,
    TARGET_LATEX,
)
from core.extraction import GenericExtractionEngine
from core.policies import (
    FieldPolicyEngine,
    GenericSchemaValidator,
    NormalizationError,
    SchemaIssue,
    ValueNormalizer,
)
from core.processing import TransformEngine, TransformError
from core.rendering import RenderEngine, RenderError


@dataclass
class GenericResult:
    """Kết quả có cấu trúc qua các stage của Knowledge Engine."""

    status: str
    values: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    errors: list[SchemaIssue] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Kết quả hoàn chỉnh khi tất cả stage sinh output thành công."""
        return self.status == STATUS_OK


class GenericKnowledgeEngine:
    """Chạy primitive đã đăng ký dựa trên definition, không xét intent cụ thể."""

    def __init__(
        self,
        extraction: GenericExtractionEngine | None = None,
        policies: FieldPolicyEngine | None = None,
        normalizer: ValueNormalizer | None = None,
        validator: GenericSchemaValidator | None = None,
        transformer: TransformEngine | None = None,
        renderer: RenderEngine | None = None,
    ) -> None:
        self.extraction = extraction or GenericExtractionEngine()
        self.policies = policies or FieldPolicyEngine()
        self.normalizer = normalizer or ValueNormalizer()
        self.validator = validator or GenericSchemaValidator()
        self.transformer = transformer or TransformEngine()
        self.renderer = renderer or RenderEngine()

    def process(
        self,
        query: str,
        definition: dict[str, Any],
        *,
        target: str = TARGET_LATEX,
        manual_values: dict[str, Any] | None = None,
    ) -> GenericResult:
        """Tiếp tục từ definition đã chọn; manual values sửa bước lấy dữ liệu."""
        stages: dict[str, dict[str, Any]] = {
            "intent_detection": {
                "status": "accepted",
                "intent": definition.get("intent"),
            }
        }
        extracted = self.extraction.extract(
            query,
            definition,
            context={"manual_values": manual_values or {}},
        )
        raw_values = dict(extracted.values)
        # Trường người dùng nhập trực tiếp thắng giá trị regex đã bóc tách.
        if manual_values:
            raw_values.update(manual_values)
        stages["extraction"] = {
            "status": STATUS_OK if not extracted.missing else "incomplete",
            "values": dict(raw_values),
            "missing": list(extracted.missing),
            "issues": [asdict(issue) for issue in extracted.issues],
        }

        schema = definition.get("input_schema", {})
        field_policies = definition.get("field_policies", {})
        policy_result = self.policies.apply(raw_values, schema, field_policies)
        stages["field_policy"] = {
            "status": STATUS_OK if not policy_result.errors else "error",
            "values": dict(policy_result.value),
            "missing": list(policy_result.missing),
            "errors": [asdict(error) for error in policy_result.errors],
        }
        if policy_result.missing or policy_result.errors:
            return GenericResult(
                STATUS_NEED_SLOTS,
                policy_result.value,
                policy_result.missing,
                policy_result.errors,
                stages=stages,
            )

        properties = schema.get("properties", {})
        normalized: dict[str, Any] = {}
        normalization_errors: list[SchemaIssue] = []
        for name, value in policy_result.value.items():
            value_schema = properties.get(name, {}) if isinstance(properties, dict) else {}
            if not isinstance(value_schema, dict):
                value_schema = {}
            try:
                normalized[name] = self.normalizer.normalize(value, value_schema, f"$.{name}")
            except NormalizationError as exc:
                normalization_errors.append(exc.issue)
        stages["normalization"] = {
            "status": "error" if normalization_errors else STATUS_OK,
            "values": dict(normalized),
            "errors": [asdict(error) for error in normalization_errors],
        }
        if normalization_errors:
            return GenericResult(
                STATUS_NEED_SLOTS,
                normalized,
                [issue.path for issue in normalization_errors],
                normalization_errors,
                stages=stages,
            )

        validation_errors = self.validator.validate(normalized, schema)
        stages["validation"] = {
            "status": "error" if validation_errors else STATUS_OK,
            "errors": [asdict(error) for error in validation_errors],
        }
        if validation_errors:
            return GenericResult(
                STATUS_NEED_SLOTS,
                normalized,
                [issue.path for issue in validation_errors],
                validation_errors,
                stages=stages,
            )

        try:
            transformed = self.transformer.transform(definition, normalized)
            stages["transform"] = {"status": STATUS_OK, "value": transformed}
        except TransformError as exc:
            stages["transform"] = {"status": "error", "error": str(exc)}
            return GenericResult(STATUS_PROCESSING_ERROR, normalized, errors=[], stages=stages)

        try:
            outputs = self.renderer.render(
                definition,
                transformed,
                targets=[target],
            )
            stages["render"] = {"status": STATUS_OK, "outputs": dict(outputs)}
        except RenderError as exc:
            stages["render"] = {"status": "error", "error": str(exc)}
            return GenericResult(STATUS_PROCESSING_ERROR, normalized, stages=stages)
        return GenericResult(STATUS_OK, normalized, outputs=outputs, stages=stages)
