"""Sandboxed Jinja2 rendering với output validation theo format.

Templates không có loader, globals hay truy cập attributes/callables Python.
Chỉ JSON-compatible context được nhận. LaTeX text phải dùng filter
`latex_escape`; kết quả LaTeX luôn chạy qua static_check, giữ nguyên guardrail
của `core.latex_tools` (compile chỉ là bước riêng, không được tuyên bố ở đây).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from jinja2 import StrictUndefined, TemplateError
from jinja2.sandbox import SandboxedEnvironment
from jinja2.runtime import LoopContext

from core.latex_tools import escape_text, static_check


class RenderError(ValueError):
    """Lỗi template, output format hoặc static security check."""

    def __init__(self, target: str, message: str) -> None:
        self.target = target
        self.message = message
        super().__init__(f"{target}: {message}")


class Renderer(Protocol):
    """Renderer protocol; output type được khai báo bằng key trong definition."""

    def render(self, template: str, context: Mapping[str, Any]) -> str:
        """Render một template với context JSON-compatible."""


class _RestrictedEnvironment(SandboxedEnvironment):
    """Jinja environment cấm object attribute và callable từ context."""

    _SAFE_LOOP_ATTRIBUTES = frozenset(
        {"index", "index0", "revindex", "revindex0", "first", "last", "length", "depth", "depth0"}
    )

    def is_safe_attribute(self, obj: Any, attr: str, value: Any) -> bool:
        del value
        return isinstance(obj, LoopContext) and attr in self._SAFE_LOOP_ATTRIBUTES

    def is_safe_callable(self, obj: Any) -> bool:
        del obj
        return False


def _latex_escape(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("latex_escape chỉ nhận chuỗi")
    return escape_text(value)


class Jinja2Renderer:
    """Render Jinja template trong sandbox không có filesystem loader."""

    def __init__(self) -> None:
        environment = _RestrictedEnvironment(
            loader=None,
            autoescape=False,
            undefined=StrictUndefined,
            enable_async=False,
        )
        environment.globals.clear()
        environment.filters["latex_escape"] = _latex_escape
        self._environment = environment

    def render(self, template: str, context: Mapping[str, Any]) -> str:
        """Render một string template, từ chối context không phải dữ liệu JSON."""
        if not isinstance(template, str):
            raise RenderError("jinja2", "template phải là chuỗi")
        if not isinstance(context, Mapping):
            raise RenderError("jinja2", "context phải là object")
        try:
            safe_context = json.loads(json.dumps(context, ensure_ascii=False))
            compiled = self._environment.from_string(template)
            return compiled.render(safe_context)
        except (TypeError, ValueError, TemplateError) as exc:
            raise RenderError("jinja2", str(exc)) from exc


class RendererRegistry:
    """Map engine names tới renderer tin cậy, không load module từ JSON."""

    def __init__(self) -> None:
        self._renderers: dict[str, Renderer] = {"jinja2": Jinja2Renderer()}

    def register(self, engine: str, renderer: Renderer, *, replace: bool = False) -> None:
        """Đăng ký renderer extension do ứng dụng quản lý."""
        if not engine or (engine in self._renderers and not replace):
            raise ValueError(f"Renderer engine rỗng hoặc đã đăng ký: {engine!r}")
        self._renderers[engine] = renderer

    def get(self, engine: str) -> Renderer | None:
        """Lấy renderer theo identifier từ definition."""
        return self._renderers.get(engine)


class RenderEngine:
    """Render một hoặc nhiều output templates từ `definition.render`."""

    def __init__(self, registry: RendererRegistry | None = None) -> None:
        self.registry = registry or RendererRegistry()

    def render(
        self,
        definition: Mapping[str, Any],
        context: Mapping[str, Any],
        targets: Sequence[str] | None = None,
    ) -> dict[str, str]:
        """Render mọi target hoặc tập target đã chọn; LaTeX luôn static-check."""
        render_config = definition.get("render")
        if not isinstance(render_config, Mapping):
            raise RenderError("unknown", "definition.render phải là object")
        engine = render_config.get("engine")
        templates = render_config.get("templates")
        if not isinstance(engine, str) or not isinstance(templates, Mapping):
            raise RenderError("unknown", "render cần engine và templates object")
        renderer = self.registry.get(engine)
        if renderer is None:
            raise RenderError(engine, "renderer chưa được đăng ký")
        selected = list(templates) if targets is None else list(targets)
        outputs: dict[str, str] = {}
        for target in selected:
            template = templates.get(target)
            if not isinstance(template, str):
                raise RenderError(target, "target không có template string")
            try:
                output = renderer.render(template, context)
            except RenderError as exc:
                raise RenderError(target, exc.message) from exc
            if target.casefold() == "latex":
                problems = static_check(output)
                if problems:
                    raise RenderError(target, "; ".join(problems))
            outputs[target] = output
        return outputs
