from __future__ import annotations

from api.server import app
import api.server as server
from api.server import AskRequest, ask_endpoint
from core.pipeline import PipelineResult, STATUS_UNSUPPORTED_TARGET


def test_api_keeps_legacy_routes_and_exposes_knowledge_management_routes() -> None:
    routes = {
        (route.path, frozenset(getattr(route, "methods", None) or ()))
        for route in app.routes
    }
    assert ("/ask", frozenset({"POST"})) in routes
    assert ("/healthz", frozenset({"GET"})) in routes
    assert ("/knowledge/publish", frozenset({"POST"})) in routes
    assert ("/knowledge/reload", frozenset({"POST"})) in routes


def test_api_returns_unsupported_target_status(monkeypatch) -> None:
    class FakePipeline:
        target = "python"
        mode = "retrieval_only"

        def predict(self, _text: str) -> PipelineResult:
            return PipelineResult(
                status=STATUS_UNSUPPORTED_TARGET,
                query="hello",
                message="No Python template",
            )

    monkeypatch.setattr(server, "create_pipeline", lambda **_kwargs: FakePipeline())
    result = ask_endpoint(AskRequest(text="hello", target="python"))
    assert result["status"] == "unsupported_target"
