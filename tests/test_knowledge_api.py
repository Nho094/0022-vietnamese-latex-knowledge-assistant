"""KA-23: knowledge-management API validation and safe publish workflow."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import HTTPException
import pytest

import api.server as server
from core.definitions import DefinitionRegistry
from core.retrieval import MockSentenceTransformer, Retriever


def _definition(intent: str = "API_CUSTOM") -> dict:
    return {
        "intent": intent,
        "definition_version": "2.0",
        "aliases": ["custom knowledge alias"],
        "describe": "Custom data driven definition",
        "category": "custom",
        "domain": "test",
        "input_schema": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        "field_policies": {
            "value": {
                "value_type": "string",
                "extraction": {"method": "manual"},
                "required": True,
                "on_missing": {"action": "ask_user"},
            }
        },
        "transform": {"engine": "jsonata", "expression": "$"},
        "render": {"engine": "jinja2", "templates": {"latex": r"\text{ {{ value | latex_escape }} }"}},
        "slots": {},
        "templates": {"latex": r"\text{ <<value>> }", "python": "", "cpp": ""},
        "kind": "snippet",
        "requires": [],
        "compilable": False,
    }


def _registry(tmp_path: Path, monkeypatch, definitions: list[dict] | None = None) -> tuple[DefinitionRegistry, Retriever]:
    bank_path = tmp_path / "intents.json"
    bank_path.write_text(json.dumps(definitions or [], ensure_ascii=False), encoding="utf-8")
    retriever = Retriever(
        model=MockSentenceTransformer(embedding_dim=16),
        intents=[],
        cache_dir=tmp_path / "cache",
        use_cache=False,
    )
    registry = DefinitionRegistry(bank_path, retriever=retriever)
    registry.attach_retriever(retriever)
    monkeypatch.setattr(server, "_get_definition_registry", lambda: registry)
    synced: dict[str, object] = {}

    class RuntimePipeline:
        _dynamic_generic_intents: set[str] = set()

        def sync_definition_snapshot(self, snapshot, dynamic_intents=None) -> None:
            synced["snapshot"] = snapshot
            synced["dynamic_intents"] = dynamic_intents
            self._dynamic_generic_intents = set(dynamic_intents or ())

    monkeypatch.setattr(server, "pipeline", RuntimePipeline())
    monkeypatch.setattr(server, "_test_synced_runtime", synced, raising=False)
    return registry, retriever


def test_validate_preview_import_export_and_publish_execute_without_python_changes(tmp_path: Path, monkeypatch) -> None:
    registry, retriever = _registry(tmp_path, monkeypatch)
    definition = _definition()

    validated = server.validate_definition(server.DefinitionRequest(definition=definition))
    imported = server.validate_import(
        server.DefinitionImportRequest(definitions=[definition])
    )
    previewed = server.preview_definition(
        server.DefinitionPreviewRequest(
            definition=definition, values={"value": "A&B"}, target="latex"
        )
    )
    published = server.publish_definition(server.DefinitionPublishRequest(definition=definition))
    listed = server.list_definitions(search="custom knowledge")
    exported = server.export_definitions(intent="API_CUSTOM")

    assert validated["valid"] is True
    assert imported["requires_overwrite_confirmation"] is False
    assert previewed["status"] == "ok"
    assert previewed["outputs"]["latex"] == r"\text{ A\&B }"
    assert published["published"] is True
    assert listed["count"] == 1
    assert exported["definitions"][0]["intent"] == "API_CUSTOM"
    assert "API_CUSTOM" in retriever
    assert registry.get("API_CUSTOM") is not None
    assert "API_CUSTOM" in server._test_synced_runtime["dynamic_intents"]


def test_duplicate_publish_requires_explicit_overwrite_confirmation(tmp_path: Path, monkeypatch) -> None:
    _registry_instance, _retriever = _registry(tmp_path, monkeypatch, [_definition()])

    with pytest.raises(HTTPException) as conflict:
        server.publish_definition(
            server.DefinitionPublishRequest(definition=_definition(), overwrite=False)
        )
    confirmed = server.publish_definition(
        server.DefinitionPublishRequest(definition=_definition(), overwrite=True)
    )

    assert conflict.value.status_code == 409
    assert confirmed["published"] is True


def test_invalid_nested_definition_is_rejected_before_publish_and_disk_is_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    registry, _retriever = _registry(tmp_path, monkeypatch)
    definition = _definition("INVALID_API")
    definition["input_schema"]["properties"]["value"] = {"type": "not-a-type"}
    before = registry.path.read_bytes()

    with pytest.raises(HTTPException) as response:
        server.publish_definition(server.DefinitionPublishRequest(definition=definition))

    assert response.value.status_code == 422
    assert registry.path.read_bytes() == before
    assert registry.get("INVALID_API") is None


def test_import_batch_detects_conflicts_without_writing_data(tmp_path: Path, monkeypatch) -> None:
    registry, _retriever = _registry(tmp_path, monkeypatch, [_definition()])
    before = registry.path.read_bytes()

    response = server.validate_import(
        server.DefinitionImportRequest(
            definitions=[_definition(), _definition("NEW_IMPORT")]
        )
    )

    assert response["valid"] is True
    assert response["conflicts"] == ["API_CUSTOM"]
    assert response["requires_overwrite_confirmation"] is True
    assert registry.path.read_bytes() == before


def test_reload_rollback_routes_use_registry_and_keep_legacy_ask_healthz(tmp_path: Path, monkeypatch) -> None:
    registry, _retriever = _registry(tmp_path, monkeypatch, [_definition()])
    first_version = registry.version
    registry.publish(_definition("SECOND_API"))

    rollback = server.rollback_definitions(server.RollbackRequest(version=first_version))
    reload_result = server.reload_definitions()
    health = server.healthz()

    assert rollback["rolled_back"] is True
    assert reload_result["reloaded"] is True
    assert health == {"status": "ok"}
    assert "API_CUSTOM" in registry.snapshot()
