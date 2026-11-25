"""KA-22: publish/reload/rollback động với index SBERT nhất quán."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from core.definitions import DefinitionRegistry, DefinitionRegistryError
from core.retrieval import MockSentenceTransformer, Retriever


def _definition(intent: str, alias: str, template: str = "{{ value }}") -> dict:
    return {
        "intent": intent,
        "definition_version": "2.0",
        "aliases": [alias],
        "describe": f"Definition {alias}",
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
        "render": {"engine": "jinja2", "templates": {"latex": template}},
        "slots": {},
        "templates": {"latex": template, "python": "", "cpp": ""},
        "kind": "snippet",
        "requires": [],
        "compilable": False,
    }


def _registry(tmp_path: Path, *definitions: dict) -> tuple[DefinitionRegistry, Retriever, Path]:
    path = tmp_path / "intents.json"
    path.write_text(json.dumps(list(definitions), ensure_ascii=False), encoding="utf-8")
    retriever = Retriever(
        model=MockSentenceTransformer(embedding_dim=16),
        intents=[],
        cache_dir=tmp_path / "cache",
        use_cache=False,
    )
    registry = DefinitionRegistry(path, retriever=retriever)
    registry.attach_retriever(retriever)
    return registry, retriever, path


def test_publish_validates_persists_backs_up_and_updates_sbert_retrieval(tmp_path: Path) -> None:
    existing = _definition("CUSTOM_EXISTING", "existing concept")
    registry, retriever, path = _registry(tmp_path, existing)
    old_version = registry.version
    new_definition = _definition("BRAND_NEW_KNOWLEDGE", "orbital matrix")

    new_version = registry.publish(new_definition)
    result = retriever.retrieve("orbital matrix", threshold_theta=-1.0)

    assert new_version != old_version
    assert registry.get("BRAND_NEW_KNOWLEDGE")["aliases"] == ["orbital matrix"]
    assert "BRAND_NEW_KNOWLEDGE" in retriever
    assert result.candidates[0].intent == "BRAND_NEW_KNOWLEDGE"
    assert path.with_name(f"intents.json.{old_version}.bak").exists()
    assert len(json.loads(path.read_text(encoding="utf-8"))) == 2


def test_publish_requires_explicit_overwrite_and_v2_definition(tmp_path: Path) -> None:
    definition = _definition("CUSTOM", "custom alias")
    registry, retriever, _path = _registry(tmp_path, definition)

    with pytest.raises(DefinitionRegistryError, match="overwrite=True"):
        registry.publish(_definition("CUSTOM", "new alias"))
    with pytest.raises(DefinitionRegistryError, match="Contract v2"):
        registry.publish({"intent": "legacy", "slots": {}, "templates": {"latex": "x"}})
    assert registry.get("CUSTOM")["aliases"] == ["custom alias"]
    assert "CUSTOM" in retriever


def test_rollback_restores_previous_disk_snapshot_and_retrieval_rows(tmp_path: Path) -> None:
    definition = _definition("ROLLBACK_TARGET", "original alias")
    registry, retriever, path = _registry(tmp_path, definition)
    original_version = registry.version
    registry.publish(_definition("ROLLBACK_TARGET", "updated alias"), overwrite=True)
    assert registry.version != original_version

    restored_version = registry.rollback(original_version)

    assert restored_version == original_version
    assert registry.get("ROLLBACK_TARGET")["aliases"] == ["original alias"]
    assert json.loads(path.read_text(encoding="utf-8"))[0]["aliases"] == ["original alias"]
    assert "ROLLBACK_TARGET" in retriever


def test_rollback_finds_persistent_backup_after_registry_restart(tmp_path: Path) -> None:
    registry, _retriever, path = _registry(
        tmp_path, _definition("PERSISTENT", "before restart")
    )
    previous_version = registry.version
    registry.publish(_definition("PERSISTENT", "after restart"), overwrite=True)

    restarted_retriever = Retriever(
        model=MockSentenceTransformer(embedding_dim=16),
        intents=[],
        cache_dir=tmp_path / "restarted-cache",
        use_cache=False,
    )
    restarted = DefinitionRegistry(path, retriever=restarted_retriever)
    restarted.attach_retriever(restarted_retriever)

    assert restarted.rollback() == previous_version
    assert restarted.get("PERSISTENT")["aliases"] == ["before restart"]


def test_reload_adopts_valid_external_change_and_rejects_invalid_without_mutating(tmp_path: Path) -> None:
    registry, retriever, path = _registry(tmp_path, _definition("BASE", "base concept"))
    old_version = registry.version
    path.write_text(
        json.dumps([_definition("BASE", "base concept"), _definition("EXTERNAL", "external concept")]),
        encoding="utf-8",
    )

    reloaded_version = registry.reload()

    assert reloaded_version != old_version
    assert "EXTERNAL" in registry.snapshot()
    assert "EXTERNAL" in retriever
    active_version = registry.version
    path.write_text("{invalid", encoding="utf-8")
    with pytest.raises(DefinitionRegistryError, match="Không đọc được"):
        registry.reload()
    assert registry.version == active_version
    assert "EXTERNAL" in registry.snapshot()


def test_failed_persist_restores_sbert_and_keeps_active_snapshot(tmp_path: Path, monkeypatch) -> None:
    registry, retriever, _path = _registry(tmp_path, _definition("BASE", "base concept"))
    old_version = registry.version
    original_atomic_write = registry._atomic_write

    def fail_definition_file(path: Path, data: bytes) -> None:
        if path == registry.path:
            raise OSError("disk full")
        original_atomic_write(path, data)

    monkeypatch.setattr(registry, "_atomic_write", fail_definition_file)

    with pytest.raises(DefinitionRegistryError, match="Publish thất bại"):
        registry.publish(_definition("TEMPORARY", "temporary new concept"))

    assert registry.version == old_version
    assert registry.get("TEMPORARY") is None
    assert "TEMPORARY" not in retriever


def test_publish_and_retrieval_are_safe_under_concurrent_reads(tmp_path: Path) -> None:
    registry, retriever, _path = _registry(tmp_path, _definition("BASE", "base concept"))

    def read_index(_index: int) -> list[str]:
        return [candidate.intent for candidate in retriever.retrieve("base concept").candidates]

    with ThreadPoolExecutor(max_workers=4) as pool:
        readers = [pool.submit(read_index, index) for index in range(8)]
        registry.publish(_definition("CONCURRENT", "concurrent concept"))
        results = [future.result() for future in readers]

    assert all("BASE" in intents for intents in results)
    assert "CONCURRENT" in retriever
