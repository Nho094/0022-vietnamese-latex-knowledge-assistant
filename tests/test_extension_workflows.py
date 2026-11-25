"""Regression checks cho các luồng mới của VS Code Extension."""

from __future__ import annotations

import json
import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import PlainTextResponse

import api.server as server
import core.config as config
from core.preprocess import preprocess_fingerprint
from core.pipeline import Pipeline
from core.definitions import DefinitionRegistry
from core.retrieval import MockSentenceTransformer, Retriever


def _definition(intent: str) -> dict:
    return {
        "intent": intent,
        "definition_version": "2.0",
        "aliases": ["một kiến thức kiểm thử"],
        "describe": "Kiến thức dùng để kiểm tra thao tác quản lý.",
        "category": "custom",
        "domain": "test",
        "input_schema": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "field_policies": {},
        "transform": {"engine": "jsonata", "expression": "$"},
        "render": {"engine": "jinja2", "templates": {"latex": r"\text{ test }"}},
        "slots": {},
        "templates": {"latex": r"\text{ test }", "python": "", "cpp": ""},
        "kind": "snippet",
        "requires": [],
        "compilable": False,
    }


def _registry(tmp_path: Path, monkeypatch, definitions: list[dict] | None = None) -> DefinitionRegistry:
    path = tmp_path / "intents.json"
    path.write_text(json.dumps(definitions or [], ensure_ascii=False), encoding="utf-8")
    retriever = Retriever(
        model=MockSentenceTransformer(embedding_dim=16),
        intents=[],
        cache_dir=tmp_path / "cache",
        use_cache=False,
    )
    registry = DefinitionRegistry(path, retriever=retriever)
    registry.attach_retriever(retriever)
    monkeypatch.setattr(server, "_get_definition_registry", lambda: registry)

    class RuntimePipeline:
        _dynamic_generic_intents: set[str] = set()

        def sync_definition_snapshot(self, snapshot, dynamic_intents=None) -> None:
            self._dynamic_generic_intents = set(dynamic_intents or ())

    monkeypatch.setattr(server, "pipeline", RuntimePipeline())
    return registry


def test_builder_compile_preview_supports_static_knowledge_without_fields() -> None:
    request = server.BuilderCompileRequest(
        name="Quy tắc lũy thừa",
        description="Hiển thị quy tắc lũy thừa cơ bản.",
        category="tex_math",
        examples=["Viết x mũ hai"],
        segments=[server.BuilderSegment(kind="text", value=r"x^{2} = x \cdot x")],
    )

    compiled = server.compile_builder_definition(request)
    preview = server.preview_definition(
        server.DefinitionPreviewRequest(
            definition=compiled["definition"], values={}, target="latex"
        )
    )

    assert compiled["valid"] is True
    assert preview["status"] == "ok"
    assert preview["outputs"]["latex"] == r"x^{2} = x \cdot x"


def test_builder_default_and_required_values_are_generated_from_form_data() -> None:
    request = server.BuilderCompileRequest(
        name="Chào người dùng",
        description="Tạo lời chào có tên người nhận.",
        category="tex_text",
        fields=[
            server.BuilderField(
                key="name", label="Tên", value_type="string", required=True
            ),
            server.BuilderField(
                key="suffix",
                label="Lời nhắn thêm",
                value_type="string",
                has_default=True,
                default="!",
            ),
        ],
        segments=[
            server.BuilderSegment(kind="text", value="Xin chào "),
            server.BuilderSegment(kind="field", value="name"),
            server.BuilderSegment(kind="field", value="suffix"),
        ],
    )

    compiled = server.compile_builder_definition(request)
    definition = compiled["definition"]
    missing = server.preview_definition(
        server.DefinitionPreviewRequest(definition=definition, values={})
    )
    output = server.preview_definition(
        server.DefinitionPreviewRequest(definition=definition, values={"name": "An"})
    )

    assert missing["status"] == "need_slots"
    assert "$.name" in missing["missing"]
    assert output["status"] == "ok"
    assert output["outputs"]["latex"] == "Xin chào An!"


def test_builder_literal_jinja_syntax_is_kept_as_literal_text() -> None:
    request = server.BuilderCompileRequest(
        name="Văn bản mẫu",
        description="In nguyên văn một biểu thức mẫu.",
        category="tex_text",
        segments=[server.BuilderSegment(kind="text", value="{{ nguy_co }}")],
    )

    compiled = server.compile_builder_definition(request)
    preview = server.preview_definition(
        server.DefinitionPreviewRequest(definition=compiled["definition"], values={})
    )

    assert preview["status"] == "ok"
    assert preview["outputs"]["latex"] == "{{ nguy_co }}"


def test_invalid_builder_field_reference_is_rejected() -> None:
    request = server.BuilderCompileRequest(
        name="Sai mẫu",
        description="Thử trường không tồn tại.",
        category="tex_text",
        segments=[server.BuilderSegment(kind="field", value="missing")],
    )

    with pytest.raises(HTTPException) as error:
        server.compile_builder_definition(request)

    assert error.value.status_code == 422


def test_delete_definition_keeps_recoverable_snapshot_and_syncs_registry(
    tmp_path: Path, monkeypatch
) -> None:
    registry = _registry(tmp_path, monkeypatch, [_definition("DELETE_ME")])
    previous_version = registry.version
    deleted = server.delete_definition("DELETE_ME")

    assert deleted["deleted"] is True
    assert registry.get("DELETE_ME") is None
    assert registry.path.with_name(
        f"intents.json.{previous_version}.bak"
    ).is_file()
    assert "DELETE_ME" not in server.pipeline._dynamic_generic_intents


def test_system_status_reports_missing_checkpoints_without_loading_models(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(server, "DEFAULT_CHECKPOINT_DIR", tmp_path / "no-classifier")
    monkeypatch.setattr(server, "SBERT_DIR", tmp_path / "no-retrieval")
    monkeypatch.setattr(config, "SBERT_DIR", tmp_path / "no-retrieval")
    monkeypatch.setattr(server, "INDEX_CACHE_DIR", tmp_path / "index-cache")
    monkeypatch.setattr(config, "PHOBERT_DIR", tmp_path / "no-tokenizer")
    monkeypatch.setattr(config, "HF_CACHE_DIR", tmp_path / "empty-hf-cache")
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-hf-hub"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf-home"))
    monkeypatch.setattr(server.importlib.util, "find_spec", lambda _name: object())

    status = server.system_status()

    assert status["classifier_available"] is False
    assert status["retrieval_model_available"] is False
    assert status["retrieval_index_available"] is False
    assert status["retrieval_cache_available"] is False
    assert status["status"] == "degraded"
    assert "model NLP khả dụng" in status["message"]


def _write_classifier_artifact(checkpoint: Path, tokenizer: Path, weight_name: str = "pytorch_model.bin") -> None:
    checkpoint.mkdir(parents=True)
    tokenizer.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "roberta", "num_labels": 2}), encoding="utf-8"
    )
    (checkpoint / weight_name).write_bytes(b"classifier-weights")
    (checkpoint / "id2label.json").write_text(json.dumps({"0": "A", "1": "B"}), encoding="utf-8")
    (checkpoint / "preprocessing_fingerprint.json").write_text(
        json.dumps({"fingerprint_sha256": preprocess_fingerprint()}), encoding="utf-8"
    )
    (tokenizer / "vocab.txt").write_text("<s>\n</s>\n", encoding="utf-8")
    (tokenizer / "bpe.codes").write_text("#version: 0.2\n", encoding="utf-8")
    (tokenizer / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def _write_retrieval_artifact(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    (model_dir / "modules.json").write_text(
        json.dumps([{"idx": 0, "name": "0", "path": "", "type": "transformer"}]),
        encoding="utf-8",
    )
    (model_dir / "config.json").write_text(json.dumps({"model_type": "bert"}), encoding="utf-8")
    (model_dir / "pytorch_model.bin").write_bytes(b"retrieval-weights")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")


def _patch_model_environment(monkeypatch, checkpoint: Path, tokenizer: Path, retrieval: Path, index: Path) -> None:
    monkeypatch.setattr(server, "DEFAULT_CHECKPOINT_DIR", checkpoint)
    monkeypatch.setattr(server, "PHOBERT_DIR", tokenizer)
    monkeypatch.setattr(config, "PHOBERT_DIR", tokenizer)
    monkeypatch.setattr(server, "SBERT_DIR", retrieval)
    monkeypatch.setattr(config, "SBERT_DIR", retrieval)
    monkeypatch.setattr(config, "HF_CACHE_DIR", index.parent / "empty-hf-cache")
    monkeypatch.setenv("HOME", str(index.parent / "empty-home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(index.parent / "empty-hf-hub"))
    monkeypatch.setenv("HF_HOME", str(index.parent / "empty-hf-home"))
    monkeypatch.setattr(server, "INDEX_CACHE_DIR", index)
    monkeypatch.setattr(server.importlib.util, "find_spec", lambda _name: object())


def test_system_status_classifier_only_reports_missing_retrieval(tmp_path: Path, monkeypatch) -> None:
    checkpoint, tokenizer = tmp_path / "ckpt", tmp_path / "tokenizer"
    _write_classifier_artifact(checkpoint, tokenizer, weight_name="model.safetensors")
    _patch_model_environment(monkeypatch, checkpoint, tokenizer, tmp_path / "missing-sbert", tmp_path / "index")

    status = server.system_status()

    assert status["classifier_available"] is True
    assert status["retrieval_model_available"] is False
    assert status["status"] == "degraded"
    assert "classifier-only" in status["message"]


def test_system_status_retrieval_only_distinguishes_model_from_index(tmp_path: Path, monkeypatch) -> None:
    retrieval = tmp_path / "sbert"
    _write_retrieval_artifact(retrieval)
    _patch_model_environment(monkeypatch, tmp_path / "missing-ckpt", tmp_path / "tokenizer", retrieval, tmp_path / "index")

    status = server.system_status()

    assert status["classifier_available"] is False
    assert status["retrieval_model_available"] is True
    assert status["retrieval_index_available"] is False
    assert "sẽ build ở lần hỏi đầu tiên" in status["retrieval_index_message"]
    assert "retrieval-only" in status["message"]


def test_system_status_reports_both_models_and_hash_valid_index(tmp_path: Path, monkeypatch) -> None:
    checkpoint, tokenizer, retrieval, index = (
        tmp_path / "ckpt", tmp_path / "tokenizer", tmp_path / "sbert", tmp_path / "index"
    )
    _write_classifier_artifact(checkpoint, tokenizer)
    _write_retrieval_artifact(retrieval)
    _patch_model_environment(monkeypatch, checkpoint, tokenizer, retrieval, index)
    intents = json.loads(server.RUNTIME_INTENTS_PATH.read_text(encoding="utf-8"))
    Retriever(
        model=MockSentenceTransformer(embedding_dim=8), intents=intents, cache_dir=index,
        model_name=config.SBERT_MODEL_NAME,
    )

    status = server.system_status()

    assert status["classifier_available"] is True
    assert status["retrieval_model_available"] is True
    assert status["retrieval_index_available"] is True
    assert status["status"] == "ready"


def test_system_status_rejects_directories_without_loadable_artifacts(tmp_path: Path, monkeypatch) -> None:
    checkpoint, retrieval = tmp_path / "ckpt", tmp_path / "sbert"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "roberta"}), encoding="utf-8")
    retrieval.mkdir()
    (retrieval / "modules.json").write_text("not-json", encoding="utf-8")
    _patch_model_environment(monkeypatch, checkpoint, tmp_path / "tokenizer", retrieval, tmp_path / "index")

    status = server.system_status()

    assert status["classifier_available"] is False
    assert status["retrieval_model_available"] is False
    assert "trọng số" in status["classifier_message"]


def test_system_status_names_missing_model_dependencies(tmp_path: Path, monkeypatch) -> None:
    checkpoint, tokenizer, retrieval = tmp_path / "ckpt", tmp_path / "tokenizer", tmp_path / "sbert"
    _write_classifier_artifact(checkpoint, tokenizer)
    _write_retrieval_artifact(retrieval)
    _patch_model_environment(monkeypatch, checkpoint, tokenizer, retrieval, tmp_path / "index")
    monkeypatch.setattr(
        server.importlib.util,
        "find_spec",
        lambda name: None if name in {"torch", "sentence_transformers"} else object(),
    )

    status = server.system_status()

    assert status["classifier_available"] is False
    assert status["classifier_missing_dependencies"] == ["torch"]
    assert status["retrieval_model_available"] is False
    assert status["retrieval_missing_dependencies"] == ["sentence-transformers"]
    assert "Thiếu thư viện" in status["message"]


def test_ask_returns_actionable_unknown_when_all_nlu_models_are_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        server,
        "pipeline",
        Pipeline(
            mode="phobert_only",
            checkpoint_dir=tmp_path / "missing-checkpoint",
            intents_path=server.DEFAULT_INTENTS_PATH,
        ),
    )

    result = server.ask_endpoint(server.AskRequest(text="Tạo một phân số"))

    assert result["status"] == "unknown"
    assert "Knowledge Manager vẫn hoạt động" in result["message"]
    assert "checkpoint PhoBERT" in result["message"]


def test_extension_registry_uses_user_storage_path_from_environment(
    tmp_path: Path, monkeypatch
) -> None:
    registry_path = tmp_path / "user-data" / "intents.json"
    monkeypatch.setenv("KA_INTENTS_PATH", str(registry_path))
    monkeypatch.setattr(server, "definition_registry", None)

    registry = server._get_definition_registry()

    assert registry.path == registry_path
    assert registry.snapshot() == {}
    assert not registry.path.exists()


def test_extension_session_middleware_rejects_requests_without_token(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KA_SESSION_TOKEN", "expected-token")

    def make_request(path: str, token: str | None = None) -> Request:
        headers = [] if token is None else [(b"x-ka-session", token.encode())]
        return Request(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": path,
                "raw_path": path.encode(),
                "query_string": b"",
                "headers": headers,
                "server": ("127.0.0.1", 8000),
                "client": ("127.0.0.1", 40000),
            }
        )

    async def call_next(_request: Request) -> PlainTextResponse:
        return PlainTextResponse("passed")

    unauthorized = asyncio.run(
        server.require_extension_session(make_request("/system/status"), call_next)
    )
    authorized = asyncio.run(
        server.require_extension_session(
            make_request("/system/status", "expected-token"), call_next
        )
    )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
