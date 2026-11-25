"""Kiểm thử Universal Knowledge Engine: TARGET_AUTO, định tuyến Tầng 2 theo languageId,
lexical keyword boost, tự học learned_aliases và sinh tập train/val mốc 10 độc lập test.csv.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from core.config import (
    ENGINE_VERSION_GENERIC,
    MILESTONE_TRAIN_THRESHOLD,
    MODE_RETRIEVAL_ONLY,
    TARGET_AUTO,
    TEST_CSV_PATH,
    resolve_language_routing,
    resolve_target_for_templates,
)
from core.learning import KnowledgeLearningManager
from core.pipeline import CandidateChoice, Pipeline, _apply_lexical_keyword_boost
from core.retrieval import MockSentenceTransformer, Retriever
import api.server as server


def _universal_definitions() -> list[dict]:
    return [
        {
            "intent": "USER_SETUP_SUNSHINE",
            "definition_version": "2.0",
            "display_name": "Cài đặt Sunshine Linux",
            "aliases": ["cài sunshine trên linux", "mở cổng stream sunshine"],
            "describe": "Hướng dẫn cài Sunshine và mở port 47989 trên Linux",
            "category": "sysadmin",
            "domain": "user-created",
            "input_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "field_policies": {},
            "transform": {"engine": "jsonata", "expression": "$", "mode": "identity"},
            "render": {
                "engine": "jinja2",
                "templates": {"text": "1. Cài Sunshine AppImage/Flatpak\n2. Mở port 47989 UDP/TCP"},
            },
            "slots": {},
            "templates": {"latex": "", "python": "", "cpp": ""},
            "kind": "snippet",
            "requires": [],
            "compilable": False,
        },
        {
            "intent": "USER_PYTHON_FIBONACCI",
            "definition_version": "2.0",
            "display_name": "Hàm Fibonacci Python",
            "aliases": ["viết hàm fibonacci", "dãy số fibonacci python"],
            "learned_aliases": ["hàm fib đệ quy tối ưu"],
            "describe": "Sinh hàm Fibonacci chuẩn PEP8 bằng Python",
            "category": "python",
            "domain": "user-created",
            "input_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "field_policies": {},
            "transform": {"engine": "jsonata", "expression": "$", "mode": "identity"},
            "render": {
                "engine": "jinja2",
                "templates": {
                    "python": "def fib(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
                    "text": "Hàm Fibonacci dùng vòng lặp O(n).",
                },
            },
            "slots": {},
            "templates": {"latex": "", "python": "", "cpp": ""},
            "kind": "snippet",
            "requires": [],
            "compilable": False,
        },
        {
            "intent": "MATH_PHAN_SO_DEMO",
            "definition_version": "2.0",
            "display_name": "Phân số LaTeX",
            "aliases": ["viết phân số", "tạo phân số"],
            "describe": "Tạo biểu thức phân số",
            "category": "tex_math",
            "domain": "latex",
            "input_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {"tu": {"type": "integer"}, "mau": {"type": "integer"}},
                "required": ["tu", "mau"],
                "additionalProperties": False,
            },
            "field_policies": {
                "tu": {"value_type": "integer", "extraction": {"method": "regex", "pattern": r"(\d+)"}, "required": True, "on_missing": {"action": "ask_user"}},
                "mau": {"value_type": "integer", "extraction": {"method": "regex", "pattern": r"/(\d+)"}, "required": True, "on_missing": {"action": "ask_user"}},
            },
            "transform": {"engine": "jsonata", "expression": "$", "mode": "identity"},
            "render": {
                "engine": "jinja2",
                "templates": {"latex": r"\frac{ {{ tu }} }{ {{ mau }} }", "obsidian": r"$\frac{ {{ tu }} }{ {{ mau }} }$"},
            },
            "slots": {},
            "templates": {"latex": "", "python": "", "cpp": ""},
            "kind": "snippet",
            "requires": [],
            "compilable": False,
        },
    ]


def test_language_routing_and_target_auto_fallback() -> None:
    assert resolve_language_routing("python") == ("python", ("python",))
    assert resolve_language_routing("plaintext") == (TARGET_AUTO, ())
    assert resolve_language_routing("latex")[0] == "latex"

    templates = {"python": "def f(): pass", "text": "Mo ta"}
    assert resolve_target_for_templates(templates, TARGET_AUTO) == "text"
    assert resolve_target_for_templates(templates, TARGET_AUTO, preferred_target="python") == "python"
    assert resolve_target_for_templates({"text": "Guide"}, TARGET_AUTO, preferred_target="python") == "text"
    assert resolve_target_for_templates({"text": "Guide"}, "latex") is None


def test_lexical_boost_and_learned_aliases_preserve_retriever_base_scores(tmp_path: Path) -> None:
    defs = _universal_definitions()
    retriever = Retriever(
        model=MockSentenceTransformer(embedding_dim=16),
        intents=defs,
        cache_dir=tmp_path / "cache",
        use_cache=False,
    )
    # Index đã nạp cả learned_aliases
    row_texts = [row.text for row in retriever._rows if row.intent == "USER_PYTHON_FIBONACCI"]
    assert any("hàm fib đệ quy tối ưu" in text for text in row_texts)

    intents_bank = {item["intent"]: item for item in defs}
    candidates = [
        CandidateChoice(intent="MATH_PHAN_SO_DEMO", score=0.72, describe="Tạo biểu thức phân số", category="tex_math"),
        CandidateChoice(intent="USER_SETUP_SUNSHINE", score=0.68, describe="Hướng dẫn cài Sunshine", category="sysadmin"),
    ]
    boosted = _apply_lexical_keyword_boost("setup sunshine linux", candidates, intents_bank)
    assert boosted[0].intent == "USER_SETUP_SUNSHINE"
    assert boosted[0].score > 0.68


def test_auto_target_pipeline_renders_universal_text_inside_python_editor(tmp_path: Path) -> None:
    defs = _universal_definitions()
    bank_path = tmp_path / "intents.json"
    bank_path.write_text(json.dumps(defs, ensure_ascii=False), encoding="utf-8")
    retriever = Retriever(
        model=MockSentenceTransformer(embedding_dim=16),
        intents=defs,
        cache_dir=tmp_path / "cache",
        use_cache=False,
    )
    pipeline = Pipeline(
        intents_path=bank_path,
        mode=MODE_RETRIEVAL_ONLY,
        engine_version=ENGINE_VERSION_GENERIC,
        target=TARGET_AUTO,
        retriever=retriever,
        theta=0.0,
        auto_accept_retrieval=True,
    )
    # Trong file .py nhưng hỏi kiến thức phổ thông chỉ có template 'text' -> vẫn ra 'ok' với outputs['text']
    res = pipeline.predict("cài sunshine trên linux", language_id="python")
    payload = res.to_dict()
    assert res.status == "ok"
    assert res.intent == "USER_SETUP_SUNSHINE"
    assert "text" in payload["outputs"]
    assert "47989" in payload["outputs"]["text"]

    # Khi hỏi kiến thức có cả 'python' và 'text' trong file .py -> ưu tiên xuất target 'python'
    py_res = pipeline.predict("viết hàm fibonacci", language_id="python")
    assert py_res.status == "ok"
    assert py_res.intent == "USER_PYTHON_FIBONACCI"
    assert "python" in py_res.outputs


def test_learning_manager_milestone_and_dataset_generation_never_touches_test_csv(tmp_path: Path) -> None:
    test_sha_before = hashlib.sha256(TEST_CSV_PATH.read_bytes()).hexdigest()

    bank_path = tmp_path / "intents.json"
    defs = _universal_definitions()
    # Tạo đủ 10 intent user-created để kích hoạt mốc 10
    for idx in range(len(defs), MILESTONE_TRAIN_THRESHOLD + 1):
        defs.append(
            {
                **defs[0],
                "intent": f"USER_NOTE_{idx}",
                "display_name": f"Mẹo hệ thống {idx}",
                "aliases": [f"hướng dẫn mẹo hệ thống số {idx}", f"cách cấu hình bước {idx}"],
                "describe": f"Ghi chú cài đặt phần mềm số {idx}",
            }
        )
    bank_path.write_text(json.dumps(defs, ensure_ascii=False), encoding="utf-8")
    snapshot = {d["intent"]: d for d in defs}

    mgr = KnowledgeLearningManager(bank_path)
    user_intents = [d["intent"] for d in defs if d["domain"] == "user-created"]
    status = mgr.mark_published(snapshot, user_intents)
    assert status["pending_count"] == len(user_intents)
    assert status["milestone_ready"] is True

    fb = mgr.record_feedback(query="hướng dẫn sunshine linux lan", intent="USER_SETUP_SUNSHINE", target="text")
    assert fb["recorded"] is True
    assert mgr.status()["feedback_count"] == 1

    ds = mgr.build_milestone_datasets(snapshot)
    assert ds["train_rows"] > 0
    assert ds["val_rows"] > 0
    assert Path(ds["train_csv"]).is_file()
    assert Path(ds["val_csv"]).is_file()

    train_texts = {
        " ".join(row["text"].strip().lower().split())
        for row in csv.DictReader(Path(ds["train_csv"]).open("r", encoding="utf-8-sig"))
    }
    val_texts = {
        " ".join(row["text"].strip().lower().split())
        for row in csv.DictReader(Path(ds["val_csv"]).open("r", encoding="utf-8-sig"))
    }
    assert train_texts.isdisjoint(val_texts)
    assert hashlib.sha256(TEST_CSV_PATH.read_bytes()).hexdigest() == test_sha_before


def test_api_ask_auto_and_learn_alias_endpoint(tmp_path: Path, monkeypatch) -> None:
    from core.definitions import DefinitionRegistry

    defs = _universal_definitions()
    bank_path = tmp_path / "intents.json"
    bank_path.write_text(json.dumps(defs, ensure_ascii=False), encoding="utf-8")
    retriever = Retriever(
        model=MockSentenceTransformer(embedding_dim=16),
        intents=defs,
        cache_dir=tmp_path / "cache",
        use_cache=False,
    )
    registry = DefinitionRegistry(bank_path, retriever=retriever)
    registry.attach_retriever(retriever)

    base_pipeline = Pipeline(
        intents_path=bank_path,
        mode=MODE_RETRIEVAL_ONLY,
        engine_version=ENGINE_VERSION_GENERIC,
        target=TARGET_AUTO,
        retriever=retriever,
        theta=0.0,
        auto_accept_retrieval=True,
    )
    mgr = KnowledgeLearningManager(bank_path)

    monkeypatch.setattr(server, "pipeline", base_pipeline)
    monkeypatch.setattr(server, "_TARGET_PIPELINES", {})
    monkeypatch.setattr(server, "_get_definition_registry", lambda: registry)
    monkeypatch.setattr(server, "_get_learning_manager", lambda: mgr)

    res = server.ask_endpoint(
        server.AskRequest(text="cài sunshine trên linux", target=TARGET_AUTO, language_id="python")
    )
    assert res["status"] == "ok"
    assert res["intent"] == "USER_SETUP_SUNSHINE"
    assert res["resolved_target"] == "text"

    learned = server.learn_alias_endpoint(
        server.LearnAliasRequest(intent="USER_SETUP_SUNSHINE", query="chỉnh port 47989 cho sunshine")
    )
    assert learned["learned"] is True
    assert "chỉnh port 47989 cho sunshine" in registry.get("USER_SETUP_SUNSHINE")["learned_aliases"]
