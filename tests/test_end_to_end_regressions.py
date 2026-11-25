"""End-to-end regression tests cho pipeline NLU -> slot -> target -> render."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import api.server as server
import core.config as config
from core.config import (
    DEFAULT_INTENTS_PATH,
    ENGINE_VERSION_GENERIC,
    ENGINE_VERSION_LEGACY,
    MODE_PHOBERT_ONLY,
    MODE_RETRIEVAL_ONLY,
    TEMPLATE_TARGET_FALLBACKS,
    classifier_artifact_status,
    resolve_runtime_mode,
    retrieval_model_status,
)
from core.definitions import DefinitionLoader
from core.pipeline import Pipeline, PipelineStatus
from core.preprocess import preprocess_fingerprint


def _bank_by_intent() -> dict[str, dict]:
    items = json.loads(DEFAULT_INTENTS_PATH.read_text(encoding="utf-8"))
    return {item["intent"]: item for item in items}


class _FixedIntentClassifier:
    def __init__(self, intent: str, confidence: float = 0.99, category: str = "math_shared") -> None:
        self.intent = intent
        self.confidence = confidence
        self.category = category

    def predict(self, _text: str) -> SimpleNamespace:
        return SimpleNamespace(
            intent=self.intent,
            confidence=self.confidence,
            category=self.category,
            all_probabilities={self.intent: self.confidence},
        )


def _mock_retrieval_for(intent: str, score: float = 0.96) -> SimpleNamespace:
    bank = _bank_by_intent()
    item = bank[intent]
    cand = SimpleNamespace(
        intent=intent,
        score=score,
        describe=item.get("describe", ""),
        category=item.get("category", ""),
    )
    return SimpleNamespace(retrieve=lambda _q, top_k=3: SimpleNamespace(candidates=[cand]))


@pytest.mark.parametrize("engine_version", [ENGINE_VERSION_LEGACY, ENGINE_VERSION_GENERIC])
def test_e2e_matrix_2x2_both_targets_exact_two_backslashes(engine_version: str) -> None:
    query = "Tạo ma trận 2x2 gồm a, b, An, 20"

    latex_pipe = Pipeline(
        mode=MODE_PHOBERT_ONLY,
        target="latex",
        engine_version=engine_version,
        classifier=_FixedIntentClassifier("MATH_MATRIX_MA_TRAN_2X2"),
        tau=0.8,
    )
    latex_res = latex_pipe.predict(query)
    assert latex_res.status == PipelineStatus.OK.value
    assert latex_res.intent == "MATH_MATRIX_MA_TRAN_2X2"
    assert latex_res.slots == {"a": "a", "b": "b", "c": "An", "d": "20"}
    assert latex_res.requires == ["amsmath"]
    assert latex_res.static_check_passed is True
    assert latex_res.latex_code == "\\[\n\\begin{matrix}\na & b \\\\\nAn & 20\n\\end{matrix}\n\\]"
    assert "\\\\\\\\" not in latex_res.latex_code

    mj_pipe = Pipeline(
        mode=MODE_RETRIEVAL_ONLY,
        target="mathjax",
        engine_version=engine_version,
        retriever=_mock_retrieval_for("MATH_MATRIX_MA_TRAN_2X2"),
        auto_accept_retrieval=True,
    )
    mj_res = mj_pipe.predict(query)
    assert mj_res.status == PipelineStatus.OK.value
    assert mj_res.latex_code == "$$\\begin{matrix}\na & b \\\\\nAn & 20\n\\end{matrix}$$"
    assert "\\\\\\\\" not in mj_res.latex_code


@pytest.mark.parametrize("engine_version", [ENGINE_VERSION_LEGACY, ENGINE_VERSION_GENERIC])
def test_e2e_matrix_norm_requests_missing_content_instead_of_dimension_number(
    engine_version: str,
) -> None:
    pipe = Pipeline(
        mode=MODE_PHOBERT_ONLY,
        target="latex",
        engine_version=engine_version,
        classifier=_FixedIntentClassifier("MATH_MATRIX_CHUAN_MA_TRAN"),
        tau=0.8,
    )

    missing_res = pipe.predict("Tạo chuẩn ma trận 2x2")
    assert missing_res.status == PipelineStatus.NEED_SLOTS.value
    assert missing_res.intent == "MATH_MATRIX_CHUAN_MA_TRAN"
    assert missing_res.slots.get("matrix_content") != "2"
    assert any(slot.endswith("matrix_content") for slot in missing_res.missing_slots)

    for filled_query in (
        "Tạo chuẩn ma trận:\n1 2\n3 4",
        "Tạo chuẩn ma trận 2x2 gồm 1, 2, 3, 4",
        "Tạo chuẩn ma trận gồm 1 2; 3 4",
        "Tính chuẩn của ma trận [[1, 2], [3, 4]]",
        "Chuẩn ma trận hàng 1 là 1 2 hàng 2 là 3 4",
    ):
        ok_res = pipe.predict(filled_query)
        assert ok_res.status == PipelineStatus.OK.value, filled_query
        assert ok_res.requires == ["amsmath"]
        assert "\\begin{Vmatrix}\n1 & 2 \\\\\n3 & 4\n\\end{Vmatrix}" in ok_res.latex_code
        assert "\\begin{matrix}" not in ok_res.latex_code


@pytest.mark.parametrize("engine_version", [ENGINE_VERSION_LEGACY, ENGINE_VERSION_GENERIC])
@pytest.mark.parametrize(
    "query",
    [
        "Tạo phân số có tử số là 2 và mẫu số là 5",
        "Tạo phân số tử là 2 mẫu là 5",
    ],
)
def test_e2e_fraction_latex_and_mathjax_targets_distinct(engine_version: str, query: str) -> None:
    latex_res = Pipeline(
        mode=MODE_PHOBERT_ONLY,
        target="latex",
        engine_version=engine_version,
        classifier=_FixedIntentClassifier("MATH_MATH_BASIC_PHAN_SO"),
        tau=0.8,
    ).predict(query)
    assert latex_res.status == PipelineStatus.OK.value
    assert latex_res.slots == {"num": "2", "den": "5"}
    assert latex_res.latex_code.startswith("\\[") and latex_res.latex_code.endswith("\\]")
    assert "$$" not in latex_res.latex_code

    mj_res = Pipeline(
        mode=MODE_PHOBERT_ONLY,
        target="mathjax",
        engine_version=engine_version,
        classifier=_FixedIntentClassifier("MATH_MATH_BASIC_PHAN_SO"),
        tau=0.8,
    ).predict(query)
    assert mj_res.status == PipelineStatus.OK.value
    assert mj_res.latex_code.startswith("$$") and mj_res.latex_code.endswith("$$")


def test_no_silent_latex_to_mathjax_fallback_and_no_intent_swap_on_unsupported_target() -> None:
    assert TEMPLATE_TARGET_FALLBACKS == {}
    res = Pipeline(
        mode=MODE_PHOBERT_ONLY,
        target="latex",
        engine_version=ENGINE_VERSION_LEGACY,
        classifier=_FixedIntentClassifier("OBSIDIAN_CALLOUTS_GHI_CHU", category="obsidian"),
        tau=0.8,
    ).predict("tạo ghi chú quan trọng")
    assert res.status == PipelineStatus.UNSUPPORTED_TARGET.value
    assert res.intent == "OBSIDIAN_CALLOUTS_GHI_CHU"


def test_legacy_obsidian_substitutes_slots_and_mermaid_uses_real_newline() -> None:
    pipe = Pipeline(
        mode=MODE_PHOBERT_ONLY,
        target="obsidian",
        engine_version=ENGINE_VERSION_LEGACY,
        classifier=_FixedIntentClassifier("OBSIDIAN_MERMAID_SO_O_LUONG", category="obsidian"),
        tau=0.8,
    )
    res = pipe.choose("OBSIDIAN_MERMAID_SO_O_LUONG", query="sơ đồ", slots={"diagram": "A --> B"})
    assert res.status == PipelineStatus.OK.value
    assert res.latex_code == "```mermaid\ngraph TD\nA --> B\n```"


def test_all_multiline_templates_have_clean_escaping_requires_and_v2_parity() -> None:
    bank = _bank_by_intent()
    loaded = {item.intent: item.definition for item in DefinitionLoader().load(DEFAULT_INTENTS_PATH)}
    assert set(loaded) == set(bank)
    for name, item in bank.items():
        if item.get("category") == "math_shared":
            assert item["templates"]["latex"].strip(), name
            assert item["templates"]["mathjax"].strip(), name
            assert item["templates"]["latex"] != item["templates"]["mathjax"] or "\\begin{equation}" in item["templates"]["latex"] or "\\tag" in item["templates"]["latex"], name
        for tname, tpl in item["templates"].items():
            assert "\\\\" * 2 not in tpl, (name, tname)
            assert not re.search(r"\\n(?![A-Za-z])", tpl), (name, tname)
        combined = "\n".join(item["templates"].values())
        if any(tok in combined for tok in ("\\begin{matrix}", "\\begin{Vmatrix}", "\\begin{cases}", "\\begin{aligned}")):
            assert "amsmath" in item["requires"], name
        if "\\mathbb" in combined or "\\vDash" in combined:
            assert "amssymb" in item["requires"], name
        if "\\mathscr" in combined:
            assert "mathrsfs" in item["requires"], name
        if "\\color{" in combined:
            assert "xcolor" in item["requires"], name


def test_artifact_checks_reject_empty_weight_and_tokenizer_files(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    tok = tmp_path / "tok"
    ckpt.mkdir()
    tok.mkdir()
    (ckpt / "config.json").write_text(json.dumps({"model_type": "roberta"}), encoding="utf-8")
    (ckpt / "pytorch_model.bin").write_bytes(b"")  # 0-byte weight file
    (ckpt / "id2label.json").write_text(json.dumps({"0": "A"}), encoding="utf-8")
    (ckpt / "preprocessing_fingerprint.json").write_text(
        json.dumps({"fingerprint_sha256": preprocess_fingerprint()}), encoding="utf-8"
    )
    (tok / "vocab.txt").write_text("<s>\n", encoding="utf-8")
    (tok / "bpe.codes").write_text("#v\n", encoding="utf-8")
    ok, _ = classifier_artifact_status(ckpt, tokenizer_dir=tok)
    assert ok is False

    sbert = tmp_path / "sbert"
    sbert.mkdir()
    (sbert / "modules.json").write_text(
        json.dumps([{"idx": 0, "name": "0", "path": "", "type": "transformer"}]), encoding="utf-8"
    )
    (sbert / "config.json").write_text(json.dumps({"model_type": "roberta"}), encoding="utf-8")
    (sbert / "pytorch_model.bin").write_bytes(b"")  # 0-byte weight file
    (sbert / "vocab.txt").write_text("<s>\n", encoding="utf-8")
    (sbert / "bpe.codes").write_text("#v\n", encoding="utf-8")
    ret_ok, _, _ = retrieval_model_status(sbert)
    assert ret_ok is False


def test_system_status_and_server_target_routing(tmp_path: Path, monkeypatch) -> None:
    assert resolve_runtime_mode(False, True) == MODE_RETRIEVAL_ONLY
    assert resolve_runtime_mode(True, True) == "phobert_retrieval"

    # Server /system/status returns "error" when dependencies are missing
    monkeypatch.setattr(server.importlib.util, "find_spec", lambda name: None if name == "torch" else object())
    monkeypatch.setattr(config, "HF_CACHE_DIR", tmp_path / "empty-hf")
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    status = server.system_status()
    assert status["status"] == "error"

    # Server /ask preserves RUNTIME_INTENTS_PATH across targets without dropping outputs
    custom_intents = tmp_path / "runtime_intents.json"
    items = json.loads(DEFAULT_INTENTS_PATH.read_text(encoding="utf-8"))
    custom_intents.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    primary = Pipeline(
        mode=MODE_PHOBERT_ONLY,
        target="latex",
        intents_path=custom_intents,
        classifier=_FixedIntentClassifier("MATH_MATH_BASIC_PHAN_SO"),
        tau=0.8,
    )
    monkeypatch.setattr(server, "RUNTIME_INTENTS_PATH", custom_intents)
    monkeypatch.setattr(server, "pipeline", primary)
    monkeypatch.setattr(server, "_TARGET_PIPELINES", {})

    res_mj = server.ask_endpoint(server.AskRequest(text="Tạo phân số có tử số là 2 và mẫu số là 5", target="mathjax"))
    assert res_mj["status"] == "ok"
    assert res_mj["latex_code"] == "$$\\frac{2}{5}$$"

    # Truy vấn tự nhiên rút gọn ("phân số tử số là 2 mẫu là 3") phải cho ra đúng LaTeX \frac{2}{3}
    res_ltx = server.ask_endpoint(server.AskRequest(text="phân số tử số là 2 mẫu là 3", target="latex"))
    assert res_ltx["status"] == "ok"
    assert res_ltx["slots"] == {"num": "2", "den": "3"}
    assert res_ltx["latex_code"] == "\\[\n\\frac{2}{3}\n\\]"
    assert "$$" not in res_ltx["latex_code"]
