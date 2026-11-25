from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cli.ka import _default_mode
from core.pipeline import Pipeline, render_latex_template
from core.retrieval import MockSentenceTransformer, Retriever
from core.slots import extract_slots, normalize_math
from core.validator import validate_intents

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


def test_default_mode_requires_production_artifacts_for_each_component() -> None:
    assert _default_mode(True, True) == "phobert_retrieval"
    assert _default_mode(True, False) == "phobert_only"
    assert _default_mode(False, True) == "retrieval_only"
    assert _default_mode(False, False) == "phobert_only"


def test_pipeline_without_models_returns_actionable_unknown(tmp_path: Path) -> None:
    pipeline = Pipeline(
        mode="phobert_only",
        checkpoint_dir=tmp_path / "missing-checkpoint",
        intents_path=ROOT / "data" / "intents.json",
    )

    result = pipeline.predict("Tạo một phân số")

    assert result.status == "unknown"
    assert "chưa có model" in result.message.casefold()
    assert "Knowledge Manager vẫn hoạt động" in result.message


def test_ka_cli_json_accepts_three_queries() -> None:
    queries = ("tạo một phân số", "viết phương trình bậc hai", "hôm nay thời tiết thế nào")
    result = subprocess.run([str(PYTHON), "cli/ka.py", "ask", "--json", "--offline-mock"], cwd=ROOT,
                            input="\n".join((*queries, "quit")) + "\n", capture_output=True,
                            text=True, check=True, timeout=60)
    for line in result.stdout.splitlines():
        payload = json.loads(line)
        assert payload["status"] in {"ok", "need_choice", "need_slots", "out_of_scope", "unknown"}


def test_ka_cli_repl_accepts_query_and_quit() -> None:
    result = subprocess.run(
        [str(PYTHON), "cli/ka.py", "repl", "--offline-mock"], cwd=ROOT,
        input="tạo một phân số\nquit\n", capture_output=True, text=True,
        check=True, timeout=60,
    )
    assert "latex-assistant>" in result.stdout
    assert "Knowledge Assistant LaTeX" in result.stdout


def test_new_knowledge_is_retrievable_without_classifier_training(tmp_path: Path) -> None:
    retriever = Retriever(model=MockSentenceTransformer(), intents=[], cache_dir=tmp_path,
                          use_cache=False, model_name="test-mock", indexing_mode="alias_only")
    toy = {
        "intent": "TOY_NEW_INTENT",
        "category": "tex_misc",
        "describe": "Thử nghiệm intent mới | tạo lệnh đồ chơi",
        "aliases": ["chèn công thức đồ chơi", "tạo lệnh đồ chơi", "viết cú pháp đồ chơi",
                    "thêm mã mẫu đồ chơi", "xuất lệnh thử nghiệm"],
        "slots": {},
        "kind": "snippet",
        "requires": [],
        "templates": {"latex": r"\toy", "python": "", "cpp": ""},
    }
    assert validate_intents([toy]) == {}
    retriever.add_intent(toy)
    result = retriever.retrieve("chèn công thức đồ chơi", top_k=1)
    assert result.top_intent == "TOY_NEW_INTENT"


def test_catch_all_text_slot_requests_value_instead_of_copying_whole_query() -> None:
    intent = {
        "intent": "GENERIC_TEXT",
        "slots": {"value": {"type": "text", "regex": "(.+)", "required": True}},
    }
    result = extract_slots("Tạo phân số có tử số a và mẫu số b", intent)
    assert result.slots == {}
    assert result.missing == ["value"]


def test_quadratic_slots_parse_implicit_coefficient_and_spoken_negative_constant() -> None:
    result = extract_slots(
        "Tạo phương trình bậc hai x bình phương cộng 3x trừ 4 bằng 0",
        "QUADRATIC_EQUATION",
    )
    assert result.slots == {"a": "1", "b": "3", "c": "-4"}
    assert result.missing == []


def test_integral_query_extracts_limits_and_spoken_power() -> None:
    result = extract_slots(
        "Tạo tích phân từ 0 đến 1 của x bình phương theo x",
        "MATH_MATH_BASIC_TICH_PHAN",
    )
    assert result.slots == {"from": "0", "to": "1", "expr": "x^2"}
    assert result.missing == []
    latex, valid, errors = render_latex_template(
        {"templates": {"latex": "$$\\int_{{{from}}}^{{{to}}} {expr}\\,dx$$"},
         "slots": {"from": {"type": "text"}, "to": {"type": "text"},
                   "expr": {"type": "math"}}},
        result.slots,
    )
    assert valid, errors
    assert latex == "$$\\int_{0}^{1} x^2\\,dx$$"


def test_math_slot_rejects_latex_commands() -> None:
    try:
        normalize_math(r"\\input{secret}")
    except ValueError:
        pass
    else:
        raise AssertionError("math slot must reject LaTeX commands")
