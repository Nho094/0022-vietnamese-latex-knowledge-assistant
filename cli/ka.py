"""CLI gọn cho pipeline, tương tác chọn intent/slot và quản lý knowledge bank."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import (
    DEFAULT_INTENTS_PATH,
    LATEX_COMPILE_TIMEOUT_SECONDS,
    MODE_PHOBERT_RETRIEVAL,
    MODE_RETRIEVAL_ONLY,
    MODE_PHOBERT_ONLY,
    classifier_artifact_status,
    retrieval_model_status,
    SUPPORTED_PIPELINE_MODES,
    SUPPORTED_TARGETS,
    TARGET_LATEX,
)
from core.latex_tools import static_check, wrap_snippet
from core.pipeline import Pipeline
from core.retrieval import MockSentenceTransformer, Retriever
from core.validator import validate_intents, validate_intents_file

KB_STATE = ROOT / "models" / "kb_state.json"


def _default_mode(
    classifier_available: bool | None = None,
    retrieval_available: bool | None = None,
) -> str:
    """Chọn đường đi chỉ từ artifact production hợp lệ, không từ smoke config."""
    if classifier_available is None:
        classifier_available = classifier_artifact_status()[0]
    if retrieval_available is None:
        retrieval_available = retrieval_model_status()[0]
    if classifier_available and retrieval_available:
        return MODE_PHOBERT_RETRIEVAL
    if classifier_available:
        return MODE_PHOBERT_ONLY
    if retrieval_available:
        return MODE_RETRIEVAL_ONLY
    return MODE_PHOBERT_ONLY


def create_pipeline(intents_path: Path = DEFAULT_INTENTS_PATH, mode: str | None = None,
                    target: str = TARGET_LATEX,
                    offline_mock: bool = False) -> Pipeline:
    intents = json.loads(intents_path.read_text(encoding="utf-8"))
    retriever = (Retriever(model=MockSentenceTransformer(), intents=intents, use_cache=False,
                           model_name="offline-mock", indexing_mode="alias_only")
                 if offline_mock else None)
    selected_mode = mode or _default_mode()
    return Pipeline(mode=selected_mode, retriever=retriever, intents_path=intents_path,
                    auto_accept_retrieval=False, target=target)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def compile_check(source: str) -> tuple[str, str]:
    """Biên dịch file tạm bằng engine đã cài, luôn tắt shell escape."""
    engine = shutil.which("pdflatex")
    if engine is None:
        return "skipped", "pdflatex không được cài đặt"
    problems = static_check(source)
    if problems:
        return "failed", "\n".join(problems)
    with tempfile.TemporaryDirectory(prefix="ka-latex-") as temporary:
        Path(temporary, "snippet.tex").write_text(source, encoding="utf-8")
        try:
            result = subprocess.run([engine, "-no-shell-escape", "-interaction=nonstopmode",
                                     "-halt-on-error", "snippet.tex"], cwd=temporary,
                                    capture_output=True, text=True,
                                    timeout=LATEX_COMPILE_TIMEOUT_SECONDS, check=False)
        except subprocess.TimeoutExpired as error:
            output = error.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            return "failed", "\n".join(str(output).splitlines()[-20:]) or "Compilation timed out"
        log = (result.stdout + "\n" + result.stderr).splitlines()[-20:]
        return ("ok" if result.returncode == 0 else "failed"), "\n".join(log)


def _jsonable(result: Any) -> dict[str, Any]:
    return result.to_dict() if hasattr(result, "to_dict") else result.__dict__


def ask(query: str, args: argparse.Namespace, pipeline: Pipeline) -> dict[str, Any]:
    result = pipeline.predict(query)
    while result.status == "need_choice" and sys.stdin.isatty() and not args.json:
        for idx, item in enumerate(result.candidates[:3], 1):
            print(f"{idx}. {item.get('intent')} ({item.get('score', 0):.3f}) — {item.get('describe', '')}")
        print("0. Nhập lại")
        choice = input("Chọn: ").strip()
        if choice == "0":
            query = input("Yêu cầu: ").strip()
            result = pipeline.predict(query)
        else:
            try:
                result = pipeline.choose(int(choice) - 1, query=query)
            except ValueError:
                # Người dùng có thể nhập luôn yêu cầu kế tiếp ở prompt chọn.
                # Xem đó là truy vấn mới để không mất câu hỏi trong phiên REPL.
                query = choice
                result = pipeline.predict(query)
    while result.status == "need_slots" and sys.stdin.isatty() and not args.json:
        additions = {}
        for name in result.missing_slots:
            additions[name] = input(f"{name}: ").strip()
        result = pipeline.choose(result.intent or "", query=query, slots=additions)
    data = _jsonable(result)
    if args.explain:
        data["explain"] = {"intent": result.intent, "p1": result.confidence,
                           "retrieval_scores": [x.get("score") for x in result.candidates],
                           "slots": result.slots}
    code = result.latex_code or ""
    if result.status == "ok":
        if args.compile and pipeline.target == TARGET_LATEX:
            intent_data = pipeline.intents_bank.get(result.intent or "", {})
            wrapped = wrap_snippet(code, requires=intent_data.get("requires", []),
                                   kind=intent_data.get("kind", "snippet"))
            status, log = compile_check(wrapped)
            data["compile"] = {"status": status, "log": "\n".join(log.splitlines()[-20:])}
            if not args.json:
                print(f"LaTeX compile: {status}")
                if log:
                    print(log)
        if args.out:
            Path(args.out).write_text(code, encoding="utf-8")
        if not args.json:
            if args.explain:
                print(f"intent={result.intent} p1={result.confidence:.4f} slots={result.slots}")
            print(code)
    elif not args.json:
        print(result.message or result.status)
    return data


def kb_add(path: Path, retriever: Retriever | None = None) -> dict[str, Any]:
    intent = input("Intent UPPER_SNAKE_CASE: ").strip()
    category = input("Category: ").strip()
    describe = input("Mô tả: ").strip()
    aliases = [x.strip() for x in input("Aliases (phân cách |, 5-8): ").split("|") if x.strip()]
    slots: dict[str, Any] = {}
    while input("Thêm slot? [y/N] ").strip().lower() == "y":
        name = input("Tên slot: ").strip()
        slots[name] = {"type": input("Kiểu (number/text/label/filename): ").strip(),
                       "required": input("Bắt buộc? [y/N] ").strip().lower() == "y",
                       "regex": input("Regex với một capturing group: ").strip()}
    kind = input("kind (snippet/document/preamble): ").strip()
    requires = [x.strip() for x in input("requires (phân cách dấu phẩy): ").split(",") if x.strip()]
    template = input("LaTeX template (placeholder <<slot>>): ")
    item = {"intent": intent, "category": category, "describe": describe, "aliases": aliases,
            "slots": slots, "kind": kind, "requires": requires,
            "domain": "latex",
            "templates": {"latex": template, "python": "", "cpp": ""}}
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_intents([*data, item])
    if errors:
        raise ValueError("; ".join(f"{key}: {'; '.join(value)}" for key, value in errors.items()))
    if retriever is not None:
        retriever.add_intent(item)
    data.append(item)
    _atomic_json(path, data)
    # Index refresh is immediate for this process; persistent KB state flags classifier retraining.
    state = json.loads(KB_STATE.read_text(encoding="utf-8")) if KB_STATE.exists() else {}
    state.setdefault("needs_classifier_retrain", []).append(intent)
    state.setdefault("needs_retrain", []).append(intent)
    _atomic_json(KB_STATE, state)
    return item


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ka")
    sub = parser.add_subparsers(dest="command", required=True)
    ask_parser = sub.add_parser("ask")
    ask_parser.add_argument("text", nargs="?")
    ask_parser.add_argument("--json", action="store_true")
    ask_parser.add_argument("--target", default=TARGET_LATEX, choices=list(SUPPORTED_TARGETS))
    ask_parser.add_argument("--compile", action="store_true")
    ask_parser.add_argument("--mode", default=None,
                            choices=list(SUPPORTED_PIPELINE_MODES))
    ask_parser.add_argument("--explain", action="store_true")
    ask_parser.add_argument("--out")
    ask_parser.add_argument("--offline-mock", action="store_true",
                            help="Dùng bộ mã hóa giả lập tất định cho smoke test offline.")
    repl_parser = sub.add_parser("repl", help="Mở phiên hỏi đáp tương tác trong terminal.")
    repl_parser.add_argument("--mode", default=None,
                             choices=list(SUPPORTED_PIPELINE_MODES))
    repl_parser.add_argument("--target", default=TARGET_LATEX, choices=list(SUPPORTED_TARGETS))
    repl_parser.add_argument("--compile", action="store_true")
    repl_parser.add_argument("--explain", action="store_true")
    repl_parser.add_argument("--out")
    repl_parser.set_defaults(json=False)
    repl_parser.add_argument("--offline-mock", action="store_true",
                             help="Dùng bộ mã hóa giả lập tất định cho smoke test offline.")
    kb = sub.add_parser("kb")
    kb_sub = kb.add_subparsers(dest="kb_command", required=True)
    kb_sub.add_parser("add")
    kb_sub.add_parser("list")
    kb_sub.add_parser("check")
    parsed = parser.parse_args(argv)
    if parsed.command == "ask":
        if parsed.mode is None and _default_mode() == MODE_RETRIEVAL_ONLY and not parsed.json:
            print("PhoBERT chưa có checkpoint thật; dùng retrieval_only.", file=sys.stderr)
        pipeline = create_pipeline(mode=parsed.mode, target=parsed.target,
                                   offline_mock=parsed.offline_mock)
        query = parsed.text
        while query is None:
            try:
                query = sys.stdin.readline().strip() if parsed.json else input("latex-assistant> ")
            except EOFError:
                return 0
            if parsed.json and not query:
                return 0
            if query.strip().lower() in {"exit", "quit", ":q"}:
                return 0
            if parsed.json:
                print(json.dumps(ask(query, parsed, pipeline), ensure_ascii=False))
            query = None
        result = ask(query, parsed, pipeline)
        if parsed.json:
            print(json.dumps(result, ensure_ascii=False))
        return 0
    if parsed.command == "repl":
        pipeline = create_pipeline(mode=parsed.mode, target=parsed.target,
                                   offline_mock=parsed.offline_mock)
        print("Knowledge Assistant LaTeX — nhập 'quit' hoặc 'exit' để thoát.")
        if parsed.mode is None and _default_mode() == MODE_RETRIEVAL_ONLY:
            print("PhoBERT chưa có checkpoint thật; dùng retrieval_only.")
        while True:
            try:
                query = input("latex-assistant> ").strip()
            except EOFError:
                break
            if query.lower() in {"exit", "quit", ":q"}:
                break
            if not query:
                continue
            ask(query, parsed, pipeline)
        return 0
    bank = json.loads(DEFAULT_INTENTS_PATH.read_text(encoding="utf-8"))
    if parsed.kb_command == "list":
        print("\n".join(x["intent"] for x in bank))
    elif parsed.kb_command == "check":
        errors = validate_intents_file(DEFAULT_INTENTS_PATH)
        print("valid" if not errors else "\n".join(
            f"{name}: {'; '.join(messages)}" for name, messages in errors.items()
        ))
        return int(bool(errors))
    else:
        pipeline = create_pipeline()
        pipeline._ensure_components_loaded()
        if pipeline.retriever is None:
            raise RuntimeError("Không nạp được retriever; cần checkpoint Sentence-BERT để thêm intent có thể tìm kiếm.")
        item = kb_add(DEFAULT_INTENTS_PATH, pipeline.retriever)
        print(item["intent"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Bọc CLI để thao tác Ctrl+C kết thúc gọn thay vì in traceback."""
    try:
        return _main(argv)
    except KeyboardInterrupt:
        print("Đã hủy.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
