"""Quét các invariant de KA-08b trên intent bank để so sánh trước/sau fix."""
from __future__ import annotations

import json
import re

from core.config import DEFAULT_INTENTS_PATH
from core.latex_tools import resolve_template
from core.slots import extract_slots
from core.validator import _is_unanchored_free_text_regex


def main() -> None:
    bank = json.loads(DEFAULT_INTENTS_PATH.read_text(encoding="utf-8"))
    unsupported = [
        item["intent"] for item in bank
        if item.get("intent") != "OUT_OF_SCOPE"
        and not (item.get("templates", {}).get("latex") or item.get("templates", {}).get("mathjax"))
    ]
    doubled = []
    naked = []
    for item in bank:
        for name, slot in item.get("slots", {}).items():
            pattern = slot.get("regex") or slot.get("pattern") or ""
            if re.search(r"\\\\[sdw.]", pattern):
                doubled.append(f"{item['intent']}.{name}")
            if slot.get("type") in {"text", "label", "filename"} and _is_unanchored_free_text_regex(pattern):
                naked.append(f"{item['intent']}.{name}")
    neutral = ["tạo giúp tôi cái này", "viết cho tôi một đoạn mã", "làm thế nào để dùng nó",
               "cho tôi ví dụ ngắn", "hello world"]
    false_fills = 0
    for query in neutral:
        for item in bank:
            result = extract_slots(query, item)
            for name, spec in item.get("slots", {}).items():
                if spec.get("required") and spec.get("default") is None:
                    if result.slots.get(name, "").strip():
                        false_fills += 1
    quad = next(item for item in bank if item["intent"] == "QUADRATIC_EQUATION")
    examples = ["x^2 - 3x + 2 = 0", "x bình phương trừ 3x cộng 2 bằng 0",
                "trừ x bình phương cộng x cộng 1 bằng 0"]
    print(f"intents={len(bank)} latex_unsupported={len(unsupported)}")
    explicit_unsupported = sum(
        item.get("intent") != "OUT_OF_SCOPE" and resolve_template(item, "latex") is None
        for item in bank
    )
    print(f"unsupported_target_resolutions={explicit_unsupported}")
    print(f"double_escaped_regex={len(doubled)} {doubled}")
    print(f"unanchored_free_text_regex={len(naked)} {naked}")
    print(f"neutral_required_slot_false_fills={false_fills}")
    for text in examples:
        result = extract_slots(text, quad)
        print(f"quadratic {text!r}: {result.slots} missing={result.missing} errors={result.errors}")


if __name__ == "__main__":
    main()
