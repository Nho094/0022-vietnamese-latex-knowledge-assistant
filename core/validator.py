"""
Module kiểm thực JSON Schema (Validator) cho Ngân hàng Kịch bản (Intent Bank / Scenario Bank).

Tuân thủ kiến trúc và quy tắc dự án trong AGENTS.md:
- Ràng buộc cấu trúc schema chặt chẽ cho data/intents.json:
  1. Tên intent duy nhất, viết hoa chuẩn UPPER_SNAKE_CASE (bắt đầu bằng chữ hoa, gồm chữ hoa, số và dấu gạch dưới).
  2. Danh mục (category) thuộc danh sách cho phép (whitelist):
     tex_doc, tex_struct, tex_text, tex_lists, tex_tables, tex_figures,
     tex_math, tex_refs, tex_layout, tex_misc, và OUT_OF_SCOPE.
  3. Mô tả (describe) là chuỗi tiếng Việt có nghĩa, không rỗng.
  4. Từ 5-8 câu diễn đạt tương đương (aliases) ngắn gọn bằng tiếng Việt (ngoại trừ OUT_OF_SCOPE có thể để rỗng).
  5. Đặc tả slot: regex có đúng một nhóm bắt giữ (single-group regex, re.compile().groups == 1)
     và kiểu dữ liệu hợp lệ: number, text, label, filename.
  6. Khớp placeholder template <<slot>>: mọi placeholder trong template LaTeX phải được khai báo
     trong slots và ngược lại mọi slot khai báo phải xuất hiện trong template.
  7. Các thuộc tính bắt buộc và tùy chọn: kind (document, snippet, preamble),
     requires (danh sách tên gói phụ thuộc), và post (hậu xử lý tùy chọn, ví dụ tidy_math).
- Honesty Rule #1: Trích xuất slot bằng regex là bước đơn giản hóa lấy cảm hứng từ RAF,
  không phải span-pointer decoder nguyên bản của Retrieve-and-Fill.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any, Final

from core.config import DEFAULT_INTENTS_PATH
from core.latex_tools import POST_ACTIONS

# Danh mục được phép trong hệ thống trợ lý LaTeX
CATEGORY_WHITELIST: Final[frozenset[str]] = frozenset({
    "tex_doc",
    "tex_struct",
    "tex_text",
    "tex_lists",
    "tex_tables",
    "tex_figures",
    "tex_math",
    "tex_refs",
    "tex_layout",
    "tex_misc",
    "obsidian",
    "math_shared",
    "OUT_OF_SCOPE",
})

# Kiểu dữ liệu tham số slot được hỗ trợ
ALLOWED_SLOT_TYPES: Final[frozenset[str]] = frozenset({
    "number",
    "math",
    "matrix",
    "text",
    "label",
    "filename",
})

# Các loại kind hợp lệ cho template LaTeX
ALLOWED_KINDS: Final[frozenset[str]] = frozenset({
    "document",
    "snippet",
    "preamble",
})

# Biểu thức chính quy cho tên intent chuẩn UPPER_SNAKE_CASE
INTENT_NAME_REGEX: Final[re.Pattern[str]] = re.compile(r"^[A-Z][A-Z0-9_]*$")

# Biểu thức chính quy trích xuất placeholder dạng <<slot_name>> từ template
PLACEHOLDER_REGEX: Final[re.Pattern[str]] = re.compile(r"<<([a-zA-Z0-9_]+)>>")

# Bi biểu thức chính quy cho tên slot (chữ thường snake_case)
SLOT_NAME_REGEX: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*$")


def _alias_key(value: str) -> str:
    """Chuẩn hóa alias để dò trùng không phân biệt dấu, hoa thường và khoảng trắng."""
    text = value.casefold().replace("đ", "d")
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", without_marks))


def slugify_intent_name(value: str) -> str:
    """Tạo ID UPPER_SNAKE_CASE tương lai, giữ ánh xạ tiếng Việt đ -> d."""
    normalized = value.casefold().replace("đ", "d")
    ascii_text = unicodedata.normalize("NFKD", normalized).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Z0-9]+", "_", ascii_text.upper()).strip("_")

# Giới hạn số lượng aliases tiếng Việt mẫu
MIN_ALIASES: Final[int] = 5
MAX_ALIASES: Final[int] = 8

# Giới hạn độ dài chuỗi alias ngắn gọn
MAX_ALIAS_LENGTH: Final[int] = 120


def _is_unanchored_free_text_regex(pattern: str) -> bool:
    """Phát hiện nhóm bắt bắt đầu ngay từ đầu mà không có từ khóa/hậu tố cố định."""
    candidate = pattern.strip()
    if candidate in {"(.+)", "(.*)", "(.+?)", "^(.+)$", "^(.*)$", "^(.+?)$"}:
        return False
    if candidate.startswith(("(?:", "(?=", "(?!", "(?<=", "(?<!", "^")):
        return False
    if not candidate.startswith("("):
        return False
    # Dấu chấm escape biểu thị hậu tố kiểu phần mở rộng; các lookaheads/nhóm
    # không bắt ở sau phần bắt cũng là neo cố định, không phải catch-all.
    if r"\." in candidate or "(?=" in candidate or "(?!" in candidate:
        return False
    escaped = False
    depth = 0
    for index, char in enumerate(candidate):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                suffix = candidate[index + 1 :]
                suffix = re.sub(r"\\[A-Za-z]+|\\.", "", suffix)
                return not bool(re.search(r"[A-Za-z0-9]", suffix))
    return True


class IntentValidationError(ValueError):
    """Ngoại lệ ném ra khi kịch bản hoặc ngân hàng intent không thỏa mãn schema ràng buộc."""

    def __init__(self, message: str, errors: dict[str, list[str]] | None = None) -> None:
        super().__init__(message)
        self.errors: dict[str, list[str]] = errors or {}


def validate_intent(
    intent_data: dict[str, Any],
    seen_intents: set[str] | None = None,
    category_whitelist: Collection[str] | None = None,
) -> list[str]:
    """
    Kiểm thực một intent đơn lẻ theo toàn bộ các quy tắc cấu trúc schema.

    Quy tắc kiểm tra:
    1. Tên intent: Duy nhất (nếu seen_intents được cung cấp), không rỗng, chuẩn UPPER_SNAKE_CASE.
    2. Category: Phải nằm trong category_whitelist (mặc định: CATEGORY_WHITELIST).
    3. Describe: Chuỗi tiếng Việt không rỗng mô tả mục đích của intent.
    4. Aliases: Danh sách 5-8 câu ngắn gọn tiếng Việt (cho phép 0 alias nếu intent là OUT_OF_SCOPE).
    5. Slots: Kiểu dữ liệu thuộc {number, text, label, filename}; regex chứa đúng 1 capturing group.
    6. Templates: Template latex chứa placeholder <<slot>> khớp chính xác 1-1 với các slot đã khai báo.
    7. Kind, requires, post: kind thuộc {document, snippet, preamble}, requires là danh sách string, post hợp lệ.

    Tham số:
        intent_data: Từ điển chứa dữ liệu kịch bản intent.
        seen_intents: Tập hợp các intent đã xuất hiện để kiểm tra tính duy nhất (sẽ được cập nhật tại chỗ).
        category_whitelist: Tập hợp danh mục cho phép (mặc định dùng CATEGORY_WHITELIST).

    Trả về:
        Danh sách các thông báo lỗi (danh sách rỗng [] nếu intent hoàn toàn hợp lệ).
    """
    errors: list[str] = []

    if not isinstance(intent_data, dict):
        return [f"Dữ liệu intent phải là dict, nhận được: {type(intent_data).__name__}"]

    # 1. Kiểm tra tên intent (unique UPPER_SNAKE_CASE)
    intent_name = intent_data.get("intent")
    if not intent_name or not isinstance(intent_name, str):
        errors.append("Trường 'intent' bắt buộc phải là chuỗi ký tự không rỗng.")
    else:
        if not INTENT_NAME_REGEX.match(intent_name):
            errors.append(
                f"Tên intent '{intent_name}' không đúng định dạng UPPER_SNAKE_CASE "
                "(phải bắt đầu bằng chữ hoa, chỉ gồm chữ cái viết hoa, chữ số và dấu gạch dưới)."
            )
        if seen_intents is not None:
            if intent_name in seen_intents:
                errors.append(f"Tên intent '{intent_name}' bị trùng lặp trong bộ dữ liệu.")
            else:
                seen_intents.add(intent_name)

    # 2. Kiểm tra danh mục (category whitelist)
    allowed_categories = set(category_whitelist) if category_whitelist is not None else CATEGORY_WHITELIST
    category = intent_data.get("category")
    if not category or not isinstance(category, str):
        errors.append("Trường 'category' bắt buộc phải là chuỗi ký tự không rỗng.")
    elif category not in allowed_categories:
        errors.append(
            f"Danh mục '{category}' không thuộc whitelist danh mục hợp lệ: "
            f"{sorted(allowed_categories)}."
        )

    # 3. Kiểm tra mô tả (describe format)
    describe = intent_data.get("describe")
    if not isinstance(describe, str) or not describe.strip():
        errors.append("Trường 'describe' phải là chuỗi tiếng Việt mô tả kịch bản, không được để rỗng.")

    # 4. Kiểm tra danh sách câu diễn đạt tương đương (5-8 short Vietnamese aliases)
    aliases = intent_data.get("aliases")
    if not isinstance(aliases, list):
        errors.append("Trường 'aliases' bắt buộc phải là một danh sách (list).")
    else:
        # Riêng intent OUT_OF_SCOPE cho phép rỗng hoặc 5-8 câu
        is_out_of_scope = (intent_name == "OUT_OF_SCOPE" or category == "OUT_OF_SCOPE")
        if is_out_of_scope:
            if len(aliases) > 0 and not (MIN_ALIASES <= len(aliases) <= MAX_ALIASES):
                errors.append(
                    f"Intent OUT_OF_SCOPE nếu có aliases thì phải từ {MIN_ALIASES} đến {MAX_ALIASES} câu, "
                    f"hiện có: {len(aliases)}."
                )
        else:
            if not (MIN_ALIASES <= len(aliases) <= MAX_ALIASES):
                errors.append(
                    f"Số lượng aliases phải từ {MIN_ALIASES} đến {MAX_ALIASES} câu ngắn gọn tiếng Việt, "
                    f"hiện có {len(aliases)} câu."
                )

        for idx, alias in enumerate(aliases):
            if not isinstance(alias, str) or not alias.strip():
                errors.append(f"Alias thứ {idx + 1} phải là chuỗi ký tự không rỗng.")
            elif len(alias.strip()) > MAX_ALIAS_LENGTH:
                errors.append(
                    f"Alias '{alias[:30]}...' quá dài ({len(alias)} ký tự > {MAX_ALIAS_LENGTH}), "
                    "cần là câu lệnh ngắn gọn."
                )
            elif re.search(r"\d", alias):
                errors.append(f"Alias '{alias}' không được chứa chữ số; dùng cách viết bằng chữ.")

        if isinstance(intent_name, str) and intent_name != "OUT_OF_SCOPE":
            slug_words = _alias_key(intent_name.replace("_", " "))
            copied = [alias for alias in aliases
                      if isinstance(alias, str) and slug_words and slug_words in _alias_key(alias)]
            if len(copied) > 1:
                errors.append(
                    "Tối đa một alias được là bản sao không dấu của tên intent; "
                    f"phát hiện {len(copied)}: {copied}."
                )

    # 5. Kiểm tra đặc tả slots (single-group regex and allowed types)
    slots = intent_data.get("slots")
    declared_slots: set[str] = set()

    if not isinstance(slots, dict):
        errors.append("Trường 'slots' bắt buộc phải là từ điển (dict).")
    else:
        for slot_name, slot_spec in slots.items():
            if not isinstance(slot_name, str) or not SLOT_NAME_REGEX.match(slot_name):
                errors.append(
                    f"Tên slot '{slot_name}' không hợp lệ (phải bắt đầu bằng chữ thường, gồm a-z, 0-9, _)."
                )

            declared_slots.add(slot_name)

            if not isinstance(slot_spec, dict):
                errors.append(f"Cấu hình của slot '{slot_name}' phải là dict.")
                continue

            # Kiểm tra kiểu dữ liệu của slot
            slot_type = slot_spec.get("type")
            if slot_type not in ALLOWED_SLOT_TYPES:
                errors.append(
                    f"Slot '{slot_name}' có kiểu '{slot_type}' không hợp lệ. "
                    f"Các kiểu được phép: {sorted(ALLOWED_SLOT_TYPES)}."
                )

            # Kiểm tra biểu thức chính quy đơn nhóm (single-group regex)
            # Hỗ trợ cả khóa 'regex' hoặc 'pattern' để tương thích ngược
            pattern_str = slot_spec.get("regex") or slot_spec.get("pattern")
            if not pattern_str or not isinstance(pattern_str, str):
                errors.append(f"Slot '{slot_name}' thiếu biểu thức regex hợp lệ.")
            else:
                try:
                    compiled_regex = re.compile(pattern_str)
                    if compiled_regex.groups != 1:
                        errors.append(
                            f"Biểu thức regex của slot '{slot_name}' ('{pattern_str}') "
                            f"phải chứa chính xác đúng 1 capturing group, hiện có {compiled_regex.groups} nhóm."
                        )
                    if re.search(r"\\\\[sdw.]", pattern_str):
                        errors.append(
                            f"Regex của slot '{slot_name}' chứa escape kép; dùng một dấu gạch chéo "
                            "ngược cho các token như \\s, \\d, \\w hoặc \\. ."
                        )
                    if slot_type in {"text", "label", "filename"} and _is_unanchored_free_text_regex(
                        pattern_str
                    ):
                        errors.append(
                            f"Regex của slot '{slot_name}' bắt văn bản tự do không có từ khóa "
                            "hoặc hậu tố neo cố định."
                        )
                except re.error as ex:
                    errors.append(
                        f"Biểu thức regex của slot '{slot_name}' không hợp lệ ({pattern_str}): {ex}."
                    )

            # Kiểm tra trường required và default
            if "required" in slot_spec and not isinstance(slot_spec["required"], bool):
                errors.append(f"Trường 'required' của slot '{slot_name}' phải là boolean (True/False).")
            if "absent_means" in slot_spec:
                absent_means = slot_spec["absent_means"]
                if slot_type != "number" or isinstance(absent_means, bool) or not isinstance(
                    absent_means, (int, float)
                ):
                    errors.append(
                        f"Trường 'absent_means' của slot '{slot_name}' chỉ nhận giá trị số "
                        "trên slot kiểu number."
                    )
            if "not_equal" in slot_spec:
                not_equal = slot_spec["not_equal"]
                if slot_type != "number" or isinstance(not_equal, bool) or not isinstance(
                    not_equal, (int, float)
                ):
                    errors.append(
                        f"Trường 'not_equal' của slot '{slot_name}' chỉ nhận giá trị số "
                        "trên slot kiểu number."
                    )
                if "constraint_error" in slot_spec and not isinstance(
                    slot_spec["constraint_error"], str
                ):
                    errors.append(
                        f"Trường 'constraint_error' của slot '{slot_name}' phải là chuỗi."
                    )

    # 6. Kiểm tra template placeholders (<<slot>>), escaping, và cân bằng môi trường
    templates = intent_data.get("templates")
    if not isinstance(templates, dict):
        errors.append("Trường 'templates' bắt buộc phải là từ điển (dict).")
    elif "latex" not in templates:
        errors.append("Trường 'templates' bắt buộc phải chứa khóa 'latex'.")
    else:
        for target_name, target_template in templates.items():
            if not isinstance(target_template, str):
                errors.append(f"Khóa '{target_name}' trong 'templates' phải là chuỗi ký tự.")
                continue
            if re.search(r"\\n(?![A-Za-z])", target_template):
                errors.append(
                    f"Template target '{target_name}' chứa literal \\n; dùng ký tự xuống dòng thật."
                )
            if "\\\\" * 2 in target_template:
                errors.append(
                    f"Template target '{target_name}' chứa chuỗi 4 backslash liên tiếp; dùng đúng \\\\ để xuống dòng."
                )
            if not target_template.strip():
                continue
            if target_name in {"latex", "mathjax"}:
                begins = Counter(re.findall(r"\\begin\{([^{}]+)\}", target_template))
                ends = Counter(re.findall(r"\\end\{([^{}]+)\}", target_template))
                if begins != ends:
                    errors.append(
                        f"Template target '{target_name}' mất cân bằng \\begin/\\end: "
                        f"begin={dict(begins)}, end={dict(ends)}."
                    )
            placeholders_in_template = set(PLACEHOLDER_REGEX.findall(target_template))
            if target_name != "latex" and not placeholders_in_template:
                placeholders_in_template.update(
                    slot_name for slot_name in declared_slots
                    if f"{{{slot_name}}}" in target_template
                )
            undeclared = placeholders_in_template - declared_slots
            if undeclared:
                errors.append(
                    f"Placeholder trong template '{target_name}' chưa được khai báo trong 'slots': "
                    f"{sorted(undeclared)}."
                )
            if intent_name != "OUT_OF_SCOPE":
                unused = declared_slots - placeholders_in_template
                if unused:
                    errors.append(
                        f"Slot được khai báo trong 'slots' nhưng không xuất hiện trong template "
                        f"'{target_name}': {sorted(unused)}."
                    )

        domain = intent_data.get("domain") or intent_data.get("category", "latex")
        primary_key = "obsidian" if domain == "obsidian" else "latex"
        primary_template = templates.get(primary_key, "")
        if intent_name != "OUT_OF_SCOPE" and isinstance(primary_template, str):
            primary_placeholders = set(PLACEHOLDER_REGEX.findall(primary_template))
            if primary_key != "latex" and not primary_placeholders:
                primary_placeholders.update(
                    slot_name for slot_name in declared_slots
                    if f"{{{slot_name}}}" in primary_template
                )
            if not primary_template.strip() and templates.get("mathjax"):
                # Giữ kiểm tra tối thiểu nếu thư mục/fixture cũ chỉ có mathjax
                fallback_tpl = templates["mathjax"]
                fb_placeholders = set(PLACEHOLDER_REGEX.findall(fallback_tpl)) or {
                    slot_name for slot_name in declared_slots if f"{{{slot_name}}}" in fallback_tpl
                }
                if intent_name != "OUT_OF_SCOPE":
                    unused = declared_slots - fb_placeholders
                    if unused:
                        errors.append(
                            f"Slot được khai báo trong 'slots' nhưng không xuất hiện dưới dạng <<slot>> "
                            f"trong LaTeX template: {sorted(unused)}."
                        )
            elif primary_template.strip():
                unused_primary = declared_slots - primary_placeholders
                if unused_primary:
                    errors.append(
                        f"Slot được khai báo trong 'slots' nhưng không xuất hiện dưới dạng <<slot>> "
                        f"trong LaTeX template: {sorted(unused_primary)}."
                    )

    render_spec = intent_data.get("render")
    if isinstance(render_spec, dict) and isinstance(render_spec.get("templates"), dict):
        for target_name, target_template in render_spec["templates"].items():
            if not isinstance(target_template, str) or not target_template.strip():
                continue
            if re.search(r"\\n(?![A-Za-z])", target_template):
                errors.append(
                    f"render.templates.{target_name} chứa literal \\n; dùng ký tự xuống dòng thật."
                )
            if "\\\\" * 2 in target_template:
                errors.append(
                    f"render.templates.{target_name} chứa chuỗi 4 backslash liên tiếp."
                )
            if re.search(r"\\[A-Za-z]+\{\{\s*[a-zA-Z0-9_]+\s*\}\}", target_template):
                errors.append(
                    f"render.templates.{target_name} thiếu ngoặc nhọn LaTeX bọc ngoài biến Jinja."
                )
            if target_name in {"latex", "mathjax"}:
                begins = Counter(re.findall(r"\\begin\{([^{}]+)\}", target_template))
                ends = Counter(re.findall(r"\\end\{([^{}]+)\}", target_template))
                if begins != ends:
                    errors.append(
                        f"render.templates.{target_name} mất cân bằng \\begin/\\end: "
                        f"begin={dict(begins)}, end={dict(ends)}."
                    )
            render_vars = set(re.findall(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}", target_template))
            undeclared_render = render_vars - declared_slots
            if undeclared_render:
                errors.append(
                    f"Biến trong render.templates.{target_name} chưa được khai báo trong 'slots': "
                    f"{sorted(undeclared_render)}."
                )

    # 7. Kiểm tra kind, requires, và post attributes
    kind = intent_data.get("kind")
    if kind not in ALLOWED_KINDS:
        errors.append(
            f"Thuộc tính 'kind' có giá trị '{kind}' không hợp lệ. "
            f"Các giá trị được phép: {sorted(ALLOWED_KINDS)}."
        )

    requires = intent_data.get("requires")
    if not isinstance(requires, list) or not all(isinstance(r, str) for r in requires):
        errors.append(
            "Thuộc tính 'requires' phải là danh sách chứa các chuỗi tên gói LaTeX phụ thuộc (list[str])."
        )

    if "post" in intent_data:
        post = intent_data.get("post")
        if post is not None and (not isinstance(post, str) or not post.strip()):
            errors.append(
                f"Thuộc tính 'post' nếu có phải là chuỗi không rỗng hoặc None, nhận được: {post!r}."
            )
        elif post is not None and post not in POST_ACTIONS:
            errors.append(f"Post action '{post}' chưa được đăng ký trong POST_ACTIONS.")

    return errors


def validate_intents(
    intents: Sequence[dict[str, Any]],
    category_whitelist: Collection[str] | None = None,
    raise_on_error: bool = False,
) -> dict[str, list[str]]:
    """
    Kiểm thực danh sách nhiều intents trong ngân hàng kịch bản.

    Tham số:
        intents: Danh sách các intent từ điển.
        category_whitelist: Whitelist danh mục cho phép (mặc định: CATEGORY_WHITELIST).
        raise_on_error: Nếu True, ném ra ngoại lệ IntentValidationError khi có lỗi.

    Trả về:
        Từ điển ánh xạ từ tên intent (hoặc vị trí index) sang danh sách các lỗi.
        Chỉ các intent có lỗi mới xuất hiện trong từ điển (từ điển rỗng {} nếu tất cả hợp lệ).
    """
    all_errors: dict[str, list[str]] = {}
    seen_intents: set[str] = set()

    for idx, item in enumerate(intents):
        name = item.get("intent", f"<item_{idx}>") if isinstance(item, dict) else f"<item_{idx}>"
        item_errors = validate_intent(
            item,
            seen_intents=seen_intents,
            category_whitelist=category_whitelist,
        )
        if item_errors:
            all_errors[name] = item_errors

    # Bí danh dùng chung khiến retriever không thể phân biệt intent.
    alias_owners: dict[str, set[str]] = {}
    for item in intents:
        if not isinstance(item, dict):
            continue
        owner = str(item.get("intent", "<UNKNOWN>"))
        for alias in item.get("aliases", []) if isinstance(item.get("aliases"), list) else []:
            if isinstance(alias, str) and alias.strip():
                alias_owners.setdefault(_alias_key(alias), set()).add(owner)
    for alias_key, owners in alias_owners.items():
        if len(owners) > 1:
            message = f"Alias chuẩn hóa '{alias_key}' được dùng bởi nhiều intent: {sorted(owners)}."
            for owner in owners:
                all_errors.setdefault(owner, []).append(message)

    if raise_on_error and all_errors:
        total_err_count = sum(len(errs) for errs in all_errors.values())
        summary = "; ".join(f"{k}: {', '.join(v[:2])}" for k, v in list(all_errors.items())[:3])
        raise IntentValidationError(
            f"Kiểm thực schema thất bại với {total_err_count} lỗi trên {len(all_errors)} intent: {summary}",
            errors=all_errors,
        )

    return all_errors


def validate_intents_file(
    path: str | Path = DEFAULT_INTENTS_PATH,
    category_whitelist: Collection[str] | None = None,
    domain: str | None = None,
    raise_on_error: bool = False,
) -> dict[str, list[str]]:
    """
    Đọc tệp JSON intents và thực hiện kiểm thực schema toàn diện.

    Tham số:
        path: Đường dẫn tệp JSON cần kiểm thực.
        category_whitelist: Whitelist danh mục cho phép.
        domain: Bộ lọc domain tùy chọn (ví dụ: 'latex' để lọc riêng các intent thuộc phạm vi LaTeX).
        raise_on_error: Nếu True, ném ra ngoại lệ IntentValidationError khi có lỗi.

    Trả về:
        Từ điển ánh xạ intent -> danh sách lỗi phát hiện.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp ngân hàng kịch bản: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Dữ liệu kịch bản phải là một danh sách JSON, nhận được: {type(data).__name__}")

    if domain is not None:
        data = [item for item in data if isinstance(item, dict) and item.get("domain") == domain]

    return validate_intents(data, category_whitelist=category_whitelist, raise_on_error=raise_on_error)


def is_valid_intent(
    intent_data: dict[str, Any],
    category_whitelist: Collection[str] | None = None,
) -> bool:
    """Kiểm tra nhanh xem một intent có hoàn toàn hợp lệ theo schema hay không."""
    return len(validate_intent(intent_data, category_whitelist=category_whitelist)) == 0


class IntentValidator:
    """
    Lớp đối tượng kiểm thực schema cho ngân hàng kịch bản và intent.
    Cung cấp giao diện hướng đối tượng đóng gói các ràng buộc kiểm thực.
    """

    def __init__(self, category_whitelist: Collection[str] | None = None) -> None:
        self.category_whitelist: frozenset[str] = (
            frozenset(category_whitelist) if category_whitelist is not None else CATEGORY_WHITELIST
        )

    def validate_intent(self, intent_data: dict[str, Any], seen_intents: set[str] | None = None) -> list[str]:
        """Kiểm thực một intent đơn lẻ."""
        return validate_intent(intent_data, seen_intents=seen_intents, category_whitelist=self.category_whitelist)

    def validate_intents(
        self, intents: Sequence[dict[str, Any]], raise_on_error: bool = False
    ) -> dict[str, list[str]]:
        """Kiểm thực danh sách nhiều intents."""
        return validate_intents(intents, category_whitelist=self.category_whitelist, raise_on_error=raise_on_error)

    def validate_file(
        self,
        path: str | Path = DEFAULT_INTENTS_PATH,
        domain: str | None = None,
        raise_on_error: bool = False,
    ) -> dict[str, list[str]]:
        """Kiểm thực toàn bộ tệp intents.json."""
        return validate_intents_file(
            path,
            category_whitelist=self.category_whitelist,
            domain=domain,
            raise_on_error=raise_on_error,
        )


def main() -> int:
    """Giao diện dòng lệnh CLI để kiểm thực tệp intents.json."""
    parser = argparse.ArgumentParser(
        description="Kiểm thực JSON schema cho ngân hàng kịch bản (Intent Bank Validator)."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_INTENTS_PATH,
        help=f"Đường dẫn tới tệp intents.json (mặc định: {DEFAULT_INTENTS_PATH}).",
    )
    parser.add_argument("--domain", type=str, default=None,
                        help="Bộ lọc domain tùy chọn; mặc định kiểm toàn bộ bank.")

    args = parser.parse_args()
    domain_filter = args.domain

    print(f"[Validator] Đang kiểm thực: {args.path} (bộ lọc domain: {domain_filter or 'toàn bộ'})...")
    try:
        errors = validate_intents_file(args.path, domain=domain_filter)
    except Exception as ex:
        print(f"[LỖI] Không thể đọc hoặc kiểm thực tệp: {ex}", file=sys.stderr)
        return 1

    if not errors:
        print("[Thành công] Tất cả kịch bản đã vượt qua kiểm thực schema nghiêm ngặt!")
        return 0

    print(f"[Cảnh báo] Phát hiện {len(errors)} kịch bản có lỗi cấu trúc:", file=sys.stderr)
    for intent_name, err_list in list(errors.items())[:10]:
        print(f" - {intent_name}:", file=sys.stderr)
        for err in err_list:
            print(f"    * {err}", file=sys.stderr)
    if len(errors) > 10:
        print(f" ... và {len(errors) - 10} kịch bản khác có lỗi.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
