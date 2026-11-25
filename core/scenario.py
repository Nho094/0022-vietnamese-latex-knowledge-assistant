"""
Module tạo biểu diễn chuỗi tự nhiên cho kịch bản (Scenario Natural String Representation).

Tuân thủ đặc tả mô hình Retrieve-and-Fill (RAF) theo tài liệu khoa học:
- Akshat Shrivastava et al., "Retrieve-and-Fill for Scenario-based Task-Oriented Semantic Parsing",
  arXiv:2202.00901, EACL.
- Mục 3.2 (Scenario Representation): Tối đa hóa hàm tương đồng sim(u, s) = E_U(u)^T * E_S(R(s)),
  trong đó R(s) là phép biến đổi chuyển kịch bản cấu trúc thành chuỗi biểu diễn tự nhiên (intrinsic modeling).
- Bảng 10 (Table 10: List of intrinsic representations used for encoding scenarios in RAF):
  + Biến thể 'name': [ intent | <name part of describe> ]
  + Biến thể 'slots': [ intent | <name> [ slot | <desc or slot name> ] ... ]
  + Biến thể 'primed': [ intent | <name> [ slot | desc | val1 / val2 ] ... ]
  + Biến thể 'handmade': Chuỗi biểu diễn primed đầy đủ dùng khi suy luận (inference).
- Tuân thủ Honesty Rule #1 (AGENTS.md): Đây là hệ thống scenario-based đơn giản hóa lấy cảm hứng từ RAF.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from core.config import (
    DEFAULT_INTENTS_PATH,
    DEFAULT_SCENARIO_VARIANT,
    MAX_SLOT_EXAMPLES,
    SCENARIO_VARIANTS,
)

# Danh sách các biến thể biểu diễn hợp lệ
SUPPORTED_VARIANTS: Final[Sequence[str]] = SCENARIO_VARIANTS


def extract_name_part(describe: str) -> str:
    """
    Trích xuất phần tên tác vụ (<name part>) từ chuỗi mô tả (describe).

    Trong ngân hàng kịch bản (intents.json), trường 'describe' thường có định dạng:
    '<tên tác vụ> | <danh sách slot>' (ví dụ: 'phân số | num, den' -> 'phân số').
    Nếu là câu văn tiếng Việt mô tả đầy đủ chứa từ 'dấu |' (như trong trường hợp định nghĩa cú pháp bảng),
    toàn bộ chuỗi được giữ nguyên để bảo đảm tính toàn vẹn ngữ nghĩa.

    Tham số:
        describe: Chuỗi mô tả gốc của intent.

    Trả về:
        Phần tên tác vụ đã loại bỏ khoảng trắng thừa.
    """
    desc = describe.strip()
    if "|" in desc:
        # Nếu chuỗi chứa ngữ cảnh ký tự '|' của LaTeX (ví dụ: 'dấu | trong khai báo cột')
        if "dấu |" in desc:
            return desc
        parts = desc.split("|", 1)
        return parts[0].strip()
    return desc


def extract_slot_examples(
    raw_examples: Any,
    max_examples: int = MAX_SLOT_EXAMPLES,
) -> list[str]:
    """
    Trích xuất và chuẩn hóa danh sách ví dụ mẫu cho slot (tối đa max_examples giá trị).

    Theo phương pháp Example Priming (Mục 3.2.2 trong RAF), mỗi slot được mở rộng tối đa
    2 ví dụ giá trị thực tế phân tách bằng dấu gạch chéo ' / ' nhằm giảm tính nhập nhằng (slot ambiguity).

    Tham số:
        raw_examples: Dữ liệu ví dụ (danh sách, chuỗi, hoặc None).
        max_examples: Số lượng ví dụ tối đa cần giữ lại (mặc định 2).

    Trả về:
        Danh sách chuỗi ví dụ hợp lệ.
    """
    if raw_examples is None:
        return []
    if isinstance(raw_examples, str):
        raw_str = raw_examples.strip()
        if not raw_str:
            return []
        if " / " in raw_str:
            return [x.strip() for x in raw_str.split(" / ") if x.strip()][:max_examples]
        if "," in raw_str:
            return [x.strip() for x in raw_str.split(",") if x.strip()][:max_examples]
        return [raw_str]
    if isinstance(raw_examples, (list, tuple, set)):
        cleaned = [str(x).strip() for x in raw_examples if str(x).strip()]
        return cleaned[:max_examples]
    return []


def format_slot_string(
    slot_name: str,
    slot_def: Any,
    include_examples: bool = False,
    max_examples: int = MAX_SLOT_EXAMPLES,
) -> str:
    """
    Định dạng một slot thành chuỗi biểu diễn tự nhiên theo quy chuẩn RAF.

    Định dạng:
    - Không ví dụ: '[ slot | <desc or slot name> ]'
    - Có ví dụ: '[ slot | <desc or slot name> | val1 / val2 ]' (nếu có ví dụ)

    Tham số:
        slot_name: Tên định danh của slot.
        slot_def: Định nghĩa của slot (dict, SlotDefinition, hoặc chuỗi).
        include_examples: Có kèm ví dụ mẫu hay không (dành cho biến thể primed/handmade).
        max_examples: Số lượng ví dụ tối đa.

    Trả về:
        Chuỗi biểu diễn slot đã đóng ngoặc vuông chuẩn.
    """
    name = str(slot_name).strip()
    desc = name
    raw_examples: Any = None

    if isinstance(slot_def, dict):
        desc_val = slot_def.get("desc") or slot_def.get("description")
        if desc_val and str(desc_val).strip():
            desc = str(desc_val).strip()
        raw_examples = slot_def.get("examples") or slot_def.get("example")
    elif hasattr(slot_def, "desc") and getattr(slot_def, "desc", None):
        desc = str(slot_def.desc).strip()
        raw_examples = getattr(slot_def, "examples", None)
    elif hasattr(slot_def, "description") and getattr(slot_def, "description", None):
        desc = str(slot_def.description).strip()
        raw_examples = getattr(slot_def, "examples", None)
    elif isinstance(slot_def, str) and slot_def.strip():
        desc = slot_def.strip()

    if include_examples:
        examples = extract_slot_examples(raw_examples, max_examples=max_examples)
        if examples:
            examples_str = " / ".join(examples)
            return f"[ slot | {desc} | {examples_str} ]"

    return f"[ slot | {desc} ]"


@functools.lru_cache(maxsize=1)
def _load_intent_lookup(path: str | Path = DEFAULT_INTENTS_PATH) -> dict[str, dict[str, Any]]:
    """Tải và lưu đệm ngân hàng kịch bản theo bảng tra cứu intent -> dict."""
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {
            str(item["intent"]): item
            for item in data
            if isinstance(item, dict) and "intent" in item
        }
    elif isinstance(data, dict):
        return data
    return {}


def scenario_string(
    intent: dict[str, Any] | Any | str,
    variant: str = DEFAULT_SCENARIO_VARIANT,
    bank_path: str | Path = DEFAULT_INTENTS_PATH,
) -> str:
    """
    Tạo biểu diễn chuỗi tự nhiên cho kịch bản (Scenario Natural String Representation)
    tuân thủ nghiêm ngặt đặc tả Retrieve-and-Fill (RAF - arXiv:2202.00901, Table 10).

    Công thức và ánh xạ 4 biến thể:
    1. 'name':
       Biểu diễn chỉ chứa tên ý định:
       '[ intent | <name part of describe> ]'
    2. 'slots':
       Biểu diễn bổ sung các slot tham số (không kèm ví dụ):
       '[ intent | <name> [ slot | <desc or slot name> ] ... ]'
    3. 'primed':
       Biểu diễn mở rộng slot với tối đa 2 ví dụ phân tách bằng ' / ':
       '[ intent | <name> [ slot | <desc> | val1 / val2 ] ... ]'
    4. 'handmade':
       Chuỗi biểu diễn primed đầy đủ dùng chính thức trong giai đoạn suy luận (inference).

    Tham số:
        intent: Đối tượng kịch bản (từ điển dict, đối tượng Scenario, hoặc chuỗi tên intent).
        variant: Biến thể mong muốn ('name', 'slots', 'primed', 'handmade'). Mặc định 'handmade'.
        bank_path: Đường dẫn tệp intents.json dự phòng khi truyền chuỗi tên intent.

    Trả về:
        Chuỗi tự nhiên chuẩn hóa định dạng RAF.

    Ngoại lệ:
        ValueError: Nếu tên biến thể (variant) không hợp lệ.
        KeyError: Nếu truyền chuỗi tên intent nhưng không tìm thấy trong ngân hàng kịch bản.
    """
    variant_clean = str(variant).strip().lower()
    if variant_clean not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"Biến thể kịch bản '{variant}' không được hỗ trợ. "
            f"Các biến thể hợp lệ: {list(SUPPORTED_VARIANTS)}"
        )

    # 1. Chuẩn hóa dữ liệu intent về dict hoặc truy xuất thuộc tính
    intent_data: dict[str, Any]
    if isinstance(intent, str):
        lookup = _load_intent_lookup(bank_path)
        if intent not in lookup:
            raise KeyError(
                f"Không tìm thấy intent '{intent}' trong ngân hàng kịch bản ({bank_path})."
            )
        intent_data = lookup[intent]
    elif isinstance(intent, dict):
        intent_data = intent
    elif hasattr(intent, "to_dict") and callable(intent.to_dict):
        intent_data = intent.to_dict()
    else:
        # Hỗ trợ duck-typing cho các đối tượng Scenario
        intent_data = {
            "intent": getattr(intent, "intent", ""),
            "describe": getattr(intent, "describe", ""),
            "slots": getattr(intent, "slots", {}),
        }

    # 2. Trích xuất phần tên mô tả (<name part of describe>)
    describe_text = str(intent_data.get("describe", "")).strip()
    name_part = extract_name_part(describe_text)
    if not name_part:
        # Nếu không có describe, sử dụng mã intent làm tên dự phòng
        name_part = str(intent_data.get("intent", "")).strip() or "unnamed_intent"

    # 3. Biến thể 'name': chỉ trả về intent và phần tên
    if variant_clean == "name":
        return f"[ intent | {name_part} ]"

    # 4. Trích xuất danh sách slot
    slots_raw = intent_data.get("slots", {})
    slots_items: list[tuple[str, Any]] = []

    if isinstance(slots_raw, dict):
        slots_items = list(slots_raw.items())
    elif isinstance(slots_raw, (list, tuple)):
        for item in slots_raw:
            if isinstance(item, str):
                slots_items.append((item, {}))
            elif isinstance(item, dict):
                slot_name = str(item.get("name") or item.get("slot") or "").strip()
                slots_items.append((slot_name, item))

    # Nếu không có slot nào, các biến thể 'slots', 'primed', 'handmade' đồng nhất với 'name'
    if not slots_items:
        return f"[ intent | {name_part} ]"

    # 5. Biến thể 'slots', 'primed', 'handmade'
    include_examples = variant_clean in ("primed", "handmade")
    formatted_slots = [
        format_slot_string(s_name, s_def, include_examples=include_examples)
        for s_name, s_def in slots_items
    ]

    slots_content = " ".join(formatted_slots)
    return f"[ intent | {name_part} {slots_content} ]"
