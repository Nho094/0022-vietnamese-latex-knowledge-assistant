"""
Module tải dữ liệu ngoại bộ PhoATIS, lấy mẫu câu ngoài phạm vi (OOS) và khai phá công thức toán LaTeX.

Tuân thủ nghiêm ngặt các quy tắc kiến trúc và tính trung thực trong AGENTS.md:
- Honesty Rule #1: Hệ thống lấy cảm hứng từ Retrieve-and-Fill (RAF), điền slot bằng regex.
- Honesty Rule #3: Không chỉnh sửa, sinh ngẫu nhiên hay mở rộng data/test.csv.
- Honesty Rule #4: Các phân vùng train/val/test không được chia sẻ bất kỳ câu trùng lặp nào
  (so khớp không phân biệt hoa thường và khoảng trắng).
- Honesty Rule #5: Số liệu công bố phải bắt nguồn từ kết quả thực tế, trích dẫn PhoATIS (Dao et al., 2021).
- Ứng viên công thức toán trong data/mined_candidates.csv luôn mang trạng thái 'pending_manual_review'
  và chỉ phục vụ thẩm định thủ công, không tự động trộn vào tập đánh giá hay huấn luyện.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from core.config import (
    DATA_DIR,
    DEFAULT_SEED,
    MINED_CANDIDATES_PATH,
    PHOATIS_DEV_URL,
    PHOATIS_DIR,
    PHOATIS_TEST_URL,
    PHOATIS_TRAIN_URL,
    set_seed,
)

# Bản đồ liên kết URL chính thức của PhoATIS cấp âm tiết (syllable-level seq.in) từ VinAI JointIDSF
PHOATIS_SPLIT_URLS: Final[dict[str, str]] = {
    "train": PHOATIS_TRAIN_URL,
    "dev": PHOATIS_DEV_URL,
    "test": PHOATIS_TEST_URL,
}

# Tập câu đàm thoại tiếng Việt thông thường (casual everyday Vietnamese queries)
# Các tập câu được thiết kế độc lập và hoàn toàn không giao thoa giữa train, val và test (Honesty Rule #4)
CASUAL_TRAIN_POOL: Final[list[str]] = [
    "Xin chào bạn, hôm nay bạn có khỏe không?",
    "Chào buổi sáng trợ lý ảo, chúc một ngày tốt lành.",
    "Hôm nay thời tiết ở Hà Nội có mưa không?",
    "Bây giờ là mấy giờ rồi bạn ơi?",
    "Cho tôi nghe một bài hát của Sơn Tùng MTP.",
    "Gợi ý cho mình một vài quán ăn trưa ngon gần đây.",
    "Lịch thi đấu bóng đá tối nay có trận nào hay không?",
    "Đường từ đây đến sân bay Nội Bài đi hướng nào nhanh nhất?",
    "Giá vàng hôm nay tăng hay giảm bao nhiêu?",
    "Bạn tên là gì và do nhóm nào phát triển?",
    "Đặt báo thức giúp tôi lúc sáu giờ ba mươi sáng mai.",
    "Hôm nay là thứ mấy và ngày bao nhiêu âm lịch?",
    "Tắt đèn phòng khách và hạ nhiệt độ điều hòa xuống nhé.",
    "Tìm kiếm công thức nấu món phở bò truyền thống.",
    "Kết quả xổ số miền Bắc chiều nay ra con gì?",
    "Mở ứng dụng bản đồ chỉ đường về nhà.",
    "Thủ đô của nước Pháp tên là gì vậy bạn?",
    "Nhắc tôi mua sữa và hoa quả lúc năm giờ chiều.",
    "Một năm ánh sáng bằng bao nhiêu kilômét?",
    "Kể cho tôi một câu chuyện cười ngắn để giải trí.",
    "Bạn có biết nấu món canh chua cá lóc không?",
    "Thời gian bay từ Hà Nội vào Sài Gòn mất bao lâu?",
    "Dịch câu xin chào sang tiếng Tây Ban Nha giúp tôi.",
    "Ai là vị vua đầu tiên của triều đại nhà Lý?",
    "Kiểm tra xem ngày mai có kẹt xe trên đường Nguyễn Trãi không.",
]

CASUAL_VAL_POOL: Final[list[str]] = [
    "Chào bạn, ngày mới tràn đầy năng lượng nhé.",
    "Nhiệt độ hiện tại ở Thành phố Hồ Chí Minh là bao nhiêu?",
    "Tìm giúp tôi quán cà phê yên tĩnh để đọc sách.",
    "Tỷ giá đồng đô la Mỹ hôm nay so với tiền Việt ra sao?",
    "Bật bình nóng lạnh trong phòng tắm lên giúp tôi.",
    "Bao nhiêu ngày nữa là đến kỳ nghỉ Tết Nguyên Đán?",
    "Tác giả của cuốn tiểu thuyết Tắt đèn là ai?",
    "Giá vé xem phim tại rạp CGV cuối tuần này bao nhiêu?",
    "Đoạn đường cao tốc Pháp Vân Cầu Giẽ dài bao nhiêu km?",
    "Mấy giờ chuyến tàu hỏa Thống Nhất Bắc Nam khởi hành?",
    "Gợi ý thực đơn ăn kiêng lành mạnh cho cả tuần.",
    "Bật bài hát nhạc không lời để tập trung làm việc.",
]

CASUAL_TEST_POOL: Final[list[str]] = [
    "Xin chào, chúc bạn một ngày làm việc thật hiệu quả.",
    "Dự báo thời tiết cuối tuần này ở Đà Nẵng thế nào?",
    "Hướng dẫn cách làm bánh mì kẹp thịt Việt Nam ngon chuẩn vị.",
    "Giá xăng dầu tuần này được liên bộ điều chỉnh tăng hay giảm?",
    "Tắt máy hút bụi tự động sau mười lăm phút nữa.",
    "Ai là người sáng lập ra tập đoàn công nghệ Microsoft?",
    "Mặt trời cách Trái Đất khoảng cách bao nhiêu triệu km?",
    "Tìm cây xăng Petrolimex gần vị trí hiện tại nhất.",
    "Bộ phim điện ảnh nào đang đứng đầu phòng vé hiện nay?",
    "Nhắc tôi uống thuốc sau bữa ăn tối ba mươi phút.",
    "Sân vận động Mỹ Đình có sức chứa bao nhiêu khán giả?",
    "Quy đổi một trăm Euro sang tiền đồng Việt Nam.",
]


def normalize_text_for_dedup(text: str) -> str:
    """
    Chuẩn hóa câu để so khớp chống trùng lặp theo Honesty Rule #4.
    Loại bỏ khoảng trắng thừa và chuyển về chữ thường.
    """
    return re.sub(r"\s+", " ", text.strip().lower())


def check_network_connection(urls: Sequence[str] | str | None = None, timeout: float = 4.0) -> bool:
    """
    Kiểm tra tính khả dụng của kết nối mạng tới các máy chủ chứa dữ liệu ngoại bộ.

    Tham số:
        urls: Danh sách URL cần kiểm tra hoặc None (kiểm tra mặc định PhoATIS URLs).
        timeout: Thời gian chờ tối đa (giây).

    Trả về:
        True nếu kết nối thành công, False nếu mất mạng hoặc quá thời gian chờ.
    """
    if urls is None:
        check_targets = list(PHOATIS_SPLIT_URLS.values())
    elif isinstance(urls, str):
        check_targets = [urls]
    else:
        check_targets = list(urls)

    headers = {"User-Agent": "Vietnamese-LaTeX-Knowledge-Assistant/1.0"}
    for target in check_targets:
        try:
            req = urllib.request.Request(target, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
    return False


def format_manual_download_instructions(target_dir: Path | None = None) -> str:
    """
    Tạo hướng dẫn tải dữ liệu thủ công chi tiết khi hệ thống không có kết nối internet.
    Giải thích rõ ràng đường dẫn lưu trữ, lệnh curl/wget và URL chính thức.
    """
    base_dest = target_dir or PHOATIS_DIR
    lines = [
        "=" * 78,
        "[LỖI KẾT NỐI MẠNG] Không thể tải dữ liệu PhoATIS trực tiếp từ GitHub.",
        "Vui lòng thực hiện tải thủ công theo các bước sau và đặt vào thư mục dự án:",
        "=" * 78,
        f"Thư mục đích lưu trữ: {base_dest}",
        "",
        "Bước 1: Tạo các thư mục lưu trữ phân vùng:",
        f"  mkdir -p {base_dest}/train {base_dest}/dev {base_dest}/test",
        "",
        "Bước 2: Tải từng tệp seq.in cấp âm tiết (syllable-level) bằng lệnh curl hoặc wget:",
        f"  curl -fsSL {PHOATIS_TRAIN_URL} -o {base_dest}/train/seq.in",
        f"  curl -fsSL {PHOATIS_DEV_URL} -o {base_dest}/dev/seq.in",
        f"  curl -fsSL {PHOATIS_TEST_URL} -o {base_dest}/test/seq.in",
        "",
        "Hoặc mở liên kết trình duyệt và lưu tệp văn bản:",
        f"  - Train: {PHOATIS_TRAIN_URL}",
        f"  - Dev:   {PHOATIS_DEV_URL}",
        f"  - Test:  {PHOATIS_TEST_URL}",
        "=" * 78,
    ]
    return "\n".join(lines)


def download_phoatis_data(
    target_dir: Path = PHOATIS_DIR,
    force: bool = False,
    timeout: float = 10.0,
) -> dict[str, Path]:
    """
    Tải các tệp dữ liệu PhoATIS cấp âm tiết (seq.in) về thư mục chỉ định.
    Xử lý kiểm tra kết nối mạng và dừng kèm thông báo hướng dẫn thủ công nếu không thể kết nối.

    Tham số:
        target_dir: Thư mục gốc lưu trữ dữ liệu PhoATIS (mặc định data/external/phoatis).
        force: Nếu True, ghi đè lại các tệp đã có.
        timeout: Thời gian chờ tối đa cho mỗi lượt tải tệp.

    Trả về:
        Từ điển đường dẫn tới các tệp seq.in của từng phân vùng {'train', 'dev', 'test'}.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    # Kiểm tra xem toàn bộ các tệp đã tồn tại đầy đủ tại đích hay chưa
    all_exist = True
    for split in ("train", "dev", "test"):
        split_file = target_dir / split / "seq.in"
        if not split_file.is_file() or split_file.stat().st_size == 0:
            all_exist = False
            break

    if all_exist and not force:
        for split in ("train", "dev", "test"):
            results[split] = target_dir / split / "seq.in"
        return results

    # Kiểm tra kết nối mạng trước khi tải
    is_online = check_network_connection(timeout=timeout / 2)
    if not is_online and not all_exist:
        instructions = format_manual_download_instructions(target_dir)
        print(instructions, file=sys.stderr)
        raise RuntimeError(
            "Không có kết nối mạng tới kho lưu trữ PhoATIS và tệp nội bộ chưa có sẵn.\n"
            + instructions
        )

    headers = {"User-Agent": "Vietnamese-LaTeX-Knowledge-Assistant/1.0"}
    for split, url in PHOATIS_SPLIT_URLS.items():
        split_dir = target_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        dest_path = split_dir / "seq.in"

        if dest_path.is_file() and not force and dest_path.stat().st_size > 0:
            results[split] = dest_path
            continue

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content = response.read().decode("utf-8")
                dest_path.write_text(content, encoding="utf-8")
                # Tạo tệp phẳng kèm tên phân vùng để tiện truy xuất
                flat_path = target_dir / f"{split}_seq.in"
                flat_path.write_text(content, encoding="utf-8")
                results[split] = dest_path
        except Exception as exc:
            instructions = format_manual_download_instructions(target_dir)
            print(instructions, file=sys.stderr)
            raise RuntimeError(
                f"Lỗi khi tải phân vùng PhoATIS '{split}' từ {url}: {exc}\n{instructions}"
            ) from exc

    return results


def sample_out_of_scope_splits(
    phoatis_dir: Path = PHOATIS_DIR,
    output_dir: Path = PHOATIS_DIR,
    seed: int = DEFAULT_SEED,
    n_train: int = 40,
    n_val: int = 15,
    n_test: int = 15,
) -> dict[str, list[dict[str, str]]]:
    """
    Lấy mẫu các câu ngoài phạm vi (Out-of-Scope / OOS) có hạt giống xác định (seed)
    cho các phân vùng train/val/test, kết hợp với các câu hội thoại tiếng Việt thông thường.

    Tuân thủ Honesty Rule #4:
    - Đảm bảo 100% không có câu trùng lặp giữa các phân vùng train, val, và test
      (so khớp không phân biệt chữ hoa, chữ thường và khoảng trắng).
    - Tập OOS được lưu trữ riêng biệt tại data/external/phoatis/ để phục vụ đánh giá/huấn luyện,
      không trực tiếp sửa đổi data/test.csv (bảo toàn Honesty Rule #3).

    Tham số:
        phoatis_dir: Thư mục chứa các tệp PhoATIS đã tải.
        output_dir: Thư mục xuất các tệp CSV phân vùng OOS.
        seed: Hạt giống ngẫu nhiên tái lập.
        n_train: Số câu PhoATIS lấy mẫu cho train.
        n_val: Số câu PhoATIS lấy mẫu cho val.
        n_test: Số câu PhoATIS lấy mẫu cho test.

    Trả về:
        Từ điển chứa danh sách các bản ghi cho từng phân vùng.
    """
    set_seed(seed)
    rng = random.Random(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    def load_lines(split_name: str) -> list[str]:
        # Hỗ trợ cả cấu trúc split/seq.in lẫn flat split_seq.in
        path1 = phoatis_dir / split_name / "seq.in"
        path2 = phoatis_dir / f"{split_name}_seq.in"
        target = path1 if path1.is_file() else path2
        if not target.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy tệp PhoATIS seq.in cho phân vùng {split_name} tại {target}."
            )
        raw_lines = target.read_text(encoding="utf-8").splitlines()
        # Loại bỏ dòng rỗng và làm sạch khoảng trắng
        cleaned: list[str] = []
        for line in raw_lines:
            s = line.strip()
            if s:
                cleaned.append(s)
        return cleaned

    phoatis_train_lines = load_lines("train")
    phoatis_dev_lines = load_lines("dev")
    phoatis_test_lines = load_lines("test")

    # Lấy mẫu có seed từ từng phân vùng tương ứng của PhoATIS
    sampled_phoatis_train = rng.sample(phoatis_train_lines, min(n_train, len(phoatis_train_lines)))
    sampled_phoatis_val = rng.sample(phoatis_dev_lines, min(n_val, len(phoatis_dev_lines)))
    sampled_phoatis_test = rng.sample(phoatis_test_lines, min(n_test, len(phoatis_test_lines)))

    # Kết hợp với tập câu giao tiếp thông thường (casual pools)
    train_records: list[dict[str, str]] = []
    val_records: list[dict[str, str]] = []
    test_records: list[dict[str, str]] = []

    seen_normalized: set[str] = set()

    # Xây dựng train OOS
    for text in sampled_phoatis_train:
        norm = normalize_text_for_dedup(text)
        if norm not in seen_normalized:
            seen_normalized.add(norm)
            train_records.append(
                {"text": text, "label": "OUT_OF_SCOPE", "category": "tex_misc", "source": "phoatis_train"}
            )

    for text in CASUAL_TRAIN_POOL:
        norm = normalize_text_for_dedup(text)
        if norm not in seen_normalized:
            seen_normalized.add(norm)
            train_records.append(
                {"text": text, "label": "OUT_OF_SCOPE", "category": "tex_misc", "source": "casual_train"}
            )

    # Xây dựng val OOS
    val_seen: set[str] = set()
    for text in sampled_phoatis_val:
        norm = normalize_text_for_dedup(text)
        # Bắt buộc không trùng với tập train
        if norm not in seen_normalized and norm not in val_seen:
            val_seen.add(norm)
            val_records.append(
                {"text": text, "label": "OUT_OF_SCOPE", "category": "tex_misc", "source": "phoatis_dev"}
            )

    for text in CASUAL_VAL_POOL:
        norm = normalize_text_for_dedup(text)
        if norm not in seen_normalized and norm not in val_seen:
            val_seen.add(norm)
            val_records.append(
                {"text": text, "label": "OUT_OF_SCOPE", "category": "tex_misc", "source": "casual_val"}
            )

    # Cập nhật danh sách câu đã thấy từ val
    seen_normalized.update(val_seen)

    # Xây dựng test OOS
    test_seen: set[str] = set()
    for text in sampled_phoatis_test:
        norm = normalize_text_for_dedup(text)
        # Bắt buộc không trùng với cả train lẫn val
        if norm not in seen_normalized and norm not in test_seen:
            test_seen.add(norm)
            test_records.append(
                {"text": text, "label": "OUT_OF_SCOPE", "category": "tex_misc", "source": "phoatis_test"}
            )

    for text in CASUAL_TEST_POOL:
        norm = normalize_text_for_dedup(text)
        if norm not in seen_normalized and norm not in test_seen:
            test_seen.add(norm)
            test_records.append(
                {"text": text, "label": "OUT_OF_SCOPE", "category": "tex_misc", "source": "casual_test"}
            )

    # Kiểm tra tính toàn vẹn và không trùng lặp (Honesty Rule #4)
    train_norms = {normalize_text_for_dedup(r["text"]) for r in train_records}
    val_norms = {normalize_text_for_dedup(r["text"]) for r in val_records}
    test_norms = {normalize_text_for_dedup(r["text"]) for r in test_records}

    assert len(train_norms & val_norms) == 0, "Phát hiện câu trùng lặp giữa Train OOS và Val OOS!"
    assert len(train_norms & test_norms) == 0, "Phát hiện câu trùng lặp giữa Train OOS và Test OOS!"
    assert len(val_norms & test_norms) == 0, "Phát hiện câu trùng lặp giữa Val OOS và Test OOS!"

    # Lưu trữ các tệp CSV phân vùng OOS
    split_data = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }

    csv_fields = ["text", "label", "category", "source"]
    all_records: list[dict[str, str]] = []

    for split_name, records in split_data.items():
        split_csv = output_dir / f"out_of_scope_{split_name}.csv"
        with open(split_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for rec in records:
                writer.writerow(rec)
                all_records.append(rec)

    # Lưu tệp tổng hợp
    all_csv = output_dir / "out_of_scope_all.csv"
    with open(all_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for rec in all_records:
            writer.writerow(rec)

    return split_data


def mine_latex_math_candidates(
    sources_dir: Path = DATA_DIR,
    output_path: Path = MINED_CANDIDATES_PATH,
) -> list[dict[str, str]]:
    """
    Khai phá các câu mẫu và biểu thức công thức toán học LaTeX từ các nguồn tri thức có sẵn
    (tài liệu Overleaf tutorial, Obsidian Knowledge Base, và các khối phân tích section).

    Toàn bộ ứng viên khai phá được gán cờ 'pending_manual_review' (chờ thẩm định thủ công)
    để người phụ trách dự án kiểm tra trước khi sử dụng (Honesty Rule & Data Governance).

    Tham số:
        sources_dir: Thư mục chứa các tệp dữ liệu tri thức.
        output_path: Đường dẫn tệp CSV lưu trữ kết quả khai phá.

    Trả về:
        Danh sách các bản ghi ứng viên công thức toán đã khai phá.
    """
    candidates: list[dict[str, str]] = []
    seen_expressions: set[str] = set()
    candidate_counter = 1

    # 1. Khai phá từ data/knowledge/obsidian_knowledge.json
    obsidian_json_path = sources_dir / "knowledge" / "obsidian_knowledge.json"
    if obsidian_json_path.is_file():
        try:
            obs_data = json.loads(obsidian_json_path.read_text(encoding="utf-8"))
            records = obs_data.get("records", [])
            for rec in records:
                topic = rec.get("topic", "")
                title = rec.get("title", "")
                tmpl = rec.get("syntax_template", "").strip()
                # Lọc các bản ghi toán học có chứa lệnh LaTeX hoặc ký tự toán học
                has_math_tokens = any(
                    token in tmpl for token in ("\\", "$", "^", "_", "{", "}")
                )
                is_markdown_only = topic in ("obsidian_basic", "callouts", "tags", "mermaid", "footnotes") and not any(
                    cmd in tmpl for cmd in ("\\frac", "\\sqrt", "\\begin", "\\alpha", "\\int", "\\sum")
                )
                if has_math_tokens and not is_markdown_only and tmpl:
                    norm_expr = normalize_text_for_dedup(tmpl)
                    if norm_expr not in seen_expressions:
                        seen_expressions.add(norm_expr)
                        c_id = f"MATH_CAND_{candidate_counter:04d}"
                        candidate_counter += 1
                        candidates.append(
                            {
                                "id": c_id,
                                "text": f"Cách viết biểu thức {title} trong LaTeX hoặc MathJax",
                                "suggested_intent": f"tex_math_{topic}" if not topic.startswith("tex_") else topic,
                                "latex_code": tmpl,
                                "source_file": "obsidian_knowledge_base.txt",
                                "source_location": rec.get("id", "OBS"),
                                "notes": f"Chủ đề: {topic}; Kích hoạt: {title}",
                                "status": "pending_manual_review",
                            }
                        )
        except Exception as err:
            print(f"[CẢNH BÁO] Không thể đọc {obsidian_json_path}: {err}", file=sys.stderr)

    # 2. Khai phá từ data/knowledge/latex_tutorial_pages.json
    tutorial_pages_path = sources_dir / "knowledge" / "latex_tutorial_pages.json"
    if tutorial_pages_path.is_file():
        try:
            tut_data = json.loads(tutorial_pages_path.read_text(encoding="utf-8"))
            pages = tut_data.get("pages", [])
            # Regex phát hiện công thức inline, display, hoặc khối equation/matrix trong văn bản
            math_regex = re.compile(
                r"(\$[^$\n]+\$|\\\[[\s\S]+?\\\]|\\begin\{equation\*?\}[\s\S]+?\\end\{equation\*?\}|\\begin\{(?:matrix|pmatrix|bmatrix|vmatrix)\}[\s\S]+?\\end\{(?:matrix|pmatrix|bmatrix|vmatrix)\})",
                flags=re.MULTILINE,
            )
            for p in pages:
                page_num = p.get("page", 0)
                content = p.get("content", "")
                matches = math_regex.findall(content)
                for raw_match in matches:
                    clean_match = raw_match.strip()
                    # Loại bỏ các placeholder quá ngắn hoặc vô nghĩa như '$ ... $'
                    if len(clean_match) < 4 or clean_match in ("$ ... $", "\\[ ... \\]"):
                        continue
                    norm_expr = normalize_text_for_dedup(clean_match)
                    if norm_expr not in seen_expressions:
                        seen_expressions.add(norm_expr)
                        c_id = f"MATH_CAND_{candidate_counter:04d}"
                        candidate_counter += 1
                        candidates.append(
                            {
                                "id": c_id,
                                "text": f"Ví dụ công thức toán học trích xuất từ tài liệu Overleaf trang {page_num}",
                                "suggested_intent": "tex_math_formula",
                                "latex_code": clean_match,
                                "source_file": "learn_latex_tutorial.pdf",
                                "source_location": f"Page_{page_num}",
                                "notes": f"Khai phá từ khối toán học trang {page_num}",
                                "status": "pending_manual_review",
                            }
                        )
        except Exception as err:
            print(f"[CẢNH BÁO] Không thể đọc {tutorial_pages_path}: {err}", file=sys.stderr)

    # 3. Khai phá từ data/sources/raw/obsidian_knowledge_base.txt
    raw_obs_path = sources_dir / "sources" / "raw" / "obsidian_knowledge_base.txt"
    if raw_obs_path.is_file():
        try:
            raw_text = raw_obs_path.read_text(encoding="utf-8")
            # Tìm các khối mã LaTeX nằm trong dấu backtick ```latex ... ```
            code_blocks = re.findall(r"```latex\n([\s\S]*?)```", raw_text)
            for idx, block in enumerate(code_blocks, 1):
                clean_block = block.strip()
                if any(m in clean_block for m in ("$", "\\[", "\\begin{equation", "\\begin{matrix", "\\frac", "\\int")):
                    norm_expr = normalize_text_for_dedup(clean_block)
                    if norm_expr not in seen_expressions:
                        seen_expressions.add(norm_expr)
                        c_id = f"MATH_CAND_{candidate_counter:04d}"
                        candidate_counter += 1
                        candidates.append(
                            {
                                "id": c_id,
                                "text": f"Đoạn mã ví dụ LaTeX trích từ tệp ghi chú Obsidian khối {idx}",
                                "suggested_intent": "tex_math_block",
                                "latex_code": clean_block,
                                "source_file": "obsidian_knowledge_base.txt",
                                "source_location": f"CodeBlock_{idx}",
                                "notes": "Khai phá từ khối ```latex``` thô",
                                "status": "pending_manual_review",
                            }
                        )
        except Exception as err:
            print(f"[CẢNH BÁO] Không thể đọc {raw_obs_path}: {err}", file=sys.stderr)

    # Ghi toàn bộ ứng viên ra tệp CSV đích
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "text",
        "suggested_intent",
        "latex_code",
        "source_file",
        "source_location",
        "notes",
        "status",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cand in candidates:
            writer.writerow(cand)

    return candidates


def main(argv: Sequence[str] | None = None) -> int:
    """
    Điểm vào dòng lệnh (CLI) thực thi toàn bộ quy trình thu nhận dữ liệu ngoại bộ.
    """
    parser = argparse.ArgumentParser(
        description="Thu nhận dữ liệu PhoATIS, lấy mẫu OOS và khai phá ứng viên công thức LaTeX toán học."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Chỉ tải các tệp PhoATIS seq.in cấp âm tiết từ GitHub.",
    )
    parser.add_argument(
        "--sample-oos",
        action="store_true",
        help="Lấy mẫu câu Out-of-Scope (OOS) có seed cho các phân vùng train/val/test.",
    )
    parser.add_argument(
        "--mine-math",
        action="store_true",
        help="Khai phá ứng viên công thức toán LaTeX vào data/mined_candidates.csv.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Thực hiện tuần tự tất cả các bước (mặc định nếu không truyền tham số cụ thể).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bắt buộc tải lại tệp PhoATIS ngay cả khi đã tồn tại trên đĩa.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Hạt giống ngẫu nhiên (mặc định: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=PHOATIS_DIR,
        help=f"Thư mục lưu trữ PhoATIS (mặc định: {PHOATIS_DIR}).",
    )

    args = parser.parse_args(argv)

    # Nếu không chọn cờ đơn lẻ nào thì mặc định thực hiện tất cả
    run_all = args.all or not (args.download or args.sample_oos or args.mine_math)

    print("=== Vietnamese LaTeX Knowledge Assistant: External Data Acquisition ===")

    # 1. Tải PhoATIS
    if run_all or args.download or args.sample_oos:
        print("\n[Bước 1/3] Kiểm tra và tải dữ liệu PhoATIS cấp âm tiết...")
        try:
            download_results = download_phoatis_data(target_dir=args.target_dir, force=args.force)
            for split, path in download_results.items():
                size_kb = path.stat().st_size / 1024
                print(f"  + [{split.upper()}] {path} ({size_kb:.1f} KB)")
        except RuntimeError as e:
            print(f"\n[LỖI] Dừng tiến trình thu nhận do không có kết nối mạng:\n{e}", file=sys.stderr)
            return 1

    # 2. Lấy mẫu Out-of-Scope
    if run_all or args.sample_oos:
        print("\n[Bước 2/3] Lấy mẫu câu Out-of-Scope (OOS) có seed và kiểm tra không trùng lặp...")
        try:
            splits = sample_out_of_scope_splits(
                phoatis_dir=args.target_dir,
                output_dir=args.target_dir,
                seed=args.seed,
            )
            print(f"  + TRAIN OOS: {len(splits['train'])} câu")
            print(f"  + VAL OOS:   {len(splits['val'])} câu")
            print(f"  + TEST OOS:  {len(splits['test'])} câu")
            print("  + Kiểm tra Honesty Rule #4 (trùng lặp = 0): ĐẠT")
        except Exception as err:
            print(f"\n[LỖI] Thất bại khi lấy mẫu OOS: {err}", file=sys.stderr)
            return 1

    # 3. Khai phá công thức toán LaTeX
    if run_all or args.mine_math:
        print("\n[Bước 3/3] Khai phá ứng viên công thức LaTeX toán học...")
        try:
            candidates = mine_latex_math_candidates(
                sources_dir=DATA_DIR,
                output_path=MINED_CANDIDATES_PATH,
            )
            print(f"  + Khai phá thành công {len(candidates)} ứng viên vào {MINED_CANDIDATES_PATH}")
            print("  + Trạng thái toàn bộ: 'pending_manual_review' (Chờ duyệt thủ công)")
        except Exception as err:
            print(f"\n[LỖI] Thất bại khi khai phá công thức toán: {err}", file=sys.stderr)
            return 1

    print("\nHoàn tất toàn bộ quy trình thu nhận dữ liệu ngoại bộ.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
