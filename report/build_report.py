"""Sinh báo cáo từ artifact hiện có; số liệu lấy trực tiếp từ JSON/CSV."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "REPORT.md"
TEST_CSV = ROOT / "data" / "test.csv"
PROGRESS = ROOT / "PROGRESS.md"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import PHOBERT_DIR


def _json_files() -> dict[str, Any]:
    data: dict[str, Any] = {}
    expected_hash = hashlib.sha256(TEST_CSV.read_bytes()).hexdigest()
    progress_hashes = re.findall(r"[0-9a-f]{64}", PROGRESS.read_text(encoding="utf-8"))
    if expected_hash not in progress_hashes:
        raise ValueError("data/test.csv hash does not match PROGRESS.md")
    with TEST_CSV.open(encoding="utf-8-sig", newline="") as stream:
        expected_rows = max(sum(1 for _ in csv.reader(stream)) - 1, 0)
    for path in sorted((ROOT / "report").glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                recorded_hash = value.get("test_csv_sha256")
                if recorded_hash is not None and recorded_hash != expected_hash:
                    continue
                if value.get("smoke_mode") is True:
                    continue
                split_counts = value.get("data_split_counts", {})
                if split_counts.get("test") is not None and split_counts["test"] != expected_rows:
                    continue
            data[path.name] = value
        except (OSError, json.JSONDecodeError):
            continue
    return data


def _csv_files() -> dict[str, list[dict[str, str]]]:
    data: dict[str, list[dict[str, str]]] = {}
    for path in sorted((ROOT / "report").glob("*.csv")):
        try:
            with path.open(encoding="utf-8", newline="") as stream:
                data[path.name] = list(csv.DictReader(stream))
        except (OSError, csv.Error):
            continue
    return data


def build_report() -> str:
    artifacts = _json_files()
    csv_artifacts = _csv_files()
    for name, rows in csv_artifacts.items():
        artifacts[name] = {"rows": len(rows), "columns": list(rows[0]) if rows else []}
    data_report = ROOT / "data" / "data_report.json"
    dataset: dict[str, Any] = {}
    if data_report.exists():
        dataset = json.loads(data_report.read_text(encoding="utf-8"))
        artifacts["data_report.json"] = {
            key: dataset[key]
            for key in ("source_files", "intent_count", "split_counts", "exact_duplicate_counts")
            if key in dataset
        }
    split_table = "\n".join(
        f"| {split} | {count} |" for split, count in dataset.get("split_counts", {}).items()
    ) or "| Chưa có artifact | Chưa có số liệu |"
    dataset_notes = "\n".join(f"- {note}" for note in dataset.get("notes", []))
    barriers: list[str] = []
    if not (ROOT / "data" / "slot_gold.csv").exists():
        barriers.append("- Chưa có `data/slot_gold.csv`; chưa thể chạy đánh giá gold spans.")
    if not any((PHOBERT_DIR / filename).exists() for filename in ("pytorch_model.bin", "model.safetensors")):
        barriers.append("- Cache PhoBERT chưa có file trọng số; trainer chỉ được kiểm tra bằng encoder nhỏ trong unit test.")
    if shutil.which("pdflatex") is None:
        barriers.append("- BLOCKED: `pdflatex` chưa có; LaTeX hiện chỉ được kiểm tra tĩnh, chưa compile.")
    barrier_notes = "\n".join(barriers)
    claim_rows = [
        "| [C01] PhoBERT dùng phân đoạn từ tiếng Việt và BPE trước fine-tuning. | `core/preprocess.py`, `core/config.py`, `Prompt_Pack_Knowledge_Assistant_LaTeX.md` §13 |",
        "| [C02] JointBERT dự đoán intent từ biểu diễn token đầu và cộng loss intent với loss slot trung bình theo từ. | `train/joint_intent_slot.py` (`JointIntentSlotModel.forward`) |",
        "| [C03] BiJoint tạo attention theo intent, ghép vector attention với biểu diễn từ, rồi tổng các xác suất slot để dự đoán intent cuối. | `train/bijoint_nlu.py` (`BiJointNLU.forward`) |",
        "| [C04] SBERT lập chỉ mục embedding và xếp hạng bằng cosine similarity; ba loss bi-encoder được triển khai trong code huấn luyện. | `core/retrieval.py`, `train/losses.py`, `train/triplet_sbert.py` |",
        "| [C05] Scenario bank biểu diễn intent bằng chuỗi scenario và aliases theo hướng tiếp cận lấy cảm hứng từ RAF. | `core/scenario.py`, `core/retrieval.py`, `AGENTS.md` (Honesty rule 1) |",
        "| [C06] Trợ lý định tuyến yêu cầu qua classifier/retriever, slot regex và LaTeX template renderer. | `core/pipeline.py` (`Pipeline.predict`), `core/slots.py`, `core/latex_tools.py` |",
        "| [C07] Dữ liệu được lấy từ các nguồn và có kích thước split nêu trong bảng. | `data/data_report.json` (`source_files`, `split_counts`) |",
        "| [C08] Kết quả chỉ được đưa vào report nếu artifact dùng đúng test hash hiện tại và không phải smoke hoặc partial-test run. | `report/build_report.py` (`_json_files`) |",
        "| [C09] Kiểm định intent dùng McNemar với hiệu chỉnh Yates trên cùng câu và seed. | `train/nlu_training.py` (`compare_intent_predictions`), `train/evaluate.py` |",
        "| [C10] Slot filler hiện là regex; CRF chưa được triển khai. | `core/slots.py`, `train/joint_intent_slot.py`, `train/bijoint_nlu.py` |",
        "| [C11] PhoATIS OOS có trong train/val và test riêng; chưa có owner gold slot spans. | `data/data_report.json`, `data/test_oos.csv`, `data/SLOT_GOLD_SCHEMA.md` |",
        "| [C12] LaTeX hiện chỉ qua kiểm tra tĩnh; compile chưa được xác nhận trong lần build này. | `core/latex_tools.py`, `cli/ka.py`, `AGENTS.md` (Honesty rule 7) |",
    ]
    claim_table = "\n".join(claim_rows)
    artifact_lines = [f"- `{name}`: `{json.dumps(value, ensure_ascii=False, sort_keys=True)[:500]}`"
                      for name, value in artifacts.items()]
    metric_rows: list[str] = []

    def add_metric_row(name: str, payload: dict[str, Any]) -> None:
        aggregate = payload.get("aggregate_test", {})
        if not aggregate:
            return
        intent = aggregate.get("intent_accuracy", {})
        slot = aggregate.get("slot_f1", {})
        exact = aggregate.get("exact_slot_accuracy", {})
        sentence = aggregate.get("sentence_accuracy", {})
        metric_rows.append(
            f"| {name} | {intent.get('mean', '—')} ± {intent.get('std', '—')} "
            f"| {slot.get('mean', '—')} ± {slot.get('std', '—')} "
            f"| {exact.get('mean', '—')} ± {exact.get('std', '—')} "
            f"| {sentence.get('mean', '—')} ± {sentence.get('std', '—')} |"
        )

    for name, payload in artifacts.items():
        if not isinstance(payload, dict):
            continue
        add_metric_row(str(payload.get("architecture", name)), payload)
        regex_metrics = payload.get("regex_baseline_test")
        if (isinstance(regex_metrics, dict)
                and payload.get("architecture") in {"joint", "bijoint"}):
            metric_rows.append(
                f"| Regex extractor KA-08 ({name}) | — | {regex_metrics.get('slot_f1', '—')} "
                f"| {regex_metrics.get('exact_slot_accuracy', '—')} | — |"
            )
        for key in ("ka05_intent_only", "no_joint_ablation"):
            if isinstance(payload.get(key), dict):
                add_metric_row(key, payload[key])
    metric_table = "\n".join(metric_rows) or "| Chưa có kết quả KA-11/KA-12 hợp lệ | — | — | — | — |"
    comparison_rows: list[str] = []
    for name, payload in artifacts.items():
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if key.startswith("intent_mcnemar_") and isinstance(value, dict):
                comparison_rows.append(
                    f"| {name}: {key.removeprefix('intent_mcnemar_')} | {value.get('b', '—')} "
                    f"| {value.get('c', '—')} | {value.get('statistic_yates', '—')} "
                    f"| {value.get('p_value', '—')} |"
                )
    comparison_table = "\n".join(comparison_rows) or "| Chưa có cặp dự đoán hợp lệ | — | — | — | — |"
    if not artifact_lines:
        artifact_lines = ["- Chưa có artifact số liệu; không công bố số thực nghiệm."]
    content = """# Báo cáo Knowledge Assistant LaTeX

## 1. Bài toán và phạm vi

Trợ lý nhận yêu cầu tiếng Việt, nhận diện intent, trích slot bằng regex và kết xuất template LaTeX đã kiểm tra tĩnh.

## 2. Cơ sở lý thuyết

- [C01] PhoBERT dùng phân đoạn từ tiếng Việt và BPE, sau đó fine-tuning.
- [C02] JointBERT dự đoán intent từ token đầu `y_i = softmax(W_i h_0 + b_i)` và cộng intent loss với slot loss trung bình theo từ.
- [C03] BiJoint dùng intent tạm, attention trên các từ, vector ghép và tổng xác suất slot để tạo intent cuối; loss gồm ba nhánh có trọng số.
- [C04] SBERT tạo embedding mean-pooling và xếp hạng bằng cosine; code có contrastive in-batch, contrastive hard-negative và triplet loss.
- [C05] Scenario bank dùng scenario string và aliases. Đây là **phiên bản đơn giản hóa lấy cảm hứng từ Retrieve-and-Fill** (*phien ban don gian hoa lay cam hung tu Retrieve-and-Fill*).

## 3. Kiến trúc hệ thống

[C06] Pipeline nối tiền xử lý, phân loại/truy hồi, slot regex, kiểm tra an toàn và render template. CLI/API dùng chung pipeline.

## 4. Dữ liệu

[C07] Nguồn dữ liệu: {{sources}}.

| Phân vùng | Số mẫu |
|---|---:|
{{split_table}}

## 5. Thực nghiệm

[C08] Artifact được đọc từ report JSON/CSV. Smoke artifact nằm trong `report/smoke/` bị bỏ qua; artifact có test hash khác PROGRESS.md bị loại.

| Mô hình | Intent accuracy mean ± std | Slot F1 mean ± std | Exact-slot mean ± std | Sentence accuracy mean ± std |
|---|---:|---:|---:|---:|
{{metric_table}}

""" + "\n".join(artifact_lines) + """

## 6. Phân tích lỗi

[C09] McNemar dùng hiệu chỉnh Yates; chỉ so sánh dự đoán trên cùng câu gold và cùng seed.

| So sánh | b (A đúng, B sai) | c (A sai, B đúng) | Chi-square | p-value |
|---|---:|---:|---:|---:|
{{comparison_table}}

## 7. Hạn chế

[C10] Slot filler hiện là regex và các mô hình sequence không có CRF. LaTeX chỉ được gọi là đã biên dịch khi lệnh compiler chạy thành công.
[C11] PhoATIS được dùng trong OOS train/val và test riêng; `data/slot_gold.csv` chưa có nên chưa thể báo cáo kết quả slot gold.

`data/test.csv` là test do AI soạn, chưa phải benchmark cuối. Kết quả trên đó là sơ bộ; kết luận cuối cần `data/test_owner.csv` do owner viết. PhoATIS có 65/27 OOS rows trong train/val và 27 rows riêng trong `data/test_oos.csv`.
LaTeX: static check only; chưa xác nhận biên dịch bằng TeX Live.

{{dataset_notes}}
{{barrier_notes}}

## 8. Hướng phát triển

Đánh giá JointBERT/BiJoint trên gold spans, mở rộng dữ liệu viết tay, và cải thiện trích slot sau khi đối chiếu kết quả.

## Phụ lục: công thức và checklist

| Thành phần | Công thức/kiểm tra |
|---|---|
| Intent | `softmax(W h_0 + b)` |
| Retrieval | cosine similarity trên embedding đã chuẩn hóa |
| JointBERT | `L_intent + L_slot` |
| BiJoint | `lambda_tmp L_tmp + lambda_ST L_ST + lambda_final L_final`; `q = sum_j p(slot_j)` |
| Tính trung thực | Không chỉnh test; paper numbers chỉ ghi là số được công bố; kiểm tra static không đồng nghĩa biên dịch |

## Bảng claims và nguồn

| Claim trong báo cáo | File/giá trị hỗ trợ |
|---|---|
{{claim_table}}
"""
    content = content.replace("{{sources}}", ", ".join(dataset.get("source_files", [])) or "chưa ghi trong artifact")
    content = content.replace("{{split_table}}", split_table)
    content = content.replace("{{dataset_notes}}", dataset_notes)
    content = content.replace("{{barrier_notes}}", barrier_notes)
    content = content.replace("{{claim_table}}", claim_table)
    content = content.replace("{{metric_table}}", metric_table)
    content = content.replace("{{comparison_table}}", comparison_table)
    if re.search(r"\bTODO\b|\bTBD\b|\bXXX\b|\bN/A\b|\[\s*(?:number|value|metric)\s*\]", content, re.IGNORECASE):
        raise ValueError("Báo cáo còn TODO hoặc placeholder số liệu")
    return content


def main() -> None:
    REPORT.write_text(build_report(), encoding="utf-8")
    print(f"Đã sinh {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
