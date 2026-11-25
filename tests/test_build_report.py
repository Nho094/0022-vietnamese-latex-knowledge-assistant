from __future__ import annotations

from report.build_report import build_report


def test_report_has_all_sections_and_no_placeholders() -> None:
    report = build_report()
    for heading in (
        "## 1. Bài toán", "## 2. Cơ sở lý thuyết", "## 3. Kiến trúc",
        "## 4. Dữ liệu", "## 5. Thực nghiệm", "## 6. Phân tích lỗi",
        "## 7. Hạn chế", "## 8. Hướng phát triển", "## Bảng claims và nguồn",
    ):
        assert heading in report
    assert "phien ban don gian hoa lay cam hung tu Retrieve-and-Fill" in report
    assert "{{" not in report and "TODO" not in report and "TBD" not in report
    assert "| train | 3825 |" in report
    assert "| val | 967 |" in report
    assert "| test | 940 |" in report
    assert "test do AI soạn, chưa phải benchmark cuối" in report
    assert "PhoATIS được dùng trong OOS train/val và test riêng" in report
    assert "LaTeX: static check only" in report
    assert "models/smoke" in report or "report/smoke" in report
