# Data package — Vietnamese LaTeX Knowledge Assistant

## Thiết kế theo Prompt Pack

- `intents.json`: scenario bank KA-01, chỉ chứa các intent sinh mã LaTeX theo nhóm `tex_*`; template dùng placeholder `<<slot>>`; `python` và `cpp` để trống.
- `knowledge/knowledge_base.json`: kho truy xuất kết hợp kiến thức LaTeX và Obsidian/MathJax.
- `knowledge/latex_tutorial.json`: kiến thức chia theo section của tutorial Overleaf.
- `knowledge/latex_tutorial_pages.json`: text trích xuất theo từng trang PDF để kiểm tra độ phủ.
- `knowledge/obsidian_knowledge.json`: các mục INTENT/OUTPUT/NOTES của knowledge_base Obsidian, đồng thời giữ `raw_text` nguyên bản.
- `train.csv`: dữ liệu TRAIN gồm ví dụ cũ phù hợp với intent LaTeX và câu synthetic được sinh có kiểm soát từ aliases. Xem `train_metadata.csv` để phân biệt nguồn.
- `val.csv`, `test.csv`, `raw_sentences.csv`: để trống có chủ ý. Prompt Pack yêu cầu VAL do chủ sở hữu duyệt và TEST do chủ sở hữu tự viết; không tự tạo hoặc sửa TEST.
- `source_coverage.csv`: truy vết intent, section, trang PDF và từng mục Obsidian.
- `legacy_original/`: bản sao các file data.zip cũ để khôi phục/đối chiếu.
- `sources/raw/`: bản gốc PDF và knowledge_base.txt.

## Trạng thái

Gói này đã có kho kiến thức để truy xuất và TRAIN khởi tạo. Chưa thể huấn luyện/đánh giá hoàn chỉnh theo KA-04/KA-05/KA-10 cho đến khi có VAL/TEST do chủ sở hữu viết hoặc duyệt và có PhoATIS cho lớp OUT_OF_SCOPE. Không dùng TRAIN làm VAL/TEST.

## Lưu ý về phạm vi

Prompt Pack hiện đặt `TARGET='latex'` và chỉ định các nhóm intent `tex_*`. Vì vậy cú pháp Obsidian (wikilink, callout, tag, Mermaid, Markdown...) được lưu đầy đủ trong kho kiến thức nhưng không tự chuyển thành intent sinh LaTeX. Nếu muốn hệ thống sinh cả Obsidian thì đó là thay đổi phạm vi/kiến trúc cần quyết định riêng.
