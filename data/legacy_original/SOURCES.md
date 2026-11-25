# Nguồn dữ liệu và quy tắc bảo toàn

1. `Learn LaTeX in 30 minutes - Overleaf, Online LaTeX Editor.pdf`: 40 trang. Text được trích bằng `pdftotext -layout`, lưu từng trang trong `knowledge/latex_tutorial_pages.json`, và chia section trong `knowledge/latex_tutorial.json`.
2. `knowledge_base.txt`: tài liệu runtime Obsidian/MathJax do người dùng cung cấp. Các cặp INTENT/OUTPUT/NOTES được parse thành `knowledge/obsidian_knowledge.json`; trường `raw_text` giữ toàn bộ văn bản nguồn.
3. Các file gốc được giữ trong `sources/raw/`. Các file data cũ được giữ trong `legacy_original/`.

Không có câu TEST/VAL nào được tự sinh. Câu synthetic chỉ nằm trong TRAIN và được đánh dấu ở `train_metadata.csv`.
