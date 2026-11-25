# Hướng dẫn tạo benchmark owner-written

`data/test.csv` hiện được phân loại là `ai_draft`: các câu do AI soạn, chưa phải benchmark cuối cùng. Không sửa tệp này.

1. Sao chép `data/test_owner_template.csv` thành `data/test_owner.csv`.
2. Viết câu tiếng Việt tự nhiên, ngắn từ 10–25 từ, theo category của từng dòng. Không sao chép boilerplate, alias hoặc câu từ train/val/test AI.
3. Điền `label` bằng intent có trong `data/intents.json`; `written_by` ghi tên/định danh người viết. Mỗi category có 3 dòng mẫu trống; có thể thêm dòng nếu cần.
4. Giữ nguyên header `text,label,category,written_by`; không để trường trống trong benchmark hoàn chỉnh.
5. Trước đánh giá, chạy `python data/verify_owner_test.py`. Script kiểm tra schema, nhãn/category, trùng nội bộ và overlap với train/val/AI draft; sau đó ghi SHA-256 trong `PROGRESS.md`.
6. Khi file hợp lệ tồn tại, các đánh giá cuối dùng `test_owner.csv` làm primary test. `test.csv` vẫn giữ nguyên và mọi kết quả của nó phải ghi rõ: **test do AI soạn, chưa phải benchmark cuối**.

Tập mẫu hiện tại có ba dòng trống cho mỗi category để chủ sở hữu tự điền; các dòng đó không phải dữ liệu benchmark.
