# Unified LaTeX + Obsidian dataset

## Split
- TRAIN: 10 câu/intent, câu tổng hợp có seed.
- VAL: 5 câu/intent, câu do AI soạn khác cách diễn đạt TRAIN; cần owner review.
- TEST: 5 câu/intent, câu do AI soạn theo yêu cầu mới của owner; cần owner review. Đây là ngoại lệ so với KA-02 gốc.
- OUT_OF_SCOPE: chưa có PhoATIS, nên hiện chưa có ví dụ.

## Schema mở rộng
- `tex_*`: tạo LaTeX.
- `obsidian`: cú pháp riêng Obsidian/Markdown.
- `math_shared`: cú pháp công thức MathJax dùng chung.
- `templates.latex`, `templates.obsidian`, `templates.mathjax`; `python` và `cpp` để rỗng.

`knowledge/knowledge_base.json` là lớp retrieval; `intents.json` là lớp intent/template. Không trộn hai vai trò thành một.


## Knowledge Engine Data Contract v2

`intents.json` giữ nguyên các trường legacy (`slots`, `templates`, aliases, category, ...), đồng thời mỗi Intent có thêm các trường chuẩn hóa:

- `definition_version`: phiên bản định nghĩa.
- `input_schema`: JSON Schema Draft 2020-12 mô tả payload đầu vào sau khi trích xuất.
- `field_policies`: quy tắc trích xuất từng trường, kiểu dữ liệu, bắt buộc/tuỳ chọn và hành vi khi thiếu.
- `transform`: cấu hình JSONata. Mặc định `$` là phép biến đổi identity (giữ nguyên payload).
- `render`: template Jinja2 theo từng định dạng đầu ra không rỗng.

### Nguyên tắc tương thích

- Không xóa hoặc đổi tên trường legacy để tránh làm hỏng code và dữ liệu huấn luyện hiện tại.
- Không thay đổi CSV, nhãn BIO, nguồn dữ liệu hoặc nội dung knowledge corpus trong lần migration này.
- Các corpus trong `knowledge/` là lớp retrieval/tài liệu tham khảo; chúng không phải Intent definitions.
- `OUT_OF_SCOPE` không có template đầu ra vì đây là nhãn ngoài phạm vi.

### Thêm kiến thức mới

GUI nên tạo `input_schema` trước, sau đó tạo `field_policies` tương ứng cho từng field và cấu hình `transform`/`render`. Với dữ liệu phức tạp, `input_schema` có thể khai báo object, array, nested object/array theo JSON Schema. Không giới hạn cấu trúc ở các scalar Slot cũ.

### An toàn khi thực thi

- Không dùng `eval()` hoặc thực thi Python tùy ý từ JSON.
- Chỉ cho phép JSONata expressions trong engine JSONata đã kiểm soát.
- Render Jinja2 trong môi trường sandbox; không bật quyền truy cập đối tượng/hàm Python tùy ý.
- Validate schema và kiểm tra template trước khi cho phép lưu/publish kiến thức.
- Regex trong dữ liệu cần giới hạn độ dài/timeout để tránh biểu thức gây backtracking quá mức.
