# Architecture Decision Record (ADR)

## ADR-001: Độ lệch thiết kế so với Retrieve-and-Fill (RAF) — Slot tùy chọn và giá trị mặc định cho Name và Caption

- **Trạng thái (Status)**: Đã chấp thuận (Accepted)
- **Ngày quyết định (Date)**: 2026-11-23
- **Phạm vi tác động (Scope)**: `data/intents.json`, `core/validator.py`, `src/pipeline.py`, `src/scenarios.py`

---

### 1. Bối cảnh kiến trúc (Context)

Hệ thống **Vietnamese LaTeX Knowledge Assistant** tuân thủ luồng kiến trúc cốt lõi được quy định trong `AGENTS.md`:
```text
text -> preprocess (NFC, math symbol normalization, pyvi word segmentation for PhoBERT)
  -> PhoBERT classifier: p = softmax over intents (or over categories in two-tier mode)
     - if max p >= tau: intent accepted
     - else retrieval top-3 over aliases + scenario strings -> ask the user to choose (if best score < theta: say not understood)
  -> slot extraction (regex per intent) -> if a required slot is missing: ask for exactly the missing slots
  -> render latex template (text slots escaped) -> output
```

Dự án lấy cảm hứng từ mô hình **Retrieve-and-Fill (RAF)** trong các hệ thống đối thoại hướng tác vụ dựa trên kịch bản (Scenario-based Task-oriented Dialogue). 

Theo **Honesty Rule #1 (AGENTS.md)**:
> *"The filling step is regex, far simpler than the span-pointer decoder of Retrieve-and-Fill (RAF). Always describe the project as a simplified scenario-based system inspired by RAF, never as a re-implementation."*
> *(Bước điền slot dựa trên biểu thức chính quy (regex), đơn giản hơn nhiều so với bộ giải mã trỏ đoạn span-pointer decoder của Retrieve-and-Fill (RAF). Luôn mô tả dự án là một hệ thống dựa trên kịch bản đơn giản hóa lấy cảm hứng từ RAF, không bao giờ tự nhận là bản tái hiện nguyên gốc).*

---

### 2. Vấn đề phát sinh với cơ chế RAF nguyên bản (Problem Statement)

Trong kiến trúc RAF truyền thống:
1. Bộ giải mã span-pointer decoder cố gắng trích xuất các phân đoạn (spans) trực tiếp từ câu truy vấn của người dùng để điền vào các vị trí trống của mẫu kịch bản (template slots).
2. Nếu một slot không tìm thấy span tương ứng trong câu người dùng, hệ thống RAF nguyên bản coi đó là thiếu thông tin và kích hoạt cơ chế hỏi lại người dùng qua nhiều lượt hội thoại (multi-turn clarification dialog) hoặc để trống.

Tuy nhiên, trong bài toán sinh mã LaTeX cho người dùng Việt Nam:
- Các câu lệnh tự nhiên thường ngắn gọn và mang tính hành động trực tiếp, ví dụ:
  - *"chèn hình ảnh sơ đồ graph.png"*
  - *"tạo môi trường figure ảnh diagram.pdf"*
  - *"tạo bảng 2 cột"*
  - *"tham chiếu đến nhãn fig:ketqua"*
- Các thuộc tính như **chú thích (caption)**, **nhãn tham chiếu (label)**, **tiêu đề (title)**, **tác giả (author)** hay **tỷ lệ chiều rộng (width)** rất hay bị người dùng lược bỏ trong câu yêu cầu đầu tiên.
- Nếu áp dụng nghiêm ngặt cơ chế span extraction của RAF và bắt buộc mọi slot trong template phải được trích xuất từ câu truy vấn, hệ thống sẽ liên tục ngắt quãng người dùng bằng các câu hỏi làm rõ (ví dụ: *"Vui lòng nhập caption cho hình"*, *"Vui lòng nhập nhãn label"*), gây ra trải nghiệm người dùng (UX) rất kém và làm mất đi tính tiện dụng của việc sinh mã LaTeX tức thì.

---

### 3. Quyết định kiến trúc (Decision: RAF Design Deviation)

Chúng tôi quyết định áp dụng một **độ lệch thiết kế có chủ ý (deliberate design deviation)** so với RAF nguyên bản:

1. **Phân tách rõ ràng giữa Required Slots và Optional Slots có Default Values**:
   - Mỗi slot trong ngân hàng kịch bản `data/intents.json` được định nghĩa tường minh với các thuộc tính:
     - `type`: Kiểu dữ liệu (`number`, `text`, `label`, `filename`).
     - `regex`: Biểu thức chính quy trích xuất (chứa đúng 1 capturing group).
     - `required`: Boolean (`True` nếu bắt buộc, `False` nếu tùy chọn).
     - `default`: Giá trị mặc định (chuỗi, số, hoặc `null`).

2. **Cơ chế xử lý Slot tùy chọn với Giá trị mặc định (Optional Slots with Defaults)**:
   - Đối với các slot mang tính bổ trợ như `caption`, `label`, `title`, `author`, `width`, `options`:
     - Thiết lập `required: false`.
     - Cung cấp giá trị mặc định hợp lý và sẵn sàng biên dịch (compilable defaults), ví dụ:
       - `caption`: `"Hình minh họa"` hoặc `"Nội dung chú thích"`.
       - `label`: `"fig:hinh"`, `"tab:bang"` hoặc `"fig:ten-nhan"`.
       - `width`: `0.8` hoặc `0.75`.
       - `title`: `"Tiêu đề tài liệu"`.
       - `author`: `"Tên tác giả"`.
   - Khi người dùng cung cấp cụ thể giá trị trong câu (ví dụ: *"chèn hình test.png với chú thích Biểu đồ tăng trưởng"*), biểu thức regex sẽ bóc tách được và ghi đè lên giá trị mặc định.
   - Khi người dùng không đề cập trong câu, hệ thống tự động điền giá trị mặc định vào placeholder `<<slot>>`, cho phép sinh ra đoạn mã LaTeX hoàn chỉnh có thể biên dịch được ngay mà không cần dừng lại hỏi người dùng.

3. **Chỉ kích hoạt Clarification khi thiếu Slot bắt buộc (`required: true`)**:
   - Hệ thống chỉ hỏi lại người dùng đúng các slot bắt buộc bị thiếu (`required: true` và không có `default`, ví dụ: tên tệp `filename` trong `LATEX_CHEN_HINH_ANH`, tên gói `package_name` trong `LATEX_NAP_PACKAGE`).
   - Đúng theo quy tắc tại `AGENTS.md`: *"if a required slot is missing: ask for exactly the missing slots"*.

---

### 4. Hệ quả và Đánh giá (Consequences)

#### Ưu điểm:
- **Tối ưu hóa trải nghiệm người dùng (UX)**: Người dùng nhận được mã LaTeX hoàn chỉnh ngay trong 1 lượt tương tác (single-turn) cho đa số các nhu cầu phổ biến.
- **Tính khả biên dịch (Compilability)**: Các môi trường phức tạp như `figure`, `table`, `article` luôn có đủ tham số hợp lệ để biên dịch thông qua `pdflatex -no-shell-escape` mà không bị lỗi cú pháp thiếu tham số bắt buộc của gói LaTeX.
- **Kiểm soát chặt chẽ qua Schema Validator**: Module `core/validator.py` kiểm thực tính nhất quán giữa danh sách slot được khai báo và các placeholder `<<slot>>` trong template LaTeX.

#### Nhược điểm và Giải pháp giảm thiểu:
- Người dùng có thể cần chỉnh sửa lại giá trị mặc định (ví dụ đổi `"Hình minh họa"` thành chú thích cụ thể của họ) trong file TeX của họ. Đây là hành vi hoàn toàn tự nhiên và mong muốn đối với người soạn thảo LaTeX.
- Biểu thức regex cho các slot tùy chọn phải được thiết kế cẩn thận với capturing group đơn (`groups == 1`) để không bắt nhầm các từ khóa thông thường trong tiếng Việt.

---

### 5. Tuân thủ Quy tắc trung thực (Honesty Rules Compliance)

Quyết định này tuân thủ tuyệt đối **Honesty Rule #1** trong `AGENTS.md`:
- Tài liệu hóa minh bạch rằng hệ thống sử dụng phương pháp **Regex Slot Extraction kết hợp Default Values**, không sử dụng và không giả lập bộ giải mã mạng nơ-ron span-pointer decoder của RAF.
- Việc sử dụng giá trị mặc định cho caption, label, và name là một cải tiến thực dụng (pragmatic heuristic) phục vụ mục tiêu hỗ trợ người dùng soạn thảo LaTeX nhanh chóng và chính xác.
