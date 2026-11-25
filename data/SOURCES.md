# Nguồn dữ liệu và Quy tắc Quản trị Dữ liệu (Data Sources & Governance)

Tài liệu này ghi nhận chi tiết nguồn gốc dữ liệu (provenance), điều khoản cấp phép bản quyền (licensing terms), và quy tắc phân chia tập dữ liệu (split rules) cho dự án Vietnamese LaTeX Knowledge Assistant theo đúng quy chuẩn kiến trúc và nguyên tắc trung thực (Honesty Rules) trong `AGENTS.md`.

---

## 1. Nguồn gốc dữ liệu (Dataset Provenance)

Hệ thống kết hợp tri thức LaTeX/Obsidian và PhoATIS cho huấn luyện/đánh giá OOS. Theo artifact `data/data_report.json`, dữ liệu train/val/test hiện lần lượt có 3825/967/940 dòng; các thống kê nhãn và split-source counts được sinh từ CSV trong artifact đó:

### 1.1. Bộ dữ liệu bên ngoài PhoATIS (VinAI Research)
- **Tên bộ dữ liệu:** PhoATIS (A Joint Segmenting, Slot-filling, and Intent-linking Dataset for Vietnamese Spoken Language Understanding).
- **Tác giả:** Dao et al., công bố tại Interspeech 2021.
- **Kho lưu trữ chính thức:** [VinAIResearch/JointIDSF](https://github.com/VinAIResearch/JointIDSF) trên GitHub.
- **Tệp thu nhận:** Cấp độ âm tiết (syllable-level) định dạng `seq.in`:
  - `train/seq.in` (lấy về `data/external/phoatis/train/seq.in`)
  - `dev/seq.in` (lấy về `data/external/phoatis/dev/seq.in`)
  - `test/seq.in` (lấy về `data/external/phoatis/test/seq.in`)
- **Vai trò trong hệ thống:** Cung cấp nguồn câu thoại tự nhiên tiếng Việt thực tế trong miền đặt vé máy bay/du lịch. Đây là nguồn dữ liệu âm tính ngoại phạm vi (Out-of-Scope / OOS) chất lượng cao, giúp bộ phân loại PhoBERT phân biệt chính xác giữa yêu cầu tạo LaTeX và các câu hỏi ngoài phạm vi hệ thống.

### 1.2. Hướng dẫn Overleaf LaTeX Tutorial
- **Nguồn tệp:** `sources/raw/learn_latex_tutorial.pdf` (tài liệu "Learn LaTeX in 30 minutes" từ Overleaf Online LaTeX Editor, 40 trang).
- **Xử lý trích xuất:**
  - `knowledge/latex_tutorial_pages.json`: Văn bản thô trích xuất theo từng trang kèm metadata.
  - `knowledge/latex_tutorial.json`: Cấu trúc 32 phân mục bài học LaTeX (cấu trúc tài liệu, môi trường toán, hình ảnh, bảng biểu, danh sách, tham chiếu).
- **Vai trò:** Cung cấp tri thức mẫu chuẩn cho các intent nhóm `tex_*`.

### 1.3. Kho tri thức Obsidian & MathJax
- **Nguồn tệp:** `sources/raw/obsidian_knowledge_base.txt` (tái cấu trúc từ `knowledge_base(1).txt` do người dùng cung cấp).
- **Xử lý trích xuất:** `knowledge/obsidian_knowledge.json` gồm 146 bản ghi thuộc 19 chủ đề toán học và ghi chú Obsidian (callouts, symbols, matrices, cases, arrays, font, etc.).
- **Vai trò:** Cung cấp mẫu cú pháp cho nhóm ý định `obsidian` và `math_shared`.

### 1.4. Ngân hàng kịch bản ý định (`data/intents.json`)
- Ngân hàng chứa 189 ý định chuẩn hóa (188 ý định chuyên môn thuộc các nhóm `tex_struct`, `tex_doc`, `tex_math`, `obsidian`, `math_shared`,... và 1 ý định đặc biệt `OUT_OF_SCOPE`).
- Validator toàn bank kiểm tra schema. Bốn intent có alias do quy tắc sinh xác định và được đánh dấu `review_needed`; chi tiết ở `docs/ALIAS_REVIEW.md`.

### 1.5. Ứng viên công thức toán khai phá (`data/mined_candidates.csv`)
- Được công cụ `data/fetch_external.py` tự động quét và khai phá từ các khối công thức toán trong tutorial Overleaf và Obsidian KB.
- **Mục đích:** Chỉ lưu trữ để người quản trị duyệt thủ công (manual review only); tuyệt đối không tự động nạp vào tập huấn luyện hay kiểm thử.

---

## 2. Điều khoản cấp phép (Licensing Terms)

| Nguồn dữ liệu / Thành phần | Bản quyền & Giấy phép | Điều khoản sử dụng |
|---|---|---|
| **PhoATIS (VinAI Research)** | MIT License / CC BY-NC 4.0 | Cho phép sử dụng cho mục đích nghiên cứu, học thuật và phát triển phi thương mại. Bắt buộc trích dẫn công trình gốc của VinAI Research (Dao et al., 2021). |
| **Overleaf LaTeX Tutorial** | Bản quyền thuộc Overleaf / Digital Science | Sử dụng theo nguyên tắc sử dụng hợp lý (Fair Use) cho mục đích giáo dục và nghiên cứu; không phân phối lại nguyên bản dưới dạng ấn phẩm thương mại. |
| **Obsidian KB (`knowledge_base(1).txt`)** | Thuộc quyền sở hữu của người dùng | Dữ liệu nội bộ dự án, được người dùng cấp quyền biên soạn và tái cấu trúc cho trợ lý LaTeX. |
| **Dữ liệu câu tổng hợp nội bộ (`train/val/test`)** | Giấy phép dự án mở (MIT / Apache-2.0) | Tự do sử dụng trong khuôn khổ dự án trợ lý tri thức tiếng Việt. |

---

## 3. Quy tắc phân chia tập dữ liệu (Split Rules & Data Governance)

Nhằm đảm bảo tính khách quan khoa học và tuân thủ các quy tắc trung thực trong `AGENTS.md`, quy trình phân chia dữ liệu áp dụng các ràng buộc nghiêm ngặt sau:

### 3.1. Phân bổ kích thước và vai trò
- **TRAIN (`data/train.csv`):** 10 câu mẫu / intent (tổng hợp có seed tái lập). Dùng để huấn luyện bộ phân loại ý định PhoBERT.
- **VAL (`data/val.csv`):** 5 câu mẫu / intent (cách diễn đạt độc lập với TRAIN). Dùng để tinh chỉnh siêu tham số và xác định ngưỡng tin cậy $\tau$.
- **TEST (`data/test.csv`):** test do AI soạn, chưa phải benchmark cuối; 940 dòng, hiện không có OUT_OF_SCOPE. Không sửa tệp này. Báo cáo kết quả trên file này phải ghi “test do AI soạn, chưa phải benchmark cuối”. Benchmark primary sẽ là `data/test_owner.csv` sau khi chủ sở hữu tự viết và chạy `python data/verify_owner_test.py`.

### 3.2. Tập kiểm thử ngoài phạm vi riêng (Out-of-Scope Sampling)

`data/test_oos.csv` được lập từ `data/external/phoatis/out_of_scope_test.csv` (các hàng PhoATIS test và casual test đã có provenance), sau khi loại bỏ trùng với train/val. OOS metrics phải dùng file này; không trộn thêm vào `data/test.csv`.

### 3.3. Quy tắc lấy mẫu câu ngoài phạm vi (Out-of-Scope Sampling)
- Dữ liệu OOS được lấy mẫu ngẫu nhiên có hạt giống cố định (`seed=42`) thông qua hàm tập trung `core.config.set_seed`.
- **Nguồn OOS:**
  1. Phân vùng tương ứng từ PhoATIS syllable `seq.in`:
     - OOS TRAIN lấy mẫu từ `data/external/phoatis/train/seq.in`.
     - OOS VAL lấy mẫu từ `data/external/phoatis/dev/seq.in`.
     - OOS TEST lấy mẫu từ `data/external/phoatis/test/seq.in`.
  2. Bổ sung các câu giao tiếp thường nhật (casual additions) bằng tiếng Việt (chào hỏi, hỏi giờ, thời tiết, thường thức đời sống). Ba tập câu giao tiếp này được thiết kế tách biệt hoàn toàn giữa Train, Val, và Test.

### 3.4. Ràng buộc bất biến Honesty Rules (AGENTS.md)
1. **Honesty Rule #3 (Bất biến tập Test):**
   - Tuyệt đối không tinh chỉnh trên tập Test.
   - Không tự ý chỉnh sửa, tự sinh hay thêm bớt dữ liệu trực tiếp vào `data/test.csv`.
   - Mã băm SHA-256 của `data/test.csv` được theo dõi và xác thực trước mỗi lần đánh giá cuối cùng.
2. **Honesty Rule #4 (Không trùng lặp xuyên tập):**
   - Ba tập dữ liệu Train, Val và Test **tuyệt đối không chia sẻ bất kỳ câu trùng lặp nào** (so khớp không phân biệt chữ hoa, chữ thường và khoảng trắng).
   - Quy trình lấy mẫu ngoại phạm vi OOS luôn chạy thuật toán kiểm tra giao tập (`intersection == 0`) trên toàn bộ các câu đã chuẩn hóa.
3. **Honesty Rule #5 & #7 (Minh bạch số liệu):**
   - Mọi số liệu trong các báo cáo phải xuất phát từ script thực thi thực tế, không bịa đặt số liệu.
   - Bảng ứng viên `data/mined_candidates.csv` luôn giữ cờ `pending_manual_review` và phải được người dùng thẩm định trước khi đưa vào bất kỳ quy trình đào tạo nào.

### 3.5. Benchmark do chủ sở hữu viết

`data/test_owner_template.csv` có ba dòng mẫu trống cho mỗi category; điền theo `docs/OWNER_TEST_GUIDE.md`. Khi `data/test_owner.csv` tồn tại, xác thực schema, label/category, duplicate và overlap trước khi xem đây là primary test. SHA-256 hợp lệ được ghi trong `PROGRESS.md`.
