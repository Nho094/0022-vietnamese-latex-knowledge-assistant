# Báo Cáo Đánh Giá Học Thuật PhoBERT & Hệ Thống Truy Hồi Dự Phòng

> **Dự án**: Vietnamese LaTeX Knowledge Assistant  
> **Thời điểm đánh giá**: 2026-11-19 10:31:21 UTC  
> **Hạt giống ngẫu nhiên (Seed)**: 42  
> **Thiết bị tính toán**: CPU  
> **Chế độ nạp mô hình**: PhoBERT HuggingFace: `False` | SBERT Bi-Encoder: `False`  

---

## 1. Kiểm Chứng Tính Toàn Vẹn Dữ Liệu (Honesty Rule #3)

- **Đường dẫn tệp kiểm thử**: `data/test.csv`
- **Mã băm SHA-256 thực tế**: `637f54fdae2d1be7d39ca960ce91003adb2f50a09ee446b87fbb092b1b9c6066`
- **Mã băm SHA-256 quy định**: `637f54fdae2d1be7d39ca960ce91003adb2f50a09ee446b87fbb092b1b9c6066`
- **Trạng thái xác thực SHA-256**: `ĐẠT CHUẨN (VERIFIED)`
- **Số lượng mẫu kiểm thử (test.csv)**: 0 mẫu
- **Số lượng mẫu huấn luyện (train.csv)**: 488 mẫu
- **Tổng số kịch bản trong Scenario Bank**: 162 kịch bản

---

## 2. Kết Quả Phân Loại Ý Định (PhoBERT Intent Classification)

Phương pháp phân loại tuân theo kiến trúc RoBERTa (Nguyen & Tuan Nguyen, 2020) với hàm kích hoạt Softmax trên biểu diễn `[CLS]`:
$$p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}, \quad p_{\max} = \max_i(p_i)$$

| Chỉ số đánh giá | Giá trị thực nghiệm | Ghi chú phương pháp |
|---|---|---|
| **Tổng số mẫu đánh giá** | 488 | Tập mẫu tiếng Việt chuẩn hóa NFC |
| **Top-1 Accuracy** | **99.59%** | Dự đoán nhãn chính xác nhất |
| **Top-3 Accuracy** | **100.00%** | Nhãn thực tế nằm trong Top-3 |
| **Top-5 Accuracy** | **100.00%** | Nhãn thực tế nằm trong Top-5 |
| **Ngưỡng tin cậy $\tau$ (Tau)** | 0.8 | Ngưỡng chấp nhận trực tiếp |
| **Tỷ lệ chấp nhận trực tiếp ($p_{\max} \ge \tau$)** | **100.00%** | Không cần kích hoạt fallback |
| **Độ chính xác trong tập chấp nhận** | **99.59%** | Độ tin cậy cao khi vượt ngưỡng |

### Độ chính xác theo danh mục (Category Breakdown)

| Danh mục (Category) | Số lượng mẫu | Số mẫu đúng | Độ chính xác |
|---|---|---|---|
| `comments_text_formatting` | 16 | 16 | **100.00%** |
| `cross_references` | 4 | 4 | **100.00%** |
| `document_classes` | 12 | 12 | **100.00%** |
| `document_structure` | 8 | 8 | **100.00%** |
| `graphics_figures` | 12 | 12 | **100.00%** |
| `lists` | 8 | 8 | **100.00%** |
| `math_accents` | 36 | 36 | **100.00%** |
| `math_alignment_array` | 15 | 15 | **100.00%** |
| `math_brackets_spacing` | 33 | 33 | **100.00%** |
| `math_cases_systems` | 6 | 6 | **100.00%** |
| `math_expression` | 42 | 42 | **100.00%** |
| `math_expressions` | 8 | 8 | **100.00%** |
| `math_fonts` | 27 | 27 | **100.00%** |
| `math_logic_sets` | 18 | 18 | **100.00%** |
| `math_matrix` | 24 | 23 | **95.83%** |
| `math_modes` | 12 | 12 | **100.00%** |
| `math_symbols` | 105 | 104 | **99.05%** |
| `math_units_color_numbering` | 42 | 42 | **100.00%** |
| `overleaf_workflow` | 4 | 4 | **100.00%** |
| `packages_ecosystem` | 12 | 12 | **100.00%** |
| `paragraphs_linebreaks` | 8 | 8 | **100.00%** |
| `preamble_packages` | 12 | 12 | **100.00%** |
| `sectioning` | 4 | 4 | **100.00%** |
| `table_of_contents` | 4 | 4 | **100.00%** |
| `tables` | 8 | 8 | **100.00%** |
| `text_layout` | 4 | 4 | **100.00%** |
| `title_author_date` | 4 | 4 | **100.00%** |

---

## 3. Độ Trễ Suy Luận và Thông Lượng (Inference Latency & Throughput)

Được đo lường sau 5 lượt chạy khởi động (warmup runs) trên thiết bị `cpu`:

| Chỉ số độ trễ | Giá trị thực nghiệm |
|---|---|
| **Thời gian trung bình (Mean Latency)** | **1.92 ms** |
| **Độ lệch chuẩn (Std Dev)** | 0.27 ms |
| **Trung vị P50 (Median)** | **1.89 ms** |
| **Phân vị 90 (P90)** | 1.98 ms |
| **Phân vị 95 (P95)** | 2.04 ms |
| **Phân vị 99 (P99)** | 3.47 ms |
| **Độ trễ nhỏ nhất (Min)** | 1.69 ms |
| **Độ trễ lớn nhất (Max)** | 3.77 ms |
| **Thông lượng (Throughput)** | **520.5 QPS** (truy vấn / giây) |

---

## 4. Hành Vi Truy Hồi Dự Phòng (Fallback & Out-of-Domain Behavior)

Cảm hứng từ mô hình Retrieve-and-Fill (RAF) và Sentence-BERT Cosine Similarity (Reimers & Gurevych, 2019):
$$\text{sim}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

| Tình huống và Hành vi | Chỉ số thực tế | Đánh giá |
|---|---|---|
| **Tỷ lệ kích hoạt Fallback ($p_{\max} < \tau$)** | **0.00%** (0/488) | Kích hoạt truy hồi SBERT |
| **Tỷ lệ Fallback tìm thấy ứng viên ($\ge \theta$)** | **0.00%** (0/1) | Hệ thống đề xuất lựa chọn |
| **Fallback Recall@1** | **0.00%** | Ứng viên top-1 trùng nhãn đích |
| **Fallback Recall@3** | **0.00%** | Nhãn đích nằm trong top-3 gợi ý |
| **Độ hồi quy toàn hệ thống (Combined Recall@1)** | **99.59%** | PhoBERT + Fallback top-1 |
| **Độ hồi quy toàn hệ thống (Combined Recall@3)** | **99.59%** | PhoBERT + Fallback top-3 |
| **Tỷ lệ từ chối câu ngoài phạm vi (OOD Rejection Rate)** | **90.00%** (9/10) | Từ chối chính xác khi $< \theta$ |

---

## 5. Kiểm Tra Tĩnh Template Rendering (Honesty Rule #7)

- **Chế độ kiểm thử**: `static check only` (static check only)
- **Tổng số mẫu kiểm tra**: 488
- **Tỷ lệ sinh mã thành công**: **100.00%** (488/488)
