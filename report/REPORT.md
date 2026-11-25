# Báo cáo Knowledge Assistant LaTeX

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

[C07] Nguồn dữ liệu: Learn LaTeX in 30 minutes - Overleaf, Online LaTeX Editor.pdf, knowledge_base(1).txt.

| Phân vùng | Số mẫu |
|---|---:|
| train | 3825 |
| val | 967 |
| test | 940 |

## 5. Thực nghiệm

[C08] Artifact được đọc từ report JSON/CSV. Smoke artifact nằm trong `report/smoke/` bị bỏ qua; artifact có test hash khác PROGRESS.md bị loại.

| Mô hình | Intent accuracy mean ± std | Slot F1 mean ± std | Exact-slot mean ± std | Sentence accuracy mean ± std |
|---|---:|---:|---:|---:|
| Chưa có kết quả KA-11/KA-12 hợp lệ | — | — | — | — |

- `baseline_category_results.json`: `{"data_split_counts": {"test": 940, "train": 3825, "val": 967}, "fuzzy_matching": {"confusion_matrix_figure": "report/figures/baseline_category_fuzzy.png", "method": "rapidfuzz_token_set_ratio", "test_bootstrap_ci_95": {"accuracy": [0.3893617021276596, 0.45], "macro_f1": [0.3647490870435433, 0.45096192172650945], "weighted_f1": [0.32799593650867814, 0.396793771316119]}, "test_metrics": {"accuracy": 0.41914893617021276, "macro_f1": 0.41398109820092, "per_class": {"math_shared": {"f1": 0.228855721`
- `baseline_intent_results.json`: `{"data_split_counts": {"test": 940, "train": 3825, "val": 967}, "fuzzy_matching": {"confusion_matrix_figure": "report/figures/baseline_intent_fuzzy.png", "method": "rapidfuzz_token_set_ratio", "test_bootstrap_ci_95": {"accuracy": [0.13191489361702127, 0.1776595744680851], "macro_f1": [0.12086198227272651, 0.13604735592208703], "weighted_f1": [0.1118145970218902, 0.15456932369763843]}, "test_metrics": {"accuracy": 0.15425531914893617, "macro_f1": 0.13100199951119731, "per_class": {"LATEX_BANG_CO_`
- `baseline_results.json`: `{"category": {"data_split_counts": {"test": 940, "train": 3825, "val": 967}, "fuzzy_matching": {"confusion_matrix_figure": "report/figures/baseline_category_fuzzy.png", "method": "rapidfuzz_token_set_ratio", "test_bootstrap_ci_95": {"accuracy": [0.3893617021276596, 0.45], "macro_f1": [0.3647490870435433, 0.45096192172650945], "weighted_f1": [0.32799593650867814, 0.396793771316119]}, "test_metrics": {"accuracy": 0.41914893617021276, "macro_f1": 0.41398109820092, "per_class": {"math_shared": {"f1"`
- `split_shift.json`: `{"most_frequent_sentence": "biểu thức cần hiển thị bằng mathjax.", "source": "data/train.csv, data/val.csv, data/test.csv (test is ai_draft)", "splits": {"test": {"count": 940, "label_count": {"LATEX_BANG_CO_UONG_KE": 5, "LATEX_BANG_TABULAR_CO_BAN": 5, "LATEX_CAN_GIUA_NOI_DUNG": 5, "LATEX_CHEN_HINH_ANH": 5, "LATEX_CHI_SO_DUOI_VA_SO_MU": 5, "LATEX_CHUONG_VA_MUC": 5, "LATEX_CHU_AM": 5, "LATEX_CHU_NGHIENG": 5, "LATEX_CHU_THICH_HINH": 5, "LATEX_CHU_THICH_MA_LATEX": 5, "LATEX_DANH_SACH_CO_THU_TU": 5,`
- `ablation.csv`: `{"columns": ["mode", "accuracy", "macro-F1", "in_scope_accuracy", "OOS_rejection_rate", "false_rejection_rate", "automatic_answer_share", "need_choice_rate", "top3_hit_rate", "median_latency_ms", "accuracy_ci95", "macro_f1_ci95", "status"], "rows": 6}`
- `data_report.json`: `{"exact_duplicate_counts": {"train_test": 0, "train_val": 0, "val_test": 0}, "intent_count": 189, "source_files": ["Learn LaTeX in 30 minutes - Overleaf, Online LaTeX Editor.pdf", "knowledge_base(1).txt"], "split_counts": {"test": 940, "train": 3825, "val": 967}}`

## 6. Phân tích lỗi

[C09] McNemar dùng hiệu chỉnh Yates; chỉ so sánh dự đoán trên cùng câu gold và cùng seed.

| So sánh | b (A đúng, B sai) | c (A sai, B đúng) | Chi-square | p-value |
|---|---:|---:|---:|---:|
| Chưa có cặp dự đoán hợp lệ | — | — | — | — |

## 7. Hạn chế

[C10] Slot filler hiện là regex và các mô hình sequence không có CRF. LaTeX chỉ được gọi là đã biên dịch khi lệnh compiler chạy thành công.
[C11] PhoATIS được dùng trong OOS train/val và test riêng; `data/slot_gold.csv` chưa có nên chưa thể báo cáo kết quả slot gold.

`data/test.csv` là test do AI soạn, chưa phải benchmark cuối. Kết quả trên đó là sơ bộ; kết luận cuối cần `data/test_owner.csv` do owner viết. PhoATIS có 65/27 OOS rows trong train/val và 27 rows riêng trong `data/test_oos.csv`.
LaTeX: static check only; chưa xác nhận biên dịch bằng TeX Live.

- VAL/TEST được tạo theo yêu cầu mới của chủ sở hữu; đây là bản nháp do AI soạn, cần chủ sở hữu rà soát trước khi dùng làm benchmark cuối cùng. Điều này là ngoại lệ so với KA-02 gốc vốn yêu cầu owner-written TEST và cấm LLM viết TEST.
- Obsidian và MathJax đã được đưa vào intents chung bằng schema mở rộng: templates.obsidian và templates.mathjax; python/cpp giữ rỗng.
- MathJax dùng chung được gắn category math_shared; intent LaTeX và Obsidian vẫn tách domain để tránh nhầm cú pháp đầu ra.
- data/test.csv là test do AI soạn, chưa phải benchmark cuối; không chỉnh sửa tệp này.
- PhoATIS có mặt trong nguồn OOS train/val và tập OOS riêng; test.csv AI không chứa OUT_OF_SCOPE.
- Chưa có `data/slot_gold.csv`; chưa thể chạy đánh giá gold spans.
- Cache PhoBERT chưa có file trọng số; trainer chỉ được kiểm tra bằng encoder nhỏ trong unit test.
- BLOCKED: `pdflatex` chưa có; LaTeX hiện chỉ được kiểm tra tĩnh, chưa compile.

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
| [C01] PhoBERT dùng phân đoạn từ tiếng Việt và BPE trước fine-tuning. | `core/preprocess.py`, `core/config.py`, `Prompt_Pack_Knowledge_Assistant_LaTeX.md` §13 |
| [C02] JointBERT dự đoán intent từ biểu diễn token đầu và cộng loss intent với loss slot trung bình theo từ. | `train/joint_intent_slot.py` (`JointIntentSlotModel.forward`) |
| [C03] BiJoint tạo attention theo intent, ghép vector attention với biểu diễn từ, rồi tổng các xác suất slot để dự đoán intent cuối. | `train/bijoint_nlu.py` (`BiJointNLU.forward`) |
| [C04] SBERT lập chỉ mục embedding và xếp hạng bằng cosine similarity; ba loss bi-encoder được triển khai trong code huấn luyện. | `core/retrieval.py`, `train/losses.py`, `train/triplet_sbert.py` |
| [C05] Scenario bank biểu diễn intent bằng chuỗi scenario và aliases theo hướng tiếp cận lấy cảm hứng từ RAF. | `core/scenario.py`, `core/retrieval.py`, `AGENTS.md` (Honesty rule 1) |
| [C06] Trợ lý định tuyến yêu cầu qua classifier/retriever, slot regex và LaTeX template renderer. | `core/pipeline.py` (`Pipeline.predict`), `core/slots.py`, `core/latex_tools.py` |
| [C07] Dữ liệu được lấy từ các nguồn và có kích thước split nêu trong bảng. | `data/data_report.json` (`source_files`, `split_counts`) |
| [C08] Kết quả chỉ được đưa vào report nếu artifact dùng đúng test hash hiện tại và không phải smoke hoặc partial-test run. | `report/build_report.py` (`_json_files`) |
| [C09] Kiểm định intent dùng McNemar với hiệu chỉnh Yates trên cùng câu và seed. | `train/nlu_training.py` (`compare_intent_predictions`), `train/evaluate.py` |
| [C10] Slot filler hiện là regex; CRF chưa được triển khai. | `core/slots.py`, `train/joint_intent_slot.py`, `train/bijoint_nlu.py` |
| [C11] PhoATIS OOS có trong train/val và test riêng; chưa có owner gold slot spans. | `data/data_report.json`, `data/test_oos.csv`, `data/SLOT_GOLD_SCHEMA.md` |
| [C12] LaTeX hiện chỉ qua kiểm tra tĩnh; compile chưa được xác nhận trong lần build này. | `core/latex_tools.py`, `cli/ka.py`, `AGENTS.md` (Honesty rule 7) |
