# VS Code Extension: cài đặt và giới hạn hiện tại

## Dùng bản VSIX

Mở **Extensions → … → Install from VSIX…** trong VS Code và chọn file `.vsix`. Extension tự tìm Python trong `PATH` hoặc dùng đường dẫn ở `vietnameseKnowledgeAssistant.pythonPath`. Backend chỉ mở trên `127.0.0.1`, dùng token ngẫu nhiên cho mỗi phiên và dừng tiến trình do Extension tạo khi VS Code đóng. Lần chạy đầu, Extension chép `intents.json` đóng gói vào `globalStorage` nếu chưa có; các lần nâng cấp không ghi đè dữ liệu người dùng.

VSIX chứa mã Python của backend và dữ liệu runtime cần thiết, nhưng không chứa Python, thư viện Python, cache/checkpoint mô hình hoặc môi trường `AI/`. Trên máy sạch, người dùng cần có một Python đã cài các dependency của dự án và cấu hình đường dẫn trong Settings. Extension không chạy lệnh cài thư viện, không tải mô hình và không gửi dữ liệu ra Internet. Nếu thiếu dependency, Settings/Output sẽ báo trạng thái thay vì giả lập rằng mô hình sẵn sàng. Dự án hiện chưa có gói cài backend tự chứa; vì vậy cài mỗi VSIX trên máy sạch chưa đủ để có đầy đủ chức năng.

## Model runtime

Extension không cần nhúng checkpoint vào VSIX. Theo mặc định, nó tìm dưới `<workspace>/models`:

```text
models/
  checkpoints/phobert_classifier/   # PhoBERT đã fine-tune: config, weights, label map, fingerprint
  hf_cache/phobert-base-v2/         # tokenizer (vocab.txt, bpe.codes), hoặc snapshot trong HF hub cache
  hf_cache/vietnamese-bi-encoder/   # SentenceTransformer model, hoặc snapshot trong HF hub cache
  index_*.npz                      # index retrieval (nếu chưa có, sẽ tạo khi chạy lần đầu)
```

Có thể đặt đường dẫn khác trong Settings → `Vietnamese Knowledge Assistant: Model Root`, hoặc cấu hình JSON:

```json
{ "vietnameseKnowledgeAssistant.modelRoot": "/duong/dan/toi/models" }
```

Đường dẫn tương đối được tính từ workspace đang mở. Nếu không mở workspace, mặc định là `globalStorage/models`. Các biến `KA_MODEL_ROOT`, `KA_CHECKPOINT_DIR`, `KA_SBERT_DIR`, `KA_HF_CACHE_DIR` và `KA_INDEX_CACHE_DIR` cũng có thể dùng khi chạy Python ngoài Extension. Runtime mặc định offline; nó tìm cache theo `modelRoot` trước, rồi cache Hugging Face chuẩn (`HF_HUB_CACHE`, `HF_HOME`, `~/.cache/huggingface`).

Settings phân biệt checkpoint PhoBERT, retrieval model và index retrieval. Model retrieval có mà index chưa có vẫn dùng được: backend build index lần đầu. Khi model thiếu, backend/Knowledge Manager vẫn chạy nhưng `/ask` trả trạng thái `unknown` với hướng dẫn artifact cần có. Trạng thái `ready` chỉ hiện khi cả classifier và retrieval model đều hợp lệ. PhoBERT cần `config.json`, weights `.bin`/`.safetensors`, label map, fingerprint khớp và tokenizer; model retrieval cần modules/config, weights và tokenizer.

Knowledge Builder thủ công không phụ thuộc AI; đề xuất tự động bằng AI chưa được tích hợp và được báo rõ trong giao diện.

## Phát triển và đóng gói

Môi trường phát triển cần Node.js, npm và Python hiện có của dự án:

```sh
npm install
npm run typecheck
npm run build
npm run test:extension
npm run package
```

Lệnh cuối tạo `dist/vietnamese-knowledge-assistant-0.1.0.vsix`. Sau khi VSIX đã được tạo thành công, cài bằng:

```sh
code --install-extension "/đường/dẫn/tuyệt/đối/tới/dist/vietnamese-knowledge-assistant-0.1.0.vsix"
```

Mở VS Code, chạy lệnh **Vietnamese Knowledge Assistant: Mở bảng điều khiển** từ Command Palette. Extension tự khởi động backend nếu tìm thấy Python hợp lệ. Nếu cần chọn interpreter, mở Settings và đặt `vietnameseKnowledgeAssistant.pythonPath` tới Python có dependency của dự án.

Kiểm tra API/backend bằng:

```sh
AI/bin/python -m pytest tests/test_extension_workflows.py tests/test_knowledge_api.py tests/test_definition_registry.py -q
AI/bin/python -m ruff check api/server.py core/definitions/registry.py tests/test_extension_workflows.py
```

Build Webview cần truy cập npm registry nếu các package chưa có trong npm cache. Không sử dụng dashboard mẫu như runtime; chỉ một số UI primitive/pattern của nó được điều chỉnh vào Webview. Attribution và MIT notice đi kèm trong `extension/ATTRIBUTION.md`.

## Kết quả kiểm tra hiện tại

Lần kiểm tra runtime gần nhất: `npm run typecheck`, `npm run build`, `npm run test:extension` (3/3), `npm run package`, Ruff, validator, test-set verifier và full pytest đều đạt. Full pytest: 578 passed, 1 skipped, 7 warnings. VSIX thực tế: `dist/vietnamese-knowledge-assistant-0.1.0.vsix` (62 file, khoảng 366 KB). Archive hợp lệ, có Extension Host, Webview JS/CSS production, backend, intent bank và knowledge corpora; không chứa `data/test.csv`, model weights, Python cache, `node_modules` hay Python runtime. VS Code CLI cài VSIX vào profile tạm thành công.

Runtime status trên môi trường audit: PhoBERT fine-tuned checkpoint không có; SentenceTransformer được tìm thấy trong `~/.cache/huggingface`, index hash hợp lệ, nên backend báo `degraded` và chọn retrieval-only. Một câu hỏi CLI ở chế độ offline đã nạp 199 weight tensors từ cache ngoài repository và trả về `need_choice`, xác nhận model retrieval bên ngoài VSIX có thể được dùng. Đây không phải kết quả đánh giá chất lượng mô hình. Chưa chạy được HTTP/VS Code UI smoke trong sandbox; cài vào profile đang dùng bị `EROFS` vì `~/.config/Code` read-only. VSCE còn cảnh báo chưa có LICENSE ở thư mục gốc.

Lệnh cài vào profile VS Code thông thường:

```sh
code --install-extension "/đường/dẫn/tuyệt/đối/tới/dự-án/dist/vietnamese-knowledge-assistant-0.1.0.vsix"
```

VSIX không độc lập: máy người dùng cần Python cùng dependencies; PhoBERT/retrieval chỉ hoạt động khi artifact có sẵn ở workspace/modelRoot hoặc HF cache. Interactive VS Code, socket backend, Builder qua UI, và khởi động với một checkpoint PhoBERT production chưa được xác minh trong sandbox. Các test Extension kiểm tra runtime path, CSP assets và directive nhưng chưa có browser automation.
