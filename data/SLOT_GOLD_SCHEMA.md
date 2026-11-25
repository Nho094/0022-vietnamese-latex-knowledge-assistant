# Định dạng `slot_gold.csv`

Đây là hướng dẫn schema, không chứa ví dụ được dùng làm validation/test. Các câu ở hai split này cần do chủ dự án viết tay.

CSV UTF-8 có các cột:

| Cột | Định dạng |
|---|---|
| `split` | `val` hoặc `test` |
| `text` | Câu gốc, offsets tính trên chuỗi này theo Unicode code point |
| `intent` | Mã intent có trong `data/intents.json` |
| `spans` | JSON array: `[ {"start": 10, "end": 11, "slot": "a"} ]`; start inclusive, end exclusive |

Mỗi span phải nằm trong `text`, tên slot phải khai báo cho intent đó, và các câu phải khác nhau giữa train/val/test sau NFC, lowercase và gộp khoảng trắng. Tập train BIO được sinh từ các frame KA-02; không chép câu val/test vào train. Sau khi tạo file, chạy:

```bash
python data/make_bio.py --synthetic-train
python train/joint_intent_slot.py --data data/slot_gold_bio.csv
python train/bijoint_nlu.py --data data/slot_gold_bio.csv
```

`data/make_bio.py` sẽ từ chối nếu không sinh đủ câu train khác biệt; trainer yêu cầu có cả `train`, `val` và `test`.
