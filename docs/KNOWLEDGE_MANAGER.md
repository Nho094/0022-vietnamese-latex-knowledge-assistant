# Local Knowledge Manager

Khởi chạy giao diện và API trên máy local:

```bash
AI/bin/python api/server.py
```

Mở `http://127.0.0.1:8000/ui/`. Trang tổng quan đọc intent definitions qua management API; knowledge corpora được hiển thị riêng và không thể sửa từ giao diện.

## Tạo và publish definition

Definition Builder khai báo metadata, JSON Schema, field policies, JSONata transform và Jinja2 render templates. Ở bước cuối:

1. Chọn **Validate definition** để kiểm tra Contract v2.
2. Nhập object JSON có các field cần thiết rồi chọn **Preview output**. Definition phải preview thành công trước khi publish.
3. Chọn **Publish definition**. Nếu intent đã có, xác nhận ghi đè. Giao diện publish qua API rồi reload registry.
4. Dùng **Thử assistant** để gửi câu hỏi qua `/ask`; kết quả phụ thuộc model/runtime đang sẵn sàng.

## Import, export và phục hồi

- Export tải toàn bộ definition bank dưới dạng JSON array.
- Import nhận JSON array hoặc object có key `definitions`. API validate cả batch trước khi hỏi xác nhận; từng definition sau đó được publish tuần tự để cập nhật SBERT index. Nếu một publish lỗi giữa chừng, giao diện báo số lượng đã ghi và giữ nguyên những definition đã publish thành công; batch hiện chưa có transaction toàn phần.
- Reload đọc lại bank trên đĩa và đồng bộ SBERT index.
- Rollback phục hồi backup gần nhất trước phiên bản hiện tại, kể cả khi server đã khởi động lại. Hãy xác nhận vì thao tác thay toàn bộ definition bank.

Các thao tác ghi chỉ đi qua API và registry; frontend không sửa trực tiếp file knowledge. Import/publish chỉ chấp nhận Contract v2.
