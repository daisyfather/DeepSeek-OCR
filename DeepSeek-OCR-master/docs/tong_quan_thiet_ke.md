# Tài liệu Thiết kế Overview DeepSeek-OCR

## Mục tiêu sản phẩm
DeepSeek-OCR cung cấp một mô hình thị giác-ngôn ngữ tối ưu cho bài toán nhận dạng văn bản, trích xuất bố cục và mô tả tài liệu phức tạp.
Hệ thống được thiết kế để xử lý linh hoạt từ ảnh đơn, tài liệu đa trang đến lô dữ liệu lớn, đồng thời duy trì tính chính xác và tốc độ suy luận cao.

## Thành phần chính
| Nhóm thành phần | Vai trò | Hiện thực trong kho mã |
|-----------------|---------|-------------------------|
| **Mô hình ngôn ngữ đa phương thức** | Kết hợp bộ mã hóa ảnh và LLM để giải mã nội dung tài liệu | `DeepSeek-OCR-vllm/deepseek_ocr.py` triển khai lớp `DeepseekOCRForCausalLM` tùy biến cho vLLM.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py†L257-L329】 |
| **Bộ xử lý ảnh & tokenizer** | Chuẩn hóa ảnh, sinh token `<image>` tương ứng và liên kết với prompt | `DeepSeek-OCR-vllm/process/image_process.py` và lớp `DeepseekOCRProcessor` cho phép chuẩn bị tensor đầu vào đồng bộ.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/image_process.py†L45-L188】 |
| **Cấu hình suy luận** | Tập trung các tham số vận hành (kích thước ảnh, chế độ crop, template prompt) | `DeepSeek-OCR-vllm/config.py` chứa các profile Tiny → Gundam và đường dẫn tài nguyên.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py†L1-L42】 |
| **Trình điều phối suy luận** | Khởi tạo mô hình, nhận dữ liệu đầu vào, phát luồng token và ghi kết quả | Các script `run_dpsk_ocr_image.py`, `run_dpsk_ocr_pdf.py`, `run_dpsk_ocr_eval_batch.py` thực thi workflow cụ thể.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py†L146-L303】【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py†L84-L170】 |

## Trải nghiệm triển khai
1. **Thiết lập môi trường**: sử dụng CUDA 11.8 và PyTorch 2.6.0, kèm bản vLLM tương thích theo hướng dẫn trong README chính.【F:README.md†L42-L81】
2. **Chọn chế độ vận hành**: chỉnh sửa `config.py` để tùy chọn kích thước ảnh, đường dẫn dữ liệu và cấu hình tokenizer.
3. **Chạy suy luận**:
   - Với vLLM: `run_dpsk_ocr_image.py` cho luồng ảnh đơn, `run_dpsk_ocr_pdf.py` cho PDF.
   - Với Transformers: tạo instance thông qua `AutoModel` để tái sử dụng mô hình trong pipeline có sẵn.【F:README.md†L83-L132】
4. **Theo dõi kết quả**: script xuất ra markdown, ảnh crop, và log chi tiết phục vụ hậu kiểm.

## Khả năng mở rộng
- **Điều chỉnh chi phí**: các profile Tiny → Large → Gundam giúp cân bằng tài nguyên GPU và độ chính xác.【F:README.md†L133-L150】
- **Tích hợp hệ thống lớn**: mô hình được đăng ký với vLLM nên tương thích với cơ chế batching, prefix caching, và logits processor mở rộng.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py†L150-L255】【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py†L177-L199】
- **Đa dạng bài toán**: prompt có thể chuyển đổi giữa OCR thuần, trích xuất bảng, mô tả bố cục nhờ `multi_modal_data` và template linh hoạt.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py†L165-L215】

## Lộ trình vận hành đề xuất
1. Đánh giá yêu cầu độ chính xác → chọn profile phù hợp.
2. Thiết lập prompt mẫu cho từng loại tài liệu.
3. Tối ưu hạ tầng vLLM: bật streaming nếu cần phản hồi nhanh hoặc batching cho throughput cao.
4. Giám sát log và số liệu n-gram để tinh chỉnh bộ lọc lặp lại (ngăn lỗi bảng, HTML) trước khi đưa vào sản phẩm.
