# Tài liệu Kiến trúc kỹ thuật DeepSeek-OCR

## Tổng quan kiến trúc
Hệ thống DeepSeek-OCR được xây dựng như một dịch vụ suy luận đa phương thức gồm ba lớp:
1. **Lớp giao diện suy luận**: các script vLLM/Transformers nhận yêu cầu, ánh xạ thành cấu trúc chuẩn và trả kết quả.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py†L146-L303】【F:README.md†L83-L132】
2. **Lớp xử lý dữ liệu**: `DeepseekOCRProcessor` chịu trách nhiệm chuẩn hóa ảnh, cắt lát động, và tạo token `<image>` tương thích với prompt.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/image_process.py†L45-L329】
3. **Lớp mô hình hóa**: `DeepseekOCRForCausalLM` hợp nhất encoder ảnh (SAM + CLIP-L) và ngôn ngữ, trả về chuỗi token văn bản giàu ngữ cảnh.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py†L257-L465】

## Luồng xử lý ảnh chi tiết
1. **Tiền xử lý**
   - Ảnh được chuyển đổi sang RGB, chuẩn hóa kích thước dựa trên profile cấu hình và tạo thêm bản tile nếu bật `crop_mode` động.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/image_process.py†L330-L435】
   - Metadata về số tile, kích thước và vị trí được gắn vào `processor_config` để lớp mô hình sử dụng khi ánh xạ token.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py†L50-L147】
2. **Mã hóa đa tầm nhìn**
   - Hàm `_pixel_values_to_embedding` đẩy các bản tile qua encoder SAM, sau đó qua CLIP-L để lấy đặc trưng không gian.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py†L364-L438】
   - Bộ tách `vision_sep_embed` chèn token phân cách giữa tile toàn cục và tile cục bộ, giúp LLM hiểu ngữ cảnh bố cục.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py†L303-L409】
3. **Hợp nhất với dòng văn bản**
   - Các embedding ảnh được chèn vào vị trí `<image>` trong chuỗi token văn bản, sau đó decoder sinh đầu ra theo sampling param được cấu hình.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py†L498-L553】【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py†L177-L215】

## Quản lý tài nguyên và cấu hình
- `config.py` định nghĩa kích thước ảnh, batch size, chế độ streaming và đường dẫn dữ liệu, cho phép triển khai nhanh ở nhiều môi trường GPU.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py†L1-L42】
- Các script suy luận nhận tham số đường dòng lệnh, hỗ trợ thiết lập đường dẫn đầu vào/đầu ra, bật lưu crop và bật chế độ đánh giá lô.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py†L1-L83】【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_eval_batch.py†L1-L116】
- N-gram logits processor (`NGramPerReqLogitsProcessor`) giúp tránh lặp token, đặc biệt với thẻ bảng HTML, đảm bảo đầu ra sạch.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py†L147-L199】

## Tương thích hệ sinh thái
- DeepSeek-OCR được đăng ký như mô hình tùy biến trong vLLM, do đó kế thừa cơ chế quản lý cache, phân bổ GPU và xử lý truy vấn song song của vLLM.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py†L150-L255】
- Bộ trọng số Hugging Face được ánh xạ lại để khớp với cấu trúc module của vLLM, đảm bảo có thể nâng cấp hoặc thay thế từng phần mà không phá vỡ tương thích ngược.【F:DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py†L564-L582】
- Pipeline Transformers sử dụng cùng checkpoint nên dễ dàng chia sẻ trong các hệ thống không sử dụng vLLM, tăng tính linh hoạt triển khai.【F:README.md†L83-L132】

## Định hướng nâng cấp
1. **Tối ưu hóa chi phí**: bổ sung profile trung gian giữa Large và Gundam để tiết kiệm bộ nhớ khi chạy tài liệu dài.
2. **Giám sát hiệu năng**: tích hợp metrics GPU, thời gian encode ảnh, và tốc độ token để chủ động phát hiện nút thắt.
3. **Chuẩn hóa giao thức**: xây dựng lớp API REST/gRPC bao quanh script hiện tại nhằm phục vụ tích hợp với nền tảng doanh nghiệp.
