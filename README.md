# Bài 2 chung kết — ảnh + chữ (NLP + thị giác máy tính)

Bộ làm bài 2 chạy trên **Jupyter có GPU của BTC**. Đề chưa công bố, nên đây là **khung làm bài tự động** cho dạng hay gặp nhất: đoán **nhãn** hoặc **con số** từ **ảnh + chữ** (hoặc chỉ ảnh, chỉ chữ), kèm công cụ cho các dạng khác.

**Hướng dẫn từng bước:** tải file [`HUONG_DAN_BAI2.docx`](HUONG_DAN_BAI2.docx) (bấm vào → nút Download ⬇).

---

## Chạy nhanh

1. Đưa thư mục này lên Jupyter (upload zip rồi giải nén, hoặc `git clone https://github.com/nguyenthithanhhue-13/B2`).
2. Mở `notebooks/B0_cai_dat.ipynb` → chạy từng ô (Shift + Enter): cài thư viện, kiểm tra GPU, tải mô hình.
3. Mở `notebooks/CHAY_TU_DONG.ipynb` → ô ✏️ dán đường dẫn file train, test, file nộp mẫu, thư mục ảnh → **Run → Run All Cells**.
4. Thấy `>>> NỘP NGAY FILE NÀY` → tải `submission.csv` về nộp. Cuối cùng bảng **TÓM TẮT** ghi file nên nộp.

## CHAY_TU_DONG làm gì

```
dữ liệu đề ─► tự đoán cột (id, ảnh, chữ, nhãn) ─► bảng chuẩn ─► kiểm tra dữ liệu (ảnh hỏng / trùng, chữ rỗng, test khác train)
          ─► ảnh → vector (CLIP, DINOv2) · chữ → vector (multilingual-e5, vietnamese-bi-encoder)
          ─► làm sạch nhãn sai (chấm chéo, ý tưởng Cleanlab)
          ─► MÔ HÌNH NHANH: hồi quy logistic / LightGBM / CLIP không học, chấm chéo 5 phần ─► NỘP 1  (~10–15 phút)
          ─► TINH CHỈNH trên GPU: nhánh ảnh CLIP + nhánh chữ e5 + đầu phân loại / hồi quy, fp16 ─► NỘP 2
          ─► TRỘN hai cách ─► NỘP 3 ─► bảng tóm tắt + lời khuyên nộp file nào
```

## Mô hình mẫu (sample model)

Model của bài 2 được **train trong giờ thi** trên dữ liệu đề (chưa có đề nên không có trọng số train sẵn). Thứ có sẵn là **kiến trúc + code train** đã chạy thử:

| mô hình | ở đâu | là gì |
|---|---|---|
| mô hình nhanh | `b2lib.mo_hinh_nhanh` | vector ảnh/chữ từ mô hình học sẵn → hồi quy logistic, LightGBM, Ridge; chấm chéo 5 phần, trung bình top 3 |
| **mô hình tinh chỉnh ảnh + chữ** | `b2lib.tinh_chinh` | nhánh ảnh (CLIP / SigLIP / DINOv2) + nhánh chữ (multilingual-e5 / vietnamese-bi-encoder, mean pooling) → nối → LayerNorm → MLP → nhãn (hoặc 1 số). AdamW, OneCycle, label smoothing, fp16, lật ảnh, giữ 15% để chấm, lưu bản tốt nhất |
| CLIP đoán theo tên nhãn | `b2lib.clip_doan_theo_ten` | không học, so ảnh với câu "a photo of a <nhãn>" |
| hỏi đáp về ảnh | `b2lib.vlm_tra_loi` (B5) | Qwen2.5-VL-3B, không học: trả lời câu hỏi, mô tả ảnh |
| khoanh vật thể | `yolo_mau.py` | YOLO11n (trọng số trong `mo_hinh/`), học trên dữ liệu đề |
| đọc chữ trong ảnh | `ocr_mau.py` | EasyOCR tiếng Việt + Anh |

Mô hình học sẵn (CLIP, DINOv2, e5…) là mô hình công khai trên Hugging Face, notebook B0 tải về.

## Các file

| file | dùng |
|---|---|
| `notebooks/CHAY_TU_DONG.ipynb` | **cách chính**, chạy cả bài một lần |
| `notebooks/B0_cai_dat.ipynb` | cài đặt, chạy đầu tiên |
| `notebooks/B1` … `B6` | từng bước của CHAY_TU_DONG tách riêng, để chỉnh tay khi cần |
| `notebooks/B5_hoi_dap_anh.ipynb` | đề bắt viết chữ từ ảnh (VQA, mô tả ảnh) |
| `notebooks/b2lib.py` | thư viện chung (mọi notebook gọi) |
| `bai2.py` | hỏi đáp trong Terminal cho dạng bảng, văn bản, âm thanh, video, **dịch** |
| `yolo_mau.py`, `ocr_mau.py` | khoanh vật thể, đọc chữ trong ảnh (Terminal) |
| `*.bat` | chỉ dùng trên laptop Windows, không dùng trên Jupyter |

Đã chạy thử trên CPU với dữ liệu tự tạo (đoán nhãn và đoán số). Chưa chạy thử trên GPU thật và phần B5.
