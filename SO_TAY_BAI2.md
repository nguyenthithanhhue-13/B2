# SỔ TAY BÀI 2 — bài mới chưa biết đề

Bài 2 là bài mới hoàn toàn. Bộ này không giải sẵn được bài 2; nó cho **điểm nền nhanh** nếu bài rơi vào một trong 7 kiểu dưới, và là **khung code có sẵn** để nhờ AI của BTC sửa (rẻ token hơn nhiều so với nhờ viết từ đầu).

Thư mục: `C:\Users\Thanh Hue\Downloads\Phenikaa_Campus_Courier_2026\KIT_BAI2`

---

## PHẦN A. Đọc đề, xác định kiểu bài (5 phút)

Tìm trong đề ba thứ: **dữ liệu là gì**, **phải đoán cái gì**, **file nộp trông thế nào**.

| đề cho | phải đoán | kiểu | bộ này làm được? |
|---|---|---|---|
| file bảng (`.csv`, `.xlsx`), mỗi dòng một mẫu, nhiều cột số / chữ | một cột (nhãn hoặc con số) | **1. BẢNG** | có |
| chuỗi theo thời gian đã xếp thành bảng (mỗi dòng một mẫu) | một cột | **1. BẢNG** | có |
| câu / đoạn văn | nhãn hoặc điểm số của câu | **2. VĂN BẢN** | có |
| ảnh | nhãn của cả tấm ảnh | **3. ẢNH** | có |
| file tiếng (`.wav`, `.mp3`...) | nhãn của file | **4. ÂM THANH** | có (mức cơ bản) |
| dữ liệu không có nhãn | chia thành nhóm | **5. KHÔNG CÓ NHÃN** | có |
| video (vd người làm ký hiệu tay) | nhãn của cả đoạn video | **6. VIDEO** | có (mức cơ bản: nhìn 8 khung hình, không hiểu chuyển động) |
| câu tiếng Trung / Anh / Việt | bản dịch sang tiếng kia | **7. DỊCH** | có (dùng mô hình dịch có sẵn, không học thêm từ train) |
| ảnh | **khoanh vật thể** (hộp), **đếm** vật thể | **8. KHOANH** (`CHAY_YOLO.bat`) | có → Phần B2 |
| ảnh | **tô vùng** (mask, segmentation) | **8. KHOANH** việc 3–4 | có (cần dữ liệu định dạng YOLO) → Phần B2 |
| ảnh có chữ (biển số, hoá đơn, biển hiệu...) | **đọc chữ** | **9. ĐỌC CHỮ** (`CHAY_OCR.bat`) | có → Phần B3 |
| ảnh | sinh ảnh mới | — | **không** → Phần D |
| văn bản | viết ra câu trả lời, tóm tắt | — | **không** → Phần D |
| tiếng | chép lời nói thành chữ | — | **không** → Phần D |

Nếu đề có nhiều loại dữ liệu cùng lúc (ví dụ bảng + ảnh): chạy kiểu chứa nhiều thông tin nhất trước để có điểm, rồi mới tính tiếp.

---

## PHẦN B. Chạy (kiểu 1–7)

1. Trong File Explorer, bấm đúp **`CHAY_BAI2.bat`**. Cửa sổ đen hiện ra và hỏi lần lượt. Mỗi câu: **kéo thả file vào cửa sổ** hoặc gõ trả lời, rồi **Enter**. Trong ngoặc vuông `[...]` là câu trả lời máy đoán sẵn; đúng thì chỉ cần Enter.

| máy hỏi | trả lời |
|---|---|
| `Kiểu bài? 1=BẢNG 2=VĂN BẢN 3=ẢNH 4=ÂM THANH 5=KHÔNG CÓ NHÃN 6=VIDEO 7=DỊCH` | gõ số theo Phần A |
| `Đặt tên cho lần chạy này` | gõ tên ngắn không dấu, vd `lan1` |
| `Kéo thả FILE train` | kéo file train vào. Với ảnh / âm thanh: nếu đề cho thư mục mà mỗi lớp là một thư mục con thì kéo cả **thư mục train** vào |
| `Kéo thả FILE test` | kéo file test (hoặc thư mục test) vào |
| `Kéo thả FILE NỘP MẪU` | kéo file `sample_submission` vào nếu đề có. **Rất nên có**: máy dựa vào nó để ghi file nộp đúng định dạng |
| (máy in danh sách cột, có số thứ tự) | xem để trả lời hai câu dưới |
| `Cột nào CẦN ĐOÁN (nhãn)?` | gõ **số thứ tự** hoặc tên cột mà đề yêu cầu đoán |
| `Cột nào là MÃ dòng (id)?` | gõ số / tên cột mã (thường tên `id`); không có thì Enter |
| `Các cột VĂN BẢN` (kiểu 2) | các cột chứa câu chữ, cách nhau dấu phẩy |
| `Cột nào chứa TÊN / ĐƯỜNG DẪN file?` (kiểu 3, 4, 6) | cột ghi tên file ảnh / tiếng / video |
| `Kéo thả THƯ MỤC chứa các file đó` (kiểu 3, 4, 6) | kéo thư mục chứa ảnh / tiếng / video vào |
| `Bài toán: 1=PHÂN LOẠI 2=ĐOÁN CON SỐ` | 1 nếu đoán nhãn / loại; 2 nếu đoán một con số (giá, điểm, nhiệt độ...) |
| `TINH CHỈNH thêm mạng nơ-ron? 1=CÓ 2=KHÔNG` (kiểu 3) | Enter (= CÓ). Máy **ghi file nộp chính trước**, rồi mới tinh chỉnh |
| `Tinh chỉnh tối đa bao nhiêu phút?` | Enter (= 20). Ít thời gian thì gõ 10 |

2. Máy tự học, in bảng điểm của từng cách làm, rồi dòng `=> CHỌN: ...` và `>>> FILE NỘP: ...`.
3. **Nộp file** `KIT_BAI2\ket_qua\<tên lần chạy>\submission.csv`. Trước khi nộp, mở file ra so với file nộp mẫu của đề: cùng tên cột, cùng số dòng là được.

**Riêng kiểu 7 (DỊCH)**, máy hỏi khác một chút:

| máy hỏi | trả lời |
|---|---|
| `Dịch hướng nào? 1=Trung->Việt 2=Việt->Trung 3=Anh->Việt 4=Việt->Anh 5=cặp khác` | gõ số; chọn 5 nếu là ngôn ngữ khác (Nhật, Hàn, Pháp...) |
| `Máy dịch? 1=NHANH 2=TỐT HƠN nhưng chậm` | lần đầu chọn **1** để có file nộp nhanh; còn thời gian thì chạy lại chọn **2** và nộp bản nào điểm cao hơn |
| `Kéo thả FILE test (các câu cần dịch)` | kéo file test vào |
| `Cột nào chứa câu CẦN DỊCH?` | cột chứa câu gốc |
| `Kéo thả FILE train (các cặp câu đã dịch)` | kéo vào nếu đề có; máy dùng để tự chấm điểm BLEU và lấy lại bản dịch của những câu trùng |

Máy in vài câu mẫu (`nguồn / máy / đúng`) để em nhìn chất lượng, và điểm `BLEU`, `chrF` (càng cao càng tốt).
Tốc độ đo trên máy này với câu ngắn: máy NHANH 2–6 câu mỗi giây; máy TỐT HƠN khoảng 1,5 câu mỗi giây (câu dài sẽ chậm hơn). 1.000 câu ngắn: khoảng 3–8 phút hoặc 10–15 phút. Việt → Trung bằng máy NHANH phải đi vòng qua tiếng Anh nên kém; với hướng này nên chọn máy TỐT HƠN.

**Đọc bảng điểm:**

- `accuracy`, `f1_macro`: càng **cao** càng tốt (tối đa 1).
- `rmse`, `mae`: càng **thấp** càng tốt. `r2`: càng cao càng tốt.
- Đây là điểm tự chấm trên train; điểm thật trên web thường thấp hơn một chút.
- Nếu đề chấm bằng F1 mà máy chọn theo accuracy: không sao, hai số thường đi cùng nhau.

**Nếu dòng FILE NỘP ghi `ghép THEO THỨ TỰ DÒNG (id không khớp ...)`:** máy không khớp được mã dòng giữa test và file mẫu. Chạy lại và chọn đúng cột id.

**Chạy lại nhanh** (không phải trả lời lại): kéo file `ket_qua\<tên>\cau_hinh.json` thả lên **`CHAY_LAI_BAI2.bat`**.

**Riêng kiểu 3 (ẢNH):** máy thử 3 bộ đọc ảnh có sẵn (ResNet50, CLIP, DINOv2), ghép cả ba, và "CLIP đoán theo tên nhãn" (không học, chỉ hợp khi nhãn là chữ tiếng Anh như `cat`, `dog`). Khi đã có file nộp chính, nếu chọn tinh chỉnh, máy học tiếp cả mạng MobileNetV3 trên ảnh của đề, rồi in một trong hai dòng:

| dòng hiện ra | em làm gì |
|---|---|
| `>>> Tinh chỉnh TỐT HƠN ... Nên nộp: ...\tinh_chinh\submission.csv` | nộp file đó |
| `>>> Tinh chỉnh KHÔNG hơn rõ ... Giữ file nộp chính` | không nộp gì thêm |

Còn lượt nộp thì thử thêm `ket_qua\<tên>\tron_ca_hai\submission.csv` (trộn hai cách, thường tốt nhất). Web lấy lần nộp cao nhất nên thử không mất gì.

### B2. Kiểu 8: KHOANH VẬT THỂ / TÔ VÙNG (bấm đúp `CHAY_YOLO.bat`)

| máy hỏi | trả lời |
|---|---|
| `Việc gì?` | **1** = chỉ khoanh / đếm các vật thể thường gặp (người, xe, chó, mèo, chai, ghế... 80 loại), không cần học. **2** = đề có ảnh train kèm hộp → máy học rồi khoanh. **3** = tô vùng có sẵn. **4** = học tô vùng (đề phải cho dữ liệu định dạng YOLO) |
| `Kéo thả file data.yaml ... HOẶC file csv ghi các hộp` | đề cho thư mục có `data.yaml` thì kéo file đó; đề cho file csv (mỗi dòng: tên ảnh, nhãn, 4 số toạ độ) thì kéo csv, máy hỏi tiếp tên các cột và tự chuyển |
| `Toạ độ hộp kiểu gì?` | xem đề: `xmin, ymin, xmax, ymax` → **1**; `x, y, width, height` → **2**; số nhỏ hơn 1 (đã chia cho cỡ ảnh) → **3** |
| `Học tối đa bao nhiêu phút?` | Enter (= 30) |
| `Cỡ ảnh` | Enter (= 640) |
| `THƯ MỤC ảnh test` | kéo thư mục ảnh test vào |
| `Ngưỡng tự tin` | đề chấm bằng **mAP** → gõ `0.01`; đề **đếm** số lượng → Enter (= 0.25) |

Máy in `TỰ CHẤM ... mAP50 = ...` (càng gần 1 càng tốt) rồi ghi:

- `ket_qua\<tên>\du_doan.csv`: mỗi dòng một hộp (tên ảnh, loại, độ tin cậy, toạ độ theo cả hai kiểu);
- `dem_so_luong.csv`: số vật thể mỗi ảnh;
- `anh_ve_hop\`: 20 ảnh có vẽ hộp, mở ra xem máy khoanh có đúng không.

File nộp của đề thường có định dạng riêng → dùng **mẫu D** (Phần D) nhờ AI viết đoạn đổi `du_doan.csv` sang đúng định dạng. Chạy lại: kéo `cau_hinh.json` thả lên `CHAY_LAI_YOLO.bat`.

Tốc độ trên CPU: khoanh khoảng 10–20 ảnh mỗi giây; học khoảng 1–3 phút mỗi vòng cho 500 ảnh. 30 phút học thường đủ cho vài trăm ảnh.

### B3. Kiểu 9: ĐỌC CHỮ TRONG ẢNH (bấm đúp `CHAY_OCR.bat`)

Trả lời: tên lần chạy → ngôn ngữ (Enter = Việt + Anh) → kéo thư mục ảnh → (nếu đề có file chữ đúng) kéo file đó, chọn cột tên ảnh và cột chữ để máy tự chấm `CER` (càng thấp càng tốt).
Kết quả: `ket_qua\<tên>\chu_doc_duoc.csv` (mỗi ảnh một dòng chữ), `tung_dong_chu.csv` (từng đoạn chữ + toạ độ). Định dạng nộp khác thì dùng mẫu D. Tốc độ: khoảng 1–3 giây mỗi ảnh.

---

## PHẦN C. Gặp lỗi

Máy in `!!! LỖI: ...` kèm khoảng 20 dòng phía trên.

| dòng lỗi có chữ | cách xử lý |
|---|---|
| `Không thấy file / thư mục đó` | kéo thả lại đúng file |
| `Không thấy ... file (vd ...)` | sai thư mục chứa ảnh / tiếng: chạy lại, kéo đúng thư mục ở câu `THƯ MỤC chứa các file đó` |
| `Thư mục train phải có thư mục con cho từng lớp` | thư mục ảnh không chia theo lớp: dùng file bảng (csv) của đề thay cho thư mục |
| `KeyError: '<tên cột>'` | gõ sai tên cột: chạy lại, gõ **số thứ tự** cột cho chắc |
| `No module named ...` | đang chạy nhầm Python: phải chạy bằng file `.bat` |
| `MemoryError` hoặc máy đơ | dữ liệu quá lớn: hỏi AI theo mẫu A, thêm câu "dữ liệu lớn, hãy lấy mẫu bớt" |
| lỗi khác | dùng mẫu A ở Phần D |

---

## PHẦN D. Dùng AI của BTC cho bài 2

Bài 2 là nơi đáng dùng token nhất, vì bài 1 đã chạy tự động.

**Thứ tự nên làm (lời thầy: nộp nhanh một bản có điểm trước, cải tiến sau):** đọc đề → chạy `CHAY_BAI2.bat` nếu khớp một trong 5 kiểu → nộp lấy điểm nền → **rồi mới** dùng AI để tăng điểm hoặc làm phần chưa khớp.

**Mẫu A — có lỗi:**

```
Tôi chạy một script Python và bị lỗi. 20 dòng cuối:
<dán>

Hàm chứa dòng lỗi (file bai2.py):
<dán đúng hàm đó>

Chỉ ra nguyên nhân và cách sửa ngắn nhất. Trả lời ngắn: chỉ đưa lại hàm đã sửa, đầy đủ.
```

**Mẫu B — đề không khớp kiểu nào:**

```
Đề bài: <dán nguyên đoạn mô tả nhiệm vụ, dữ liệu, cách chấm, định dạng file nộp>

5 dòng đầu của file train: <dán>
5 dòng đầu của file nộp mẫu: <dán>

Máy tôi: Windows, CPU (không GPU), Python 3.13, KHÔNG có mạng. Đã cài: numpy, pandas, scikit-learn, lightgbm, torch, torchvision, transformers, sentence-transformers, sentencepiece, sacrebleu, opencv, pillow, soundfile, scipy, matplotlib.
Đã cài thêm: ultralytics (YOLO11), easyocr. Mô hình có sẵn offline: ResNet18/50, MobileNetV3, EfficientNet-B0 (torchvision); YOLO11n và YOLO11n-seg ở KIT_BAI2\mo_hinh\yolo11n.pt, yolo11n-seg.pt; EasyOCR vi+en, ch_sim+en; facebook/dinov2-small; openai/clip-vit-base-patch32; intfloat/multilingual-e5-small; bkai-foundation-models/vietnamese-bi-encoder; paraphrase-multilingual-MiniLM-L12-v2; Helsinki-NLP/opus-mt-zh-vi, opus-mt-en-vi, opus-mt-vi-en, opus-mt-en-zh; facebook/nllb-200-distilled-600M.

Viết MỘT file Python hoàn chỉnh, chạy được ngay, đọc train/test từ đường dẫn tôi sẽ điền ở đầu file, và ghi submission.csv đúng định dạng file nộp mẫu. Ưu tiên cách đơn giản, chạy nhanh trên CPU. Không giải thích dài.
```

**Mẫu C — đã có điểm nền, muốn tăng điểm:**

```
Bài toán: <1-2 câu>. Tôi đang dùng <dán dòng "=> CHỌN: ..." và bảng điểm máy in ra>.
Thước đo của đề: <tên thước đo>.

Đây là hàm tạo đặc trưng tôi đang dùng (file bai2.py): <dán hàm table_features / text_features / image_features tương ứng>

Đề xuất 2-3 thay đổi có khả năng tăng điểm nhất và đưa lại hàm đã sửa, đầy đủ. Trả lời ngắn.
```

**Mẫu D — đổi kết quả sang đúng định dạng nộp (hay dùng với kiểu 8, 9):**

```
Tôi có file kết quả du_doan.csv, 5 dòng đầu:
<dán>

Đề yêu cầu file nộp dạng (mô tả của đề + 5 dòng đầu file nộp mẫu):
<dán>

Viết đoạn Python ngắn (pandas) đọc du_doan.csv và ghi submission.csv đúng định dạng trên. Ảnh không có vật thể nào vẫn phải có dòng nếu đề yêu cầu. Chỉ đưa code.
```

**Cách chạy code AI viết (mẫu B, D):** trong VS Code tạo file mới trong `KIT_BAI2`, đặt tên `giai.py`, dán code vào, sửa đường dẫn ở đầu file, lưu. Mở Terminal trong VS Code (**Ctrl + `**) và gõ:
`& "C:\Users\Thanh Hue\.venv\Scripts\python.exe" giai.py`

**Cách sửa một hàm trong `bai2.py` (mẫu A, C):** sao lưu file (Ctrl + C, Ctrl + V trong File Explorer) → mở `bai2.py` → Ctrl + F gõ `def <tên hàm>` → bôi đen cả hàm → dán hàm mới đè lên → Ctrl + S → kéo `cau_hinh.json` thả lên `CHAY_LAI_BAI2.bat`.

Các hàm chính trong `bai2.py`:

| việc | hàm |
|---|---|
| đọc file bảng | `read_table` |
| tạo đặc trưng cho bảng | `table_features` |
| tạo đặc trưng cho văn bản | `text_features`, `text_embed` |
| tạo đặc trưng cho ảnh | `image_features`, `image_embed` |
| tạo đặc trưng cho âm thanh | `audio_features`, `audio_feats_one` |
| các mô hình được thử | `make_models` |
| chấm chéo và chọn mô hình | `fit_predict` |
| lấy khung hình từ video | `frames_of` |
| dịch | `translate`, `run_translate` |
| ghi file nộp | `write_submission` |

---

## PHẦN E. Bộ này dùng gì (đối chiếu với danh mục kiến thức IOAI)

| mục trong danh mục | có trong bộ này |
|---|---|
| Pandas, NumPy, scikit-learn, xử lý dữ liệu thiếu, cột loại, cột ngày giờ | có (kiểu 1) |
| Hồi quy tuyến tính / logistic, Gradient Boosting, chấm chéo, accuracy / F1 / RMSE | có (mọi kiểu) |
| K-Means, PCA | có (kiểu 5) |
| Nhúng văn bản bằng mô hình có sẵn, phân loại văn bản, TF-IDF | có (kiểu 2) |
| Bộ mã hóa ảnh có sẵn (ResNet), CLIP, DINOv2, phân loại ảnh, tinh chỉnh (transfer learning), tăng cường dữ liệu (lật, cắt) | có (kiểu 3) |
| Phát hiện vật thể (YOLO), phân vùng ảnh | có (kiểu 8, YOLO11n bản nhỏ, học trên CPU) |
| Đọc chữ trong ảnh (OCR) | có (kiểu 9, EasyOCR) |
| Âm thanh | chỉ có đặc trưng phổ mel; **không** có HuBERT / Whisper |
| Mô hình mã hóa - giải mã (dịch máy) | có (kiểu 7): opus-mt Trung→Việt, Anh↔Việt, Anh→Trung; NLLB-200 cho mọi cặp. Chỉ dùng nguyên trạng, không tinh chỉnh |
| Video | có (kiểu 6), mức cơ bản |
| CNN tự viết từ đầu, mạng lớn | **không** (máy không có GPU) |
| U-Net, GAN, Diffusion | **không** |
| Mô hình ngôn ngữ, sinh văn bản, tóm tắt | **không** |
| t-SNE, UMAP, DBSCAN | **không** (nhờ AI thêm nếu đề cần; t-SNE và DBSCAN có sẵn trong scikit-learn) |

---

## PHẦN F. Ghi chú

- Kiểu 6 (VIDEO) lấy 8 khung hình cách đều của mỗi video rồi lấy trung bình, nên nó nhận ra "trong video có gì" chứ không nhận ra "chuyển động thế nào". Với ngôn ngữ ký hiệu, những ký hiệu khác nhau chủ yếu ở hình dáng bàn tay sẽ đoán được; những ký hiệu chỉ khác ở hướng chuyển động sẽ dễ nhầm. Nếu đề cho sẵn tọa độ khớp tay / khung xương theo từng khung hình (file số) thay vì video: đó là dữ liệu dạng bảng hoặc chuỗi, dùng mẫu B nhờ AI viết.
- Kiểu 7 (DỊCH) không học từ dữ liệu train của đề (tinh chỉnh mô hình dịch trên CPU quá chậm). Nếu đề yêu cầu phong cách dịch riêng hay có thuật ngữ riêng thì điểm sẽ hạn chế.

- Mọi mô hình có sẵn nằm trong bộ nhớ đệm của **máy Huệ** (`C:\Users\Thanh Hue\.cache\huggingface` và `.cache\torch`), không nằm trong thư mục này. Máy khác muốn dùng kiểu 2 và 3 phải tự tải các mô hình đó khi còn mạng.
- Kiểu 3 trên CPU: khoảng 10–15 ảnh mỗi giây cho mỗi mô hình. 5.000 ảnh mất chừng 10–15 phút.
- Kiểu 2: khoảng 15 câu mỗi giây cho mỗi mô hình nhúng. 10.000 câu mất chừng 20 phút; nếu gấp thì kết quả `TF-IDF` (có ngay) thường đã khá.
- Kết quả tính đặc trưng được lưu trong `cache\`, chạy lại lần hai sẽ nhanh.
