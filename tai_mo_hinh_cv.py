"""Tải trước (KHI CÒN MẠNG) mọi mô hình mà bài 2 dùng: ảnh, YOLO, OCR, đọc câu, dịch.
Chạy một lần:  python tai_mo_hinh_cv.py            (máy Huệ đã tải rồi; máy khác / JupyterHub thì chạy lệnh này)
               python tai_mo_hinh_cv.py --nhanh    (bỏ mô hình dịch NLLB 2,4 GB và EasyOCR tiếng Trung)
Cần cài trước:  pip install ultralytics easyocr transformers sentence-transformers sentencepiece sacremoses
"""
import os
import sys
from pathlib import Path

for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "YOLO_OFFLINE"):
    os.environ.pop(k, None)
for _st in (sys.stdout, sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
HERE = Path(__file__).resolve().parent
W = HERE / "mo_hinh"
W.mkdir(exist_ok=True)
NHANH = "--nhanh" in sys.argv


def step(name, fn):
    try:
        fn()
        print(f"[ok]  {name}", flush=True)
    except Exception as e:
        print(f"[LỖI] {name}: {type(e).__name__}: {str(e)[:200]}", flush=True)


def tv():
    import torchvision.models as M
    for w in (M.ResNet50_Weights.DEFAULT, M.ResNet18_Weights.DEFAULT, M.MobileNet_V3_Large_Weights.DEFAULT, M.EfficientNet_B0_Weights.DEFAULT):
        w.get_state_dict(progress=False)


def hf():
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    CLIPModel.from_pretrained("openai/clip-vit-base-patch32"), CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    AutoModel.from_pretrained("facebook/dinov2-small"), AutoImageProcessor.from_pretrained("facebook/dinov2-small")


def yolo():
    import shutil
    from ultralytics import YOLO
    from ultralytics.utils.checks import check_font
    cwd = os.getcwd()
    os.chdir(W)      # ultralytics tải file .pt vào thư mục hiện tại
    try:
        for n in ("yolo11n.pt", "yolo11n-seg.pt"):
            if not (W / n).exists():
                YOLO(n)
    finally:
        os.chdir(cwd)
    check_font("Arial.ttf"), check_font("Arial.Unicode.ttf")      # phông chữ để vẽ nhãn lên ảnh
    for n in ("yolo11n.pt", "yolo11n-seg.pt"):
        assert (W / n).exists(), n


def ocr():
    import easyocr
    easyocr.Reader(["vi", "en"], gpu=False, verbose=False)
    if not NHANH:
        easyocr.Reader(["ch_sim", "en"], gpu=False, verbose=False)


TEXT = ["intfloat/multilingual-e5-small", "bkai-foundation-models/vietnamese-bi-encoder", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]
MT = ["Helsinki-NLP/opus-mt-zh-vi", "Helsinki-NLP/opus-mt-en-vi", "Helsinki-NLP/opus-mt-vi-en", "Helsinki-NLP/opus-mt-en-zh"]


def text():
    from transformers import AutoModel, AutoTokenizer
    for m in TEXT:
        AutoModel.from_pretrained(m), AutoTokenizer.from_pretrained(m)


def mt():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    for m in MT + ([] if NHANH else ["facebook/nllb-200-distilled-600M"]):
        AutoModelForSeq2SeqLM.from_pretrained(m), AutoTokenizer.from_pretrained(m)
        print(f"      đã tải {m}", flush=True)


step("torchvision: ResNet50, ResNet18, MobileNetV3, EfficientNet-B0", tv)
step("CLIP + DINOv2", hf)
step("YOLO11n (khoanh hộp) + YOLO11n-seg (tô vùng)", yolo)
step("EasyOCR: Việt + Anh" + ("" if NHANH else ", Trung + Anh"), ocr)
step("mô hình đọc câu (e5, vietnamese-bi-encoder, MiniLM)", text)
step("mô hình dịch (opus-mt" + ("" if NHANH else " + NLLB-200") + ")", mt)
print("Xong.")
