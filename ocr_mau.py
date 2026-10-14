"""BÀI 2 - ĐỌC CHỮ TRONG ẢNH (OCR) bằng EasyOCR: tiếng Việt + tiếng Anh (tuỳ chọn thêm tiếng Trung giản thể).

Cách chạy: bấm đúp CHAY_OCR.bat  (hoặc: python ocr_mau.py)
Kết quả: ket_qua\\<tên>\\chu_doc_duoc.csv  (mỗi ảnh một dòng: tên ảnh, toàn bộ chữ đọc được)
         ket_qua\\<tên>\\tung_dong_chu.csv (mỗi đoạn chữ một dòng: tên ảnh, chữ, độ tin cậy, toạ độ khung)
Nếu đề cho file train có sẵn chữ đúng: máy tự chấm (tỉ lệ ký tự sai CER, càng thấp càng tốt).
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
for _st in (sys.stdout, sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bai2 import IMG_EXT, ask, ask_path, clean_path, pick_col, read_table, show_cols

OUT = HERE / "ket_qua"


def cer(a, b):
    """Tỉ lệ ký tự sai (khoảng cách sửa chuỗi / độ dài chữ đúng)."""
    a, b = str(a), str(b)
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[len(b)] / max(1, len(b))


def run(cfg):
    import easyocr
    name = cfg.get("ten") or time.strftime("ocr_%H%M")
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "cau_hinh.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    langs = cfg["ngon_ngu"]
    readers = [easyocr.Reader(["ch_sim", "en"], gpu=False, verbose=False)] if "ch_sim" in langs else []
    if any(l in langs for l in ("vi", "en")):
        readers.insert(0, easyocr.Reader([l for l in langs if l in ("vi", "en")], gpu=False, verbose=False))
    paths = sorted(str(p) for p in Path(clean_path(cfg["anh"])).rglob("*") if p.suffix.lower() in IMG_EXT)
    print(f"Đang đọc chữ trong {len(paths)} ảnh ...", flush=True)
    t0, full, parts = time.time(), [], []
    for i, p in enumerate(paths):
        found = []
        for rd in readers:
            try:
                found += rd.readtext(p, paragraph=False)
            except Exception as e:
                print(f"   !!! không đọc được {Path(p).name}: {type(e).__name__}")
        found.sort(key=lambda r: (round(min(pt[1] for pt in r[0]) / 20), min(pt[0] for pt in r[0])))      # trên xuống dưới, trái sang phải
        for box, txt, conf in found:
            xs, ys = [pt[0] for pt in box], [pt[1] for pt in box]
            parts.append({"image_id": Path(p).stem, "file": Path(p).name, "text": txt, "confidence": round(float(conf), 4),
                          "x1": int(min(xs)), "y1": int(min(ys)), "x2": int(max(xs)), "y2": int(max(ys))})
        full.append({"image_id": Path(p).stem, "file": Path(p).name, "text": " ".join(t for _, t, _ in found)})
        if i % 20 == 0:
            print(f"   {i + 1}/{len(paths)}   ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(full)
    df.to_csv(folder / "chu_doc_duoc.csv", index=False, encoding="utf-8")
    pd.DataFrame(parts).to_csv(folder / "tung_dong_chu.csv", index=False, encoding="utf-8")
    if cfg.get("dung"):      # tự chấm nếu có chữ đúng
        g = read_table(cfg["dung"])
        key = dict(zip(g[cfg["cot_anh"]].astype(str).map(lambda s: Path(s).stem), g[cfg["cot_chu"]].astype(str)))
        m = df[df["image_id"].isin(key)]
        if len(m):
            c = [cer(t, key[k]) for t, k in zip(m["text"], m["image_id"])]
            print(f"\n   TỰ CHẤM trên {len(m)} ảnh có chữ đúng: CER = {sum(c) / len(c):.4f} (0 = đúng hết), đúng nguyên văn {sum(x == 0 for x in c) / len(c):.1%}")
    print(f"\n>>> KẾT QUẢ: {folder / 'chu_doc_duoc.csv'}   ({len(df)} ảnh)")
    print(f"    từng đoạn chữ + toạ độ: {folder / 'tung_dong_chu.csv'}")
    print(f"Xong sau {(time.time() - t0) / 60:.1f} phút.")


def wizard():
    print(__doc__)
    cfg = {"ten": ask("Đặt tên cho lần chạy này (không dấu, không cách)", time.strftime("ocr_%H%M"))}
    l = ask("Ngôn ngữ của chữ? 1=Việt+Anh  2=chỉ Anh (nhanh hơn)  3=Trung giản thể+Anh  4=Việt+Anh+Trung", "1")
    cfg["ngon_ngu"] = {"1": ["vi", "en"], "2": ["en"], "3": ["ch_sim"], "4": ["vi", "en", "ch_sim"]}.get(l, ["vi", "en"])
    cfg["anh"] = ask_path("Kéo thả THƯ MỤC ảnh cần đọc chữ vào đây")
    cfg["dung"] = ask_path("Có file chứa CHỮ ĐÚNG của một số ảnh (vd train.csv) để tự chấm? Kéo thả vào (Enter = không có)", allow_empty=True)
    if cfg["dung"]:
        g = read_table(cfg["dung"])
        show_cols(g)
        cfg["cot_anh"] = pick_col(g, "Cột nào là TÊN ẢNH?", str(g.columns[0]))
        cfg["cot_chu"] = pick_col(g, "Cột nào là CHỮ ĐÚNG?", str(g.columns[-1]))
    return cfg


if __name__ == "__main__":
    try:
        cfg = json.loads(clean_path(sys.argv[sys.argv.index("--lai") + 1]).read_text(encoding="utf-8")) if "--lai" in sys.argv else wizard()
        run(cfg)
    except KeyboardInterrupt:
        print("\nĐã dừng.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n!!! LỖI: {type(e).__name__}: {e}")
        print("    Chép 20 dòng cuối ở trên cho AI của BTC (mẫu A trong SO_TAY_BAI2.md).")
