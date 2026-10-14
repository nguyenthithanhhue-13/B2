"""BÀI 2 - KHOANH VẬT THỂ / TÔ VÙNG bằng YOLO11 (ultralytics). Hỏi vài câu rồi tự chạy, ghi kết quả ra file csv.

Cách chạy: bấm đúp CHAY_YOLO.bat  (hoặc: python yolo_mau.py)

4 việc:
  1. KHOANH vật thể bằng mô hình CÓ SẴN (80 loại thường gặp: người, xe đạp, ô tô, chó, mèo, chai, ghế...) - không cần học
  2. HỌC khoanh vật thể trên dữ liệu của đề, rồi khoanh trên ảnh test
  3. TÔ VÙNG (segmentation) bằng mô hình có sẵn (80 loại như trên)
  4. HỌC tô vùng trên dữ liệu của đề (dữ liệu phải ở định dạng YOLO, có file data.yaml)
Dữ liệu học (việc 2) có thể là: file data.yaml (định dạng YOLO) HOẶC một file csv ghi hộp (mỗi dòng: tên ảnh, nhãn, toạ độ hộp)
-> máy tự chuyển sang định dạng YOLO.
Kết quả: ket_qua\\<tên>\\du_doan.csv (mỗi dòng một hộp), dem_so_luong.csv (số vật thể mỗi ảnh), anh_ve_hop\\ (vài ảnh có vẽ hộp để xem).
Định dạng nộp của đề khác thì nhờ AI của BTC viết đoạn đổi từ du_doan.csv sang định dạng đó (mẫu D trong SO_TAY_BAI2.md).
"""
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
for _st in (sys.stdout, sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bai2 import IMG_EXT, ask, ask_path, clean_path, pick_col, read_table, show_cols      # dùng lại phần hỏi đáp của bai2.py

OUT = HERE / "ket_qua"
W = HERE / "mo_hinh"
def DEV():
    import torch
    return 0 if torch.cuda.is_available() else "cpu"


WEIGHTS = {"hop": W / "yolo11n.pt", "vung": W / "yolo11n-seg.pt"}


def images_in(folder):
    return sorted(str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)


def csv_to_yolo(cfg, folder):
    """Đổi file csv ghi hộp -> thư mục định dạng YOLO (images/, labels/, data.yaml). Chia 85% học, 15% tự chấm theo ảnh."""
    from PIL import Image
    df = read_table(cfg["csv_hop"])
    c = cfg["cot"]
    names = sorted(df[c["nhan"]].astype(str).unique())
    cid = {n: i for i, n in enumerate(names)}
    root = clean_path(cfg["thu_muc_anh"])
    files = sorted(df[c["anh"]].astype(str).unique())
    rng = np.random.default_rng(0)
    val = set(rng.choice(files, max(1, int(0.15 * len(files))), replace=False))
    D = folder / "yolo_data"
    bad = 0
    for f in files:
        src = Path(f) if Path(f).is_absolute() else root / f
        if not src.exists():
            cand = [root / (f + e) for e in IMG_EXT if (root / (f + e)).exists()]
            if not cand:
                bad += 1
                continue
            src = cand[0]
        part = "val" if f in val else "train"
        (D / "images" / part).mkdir(parents=True, exist_ok=True)
        (D / "labels" / part).mkdir(parents=True, exist_ok=True)
        dst = D / "images" / part / src.name
        if not dst.exists():
            shutil.copy(src, dst)
        w, h = Image.open(src).size
        lines = []
        for _, r in df[df[c["anh"]].astype(str) == f].iterrows():
            a, b, cc, d = (float(r[k]) for k in c["toa_do"])
            if cfg["kieu_toa_do"] == "1":        # x1 y1 x2 y2 (điểm ảnh)
                xc, yc, bw, bh = (a + cc) / 2 / w, (b + d) / 2 / h, (cc - a) / w, (d - b) / h
            elif cfg["kieu_toa_do"] == "2":      # x y rộng cao, (x, y) là góc trên trái (điểm ảnh)
                xc, yc, bw, bh = (a + cc / 2) / w, (b + d / 2) / h, cc / w, d / h
            else:                                # YOLO: tâm x, tâm y, rộng, cao, đã chia cho cỡ ảnh (0..1)
                xc, yc, bw, bh = a, b, cc, d
            lines.append(f"{cid[str(r[c['nhan']])]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        (D / "labels" / part / (src.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
    if bad:
        print(f"   !!! {bad} ảnh trong csv không tìm thấy trong thư mục ảnh (bỏ qua)")
    y = D / "data.yaml"
    y.write_text(f"path: {D.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {json.dumps(n, ensure_ascii=False)}\n" for i, n in enumerate(names)), encoding="utf-8")
    print(f"   đã đổi sang định dạng YOLO: {len(files) - bad} ảnh, {len(names)} loại vật thể -> {y}")
    return str(y)


def run(cfg):
    from ultralytics import YOLO
    name = cfg.get("ten") or time.strftime("yolo_%H%M")
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "cau_hinh.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    t0 = time.time()
    seg = cfg["viec"] in ("3", "4")
    model = YOLO(str(WEIGHTS["vung" if seg else "hop"]))
    if cfg["viec"] in ("2", "4"):
        data = cfg.get("data_yaml") or csv_to_yolo(cfg, folder)
        print(f"\nĐANG HỌC (tối đa {cfg['phut']} phút, ảnh cỡ {cfg['co_anh']}). Mỗi vòng in một bảng; cột mAP50 càng cao càng tốt (tối đa 1).", flush=True)
        model.train(data=data, epochs=int(cfg.get("so_vong", 100)), time=float(cfg["phut"]) / 60, imgsz=int(cfg["co_anh"]), device=DEV(),
                    batch=16 if DEV() == 0 else 8, workers=0, project=str(folder), name="hoc", exist_ok=True, plots=False, verbose=False, patience=20, seed=0)
        best = folder / "hoc" / "weights" / "best.pt"
        model = YOLO(str(best))
        m = model.val(data=data, imgsz=int(cfg["co_anh"]), device=DEV(), workers=0, plots=False, verbose=False, project=str(folder), name="cham", exist_ok=True)
        print(f"\n   TỰ CHẤM trên 15% ảnh giữ lại: mAP50 = {m.box.map50:.4f}   mAP50-95 = {m.box.map:.4f}" + (f"   (vùng: mAP50 = {m.seg.map50:.4f})" if seg else ""))
        print(f"   mô hình đã học: {best}")
    paths = images_in(cfg["test"])
    print(f"\nĐANG KHOANH {len(paths)} ảnh test ...", flush=True)
    rows, counts = [], []
    (folder / "anh_ve_hop").mkdir(exist_ok=True)
    for i in range(0, len(paths), 16):
        for r in model.predict(paths[i:i + 16], imgsz=int(cfg["co_anh"]), conf=float(cfg["nguong"]), device=DEV(), verbose=False):
            p = Path(r.path)
            h, w = r.orig_shape
            n = 0
            if r.boxes is not None:
                xyxy, cls, cf = r.boxes.xyxy.numpy(), r.boxes.cls.numpy().astype(int), r.boxes.conf.numpy()
                polys = r.masks.xyn if (seg and r.masks is not None) else [None] * len(cls)
                for (x1, y1, x2, y2), k, s, poly in zip(xyxy, cls, cf, polys):
                    row = {"image_id": p.stem, "file": p.name, "class_id": k, "class_name": r.names[k], "confidence": round(float(s), 4),
                           "x1": round(float(x1), 1), "y1": round(float(y1), 1), "x2": round(float(x2), 1), "y2": round(float(y2), 1),
                           "x_center": round((x1 + x2) / 2 / w, 6), "y_center": round((y1 + y2) / 2 / h, 6),
                           "width": round((x2 - x1) / w, 6), "height": round((y2 - y1) / h, 6), "img_w": w, "img_h": h}
                    if seg:
                        row["polygon"] = " ".join(f"{a:.4f} {b:.4f}" for a, b in (poly if poly is not None else []))
                    rows.append(row)
                    n += 1
            counts.append({"image_id": p.stem, "file": p.name, "so_vat_the": n})
            if len(counts) <= 20:
                try:
                    import cv2
                    cv2.imwrite(str(folder / "anh_ve_hop" / (p.stem + ".jpg")), r.plot())
                except Exception:
                    pass
        print(f"   {min(i + 16, len(paths))}/{len(paths)}", flush=True)
    cols = ["image_id", "file", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "x_center", "y_center", "width", "height", "img_w", "img_h"] + (["polygon"] if seg else [])
    pd.DataFrame(rows, columns=cols).to_csv(folder / "du_doan.csv", index=False, encoding="utf-8")
    cnt = pd.DataFrame(counts)
    if rows:
        per = pd.DataFrame(rows).pivot_table(index="image_id", columns="class_name", values="confidence", aggfunc="count", fill_value=0)
        cnt = cnt.merge(per, left_on="image_id", right_index=True, how="left").fillna(0)
    cnt.to_csv(folder / "dem_so_luong.csv", index=False, encoding="utf-8")
    print(f"\n>>> KẾT QUẢ: {folder / 'du_doan.csv'}   ({len(rows)} hộp trên {len(paths)} ảnh)")
    print(f"    số vật thể mỗi ảnh: {folder / 'dem_so_luong.csv'}")
    print(f"    vài ảnh có vẽ hộp để xem bằng mắt: {folder / 'anh_ve_hop'}")
    print("    Toạ độ: x1 y1 x2 y2 = góc trên trái / dưới phải (điểm ảnh); x_center y_center width height = kiểu YOLO (0..1).")
    print(f"Xong sau {(time.time() - t0) / 60:.1f} phút. Chạy lại: kéo {folder / 'cau_hinh.json'} thả lên CHAY_LAI_YOLO.bat")


def wizard():
    print(__doc__)
    cfg = {"viec": ask("Việc gì? 1=KHOANH (có sẵn)  2=HỌC khoanh trên dữ liệu đề  3=TÔ VÙNG (có sẵn)  4=HỌC tô vùng (cần data.yaml)", "2")}
    cfg["ten"] = ask("Đặt tên cho lần chạy này (không dấu, không cách)", time.strftime("yolo_%H%M"))
    if cfg["viec"] in ("2", "4"):
        p = ask_path("Kéo thả file data.yaml (định dạng YOLO) HOẶC file csv ghi các hộp vào đây")
        if p.lower().endswith((".yaml", ".yml")):
            cfg["data_yaml"] = p
        else:
            cfg["csv_hop"] = p
            df = read_table(p)
            show_cols(df)
            cfg["cot"] = {"anh": pick_col(df, "Cột nào là TÊN ẢNH?", str(df.columns[0])), "nhan": pick_col(df, "Cột nào là NHÃN (loại vật thể)?")}
            cfg["kieu_toa_do"] = ask("Toạ độ hộp kiểu gì? 1=x1,y1,x2,y2 (góc trên trái, góc dưới phải)  2=x,y,rộng,cao (x,y là góc trên trái)  3=YOLO (tâm, đã chia 0..1)", "1")
            cfg["cot"]["toa_do"] = [pick_col(df, f"Cột toạ độ thứ {i + 1} trong 4 cột ({['x1 / x', 'y1 / y', 'x2 / rộng', 'y2 / cao'][i]})?") for i in range(4)]
            cfg["thu_muc_anh"] = ask_path("Kéo thả THƯ MỤC chứa ảnh train vào đây")
        m = ask("Học tối đa bao nhiêu phút?", "30")
        cfg["phut"] = float(m) if m.replace(".", "").isdigit() else 30
    cfg["co_anh"] = ask("Cỡ ảnh khi chạy (640 = chuẩn; 416 = nhanh hơn, kém hơn chút; vật thể rất nhỏ thì 960)", "640")
    cfg["test"] = ask_path("Kéo thả THƯ MỤC ảnh test vào đây")
    cfg["nguong"] = ask("Ngưỡng tự tin (chỉ giữ hộp có độ tin cậy >= ngưỡng; đề chấm bằng mAP thì để thấp 0.01, đề đếm số lượng thì 0.25)", "0.25")
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
