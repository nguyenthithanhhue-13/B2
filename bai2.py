"""BÀI 2 - bộ khung chung cho bài mới chưa biết đề: hỏi vài câu rồi tự học và ghi file nộp.

Cách chạy:  bấm đúp CHAY_BAI2.bat   (hoặc: python bai2.py)          -> hỏi đáp từng bước
            python bai2.py --lai ket_qua\\<tên>\\cau_hinh.json          -> chạy lại đúng cấu hình cũ (sau khi sửa code)

Làm được 5 kiểu bài:
  1. BẢNG số liệu (csv / xlsx): đoán một cột (phân loại hoặc con số)           LightGBM
  2. VĂN BẢN: phân loại / chấm điểm câu, đoạn văn                               TF-IDF + mô hình nhúng câu có sẵn trong máy
  3. ẢNH: phân loại ảnh                                                          ResNet50 / CLIP / DINOv2 có sẵn + hồi quy logistic,
                                                                                 CLIP đoán không cần học, tuỳ chọn tinh chỉnh MobileNetV3
  4. ÂM THANH: phân loại file tiếng (wav / flac / ogg / mp3)                    phổ mel + LightGBM
  5. KHÔNG CÓ NHÃN: chia nhóm (phân cụm) + vẽ hình 2 chiều                      KMeans + PCA
  6. VIDEO: phân loại đoạn phim (vd ngôn ngữ ký hiệu)                            lấy 8 khung hình mỗi video -> ResNet50 / CLIP
  7. DỊCH: Trung <-> Việt, Anh <-> Việt                                          mô hình dịch opus-mt có sẵn trong máy
Mọi mô hình đều chạy offline (đã tải sẵn trong máy). Kết quả: ket_qua\\<tên>\\submission.csv và bao_cao.txt.
"""
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _st in (sys.stdout, sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "ket_qua"
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
SEED = 0
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
AUD_EXT = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}
MT_MODELS = {"zh-vi": "Helsinki-NLP/opus-mt-zh-vi", "en-vi": "Helsinki-NLP/opus-mt-en-vi", "vi-en": "Helsinki-NLP/opus-mt-vi-en", "en-zh": "Helsinki-NLP/opus-mt-en-zh"}
NLLB = "facebook/nllb-200-distilled-600M"      # mô hình dịch đa ngôn ngữ (615 triệu tham số, chậm hơn nhưng dịch tốt hơn và dịch được mọi cặp)
NLLB_LANG = {"vi": "vie_Latn", "zh": "zho_Hans", "en": "eng_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang", "fr": "fra_Latn", "de": "deu_Latn",
             "th": "tha_Thai", "lo": "lao_Laoo", "km": "khm_Khmr", "ru": "rus_Cyrl", "es": "spa_Latn", "id": "ind_Latn"}
CLIP = "openai/clip-vit-base-patch32"
DINO = "facebook/dinov2-small"      # bộ đọc ảnh tự học (Meta), đặc trưng thường tốt hơn ResNet50 khi chỉ học lớp cuối
TEXT_MODELS = ["intfloat/multilingual-e5-small", "bkai-foundation-models/vietnamese-bi-encoder"]
LOG = []
NL = chr(10)


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ======================= đọc dữ liệu =======================
def clean_path(s):
    p = Path(str(s).strip().strip('"').strip("'")).expanduser()
    if not p.exists() and not p.is_absolute() and (Path.home() / p).exists():      # Jupyter 'Copy Path' tính từ thư mục nhà
        p = Path.home() / p
    return p


def read_table(path):
    p = clean_path(path)
    suf = p.suffix.lower()
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if suf == ".json":
        try:
            return pd.read_json(p)
        except ValueError:
            return pd.read_json(p, lines=True)
    if suf == ".jsonl":
        return pd.read_json(p, lines=True)
    if suf == ".parquet":
        return pd.read_parquet(p)
    if suf == ".tsv":
        return pd.read_csv(p, sep="\t")
    for enc in ("utf-8-sig", "utf-16", "latin-1"):      # csv / txt: thử dấu phẩy trước, rồi ; tab |
        try:
            df = pd.read_csv(p, encoding=enc)
        except UnicodeError:
            continue
        if df.shape[1] == 1:
            head = str(df.columns[0])
            for sep in (";", "	", "|"):
                if sep in head:
                    df = pd.read_csv(p, encoding=enc, sep=sep)
                    break
        return df
    raise RuntimeError(f"Không đọc được file {p}")


def list_files(folder, exts):
    folder = clean_path(folder)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def folder_to_table(folder, exts, labeled):
    """Thư mục -> bảng (duong_dan, nhan). labeled=True: mỗi thư mục con là một lớp."""
    folder = clean_path(folder)
    rows = []
    for p in list_files(folder, exts):
        rel = p.relative_to(folder)
        rows.append({"id": rel.as_posix() if len(rel.parts) > 1 and not labeled else p.name, "duong_dan": str(p),
                     **({"nhan": rel.parts[0]} if labeled and len(rel.parts) > 1 else {})})
    return pd.DataFrame(rows)


def guess_task(y):
    yy = pd.Series(y).dropna()
    if not pd.api.types.is_numeric_dtype(yy):
        return "phanloai"
    return "phanloai" if (yy.nunique() <= 20 and np.allclose(yy, yy.round())) else "hoiquy"


# ======================= học + chấm chéo =======================
def cv_splits(y, task, n=5):
    from sklearn.model_selection import KFold, StratifiedKFold
    if task == "phanloai":
        m = int(pd.Series(y).value_counts().min())
        if m >= 2:
            return list(StratifiedKFold(min(n, m), shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    return list(KFold(min(n, len(y)), shuffle=True, random_state=SEED).split(np.zeros(len(y))))


def make_models(task, kind, n_classes):
    """kind: 'bang' (DataFrame có cột loại), 'day' (ma trận đặc), 'thua' (ma trận thưa TF-IDF)."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    M = {}
    lgb_kw = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  random_state=SEED, verbose=-1, n_jobs=max(1, (os.cpu_count() or 4) - 2))
    if task == "phanloai":
        if kind in ("bang", "day"):
            M["LightGBM"] = lambda: lgb.LGBMClassifier(**lgb_kw)
        if kind in ("day", "tuyentinh"):
            M["HồiQuyLogistic"] = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
        if kind == "thua":
            M["HồiQuyLogistic"] = lambda: LogisticRegression(C=10.0, max_iter=3000)
    else:
        if kind in ("bang", "day"):
            M["LightGBM"] = lambda: lgb.LGBMRegressor(**lgb_kw)
        if kind in ("day", "tuyentinh"):
            M["Ridge"] = lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        if kind == "thua":
            M["Ridge"] = lambda: Ridge(alpha=1.0)
    return M


def score(task, y, pred):
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
    if task == "phanloai":
        lab = pred.argmax(1) if pred.ndim == 2 else pred
        return {"accuracy": float(accuracy_score(y, lab)), "f1_macro": float(f1_score(y, lab, average="macro"))}
    return {"rmse": float(np.sqrt(mean_squared_error(y, pred))), "mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred))}


def take(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def fit_predict(feature_sets, y, task, metric=None):
    """feature_sets: {tên: (Xtrain, Xtest, kind)}. Chấm chéo mọi (đặc trưng x mô hình), thêm 'trung bình', chọn cái tốt nhất,
    học lại trên toàn bộ train rồi đoán test. Trả về (dự đoán test, xác suất test hoặc None, bảng điểm, tên được chọn)."""
    classes = None
    if task == "phanloai":
        classes, y = np.unique(y, return_inverse=True)
    else:
        y = np.asarray(y, dtype=float)
    n_classes = len(classes) if classes is not None else 0
    metric = metric or ("accuracy" if task == "phanloai" else "rmse")
    better = (lambda a, b: a < b) if metric in ("rmse", "mae") else (lambda a, b: a > b)
    splits = cv_splits(y, task)
    oof, test_pred, board = {}, {}, {}
    for fname, (Xtr, Xte, kind) in feature_sets.items():
        if kind == "san":      # xác suất đã có sẵn (vd CLIP đoán không cần học): chỉ chấm điểm, không học gì
            if task == "phanloai":
                oof[fname], test_pred[fname], board[fname] = Xtr, Xte, score(task, y, Xtr)
                say(f"   {fname:52s} " + "  ".join(f"{k}={v:.4f}" for k, v in board[fname].items()))
            continue
        for mname, make in make_models(task, kind, n_classes).items():
            name = f"{fname} + {mname}"
            t = time.time()
            try:
                o = np.zeros((len(y), n_classes)) if task == "phanloai" else np.zeros(len(y))
                for tr, va in splits:
                    m = make().fit(take(Xtr, tr), y[tr])
                    if task == "phanloai":
                        p = m.predict_proba(take(Xtr, va))
                        o[np.ix_(va, m.classes_)] = p
                    else:
                        o[va] = m.predict(take(Xtr, va))
                m = make().fit(Xtr, y)
                if task == "phanloai":
                    tp = np.zeros((Xte.shape[0], n_classes))
                    tp[:, m.classes_] = m.predict_proba(Xte)
                else:
                    tp = m.predict(Xte)
                oof[name], test_pred[name], board[name] = o, tp, score(task, y, o)
                say(f"   {name:52s} " + "  ".join(f"{k}={v:.4f}" for k, v in board[name].items()) + f"   ({time.time() - t:.0f}s)")
            except Exception as e:
                say(f"   {name}: lỗi, bỏ qua ({type(e).__name__}: {str(e)[:120]})")
    if not oof:
        raise RuntimeError("Không mô hình nào học được. Xem các dòng lỗi phía trên.")
    if len(oof) >= 2:
        top = sorted(oof, key=lambda k: board[k][metric], reverse=metric not in ("rmse", "mae"))[:3]
        name = "TRUNG BÌNH " + str(len(top)) + " mô hình tốt nhất"
        oof[name] = sum(oof[k] for k in top) / len(top)
        test_pred[name] = sum(test_pred[k] for k in top) / len(top)
        board[name] = score(task, y, oof[name])
        say(f"   {name:52s} " + "  ".join(f"{k}={v:.4f}" for k, v in board[name].items()))
    best = None
    for k in board:
        if best is None or better(board[k][metric], board[best][metric]):
            best = k
    say(f"=> CHỌN: {best}   (điểm tự chấm {metric} = {board[best][metric]:.4f}, chấm chéo {len(splits)} phần)")
    tp = test_pred[best]
    fit_predict.last = (best, board[best][metric], metric)      # để phần tinh chỉnh so sánh
    if task == "phanloai":
        return classes[tp.argmax(1)], pd.DataFrame(tp, columns=[str(c) for c in classes]), board, best
    return tp, None, board, best


# ======================= đặc trưng: BẢNG =======================
def table_features(train, test, drop):
    tr, te = train.drop(columns=[c for c in drop if c in train.columns]), test.drop(columns=[c for c in drop if c in test.columns])
    cols = [c for c in tr.columns if c in te.columns]
    A, B = pd.DataFrame(index=tr.index), pd.DataFrame(index=te.index)
    notes = []
    for c in cols:
        a, b = tr[c], te[c]
        if pd.api.types.is_bool_dtype(a):
            A[c], B[c] = a.astype(float), b.astype(float)
        elif pd.api.types.is_numeric_dtype(a):
            A[c], B[c] = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
        else:
            a, b = a.astype("string"), b.astype("string")
            d = pd.to_datetime(a, errors="coerce")
            if d.notna().mean() > 0.8 and a.str.len().median() >= 6:      # cột ngày giờ -> tách phần
                db = pd.to_datetime(b, errors="coerce")
                for part in ("year", "month", "day", "dayofweek", "hour"):
                    A[f"{c}_{part}"], B[f"{c}_{part}"] = getattr(d.dt, part), getattr(db.dt, part)
                notes.append(f"{c}: ngày giờ")
            elif a.nunique() <= max(50, 0.05 * len(a)):                   # cột loại
                cats = pd.CategoricalDtype(sorted(set(a.dropna()) | set(b.dropna())))
                A[c], B[c] = a.astype(object).astype(cats), b.astype(object).astype(cats)
                notes.append(f"{c}: loại ({a.nunique()} giá trị)")
            else:                                                          # cột chữ tự do -> TF-IDF rút gọn (chỉ học trên train)
                from sklearn.decomposition import TruncatedSVD
                from sklearn.feature_extraction.text import TfidfVectorizer
                v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=30000, sublinear_tf=True)
                Xa, Xb = v.fit_transform(a.fillna("")), v.transform(b.fillna(""))
                k = int(min(24, Xa.shape[1] - 1, len(a) - 1))
                if k >= 2:
                    s = TruncatedSVD(k, random_state=SEED).fit(Xa)
                    Za, Zb = s.transform(Xa), s.transform(Xb)
                    for j in range(k):
                        A[f"{c}_chu{j}"], B[f"{c}_chu{j}"] = Za[:, j], Zb[:, j]
                A[f"{c}_dai"], B[f"{c}_dai"] = a.str.len().astype(float), b.str.len().astype(float)
                notes.append(f"{c}: chữ tự do")
    A.columns = [str(c).replace('"', "").replace(":", "_").replace(",", "_").replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")") for c in A.columns]
    B.columns = A.columns
    say(f"   {A.shape[1]} đặc trưng từ {len(cols)} cột. " + ("; ".join(notes[:12]) if notes else "tất cả là số"))
    return A, B


# ======================= đặc trưng: VĂN BẢN =======================
def cached(key, fn):
    p = CACHE / (hashlib.md5(key.encode("utf-8")).hexdigest() + ".npy")
    if p.exists():
        return np.load(p)
    v = fn()
    np.save(p, v)
    return v


def text_embed(texts, model_name):
    key = model_name + "|" + hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()

    def run():
        import torch
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        pre = "query: " if "e5" in model_name else ""
        return m.encode([pre + t for t in texts], batch_size=32, show_progress_bar=len(texts) > 500, normalize_embeddings=True)

    return np.asarray(cached(key, run), dtype=np.float32)


def text_features(tr_texts, te_texts):
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    F = {}
    w = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_features=200000)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, max_features=300000)
    F["TF-IDF"] = (hstack([w.fit_transform(tr_texts), c.fit_transform(tr_texts)]).tocsr(),
                   hstack([w.transform(te_texts), c.transform(te_texts)]).tocsr(), "thua")
    for name in TEXT_MODELS:
        try:
            t = time.time()
            F["nhúng " + name.split("/")[-1]] = (text_embed(tr_texts, name), text_embed(te_texts, name), "day")
            say(f"   đã nhúng câu bằng {name} ({time.time() - t:.0f}s)")
        except Exception as e:
            say(f"   không dùng được mô hình {name} ({type(e).__name__}: {str(e)[:100]})")
    return F


# ======================= đặc trưng: ẢNH =======================
def frames_of(path, n=8):
    """Lấy n khung hình cách đều của một video (ảnh PIL)."""
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    if total > 0:
        for i in np.linspace(0, max(0, total - 1), n).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok:
                out.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    if not out:      # không biết số khung: đọc hết rồi lấy đều
        cap.release()
        cap = cv2.VideoCapture(str(path))
        allf = []
        while len(allf) < 2000:
            ok, fr = cap.read()
            if not ok:
                break
            allf.append(fr)
        if allf:
            out = [Image.fromarray(cv2.cvtColor(allf[i], cv2.COLOR_BGR2RGB)) for i in np.linspace(0, len(allf) - 1, min(n, len(allf))).astype(int)]
    cap.release()
    return out


def image_embed(paths, which, video=False):
    key = which + ("|video" if video else "") + "|" + hashlib.md5("\n".join(f"{p}:{Path(p).stat().st_size}" for p in paths).encode("utf-8")).hexdigest()

    def run():
        import torch
        from PIL import Image
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        out = []
        if which == "resnet50":
            import torchvision.models as M
            W = M.ResNet50_Weights.DEFAULT
            net = M.resnet50(weights=W)
            net.fc = torch.nn.Identity()
            net.eval().to(dev)
            prep = W.transforms()
            fwd = lambda ims: net(torch.stack([prep(im) for im in ims]).to(dev))
        elif which == "clip":
            from transformers import CLIPModel, CLIPProcessor
            net = CLIPModel.from_pretrained(CLIP).eval().to(dev)
            proc = CLIPProcessor.from_pretrained(CLIP)

            def fwd(ims):
                o = net.get_image_features(**{k: v.to(dev) for k, v in proc(images=ims, return_tensors="pt").items()})
                return o if isinstance(o, torch.Tensor) else o.pooler_output
        else:      # dinov2: nối token [CLS] với trung bình các mảnh ảnh
            from transformers import AutoImageProcessor, AutoModel
            net = AutoModel.from_pretrained(DINO).eval().to(dev)
            proc = AutoImageProcessor.from_pretrained(DINO)

            def fwd(ims):
                h = net(**{k: v.to(dev) for k, v in proc(images=ims, return_tensors="pt").items()}).last_hidden_state
                return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)
        with torch.no_grad():
            step = 4 if video else 32
            for i in range(0, len(paths), step):
                groups = []
                for p in paths[i:i + step]:
                    try:
                        g = frames_of(p) if video else [Image.open(p).convert("RGB")]
                    except Exception:
                        g = []
                    groups.append(g or [Image.new("RGB", (224, 224))])
                flat = [im for g in groups for im in g]
                f = np.concatenate([fwd(flat[j:j + 32]).float().cpu().numpy() for j in range(0, len(flat), 32)])
                pos = 0
                for g in groups:      # video: trung bình đặc trưng của các khung hình
                    out.append(f[pos:pos + len(g)].mean(0))
                    pos += len(g)
                if (i // step) % 10 == 0:
                    print(f"      {which}: {min(i + step, len(paths))}/{len(paths)} {'video' if video else 'ảnh'}", flush=True)
        return np.stack(out)

    return np.asarray(cached(key, run), dtype=np.float32)


def clip_zero_shot(img_tr, img_te, classes):
    """CLIP đoán nhãn chỉ từ TÊN nhãn (không học từ train): so ảnh với câu 'a photo of a <nhãn>'."""
    import torch
    from transformers import CLIPModel, CLIPTokenizer
    net = CLIPModel.from_pretrained(CLIP).eval()
    tok = CLIPTokenizer.from_pretrained(CLIP)
    names = [str(c).replace("_", " ").replace("-", " ") for c in classes]
    with torch.no_grad():
        t = net.get_text_features(**tok([f"a photo of a {n}" for n in names], padding=True, return_tensors="pt"))
        t = (t if isinstance(t, torch.Tensor) else t.pooler_output).numpy()
    t /= np.linalg.norm(t, axis=1, keepdims=True)

    def probs(F):
        F = F / np.linalg.norm(F, axis=1, keepdims=True)
        z = 100.0 * F @ t.T
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)
    return probs(img_tr), probs(img_te)


def image_features(tr_paths, te_paths, video=False, y=None):
    F = {}
    for which in ("resnet50", "clip", "dinov2"):
        try:
            t = time.time()
            F[which] = (image_embed(tr_paths, which, video), image_embed(te_paths, which, video), "day")
            say(f"   đã rút đặc trưng ảnh bằng {which} ({time.time() - t:.0f}s)")
        except Exception as e:
            say(f"   không dùng được {which} ({type(e).__name__}: {str(e)[:100]})")
    if len(F) >= 2:
        ks = list(F)
        F["+".join(ks)] = (np.hstack([F[k][0] for k in ks]), np.hstack([F[k][1] for k in ks]), "day")
    if "clip" in F and y is not None and not video:
        try:
            F["CLIP đoán theo tên nhãn (không học)"] = (*clip_zero_shot(F["clip"][0], F["clip"][1], np.unique(y)), "san")
        except Exception as e:
            say(f"   bỏ qua CLIP đoán theo tên nhãn ({type(e).__name__}: {str(e)[:100]})")
    return F


def finetune_images(trp, y, tep, minutes=20, epochs=8):
    """Tinh chỉnh cả mạng MobileNetV3 (đã học sẵn trên ImageNet) trên ảnh của đề, trên CPU.
    Giữ lại 15% ảnh train để tự chấm. Dừng khi hết số vòng hoặc hết 'minutes' phút.
    Trả về (điểm accuracy trên phần giữ lại, xác suất test [n_test x n_lớp], các lớp)."""
    import torch
    import torchvision.models as M
    import torchvision.transforms as T
    from PIL import Image
    from sklearn.model_selection import train_test_split
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    classes, yi = np.unique(y, return_inverse=True)
    idx = np.arange(len(yi))
    strat = yi if pd.Series(yi).value_counts().min() >= 2 else None
    tr_i, va_i = train_test_split(idx, test_size=0.15, random_state=SEED, stratify=strat)
    keep = len(trp) + len(tep) <= 12000      # ít ảnh thì giữ sẵn trong bộ nhớ (đã thu nhỏ) cho nhanh

    def load(p):
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            im = Image.new("RGB", (256, 256))
        im.thumbnail((320, 320))
        return im
    mem = {p: load(p) for p in set(trp)} if keep else None
    get = (lambda p: mem[p]) if keep else load
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    aug = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), norm])
    plain = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = M.mobilenet_v3_large(weights=M.MobileNet_V3_Large_Weights.DEFAULT)
    net.classifier[3] = torch.nn.Linear(net.classifier[3].in_features, len(classes))
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    def predict(paths, tta=True):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(paths), 64):
                x = torch.stack([plain(get(p) if keep and p in mem else load(p)) for p in paths[i:i + 64]]).to(dev)
                p = net(x).softmax(1)
                if tta:      # lật ngang rồi lấy trung bình
                    p = (p + net(torch.flip(x, dims=[3])).softmax(1)) / 2
                out.append(p.cpu().numpy())
        return np.concatenate(out)

    t0, best, best_state = time.time(), -1.0, None
    rng = np.random.default_rng(SEED)
    for ep in range(epochs):
        net.train()
        order = rng.permutation(tr_i)
        for i in range(0, len(order), 32):
            b = order[i:i + 32]
            if len(b) < 2:
                continue
            x = torch.stack([aug(get(trp[j])) for j in b]).to(dev)
            loss = lossf(net(x), torch.as_tensor(yi[b]).to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        acc = float((predict([trp[j] for j in va_i], tta=False).argmax(1) == yi[va_i]).mean())
        el = (time.time() - t0) / 60
        say(f"   tinh chỉnh vòng {ep + 1}/{epochs}: accuracy phần giữ lại = {acc:.4f}   ({el:.1f} phút)")
        if acc > best:
            best, best_state = acc, {k: v.detach().clone() for k, v in net.state_dict().items()}
        if el / (ep + 1) * (ep + 2) > minutes:
            say(f"   dừng sớm cho kịp giờ (giới hạn {minutes} phút)")
            break
    net.load_state_dict(best_state)
    return best, predict(tep), classes


# ======================= đặc trưng: ÂM THANH =======================
def mel_bank(sr, n_fft, n_mels=40):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(20), hz2mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        a, b, c = bins[m - 1], bins[m], bins[m + 1]
        if b > a:
            fb[m - 1, a:b] = (np.arange(a, b) - a) / (b - a)
        if c > b:
            fb[m - 1, b:c] = (c - np.arange(b, c)) / (c - b)
    return fb


def audio_feats_one(path, sr=16000):
    import soundfile as sf
    from scipy.signal import resample_poly, stft
    x, r = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    if r != sr:
        g = np.gcd(int(r), sr)
        x = resample_poly(x, sr // g, int(r) // g)
    if len(x) < 1024:
        x = np.pad(x, (0, 1024 - len(x)))
    _, _, Z = stft(x, fs=sr, nperseg=400, noverlap=240, nfft=512)
    S = np.log(mel_bank(sr, 512) @ (np.abs(Z) ** 2) + 1e-6)          # log-mel (40, T)
    d = np.diff(S, axis=1) if S.shape[1] > 1 else np.zeros_like(S)
    return np.concatenate([S.mean(1), S.std(1), S.max(1), S.min(1), d.std(1),
                           [len(x) / sr, float(np.sqrt((x ** 2).mean())), float((np.abs(np.diff(np.sign(x))) > 0).mean())]])


def audio_features(tr_paths, te_paths):
    def run(paths):
        out, bad = [], 0
        for i, p in enumerate(paths):
            try:
                out.append(audio_feats_one(p))
            except Exception:
                out.append(np.zeros(203))
                bad += 1
            if i % 200 == 0:
                print(f"      âm thanh: {i + 1}/{len(paths)}", flush=True)
        if bad:
            say(f"   !!! {bad}/{len(paths)} file âm thanh không đọc được (coi như im lặng)")
        return np.array(out, dtype=np.float32)

    return {"phổ mel": (run(tr_paths), run(te_paths), "day")}


# ======================= ghi file nộp =======================
def write_submission(folder, ids, pred, proba, id_col, target_col, sample_path):
    folder.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({id_col or "id": list(ids), target_col or "prediction": list(pred)})
    how = "tự tạo (cột id + cột dự đoán)"
    if sample_path:
        s = read_table(sample_path)
        common = [c for c in s.columns if c == (id_col or "id")] or [s.columns[0]]
        key, rest = common[0], [c for c in s.columns if c != common[0]]
        mp = dict(zip([str(i) for i in ids], range(len(ids))))
        order = [mp.get(str(v)) for v in s[key]]
        if any(o is None for o in order):      # thử khớp theo tên file bỏ đuôi (vd 't0.png' với 't0')
            stem = lambda v: Path(str(v)).stem if Path(str(v)).suffix.lower() in IMG_EXT | AUD_EXT | VID_EXT else str(v)
            mp2 = {stem(i): k for k, i in enumerate(ids)}
            o2 = [mp2.get(stem(v)) for v in s[key]]
            if sum(o is not None for o in o2) > sum(o is not None for o in order):
                order = o2
        if any(o is None for o in order):
            if len(s) == len(ids):
                order = list(range(len(ids)))
                how = "theo file mẫu, ghép THEO THỨ TỰ DÒNG (id không khớp - hãy kiểm tra lại!)"
            else:
                order = None
        else:
            how = "theo file mẫu, ghép theo id"
        if order is not None and rest:
            out = s.copy()
            if proba is not None and len(rest) > 1 and all(str(c) in proba.columns for c in rest):
                for c in rest:
                    out[c] = proba[str(c)].to_numpy()[order]
                how += " (điền xác suất từng lớp)"
            else:
                vals = np.asarray(list(pred), dtype=object)[order]
                if proba is not None and proba.shape[1] == 2 and pd.api.types.is_float_dtype(s[rest[0]]) and not np.allclose(s[rest[0]].dropna(), s[rest[0]].dropna().round()):
                    vals = proba.iloc[:, 1].to_numpy()[order]
                    how += " (điền xác suất lớp dương)"
                out[rest[0]] = vals
            sub = out
    p = folder / "submission.csv"
    sub.to_csv(p, index=False, encoding="utf-8")
    if proba is not None:
        pr = proba.copy()
        pr.insert(0, id_col or "id", list(ids))
        pr.to_csv(folder / "xac_suat.csv", index=False, encoding="utf-8")
    say(f"\n>>> FILE NỘP: {p}   ({len(sub)} dòng, định dạng: {how})")
    say("    5 dòng đầu:\n" + sub.head().to_string(index=False))
    return p


# ======================= các kiểu bài =======================
def run(cfg):
    kind = cfg["kieu"]
    name = cfg.get("ten") or time.strftime("lan_%H%M")
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "cau_hinh.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    t0 = time.time()
    if kind == "dich":
        return run_translate(cfg, folder)
    if kind == "cum":
        return run_cluster(cfg, folder)
    exts = {"anh": IMG_EXT, "video": VID_EXT}.get(kind, AUD_EXT)
    # ---- nạp bảng train / test ----
    if kind in ("anh", "amthanh", "video") and clean_path(cfg["train"]).is_dir():
        train = folder_to_table(cfg["train"], exts, labeled=True)
        cfg.setdefault("cot_duong_dan", "duong_dan"), cfg.setdefault("cot_dich", "nhan")
        if "nhan" not in train.columns:
            raise RuntimeError("Thư mục train phải có thư mục con cho từng lớp (vd train\\meo\\..., train\\cho\\...).")
    else:
        train = read_table(cfg["train"])
    if kind in ("anh", "amthanh", "video") and clean_path(cfg["test"]).is_dir():
        test = folder_to_table(cfg["test"], exts, labeled=False)
        cfg.setdefault("cot_id", "id")
    else:
        test = read_table(cfg["test"])
    target, idc = cfg["cot_dich"], cfg.get("cot_id")
    train = train[train[target].notna()].reset_index(drop=True)
    y = train[target].to_numpy()
    task = cfg.get("bai_toan") or guess_task(y)
    ids = test[idc] if idc and idc in test.columns else pd.Series(range(len(test)), name="id")
    say(f"Kiểu bài: {kind} | train {train.shape} | test {test.shape} | cột cần đoán: {target} | bài toán: {'PHÂN LOẠI' if task == 'phanloai' else 'ĐOÁN CON SỐ'}")
    if task == "phanloai":
        vc = pd.Series(y).value_counts()
        say("Số mẫu mỗi lớp: " + ", ".join(f"{k}={v}" for k, v in vc.head(15).items()) + (" ..." if len(vc) > 15 else ""))
    # ---- đặc trưng ----
    if kind == "bang":
        A, B = table_features(train, test, [target] + ([idc] if idc else []) + list(cfg.get("bo_cot") or []))
        F = {"bảng": (A, B, "bang")}
        try:      # bản "tuyến tính": điền chỗ trống bằng trung vị, cột loại -> one-hot (hợp với dữ liệu ít dòng)
            num = [c for c in A.columns if str(A[c].dtype) != "category"]
            cat = [c for c in A.columns if c not in num and A[c].cat.categories.size <= 30]
            med = A[num].median()
            La = pd.concat([A[num].fillna(med).fillna(0)] + [pd.get_dummies(A[c], prefix=str(c), dtype=float) for c in cat], axis=1)
            Lb = pd.concat([B[num].fillna(med).fillna(0)] + [pd.get_dummies(B[c], prefix=str(c), dtype=float) for c in cat], axis=1)
            F["bảng (tuyến tính)"] = (La.to_numpy(dtype=float), Lb.reindex(columns=La.columns, fill_value=0).to_numpy(dtype=float), "tuyentinh")
        except Exception as e:
            say(f"   (bỏ qua bản tuyến tính: {type(e).__name__})")
    elif kind == "vanban":
        cols = cfg["cot_van_ban"]
        join = lambda d: d[cols].fillna("").astype(str).agg(" [SEP] ".join, axis=1).tolist()
        F = text_features(join(train), join(test))
    else:
        pc = cfg["cot_duong_dan"]
        base = clean_path(cfg["thu_muc_goc"]) if cfg.get("thu_muc_goc") else None

        def full(d, src):
            root = base or (clean_path(src) if clean_path(src).is_dir() else clean_path(src).parent)
            out = []
            for v in d[pc].astype(str):
                p = Path(v)
                out.append(str(p if p.is_absolute() else root / p))
            return out

        trp, tep = full(train, cfg["train"]), full(test, cfg["test"])
        miss = [p for p in trp + tep if not Path(p).exists()]
        if miss:
            raise RuntimeError(f"Không thấy {len(miss)} file (vd {miss[0]}). Kiểm tra lại 'thư mục gốc' của đường dẫn.")
        F = image_features(trp, tep, video=(kind == "video"), y=y if task == "phanloai" else None) if kind in ("anh", "video") else audio_features(trp, tep)
    pred, proba, board, best = fit_predict(F, y, task, cfg.get("chi_so"))
    write_submission(folder, ids, pred, proba, idc, target, cfg.get("mau_nop"))
    if kind == "anh" and task == "phanloai" and cfg.get("tinh_chinh"):
        b_name, b_score, b_metric = fit_predict.last
        say(f"\nTINH CHỈNH MobileNetV3 (tối đa {cfg.get('phut_tinh_chinh', 20)} phút). Bản đang có: {b_name}, {b_metric} = {b_score:.4f}")
        try:
            acc, tp, cl = finetune_images(trp, y, tep, minutes=float(cfg.get("phut_tinh_chinh", 20)))
            pr = pd.DataFrame(tp, columns=[str(c) for c in cl])
            say(f"   tinh chỉnh: accuracy phần giữ lại = {acc:.4f}")
            write_submission(folder / "tinh_chinh", ids, cl[tp.argmax(1)], pr, idc, target, cfg.get("mau_nop"))
            if proba is not None and list(proba.columns) == list(pr.columns):
                mix = (proba + pr) / 2
                write_submission(folder / "tron_ca_hai", ids, cl[mix.to_numpy().argmax(1)], mix, idc, target, cfg.get("mau_nop"))
            if acc > b_score + 0.01:
                say(f">>> Tinh chỉnh TỐT HƠN ({b_score:.4f} -> {acc:.4f}). Nên nộp: {folder / 'tinh_chinh' / 'submission.csv'}")
                say(f"    Còn lượt nộp thì thử thêm bản trộn hai cách: {folder / 'tron_ca_hai' / 'submission.csv'}")
            else:
                say(f">>> Tinh chỉnh KHÔNG hơn rõ ({b_score:.4f} -> {acc:.4f}). Giữ file nộp chính: {folder / 'submission.csv'}")
                say(f"    Còn lượt nộp thì có thể thử bản trộn hai cách: {folder / 'tron_ca_hai' / 'submission.csv'}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            say(f"   tinh chỉnh lỗi, bỏ qua ({type(e).__name__}: {str(e)[:150]}). File nộp chính vẫn dùng được.")
    say(f"Xong sau {(time.time() - t0) / 60:.1f} phút. Cấu hình đã lưu: {folder / 'cau_hinh.json'}")
    (folder / "bao_cao.txt").write_text("\n".join(LOG), encoding="utf-8")


def translate(texts, direction, batch=8, engine="opus"):
    """engine: 'opus' = mô hình nhỏ, nhanh (chỉ zh-vi, en-vi, vi-en, en-zh; vi-zh đi vòng qua tiếng Anh);
               'nllb' = mô hình đa ngôn ngữ, chậm hơn khoảng 3-5 lần, thường dịch tốt hơn."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a, b = direction.split("-")
    gen_kw = {}
    if engine == "nllb" or (direction not in MT_MODELS and direction != "vi-zh"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(NLLB, src_lang=NLLB_LANG[a])
        net = AutoModelForSeq2SeqLM.from_pretrained(NLLB).eval().to(dev)
        gen_kw = {"forced_bos_token_id": tok.convert_tokens_to_ids(NLLB_LANG[b])}
        batch = min(batch, 4)
    elif direction == "vi-zh":      # không có mô hình nhỏ dịch thẳng Việt -> Trung: dịch Việt -> Anh -> Trung
        return translate(translate(texts, "vi-en", batch), "en-zh", batch)
    else:
        from transformers import MarianMTModel, MarianTokenizer
        tok = MarianTokenizer.from_pretrained(MT_MODELS[direction])
        net = MarianMTModel.from_pretrained(MT_MODELS[direction]).eval().to(dev)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))      # câu dài ngắn gần nhau vào cùng lô cho nhanh
    out = [""] * len(texts)
    t = time.time()
    with torch.no_grad():
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            enc = tok([texts[i] if texts[i].strip() else "." for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)
            # chặn dịch lặp vô tận: giới hạn độ dài theo câu nguồn + cấm lặp cụm 4 từ
            gen = net.generate(**enc, num_beams=4, max_new_tokens=int(min(256, 2 * enc["input_ids"].shape[1] + 16)), no_repeat_ngram_size=4, **gen_kw)
            for i, o in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
                out[i] = o if texts[i].strip() else ""
            if (b // batch) % 10 == 0:
                print(f"      đã dịch {min(b + batch, len(order))}/{len(order)} câu ({time.time() - t:.0f}s)", flush=True)
    return out


def run_translate(cfg, folder):
    direction = cfg["huong"]
    test = read_table(cfg["test"])
    src = cfg["cot_nguon"]
    idc = cfg.get("cot_id")
    texts = test[src].fillna("").astype(str).tolist()
    engine = cfg.get("may_dich", "opus")
    say(f"Kiểu bài: dịch {direction} | test {test.shape} | cột nguồn: {src} | máy dịch: {engine}")
    memo = {}
    tcol = cfg.get("cot_dich")
    if cfg.get("train"):
        tr = read_table(cfg["train"])
        if src in tr.columns and tcol in tr.columns:
            memo = dict(zip(tr[src].astype(str).str.strip(), tr[tcol].astype(str)))
            try:      # chấm thử mô hình dịch trên tối đa 100 câu train
                import sacrebleu
                sm = tr.dropna(subset=[src, tcol]).sample(min(100, len(tr)), random_state=SEED)
                hyp = translate(sm[src].astype(str).tolist(), direction, engine=engine)
                ref = sm[tcol].astype(str).tolist()
                tk = "zh" if direction.endswith("zh") else "13a"
                say(f"   điểm tự chấm trên {len(sm)} câu train: BLEU = {sacrebleu.corpus_bleu(hyp, [ref], tokenize=tk).score:.1f}   chrF = {sacrebleu.corpus_chrf(hyp, [ref]).score:.1f}")
                for a, b, c in list(zip(sm[src].astype(str), hyp, ref))[:3]:
                    say(f"      nguồn: {a[:80]}")
                    say(f"      máy  : {b[:80]}")
                    say(f"      đúng : {c[:80]}")
            except Exception as e:
                say(f"   (không chấm thử được: {type(e).__name__}: {str(e)[:100]})")
    need = [i for i, t in enumerate(texts) if t.strip() not in memo]
    out = [memo.get(t.strip(), "") for t in texts]
    if memo:
        say(f"   {len(texts) - len(need)} câu test có sẵn bản dịch trong train (dùng luôn), còn {len(need)} câu phải dịch")
    for i, o in zip(need, translate([texts[i] for i in need], direction, engine=engine)):
        out[i] = o
    ids = test[idc] if idc and idc in test.columns else pd.Series(range(len(test)), name="id")
    write_submission(folder, ids, out, None, idc, tcol or "ban_dich", cfg.get("mau_nop"))
    (folder / "bao_cao.txt").write_text(NL.join(LOG), encoding="utf-8")


def run_cluster(cfg, folder):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    df = read_table(cfg["train"])
    idc = cfg.get("cot_id")
    if cfg.get("cot_van_ban"):
        X = text_embed(df[cfg["cot_van_ban"]].fillna("").astype(str).agg(" ".join, axis=1).tolist(), TEXT_MODELS[0])
    else:
        A, _ = table_features(df, df.head(1), [idc] if idc else [])
        for c in A.columns:
            if str(A[c].dtype) == "category":
                A[c] = A[c].cat.codes.astype(float)
        X = StandardScaler().fit_transform(A.fillna(A.median(numeric_only=True)).fillna(0).to_numpy(dtype=float))
    k = cfg.get("so_nhom")
    if not k:
        best = (-2, 2)
        for kk in range(2, min(11, len(X))):
            lab = KMeans(kk, n_init=5, random_state=SEED).fit_predict(X)
            s = silhouette_score(X, lab, sample_size=min(3000, len(X)), random_state=SEED)
            say(f"   {kk} nhóm: silhouette = {s:.3f}")
            best = max(best, (s, kk))
        k = best[1]
        say(f"=> chọn {k} nhóm")
    lab = KMeans(int(k), n_init=10, random_state=SEED).fit_predict(X)
    out = pd.DataFrame({idc or "id": df[idc] if idc else range(len(df)), "nhom": lab})
    out.to_csv(folder / "submission.csv", index=False, encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Z = PCA(2, random_state=SEED).fit_transform(X)
        plt.figure(figsize=(7, 6))
        plt.scatter(Z[:, 0], Z[:, 1], c=lab, s=8, cmap="tab10")
        plt.title(f"{k} nhom (PCA 2 chieu)")
        plt.savefig(folder / "hinh_nhom.png", dpi=120)
    except Exception:
        pass
    say(f"\n>>> FILE KẾT QUẢ: {folder / 'submission.csv'}   (cột 'nhom' là số thứ tự nhóm; hình: hinh_nhom.png)")
    say("Số mẫu mỗi nhóm: " + ", ".join(f"{a}={b}" for a, b in pd.Series(lab).value_counts().sort_index().items()))
    (folder / "bao_cao.txt").write_text("\n".join(LOG), encoding="utf-8")


# ======================= hỏi đáp =======================
def ask(q, default=None):
    s = input(f"{q}" + (f" [{default}]" if default not in (None, "") else "") + ": ").strip().strip('"')
    return s if s else (default if default is not None else "")


def ask_path(q, allow_empty=False):
    """Hỏi một đường dẫn (kéo thả vào cửa sổ); hỏi lại cho tới khi file / thư mục tồn tại."""
    while True:
        s = ask(q, "" if allow_empty else None)
        if not s and allow_empty:
            return None
        try:
            if s and clean_path(s).exists():
                return str(clean_path(s))
        except OSError:
            pass
        print("   Không thấy file / thư mục đó. Kéo thả lại (hoặc dán đường dẫn) rồi Enter.")


def pick_col(df, q, default=None, allow_empty=False):
    while True:
        s = ask(q + (" (Enter = bỏ qua)" if allow_empty and not default else ""), default)
        if not s and allow_empty:
            return None
        if s.isdigit() and int(s) < len(df.columns):
            return df.columns[int(s)]
        if s in df.columns:
            return s
        print("   Không có cột đó. Gõ SỐ THỨ TỰ hoặc đúng TÊN cột.")


def show_cols(df):
    print(f"   Bảng có {len(df)} dòng, {len(df.columns)} cột:")
    for i, c in enumerate(df.columns):
        print(f"     {i:2d}. {str(c):28s} vd: {str(df[c].dropna().iloc[0])[:50] if df[c].notna().any() else '(trống)'}   ({df[c].nunique()} giá trị khác nhau)")


def wizard():
    print(__doc__)
    k = ask("Kiểu bài? 1=BẢNG  2=VĂN BẢN  3=ẢNH  4=ÂM THANH  5=KHÔNG CÓ NHÃN (chia nhóm)  6=VIDEO  7=DỊCH", "1")
    cfg = {"kieu": {"1": "bang", "2": "vanban", "3": "anh", "4": "amthanh", "5": "cum", "6": "video", "7": "dich"}.get(k, "bang")}
    cfg["ten"] = ask("Đặt tên cho lần chạy này (không dấu, không cách)", time.strftime("lan_%H%M"))
    media = cfg["kieu"] in ("anh", "amthanh", "video")
    if cfg["kieu"] == "dich":
        h = ask("Dịch hướng nào? 1=Trung->Việt  2=Việt->Trung  3=Anh->Việt  4=Việt->Anh  5=cặp khác", "1")
        cfg["huong"] = {"1": "zh-vi", "2": "vi-zh", "3": "en-vi", "4": "vi-en"}.get(h, "zh-vi")
        if h == "5":
            print("   Mã ngôn ngữ: " + " ".join(NLLB_LANG))
            cfg["huong"] = ask("Gõ cặp ngôn ngữ dạng nguồn-đích, vd vi-ja", "vi-en")
        m = ask("Máy dịch? 1=NHANH (mô hình nhỏ)  2=TỐT HƠN nhưng chậm gấp 3-5 lần (NLLB)", "1")
        cfg["may_dich"] = "nllb" if m == "2" or cfg["huong"] not in list(MT_MODELS) + ["vi-zh"] else "opus"
        cfg["test"] = ask_path("Kéo thả FILE test (các câu cần dịch) vào đây")
        te = read_table(cfg["test"])
        show_cols(te)
        cfg["cot_nguon"] = pick_col(te, "Cột nào chứa câu CẦN DỊCH?", str(next((c for c in te.columns if not pd.api.types.is_numeric_dtype(te[c])), te.columns[-1])))
        cfg["cot_id"] = pick_col(te, "Cột nào là MÃ dòng (id)?", next((str(c) for c in te.columns if c != cfg["cot_nguon"]), None), allow_empty=True)
        cfg["mau_nop"] = ask_path("Kéo thả FILE NỘP MẪU vào đây (Enter = không có)", allow_empty=True)
        cfg["train"] = ask_path("Kéo thả FILE train (các cặp câu đã dịch) vào đây (Enter = không có)", allow_empty=True)
        if cfg["train"]:
            tr = read_table(cfg["train"])
            show_cols(tr)
            cfg["cot_dich"] = pick_col(tr, "Trong train, cột nào chứa BẢN DỊCH đúng?", str(next((c for c in tr.columns if c not in te.columns), tr.columns[-1])))
        elif cfg["mau_nop"]:
            sm = read_table(cfg["mau_nop"])
            cfg["cot_dich"] = str(next((c for c in sm.columns if c != cfg["cot_id"]), sm.columns[-1]))
        return cfg
    cfg["train"] = ask_path("Kéo thả FILE train (csv / xlsx)" + (" HOẶC THƯ MỤC train (mỗi lớp một thư mục con)" if media else "") + " vào đây")
    if cfg["kieu"] == "cum":
        df = read_table(cfg["train"])
        show_cols(df)
        cfg["cot_id"] = pick_col(df, "Cột nào là MÃ dòng (id)?", allow_empty=True)
        t = ask("Nếu chia nhóm theo VĂN BẢN: gõ tên / số các cột chữ, cách nhau dấu phẩy (Enter = dùng các cột số)", "")
        if t:
            cfg["cot_van_ban"] = [df.columns[int(x)] if x.strip().isdigit() else x.strip() for x in t.split(",")]
        n = ask("Chia thành mấy nhóm? (Enter = để máy tự chọn)", "")
        cfg["so_nhom"] = int(n) if n.isdigit() else None
        return cfg
    is_dir = clean_path(cfg["train"]).is_dir()
    cfg["test"] = ask_path("Kéo thả FILE test" + (" HOẶC THƯ MỤC test" if media else "") + " vào đây")
    cfg["mau_nop"] = ask_path("Kéo thả FILE NỘP MẪU (sample submission) vào đây (Enter = không có)", allow_empty=True)
    if not is_dir:
        df = read_table(cfg["train"])
        show_cols(df)
        te = read_table(cfg["test"]) if not clean_path(cfg["test"]).is_dir() else None
        sm = read_table(cfg["mau_nop"]) if cfg["mau_nop"] else None
        # đoán cột nhãn và cột id: ưu tiên theo file nộp mẫu, rồi theo cột có ở train mà không có ở test, rồi theo tên
        guess_t = next((c for c in (sm.columns if sm is not None else []) if c in df.columns and (te is None or c not in te.columns)), None)
        if guess_t is None:
            guess_t = next((c for c in df.columns if te is not None and c not in te.columns), df.columns[-1])
        cfg["cot_dich"] = pick_col(df, "Cột nào CẦN ĐOÁN (nhãn)?", str(guess_t))
        guess_i = next((c for c in (sm.columns if sm is not None else []) if te is not None and c in te.columns), None)
        if guess_i is None:
            guess_i = next((c for c in df.columns if str(c).lower() in ("id", "index", "idx", "ma", "stt") or str(c).lower().endswith("_id")), None)
        if guess_i is None and te is not None:
            guess_i = next((c for c in te.columns if c != cfg["cot_dich"] and te[c].nunique() == len(te) and c in df.columns and df[c].nunique() == len(df)
                            and not pd.api.types.is_float_dtype(te[c])), None)
        cfg["cot_id"] = pick_col(df, "Cột nào là MÃ dòng (id)?", str(guess_i) if guess_i is not None else None, allow_empty=True)
        if cfg["kieu"] == "vanban":
            cand = [c for c in df.columns if c not in (cfg["cot_dich"], cfg["cot_id"]) and not pd.api.types.is_numeric_dtype(df[c])]
            t = ask("Các cột VĂN BẢN (tên / số, cách nhau dấu phẩy)", ",".join(map(str, cand)))
            cfg["cot_van_ban"] = [df.columns[int(x)] if x.strip().isdigit() else x.strip() for x in t.split(",")]
        if media:
            cand = next((c for c in df.columns if df[c].astype(str).str.lower().str.endswith(tuple(IMG_EXT | AUD_EXT | VID_EXT)).mean() > 0.5), df.columns[0])
            cfg["cot_duong_dan"] = pick_col(df, "Cột nào chứa TÊN / ĐƯỜNG DẪN file?", str(cand))
            cfg["thu_muc_goc"] = ask_path("Kéo thả THƯ MỤC chứa các file đó vào đây (Enter = cùng chỗ với file bảng)", allow_empty=True)
        y = df[cfg["cot_dich"]]
        g = guess_task(y)
        a = ask(f"Bài toán: 1=PHÂN LOẠI (đoán nhãn)  2=ĐOÁN CON SỐ. Máy đoán là {'1' if g == 'phanloai' else '2'}", "1" if g == "phanloai" else "2")
        cfg["bai_toan"] = "phanloai" if a == "1" else "hoiquy"
    if cfg["kieu"] == "anh" and cfg.get("bai_toan", "phanloai") == "phanloai":
        a = ask("Sau khi có file nộp, TINH CHỈNH thêm mạng nơ-ron (thường tăng điểm, mất thêm 10-30 phút)? 1=CÓ  2=KHÔNG", "1")
        cfg["tinh_chinh"] = a == "1"
        if cfg["tinh_chinh"]:
            m = ask("Tinh chỉnh tối đa bao nhiêu phút?", "20")
            cfg["phut_tinh_chinh"] = float(m) if m.replace(".", "").isdigit() else 20
    return cfg


if __name__ == "__main__":
    try:
        if "--lai" in sys.argv:
            cfg = json.loads(clean_path(sys.argv[sys.argv.index("--lai") + 1]).read_text(encoding="utf-8"))
        else:
            cfg = wizard()
        run(cfg)
    except KeyboardInterrupt:
        print("\nĐã dừng.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        say(f"\n!!! LỖI: {type(e).__name__}: {e}")
        say("    Chép 20 dòng cuối ở trên cho AI của BTC (mẫu A trong SO_TAY_BAI2.md), hoặc thử lại với câu trả lời khác.")
