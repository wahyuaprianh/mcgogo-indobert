# app.py — FINAL (Strict override + Secrets-aware + debug panel)
# - Pakai MODEL_ID_CPU_OVERRIDE (FP32) dari Secrets untuk CPU. Jika gagal load, tampilkan error & stop (tidak fallback).
# - Opsional REMOTE_URL + REMOTE_TOKEN untuk inference endpoint (jika ingin pakai GPU remote).
# - Opsional HF_TOKEN untuk akses repo private di Hugging Face.
# - Halaman: Beranda, Scraping Data, Preprocessing, Modeling & Evaluasi, Prediksi.

import os, re, base64, time, datetime
from typing import Optional, Tuple, List
from functools import lru_cache
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
import requests

# ---------------- Secrets helper ----------------
def get_secret(name: str, default: str = "") -> str:
    try:
        v = st.secrets.get(name)
        if v is not None:
            return str(v)
    except Exception:
        pass
    return os.getenv(name, default) or default

# ================== ENV (REMOTE/LOCAL) ==================
MODEL_ID        = get_secret("MODEL_ID", "wahyuaprian/indobert-sentiment-mcgogo-8bit")
MODEL_ID_CPU_OVERRIDE = (get_secret("MODEL_ID_CPU_OVERRIDE", "").strip() or None)
LABEL_MAP_RAW   = get_secret("LABEL_MAP", "").strip()

REMOTE_URL   = (get_secret("REMOTE_URL", "").strip()   or None)
REMOTE_TOKEN = (get_secret("REMOTE_TOKEN", "").strip() or None)
USE_REMOTE   = bool(REMOTE_URL and REMOTE_TOKEN)

HF_TOKEN     = (get_secret("HF_TOKEN", "").strip() or None)  # jika repo HF private
AUTH = {"token": HF_TOKEN} if HF_TOKEN else {}

# ================== Threading & Perf ==================
try:
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, min(os.cpu_count() or 1, 4)))
    os.environ.setdefault("OMP_NUM_THREADS", str(torch.get_num_threads()))
    os.environ.setdefault("MKL_NUM_THREADS", str(torch.get_num_threads()))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(torch.get_num_threads()))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(torch.get_num_threads()))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

# ================== NLP utils ==================
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from nltk.probability import FreqDist
from nltk import ngrams
from wordcloud import WordCloud
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ================== HF Transformers ==================
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AutoConfig,
)
from sklearn.metrics import confusion_matrix, classification_report

# ================== Scraper & UI ==================
from google_play_scraper import Sort, reviews
from streamlit_carousel import carousel

# ========= App Config =========
st.set_page_config(layout="wide", page_title="Analisis Sentimen Magic Chess : Go Go Menggunakan IndoBERT")
APP_ID = "com.mobilechess.gp"

COLOR_MAP = {'positive': 'green', 'neutral': 'blue', 'negative': 'red'}
VALID_LABELS = set(COLOR_MAP.keys())
DEFAULT_ID2LABEL = {0:"negative", 1:"neutral", 2:"positive"}
DEFAULT_LABEL_ORDER = ["positive", "neutral", "negative"]

# ====== Panel kecil untuk cek Secrets ======
with st.sidebar.expander("🔎 Secrets debug", expanded=False):
    st.write({
        "MODEL_ID_CPU_OVERRIDE": MODEL_ID_CPU_OVERRIDE,
        "USE_REMOTE": USE_REMOTE,
        "HF_TOKEN_set": bool(HF_TOKEN),
        "LABEL_MAP": LABEL_MAP_RAW
    })

# ========= UI Helpers =========
@st.cache_data
def get_image_as_base64(path: str):
    if not os.path.exists(path): return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

header_image_path = 'image/fix.png'
img_base64 = get_image_as_base64(header_image_path)
background_style = (
    f"background-image: url(data:image/png;base64,{img_base64}); background-size: cover; background-position: center;"
    if img_base64 else "background-color: #27272a;"
)

st.markdown(f"""
<style>
[data-testid="stSidebar"] {{ background-color: #1f1f2e; color: white; }}
.sidebar-title {{ font-size: 20px; font-weight: 700; padding-bottom: 10px; }}
.sidebar-button {{ background-color: transparent; color: #fff; border: none; text-align: left;
  padding: 0.5rem 1rem; border-radius: 8px; width: 100%; transition: 0.2s; font-size: 16px; }}
.sidebar-button:hover {{ background-color: #4CAF50; }}
.sidebar-button-active {{ background-color: #4CAF50; font-weight: bold; }}
.main-card {{ {background_style} padding: 2em; border-radius: 12px; color: white; }}
.main-card h1, .main-card h2, .main-card h3, .main-card p, .main-card div[data-testid="stMarkdown"] {{ color: white !important; }}
.author-table {{ width: 100%; border-collapse: collapse; margin-top: 1em; }}
.author-table td {{ padding: 4px; vertical-align: top; }}
.author-table td:first-child {{ font-weight: 600; width: 150px; }}
</style>
""", unsafe_allow_html=True)

# ========= NLTK Guards =========
@st.cache_data
def ensure_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
ensure_nltk()

# ========= Preprocessing (EDA) =========
@st.cache_data
def load_stopwords():
    sw = set(stopwords.words('indonesian'))
    sw.update(["yg","dg","rt","dgn","ny","d","klo","kalo","amp","biar","bikin",
               "bilang","gak","ga","krn","nya","nih","sih","si","tau","tdk","tuh",
               "utk","ya","jd","jgn","sdh","aja","n","t","nyg","hehe","pen","u",
               "nan","loh","yah","dr","gw","gue"])
    path = './data/stopwords_id.txt'
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    w = line.strip()
                    if w: sw.add(w)
        except Exception:
            pass
    return sw
LIST_STOPWORDS = load_stopwords()

@st.cache_resource
def get_tokenizer_toktok():
    return ToktokTokenizer()
TOKTOK = get_tokenizer_toktok()

@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()
STEMMER = load_stemmer()

@lru_cache(maxsize=100_000)
def _stem_cached(word: str) -> str:
    return STEMMER.stem(word)

@st.cache_data
def load_kamus_baku():
    path = './data/kamus_baku.csv'
    if not os.path.exists(path): return {}
    df = pd.read_csv(path, encoding='latin-1')
    return dict(zip(df.iloc[:,0], df.iloc[:,1]))
KAMUS_BAKU = load_kamus_baku()

_repeat_re = re.compile(r'(.)\1{2,}')
def repeatchar_clean(s: str) -> str:
    return _repeat_re.sub(r'\1', s)

def clean_review(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]+", " ", text)
    text = re.sub(r'#(\S+)', r'\1', text)
    text = re.sub('[^a-zA-Z]+', ' ', text)
    text = repeatchar_clean(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str):
    text = "" if text is None else str(text)
    try:
        return list(TOKTOK.tokenize(text))
    except Exception:
        return re.findall(r"[A-Za-z]+", text)

def remove_stopwords(tokens): return [w for w in tokens if w not in LIST_STOPWORDS]
def stem(tokens): return [_stem_cached(w) for w in tokens]
def normalize(tokens): return [KAMUS_BAKU.get(t, t) for t in tokens]

def preprocess_dataframe(df_raw_input: pd.DataFrame) -> pd.DataFrame:
    df = df_raw_input.copy()
    with st.expander("Langkah 1: Case Folding & Cleaning"):
        df['review_text_cleaned'] = df['review_text'].apply(clean_review)
        st.dataframe(df[['review_text', 'review_text_cleaned']].head())

    with st.expander("Langkah 2: Tokenization & Stopwords Removal"):
        df['review_text_tokens'] = df['review_text_cleaned'].apply(tokenize)
        df['review_text_tokens_WSW'] = df['review_text_tokens'].apply(remove_stopwords)
        st.dataframe(df[['review_text_cleaned','review_text_tokens_WSW']].head())

    with st.expander("Langkah 3: Stemming"):
        st.info("Mengubah kata berimbuhan menjadi kata dasar. Proses ini bisa agak lama, mohon tunggu 🙏")
        tokens_series = df['review_text_tokens_WSW'].tolist()
        total = len(tokens_series)
        progress_bar = st.progress(0)
        status_text = st.empty()
        t0 = time.time()
        stemmed_out = []
        for i, tokens in enumerate(tokens_series, start=1):
            stemmed_out.append([_stem_cached(w) for w in tokens])
            if (i % 20 == 0) or (i == total):
                p = i / total
                elapsed = time.time() - t0
                eta = (elapsed / i) * (total - i) if i > 0 else 0
                status_text.text(f"Stemming {i}/{total} ({p*100:.1f}%) | ETA: {datetime.timedelta(seconds=int(eta))}")
                progress_bar.progress(int(p * 100))
        df['review_text_stemmed'] = stemmed_out
        st.dataframe(df[['review_text_tokens_WSW','review_text_stemmed']].head())

    with st.expander("Langkah 4: Normalisasi"):
        df['review_text_normalized'] = df['review_text_stemmed'].apply(normalize)
        st.dataframe(df[['review_text_stemmed','review_text_normalized']].head())

    df["review_text_normalizedjoin"] = df["review_text_normalized"].apply(lambda x: " ".join(x).strip())
    empty = (df["review_text_normalizedjoin"].str.len() == 0)
    df.loc[empty, "review_text_normalizedjoin"] = df.loc[empty, "review_text_cleaned"].replace("", "netral")
    return df

# ========= Preprocessing ringan untuk MODEL =========
_light_re = re.compile(r"(http\S+|www\S+|[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF])")
def preprocess_for_model(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = _light_re.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "netral"

# ========= Scraping helpers =========
def map_score_to_sentiment(score: int) -> str:
    if score in (1,2): return 'negative'
    if score == 3:     return 'neutral'
    if score in (4,5): return 'positive'
    return 'unknown'

@st.cache_data
def get_top_ngrams(corpus: str, n=2, top=15):
    tokens = corpus.split()
    fdist = FreqDist(ngrams(tokens, n))
    return pd.DataFrame(fdist.most_common(top), columns=["Ngram", "Frequency"])

def fetch_reviews_by_date(
    app_id: str,
    start_dt: datetime.datetime,
    end_dt: datetime.datetime,
    lang: str = "id",
    country: str = "id",
    page_size: int = 200,
    max_pages: int = 200,
    show_progress: bool = True,
) -> pd.DataFrame:
    all_rows: List[dict] = []
    token: Optional[Tuple] = None
    prog = st.progress(0) if show_progress else None
    info = st.empty() if show_progress else None
    for page in range(max_pages):
        batch, token = reviews(
            app_id, lang=lang, country=country, sort=Sort.NEWEST,
            count=page_size, continuation_token=token,
        )
        if not batch:
            break
        all_rows.extend(batch)
        if show_progress:
            pct = int(min(100, (page + 1) / max_pages * 100))
            prog.progress(pct)
            last_dt = batch[-1].get("at")
            if isinstance(last_dt, datetime.datetime):
                info.text(f"Halaman {page+1}/{max_pages} | Oldest in batch: {last_dt}")
        last_at = batch[-1].get("at")
        if isinstance(last_at, datetime.datetime) and last_at.tzinfo is not None:
            last_at = last_at.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        if isinstance(last_at, datetime.datetime) and last_at < start_dt:
            break
        if token is None:
            break
    if show_progress and prog: prog.progress(100)
    if not all_rows: return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    ts = pd.to_datetime(df["at"], errors="coerce", utc=True)
    df["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)].reset_index(drop=True)
    if "content" in df.columns and "review_text" not in df.columns:
        df = df.rename(columns={"content": "review_text"})
    if "review_text" not in df.columns:
        df["review_text"] = df.get("body", "")
    if "score" in df.columns:
        df["category"] = df["score"].apply(map_score_to_sentiment)
    else:
        df["category"] = "unknown"
    if "reviewId" in df.columns:
        df = df.drop_duplicates("reviewId")
    if "at" in df.columns:
        df = df.drop(columns=["at"])
    return df

# ========= Label utils =========
def _standardize_label(lbl: str) -> str:
    l = (lbl or "").lower()
    if l.startswith("pos"): return "positive"
    if l.startswith("neu"): return "neutral"
    if l.startswith("neg"): return "negative"
    if l in VALID_LABELS:   return l
    return l

def _norm_lbl(x: str) -> str:
    x = (x or "").strip().lower()
    aliases = {"pos":"positive","positive":"positive","neg":"negative","negative":"negative","neu":"neutral","neutral":"neutral"}
    return aliases.get(x, x)

# ========= Runtime params by mode =========
def get_runtime_params(mode: str, device: torch.device):
    if mode == "Akurasi Tinggi":
        max_len = 512; batch_size = 16 if device.type == "cpu" else 64; use_quant = False
    else:
        max_len = 256; batch_size = 64 if device.type == "cpu" else 128; use_quant = True if device.type == "cpu" else False
    return max_len, batch_size, use_quant

# ========= Remote predict =========
def remote_predict_batch(texts: List[str], return_conf: bool = False):
    payload = {"inputs": [preprocess_for_model(t) for t in texts]}
    headers = {"Authorization": f"Bearer {REMOTE_TOKEN}"}
    r = requests.post(REMOTE_URL, json=payload, headers=headers, timeout=120)
    r.raise_for_status()
    out = r.json()
    preds, confs = [], []
    if isinstance(out, dict): raise RuntimeError(out.get("error","Remote error"))
    if len(texts) == 1 and isinstance(out, list) and out and isinstance(out[0], dict):
        out = [out]
    for per_text in out:
        if not per_text: preds.append("neutral"); confs.append(0.0); continue
        best = max(per_text, key=lambda x: x.get("score", 0.0))
        lbl = _norm_lbl(best.get("label", "neutral"))
        preds.append(lbl); confs.append(float(best.get("score", 0.0))*100.0)
    return (preds, confs) if return_conf else preds

# ========= Loader (STRICT OVERRIDE) =========
@st.cache_resource
def load_model_and_tokenizer(quantize: bool = False, _v: int = 8):
    from transformers import BertTokenizer, BertForSequenceClassification, AutoConfig

    errors = []
    use_cuda = torch.cuda.is_available()
    loaded_bnb_8bit = False
    tokenizer = None
    model = None
    device = None
    final_model_id = None

    # 1) PRIORITAS: pakai MODEL_ID_CPU_OVERRIDE apa pun yang terjadi
    if MODEL_ID_CPU_OVERRIDE:
        try:
            cfg = AutoConfig.from_pretrained(MODEL_ID_CPU_OVERRIDE, **AUTH)
            # buang jejak quantization di config kalau ada
            if hasattr(cfg, "quantization_config"):
                try: delattr(cfg, "quantization_config")
                except: pass
                try: cfg.__dict__.pop("quantization_config", None)
                except: pass
            if getattr(cfg, "num_labels", None) != 3:
                cfg.num_labels = 3

            # CPU path (Streamlit Cloud)
            model = BertForSequenceClassification.from_pretrained(
                MODEL_ID_CPU_OVERRIDE, config=cfg, torch_dtype=torch.float32, **AUTH
            )
            device = torch.device("cpu")
            model.to(device)
            model.float()
            final_model_id = MODEL_ID_CPU_OVERRIDE

        except Exception as e:
            errors.append(f"OVERRIDE load failed for {MODEL_ID_CPU_OVERRIDE}: {e}")

            # tampilkan error & JANGAN fallback
            with st.sidebar.expander("⚙ Model Info", expanded=True):
                st.error(f"Gagal memuat MODEL_ID_CPU_OVERRIDE = `{MODEL_ID_CPU_OVERRIDE}`")
                if errors:
                    st.write("Load notes:")
                    st.code("\n".join(errors))
            st.stop()  # hentikan app di sini supaya penyebab terlihat jelas

    # 2) Jika tidak ada override (atau kamu hapus st.stop di atas), jalan normal
    if model is None:
        if use_cuda:
            try:
                model = BertForSequenceClassification.from_pretrained(
                    MODEL_ID, load_in_8bit=True, device_map="auto", **AUTH
                )
                device = next(model.parameters()).device
                loaded_bnb_8bit = True
                final_model_id = MODEL_ID
            except Exception as e:
                errors.append(f"cuda 8bit load failed: {e}")
                try:
                    cfg = AutoConfig.from_pretrained(MODEL_ID, **AUTH)
                    if hasattr(cfg, "quantization_config"):
                        try: delattr(cfg, "quantization_config")
                        except: pass
                        try: cfg.__dict__.pop("quantization_config", None)
                        except: pass
                    model = BertForSequenceClassification.from_pretrained(MODEL_ID, config=cfg, **AUTH)
                    device = torch.device("cuda")
                    model.to(device)
                    final_model_id = MODEL_ID
                except Exception as e2:
                    errors.append(f"cuda fp32 load failed: {e2}")
        # CPU fallback ke base jika semua gagal & tidak ada override
        if model is None:
            base_id = "indobenchmark/indobert-base-p1"
            cfg = AutoConfig.from_pretrained(base_id, num_labels=3, **AUTH)
            model = BertForSequenceClassification.from_pretrained(
                base_id, config=cfg, torch_dtype=torch.float32, **AUTH
            )
            device = torch.device("cpu")
            model.to(device)
            model.float()
            final_model_id = base_id

    # Quantization CPU opsional
    if device.type == "cpu" and quantize and not loaded_bnb_8bit:
        try:
            from torchao.quantization import quantize_, int8_dynamic
            quantize_(model, int8_dynamic())
        except Exception:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="torch.ao.quantization is deprecated", category=DeprecationWarning)
                try:
                    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                except Exception:
                    pass

    model.eval()

    # Tokenizer
    try:
        tokenizer = BertTokenizer.from_pretrained(final_model_id, use_fast=True, **AUTH)
    except Exception:
        tokenizer = BertTokenizer.from_pretrained(final_model_id, **AUTH)

    # Label map dari config
    try:
        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
        if len(id2label) != getattr(model.config, "num_labels", 3):
            raise KeyError
    except Exception:
        id2label = {0: "negative", 1: "neutral", 2: "positive"}
        label2id = {v: k for k, v in id2label.items()}

    # Override LABEL_MAP (secrets)
    if LABEL_MAP_RAW:
        for pair in LABEL_MAP_RAW.split(","):
            i, name = pair.split(":", 1)
            id2label[int(i)] = str(name).strip().lower()
        label2id = {v: k for k, v in id2label.items()}

    with st.sidebar.expander("⚙ Model Info", expanded=False):
        st.write(f"Use Remote: `{USE_REMOTE}`")
        st.write(f"Model: `{final_model_id}`")
        st.write(f"Device: `{device.type}`")
        if errors:
            st.warning("Load notes:\n- " + "\n- ".join(errors))

    return tokenizer, model, device, id2label, label2id

# ========= Sidebar & Mode =========
if "page" not in st.session_state:
    st.session_state.page = "Beranda"

st.sidebar.markdown('<div class="sidebar-title">Menu</div>', unsafe_allow_html=True)
for key, label in {"Beranda":"🏠 Beranda","Scraping Data":"📥 Scraping Data","Preprocessing":"🧹 Preprocessing","Modeling & Evaluasi":"📊 Modeling & Evaluasi","Prediksi":"🔮 Prediksi"}.items():
    if st.sidebar.button(label, key=f"menu_{key}", use_container_width=True):
        st.session_state.page = key; st.rerun()

mode_choice = st.sidebar.radio("Mode Inference",
    options=["Cepat (disarankan)", "Akurasi Tinggi"],
    help="Cepat: MAX_LEN=256, batching besar, quantization CPU.\nAkurasi Tinggi: MAX_LEN=512, tanpa quantization."
)

# ========= Load model/tokenizer (skip kalau remote) =========
if USE_REMOTE:
    device = torch.device("cpu")
    ID2LABEL = DEFAULT_ID2LABEL.copy()
    if LABEL_MAP_RAW:
        for pair in LABEL_MAP_RAW.split(","):
            i, name = pair.split(":",1)
            ID2LABEL[int(i)] = _standardize_label(name)
    LABEL2ID = {v:k for k,v in ID2LABEL.items()}
    tokenizer = None; model = None
else:
    _tmp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, use_quant_tmp = get_runtime_params(mode_choice, _tmp_device)
    tokenizer, model, device, ID2LABEL, LABEL2ID = load_model_and_tokenizer(
        quantize=use_quant_tmp, _v=8  # bust cache saat perlu
    )

MAX_LEN, BATCH_SIZE, USE_QUANT = get_runtime_params(mode_choice, device)
page = st.session_state.page

# ========= Prediksi utility =========
def predict_texts_dynamic(texts: List[str], batch_size: int = BATCH_SIZE, return_conf: bool = False):
    if USE_REMOTE:
        preds, confs = [], []
        prog = st.progress(0); info = st.empty()
        N = len(texts); processed = 0
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            p, c = remote_predict_batch(texts[start:end], return_conf=True)
            preds.extend(p); confs.extend(c)
            processed += (end - start)
            prog.progress(int(processed / N * 100))
            info.text(f"Memproses {processed}/{N} ({processed/N*100:.1f}%)")
        return (preds, confs) if return_conf else preds

    preds, confs = [], []
    prog = st.progress(0); info = st.empty()
    N = len(texts); processed = 0; last_ui = time.time()
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_texts = [preprocess_for_model(t) for t in texts[start:end]]
        enc = tokenizer(
            batch_texts, return_tensors='pt', truncation=True, padding=True,
            max_length=MAX_LEN, pad_to_multiple_of=(8 if device.type != "cpu" else None),
        )
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model(**enc).logits
            else:
                logits = model(**enc).logits
            if return_conf:
                probs = F.softmax(logits, dim=-1)
                pred_ids = probs.argmax(dim=-1).tolist()
                preds.extend([ID2LABEL.get(int(i), "unknown") for i in pred_ids])
                confs.extend((probs.max(dim=-1).values * 100).tolist())
            else:
                pred_ids = logits.argmax(dim=-1).tolist()
                preds.extend([ID2LABEL.get(int(i), "unknown") for i in pred_ids])
        processed += (end - start)
        if time.time() - last_ui > 0.25:
            prog.progress(int(processed / N * 100))
            info.text(f"Memproses {processed}/{N} ({processed/N*100:.1f}%)")
            last_ui = time.time()
    prog.progress(100)
    return (preds, confs) if return_conf else preds

# ========= Page Layout =========
st.markdown('<div class="main-card">', unsafe_allow_html=True)

if page == "Beranda":
    st.title("📊 Analisis Sentimen Magic Chess : Go Go Menggunakan Model IndoBERT")
    try:
        st.image("image/home.jpg", use_container_width=True)
    except Exception:
        st.image("https://placehold.co/1200x400/1a202c/ffffff?text=Magic+Chess+Home", use_container_width=True)
    st.markdown("""
    Selamat datang di dasbor **Analisis Sentimen Ulasan Aplikasi Magic Chess: Go Go**.
    Aplikasi ini memanfaatkan **IndoBERT** untuk mengklasifikasikan sentimen ulasan pengguna.
    """)
    st.markdown("""
    <table class="author-table">
        <tr><td>Nama</td><td>Wahyu Aprian Hadiansyah</td></tr>
        <tr><td>NPM</td><td>11121284</td></tr>
        <tr><td>Kelas</td><td>4KA23</td></tr>
        <tr><td>Program Studi</td><td>Sistem Informasi</td></tr>
        <tr><td>Fakultas</td><td>Ilmu Komputer dan Teknologi Informasi</td></tr>
        <tr><td>Universitas</td><td>Universitas Gunadarma</td></tr>
        <tr><td>Tahun</td><td>2025</td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.subheader("Sinergi Hero Magic Chess Go Go")
    items = [
        {"title": "", "text": "", "img": "image/dragon altar.jpg"},
        {"title": "", "text": "", "img": "image/astro power.jpg"},
        {"title": "", "text": "", "img": "image/doomsworn.jpg"},
        {"title": "", "text": "", "img": "image/eruditio.jpg"},
        {"title": "", "text": "", "img": "image/nature spirit.jpg"},
        {"title": "", "text": "", "img": "image/emberlord.jpg"},
        {"title": "", "text": "", "img": "image/exorcist.jpg"},
        {"title": "", "text": "", "img": "image/faeborn.jpg"},
        {"title": "", "text": "", "img": "image/los pecados.jpg"},
        {"title": "", "text": "", "img": "image/necrokeep.jpg"},
        {"title": "", "text": "", "img": "image/northen vale.jpg"},
        {"title": "", "text": "", "img": "image/shadeweaver.jpg"},
        {"title": "", "text": "", "img": "image/the inferno.jpg"},
        {"title": "", "text": "", "img": "image/vonetis sea.jpg"},
    ]
    try:
        carousel(items=items)
    except Exception as e:
        st.warning(f"Carousel tidak dapat ditampilkan: {e}. Pastikan file gambarnya ada.")

elif page == "Scraping Data":
    st.header("📥 Scraping Data dari Google Play Store")
    st.write(f"Mengambil ulasan untuk **Magic Chess: Bang Bang** (App ID: `{APP_ID}`).")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Tanggal Mulai", datetime.date.today() - datetime.timedelta(days=30))
        num_reviews = st.number_input("Jumlah ulasan cepat (mode cepat saja):", min_value=10, max_value=20000, value=200, step=10)
        mode_scrape = st.radio("Metode Pengambilan", ["By rentang tanggal (disarankan)", "Cepat (terbaru saja)"])
    with col2:
        end_date = st.date_input("Tanggal Selesai", datetime.date.today())
        lang = st.selectbox("Bahasa", options=['id','en'], index=0)
        country = st.selectbox("Negara", options=['id','us'], index=0)
    start_dt = datetime.datetime.combine(start_date, datetime.time.min)
    end_dt   = datetime.datetime.combine(end_date,   datetime.time.max)
    if mode_scrape == "By rentang tanggal (disarankan)":
        with st.expander("⚙️ Opsi Lanjutan (Mode Tanggal)"):
            page_size = st.slider("Ukuran batch per halaman", 100, 250, 200, step=10)
            max_pages = st.slider("Maksimal halaman untuk dijelajahi", 10, 1000, 200, step=10)
    if st.button("Mulai Scraping", use_container_width=True):
        with st.spinner("Mengambil ulasan..."):
            try:
                if mode_scrape == "By rentang tanggal (disarankan)":
                    df = fetch_reviews_by_date(APP_ID, start_dt, end_dt, lang=lang, country=country, page_size=page_size, max_pages=max_pages, show_progress=True)
                    if df.empty:
                        st.error("Tidak ada ulasan pada rentang tanggal tersebut.")
                        st.session_state.df_scraped = pd.DataFrame()
                    else:
                        st.success(f"✅ Berhasil mengambil {len(df)} ulasan.")
                        st.dataframe(df[['review_text','category','score','timestamp']].head())
                        st.session_state.df_scraped = df
                        st.download_button("Unduh Hasil Scraping", df.to_csv(index=False).encode('utf-8'), "scraped_data.csv", "text/csv")
                else:
                    raw, _ = reviews(APP_ID, lang=lang, country=country, sort=Sort.NEWEST, count=int(num_reviews))
                    if not raw:
                        st.warning("Tidak ada ulasan ditemukan."); st.session_state.df_scraped = pd.DataFrame()
                    else:
                        df = pd.DataFrame(raw)
                        if 'at' not in df.columns:
                            st.error("Struktur data Play Store berubah. Kolom 'at' tidak ada."); st.stop()
                        ts = pd.to_datetime(df['at'], errors='coerce', utc=True)
                        df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
                        df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)].reset_index(drop=True)
                        if df.empty:
                            st.error("Tidak ada ulasan pada rentang ini dari batch terbaru.")
                            st.session_state.df_scraped = pd.DataFrame()
                        else:
                            df['category'] = df.get('score', np.nan).apply(map_score_to_sentiment)
                            if 'content' in df.columns: df = df.rename(columns={'content':'review_text'})
                            elif 'review_text' not in df.columns: df['review_text'] = df.get('body', '')
                            if 'at' in df.columns: df = df.drop(columns=['at'])
                            st.success(f"✅ Berhasil mengambil {len(df)} ulasan dari batch terbaru.")
                            st.dataframe(df[['review_text','category','score','timestamp']].head())
                            st.session_state.df_scraped = df
                            st.download_button("Unduh Hasil Scraping", df.to_csv(index=False).encode('utf-8'), "scraped_data.csv", "text/csv")
            except Exception as e:
                st.error(f"Gagal mengambil data: {e}")

elif page == "Preprocessing":
    st.header("🧹 Preprocessing Data Ulasan")
    df_raw = None
    if 'df_scraped' in st.session_state and not getattr(st.session_state, "df_scraped", pd.DataFrame()).empty:
        if st.checkbox("Gunakan data hasil scraping", value=True):
            df_raw = st.session_state.df_scraped
            cols = ['review_text','category'] if 'category' in df_raw.columns else ['review_text']
            st.subheader("📄 Data Asli (Scraping)"); st.dataframe(df_raw[cols].head())
    if df_raw is None:
        up = st.file_uploader("Atau unggah TSV/CSV", type=["tsv","csv"])
        if up is not None:
            if up.name.endswith(".tsv"):
                df_raw = pd.read_csv(up, sep='\t', names=['review_text','category'])
            else:
                df_raw = pd.read_csv(up)
                if df_raw.shape[1] == 1:
                    df_raw.columns = ['review_text']; df_raw['category'] = 'unknown'
            st.subheader("📄 Data Asli (File)")
            cols = ['review_text','category'] if 'category' in df_raw.columns else ['review_text']
            st.dataframe(df_raw[cols].head())
    if df_raw is not None and not df_raw.empty:
        if 'category' in df_raw.columns:
            tmp = df_raw[df_raw['category'].isin(VALID_LABELS)]
            if not tmp.empty:
                st.subheader("➡️ Distribusi Sentimen Data Asli (label valid)")
                counts = tmp['category'].value_counts().reindex(DEFAULT_LABEL_ORDER).fillna(0)
                fig = px.bar(x=counts.index, y=counts.values, labels={'x':'category','y':'count'},
                             title='Distribusi Sentimen Data Asli', color=counts.index, color_discrete_map=COLOR_MAP)
                fig.update_layout(showlegend=False); st.plotly_chart(fig, use_container_width=True)
        if st.button("🚀 Mulai Preprocessing", use_container_width=True):
            dfp = preprocess_dataframe(df_raw.copy()); st.success("✅ Preprocessing selesai!")
            cols_show = ['review_text','review_text_cleaned','review_text_tokens','review_text_tokens_WSW',
                         'review_text_stemmed','review_text_normalized','review_text_normalizedjoin']
            if 'category' in dfp.columns: cols_show.append('category')
            st.dataframe(dfp[cols_show].head())
            st.subheader("➡️ Distribusi Panjang Ulasan")
            dfp['length_original']   = df_raw['review_text'].astype(str).apply(lambda s: len(s.split()))
            dfp['length_preprocessed'] = dfp['review_text_normalizedjoin'].astype(str).apply(lambda s: len(s.split()))
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=dfp['length_original'], name='Asli'))
            fig_hist.add_trace(go.Histogram(x=dfp['length_preprocessed'], name='Preprocessed'))
            fig_hist.update_layout(barmode='overlay', title='Distribusi Panjang Ulasan',
                                   xaxis_title='Jumlah Kata', yaxis_title='Frekuensi')
            fig_hist.update_traces(opacity=0.75); st.plotly_chart(fig_hist, use_container_width=True)
            corpus = " ".join(dfp['review_text_normalizedjoin'].astype(str))
            if corpus.strip():
                st.subheader("➡️ Word Cloud Kata Terpopuler")
                wc = WordCloud(background_color="white", max_words=100).generate(corpus)
                fig_wc, ax_wc = plt.subplots(figsize=(10,5)); ax_wc.imshow(wc, interpolation="bilinear"); ax_wc.axis("off")
                st.pyplot(fig_wc)
                st.subheader("➡️ 20 Kata Paling Sering Muncul")
                tokens = sum((t for t in dfp['review_text_normalized'] if isinstance(t, list)), [])
                freqdist = FreqDist(tokens); df_freq = pd.DataFrame(freqdist.most_common(20), columns=['word','freq'])
                st.plotly_chart(px.bar(df_freq, x='word', y='freq', title='20 Kata Paling Sering Muncul'), use_container_width=True)
                st.subheader("➡️ 15 Bigram Paling Sering Muncul")
                df_bi = get_top_ngrams(corpus, n=2, top=15); df_bi['Ngram'] = df_bi['Ngram'].apply(lambda x: ' '.join(x))
                st.plotly_chart(px.bar(df_bi, x='Ngram', y='Frequency', title='15 Bigram Paling Sering Muncul'), use_container_width=True)
            st.download_button("💾 Download Hasil Preprocessing", dfp.to_csv(index=False).encode('utf-8'), "preprocessed.csv", "text/csv", use_container_width=True)
            st.session_state.df_preprocessed = dfp
    else:
        st.info("Silakan unggah atau scraping data terlebih dahulu.")

elif page == "Modeling & Evaluasi":
    st.header("📊 Modeling & Evaluasi IndoBERT")
    df_eval = None
    if 'df_preprocessed' in st.session_state:
        df_eval = st.session_state.df_preprocessed; st.write("Menggunakan data hasil preprocessing.")
    else:
        up = st.file_uploader("Upload File Preprocessed / Raw", type=["tsv","csv"], key="eval_upload")
        if up is not None:
            df_eval = pd.read_csv(up, sep='\t' if up.name.endswith('.tsv') else ',')
    if df_eval is not None and not df_eval.empty:
        if 'review_text' not in df_eval.columns or 'category' not in df_eval.columns:
            st.error("File harus memiliki kolom 'review_text' dan 'category'.")
        else:
            df_eval = df_eval[df_eval['category'].isin(VALID_LABELS)].reset_index(drop=True)
            if df_eval.empty:
                st.warning("Tidak ada baris dengan label valid untuk evaluasi.")
            else:
                st.dataframe(df_eval.head())
                if st.button("⚡ Mulai Evaluasi Model", use_container_width=True):
                    texts = df_eval['review_text'].astype(str).tolist()
                    with st.spinner("Inferensi cepat..."):
                        preds = predict_texts_dynamic(texts, batch_size=BATCH_SIZE, return_conf=False)
                    df_eval['predicted_category'] = preds; st.success("Evaluasi selesai! ✅")
                    order_names = DEFAULT_LABEL_ORDER
                    y_true = df_eval['category'].str.lower().tolist()
                    y_pred = df_eval['predicted_category'].str.lower().tolist()
                    st.subheader("🔢 Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred, labels=order_names)
                    fig_cm = px.imshow(cm, x=order_names, y=order_names, text_auto=True,
                                       labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                                       title="Confusion Matrix")
                    st.plotly_chart(fig_cm, use_container_width=True)
                    st.subheader("📝 Classification Report")
                    report = classification_report(y_true, y_pred, labels=order_names, target_names=order_names, output_dict=True, zero_division=0)
                    st.dataframe(pd.DataFrame(report).T)
                    st.subheader("🥧 Proporsi Sentimen Prediksi")
                    counts = pd.Series(y_pred).value_counts().reindex(order_names).fillna(0)
                    st.plotly_chart(px.pie(values=counts.values, names=counts.index, title='Proporsi Sentimen Prediksi',
                                           color=counts.index, color_discrete_map=COLOR_MAP), use_container_width=True)
    else:
        st.info("Silakan proses data di 'Preprocessing' atau unggah file yang sudah diproses.")

elif page == "Prediksi":
    st.header("🔮 Prediksi Sentimen")
    st.subheader("Prediksi dari Data yang Diproses")
    if 'df_preprocessed' in st.session_state and not st.session_state.df_preprocessed.empty:
        if st.button("Mulai Prediksi Batch", use_container_width=True):
            dfp = st.session_state.df_preprocessed.copy()
            texts = dfp['review_text'].astype(str).tolist()
            with st.spinner("Inferensi cepat (beserta confidence)..."):
                preds, confs = predict_texts_dynamic(texts, batch_size=BATCH_SIZE, return_conf=True)
            dfp['predicted_category'] = preds; dfp['confidence'] = [f"{c:.2f}%" for c in confs]
            st.success("Prediksi batch selesai! ✅")
            st.dataframe(dfp[['review_text','predicted_category','confidence']].head())
            counts = dfp['predicted_category'].value_counts().reindex(DEFAULT_LABEL_ORDER).fillna(0)
            st.plotly_chart(px.pie(values=counts.values, names=counts.index, title='Distribusi Sentimen Hasil Prediksi',
                                   color=counts.index, color_discrete_map=COLOR_MAP), use_container_width=True)
            st.download_button("Unduh Hasil Prediksi", dfp.to_csv(index=False).encode('utf-8'), "predicted_data.csv", "text/csv", use_container_width=True)
    else:
        st.info("Silakan proses data terlebih dahulu di halaman 'Preprocessing'.")
    st.subheader("Prediksi Ulasan Tunggal")
    user_input = st.text_area("Masukkan ulasan:", height=150)
    if st.button("🎯 Hasil Deteksi", use_container_width=True):
        if user_input.strip():
            if USE_REMOTE:
                preds, confs = remote_predict_batch([user_input], return_conf=True)
                predicted, conf = preds[0], confs[0]
            else:
                text_for_model = preprocess_for_model(user_input)
                enc = tokenizer(text_for_model, return_tensors='pt', truncation=True, padding=True,
                                max_length=MAX_LEN, pad_to_multiple_of=(8 if device.type != "cpu" else None))
                enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
                with torch.inference_mode():
                    if device.type == "cuda":
                        with torch.autocast('cuda', dtype=torch.float16):
                            logits = model(**enc).logits
                    else:
                        logits = model(**enc).logits
                    probs = F.softmax(logits, dim=-1).detach().cpu().numpy().squeeze()
                    lid = int(np.argmax(probs)); conf = float(probs[lid]) * 100.0
                    predicted = ID2LABEL.get(lid, "unknown")
            msg = f"Sentimen: **{predicted}** ({conf:.2f}%)"
            if predicted == 'positive': st.success(msg)
            elif predicted == 'negative': st.error(msg)
            else: st.info(msg)
        else:
            st.warning("Mohon masukkan ulasan untuk dianalisis.")

st.markdown('</div>', unsafe_allow_html=True)
