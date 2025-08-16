# app.py — FINAL (token-only HF auth, force re-download, clearer errors, tokenizer fallback)

import os, re, base64, time, datetime, json
from typing import Optional, Tuple, List, Dict
from functools import lru_cache
from itertools import permutations
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
MODEL_ID        = get_secret("MODEL_ID", "wahyuaprian/indobert-sentiment-mcgogo-fp32")
MODEL_ID_CPU_OVERRIDE = (get_secret("MODEL_ID_CPU_OVERRIDE", "").strip() or None)
LABEL_MAP_RAW   = get_secret("LABEL_MAP", "").strip()

REMOTE_URL   = (get_secret("REMOTE_URL", "").strip()   or None)
REMOTE_TOKEN = (get_secret("REMOTE_TOKEN", "").strip() or None)
USE_REMOTE   = bool(REMOTE_URL and REMOTE_TOKEN)

HF_TOKEN     = (get_secret("HF_TOKEN", "").strip() or None)  # jika repo HF private
HF_REVISION  = (get_secret("HF_REVISION", "").strip() or "main")  # default "main"

# Penting: gunakan HANYA 'token' (tanpa 'use_auth_token') → hindari ValueError di Transformers 4.55+
def _hf_kwargs(force_download: bool = False) -> dict:
    kw = dict(revision=HF_REVISION, local_files_only=False, force_download=force_download)
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    return kw

# --- Single-pred tuning (defaults; bisa diubah via sidebar) ---
TEMP = float(get_secret("TEMP", "1.2"))
CONF_MIN = float(get_secret("CONF_MIN", "0.60"))
NEUTRAL_MARGIN = float(get_secret("NEUTRAL_MARGIN", "0.10"))
NEUTRAL_CAP_DEFAULT = float(get_secret("NEUTRAL_CAP", "0.60"))
SINGLE_MAXLEN = int(get_secret("SINGLE_MAXLEN", "512"))
STRIDE_RATIO = float(get_secret("STRIDE_RATIO", "0.40"))
CHUNK_AGG = "avg"

# Toggle perilaku default
DEFAULT_USE_CONTRAST = (get_secret("USE_CONTRAST", "true").lower() != "false")
DEFAULT_USE_LEXICON  = (get_secret("USE_LEXICON", "true").lower()  != "false")

# Kata penghubung KONTRAS (Bahasa Indonesia)
CONTRAST_CUES = [
    "tetapi", "tapi", "namun", "akan tetapi",
    "semenjak", "sejak", "sayangnya", "padahal"
]

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
try:
    from streamlit_carousel import carousel
except Exception:
    carousel = None

# ========= App Config =========
st.set_page_config(layout="wide", page_title="Analisis Sentimen Magic Chess : Go Go Menggunakan IndoBERT")
APP_ID = "com.mobilechess.gp"

COLOR_MAP = {'positive': 'green', 'neutral': 'blue', 'negative': 'red'}
VALID_LABELS = set(COLOR_MAP.keys())

# >>> Default mapping konsisten dgn config.json yang benar
DEFAULT_ID2LABEL = {0:"negative", 1:"neutral", 2:"positive"}
DEFAULT_LABEL_ORDER = ["positive", "neutral", "negative"]
LABEL_NAMES = ["positive","neutral","negative"]

# ====== Panel kecil untuk cek Secrets ======
with st.sidebar.expander("🔎 Secrets debug", expanded=False):
    st.write({
        "MODEL_ID": MODEL_ID,
        "MODEL_ID_CPU_OVERRIDE": MODEL_ID_CPU_OVERRIDE,
        "USE_REMOTE": USE_REMOTE,
        "HF_TOKEN_set": bool(HF_TOKEN),
        "HF_REVISION": HF_REVISION,
        "LABEL_MAP": LABEL_MAP_RAW,
        "TEMP": TEMP,
        "CONF_MIN": CONF_MIN,
        "NEUTRAL_MARGIN": NEUTRAL_MARGIN,
        "NEUTRAL_CAP": NEUTRAL_CAP_DEFAULT,
        "SINGLE_MAXLEN": SINGLE_MAXLEN,
        "STRIDE_RATIO": STRIDE_RATIO
    })

# ========= UI Theme Helpers =========
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

LABEL_INDEX_RE = re.compile(r"label[_\s\-]?(\d+)$", re.I)
def resolve_label_name(raw: str, id2label_fallback: Dict[int, str]) -> str:
    s = (raw or "").strip().lower()
    s = _standardize_label(s)
    if s in {"positive","neutral","negative"}:
        return s
    m = LABEL_INDEX_RE.search(s)
    if m:
        idx = int(m.group(1))
        return id2label_fallback.get(idx, DEFAULT_ID2LABEL.get(idx, "neutral"))
    return "neutral"

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
        active_id2, _ = get_active_maps()
        lbl = resolve_label_name(best.get("label", "neutral"), active_id2)
        preds.append(lbl); confs.append(float(best.get("score", 0.0))*100.0)
    return (preds, confs) if return_conf else preds

# ========= Loader (mapping-safe, robust fallback) =========
@st.cache_resource
def load_model_and_tokenizer(quantize: bool = False, force_dl: bool = False, _v: int = 26):
    errors = []
    use_cuda = torch.cuda.is_available()
    loaded_bnb_8bit = False
    tokenizer = None
    model = None
    device = None
    final_model_id = None
    force_kwargs = _hf_kwargs(force_download=force_dl)
    noforce_kwargs = _hf_kwargs(force_download=False)

    # (0) CPU override (opsional)
    if MODEL_ID_CPU_OVERRIDE:
        try:
            cfg = AutoConfig.from_pretrained(MODEL_ID_CPU_OVERRIDE, **force_kwargs)
            if hasattr(cfg, "quantization_config"):
                try: delattr(cfg, "quantization_config")
                except: pass
                try: cfg.__dict__.pop("quantization_config", None)
                except: pass
            if getattr(cfg, "num_labels", None) != 3:
                cfg.num_labels = 3
            model = BertForSequenceClassification.from_pretrained(
                MODEL_ID_CPU_OVERRIDE, config=cfg, torch_dtype=torch.float32, **force_kwargs
            )
            device = torch.device("cpu"); model.to(device); model.float()
            final_model_id = MODEL_ID_CPU_OVERRIDE
        except Exception as e:
            errors.append(f"OVERRIDE load failed for {MODEL_ID_CPU_OVERRIDE}: {e}")
            model = None

    # (1) Coba MODEL_ID di GPU (8-bit, lalu FP32)
    if model is None and use_cuda:
        try:
            model = BertForSequenceClassification.from_pretrained(
                MODEL_ID, load_in_8bit=True, device_map="auto", **force_kwargs
            )
            device = next(model.parameters()).device
            loaded_bnb_8bit = True
            final_model_id = MODEL_ID
        except Exception as e:
            errors.append(f"cuda 8bit load failed ({MODEL_ID}): {e}")
            try:
                cfg = AutoConfig.from_pretrained(MODEL_ID, **force_kwargs)
                if hasattr(cfg, "quantization_config"):
                    try: delattr(cfg, "quantization_config")
                    except: pass
                    try: cfg.__dict__.pop("quantization_config", None)
                    except: pass
                model = BertForSequenceClassification.from_pretrained(MODEL_ID, config=cfg, **force_kwargs)
                device = torch.device("cuda"); model.to(device)
                final_model_id = MODEL_ID
            except Exception as e2:
                errors.append(f"cuda fp32 load failed ({MODEL_ID}): {e2}")
                model = None

    # (2) Coba MODEL_ID di CPU (FP32) – force download dulu, lalu coba cache
    if model is None:
        try:
            cfg = AutoConfig.from_pretrained(MODEL_ID, **force_kwargs)
            if hasattr(cfg, "quantization_config"):
                try: delattr(cfg, "quantization_config")
                except: pass
                try: cfg.__dict__.pop("quantization_config", None)
                except: pass
            if getattr(cfg, "num_labels", None) != 3:
                cfg.num_labels = 3
            model = BertForSequenceClassification.from_pretrained(
                MODEL_ID, config=cfg, torch_dtype=torch.float32, **force_kwargs
            )
            device = torch.device("cpu"); model.to(device); model.float()
            final_model_id = MODEL_ID
        except Exception as e:
            errors.append(f"cpu load failed (force) ({MODEL_ID}): {e}")
            try:
                cfg = AutoConfig.from_pretrained(MODEL_ID, **noforce_kwargs)
                if hasattr(cfg, "quantization_config"):
                    try: delattr(cfg, "quantization_config")
                    except: pass
                    try: cfg.__dict__.pop("quantization_config", None)
                    except: pass
                if getattr(cfg, "num_labels", None) != 3:
                    cfg.num_labels = 3
                model = BertForSequenceClassification.from_pretrained(
                    MODEL_ID, config=cfg, torch_dtype=torch.float32, **noforce_kwargs
                )
                device = torch.device("cpu"); model.to(device); model.float()
                final_model_id = MODEL_ID
            except Exception as e2:
                errors.append(f"cpu load failed (cache) ({MODEL_ID}): {e2}")
                model = None

    # (3) Fallback terakhir: base encoder
    fallback_used = False
    if model is None:
        fallback_used = True
        base_id = "indobenchmark/indobert-base-p1"
        try:
            cfg = AutoConfig.from_pretrained(base_id, num_labels=3, **noforce_kwargs)
            model = BertForSequenceClassification.from_pretrained(
                base_id, config=cfg, torch_dtype=torch.float32, **noforce_kwargs
            )
            device = torch.device("cpu"); model.to(device); model.float()
            final_model_id = base_id
        except Exception as e:
            errors.append(f"fallback base load failed: {e}")
            raise RuntimeError("Tidak dapat memuat model apa pun. Lihat error detail di sidebar.")

    # (4) Quantize CPU (opsional)
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

    # (5) Tokenizer
    tok = None
    tok_errors = []
    try:
        tok = BertTokenizer.from_pretrained(final_model_id, use_fast=True, **force_kwargs)
    except Exception as e:
        tok_errors.append(f"tokenizer fast load failed ({final_model_id}): {e}")
        try:
            tok = BertTokenizer.from_pretrained(final_model_id, **force_kwargs)
        except Exception as e2:
            tok_errors.append(f"tokenizer load failed ({final_model_id}): {e2}")
            # fallback tokenizer base
            try:
                tok = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1", **noforce_kwargs)
                tok_errors.append("Tokenizer fallback ke indobenchmark/indobert-base-p1")
            except Exception as e3:
                tok_errors.append(f"tokenizer base fallback failed: {e3}")
                raise RuntimeError("Tokenizer tidak dapat dimuat. Lihat error di sidebar.")

    # (6) Mapping label dari config (atau default) + override via secrets + SANITIZER
    try:
        id2label_cfg = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        if len(id2label_cfg) != getattr(model.config, "num_labels", 3):
            raise KeyError
    except Exception:
        id2label_cfg = {0:"negative", 1:"neutral", 2:"positive"}

    if LABEL_MAP_RAW:
        for pair in LABEL_MAP_RAW.split(","):
            i, name = pair.split(":", 1)
            id2label_cfg[int(i)] = str(name).strip().lower()

    if any(str(v).lower().startswith("label_") for v in id2label_cfg.values()) or \
       not set(map(str, id2label_cfg.values())).issubset({"negative", "neutral", "positive"}):
        id2label_cfg = {0:"negative", 1:"neutral", 2:"positive"}

    label2id_cfg = {v: k for k, v in id2label_cfg.items()}

    try:
        model.config.id2label = {int(k): v for k, v in id2label_cfg.items()}
        model.config.label2id = {v: int(k) for k, v in label2id_cfg.items()}
    except Exception:
        pass

    with st.sidebar.expander("⚙ Model Info", expanded=False):
        st.write(f"Use Remote: `{USE_REMOTE}`")
        st.write(f"Model: `{final_model_id}`")
        st.write(f"Device: `{device.type}`")
        if fallback_used:
            st.warning("Model custom gagal dimuat → Fallback ke base `indobenchmark/indobert-base-p1`.")
        if errors:
            st.code("\n\n".join(errors))
        if tok_errors:
            st.code("Tokenizer notes:\n" + "\n".join(tok_errors))

    return tok, model, device, id2label_cfg, label2id_cfg

# ========= Sidebar & Mode =========
if "page" not in st.session_state:
    st.session_state.page = "Beranda"

st.sidebar.markdown('<div class="sidebar-title">Menu</div>', unsafe_allow_html=True)
menu_items = {
    "Beranda":"🏠 Beranda",
    "Scraping Data":"📥 Scraping Data",
    "Preprocessing":"🧹 Preprocessing",
    "Modeling & Evaluasi":"📊 Modeling & Evaluasi",
    "Prediksi":"🔮 Prediksi",
    "Tes Chunking":"🧪 Tes Chunking & Prob Detail (Single)"
}
for key, label in menu_items.items():
    if st.sidebar.button(label, key=f"menu_{key}", use_container_width=True):
        st.session_state.page = key; st.rerun()

mode_choice = st.sidebar.radio("Mode Inference",
    options=["Cepat (disarankan)", "Akurasi Tinggi"],
    help="Cepat: MAX_LEN=256, batching besar, quantization CPU.\nAkurasi Tinggi: MAX_LEN=512, tanpa quantization."
)

st.sidebar.markdown("---")
use_contrast = st.sidebar.checkbox("Gunakan Contrast-aware", value=DEFAULT_USE_CONTRAST)
use_lexicon  = st.sidebar.checkbox("Gunakan Lexicon override", value=DEFAULT_USE_LEXICON)

FORCE_HF_DOWNLOAD = st.sidebar.checkbox("🔄 Force re-download HF files", value=False,
                                        help="Centang ini agar snapshot terbaru dari Hugging Face diunduh ulang (menghindari cache lama).")

# --- Anti-over-neutral controls ---
st.sidebar.markdown("### ⚙️ Tuning Prediksi")
TEMP = st.sidebar.slider("Temperature (lebih kecil = lebih tegas)", 0.50, 1.50, float(TEMP), 0.05)
NEUTRAL_MARGIN = st.sidebar.slider("Neutral margin (selisih top-2)", 0.00, 0.20, float(NEUTRAL_MARGIN), 0.01)
NEUTRAL_CAP = st.sidebar.slider("Neutral cap (maks top_p utk netral)", 0.50, 0.80, float(NEUTRAL_CAP_DEFAULT), 0.01)
CHUNK_AGG = st.sidebar.radio("Chunk aggregation", ["avg", "max"], index=1,
    help="avg: rata-rata semua chunk; max: pilih chunk paling yakin (biasanya mengurangi netral).")

# ========= Load model/tokenizer =========
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
        quantize=use_quant_tmp, force_dl=FORCE_HF_DOWNLOAD, _v=26
    )

MAX_LEN, BATCH_SIZE, USE_QUANT = get_runtime_params(mode_choice, device)

# ====== mapping override helpers ======
def set_label_override(id2label_new: Dict[int,str]):
    st.session_state.ID2LABEL_override = {int(k): str(v).lower() for k, v in id2label_new.items()}
    st.session_state.LABEL2ID_override = {v: k for k, v in st.session_state.ID2LABEL_override.items()}

def get_active_maps():
    id2 = st.session_state.get("ID2LABEL_override", ID2LABEL)
    lab2 = st.session_state.get("LABEL2ID_override", LABEL2ID)
    return id2, lab2

# ====== Proba util (opsional) ======
def _predict_proba(texts: List[str], max_len: int = 512) -> np.ndarray:
    if USE_REMOTE:
        raise RuntimeError("Proba util tidak tersedia untuk remote.")
    N = len(texts)
    C = len(get_active_maps()[0])
    out = []
    for i in range(0, N, 64):
        batch = [preprocess_for_model(t) for t in texts[i:i+64]]
        enc = tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        out.append(probs)
    return np.vstack(out) if out else np.zeros((0, C), dtype=np.float32)

def calibrate_label_map_probes() -> Tuple[dict, pd.DataFrame, dict]:
    probes = {
        "positive": [
            "gamenya seru sekali dan menyenangkan",
            "saya suka dan puas dengan permainan ini",
            "bagus mantap lancar tanpa lag",
            "pengalaman bermain sangat baik rekomendasi"
        ],
        "neutral": [
            "permainannya biasa saja tidak terlalu buruk",
            "fitur cukup standar tidak ada keluhan berarti",
            "lumayan namun tidak istimewa",
            "biasa saja sesuai ekspektasi"
        ],
        "negative": [
            "game ini buruk dan mengecewakan penuh bug",
            "sangat jelek bikin kesal dan sering kalah",
            "lag parah tidak seimbang dan ngawur",
            "fiturnya tidak jelas dan ampas"
        ]
    }
    rows = []
    for lab, sents in probes.items():
        probs = _predict_proba(sents, max_len=512)
        for p in probs:
            rows.append({"label": lab, **{f"p_{i}": float(p[i]) for i in range(len(p))}})
    df = pd.DataFrame(rows)
    means = df.groupby("label").mean(numeric_only=True)
    mapping = {}
    used = set()
    for lab in ["positive","neutral","negative"]:
        order = list(np.argsort(-means.loc[lab].values))
        for j in order:
            if j not in used:
                mapping[j] = lab
                used.add(j)
                break
    left_ids = set(range(means.shape[1])) - set(mapping.keys())
    left_labs = {"positive","neutral","negative"} - set(mapping.values())
    for j, lab in zip(left_ids, left_labs):
        mapping[j] = lab
    return {int(k):v for k,v in mapping.items()}, df, {k:v for k,v in means.to_dict().items()}

# ========= Prediksi batch utility =========
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

    ACTIVE_ID2LABEL, _ = get_active_maps()

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
                preds.extend([ACTIVE_ID2LABEL.get(int(i), "unknown") for i in pred_ids])
                confs.extend((probs.max(dim=-1).values * 100).tolist())
            else:
                pred_ids = logits.argmax(dim=-1).tolist()
                preds.extend([ACTIVE_ID2LABEL.get(int(i), "unknown") for i in pred_ids])
        processed += (end - start)
        if time.time() - last_ui > 0.25:
            prog.progress(int(processed / N * 100))
            info.text(f"Memproses {processed}/{N} ({processed/N*100:.1f}%)")
            last_ui = time.time()
    prog.progress(100)
    return (preds, confs) if return_conf else preds

# ====== Kamus sentimen ringan untuk override ======
SENTI_LEX_NEG = {
    "ngawur","ga "," gak","tidak","tdk","buruk","ampas","hapus","kalah","jelek","lag","parah",
    "frustasi","kesal","error","bug","rusak","tidak adil","no counter","ga jelas","payah",
    "ngeframe","nge-lag","ngefreeze","berat","crash","ngaco","cacat","curang","toxic","rigged",
    "tidak sinkron","ga sinkron","mengecewakan","benci","tidak suka","anjir","anjay"
}
SENTI_LEX_POS = {
    "bagus","seru","keren","mantap","menarik","suka","stabil","lancar","bagus banget",
    "mantul","recommended","rekomen","top","puas","senang","memuaskan","enak","oke","keren banget",
    "asyik","asik","mantap sekali","bagus sekali","the best"
}
def _lexicon_score(text: str) -> int:
    t = preprocess_for_model(text).lower()
    neg = sum(1 for w in SENTI_LEX_NEG if w in t)
    pos = sum(1 for w in SENTI_LEX_POS if w in t)
    return pos - neg  # <0 = negatif

# ==== Chunking helper ====
def _chunk_ids_for_model(text: str, max_len: int = None, stride_ratio: float = 0.5):
    if USE_REMOTE:
        return []
    if max_len is None:
        max_len = MAX_LEN
    body_max = max_len - 2
    ids = tokenizer.encode(preprocess_for_model(text), add_special_tokens=False)
    if len(ids) <= body_max:
        return [tokenizer.build_inputs_with_special_tokens(ids)]
    stride = max(16, int(body_max * stride_ratio))
    step = max(1, body_max - stride)
    chunks = []
    i = 0
    while i < len(ids):
        body = ids[i:i + body_max]
        if not body:
            break
        chunks.append(tokenizer.build_inputs_with_special_tokens(body))
        if i + body_max >= len(ids):
            break
        i += step
    return chunks

# ==== Robust single-text prediction (anti-over-neutral) ====
def predict_single_robust(text: str, star_score: Optional[int] = None):
    if USE_REMOTE:
        p, c = remote_predict_batch([text], return_conf=True)
        return p[0], c[0]

    ACTIVE_ID2LABEL, _ = get_active_maps()

    ids_chunks = _chunk_ids_for_model(text, max_len=SINGLE_MAXLEN, stride_ratio=STRIDE_RATIO)
    if not ids_chunks:
        ids_chunks = [tokenizer.encode(preprocess_for_model(text), add_special_tokens=True)]

    # Prob per chunk
    chunk_probs = []
    with torch.inference_mode():
        for ids in ids_chunks:
            input_ids = torch.tensor([ids], device=device)
            attn = torch.ones_like(input_ids)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            logits = logits / max(1e-6, TEMP)  # TEMP < 1 => lebih tajam
            probs = F.softmax(logits, dim=-1).float().cpu().squeeze(0).numpy()
            chunk_probs.append(probs)
    chunk_probs = np.vstack(chunk_probs)  # (n_chunk, C)

    # Aggregator
    if CHUNK_AGG == "max":
        idx = int(np.argmax(chunk_probs.max(axis=1)))
        agg_probs = chunk_probs[idx]
    else:
        weights = []
        for ids in ids_chunks:
            w = max(1, len(ids) - 2)
            weights.append(w)
        weights = np.array(weights, dtype=np.float32)
        agg_probs = (chunk_probs * weights[:, None]).sum(axis=0) / max(1.0, weights.sum())

    top2 = np.argsort(-agg_probs)[:2]
    top, second = int(top2[0]), int(top2[1])
    top_p, second_p = float(agg_probs[top]), float(agg_probs[second])
    pred = ACTIVE_ID2LABEL.get(top, "neutral")
    conf = top_p * 100.0

    if (top_p - second_p) < NEUTRAL_MARGIN and top_p < NEUTRAL_CAP:
        pred = "neutral"
        conf = max(conf, (1.0 - (top_p - second_p)) * 100.0 * 0.5)

    if star_score is not None and conf < (CONF_MIN * 100.0):
        if star_score in (1, 2):
            pred = "negative"
        elif star_score == 3:
            pred = "neutral"
        elif star_score in (4, 5):
            pred = "positive"

    return pred, conf

# ==== Contrast-aware wrapper ====
def _extract_after_contrast(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None
    low = t.lower()
    best = None
    best_pos = 10**9
    for cue in CONTRAST_CUES:
        i = low.find(cue)
        if i != -1 and i < best_pos:
            best_pos = i
            best = cue
    if best is None:
        return None
    j = best_pos + len(best)
    tail = t[j:].strip(" ,.:;!?\n\t-—")
    return tail if len(tail.split()) >= 5 else None

def predict_single_with_contrast(text: str, star_score: Optional[int] = None, use_contrast_flag: bool = True, use_lexicon_flag: bool = True):
    if use_contrast_flag:
        tail = _extract_after_contrast(text)
        if tail:
            pred_tail, conf_tail = predict_single_robust(tail, star_score=star_score)
            if (pred_tail == "negative" and conf_tail >= 55.0) or (pred_tail == "neutral" and conf_tail >= 60.0):
                return pred_tail, conf_tail
            if pred_tail == "positive" and conf_tail >= 70.0:
                return pred_tail, conf_tail

    pred, conf = predict_single_robust(text, star_score=star_score)

    if use_lexicon_flag and conf < 95.0:
        score = _lexicon_score(text)
        if score <= -2 and pred != "negative":
            pred = "negative"
        elif score >= 2 and pred != "positive":
            pred = "positive"

    return pred, conf

# === Auto-detect label order (opsional; untuk evaluasi berlabel) ===
def predict_ids_only(texts: List[str], batch_size: int = BATCH_SIZE) -> List[int]:
    preds = []
    N = len(texts)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_texts = [preprocess_for_model(t) for t in texts[start:end]]
        enc = tokenizer(
            batch_texts, return_tensors='pt', truncation=True, padding=True,
            max_length=MAX_LEN, pad_to_multiple_of=(8 if device.type != "cpu" else None),
        )
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        with torch.inference_mode():
            logits = model(**enc).logits
            pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
            preds.extend(pred_ids)
    return preds

def autodetect_label_mapping(df_labeled: pd.DataFrame, sample_n: int = 500):
    if df_labeled.empty:
        return DEFAULT_ID2LABEL.copy(), 0.0
    samp = df_labeled.sample(n=min(sample_n, len(df_labeled)), random_state=42)
    texts = samp["review_text"].astype(str).tolist()
    true_labels = samp["category"].str.lower().tolist()
    pred_ids = predict_ids_only(texts)

    y_true = np.array(true_labels)
    best_perm, best_acc = None, -1.0
    for perm in permutations(LABEL_NAMES):
        mapped = np.array([perm[i] for i in pred_ids])
        acc = (mapped == y_true).mean()
        if acc > best_acc:
            best_acc, best_perm = acc, perm
    detected = {0: best_perm[0], 1: best_perm[1], 2: best_perm[2]}
    return detected, float(best_acc)

# ========= Page Layout =========
st.markdown('<div class="main-card">', unsafe_allow_html=True)

page = st.session_state.page
if page == "Beranda":
    st.title("📊 Analisis Sentimen Magic Chess : Go Go Menggunakan Model IndoBERT")
    try:
        st.image("image/home.jpg", use_column_width=True)
    except Exception:
        st.image("https://placehold.co/1200x400/1a202c/ffffff?text=Magic+Chess+Home", use_column_width=True)
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
    if carousel is not None:
        try:
            carousel(items=items)
        except Exception as e:
            st.warning(f"Carousel tidak dapat ditampilkan: {e}. Pastikan file gambarnya ada.")
    else:
        st.caption("streamlit_carousel tidak terpasang. Lewati carousel.")

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

                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("🧪 Kalibrasi mapping via Probes (opsional)", use_container_width=True, disabled=USE_REMOTE):
                        try:
                            det, _, _ = calibrate_label_map_probes()
                            set_label_override(det)
                            st.success(f"Mapping terdeteksi (probes): {det}")
                        except Exception as e:
                            st.error(f"Gagal kalibrasi probes: {e}")
                with c2:
                    if st.button("♻️ Reset mapping ke default", use_container_width=True):
                        st.session_state.pop("ID2LABEL_override", None)
                        st.session_state.pop("LABEL2ID_override", None)
                        st.rerun()
                with c3:
                    st.caption("Config.json sudah benar, jadi mapping default sudah aman ✅")

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
    star_opt = st.selectbox("(Opsional) Rating bintang ulasan:", ["Tidak ada", "★1", "★2", "★3", "★4", "★5"], index=0)
    star_val = None if star_opt == "Tidak ada" else int(star_opt.replace("★",""))

    if st.button("🎯 Hasil Deteksi", use_container_width=True):
        if user_input.strip():
            predicted, conf = predict_single_with_contrast(
                user_input, star_score=star_val,
                use_contrast_flag=use_contrast,
                use_lexicon_flag=use_lexicon
            )
            msg = f"Sentimen: **{predicted}** ({conf:.2f}%)"
            if predicted == 'positive': st.success(msg)
            elif predicted == 'negative': st.error(msg)
            else: st.info(msg)
        else:
            st.warning("Mohon masukkan ulasan untuk dianalisis.")

elif page == "Tes Chunking":
    st.header("🧪 Tes Chunking & Prob Detail (Single)")
    txt = st.text_area("Masukkan kalimat untuk inspeksi logits/prob:", height=160,
                       value="dari segi gameplay cukup menarik bagi saya, tetapi balancing sinergi buruk dan sering lag")
    if st.button("👀 Inspect", use_container_width=True, disabled=USE_REMOTE):
        ACTIVE_ID2LABEL, _ = get_active_maps()
        chunks = _chunk_ids_for_model(txt, max_len=SINGLE_MAXLEN, stride_ratio=STRIDE_RATIO)
        rows = []
        with torch.inference_mode():
            for ci, ids in enumerate(chunks):
                input_ids = torch.tensor([ids], device=device)
                attn = torch.ones_like(input_ids)
                logits = model(input_ids=input_ids, attention_mask=attn).logits
                logits = logits / max(1e-6, TEMP)
                probs = F.softmax(logits, dim=-1).float().cpu().numpy().squeeze(0)
                row = {f"prob_{i}": float(probs[i]) for i in range(len(probs))}
                row["chunk_idx"] = ci
                row["len_ids"] = int(input_ids.shape[1])
                row["top_id"] = int(np.argmax(probs))
                row["top_label"] = ACTIVE_ID2LABEL.get(row["top_id"], "unknown")
                rows.append(row)
        st.dataframe(pd.DataFrame(rows))

# ======= Sidebar: panel mapping aktif & tools =======
with st.sidebar.expander("🧭 Active label map", expanded=True):
    cur_id2, cur_l2 = get_active_maps()
    st.write(cur_id2)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Reset mapping", use_container_width=True):
            st.session_state.pop("ID2LABEL_override", None)
            st.session_state.pop("LABEL2ID_override", None)
            st.rerun()
    with c2:
        if st.button("Swap Pos↔Neg", use_container_width=True):
            id2 = cur_id2.copy()
            pos_id = [k for k,v in id2.items() if v=="positive"]
            neg_id = [k for k,v in id2.items() if v=="negative"]
            if pos_id and neg_id:
                id2[pos_id[0]], id2[neg_id[0]] = id2[neg_id[0]], id2[pos_id[0]]
                set_label_override(id2)
                st.rerun()
    with c3:
        if st.button("Set default P/N/Neu", use_container_width=True):
            set_label_override(DEFAULT_ID2LABEL.copy())
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
