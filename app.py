# app.py (FINAL)

import os, re, base64, time, datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F

# NLTK & Preproc
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from nltk.probability import FreqDist
from nltk import ngrams
from wordcloud import WordCloud
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Model
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report

# Scraper
from google_play_scraper import Sort, reviews

# Carousel
from streamlit_carousel import carousel

# ========= App Config =========
st.set_page_config(layout="wide", page_title="Analisis Sentimen Magic Chess : Go Go Menggunakan IndoBERT")
torch.set_grad_enabled(False)

MODEL_ID = "wahyuaprian/indobert-sentiment-mcgogo-8bit"
APP_ID = "com.mobilechess.gp"

# warna konsisten untuk semua chart
COLOR_MAP = {'positive': 'green', 'neutral': 'blue', 'negative': 'red'}
VALID_LABELS = set(COLOR_MAP.keys())

# ========= UI Helpers =========
@st.cache_data
def get_image_as_base64(path: str):
    if not os.path.exists(path):
        return None
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
[data-testid="stSidebar"] {{
  background-color: #1f1f2e; color: white;
}}
.sidebar-title {{ font-size: 20px; font-weight: 700; padding-bottom: 10px; }}
.sidebar-button {{
  background-color: transparent; color: #fff; border: none; text-align: left;
  padding: 0.5rem 1rem; border-radius: 8px; width: 100%; transition: 0.2s; font-size: 16px;
}}
.sidebar-button:hover {{ background-color: #4CAF50; }}
.sidebar-button-active {{ background-color: #4CAF50; font-weight: bold; }}
.main-card {{
  {background_style}
  padding: 2em; border-radius: 12px; color: white;
}}
.main-card h1, .main-card h2, .main-card h3, .main-card p, .main-card div[data-testid="stMarkdown"] {{
  color: white !important;
}}
.author-table {{ width: 100%; border-collapse: collapse; margin-top: 1em; }}
.author-table td {{ padding: 4px; vertical-align: top; }}
.author-table td:first-child {{ font-weight: 600; width: 150px; }}
</style>
""", unsafe_allow_html=True)

# ========= NLTK Guards (stopwords only; tidak perlu punkt/punkt_tab) =========
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
    sw.update([
        "yg","dg","rt","dgn","ny","d","klo","kalo","amp","biar","bikin",
        "bilang","gak","ga","krn","nya","nih","sih","si","tau","tdk","tuh",
        "utk","ya","jd","jgn","sdh","aja","n","t","nyg","hehe","pen","u",
        "nan","loh","yah","dr","gw","gue"
    ])
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
def get_tokenizer():
    return ToktokTokenizer()
TOKTOK = get_tokenizer()

@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()
STEMMER = load_stemmer()

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
def stem(tokens): return [STEMMER.stem(w) for w in tokens]
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
        df['review_text_stemmed'] = df['review_text_tokens_WSW'].apply(stem)
        st.dataframe(df[['review_text_tokens_WSW','review_text_stemmed']].head())

    with st.expander("Langkah 4: Normalisasi"):
        df['review_text_normalized'] = df['review_text_stemmed'].apply(normalize)
        st.dataframe(df[['review_text_stemmed','review_text_normalized']].head())

    df["review_text_normalizedjoin"] = df["review_text_normalized"].apply(lambda x: " ".join(x).strip())
    empty = (df["review_text_normalizedjoin"].str.len() == 0)
    df.loc[empty, "review_text_normalizedjoin"] = df.loc[empty, "review_text_cleaned"].replace("", "netral")
    return df

# ========= Preprocessing untuk MODEL (light) =========
_light_re = re.compile(r"(http\S+|www\S+|[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF])")
def preprocess_for_model(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = _light_re.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "netral"

# ========= Scraping helper =========
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

# ========= Model Loader (ambil mapping dari config) =========
def _standardize_label(lbl: str) -> str:
    l = (lbl or "").lower()
    if l.startswith("pos"): return "positive"
    if l.startswith("neu"): return "neutral"
    if l.startswith("neg"): return "negative"
    if l in VALID_LABELS:   return l
    return l  # fallback: biarkan apa adanya

@st.cache_resource
def load_model_and_tokenizer():
    try:
        # coba 8-bit
        tokenizer = BertTokenizer.from_pretrained(MODEL_ID)
        model = BertForSequenceClassification.from_pretrained(MODEL_ID, load_in_8bit=True, device_map="auto")
        device = next(model.parameters()).device
    except Exception:
        tokenizer = BertTokenizer.from_pretrained(MODEL_ID)
        model = BertForSequenceClassification.from_pretrained(MODEL_ID)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    model.eval()

    # mapping dari config -> distandardisasi
    cfg = model.config
    id2label_raw = getattr(cfg, "id2label", {0: "positive", 1: "neutral", 2: "negative"})
    label2id_raw = getattr(cfg, "label2id", {"positive": 0, "neutral": 1, "negative": 2})
    id2label = {int(k): _standardize_label(v) for k, v in (id2label_raw.items() if isinstance(id2label_raw, dict) else enumerate(id2label_raw))}
    label2id = { _standardize_label(k): int(v) for k, v in label2id_raw.items() }

    return tokenizer, model, device, id2label, label2id

tokenizer, model, device, ID2LABEL, LABEL2ID = load_model_and_tokenizer()

# ========= Sidebar =========
if "page" not in st.session_state:
    st.session_state.page = "Beranda"

menu_items = {
    "Beranda": "🏠 Beranda",
    "Scraping Data": "📥 Scraping Data",
    "Preprocessing": "🧹 Preprocessing",
    "Modeling & Evaluasi": "📊 Modeling & Evaluasi",
    "Prediksi": "🔮 Prediksi",
}
st.sidebar.markdown('<div class="sidebar-title">Menu</div>', unsafe_allow_html=True)
for key, label in menu_items.items():
    if st.sidebar.button(label, key=f"menu_{key}", use_container_width=True):
        st.session_state.page = key
        st.rerun()
page = st.session_state.page

# ========= Halaman =========
st.markdown('<div class="main-card">', unsafe_allow_html=True)

if page == "Beranda":
    st.title("📊 Analisis Sentimen Magic Chess : Go Go Menggunakan Model IndoBERT")
    try:
        st.image("image/home.jpg", use_container_width=True)
    except Exception:
        st.image("https://placehold.co/1200x400/1a202c/ffffff?text=Magic+Chess+Home", use_container_width=True)
        st.info("Gambar 'image/home.jpg' tidak ditemukan. Menampilkan placeholder.")

    st.markdown("""
    Selamat datang di dasbor **Analisis Sentimen Ulasan Aplikasi Magic Chess: Go Go**.
    Aplikasi ini memanfaatkan **IndoBERT** untuk mengklasifikasikan sentimen ulasan pengguna (positif, netral, negatif).
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

    # ==== SINERGI + CAROUSEL ====
    st.subheader("Sinergi Hero Magic Chess Go Go")
    items = [
        {"title": "", "text": "", "img": "image/dragon altar.jpg"},
        {"title": "", "text": "", "img": "image/astro power.jpg"},
        {"title": "", "text": "", "img": "image/doomsworn.jpg"},
        {"title": "", "text": "", "img": "image/eruditio.jpg"},
        {"title": "", "text": "", "img": "image/nature spirit.jpg"},
        {"title": "", "text": "", "img": "image/emberlord.jpg"},
        {"title": "", "text": "", "img": "image/eruditio.jpg"},
        {"title": "", "text": "", "img": "image/exorcist.jpg"},
        {"title": "", "text": "", "img": "image/faeborn.jpg"},
        {"title": "", "text": "", "img": "image/los pecados.jpg"},
        {"title": "", "text": "", "img": "image/nature spirit.jpg"},
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
        num_reviews = st.number_input("Jumlah ulasan:", min_value=10, max_value=20000, value=200, step=10)
    with col2:
        end_date = st.date_input("Tanggal Selesai", datetime.date.today())
        lang = st.selectbox("Bahasa", options=['id','en'], index=0)
        country = st.selectbox("Negara", options=['id','us'], index=0)

    start_dt = datetime.datetime.combine(start_date, datetime.time.min)
    end_dt   = datetime.datetime.combine(end_date,   datetime.time.max)

    if st.button("Mulai Scraping", use_container_width=True):
        with st.spinner("Mengambil ulasan..."):
            try:
                raw, _ = reviews(APP_ID, lang=lang, country=country, sort=Sort.NEWEST, count=int(num_reviews*2))
                if not raw:
                    st.warning("Tidak ada ulasan ditemukan.")
                    st.session_state.df_scraped = pd.DataFrame()
                else:
                    df = pd.DataFrame(raw)
                    if 'at' not in df.columns:
                        st.error("Struktur data Play Store berubah. Kolom 'at' tidak ada.")
                        st.stop()
                    df['timestamp'] = pd.to_datetime(df['at'])
                    df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)].reset_index(drop=True)
                    if df.empty:
                        st.warning("Tidak ada ulasan pada rentang tanggal tersebut.")
                        st.session_state.df_scraped = pd.DataFrame()
                    else:
                        df = df.head(int(num_reviews)).copy()
                        df['category'] = df.get('score', np.nan).apply(map_score_to_sentiment)
                        if 'content' in df.columns:
                            df = df.rename(columns={'content':'review_text'})
                        elif 'review_text' not in df.columns:
                            df['review_text'] = df.get('body', '')
                        if 'at' in df.columns: df = df.drop(columns=['at'])
                        st.success(f"✅ Berhasil mengambil {len(df)} ulasan.")
                        st.dataframe(df[['review_text','category','score','timestamp']].head())
                        st.session_state.df_scraped = df
                        st.download_button("Unduh Hasil Scraping", df.to_csv(index=False).encode('utf-8'),
                                           "scraped_data.csv", "text/csv")
            except Exception as e:
                st.error(f"Gagal mengambil data: {e}")

elif page == "Preprocessing":
    st.header("🧹 Preprocessing Data Ulasan")

    df_raw = None
    if 'df_scraped' in st.session_state and not getattr(st.session_state, "df_scraped", pd.DataFrame()).empty:
        if st.checkbox("Gunakan data hasil scraping", value=True):
            df_raw = st.session_state.df_scraped
            cols = ['review_text','category'] if 'category' in df_raw.columns else ['review_text']
            st.subheader("📄 Data Asli (Scraping)")
            st.dataframe(df_raw[cols].head())

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
        # Distribusi label valid
        if 'category' in df_raw.columns:
            tmp = df_raw[df_raw['category'].isin(VALID_LABELS)]
            if not tmp.empty:
                st.subheader("➡️ Distribusi Sentimen Data Asli (label valid)")
                counts = tmp['category'].value_counts()
                order = ['positive','neutral','negative']
                counts = counts.reindex(order).fillna(0)
                fig = px.bar(
                    x=counts.index, y=counts.values,
                    labels={'x':'category','y':'count'},
                    title='Distribusi Sentimen Data Asli',
                    color=counts.index, color_discrete_map=COLOR_MAP
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        if st.button("🚀 Mulai Preprocessing", use_container_width=True):
            dfp = preprocess_dataframe(df_raw.copy())
            st.success("✅ Preprocessing selesai!")
            cols_show = [
                'review_text','review_text_cleaned','review_text_tokens','review_text_tokens_WSW',
                'review_text_stemmed','review_text_normalized','review_text_normalizedjoin'
            ]
            if 'category' in dfp.columns: cols_show.append('category')
            st.dataframe(dfp[cols_show].head())

            # Distribusi panjang
            st.subheader("➡️ Distribusi Panjang Ulasan")
            dfp['length_original'] = df_raw['review_text'].astype(str).apply(lambda s: len(s.split()))
            dfp['length_preprocessed'] = dfp['review_text_normalizedjoin'].astype(str).apply(lambda s: len(s.split()))
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=dfp['length_original'], name='Asli'))
            fig_hist.add_trace(go.Histogram(x=dfp['length_preprocessed'], name='Preprocessed'))
            fig_hist.update_layout(barmode='overlay', title='Distribusi Panjang Ulasan',
                                   xaxis_title='Jumlah Kata', yaxis_title='Frekuensi')
            fig_hist.update_traces(opacity=0.75)
            st.plotly_chart(fig_hist, use_container_width=True)

            corpus = " ".join(dfp['review_text_normalizedjoin'].astype(str))
            if corpus.strip():
                st.subheader("➡️ Word Cloud Kata Terpopuler")
                wc = WordCloud(background_color="white", max_words=100).generate(corpus)
                fig_wc, ax_wc = plt.subplots(figsize=(10,5))
                ax_wc.imshow(wc, interpolation="bilinear"); ax_wc.axis("off")
                st.pyplot(fig_wc)

                st.subheader("➡️ 20 Kata Paling Sering Muncul")
                tokens = sum((t for t in dfp['review_text_normalized'] if isinstance(t, list)), [])
                freqdist = FreqDist(tokens)
                top_words = freqdist.most_common(20)
                df_freq = pd.DataFrame(top_words, columns=['word','freq'])
                st.plotly_chart(px.bar(df_freq, x='word', y='freq', title='20 Kata Paling Sering Muncul'),
                                use_container_width=True)

                st.subheader("➡️ 15 Bigram Paling Sering Muncul")
                df_bi = get_top_ngrams(corpus, n=2, top=15); df_bi['Ngram'] = df_bi['Ngram'].apply(lambda x: ' '.join(x))
                st.plotly_chart(px.bar(df_bi, x='Ngram', y='Frequency', title='15 Bigram Paling Sering Muncul'),
                                use_container_width=True)

            st.download_button("💾 Download Hasil Preprocessing",
                               dfp.to_csv(index=False).encode('utf-8'),
                               "preprocessed.csv", "text/csv", use_container_width=True)
            st.session_state.df_preprocessed = dfp
    else:
        st.info("Silakan unggah atau scraping data terlebih dahulu.")

elif page == "Modeling & Evaluasi":
    st.header("📊 Modeling & Evaluasi IndoBERT")
    df_eval = None
    if 'df_preprocessed' in st.session_state:
        df_eval = st.session_state.df_preprocessed
        st.write("Menggunakan data hasil preprocessing.")
    else:
        up = st.file_uploader("Upload File Preprocessed / Raw", type=["tsv","csv"], key="eval_upload")
        if up is not None:
            df_eval = pd.read_csv(up, sep='\t' if up.name.endswith('.tsv') else ',')

    if df_eval is not None and not df_eval.empty:
        if 'review_text' not in df_eval.columns or 'category' not in df_eval.columns:
            st.error("File harus memiliki kolom 'review_text' dan 'category'.")
        else:
            # filter label ground-truth yang valid
            df_eval = df_eval[df_eval['category'].isin(VALID_LABELS)].reset_index(drop=True)
            if df_eval.empty:
                st.warning("Tidak ada baris dengan label valid untuk evaluasi.")
            else:
                st.dataframe(df_eval.head())

                if st.button("⚡ Mulai Evaluasi Model", use_container_width=True):
                    prog = st.progress(0); info = st.empty()
                    preds = []; total = len(df_eval); t0 = time.time()

                    for i, raw in enumerate(df_eval['review_text'].astype(str)):
                        text_for_model = preprocess_for_model(raw)
                        enc = tokenizer(text_for_model, return_tensors='pt', truncation=True, padding=True, max_length=512)
                        enc = {k: v.to(device) for k, v in enc.items()}
                        logits = model(**enc).logits
                        label_id = int(torch.argmax(logits, dim=-1).item())
                        preds.append(ID2LABEL.get(label_id, "unknown"))

                        p = (i+1)/total; eta = (time.time()-t0)/(i+1) * (total-i-1)
                        info.text(f"Memproses {i+1}/{total} ({p*100:.1f}%) | ETA: {datetime.timedelta(seconds=int(eta))}")
                        prog.progress(int(p*100))

                    df_eval['predicted_category'] = preds
                    st.success("Evaluasi selesai! ✅")

                    # mapping ke id menggunakan LABEL2ID (dari config yang sudah distandardisasi)
                    y_true = df_eval['category'].map(LABEL2ID).astype(int).to_numpy()
                    y_pred = df_eval['predicted_category'].map(LABEL2ID).astype(int).to_numpy()

                    # urutan label konsisten
                    order_names = ['positive','neutral','negative']
                    order_ids = [LABEL2ID.get(n, i) for i, n in enumerate(order_names)]

                    st.subheader("🔢 Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred, labels=order_ids)
                    fig_cm = px.imshow(
                        cm, x=order_names, y=order_names, text_auto=True,
                        labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                        title="Confusion Matrix"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

                    st.subheader("📝 Classification Report")
                    report = classification_report(y_true, y_pred,
                        target_names=order_names, output_dict=True, zero_division=0)
                    st.dataframe(pd.DataFrame(report).T)

                    st.subheader("🥧 Proporsi Sentimen Prediksi")
                    counts = df_eval['predicted_category'].value_counts()
                    counts = counts.reindex(order_names).fillna(0)
                    fig_pie = px.pie(
                        values=counts.values, names=counts.index,
                        title='Proporsi Sentimen Prediksi',
                        color=counts.index, color_discrete_map=COLOR_MAP
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Silakan proses data di 'Preprocessing' atau unggah file yang sudah diproses.")

elif page == "Prediksi":
    st.header("🔮 Prediksi Sentimen")

    st.subheader("Prediksi dari Data yang Diproses")
    if 'df_preprocessed' in st.session_state and not st.session_state.df_preprocessed.empty:
        if st.button("Mulai Prediksi Batch", use_container_width=True):
            dfp = st.session_state.df_preprocessed.copy()
            prog = st.progress(0); info = st.empty()
            preds, confs = [], []; total = len(dfp); t0 = time.time()

            for i, raw in enumerate(dfp['review_text'].astype(str)):
                text_for_model = preprocess_for_model(raw)
                enc = tokenizer(text_for_model, return_tensors='pt', truncation=True, padding=True, max_length=512)
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits
                probs = F.softmax(logits, dim=-1).detach().cpu().numpy().squeeze()
                lid = int(np.argmax(probs)); conf = float(probs[lid]) * 100.0
                preds.append(ID2LABEL.get(lid, "unknown")); confs.append(f"{conf:.2f}%")

                p = (i+1)/total
                info.text(f"Memproses {i+1}/{total} ({p*100:.1f}%)")
                prog.progress(int(p*100))

            dfp['predicted_category'] = preds
            dfp['confidence'] = confs
            st.success("Prediksi batch selesai! ✅")
            st.dataframe(dfp[['review_text','predicted_category','confidence']].head())

            counts = dfp['predicted_category'].value_counts()
            order = ['positive','neutral','negative']
            counts = counts.reindex(order).fillna(0)
            fig_pie = px.pie(
                values=counts.values, names=counts.index,
                title='Distribusi Sentimen Hasil Prediksi',
                color=counts.index, color_discrete_map=COLOR_MAP
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            st.download_button("Unduh Hasil Prediksi",
                               dfp.to_csv(index=False).encode('utf-8'),
                               "predicted_data.csv", "text/csv", use_container_width=True)
    else:
        st.info("Silakan proses data terlebih dahulu di halaman 'Preprocessing'.")

    st.subheader("Prediksi Ulasan Tunggal")
    user_input = st.text_area("Masukkan ulasan:", height=150)
    if st.button("🎯 Hasil Deteksi", use_container_width=True):
        if user_input.strip():
            text_for_model = preprocess_for_model(user_input)
            enc = tokenizer(text_for_model, return_tensors='pt', truncation=True, padding=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
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
