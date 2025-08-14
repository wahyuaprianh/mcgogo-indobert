# app.py
import os
import re
import base64
import time
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk import ngrams
from wordcloud import WordCloud
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from sklearn.metrics import confusion_matrix, classification_report

from google_play_scraper import Sort, reviews

# -----------------------------
# Konfigurasi umum
# -----------------------------
st.set_page_config(layout="wide", page_title="Analisis Sentimen Magic Chess : Go Go Menggunakan IndoBERT")
torch.set_grad_enabled(False)

MODEL_ID = "wahyuaprian/indobert-sentiment-mcgogo-8bit"  # HuggingFace repo
APP_ID = "com.mobilechess.gp"

LABEL_TO_INDEX = {'positive': 0, 'neutral': 1, 'negative': 2}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}
VALID_LABELS = set(LABEL_TO_INDEX.keys())

# -----------------------------
# Utilities UI
# -----------------------------
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
    if img_base64 else
    "background-color: #27272a;"
)

st.markdown(f"""
<style>
[data-testid="stSidebar"] {{
    background-color: #1f1f2e; color: white;
}}
.sidebar-title {{ font-size: 20px; font-weight: 700; padding-bottom: 10px; color: #fff; }}
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

# -----------------------------
# NLTK / Preprocessing helpers
# -----------------------------
import nltk
@st.cache_data
def ensure_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
ensure_nltk()

@st.cache_data
def load_stopwords():
    sw = set(stopwords.words('indonesian'))
    sw.update([
        "yg","dg","rt","dgn","ny","d","klo","kalo","amp","biar","bikin",
        "bilang","gak","ga","krn","nya","nih","sih","si","tau","tdk","tuh",
        "utk","ya","jd","jgn","sdh","aja","n","t","nyg","hehe","pen","u",
        "nan","loh","yah","dr","gw","gue"
    ])
    # dari file lokal opsional
    stopwords_file_path = './data/stopwords_id.txt'
    if os.path.exists(stopwords_file_path):
        try:
            with open(stopwords_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    w = line.strip()
                    if w:
                        sw.add(w)
        except Exception:
            pass
    return sw
LIST_STOPWORDS = load_stopwords()

@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()
STEMMER = load_stemmer()

@st.cache_data
def load_kamus_baku():
    path = './data/kamus_baku.csv'
    if not os.path.exists(path):
        return {}
    df_k = pd.read_csv(path, encoding='latin-1')
    return dict(zip(df_k.iloc[:, 0], df_k.iloc[:, 1]))
KAMUS_BAKU = load_kamus_baku()

_repeat_re = re.compile(r'(.)\1{2,}')
def repeatchar_clean(s: str) -> str:
    return _repeat_re.sub(r'\1', s)

def clean_review(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]+", " ", text)
    text = re.sub(r'#(\S+)', r'\1', text)
    text = re.sub('[^a-zA-Z]+', ' ', text)
    text = repeatchar_clean(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text): return word_tokenize(text)
def remove_stopwords(tokens): return [w for w in tokens if w not in LIST_STOPWORDS]
def stem(tokens): return [STEMMER.stem(w) for w in tokens]
def normalize(tokens): return [KAMUS_BAKU.get(t, t) for t in tokens]

def preprocess_single_text(raw_text: str) -> str:
    """Selalu kembalikan string non-kosong agar tokenizer tidak error."""
    cleaned = clean_review(raw_text or "")
    tokens = normalize(stem(remove_stopwords(tokenize(cleaned))))
    txt = " ".join(tokens).strip()
    if not txt:                      # fallback jika terlalu agresif
        txt = cleaned.strip()
    if not txt:                      # masih kosong → pakai token netral
        txt = "netral"
    return txt

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

    # join + safety: jangan biarkan kosong
    df["review_text_normalizedjoin"] = df["review_text_normalized"].apply(lambda x: " ".join(x).strip())
    # Fallback kalau join kosong → gunakan cleaned
    empty_mask = (df["review_text_normalizedjoin"].str.len() == 0)
    df.loc[empty_mask, "review_text_normalizedjoin"] = df.loc[empty_mask, "review_text_cleaned"].replace("", "netral")

    return df

def map_score_to_sentiment(score: int) -> str:
    if score in (1, 2): return 'negative'
    if score == 3:      return 'neutral'
    if score in (4, 5): return 'positive'
    return 'unknown'

@st.cache_data
def get_top_ngrams(corpus: str, n=2, top=15):
    tokens = corpus.split()
    fdist = FreqDist(ngrams(tokens, n))
    return pd.DataFrame(fdist.most_common(top), columns=["Ngram", "Frequency"])

# -----------------------------
# Model loader (8-bit fallback)
# -----------------------------
@st.cache_resource
def load_model_and_tokenizer():
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_ID)
        config = BertConfig.from_pretrained(MODEL_ID)
        config.num_labels = 3

        # coba 8-bit bila tersedia
        try:
            model = BertForSequenceClassification.from_pretrained(
                MODEL_ID, config=config, load_in_8bit=True, device_map="auto"
            )
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = BertForSequenceClassification.from_pretrained(MODEL_ID, config=config).to(device)

        model.eval()
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Gagal memuat model/tokenizer: {e}")
        st.stop()

tokenizer, model, device = load_model_and_tokenizer()

# -----------------------------
# Sidebar
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "Beranda"

menu_items = {
    "Beranda": "🏠 Beranda",
    "Scraping Data": "📥 Scraping Data",
    "Preprocessing": "🧹 Preprocessing",
    "Modeling & Evaluasi": "📊 Modeling & Evaluasi",
    "Prediksi": "🔮 Prediksi"
}

st.sidebar.markdown('<div class="sidebar-title">Menu</div>', unsafe_allow_html=True)
for key, label in menu_items.items():
    if st.sidebar.button(label, key=f"menu_{key}", use_container_width=True):
        st.session_state.page = key
        st.rerun()
page = st.session_state.page

# -----------------------------
# Halaman Konten
# -----------------------------
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

elif page == "Scraping Data":
    st.header("📥 Scraping Data dari Google Play Store")
    st.write(f"Mengambil ulasan untuk **Magic Chess: Bang Bang** (App ID: `{APP_ID}`).")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Tanggal Mulai", datetime.date.today() - datetime.timedelta(days=30))
        num_reviews = st.number_input("Jumlah ulasan yang diinginkan:", min_value=10, max_value=20000, value=200, step=10)
    with col2:
        end_date = st.date_input("Tanggal Selesai", datetime.date.today())
        lang = st.selectbox("Pilih bahasa:", options=['id', 'en'], index=0)
        country = st.selectbox("Pilih negara:", options=['id', 'us'], index=0)

    start_dt = datetime.datetime.combine(start_date, datetime.time.min)
    end_dt   = datetime.datetime.combine(end_date,   datetime.time.max)

    if st.button("Mulai Scraping", use_container_width=True):
        with st.spinner("Mengambil ulasan..."):
            try:
                raw, _ = reviews(
                    APP_ID, lang=lang, country=country, sort=Sort.NEWEST, count=int(num_reviews*2)
                )
                if not raw:
                    st.warning("Tidak ada ulasan yang ditemukan.")
                    st.session_state.df_scraped = pd.DataFrame()
                else:
                    df = pd.DataFrame(raw)
                    if 'at' not in df.columns:
                        st.error("Struktur data Play Store berubah. Kolom 'at' tidak ditemukan.")
                        st.stop()
                    df['timestamp'] = pd.to_datetime(df['at'])
                    df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)].reset_index(drop=True)

                    if df.empty:
                        st.warning("Tidak ada ulasan pada rentang tanggal tersebut.")
                        st.session_state.df_scraped = pd.DataFrame()
                    else:
                        df = df.head(int(num_reviews)).copy()
                        # map kategori
                        df['category'] = df.get('score', np.nan).apply(map_score_to_sentiment)
                        # pastikan kolom text ada
                        if 'content' in df.columns:
                            df = df.rename(columns={'content': 'review_text'})
                        elif 'review_text' not in df.columns:
                            df['review_text'] = df.get('body', '')
                        if 'at' in df.columns:
                            df = df.drop(columns=['at'])

                        st.success(f"✅ Berhasil mengambil {len(df)} ulasan.")
                        st.dataframe(df[['review_text','category','score','timestamp']].head())
                        st.session_state.df_scraped = df

                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("Unduh Hasil Scraping", csv, "scraped_data.csv", "text/csv")
            except Exception as e:
                st.error(f"Gagal mengambil data: {e}")

elif page == "Preprocessing":
    st.header("🧹 Preprocessing Data Ulasan")

    df_raw = None
    if 'df_scraped' in st.session_state and not getattr(st.session_state, "df_scraped", pd.DataFrame()).empty:
        use_scraped = st.checkbox("Gunakan data yang telah di-scraping", value=True)
        if use_scraped:
            df_raw = st.session_state.df_scraped
            cols = ['review_text','category'] if 'category' in df_raw.columns else ['review_text']
            st.subheader("📄 Data Asli (Scraping)")
            st.dataframe(df_raw[cols].head())

    if df_raw is None:
        up = st.file_uploader("Atau unggah file TSV/CSV", type=["tsv","csv"])
        if up is not None:
            if up.name.endswith(".tsv"):
                df_raw = pd.read_csv(up, sep='\t', names=['review_text','category'])
            else:
                df_raw = pd.read_csv(up)
                if df_raw.shape[1] == 1:
                    df_raw.columns = ['review_text']
                    df_raw['category'] = 'unknown'
            st.subheader("📄 Data Asli (File)")
            cols = ['review_text','category'] if 'category' in df_raw.columns else ['review_text']
            st.dataframe(df_raw[cols].head())

    if df_raw is not None and not df_raw.empty:
        # tampilkan distribusi label "as is" tapi difilter hanya 3 label valid
        if 'category' in df_raw.columns:
            tmp = df_raw[df_raw['category'].isin(VALID_LABELS)]
            if not tmp.empty:
                st.subheader("➡️ Distribusi Sentimen Data Asli (label valid saja)")
                counts = tmp['category'].value_counts().reindex(['positive','neutral','negative']).fillna(0)
                fig = px.bar(x=counts.index, y=counts.values, labels={'x':'category','y':'count'},
                             title='Distribusi Sentimen Data Asli')
                st.plotly_chart(fig, use_container_width=True)

        if st.button("🚀 Mulai Preprocessing", use_container_width=True):
            df_processed = preprocess_dataframe(df_raw.copy())
            st.success("✅ Preprocessing selesai!")
            cols_show = [
                'review_text', 'review_text_cleaned', 'review_text_tokens',
                'review_text_tokens_WSW', 'review_text_stemmed',
                'review_text_normalized', 'review_text_normalizedjoin'
            ]
            if 'category' in df_processed.columns: cols_show.append('category')
            st.dataframe(df_processed[cols_show].head())

            # Analitik panjang dan wordcloud
            st.subheader("➡️ Distribusi Panjang Ulasan")
            df_processed['length_original'] = df_raw['review_text'].astype(str).apply(lambda s: len(s.split()))
            df_processed['length_preprocessed'] = df_processed['review_text_normalizedjoin'].astype(str).apply(lambda s: len(s.split()))
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=df_processed['length_original'], name='Asli'))
            fig_hist.add_trace(go.Histogram(x=df_processed['length_preprocessed'], name='Preprocessed'))
            fig_hist.update_layout(barmode='overlay', title='Distribusi Panjang Ulasan', xaxis_title='Jumlah Kata', yaxis_title='Frekuensi')
            fig_hist.update_traces(opacity=0.75)
            st.plotly_chart(fig_hist, use_container_width=True)

            corpus = " ".join(df_processed['review_text_normalizedjoin'].astype(str))
            if corpus.strip():
                st.subheader("➡️ Word Cloud Kata Terpopuler")
                wc = WordCloud(background_color="white", max_words=100).generate(corpus)
                fig_wc, ax_wc = plt.subplots(figsize=(10,5))
                ax_wc.imshow(wc, interpolation="bilinear"); ax_wc.axis("off")
                st.pyplot(fig_wc)

                st.subheader("➡️ 20 Kata Paling Sering Muncul")
                tokens = sum((t for t in df_processed['review_text_normalized'] if isinstance(t, list)), [])
                freqdist = FreqDist(tokens)
                top_words = freqdist.most_common(20)
                df_freq = pd.DataFrame(top_words, columns=['word','freq'])
                fig2 = px.bar(df_freq, x='word', y='freq', title='20 Kata Paling Sering Muncul')
                st.plotly_chart(fig2, use_container_width=True)

                st.subheader("➡️ 15 Bigram Paling Sering Muncul")
                df_bi = get_top_ngrams(corpus, n=2, top=15)
                df_bi['Ngram'] = df_bi['Ngram'].apply(lambda x: ' '.join(x))
                fig3 = px.bar(df_bi, x='Ngram', y='Frequency', title='15 Bigram Paling Sering Muncul')
                st.plotly_chart(fig3, use_container_width=True)

            csv = df_processed.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Hasil Preprocessing", csv, "preprocessed.csv", "text/csv", use_container_width=True)
            st.session_state.df_preprocessed = df_processed
    else:
        st.info("Silakan unggah atau scraping data terlebih dahulu.")

elif page == "Modeling & Evaluasi":
    st.header("📊 Modeling & Evaluasi IndoBERT")
    df_eval = None
    if 'df_preprocessed' in st.session_state:
        df_eval = st.session_state.df_preprocessed
        st.write("Menggunakan data hasil preprocessing.")
    else:
        up = st.file_uploader("Upload File Preprocessed", type=["tsv","csv"], key="eval_upload")
        if up is not None:
            df_eval = pd.read_csv(up, sep='\t' if up.name.endswith('.tsv') else ',')

    if df_eval is not None and not df_eval.empty:
        if not {'review_text_normalizedjoin','category'}.issubset(df_eval.columns):
            st.error("File harus memiliki kolom 'review_text_normalizedjoin' dan 'category'.")
        else:
            # filter hanya label valid
            df_eval = df_eval[df_eval['category'].isin(VALID_LABELS)].reset_index(drop=True)
            if df_eval.empty:
                st.warning("Tidak ada baris dengan label valid untuk evaluasi.")
            else:
                st.dataframe(df_eval.head())

                if st.button("⚡ Mulai Evaluasi Model", use_container_width=True):
                    progress = st.progress(0); status = st.empty()
                    preds = []; total = len(df_eval); t0 = time.time()

                    for i, text in enumerate(df_eval['review_text_normalizedjoin'].astype(str)):
                        text_safe = preprocess_single_text(text)
                        enc = tokenizer(text_safe, return_tensors='pt', truncation=True, padding=True, max_length=512)
                        enc = {k: v.to(device) for k, v in enc.items()}
                        logits = model(**enc).logits
                        label_id = int(torch.argmax(logits, dim=-1).item())
                        preds.append(INDEX_TO_LABEL[label_id])

                        p = (i+1)/total
                        eta = (time.time()-t0)/(i+1) * (total-i-1)
                        status.text(f"Memproses {i+1}/{total} ({p*100:.1f}%) | ETA: {datetime.timedelta(seconds=int(eta))}")
                        progress.progress(int(p*100))

                    df_eval['predicted_category'] = preds
                    st.success("Evaluasi selesai! ✅")

                    y_true = df_eval['category'].map(LABEL_TO_INDEX).astype(int).to_numpy()
                    y_pred = df_eval['predicted_category'].map(LABEL_TO_INDEX).astype(int).to_numpy()

                    st.subheader("🔢 Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
                    fig_cm = px.imshow(
                        cm, x=['positive','neutral','negative'], y=['positive','neutral','negative'],
                        text_auto=True, labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                        title="Confusion Matrix"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

                    st.subheader("📝 Classification Report")
                    report = classification_report(
                        y_true, y_pred, target_names=['positive','neutral','negative'], output_dict=True, zero_division=0
                    )
                    st.dataframe(pd.DataFrame(report).T)

                    st.subheader("🥧 Proporsi Sentimen Prediksi")
                    counts = df_eval['predicted_category'].value_counts()
                    fig_pie = px.pie(values=counts.values, names=counts.index, title='Proporsi Sentimen Prediksi',
                                     color_discrete_map={'positive':'green','negative':'red','neutral':'blue'})
                    st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Silakan proses data di 'Preprocessing' atau unggah file yang sudah diproses.")

elif page == "Prediksi":
    st.header("🔮 Prediksi Sentimen")

    # Batch prediction dari hasil preprocessing
    st.subheader("Prediksi dari Data yang Diproses")
    if 'df_preprocessed' in st.session_state and not st.session_state.df_preprocessed.empty:
        if st.button("Mulai Prediksi Batch", use_container_width=True):
            dfp = st.session_state.df_preprocessed.copy()
            progress = st.progress(0); status = st.empty()
            preds, confs = [], []
            total = len(dfp); t0 = time.time()

            for i, text in enumerate(dfp['review_text_normalizedjoin'].astype(str)):
                text_safe = preprocess_single_text(text)
                enc = tokenizer(text_safe, return_tensors='pt', truncation=True, padding=True, max_length=512)
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits
                probs = F.softmax(logits, dim=-1).detach().cpu().numpy().squeeze()
                lid = int(np.argmax(probs))
                conf = float(probs[lid]) * 100.0
                preds.append(INDEX_TO_LABEL[lid])
                confs.append(f"{conf:.2f}%")

                p = (i+1)/total
                status.text(f"Memproses {i+1}/{total} ({p*100:.1f}%)")
                progress.progress(int(p*100))

            dfp['predicted_category'] = preds
            dfp['confidence'] = confs
            st.success("Prediksi batch selesai! ✅")
            st.dataframe(dfp[['review_text','predicted_category','confidence']].head())

            st.subheader("🥧 Distribusi Sentimen Hasil Prediksi")
            counts = dfp['predicted_category'].value_counts()
            fig_pie = px.pie(values=counts.values, names=counts.index, title='Distribusi Sentimen Hasil Prediksi',
                             color_discrete_map={'positive':'green','negative':'red','neutral':'blue'})
            st.plotly_chart(fig_pie, use_container_width=True)

            csv = dfp.to_csv(index=False).encode('utf-8')
            st.download_button("Unduh Hasil Prediksi", csv, "predicted_data.csv", "text/csv", use_container_width=True)
    else:
        st.info("Silakan proses data terlebih dahulu di halaman 'Preprocessing' untuk memulai prediksi batch.")

    # Single prediction
    st.subheader("Prediksi Ulasan Tunggal")
    user_input = st.text_area("Masukkan ulasan:", height=150)
    if st.button("🎯 Hasil Deteksi", use_container_width=True):
        if user_input.strip():
            text_safe = preprocess_single_text(user_input)
            enc = tokenizer(text_safe, return_tensors='pt', truncation=True, padding=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy().squeeze()
            lid = int(np.argmax(probs)); conf = float(probs[lid]) * 100.0
            predicted = INDEX_TO_LABEL[lid]

            st.subheader("Hasil Prediksi:")
            msg = f"Sentimen: **{predicted}** ({conf:.2f}%)"
            if predicted == 'positive': st.success(msg)
            elif predicted == 'negative': st.error(msg)
            else: st.info(msg)
        else:
            st.warning("Mohon masukkan ulasan untuk dianalisis.")

st.markdown('</div>', unsafe_allow_html=True)
