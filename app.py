# app.py (final revised)
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns  # tetap diimport jika nanti Anda butuh
import plotly.express as px
import plotly.graph_objects as go

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import time
import datetime

# PENTING: gunakan reviews() agar bisa batasi jumlah ulasan
from google_play_scraper import Sort, reviews
from nltk import ngrams
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from streamlit_carousel import carousel
import base64
import os

# =========================
# NLTK resource setup (safe)
# =========================
# Try to download only if not present; wrap in try/except so app doesn't crash in restricted env.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception:
        pass

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except Exception:
        pass

# punkt_tab might be missing in some envs; attempt to download but continue if fails.
try:
    nltk.data.find('tokenizers/punkt_tab/indonesian')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        # If cannot download, we'll still continue and use fallback tokenization
        pass

# =========================
# Helper: safe word_tokenize (fallback to simple split)
# =========================
def safe_word_tokenize(text: str):
    try:
        return word_tokenize(text)
    except Exception:
        # fallback: quick split on whitespace and punctuation removal
        text = re.sub(r'[' + re.escape(string.punctuation) + r']', ' ', str(text))
        return [t for t in text.split() if t]

# =========================
# Fungsi untuk mengubah gambar menjadi Base64
# =========================
@st.cache_data
def get_image_as_base64(path):
    """Baca file gambar lokal dan ubah menjadi Base64."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# =========================
# Konfigurasi Awal Streamlit
# =========================
st.set_page_config(layout="wide", page_title="Analisis Sentimen Magic Chess : Go Go Menggunakan IndoBERT")

# =========================
# Styling Modern Minimalis
# =========================
header_image_path = 'image/fix.png'
img_base64 = get_image_as_base64(header_image_path)

if img_base64:
    background_style = f"""
        background-image: url(data:image/png;base64,{img_base64});
        background-size: cover;
        background-position: center;
    """
else:
    background_style = "background-color: #27272a;"

st.markdown(f"""
<style>
[data-testid="stSidebar"] {{
    background-color: #1f1f2e;
    color: white;
}}
.sidebar-title {{
    font-size: 20px;
    font-weight: bold;
    padding-bottom: 10px;
    color: #ffffff;
}}
.sidebar-button {{
    background-color: transparent;
    color: #ffffff;
    border: none;
    text-align: left;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    width: 100%;
    transition: 0.2s ease;
    font-size: 16px;
}}
.sidebar-button:hover {{
    background-color: #4CAF50;
}}
.sidebar-button-active {{
    background-color: #4CAF50;
    font-weight: bold;
}}
.main-card {{
    {background_style}
    padding: 2em;
    border-radius: 12px;
    color: white;
}}
.main-card h1, .main-card h2, .main-card h3, .main-card p, .main-card div[data-testid="stMarkdown"] {{
    color: white !important;
}}
.author-table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 1em;
}}
.author-table td {{
    padding: 4px;
    vertical-align: top;
}}
.author-table td:first-child {{
    font-weight: bold;
    width: 150px;
}}
</style>
""", unsafe_allow_html=True)

# =========================
# Path Model & Label Mapping
# =========================
MODEL_PATH = 'wahyuaprian/indobert-sentiment-mcgogo-8bit'
APP_ID = "com.mobilechess.gp"
LABEL_TO_INDEX = {'positive': 0, 'neutral': 1, 'negative': 2}
INDEX_TO_LABEL = {0: 'positive', 1: 'neutral', 2: 'negative'}

# =========================
# Load Model dan Tokenizer (dengan fallback)
# =========================
@st.cache_resource
def load_model_and_tokenizer():
    """
    Load tokenizer and model once (cached). Return (tokenizer, model, device).
    Handles 8-bit loading attempt and fallback to float32.
    """
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    except Exception as e:
        st.error(f"Gagal memuat tokenizer dari {MODEL_PATH}: {e}")
        st.stop()

    try:
        config = BertConfig.from_pretrained(MODEL_PATH)
        config.num_labels = len(LABEL_TO_INDEX)
    except Exception:
        config = BertConfig.from_pretrained(MODEL_PATH, num_labels=len(LABEL_TO_INDEX))

    model = None
    device = torch.device("cpu")
    # Try load in 8-bit if possible (and bitsandbytes available)
    try:
        model = BertForSequenceClassification.from_pretrained(
            MODEL_PATH,
            config=config,
            load_in_8bit=True,
            device_map="auto"
        )
        # detect device (could be 'cpu' or 'cuda:x')
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cpu")
    except Exception:
        # fallback: normal float32 load
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = BertForSequenceClassification.from_pretrained(MODEL_PATH, config=config).to(device)
        except Exception as e:
            st.error(f"Gagal memuat model dari {MODEL_PATH}: {e}")
            st.stop()

    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model_and_tokenizer()

# =========================
# Fungsi Preprocessing
# =========================
def load_stopwords():
    """Memuat stopwords dari NLTK, daftar kustom manual, dan file lokal."""
    try:
        list_stop = set(stopwords.words('indonesian'))
    except Exception:
        # fallback to English stopwords if indonesian not present
        try:
            list_stop = set(stopwords.words('english'))
        except Exception:
            list_stop = set()

    list_stop.update([
        "yg","dg","rt","dgn","ny","d","klo","kalo","amp","biar","bikin",
        "bilang","gak","ga","krn","nya","nih","sih","si","tau","tdk","tuh",
        "utk","ya","jd","jgn","sdh","aja","n","t","nyg","hehe","pen","u",
        "nan","loh","yah","dr","gw","gue"
    ])
    stopwords_file_path = './data/stopwords_id.txt'
    if os.path.exists(stopwords_file_path):
        try:
            with open(stopwords_file_path, 'r', encoding='utf-8') as file:
                stopwords_from_file = [line.strip() for line in file if line.strip()]
                list_stop.update(stopwords_from_file)
        except Exception as e:
            st.warning(f"Gagal memuat stopwords lokal: {e}. Menggunakan bawaan NLTK + manual.")
    else:
        # silent: don't warn repeatedly on cloud
        pass
    return list_stop

list_stopwords = load_stopwords()

@st.cache_data
def load_kamus_baku():
    kamus_baku_path = './data/kamus_baku.csv'
    if not os.path.exists(kamus_baku_path):
        # silent fallback
        return {}
    try:
        df_kamus = pd.read_csv(kamus_baku_path, encoding='latin-1')
        return dict(zip(df_kamus.iloc[:,0], df_kamus.iloc[:,1]))
    except Exception:
        return {}

normalizad_word_dict = load_kamus_baku()

# Optimasi: rapikan karakter berulang (>=3 jadi 1)
_repeat_re = re.compile(r'(.)\1{2,}')
def repeatcharClean(text: str) -> str:
    return _repeat_re.sub(r'\1', text)

def clean_review(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]+", flags=re.UNICODE)
    text = emoji_pattern.sub(' ', text)
    text = re.sub(r'#(\S+)', r'\1', text)
    # keep only letters and spaces (Indonesian uses Latin letters)
    text = re.sub('[^a-zA-Z ]+', ' ', text)
    text = repeatcharClean(text)
    text = re.sub('[ ]+',' ',text).strip()
    return text

def tokenization_review_func(text):
    # use safe_word_tokenize which falls back to simple split
    return safe_word_tokenize(text)

def stopwords_removal_func_wrapper(words):
    try:
        return [w for w in words if w and w not in list_stopwords]
    except Exception:
        return [w for w in words if w]

@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

stemmer = load_stemmer()

def stemming_func_wrapper(words):
    try:
        return [stemmer.stem(w) for w in words]
    except Exception:
        return words

def normalized_term_func_wrapper(document):
    return [normalizad_word_dict.get(term, term) for term in document]

def preprocess_dataframe(df_raw_input):
    df_processed = df_raw_input.copy()

    # ensure 'review_text' exists
    if 'review_text' not in df_processed.columns:
        # try to find a plausible column name
        possible_cols = [c for c in df_processed.columns if 'review' in c.lower() or 'content' in c.lower()]
        if possible_cols:
            df_processed = df_processed.rename(columns={possible_cols[0]: 'review_text'})
        else:
            df_processed['review_text'] = df_processed.iloc[:,0].astype(str)

    # 1. Case folding & cleaning
    df_processed['review_text_cleaned'] = df_processed['review_text'].fillna("").astype(str).apply(clean_review)

    # 2. Tokenization & stopwords removal
    df_processed['review_text_tokens'] = df_processed['review_text_cleaned'].apply(tokenization_review_func)
    df_processed['review_text_tokens_WSW'] = df_processed['review_text_tokens'].apply(stopwords_removal_func_wrapper)

    # 3. Stemming
    df_processed['review_text_stemmed'] = df_processed['review_text_tokens_WSW'].apply(stemming_func_wrapper)

    # 4. Normalisasi
    df_processed['review_text_normalized'] = df_processed['review_text_stemmed'].apply(normalized_term_func_wrapper)

    # Join normalized tokens
    df_processed["review_text_normalizedjoin"] = df_processed["review_text_normalized"].apply(lambda l: ' '.join(l) if isinstance(l, (list, tuple)) else str(l))

    # Replace truly empty strings with NaN and drop rows with missing text
    df_processed['review_text_normalizedjoin'] = df_processed['review_text_normalizedjoin'].replace(r'^\s*$', np.nan, regex=True)
    df_processed.dropna(subset=['review_text_normalizedjoin'], inplace=True)

    return df_processed

def preprocess_single_text(text):
    text = clean_review(text)
    tokens = tokenization_review_func(text)
    tokens_wsw = stopwords_removal_func_wrapper(tokens)
    stemmed_tokens = stemming_func_wrapper(tokens_wsw)
    normalized_tokens = normalized_term_func_wrapper(stemmed_tokens)
    return " ".join(normalized_tokens)

def map_score_to_sentiment(score):
    try:
        score = int(score)
    except Exception:
        return 'unknown'
    if score in [1, 2]:
        return 'negative'
    elif score == 3:
        return 'neutral'
    elif score in [4, 5]:
        return 'positive'
    return 'unknown'

@st.cache_data
def get_top_ngrams(corpus, n=2, top=15):
    corpus_tokens = corpus.split()
    n_grams_list = ngrams(corpus_tokens, n)
    fdist = FreqDist(n_grams_list)
    return pd.DataFrame(fdist.most_common(top), columns=["Ngram", "Frequency"])

# =========================
# Sidebar Navigasi
# =========================
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

# =========================
# Konten Halaman
# =========================
st.markdown('<div class="main-card">', unsafe_allow_html=True)

# -------------------------
# Beranda (unchanged)
# -------------------------
if page == "Beranda":
    st.title("📊 Analisis Sentimen Magic Chess : Go Go Menggunakan Model IndoBERT")
    try:
        st.image("image/home.jpg", use_container_width=True)
    except Exception:
        st.image("https://placehold.co/1200x400/1a202c/ffffff?text=Magic+Chess+Home", use_container_width=True)
        st.info("Gambar 'image/home.jpg' tidak ditemukan. Menampilkan gambar placeholder.")
    st.markdown("""
    Selamat datang di dasbor **Analisis Sentimen Ulasan Aplikasi Magic Chess: Go Go**.
    """, unsafe_allow_html=True)

    st.markdown("""
    <table class="author-table">
        <tr><td>Nama</td><td>Wahyu Aprian Hadiansyah</td></tr>
        <tr><td>NPM</td><td>11121284</td></tr>
        <tr><td>Kelas</td><td>4KA23</td></tr>
        <tr><td>Program Studi</td><td>Sistem Informasi</td></tr>
        <tr><td>Universitas</td><td>Universitas Gunadarma</td></tr>
        <tr><td>Tahun</td><td>2025</td></tr>
    </table>
    """, unsafe_allow_html=True)

    # carousel (keep original)
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
        st.warning(f"Carousel tidak dapat ditampilkan: {e}")

# -------------------------
# Scraping Data (kept)
# -------------------------
elif page == "Scraping Data":
    st.header("📥 Scraping Data dari Google Play Store")
    st.write(f"Mengambil ulasan untuk **Magic Chess: Bang Bang** (App ID: `{APP_ID}`).")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Tanggal Mulai", datetime.date.today() - datetime.timedelta(days=30))
    with col2:
        end_date = st.date_input("Tanggal Selesai", datetime.date.today())

    col3, col4 = st.columns(2)
    with col3:
        num_reviews = st.number_input("Jumlah ulasan yang diinginkan:", min_value=10, max_value=20000, value=100, step=10)
    with col4:
        lang = st.selectbox("Pilih bahasa:", options=['id', 'en'], index=0)
        country = st.selectbox("Pilih negara:", options=['id', 'us'], index=0)

    start_dt = datetime.datetime.combine(start_date, datetime.time.min)
    end_dt = datetime.datetime.combine(end_date, datetime.time.max)

    st.info("Aplikasi akan mengambil ulasan terbaru, memfilternya sesuai rentang tanggal, lalu memilih jumlah yang Anda inginkan.")

    if st.button("Mulai Scraping", use_container_width=True):
        if num_reviews <= 0:
            st.error("Jumlah ulasan tidak boleh 0.")
        else:
            with st.spinner("Mengambil ulasan..."):
                try:
                    raw, _ = reviews(
                        APP_ID,
                        lang=lang,
                        country=country,
                        sort=Sort.NEWEST,
                        count=int(num_reviews*2)
                    )
                    if not raw:
                        st.warning("Tidak ada ulasan yang ditemukan. Silakan cek rentang tanggal.")
                        st.session_state.df_scraped = pd.DataFrame()
                    else:
                        df_scraped = pd.DataFrame(raw)
                        if 'at' not in df_scraped.columns:
                            st.error("Struktur data dari Play Store berubah. Kolom 'at' tidak ditemukan.")
                            st.stop()

                        df_scraped['timestamp'] = pd.to_datetime(df_scraped['at'])
                        df_filtered_by_date = df_scraped[(df_scraped['timestamp'] >= start_dt) & (df_scraped['timestamp'] <= end_dt)].reset_index(drop=True)

                        if df_filtered_by_date.empty:
                            st.warning("Tidak ada ulasan pada rentang tanggal tersebut. Coba rentang yang lebih luas.")
                            st.session_state.df_scraped = pd.DataFrame()
                        else:
                            final_df = df_filtered_by_date.head(int(num_reviews)).copy()
                            if 'score' in final_df.columns:
                                final_df['category'] = final_df['score'].apply(map_score_to_sentiment)
                            else:
                                final_df['category'] = 'unknown'

                            if 'content' in final_df.columns:
                                final_df = final_df.rename(columns={'content': 'review_text'})
                            elif 'review_text' not in final_df.columns:
                                final_df['review_text'] = final_df.get('body', '')

                            if 'at' in final_df.columns:
                                final_df = final_df.drop(columns=['at'])

                            st.success(f"✅ Berhasil mengambil {len(final_df)} ulasan dalam rentang tanggal yang dipilih!")
                            st.dataframe(final_df[['review_text', 'category', 'score', 'timestamp']].head())
                            st.session_state.df_scraped = final_df

                            csv = final_df.to_csv(index=False).encode('utf-8')
                            st.download_button("Unduh Hasil Scraping", csv, "scraped_data.csv", "text/csv")
                except Exception as e:
                    st.error(f"Gagal mengambil data. Error: {e}")

# -------------------------
# Preprocessing (kept but safe)
# -------------------------
elif page == "Preprocessing":
    st.header("🧹 Preprocessing Data Ulasan")
    st.write("Disarankan untuk mengunggah file yang sudah memiliki label sentimen (positive, negative, neutral) jika ingin melakukan evaluasi model.")

    df_raw = None
    if 'df_scraped' in st.session_state and not getattr(st.session_state, "df_scraped", pd.DataFrame()).empty:
        st.subheader("Pilih Sumber Data")
        use_scraped = st.checkbox("Gunakan data yang telah di-scraping", value=True)
        if use_scraped:
            df_raw = st.session_state.df_scraped
            st.subheader("📄 Data Asli (dari Scraping)")
            cols_to_show = ['review_text', 'category'] if 'category' in df_raw.columns else ['review_text']
            st.dataframe(df_raw[cols_to_show].head())

    if df_raw is None:
        uploaded_file = st.file_uploader("Atau, pilih file TSV/CSV", type=["tsv", "csv"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.tsv'):
                    df_raw = pd.read_csv(uploaded_file, sep='\t', names=['review_text', 'category'])
                else:
                    df_raw = pd.read_csv(uploaded_file)
                    if df_raw.shape[1] == 1:
                        df_raw.columns = ['review_text']
                        df_raw['category'] = 'unknown'
                st.subheader("📄 Data Asli (dari File)")
                cols_to_show = ['review_text', 'category'] if 'category' in df_raw.columns else ['review_text']
                st.dataframe(df_raw[cols_to_show].head())
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

    if df_raw is not None and not df_raw.empty:
        is_labeled = ('category' in df_raw.columns and df_raw['category'].isin(LABEL_TO_INDEX.keys()).any())

        if is_labeled:
            st.subheader("➡️ Distribusi Sentimen Data Asli")
            if 'timestamp' in df_raw.columns:
                df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
                df_raw['bulan_tahun'] = df_raw['timestamp'].dt.to_period('M').astype(str)

                df_counts = df_raw.groupby(['bulan_tahun', 'category']).size().reset_index(name='Jumlah Ulasan')

                fig = go.Figure()
                color_map = {'positive': 'green', 'negative': 'red', 'neutral': 'blue'}

                for category, color in color_map.items():
                    df_category = df_counts[df_counts['category'] == category]
                    fig.add_trace(go.Bar(
                        x=df_category['bulan_tahun'],
                        y=df_category['Jumlah Ulasan'],
                        name=category,
                        marker_color=color
                    ))

                fig.update_layout(
                    barmode='group',
                    xaxis_tickangle=-45,
                    title='Jumlah Ulasan Berdasarkan Sentimen per Bulan',
                    xaxis_title='Bulan dan Tahun',
                    yaxis_title='Jumlah Ulasan'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                counts = df_raw['category'].value_counts().reset_index()
                counts.columns = ['category', 'count']
                fig = px.bar(counts, x='category', y='count', title='Distribusi Sentimen Data Asli',
                             color='category', color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'blue'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Data yang Anda gunakan tidak memiliki label sentimen yang valid. Grafik distribusi sentimen tidak dapat dibuat.")

        if st.button("🚀 Mulai Preprocessing", use_container_width=True):
            df_processed = preprocess_dataframe(df_raw.copy())
            st.success("✅ Preprocessing selesai!")
            st.subheader("✅ Hasil Preprocessing Akhir")

            cols_show = ['review_text', 'review_text_cleaned', 'review_text_tokens', 'review_text_tokens_WSW', 'review_text_normalized', 'review_text_normalizedjoin']
            if 'category' in df_processed.columns:
                cols_show.append('category')
            st.dataframe(df_processed[cols_show].head())

            st.subheader("➡️ Distribusi Panjang Ulasan")
            df_processed['length_original'] = df_raw['review_text'].apply(lambda x: len(str(x).split()))
            df_processed['length_preprocessed'] = df_processed['review_text_normalizedjoin'].apply(lambda x: len(str(x).split()))

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=df_processed['length_original'], name='Panjang Asli'))
            fig_hist.add_trace(go.Histogram(x=df_processed['length_preprocessed'], name='Panjang Setelah Preprocessing'))
            fig_hist.update_layout(barmode='overlay', title_text='Distribusi Panjang Ulasan', xaxis_title_text='Jumlah Kata', yaxis_title_text='Frekuensi')
            fig_hist.update_traces(opacity=0.75)
            st.plotly_chart(fig_hist, use_container_width=True)

            corpus = " ".join(df_processed['review_text_normalizedjoin'])
            if corpus.strip():
                st.subheader("➡️ Word Cloud Kata Terpopuler")
                wc = WordCloud(background_color="white", max_words=100).generate(corpus)
                fig_wc, ax_wc = plt.subplots(figsize=(10,5))
                ax_wc.imshow(wc, interpolation="bilinear")
                ax_wc.axis("off")
                st.pyplot(fig_wc)

                st.subheader("➡️ 20 Kata Paling Sering Muncul")
                tokens = sum([tokens for tokens in df_processed['review_text_normalized'] if isinstance(tokens, list)], [])
                freqdist = FreqDist(tokens)
                top_words = freqdist.most_common(20)
                df_freq = pd.DataFrame(top_words, columns=['word','freq'])
                fig2 = px.bar(df_freq, x='word', y='freq', title='20 Kata Paling Sering Muncul')
                st.plotly_chart(fig2, use_container_width=True)

                st.subheader("➡️ 15 Bigram Paling Sering Muncul")
                df_bigrams = get_top_ngrams(corpus, n=2, top=15)
                df_bigrams['Ngram'] = df_bigrams['Ngram'].apply(lambda x: ' '.join(x))
                fig3 = px.bar(df_bigrams, x='Ngram', y='Frequency', title='15 Bigram Paling Sering Muncul')
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning("Tidak ada kata-kata yang tersisa setelah preprocessing. Visualisasi tidak dapat dibuat.")

            csv = df_processed.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Hasil Preprocessing", csv, "preprocessed.csv", "text/csv", use_container_width=True)
            st.session_state.df_preprocessed = df_processed
    else:
        st.info("Silakan unggah atau scraping data terlebih dahulu.")

# -------------------------
# Modeling & Evaluasi (kept)
# -------------------------
elif page == "Modeling & Evaluasi":
    st.header("📊 Modeling & Evaluasi IndoBERT")
    df_eval_input = None
    if 'df_preprocessed' in st.session_state:
        df_eval_input = st.session_state.df_preprocessed
        st.write("Menggunakan data yang sudah diproses dari halaman Preprocessing.")
    else:
        uploaded_eval_file = st.file_uploader("Upload File Preprocessed", type=["tsv", "csv"], key="eval_upload")
        if uploaded_eval_file is not None:
            try:
                df_eval_input = pd.read_csv(uploaded_eval_file, sep='\t' if uploaded_eval_file.name.endswith('.tsv') else ',')
            except Exception as e:
                st.error(f"Gagal membaca file evaluasi: {e}")

    if df_eval_input is not None and not df_eval_input.empty:
        if 'review_text_normalizedjoin' not in df_eval_input.columns or 'category' not in df_eval_input.columns:
            st.error("File harus memiliki kolom 'review_text_normalizedjoin' dan 'category'")
        elif not df_eval_input['category'].isin(LABEL_TO_INDEX.keys()).any():
            st.warning("Data yang diinput tidak memiliki label sentimen yang valid ('positive', 'negative', 'neutral') untuk evaluasi. Silakan unggah dataset yang sudah diberi label.")
        else:
            st.dataframe(df_eval_input.head())
            if st.button("⚡ Mulai Evaluasi Model", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                predictions = []
                total = len(df_eval_input)
                start_time = time.time()

                for idx, text in enumerate(df_eval_input['review_text_normalizedjoin'].fillna("").astype(str)):
                    # safe single inference (kept small)
                    enc = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                    enc = {k: v.to(device) for k, v in enc.items()}
                    with torch.no_grad():
                        logits = model(**enc).logits
                    # safe label extraction
                    try:
                        label_id = int(torch.argmax(logits, dim=-1).item())
                        predictions.append(INDEX_TO_LABEL.get(label_id, 'unknown'))
                    except Exception:
                        predictions.append('unknown')

                    progress = (idx + 1) / total
                    elapsed = time.time() - start_time
                    avg_time_per_item = elapsed / (idx + 1)
                    remaining_time = avg_time_per_item * (total - idx - 1)
                    eta = datetime.timedelta(seconds=int(remaining_time))
                    status_text.text(f"Memproses data {idx+1}/{total} ({progress*100:.1f}%) | ETA: {eta}")
                    progress_bar.progress(int(progress * 100))

                df_eval_input['predicted_category'] = predictions
                st.success("Evaluasi selesai! ✅")

                y_true = df_eval_input['category'].map(LABEL_TO_INDEX)
                y_pred = df_eval_input['predicted_category'].map(LABEL_TO_INDEX)

                valid_labels_mask = y_true.notna()
                y_true_filtered = y_true[valid_labels_mask]
                y_pred_filtered = y_pred[valid_labels_mask]

                if not y_true_filtered.empty:
                    st.subheader("🔢 Confusion Matrix")
                    cm = confusion_matrix(y_true_filtered.astype(int), y_pred_filtered.astype(int))
                    fig_cm = px.imshow(
                        cm,
                        x=list(LABEL_TO_INDEX.keys()),
                        y=list(LABEL_TO_INDEX.keys()),
                        text_auto=True,
                        labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                        title="Confusion Matrix"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

                    st.subheader("📝 Classification Report")
                    report_dict = classification_report(
                        y_true_filtered.astype(int),
                        y_pred_filtered.astype(int),
                        target_names=list(LABEL_TO_INDEX.keys()),
                        output_dict=True
                    )
                    df_report = pd.DataFrame(report_dict).T
                    st.dataframe(df_report)

                    st.subheader("🥧 Proporsi Sentimen Prediksi")
                    counts = df_eval_input['predicted_category'].value_counts()
                    fig_pie = px.pie(values=counts, names=counts.index, title='Proporsi Sentimen Prediksi',
                                     color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'blue'})
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.warning("Tidak ada data dengan label yang valid untuk evaluasi.")
    else:
        st.info("Silakan proses data terlebih dahulu di halaman 'Preprocessing' atau unggah file yang sudah diproses.")

# -------------------------
# Prediksi (REVISED: safe batch inference)
# -------------------------
elif page == "Prediksi":
    st.header("🔮 Prediksi Sentimen")

    st.subheader("Prediksi dari Data yang Diproses")
    if 'df_preprocessed' in st.session_state and not st.session_state.df_preprocessed.empty:
        if st.button("Mulai Prediksi Batch", use_container_width=True):
            df_to_predict = st.session_state.df_preprocessed.copy()

            # 1) Ensure text column exists & safe
            if 'review_text_normalizedjoin' not in df_to_predict.columns:
                st.error("Kolom 'review_text_normalizedjoin' tidak ditemukan di data. Silakan preprocessing terlebih dahulu.")
            else:
                # Fill NaN, strip, then drop truly empty rows
                df_to_predict['review_text_normalizedjoin'] = df_to_predict['review_text_normalizedjoin'].fillna("").astype(str).str.strip()
                df_to_predict = df_to_predict[df_to_predict['review_text_normalizedjoin'] != ""].reset_index(drop=True)

                if df_to_predict.empty:
                    st.warning("Tidak ada teks valid untuk diprediksi setelah preprocessing.")
                else:
                    texts = df_to_predict['review_text_normalizedjoin'].tolist()
                    total = len(texts)

                    # Tokenize batch (let tokenizer handle padding)
                    try:
                        enc = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
                    except Exception:
                        # Fallback: tokenize per item into a list to avoid tokenizer side effects
                        enc_list = [tokenizer(t, return_tensors='pt', truncation=True, padding='max_length', max_length=512) for t in texts]
                        # Merge into batch tensors manually (less common path)
                        enc = {}
                        for k in enc_list[0].keys():
                            enc[k] = torch.cat([e[k] for e in enc_list], dim=0)

                    # Move tensors to correct device
                    enc = {k: v.to(device) for k, v in enc.items()}

                    # Run inference in batch safely
                    with torch.no_grad():
                        try:
                            logits = model(**enc).logits  # shape: (batch_size, num_labels)
                        except Exception as e:
                            st.error(f"Gagal melakukan inferensi batch: {e}")
                            logits = None

                    predictions = []
                    confidence_scores = []

                    if logits is None:
                        # fallback: run per-sample safe inference
                        for text in texts:
                            if not text.strip():
                                predictions.append('unknown')
                                confidence_scores.append("0.00%")
                                continue
                            enc_single = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                            enc_single = {k: v.to(device) for k, v in enc_single.items()}
                            with torch.no_grad():
                                out = model(**enc_single).logits
                            out = out.detach().cpu().float().squeeze()
                            if torch.isnan(out).any() or torch.isinf(out).any() or out.numel() == 0:
                                predictions.append('unknown')
                                confidence_scores.append("0.00%")
                            else:
                                probs = F.softmax(out, dim=-1).numpy()
                                label_id = int(np.argmax(probs))
                                conf = float(probs[label_id]) * 100.0
                                predictions.append(INDEX_TO_LABEL.get(label_id, 'unknown'))
                                confidence_scores.append(f"{conf:.2f}%")
                    else:
                        # process batch logits safely
                        logits_cpu = logits.detach().cpu().float().numpy()  # shape (N, C)
                        # check for nan/inf rows
                        for row in logits_cpu:
                            if np.isnan(row).any() or np.isinf(row).any() or row.size == 0:
                                predictions.append('unknown')
                                confidence_scores.append("0.00%")
                            else:
                                probs = np.exp(row - np.max(row))  # stable softmax numerator
                                probs = probs / probs.sum()
                                label_id = int(np.argmax(probs))
                                conf = float(probs[label_id]) * 100.0
                                predictions.append(INDEX_TO_LABEL.get(label_id, 'unknown'))
                                confidence_scores.append(f"{conf:.2f}%")

                    df_to_predict['predicted_category'] = predictions
                    df_to_predict['confidence'] = confidence_scores

                    st.success("Prediksi batch selesai! ✅")
                    st.dataframe(df_to_predict[['review_text', 'predicted_category', 'confidence']].head())

                    st.subheader("🥧 Distribusi Sentimen Hasil Prediksi")
                    counts = df_to_predict['predicted_category'].value_counts()
                    fig_pie = px.pie(values=counts, names=counts.index, title='Distribusi Sentimen Hasil Prediksi',
                                     color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'blue'})
                    st.plotly_chart(fig_pie, use_container_width=True)

                    csv = df_to_predict.to_csv(index=False).encode('utf-8')
                    st.download_button("Unduh Hasil Prediksi", csv, "predicted_data.csv", "text/csv", use_container_width=True)
    else:
        st.info("Silakan proses data terlebih dahulu di halaman 'Preprocessing' untuk memulai prediksi batch.")

    # -------------------------
    # Prediksi Ulasan Tunggal (REVISED: safe)
    # -------------------------
    st.subheader("Prediksi Ulasan Tunggal")
    user_input = st.text_area("Masukkan ulasan:", height=150)
    if st.button("🎯 Hasil Deteksi", use_container_width=True):
        if user_input.strip():
            processed_text = preprocess_single_text(user_input)
            if not processed_text.strip():
                st.warning("Teks setelah preprocessing menjadi kosong. Coba masukkan teks lain.")
            else:
                try:
                    enc = tokenizer(processed_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                    enc = {k: v.to(device) for k, v in enc.items()}
                    with torch.no_grad():
                        logits = model(**enc).logits
                    # ensure vector
                    logits = logits.detach().cpu().float().squeeze()
                    if torch.isnan(torch.tensor(logits)).any() or torch.isinf(torch.tensor(logits)).any():
                        label = 'unknown'
                        conf = 0.0
                    else:
                        probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
                        label_id = int(np.argmax(probs))
                        label = INDEX_TO_LABEL.get(label_id, 'unknown')
                        conf = float(probs[label_id]) * 100.0
                except Exception as e:
                    st.error(f"Gagal memprediksi: {e}")
                    label = 'unknown'
                    conf = 0.0

                st.subheader("Hasil Prediksi:")
                if label == 'positive':
                    st.success(f"Sentimen: **{label}** ({conf:.2f}%)")
                elif label == 'negative':
                    st.error(f"Sentimen: **{label}** ({conf:.2f}%)")
                elif label == 'unknown':
                    st.warning("Hasil prediksi tidak dapat ditentukan (unknown).")
                else:
                    st.info(f"Sentimen: **{label}** ({conf:.2f}%)")

st.markdown('</div>', unsafe_allow_html=True)
