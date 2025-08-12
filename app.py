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
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import time
import datetime
from google_play_scraper import Sort, reviews_all
from nltk import ngrams
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from streamlit_carousel import carousel
import base64 # Modul untuk encoding Base64
import os # Modul untuk memeriksa path file

# =========================
# Fungsi untuk mengubah gambar menjadi Base64
# =========================
@st.cache_data
def get_image_as_base64(path):
    """Fungsi ini membaca file gambar lokal dan mengubahnya menjadi string Base64."""
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
# Panggil fungsi untuk mendapatkan gambar header dalam format Base64
# Pastikan Anda memiliki file 'header.png' di dalam folder 'image'
header_image_path = 'image/fix.png'
img_base64 = get_image_as_base64(header_image_path)

# Siapkan style CSS. Jika gambar ditemukan, gunakan sebagai background. Jika tidak, gunakan warna solid.
if img_base64:
    background_style = f"""
        background-image: url(data:image/png;base64,{img_base64});
        background-size: cover;
        background-position: center;
    """
else:
    # Fallback jika gambar tidak ditemukan
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
    color: white; /* Mengubah warna teks default di dalam card menjadi putih */
}}

/* Memastikan semua teks utama di dalam card menjadi putih agar kontras */
.main-card h1, .main-card h2, .main-card h3, .main-card p, .main-card div[data-testid="stMarkdown"], .main-card .st-emotion-cache-1g6goon {{
    color: white !important;
}}

/* Styling untuk tabel informasi penyusun */
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
    width: 150px; /* Atur lebar kolom label */
}}
</style>
""", unsafe_allow_html=True)

# =========================
# Path Model & Label Mapping
# =========================
MODEL_PATH = 'wahyuaprian/indobert-sentiment-mcgogo'
APP_ID = "com.mobilechess.gp"
LABEL_TO_INDEX = {'positive': 0, 'neutral': 1, 'negative': 2}
INDEX_TO_LABEL = {0: 'positive', 1: 'neutral', 2: 'negative'}

# =========================
# Load Model dan Tokenizer
# =========================
@st.cache_resource
def load_model_and_tokenizer():
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        config = BertConfig.from_pretrained(MODEL_PATH)
        config.num_labels = len(LABEL_TO_INDEX)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH, config=config)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Gagal memuat model atau tokenizer: {e}. Pastikan Anda berada di direktori proyek yang benar.")
        st.stop()

tokenizer, model, device = load_model_and_tokenizer()

# =========================
# Fungsi Preprocessing
# =========================
@st.cache_data
def load_stopwords():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')
        
    list_stopwords = stopwords.words('indonesian')
    list_stopwords.extend([
        "yg","dg","rt","dgn","ny","d","klo","kalo","amp","biar","bikin",
        "bilang","gak","ga","krn","nya","nih","sih","si","tau","tdk","tuh",
        "utk","ya","jd","jgn","sdh","aja","n","t","nyg","hehe","pen","u",
        "nan","loh","yah","dr","gw","gue"
    ])
    return set(list_stopwords)

list_stopwords = load_stopwords()

@st.cache_data
def load_kamus_baku():
    kamus_baku_path = './data/kamus_baku.csv'
    df_kamus = pd.read_csv(kamus_baku_path, encoding='latin-1')
    return dict(zip(df_kamus.iloc[:,0], df_kamus.iloc[:,1]))

normalizad_word_dict = load_kamus_baku()

character_for_cleaning = list(string.ascii_letters) + list(".,;:-...?!()[]{}<>\"/\\#-@")

def repeatcharClean(text):
    for char in character_for_cleaning:
        for n in range(5,2,-1):
            text = text.replace(char*n, char)
    return text

def clean_review(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'#(\S+)', r'\1', text)
    text = re.sub('[^a-zA-Z]+', ' ', text)
    text = repeatcharClean(text)
    text = re.sub('[ ]+',' ',text).strip()
    return text

def tokenization_review_func(text):
    return word_tokenize(text)

def stopwords_removal_func_wrapper(words):
    return [w for w in words if w not in list_stopwords]

@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

stemmer = load_stemmer()

def stemming_func_wrapper(words):
    return [stemmer.stem(w) for w in words]

def normalized_term_func_wrapper(document):
    return [normalizad_word_dict.get(term, term) for term in document]

def preprocess_dataframe(df_raw_input):
    df_processed = df_raw_input.copy()
    
    with st.expander("Langkah 1: Case Folding & Cleaning"):
        st.info("Mengubah semua teks menjadi huruf kecil, menghapus URL, emoji, dan karakter non-alfabet.")
        df_processed['review_text_cleaned'] = df_processed['review_text'].apply(clean_review)
        st.dataframe(df_processed[['review_text', 'review_text_cleaned']].head())
    
    with st.expander("Langkah 2: Tokenization & Stopwords Removal"):
        st.info("Memisahkan teks menjadi kata-kata (token) dan menghapus kata-kata umum (stopwords).")
        df_processed['review_text_tokens'] = df_processed['review_text_cleaned'].apply(tokenization_review_func)
        df_processed['review_text_tokens_WSW'] = df_processed['review_text_tokens'].apply(stopwords_removal_func_wrapper)
        st.dataframe(df_processed[['review_text_cleaned', 'review_text_tokens_WSW']].head())

    with st.expander("Langkah 3: Stemming"):
        st.info("Mengubah kata berimbuhan menjadi kata dasar.")
        df_processed['review_text_stemmed'] = df_processed['review_text_tokens_WSW'].apply(stemming_func_wrapper)
        st.dataframe(df_processed[['review_text_tokens_WSW', 'review_text_stemmed']].head())

    with st.expander("Langkah 4: Normalisasi"):
        st.info("Mengubah kata-kata tidak baku menjadi kata baku berdasarkan kamus.")
        df_processed['review_text_normalized'] = df_processed['review_text_stemmed'].apply(normalized_term_func_wrapper)
        st.dataframe(df_processed[['review_text_stemmed', 'review_text_normalized']].head())
    
    df_processed["review_text_normalizedjoin"] = [' '.join(word) for word in df_processed["review_text_normalized"]]
    df_processed.replace(['',' '], np.nan, inplace=True)
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
    btn_class = "sidebar-button-active" if st.session_state.page == key else "sidebar-button"
    if st.sidebar.button(label, key=f"menu_{key}", use_container_width=True):
        st.session_state.page = key
        st.rerun()

page = st.session_state.page

# =========================
# Konten Halaman
# =========================
st.markdown('<div class="main-card">', unsafe_allow_html=True)

if page == "Beranda":
    st.title("📊 Analisis Sentimen Magic Chess : Go Go Menggunakan Model IndoBERT")
    
    # Bagian untuk gambar utama (DIPINDAHKAN KE ATAS)
    try:
        st.image("image/home.jpg", use_container_width=True)
    except Exception:
        st.image("https://placehold.co/1200x400/1a202c/ffffff?text=Magic+Chess+Home", use_container_width=True)
        st.info("Gambar 'image/home.jpg' tidak ditemukan. Menampilkan gambar placeholder.")
    
    # --- Teks Deskripsi (REVISI) ---
    st.markdown("""
    Selamat datang di dasbor **Analisis Sentimen Ulasan Aplikasi Magic Chess: Go Go**.

    Aplikasi ini dirancang untuk menganalisis dan mengklasifikasikan sentimen dari ulasan pengguna aplikasi *Magic Chess: Go Go* di Google Play Store. Dengan memanfaatkan kecanggihan model **IndoBERT**, aplikasi ini mampu mengubah data kualitatif (teks ulasan) menjadi wawasan kuantitatif (positif, negatif, dan netral) melalui antarmuka yang interaktif.

    Dasbor ini dibuat sebagai bagian dari pemenuhan tugas akhir yang disusun oleh:
    """)
    
    # --- Tabel Informasi Penyusun (REVISI) ---
    st.markdown("""
    <table class="author-table">
        <tr>
            <td>Nama</td>
            <td>: Wahyu Aprian Hadiansyah</td>
        </tr>
        <tr>
            <td>NPM</td>
            <td>: 11121284</td>
        </tr>
        <tr>
            <td>Kelas</td>
            <td>: 4KA23</td>
        </tr>
        <tr>
            <td>Program Studi</td>
            <td>: Sistem Informasi</td>
        </tr>
        <tr>
            <td>Fakultas</td>
            <td>: Ilmu Komputer dan Teknologi Informasi</td>
        </tr>
        <tr>
            <td>Universitas</td>
            <td>: Universitas Gunadarma</td>
        </tr>
        <tr>
            <td>Tahun</td>
            <td>: 2025</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)

    # --- BAGIAN CAROUSEL ---
    st.subheader("Sinergi Hero Magic Chess Go Go")
    
    # Data untuk carousel
    # Pastikan path gambar lokal Anda sudah benar.
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
    
    carousel(items=items)

elif page == "Scraping Data":
    st.header("📥 Scraping Data dari Google Play Store")
    st.write(f"Mengambil ulasan untuk **Magic Chess: Bang Bang** (App ID: `{APP_ID}`).")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Tanggal Mulai", datetime.date.today() - datetime.timedelta(days=30))
        num_reviews = st.number_input("Jumlah ulasan yang diinginkan:", min_value=10, max_value=20000, value=100)
    with col2:
        end_date = st.date_input("Tanggal Selesai", datetime.date.today())
        lang = st.selectbox("Pilih bahasa:", options=['id', 'en'], index=0)
        country = st.selectbox("Pilih negara:", options=['id', 'us'], index=0)

    start_dt = datetime.datetime.combine(start_date, datetime.time.min)
    end_dt = datetime.datetime.combine(end_date, datetime.time.max)
    
    st.info("Aplikasi akan mengambil ulasan terbaru, memfilternya sesuai rentang tanggal, lalu memilih jumlah yang Anda inginkan.")

    if st.button("Mulai Scraping", use_container_width=True):
        if num_reviews == 0:
            st.error("Jumlah ulasan tidak boleh 0.")
        else:
            with st.spinner("Mengambil sejumlah besar ulasan..."):
                try:
                    scraped_reviews = reviews_all(
                        APP_ID,
                        lang=lang,
                        country=country,
                        sort=Sort.NEWEST,
                        count=num_reviews * 2 # Ambil lebih banyak untuk difilter
                    )
                    
                    if not scraped_reviews:
                        st.warning("Tidak ada ulasan yang ditemukan. Silakan cek rentang tanggal.")
                        st.session_state.df_scraped = pd.DataFrame()
                    else:
                        df_scraped = pd.DataFrame(scraped_reviews)
                        df_scraped['timestamp'] = pd.to_datetime(df_scraped['at'])
                        
                        df_filtered_by_date = df_scraped[(df_scraped['timestamp'] >= start_dt) & (df_scraped['timestamp'] <= end_dt)].reset_index(drop=True)
                        
                        if df_filtered_by_date.empty:
                            st.warning("Tidak ada ulasan yang ditemukan dalam rentang tanggal tersebut. Coba rentang tanggal yang lebih luas.")
                            st.session_state.df_scraped = pd.DataFrame()
                        else:
                            final_df = df_filtered_by_date.head(num_reviews)
                            
                            final_df['category'] = final_df['score'].apply(map_score_to_sentiment)

                            final_df = final_df.rename(columns={'content': 'review_text'})
                            final_df = final_df.drop(columns=['at'])

                            st.success(f"✅ Berhasil mengambil {len(final_df)} ulasan dalam rentang tanggal yang dipilih!")
                            st.dataframe(final_df[['review_text', 'category', 'score', 'timestamp']].head())
                            st.session_state.df_scraped = final_df
                            
                            csv = final_df.to_csv(index=False).encode('utf-8')
                            st.download_button("Unduh Hasil Scraping", csv, "scraped_data.csv", "text/csv")
                
                except Exception as e:
                    st.error(f"Gagal mengambil data. Error: {e}")

elif page == "Preprocessing":
    st.header("🧹 Preprocessing Data Ulasan")
    st.write("Disarankan untuk mengunggah file yang sudah memiliki label sentimen (positive, negative, neutral) jika ingin melakukan evaluasi model.")

    df_raw = None
    if 'df_scraped' in st.session_state and not st.session_state.df_scraped.empty:
        st.subheader("Pilih Sumber Data")
        use_scraped = st.checkbox("Gunakan data yang telah di-scraping", value=True)
        if use_scraped:
            df_raw = st.session_state.df_scraped
            st.subheader("📄 Data Asli (dari Scraping)")
            st.dataframe(df_raw[['review_text', 'category']].head())
    
    if df_raw is None:
        uploaded_file = st.file_uploader("Atau, pilih file TSV/CSV", type=["tsv", "csv"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.tsv'):
                df_raw = pd.read_csv(uploaded_file, sep='\t', names=['review_text', 'category'])
            else:
                df_raw = pd.read_csv(uploaded_file)
                if df_raw.shape[1] == 1:
                    df_raw.columns = ['review_text']
                    df_raw['category'] = 'unknown'
            st.subheader("📄 Data Asli (dari File)")
            st.dataframe(df_raw[['review_text', 'category']].head())
            
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
                st.plotly_chart(fig)
            else:
                counts = df_raw['category'].value_counts().reset_index()
                counts.columns = ['category', 'count']
                fig = px.bar(counts, x='category', y='count', title='Distribusi Sentimen Data Asli',
                             color='category', color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'blue'})
                st.plotly_chart(fig)
        else:
            st.info("Data yang Anda gunakan tidak memiliki label sentimen yang valid. Grafik distribusi sentimen tidak dapat dibuat.")

        if st.button("🚀 Mulai Preprocessing", use_container_width=True):
            df_processed = preprocess_dataframe(df_raw.copy())
            st.success("✅ Preprocessing selesai!")
            st.subheader("✅ Hasil Preprocessing Akhir")
            
            st.dataframe(df_processed[['review_text', 'review_text_cleaned', 'review_text_tokens', 'review_text_tokens_WSW', 'review_text_normalized', 'review_text_normalizedjoin', 'category']].head())

            st.subheader("➡️ Distribusi Panjang Ulasan")
            df_processed['length_original'] = df_raw['review_text'].apply(lambda x: len(str(x).split()))
            df_processed['length_preprocessed'] = df_processed['review_text_normalizedjoin'].apply(lambda x: len(str(x).split()))
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=df_processed['length_original'], name='Panjang Asli'))
            fig_hist.add_trace(go.Histogram(x=df_processed['length_preprocessed'], name='Panjang Setelah Preprocessing'))
            fig_hist.update_layout(barmode='overlay', title_text='Distribusi Panjang Ulasan', xaxis_title_text='Jumlah Kata', yaxis_title_text='Frekuensi')
            fig_hist.update_traces(opacity=0.75)
            st.plotly_chart(fig_hist)
            
            corpus = " ".join(df_processed['review_text_normalizedjoin'])
            if corpus:
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
                st.plotly_chart(fig2)

                st.subheader("➡️ 15 Bigram Paling Sering Muncul")
                df_bigrams = get_top_ngrams(corpus, n=2, top=15)
                df_bigrams['Ngram'] = df_bigrams['Ngram'].apply(lambda x: ' '.join(x))
                fig3 = px.bar(df_bigrams, x='Ngram', y='Frequency', title='15 Bigram Paling Sering Muncul')
                st.plotly_chart(fig3)
            else:
                st.warning("Tidak ada kata-kata yang tersisa setelah preprocessing. Visualisasi tidak dapat dibuat.")

            csv = df_processed.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Hasil Preprocessing", csv, "preprocessed.csv", "text/csv", use_container_width=True)
            st.session_state.df_preprocessed = df_processed
    else:
        st.info("Silakan unggah atau scraping data terlebih dahulu.")

elif page == "Modeling & Evaluasi":
    st.header("📊 Modeling & Evaluasi IndoBERT")
    df_eval_input = None
    if 'df_preprocessed' in st.session_state:
        df_eval_input = st.session_state.df_preprocessed
        st.write("Menggunakan data yang sudah diproses dari halaman Preprocessing.")
    else:
        uploaded_eval_file = st.file_uploader("Upload File Preprocessed", type=["tsv", "csv"], key="eval_upload")
        if uploaded_eval_file is not None:
            df_eval_input = pd.read_csv(uploaded_eval_file, sep='\t' if uploaded_eval_file.name.endswith('.tsv') else ',')

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

                for idx, text in enumerate(df_eval_input['review_text_normalizedjoin']):
                    enc = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                    enc = {k: v.to(device) for k, v in enc.items()}
                    with torch.no_grad():
                        logits = model(**enc).logits
                    label_id = torch.argmax(logits, dim=-1).item()
                    predictions.append(INDEX_TO_LABEL[label_id])

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
                    cm = confusion_matrix(y_true_filtered, y_pred_filtered)
                    fig_cm = px.imshow(cm, x=list(LABEL_TO_INDEX.keys()), y=list(LABEL_TO_INDEX.keys()), text_auto=True,
                                       labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                                       title="Confusion Matrix")
                    st.plotly_chart(fig_cm)

                    st.subheader("📝 Classification Report")
                    report_dict = classification_report(y_true_filtered, y_pred_filtered, target_names=LABEL_TO_INDEX.keys(), output_dict=True)
                    df_report = pd.DataFrame(report_dict).T
                    st.dataframe(df_report)

                    st.subheader("🥧 Proporsi Sentimen Prediksi")
                    counts = df_eval_input['predicted_category'].value_counts()
                    fig_pie = px.pie(values=counts, names=counts.index, title='Proporsi Sentimen Prediksi',
                                     color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'blue'})
                    st.plotly_chart(fig_pie)
                else:
                    st.warning("Tidak ada data dengan label yang valid untuk evaluasi.")
    else:
        st.info("Silakan proses data terlebih dahulu di halaman 'Preprocessing' atau unggah file yang sudah diproses.")

elif page == "Prediksi":
    st.header("🔮 Prediksi Sentimen")
    
    st.subheader("Prediksi dari Data yang Diproses")
    if 'df_preprocessed' in st.session_state and not st.session_state.df_preprocessed.empty:
        if st.button("Mulai Prediksi Batch", use_container_width=True):
            df_to_predict = st.session_state.df_preprocessed.copy()
            progress_bar = st.progress(0)
            status_text = st.empty()
            predictions = []
            confidence_scores = []
            total = len(df_to_predict)
            start_time = time.time()
            
            for idx, text in enumerate(df_to_predict['review_text_normalizedjoin']):
                enc = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    logits = model(**enc).logits
                label_id = torch.argmax(logits, dim=-1).item()
                conf = F.softmax(logits, dim=-1).squeeze()[label_id].item()*100
                predictions.append(INDEX_TO_LABEL[label_id])
                confidence_scores.append(f"{conf:.2f}%")
                
                progress = (idx + 1) / total
                status_text.text(f"Memproses data {idx+1}/{total} ({progress*100:.1f}%)")
                progress_bar.progress(int(progress * 100))

            df_to_predict['predicted_category'] = predictions
            df_to_predict['confidence'] = confidence_scores
            st.success("Prediksi batch selesai! ✅")
            st.dataframe(df_to_predict[['review_text', 'predicted_category', 'confidence']].head())
            
            st.subheader("🥧 Distribusi Sentimen Hasil Prediksi")
            counts = df_to_predict['predicted_category'].value_counts()
            fig_pie = px.pie(values=counts, names=counts.index, title='Distribusi Sentimen Hasil Prediksi',
                             color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'blue'})
            st.plotly_chart(fig_pie)
            
            csv = df_to_predict.to_csv(index=False).encode('utf-8')
            st.download_button("Unduh Hasil Prediksi", csv, "predicted_data.csv", "text/csv", use_container_width=True)
    else:
        st.info("Silakan proses data terlebih dahulu di halaman 'Preprocessing' untuk memulai prediksi batch.")


    st.subheader("Prediksi Ulasan Tunggal")
    user_input = st.text_area("Masukkan ulasan:", height=150)
    if st.button("🎯 Hasil Deteksi", use_container_width=True):
        if user_input.strip():
            processed_text = preprocess_single_text(user_input)
            enc = tokenizer(processed_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
            label_id = torch.argmax(logits, dim=-1).item()
            conf = F.softmax(logits, dim=-1).squeeze()[label_id].item()*100
            predicted_label = INDEX_TO_LABEL[label_id]
            st.subheader("Hasil Prediksi:")
            if predicted_label == 'positive':
                st.success(f"Sentimen: **{predicted_label}** ({conf:.2f}%)")
            elif predicted_label == 'negative':
                st.error(f"Sentimen: **{predicted_label}** ({conf:.2f}%)")
            else:
                st.info(f"Sentimen: **{predicted_label}** ({conf:.2f}%)")
        else:
            st.warning("Mohon masukkan ulasan untuk dianalisis.")

st.markdown('</div>', unsafe_allow_html=True)
