import streamlit as st
import pandas as pd
import os
import gdown
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import requests
from deep_translator import GoogleTranslator

# ================================
# KONFIGURASI STREAMLIT
# ================================
st.set_page_config(page_title="🍜 Sistem Rekomendasi Anime", layout="wide")
st.markdown("<h1 style='text-align: center;'>🍜 Sistem Rekomendasi Anime</h1>", unsafe_allow_html=True)
st.caption("Powered by K-Nearest Neighbors, Jikan API & Google Drive")

# ================================
# AMBIL CSV DARI GOOGLE DRIVE
# ================================
@st.cache_data
def download_and_load_csv(file_id, filename):
    output = f"/tmp/{filename}"
    if not os.path.exists(output):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
    return pd.read_csv(output)

# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_data():
    anime_file_id = "1QeLqFognHnifo9EDQz_19NfNiwbPIV3x"   # ganti dengan file ID anime.csv
    rating_file_id = "1rLbB5n1LBTUPAsU9g-5SX1ru1IeOy3Ab"  # ganti dengan file ID rating.csv

    anime = download_and_load_csv(anime_file_id, "anime.csv")[["anime_id", "name"]].dropna().drop_duplicates("anime_id")
    ratings = download_and_load_csv(rating_file_id, "rating.csv")
    ratings = ratings[ratings["rating"] > 0]
    data = ratings.merge(anime, on="anime_id")
    return anime, data

# ================================
# SIAPKAN MATRIX
# ================================
@st.cache_data
def prepare_matrix(data, num_users=800, num_anime=400):
    top_users = data['user_id'].value_counts().head(num_users).index
    top_anime = data['name'].value_counts().head(num_anime).index
    filtered = data[data['user_id'].isin(top_users) & data['name'].isin(top_anime)]
    matrix = filtered.pivot_table(index='name', columns='user_id', values='rating').fillna(0)
    return matrix.astype('float32')

@st.cache_resource
def train_model(matrix):
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(csr_matrix(matrix.values))
    return model

# ================================
# GET REKOMENDASI
# ================================
def get_recommendations(title, matrix, model, n=5):
    if title not in matrix.index:
        return []
    idx = matrix.index.get_loc(title)
    dists, idxs = model.kneighbors(matrix.iloc[idx, :].values.reshape(1, -1), n_neighbors=n+1)
    return [(matrix.index[i], 1 - dists.flatten()[j]) for j, i in enumerate(idxs.flatten()[1:])]

# ================================
# API JIKAN: GAMBAR + SINOPSIS
# ================================
def get_anime_details(anime_title):
    try:
        response = requests.get("https://api.jikan.moe/v4/anime", params={"q": anime_title, "limit": 1})
        if response.status_code == 200 and response.json()["data"]:
            data = response.json()["data"][0]
            image = data["images"]["jpg"].get("image_url", "")
            synopsis_en = data.get("synopsis", "Sinopsis tidak tersedia.")
            genres = ", ".join([g["name"] for g in data.get("genres", [])])
            synopsis_id = GoogleTranslator(source='auto', target='id').translate(synopsis_en)
            return image, synopsis_id, genres
    except:
        pass
    return "", "Sinopsis tidak tersedia.", "-"

# ================================
# TOP 5 LEADERBOARD
# ================================
@st.cache_data
def get_top_5_anime(data):
    grouped = data.groupby("name").agg(
        avg_rating=("rating", "mean"),
        num_ratings=("rating", "count")
    ).reset_index()
    top_anime = grouped[grouped["num_ratings"] > 10].sort_values(by="avg_rating", ascending=False).head(5)
    return top_anime

# ================================
# LOAD DATA
# ================================
with st.spinner("🔄 Memuat data..."):
    anime, data = load_data()
    matrix = prepare_matrix(data)
    model = train_model(matrix)

# ================================
# LEADERBOARD
# ================================
st.subheader("🏆 Top 5 Anime Berdasarkan Rating")
top5_df = get_top_5_anime(data)

# Buat kolom sebanyak jumlah top 5
cols = st.columns(5)

for i, row in enumerate(top5_df.itertuples()):
    with cols[i]:
        image_url, _, _ = get_anime_details(row.name)
        st.image(image_url, caption=row.name, use_container_width=True)
        st.markdown(f"⭐ **Rating:** `{row.avg_rating:.2f}`")
        st.markdown(f"👥 **Jumlah Rating:** `{row.num_ratings}`")

# ================================
# FITUR REKOMENDASI
# ================================
st.markdown("## 🎮 Pilih Anime Favorit Kamu")
anime_list = list(matrix.index)
selected_anime = st.selectbox("Pilih anime yang kamu suka:", anime_list)

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("🔍 Tampilkan Rekomendasi"):
    st.session_state.history.append(selected_anime)
    rekomendasi = get_recommendations(selected_anime, matrix, model, n=5)

    st.subheader(f"✨ Rekomendasi berdasarkan: {selected_anime}")
    cols = st.columns(5)
    for i, (rec_title, similarity) in enumerate(rekomendasi):
        with cols[i % 5]:
            image_url, synopsis, genres = get_anime_details(rec_title)
            st.image(image_url, caption=rec_title, use_container_width=True)
            st.markdown(f"*Genre:* {genres}")
            st.markdown(f"🔗 Kemiripan: `{similarity:.2f}`")
            with st.expander("📓 Lihat Sinopsis"):
                st.markdown(synopsis)

# ================================
# RIWAYAT PILIHAN
# ================================
if st.session_state.history:
    st.markdown("### 🕓 Riwayat Anime yang Kamu Pilih:")
    history = st.session_state.history[-5:]
    cols = st.columns(len(history))
    for i, title in enumerate(reversed(history)):
        with cols[i]:
            image_url, _, _ = get_anime_details(title)
            st.image(image_url, caption=title, use_container_width=True)