import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import requests
import gdown
import os
from deep_translator import GoogleTranslator

# ================================
# KONFIGURASI STREAMLIT
# ================================
st.set_page_config(page_title="🎌 Sistem Rekomendasi Anime", layout="wide")
st.markdown("<h1 style='text-align: center;'>🎌 Sistem Rekomendasi Anime</h1>", unsafe_allow_html=True)
st.caption("Powered by K-Nearest Neighbors & Jikan API + Google Drive")

# ================================
# FUNGSI AMBIL DATA DARI GDRIVE
# ================================
@st.cache_data
def download_and_load_csv(file_id, filename):
    output = f"/tmp/{filename}"
    if not os.path.exists(output):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
    return pd.read_csv(output)

# ================================
# AMBIL DAN PROSES DATA
# ================================
@st.cache_data
def load_data():
    anime_file_id = "1QelqFognHnifo9EDQz_19NfNiwbPIV3x"  # Ganti dengan file ID Google Drive anime.csv
    rating_file_id = "1rlBb5n1LBTUPAsU9g-S5X1ru1IeOy3Ab"  # Ganti dengan file ID Google Drive rating.csv

    anime = download_and_load_csv(anime_file_id, "anime.csv")[["anime_id", "name"]].dropna().drop_duplicates("anime_id")
    ratings = download_and_load_csv(rating_file_id, "rating.csv")
    ratings = ratings[ratings["rating"] > 0]

    data = ratings.merge(anime, on="anime_id")
    return anime, data

@st.cache_data
def prepare_matrix(data, num_users=800, num_anime=1000):
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
# FUNGSI REKOMENDASI
# ================================
def get_recommendations(title, matrix, model, n=5):
    if title not in matrix.index:
        return []
    idx = matrix.index.get_loc(title)
    dists, idxs = model.kneighbors(matrix.iloc[idx, :].values.reshape(1, -1), n_neighbors=n+1)
    return [
        (matrix.index[i], 1 - dists.flatten()[j])
        for j, i in enumerate(idxs.flatten()[1:])
        if matrix.index[i] != title
    ]

# ================================
# JIKAN API AMBIL GAMBAR & GENRE
# ================================
def get_anime_details(anime_title):
    try:
        response = requests.get("https://api.jikan.moe/v4/anime", params={"q": anime_title, "limit": 1})
        if response.status_code == 200 and response.json()["data"]:
            data = response.json()["data"][0]
            image = data["images"]["jpg"].get("image_url", "")
            synopsis_en = data.get("synopsis", "")
            genres = ", ".join([g["name"] for g in data.get("genres", [])])
            synopsis_id = GoogleTranslator(source='auto', target='id').translate(synopsis_en)
            return image, synopsis_id, genres
    except Exception as e:
        print(f"Error for {anime_title}: {e}")
    return "", "Sinopsis tidak tersedia.", "-"
# ================================
# LOAD DATASET & MODEL
@@ -104,43 +92,27 @@
    matrix = prepare_matrix(data)
    model = train_model(matrix)
# ================================
# UI: PILIH FAVORIT
# ================================
st.markdown("## 🎮 Pilih Anime Favorit Kamu")
anime_titles = list(matrix.index)
selected_anime = st.selectbox("Pilih anime yang kamu suka:", anime_titles)

# ================================
# TAMPILKAN REKOMENDASI
# ================================
if st.button("🔍 Tampilkan Rekomendasi"):
    recommendations = get_recommendations(selected_anime, matrix, model)

    if recommendations:
        st.subheader(f"✨ Rekomendasi berdasarkan: {selected_anime}")
        cols = st.columns(5)
        for i, (rec_title, similarity) in enumerate(recommendations):
            with cols[i % 5]:
                image_url, synopsis, genres = get_anime_details(rec_title)
                st.image(image_url, caption=rec_title, use_container_width=True)
                st.markdown(f"**Genre:** {genres}")
                st.markdown(f"🔗 *Kemiripan:* `{similarity:.2f}`")
                with st.expander("📖 Sinopsis"):
                    st.write(synopsis)
    else:
        st.warning("❗ Anime tidak ditemukan dalam dataset.")
