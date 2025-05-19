import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import requests
from deep_translator import GoogleTranslator

# ================================
# CONFIGURASI HALAMAN
# ================================
st.set_page_config(page_title="🎌 Rekomendasi Anime", layout="wide")
st.title("🎌 Sistem Rekomendasi Anime")

# ================================
# BACA DATA DARI GOOGLE DRIVE
# ================================
@st.cache_data
def load_from_gdrive(file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    return pd.read_csv(url)

@st.cache_data
def load_data():
    anime_file_id = "1QeLqFognHnifo9EDQz_19NFNiwbPIV3x"
    rating_file_id = "1rLbB5n1LBTUPAsU9g-SSX1ru1IeOy3Ab"

    anime = load_from_gdrive(anime_file_id)[["anime_id", "name"]].dropna().drop_duplicates("anime_id")
    ratings = load_from_gdrive(rating_file_id)
    ratings = ratings[ratings["rating"] > 0]
    data = ratings.merge(anime, on="anime_id")
    return anime, data

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
# GET INFO DARI JIKAN API
# ================================
def get_anime_details(anime_title):
    try:
        response = requests.get("https://api.jikan.moe/v4/anime", params={"q": anime_title, "limit": 1})
        if response.status_code == 200 and response.json()["data"]:
            data = response.json()["data"][0]
            image = data["images"]["jpg"].get("image_url", "assets/placeholder.jpg")
            synopsis_en = data.get("synopsis", "Sinopsis tidak tersedia.")
            genres = ", ".join([g["name"] for g in data.get("genres", [])])
            synopsis_id = GoogleTranslator(source='auto', target='id').translate(synopsis_en)
            return image, synopsis_id, genres
    except:
        pass
    return "assets/placeholder.jpg", "Sinopsis tidak tersedia.", "-"

# ================================
# FUNGSI REKOMENDASI
# ================================
def get_recommendations(title, matrix, model, n=5):
    if title not in matrix.index:
        return []
    idx = matrix.index.get_loc(title)
    dists, idxs = model.kneighbors(matrix.iloc[idx].values.reshape(1, -1), n_neighbors=n+1)
    return [(matrix.index[i], 1 - dists.flatten()[j]) for j, i in enumerate(idxs.flatten()[1:])]

# ================================
# LOAD & TRAIN
# ================================
st.sidebar.header("⚙️ Pengaturan Dataset")
num_users = st.sidebar.slider("Jumlah User Teratas", 100, 2000, 800, 100)
num_anime = st.sidebar.slider("Jumlah Anime Teratas", 100, 1000, 400, 50)

with st.spinner("Memuat data..."):
    anime, data = load_data()
    matrix = prepare_matrix(data, num_users, num_anime)
    model = train_model(matrix)

# ================================
# SELECT DAN TAMPILKAN REKOMENDASI
# ================================
st.markdown("## 🎮 Pilih anime favorit kamu:")
anime_titles = list(matrix.index)
selected_anime = st.selectbox("Pilih anime:", anime_titles)

if st.button("🔍 Tampilkan Rekomendasi"):
    st.markdown(f"### ✨ Rekomendasi berdasarkan: **{selected_anime}**")
    recommendations = get_recommendations(selected_anime, matrix, model)

    if recommendations:
        cols = st.columns(5)
        for i, (rec_title, similarity) in enumerate(recommendations):
            with cols[i % 5]:
                image_url, synopsis, genres = get_anime_details(rec_title)
                st.image(image_url, caption=rec_title, use_container_width=True)
                st.markdown(f"🎭 *Genre:* {genres}")
                st.markdown(f"🔗 *Kemiripan:* `{similarity:.2f}`")
                with st.expander("📖 Sinopsis"):
                    st.write(synopsis)
    else:
        st.warning("❗ Anime tidak ditemukan atau tidak memiliki cukup data.")

# ================================
# TAMPILKAN RIWAYAT PILIHAN
# ================================
if "history" not in st.session_state:
    st.session_state.history = []

if selected_anime not in st.session_state.history:
    st.session_state.history.append(selected_anime)

if st.session_state.history:
    st.markdown("## 🕓 Riwayat Anime yang Kamu Pilih:")
    cols = st.columns(min(5, len(st.session_state.history)))
    for i, title in enumerate(reversed(st.session_state.history[-5:])):
        with cols[i % 5]:
            image_url, _, _ = get_anime_details(title)
            st.image(image_url, caption=title, width=150)
