import streamlit as st
import cv2
import numpy as np
from PIL import Image
from joblib import load
from skimage.feature import greycomatrix, greycoprops

# Load model, scaler, dan PCA
model = load("best_knn_model.pkl")
scaler = load("scaler.pkl")
pca = load("pca.pkl")

# Label kelas
label_mapping = {
    0: "Jalan Retak",
    1: "Jalan Lubang",
    2: "Jalan Tidak Rusak"
}

# Ekstraksi fitur GLCM (harus identik dengan training!)
def extract_glcm_features(image):
    # Resize agar sama dengan data training
    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    glcm = greycomatrix(gray, distances=distances, angles=angles, symmetric=True, normed=True)
    features = []

    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in properties:
        vals = greycoprops(glcm, prop)
        features.append(np.mean(vals))  # rata-rata semua kombinasi

    return features

# UI Streamlit
st.set_page_config(page_title="Deteksi Kerusakan Jalan", page_icon="🛣️", layout="centered")
st.markdown("<h4 style='text-align: center;'>Deteksi Kerusakan Jalan</h4>", unsafe_allow_html=True)
st.caption("Upload gambar jalan untuk prediksi jenis kerusakannya.")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption=None, width=300)
    img_cv = np.array(image)

    # Ekstraksi fitur dan prediksi
    features = extract_glcm_features(img_cv)
    features_scaled = scaler.transform([features])
    features_pca = pca.transform(features_scaled)

    prediction = model.predict(features_pca)[0]
    proba = model.predict_proba(features_pca)[0]

    st.markdown("---")
    st.markdown(f"<p style='font-size: 16px;'>🧠 <b>Prediksi:</b> {label_mapping[prediction]}</p>", unsafe_allow_html=True)

    st.markdown("<p style='margin-bottom: 4px;'>📊 <b>Probabilitas:</b></p>", unsafe_allow_html=True)
    for i, p in enumerate(proba):
        st.markdown(f"<small>{label_mapping[i]}: {p*100:.1f}%</small>", unsafe_allow_html=True)
        st.progress(float(p))
