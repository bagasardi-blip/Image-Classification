import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Aplikasi Klasifikasi Gambar",
    page_icon="📷",
    layout="centered"
)

# Judul Aplikasi
st.title("📷 Aplikasi Klasifikasi Gambar")
st.write("Unggah gambar untuk mendeteksi apakah termasuk kategori **Negative** atau **Positive**.")

# Tentukan nama kelas sesuai dataset Anda
CLASS_NAMES = ['Negative', 'Positive']
MODEL_PATH = 'image_classification_model.h5'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None
    else:
        return None

# Memuat model
model = load_model()

if model is None:
    st.warning(f"Berkas model `{MODEL_PATH}` tidak ditemukan di direktori lokal. Pastikan Anda telah melatih model dan menyimpannya dengan nama tersebut, atau letakkan berkas model di folder yang sama dengan skrip ini.")
else:
    st.success("Model berhasil dimuat!")

# Widget unggah gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
    
    if model is not None:
        with st.spinner("Sedang mengklasifikasikan..."):
            # Praproses gambar agar sesuai dengan input model (150x150)
            img_resized = image.resize((150, 150))
            img_array = np.array(img_resized)
            
            # Memastikan gambar memiliki 3 channel (RGB)
            if img_array.shape[-1] != 3:
                img_array = img_resized.convert("RGB")
                img_array = np.array(img_array)
                
            img_batch = np.expand_dims(img_array, axis=0) # Tambah dimensi batch
            
            # Prediksi menggunakan model
            predictions = model.predict(img_batch)
            
            # Jika model menggunakan output logits tanpa softmax di akhir
            score = tf.nn.softmax(predictions[0])
            predicted_class = CLASS_NAMES[np.argmax(score)]
            confidence = 100 * np.max(score)
            
            # Menampilkan hasil
            st.write("---")
            st.subheader("Hasil Prediksi:")
            
            if predicted_class == 'Positive':
                st.error(f"Kategori: **{predicted_class}**")
            else:
                st.success(f"Kategori: **{predicted_class}**")
                
            st.info(f"Tingkat Keyakinan (Confidence): **{confidence:.2f}%**")
