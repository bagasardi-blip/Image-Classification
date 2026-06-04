import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Aplikasi Klasifikasi Gambar Massal",
    page_icon="📷",
    layout="centered"
)

# Judul Aplikasi
st.title("📷 Aplikasi Klasifikasi Gambar Massal")
st.write("Pilih dan unggah **banyak gambar sekaligus** (tanpa perlu di-ZIP) untuk dideteksi secara otomatis.")

# Tentukan nama kelas sesuai dengan model Anda
CLASS_NAMES = ['Negative', 'Positive']
MODEL_PATH = 'image_classification_model.h5'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            # Menggunakan compile=False agar menghemat memori RAM server
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None
    else:
        return None

# Memuat model otomatis
model = load_model()

if model is None:
    st.warning(f"Berkas model `{MODEL_PATH}` tidak ditemukan. Pastikan file model tersebut berada di folder yang sama dengan `app.py`.")
else:
    st.success("Model berhasil dimuat dan siap digunakan!")

# Widget unggah banyak file sekaligus (accept_multiple_files=True)
uploaded_files = st.file_uploader(
    "Pilih atau seret semua gambar gambar di sini...", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files and model is not None:
    st.write("---")
    total_images = len(uploaded_files)
    st.info(f"Ditemukan {total_images} gambar siap diproses otomatis.")
    
    # Komponen visual progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    
    # Proses gambar satu per satu secara langsung
    for idx, file_slot in enumerate(uploaded_files):
        status_text.text(f"Memproses gambar ({idx + 1}/{total_images}): {file_slot.name}")
        try:
            # Membaca file langsung dari slot input upload
            image = Image.open(file_slot)
            
            # Resize & paksa ke format RGB jika ada channel alpha (RGBA)
            img_resized = image.resize((150, 150))
            if img_resized.mode != 'RGB':
                img_resized = img_resized.convert('RGB')
                
            img_array = np.array(img_resized)
            img_batch = np.expand_dims(img_array, axis=0)
            
            # Prediksi menggunakan model
            predictions = model.predict(img_batch, verbose=0)
            score = tf.nn.softmax(predictions[0])
            predicted_class = CLASS_NAMES[np.argmax(score)]
            confidence = 100 * np.max(score)
            
            results.append({
                "Nama File": file_slot.name,
                "Prediksi": predicted_class,
                "Tingkat Keyakinan": f"{confidence:.2f}%"
            })
            
        except Exception as img_err:
            results.append({
                "Nama File": file_slot.name,
                "Prediksi": "Gagal diproses",
                "Tingkat Keyakinan": "0%"
            })
        
        # Pengosongan memori berkala setiap kali selesai memproses gambar
        tf.keras.backend.clear_session()
        
        # Update progress bar
        progress_bar.progress((idx + 1) / total_images)
    
    # Bersihkan teks status running setelah selesai
    status_text.empty()
    
    # Tampilkan tabel hasil akhir
    st.write("### 📊 Hasil Klasifikasi Keseluruhan:")
    st.dataframe(results, use_container_width=True)
    
    # Statistik hasil
    total_pos = sum(1 for r in results if r["Prediksi"] == "Positive")
    total_neg = sum(1 for r in results if r["Prediksi"] == "Negative")
    
    col1, col2 = st.columns(2)
    col1.metric("Total Positive", total_pos)
    col2.metric("Total Negative", total_neg)
