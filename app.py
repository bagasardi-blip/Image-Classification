import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import zipfile
import io
import gc

# Konfigurasi Halaman Web
st.set_page_config(page_title="Klasifikasi Gambar ZIP", page_icon="📷", layout="centered")

st.title("📷 Aplikasi Klasifikasi Gambar via ZIP")
st.write("Unggah berkas **.zip** berisi kumpulan foto beton untuk mendeteksi kategori secara otomatis.")

CLASS_NAMES = ['Negative', 'Positive']
MODEL_PATH = 'image_classification_model.h5'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            # Menggunakan compile=False agar hemat memori RAM server gratisan
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            st.error(f"Gagal memuat file model h5: {e}")
            return None
    return None

model = load_model()

if model is None:
    st.warning(f"⚠️ Berkas model `{MODEL_PATH}` tidak ditemukan di GitHub. Pastikan file model .h5 sudah di-upload ke folder yang sama dengan app.py!")
else:
    st.success("✅ Model berhasil dimuat!")

# Widget upload dikunci khusus untuk menerima file ZIP saja
uploaded_zip = st.file_uploader("Pilih dan unggah file ZIP berisi kumpulan foto...", type=["zip"])

if uploaded_zip is not None:
    if model is None:
        st.error("Proses dihentikan karena file model .h5 belum terdeteksi di GitHub Anda.")
    else:
        st.write("---")
        st.info("📦 Berkas ZIP terdeteksi! Mengekstrak isi file...")
        
        try:
            with zipfile.ZipFile(uploaded_zip) as z:
                all_files = z.namelist()
                valid_extensions = ('.jpg', '.jpeg', '.png')
                
                # Saring hanya file gambar asli dan buang file sampah tersembunyi
                image_files = [
                    f for f in all_files 
                    if f.lower().endswith(valid_extensions) 
                    and not f.startswith('__MACOSX/') 
                    and not os.path.basename(f).startswith('.')
                ]
                
                total_images = len(image_files)
                
                if total_images == 0:
                    st.warning("⚠️ Tidak ditemukan file gambar (.jpg/.png) yang valid di dalam ZIP Anda.")
                else:
                    st.success(f"🚀 Menemukan {total_images} gambar. Memulai klasifikasi otomatis...")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = []

                    # Proses gambar satu per satu
                    for idx, file_name in enumerate(image_files):
                        status_text.text(f"Menganalisis ({idx + 1}/{total_images}): {os.path.basename(file_name)}")
                        
                        try:
                            img_data = z.read(file_name)
                            with Image.open(io.BytesIO(img_data)) as img:
                                # Paksa konversi ke RGB (mengatasi masalah gambar 4-channel / RGBA)
                                img_rgb = img.convert('RGB')
                                img_resized = img_rgb.resize((150, 150))
                                img_array = np.array(img_resized)
                            
                            if img_array.shape == (150, 150, 3):
                                img_batch = np.expand_dims(img_array, axis=0)
                                
                                # Jalankan prediksi model
                                predictions = model.predict(img_batch, verbose=0)
                                score = tf.nn.softmax(predictions[0])
                                
                                predicted_class = CLASS_NAMES[np.argmax(score)]
                                confidence = 100 * np.max(score)
                                
                                results.append({
                                    "Nama File": os.path.basename(file_name),
                                    "Prediksi": predicted_class,
                                    "Tingkat Keyakinan": f"{confidence:.2f}%"
                                })
                        except Exception:
                            continue
                        
                        progress_bar.progress((idx + 1) / total_images)
                        
                        # Bersihkan RAM setiap 5 gambar agar server Streamlit tidak tumbang
                        if (idx + 1) % 5 == 0:
                            tf.keras.backend.clear_session()
                            gc.collect()

                    status_text.empty()
                    
                    if results:
                        st.write("### 📊 Hasil Klasifikasi Keseluruhan:")
                        st.dataframe(results, use_container_width=True)
                        
                        total_pos = sum(1 for r in results if r["Prediksi"] == "Positive")
                        total_neg = sum(1 for r in results if r["Prediksi"] == "Negative")
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Total Kategori Positive", total_pos)
                        col2.metric("Total Kategori Negative", total_neg)
                        
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat membaca file ZIP: {e}")
