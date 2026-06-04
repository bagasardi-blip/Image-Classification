import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import zipfile
import io
import gc

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Klasifikasi Gambar ZIP", page_icon="📷")

st.title("📷 Klasifikasi Gambar Massal (ZIP)")
st.write("Aplikasi ini dirancang untuk memproses banyak gambar dengan penggunaan memori rendah.")

CLASS_NAMES = ['Negative', 'Positive']
MODEL_PATH = 'image_classification_model.h5'

# 2. Fungsi Load Model dengan Decorator Cache
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            # Memuat model tanpa optimizer untuk menghemat RAM
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            return model
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None
    return None

model = load_model()

if model is None:
    st.warning("File model `.h5` tidak ditemukan. Pastikan sudah diunggah ke GitHub.")
else:
    st.success("Model siap!")

# 3. Antarmuka Unggah
uploaded_zip = st.file_uploader("Unggah file ZIP gambar", type=["zip"])

if uploaded_zip is not None and model is not None:
    try:
        with zipfile.ZipFile(uploaded_zip) as z:
            # Ambil daftar file
            all_info = z.infolist()
            valid_extensions = ('.jpg', '.jpeg', '.png')
            
            # Saring file gambar (abaikan folder dan file sistem)
            image_files = [
                info.filename for info in all_info 
                if info.filename.lower().endswith(valid_extensions) 
                and not info.filename.startswith('__MACOSX/') 
                and not os.path.basename(info.filename).startswith('.')
            ]
            
            total = len(image_files)
            
            if total == 0:
                st.warning("Tidak ada gambar valid di dalam ZIP.")
            else:
                st.info(f"Memproses {total} gambar...")
                
                # Wadah hasil
                progress_bar = st.progress(0)
                placeholder = st.empty()
                results = []

                # 4. Proses Satu Per Satu dengan Manajemen Memori Ketat
                for i, file_path in enumerate(image_files):
                    with placeholder.container():
                        st.text(f"Sedang memproses ({i+1}/{total}): {os.path.basename(file_path)}")
                    
                    try:
                        # Baca gambar
                        img_data = z.read(file_path)
                        with Image.open(io.BytesIO(img_data)) as img:
                            img = img.convert('RGB').resize((150, 150))
                            img_array = np.array(img) / 255.0 # Normalisasi manual jika diperlukan
                        
                        img_batch = np.expand_dims(img_array, axis=0)
                        
                        # Prediksi
                        preds = model.predict(img_batch, verbose=0)
                        score = tf.nn.softmax(preds[0])
                        
                        label = CLASS_NAMES[np.argmax(score)]
                        conf = f"{100 * np.max(score):.2f}%"
                        
                        results.append({"File": os.path.basename(file_path), "Hasil": label, "Skor": conf})
                        
                    except Exception as e:
                        results.append({"File": os.path.basename(file_path), "Hasil": "Error", "Skor": "0%"})
                    
                    # UPDATE PROGRESS
                    progress_bar.progress((i + 1) / total)
                    
                    # PAKSA BERSIHKAN MEMORI (CRITICAL)
                    if (i + 1) % 5 == 0:
                        tf.keras.backend.clear_session()
                        gc.collect() # Garbagge Collector Python

                # 5. Tampilkan Hasil Akhir
                placeholder.empty()
                st.write("### ✅ Selesai! Hasil Analisis:")
                st.dataframe(results, use_container_width=True)
                
                # Ringkasan
                pos = sum(1 for r in results if r["Hasil"] == "Positive")
                neg = sum(1 for r in results if r["Hasil"] == "Negative")
                st.write(f"**Ringkasan:** Positive: {pos} | Negative: {neg}")

    except Exception as e:
        st.error(f"Terjadi kesalahan ZIP: {e}")
