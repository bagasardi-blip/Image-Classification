import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import zipfile
import io
import gc
import urllib.request

# Konfigurasi Halaman Web
st.set_page_config(page_title="Klasifikasi Gambar ZIP", page_icon="📷", layout="centered")

st.title("📷 Aplikasi Klasifikasi Gambar via ZIP")
st.write("Unggah berkas **.zip** berisi kumpulan foto beton untuk mendeteksi kategori secara otomatis.")

CLASS_NAMES = ['Negative', 'Positive']
MODEL_PATH = 'image_classification_model.h5'

# =========================================================================
# PASTE LINK GOOGLE DRIVE KAMU DI BAWAH INI (Sudah otomatis aman)
# =========================================================================
GDrive_Link = "https://drive.google.com/file/d/1zYttdVoEhptiajyCjxuAcClAXxYxllJT/view?usp=sharing" 

def get_direct_download_link(url):
    if "drive.google.com" in url:
        if "/file/d/" in url:
            file_id = url.split("/file/d/")[1].split("/")[0]
            return f"https://docs.google.com/uc?export=download&id={file_id}"
    return url

@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        if "drive.google.com" not in GDrive_Link and "export=download" not in GDrive_Link:
            st.warning("⚠️ Kamu belum memasukkan link Google Drive dengan benar di dalam kode app.py!")
            return None
        
        with st.spinner("⏳ Mengunduh file model dari Google Drive..."):
            try:
                direct_link = get_direct_download_link(GDrive_Link)
                urllib.request.urlretrieve(direct_link, MODEL_PATH)
                st.success("✅ Model berhasil diunduh!")
            except Exception as e:
                st.error(f"❌ Gagal mengunduh model: {e}")
                return None
                
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Gagal memuat file model h5: {e}")
        return None

model = load_model_from_drive()

if model is not None:
    st.success("✅ Model AI Siap Digunakan!")

uploaded_zip = st.file_uploader("Pilih dan unggah file ZIP berisi kumpulan foto...", type=["zip"])

if uploaded_zip is not None and model is not None:
    st.write("---")
    st.info("📦 Berkas ZIP terdeteksi! Mengekstrak isi file...")
    
    try:
        with zipfile.ZipFile(uploaded_zip) as z:
            all_files = z.namelist()
            valid_extensions = ('.jpg', '.jpeg', '.png')
            
            image_files = [
                f for f in all_files 
                if f.lower().endswith(valid_extensions) 
                and not f.startswith('__MACOSX/') 
                and not os.path.basename(f).startswith('.')
            ]
            
            total_images = len(image_files)
            
            if total_images == 0:
                st.warning("⚠️ Tidak ditemukan file gambar yang valid di dalam ZIP Anda.")
            else:
                st.success(f"🚀 Menemukan {total_images} gambar. Memulai klasifikasi otomatis...")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []

                for idx, file_name in enumerate(image_files):
                    status_text.text(f"Menganalisis ({idx + 1}/{total_images}): {os.path.basename(file_name)}")
                    
                    try:
                        img_data = z.read(file_name)
                        with Image.open(io.BytesIO(img_data)) as img:
                            img_rgb = img.convert('RGB')
                            img_resized = img_rgb.resize((150, 150))
                            # NORMALISASI: Diubah ke skala 0-1 (membantu akurasi prediksi model)
                            img_array = np.array(img_resized) / 255.0
                        
                        if img_array.shape == (150, 150, 3):
                            img_batch = np.expand_dims(img_array, axis=0)
                            
                            # Jalankan prediksi model CNN
                            predictions = model.predict(img_batch, verbose=0)
                            
                            # SOLUSI LOGIKA PERBAIKAN: Cek arsitektur output layer model kamu
                            if predictions.shape[-1] == 1:
                                # Jika model kamu jenis Binary (1 Output Neuron / Sigmoid)
                                pred_value = predictions[0][0]
                                if pred_value >= 0.5:
                                    predicted_class = "Positive"
                                    confidence = pred_value * 100
                                else:
                                    predicted_class = "Negative"
                                    confidence = (1 - pred_value) * 100
                            else:
                                # Jika model kamu jenis Multi-class (2 atau lebih Output Neuron)
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
                    col1.metric("Total Kategori Positive (Retak)", total_pos)
                    col2.metric("Total Kategori Negative (Aman)", total_neg)
                        
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat membaca file ZIP: {e}")
