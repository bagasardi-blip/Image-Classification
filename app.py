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
st.write("Unggah berkas **.zip** berisi kumpulan foto untuk mendeteksi kategori secara otomatis.")

CLASS_NAMES = ['Negative', 'Positive']
MODEL_PATH = 'image_classification_model.h5'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            st.error(f"Gagal memuat file model: {e}")
            return None
    return None

model = load_model()

if model is None:
    st.error(f"❌ File model `{MODEL_PATH}` tidak ditemukan di GitHub Anda.")
else:
    st.success("✅ Model berhasil dimuat dan siap digunakan!")

uploaded_zip = st.file_uploader("Pilih dan unggah file ZIP berisi foto gambar...", type=["zip"])

if uploaded_zip is not None and model is not None:
    st.write("---")
    st.info("📦 Berkas ZIP terdeteksi! Memproses...")
    
    try:
        with zipfile.ZipFile(uploaded_zip) as z:
            all_files = z.namelist()
            valid_extensions = ('.jpg', '.jpeg', '.png')
            
            # Filter ketat: Hanya ambil file gambar asli dan abaikan semua file sistem tersembunyi
            image_files = [
                f for f in all_files 
                if f.lower().endswith(valid_extensions) 
                and not f.startswith('__MACOSX/') 
                and not os.path.basename(f).startswith('.')
                and not os.path.basename(f).startswith('_')
            ]
            
            total_images = len(image_files)
            
            if total_images == 0:
                st.warning("⚠️ Tidak ditemukan file gambar yang valid di dalam file ZIP Anda.")
            else:
                st.success(f"🚀 Menemukan {total_images} gambar. Memulai prediksi otomatis...")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []

                for idx, file_name in enumerate(image_files):
                    status_text.text(f"Menganalisis ({idx + 1}/{total_images}): {os.path.basename(file_name)}")
                    
                    try:
                        img_data = z.read(file_name)
                        
                        # Membuka gambar menggunakan PIL
                        with Image.open(io.BytesIO(img_data)) as img:
                            # OBAT NYATA: Paksa konversi ke 'RGB' (Menghapus Channel ke-4/Alpha penyebab error)
                            img_rgb = img.convert('RGB')
                            # Samakan ukuran pixel (150x150)
                            img_resized = img_rgb.resize((150, 150))
                            img_array = np.array(img_resized)
                        
                        # Pastikan bentuk array benar-benar (150, 150, 3) sebelum masuk ke model
                        if img_array.shape == (150, 150, 3):
                            img_batch = np.expand_dims(img_array, axis=0)
                            
                            # Prediksi
                            predictions = model.predict(img_batch, verbose=0)
                            score = tf.nn.softmax(predictions[0])
                            
                            predicted_class = CLASS_NAMES[np.argmax(score)]
                            confidence = 100 * np.max(score)
                            
                            results.append({
                                "Nama File": os.path.basename(file_name),
                                "Prediksi": predicted_class,
                                "Tingkat Keyakinan": f"{confidence:.2f}%"
                            })
                        else:
                            # Jika ukuran dimensi tetap tidak cocok, skip gambar ini jangan bikin crash
                            continue
                            
                    except Exception as img_err:
                        # Jika ada file error/corrupt, skip ke gambar selanjutnya
                        continue
                    
                    # Update progress bar
                    progress_bar.progress((idx + 1) / total_images)
                    
                    # Pengosongan RAM berkala
                    if (idx + 1) % 5 == 0:
                        tf.keras.backend.clear_session()
                        gc.collect()

                status_text.empty()
                
                # Menampilkan Hasil Tabel Akhir
                if results:
                    st.write("### 📊 Hasil Klasifikasi Keseluruhan:")
                    st.dataframe(results, use_container_width=True)
                    
                    total_pos = sum(1 for r in results if r["Prediksi"] == "Positive")
                    total_neg = sum(1 for r in results if r["Prediksi"] == "Negative")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Total Kategori Positive", total_pos)
                    col2.metric("Total Kategori Negative", total_neg)
                else:
                    st.error("Gagal memproses gambar. Pastikan format foto Anda adalah JPG/PNG standar.")
                    
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan berkas ZIP: {e}")
