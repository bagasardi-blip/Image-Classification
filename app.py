import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import zipfile
import io
import gc

# Konfigurasi Halaman
st.set_page_config(page_title="Klasifikasi Gambar ZIP", page_icon="📷", layout="centered")

st.title("📷 Klasifikasi Gambar Massal via ZIP")
st.write("Aplikasi akan **otomatis berjalan** sesaat setelah file `.zip` selesai diunggah.")

CLASS_NAMES = ['Negative', 'Positive']
MODEL_PATH = 'image_classification_model.h5'

# Memuat Model
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None
    return None

model = load_model()

if model is None:
    st.error(f"❌ File model `{MODEL_PATH}` tidak ditemukan di repositori GitHub Anda. Pastikan file model sudah di-upload ke folder yang sama dengan `app.py`!")
else:
    st.success("✅ Model berhasil dimuat dan siap digunakan!")

# Widget Unggah ZIP (Otomatis memicu kode di bawahnya saat upload selesai)
uploaded_zip = st.file_uploader("Unggah file ZIP gambar di sini...", type=["zip"])

if uploaded_zip is not None:
    if model is None:
        st.error("Proses dihentikan karena model belum berhasil dimuat.")
    else:
        st.write("---")
        st.info("📦 Berkas ZIP terdeteksi! Membuka isi file...")
        
        try:
            with zipfile.ZipFile(uploaded_zip) as z:
                # Ambil daftar semua berkas di dalam ZIP
                all_files = z.namelist()
                
                # Tampilkan log pencarian untuk kebutuhan pelacakan Anda
                st.write(f"🔍 Total item di dalam ZIP (termasuk folder): {len(all_files)}")
                
                # Saring hanya file gambar valid (.jpg, .jpeg, .png)
                valid_extensions = ('.jpg', '.jpeg', '.png')
                image_files = [
                    f for f in all_files 
                    if f.lower().endswith(valid_extensions) and not f.startswith('__MACOSX/') and not os.path.basename(f).startswith('.')
                ]
                
                total_images = len(image_files)
                
                if total_images == 0:
                    st.warning("⚠️ AWAS: Tidak ditemukan file gambar (.jpg, .jpeg, .png) di dalam file ZIP Anda!")
                    st.write("Daftar file yang Anda unggah justru berisi ini:")
                    st.code(all_files[:10]) # Menampilkan 10 file pertama yang ada di dalam ZIP untuk cek kesalahan
                else:
                    st.success(f"🚀 Menemukan {total_images} gambar valid. Memulai prediksi otomatis saat ini juga...")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = []

                    # Proses iterasi satu per satu gambar
                    for idx, file_name in enumerate(image_files):
                        status_text.text(f"Menganalisis ({idx + 1}/{total_images}): {os.path.basename(file_name)}")
                        
                        try:
                            # Membaca data gambar langsung dari memori ZIP
                            img_data = z.read(file_name)
                            with Image.open(io.BytesIO(img_data)) as img:
                                # Paksa konversi ke RGB dan samakan ukuran input model (150x150)
                                img_resized = img.convert('RGB').resize((150, 150))
                                img_array = np.array(img_resized)
                            
                            # Menyelaraskan dimensi input batch (1, 150, 150, 3)
                            img_batch = np.expand_dims(img_array, axis=0)
                            
                            # Prediksi lewat model h5
                            predictions = model.predict(img_batch, verbose=0)
                            score = tf.nn.softmax(predictions[0])
                            
                            predicted_class = CLASS_NAMES[np.argmax(score)]
                            confidence = 100 * np.max(score)
                            
                            results.append({
                                "Nama File": os.path.basename(file_name),
                                "Prediksi": predicted_class,
                                "Tingkat Keyakinan": f"{confidence:.2f}%"
                            })
                        except Exception as img_err:
                            results.append({
                                "Nama File": os.path.basename(file_name),
                                "Prediksi": "Gagal (Format Rusak)",
                                "Tingkat Keyakinan": "0%"
                            })
                        
                        # Update progress bar secara langsung
                        progress_bar.progress((idx + 1) / total_images)
                        
                        # Pengosongan memori berkala untuk mencegah server crash
                        if (idx + 1) % 10 == 0:
                            tf.keras.backend.clear_session()
                            gc.collect()

                    # Bersihkan teks status berjalan
                    status_text.empty()
                    
                    # Tampilkan tabel output akhir
                    st.write("### 📊 Hasil Klasifikasi Keseluruhan:")
                    st.dataframe(results, use_container_width=True)
                    
                    # Tampilkan statistik jumlah ringkas
                    total_pos = sum(1 for r in results if r["Prediksi"] == "Positive")
                    total_neg = sum(1 for r in results if r["Prediksi"] == "Negative")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Total Kategori Positive", total_pos)
                    col2.metric("Total Kategori Negative", total_neg)
                    
        except Exception as e:
            st.error(f"❌ File ZIP Anda rusak atau tidak bisa dibuka oleh sistem server: {e}")
