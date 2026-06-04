import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import zipfile
import io

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Aplikasi Klasifikasi Gambar (ZIP)",
    page_icon="📷",
    layout="centered"
)

# Judul Aplikasi
st.title("📷 Aplikasi Klasifikasi Gambar via ZIP")
st.write("Unggah file **.zip** yang berisi kumpulan gambar untuk mendeteksi kategori secara massal.")

# Tentukan nama kelas sesuai dengan model Anda
CLASS_NAMES = ['Negative', 'Positive']
MODEL_PATH = 'image_classification_model.h5'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            # Menggunakan compile=False agar menghemat memori saat loading awal
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None
    else:
        return None

# Memuat model otomatis
model = load_model()

if model is None:
    st.warning(f"Berkas model `{MODEL_PATH}` tidak ditemukan. Pastikan file model tersebut sudah Anda unggah ke folder yang sama dengan `app.py` di GitHub.")
else:
    st.success("Model berhasil dimuat dan siap digunakan!")

# Widget unggah berkas ZIP
uploaded_zip = st.file_uploader("Pilih file ZIP Anda...", type=["zip"])

if uploaded_zip is not None and model is not None:
    st.write("---")
    st.subheader("📦 Memproses Berkas ZIP...")
    
    try:
        with zipfile.ZipFile(uploaded_zip) as z:
            all_files = z.namelist()
            
            # Saring file gambar valid dan abaikan file sampah tersembunyi
            valid_extensions = ('.jpg', '.jpeg', '.png')
            image_files = [
                f for f in all_files 
                if f.lower().endswith(valid_extensions) and not f.startswith('__MACOSX/') and not os.path.basename(f).startswith('.')
            ]
            
            total_images = len(image_files)
            
            if total_images == 0:
                st.warning("Tidak ditemukan file gambar (.jpg, .jpeg, .png) yang valid di dalam file ZIP tersebut.")
            else:
                st.info(f"Ditemukan {total_images} gambar. Memulai proses prediksi otomatis...")
                
                # Gunakan fungsi kosong penampung agar Streamlit tidak melakukan render ulang komponen yang tidak perlu
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                
                # Proses gambar satu per satu
                for idx, file_name in enumerate(image_files):
                    status_text.text(f"Memproses gambar ({idx + 1}/{total_images}): {os.path.basename(file_name)}")
                    try:
                        # Membaca gambar dari memori ZIP
                        img_data = z.read(file_name)
                        image = Image.open(io.BytesIO(img_data))
                        
                        # Resize & paksa ke format RGB
                        img_resized = image.resize((150, 150))
                        if img_resized.mode != 'RGB':
                            img_resized = img_resized.convert('RGB')
                            
                        img_array = np.array(img_resized)
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
                        
                    except Exception as img_err:
                        results.append({
                            "Nama File": os.path.basename(file_name),
                            "Prediksi": "Gagal diproses",
                            "Tingkat Keyakinan": "0%"
                        })
                    
                    # Manajemen Pengosongan Memori Kritis (Pencegah Crash)
                    if idx % 10 == 0:  # Setiap 10 gambar, bersihkan sisa memori TensorFlow backend
                        tf.keras.backend.clear_session()
                    
                    # Update status progres
                    progress_bar.progress((idx + 1) / total_images)
                
                # Bersihkan teks status berjalan jika sudah selesai seluruhnya
                status_text.empty()
                
                # Tampilkan hasil rekapitulasi data akhir
                st.write("### 📊 Hasil Klasifikasi Keseluruhan:")
                st.dataframe(results, use_container_width=True)
                
                total_pos = sum(1 for r in results if r["Prediksi"] == "Positive")
                total_neg = sum(1 for r in results if r["Prediksi"] == "Negative")
                total_fail = sum(1 for r in results if r["Prediksi"] == "Gagal diproses")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Positive", total_pos)
                col2.metric("Total Negative", total_neg)
                if total_fail > 0:
                    col3.metric("Gagal", total_fail)
                
    except Exception as e:
        st.error(f"Terjadi kesalahan sistem saat membuka berkas ZIP: {e}")
