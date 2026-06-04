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
            # Gunakan compile=False agar tidak terjadi error versi optimizer saat memuat model
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
    st.success("Model berhasil dimuat!")

# Widget unggah berkas ZIP
uploaded_zip = st.file_uploader("Pilih file ZIP Anda...", type=["zip"])

if uploaded_zip is not None and model is not None:
    st.write("---")
    st.subheader("📦 Memproses Berkas ZIP...")
    
    try:
        with zipfile.ZipFile(uploaded_zip) as z:
            # Ambil semua daftar file di dalam ZIP
            file_list = z.namelist()
            
            # Saring file: Hanya ambil file gambar dan abaikan folder tersembunyi sistem seperti __MACOSX atau .DS_Store
            valid_extensions = ('.jpg', '.jpeg', '.png')
            image_files = [
                f for f in file_list 
                if f.lower().endswith(valid_extensions) and not f.startswith('__MACOSX/') and not os.path.basename(f).startswith('.')
            ]
            
            if not image_files:
                st.warning("Tidak ditemukan file gambar (.jpg, .jpeg, .png) yang valid di dalam file ZIP tersebut.")
            else:
                st.info(f"Ditemukan {len(image_files)} gambar. Memulai proses prediksi...")
                
                # Progress bar untuk visualisasi komponen web
                progress_bar = st.progress(0)
                results = []
                
                for idx, file_name in enumerate(image_files):
                    try:
                        # Membaca data gambar langsung dari memori ZIP
                        img_data = z.read(file_name)
                        image = Image.open(io.BytesIO(img_data))
                        
                        # Ubah ukuran gambar sesuai input arsitektur model Anda (150x150)
                        img_resized = image.resize((150, 150))
                        
                        # Pastikan gambar dikonversi ke RGB (mengatasi masalah gambar bertipe RGBA/Grayscale)
                        if img_resized.mode != 'RGB':
                            img_resized = img_resized.convert('RGB')
                            
                        img_array = np.array(img_resized)
                        
                        # Tambahkan dimensi batch (1, 150, 150, 3)
                        img_batch = np.expand_dims(img_array, axis=0)
                        
                        # Prediksi menggunakan model
                        predictions = model.predict(img_batch, verbose=0)
                        
                        # Hitung probabilitas menggunakan Softmax jika output model berupa logits
                        score = tf.nn.softmax(predictions[0])
                        predicted_class = CLASS_NAMES[np.argmax(score)]
                        confidence = 100 * np.max(score)
                        
                        # Simpan data hasil prediksi
                        results.append({
                            "Nama File": os.path.basename(file_name),
                            "Prediksi": predicted_class,
                            "Tingkat Keyakinan": f"{confidence:.2f}%"
                        })
                    except Exception as img_err:
                        # Jika ada satu gambar rusak, proses tidak akan berhenti total
                        results.append({
                            "Nama File": os.path.basename(file_name),
                            "Prediksi": "Gagal diproses",
                            "Tingkat Keyakinan": "0%"
                        })
                    
                    # Update status progress bar
                    progress_bar.progress((idx + 1) / len(image_files))
                
                # Tampilkan tabel hasil akhir
                st.write("### 📊 Hasil Klasifikasi Keseluruhan:")
                st.dataframe(results, use_container_width=True)
                
                # Menghitung statistik rekapitulasi hasil prediksi
                total_pos = sum(1 for r in results if r["Prediksi"] == "Positive")
                total_neg = sum(1 for r in results if r["Prediksi"] == "Negative")
                total_fail = sum(1 for r in results if r["Prediksi"] == "Gagal diproses")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Positive", total_pos)
                col2.metric("Total Negative", total_neg)
                if total_fail > 0:
                    col3.metric("Gagal", total_fail)
                
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membuka berkas ZIP: {e}")
