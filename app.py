import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import zipfile
import io
import gc

# 1. Konfigurasi Halaman Web Streamlit
st.set_page_config(
    page_title="Aplikasi Klasifikasi Gambar (ZIP)",
    page_icon="📷",
    layout="centered"
)

# Judul Utama di Layar
st.title("📷 Aplikasi Klasifikasi Gambar via ZIP")
st.write("Unggah berkas **.zip** berisi kumpulan foto/gambar untuk mendeteksi kategori secara otomatis.")

# Pengaturan Kelas dan Nama File Model h5 Anda
CLASS_NAMES = ['Negative', 'Positive']
MODEL_PATH = 'image_classification_model.h5'

# 2. Fungsi untuk Memuat Model Keras (.h5) secara Aman
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            # Menggunakan compile=False agar aman dari perbedaan versi library di server
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            st.error(f"Gagal memuat file model: {e}")
            return None
    return None

# Memuat model secara otomatis saat web dibuka
model = load_model()

if model is None:
    st.error(f"❌ File model `{MODEL_PATH}` tidak ditemukan di GitHub Anda. Pastikan file model .h5 sudah di-upload ke folder yang sama dengan app.py!")
else:
    st.success("✅ Model berhasil dimuat dan siap digunakan!")

# 3. Tombol Unggah File ZIP Gambar
uploaded_zip = st.file_uploader("Pilih dan unggah file ZIP berisi foto gambar...", type=["zip"])

# Jalankan proses klasifikasi otomatis jika file ZIP sudah selesai di-upload
if uploaded_zip is not None:
    if model is None:
        st.error("Proses tidak dapat dijalankan karena model .h5 belum berhasil dimuat.")
    else:
        st.write("---")
        st.info("📦 Berkas ZIP terdeteksi! Membuka dan membaca isi file...")
        
        try:
            with zipfile.ZipFile(uploaded_zip) as z:
                # Mengambil semua daftar file di dalam ZIP
                all_files = z.namelist()
                
                # Saring hanya file yang berupa gambar (.jpg, .jpeg, .png)
                # Serta otomatis mengabaikan file sampah sistem bawaan Mac/Windows (__MACOSX atau .DS_Store)
                valid_extensions = ('.jpg', '.jpeg', '.png')
                image_files = [
                    f for f in all_files 
                    if f.lower().endswith(valid_extensions) and not f.startswith('__MACOSX/') and not os.path.basename(f).startswith('.')
                ]
                
                total_images = len(image_files)
                
                if total_images == 0:
                    st.warning("⚠️ AWAS: Tidak ditemukan file gambar (.jpg, .jpeg, .png) yang valid di dalam file ZIP Anda!")
                    st.write("Isi file ZIP yang Anda unggah justru mendeteksi file-file ini:")
                    st.code(all_files[:10])
                else:
                    st.success(f"🚀 Menemukan {total_images} gambar. Memulai proses prediksi otomatis...")
                    
                    # Membuat visualisasi loading progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = []

                    # Perulangan (looping) untuk memproses gambar satu per satu
                    for idx, file_name in enumerate(image_files):
                        status_text.text(f"Menganalisis ({idx + 1}/{total_images}): {os.path.basename(file_name)}")
                        
                        try:
                            # Membaca bit data gambar langsung dari memori ZIP
                            img_data = z.read(file_name)
                            
                            with Image.open(io.BytesIO(img_data)) as img:
                                # SOLUSI UTAMA: Paksa konversi ke 'RGB' untuk membuang channel ke-4 (Alpha/Transparansi)
                                # Langkah ini mencegah error shape=(None, 150, 150, 4)
                                img_rgb = img.convert('RGB')
                                
                                # Ubah resolusi gambar ke 150x150 sesuai dengan arsitektur model CNN Anda
                                img_resized = img_rgb.resize((150, 150))
                                
                                # Mengubah gambar menjadi bentuk array matriks angka
                                img_array = np.array(img_resized)
                            
                            # Menambahkan dimensi batch agar bentuknya pas menjadi (1, 150, 150, 3)
                            img_batch = np.expand_dims(img_array, axis=0)
                            
                            # Melakukan prediksi klasifikasi menggunakan model h5 Anda
                            predictions = model.predict(img_batch, verbose=0)
                            
                            # Menggunakan fungsi Softmax untuk mengambil keputusan kelas tertinggi
                            score = tf.nn.softmax(predictions[0])
                            predicted_class = CLASS_NAMES[np.argmax(score)]
                            confidence = 100 * np.max(score)
                            
                            # Menyimpan hasil data ke dalam list rekapitulasi
                            results.append({
                                "Nama File": os.path.basename(file_name),
                                "Prediksi": predicted_class,
                                "Tingkat Keyakinan": f"{confidence:.2f}%"
                            })
                            
                        except Exception as img_err:
                            # Jika ada salah satu gambar yang rusak, program tidak akan mati total
                            results.append({
                                "Nama File": os.path.basename(file_name),
                                "Prediksi": "Gagal (Format Rusak)",
                                "Tingkat Keyakinan": "0%"
                            })
                        
                        # Memperbarui tampilan progress bar di halaman web
                        progress_bar.progress((idx + 1) / total_images)
                        
                        # MANAJEMEN MEMORI RAM: Kosongkan sisa sampah memori setiap kelipatan 10 gambar
                        if (idx + 1) % 10 == 0:
                            tf.keras.backend.clear_session()
                            gc.collect()

                    # Menghapus teks status berjalan jika seluruh proses sudah selesai
                    status_text.empty()
                    
                    # 4. Menampilkan Hasil Akhir di Layar Web dalam Bentuk Tabel
                    st.write("### 📊 Hasil Klasifikasi Keseluruhan:")
                    st.dataframe(results, use_container_width=True)
                    
                    # Menghitung total statistik prediksi kelas
                    total_pos = sum(1 for r in results if r["Prediksi"] == "Positive")
                    total_neg = sum(1 for r in results if r["Prediksi"] == "Negative")
                    
                    # Menampilkan metrik total angka di bawah tabel
                    col1, col2 = st.columns(2)
                    col1.metric("Total Kategori Positive (Retak)", total_pos)
                    col2.metric("Total Kategori Negative (Aman)", total_neg)
                    
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan sistem saat membuka berkas ZIP: {e}")
