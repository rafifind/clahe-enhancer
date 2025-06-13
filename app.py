import streamlit as st
from PIL import Image
import numpy as np
import cv2
from image_utils import apply_clahe # Mengimpor fungsi dari image_utils.py

# Konfigurasi halaman
st.set_page_config(page_title="Aplikasi Peningkatan Citra CLAHE", layout="wide")

# Judul Aplikasi
st.title("🖼️ Aplikasi Peningkatan Citra menggunakan CLAHE")
st.markdown("""
Aplikasi ini untuk meng-upload gambar dan menerapkan 
*Contrast Limited Adaptive Histogram Equalization* (CLAHE) untuk meningkatkan kontras lokalnya. 
Anda dapat menyesuaikan parameter CLAHE melalui menu di sebelah kiri.
""")

# Sidebar untuk input parameter
st.sidebar.title("🔧 Parameter CLAHE")
clip_limit = st.sidebar.slider(
    "Clip Limit", 
    min_value=1.0, 
    max_value=40.0,  # Batas atas yang lebih realistis untuk clipLimit
    value=2.0,       # Nilai default yang umum
    step=0.1,
    help="Mengontrol batas amplifikasi kontras. Nilai yang lebih tinggi berarti kontras yang lebih kuat, namun bisa meningkatkan noise."
)
tile_rows = st.sidebar.slider(
    "Tile Grid Rows", 
    min_value=2, 
    max_value=32,    # Batas atas yang lebih realistis
    value=8, 
    step=1,
    help="Jumlah baris dalam grid tile. Tile yang lebih kecil dapat menangkap variasi lokal dengan lebih baik."
)
tile_cols = st.sidebar.slider(
    "Tile Grid Columns", 
    min_value=2, 
    max_value=32,    # Batas atas yang lebih realistis
    value=8, 
    step=1,
    help="Jumlah kolom dalam grid tile."
)

# File uploader
uploaded_file = st.file_uploader("Upload gambar (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membaca gambar yang diunggah
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np_rgb = np.array(image_pil)
    
    # Konversi RGB (dari PIL) ke BGR (untuk OpenCV)
    image_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)

    # Membuat dua kolom untuk menampilkan gambar asli dan hasil
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gambar Asli")
        st.image(image_pil, use_container_width=True, caption="Gambar yang diunggah")

    # Terapkan CLAHE
    # Fungsi apply_clahe akan mengembalikan citra grayscale
    enhanced_image_gray = apply_clahe(image_bgr, clip_limit=clip_limit, tile_grid_size=(tile_rows, tile_cols))

    with col2:
        st.subheader("Hasil Peningkatan CLAHE")
        # Menampilkan citra grayscale hasil CLAHE
        st.image(enhanced_image_gray, use_container_width=True, caption="Gambar setelah CLAHE", channels="GRAY", clamp=True)
        
    # Opsi untuk mengunduh gambar hasil
    # Konversi numpy array ke format yang bisa diunduh (misalnya, PNG dalam bytes)
    try:
        enhanced_pil = Image.fromarray(enhanced_image_gray)
        from io import BytesIO
        buf = BytesIO()
        enhanced_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(
            label="Download Gambar Hasil",
            data=byte_im,
            file_name="enhanced_clahe_image.png",
            mime="image/png"
        )
    except Exception as e:
        st.error(f"Gagal membuat tombol unduh: {e}")


else:
    st.info("ℹ️ Silakan unggah gambar untuk memulai.")

# Penjelasan tentang CLAHE
st.markdown("---")
st.subheader("Apa itu CLAHE?")
st.markdown("""
**Contrast Limited Adaptive Histogram Equalization (CLAHE)** adalah sebuah teknik dalam pengolahan citra yang digunakan untuk meningkatkan kontras pada gambar. Berbeda dengan *Histogram Equalization* biasa yang bekerja secara global pada seluruh gambar, CLAHE bekerja pada bagian-bagian kecil dari gambar yang disebut *tiles*.

**Cara Kerja Singkat:**
1.  Gambar dibagi menjadi beberapa *tile* (blok kecil).
2.  Untuk setiap *tile*, histogram intensitas piksel dihitung.
3.  Histogram tersebut kemudian di-"clip" (dipotong) berdasarkan parameter `Clip Limit` untuk mencegah amplifikasi noise yang berlebihan.
4.  *Histogram Equalization* diterapkan pada histogram yang sudah di-clip tersebut.
5.  Hasil dari setiap *tile* kemudian digabungkan kembali menggunakan interpolasi bilinear untuk menghindari batas antar *tile* yang terlihat jelas.

**Parameter Utama:**
-   **Clip Limit:** Mengontrol seberapa besar kontras dapat ditingkatkan. Nilai yang lebih rendah menghasilkan peningkatan kontras yang lebih halus, sementara nilai yang lebih tinggi menghasilkan kontras yang lebih tajam tetapi berpotensi memperkuat noise.
-   **Tile Grid Size (Rows/Cols):** Menentukan ukuran dan jumlah *tile* tempat histogram lokal dihitung. Ukuran *tile* yang lebih kecil memungkinkan adaptasi yang lebih baik terhadap variasi kontras lokal, tetapi juga bisa lebih sensitif terhadap noise.
""")
st.markdown("Sumber implementasi CLAHE: OpenCV")