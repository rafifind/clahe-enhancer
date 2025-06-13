import cv2
import numpy as np

def apply_clahe(image_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Menerapkan CLAHE pada citra input BGR.
    Citra dikonversi ke grayscale, CLAHE diterapkan, lalu dikembalikan sebagai citra grayscale.

    Args:
        image_bgr (numpy.ndarray): Citra input dalam format BGR (Blue, Green, Red).
                                Ini adalah format default yang digunakan oleh OpenCV.
        clip_limit (float): Ambang batas untuk pembatasan kontras (contrast limiting).
                            Nilai yang lebih tinggi menghasilkan kontras yang lebih kuat.
        tile_grid_size (tuple): Ukuran grid (baris, kolom) untuk CLAHE.
                                Menentukan seberapa lokal adaptasi histogram akan dilakukan.

    Returns:
        numpy.ndarray: Citra grayscale yang telah ditingkatkan dengan CLAHE.
                       Jika terjadi error selama proses CLAHE, akan mencoba mengembalikan
                       citra grayscale asli atau channel pertama dari citra BGR input.
    """
    try:
        # Langkah 1: Konversi citra BGR ke grayscale
        # CLAHE biasanya diterapkan pada citra single-channel (grayscale).
        gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Langkah 2: Buat objek CLAHE dengan parameter yang diberikan
        # clipLimit: Batas kontras. Mencegah amplifikasi noise yang berlebihan.
        # tileGridSize: Ukuran grid untuk pemrosesan adaptif lokal.
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        # Langkah 3: Terapkan CLAHE pada citra grayscale
        enhanced_gray_image = clahe_obj.apply(gray_image)

        return enhanced_gray_image
    except Exception as e:
        # Penanganan error dasar jika terjadi masalah selama proses CLAHE
        print(f"Error in apply_clahe: {e}")
        # Fallback: Jika konversi ke grayscale berhasil tetapi CLAHE gagal, kembalikan grayscale asli.
        if 'gray_image' in locals():
            return gray_image
        else:
            # Fallback lebih lanjut: Jika bahkan konversi ke grayscale gagal,
            # coba kembalikan channel pertama dari citra BGR (misalnya, channel Biru).
            # Ini adalah fallback kasar dan mungkin perlu penanganan yang lebih baik
            # tergantung pada kebutuhan aplikasi (misalnya, log error yang lebih detail atau raise exception).
            # Memastikan citra memiliki 3 dimensi (untuk citra berwarna) sebelum mencoba slicing.
            return image_bgr[:,:,0] if len(image_bgr.shape) == 3 and image_bgr.shape[2] == 3 else image_bgr


# def to_grayscale(image_bgr):
#     """Konversi citra BGR ke Grayscale."""
#     return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# def gaussian_blur(image_bgr, ksize=(5,5)):
#     """Terapkan Gaussian Blur pada citra BGR."""
#     return cv2.GaussianBlur(image_bgr, ksize, 0)