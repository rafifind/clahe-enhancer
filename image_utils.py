import cv2
import numpy as np

def apply_clahe(image_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Menerapkan CLAHE pada citra input BGR.
    Citra dikonversi ke grayscale, CLAHE diterapkan, lalu dikembalikan.

    Args:
        image_bgr (numpy.ndarray): Citra input dalam format BGR.
        clip_limit (float): Ambang batas untuk pembatasan kontras.
        tile_grid_size (tuple): Ukuran grid (baris, kolom) untuk CLAHE.

    Returns:
        numpy.ndarray: Citra grayscale yang telah ditingkatkan dengan CLAHE.
    """
    try:
        # Konversi ke grayscale
        gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Buat objek CLAHE
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        # Terapkan CLAHE
        enhanced_gray_image = clahe_obj.apply(gray_image)

        return enhanced_gray_image
    except Exception as e:
        print(f"Error in apply_clahe: {e}")
        # Kembalikan citra grayscale asli jika terjadi error
        if 'gray_image' in locals():
            return gray_image
        else:
            # Jika konversi ke gray juga gagal, coba kembalikan channel pertama dari BGR
            # Ini adalah fallback kasar dan mungkin perlu penanganan yang lebih baik
            return image_bgr[:,:,0] if len(image_bgr.shape) == 3 else image_bgr

# Anda bisa menambahkan fungsi pengolahan citra lainnya di sini jika diperlukan
# Contoh:
# def to_grayscale(image_bgr):
#     return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# def gaussian_blur(image_bgr, ksize=(5,5)):
#     return cv2.GaussianBlur(image_bgr, ksize, 0)