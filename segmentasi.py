import cv2
import numpy as np
import os

# ==============================================================================
# 1. FUNGSI BANTU UMUM (HELPER FUNCTIONS)
# ==============================================================================

def ensure_dir(path):
    """Memastikan folder output tersedia."""
    if not os.path.exists(path):
        os.makedirs(path)

def load_as_gray(path):
    """
    Membaca citra dari path dan memastikan format Grayscale (uint8).
    Mengembalikan None jika citra tidak ditemukan.
    """
    if not os.path.exists(path):
        print(f"[ERROR] File tidak ditemukan: {path}")
        return None
    
    img = cv2.imread(path)
    if img is None:
        return None
        
    # Konversi ke grayscale jika citra berwarna
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img

def normalize_to_uint8(img):
    """
    Normalisasi array float32 ke rentang 0-255 (uint8).
    Rumus: (pixel - min) / (max - min) * 255
    """
    img_float = img.astype(np.float32)
    min_val = img_float.min()
    max_val = img_float.max()
    
    if max_val - min_val == 0:
        return np.zeros(img.shape, dtype=np.uint8)
    
    norm = (img_float - min_val) / (max_val - min_val) * 255.0
    return norm.astype(np.uint8)

# ==============================================================================
# 2. IMPLEMENTASI KONVOLUSI MANUAL
# ==============================================================================

def convolve_2d(image, kernel):
    """
    Melakukan konvolusi 2D manual antara citra grayscale dan kernel.
    
    Syarat Tugas:
    1. Menggunakan tipe data float32 untuk presisi.
    2. Melakukan FLIP kernel (Konvolusi sebenar-benarnya, bukan korelasi).
    3. Menggunakan Padding Tepi (BORDER_REPLICATE) untuk akurasi tepi.
    """
    # Ambil dimensi citra dan kernel
    img_h, img_w = image.shape
    kh, kw = kernel.shape
    
    # Hitung padding (asumsi kernel ganjil/pusat di tengah, atau 2x2 roberts)
    pad_h = kh // 2
    pad_w = kw // 2
    
    # Khusus Roberts (2x2), padding perlu penyesuaian agar tidak out of bound
    # Namun, standar padding biasanya floor(size/2).
    # Untuk 2x2: pad=1 (agar bisa proses pixel terakhir).
    if kh % 2 == 0: pad_h = 1
    if kw % 2 == 0: pad_w = 1

    # Buat citra padding dengan metode REPLICATE (mengulang pixel pinggir)
    # Ini penting agar deteksi tepi di pinggir frame tidak hitam (nol) [cite: 59]
    image_padded = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
    
    # Konversi ke float32 untuk perhitungan
    image_padded = image_padded.astype(np.float32)
    
    # FLIP KERNEL (Definisi Matematika Konvolusi: g(x,y) = f * h)
    kernel_flipped = np.flipud(np.fliplr(kernel))
    
    output = np.zeros((img_h, img_w), dtype=np.float32)
    
    # Loop manual (Sliding Window)
    for y in range(img_h):
        for x in range(img_w):
            # Ambil region of interest (ROI)
            # Koordinat padding disesuaikan agar pusat kernel bertemu pixel (y,x)
            # Khusus kernel genap (Roberts), anchor biasanya di pojok kiri atas.
            if kh == 2: # Penanganan khusus Roberts 2x2
                 region = image_padded[y:y+kh, x:x+kw]
            else: # Kernel ganjil (3x3)
                 region = image_padded[y:y+kh, x:x+kw]

            # Operasi konvolusi: Sum of Product
            # Pastikan ukuran region sama dengan kernel (menangani batas loop)
            if region.shape == kernel.shape:
                value = np.sum(region * kernel_flipped)
                output[y, x] = value
                
    return output

# ==============================================================================
# 3. OPERATOR DETEKSI TEPI (DISCONTINUITY)
# ==============================================================================

def calculate_magnitude(gx, gy):
    """Menghitung magnitude gradient: M = sqrt(Gx^2 + Gy^2)"""
    mag = np.sqrt(gx**2 + gy**2)
    return normalize_to_uint8(mag)

def roberts_operator(img_gray):
    """
    Operator Roberts (2x2).
    Referensi: Halaman 12-13 PDF [cite: 73]
    Menggunakan kernel silang diagonal.
    """
    # Kernel Gx (Diagonal Utama)
    kx = np.array([[1, 0],
                   [0, -1]], dtype=np.float32)
    
    # Kernel Gy (Diagonal Sekunder)
    ky = np.array([[0, 1],
                   [-1, 0]], dtype=np.float32)
    
    gx = convolve_2d(img_gray, kx)
    gy = convolve_2d(img_gray, ky)
    
    return calculate_magnitude(gx, gy)

def prewitt_operator(img_gray):
    """
    Operator Prewitt (3x3).
    Referensi: Halaman 15 PDF [cite: 77]
    Lebih sensitif terhadap tepi vertikal dan horizontal.
    """
    # Kernel Gx (Gradient Horizontal -> Deteksi Tepi Vertikal)
    kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=np.float32)
    
    # Kernel Gy (Gradient Vertikal -> Deteksi Tepi Horizontal)
    ky = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]], dtype=np.float32)
    
    gx = convolve_2d(img_gray, kx)
    gy = convolve_2d(img_gray, ky)
    
    return calculate_magnitude(gx, gy)

def sobel_operator(img_gray):
    """
    Operator Sobel (3x3).
    Referensi: Halaman 17 PDF [cite: 78]
    Memberikan bobot 2 pada piksel tengah (smoothing).
    """
    # Kernel Gx
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    
    # Kernel Gy
    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)
    
    gx = convolve_2d(img_gray, kx)
    gy = convolve_2d(img_gray, ky)
    
    return calculate_magnitude(gx, gy)

def frei_chen_operator(img_gray):
    """
    Operator Frei-Chen (3x3).
    Referensi: Halaman 19 PDF [cite: 80]
    Isotropik, menggunakan akar 2 untuk pembobotan agar hasil lebih halus di segala arah.
    """
    sqrt2 = np.sqrt(2)
    
    # Kernel Gx (Berdasarkan referensi pers. R(y,x) hal 19)
    kx = np.array([[1,   0, -1],
                   [sqrt2, 0, -sqrt2],
                   [1,   0, -1]], dtype=np.float32)
    
    # Kernel Gy
    ky = np.array([[-1, -sqrt2, -1],
                   [ 0,   0,     0],
                   [ 1,  sqrt2,  1]], dtype=np.float32)
    
    # Catatan: PDF mungkin menampilkan orientasi berbeda, 
    # namun struktur matematisnya (pembobotan sqrt2) adalah ciri khas Frei-Chen.
    
    gx = convolve_2d(img_gray, kx)
    gy = convolve_2d(img_gray, ky)
    
    return calculate_magnitude(gx, gy)

# ==============================================================================
# 4. FUNGSI VISUALISASI PANEL
# ==============================================================================

def add_label(image, text):
    """Menambahkan teks label pada bagian atas citra."""
    h, w = image.shape
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Tambahkan bar putih di atas untuk teks
    bar_height = 30
    bar = np.full((bar_height, w, 3), 255, dtype=np.uint8)
    
    cv2.putText(bar, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 0, 0), 1, cv2.LINE_AA)
    
    return np.vstack((bar, img_color))

def save_comparison_panel(original, roberts, prewitt, sobel, freichen, filename):
    """
    Membuat grid 2x3:
    [Original] [Roberts] [Prewitt]
    [Sobel]    [FreiChen] [Kosong/Hitam]
    """
    # Beri label pada setiap citra
    img1 = add_label(original, "Original")
    img2 = add_label(roberts, "Roberts (2x2)")
    img3 = add_label(prewitt, "Prewitt (3x3)")
    img4 = add_label(sobel, "Sobel (3x3)")
    img5 = add_label(freichen, "Frei-Chen (3x3)")
    
    # Samakan ukuran jika perlu (biasanya hasil konvolusi sama ukuran dgn input)
    # Buat slot kosong untuk panel ke-6
    h, w, c = img1.shape
    img6 = np.zeros((h, w, c), dtype=np.uint8) # Panel kosong hitam
    
    # Gabungkan Horizontal (Baris 1 dan Baris 2)
    row1 = np.hstack((img1, img2, img3))
    row2 = np.hstack((img4, img5, img6))
    
    # Gabungkan Vertikal
    panel = np.vstack((row1, row2))
    
    # Simpan
    cv2.imwrite(filename, panel)
    print(f"   [INFO] Panel disimpan: {filename}")

# ==============================================================================
# 5. MAIN PROGRAM
# ==============================================================================

def main():
    # --- KONFIGURASI FOLDER ---
    INPUT_FOLDER = "images"       # Ganti dengan nama folder input Anda
    OUTPUT_FOLDER = "hasil_segmentasi"
    
    # Daftar file citra tugas restorasi Anda (Pastikan file ada di folder images)
    IMAGE_LIST = [
        "potrait.jpg",  # Ganti dengan nama file asli Anda
        "potrait-gray.jpg",        # Ganti dengan nama file noise salt & pepper
        "potrait-gray-saltpepper-2.jpg",     # Ganti dengan nama file noise gaussian
        "potrait-gray-gaussian-2.jpg"        # Ganti dengan nama file hasil restorasi
    ]

    ensure_dir(OUTPUT_FOLDER)
    print("=== MULAI PROSES SEGMENTASI TEPI (DISCONTINUITY) ===")
    print(f"Input Folder: {INPUT_FOLDER}")
    print(f"Output Folder: {OUTPUT_FOLDER}\n")

    for img_name in IMAGE_LIST:
        input_path = os.path.join(INPUT_FOLDER, img_name)
        print(f"--- Memproses: {img_name} ---")
        
        # 1. Load Citra
        img = load_as_gray(input_path)
        if img is None:
            print(f"   [SKIP] Gagal memuat gambar: {input_path}")
            continue

        # 2. Proses Deteksi Tepi
        print("   -> Menghitung Roberts...")
        res_roberts = roberts_operator(img)
        
        print("   -> Menghitung Prewitt...")
        res_prewitt = prewitt_operator(img)
        
        print("   -> Menghitung Sobel...")
        res_sobel = sobel_operator(img)
        
        print("   -> Menghitung Frei-Chen...")
        res_frei = frei_chen_operator(img)

        # 3. Simpan Hasil Individu (Opsional, sesuai kebutuhan laporan)
        base_name = os.path.splitext(img_name)[0]
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{base_name}_roberts.png"), res_roberts)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{base_name}_prewitt.png"), res_prewitt)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{base_name}_sobel.png"), res_sobel)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{base_name}_freichen.png"), res_frei)

        # 4. Buat dan Simpan Panel Perbandingan (Penting untuk Laporan)
        panel_filename = os.path.join(OUTPUT_FOLDER, f"{base_name}_PANEL.png")
        save_comparison_panel(img, res_roberts, res_prewitt, res_sobel, res_frei, panel_filename)
        
    print("\n=== PROSES SELESAI ===")
    print("Silakan cek folder output untuk hasil gambar dan panel perbandingan.")

if __name__ == "__main__":
    main()