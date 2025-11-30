# Program Segmentasi Citra - Deteksi Tepi (Edge Detection)

Program ini dibuat untuk memenuhi tugas **Segmentasi Citra** dengan pendekatan *discontinuity* (deteksi tepi). Program ini ditulis dalam bahasa Python dalam satu file tunggal dan mengimplementasikan operasi konvolusi manual tanpa menggunakan fungsi deteksi tepi bawaan library.

## 📋 Fitur Utama

1.  **Implementasi Konvolusi Manual**:
      * Menggunakan operasi *sliding window* manual.
      * Menggunakan tipe data `float32` untuk presisi perhitungan.
      * Menerapkan *kernel flipping* sesuai definisi matematis konvolusi.
      * Menggunakan *Padding Tepi* (`BORDER_REPLICATE`) untuk hasil tepian yang lebih akurat.
2.  **4 Metode Operator Tepi**:
      * [cite\_start]**Roberts** (Kernel 2x2) [cite: 73]
      * [cite\_start]**Prewitt** (Kernel 3x3) [cite: 77]
      * [cite\_start]**Sobel** (Kernel 3x3) [cite: 78]
      * [cite\_start]**Frei-Chen** (Kernel 3x3 Isotropik) [cite: 80]
3.  **Visualisasi Panel**: Secara otomatis membuat gambar panel perbandingan (Original vs 4 Metode) untuk memudahkan analisis laporan.

## 🛠️ Persyaratan Sistem

Pastikan Anda telah menginstal Python dan library berikut:

  * **Python** (3.x)
  * **OpenCV** (`cv2`)
  * **NumPy**

Untuk menginstal library yang dibutuhkan, jalankan perintah berikut di terminal/command prompt:

```bash
pip install opencv-python numpy
```

## 📂 Struktur Folder

Agar program berjalan dengan benar, pastikan struktur folder Anda terlihat seperti ini:

```text
folder_tugas_anda/
│
├── tugas_segmentasi.py      # File kode program utama
├── README.md                # File dokumentasi ini
├── images/                  # FOLDER INPUT: Masukkan citra tugas di sini
│   ├── potrait.jpg
│   ├── potrait-gray.jpg
│   ├── potrait-gray-saltpepper-2.jpg
│   └── potrait-gray-gaussian-2.jpg
│
└── hasil_segmentasi/        # FOLDER OUTPUT: Akan dibuat otomatis oleh program
```

## 🚀 Cara Menggunakan Program

1.  **Siapkan Citra**: Masukkan 4 citra yang akan diproses (Citra Asli, Citra Grayscale, Noise Salt & Pepper, Noise Gaussian) ke dalam folder `images`.

2.  **Konfigurasi Nama File**:
    Buka file `tugas_segmentasi.py` menggunakan text editor (VS Code, Notepad++, dll). Cari bagian `def main():` dan sesuaikan list `IMAGE_LIST` dengan nama file gambar Anda yang sebenarnya.

    ```python
    # Contoh di dalam kode:
    IMAGE_LIST = [
        "nama_file_asli.jpg",
        "nama_file_grayscale.jpg",
        "nama_file_noise_sp.jpg",
        "nama_file_noise_gauss.jpg"
    ]
    ```

3.  **Jalankan Program**:
    Buka terminal atau command prompt di folder proyek, lalu jalankan:

    ```bash
    python tugas_segmentasi.py
    ```

4.  **Cek Hasil**:
    Hasil segmentasi akan muncul di folder `hasil_segmentasi`.

## 📄 Output Program

Untuk setiap citra input, program akan menghasilkan 5 file output:

1.  `namafile_roberts.png`: Hasil deteksi tepi metode Roberts.
2.  `namafile_prewitt.png`: Hasil deteksi tepi metode Prewitt.
3.  `namafile_sobel.png`: Hasil deteksi tepi metode Sobel.
4.  `namafile_freichen.png`: Hasil deteksi tepi metode Frei-Chen.
5.  `namafile_PANEL.png`: **Gambar gabungan** yang menampilkan citra asli dan keempat hasil metode di atas dalam satu bingkai (sangat berguna untuk laporan).

## 📝 Catatan Implementasi

  * Program secara otomatis mengonversi citra input menjadi **Grayscale** sebelum diproses.
  * Magnitude gradien dihitung menggunakan rumus $M = \sqrt{G_x^2 + G_y^2}$.
  * Hasil akhir dinormalisasi ke rentang 0-255 (`uint8`) agar dapat disimpan sebagai gambar `.png`.