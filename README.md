# Analisis Performa Aiming - Tugas Akhir Pengolahan Citra

Proyek ini adalah sistem berbasis deteksi objek YOLOv8 untuk menganalisis performa aiming (membidik) pemain pada game FPS (seperti Valorant). Sistem ini mendeteksi posisi target (kepala/tubuh musuh) dan menghitung metrik akurasi bidikan secara otomatis dari rekaman gameplay maupun tangkapan layar langsung (real-time).

## Fitur Utama

* **Analisis Video**: Menganalisis rekaman video gameplay secara frame-demi-frame.
* **Analisis Real-Time**: Tangkapan layar langsung menggunakan DXcam untuk mendeteksi target saat bermain.
* **Metrik Evaluasi**: Menghitung akurasi aiming, jarak rata-rata crosshair ke target, rasio overshoot (bidikan melampaui target), serta mengklasifikasikan tingkat kemahiran pemain (Pemula, Menengah, Mahir).
* **Perbandingan Sesi**: Membandingkan statistik performa aiming antara beberapa sesi latihan yang tersimpan.

## Persyaratan Sistem

* **Python**: Versi 3.9 ke atas (diuji pada Python 3.10.6)
* **Resolusi Game/Video**: Dioptimalkan untuk resolusi 1920×1080
* **Sistem Operasi**: Windows 10/11

## Cara Menjalankan Aplikasi

Aplikasi ini dapat dijalankan menggunakan dua jenis antarmuka (Desktop GUI atau Web Dashboard). Terdapat berkas launcher otomatis (.bat) untuk memudahkan proses menjalankan aplikasi:

### 1. Menjalankan Aplikasi Desktop (CustomTkinter)
Klik dua kali pada berkas launcher berikut:
* `Run_Desktop_Analyzer.bat`

### 2. Menjalankan Dashboard Web (Streamlit)
Klik dua kali pada berkas launcher berikut:
* `Run_Streamlit_AppMode.bat`

---

### Instalasi Manual (Alternatif)
Jika ingin menjalankan secara manual via Command Prompt / PowerShell:
1. Buka Command Prompt di direktori proyek ini.
2. Buat dan aktifkan virtual environment (opsional).
3. Instal semua dependensi:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi desktop:
   ```bash
   python desktop_app.py
   ```
5. Atau jalankan dashboard web Streamlit:
   ```bash
   streamlit run app.py
   ```
