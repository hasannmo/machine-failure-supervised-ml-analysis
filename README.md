# Machine Failure — Supervised ML Analysis

Analisis klasifikasi untuk memprediksi **kegagalan mesin** dari data [AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset). Proyek ini membandingkan Logistic Regression, Decision Tree, dan K-Nearest Neighbors (KNN) pada tiga kelompok fitur: data sensor asli (A), fitur polinomial (B), dan fitur berbasis rumus fisika (C). Seluruh analisis dijalankan melalui Jupyter Notebook; demo prediksi lokal tersedia melalui Gradio.

## Cara menjalankan

### 1. Ambil repo dan pasang dependensi

Pastikan **Python dan Git** sudah terpasang. Jalankan perintah berikut di terminal:

```bash
git clone https://github.com/hasannmo/machine-failure-supervised-ml-analysis.git
cd machine-failure-supervised-ml-analysis
python -m venv .venv
```

Aktifkan virtual environment sesuai sistem operasi:

| Sistem | Perintah |
| --- | --- |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

Lalu pasang dependensi dari repo:

```bash
python -m pip install -r requirements.txt
```

### 2. Buka notebook

Jalankan Jupyter dari **direktori utama repo**:

```bash
jupyter notebook
```

Di halaman Jupyter yang terbuka, masuk ke folder `notebooks/`, pilih kernel dari virtual environment `.venv`, lalu jalankan sel dengan **Run All** atau satu per satu dari atas ke bawah.

**Untuk melihat demo dengan cepat:** buka `notebooks/05_demo_gradio.ipynb` dan jalankan seluruh sel. Notebook ini memuat model serta `results/feature_splits.joblib` yang sudah ada di repo, kemudian membuka antarmuka Gradio di alamat lokal yang ditampilkan pada output (biasanya `http://127.0.0.1:7860`). Pilih contoh input atau isi nilai sensor, skenario fitur, dan model, lalu jalankan prediksi. Demo ini memakai model tersimpan; tidak perlu melatih ulang model terlebih dahulu.

**Untuk mengulang seluruh analisis:** jalankan notebook berikut **sesuai urutan**:

| Urutan | Notebook | Fungsi |
| --- | --- | --- |
| 1 | `notebooks/01_eda.ipynb` | Eksplorasi data dan visualisasi awal. |
| 2 | `notebooks/02_features.ipynb` | Pemisahan data serta pembuatan fitur skenario A, B, dan C; menyimpan `results/feature_splits.joblib`. |
| 3 | `notebooks/03_models.ipynb` | Pencarian parameter, pelatihan, dan evaluasi tiga model pada setiap skenario; menyimpan model ke `models/` dan metrik ke `results/`. |
| 4 | `notebooks/04_error_analysis.ipynb` | Analisis kesalahan prediksi dan perbandingan dengan aturan referensi. |
| 5 | `notebooks/05_demo_gradio.ipynb` | Demo interaktif memakai model yang telah dilatih. |

Pelatihan pada notebook ketiga memakai pencarian parameter dengan validasi silang, sehingga dapat memerlukan waktu lebih lama daripada membuka demo.

## Isi repo

```text
data/
  ai4i2020.csv           Dataset utama (sudah tersedia)
  demo_input.csv         Contoh masukan demo
notebooks/               Lima notebook analisis dan demo
src/                     Fungsi preprocessing, pembuatan fitur, dan aturan referensi
models/                  Sembilan model .joblib yang sudah dilatih
results/                 Pemisahan fitur, metrik, dan gambar hasil analisis
requirements.txt         Daftar dependensi Python
```

Target klasifikasi adalah kolom `Machine failure`. Masukan model berasal dari `Type`, suhu udara, suhu proses, kecepatan putar, torsi, dan keausan alat. Kolom jenis kegagalan (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) dipakai untuk analisis, bukan sebagai masukan model.

## Jika ada kendala

- **`ModuleNotFoundError` atau kernel tidak menemukan paket:** pastikan `.venv` aktif saat menjalankan `jupyter notebook` dan kernel notebook memakai environment yang sama. Ulangi `python -m pip install -r requirements.txt` bila perlu.
- **`FileNotFoundError` untuk data, model, atau `feature_splits.joblib`:** jalankan Jupyter dari direktori utama repo. Jika artefak hasil telah terhapus, jalankan `02_features.ipynb` lalu `03_models.ipynb` untuk membuatnya kembali.
- **Port Gradio sudah digunakan:** lihat alamat lokal yang benar pada output sel terakhir `05_demo_gradio.ipynb`.

Model `.joblib` sebaiknya dibuka hanya dari repo/sumber yang dipercaya. Versi paket yang digunakan proyek tercantum di `requirements.txt`.
