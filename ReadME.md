# Banana Leaf Disease Classification Web Application

Aplikasi web berbasis **Flask** untuk mengklasifikasikan penyakit daun pisang menggunakan metode **Machine Learning** dengan ekstraksi fitur citra (warna, tekstur, dan bentuk). Pengguna hanya perlu mengunggah gambar daun pisang, kemudian sistem akan melakukan preprocessing, ekstraksi fitur, dan memberikan hasil klasifikasi beserta tingkat kepercayaan (confidence), deskripsi penyakit, dan rekomendasi penanganan.

---

# 1. Penjelasan Program

Program ini dikembangkan untuk membantu proses identifikasi penyakit daun pisang secara otomatis melalui citra digital.

Sistem mengenali empat kelas daun pisang:

- Healthy
- Cordana
- Pestalotiopsis
- Sigatoka

Model Machine Learning dilatih menggunakan dataset daun pisang yang telah melalui proses ekstraksi fitur sehingga proses prediksi menjadi lebih cepat dibandingkan menggunakan pengolahan citra secara langsung.

---

# 2. Cara Install

## 1. Clone / Download Project

```bash
git clone <repository-url>
```

atau ekstrak file ZIP.

---

## 2. Masuk ke Folder Project

```bash
cd classification_penyakit_pohon_pisang
```

---

## 3. Buat Virtual Environment

Windows

```bash
python -m venv .venv
```

---

## 4. Aktifkan Virtual Environment

### CMD

```cmd
.venv\Scripts\activate.bat
```

### PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### Git Bash

```bash
source .venv/Scripts/activate
```

---

## 5. Install Dependency

```bash
pip install -r requirements.txt
```

---

## 6. Konfigurasi Environment

Buat file `.env`

```text
SECRET_KEY=penyakit_pohon_pisang
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=5242880
MODEL_PATH=models/svm_model.pkl
TEMPLATES_FOLDER=app/templates
STATIC_FOLDER=app/static
STATIC_URL_PATH=/static
```

---

# 3. Cara Menggunakan

## Jalankan Flask

```bash
python app.py
```

Buka browser

```
http://127.0.0.1:5000
```

Langkah penggunaan:

1. Klik **Choose Image**.
2. Pilih gambar daun pisang.
3. Klik **Predict**.
4. Tunggu proses klasifikasi.
5. Sistem akan menampilkan:

- Preview gambar
- Jenis penyakit
- Confidence
- Deskripsi
- Rekomendasi penanganan

---

# 4. Diagram Folder

```
classification_penyakit_pohon_pisang/
│
├── app/                        # Source code aplikasi Flask
│   ├── routes/                 # Routing
│   ├── services/               # Business logic
│   ├── static/                 # CSS, JS, Images
│   └── templates/              # HTML Template
│
├── dataset/
│   ├── OriginalSet/            # Dataset asli
│   └── AugmentedSet/           # Dataset hasil augmentasi
│
├── extracted_dataset/          # Dataset hasil ekstraksi fitur
│
├── models/                     # Model Machine Learning (.pkl)
│
├── training/                   # Script training model
│
├── uploads/                    # Gambar yang diupload user
│
├── app.py                      # Entry point aplikasi
├── config.py                   # Konfigurasi aplikasi
├── requirements.txt            # Dependency Python
├── PROJECT_DOCUMENTATION.md    # Dokumentasi lengkap
└── README.md                   # Dokumentasi singkat
```

---

## Teknologi yang Digunakan

- Python
- Flask
- OpenCV
- NumPy
- Pandas
- Scikit-Learn
- Scikit-Image
- Joblib
- Bootstrap 5
