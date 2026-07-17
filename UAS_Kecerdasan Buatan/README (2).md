# SkinVision AI — Klasifikasi Kondisi Kulit Berbasis Computer Vision

Model klasifikasi citra kondisi kulit menggunakan **Transfer Learning MobileNetV2**, dikembangkan sebagai proyek UAS mata kuliah Kecerdasan Buatan. Model menerima input gambar kulit, melakukan preprocessing, lalu menghasilkan prediksi kelas kondisi kulit beserta *confidence score*.

> ⚠️ **Disclaimer:** Proyek ini hanya untuk keperluan demo/edukasi (UAS). Hasil prediksi **tidak menggantikan pemeriksaan/diagnosis dokter kulit (dermatolog)**.

---

## 📋 Daftar Isi

- [Tentang Proyek](#tentang-proyek)
- [Kelas Kondisi Kulit](#kelas-kondisi-kulit)
- [Dataset](#dataset)
- [Arsitektur Model](#arsitektur-model)
- [Hasil & Performa](#hasil--performa)
- [Instalasi](#instalasi)
- [Cara Penggunaan](#cara-penggunaan)
- [Struktur Proyek](#struktur-proyek)
- [Keterbatasan](#keterbatasan)
- [Rencana Pengembangan](#rencana-pengembangan)
- [Referensi](#referensi)

---

## Tentang Proyek

Kondisi kulit wajah seperti jerawat, komedo, flek hitam, kulit kering, kulit berminyak, hingga kerutan adalah permasalahan umum, namun tidak semua orang punya akses cepat dan terjangkau ke dokter kulit. **SkinVision AI** dibangun untuk memberikan estimasi/skrining awal kondisi kulit dari sebuah foto, memanfaatkan model *pre-trained* **MobileNetV2** (ImageNet) yang di-*fine-tune* pada dataset gambar kulit.

**Tujuan:**
- Membangun model klasifikasi gambar kulit ke dalam 7 kelas kondisi.
- Menerapkan *Transfer Learning* MobileNetV2 agar lebih kuat dibanding CNN sederhana yang dilatih dari nol.
- Melakukan *fine-tuning* pada layer akhir MobileNetV2 agar fitur lebih sesuai karakteristik kulit.
- Menghasilkan output prediksi + *confidence score* yang bisa ditampilkan di aplikasi web.

## Kelas Kondisi Kulit

Model mengklasifikasikan gambar ke dalam 7 kelas:

| Kelas | Keterangan |
| :--- | :--- |
| Acne | Jerawat |
| Blackheads | Komedo hitam |
| Dark Spots | Flek hitam |
| Dry Skin | Kulit kering |
| Normal Skin | Kulit normal |
| Oily Skin | Kulit berminyak |
| Wrinkles | Kerutan |

## Dataset

- Sumber: kumpulan gambar kulit internal (ZIP), dibagi ke folder `train/` dan `val/` per kelas.
- Total **701 gambar** pada 7 kelas (71–141 gambar per kelas — sedikit tidak seimbang).
- Split training/validasi: **563 gambar latih** dan **138 gambar validasi** (`validation_split=0.2`).
- Preprocessing: resize ke 224×224 px, normalisasi piksel (0–1), serta augmentasi (rotasi, zoom, shift, shear, brightness, flip horizontal) untuk data latih.

Detail lengkap distribusi data dan EDA ada di [`Laporan_UAS_SkinVisionAI.md`](Laporan_UAS_SkinVisionAI.md).

## Arsitektur Model

**Base model:** MobileNetV2 (`weights="imagenet"`, `include_top=False`), input 224×224×3

**Custom head:**
```
GlobalAveragePooling2D
→ Dense(256, relu) → BatchNorm → Dropout(0.4)
→ Dense(128, relu) → BatchNorm → Dropout(0.3)
→ Dense(7, softmax)
```

**Strategi training (2 tahap):**
1. **Head training** — base model dibekukan penuh, hanya custom head yang dilatih (`lr=0.0001`, hingga 30 epoch).
2. **Fine-tuning** — 40 layer terakhir MobileNetV2 dibuka kembali (`lr=0.00001`, hingga 15 epoch).

Callback yang digunakan: `ModelCheckpoint`, `EarlyStopping` (patience 7), `ReduceLROnPlateau` (patience 3).

## Hasil & Performa

| Metrik | Nilai |
| :--- | :---: |
| Validation Accuracy | **52,90%** |
| Validation Loss | 1,4145 |

- Performa terbaik pada kelas dengan pola visual khas: **Blackheads** (F1 0,72) dan **Dark Spots** (F1 0,73).
- Performa terendah pada kelas yang mirip secara visual: **Normal Skin** (F1 0,21) dan **Dry Skin** (F1 0,22).
- Terindikasi *overfitting ringan* — akurasi training terus naik, akurasi validasi stagnan di 50–55%.

Classification report lengkap, confusion matrix, dan kurva training tersedia di [`Laporan_UAS_SkinVisionAI.md`](Laporan_UAS_SkinVisionAI.md#7-evaluation--pengujian).

## Instalasi

Notebook dikembangkan dan dijalankan di **Google Colab**. Untuk menjalankan secara lokal:

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

## Cara Penggunaan

1. Buka `Klasifikasi_Kulit_Berbasis_Computer_Vision.ipynb`.
2. Jalankan sel secara berurutan — unggah dataset (ZIP berisi folder per kelas) saat diminta.
3. Notebook akan otomatis mendeteksi folder dataset, melakukan preprocessing, augmentasi, dan training 2 tahap.
4. Setelah training selesai, model dan label kelas tersimpan sebagai:
   - `skin_disease_model.h5`
   - `class_names.txt`
5. Untuk uji prediksi satu gambar, jalankan sel prediksi dan unggah gambar kulit (`.jpg`/`.jpeg`/`.png`) — sistem akan menampilkan Top-3 prediksi beserta persentase *confidence*.

## Struktur Proyek

```
.
├── Klasifikasi_Kulit_Berbasis_Computer_Vision.ipynb   # Notebook utama (training & evaluasi)
├── Laporan_UAS_SkinVisionAI.md                        # Laporan lengkap (EDA, modeling, evaluasi)
├── skin_disease_model.h5                              # Model hasil training (dihasilkan setelah run)
├── class_names.txt                                    # Daftar nama kelas (dihasilkan setelah run)
└── images/                                             # Grafik hasil EDA & evaluasi
```

## Keterbatasan

- Jumlah data latih per kelas masih minim (rata-rata puluhan gambar/kelas) dan tidak seimbang antar kelas.
- Beberapa kelas memiliki kemiripan visual tinggi (Normal Skin, Oily Skin, Dry Skin) sehingga mudah tertukar.
- Model belum divalidasi oleh tenaga medis/dermatolog — **bukan alat diagnosis resmi**.

## Rencana Pengembangan

- Menambah jumlah data, terutama pada kelas dengan performa rendah (Dry Skin, Normal Skin).
- Menerapkan `class_weight` untuk mengatasi ketidakseimbangan kelas.
- Menambah regularisasi/augmentasi untuk mengurangi overfitting.
- Membandingkan dengan arsitektur lain (EfficientNet, ResNet).
- Deployment ke aplikasi web dengan disclaimer medis yang jelas.

## Referensi

1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*.
2. Abadi, M., et al. (2016). TensorFlow: A system for large-scale machine learning. *OSDI 16*.
3. Chollet, F., et al. (2015). *Keras*. https://keras.io
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825-2830.
5. Deng, J., et al. (2009). ImageNet: A large-scale hierarchical image database. *CVPR*.

---

Laporan lengkap tersedia di [`Laporan_UAS_SkinVisionAI.md`](Laporan_UAS_SkinVisionAI.md).
