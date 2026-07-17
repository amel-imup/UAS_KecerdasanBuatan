# LAPORAN UJIAN AKHIR SEMESTER (UAS) KECERDASAN BUATAN

**Mata Kuliah:** Kecerdasan Buatan

**Topik Proyek:** Klasifikasi Kondisi Kulit Berbasis Computer Vision dengan Transfer Learning MobileNetV2 (SkinVision AI)

**Anggota Kelompok:**

1. _(Ina Amelia Cantika 2406122)_
2. _(Aprilia Fauziah 2406090)_

**LINK/SUMBER Dataset:** (https://www.kaggle.com/datasets/rajeshpandhare/skin-classification-normal-acne-oily-dry-wrinkle)

---

## 1. Judul Proyek

**"Klasifikasi Kondisi Kulit Berbasis Computer Vision"**

---

## 2. Business Understanding

### Latar Belakang Masalah (Domain Proyek)

Kondisi kulit wajah seperti jerawat, komedo, flek hitam, kulit kering, kulit berminyak, hingga kerutan merupakan permasalahan yang umum dialami masyarakat, namun tidak semua orang memiliki akses cepat dan terjangkau untuk berkonsultasi dengan dokter kulit (dermatolog). Banyak orang mengandalkan pengamatan visual sendiri atau pencarian informasi di internet yang belum tentu akurat untuk mengenali jenis kondisi kulit yang dialami.

Perkembangan teknologi *Computer Vision* dan *Deep Learning*, khususnya pendekatan *Transfer Learning*, memungkinkan dibangunnya sistem yang dapat mengenali pola visual pada gambar kulit secara otomatis. Dengan memanfaatkan model pre-trained seperti **MobileNetV2** yang sudah dilatih pada jutaan gambar (ImageNet), proses pelatihan model klasifikasi gambar kulit dapat dilakukan lebih efisien meskipun dengan jumlah data yang terbatas.

### Permasalahan Dunia Nyata

1. **Keterbatasan Akses Konsultasi Dermatologi:** Tidak semua masyarakat dapat dengan mudah dan cepat berkonsultasi ke dokter kulit untuk sekadar mengetahui kondisi kulit dasar yang dialami.
2. **Kesulitan Mengenali Kondisi Kulit Secara Mandiri:** Masyarakat awam sulit membedakan secara visual antara kondisi kulit yang mirip, misalnya antara kulit berminyak dan kulit normal, atau antara flek hitam dan komedo.
3. **Kebutuhan Skrining Awal yang Cepat:** Diperlukan alat bantu skrining awal berbasis gambar yang dapat memberikan estimasi kondisi kulit secara instan sebelum pengguna memutuskan untuk berkonsultasi lebih lanjut ke tenaga medis.

### Tujuan Proyek

1. **Membangun Model Klasifikasi Gambar Kulit:** Mengembangkan model *Computer Vision* yang mampu mengklasifikasikan gambar kulit ke dalam 7 kelas kondisi: Acne, Blackheads, Dark Spots, Dry Skin, Normal Skin, Oily Skin, dan Wrinkles.
2. **Menerapkan Transfer Learning:** Memanfaatkan arsitektur **MobileNetV2** yang telah dilatih pada dataset ImageNet sebagai basis ekstraksi fitur, agar model lebih kuat dibanding CNN sederhana yang dilatih dari nol (*from scratch*).
3. **Melakukan Fine-Tuning:** Melatih ulang sebagian layer akhir MobileNetV2 agar representasi fitur lebih sesuai dengan karakteristik gambar kulit.
4. **Menyediakan Output yang Informatif:** Menghasilkan prediksi kelas kondisi kulit beserta *confidence score*, sehingga dapat ditampilkan pada aplikasi web sebagai hasil skrining awal.

### Target Pengguna (User)

1. **Masyarakat Umum:** Sebagai pengguna utama yang ingin mengetahui estimasi kondisi kulitnya secara cepat dan gratis sebelum berkonsultasi ke dokter.
2. **Pelaku Industri Kecantikan/Skincare:** Sebagai alat bantu awal untuk merekomendasikan produk perawatan kulit sesuai kondisi yang terdeteksi.
3. **Mahasiswa/Peneliti Bidang AI Kesehatan:** Sebagai referensi penerapan *Transfer Learning* pada studi kasus klasifikasi citra medis/kesehatan kulit.

### Solusi dan Manfaat

Proyek ini membangun model klasifikasi citra kulit menggunakan pendekatan *Transfer Learning* dengan basis **MobileNetV2**, dilatih melalui dua tahap: (1) pelatihan layer tambahan di atas *base model* yang dibekukan (*frozen*), dan (2) *fine-tuning* pada 40 layer terakhir MobileNetV2 agar model lebih adaptif terhadap karakteristik visual gambar kulit.

- **Skrining Cepat:** Pengguna mendapatkan estimasi kondisi kulit hanya dengan mengunggah satu foto, tanpa perlu menunggu jadwal konsultasi.
- **Efisiensi Model:** MobileNetV2 dipilih karena arsitekturnya ringan (*lightweight*), sehingga cocok untuk deployment pada aplikasi web maupun perangkat dengan sumber daya terbatas.
- **Transparansi Hasil:** Sistem menampilkan Top-3 prediksi beserta persentase *confidence*, serta status keyakinan model (cukup kuat / belum yakin) sehingga pengguna memahami batas kemampuan model.
- **Disclaimer Etis:** Aplikasi secara eksplisit dinyatakan sebagai demo proyek UAS dan hasil prediksinya **tidak menggantikan pemeriksaan medis oleh dokter**.

---

## 3. Data Understanding

### Sumber Dataset

Dataset berupa kumpulan gambar kulit dalam format ZIP (`archive (4).zip`) yang diunggah langsung ke Google Colab. Setelah diekstraksi, struktur dataset berbentuk:

```
skinvision_dataset/
  skin_dataset/
    train/
      Acne/
      Blackheads/
      Dark Spots/
      Dry Skin/
      Normal Skin/
      Oily Skin/
      Wrinkles/
    val/
```

Folder dataset yang digunakan untuk pelatihan terdeteksi secara otomatis pada path `skinvision_dataset/skin_dataset/train`, dengan 7 subfolder kelas sesuai nama kondisi kulit.

### Karakteristik dan Spesifikasi Data

| Class | Jumlah Gambar |
| :--- | :---: |
| Acne | 106 |
| Blackheads | 90 |
| Dark Spots | 98 |
| Dry Skin | 71 |
| Normal Skin | 92 |
| Oily Skin | 103 |
| Wrinkles | 141 |
| **Total** | **701** |

Setelah pembagian data menggunakan `validation_split=0.2` pada `ImageDataGenerator`, diperoleh **563 gambar untuk data latih (training)** dan **138 gambar untuk data validasi**, keduanya tersebar ke dalam 7 kelas yang sama.

**7 kelas kondisi kulit** yang dicakup: Acne (jerawat), Blackheads (komedo hitam), Dark Spots (flek hitam), Dry Skin (kulit kering), Normal Skin (kulit normal), Oily Skin (kulit berminyak), dan Wrinkles (kerutan).

### Validasi Data

Dari hasil eksplorasi jumlah gambar per kelas, terlihat bahwa dataset **tidak seimbang secara sempurna** (*mild class imbalance*) — jumlah gambar berkisar antara 71 (Dry Skin) hingga 141 (Wrinkles), dengan selisih hampir dua kali lipat antara kelas terkecil dan terbesar. Kondisi ini menjadi salah satu faktor yang perlu diperhatikan pada tahap modeling dan evaluasi.

---

## 4. Exploratory Data Analysis (EDA)

Sebelum masuk ke tahap pemodelan, dilakukan eksplorasi data untuk memahami distribusi jumlah gambar per kelas serta karakteristik visual dataset.

### 4.1 Distribusi Jumlah Gambar per Kelas

![Jumlah Gambar per Class](image/1_class_distribution.png)

Kelas **Wrinkles** memiliki jumlah gambar terbanyak (141 gambar), sedangkan kelas **Dry Skin** memiliki jumlah gambar paling sedikit (71 gambar). Ketimpangan ini berpotensi membuat model lebih condong (*bias*) dalam mengenali kelas dengan jumlah data lebih besar, dan menjadi salah satu kandidat penyebab performa yang lebih rendah pada kelas dengan data minim (lihat Bagian 7).

### 4.2 Contoh Gambar per Kelas


![Contoh Gambar per Class](../image/2_sample_images.png)

Visualisasi satu sampel gambar dari masing-masing kelas menunjukkan bahwa antar-kelas memiliki kemiripan visual yang cukup tinggi, terutama antara **Normal Skin**, **Oily Skin**, dan **Dry Skin**, yang secara kasat mata sulit dibedakan bahkan oleh manusia awam tanpa pencahayaan dan sudut pengambilan gambar yang konsisten. Kemiripan visual ini menjadi tantangan tersendiri bagi model klasifikasi.

### 4.3 Insight Awal dari EDA

1. Dataset berjumlah 701 gambar untuk 7 kelas tergolong **kecil** untuk standar *Computer Vision*, sehingga pendekatan *Transfer Learning* (dibanding melatih CNN dari nol) menjadi pilihan yang tepat.
2. Terdapat **ketidakseimbangan kelas ringan hingga sedang**, dengan rasio kelas terbesar terhadap terkecil mencapai ±2:1 (Wrinkles vs Dry Skin).
3. Kemiripan visual antar beberapa kelas (terutama kondisi kulit yang berkaitan dengan tekstur/kelembapan seperti Normal, Oily, dan Dry Skin) berpotensi menjadi sumber utama kesalahan klasifikasi model.
4. Augmentasi data menjadi krusial mengingat jumlah data latih yang terbatas per kelas (rata-rata hanya puluhan gambar per kelas untuk training).

---

## 5. Data Preparation & Preprocessing

Sebelum masuk ke tahap pemodelan, seluruh data gambar diproses melalui pipeline berikut:

1. **Ekstraksi & Deteksi Otomatis Folder Dataset:** File ZIP diekstrak, kemudian sistem secara otomatis mencari folder yang berisi minimal 3 dari 7 nama kelas yang diharapkan (Acne, Blackheads, Dark Spots, Dry Skin, Normal Skin, Oily Skin, Wrinkles) untuk menentukan direktori dataset yang valid.

2. **Resize Gambar:** Seluruh gambar diseragamkan ukurannya menjadi **224×224 piksel** (`IMG_SIZE = 224`), sesuai dengan ukuran input standar MobileNetV2.

3. **Normalisasi Piksel:** Nilai piksel gambar diskalakan dari rentang 0–255 menjadi 0–1 (`rescale=1./255`).

4. **Data Augmentation (khusus data latih):** Untuk memperkaya variasi data latih dan mengurangi risiko *overfitting* akibat jumlah data yang terbatas, diterapkan augmentasi berupa:
   - Rotasi acak hingga 30°
   - Zoom acak hingga 25%
   - Pergeseran horizontal & vertikal hingga 15%
   - *Shear* hingga 15%
   - Perubahan kecerahan (*brightness*) pada rentang 0.8–1.2
   - *Flip* horizontal
   - Mode pengisian piksel kosong `nearest`

5. **Split Data:** Data dibagi menjadi data latih dan data validasi menggunakan `validation_split=0.2` (80% latih, 20% validasi), menghasilkan **563 gambar latih** dan **138 gambar validasi** dengan `BATCH_SIZE = 16`.

6. **Encoding Label:** Label kelas dikonversi ke format *categorical* (one-hot encoding) melalui `class_mode="categorical"` pada `ImageDataGenerator`.

---

## 6. Modeling / Arsitektur Sistem

### Arsitektur Model: Transfer Learning MobileNetV2

Model dibangun menggunakan **MobileNetV2** sebagai *base model* (`weights="imagenet"`, `include_top=False`), dengan tambahan *custom head* untuk klasifikasi 7 kelas:

| Layer | Output Shape | Keterangan |
| :--- | :--- | :--- |
| MobileNetV2 (base) | (7, 7, 1280) | Base model, awalnya dibekukan (`trainable=False`) |
| GlobalAveragePooling2D | (1280,) | Meringkas fitur spasial menjadi vektor |
| Dense (256, ReLU) | (256,) | + BatchNormalization + Dropout (0.4) |
| Dense (128, ReLU) | (128,) | + BatchNormalization + Dropout (0.3) |
| Dense (7, Softmax) | (7,) | Output probabilitas 7 kelas |

**Total parameter:** 2.621.255 — dengan **362.503 parameter *trainable*** dan 2.258.752 parameter *non-trainable* (dibekukan dari MobileNetV2) pada tahap awal pelatihan.

**Konfigurasi kompilasi:** optimizer `Adam` (`learning_rate=0.0001`), *loss* `categorical_crossentropy`, metrik `accuracy`.

### Tahap Pelatihan 1: Training Head Layer

Base model MobileNetV2 dibekukan penuh (`base_model.trainable = False`), hanya melatih layer *custom head* di atasnya selama maksimum 30 *epoch*, dengan tiga *callback*:

- **ModelCheckpoint:** menyimpan model terbaik berdasarkan `val_accuracy`.
- **EarlyStopping:** menghentikan pelatihan jika `val_loss` tidak membaik selama 7 *epoch* berturut-turut, sekaligus mengembalikan bobot terbaik.
- **ReduceLROnPlateau:** menurunkan *learning rate* sebesar faktor 0.2 jika `val_loss` stagnan selama 3 *epoch*.

Akurasi validasi meningkat bertahap dari **15,9% (epoch 1)** hingga mencapai **55,07% (epoch 18)**, yang menjadi titik terbaik pada tahap ini.

### Tahap Pelatihan 2: Fine-Tuning

Setelah tahap pertama, dilakukan *fine-tuning* dengan membuka kembali seluruh layer MobileNetV2 (`base_model.trainable = True`), namun tetap membekukan seluruh layer **kecuali 40 layer terakhir** (`base_model.layers[:-40]` dibekukan). *Learning rate* diturunkan menjadi **0.00001** agar bobot pre-trained tidak rusak drastis, kemudian dilatih ulang hingga maksimum 15 *epoch* dengan callback yang sama.

Pada tahap *fine-tuning* ini, akurasi validasi terbaik yang tercatat tetap berada di **55,07%** (tidak melampaui hasil tahap 1), mengindikasikan model belum menunjukkan peningkatan tambahan pada fase ini.

### Penyimpanan Model

Model akhir disimpan dalam format `.h5` (`skin_disease_model.h5`) menggunakan `model.save()`, beserta daftar nama kelas pada `class_names.txt`, agar dapat dimuat ulang untuk keperluan *deployment* pada aplikasi web tanpa perlu melatih ulang dari awal.

---

## 7. Evaluation & Pengujian

### 7.1 Kurva Akurasi dan Loss

**Tahap 1 (Training Head Layer):**

![Akurasi Tahap 1](image/3_accuracy.png)
![Loss Tahap 1](image/4_loss.png)

**Tahap 2 (Fine-Tuning):**

![Akurasi Fine-Tuning](image/5_fine_tuning_accuracy.png)
![Loss Fine-Tuning](image/6_fine_tuning_loss.png)

Kurva menunjukkan akurasi *training* terus meningkat pada kedua tahap, namun akurasi *validation* cenderung stagnan/mendatar di kisaran 50–55% setelah beberapa *epoch*, sementara *loss training* terus menurun. Pola ini mengindikasikan gejala **overfitting ringan**, di mana model semakin menghafal data latih namun tidak diikuti peningkatan performa yang sepadan pada data validasi.

### 7.2 Akurasi Akhir pada Data Validasi

| Metrik | Nilai |
| :--- | :---: |
| Validation Loss | 1,4145 |
| **Validation Accuracy** | **52,90%** |

### 7.3 Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| Acne | 0,60 | 0,57 | 0,59 | 21 |
| Blackheads | 0,67 | 0,78 | 0,72 | 18 |
| Dark Spots | 0,64 | 0,84 | 0,73 | 19 |
| Dry Skin | 0,50 | 0,14 | 0,22 | 14 |
| Normal Skin | 0,27 | 0,17 | 0,21 | 18 |
| Oily Skin | 0,39 | 0,55 | 0,46 | 20 |
| Wrinkles | 0,52 | 0,54 | 0,53 | 28 |
| **Accuracy** | | | **0,53** | **138** |
| Macro avg | 0,51 | 0,51 | 0,49 | 138 |
| Weighted avg | 0,51 | 0,53 | 0,51 | 138 |

### 7.4 Confusion Matrix

![Confusion Matrix](image/7_confussion matrix.png)

### 7.5 Analisis Kesalahan Klasifikasi

1. **Kelas Blackheads dan Dark Spots** menunjukkan performa terbaik (F1-score 0,72 dan 0,73), menandakan pola visual keduanya cukup khas dan mudah dikenali model.
2. **Kelas Normal Skin dan Dry Skin** menunjukkan performa terendah (F1-score 0,21 dan 0,22), dengan *recall* Dry Skin hanya 0,14 — artinya sebagian besar gambar Dry Skin justru salah diklasifikasikan ke kelas lain. Hal ini konsisten dengan temuan EDA bahwa Normal Skin, Oily Skin, dan Dry Skin memiliki kemiripan visual yang tinggi.
3. **Jumlah data latih yang minim per kelas** (rata-rata ±80 gambar/kelas sebelum split) diduga menjadi kandidat utama penyebab performa yang belum optimal, ditambah ketimpangan jumlah data antar kelas yang teridentifikasi pada Bagian 4.
4. **Gejala overfitting** yang terlihat pada kurva *loss* (Bagian 7.1) turut menjadi faktor pembatas peningkatan akurasi validasi meski akurasi *training* terus naik.

### 7.6 Pengujian Prediksi Gambar Tunggal

Dilakukan uji coba *end-to-end* dengan mengunggah satu gambar kulit baru (kulit berminyak):

![Uji Prediksi Satu Gambar](images/8_prediction_result.png)

```
Top 3 Prediksi:
Oily Skin    : 80.23%
Normal Skin  : 9.00%
Dark Spots   : 6.24%

Hasil utama : Oily Skin
Confidence  : 80.23%
Status      : Prediksi cukup kuat
```

Pada kasus uji ini, model berhasil memprediksi kelas dengan benar (Oily Skin) dan tingkat keyakinan (*confidence*) yang tinggi (80,23%), menunjukkan bahwa meskipun akurasi rata-rata keseluruhan masih di kisaran 53%, model tetap mampu memberikan prediksi yang kuat pada kasus-kasus dengan pola visual yang jelas.

---

## 8. Kesimpulan dan Rekomendasi

### Kesimpulan Hasil

1. **Sistem berhasil dibangun end-to-end**, mulai dari ekstraksi dataset, eksplorasi data, augmentasi gambar, pelatihan model dengan *Transfer Learning* MobileNetV2 (dua tahap: *head training* dan *fine-tuning*), evaluasi, hingga pengujian prediksi gambar tunggal.
2. Model mencapai **akurasi validasi 52,90%** pada klasifikasi 7 kelas kondisi kulit — cukup sebagai *baseline* mengingat jumlah dataset yang relatif kecil (701 gambar untuk 7 kelas), namun masih jauh dari performa yang layak untuk digunakan sebagai alat bantu diagnosis yang andal.
3. **Performa antar kelas tidak merata**: kelas dengan pola visual khas (Blackheads, Dark Spots) diklasifikasikan dengan baik, sementara kelas dengan kemiripan visual tinggi (Normal Skin, Dry Skin, Oily Skin) sering tertukar satu sama lain.
4. Terdapat indikasi **overfitting ringan**, ditunjukkan oleh kesenjangan antara akurasi *training* yang terus meningkat dan akurasi *validation* yang stagnan di kisaran 50–55%.
5. **Keterbatasan utama:** jumlah data latih per kelas yang minim (71–141 gambar/kelas) dan distribusi kelas yang tidak seimbang menjadi kandidat penyebab utama akurasi yang belum optimal.

### Rekomendasi Pengembangan Masa Depan

- **Penambahan Data:** Memperbanyak jumlah gambar per kelas, khususnya untuk kelas dengan performa rendah (Dry Skin, Normal Skin), serta menyeimbangkan jumlah data antar kelas.
- **Regularisasi Tambahan:** Menambah *dropout*, *weight decay*, atau augmentasi yang lebih variatif untuk mengurangi gejala *overfitting* yang teramati.
- **Eksperimen Arsitektur Lain:** Membandingkan MobileNetV2 dengan arsitektur *transfer learning* lain seperti EfficientNet atau ResNet untuk melihat potensi peningkatan akurasi.
- **Class Weighting:** Menerapkan `class_weight` saat training untuk mengatasi ketidakseimbangan jumlah data antar kelas.
- **Evaluasi Klinis:** Melibatkan tenaga medis/dermatolog untuk validasi label dataset dan menilai kelayakan model sebelum digunakan sebagai alat bantu skrining nyata.
- **Deployment:** Mengintegrasikan model (`skin_disease_model.h5`) ke dalam aplikasi web SkinVision AI, lengkap dengan disclaimer bahwa hasil prediksi bersifat estimasi awal dan tidak menggantikan pemeriksaan medis profesional.

---

## 9. Referensi (APA Style)

1. agarwal, R., & Godavarthi, D. (2023). *EAI Endorsed Transactions on Pervasive Health and Technology, 9*, 1–8. https://doi.org/10.4108/eetpht.9.4039
2. Studi, P., Terapan, S., Rekayasa, T., Lunak, P., Informasi, J. T., & Bali, P. N. (2025). Aplikasi berbasis web untuk deteksi menggunakan convolutional neural network.
3. Velasco, J., Pascion, C., Alberio, J. W., Apuang, J., Cruz, J. S., Gomez, A., Molina, B. J., Tuala, L., Thio-ac, A., & Jorda, R. J. (2019). A smartphone-based skin disease classification using MobileNet CNN. *International Journal of Advanced Trends in Computer Science and Engineering*, 8(5), 3–8.
4. Watef, L., & Mahanto, F. (2024). Deteksi dan visualisasi berbasis computer vision untuk analisis gambar dermatologis dalam penilaian keparahan jerawat, 6(1), 45–53.
5. Yolov, B., & Resnet, D. A. N. (2024). *JUTEKOM: Jurnal Teknologi dan Ilmu Komputer*, 1(3), 2–9.

## Lampiran 
dataset_path = "dataset"
