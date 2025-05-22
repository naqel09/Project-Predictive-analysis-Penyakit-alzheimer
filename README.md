# Laporan Proyek Machine Learning - Andry Septian Syahputra Tumaruk
---
## Domain Proyek
Proyek ini berada dalam domain kesehatan, dengan fokus utama pada deteksi dini penyakit Alzheimer menggunakan pendekatan Machine Learning. Alzheimer adalah penyakit neurodegeneratif yang menyebabkan gangguan memori, kemampuan berpikir, dan perilaku. Seiring perkembangan penyakit, penderita menjadi tidak mampu melakukan aktivitas sehari-hari secara mandiri, yang berdampak besar pada kualitas hidup penderita dan beban keluarga serta tenaga medis.

**Mengapa Masalah Ini Harus Diselesaikan**

Menurut laporan WHO, jumlah penderita demensia secara global diperkirakan mencapai lebih dari 55 juta orang, dan angka ini diperkirakan akan meningkat menjadi 139 juta pada tahun 2050 jika tidak ada intervensi yang signifikan (World Health Organization, 2021). Alzheimer menyumbang hingga 60–70% dari seluruh kasus demensia, menjadikannya salah satu prioritas dalam bidang kesehatan global.

Masalah ini harus diselesaikan karena:
- **Beban ekonomi dan sosial**: Alzheimer memerlukan perawatan jangka panjang yang mahal. Pada tahun 2020, total biaya global akibat demensia mencapai USD 1 triliun dan akan meningkat drastis (Alzheimer’s Disease International, 2020).
- **Deteksi dini dapat memperlambat progresi penyakit**: Terapi atau perubahan gaya hidup bisa lebih efektif jika dilakukan pada tahap awal penyakit (Livingston et al., 2020).
- **Keterbatasan diagnosis konvensional**: Saat ini, diagnosis Alzheimer seringkali mengandalkan tes kognitif dan pencitraan otak yang mahal dan tidak selalu tersedia di fasilitas layanan primer. Machine Learning dapat menawarkan solusi non-invasif, efisien, dan terjangkau.

**Bagaimana Masalah Ini Dapat Diselesaikan**

Masalah ini dapat didekati melalui penerapan Machine Learning (ML) yang mampu menganalisis data klinis pasien secara efisien untuk mendeteksi pola yang berkaitan dengan Alzheimer. ML mampu:
- Mengidentifikasi fitur penting (misal usia, pola tidur, aktivitas fisik) yang berkaitan dengan risiko Alzheimer.
- Mengklasifikasikan pasien ke dalam kategori berisiko tinggi atau rendah.Membantu dokter dalam pengambilan keputusan berbasis data (data-driven decision making).
- Melalui model klasifikasi berbasis ML yang dibangun dan dievaluasi secara akurat, kita dapat menciptakan sistem deteksi awal yang dapat digunakan secara luas, bahkan di fasilitas layanan kesehatan dasar.

**Referensi:**
- [Alzheimer’s Disease International. (2020). World Alzheimer Report 2020: Design, dignity, dementia.](https://www.alzint.org/u/WorldAlzheimerReport2020Vol1.pdf)
- [World Health Organization (WHO). (2021). Global status report on the public health response to dementia.](https://www.who.int/publications/i/item/9789240033245)
---
## Business Understanding
### Problem Statements
- Bagaimana cara mengklasifikasikan apakah seorang pasien memiliki penyakit Alzheimer berdasarkan data medis?
- Fitur atau variabel apa saja yang paling berpengaruh terhadap diagnosis Alzheimer?

### Goals
- Membangun model klasifikasi yang dapat memprediksi diagnosis Alzheimer dengan akurasi yang tinggi.
- Mengidentifikasi fitur penting yang dapat menjadi indikator utama terhadap kemungkinan pasien terkena Alzheimer.

### Solution Statements
1. Menerapkan dan membandingkan beberapa algoritma machine learning seperti SVC, Random Forest, XGBoost, dan MLP untuk memprediksi diagnosis Alzheimer.
2. Melakukan perbaikan performa model dengan feature selection dan hyperparameter tuning.
3. Metrik yang digunakan adalah akurasi, precision, recall, dan F1-score untuk mengukur kinerja masing-masing model.
---
## Data Understanding
### URL/tautan sumber data
Dataset digunakan berisi data pasien lansia dan berbagai atribut klinis serta gaya hidup. Dataset ini dimuat dalam format CSV dan datasetnya diambil dari sumber kaggle: [**Link Alzheimer Dataset**](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)

### Variabel-variabel:
- **`Age`**: Umur pasien
- **`Gender`**: Jenis kelamin
- **`Education Level`**: Tingkat pendidikan
- **`BMI`**: Indeks massa tubuh
- **`Smoking Status`**: Status merokok
- **`Alcohol Consumption`**: Konsumsi alkohol
- **`Sleep Quality`**: Kualitas tidur
- **`Physical Activity`**: Aktivitas fisik
- **`Diet`**: Pola makan
- **`Comorbidities`**: Penyakit penyerta
- **`Cognitive Decline`**: Tingkat penurunan kognitif
- **`Independence Level`**: Tingkat kemandirian
- **`Diagnosis`**: Label target (Alzheimer / Tidak)

---
### Exploratory Data Analysis (EDA)
- Mayoritas pasien berusia 60–90 tahun.
- Perbandingan gender cukup seimbang.
- Sebagian besar pasien memiliki pendidikan menengah ke atas.
- Rata-rata BMI menunjukkan pasien dalam kategori overweight.
- Sebagian besar bukan perokok dan memiliki pola tidur yang baik.
- Diagnosis Alzheimer ditemukan pada sekitar 33% pasien.
---
## Data Preparation
Langkah-langkah persiapan data meliputi:
1. **Menghapus kolom irrelevan**: Misalnya ID pasien atau nama dokter yang tidak relevan dengan prediksi.
2. **Menangani missing values**: Tidak ditemukan missing value pada dataset.
3. **Menghapus data duplikat**: Dataset telah dicek dan tidak ditemukan duplikasi.
4. **Univariate & Multivariate Analysis**: Distribusi variabel dan korelasi antar fitur dianalisis untuk mengidentifikasi relasi penting.
---
## Modeling
Beberapa model klasifikasi yang digunakan adalah:
- **SVC (Support Vector Classifier)**
- **Random Forest**
- **XGBoost**
- **MLP (Multilayer Perceptron)**

Masing-masing model di-fit dengan data yang telah dibersihkan dan di-preprocess. Model dilatih menggunakan data training dan dievaluasi menggunakan data testing dengan metrik:
Accuracy
- **Precision**
- **Recall**
- **F1-score**

Kelebihan dan Kekurangan Model:
- **SVC**: Baik untuk data dengan margin yang jelas, tetapi performanya rendah pada dataset ini.
- **Random Forest**: Model ensemble yang kuat dan dapat menangani data non-linear.
- **XGBoost**: Performa terbaik, efisien, dan sangat akurat.
- **MLP**: Kurang akurat pada dataset ini, mungkin karena kompleksitas model dan kebutuhan tuning yang lebih tinggi.
---
## Evaluation
Berikut hasil evaluasi performa model (lihat tabel visualisasi sebelumnya):
| Model        | Akurasi  | Precision | Recall   | F1-score |
| ------------ | -------- | --------- | -------- | -------- |
| SVC          | 0.820930 | 0.775362  | 0.699346 | 0.735395 |
| RandomForest | 0.941860 | 0.957143  | 0.875817 | 0.914676 |
| XGBoost      | 0.953488 | 0.952381  | 0.915033 | 0.933333 |
| MLP          | 0.804651 | 0.741259  | 0.692810 | 0.716216 |

## Analisis:
- **XGBoost** menunjukkan performa tertinggi secara keseluruhan (F1-score tertinggi: 93.33%).
- **RandomForest** memiliki precision tertinggi, cocok untuk kasus dengan fokus pada minimisasi false positive.
- **MLP** dan **SVC** memiliki performa yang kurang optimal pada dataset ini.

## Formula Evaluasi:
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1-score** = 2 * (Precision * Recall) / (Precision + Recall)
---
## Kesimpulan
Model XGBoost dipilih sebagai model terbaik karena memiliki **akurasinya tinggi**, **recall yang baik**, serta **f1-score terbaik**, yang cocok untuk klasifikasi diagnosis medis seperti Alzheimer. Model ini dapat digunakan sebagai sistem pendukung keputusan dalam diagnosa awal penyakit Alzheimer.
