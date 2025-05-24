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

### Dataset Awal
- Jumlah Entri: 16.714 baris
- Jumlah Kolom: 35 (10 fitur numerik + 1 target)
- Tidak Ada Nilai Kosong: Semua kolom lengkap tanpa missing values
- Duplikat: semua kolom tidak memiliki duplikat data
- Outlier: Semua fitur tidak memiliki outlier
- Tipe Data:
- float64: 12 kolom (semua fitur)
- int64 : 12 kolom (11 fitur + 1target)
- object: 1 kolom (DoctorInCharge)
### Variabel-variabel:
- **`PatientID`**: Identifikasi unik yang diberikan kepada setiap pasien (4751 hingga 6900).
- **`Age`**: Umur pasien
- **`Gender`**: Jenis kelamin
- **`Education Level`**: Tingkat pendidikan
- **`BMI`**: Indeks massa tubuh
- **`Smoking Status`**: Status merokok
- **`Alcohol Consumption`**: Konsumsi alkohol
- **`Sleep Quality`**: Kualitas tidur
- **`Physical Activity`**: Aktivitas fisik
- **`FamilyHistoryAlzheimers`**: Riwayat keluarga terhadap penyakit Alzheimer, di mana 0 berarti Tidak dan 1 berarti Ya.
- **`CardiovascularDisease`**: Kehadiran penyakit kardiovaskular, 0 berarti Tidak dan 1 berarti Ya.
- **`Diabetes`**: Kehadiran penyakit diabetes, 0 berarti Tidak dan 1 berarti Ya.
- **`Depression`**: Kehadiran depresi, 0 berarti Tidak dan 1 berarti Ya.
- **`HeadInjury`**: Riwayat cedera kepala, 0 berarti Tidak dan 1 berarti Ya.
- **`Hypertension`**: Kehadiran hipertensi, 0 berarti Tidak dan 1 berarti Ya.
- **`SystolicBP`**: Tekanan darah sistolik, berkisar antara 90 hingga 180 mmHg.
- **`DiastolicBP`**: Tekanan darah diastolik, berkisar antara 60 hingga 120 mmHg.
- **`CholesterolTotal`**: Kadar kolesterol total, berkisar antara 150 hingga 300 mg/dL.
- **`CholesterolLDL`**: Kadar kolesterol LDL (low-density lipoprotein), berkisar antara 50 hingga 200 mg/dL.
- **`CholesterolHDL`**: Kadar kolesterol HDL (high-density lipoprotein), berkisar antara 20 hingga 100 mg/dL.
- **`CholesterolTriglycerides`**: Kadar trigliserida, berkisar antara 50 hingga 400 mg/dL.
- **`MMSE`**: Skor Pemeriksaan Mental Mini (Mini-Mental State Examination), berkisar antara 0 hingga 30. Skor lebih rendah menunjukkan gangguan kognitif.
- **`FunctionalAssessment`**: Skor penilaian fungsional, berkisar antara 0 hingga 10. Skor lebih rendah menunjukkan gangguan yang lebih berat.
- **`MemoryComplaints`**: Keluhan terhadap daya ingat, 0 berarti Tidak dan 1 berarti Ya.
- **`BehavioralProblems`**: Masalah perilaku, 0 berarti Tidak dan 1 berarti Ya.
- **`ADL`**: Skor aktivitas kehidupan sehari-hari (Activities of Daily Living), berkisar antara 0 hingga 10. Skor lebih rendah menunjukkan gangguan yang lebih berat.
- **`Confusion`**: Kehadiran kebingungan, 0 berarti Tidak dan 1 berarti Ya.
- **`Disorientation`**: Kehadiran disorientasi, 0 berarti Tidak dan 1 berarti Ya.
- **`PersonalityChanges`**: Perubahan kepribadian, 0 berarti Tidak dan 1 berarti Ya.
- **`DifficultyCompletingTasks`**: Kesulitan dalam menyelesaikan tugas, 0 berarti Tidak dan 1 berarti Ya.
- **`Forgetfulness`**: Pelupa, 0 berarti Tidak dan 1 berarti Ya.
- **`Diagnosis`**: Status diagnosis penyakit Alzheimer, 0 berarti Tidak dan 1 berarti Ya.
- **`DoctorInCharge`**: Kolom ini berisi informasi rahasia tentang dokter yang menangani, dengan nilai "XXXConfid" untuk semua pasien.
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
### Model 1: Support Vector Machine (SVM)
- **Cara Kerja**: Support Vector Machine bekerja dengan mencari hyperplane terbaik yang memisahkan dua kelas data dengan margin maksimum. Dalam kasus dengan kernel linear, model berusaha mencari garis lurus yang memisahkan data pada ruang berdimensi tinggi.
- **Parameter**:
 - `C=1.0`: Parameter regularisasi default yang mengontrol trade-off antara margin maksimum dan kesalahan klasifikasi. Nilai lebih kecil memberikan margin lebih besar dengan toleransi error lebih tinggi.
 - `kernel='linear'`: Kernel default yang digunakan dalam notebook. Kernel linear cocok digunakan saat data relatif linier terpisah.
 - `gamma=10`: Walau disebutkan, gamma tidak relevan untuk kernel linear, sehingga tidak berpengaruh dalam model ini.

### **Model 2: Random Forest**
- **Cara Kerja:** Random Forest adalah metode ensemble learning yang membangun banyak decision tree dari subset acak data dan fitur, kemudian menggabungkan hasil prediksi dari seluruh tree melalui voting untuk klasifikasi.
- **Parameter:**
 - `random_state=42`: Parameter untuk memastikan hasil reproduktif. Parameter lain menggunakan default:
 - `n_estimators=100`: Jumlah pohon dalam hutan.
 - `max_depth=None`: Tanpa batasan kedalaman pohon.
 - `min_samples_split=2`,
 - `min_samples_leaf=1`,
 - `bootstrap=True`.

### **Model 3: XGBoost**
- **Cara Kerja**: XGBoost adalah algoritma boosting berbasis pohon yang membangun model secara bertahap dan mengoptimalkan kesalahan dari model sebelumnya. Ini menghasilkan model yang sangat akurat dan efisien.
- **Parameter:**
 - `use_label_encoder=False`: Untuk menghindari penggunaan encoder label bawaan XGBoost yang deprecated.
 -`eval_metric='logloss'`: Metrik evaluasi default untuk klasifikasi biner.
 - `n_estimators=100`,
 - `learning_rate=0.1`,
 - `max_depth=3`,
 - `subsample=1`,
 - `colsample_bytree=1`.

### **Model 4: Neural Network (MLPClassifier)**
- **Cara Kerja:** Multi-layer Perceptron adalah jaringan saraf tiruan yang terdiri dari lapisan input, hidden layer, dan output layer. Model ini belajar memetakan fitur ke target melalui fungsi aktivasi non-linear.

- **Parameter:**
 - `hidden_layer_sizes=(100,50,50)`: Arsitektur jaringan terdiri dari tiga hidden layer dengan masing-masing 100, 50, dan 50 neuron.
 - `max_iter=1000`: Batas maksimum iterasi pelatihan.
 - `activation='relu'`: Fungsi aktivasi Rectified Linear Unit.
 - `solver='adam'`: Optimizer berbasis stochastic gradient descent yang efisien.
 - `learning_rate_init=0.001`: Learning rate awal.
 - `alpha=0.0001`,
 - `batch_size='auto'`,
 - `early_stopping=False`.
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
