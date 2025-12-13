# Tugas Besar 2 IF3070 – Implementasi Algoritma Machine Learning

## Deskripsi
Repository ini berisi implementasi **algoritma Machine Learning from scratch** untuk menyelesaikan permasalahan **klasifikasi fraud** sesuai dengan spesifikasi **Tugas Besar 2 IF3070 Dasar Inteligensi Artifisial**.

Algoritma yang diimplementasikan meliputi:
- **Decision Tree Learning (DTL)**
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**

Selain implementasi from scratch, proyek ini juga mencakup **perbandingan performa dengan scikit-learn**, **evaluasi menggunakan metrik klasifikasi**, serta **Kaggle submission**.

---

## Algoritma yang Diimplementasikan

### 1. Decision Tree Learning (DTL)
- Algoritma: **C4.5**
- Mendukung:
  - Fitur numerik
  - Fitur kategorikal
  - Missing values
- Kriteria pemilihan split:
Pemilihan atribut dan threshold dilakukan dengan menghitung information gain yang merupakan selisih antara entropy parent node dan weighted entropy setelah data dibagi menjadi subset kiri dan subset kanan. Split dilakukan dengan memilih atribut yang memiliki nilai information gain terbesar.

### 2. Logistic Regression
- Implementasi Logistic Regression from scratch
- Optimasi yang dilakukan adalah learning rate dipertahankan pada 0.01 sebagai sweet spot untuk konvergensi stabil, jumlah iterasi dioptimalkan menjadi 8000 untuk memastikan cukup pembelajaran tanpa overfitting, dan lambda regularisasi diset 0.02 untuk balans optimal antara bias dan variance. Konfigurasi terbaik juga mengaktifkan class weighting untuk mengatasi imbalance data dan menggunakan undersampling ratio 2.0, yang secara signifikan meningkatkan performa model dalam mendeteksi fraud

### 3. K-Nearest Neighbors (KNN)
- Implementasi KNN from scratch
- Metrik jarak:
  - Euclidean Distance: 
  - Manhattan Distance
  - Cosine Similarity
- Menggunakan **weighted voting** untuk klasifikasi sehingga neighbor yang lebih dekat memiliki pengaruh lebih besar terhadap hasil klasifikasi, sehingga prediksi menjadi lebih stabil dan representatif.

## Perbandingan dengan Scikit-learn
Sebagai pembanding, model juga diimplementasikan menggunakan pustaka **scikit-learn**, yaitu:
- `DecisionTreeClassifier`
- `LogisticRegression`
- `KNeighborsClassifier` 

Hasil prediksi dari implementasi from scratch dibandingkan dengan pustaka menggunakan metrik evaluasi yang sesuai dengan permasalahan klasifikasi fraud.

---

## Preprocessing Data
Tahapan preprocessing yang dilakukan meliputi:
- Penanganan missing values
- Penanganan data duplikat
- Encoding fitur kategorikal
- Normalisasi / scaling fitur numerik
- Penanganan imbalanced dataset
- Feature selection untuk meningkatkan performa model

---

## Evaluasi Model
Evaluasi performa model dilakukan menggunakan metrik:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

Evaluasi dilakukan pada data validasi serta melalui **Kaggle submission**.

---

## Penyimpanan Model
- Model dapat disimpan (save) dan dimuat kembali (load)
- Format penyimpanan model: .csv

---

## Cara Menjalankan Program

```bash
git clone https://github.com/IlllIllIIIlIlIll/Fraud-Detection-using-Machine-Learning-Algorithm.git
```

## 👥 Tim Pengembang - Welfarella (Kelompok 31)

| NIM | Nama | Kontribusi |
|-----|------|------------|
| 18222070 | Favian Izza Diasputra | Feature encoding, feature scaling, data normalization, exploratory data analysis, compile preprocessing pipeline, submissions, dan algoritma DTL, logistic regression, KNN dengan menggunakan scikit-learn |
| 18223008 | Henrycus Hugatama Risaldy | Feature engineering, dimensionality reduction, dan implementasi algoritma logistic regression from scratch |
| 18223046 | Farella Kamala Budianto | Split training, validation, and testing set, handling missing data, exploratory data analysis, dan implementasi algoritma K-Nearest Neighbors from scratch |
| 18223098 | Kenlyn Tesalonika Winata | Dealing with outliers, remove duplicates, handling imbalance dataset, dan implementasi algoritma DTL from scratch |
