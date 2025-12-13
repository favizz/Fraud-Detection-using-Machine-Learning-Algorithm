# Tugas Besar 2 IF3070 – Implementasi Algoritma Machine Learning

## 📌 Deskripsi
Repository ini berisi implementasi **algoritma Machine Learning from scratch** untuk menyelesaikan permasalahan **klasifikasi fraud** sesuai dengan spesifikasi **Tugas Besar 2 IF3070 Dasar Inteligensi Artifisial**.

Algoritma yang diimplementasikan meliputi:
- **Decision Tree Learning (DTL)**
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**

Selain implementasi from scratch, proyek ini juga mencakup **perbandingan performa dengan scikit-learn**, **evaluasi menggunakan metrik klasifikasi**, serta **Kaggle submission**.

---

## 🤖 Algoritma yang Diimplementasikan

### 1. Decision Tree Learning (DTL)
- Algoritma: **C4.5**
- Mendukung:
  - Fitur numerik
  - Fitur kategorikal
  - Missing values
- Kriteria pemilihan split:
  **ISI SENDIRI BAGIAN INI (entropy / gini / lainnya)**

### 2. Logistic Regression
- Implementasi Logistic Regression from scratch
- Optimasi parameter sesuai algoritma yang diajarkan di kelas
**ISI SENDIRI BAGIAN INI (pilih salah satu)**

#### 3. K-Nearest Neighbors (KNN)
- Implementasi KNN from scratch
- Metrik jarak:
  - Euclidean Distance
  - Manhattan Distance
  - Cosine Similarity
- Menggunakan **weighted voting** untuk klasifikasi



## 🔄 Perbandingan dengan Scikit-learn
Sebagai pembanding, model juga diimplementasikan menggunakan pustaka **scikit-learn**, yaitu:
- `DecisionTreeClassifier`
- `LogisticRegression`
- `KNeighborsClassifier` 

Hasil prediksi dari implementasi from scratch dibandingkan dengan pustaka menggunakan metrik evaluasi yang sesuai dengan permasalahan klasifikasi fraud.

---

## 🧹 Preprocessing Data
Tahapan preprocessing yang dilakukan meliputi:
- Penanganan missing values
- Penanganan data duplikat
- Encoding fitur kategorikal
- Normalisasi / scaling fitur numerik
- Penanganan imbalanced dataset
- Feature selection untuk meningkatkan performa model

---

## 📊 Evaluasi Model
Evaluasi performa model dilakukan menggunakan metrik:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

Evaluasi dilakukan pada data validasi serta melalui **Kaggle submission**.

---

## 💾 Penyimpanan Model
- Model dapat disimpan (save) dan dimuat kembali (load)
- Format penyimpanan model: .csv

---

## 🚀 Cara Menjalankan Program

### 1. Clone Repository
```bash
git clone ISI_SENDIRI_LINK_REPOSITORY
cd ISI_SENDIRI_NAMA_FOLDER



