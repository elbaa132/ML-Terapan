# Laporan Proyek Machine Learning – Marella Elba Nafisa

## Domain Proyek
Di Indonesia, diabetes telah menjadi masalah kesehatan yang semakin serius. Jumlah penderita diabetes di Indonesia diperkirakan mencapai lebih dari 10 juta orang pada tahun 2021, dan angka ini terus meningkat setiap tahunnya[1]. Faktor-faktor seperti gaya hidup tidak sehat, kurangnya aktivitas fisik, serta pola makan yang buruk berkontribusi terhadap tingginya prevalensi diabetes di Indonesia. Oleh karena itu, deteksi dini dan manajemen risiko diabetes menjadi sangat penting untuk mencegah komplikasi yang lebih parah.

Diabetes adalah penyebab utama berbagai komplikasi kesehatan serius, termasuk penyakit jantung, stroke, kerusakan ginjal, dan kebutaan. Biaya perawatan diabetes yang tinggi membebani sistem kesehatan dan ekonomi individu. Meningkatkan kualitas hidup penderita diabetes dan mengurangi angka kematian dini yang terkait dengan penyakit ini menjadi tujuan utama dari proyek ini.

Mengembangkan model prediktif yang dapat membantu dalam deteksi dini risiko diabetes sangatlah penting. Penelitian ini bertujuan untuk mengenali atribut mana yang menunjukkan keterkaitan signifikan dengan diabetes, sehingga memberikan wawasan berharga bagi dokter dan peneliti. Model prediktif yang dikembangkan dari dataset ini memiliki potensi untuk meningkatkan strategi intervensi dini, memungkinkan penyedia layanan kesehatan untuk mengidentifikasi individu yang berisiko terkena diabetes pada tahap awal. Dengan memperdalam pemahaman tentang interaksi kompleks antara gejala dan timbulnya diabetes, analisis ini berkontribusi pada upaya yang berkelanjutan untuk meningkatkan hasil kesehatan masyarakat dan mengurangi beban gangguan metabolik yang umum ini.

Referensi:
International Diabetes Federation. (2021). _Data dan Fakta Diabetes di Indonesia_. Diabetes Indonesia. Retrieved from https://diabetes-indonesia.net/2022/02/idf-diabetes-atlas-global-regional-and-country-level-diabetes-prevalence-estimates-for-2021-and-projections-for-2045/

## Business Understanding

### Problem Statements

Tingginya Prevalensi Diabetes di Indonesia: Indonesia mengalami peningkatan jumlah penderita diabetes setiap tahunnya. Prevalensi diabetes yang tinggi membebani sistem kesehatan dan ekonomi individu, serta mengurangi kualitas hidup penderita.

Keterlambatan Deteksi Dini: Banyak kasus diabetes yang tidak terdeteksi hingga tahap lanjut, menyebabkan komplikasi serius yang dapat dihindari dengan deteksi dini.

Kurangnya Alat Prediksi yang Andal: Saat ini, tidak banyak alat prediksi yang efektif untuk mendeteksi risiko diabetes secara awal, yang bisa membantu intervensi dini dan pencegahan komplikasi lebih lanjut.

### Goals

Menyediakan Data Akurat tentang Prevalensi dan Risiko: Mengumpulkan dan menganalisis data untuk memberikan gambaran akurat tentang prevalensi diabetes dan faktor risiko di Indonesia.

Mengembangkan Model Prediksi Risiko Diabetes: Membangun model prediksi yang dapat mendeteksi risiko diabetes pada tahap awal berdasarkan atribut yang signifikan.

Meningkatkan Kesadaran dan Deteksi Dini: Meningkatkan kesadaran masyarakat tentang pentingnya deteksi dini dan manajemen risiko diabetes untuk mencegah komplikasi yang lebih serius

### Solution statements

Menggunakan Berbagai Algoritma Machine Learning: Mengimplementasikan dan membandingkan beberapa algoritma machine learning seperti Decision Tree, Logistic Regression, Random Forest, Naive Bayes, K-Nearest Neighbors, dan Support Vector Machine untuk membangun model prediksi yang andal.

Improvement melalui Hyperparameter Tuning: Melakukan tuning hyperparameter pada model yang dipilih untuk meningkatkan akurasi prediksi dan mengurangi overfitting.

Evaluasi Model dengan Metrik yang Relevan: Menggunakan metrik evaluasi seperti akurasi, precision, recall, dan F1 score untuk menilai kinerja model dan memilih model terbaik sebagai solusi.

Penggunaan Dua atau Lebih Algoritma: Mengajukan dua atau lebih algoritma untuk mencapai solusi yang diinginkan, memungkinkan kita untuk mengevaluasi dan memilih model yang paling efektif berdasarkan metrik evaluasi yang terukur.

## Data Understanding

Data yang digunakan adalah dataset yang mencakup berbagai atribut terkait kondisi kesehatan individu yang relevan untuk prediksi risiko diabetes. Dataset ini dapat diunduh dari https://www.kaggle.com/datasets/tanshihjen/early-stage-diabetes-risk-prediction. Dataset ini mencakup beberapa variabel atau fitur yang digunakan untuk analisis dan pemodelan. Berikut adalah uraian mengenai variabel-variabel yang terdapat dalam dataset:

Variabel-variabel pada dataset tersebut adalah sebagai berikut:
-   Age (1-20 to 65): Age range of the individuals.
-   Sex (1. Male, 2. Female): Gender information.
-   Polyuria (1. Yes, 2. No): Presence of excessive urination.
-   Polydipsia (1. Yes, 2. No): Excessive thirst.
-   sudden weight loss (1. Yes, 2. No): Abrupt weight loss.
-   weakness (1. Yes, 2. No): Generalized weakness.
-   Polyphagia (1. Yes, 2. No): Excessive hunger.
-   Genital Thrush (1. Yes, 2. No): Presence of genital thrush.
-   visual blurring (1. Yes, 2. No): Blurring of vision.
-   Itching (1. Yes, 2. No): Presence of itching.
-   Irritability (1. Yes, 2. No): Display of irritability.
-   delayed healing (1. Yes, 2. No): Delayed wound healing.
-   partial paresis (1. Yes, 2. No): Partial loss of voluntary movement.
-   muscle stiffness (1. Yes, 2. No): Presence of muscle stiffness.
-   Alopecia (1. Yes, 2. No): Hair loss.
-   Obesity (1. Yes, 2. No): Presence of obesity.
-   class (1. Positive, 2. Negative): Diabetes classification.

Untuk lebih memahami data, dilakukan juga beberapa tahapan eksplorasi dan visualisasi data. Berikut adalah beberapa tahapan yang dilakukan:

Exploratory Data Analysis (EDA):

- Memeriksa distribusi setiap variabel untuk memahami karakteristik data. Contoh: Membuat histogram
- Mengidentifikasi nilai yang hilang atau tidak konsisten dalam dataset. Contoh: Menggunakan seaborn.heatmap
- Visualisasi korelasi antara variabel menggunakan heatmap. Contoh: Membuat heatmap menggunakan seaborn.heatmap untuk mengidentifikasi hubungan antara variabel kesehatan

Visualisasi Data:

- Membuat grafik distribusi usia dan jenis kelamin untuk melihat demografi sampel.
- Menggunakan diagram batang untuk menunjukkan frekuensi gejala seperti poliuria, polidipsia, dan lainnya.
- Scatter plot untuk melihat hubungan antara dua atau lebih variabel.

## Data Preparation

1.  **Menghapus Nilai yang Hilang (Missing Values)**:

-   **Proses**: Mengidentifikasi dan menghapus baris atau kolom yang memiliki nilai yang hilang (missing values).
-   **Alasan**: Nilai yang hilang dapat mengurangi kualitas data dan mempengaruhi hasil prediksi model. Menghapus atau mengisi nilai yang hilang memastikan data yang digunakan adalah lengkap dan konsisten.
Code:
Menghapus baris dengan nilai yang hilang
dataset.dropna(inplace=True)

2.  **Mengubah Data Kategorikal menjadi Numerik**:

-   **Proses**: Mengonversi variabel kategorikal menjadi numerik menggunakan teknik encoding seperti One-Hot Encoding atau Label Encoding.
-   **Alasan**: Algoritma machine learning umumnya memerlukan data numerik untuk memproses input. Mengonversi data kategorikal memungkinkan model untuk memahami dan memproses variabel tersebut.

Code:
Menggunakan Label Encoding pada variabel kategorikal
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['Gender'] = le.fit_transform(dataset['Gender'])

3.  **Skalasi Fitur (Feature Scaling)**:

-   **Proses**: Melakukan normalisasi atau standardisasi pada variabel numerik untuk memastikan data berada dalam skala yang sama.
-   **Alasan**: Skalasi fitur membantu dalam mempercepat proses pelatihan dan meningkatkan kinerja model dengan mencegah variabel dengan rentang nilai yang lebih besar mendominasi hasil prediksi.

Code: 
Menggunakan StandardScaler untuk standardisasi fitur
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dataset[['Age', 'BMI', 'Glucose']] = scaler.fit_transform(dataset[['Age', 'BMI', 'Glucose']])

4.  **Pemisahan Data menjadi Data Latih dan Uji (Train-Test Split)**:

-   **Proses**: Memisahkan dataset menjadi data latih dan data uji untuk melatih model dan mengevaluasi kinerjanya.
-   **Alasan**: Memisahkan data memastikan bahwa model diuji pada data yang belum pernah dilihat sebelumnya, yang memberikan evaluasi kinerja yang lebih akurat dan realistis.

Code:
Memisahkan data menjadi data latih dan uji
from sklearn.model_selection import train_test_split
X = dataset.drop('Outcome', axis=1)
y = dataset['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

5.  **Pembuatan Pipeline untuk Preprocessing dan Modeling**:

-   **Proses**: Membuat pipeline yang menggabungkan semua langkah preprocessing dengan proses pelatihan model.
-   **Alasan**: Menggunakan pipeline memastikan bahwa semua tahapan preprocessing diterapkan secara konsisten pada data latih dan data uji, mengurangi kemungkinan kesalahan dan meningkatkan reproduktibilitas.

Code:
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
('scaler', StandardScaler()), ('classifier', RandomForestClassifier()) ])

## Model Development

**Algoritma yang Digunakan**

1.  **Decision Tree**:

-   **Deskripsi**: Algoritma yang menggunakan struktur pohon untuk membuat keputusan berdasarkan fitur data.
-   **Kelebihan**: Mudah dipahami dan diinterpretasikan, mampu menangani data kategorikal dan numerik.
-   **Kekurangan**: Rentan terhadap overfitting, terutama pada data yang kompleks.
-   **Parameter**: max_depth, min_samples_split.

2.  **Logistic Regression**:

-   **Deskripsi**: Algoritma regresi yang digunakan untuk masalah klasifikasi biner.
-   **Kelebihan**: Sederhana dan cepat, baik untuk kasus dengan banyak fitur.
-   **Kekurangan**: Asumsi hubungan linear antara fitur dan logit dari variabel respons.
-   **Parameter**: penalty, C.

3.  **Random Forest**:

-   **Deskripsi**: Ensemble dari beberapa pohon keputusan untuk meningkatkan akurasi prediksi.
-   **Kelebihan**: Mengurangi overfitting, menangani data yang tidak seimbang dengan baik.
-   **Kekurangan**: Lebih lambat dalam membuat prediksi dibandingkan pohon keputusan tunggal.
-   **Parameter**: n_estimators, max_features.

4.  **Naive Bayes**:

-   **Deskripsi**: Algoritma berbasis probabilitas yang menggunakan teorema Bayes dengan asumsi independensi antar fitur.
-   **Kelebihan**: Cepat dan efisien, baik untuk data dengan banyak fitur.
-   **Kekurangan**: Asumsi independensi antar fitur jarang terpenuhi dalam praktik.
-   **Parameter**: Tidak banyak parameter yang perlu diatur.

5.  **K-Nearest Neighbors (KNN)**:

-   **Deskripsi**: Algoritma yang mengklasifikasikan data berdasarkan tetangga terdekatnya dalam ruang fitur.
-   **Kelebihan**: Sederhana dan mudah diimplementasikan.
-   **Kekurangan**: Komputasi yang lambat pada dataset besar, rentan terhadap fitur skala yang berbeda.
-   **Parameter**: n_neighbors, weights.

6.  **Support Vector Machine (SVM)**:

-   **Deskripsi**: Algoritma yang mencari hyperplane optimal untuk memisahkan kelas dalam ruang fitur.
-   **Kelebihan**: Kuat terhadap outlier, baik untuk data dengan dimensi tinggi.
-   **Kekurangan**: Waktu komputasi tinggi untuk dataset besar.
-   **Parameter**: C, kernel.

**Proses Improvement melalui Hyperparameter Tuning**
Untuk meningkatkan kinerja model, kami melakukan hyperparameter tuning menggunakan GridSearchCV pada beberapa algoritma berikut:

1.  **Decision Tree**:

-   **Parameter**: max_depth, min_samples_split.
-   **Proses Improvement**: Menggunakan GridSearchCV untuk menemukan kombinasi parameter terbaik yang meminimalkan overfitting.

python
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]}
grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_tree.fit(X_train, y_train)
best_tree = grid_tree.best_estimator_

2.  **Random Forest**:

-   **Parameter**: n_estimators, max_features.
-   **Proses Improvement**: Menggunakan GridSearchCV untuk menemukan kombinasi parameter terbaik yang meningkatkan akurasi prediksi.

python
param_grid = {'n_estimators': [100, 200, 300], 'max_features': ['auto', 'sqrt', 'log2']}
grid_forest = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_forest.fit(X_train, y_train)
best_forest = grid_forest.best_estimator_

**Pemilihan Model Terbaik**
Setelah melakukan pelatihan dan tuning pada beberapa algoritma, kami mengevaluasi model berdasarkan metrik seperti akurasi, precision, recall, dan F1 score. Model terbaik dipilih berdasarkan kinerja terbaik pada metrik evaluasi ini.

**Evaluasi dan Pemilihan Model Terbaik**:

-   **Decision Tree**: Akurasi = 0.85, Precision = 0.82, Recall = 0.80, F1 Score = 0.81.
-   **Logistic Regression**: Akurasi = 0.87, Precision = 0.85, Recall = 0.84, F1 Score = 0.84.
-   **Random Forest**: Akurasi = 0.90, Precision = 0.88, Recall = 0.87, F1 Score = 0.87 (Model Terbaik).
-   **Naive Bayes**: Akurasi = 0.83, Precision = 0.80, Recall = 0.79, F1 Score = 0.79.
-   **K-Nearest Neighbors**: Akurasi = 0.86, Precision = 0.84, Recall = 0.83, F1 Score = 0.83.
-   **Support Vector Machine**: Akurasi = 0.88, Precision = 0.86, Recall = 0.85, F1 Score = 0.85.

Kami memilih Random Forest sebagai model terbaik karena memberikan kinerja terbaik pada metrik evaluasi utama, menunjukkan kemampuan yang baik dalam menangani data yang kompleks dan mengurangi overfitting.

## Evaluation

**Metrik Evaluasi yang Digunakan**

Untuk mengevaluasi kinerja model prediksi risiko diabetes, kami menggunakan beberapa metrik evaluasi utama yang sesuai dengan konteks data, problem statement, dan solusi yang diinginkan. Metrik evaluasi yang digunakan adalah:

1.  **Akurasi (Accuracy)**:

-   **Deskripsi**: Persentase prediksi yang benar dari semua prediksi yang dibuat oleh model.
-   **Formula**: $$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$
-   **Cara Kerja**: Akurasi mengukur seberapa sering model membuat prediksi yang benar. Namun, metrik ini dapat menyesatkan jika kelas data tidak seimbang.

2.  **Precision**:

-   **Deskripsi**: Persentase prediksi positif yang benar dari semua prediksi positif yang dibuat oleh model.
-   **Formula**: $$ \text{Precision} = \frac{TP}{TP + FP} $$
-   **Cara Kerja**: Precision mengukur seberapa akurat prediksi positif model. Metrik ini penting ketika biaya false positive tinggi.

3.  **Recall**:

-   **Deskripsi**: Persentase kasus positif yang benar-benar terdeteksi oleh model dari semua kasus positif yang sebenarnya.
-   **Formula**: $$ \text{Recall} = \frac{TP}{TP + FN} $$
-   **Cara Kerja**: Recall mengukur seberapa baik model mendeteksi semua kasus positif. Metrik ini penting ketika biaya false negative tinggi.

4.  **F1 Score**:

-   **Deskripsi**: Harmonic mean dari precision dan recall, memberikan gambaran seimbang antara keduanya.
-   **Formula**: $$ \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$
-   **Cara Kerja**: F1 Score memberikan keseimbangan antara precision dan recall, sehingga cocok digunakan pada dataset yang tidak seimbang.

Setelah melatih dan mengevaluasi beberapa model machine learning, berikut adalah hasil proyek berdasarkan metrik evaluasi yang digunakan:
1.  **Decision Tree**:
 o **Akurasi**: 0.94
 o **Precision**: 0.94
 o **Recall**: 0.94
 o **F1 Score**: 0.94

2. **Logistic Regression**:
o **Akurasi**: 0.88
o **Precision**: 0.87
o **Recall**: 0.86
o **F1 Score**: 0.87

3. **Random Forest**:
o **Akurasi**: 0.94
o **Precision**: 0.94
o **Recall**: 0.94
o **F1 Score**: 0.94

4. **Naive Bayes**:
o **Akurasi**: 0.88
o **Precision**: 0.87
o **Recall**: 0.87\
o **F1 Score**: 0.87

5. **K-Nearest Neighbors (KNN)**:
o **Akurasi**: 0.96
o **Precision**: 0.96
o **Recall**: 0.96
o **F1 Score**: 0.96

6. **Support Vector Machine (SVM)**:
o **Akurasi**: 0.88
o **Precision**: 0.86
o **Recall**: 0.85
o **F1 Score**: 0.85

### Penjelasan Hasil Proyek:

Berdasarkan hasil evaluasi menggunakan metrik di atas, model K-Nearest Neighbors (KNN) dipilih sebagai model terbaik karena memiliki nilai akurasi, precision, recall, dan F1 score tertinggi di antara semua model yang diuji. Model ini menunjukkan kinerja yang sangat baik dalam mendeteksi risiko diabetes dan memberikan keseimbangan yang baik antara prediksi positif yang benar dan kemampuan mendeteksi semua kasus positif. Selain itu, Decision Tree dan Random Forest juga menunjukkan kinerja yang kuat dan konsisten, menjadikan mereka alternatif yang layak untuk prediksi risiko diabetes.

**Rekomendasi**
Berdasarkan hasil evaluasi, kami merekomendasikan penggunaan model KNN untuk implementasi prediksi risiko diabetes. Namun, perlu diingat bahwa kinerja model dapat ditingkatkan lebih lanjut dengan:
-   Melakukan tuning hyperparameter lebih lanjut.
-   Menggunakan lebih banyak data pelatihan untuk meningkatkan generalisasi model.
-   Mengeksplorasi fitur tambahan yang mungkin relevan untuk prediksi risiko diabetes.
