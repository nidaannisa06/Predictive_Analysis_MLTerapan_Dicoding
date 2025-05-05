
# Laporan Proyek Machine Learning - Nida Annisa Sholeha

## Domain Proyek
Sustainable Development Goals (SDGS) atau yang dikenal sebagai tujuan pembangunan berkelanjutan adalah program yang ditetapkan oleh Perserikatan Bangsa-Bangsa dengan tujuan meningkatkan kesejahteraan masyarakat. Program ini terdiri dari 17 tujuan yang saling terkait dengan 169 target untuk mencapai pembangunan berkelanjutan pada tahun 2030 (United Nations, n.d.). Salah satu yang menjadi perhatian pada program ini adalah perubahan iklim yang tercantum pada tujuan ke-13 (Katila et al., 2019). Dampak yang terjadi akibat perubahan iklim dapat mempengaruhi pertanian, peternakan dan ketersediaan air (Kabir et al. 2023). Selain hal tersebut, terjadinya perubahan iklim juga dapat berpengaruh pada kesehatan manusia (Rocque et al., 2021). 

Menurut IPCC (Intergovernmental Panel on Climate Change, 2014) penyumbang gas emisi rumah kaca terbesar bersumber dari emisi emisi CO2. Gas karbon menjadi penyumbang utama dalam pemanasan global dan menjadi penyebab utama perubahan iklim yang disebabkan oleh manusia (Sampaio et al., 2025). Sekitar 90% dari total emisi karbon berasal dari pembakaran bahan bakar fosil (IEA, 2015). Penggunaan transportasi kendaraan bermotor yang menggunakan bahan bakar minyak sebagai sumber energi turut menjadi penyebab peningkatan emisi CO2 (Sudjoko, 2021). Indonesia menempati urutan keenam negara penghasil emisi CO2 terbesar dunia menurut World Resource Institute (WRI, 2016). Banyaknya jumlah kendaraan bermotor di Indonesia ini disebabkan karena seiring bertambahnya jumlah penduduk diikuti dengan pertambahan jumlah kendaraan bermotor yang akan terus meningkat mengikuti angka jumlah penduduk. Meningkatnya jumlah kendaraan bermotor inilah yang membuat angka emisi CO2 tinggi (Mubarak & Juni, 2025). Studi kasus analisis potensi emisi CO2 oleh kendaraan bermotor di Jalan Raya Kemantren Kabupaten Sidoarjo menyatakan jika jenis kendaraan bermotor berupa sepeda motor dan mobil ialah penyumbang terbanyak dengan potensi emisi CO2 sebesar 67,568.26 (g.30/menit/km) dan mobil sebesar 63,335.30 (g.30/menit/km) (Sudarti et al., 2022).

Salah satu strategi yang relevan dalam memprediksi emisi CO₂ adalah dengan memanfaatkan kecerdasan buatan (Artificial Intelligence/AI). AI dapat berperan sebagai alat pendukung pengambilan keputusan berbasis data untuk menurunkan emisi, baik melalui optimalisasi sistem transportasi maupun dalam penyusunan kebijakan lingkungan yang lebih adaptif (Mobasshir et al., 2025).
Dalam proyek ini, dikembangkan sebuah model prediksi emisi CO₂ berbasis machine learning yang dirancang untuk memperkirakan jumlah emisi yang dihasilkan oleh kendaraan secara akurat. Model ini dibangun menggunakan data teknis kendaraan dari Kaggle, mencakup variabel seperti kapasitas mesin, brand kendaraan dan lainnya.

# Business Understanding
## **Problem Statements**
1. Bagaimanakah cara melakukan prediksi emisi CO2 menggunakan model Machine Learning berdasarkan fitur seperti ukuran mesin, jenis bahan bakar dan penggunaan bahan bakar?
2. Bagaimana hubungan antar variabel fitur terhadap emisi CO2?

## **Goals**
1. Membuat model machine learning untuk memprediksi emisi CO2 berdasarkan fitur seperti ukuran mesin, jenis bahan bakar dan penggunaan bahan bakar.
2. Mengidentifikasi hubungan antar variabel fitur terhadap emisi CO2.

## **Solution Statements**

Berdasarkan masalah dan tujuan di atas, maka dapat diterapkan solusi sebagai berikut:
- Menggunakan dataset yang mencakup beberapa fitur yaitu:
    1. Spesifikasi : Brand, model kendaraan, ukuran mesin, jumlah silinder, jenis bahan bakar serta jenis transmisi.
    2. Efisiensi pembakaran : penggunaan bahan bakar di perkotaan, jalan raya/jalan tol dan kombinasi dalam L/100 km atau mpg (mil per galon).
    3. Variabel target prediksi : emisi CO2 (g/km).

- Pembuatan model Machine Learning untuk memprediksi emisi CO2 dilakukan menggunakan 5 model yaitu :
    - Linear Regression
    - Random Forest Regressor
    - Decision Tree Regressor
    - Gradient Boosting Regressor
    - K-Nearest Neighbor (KNN)

# Data Understanding

- Dataset yang digunakan adalah dataset [CO2 Emission by Vehicle](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles) yang diambil dari platform penyedia data Kaggle. File yang digunakan berekstensi .csv yaitu CO2 Emissions_Canada.csv.

- Dataset memiliki informasi tentang variasi emisi CO2 dari sebuah kendaraan tergantung pada berbagai fiturnya (ukuran mesin, jumlah silinder, jumlah transmisi, tipe bahan bakar, konsumsi bahan).

## Exploratory Data Analysis (EDA)

**EDA - Deskripsi Fitur**
- Terdapat 7385 baris dengan 12 kolom yang berisi informasi tentang emisi O2 oleh kendaraan.
- Dataset CO2 Emissions by Vehicle memiliki 12 kolom. Kolom-kolom dalam dataset ini dapat dijelaskan sebagai berikut:

    - Make: Merek kendaraan.
    - Model: Model kendaraan.
    - Vehicle Class: Kelas kendaraan (misalnya: compact, SUV).
    - Engine Size (L): Ukuran mesin dalam liter.
    - Cylinders: Jumlah silinder pada mesin kendaraan.
    - Transmission: Jenis transmisi (misalnya: otomatis, manual).
    - Fuel Type: Jenis bahan bakar yang digunakan (misalnya: bensin, diesel).
    - Fuel Consumption City (L/100 km): Konsumsi bahan bakar di dalam kota (liter per 100 kilometer).
    - Fuel Consumption Hwy (L/100 km): Konsumsi bahan bakar di jalan raya (luar kota).
    - Fuel Consumption Comb (L/100 km): Konsumsi bahan bakar gabungan (rata-rata kota dan jalan raya).
    - Fuel Consumption Comb (mpg): Konsumsi bahan bakar gabungan dalam satuan mil per galon. (Semakin tinggi nilainya, semakin efisien—lebih jauh dengan lebih sedikit bahan bakar.)
    - CO2 Emissions (g/km): Emisi CO₂ dalam gram per kilometer.

- Penjelasan tipe bahan bakar (Fuel Type):
    - X = Regular gasoline
    - Z = Premium gasoline
    - D = Diesel
    - E = Ethanol (E85)
    - N = Natural gas

- Dari 12 kolom tersebut terdapat 7 data numerik (Engine Size, Cylinders, Fuel Consumptions City, Fuel Consumption Hwy, Fuel Consumption Comb, CO2 Emission) dan 5 data kategoris (Make, Model, Vehicle Class, Transmission, Fuel Type).

## **EDA - Univariate Analisis**
### Tabel Deskripsi Statistik

| Fitur                                | Count  | Mean       | Std        | Min   | 25%   | 50%   | 75%   | Max   |
|--------------------------------------|--------|------------|------------|-------|-------|-------|-------|--------|
| Engine Size (L)                      | 7385.0 | 3.160068   | 1.354170   | 0.9   | 2.0   | 3.0   | 3.7   | 8.4    |
| Cylinders                            | 7385.0 | 5.615030   | 1.828307   | 3.0   | 4.0   | 6.0   | 6.0   | 16.0   |
| Fuel Consumption City (L/100 km)     | 7385.0 | 12.556534  | 3.500274   | 4.2   | 10.1  | 12.1  | 14.6  | 30.6   |
| Fuel Consumption Hwy (L/100 km)      | 7385.0 | 9.041706   | 2.224456   | 4.0   | 7.5   | 8.7   | 10.2  | 20.6   |
| Fuel Consumption Comb (L/100 km)     | 7385.0 | 10.975071  | 2.892506   | 4.1   | 8.9   | 10.6  | 12.6  | 26.1   |
| Fuel Consumption Comb (mpg)          | 7385.0 | 27.481652  | 7.231879   | 11.0  | 22.0  | 27.0  | 32.0  | 69.0   |
| CO₂ Emissions (g/km)                 | 7385.0 | 250.584699 | 58.512679  | 96.0  | 208.0 | 246.0 | 288.0 | 522.0  |

### Penjelasan Statistik Deskriptif

1. **Engine Size (L):**  
   - Ukuran rata-rata mesin kendaraan sebesar **3.16 liter**, sementara nilai tengah (median) adalah **3.00 liter**.  
   - Mesin terkecil dalam data ini berkapasitas **0.90 L**, dan yang terbesar mencapai **8.40 L**.  
   - Nilai **standar deviasi 1.35 L** menunjukkan sebaran ukuran mesin yang cukup luas.  
   - Setengah dari kendaraan memiliki ukuran mesin antara **2.00 L (Q1)** hingga **3.70 L (Q3)**.

2. **Cylinders:**  
   - Jumlah rata-rata silinder per kendaraan adalah **5.62**, dengan median **6 buah**.  
   - Jumlah silinder berkisar dari **3 hingga 16**, yang berarti terdapat model kendaraan dari yang hemat bahan bakar hingga performa tinggi.  
   - **Standar deviasi sebesar 1.83** memperlihatkan variasi jumlah silinder yang signifikan.  
   - Mayoritas kendaraan (50%) berada dalam rentang **4 sampai 6 silinder**.

3. **Fuel Consumption City (L/100 km):**  
   - Konsumsi bahan bakar saat berkendara di area kota rata-rata sebesar **12.56 L/100 km**, dengan nilai tengah **12.10**.  
   - Konsumsi paling rendah tercatat **4.20 L/100 km**, sementara tertinggi **30.60 L/100 km**, menunjukkan kontras efisiensi antar kendaraan.  
   - **Standar deviasi 3.50** menandakan adanya ketidakhomogenan dalam konsumsi bahan bakar kota.  
   - Nilai kuartil menunjukkan bahwa 50% kendaraan mengonsumsi bahan bakar di kisaran **10.10 hingga 14.60 L/100 km**.

4. **Fuel Consumption Hwy (L/100 km):**  
   - Di jalan tol, konsumsi rata-rata bahan bakar tercatat **9.04 L/100 km**, dengan median **8.70**.  
   - Konsumsi minimum **4.00** dan maksimum **20.60 L/100 km**.  
   - **Standar deviasi 2.22** menunjukkan variasi antar kendaraan lebih kecil dibandingkan konsumsi dalam kota.  
   - Sebagian besar kendaraan memiliki konsumsi bahan bakar tol antara **7.50 hingga 10.20 L/100 km**.

5. **Fuel Consumption Comb (L/100 km):**  
   - Untuk konsumsi kombinasi kota dan tol, rata-rata berada di angka **10.98 L/100 km**, median **10.60**.  
   - Rentang nilai berkisar antara **4.10 hingga 26.10**, dengan **standar deviasi 2.89**.  
   - Setengah dari total kendaraan tercatat mengonsumsi bahan bakar kombinasi antara **8.90 hingga 12.60 L/100 km**.

6. **Fuel Consumption Comb (mpg):**  
   - Efisiensi rata-rata kendaraan dalam satuan mil per galon (mpg) adalah **27.48**, dan nilai tengah **27.00**.  
   - Nilai tertinggi mencapai **69 mpg**, dan yang terendah **11 mpg**, menandakan adanya kendaraan yang sangat hemat hingga sangat boros.  
   - **Standar deviasi 7.23** mencerminkan perbedaan efisiensi antar kendaraan.  
   - Sebagian besar kendaraan memiliki efisiensi antara **22 hingga 32 mpg**.

7. **CO2 Emissions (g/km):**  
   - Emisi karbon dioksida rata-rata kendaraan adalah **250.58 gram per kilometer**, dengan median **246.00 g/km**.  
   - Emisi terendah tercatat **96 g/km**, dan tertinggi **522 g/km**.  
   - **Standar deviasi sebesar 58.51 g/km** menunjukkan keragaman signifikan dalam tingkat emisi.  
   - Sekitar 50% kendaraan dalam data ini menghasilkan emisi CO2 antara **208 hingga 288 g/km**.


**EDA - Tabel Deskripsi Fitur Kategoris**
### Tabel Statistik Deskriptif (Data Kategorikal)

| Fitur            | Count | Unique | Kategori Terbanyak (Top) | Frekuensi Terbanyak (Freq) |
|------------------|--------|--------|---------------------------|-----------------------------|
| Make             | 7385   | 42     | FORD                      | 628                         |
| Model            | 7385   | 2053   | F-150 FFV                 | 32                          |
| Vehicle Class    | 7385   | 16     | SUV - SMALL               | 1217                        |
| Transmission     | 7385   | 27     | AS6                       | 1324                        |
| Fuel Type        | 7385   | 5      | X                         | 3637                        |

### Penjelasan Statistik Deskriptif (Data Kategorikal)

1. **Make (Merek Mobil):**  
   - Terdapat total **7.385 entri** untuk merek mobil dengan **42 merek berbeda**.  
   - Merek yang paling sering muncul adalah **FORD**, yang muncul sebanyak **628 kali**.  
   - Ini menunjukkan bahwa merek FORD cukup mendominasi dalam kumpulan data ini.

2. **Model (Model Kendaraan):**  
   - Dataset mencakup **2.053 model mobil unik**, dengan total **7.385 data**.  
   - Model yang paling sering muncul adalah **F-150 FFV**, yang muncul sebanyak **32 kali**.  
   - Hal ini menunjukkan bahwa distribusi model sangat bervariasi dan sebagian besar model hanya muncul sedikit kali.

3. **Vehicle Class (Kelas Kendaraan):**  
   - Ada **16 kelas kendaraan berbeda** dalam dataset ini.  
   - Kelas **SUV - SMALL** menjadi yang paling dominan dengan **1.217 entri** dari total **7.385**.  
   - Artinya, jenis kendaraan SUV berukuran kecil cukup populer di antara data yang tersedia.

4. **Transmission (Transmisi):**  
   - Tipe transmisi yang tercatat ada **27 jenis**.  
   - Jenis transmisi **AS6** (otomatis 6 percepatan) paling sering digunakan, muncul sebanyak **1.324 kali**.  
   - Ini menunjukkan preferensi umum terhadap transmisi otomatis 6-speed di kendaraan yang ada.

5. **Fuel Type (Tipe Bahan Bakar):**  
- Terdapat **5 jenis bahan bakar** yang digunakan:
    - X = Regular gasoline
    - Z = Premium gasoline
    - D = Diesel
    - E = Ethanol (E85)
    - N = Natural gas
- **Regular Gasoline (X)** menjadi tipe bahan bakar yang paling umum, tercatat sebanyak **3.637 kali**.  

```python
df.rename(columns={'make': 'brands'}, inplace=True)
df.columns = df.columns.str.replace(" (L/100 km)", "", regex=False).str.replace("(L)", "", regex=False).str.replace("(g/km)", "", regex=False).str.replace(" ", "_", regex=False).str.lower()
df.columns
```
- Sintaks ini digunakan untuk membersihkan dan menstandarisasi nama-nama kolom (fitur) dalam DataFrame df. Tujuannya adalah untuk membuat nama kolom menjadi lebih konsisten, mudah digunakan, dan sesuai untuk analisis data atau pemodelan machine learning.

### Visualisasi Distribusi CO2

![Distribusi Emisi CO2](https://github.com/user-attachments/assets/4f4df750-19d9-4871-be57-560e59e47b3b)

Penjelasan Gambar:

Grafik ini menyajikan distribusi frekuensi emisi CO2 dari sebuah dataset. Sumbu horizontal merepresentasikan nilai emisi CO2, sedangkan sumbu vertikal menunjukkan frekuensi kemunculan nilai-nilai tersebut.

Distribusi emisi CO2 dalam dataset ini menunjukkan pola konsentrasi di sekitar nilai 200-300, dengan frekuensi tertinggi pada sekitar 250. Adanya kemiringan positif mengindikasikan bahwa meskipun sebagian besar nilai emisi relatif rendah, terdapat beberapa kejadian dengan emisi yang jauh lebih tinggi. Pemahaman terhadap distribusi ini penting untuk analisis lebih lanjut terkait faktor-faktor penyebab emisi CO2 dan potensi dampaknya.

### Visualiasasi Distribusi Kelas Kendaraan
![vehicle_type_distribution](https://github.com/user-attachments/assets/02316820-c879-4f04-bb70-f6ecdde22190)

Penjelasan:
Jenis kendaraan "COMPACT", "SUV - SMALL", dan "MID-SIZE" adalah **top three dominant** dalam data ini, sementara van, minivan, dan kendaraan khusus jumlahnya relatif sedikit.

## **EDA - Bivariate Analisis**

![CO2 Emissions by Vehicle Class](https://github.com/user-attachments/assets/104ae01e-d717-49f0-b690-563a8a2ccea8)

Penjelasan:
Grafik ini memperlihatkan bahwa kelas kendaraan yang lebih kecil seperti minicompact, subcompact, dan compact cenderung menghasilkan emisi CO2 yang lebih rendah secara keseluruhan. Sebaliknya, kelas kendaraan yang lebih besar seperti SUV standard, van kargo, dan truk standar umumnya memiliki tingkat emisi CO2 yang lebih tinggi. Selain itu, terdapat variasi emisi yang signifikan di dalam setiap kelas kendaraan, dan keberadaan outlier menunjukkan adanya beberapa kendaraan dengan tingkat emisi yang tidak biasa untuk kelasnya.

![Average Fuel Consumption](https://github.com/user-attachments/assets/cf05129e-c869-42f5-b619-58e939bc2e22)

Penjelasan: Grafik ini menunjukkan bahwa terdapat perbedaan yang signifikan dalam rata-rata konsumsi bahan bakar antar kelas kendaraan. Kendaraan yang lebih kecil seperti pickup truck kecil, station wagon kecil, dan compact cenderung paling hemat bahan bakar. Sebaliknya, kendaraan yang lebih besar dan fungsional seperti van penumpang, van kargo, dan truk standar memiliki rata-rata konsumsi bahan bakar yang paling tinggi.

## **EDA - Multivariate Analisis**

![correlation_matrix](https://github.com/user-attachments/assets/8527d541-8cd5-43ae-9bbb-f40f5982ffa1)

Penjelasan:
Matriks korelasi ini mengungkapkan hubungan yang kuat antara ukuran mesin, jumlah silinder, konsumsi bahan bakar (dalam liter/100km), dan emisi CO2. Semakin besar ukuran mesin dan jumlah silinder, cenderung semakin tinggi konsumsi bahan bakar dan emisi CO2. Sebaliknya, variabel-variabel ini berkorelasi negatif kuat dengan efisiensi bahan bakar yang diukur dalam MPG. Korelasi yang sangat kuat antara berbagai metrik konsumsi bahan bakar (city, highway, combined) dan emisi CO2 juga menegaskan hubungan langsung antara seberapa banyak bahan bakar yang digunakan dan seberapa banyak CO2 yang dihasilkan.

![pairplot](https://github.com/user-attachments/assets/a175a0b3-13a9-4744-999b-cf3fe332c61d)

Penjelasan:
Pair plot ini secara visual mengkonfirmasi hubungan yang kuat antara ukuran mesin, jumlah silinder, konsumsi bahan bakar, dan emisi CO2 yang sebelumnya ditunjukkan oleh matriks korelasi. Ukuran mesin dan jumlah silinder yang lebih besar umumnya berkaitan dengan konsumsi bahan bakar dan emisi CO2 yang lebih tinggi, serta efisiensi bahan bakar (MPG) yang lebih rendah. Sebaliknya, konsumsi bahan bakar yang tinggi (dalam L/100km) berkorelasi kuat dengan emisi CO2 yang tinggi dan efisiensi bahan bakar (MPG) yang rendah. Plot ini juga memberikan gambaran tentang distribusi masing-masing variabel secara individual.

![relationship_between_enginesize_and_CO2_accordingtofueltype](https://github.com/user-attachments/assets/960241a8-ce9e-4b6a-b65f-8cf851c6a190)

Penjelasan:
Grafik ini menunjukkan bahwa secara umum, terdapat korelasi positif antara ukuran mesin dan emisi CO2 di sebagian besar jenis bahan bakar. Artinya, semakin besar ukuran mesin, semakin tinggi emisi CO2 yang dihasilkan. Namun, hubungan ini dapat bervariasi dalam kekuatan dan sebaran data tergantung pada jenis bahan bakar yang digunakan. Satu kategori bahan bakar yaitu Natural Gas (fuel_type = N) tampak sangat jarang atau memiliki karakteristik emisi yang berbeda.

# **Data Preparation**
## Menangani Duplikasi Data dalam DataFrame

```python
if duplicate_rows > 0:
   df = df.drop_duplicates()
   print("Data duplikat telah dihapus")
```
### Tujuan
- Blok kode ini bertujuan untuk mengidentifikasi dan menghapus baris-baris duplikat yang terdapat dalam sebuah DataFrame (`df`). Penghapusan data duplikat ini penting untuk memastikan analisis data yang akurat dan menghindari bias yang mungkin timbul akibat adanya data yang berulang.

```python
# Menghapus fitur 'model' karena kardinalitas tinggi
df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
X = df.drop('co2_emissions', axis=1)
y = df['co2_emissions']

# Hapus kolom 'model' dari fitur (X)
X = X.drop('model', axis=1, errors='ignore')
```

- Kode ini bertujuan untuk melakukan **pembersihan dan persiapan data** sebelum analisis lebih lanjut atau pemodelan machine learning. Secara spesifik, kode ini melakukan langkah-langkah berikut:
   - **Menghapus Kolom Indeks Artifisial:** Menghilangkan kolom 'Unnamed: 0' yang sering muncul sebagai sisa indeks dari proses pembacaan data.
   - **Memisahkan Fitur dan Target:** Membagi DataFrame menjadi dua bagian, yaitu fitur (`X`) yang akan digunakan sebagai input model dan target (`y`) yang akan diprediksi.
   - **Menangani Kardinalitas Tinggi:** Menghapus fitur 'model' dari set fitur (`X`) karena diasumsikan memiliki terlalu banyak nilai unik, yang dapat merugikan performa model.

```python
categorical_features = X.select_dtypes(include='object').columns
print("Fitur Kategorikal setelah menghapus 'model':", categorical_features)
```
- Kode ini bertujuan untuk mengidentifikasi kolom kategorikal setelah menghapus fitur 'model'. Setelah dilakukan fitur drop, maka fitur kategorikal yang tersedia yaitu **brands, vehicle_class, transmission, fuel_type** dengan tipe data object

```python
numerical_features = X.select_dtypes(include=np.number).columns
print("Fitur Numerikal:", numerical_features)
```
- Kegunaan sintaks ini adalah untuk mengidentifikasi dan menampilkan kolom-kolom yang memiliki tipe data numerik dalam DataFrame fitur X.

- **Pembagian Dataset :** Dataset data train dan data test dengan proporsi 80% dan 20%. Dengan demikian jumlah data latih (target) sebanyak 5025 dan jumlah data test (target) sebanyak 1257.

- **Encoding Fitur Kategorikal :** Encoding dilakukan setelah splitting data agar tidak terjadi kebocoran data. Untuk mempersiapkan data dengan tipe fitur yang beragam, kita menggunakan ColumnTransformer. Langkah pertama adalah menangani fitur kategorikal. Dalam kode ini, OneHotEncoder dipilih sebagai metode encoding untuk fitur kategorikal. Alasan utama pemilihan OneHotEncoder adalah karena metode ini efektif dalam mengubah variabel kategorikal nominal (tanpa urutan intrinsik) menjadi format numerik yang dapat dipahami oleh model machine learning. One-hot encoding bekerja dengan membuat kolom biner baru untuk setiap kategori unik dalam fitur asli. Keberadaan kategori dalam suatu data poin ditandai dengan nilai 1 pada kolom yang sesuai, dan 0 untuk kolom lainnya. Selain itu, opsi handle_unknown='ignore' diaktifkan untuk mengatasi kemungkinan munculnya kategori baru dalam data uji yang tidak ada dalam data latih, sehingga mencegah error dan menjaga konsistensi pemrosesan. Sementara itu, fitur numerik pada tahap ini dibiarkan tanpa perubahan menggunakan 'passthrough', karena penanganannya (penskalaan) akan dilakukan pada langkah selanjutnya.

- **Standarisasi :** Setelah fitur kategorikal diubah menjadi numerik, langkah selanjutnya adalah mempersiapkan fitur numerik untuk algoritma K-Nearest Neighbors (KNN). KNN adalah algoritma yang sangat bergantung pada perhitungan jarak antara data poin. Oleh karena itu, skala fitur numerik menjadi sangat penting. Jika fitur memiliki rentang nilai yang jauh berbeda, fitur dengan rentang yang lebih besar dapat mendominasi perhitungan jarak, sehingga mengabaikan pengaruh fitur dengan rentang yang lebih kecil meskipun mungkin relevan. Untuk mengatasi masalah ini, dilakukan standarisasi menggunakan StandardScaler. Standarisasi mengubah setiap fitur numerik sehingga memiliki mean (rata-rata) 0 dan standar deviasi 1. Proses ini memastikan bahwa semua fitur numerik berkontribusi secara setara dalam perhitungan jarak, mencegah bias dan seringkali meningkatkan performa model KNN secara signifikan. Penerapan standarisasi dilakukan setelah encoding kategorikal untuk memastikan bahwa hanya fitur numerik yang diskalakan.
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

# **Modeling**
Dalam proyek prediksi emisi CO2 ini, beberapa algoritma regresi diterapkan untuk memodelkan hubungan antara fitur kendaraan dan tingkat emisi. Linear Regression digunakan untuk mengidentifikasi hubungan linier dasar. Decision Tree Regressor diterapkan untuk menangkap potensi hubungan non-linier melalui aturan keputusan. Random Forest Regressor, sebagai ensemble berbasis pohon, menggabungkan prediksi banyak pohon untuk meningkatkan akurasi dan mengurangi overfitting. Gradient Boosting Regressor juga merupakan teknik ensemble yang membangun model secara bertahap untuk memperbaiki kesalahan prediksi sebelumnya, seringkali menghasilkan performa tinggi. Terakhir, KNN (K-Nearest Neighbors) memprediksi emisi berdasarkan kedekatan fitur dengan data lain. Setiap algoritma ini memiliki pendekatan unik dalam mempelajari data dan mengoptimalkan prediksi emisi CO2, yang memungkinkan perbandingan efektivitas prediktifnya.

Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.
1. Linear Regression
```python
# Model 1: Linear Regression
linear_model_drop = LinearRegression()
linear_model_drop.fit(X_train_processed_drop_df, y_train)
y_pred_linear_drop = linear_model_drop.predict(X_test_processed_drop_df)
```
**Linear Regression** adalah model yang berusaha memprediksi emisi CO2 berdasarkan **hubungan linear dengan fitur-fitur**. Model ini diinisialisasi dengan *LinearRegression()* dari scikit-learn, yang secara otomatis menggunakan metode Ordinary Least Squares (OLS) untuk menemukan koefisien terbaik. Parameter utama yang sering digunakan adalah fit_intercept (secara default True untuk memperkirakan konstanta).

- Kelebihan: Interpretasi koefisien mudah, implementasi sederhana, dan komputasi cepat.
- Kekurangan: Mengasumsikan hubungan linear, sensitif terhadap outlier, dan tidak otomatis menangkap interaksi kompleks antar fitur.

2. Decision Tree Regressor
```python
tree_model_drop = DecisionTreeRegressor(random_state=42)
tree_model_drop.fit(X_train_processed_drop_df, y_train)
y_pred_tree_drop = tree_model_drop.predict(X_test_processed_drop_df)
```
Decision Tree Regressor
- Cara Kerja: Model yang memprediksi emisi CO2 dengan membuat serangkaian keputusan berdasarkan nilai fitur. Model ini diinisialisasi dengan *DecisionTreeRegressor(random_state=42)* dari scikit-learn. Parameter random_state digunakan untuk memastikan hasil yang dapat direproduksi. Model dilatih menggunakan data latih (X_train_processed_drop_df dan y_train) dan kemudian digunakan untuk memprediksi emisi pada data uji (X_test_processed_drop_df).
- Kelebihan:
   - Dapat Menangani Non-Linearitas: Mampu memodelkan hubungan yang kompleks dan non-linier antara fitur dan emisi.
   - Interpretasi Relatif Mudah: Struktur pohon dapat divisualisasikan dan aturan keputusan dapat dipahami.
   - Tidak Membutuhkan Penskalanaan Fitur: Kurang sensitif terhadap perbedaan skala antar fitur.
   - Dapat Menangani Fitur Kategorikal dan Numerik: Dapat bekerja dengan kedua jenis fitur tanpa memerlukan encoding khusus (tergantung implementasi).
- Kekurangan:
    - Rentan terhadap Overfitting: Pohon yang dalam dapat terlalu cocok dengan data latih dan gagal melakukan generalisasi dengan baik pada data baru.
   - Tidak Stabil: Perubahan kecil dalam data latih dapat menghasilkan struktur pohon yang sangat berbeda.
   - Kurang Efisien untuk Hubungan Linear Halus: Mungkin tidak seefisien regresi linear untuk hubungan yang benar-benar linier.

3. Random Forest Regressor
```python
# Random Forest Regressor (dengan parameter terbaik dari eksperimen)
best_forest_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42
)
# Latih model pada data latih yang telah diproses
best_forest_model.fit(X_train_processed_drop_df, y_train)
# Lakukan prediksi pada data uji yang telah diproses
y_pred_best_forest = best_forest_model.predict(X_test_processed_drop_df)
```
**Random Forest Regressor** adalah model ensemble yang menggunakan banyak decision tree untuk memprediksi emisi CO2 dengan parameter terbaik: n_estimators=200, max_depth=None, dan random_state=42.

**Kelebihan:** Akurasi tinggi, mengurangi overfitting karena banyak pohon, dapat memodelkan hubungan non-linier, memberikan estimasi pentingnya fitur, robust terhadap outlier, dan tidak terlalu butuh penskalaan fitur.

**Kekurangan:** Kurang interpretatif karena banyak pohon, butuh lebih banyak komputasi, berpotensi overfitting jika parameter tidak tepat, dan prediksi bisa kurang halus.

Model ini dilatih pada data latih yang diproses dan digunakan untuk prediksi pada data uji. Evaluasi (opsional dengan MSE dan R-squared) mengukur kinerja model terbaik ini, yang kini siap digunakan untuk prediksi lebih lanjut.

4. GradientBoosting Regressor
```python
gb_model_drop = GradientBoostingRegressor(random_state=42)
gb_model_drop.fit(X_train_processed_drop_df, y_train)

y_pred_gb_drop = gb_model_drop.predict(X_test_processed_drop_df)
```
**Gradient Boosting Regressor** adalah model ensemble lain yang secara iteratif membangun decision tree, dengan setiap pohon berusaha memperbaiki kesalahan dari pohon sebelumnya. Model ini diinisialisasi dengan *GradientBoostingRegressor(random_state=42)* dari scikit-learn. Parameter random_state digunakan untuk memastikan hasil yang dapat direproduksi. Model dilatih menggunakan data latih yang telah diproses (X_train_processed_drop_df dan y_train) dan kemudian digunakan untuk memprediksi emisi pada data uji (X_test_processed_drop_df).

**Kelebihan:** Akurasi sangat tinggi, dapat menangani hubungan non-linier dan interaksi fitur, fleksibel dengan berbagai fungsi loss.

**Kekurangan:** Rentan overfitting jika tidak diatur dengan baik (perlu tuning parameter), lebih sulit diinterpretasikan, membutuhkan lebih banyak waktu pelatihan.

5. K-Nearest Neighbor (KNN)
```python
knn_model = KNeighborsRegressor(n_neighbors=5) 
knn_model.fit(X_train_scaled_df, y_train)
y_pred_knn = knn_model.predict(X_test_scaled_df)
```
**K-Nearest Neighbor** memprediksi nilai emisi CO2 untuk mobil baru dengan melihat k mobil yang paling mirip dengannya dalam data pelatihan (berdasarkan fitur-fitur). Prediksi dilakukan dengan mengambil rata-rata emisi CO2 dari k mobil tetangga terdekat tersebut. Penting untuk memastikan fitur-fitur memiliki skala yang sama agar perhitungan kemiripan akurat. Nilai k (jumlah tetangga) perlu dipilih dengan hati-hati karena mempengaruhi hasil prediksi.

**Kelebihan:** Sederhana dan mudah diimplementasikan, tidak membuat asumsi tentang data (non-parametrik), dapat menangkap hubungan non-linier lokal.

**Kekurangan:** Sensitif terhadap skala fitur (perlu penskalaan), pemilihan k yang optimal sulit, mahal secara komputasi untuk dataset besar, rentan terhadap data yang tidak seimbang, interpretasi prediksi kurang jelas.

# **Evaluation**

Bagian ini akan fokus pada evaluasi performa model-model prediksi emisi CO2 yang telah dikembangkan. Dengan menggunakan metrik evaluasi standar seperti **Mean Squared Error (MSE)** dan R-squared (R²)**, kita akan menganalisis seberapa akurat model-model tersebut dalam memprediksi emisi pada data uji. MSE dan R-squared seringkali dianggap cukup untuk memberikan evaluasi yang solid terhadap model regresi. MSE berfokus pada besarnya kesalahan prediksi, sementara R-squared memberikan gambaran tentang seberapa baik model menjelaskan variasi data. Dengan menganalisis kedua metrik ini, kita dapat memperoleh pemahaman yang baik tentang seberapa akurat dan seberapa baik model-model prediksi emisi CO2 kita sesuai dengan data yang ada. Hasil evaluasi ini akan memberikan wawasan penting mengenai kualitas prediksi dari setiap model.

1. Mean Squared Error (MSE)
- Mean Squared Error (MSE) digunakan sebagai ukuran absolut dari rata-rata besarnya kesalahan prediksi. Dengan menghitung rata-rata kuadrat selisih antara nilai emisi CO2 yang diprediksi oleh model dan nilai emisi CO2 sebenarnya dari data uji, MSE memberikan indikasi seberapa jauh prediksi model menyimpang dari nilai aslinya. Nilai MSE yang lebih rendah menunjukkan bahwa model memiliki kesalahan prediksi yang lebih kecil, dan oleh karena itu, performanya lebih baik dalam memprediksi tingkat emisi CO2. Rumus MSE diformulakan sebagai berikut:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Mean Squared Error (MSE) digunakan sebagai ukuran absolut dari rata-rata besarnya kesalahan prediksi. Secara matematis, MSE dihitung dengan mengambil selisih antara setiap nilai emisi CO₂ yang diprediksi oleh model (ŷᵢ) dan nilai emisi CO₂ sebenarnya (yᵢ) dari data uji. Selisih ini kemudian dikuadratkan [(yᵢ − ŷᵢ)²] untuk memastikan bahwa semua kesalahan (baik positif maupun negatif) berkontribusi positif terhadap nilai total dan untuk memberikan bobot yang lebih besar pada kesalahan yang lebih besar. Akhirnya, nilai-nilai kuadrat selisih ini dijumlahkan untuk seluruh data uji dan dibagi dengan jumlah total sampel (n) untuk mendapatkan rata-rata kuadrat kesalahan.

Nilai MSE yang dihasilkan memberikan indikasi seberapa jauh, secara rata-rata, prediksi model menyimpang dari nilai emisi CO2 yang sebenarnya. Semakin rendah nilai MSE, semakin kecil rata-rata kuadrat kesalahan prediksi, dan oleh karena itu, performa model dalam memprediksi tingkat emisi CO2 dianggap semakin baik.

2. R-squared (R²)
- R-squared (Koefisien Determinasi) memberikan perspektif tentang seberapa baik model yang sudah dibuat cocok dengan data secara keseluruhan. Metrik ini mengukur proporsi varians dalam variabel target (emisi CO2) yang dapat dijelaskan oleh model. Nilai R-squared berkisar antara 0 dan 1, di mana nilai yang lebih mendekati 1 mengindikasikan bahwa model mampu menjelaskan sebagian besar variabilitas dalam data emisi CO2. R-squared membantu untuk memahami seberapa baik model dapat menangkap pola dan tren yang ada dalam data dibandingkan dengan model dasar yang hanya memprediksi nilai rata-rata emisi.

![Rsquared](https://github.com/user-attachments/assets/ce42f722-3465-42be-81b5-aa00f60ba7f0)

Melalui rumus ini dapat dilihat seberapa kecil kesalahan model dibandingkan dengan variasi total data. Jika kesalahan model yang dibuat kecil (SSres​ kecil), maka nilai SStot​SSres​​ juga kecil, dan R2 akan mendekati 1. Ini berarti model berhasil menjelaskan sebagian besar variasi dalam emisi CO2 dan cocok dengan data dengan baik.

Sebaliknya, jika kesalahan model besar (SSres​ besar, hampir sama dengan SStot​), maka R2 akan mendekati 0. Ini berarti model kita tidak banyak membantu dalam menjelaskan variasi emisi CO2 dan mungkin tidak lebih baik daripada sekadar menebak nilai rata-rata saja.

Jadi, nilai R-squared yang tinggi itu bagus karena artinya model kita pintar dalam 'menebak' emisi CO2 berdasarkan fitur-fitur kendaraan.

### 3. Visualisasi Perbandingan MSE dan R-squared antar Model

![MSE Antar Algoritma](https://github.com/user-attachments/assets/0917c5fa-4527-4ddc-b1f7-4c87a845d40d)

![Rsquared](https://github.com/user-attachments/assets/772d808a-149d-4f83-9150-40614b47c663)

#### **Penjelasan :**

- Setelah melalui serangkaian eksperimen dan evaluasi terhadap berbagai algoritma regresi, termasuk Linear Regression, Decision Tree, Random Forest, Gradient Boosting, dan K-Nearest Neighbors (KNN), Gradient Boosting Regressor terpilih sebagai model yang paling menjanjikan untuk memprediksi emisi CO2 kendaraan.

- Keputusan ini didasarkan pada kinerja superior Gradient Boosting dalam dua metrik evaluasi utama: Mean Squared Error (MSE) dan R-squared (R²). Nilai MSE yang rendah menunjukkan bahwa prediksi emisi CO2 oleh model ini memiliki tingkat kesalahan kuadrat rata-rata yang minimal, yang berarti prediksi cenderung sangat dekat dengan nilai sebenarnya. Sementara itu, nilai R-squared yang tinggi mengindikasikan bahwa model ini mampu menjelaskan sebagian besar variasi dalam data emisi CO2, menandakan pemahaman yang baik terhadap hubungan antara fitur-fitur kendaraan dan tingkat emisinya.

- Keunggulan Gradient Boosting terletak pada pendekatannya yang iteratif dalam membangun model ensemble. Dengan menggabungkan kekuatan banyak model yang lebih sederhana (biasanya decision tree) dan secara bertahap memperbaiki kesalahan prediksi, Gradient Boosting mampu menangkap pola-pola kompleks dan hubungan non-linear dalam data yang mungkin terlewatkan oleh model tunggal seperti Linear Regression atau Decision Tree. Meskipun Random Forest juga merupakan teknik ensemble yang kuat, hasil evaluasi menunjukkan bahwa mekanisme boosting dalam Gradient Boosting memberikan keunggulan spesifik untuk tugas prediksi emisi CO2 ini.

- Sebaliknya, model KNN menunjukkan kinerja yang kurang memuaskan, terutama dalam hal MSE yang tinggi, mengindikasikan error prediksi yang lebih besar. Sementara Linear Regression dan Decision Tree memberikan hasil yang lebih baik dari KNN, mereka tidak mencapai tingkat akurasi dan kemampuan penjelasan varians yang ditunjukkan oleh model-model ensemble, terutama Gradient Boosting.

- Secara keseluruhan, terpilihnya Gradient Boosting Regressor sebagai algoritma terbaik mengimplikasikan bahwa model ini memiliki potensi terbesar untuk memberikan prediksi emisi CO2 yang akurat dan andal berdasarkan fitur-fitur kendaraan yang tersedia. Ini menjadi landasan yang kuat untuk pengembangan lebih lanjut dan penerapan model dalam konteks yang memerlukan estimasi emisi CO2 yang presisi.

---Ini adalah bagian akhir laporan---

Referensi:

1. Ajala, A. A., Adeoye, O. L., Salami, O. M., & Jimoh, A. Y. (2025). An examination of daily CO₂ emissions prediction through a comparative analysis of machine learning, deep learning, and statistical models. Environmental Science and Pollution Research, 32, 2510–2535. https://doi.org/10.1007/s11356-024-35764-8.

2. IEA (Internasional Energi Agency). (2015). Energi and Climate Change. World Energi Outlook Special Report. IEA, Paris, France.

3. IPCC. 2014. AR5 Synthesis Report: Climate Change 2014. Diunduh 24 April 2025, dari https://www.ipcc.ch/report/ar5/syr

4. Kabir, M., Habiba, U. E., Khan, W., Shah, A., Rahim, S., De los Rios-Escalante, P. R., Zia, U. R. F., Liaqat, A. & Shafiq, M. (2023). Climate change due to increasing concentration of carbon dioxide and its impacts on environment in 21st century; a mini review. Journal of King Saud University-Science, 35(5): 1-7.

5. Katila, P., Colfer, C. J. P., De Jong, W., Galloway, G., Pacheco, P., & Winkel, G. (Eds.). (2019). Sustainable development goals. Cambridge University Press.

6. Mobasshir, M., Praveen, P., Pratibha, K., Faisal, K., Azhar, E., Osama, K., Mohd, P., Taufique, A. & Shadab, A. (2025). Green Technologies and Sustainability, 3(3) : 1-18.

7. Mubarak, N. R. & Juni, R. (2025). Penerapan Peraturan Emisi pada Penurunan Emisi Gas Rumah Kaca dari Kendaraan Bermotor di Indonesia. Bacarita Law Journal, 5(2) : 201-208.

8. Rocque, R. J., Beaudoin, C., Ndjaboue, R., Cameron, L., Poirier-Bergeron, L., Poulin-Rheault, R.-A., Fallon, C., Tricco, A. C., & Witteman, H. O. (2021). Health effects of climate change: An overview of systematic reviews. BMJ Open, 11(6), e046333. https://doi.org/10.1136/bmjopen-2020-046333

9. Sampaio, A.R.S., David, G.B.F., Joel, C.Z.J. & Arlenes, B.D.S. (2025). Artificial intelligence applied to truk emissions reduction: A novel emissions calculation model. Transportation Research Part D: Transport and Environment. 138(104533): 1-14.

10. Sudarti, Yushardi & Nur, K. (2022). Analisis Potensi Emisi CO2 oleh Berbagai Jenis Kendaraan Bermotor di Jalan Raya Kemantren Kabupaten Sidoarjo. Jurnal Sumberdaya Alam dan Lingkungan, 9(2) : 70-75.

11. Sudjoko, C. (2021). Strategi Pemanfaatan Kendaraan Listrik Berkelanjutan sebagai Solusi untuk Mengurangi Emisi Karbon. Jurnal Paradigma: Jurnal Multidisipliner Mahasiswa Pascasarjana Indonesia, 2(2): 54-68.

12. United Nations. (n.d.). Goal 13: Take urgent action to combat climate change and its impacts. United Nations Sustainable Development Goals. https://sdgs.un.org/goals/goal13.

13. World Resources Indonesia. Institute Menginterpretasikan Menilai Transparansi (WRI) 2016. Indc: Target Emisi Gas Rumah Kaca Pasca 2020 Dari 8 Negara Penyumbang Emisi Terbesar. Washington, DC: Open Climate Network (OCN).
