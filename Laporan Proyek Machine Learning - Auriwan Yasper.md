# Ethereum-Cryptocurrency-Predictive-Analytics
# Laporan Proyek Machine Learning - Auriwan Yasper

## Domain Proyek
Topik yang dipilih dalam proyek machine learning ini adalah mengenai keuangan dan cryptocurrency dengan judul proyek "Ethereum Price Predictive Analytics".

**Latar Belakang**
_Cryptocurrency_ telah mendapatkan popularitas yang cukup besar dalam beberapa tahun terakhir. Sifat _cryptocurrency_ yang tidak dapat dilacak dan tidak terkendali menarik jutaan orang di seluruh dunia [[1]](https://www.sciencedirect.com/science/article/abs/pii/S0045790618331343). Terlebih lagi dunia telah melihat berbagai kemajuan teknologi, seperti _Internet of Things_ (IoT), _e-commerce_, dan pembayaran digital.
Saat ini banyak pengguna yang terlibat dalam _cryptocurrency_ seperti _Ethereum_. Dengan peningkatan mendadak dalam setiap detik dan modal yang digunakan dalam transaksi tersebut, ada kebutuhan bagi pengguna untuk dapat memprediksi apakah pengguna akan melakukan transaksi menggunakan _ethereum_ atau tidak. Sehingga dapat mengurangi kerugian akibat rendahnya harga pasar saat melakukan transaksi.

Oleh karena Itu pada proyek kali ini akan diusulkan model _machine learning_ untuk melakukan prediksi harga pasar dan menyusun strategi transaksi yang tepat jika ingin melakukan transaksi menggunakan ethereum atau menunggu momen saat harga tertinggi terlebih dahulu untuk melakukan transaksi.

Cukup banyak model yang sudah digunakan dalam prediksi harga coin _criptocurrency_, seperti 
yang dilakukan oleh Hegazy dan Mumford (2016) [[2]](http://cs229.stanford.edu/proj2016/report/MumfordHegazy-ComparitiveAutomatedBitcoinTradingStrategies-report.pdf) yang menghitung harga bitcoin menggunakan algoritma _decision treee_ dan memprediksi perubahan harga bitcoin dengan akurasi sebesar 57.11%. 
Algoritma lain juga digunakan oleh Madan, Saluja dan Zhao (2014) [[3]](http://cs229.stanford.edu/proj2014/Isaac%20Madan,%20Shaurya%20Saluja,%20Aojia%20Zhao,Automated%20Bitcoin%20Trading%20via%20Machine%20Learning%20Algorithms.pdf) yang menggunakan _random forest_ dengan akurasi 57.4%.
Selain itu masih banyak model lainnya yang dapat digunakan seperti _Logistic Regression_, _Naive Bayes_, _Support Vector Machine_, _Auto Regressive Integrated Moving Average_, _Recurrent Neural Network_, dan masih banyak lainnya.
Pada Proyek ini akan digunakan tiga algoritma yaitu _K-Nearest Neighbors_, _Random Forest_, dan _AdaBoosting_. Dari ketiga model ini akan dipilih model dengan akurasi terbaik yang akan digunakan untuk memprediksi harga Ethereum satu bulan selanjutnya.

## Business Understanding

### Problem Statements
Berdasarkan pada latar belakang di atas, permasalahan yang dapat diselesaikan pada proyek ini adalah sebagai berikut:

- Bagaimana melakukan pengolahan dataset agar dapat memepoleh data yang baik untuk model _machine Learning_?
- Bagaimana Membangun model _machine learning_ untuk memprediksi harga satu bulan selanjutnya?

### Goals
Tujuan dibuatnya proyek ini adalah sebagai berikut:

- Melakukan pengolahan dataset _ethereum_ agar dapat digunakan dalam membangun model.
- Membangun model _machine learning_ untuk memprediksi harga satu bulan selanjutnya.

### Solution statements
Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini di antaranya:

Pengolahan data dapat dilakukan beberapa tahapan sebagai berikut:
- Melakukan perhitungan rata-rata berdasarkan data Open, Close, High dan Low.
- Melakukan pembagian dataset.
- Mengatasi data _outliers_ dengan Metode IQR.
- Standardisasi fitur numerik pada dataset.

Perancangan model dengan tiga algoritma yaitu:
  - _K-Nearest Neighbors_
  - _Random Forest_
  - _Boosting Algorithm_

## Data Understanding
- **Informasi Dataset**
  Dataset yang digunakan berupa data Historis Ethereum dari 2016 sampai 2022, informasi lebih lanjut mengenai dataset tersebut dapat lihat pada tabel berikut:

  | Jenis                   | Keterangan                                                                              |
  | ----------------------- | --------------------------------------------------------------------------------------- |
  | Sumber                  | Dataset: [Kaggle](https://www.kaggle.com/datasets/kaushiksuresh147/ethereum-cryptocurrency-historical-dataset) |
  | Dataset Owner           | Kash                                                                           |
  | Lisensi                 | CC0: Public Domain                                                                        |
  | Kategori                | Cryptorrency, Finance                                                                     |
  | Usability               | 10.00                                                                                       |
  | Jenis dan Ukuran Berkas | CSV (382.27 kB)                                                                           |
  | Versi | Version 27                                                                           |

  dataset tersebut memiliki informasi sebagai berikut :
  
  - Terdapat  2229 baris yang berisi informasi mengenai data riwayat harga **Ethereum** setiap harinya.
  - Terdapat 6 kolom yaitu `Date, High, Low, Open, Close, Volume` yang merupakan variabel - variabel pada data
  - Dari kolom-kolom tersebut terdapat 4 kolom numerik dengan tipe data float64, yaitu `High, Low, Open, Close` dan terdapat 1 kolom numerik dengan tipe data int64 yaitu `Volume` yang merupakan fitur numerik. 
  
  Untuk penjelasan mengenai variabel-variabel pada dataset dapat dilihat pada poin-poin berikut ini:
    *   Date : Tanggal pencatatan data
    *   Open : harga ketika dibuka yang dihitung perhari
    *   Close : harga ketika ditutup yang dihitung perhari
    *   Low : harga terendah perhari
    *   High : harga tertinggi perhari
    *   Volume : volume transaksi perhari

- **Pengolahan Fitur dataset**
  sebelum masuk ke tahap pengolahan data, penulis menambahkan dua buah variabel baru yaitu `Price_Average` dan `Next Month Price`.
  Sehingga variabel pada data adalah `Date, High, Low, Open, Close, Volume, Price_Average, Next_Month_Price`.

    **Pengolahan Data**
   - Mengidentifikasi Missing Value dan Outlier
    <br>
    <image src='https://raw.githubusercontent.com/auriwan/Ethereum-Cryptocurrency-Predictive-Analytics/data-gambar/Missing%20Value.PNG' width= 500/>
    <br>
     Setelah dilakukan pengecekan terdapat 8 _missing value_ pada dataset yaitu pada kolom volume, untuk menangani hal tersebut cukup dengan menghapus data yang nilainya 0. Selanjutnya cukup banyak _outliers_ pada dataset, dan untuk mengatasi ini penulis menggunakan metode InterQuartile Range (IQR) untuk menghapus semua outliers pada data.
    
  - Univariate Analysis
    <br>
    <image src='https://github.com/auriwan/Ethereum-Cryptocurrency-Predictive-Analytics/blob/data-gambar/Histogram%20Univariate.png?raw=true' width = 500/>
    <br>
    Peningkatan harga sebanding dengan penurunan jumlah sampel. Hal ini dapat kita lihat jelas dari hampir semua histogram yang grafiknya mengalami penurunan seiring dengan semakin banyaknya jumlah sampel (sumbu x).
    Terlihat pada grafik bahwa hampir semua data cenderung distribusi nilainya miring ke kanan (right-skewed). Hal ini akan berimplikasi pada model nantinya.
    
  - Multivariate Analysis
   _gambar mulitvariate analysis_
    Pada gambar diatas bisa kita lihat bahwa kebanyakan data memiliki korelasai positif yang ditandai dengan meningkatnya variabel pada sumbu y saat terjadi peningkatan variabel pada sumbu x. Selain itu ada juga data yang memiliki sebaran data acak yaitu pada grafik Volume.
    
    bisa juga kita lihat pada score korelasinya pada gambar berikut:
    _gambar matrik korelasi_
    Terlihat pada matriks korelasi di atas dapat disimpulkan bahwa kebanyakan variabel memiliki keterikatan dan korelasi yang cukup kuat antar variabel lainnya, dimana nilai korelasi antar variabel bernilai lebih dari 0.8 atau mendekati 1. Sedangkan Volume memiliki korelasi yang lemah yaitu -0.05 dan korelasinya negatif.

## Data Preparation

Selanjutnya pengolahan data yang dilakukan dengan tahapan sebagai berikut:
  - **Melakukan Penanganan _Missing Value_ dan _Outliers_**
    Setelah dilakukan pengecekan terdapat 8 _missing value_ pada dataset yaitu pada kolom volume, untuk menangani hal tersebut cukup dengan menghapus data yang nilainya 0, sehingga data tidak lagi memiliki nilai nol dan kita bisa cek nilai minimum data bukan nol, sedangkan untuk mengatasi _outliers_ penulis menggunakan _interquartile range_ atau IQR yang memanfaatkan batas bawah dan batas atas data.

  - **Melakukan pembagian dataset**
    Dataset yang kita miliki perlu dilakukan pembagian menjadi data _train_ dan data _test_. data _train_ merupakan data yang digunakan untuk melatih model dan data _test_ adalah data yang belum diketahui oleh model dan data ini akan digunakan untuk menguji model yang kita rancang. Pada proyek kali ini data untuk _train_ sebeasar 80% dan untuk _test_ sebesat 20%.  Pembagian dataset dilakukan dengan modul [train_test_split](https://scikit-learn.org/0.24/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) dari scikit-learn. Jadi sample yang digunaka untuk melatih model adalah 1316 dan sampel untuk menguji model adalah 330.
    
  - **Standardisasi data pada semua fitur numerik pada dataset**
    Proses Standardisasi sangat diperlukan dalam melatih model, karena pada dasarnya model akan lebih mudah memproses data yang seragam dan mendekati distribusi normal. Karena data merupakan fitur numerik maka teknik yang digunakan adalah StandarScaler dari library [Scikitlearn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html). StandarScaler ini melakukan standarisasi dengan mengurangkan nilai rata-rata kemudian membaginya dengan standar deviasi untuk menggeser distribusi. StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1. Sebagai tahap awal, standarisai dilakukan untuk data latih terlebih dahulu untuk menghindari kebocoran data, sedangkan untuk data uji akan dilakukan standardisasi pada tahap evaluasi.

## Modeling
Sebagaimana yang penulis sebutkan diatas model _machine learning_ yang akan dirancang akan menggunakan 3 algoritma yaitu _K-Nearest Neighbors_, _Random Forest_, _Boosting Algorithm_. Ketiga algoritma ini akan kita evaluasi dan mencari algoritma dengan akurasi terbaik.
    - **K-Nearest Neighbors**
    Algoritma KNN adalah algoritma yang memiliki karakteristik sederhana akan tetapi efective dalam melakukan _data mining_. Kelemahan utama dari teknik ini adalah ketika sejumlah data yang memiliki _noise_ dan tidak lengkap terlibat dalam model, maka akan mengakibatkan algoritma KNN menjadi tidak efficient dan tidak presisi [[4]](https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/widm.1289). KNN beroperasi dengan menggunakan kesamaan fitur untuk memprediksi nilai dari setiap data baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. Pembuatan model dilakukan menggunakan modul [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) dari library scikitlearn. K yang digunakan adalah 10 dan metric Euclidean untuk mengukur jarak antara titik. Pada model ini akan membandingkan jarak satu sampel data ke 10 sampel data tetangganya yang terdekat, agar hasil persamaan regresi yang dihasilkannya nantinya akan lebih halus, tahapan itu akan dilakukan berulang-ulang hingga mendapatkan hasil persamaan regresi dengan nilai yang maksimal. Kemudian proses selanjutnya melakukan prediksi menggunakan data uji dan melakukan pengujian. Pada tahap ini kita hanya melatih data training dan menyimpan data testing untuk tahap evaluasi.

- **Kelebihan**:
    - Algoritma KNN merupakan algoritma yang sederhana dan Effektif dalam melakukan _data mining_.
    - Dapat di implementasikan pada beberapa kasus seperti klasifikasi, regresi dan pencarian.
- **Kekurangan**:
    - Algoritma KNN menjadi lebih lambat secara signifikan seiring meningkatnya jumlah sampel dan/atau variabel independen.
    - Algoritma KNN akan menjadi tidak effisien dan kurang presisi ketika berhadapan dengan data yang memiliki banyak missing value.

- **Random Forest**
  Random Forest disusun dari banyak algoritma decision tree yang pembagian data dan fiturnya dipilih secara acak. Random Forest pada dasarnya adalah versi bagging dari algoritma decision tree. Pembuatan model dilakukan dengan menggunakan modul [RandomForestClassifier](https://scikitlearn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) dari library Scikitlearn. Ada beberapa parameter yang digunakan untuk memaksimalkan hasil dari pelatihan model sebagai berikut: 
   - Parameter pertama adalah `n_estimator` yaitu jumlah _trees_ (pohon) di _forest_. Pada proyek ini penulis memilih pohon sebanyak 50 _trees_. 
   - Parameter selanjutnya adalah `max_depth` yang merupakan kedalaman atau panjang pohon. Ini merupakan ukuran seberapa banyak pohon dapat membelah (_splitting_) untuk membagi setiap _node_ ke dalam jumlah pengamatan yang di inginkan, pada proyek ini penulis melakukan set nilai `max_depth` sebesar 16 _split_. 
   - Parameter `random_state` digunakan untuk mengontrol _random number generator_ yang digunakan. Pada proyek ini penulis melakukan _set_ nilai pada parameter `random_state` sebesar 55. 
   - Parameter `n_jobs` yaitu jumlah _job_ (pekerjaan) yang digunakan secara paralel. Ini merupakan komponen untuk mengontrol _thread_ atau proses yang berjalan secara paralel. `n_jobs = -1` artinya semua proses berjalan secara paralel.
 
    **Kelebihan** :
    - dapat menghasilkan error yang lebih rendah. 
    - memberikan hasil yang bagus dalam klasifikasi.
    - dapat mengatasi data training dalam jumlah sangat besar secara efisien dan
    - metode yang efektif untuk mengestimasi missing data.

   **Kekurangan** :
   - Algoritma Random Forest overfiting untuk beberapa kumpulan data dengan tugas klasifikasi/regresi yang _bising/noise_.
    - Untuk data yang menyertakan variabel kategorik dengan jumlah level yang berbeda, Random Forest menjadi bias dalam mendukung atribut dengan level yang lebih banyak. Oleh karena itu, skor kepentingan variabel dari Random Forest tidak dapat diandalkan untuk jenis data ini.

- **Boosting Algorithm**
  Algoritma ini bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. Pada tahap ini pembuatan model dilakukan dengan menggunakan modul [Boosting Alghoritm](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) dari library Scikitlearn. Pada proyek ini, penulis akan menggunakan metode adaptive boosting. Salah satu metode adaptive boosting yang terkenal adalah AdaBoost. Ada beberapa prameter yang digunakan dalam AdaBoosting yaitu:
    - learning_rate: bobot yang diterapkan pada setiap regressor di masing-masing proses iterasi boosting.
    - random_state: digunakan untuk mengontrol random number generator yang digunakan. 

    **Kelebihan** :
  -   Algoritma Boosting dapat mengurangi bias pada data.
  -   Prosedur Boosting cukup sederhana.
  -   Algoritma ini sangat efisien dalam meningkatkan akurasi prediksi.
  -   Algoritma boosting sering mengungguli model yang lebih sederhana seperti logistic regression dan random forest.

    **Kekurangan** :
  -   AdaBoost sangat dipengaruhi oleh outlier.

## Evaluation
Metrik yang akan penulis gunakan pada proyek ini adalah MSE atau Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. Karena penulis baru melakukan scaling pada data latih untuk menghindari kebocoran data, maka sebelum menghitung nilai mse, penulis perlu melakukan scaling fitur terlebih dahulu pada data uji. Setelah melakukan scaling data uji barulah MSE dihitung. Secara matematis dihitung dengan persamaan berikut:

_gambar rumus mse_

keterangan:
N = jumlah dataset
yi = nilai sebenarnya
yi^ = nilai prediksi

Penulis juga melakukan evaluasi dengan menggunakan metrik akurasi, yaitu tingkat keakuran data prediksi yang didasarkan dari data latih pada model. Metrik Akurasi mungkin metrik paling awam/paling diketahui pada pemodelan klasifikasi. Metrik ini adalah persentase jumlah data yang diprediksi secara benar terhadap jumlah keseluruhan data. Jika ditinjau dengan confusion matrix, akurasi adalah rasio dari jumlah elemen diagonal terhadap jumlah seluruh elemen matriks, atau:

_gambar Rumus metrik akurasi_

Berdasarkan metrik akurasi penulis mendapati bahwa model dengan akurasi tertinggi adalah _Random Forest_, yaitu 84.7%. Sama halnya dengan metrik akurasi, MSE juga menunjukkan bahwa model _Random Forest_ memberikan error yang paling kecil. Sedangkan _boosting_ memiliki error paling besar. Jadi model yang akan penulis gunakan dalam memprediksi harga satu bulan selanjutnya adalah model _Random Forest_.

Berikut ini bisa kita lihat perbandingan grafik MSE dan akurasi ketiga model

_gambar grafik mse_

Berdasarkan tingkat eror pada grafik di atas, semakin kecil tingkat eror maka semakin baik model tersebut memprediksi data. Jika dibandingkan dengan dua model lainnya, model dengan Error terkecil adalah Model Random Forest.
