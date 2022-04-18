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
Algoritma lain juga diguanakn oleh Madan, Saluja dan Zhao (2014) [[2]](http://cs229.stanford.edu/proj2014/Isaac%20Madan,%20Shaurya%20Saluja,%20Aojia%20Zhao,Automated%20Bitcoin%20Trading%20via%20Machine%20Learning%20Algorithms.pdf) yang menggunakan _random forest_ dengan akurasi 57.4%.
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
  - Tidak ada _missing value_ pada dataset. 
  
  Untuk penjelasan mengenai variabel-variabel pada dataset dapat dilihat pada poin-poin berikut ini:
    *   Date : Tanggal pencatatan data
    *   Open : harga ketika dibuka yang dihitung perhari
    *   Close : harga ketika ditutup yang dihitung perhari
    *   Low : harga terendah perhari
    *   High : harga tertinggi perhari
    *   Volume : volume transaksi perhari

- **Pengolahan Fitur dataset**
  <br> sebelum masuk ke tahap pengolahan data, perlu menambahkan dua buah variabel baru yaitu `Price_Average` dan `Next Month Price`.
  <br> Sehingga variabel pada data adalah `Date, High, Low, Open, Close, Volume, Price_Average, Next_Month_Price`.

    **Pengolahan Data**
   - Mengidentifikasi Missing Value dan Outlier
    
     Cukup banyak _outliers_ pada dataset, dan untuk mengatasi ini penulis menggunakan metode InterQuartile Range (IQR) untuk menghapus semua outliers pada data.
    
  - Univariate Analysis
  
    Peningkatan harga sebanding dengan penurunan jumlah sampel. Hal ini dapat kita lihat jelas dari semua histogram yang grafiknya mengalami penurunan seiring dengan semakin banyaknya jumlah sampel (sumbu x).
    Terlihat pada grafik bahwa semua data cenderung distribusi nilainya miring ke kanan (right-skewed). Hal ini akan berimplikasi pada model nantinya.
    
  - Multivariate Analysis
   
    Pada gambar diatas bisa kita lihat bahwa kebanyakan data memiliki korelasai positif yang ditandai dengan meningkatnya variabel pada sumbu y saat terjadi peningkatan variabel pada sumbu x. Selain itu ada juga data yang memiliki sebaran data acak yaitu pada grafik Volume.
    
    bisa juga kita lihat pada score korelasinya pada gambar berikut:
    
    Terlihat pada matriks korelasi di atas dapat disimpulkan bahwa kebanyakan variabel memiliki keterikatan dan korelasi yang cukup kuat antar variabel lainnya, dimana nilai korelasi antar variabel bernilai lebih dari 0.8 atau mendekati 1. Sedangkan Volume memiliki korelasi yang lemah yaitu -0.05 dan korelasinya negatif.

## Data Preparation

Selanjutnya pengolahan data yang dilakukan dengan tahapan sebagai berikut:
  - **Melakukan Penanganan _Missing Value_ dan _Outliers_**
    Setelah dilakukan pengecekan terdapat 8 _missing value_ pada dataset yaitu pada kolom volume, untuk menangani hal tersebut cukup dengan menghapus data yang nilainya 0, sehingga data tidak lagi memiliki nilai nol dan kita bisa cek nilai minimum data bukan nol, sedangkan untuk mengatasi _outliers_ penulis menggunakan _interquartile range_ atau IQR yang memanfaatkan batas bawah dan batas atas data.

  - **Melakukan pembagian dataset**
    Dataset yang kita miliki perlu dilakukan pembagian menjadi data _train_ dan data _test_. data _train_ merupakan data yang digunakan untuk melatih model dan data _test_ adalah data yang belum diketahui oleh model dan data ini akan digunakan untuk menguji model yang kita rancang. Pada proyek kali ini data untuk _train_ sebeasar 80% dan untuk _test_ sebesat 20%.  Pembagian dataset dilakukan dengan modul [train_test_split](https://scikit-learn.org/0.24/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) dari scikit-learn. Jadi sample yang digunaka untuk melatih model adalah 1316 dan sampel untuk menguji model adalah 330.
    
  - **Standardisasi data pada semua fitur numerik pada dataset**
    

Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
