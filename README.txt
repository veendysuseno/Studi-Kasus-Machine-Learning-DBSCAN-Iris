README.md 

Studi Kasus:
Pada Proyek kali ini akan menggunakan dataset IRIS yang dapat diunduh di https://archive.ics.uci.edu/dataset/53/iris.
Kali ini akan Melakukan Penerapan Machine Learning dengan pendekatan Unsupervised melalui clustering terhadap data-data bunga iris tersebut dengan menggunakan Algoritma DBSCAN. Untuk itu pertama-tama kita pelajari dahulu apa itu Algoritma DBSCAN.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) adalah salah satu algoritma klasterisasi yang digunakan untuk menemukan klaster dalam data berdasarkan kepadatan titik data. Algoritma ini sangat efektif untuk menemukan klaster dengan bentuk yang arbitrer dan juga mampu menangani outlier (data yang berada jauh dari klaster lainnya).
Prinsip Dasar DBSCAN:

    Parameter Utama:
        Epsilon (ε): Jarak maksimum yang digunakan untuk menentukan tetangga sekitar suatu titik. Dua titik data dianggap bertetangga jika jarak mereka kurang dari atau sama dengan ε.
        MinPts: Jumlah minimum titik data yang harus ada dalam jangkauan ε untuk dianggap sebagai klaster.

    Jenis Titik Data:
        Core Point: Sebuah titik data dianggap sebagai core point jika setidaknya terdapat MinPts titik (termasuk dirinya sendiri) dalam radius ε.
        Border Point: Sebuah titik data yang bukan core point tetapi berada dalam jarak ε dari core point.
        Noise Point (Outlier): Titik data yang bukan core point dan bukan border point.

    Cara Kerja Algoritma DBSCAN:
        Inisialisasi: Mulai dari titik data acak yang belum dikunjungi.
        Ekspansi Klaster:
            Jika titik tersebut merupakan core point (memiliki setidaknya MinPts tetangga dalam radius ε), maka bentuk klaster dengan menggabungkan semua titik tetangga.
            Lalu, periksa semua tetangga dari core point tersebut. Jika tetangga tersebut juga merupakan core point, lanjutkan untuk menggabungkan tetangga-tetangga mereka.
            Proses ini berlanjut sampai semua titik dalam klaster telah diperiksa.
        Penanganan Noise: Titik data yang tidak termasuk dalam klaster mana pun dianggap sebagai noise atau outlier.
        Iterasi: Lanjutkan dengan titik data berikutnya yang belum dikunjungi dan ulangi proses di atas.

    Hasil Akhir: Setelah semua titik data telah diproses, algoritma selesai, dan semua klaster serta outlier telah teridentifikasi.

Kelebihan dan Kekurangan DBSCAN:

Kelebihan:

    Tidak memerlukan penentuan jumlah klaster di awal, berbeda dengan algoritma K-Means.
    Mampu menemukan klaster dengan bentuk yang kompleks dan tidak beraturan.
    Dapat menangani noise dan outlier dengan baik.
    Klasterisasi tidak bergantung pada inisialisasi pusat klaster.

Kekurangan:

    Sensitif terhadap parameter ε dan MinPts. Memilih nilai parameter yang tidak tepat dapat menghasilkan klasterisasi yang buruk.
    Tidak cocok untuk data dengan kepadatan yang bervariasi; jika kepadatan sangat bervariasi, algoritma ini bisa saja gagal mengidentifikasi klaster dengan benar.
    Waktu komputasi bisa menjadi masalah untuk dataset besar, meskipun algoritma ini umumnya lebih efisien dibandingkan dengan pendekatan berbasis jarak lainnya.

Contoh Implementasi DBSCAN dengan Python dan Scikit-Learn

Berikut adalah contoh implementasi DBSCAN menggunakan scikit-learn:

python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Buat dataset
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# Inisialisasi model DBSCAN
dbscan = DBSCAN(eps=0.1, min_samples=5)

# Fit dan prediksi klaster
y_dbscan = dbscan.fit_predict(X)

# Visualisasi hasil
plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='viridis', s=50)
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

Penjelasan Kode:

    make_moons: Digunakan untuk membuat dataset dengan bentuk bulan sabit, yang sering digunakan untuk menguji algoritma klasterisasi.
    DBSCAN: Algoritma DBSCAN diinisialisasi dengan parameter eps dan min_samples.
    fit_predict: Algoritma di-fit ke data dan melakukan prediksi klaster.
    plt.scatter: Menampilkan hasil klasterisasi.

Kesimpulan

DBSCAN adalah algoritma yang kuat untuk klasterisasi data yang tidak terstruktur atau memiliki outlier, terutama ketika bentuk klaster tidak diketahui sebelumnya dan bisa bervariasi secara kompleks. Pemilihan parameter yang tepat adalah kunci keberhasilan DBSCAN dalam mengidentifikasi struktur data yang sebenarnya.


@Copyright Veendy 2024