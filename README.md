# Penerapan Machine Learning dengan Algoritma DBSCAN

## Studi Kasus

Proyek ini menggunakan dataset IRIS yang dapat diunduh di [link ini](https://archive.ics.uci.edu/dataset/53/iris). Proyek ini melakukan klasterisasi data bunga iris dengan algoritma DBSCAN, yang merupakan pendekatan _unsupervised_.

## Definisi DBSCAN

DBSCAN (_Density-Based Spatial Clustering of Applications with Noise_) adalah algoritma klasterisasi yang mendeteksi klaster berdasarkan kepadatan data. Algoritma ini sangat efektif dalam mengidentifikasi klaster berbentuk arbitrer dan menangani outlier.

### Parameter Utama:

- **Epsilon (ε)**: Jarak maksimum untuk menentukan tetangga sekitar suatu titik.
- **MinPts**: Jumlah minimum titik yang diperlukan dalam radius ε untuk membentuk klaster.

### Jenis Titik Data:

- **Core Point**: Titik dengan setidaknya MinPts tetangga dalam radius ε.
- **Border Point**: Titik yang bukan _core point_ tetapi berada dalam radius ε dari _core point_.
- **Noise Point (Outlier)**: Titik yang tidak termasuk dalam klaster mana pun.

### Cara Kerja Algoritma DBSCAN:

1. **Inisialisasi**: Pilih titik data acak yang belum dikunjungi.
2. **Ekspansi Klaster**: Jika titik adalah _core point_, bentuk klaster dengan semua tetangga dalam radius ε.
3. **Penanganan Noise**: Titik yang tidak termasuk dalam klaster akan dianggap sebagai noise.
4. **Iterasi**: Proses ini berlanjut hingga semua titik data dikunjungi.

## Kelebihan DBSCAN:

- Tidak perlu menentukan jumlah klaster di awal.
- Mampu menemukan klaster dengan bentuk kompleks.
- Menangani noise dan outlier dengan baik.

## Kekurangan DBSCAN:

- Sensitif terhadap parameter ε dan MinPts.
- Tidak cocok untuk data dengan kepadatan bervariasi.
- Waktu komputasi meningkat untuk dataset besar.

## Contoh Implementasi DBSCAN dengan Python dan Scikit-Learn

```python
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
```

## Penjelasan Kode:

- make_moons: Membuat dataset dengan bentuk bulan sabit.
- DBSCAN: Menginisialisasi algoritma DBSCAN dengan parameter eps dan min_samples.
- fit_predict: Melakukan klasterisasi pada data.
- plt.scatter: Visualisasi hasil klasterisasi.

## Kesimpulan

- DBSCAN adalah algoritma klasterisasi yang efektif untuk data tidak beraturan atau memiliki outlier. Algoritma ini tidak memerlukan penentuan jumlah klaster dan sangat cocok untuk klaster dengan bentuk kompleks. Pemilihan parameter yang tepat sangat penting untuk keberhasilan DBSCAN.

#### @Copyright Veendy 2024
