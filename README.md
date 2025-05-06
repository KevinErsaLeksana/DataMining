# DataMining
1. Import Library
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
```
Baris ini digunakan untuk mengimpor semua pustaka yang dibutuhkan. datasets untuk mengambil dataset bawaan dari scikit-learn, train_test_split untuk membagi data, DecisionTreeClassifier dan plot_tree untuk membuat dan menggambar pohon keputusan, serta matplotlib.pyplot (plt) untuk menampilkan visualisasi grafik.

2. Load dan Eksplorasi Dataset
```python
dir(datasets)
data = datasets.load_linnerud()
data
dir(data)
data.target
```
Pertama, dir(datasets) digunakan untuk melihat daftar dataset bawaan. Kemudian, load_linnerud() memuat dataset Linnerud, yaitu dataset multivariat dengan fitur fisik (seperti sit-up, chin-up) dan target berupa tiga ukuran tubuh. dir(data) digunakan untuk melihat properti yang tersedia dalam objek data, seperti .data (fitur) dan .target (label). data.target digunakan untuk melihat nilai target/output dari dataset.

3. Pisahkan Fitur dan Target
```python
x = data.data
y = data.target
x
y
```
Dataset Linnerud memiliki fitur dan target dalam bentuk array NumPy. x = data.data menyimpan fitur (input), sedangkan y = data.target menyimpan target (output). Baris x dan y digunakan untuk menampilkan isi masing-masing.

4. Bagi Data Menjadi Data Latih dan Uji
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```
Data dibagi menjadi 80% untuk pelatihan dan 20% untuk pengujian menggunakan train_test_split. Ini penting agar model dapat diuji pada data yang belum pernah dilihat.

5. Buat dan Latih Model Pohon Keputusan
```python
dtree = DecisionTreeClassifier(max_depth=3)
dtree.fit(x_train, y_train)
```
Membuat model pohon keputusan dengan kedalaman maksimum 3 menggunakan DecisionTreeClassifier. Kemudian model dilatih pada data latih (x_train, y_train) dengan .fit().

6. Visualisasi Model dalam Bentuk Pohon
```python
plt.figure(figsize=(30, 20))
plot_tree(dtree, filled=True)
plt.show()
```
Model pohon keputusan divisualisasikan menggunakan plot_tree, dengan ukuran gambar diperbesar menggunakan figsize. Opsi filled=True memberi warna untuk memudahkan membaca hasil klasifikasi. plt.show() digunakan untuk menampilkan gambar pohon.


