# Kode

```
# 1. Instalasi dan Inisialisasi PySpark di Google Colab
!pip install pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace
import matplotlib.pyplot as plt
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import pandas as pd

# Inisialisasi Spark Session
spark = SparkSession.builder.appName("GPU Analysis").getOrCreate()

# 2. Baca Data GPU dari GitHub dengan Encoding
url = "https://raw.githubusercontent.com/steprtm/datagpu/main/gpudatas.csv"
data_gpu_pd = pd.read_csv(url, encoding="ISO-8859-1")
data_gpu = spark.createDataFrame(data_gpu_pd)
data_gpu.show(5)

# Menghapus duplikasi
data_gpu = data_gpu.dropDuplicates()

# Membersihkan kolom 'price' dari simbol '$' dan mengonversinya menjadi float
# Menghapus baris dengan nilai 'price' yang kosong atau tidak valid (contoh: "N/A")
data_gpu = data_gpu.filter(col("price").isNotNull() & (col("price") != "N/A"))
data_gpu = data_gpu.withColumn("price", regexp_replace(col("price"), "[$,]", "").cast("float"))

# Membersihkan kolom 'memory' dari satuan 'GB' dan mengonversinya menjadi float
data_gpu = data_gpu.withColumn("memory", regexp_replace(col("memory"), " GB", "").cast("float"))

# Membersihkan kolom 'clock_speed' dari satuan 'MHz' dan mengonversinya menjadi float
data_gpu = data_gpu.withColumn("clock_speed", regexp_replace(col("clock_speed"), " MHz", "").cast("float"))

# Menangani nilai yang hilang dengan nilai default
data_gpu = data_gpu.na.fill({
    "price": 0.0,       # Mengganti nilai kosong pada kolom 'price' dengan 0
    "memory": 0.0,      # Mengganti nilai kosong pada kolom 'memory' dengan 0
    "clock_speed": 0.0, # Mengganti nilai kosong pada kolom 'clock_speed' dengan 0
    "brand": "Unknown"  # Mengganti nilai kosong pada kolom 'brand' dengan 'Unknown'
})

# Menghapus baris dengan harga yang bernilai 0 setelah melakukan pengisian nilai default
data_gpu = data_gpu.filter(col("price") > 0)


# 4. Analisis Eksploratif Data GPU
# Statistik deskriptif
data_gpu.describe().show()

# 4.1 Visualisasi Distribusi Harga
# Mengonversi DataFrame Spark ke Pandas untuk keperluan visualisasi
price_data = data_gpu.select("price").toPandas()

plt.figure(figsize=(10, 6))
plt.hist(price_data["price"], bins=30, color='skyblue', edgecolor='black')
plt.title("Distribusi Harga GPU")
plt.xlabel("Harga ($)")
plt.ylabel("Frekuensi")
plt.grid(axis='y')
plt.show()

# 5. Clustering GPU Berdasarkan Harga
# Menggunakan K-Means untuk segmentasi harga
# Menyiapkan fitur untuk clustering
assembler = VectorAssembler(inputCols=["price"], outputCol="features")
data_gpu_features = assembler.transform(data_gpu)

# Menjalankan K-Means clustering
kmeans = KMeans(k=3, seed=1)
model = kmeans.fit(data_gpu_features)
clusters = model.transform(data_gpu_features)

# Visualisasi hasil clustering
cluster_data = clusters.select("price", "prediction").toPandas()
plt.figure(figsize=(10, 6))
plt.scatter(cluster_data["price"], cluster_data["prediction"], c=cluster_data["prediction"], cmap="viridis", s=50, alpha=0.6)
plt.title("Clustering Harga GPU")
plt.xlabel("Harga ($)")
plt.ylabel("Cluster")
plt.colorbar(label='Cluster')
plt.grid()
plt.show()

# 6. Prediksi Harga Menggunakan Regresi Linear
# Menggunakan kolom 'memory' dan 'clock_speed' sebagai fitur untuk memprediksi harga
assembler = VectorAssembler(inputCols=["memory", "clock_speed"], outputCol="features")
data_gpu_regression = assembler.transform(data_gpu)

# Model regresi linear
lr = LinearRegression(featuresCol="features", labelCol="price")
model = lr.fit(data_gpu_regression)
predictions = model.transform(data_gpu_regression)

# Menampilkan hasil prediksi
predictions.select("memory", "clock_speed", "price", "prediction").show(5)

# Visualisasi Prediksi Harga
# Plot prediksi vs harga aktual
pred_data = predictions.select("price", "prediction").toPandas()

plt.figure(figsize=(10, 6))
plt.scatter(pred_data["price"], pred_data["prediction"], alpha=0.6, color='coral', edgecolor='black')
plt.plot([pred_data["price"].min(), pred_data["price"].max()], [pred_data["price"].min(), pred_data["price"].max()], 'k--', lw=2)
plt.title("Prediksi Harga GPU vs Harga Aktual")
plt.xlabel("Harga Aktual ($)")
plt.ylabel("Prediksi Harga ($)")
plt.grid()
plt.show()
```
