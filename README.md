# Heart Disease Classification System with MLOps

## Deskripsi Proyek

Proyek ini merupakan implementasi sistem Machine Learning End-to-End untuk melakukan klasifikasi penyakit jantung menggunakan **Heart Disease Cleveland Dataset**.

Model dikembangkan menggunakan algoritma **Random Forest Classifier**, kemudian dilakukan **Hyperparameter Tuning** menggunakan **GridSearchCV** untuk memperoleh performa terbaik.

Selain proses pelatihan model, proyek ini juga menerapkan konsep **MLOps**, meliputi:

- Experiment Tracking menggunakan MLflow
- REST API menggunakan FastAPI
- Model Inference
- Monitoring menggunakan Prometheus
- Visualisasi monitoring menggunakan Grafana

---

# Dataset

Dataset yang digunakan adalah:

**Heart Disease Cleveland Dataset**

Target klasifikasi:

- 0 → No Heart Disease
- 1 → Heart Disease

Jumlah fitur yang digunakan:

1. age
2. sex
3. cp
4. trestbps
5. chol
6. fbs
7. restecg
8. thalach
9. exang
10. oldpeak
11. slope
12. ca
13. thal

---

# Struktur Proyek

```
project/
│
├── artifacts/
│   ├── models/
│   │   ├── best_model.pkl
│   │   └── best_params.json
│   │
│   └── reports/
│       ├── classification_report.txt
│       └── confusion_matrix.csv
│
├── preprocessing/
│   ├── preprocessing.py
│   ├── heart_train.csv
│   ├── heart_test.csv
│   └── scaler.pkl
│
├── monitoring & logging/
│   └── prometheus.yml
│
├── app.py
├── inference.py
├── modelling.py
├── modelling_tuning.py
├── requirements.txt
└── README.md
```

---

# Teknologi yang Digunakan

- Python 3.13
- Pandas
- NumPy
- Scikit-Learn
- Joblib
- FastAPI
- Uvicorn
- MLflow
- Prometheus
- Grafana

---

# Instalasi

Clone repository

```bash
git clone <repository-url>
```

Masuk ke folder project

```bash
cd project
```

Install dependency

```bash
pip install -r requirements.txt
```

---

# Training Model

Melatih model Random Forest tanpa Hyperparameter Tuning

```bash
python modelling.py
```

---

# Hyperparameter Tuning

Melatih model menggunakan GridSearchCV

```bash
python modelling_tuning.py
```

Hasil training akan menghasilkan:

- best_model.pkl
- best_params.json
- classification_report.txt
- confusion_matrix.csv

Seluruh eksperimen juga akan tercatat pada MLflow.

---

# Menjalankan MLflow

```bash
mlflow ui
```

Akses

```
http://127.0.0.1:5000
```

MLflow digunakan untuk menyimpan:

- Parameter
- Metrics
- Model
- Artifacts

---

# Menjalankan API

Jalankan server

```bash
uvicorn app:app --reload
```

API tersedia pada

```
http://localhost:8000
```

Swagger Documentation

```
http://localhost:8000/docs
```

---

# Endpoint API

## GET /

Health Check

---

## POST /predict

Contoh Request

```json
{
  "age": 55,
  "sex": 1,
  "cp": 2,
  "trestbps": 130,
  "chol": 250,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.5,
  "slope": 2,
  "ca": 0,
  "thal": 2
}
```

Contoh Response

```json
{
  "prediction": 1,
  "label": "Heart Disease",
  "probability": {
    "negative": 0.1835,
    "positive": 0.8165
  }
}
```

---

# Monitoring

Prometheus digunakan untuk mengumpulkan metric dari API.

Jalankan Prometheus

```bash
prometheus.exe --config.file="monitoring & logging/prometheus.yml"
```

Dashboard Prometheus

```
http://localhost:9090
```

Metric yang dimonitor:

- prediction_requests_total
- prediction_positive_total
- prediction_negative_total
- prediction_latency_seconds

---

# Grafana

Grafana digunakan untuk memvisualisasikan metric dari Prometheus.

Dashboard yang dibuat meliputi:

- Total Prediction Requests
- Positive Prediction
- Negative Prediction
- Average Prediction Latency

Grafana berjalan pada

```
http://localhost:3000
```

---

# Hasil Model

Model terbaik diperoleh menggunakan Random Forest Classifier dengan Hyperparameter Tuning menggunakan GridSearchCV.

Parameter terbaik disimpan pada:

```
artifacts/models/best_params.json
```

Model terbaik disimpan pada:

```
artifacts/models/best_model.pkl
```

---

# Monitoring Dashboard

Dashboard Grafana menampilkan:

- Total permintaan prediksi
- Jumlah prediksi positif
- Jumlah prediksi negatif
- Rata-rata waktu inferensi

---

# Author

Muhammad Keisa

IBM SkillsBuild AI Engineer

Machine Learning & MLOps Project