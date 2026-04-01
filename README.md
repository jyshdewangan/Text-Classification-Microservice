# Text-Classification-Microservice

An end-to-end deep learning project for binary text classification. This project natively trains a TensorFlow/Keras bidirectional LSTM on the IMDB dataset and automatically pipelines the output into a heavily concurrent `FastAPI` inference REST service architecture. 

It tracks runtime, dynamically version controls model saves through timestamp tagging, applies continuous logging to api predictions, and proves inference latency and throughput under heavy concurrent async stress conditions using `aiohttp`.

---

## 🛠 Features

- **End-To-End Training Pipeline**: Utilizes an automated `Sequential` model with Vocabulary `Embedding`, `Bidirectional LSTM` mapping, `tf.data.Dataset` batching, and `EarlyStopping` validation mechanics.
- **Automated Model Versioning**: Saves artifact iterations seamlessly into a `models/` directory natively tagging them via `v1_{YYYYMMDD_HHMMSS}.keras` alongside a master `_latest` fallback.
- **FastAPI Deployment**: Leverages `uvicorn` to statically load the TensorFlow models into server RAM on startup, exposing a robust `/predict` endpoint scaling horizontally to batch requests.
- **Inference & Latency Logging**: Evaluates HTTP request/response metrics with pure Python logging integration (`app.py`), outputting inference predictions, their confidence bounds, and milliseconds (ms) request latency directly to `stdout`.
- **Concurrent Load Testing**: Uses an asynchronous python `aiohttp` routine (`stress_test.py`) to hit the prediction endpoints natively with 200 payloads and over 50 concurrent connections simultaneously, asserting strict latency (p95, p99) and server throughput capability limits.

---

## 🚀 Setup & Execution

It is heavily recommended to use [uv](https://github.com/astral-sh/uv) to manage requirements.

### 1. Create Virtual Environment and Install Dependencies
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Generate Versioned Model & Training Pipeline
Executes the data downloads, early stopping validations, F1 performance evaluations, and the automated saving mechanisms:
```bash
python train.py
```

### 3. Deploy API Node Server
Starts the `FastAPI` instance. It will load `models/text_classification_model_latest.keras` and the IMDB vocabulary map directly to RAM.
```bash
uvicorn app:app --port 8000
```

### 4. Execute Asynchronous Inference Load Testing
Tests the prediction endpoint using the `aiohttp` script under intense concurrent parallel load to compute API efficiency metrics natively.
```bash
python stress_test.py
```

### 5. Simple Sandbox Testing
Use `infer.py` directly for any small one-off raw string evaluations natively bypassing the API.
```bash
python infer.py
```

---

## 📊 Pipeline Metrics Showcase

Following evaluation on the testing set (10 Epoch limit):
- **Accuracy**: 83%
- **Precision**: 81.5%
- **Recall**: 86.1%
- **F1-Score**: 83.7%

During asynchronous stress testing against the FastAPI service (50 concurrent workers evaluating 2,000 sentences):
- **Execution Speed**: Successfully resolved 200 payload batches in **5.83 seconds**.
- **Service Throughput**: Averaged **34 requests per second**, equaling roughly **~2.9ms per individual sentence** prediction.
- **Latency Distribution**: P50: 3620ms | P95: 5818ms

---
