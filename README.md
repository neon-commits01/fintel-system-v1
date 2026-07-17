# Fin Intelligence System — V1

![Python](https://img.shields.io/badge/Python-3.13-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)
![Docker](https://img.shields.io/badge/Container-Docker-blue)
![Deployment](https://img.shields.io/badge/Deployment-Render-purple)

A financial news sentiment classification API that converts raw financial headlines into structured sentiment labels: **positive**, **neutral**, or **negative**.

This is the first version of a larger financial NLP system. V1 focuses on building a reliable classical machine learning baseline, serving it through a FastAPI inference API, containerizing it with Docker, and deploying it publicly on Render.

---

## Live API

**Swagger API Docs:**  
[https://fintel-system-v1.onrender.com/docs](https://fintel-system-v1.onrender.com/docs)

**Health Check:**  
[https://fintel-system-v1.onrender.com/health](https://fintel-system-v1.onrender.com/health)

> Note: This is an API-first ML deployment project. A user-facing frontend is not included in V1.

---

## Project Status

**V1 Complete**

Completed:

- Data cleaning and preprocessing
- Duplicate and conflicting-label handling
- Stratified development/test split
- TF-IDF + Logistic Regression baseline
- TF-IDF + Linear SVM baseline
- Final test-set evaluation
- Error analysis
- Saved sklearn pipeline with `joblib`
- FastAPI inference API
- API tests with `pytest`
- Dockerized service
- Public deployment on Render

Planned:

- V2: FinBERT comparison
- V3: Financial document intelligence

---

## The Problem

Financial news moves fast. Analysts, investors, and financial platforms often need to process large volumes of headlines and quickly understand whether the news is positive, negative, or neutral.

This project automates that first triage step.

Given a financial headline, the API returns one of three sentiment labels:

```text
positive
neutral
negative
```

Example:

```text
Input:  Operating profit fell compared with the previous year.
Output: negative
```

---

## What It Does

The deployed API exposes a trained sentiment classifier through FastAPI.

Example request:

```json
{
  "text": "Operating profit fell compared with the previous year."
}
```

Example response:

```json
{
  "sentiment": "negative",
  "model_name": "TF-IDF + Linear SVM",
  "model_version": "v1-baseline"
}
```

---

## System Overview

```text
Financial Headline
        ↓
FastAPI Endpoint
        ↓
Pydantic Input Validation
        ↓
Saved sklearn Pipeline
TF-IDF Vectorizer + Linear SVM
        ↓
Sentiment Prediction
positive / neutral / negative
```

Deployment flow:

```text
GitHub Repository
        ↓
Render Web Service
        ↓
Docker Build
        ↓
FastAPI App Starts
        ↓
Saved Model Loads
        ↓
Public API Available
```

---

## Dataset

Dataset used:

```text
FinancialPhraseBank
```

The original dataset contains **4,846 financial sentences** labeled as positive, neutral, or negative.

Final cleaned class distribution:

| Sentiment | Proportion |
|---|---:|
| neutral | 59.16% |
| positive | 28.29% |
| negative | 12.55% |

The dataset is imbalanced, with `neutral` as the majority class. Because of this, macro F1-score is used along with accuracy.

---

## Data Preparation

The data preparation process focused on creating a clean and traceable dataset before modeling.

Main cleaning steps:

- Standardized column names to `text` and `sentiment`
- Checked missing values
- Removed exact duplicate rows
- Manually reviewed same-text different-label conflicts
- Removed greetings, boilerplate, broken fragments, and contextless rows
- Preserved short but meaningful financial statements
- Maintained a removal log for auditability

Important cleaning principle:

```text
Short text was not removed just because it was short.
Rows were removed only when they lacked standalone financial or news meaning.
```

Examples of removed rows:

```text
All are welcome .
Welcome !
All rights reserved .
Why not subscribe to the magazine ?
```

Examples of short but meaningful rows that were kept:

```text
Terms were not disclosed .
The order was worth EUR 8mn .
Cargo volume grew by 7 % .
EPS dropped to EUR0.2 from EUR0.3 .
```

---

## Split Strategy

A fixed stratified split was created:

| Split | Purpose |
|---|---|
| Development set | Cross-validation and model selection |
| Test set | Final untouched evaluation |

Split ratio:

```text
85% development data
15% final test data
```

The split was stratified to preserve the original class distribution across both sets.

The original row index was preserved as `original_index` for traceability and error analysis.

---

## Majority Baseline

The majority class is:

```text
neutral
```

Majority-class baseline accuracy:

| Baseline | Accuracy |
|---|---:|
| Always predict neutral | 0.5916 |

A useful model should meaningfully outperform this baseline.

---

## Modeling Approach

Two classical NLP baselines were trained using TF-IDF features:

1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM

Both models were evaluated using stratified 5-fold cross-validation on the development set.

| Model | CV Accuracy | CV Macro F1 | Notes |
|---|---:|---:|---|
| TF-IDF + Logistic Regression | 0.7568 | 0.7148 | Strong baseline |
| TF-IDF + Linear SVM | 0.7729 | 0.7294 | Selected model |

The Linear SVM model was selected because it performed best during cross-validation and is well-suited for high-dimensional sparse text features produced by TF-IDF.

The final model was saved as a full sklearn Pipeline:

```text
TfidfVectorizer + LinearSVC
```

Saved model artifact:

```text
models/best_tfidf_linear_svm.joblib
```

---

## Final Test Results

The selected model was trained on the full development set and evaluated once on the untouched test set.

| Metric | Score |
|---|---:|
| Test Accuracy | 0.7441 |
| Test Macro F1 | 0.7077 |
| Majority-Class Accuracy Baseline | 0.5916 |

The model improves over the majority-class accuracy baseline by:

```text
+15.25 percentage points
```

Classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| negative | 0.66 | 0.71 | 0.69 | 91 |
| neutral | 0.79 | 0.82 | 0.81 | 428 |
| positive | 0.66 | 0.59 | 0.63 | 204 |

Confusion matrix:

```text
[[ 65  19   7]
 [ 22 352  54]
 [ 11  72 121]]
```

Class order:

```text
negative, neutral, positive
```

Main weakness:

```text
positive ↔ neutral confusion
```

---

## Error Analysis

Wrong predictions were manually reviewed to understand model behavior.

Most common error types:

| True Label | Predicted Label | Count |
|---|---|---:|
| positive | neutral | 72 |
| neutral | positive | 54 |
| neutral | negative | 22 |
| negative | neutral | 19 |
| positive | negative | 11 |
| negative | positive | 7 |

Main failure patterns:

1. **Positive-neutral boundary**
   - Positive business developments written in a factual tone were often predicted as neutral.

2. **Keyword overreaction**
   - Words like `increase`, `growth`, `profit`, or `agreement` sometimes pushed the model toward positive even when the full sentence was neutral or negative.

3. **Missed comparison direction**
   - The model sometimes failed to understand financial improvement or decline across reporting periods.

4. **Mixed financial signals**
   - Headlines containing both positive and negative elements were difficult for the classical model.

Example:

```text
A sentence mentioning "loss" may still be positive if the company moved from loss to profit.
A sentence mentioning "profit" may still be negative if profit declined.
```

These errors show why a contextual financial language model such as FinBERT is a logical V2 upgrade.

---

## FastAPI Inference API

The trained model is served through a FastAPI application.

Available endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service health check |
| GET | `/model-info` | Model version and metadata |
| POST | `/predict` | Classify a single headline |
| POST | `/predict_batch` | Classify multiple headlines |

The API validates:

- blank input
- whitespace-only input
- batch input containing empty strings

Interactive API documentation:

```text
https://fintel-system-v1.onrender.com/docs
```

---

## API Usage

### Health Check

```bash
curl https://fintel-system-v1.onrender.com/health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### Model Info

```bash
curl https://fintel-system-v1.onrender.com/model-info
```

Example response:

```json
{
  "model_name": "TF-IDF + Linear SVM",
  "model_version": "v1-baseline",
  "model_path": ".../models/best_tfidf_linear_svm.joblib",
  "classes": [
    "negative",
    "neutral",
    "positive"
  ]
}
```

---

### Single Prediction

Endpoint:

```text
POST /predict
```

Example request:

```bash
curl -X POST "https://fintel-system-v1.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Operating profit fell compared with the previous year.\"}"
```

Example response:

```json
{
  "sentiment": "negative",
  "model_name": "TF-IDF + Linear SVM",
  "model_version": "v1-baseline"
}
```

---

### Batch Prediction

Endpoint:

```text
POST /predict_batch
```

Example request:

```bash
curl -X POST "https://fintel-system-v1.onrender.com/predict_batch" \
  -H "Content-Type: application/json" \
  -d "{\"texts\":[\"Operating profit fell compared with the previous year.\",\"The company signed a new contract worth EUR 10 million.\",\"Financial terms were not disclosed.\"]}"
```

Example response:

```json
{
  "predictions": [
    "negative",
    "positive",
    "neutral"
  ],
  "model_name": "TF-IDF + Linear SVM",
  "model_version": "v1-baseline"
}
```

---

## Important API Note

Opening this URL directly in a browser:

```text
https://fintel-system-v1.onrender.com/predict
```

may return:

```json
{
  "detail": "Method Not Allowed"
}
```

This is expected because `/predict` is a POST endpoint, not a GET endpoint.

Use Swagger UI for browser-based testing:

```text
https://fintel-system-v1.onrender.com/docs
```

---

## Run Locally

Clone the repository:

```bash
git clone https://github.com/neon-commits01/fintel-system-v1.git
cd fintel-system-v1
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI app:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## Run With Docker

Build the Docker image:

```bash
docker build -t fintel-v1 .
```

Run the container:

```bash
docker run -p 8000:8000 fintel-v1
```

Open:

```text
http://127.0.0.1:8000/docs
```

The Dockerfile uses dynamic port binding for deployment:

```dockerfile
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

This allows local execution on port `8000` and cloud execution using Render's provided `PORT` variable.

---

## Tests

Run tests:

```bash
python -m pytest
```

Current test status:

```text
6 passed
```

Tests cover:

- health check
- model metadata
- single prediction
- batch prediction
- empty input validation
- batch empty input validation

---

## Project Structure

```text
fintel-system-v1/
├── app/
│   ├── __init__.py
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── best_tfidf_linear_svm.joblib
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_data_splitting.ipynb
│   └── 03_baseline_models.ipynb
├── reports/
│   ├── removed_rows_log.csv
│   ├── baseline_model_comparison.csv
│   ├── final_test_metrics.csv
│   ├── test_predictions.csv
│   └── error_analysis_sample.csv
├── tests/
│   └── test_api.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Built With

| Area | Tools |
|---|---|
| Language | Python |
| Data Processing | pandas, NumPy |
| ML/NLP | scikit-learn, TF-IDF, Linear SVM |
| API | FastAPI |
| Validation | Pydantic |
| Testing | pytest, FastAPI TestClient |
| Serialization | joblib |
| Containerization | Docker |
| Deployment | Render |

---

## Limitations

Current limitations:

- No user-facing frontend
- No probability or confidence score
- No FinBERT comparison yet
- No live news ingestion
- No entity extraction
- No explanation generation
- No RAG or document grounding
- Classical model has limited understanding of deep financial context

---

## What’s Next

### V2 — FinBERT Upgrade

Compare the current classical baseline against a transformer model pretrained on financial text.

Goals:

- Improve contextual understanding
- Reduce positive-neutral confusion
- Better handle financial comparison direction
- Compare performance and latency against the classical baseline

### V3 — Financial Document Intelligence

Extend beyond headlines into longer financial documents such as:

- RBI circulars
- NSE/BSE announcements
- earnings call transcripts
- annual report sections
- regulatory filings

Potential features:

- document summarization
- entity extraction
- event classification
- citation-grounded financial Q&A
- RAG-based document intelligence

---

## Project Positioning

This project is the first version of a larger financial NLP system.

V1 establishes:

- a cleaned financial sentiment dataset
- a strong classical NLP baseline
- reproducible model evaluation
- error analysis
- a saved model artifact
- a FastAPI inference API
- Docker-based deployment
- a public cloud-hosted API

Future versions will focus on contextual financial language models and document-level intelligence.
