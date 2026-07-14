# Financial Intelligence System v1

A machine learning project for classifying financial news headlines into **positive**, **neutral**, or **negative** sentiment.

The goal of this project is to build a production-ready financial sentiment intelligence system, starting with classical NLP baselines and later upgrading to transformer-based financial language models.

> **Status:** Classical baseline complete | FastAPI inference API complete locally |
> Docker deployment pending

**Test Accuracy:** 0.7441 | **Test Macro F1:** 0.7077  
**Majority-Class Accuracy Baseline:** 0.5916  
The selected model improves over the majority-class accuracy baseline by **+15.25 percentage points**.

## Project Status

Current stage: **Classical baseline model completed and served through a local FastAPI API**

Completed:

* Data loading and cleaning
* Duplicate and conflicting-label handling
* Removal log for auditability
* Stratified development/test split
* TF-IDF + Logistic Regression baseline
* TF-IDF + Linear SVM baseline
* Final test evaluation
* Error analysis on wrong predictions
* Best baseline model saved with joblib
* FastAPI inference service
* Single prediction endpoint: /predict
* Batch prediction endpoint: /predict_batch
* Health check endpoint: /health
* Model metadata endpoint: /model-info
* API input validation with Pydantic
* API tests with pytest and FastAPI TestClient

Planned next:

* Dockerization
* Deployment
* FinBERT comparison

## Problem Statement

Financial news contains large volumes of short text that investors, analysts, and financial platforms need to interpret quickly.

This project classifies financial headlines into sentiment categories:

* `positive`
* `neutral`
* `negative`

The system is designed as a first version of a financial news triage tool that can later be exposed through an API.

## Dataset

Dataset used:

* FinancialPhraseBank

The dataset contains financial news sentences labeled by sentiment.

Final cleaned class distribution:

| Sentiment | Proportion |
| --------- | ---------: |
| neutral   |     59.16% |
| positive  |     28.29% |
| negative  |     12.55% |

The dataset is imbalanced, with `neutral` as the majority class. Because of this, **macro F1** is used as the main evaluation metric instead of accuracy alone.

## Data Cleaning

Cleaning steps performed:

* Standardized schema to `text` and `sentiment`
* Checked for null values
* Removed exact duplicate rows
* Manually resolved conflicting labels
* Removed obvious boilerplate, greetings, broken fragments, and contextless rows
* Preserved short but meaningful financial statements
* Created a removal log for auditability

Important cleaning principle:

> Short text was not removed just because it was short. Rows were removed only when they lacked standalone financial or news meaning.

## Split Strategy

A fixed stratified split was created:

| Split           | Purpose                              |
| --------------- | ------------------------------------ |
| Development set | Cross-validation and model selection |
| Test set        | Final untouched evaluation           |

Split ratio:

* 85% development data
* 15% final test data

The split was stratified to preserve the original class distribution across both sets.

The original row index was preserved as `original_index` for traceability and error analysis.

## Majority Baseline

The majority class is `neutral`.

Majority-class baseline accuracy:

| Baseline               | Accuracy |
| ---------------------- | -------: |
| Always predict neutral |   59.16% |

Any useful model should meaningfully outperform this baseline.

## Models Trained

Two classical NLP baselines were trained using TF-IDF features:

1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM

Both models were evaluated using manual Stratified 5-Fold Cross-Validation on the development set.

## Cross-Validation Results

| Model                        | Accuracy Mean | Macro F1 Mean | Macro Precision Mean | Macro Recall Mean |
| ---------------------------- | ------------: | ------------: | -------------------: | ----------------: |
| TF-IDF + Logistic Regression |        0.7568 |        0.7148 |               0.7160 |            0.7148 |
| TF-IDF + Linear SVM          |        0.7729 |        0.7294 |               0.7446 |            0.7180 |

The TF-IDF + Linear SVM model performed best and was selected as the final classical baseline.

## Final Test Results

The selected TF-IDF + Linear SVM model was trained on the full development set and evaluated once on the untouched test set.

| Metric        |  Score |
| ------------- | -----: |
| Test Accuracy | 0.7441 |
| Test Macro F1 | 0.7077 |

Classification report:

| Class    | Precision | Recall | F1-score | Support |
| -------- | --------: | -----: | -------: | ------: |
| negative |      0.66 |   0.71 |     0.69 |      91 |
| neutral  |      0.79 |   0.82 |     0.81 |     428 |
| positive |      0.66 |   0.59 |     0.63 |     204 |

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

## Error Analysis

The main error pattern was confusion between `positive` and `neutral`.

Error counts:

| True Label | Predicted Label | Count |
| ---------- | --------------- | ----: |
| positive   | neutral         |    72 |
| neutral    | positive        |    54 |
| neutral    | negative        |    22 |
| negative   | neutral         |    19 |
| positive   | negative        |    11 |
| negative   | positive        |     7 |

Main observations:

* The model often misses subtle positive sentiment written in a factual business-news style.
* The model sometimes overreacts to business-growth keywords such as `increase`, `global`, `agreement`, or `stake`.
* The model sometimes misses directionality in financial comparisons, such as moving from loss to profit.
* TF-IDF + Linear SVM is useful as a classical baseline, but it does not deeply understand financial context, comparison direction, or implicit business impact.

## Current Best Model

Best baseline model:

```text
TF-IDF + Linear SVM
```

Saved model artifact:

```text
models/best_tfidf_linear_svm.joblib
```

The saved artifact is a full sklearn Pipeline containing:

```text
TfidfVectorizer + LinearSVC
```

This allows raw text to be passed directly into the model for prediction.

## Repository Structure

```text
data/
├── raw/
│   └── original dataset files
│
├── processed/
│   ├── financial_phrasebank_cleaned.csv
│   ├── dev_data.csv
│   └── test_data.csv

app/
├── __init__.py
└── main.py

tests/
└── test_api.py

models/
└── best_tfidf_linear_svm.joblib

notebooks/
├── 01_data_cleaning.ipynb
├── 02_data_splitting.ipynb
└── 03_baseline_models.ipynb

reports/
├── removed_rows_log.csv
├── baseline_model_comparison.csv
├── final_test_metrics.csv
├── test_predictions.csv
└── error_analysis_sample.csv
```

## FastAPI Inference API

The trained TF-IDF + Linear SVM pipeline is served through a FastAPI application.

Available endpoints:

| Method | Endpoint         | Purpose                                            |
| ------ | ---------------- | -------------------------------------------------- |
| GET    | `/health`        | Check API and model loading status                 |
| GET    | `/model-info`    | Return model name, version, path, and classes      |
| POST   | `/predict`       | Predict sentiment for one financial headline       |
| POST   | `/predict_batch` | Predict sentiment for multiple financial headlines |

Example request to `/predict`:

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

Example request to `/predict_batch`:

```json
{
  "texts": [
    "Operating profit fell compared with the previous year.",
    "The company signed a new contract worth EUR 10 million.",
    "Financial terms were not disclosed."
  ]
}
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

The API validates blank inputs and rejects empty strings after whitespace stripping.


## How to Run

Install dependencies:

```bash
pip install pandas scikit-learn joblib fastapi uvicorn pytest httpx
```

Run notebooks in order:

```text
01_data_cleaning.ipynb
02_data_splitting.ipynb
03_baseline_models.ipynb
```

Run the FastAPI app from the project root:

```bash
uvicorn app.main:app --reload
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

Run API tests:

```bash
python -m pytest
```

Current test status:

```text
6 passed
```

## Key Learnings

This project demonstrates:

* Text classification with financial news
* Data cleaning and audit logging
* Handling imbalanced classes
* Stratified train/test splitting
* Manual Stratified 5-Fold Cross-Validation
* TF-IDF feature extraction
* Logistic Regression and Linear SVM baselines
* Macro F1 based model comparison
* Final test-set evaluation
* Error analysis for model interpretation
* Saving sklearn pipelines with joblib
*  Building a FastAPI inference API around a saved sklearn pipeline
* Request and response validation with Pydantic
* API testing with pytest and FastAPI TestClient
* Handling single and batch model predictions

## Next Steps

Planned improvements:

1. Dockerize the FastAPI service
2. Add a production-style `requirements.txt`
3. Add API usage examples with curl
4. Deploy the API
5. Compare classical baselines against FinBERT
6. Add model latency comparison
7. Write final model card and project report

## Project Positioning

This project is a first version of a financial sentiment intelligence system. The current version establishes a reliable classical machine learning baseline before moving toward transformer-based models and production deployment.
