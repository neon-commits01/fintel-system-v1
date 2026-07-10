# Financial Intelligence System v1

A machine learning project for classifying financial news headlines into **positive**, **neutral**, or **negative** sentiment.

The goal of this project is to build a production-ready financial sentiment intelligence system, starting with classical NLP baselines and later upgrading to transformer-based financial language models.

> **Status:** Classical baseline complete | 
> FastAPI deployment in progress

**Macro F1: 0.7077** | 
Beats majority baseline (0.5916) by 11.6 points

## Project Status

Current stage: **Classical baseline modeling completed**

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

Planned next:

* FastAPI prediction service
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

## How to Run

Install dependencies:

```bash
pip install pandas scikit-learn joblib
```

Run notebooks in order:

```text
01_data_cleaning.ipynb
02_data_splitting.ipynb
03_baseline_models.ipynb
```

To load the saved model:

```python
import joblib

model = joblib.load("models/best_tfidf_linear_svm.joblib")

sample_text = ["Operating profit fell compared with the previous year."]
prediction = model.predict(sample_text)

print(prediction)
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

## Next Steps

Planned improvements:

1. Build a FastAPI inference service
2. Add `/predict` and `/predict_batch` endpoints
3. Add request validation with Pydantic
4. Dockerize the service
5. Deploy the model
6. Compare classical baselines against FinBERT
7. Add model latency comparison
8. Write final model card and project report

## Project Positioning

This project is a first version of a financial sentiment intelligence system. The current version establishes a reliable classical machine learning baseline before moving toward transformer-based models and production deployment.
