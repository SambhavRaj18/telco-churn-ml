# Telco Customer Churn Prediction — End-to-End ML Pipeline & Analytics Dashboard

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.0-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8422-emerald?style=flat)](https://github.com/SambhavRaj18/telco-churn-ml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/SambhavRaj18/telco-churn-ml)

> An end-to-end production Machine Learning pipeline, REST API, and interactive decision-support interface for predicting customer churn risk on the IBM Telco Customer Churn dataset.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Tech Stack](#-tech-stack)
- [End-to-End System Architecture](#-end-to-end-system-architecture)
- [Dataset Summary](#-dataset-summary)
- [Data Cleaning & Type Standardization](#-data-cleaning--type-standardization)
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [Correlation Analysis](#-correlation-analysis)
- [Feature Preparation & Preprocessing Pipeline](#-feature-preparation--preprocessing-pipeline)
- [Model Training & Baseline Comparison](#-model-training--baseline-comparison)
- [Cross-Validation & Hyperparameter Tuning](#-cross-validation--hyperparameter-tuning)
- [ROC-AUC & Discrimination Analysis](#-roc-auc--discrimination-analysis)
- [Probability Threshold Optimization (Recall Tuning)](#-probability-threshold-optimization-recall-tuning)
- [Final Test Performance](#-final-test-performance)
- [Confusion Matrix & Business Interpretation](#-confusion-matrix--business-interpretation)
- [Serialized Pipeline Creation](#-serialized-pipeline-creation)
- [Flask REST API Endpoint](#-flask-rest-api-endpoint)
- [TelcoShield AI Frontend Dashboard](#-telcoshield-ai-frontend-dashboard)
- [Docker Containerization & Engineering Notes](#-docker-containerization--engineering-notes)
- [End-to-End Verification](#-end-to-end-verification)
- [Project Directory Structure](#-project-directory-structure)
- [How to Run (Local & Docker)](#-how-to-run-local--docker)
- [Key Engineering & ML Concepts](#-key-engineering--ml-concepts)
- [Project Limitations](#-project-limitations)
- [Future Enhancements](#-future-enhancements)
- [Author](#-author)

---

## 🎯 Project Overview

Customer churn refers to the loss of subscribers or customers who discontinue their service contracts. In telecommunications, acquiring new customers costs significantly more than retaining existing ones; accurate early identification of churn risk enables retention teams to proactively intervene with tailored offers.

### Problem Formulation
- **ML Task:** Supervised Binary Classification.
- **Target Variable ($y$):**
  - `0` $\rightarrow$ **No Churn** (Customer retained)
  - `1` $\rightarrow$ **Churn** (Customer left within the last month)
- **Objective:** Train an ensemble model optimized for high churn recall without excessive false alarm degradation, package the model and preprocessing into a single serialized artifact, and serve inference through a validated Flask REST API and containerized web dashboard.

---

## 💻 Tech Stack

| Domain | Technologies & Libraries | Purpose |
| :--- | :--- | :--- |
| **Language & Core** | Python 3.10 | Core programming runtime |
| **Data Science & ML** | Pandas, NumPy, Scikit-Learn 1.6.0, Joblib | Data manipulation, feature engineering, pipeline serialization, model evaluation |
| **Data Visualization** | Matplotlib | Exploratory data analysis and distribution plotting |
| **Backend & API** | Flask | RESTful prediction API, payload validation, template rendering |
| **Frontend & UI** | HTML5, Vanilla JavaScript, Tailwind CSS, FontAwesome | Reactive web UI, dynamic form dependencies, animated risk gauge |
| **DevOps & Container** | Docker, Git, GitHub | Application containerization, environment isolation, version control |

---

## 🏗️ End-to-End System Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────┐
│                           1. DATA PIPELINE & TRAINING                          │
│                                                                                │
│  Raw CSV Dataset ──► Data Cleaning & Type Fix ──► EDA & Stratified Split (80/20)│
│                                                          │                     │
│  Pipeline: StandardScaler + OneHotEncoder ◄──────────────┘                     │
│         │                                                                      │
│         ▼                                                                      │
│  5-Fold Stratified CV ──► Random Forest Tuning ──► Threshold Optimization (0.33)│
│                                                          │                     │
│  final_pipeline.pkl (Serialized Inference Artifact) ◄────┘                     │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼─────────────────────────────────────────┐
│                        2. PRODUCTION SERVING & CONTAINER                       │
│                                                                                │
│  TelcoShield AI Dashboard (HTML5 / Tailwind / JS)                              │
│         │                                                                      │
│         ▼  (AJAX JSON Payload: 19 Features)                                    │
│  Flask Backend (app.py @ POST /predict)                                        │
│         ├── Request Validation (Keys, Numeric Types, Categorical Enums)        │
│         ├── Model Inference (final_pipeline.pkl)                               │
│         └── Threshold Logic: p >= 0.33 -> Churn | p < 0.33 -> No Churn         │
│                                                                                │
│  Docker Container (python:3.10-slim @ 0.0.0.0:5000)                            │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset Summary

The project utilizes the **IBM Telco Customer Churn** dataset.

- **Total Records:** 7,043 customer accounts
- **Raw Features:** 21 columns (1 ID, 19 input features, 1 target)
- **Target Distribution:**
  - `No Churn`: **5,174** (73.46%)
  - `Churn`: **1,869** (26.54%)
  - *Class Imbalance Ratio:* $\approx 2.77 : 1$

### Feature Breakdown

| Category | Features |
| :--- | :--- |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Connectivity & Phone** | `tenure`, `PhoneService`, `MultipleLines`, `InternetService` |
| **Add-on Services** | `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| **Contract & Billing** | `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |

> **Repository Note:** The raw CSV file (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) is excluded from version control via `.gitignore`. Place the dataset in `data/` or `telco-customer-churn/` for local notebook reproduction.

---

## 🧹 Data Cleaning & Type Standardization

Inspection of the raw dataset revealed a critical data type issue in `TotalCharges`:

1. **Object Storage:** `TotalCharges` was loaded as an `object` (string) column rather than a floating-point numeric.
2. **Hidden Whitespace Values:** 11 rows contained blank whitespace strings (`" "`), causing direct numeric conversion to fail.
3. **Zero-Tenure Root Cause:** Cross-referencing these 11 rows revealed that every customer with a missing `TotalCharges` had `tenure == 0` (brand new accounts prior to their first monthly billing cycle).

### Cleaning Implementation

```python
# Strip whitespace and coerce invalid string tokens to NaN
df["TotalCharges"] = df["TotalCharges"].str.strip()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Impute zero for accounts with tenure == 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)
```

After cleaning, `TotalCharges` is fully continuous with zero nulls and verified data integrity.

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed across numerical distributions and categorical crosstabs to identify baseline associations with customer churn.

### Numerical Distribution Summary

| Numerical Feature | No Churn Group | Churn Group | Observed Pattern |
| :--- | :--- | :--- | :--- |
| **`tenure`** | Median $\approx 38$ months | Median $\approx 10$ months | Churners are heavily concentrated in early account lifecycle stages. |
| **`MonthlyCharges`** | Median $\approx \$64.40$ | Median $\approx \$79.65$ | Churners exhibit higher monthly bill amounts on average. |
| **`TotalCharges`** | Median $\approx \$1,683.60$ | Median $\approx \$703.55$ | Lower overall cumulative spend due to truncated tenure. |

### Categorical Churn Rate Observations

| Category | High Churn Rate Segment | Lower Churn Rate Segment |
| :--- | :--- | :--- |
| **Contract Type** | Month-to-Month ($\approx 42.7\%$) | Two-Year ($\approx 2.8\%$), One-Year ($\approx 11.3\%$) |
| **Payment Method** | Electronic Check ($\approx 45.3\%$) | Credit Card ($\approx 15.2\%$), Bank Transfer ($\approx 16.7\%$) |
| **Internet Service** | Fiber Optic ($\approx 41.9\%$) | DSL ($\approx 19.0\%$), No Internet ($\approx 7.4\%$) |
| **Support Add-ons** | No Online Security ($\approx 41.8\%$) / No Tech Support ($\approx 41.6\%$) | Active Online Security ($\approx 14.6\%$) / Tech Support ($\approx 15.2\%$) |
| **Demographics** | Senior Citizens ($\approx 41.7\%$) | Non-Seniors ($\approx 23.6\%$) |

> **Analytical Discipline:** These patterns represent observed statistical correlations in historical data and must not be conflated with direct causality.

---

## 📈 Correlation Analysis

Pearson correlation coefficients were computed across numerical attributes to measure linear co-dependencies:

| Feature Pair | Pearson Correlation ($r$) | Domain Interpretation |
| :--- | :---: | :--- |
| **`tenure` – `TotalCharges`** | **0.8262** | Strong positive correlation: cumulative bill increases linearly with tenure. |
| **`MonthlyCharges` – `TotalCharges`** | **0.6512** | Moderate-to-strong positive correlation. |
| **`tenure` – `MonthlyCharges`** | **0.2479** | Weak positive correlation: tenure is not dictated by monthly charge tier. |

---

## ⚙️ Feature Preparation & Preprocessing Pipeline

### 1. Identifier Removal & Target Encoding
- `customerID` was dropped as it carries zero generalizable predictive signal.
- Binary target mapping: `y = df["Churn"].map({"No": 0, "Yes": 1})`.

### 2. Stratified Train-Test Split
An 80/20 train-test split was performed using `stratify=y` to preserve exact class ratios across partitions:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```
- **Training Set:** 5,634 samples (4,139 No Churn / 1,495 Churn)
- **Test Set:** 1,409 samples (1,035 No Churn / 374 Churn)

### 3. ColumnTransformer Preprocessing Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)
```

- **`StandardScaler`**: Scales continuous variables to zero mean and unit variance.
- **`OneHotEncoder(handle_unknown="ignore")`**: Converts categorical levels to binary indicator columns while preventing runtime failure on unseen categories during inference.

---

## 🤖 Model Training & Baseline Comparison

Multiple model architectures were trained and compared on the standardized training data:

| Model Architecture | Validation / Test Accuracy | ROC-AUC | Engineering Assessment |
| :--- | :---: | :---: | :--- |
| **Logistic Regression** | **80.55%** | **0.8421** | Strong linear baseline with stable discrimination. |
| **Baseline Decision Tree** | 72.11% | 0.6477 | Severe unconstrained tree overfitting; low generalizability. |
| **Depth-Limited Decision Tree** | 77.22% | 0.7810 | Improved regularization via depth pruning. |
| **Tuned Random Forest** | **76.86%** *(at 0.33 threshold)* | **0.8422** | **Selected Model:** Optimal probability discrimination and robust ensemble variance reduction. |

---

## 🔬 Cross-Validation & Hyperparameter Tuning

Hyperparameter tuning for the Random Forest classifier was conducted using **5-Fold Stratified Cross-Validation** strictly on the training set (`X_train`, `y_train`):

### Final Tuned Hyperparameters

```python
from sklearn.ensemble import RandomForestClassifier

rf_tuned = RandomForestClassifier(
    n_estimators=500,
    max_depth=7,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42
)
```

- **`n_estimators=500`**: High ensemble averaging stabilizes probability variance.
- **`max_depth=7`**: Restricts maximum tree depth to prevent leaf overfitting.
- **`min_samples_leaf=2` & `min_samples_split=5`**: Enforces minimum node density for leaf splits.
- **`max_features="sqrt"`**: Subsamples feature space per split for tree de-correlation.

---

## 📉 ROC-AUC & Discrimination Analysis

The tuned Random Forest model achieved a test-set **ROC-AUC of 0.8422 (84.22%)**.

```text
                      ROC-AUC Concept
1.0 ┌──────────────────────────────────────────────┐
    │                                    .---------│ (AUC = 0.8422)
    │                             .-----'          │
TPR │                       .----'                 │
    │                 .----'                       │
    │           .----'                             │
    │     .----'                                   │
0.0 └──────────────────────────────────────────────┘
    0.0                     FPR                  1.0
```

### Technical Distinction
- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)** measures the probability that the model ranks a randomly chosen positive instance (churner) higher than a randomly chosen negative instance (non-churner) across all possible thresholds ($0.0 \le p \le 1.0$).
- ROC-AUC is **threshold-independent** and is not equivalent to accuracy or probability calibration.

---

## ⚖️ Probability Threshold Optimization (Recall Tuning)

Standard classification algorithms use an arbitrary default decision threshold of $p = 0.50$. In retention analytics, this default is often misaligned with business economics:

$$\text{Cost}(\text{False Negative}) \gg \text{Cost}(\text{False Positive})$$

- **False Negative (Missed Churner):** Customer leaves undetected; entire customer lifetime value (LTV) is lost.
- **False Positive (False Alarm):** A loyal customer receives a proactive discount or loyalty call; minor operational outreach cost.

### Threshold Selection Process
Using out-of-fold cross-validation probabilities, precision-recall trade-offs were evaluated. Lowering the decision threshold to **`0.33`** captures significantly more churners:

$$\text{Decision Rule: } \hat{y} = \begin{cases} \text{"Churn"}, & \text{if } P(y=1 \mid \mathbf{x}) \ge 0.33 \\ \text{"No Churn"}, & \text{if } P(y=1 \mid \mathbf{x}) < 0.33 \end{cases}$$

> **Operating Parameter Note:** The `0.33` threshold is a project-specific operating threshold chosen to prioritize churn recall. It is not an arbitrary industry constant.

---

## 🏆 Final Test Performance

Evaluated on the locked test set ($N = 1,409$) with threshold $p = 0.33$:

| Metric | Score | Nature of Metric | Business Meaning |
| :--- | :---: | :--- | :--- |
| **Accuracy** | **76.86%** | Threshold-dependent | Overall correct classification rate across both classes. |
| **Precision** | **54.71%** | Threshold-dependent | When flagged as churn, 54.71% of customers were actual churners. |
| **Recall** | **74.60%** | Threshold-dependent | **74.60% of all actual churners were successfully detected.** |
| **F1 Score** | **63.12%** | Threshold-dependent | Harmonic mean of Precision and Recall. |
| **ROC-AUC** | **84.22%** | Threshold-independent | Global ranking and class separation capability. |

---

## 🧩 Confusion Matrix & Business Interpretation

```text
                       Actual Positive (Churn)     Actual Negative (No Churn)
Predicted Churn (p ≥ 0.33)       279 [TP]                     231 [FP]
Predicted No Churn (p < 0.33)     95 [FN]                     804 [TN]
```

- **True Positives (TP = 279):** 279 churning customers correctly identified for retention outreach.
- **True Negatives (TN = 804):** 804 loyal customers correctly classified with zero retention overhead.
- **False Positives (FP = 231):** 231 non-churning customers flagged; minimal cost of retention offer.
- **False Negatives (FN = 95):** 95 churning customers missed out of 374 total churners (substantially reduced from $>180$ misses at a 0.50 threshold).

---

## 📦 Serialized Pipeline Creation

To eliminate training-serving skew, the fitted `preprocessor` and `rf_tuned` estimator were bundled into a single Scikit-Learn `Pipeline`:

```python
from sklearn.pipeline import Pipeline
import joblib

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", rf_tuned)
])

# Serialize pipeline artifact
joblib.dump(final_pipeline, "final_pipeline.pkl")
```

The exported file `final_pipeline.pkl` encapsulates scaling parameters, one-hot category mappings, and all 500 decision trees for atomic single-line inference.

---

## 🔌 Flask REST API Endpoint

The backend is exposed via Flask in [`app.py`](file:///d:/udemy/ML1/app.py).

### API Specification

- **Health Check / UI Route:** `GET /` $\rightarrow$ Renders `templates/index.html`
- **Inference Route:** `POST /predict`
- **Request Content-Type:** `application/json`

### Request Payload (19 Required Features)

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.85,
  "TotalCharges": 179.70
}
```

### Successful Response Payload (`200 OK`)

```json
{
  "churn_probability": 0.7360412891045231,
  "prediction": "Churn"
}
```

### Robust Backend Validation
- **400 Bad Request:** Triggered on missing JSON body, missing keys, invalid numeric ranges/types, or categorical strings not matching fitted `OneHotEncoder` categories.
- **500 Internal Server Error:** Safely caught inside `try/except` blocks returning sanitized JSON error messages without leaking Python tracebacks.

---

## 🖥️ TelcoShield AI Frontend Dashboard

The frontend in [`templates/index.html`](file:///d:/udemy/ML1/templates/index.html) is built with modern HTML5, Tailwind CSS, and Vanilla JavaScript.

```text
┌────────────────────────────────────────────────────────────────────────┐
│ TelcoShield AI • Customer Churn Prediction Engine                      │
├───────────────────────────────────┬────────────────────────────────────┤
│ 1. Demographics                   │ PREDICTION OUTCOME                 │
│    Gender, Senior, Partner...     │                                    │
│ 2. Phone Services                 │       ╭─────────────╮              │
│    PhoneService, MultipleLines... │      │    73.60%   │ (SVG Gauge)   │
│ 3. Internet & Add-on Services     │       ╰─────────────╯              │
│    InternetService, Security...   │  [⚠️ CHURN RISK DETECTED]          │
│ 4. Contract & Financials          │                                    │
│    Contract, Payment, Charges...  │  Threshold: p ≥ 0.33               │
│                                   │                                    │
│ [ RUN CHURN RISK ANALYSIS ]       │  ► Technical Details (API Payload) │
└───────────────────────────────────┴────────────────────────────────────┘
```

### Frontend Engineering Highlights
1. **Dynamic Parent-Child Dependency Logic:**
   - `PhoneService = "No"` $\rightarrow$ Automatically locks `MultipleLines` to `"No phone service"` and disables the input.
   - `InternetService = "No"` $\rightarrow$ Automatically locks all 6 add-on services to `"No internet service"` and disables them.
   - Dynamic option rebuilding guarantees zero stale invalid values on repeated switching.
2. **Strict Client-Side Validation:** Explicit string trim checks prevent `Number("") === 0` bugs; enforces integer `tenure` $\in [0, 72]$ and non-negative charges.
3. **Continuous Risk Visualization:** SVG circular progress meter animated to the exact continuous churn probability percentage.
4. **Stale Prediction Protection:** Modifying any form parameter after inference immediately displays an `OUTDATED` badge and warning banner.
5. **Sample Profiles:** 1-click loading for standard test vectors (`Sample Profile A`, `Sample Profile B`, `Sample Profile C`).
6. **No Pseudo-AI Inventions:** Displays verified probability and threshold rules without hardcoded or fabricated feature attributions.

---

## 🐳 Docker Containerization & Engineering Notes

The application is containerized with Docker for repeatable, environment-agnostic deployment.

### `Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY final_pipeline.pkl .
COPY templates ./templates

EXPOSE 5000

CMD ["python", "app.py"]
```

### `requirements.txt`

```text
Flask
pandas
scikit-learn==1.6.0
joblib
```

### Engineering & Troubleshooting Notes

| Challenge Encountered | Root Cause | Engineering Solution |
| :--- | :--- | :--- |
| **Unpickling Attribute Error** (`_RemainderColsList`) | Version drift between serialized pipeline Scikit-Learn version and container environment. | Pinned `scikit-learn==1.6.0` explicitly in `requirements.txt`. |
| **Container Port Inaccessibility** | Flask by default binds to `127.0.0.1` (loopback internal to container). | Updated Flask entrypoint to `app.run(host="0.0.0.0", port=5000, debug=False)`. |
| **Docker Build Cache Retention** | Docker reused stale layer cache when updating dependency files. | Built image with `docker build --no-cache -t telco-churn-app .`. |

---

## 🧪 End-to-End Verification

The complete flow was verified across both local and containerized environments:

$$\text{Browser Form} \longrightarrow \text{Docker Port 5000:5000} \longrightarrow \text{Flask API} \longrightarrow \text{ColumnTransformer} \longrightarrow \text{RandomForest} \longrightarrow \text{JSON Response}$$

- **Sample Profile A (Month-to-month, Fiber optic, Electronic check):**
  - Result: `p = 0.7360 (73.60%)` $\rightarrow$ `Prediction: Churn`
- **Sample Profile B (Two year, DSL, Auto-card):**
  - Result: `p = 0.0241 (2.41%)` $\rightarrow$ `Prediction: No Churn`
- **Sample Profile C (Phone-only, No internet):**
  - Result: `p = 0.0457 (4.57%)` $\rightarrow$ `Prediction: No Churn`

Both local execution and containerized execution produce bit-for-bit identical inference probabilities.

---

## 📂 Project Directory Structure

```text
telco-churn-ml/
│
├── app.py                     # Flask REST API backend & inference routes
├── final_pipeline.pkl         # Serialized Scikit-Learn pipeline artifact
├── notebook.ipynb             # Jupyter notebook for data cleaning, EDA, tuning
├── requirements.txt           # Pinned production runtime dependencies
├── Dockerfile                 # Docker container build specification
├── .dockerignore              # Excluded files during Docker build context
├── .gitignore                 # Excluded files for Git version control
├── README.md                  # Comprehensive project documentation
│
├── telco-customer-churn/      # Local dataset directory (gitignored)
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
└── templates/
    └── index.html             # Production TelcoShield AI frontend template
```

---

## 🚀 How to Run (Local & Docker)

### Option 1: Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SambhavRaj18/telco-churn-ml.git
   cd telco-churn-ml
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the Flask server:**
   ```bash
   python app.py
   ```

5. **Access the application:** Open `http://127.0.0.1:5000` in your web browser.

---

### Option 2: Docker Setup

1. **Build the Docker image:**
   ```bash
   docker build --no-cache -t telco-churn-app .
   ```

2. **Run the container with port forwarding:**
   ```bash
   docker run -p 5000:5000 telco-churn-app
   ```

3. **Access the application:** Open `http://localhost:5000` in your web browser.

---

## 🧠 Key Engineering & ML Concepts

### Data Engineering & Statistical Analysis
- **Missing Value Forensics:** Identified zero-tenure root cause for whitespace strings in `TotalCharges`.
- **Stratification:** Maintained class distribution fidelity ($73.5\% / 26.5\%$) across partitions.
- **Correlation Boundaries:** Analyzed Pearson colinearities without making causal claims.

### Machine Learning & Decision Theory
- **Ensemble Variance Reduction:** Utilized 500-tree bagging to reduce individual tree overfitting.
- **Threshold Tuning vs Accuracy:** Optimized threshold ($p = 0.33$) for high churn recall ($74.60\%$) based on retention economics.
- **ROC-AUC Invariance:** Evaluated global ranking discrimination independent of decision cutoffs.

### Production Engineering & Deployment
- **Unified Pipeline Serialization:** Prevented train-serve skew by packaging `ColumnTransformer` with `RandomForestClassifier`.
- **API Defensive Programming:** Validated categorical sets, data types, and null inputs.
- **Container Isolation:** Standardized system environment using pinned dependencies in `python:3.10-slim`.

---

## ⚠️ Project Limitations

1. **Static Historical Data:** Model reflects stationary distributions from historical data; does not account for macro-economic shifts or external competitor pricing changes.
2. **Tabular Scope:** Does not process real-time customer service call transcripts, chat logs, or unstructured support tickets.
3. **Threshold Sensitivity:** The `0.33` operating cutoff is tied to specific retention campaign assumptions; shifting cost ratios requires recalibration.

---

## 🔮 Future Enhancements

- [ ] **SHAP Integration:** Implement local TreeSHAP explanations for feature contribution transparency per prediction.
- [ ] **Data Drift Monitoring:** Incorporate Evidently AI or Prometheus metrics to monitor prediction distribution drift in production.
- [ ] **Automated CI/CD Pipeline:** Implement GitHub Actions for automated unit testing, Docker image building, and container registry publishing.
- [ ] **Cloud Deployment:** Deploy containerized service to AWS ECS / Google Cloud Run with autoscaling.

---

## 👤 Author

**Sambhav Raj**  
- **GitHub:** [@SambhavRaj18](https://github.com/SambhavRaj18)  
- **Repository:** [https://github.com/SambhavRaj18/telco-churn-ml](https://github.com/SambhavRaj18/telco-churn-ml)
