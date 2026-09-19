# Telco Customer Churn Prediction — End-to-End ML Pipeline & Analytics Dashboard

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.0-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8422-emerald?style=flat)](https://github.com/SambhavRaj18/telco-churn-ml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/SambhavRaj18/telco-churn-ml)

> An end-to-end machine learning application, containerized inference API, and interactive decision-support dashboard for predicting customer churn risk on the IBM Telco Customer Churn dataset.

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
- [Model Training & Comparison](#-model-training--comparison)
- [Cross-Validation & Hyperparameter Tuning](#-cross-validation--hyperparameter-tuning)
- [ROC-AUC & Discrimination Analysis](#-roc-auc--discrimination-analysis)
- [Probability Threshold Optimization (Recall Tuning)](#-probability-threshold-optimization-recall-tuning)
- [Final Test Performance](#-final-test-performance)
- [Confusion Matrix & Operational Interpretation](#-confusion-matrix--operational-interpretation)
- [Serialized Pipeline Creation](#-serialized-pipeline-creation)
- [Flask REST API Endpoint](#-flask-rest-api-endpoint)
- [TelcoShield AI Frontend Dashboard](#-telcoshield-ai-frontend-dashboard)
- [Docker Containerization & Troubleshooting History](#-docker-containerization--troubleshooting-history)
- [End-to-End Verification](#-end-to-end-verification)
- [Repository Structure](#-repository-structure)
- [How to Run (Local & Docker)](#-how-to-run-local--docker)
- [Key Engineering & ML Concepts](#-key-engineering--ml-concepts)
- [Project Limitations](#-project-limitations)
- [Future Enhancements](#-future-enhancements)
- [Author](#-author)

---

## 🎯 Project Overview

Customer churn refers to subscribers discontinuing their service contracts with a telecommunications provider. Anticipating which customers are at risk of churning allows organizations to engage in proactive retention outreach.

### Problem Formulation
- **ML Task:** Supervised Binary Classification.
- **Target Variable ($y$):**
  - `0` $\rightarrow$ **No Churn** (Customer retained)
  - `1` $\rightarrow$ **Churn** (Customer departed within the observation window)
- **Objective:** Build an end-to-end machine learning pipeline that preprocesses customer data, trains an ensemble classifier with tuned probability thresholds prioritizing churn recall, packages the pipeline into a single serialized artifact, and serves predictions via a validated Flask API and containerized web dashboard.

---

## 💻 Tech Stack

| Domain | Technologies & Libraries | Purpose |
| :--- | :--- | :--- |
| **Language & Core** | Python 3.10 | Core programming runtime |
| **Data Science & ML** | Pandas, NumPy, Scikit-Learn 1.6.0, Joblib | Data processing, feature transformation, model training, evaluation, and artifact serialization |
| **Data Visualization** | Matplotlib | Exploratory data analysis and distribution plotting |
| **Backend & API** | Flask | HTTP request routing, input validation, template serving, and JSON inference endpoint |
| **Frontend & UI** | HTML5, Vanilla JavaScript, Tailwind CSS (CDN), FontAwesome (CDN) | Interactive dashboard, dynamic form dependency controls, animated SVG probability gauge |
| **Container & Version Control** | Docker, Git, GitHub | Application containerization, environment reproducibility, source control |

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
│                        2. INFERENCE SERVING & CONTAINER                        │
│                                                                                │
│  TelcoShield AI Dashboard (HTML5 / Tailwind CSS / Vanilla JS)                  │
│         │                                                                      │
│         ▼  (AJAX JSON Payload: 19 Features)                                    │
│  Flask Application (app.py @ POST /predict)                                    │
│         ├── Input Validation (19 Keys, Numeric Types, Categorical Values)      │
│         ├── Pipeline Inference (final_pipeline.pkl)                            │
│         └── Threshold Logic: p >= 0.33 -> Churn | p < 0.33 -> No Churn         │
│                                                                                │
│  Docker Container (python:3.10-slim @ 0.0.0.0:5000)                            │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset Summary

The project utilizes the **IBM Telco Customer Churn** dataset.

- **Total Records:** 7,043 customer accounts
- **Raw Features:** 21 columns (1 ID, 19 model inputs, 1 target)
- **Target Distribution:**
  - `No Churn`: **5,174** (73.46%)
  - `Churn`: **1,869** (26.54%)
  - *Class Ratio:* $\approx 2.77 : 1$

### Feature Categories

| Category | Features |
| :--- | :--- |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Account & Connectivity** | `tenure`, `PhoneService`, `MultipleLines`, `InternetService` |
| **Add-on Services** | `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| **Contract & Billing** | `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |

> **Dataset Storage Note:** The raw CSV file (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) is excluded from this repository via `.gitignore` and is not packaged inside the Docker image. For local training or notebook execution, place the dataset locally under `telco-customer-churn/` or `data/`.

---

## 🧹 Data Cleaning & Type Standardization

Inspection of the raw dataset identified a data type inconsistency in `TotalCharges`:

1. **Object Storage:** `TotalCharges` was loaded as an `object` (string) column rather than a floating-point number.
2. **Blank Whitespace Values:** 11 rows contained blank whitespace strings (`" "`).
3. **Zero-Tenure Alignment:** Cross-referencing these 11 rows showed that each customer had `tenure == 0` (new accounts that had not yet completed a monthly billing cycle).

### Cleaning Implementation

```python
# Strip whitespace and coerce invalid string tokens to NaN
df["TotalCharges"] = df["TotalCharges"].str.strip()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Impute zero for accounts with tenure == 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)
```

After cleaning, `TotalCharges` is fully numeric with zero missing values.

---

## 🔍 Exploratory Data Analysis (EDA)

Exploratory data analysis was conducted to examine numerical distributions and categorical crosstabs against churn outcomes.

### Numerical Distribution Summary

| Numerical Feature | No Churn Group | Churn Group | Observed Association |
| :--- | :--- | :--- | :--- |
| **`tenure`** | Median $\approx 38$ months | Median $\approx 10$ months | Churners in the dataset are concentrated in lower tenure ranges. |
| **`MonthlyCharges`** | Median $\approx \$64.40$ | Median $\approx \$79.65$ | Churners exhibit higher median monthly charges. |
| **`TotalCharges`** | Median $\approx \$1,683.60$ | Median $\approx \$703.55$ | Churners have lower cumulative charges due to shorter tenure. |

### Categorical Churn Rate Summary

| Category | Higher Churn Segment | Lower Churn Segment |
| :--- | :--- | :--- |
| **Contract Type** | Month-to-Month ($\approx 42.7\%$) | Two-Year ($\approx 2.8\%$), One-Year ($\approx 11.3\%$) |
| **Payment Method** | Electronic Check ($\approx 45.3\%$) | Credit Card ($\approx 15.2\%$), Bank Transfer ($\approx 16.7\%$) |
| **Internet Service** | Fiber Optic ($\approx 41.9\%$) | DSL ($\approx 19.0\%$), No Internet ($\approx 7.4\%$) |
| **Support Subscriptions** | No Online Security ($\approx 41.8\%$) / No Tech Support ($\approx 41.6\%$) | Active Online Security ($\approx 14.6\%$) / Tech Support ($\approx 15.2\%$) |
| **Demographics** | Senior Citizens ($\approx 41.7\%$) | Non-Seniors ($\approx 23.6\%$) |

> **Interpretation Note:** These patterns describe historical associations within this specific dataset and do not imply direct causality.

---

## 📈 Correlation Analysis

Pearson correlation coefficients were calculated across numerical features:

| Feature Pair | Pearson Correlation ($r$) | Description |
| :--- | :---: | :--- |
| **`tenure` – `TotalCharges`** | **0.8262** | Strong positive linear correlation as cumulative charges accumulate over tenure. |
| **`MonthlyCharges` – `TotalCharges`** | **0.6512** | Moderate-to-strong positive linear correlation. |
| **`tenure` – `MonthlyCharges`** | **0.2479** | Weak positive linear correlation. |

---

## ⚙️ Feature Preparation & Preprocessing Pipeline

### 1. Identifier Removal & Target Encoding
- `customerID` was removed because it is a unique identifier with no generalizable predictive value.
- Target mapping: `y = df["Churn"].map({"No": 0, "Yes": 1})`.

### 2. Stratified Train-Test Split
An 80/20 train-test split was performed using `stratify=y` to preserve class proportions:

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

### 3. ColumnTransformer Pipeline

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

- **`StandardScaler`**: Centers and scales continuous variables to zero mean and unit variance.
- **`OneHotEncoder(handle_unknown="ignore")`**: Converts categorical variables into binary indicator columns and ignores unseen categories during inference.

---

## 🤖 Model Training & Comparison

The following models were trained and evaluated on the dataset:

| Model | Test Accuracy | ROC-AUC | Description |
| :--- | :---: | :---: | :--- |
| **Logistic Regression** | **80.55%** | **0.8421** | Linear classification baseline. |
| **Baseline Decision Tree** | 72.11% | 0.6477 | Unconstrained single tree exhibiting training set overfitting. |
| **Tuned Random Forest** | **76.86%** *(at threshold 0.33)* | **0.8422** | **Selected Model:** 500-tree ensemble with tuned depth and split parameters. |

---

## 🔬 Cross-Validation & Hyperparameter Tuning

Hyperparameter tuning for the Random Forest model was conducted using **5-Fold Stratified Cross-Validation** on the training data (`X_train`, `y_train`):

### Final Random Forest Configuration

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

- **`n_estimators=500`**: Number of trees in the ensemble.
- **`max_depth=7`**: Limits individual tree depth to reduce leaf-level variance.
- **`min_samples_split=5` & `min_samples_leaf=2`**: Constrains node splitting to maintain minimum sample thresholds.
- **`max_features="sqrt"`**: Subsamples features per split to de-correlate individual trees.

---

## 📉 ROC-AUC & Discrimination Analysis

The tuned Random Forest model achieved a test-set **ROC-AUC of 0.8422 (84.22%)**.

### Technical Meaning
- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)** evaluates the model's ability to rank churning customers above non-churning customers across all possible decision thresholds ($0.0 \le p \le 1.0$).
- ROC-AUC is **threshold-independent** and should not be confused with accuracy or probability calibration.

---

## ⚖️ Probability Threshold Optimization (Recall Tuning)

Standard binary classifiers use a conventional default threshold of $p = 0.50$. For this project, threshold behavior was analyzed using out-of-fold predictions on the training set, evaluating the trade-off between precision, recall, and F1 score.

Because the project objective prioritized identifying a higher proportion of potential churners, the operating classification threshold was adjusted to **`0.33`**:

$$\text{Decision Rule: } \hat{y} = \begin{cases} \text{"Churn"}, & \text{if } P(y=1 \mid \mathbf{x}) \ge 0.33 \\ \text{"No Churn"}, & \text{if } P(y=1 \mid \mathbf{x}) < 0.33 \end{cases}$$

> **Project Parameter Note:** The `0.33` threshold is an operating parameter selected for this project's stated recall-focused objective. It is not a mathematically universal or industry-standard threshold.

---

## 🏆 Final Test Performance

Evaluated on the locked test set ($N = 1,409$) with threshold $p = 0.33$:

| Metric | Score | Metric Type | Description |
| :--- | :---: | :--- | :--- |
| **Accuracy** | **76.86%** | Threshold-dependent | Overall correct classification rate. |
| **Precision** | **54.71%** | Threshold-dependent | Proportion of predicted churners that actually churned. |
| **Recall** | **74.60%** | Threshold-dependent | **Proportion of actual churners successfully identified.** |
| **F1 Score** | **63.12%** | Threshold-dependent | Harmonic mean of Precision and Recall. |
| **ROC-AUC** | **84.22%** | Threshold-independent | Global ranking and class separation metric. |

---

## 🧩 Confusion Matrix & Operational Interpretation

At the `0.33` threshold on the test set ($N = 1,409$):

```text
                       Actual Positive (Churn)     Actual Negative (No Churn)
Predicted Churn (p ≥ 0.33)       279 [TP]                     231 [FP]
Predicted No Churn (p < 0.33)     95 [FN]                     804 [TN]
```

- **True Positives (TP = 279):** Churning customers correctly identified for potential retention action.
- **True Negatives (TN = 804):** Non-churning customers correctly classified.
- **False Positives (FP = 231):** Non-churning customers flagged for potential retention outreach.
- **False Negatives (FN = 95):** Churning customers that were not flagged by the model.

---

## 📦 Serialized Pipeline Creation

The preprocessing steps and tuned Random Forest model were assembled into a single Scikit-Learn `Pipeline`:

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

The exported file `final_pipeline.pkl` encapsulates feature scaling, one-hot encodings, and ensemble trees in a single artifact for deployment.

---

## 🔌 Flask REST API Endpoint

The inference service is implemented in [`app.py`](file:///d:/udemy/ML1/app.py).

### Endpoints
- **UI Route:** `GET /` $\rightarrow$ Serves `templates/index.html`
- **Inference Route:** `POST /predict`
- **Content-Type:** `application/json`

### Example Request Body (19 Features)

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

### Example Response Body (`200 OK`)

```json
{
  "churn_probability": 0.7396,
  "prediction": "Churn"
}
```

### Validation Handling
- **400 Bad Request:** Returned when required keys are missing, numerical values are invalid or negative, or categorical strings do not match expected training categories.
- **500 Internal Server Error:** Managed with defensive exception handling returning JSON error descriptions.

---

## 🖥️ TelcoShield AI Frontend Dashboard

The frontend in [`templates/index.html`](file:///d:/udemy/ML1/templates/index.html) is built with HTML5, Vanilla JavaScript, Tailwind CSS (via CDN), and FontAwesome icons (via CDN).

### Frontend Features
1. **Dynamic Service Dependencies:**
   - Selecting `PhoneService = "No"` locks `MultipleLines` to `"No phone service"` and disables the input.
   - Selecting `InternetService = "No"` locks all 6 add-on services to `"No internet service"` and disables them.
   - Switching options re-enables applicable dropdown choices without leaving invalid combinations.
2. **Client-Side Validation:** Validates that `tenure` is an integer between 0 and 72, charges are non-negative, and no required fields are blank prior to submission.
3. **Continuous Probability Gauge:** Animated SVG circular progress meter representing the exact continuous churn probability percentage.
4. **Stale Prediction Flagging:** If any input is modified after inference, an `OUTDATED` badge and banner notify the user that results reflect previous inputs.
5. **Sample Profiles:** Includes sample profiles (`Sample Profile A`, `Sample Profile B`, `Sample Profile C`) with valid configurations for demonstration.
6. **Technical Inspection Accordion:** Expandable section displaying the raw JSON payload and response.

---

## 🐳 Docker Containerization & Troubleshooting History

The application is containerized using Docker.

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

### Engineering & Troubleshooting History

1. **Initial Missing Dependencies:** `requirements.txt` was initially empty, causing missing library errors during startup (`joblib` / `scikit-learn`). Added all runtime dependencies to `requirements.txt`.
2. **Scikit-Learn Version Compatibility:** Unpickling the model artifact in a container with a newer Scikit-Learn version (1.7.x) caused an `AttributeError` (`_RemainderColsList`). Explicitly pinning `scikit-learn==1.6.0` to match the serialization environment resolved the error.
3. **Container Host Binding:** Flask defaults to listening on `127.0.0.1`, which is accessible only inside the container network namespace. Updated `app.py` to `app.run(host="0.0.0.0", port=5000, debug=False)` to enable external port forwarding.
4. **Build Cache Invalidation:** When updating dependencies, Docker layer caching retained older environments. Rebuilt using `docker build --no-cache -t telco-churn-app .` to ensure clean package installation.

---

## 🧪 End-to-End Verification

The complete application flow was verified across both local Flask execution and containerized Docker execution:

$$\text{Browser Input} \longrightarrow \text{Port 5000 Mapping} \longrightarrow \text{Flask Route} \longrightarrow \text{ColumnTransformer} \longrightarrow \text{Random Forest} \longrightarrow \text{JSON Response} \longrightarrow \text{UI Display}$$

### Verified Consistency Check
- **Customer Record:** `9305-CDSKC` (Actual churn: `Yes`)
- **Local Flask Probability:** **73.96%** (`Prediction: Churn`)
- **Docker Container Probability:** **73.96%** (`Prediction: Churn`)

Both local and Docker execution yielded identical displayed prediction probabilities of **73.96%** for the verified test record.

---

## 📂 Repository Structure

```text
telco-churn-ml/
│
├── app.py                     # Flask inference API & route definitions
├── final_pipeline.pkl         # Serialized Scikit-Learn pipeline artifact
├── notebook.ipynb             # Jupyter notebook containing data analysis & modeling
├── requirements.txt           # Python package dependencies (pinned versions)
├── Dockerfile                 # Docker container build specification
├── .dockerignore              # Files excluded from Docker build context
├── .gitignore                 # Files excluded from Git version control
├── README.md                  # Project documentation
│
└── templates/
    └── index.html             # TelcoShield AI dashboard template
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

5. **Open the application:** Navigate to `http://127.0.0.1:5000` in your browser.

---

### Option 2: Docker Setup

1. **Build the Docker image:**
   ```bash
   docker build --no-cache -t telco-churn-app .
   ```

2. **Run the container:**
   ```bash
   docker run -p 5000:5000 telco-churn-app
   ```

3. **Open the application:** Navigate to `http://localhost:5000` in your browser.

---

## 🧠 Key Engineering & ML Concepts

### Data Analysis & Cleaning
- **Zero-Tenure Imputation:** Addressed whitespace values in `TotalCharges` by identifying zero-tenure customer accounts.
- **Stratified Partitioning:** Preserved target class ratios across train/test splits.
- **Correlation Interpretation:** Evaluated linear associations while maintaining analytical boundaries regarding causality.

### Machine Learning & Evaluation
- **Ensemble Bagging:** Employed 500 trees with restricted depth to manage decision tree variance.
- **Threshold Calibration:** Selected a `0.33` operating threshold based on recall prioritization.
- **ROC-AUC Invariance:** Evaluated discrimination capability across all possible decision cutoffs.

### Engineering & Serving
- **Unified Pipeline Serialization:** Packaged preprocessing and model inference together to prevent training-serving skew.
- **Defensive API Validation:** Implemented type, range, and category validation for incoming JSON payloads.
- **Containerization:** Standardized runtime environment with pinned library versions.

---

## ⚠️ Project Limitations

1. **Static Dataset:** Model reflects historical patterns in the training data and does not capture future market shifts or external price changes.
2. **Tabular Scope:** Limited to structured customer account attributes without unstructured support ticket or call transcript data.
3. **Threshold Context:** The `0.33` operating threshold is specific to this project's recall objective and would require re-evaluation under different operational priorities.

---

## 🔮 Future Enhancements

- [ ] **Explainability Integration:** Incorporate TreeSHAP to provide individual feature contribution breakdowns for predictions.
- [ ] **Model Monitoring:** Implement data drift and prediction distribution monitoring.
- [ ] **Automated Testing:** Add automated unit tests for API routes, validation rules, and pipeline inference.
- [ ] **CI/CD Integration:** Configure GitHub Actions for automated testing and container image building.

---

## 👤 Author

**Sambhav Raj**  
- **GitHub:** [@SambhavRaj18](https://github.com/SambhavRaj18)  
- **Repository:** [https://github.com/SambhavRaj18/telco-churn-ml](https://github.com/SambhavRaj18/telco-churn-ml)
