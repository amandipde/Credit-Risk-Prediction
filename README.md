# 💳 End-to-End Credit Risk Prediction

An end-to-end Machine Learning project for predicting the probability of loan default from applicant financial, employment, loan, and credit-history information.

The project covers the complete ML workflow:

**Data → EDA → Preprocessing → Model Comparison → Cross-Validation → Hyperparameter Tuning → Threshold Optimization → Probability Calibration → SHAP Explainability → Model Serialization → FastAPI → Interactive Web UI → Render Deployment**

---

## 🎯 Project Goal

Credit risk assessment is an important problem for financial institutions. Before approving a loan, a lender needs to estimate the likelihood that an applicant will default.

The goal of this project is to build a machine learning system that takes information about a loan applicant and predicts:

1. The **probability of loan default**
2. A **High Risk / Low Risk** classification based on a decision threshold

The project is designed not only as a machine learning experiment, but as a complete **production-oriented ML application**, where the trained model is exposed through an API and integrated with an interactive web interface.

The final application allows a user to enter an applicant's information and receive a real-time credit-risk assessment.

---

# 📊 Dataset

The dataset used in this project is the **Credit Risk Prediction** dataset available on Hugging Face:

🔗 **Dataset:** [Alfazril/credit-risk-prediction on Hugging Face](https://huggingface.co/Alfazril/credit-risk-prediction?utm_source=chatgpt.com)

The dataset contains **32,581 loan applications** and **12 variables**.

The target variable is:

- `loan_status = 0` → Non-default
- `loan_status = 1` → Default

The dataset contains both numerical and categorical information about borrowers and their loans.

### Features

| Feature | Description |
|---|---|
| `person_age` | Age of the applicant |
| `person_income` | Annual income |
| `person_home_ownership` | Home ownership status |
| `person_emp_length` | Employment length in years |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Loan risk grade |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Interest rate |
| `loan_status` | Target variable: default / non-default |
| `loan_percent_income` | Loan amount relative to annual income |
| `cb_person_default_on_file` | Previous default indicator |
| `cb_person_cred_hist_length` | Length of credit history |

The dataset contains missing values in variables such as employment length and loan interest rate, which are handled during the preprocessing stage.

---

# 🔎 Exploratory Data Analysis

The first stage of the project involved understanding the structure and statistical properties of the dataset.

The analysis included:

- Dataset structure and data types
- Missing-value analysis
- Descriptive statistics
- Target-class distribution
- Numerical feature distributions
- Categorical feature distributions
- Skewness analysis
- Outlier investigation
- Relationships between features and the target
- Class imbalance analysis

A key observation is that the target variable is imbalanced:

- **Non-default:** 25,473
- **Default:** 7,108

This corresponds approximately to:

- **78.2% non-default**
- **21.8% default**

Therefore, accuracy alone is not sufficient to evaluate the model. Metrics such as **ROC-AUC, PR-AUC, precision, recall and F1-score** are also considered.

---

# ⚙️ Machine Learning Pipeline

The machine learning workflow was designed to avoid data leakage by placing preprocessing operations inside Scikit-learn pipelines.

The data was divided into:

```python
X = df.drop(columns="loan_status")
y = df["loan_status"]
```

A stratified train-test split was used so that the class distribution was preserved:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

---

## 🧹 Data Preprocessing

The preprocessing strategy depends on the type of model.

### Numerical Features

Missing numerical values are handled using median imputation.

For Logistic Regression, numerical variables are additionally standardized using `StandardScaler`.

```text
Numerical Features
        ↓
Median Imputation
        ↓
StandardScaler
```

### Categorical Features

Categorical missing values are handled using a dedicated `"Missing"` category and categorical variables are converted into numerical representations using one-hot encoding.

```text
Categorical Features
        ↓
Missing-value Imputation
        ↓
One-Hot Encoding
```

The complete preprocessing is implemented using `ColumnTransformer` and integrated directly into the ML pipelines.

This ensures that preprocessing is learned only from the training data during cross-validation and model fitting.

---

# ⚖️ Handling Class Imbalance

Because default cases represent a minority of the dataset, class imbalance was explicitly considered.

For Logistic Regression:

```python
class_weight="balanced"
```

was used.

For XGBoost, the class imbalance was incorporated through:

```python
scale_pos_weight
```

calculated from the training data:

```python
scale_pos_weight = (
    y_train.value_counts()[0] /
    y_train.value_counts()[1]
)
```

The resulting value is approximately:

```text
scale_pos_weight ≈ 3.63
```

The test set was **not artificially balanced**. It was kept representative of the original distribution for unbiased evaluation.

---

# 🤖 Models Evaluated

Several machine learning approaches were considered to understand the performance trade-offs between linear models, bagging methods, and gradient boosting algorithms.

### 1. Logistic Regression

Logistic Regression was used as the baseline classification model.

It provides an interpretable linear baseline against which more complex models can be compared.

### 2. Random Forest

Random Forest was considered as a non-linear ensemble method based on bagging and decision trees.

### 3. XGBoost

XGBoost was selected as the main model because of its strong performance on the tabular credit-risk dataset.

### 4. LightGBM

LightGBM was also considered as an efficient gradient-boosting alternative.

### 5. CatBoost

CatBoost was investigated as another gradient-boosting approach, particularly relevant for datasets containing categorical variables.

---

# 📈 Model Evaluation

Five-fold stratified cross-validation was used to evaluate the models.

The evaluation included:

- ROC-AUC
- PR-AUC / Average Precision
- Accuracy
- Precision
- Recall
- F1-score

PR-AUC was particularly important because the default class is the minority class.

---

## Logistic Regression Results

The Logistic Regression baseline achieved:

| Metric | Mean | Std |
|---|---:|---:|
| ROC-AUC | 0.8705 | ±0.0049 |
| PR-AUC | 0.7135 | ±0.0065 |
| Accuracy | 0.8116 | ±0.0046 |
| Precision | 0.5448 | ±0.0087 |
| Recall | 0.7781 | ±0.0087 |
| F1 | 0.6408 | ±0.0036 |

The model provides a useful baseline but does not capture the non-linear relationships as effectively as the boosting models.

---

# 🚀 XGBoost

XGBoost provided a substantial improvement over the Logistic Regression baseline.

The initial cross-validation results were:

| Metric | Mean | Std |
|---|---:|---:|
| ROC-AUC | **0.9472** | ±0.0045 |
| PR-AUC | **0.9026** | ±0.0057 |
| Accuracy | **0.9182** | ±0.0027 |
| Precision | **0.8195** | ±0.0151 |
| Recall | **0.7974** | ±0.0110 |
| F1 | **0.8081** | ±0.0046 |

These results motivated the selection of **XGBoost as the final production model**.

---

# 🔧 Hyperparameter Tuning

After comparing the baseline models, XGBoost was further optimized using `GridSearchCV`.

The hyperparameters explored included:

```python
param_grid = {
    "model__n_estimators": [200, 300, 500],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.05, 0.1],
    "model__min_child_weight": [1, 3],
    "model__colsample_bytree": [0.8, 1.0],
    "model__subsample": [0.8, 0.9, 1.0],
    "model__gamma": [0, 1, 2, 3, 4, 5]
}
```

The optimization was performed using:

```python
scoring="average_precision"
```

rather than accuracy.

This was chosen because PR-AUC is more informative when evaluating the minority default class.

The tuned model was then retained as:

```python
best_xgb
```

---

# 🎚️ Decision Threshold Optimization

A machine learning classifier does not necessarily need to use the default probability threshold of `0.5`.

In credit-risk applications, the threshold represents a business decision: how conservative should the system be when classifying an applicant as high risk?

The Precision-Recall curve was therefore used to investigate different thresholds.

The selected threshold was approximately:

```text
0.699
```

At this threshold, the observed performance was approximately:

```text
F1-score  = 0.844
Precision = 0.959
Recall    = 0.754
```

This illustrates an important part of the project: the model's probability output and the final business decision are treated as separate concepts.

The optimized threshold is saved separately as:

```text
best_threshold.pkl
```

so that the same decision rule can be used by the deployed API.

---

# 🎯 Probability Calibration

The raw probability estimates produced by a gradient-boosting model are not necessarily perfectly calibrated.

Therefore, probability calibration was investigated using `CalibratedClassifierCV`.

The calibration process improved the Brier score:

```text
Before calibration : 0.06134
After calibration  : 0.05231
```

This represents an approximately **14.7% reduction in Brier score**.

A lower Brier score indicates that the predicted probabilities are better aligned with the observed outcomes.

This is particularly relevant for a credit-risk application because the probability itself can be more informative than a simple binary classification.

---

# 🔍 Model Explainability with SHAP

Model performance alone is not sufficient for a credit-risk system.

It is also important to understand which variables influence the model's predictions.

SHAP (**SHapley Additive exPlanations**) was used to investigate the contribution of the input features to the XGBoost predictions.

The XGBoost model was extracted from the Scikit-learn pipeline and SHAP values were calculated after applying the trained preprocessing transformation.

This provides:

- Global feature importance
- Direction and magnitude of feature effects
- Understanding of which features drive predictions
- A basis for explaining individual model decisions

Because categorical variables are one-hot encoded, categorical levels initially appear as separate features in the SHAP analysis.

---

# 💾 Model Serialization

After training and tuning, the final model was serialized using `joblib`.

Two important artifacts are stored:

```text
credit_risk_model.pkl
best_threshold.pkl
```

### `credit_risk_model.pkl`

Contains the trained XGBoost pipeline, including the preprocessing and model components.

### `best_threshold.pkl`

Contains the optimized classification threshold used to convert the predicted probability into a final risk classification.

Keeping these artifacts separate makes the inference process reproducible and ensures that the deployed application uses exactly the same model and decision threshold established during model development.

---

# 🌐 FastAPI Backend

The trained model is exposed through a REST API using **FastAPI**.

The backend receives loan-application information, converts it into a Pandas DataFrame, passes it through the saved ML pipeline, and returns the predicted probability and risk classification.

The API accepts the following information:

```text
person_age
person_income
person_home_ownership
person_emp_length
loan_intent
loan_grade
loan_amnt
loan_int_rate
loan_percent_income
cb_person_default_on_file
cb_person_cred_hist_length
```

The target variable `loan_status` is deliberately **not** part of the API input because it is the quantity being predicted.

---

# 🔌 Prediction Endpoint

The main prediction endpoint is:

```http
POST /predict/
```

A request contains applicant information in JSON format.

The API returns:

```json
{
    "default_probability": 0.XX,
    "default_prediction": 0,
    "threshold": 0.699,
    "Result": "Low Risk"
}
```

where:

- `default_probability` is the model's estimated probability of default
- `default_prediction` is the binary decision
- `threshold` is the optimized decision threshold
- `Result` is the human-readable risk classification

The application therefore separates:

**Probability estimation → Decision threshold → Risk classification**

---

# 🖥️ Interactive Web Interface

A lightweight frontend was developed using:

- HTML
- CSS
- JavaScript

No frontend framework is required.

The interface allows users to enter the applicant's information and submit it directly to the FastAPI backend.

The JavaScript frontend sends the application to:

```javascript
fetch("/predict/", {
    method: "POST",
    ...
})
```
and dynamically displays:

- Risk classification
- Default probability
- Decision threshold
- Visual probability meter
- Model decision

The interface also includes an example application for quickly demonstrating the system.

<img width="1224" height="834" alt="Screenshot from 2026-10-08 15-41-18" src="https://github.com/user-attachments/assets/47ce69bc-9b0d-4c7a-af2c-2b4e206bd0eb" />

---

# 📁 Project Structure

```text
End_to_End_Credit_Risk_Prediction/
│
├── main.py
├── credit_risk_model.pkl
├── best_threshold.pkl
├── requirements.txt
├── render.yaml
│
└── static/
    ├── index.html
    ├── style.css
    └── script.js
```

### File Description

| File | Purpose |
|---|---|
| `main.py` | FastAPI application, model loading and prediction endpoint |
| `credit_risk_model.pkl` | Serialized trained XGBoost ML pipeline |
| `best_threshold.pkl` | Optimized probability threshold |
| `requirements.txt` | Python dependencies required by the application |
| `render.yaml` | Render deployment configuration |
| `static/index.html` | Frontend user interface |
| `static/style.css` | Styling and responsive layout |
| `static/script.js` | Frontend interaction and API communication |

---

# 🚀 Deployment

The application is designed to be deployed on **Render**.

The production server is started using:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

The `render.yaml` file provides the deployment configuration.

The deployment architecture is:

```text
                    ┌──────────────────────┐
                    │      User / Browser   │
                    └──────────┬───────────┘
                               │
                               │ HTTP
                               ▼
                    ┌──────────────────────┐
                    │    FastAPI Backend   │
                    │       main.py        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  XGBoost ML Pipeline │
                    │ credit_risk_model.pkl│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Default Probability  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Decision Threshold   │
                    │ best_threshold.pkl   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ High Risk / Low Risk │
                    └──────────────────────┘
```

The frontend is served by FastAPI itself through:

```python
app.mount(
    "/",
    StaticFiles(directory="static", html=True),
    name="static"
)
```

This allows the application to serve the web interface and API from the same deployment.

---

# 🛠️ Technology Stack

### Data Science

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

### Machine Learning

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

### Model Analysis

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- Brier score
- Precision-Recall curves
- SHAP

### API

- FastAPI
- Pydantic
- Uvicorn

### Model Persistence

- Joblib

### Frontend

- HTML
- CSS
- JavaScript

### Deployment

- Render
- GitHub

---

# 🔄 End-to-End Workflow

The complete project can be summarized as:

```text
Hugging Face Dataset
        │
        ▼
Data Understanding
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Data Cleaning & Validation
        │
        ▼
Train / Test Split
        │
        ▼
Preprocessing Pipeline
        │
        ▼
Model Comparison
        │
        ├── Logistic Regression
        ├── Random Forest
        ├── XGBoost
        ├── LightGBM
        └── CatBoost
        │
        ▼
Cross-Validation
        │
        ▼
XGBoost Selection
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Threshold Optimization
        │
        ▼
Probability Calibration
        │
        ▼
SHAP Explainability
        │
        ▼
Model Serialization
        │
        ├── credit_risk_model.pkl
        └── best_threshold.pkl
        │
        ▼
FastAPI
        │
        ▼
Interactive Web Interface
        │
        ▼
Render Deployment
```

---

# 📌 Key Results

The project demonstrates that the XGBoost model substantially outperformed the Logistic Regression baseline.

The initial cross-validation results were:

| Model | ROC-AUC | PR-AUC | F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.8705 | 0.7135 | 0.6408 |
| XGBoost | **0.9472** | **0.9026** | **0.8081** |

After threshold optimization, the selected decision threshold produced an F1-score of approximately:

```text
0.844
```

with:

```text
Precision ≈ 0.959
Recall    ≈ 0.754
```

The probability calibration experiment also reduced the Brier score from approximately `0.0613` to `0.0523`.

---

# 💡 What This Project Demonstrates

This project goes beyond training a classification model.

It demonstrates the complete lifecycle of a machine learning application:

- Data acquisition and understanding
- Exploratory data analysis
- Missing-value handling
- Categorical and numerical preprocessing
- Class-imbalance handling
- Model comparison
- Stratified cross-validation
- Gradient-boosting model development
- Hyperparameter optimization
- Probability-threshold optimization
- Probability calibration
- SHAP-based model interpretation
- Model serialization
- REST API development
- Interactive frontend development
- Cloud deployment

The project therefore demonstrates both **Machine Learning expertise** and the ability to take a trained model from experimentation to a usable application.

---

# ⚠️ Disclaimer

This project is intended for **educational and portfolio purposes**.

The predictions should not be used as the sole basis for real-world lending decisions. A production credit-risk system would require additional validation, regulatory compliance, fairness analysis, monitoring, security controls, model governance, and evaluation on appropriate real-world institutional data.

---

# 👤 Author

**Dr. Amandip De**

🔗 GitHub: [Credit Risk Prediction — GitHub Repository](https://github.com/amandipde/Credit-Risk-Prediction/tree/main?utm_source=chatgpt.com)
