# Churn-Predictor
# Customer Churn Prediction (Telco)

**Short project summary**

A machine-learning pipeline that predicts whether a telecom customer will churn using the classic Telco Customer Churn dataset. The notebook performs EDA, data cleaning, label encoding, class balancing (SMOTE), and trains/compares Decision Tree, Random Forest and XGBoost models. The final Random Forest model is exported for inference.

---

## Table of contents

* [Dataset](#dataset)
* [What the notebook does](#what-the-notebook-does)
* [Models & results](#models--results)
* [How to run](#how-to-run)
* [Quick inference example](#quick-inference-example)
* [File structure](#file-structure)
* [Notes & suggestions for improvement](#notes--suggestions-for-improvement)
* [Dependencies](#dependencies)
* [License & contact](#license--contact)

---

## Dataset

This project uses the **WA\_Fn-UseC\_-Telco-Customer-Churn.csv** (Telco Customer Churn) dataset. The notebook expects the CSV to be available in the working directory.

## What the notebook does

1. Imports dependencies (pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, imblearn).
2. Loads the Telco CSV and drops the `customerID` column.
3. Exploratory Data Analysis (histograms, boxplots, correlation heatmap) and prints unique values per column.
4. Fixes data issues: converts `TotalCharges` to numeric by replacing blanks with `0.0` and casting to `float`.
5. Encodes categorical features using `LabelEncoder` and saves encoders to `encoders.pkl`.
6. Splits data into features `X` and target `y (Churn)` and uses `train_test_split(test_size=0.2)`.
7. Uses **SMOTE** to oversample the minority class on the training set.
8. Trains and cross-validates three models: Decision Tree, Random Forest, and XGBoost.
9. Fits the chosen model (Random Forest) on the SMOTE-resampled training set and evaluates on the test set.
10. Saves the trained model and feature names to `customer_churn_model.pkl` and demonstrates an inference example.

## Models & results

* **Models tested**: Decision Tree, Random Forest, XGBoost

* **Cross-validation (5-fold) mean accuracy**:

  * Decision Tree: **0.78**
  * Random Forest: **0.84**
  * XGBoost: **0.83**

* **Random Forest — test set performance**:

  * **Accuracy**: 0.7785663591199432
  * **Confusion matrix**:

    ```
    [[878 158]
     [154 219]]
    ```
  * **Classification report (summary)**:

    * Class 0 (No churn) — precision: 0.85, recall: 0.85, f1-score: 0.85 (support: 1036)
    * Class 1 (Churn)    — precision: 0.58, recall: 0.59, f1-score: 0.58 (support: 373)
    * Weighted avg accuracy ≈ **0.78**

* **Example inference (from notebook)**: Predicted `No Churn` with probability `[0.78, 0.22]` for the sample input.

## How to run (local)

1. Clone or copy the notebook and dataset into a working folder.
2. (Recommended) Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Open and run the notebook `Customer_Churn_Prediction_using_ML.ipynb` in Jupyter / Google Colab.

**Suggested `requirements.txt`** (use latest compatible versions):

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
joblib
```

## Quick inference example (Python)

```python
import pickle
import pandas as pd

# load model + encoders
with open('customer_churn_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
model = model_data['model']
feature_names = model_data['features_names']

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# build input row as dict (keys = original column names)
input_data = {
  'gender': 'Female',
  'SeniorCitizen': 0,
  'Partner': 'Yes',
  # ... include all feature columns in same order as used during training
}

# construct DataFrame, encode categorical columns using `encoders` and call model.predict
input_df = pd.DataFrame([input_data])
for col, enc in encoders.items():
    input_df[col] = enc.transform(input_df[col])

pred = model.predict(input_df[feature_names])
prob = model.predict_proba(input_df[feature_names])
print('Prediction:', 'Churn' if pred[0] == 1 else 'No Churn')
print('Probability:', prob)
```

## File structure (expected)

```
Customer_Churn_Prediction_using_ML.ipynb
WA_Fn-UseC_-Telco-Customer-Churn.csv
encoders.pkl
customer_churn_model.pkl
README.md   <-- this file
requirements.txt
```

## Notes & suggestions for improvement

* Replace `LabelEncoder` with `OneHotEncoder` or `pd.get_dummies` for non-ordinal categorical features and use sklearn `ColumnTransformer` + `Pipeline` for robust preprocessing.
* Use a full pipeline (`Pipeline`) and `GridSearchCV` / `RandomizedSearchCV` for hyperparameter tuning.
* Try calibrating model probabilities (e.g., `CalibratedClassifierCV`) for better probability estimates.
* Explore feature engineering (interaction terms, tenure bins) and additional feature selection methods.
* Use SHAP or LIME to explain per-customer predictions and extract business insights.
* Consider stratified splitting and alternative imbalance strategies (class weights, ADASYN).
* For deployment: wrap the preprocessing + model into a single serialized `sklearn` pipeline (joblib) or serve via a small FastAPI/Flask app.

## License & contact

This repository is released under the MIT License (feel free to change). For questions or collaboration, contact the author (available in the notebook metadata or add your email here).

---    
