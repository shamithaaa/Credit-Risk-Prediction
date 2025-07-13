# Credit Risk Prediction Web App

A Streamlit web app that predicts whether a person is likely to default on a salary advance loan based on key financial indicators. Powered by **Machine Learning (Logistic Regression & XGBoost)** and deployed for public use.

---

## Problem Statement

Credit risk is the possibility of a borrower defaulting on a loan. In this project, we analyze historical loan data to **predict the probability of default** using machine learning models.

---

## Machine Learning Workflow

- **Dataset**: [HMEQ Loan Dataset](https://www.kaggle.com/datasets/ajay1735/hmeq-data)
- **Target Variable**: `BAD` (1 = default, 0 = no default)
- **Models Used**:
  - Logistic Regression
  - XGBoost Classifier
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Deployment**: Streamlit + GitHub

---

## 📁 Folder Structure
```
credit-risk-streamlit-app/
│
├── app.py
├── xgb_model.pkl
├── scaler.pkl
├── requirements.txt
└── screenshots/
```

---

## 🖥️ Features

- 🔍 Takes user financial inputs and predicts default likelihood
- 📈 Displays ROC Curve, Confusion Matrix, and Feature Importance
- 📊 Visualizes class distribution
- ⚙️ Model explainability using SHAP (optional extension)




