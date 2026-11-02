# 🚗 Used Car Price Prediction & Valuation System

An end-to-end Machine Learning pipeline and interactive web application for real-time second-hand vehicle price prediction, powered by **LightGBM**, **Scikit-Learn**, and **Streamlit**.

---

## 📌 Project Overview

Predicting second-hand vehicle market value requires modeling complex nonlinear interactions between vehicle age, mileage, technical specifications, and damage history. This project delivers a production-ready machine learning pipeline that handles high-cardinality categorical encoding, target transformation, and hyperparameter optimization to provide accurate price evaluations.

### Key Highlights
- **End-to-End ML Pipeline:** Automated preprocessing, feature engineering, and estimation enclosed within Scikit-Learn `Pipeline` and `ColumnTransformer` structures.
- **High-Cardinality Target Encoding:** Utilized cross-validated `TargetEncoder` for vehicle `series` to retain model hierarchy without causing dimensionality explosion.
- **Logarithmic Target Transformation:** Wrapped regressors with `TransformedTargetRegressor` (log1p / expm1) to normalize price skewness and reduce heavy-tail prediction error.
- **Interactive Web Application:** Built a responsive, bilingual (English/Turkish UI) web application using Streamlit for instant user predictions.

---

## 📊 Model Performance & Benchmarks

During the iterative evaluation phase across baseline linear models, tree ensembles, and hyperparameter tuning, the following test set metrics were achieved:

| Model Architecture | Features Included | R2 Score | MAE (TRY) |
| :--- | :--- | :---: | :---: |
| **Ridge Regression (Baseline)** | Standard Scaling + One-Hot Encoding | 0.6480 | ~178,000 TL |
| **Random Forest Regressor** | Default Parameters | 0.6512 | ~175,000 TL |
| **LightGBM (Baseline)** | One-Hot Encoded Categoricals | 0.7054 | ~162,000 TL |
| **LightGBM + Target Encoding** | Target Encoded series | 0.7224 | 145,481 TL |
| **LightGBM (GridSearchCV Tuned)** | **Optimized Hyperparameters** | **0.7501** | **135,197 TL** |

> **Final Model Performance:** Hyperparameter tuning via 5-fold cross-validation brought R2 to **0.7501** and reduced Mean Absolute Error (MAE) by ~43,000 TRY compared to initial baselines.

---

## 🛠️ Tech Stack & Requirements

- **Language:** Python 3.10+
- **Machine Learning:** Scikit-Learn, LightGBM, Joblib
- **Data Manipulation & Viz:** Pandas, NumPy, Matplotlib, Seaborn
- **Web Interface:** Streamlit
- **Version Control:** Git

---

## 📁 Repository Structure

arabam-used-car-price-prediction/
│
├── data/
│   ├── raw/                      # Raw scraped vehicle dataset
│   └── processed/                # Cleaned dataset (cleaned_arabam.csv)
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb    # Data parsing, missing value imputation, type conversion
│   ├── 02_eda.ipynb              # Exploratory data analysis & visualizations
│   └── 03_model_training.ipynb   # Model comparisons, Target Encoding & GridSearchCV tuning
│
├── models/
│   └── best_car_price_model.joblib # Serialized production-ready model pipeline
│
├── src/
│   └── train.py                  # Standalone modular training & serialization script
│
├── app/
│   └── app.py                    # Interactive Streamlit application
│
├── requirements.txt              # Project dependencies
├── .gitignore                    # Version control exclusions
└── README.md                     # Project documentation

---

## ⚙️ Quickstart & Installation

### 1. Clone the Repository & Set Up Environment
git clone [https://github.com/YOUR_USERNAME/arabam-used-car-price-prediction.git](https://github.com/YOUR_USERNAME/arabam-used-car-price-prediction.git)
cd arabam-used-car-price-prediction

python -m venv .venv
.\.venv\Scripts\Activate.ps1

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Train & Export Model Artifact
python src/train.py

### 4. Launch Streamlit Web App
streamlit run app/app.py

---

## 🚀 Future Roadmap & Enhancements

- Incorporate sub-model variant descriptions into secondary Target Encoders.
- Experiment with CatBoost Regressor for enhanced categorical feature handling.
- Deploy Streamlit application to Streamlit Community Cloud.