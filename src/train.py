import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from lightgbm import LGBMRegressor

def build_and_train_pipeline(data_path: str, model_save_path: str):
    """
    Cleans, splits, preprocesses, and trains the optimized LightGBM model.
    Saves the trained pipeline as a joblib artifact.
    """

    print(f"[INFO] Loading processed data from {data_path}...")
    df = pd.read_csv(data_path)

    X = df.drop(columns=["price", "model"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    high_card_col = ["series"]
    low_card_col = [c for c in X_train.select_dtypes(include=["object"]).columns if c != "series"]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()) 
        ]), num_cols),
        ("target_enc", TargetEncoder(cv=5, smooth="auto", random_state=42), high_card_col),
        ("onehot", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), low_card_col)
    ])

    lgbm_estimator = LGBMRegressor(
        learning_rate=0.08,
        max_depth=-1,
        n_estimators=150,
        num_leaves=15,
        random_state=42,
        verbose=-1
    )

    raw_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", lgbm_estimator)
    ])

    full_model_pipeline = TransformedTargetRegressor(
        regressor=raw_pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )

    print("[INFO] Training the best LightGBM pipeline...")
    full_model_pipeline.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(full_model_pipeline, model_save_path)
    print(f"[SUCCESS] Trained model pipeline saved successfully at: {model_save_path}")

if __name__ == "__main__":
    DATA_PATH = "data/processed/cleaned_arabam.csv"
    MODEL_SAVE_PATH = "models/best_car_price_model.joblib"

    build_and_train_pipeline(DATA_PATH, MODEL_SAVE_PATH)