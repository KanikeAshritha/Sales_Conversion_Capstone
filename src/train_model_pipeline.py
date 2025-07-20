# src/train_model_pipeline.py

import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.preprocessing import preprocess_data

def train_model_pipeline():
    print("[🚀 Training] Starting model training pipeline...")

    df = pd.read_csv("artifacts/reference_data.csv")  # Train on historical (reference) data

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    with mlflow.start_run(run_name="retrained_model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print(f"[✅ Model Training] Accuracy: {acc:.4f}")
