# src/train_model_pipeline.py

import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.preprocessing import preprocess_data

# This function runs the model training pipeline
def train_model_pipeline():
    # Load reference data for training
    df = pd.read_csv("artifacts/reference_data.csv")

    # Preprocess the data and get features and target
    X, y = preprocess_data(df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Start an MLflow run to log metrics and model
    with mlflow.start_run(run_name="retrained_model"):
        # Fit the model on training data
        model.fit(X_train, y_train)

        # Predict on test data
        y_pred = model.predict(X_test)

        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)

        # Log accuracy metric to MLflow
        mlflow.log_metric("accuracy", acc)

        # Log the trained model to MLflow
        mlflow.sklearn.log_model(model, "model")

        # Print the final accuracy
        print(f"Accuracy: {acc:.4f}")
