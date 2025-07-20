from src.train_model_pipeline import train_model_pipeline  
import os
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Evaluates the performance of the given model using common classification metrics
def evaluate_model(X_train, y_train, X_test, y_test, model_name, model):
    # Displays which model is being trained
    print(f"\nTraining model: {model_name}")
    
    # Trains the model on the training dataset
    model.fit(X_train, y_train)

    # Predicts labels for the test dataset
    y_pred = model.predict(X_test)

    # Predicts probabilities for the test dataset if the model supports it
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Calculates accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Calculates precision score
    precision = precision_score(y_test, y_pred, zero_division=0)

    # Calculates recall score
    recall = recall_score(y_test, y_pred, zero_division=0)

    # Calculates F1-score
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Calculates AUC-ROC score if probability predictions are available
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    # Displays all evaluation metrics
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    if auc is not None:
        print(f"AUC-ROC:   {auc:.4f}")

# Path to the file that contains the data drift detection result
DRIFT_FLAG_PATH = "airflow/artifacts/drift_flag.txt"

# Checks if drift is detected and retrains the model accordingly
def run_training_if_drifted():
    # Skips retraining if drift flag file is not present
    if not os.path.exists(DRIFT_FLAG_PATH):
        print("Drift flag not found. Skipping retraining.")
        return

    # Reads the value from the drift flag file
    with open(DRIFT_FLAG_PATH, "r") as f:
        flag = f.read().strip()

    # If drift is found, retraining is triggered
    if flag.lower() == "true":
        print("Drift Found Retraining model...") 
        train_model_pipeline()
    else:
        # Skips retraining if no drift is found
        print("No Drift, Skipping retraining.")
