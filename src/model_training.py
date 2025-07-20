from src.train_model_pipeline import train_model_pipeline  
import os
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(X_train, y_train, X_test, y_test, model_name, model):
    print(f"\n🧠 Training model: {model_name}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print(f"✅ Accuracy:  {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall:    {recall:.4f}")
    print(f"✅ F1-score:  {f1:.4f}")
    if auc is not None:
        print(f"✅ AUC-ROC:   {auc:.4f}")

DRIFT_FLAG_PATH = "airflow/artifacts/drift_flag.txt"

def run_training_if_drifted():
    if not os.path.exists(DRIFT_FLAG_PATH):
        print("Drift flag not found. Skipping retraining.")
        return

    with open(DRIFT_FLAG_PATH, "r") as f:
        flag = f.read().strip()

    if flag.lower() == "true":
        print("[⚠️ Drift Found] Retraining model...")
        train_model_pipeline()
    else:
        print("[ℹ️ No Drift] Skipping retraining.")