import os
import tempfile
import joblib
import mlflow
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# Load saved preprocessor and return a pipeline with the model
def get_pipeline(model):
    """Build full pipeline: preprocessor + model"""
    preprocessor_path = "artifacts/preprocessor.pkl"

    # Make sure preprocessor file exists
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
    
    # Load the preprocessor (not added to pipeline here as preprocessing was done separately)
    preprocessor = joblib.load(preprocessor_path)

    # Return pipeline with just the model
    return Pipeline([
        ('model', model)
    ])

# Try GridSearchCV and fallback to normal training if it fails
def perform_grid_search(pipeline, X_train, y_train, param_grid=None):
    """Perform GridSearchCV with error handling"""
    
    # Use empty parameter grid if none provided
    if param_grid is None:
        param_grid = {}

    # Set up grid search
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    try:
        # Try fitting with grid search
        search.fit(X_train, y_train)
        return search
    except Exception as e:
        # If grid search fails, fallback to simple model training
        print(f"GridSearchCV failed: {e}")
        pipeline.fit(X_train, y_train)

        # Create a mock object to hold the best estimator
        class MockSearch:
            def __init__(self, pipeline):
                self.best_estimator_ = pipeline
                self.best_params_ = {}

        return MockSearch(pipeline)

# Save, log, and register the model using MLflow
def log_and_register_model(model, model_name):
    """Log and register the best model with error handling"""
    artifact_path = "model"

    # End any current MLflow run
    if mlflow.active_run():
        mlflow.end_run()

    try:
        # Start a new MLflow run
        with mlflow.start_run(run_name=f"{model_name}_Register") as run:
            run_id = run.info.run_id
            experiment = mlflow.get_experiment(run.info.experiment_id)

            # Log model to MLflow
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path
            )

            # Prepare model URI path
            model_uri = f"runs:/{run_id}/{artifact_path}"
            client = MlflowClient()

            # If model is not yet registered, create it
            try:
                client.get_registered_model(model_name)
            except MlflowException:
                client.create_registered_model(model_name)

            # Create a version entry for the model
            version = client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id
            ).version

            # Move model to "Staging" phase
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Staging"
            )

            # Save model locally
            os.makedirs("models", exist_ok=True)
            model_save_name = "models/model.pkl"
            joblib.dump(model, model_save_name)

            # Save model URI and run ID for later use
            with open("latest_model_uri.txt", "w") as f:
                f.write(model_uri)
            with open("latest_run_id.txt", "w") as f:
                f.write(run_id)

            print(f"Model {model_name} v{version} registered and saved")

            return {
                "model_uri": model_uri,
                "model_name": model_name,
                "version": version,
                "run_id": run_id,
                "experiment_name": experiment.name,
                "model_save_path": model_save_name
            }
    except Exception as e:
        print(f"Error logging and registering model {model_name}: {e}")
        return {
            "model_uri": "",
            "model_name": model_name,
            "version": "unknown",
            "run_id": "",
            "experiment_name": "",
            "model_save_path": ""
        }

# Train and log all models using MLflow and GridSearch
def mlflow_run_with_grid_search(X_train, X_test, y_train, y_test, model_registry):
    """
    Run MLflow experiments with grid search for multiple models
    """

    # To track the best model among all
    best_model_info = {}
    best_score = 0
    best_model = None
    best_model_name = ""

    # Set MLflow tracking URI and experiment name
    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Lead_Conversion_Classification")

    print(f"Training {len(model_registry)} models...")

    # Loop through all models in the registry
    for model_name, (model, param_grid) in model_registry.items():
        print(f"\nTraining {model_name}...")

        try:
            # Start an MLflow run for this model
            with mlflow.start_run(run_name=f"{model_name}_Classifier", nested=True):
                # Build pipeline with model
                pipeline = get_pipeline(model)
                
                # Perform grid search with fallback
                search = perform_grid_search(pipeline, X_train, y_train, param_grid)

                # Get best model from search
                best_estimator = search.best_estimator_

                # Predict on test set
                y_pred = best_estimator.predict(X_test)

                # Calculate metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                # Log parameters and metrics to MLflow
                mlflow.log_params(search.best_params_)
                mlflow.log_metrics({
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1_Score": f1
                })

                # Log model artifact
                mlflow.sklearn.log_model(
                    sk_model=best_estimator,
                    artifact_path="model"
                )

                print(f"{model_name} — Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

                # Save best model based on accuracy
                if acc > best_score:
                    best_score = acc
                    best_model = best_estimator
                    best_model_name = model_name

        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue

    # Register the best model
    if best_model:
        print(f"\nBest model: {best_model_name} (Accuracy: {best_score:.4f})")
        best_model_info = log_and_register_model(best_model, best_model_name)

        # Final summary print
        print("\nFINAL BEST MODEL SUMMARY")
        print(f"Model Name     : {best_model_info['model_name']}")
        print(f"Version        : {best_model_info['version']}")
        print(f"Run ID         : {best_model_info['run_id']}")
        print(f"Saved at       : {best_model_info['model_save_path']}")
    else:
        print("No best model found. All models failed.")

    return best_model_info
