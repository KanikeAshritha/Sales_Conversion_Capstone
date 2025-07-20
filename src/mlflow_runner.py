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

# Load the saved preprocessor and return a pipeline with the given model
def get_pipeline(model):
    # Path to the saved preprocessor
    preprocessor_path = "artifacts/preprocessor.pkl"

    # Raise error if preprocessor file does not exist
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
    
    # Load the preprocessor (not used here, but can be extended later)
    preprocessor = joblib.load(preprocessor_path)

    # Return a simple pipeline with only the model for now
    return Pipeline([
        ('model', model)
    ])

# Perform grid search cross-validation with the given pipeline and training data
def perform_grid_search(pipeline, X_train, y_train, param_grid=None):
    # If no parameter grid is provided, use an empty dictionary
    if param_grid is None:
        param_grid = {}
    
    # Create GridSearchCV object to search best parameters
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    try:
        # Fit the model with cross-validation
        search.fit(X_train, y_train)
        return search
    except Exception as e:
        # If grid search fails, print error
        print(f"GridSearchCV failed: {e}")
        
        # Fit the pipeline without tuning
        pipeline.fit(X_train, y_train)
        
        # Create a mock object to mimic GridSearchCV result
        class MockSearch:
            def __init__(self, pipeline):
                self.best_estimator_ = pipeline
                self.best_params_ = {}
        
        return MockSearch(pipeline)

# Log the trained model to MLflow and register it for version control
def log_and_register_model(model, model_name):
    artifact_path = "model"

    # End any active MLflow run
    if mlflow.active_run():
        mlflow.end_run()

    try:
        # Start a new MLflow run for registration
        with mlflow.start_run(run_name=f"{model_name}_Register") as run:
            run_id = run.info.run_id

            # Get current experiment info
            experiment = mlflow.get_experiment(run.info.experiment_id)

            # Log the trained model to MLflow
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path
            )

            # Get the URI to the logged model
            model_uri = f"runs:/{run_id}/{artifact_path}"

            # Initialize MLflow client
            client = MlflowClient()

            # If model is not yet registered, register it
            try:
                client.get_registered_model(model_name)
            except MlflowException:
                client.create_registered_model(model_name)

            # Create a new version of the model
            version = client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id
            ).version

            # Move model to the "Staging" stage
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Staging"
            )

            # Save model locally
            os.makedirs("models", exist_ok=True)
            model_save_name = "models/model.pkl"
            joblib.dump(model, model_save_name)

            # Save model URI and run ID to text files
            with open("latest_model_uri.txt", "w") as f:
                f.write(model_uri)
            with open("latest_run_id.txt", "w") as f:
                f.write(run_id)

            # Print success message
            print(f"Model {model_name} v{version} registered and saved")

            # Return model info
            return {
                "model_uri": model_uri,
                "model_name": model_name,
                "version": version,
                "run_id": run_id,
                "experiment_name": experiment.name,
                "model_save_path": model_save_name
            }
    except Exception as e:
        # Print error message if registration fails
        print(f"Error logging and registering model {model_name}: {e}")
        return {
            "model_uri": "",
            "model_name": model_name,
            "version": "unknown",
            "run_id": "",
            "experiment_name": "",
            "model_save_path": ""
        }

# Run training for multiple models, compare them, and register the best one
def mlflow_run_with_grid_search(X_train, X_test, y_train, y_test, model_registry):
    # Initialize variables to store the best model info
    best_model_info = {}
    best_score = 0
    best_model = None
    best_model_name = ""

    # Set MLflow tracking URI (your MLflow server URL)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Set experiment name in MLflow
    mlflow.set_experiment("Lead_Conversion_Classification")

    print(f"Training {len(model_registry)} models...")

    # Loop through each model in the registry
    for model_name, (model, param_grid) in model_registry.items():
        print(f"\nTraining {model_name}...")
        
        try:
            # Start an MLflow run for this model
            with mlflow.start_run(run_name=f"{model_name}_Classifier", nested=True):
                # Create pipeline using the model
                pipeline = get_pipeline(model)
                
                # Perform grid search to tune hyperparameters
                search = perform_grid_search(pipeline, X_train, y_train, param_grid)
                
                # Get the best model found by grid search
                best_estimator = search.best_estimator_
                
                # Make predictions on the test set
                y_pred = best_estimator.predict(X_test)

                # Calculate evaluation metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                # Log best parameters and metrics to MLflow
                mlflow.log_params(search.best_params_)
                mlflow.log_metrics({
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1_Score": f1
                })

                # Log the best model
                mlflow.sklearn.log_model(
                    sk_model=best_estimator,
                    artifact_path="model"
                )

                # Print model performance
                print(f"{model_name} — Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

                # If this model is the best so far, update tracking variables
                if acc > best_score:
                    best_score = acc
                    best_model = best_estimator
                    best_model_name = model_name

        except Exception as e:
            # Print error and continue with next model
            print(f"Error training {model_name}: {e}")
            continue

    # If at least one model trained successfully, register the best one
    if best_model:
        print(f"\nBest model: {best_model_name} (Accuracy: {best_score:.4f})")
        best_model_info = log_and_register_model(best_model, best_model_name)
        
        # Print final summary
        print("\nFINAL BEST MODEL SUMMARY")
        print(f"Model Name     : {best_model_info['model_name']}")
        print(f"Version        : {best_model_info['version']}")
        print(f"Run ID         : {best_model_info['run_id']}")
        print(f"Saved at       : {best_model_info['model_save_path']}")
    else:
        # If all models failed
        print("No best model found. All models failed.")

    # Return the best model's metadata
    return best_model_info
