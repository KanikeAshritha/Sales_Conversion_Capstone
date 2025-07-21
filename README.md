<pre>

final_capstone/
├── airflow/                             # Airflow orchestration for drift detection & retraining
│   ├── dags/                            # Custom DAGs to automate ML workflows
│   │   ├── drift_detection_dag.py       # Uses Evidently to detect data drift
│   │   └── retrain_model_dag.py         # Retrains model if drift is detected
│   ├── airflow_home/                    # Airflow base folder (now cleaned)
│   │   ├── logs/                        # Airflow task & scheduler logs
│   ├── artifacts/                       # Artifacts for drift tasks (e.g., drift flag, pickle files)
│   ├── data/                            # Reference & incoming data for drift comparison
│   │   ├── reference_data.csv           # Clean data snapshot to compare against
│   │   └── new_data.csv                 # New data to be tested for drift
│   ├── logs/                            # Logs for Airflow jobs
│   ├── models/                          # Trained models for Airflow retraining
│   ├── .env                             # Environment variables (used in Airflow DAGs)
│   ├── airflow.cfg                      # Airflow config file
│   ├── airflow.db                       # SQLite DB for Airflow metadata
│   ├── airflow-webserver.pid            # Webserver process ID
│   ├── latest_model_uri.txt             # URI to latest MLflow Production model
│   ├── latest_run_id.txt                # MLflow run ID of the best model
│   ├── requirements.txt                 # Python dependencies for Airflow setup
│   └── webserver_config.py              # Airflow webserver configuration

├── api/                                 # Flask API for serving predictions
│   ├── templates/                       # HTML templates for Flask UI
│   │   ├── index.html                   # Upload form
│   │   └── results.html                 # Displays prediction output
│   └── app.py                           # Main Flask app: loads model, serves UI, predicts

├── artifacts/                           # Global project artifacts (not just for Airflow)
│   ├── *.pkl                            # Transformers (preprocessor, encoders, dropper, etc.)
│   ├── drift_flag.json                  # Set when drift is detected
│   ├── drift_report.html                # Visual drift report (Evidently)
│   └── reference_data.csv               # Same as airflow/data (duplicate for shared access)

├── mlartifacts/                         # Optional: for logs, plots, or artifacts
├── mlruns/                              # MLflow run data — experiment tracking and metadata
├── models/                              # Trained models (local copies)

├── notebooks/                           # Final EDA or research notebooks
│   └── eda.ipynb

├── Outputs/                             # Optional: prediction results, screenshots, etc.

├── src/                                 # Core Python ML pipeline code
│   ├── __pycache__/                     # Python bytecode cache
│   ├── artifacts/                       # Optional: stores pickled transformers, encoders
│   ├── models/                          # Optional: model saving logic
│   ├── detect_drift.py                  # Uses Evidently to check for drift and generate flag
│   ├── eda.py                           # EDA utility scripts and plots
│   ├── mlflow_runner.py                 # Manages MLflow logging, registration, promotion
│   ├── model_training.py                # Trains ML models (e.g., XGBoost, RandomForest)
│   ├── postgres_utils.py                # Connects to PostgreSQL and fetches data
│   ├── preprocessing.py                 # Handles encoding, scaling, cleaning, splitting
│   ├── shap.py                          # SHAP explainability for model interpretation
│   ├── train_model_pipeline.py          # Pipeline orchestration: preprocessing → training → logging


├── Capstone_Crispml(q).xlsx             # CRISP-ML(Q) methodology planning document
├── docker-compose.yml                   # Docker Compose setup for full app (Airflow, API, MLflow)
├── Dockerfile                           # Docker image definition for app training or inference
├── drift_flag                           # Placeholder or flag file (optional or deprecated)
├── latest_model_uri                     # Stores path/URI to the most recent model
├── latest_run_id                        # Stores run ID of the latest MLflow model
├── Lead Scoring.csv                     # Sample or real-world business dataset
├── main.py                              # Entry point to run the ML pipeline manually


</pre>
