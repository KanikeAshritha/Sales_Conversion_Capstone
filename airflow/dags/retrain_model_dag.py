from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess
from mlflow_runner import mlflow_run_with_grid_search
from main import main

def retrain_model():
    print("📦 Starting retraining process")
    main()
    print("✅ Retraining and logging complete")

with DAG(
    dag_id="retrain_model_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    retrain = PythonOperator(
        task_id="retrain_best_model",
        python_callable=retrain_model
    )

    retrain  # ✅ This line ensures the task is registered in the DAG
