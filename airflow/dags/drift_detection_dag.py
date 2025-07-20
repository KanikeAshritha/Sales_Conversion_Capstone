from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import os
import json

# Path to the drift flag file (written by drift detection logic)
DRIFT_FLAG_PATH = "/opt/airflow/dags/drift_flag.json"

# Path to the log file that stores drift check results
DRIFT_LOG_PATH = '/opt/airflow/dags/drift_detected_log.txt'

# ID of the DAG to trigger if drift is detected
RETRAIN_DAG_ID = "retrain_model_dag"

# Default settings for all tasks in the DAG
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

# Define the DAG
with DAG(
    dag_id="drift_detection_dag",              # Name of the DAG
    default_args=default_args,                 # Use the default arguments above
    start_date=datetime(2024, 1, 1),           # When the DAG starts running
    schedule_interval="@daily",                # Run this DAG once every day
    catchup=False                              # Do not run past DAG runs
) as dag:

    # Function to check if drift has been detected
    def check_drift_flag(**kwargs):
        # If the flag file does not exist, end the pipeline
        if not os.path.exists(DRIFT_FLAG_PATH):
            print("Drift flag file not found.")
            return "end_pipeline"
        
        # Read the contents of the drift flag file
        with open(DRIFT_FLAG_PATH, 'r') as f:
            flag_data = json.load(f)

        # If drift is detected, log it and return the task ID to trigger retraining
        if flag_data.get("drift_detected", False):
            with open(DRIFT_LOG_PATH, 'a') as log:
                log.write(f"{datetime.now()}: Drift detected. Triggering retrain DAG.\n")
            return "trigger_retrain_dag"
        else:
            # If no drift is detected, log it and return the end task
            with open(DRIFT_LOG_PATH, 'a') as log:
                log.write(f"{datetime.now()}: No drift detected.\n")
            return "end_pipeline"

    # Function to reset the drift flag after retraining
    def reset_drift_flag():
        with open(DRIFT_FLAG_PATH, 'w') as f:
            json.dump({"drift_detected": False}, f)
        print("Drift flag reset.")

    # Task to decide whether to retrain or end pipeline
    decide = BranchPythonOperator(
        task_id="decide_drift_action",
        python_callable=check_drift_flag,
        provide_context=True
    )

    # Task to trigger the retraining DAG
    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain_dag",
        trigger_dag_id=RETRAIN_DAG_ID,
        wait_for_completion=False,
        reset_dag_run=True
    )

    # Task that runs if no drift is detected
    end_pipeline = DummyOperator(task_id="end_pipeline")

    # Task to reset the drift flag after retraining
    reset_flag = PythonOperator(
        task_id="reset_drift_flag",
        python_callable=reset_drift_flag
    )

    # Define the order in which tasks should run
    decide >> [trigger_retrain, end_pipeline]
    trigger_retrain >> reset_flag
