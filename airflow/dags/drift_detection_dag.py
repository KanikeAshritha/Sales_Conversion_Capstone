# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from datetime import datetime, timedelta
# import sys
# import os

# # Add /opt/airflow/src to path to import detect_drift
# sys.path.append("/opt/airflow")
# from src.detect_drift import run_drift_detection    

# default_args = {
#     'owner': 'airflow',
#     'retries': 1,
#     'retry_delay': timedelta(minutes=1)
# }

# with DAG(
#     dag_id='drift_detection_dag',
#     default_args=default_args,
#     description='Daily data drift check',
#     schedule_interval='@daily',
#     start_date=datetime(2024, 1, 1),
#     catchup=False
# ) as dag:

#     drift_check_task = PythonOperator(
#         task_id='check_drift',
#         python_callable=run_drift_detection,
#     )



# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from datetime import datetime, timedelta
# import sys
# import os

# # Add src folder to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'src')))

# from detect_drift import run_drift_detection

# default_args = {
#     'owner': 'airflow',
#     'retries': 1,
#     'retry_delay': timedelta(minutes=1),
# }

# with DAG(
#     dag_id='drift_detection_dag',
#     default_args=default_args,
#     description='Detects drift between reference and current datasets',
#     start_date=datetime(2023, 1, 1),
#     schedule_interval='@daily',  # ← or None if manual
#     catchup=False,
#     tags=['drift', 'evidently'],
# ) as dag:

#     detect_drift_task = PythonOperator(
#         task_id='check_drift',
#         python_callable=run_drift_detection,
#         provide_context=True,  # needed for XCom
#     )

#     detect_drift_task




from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import os
import json

# ✅ Paths
DRIFT_FLAG_PATH = "/opt/airflow/dags/drift_flag.json"
DRIFT_LOG_PATH = '/opt/airflow/dags/drift_detected_log.txt' 
RETRAIN_DAG_ID = "retrain_model_dag"

# ✅ Default args
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

# ✅ DAG definition
with DAG(
    dag_id="drift_detection_dag",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False
) as dag:

    def check_drift_flag(**kwargs):
        if not os.path.exists(DRIFT_FLAG_PATH):
            print("❌ Drift flag file not found.")
            return "end_pipeline"
        
        with open(DRIFT_FLAG_PATH, 'r') as f:
            flag_data = json.load(f)

        if flag_data.get("drift_detected", False):
            with open(DRIFT_LOG_PATH, 'a') as log:
                log.write(f"{datetime.now()}: ✅ Drift detected. Triggering retrain DAG.\n")
            return "trigger_retrain_dag"
        else:
            with open(DRIFT_LOG_PATH, 'a') as log:
                log.write(f"{datetime.now()}: ❌ No drift detected.\n")
            return "end_pipeline"

    def reset_drift_flag():
        with open(DRIFT_FLAG_PATH, 'w') as f:
            json.dump({"drift_detected": False}, f)
        print("✅ Drift flag reset.")

    # ✅ Branching to decide path
    decide = BranchPythonOperator(
        task_id="decide_drift_action",
        python_callable=check_drift_flag,
        provide_context=True
    )

    # ✅ Trigger retrain DAG
    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain_dag",
        trigger_dag_id=RETRAIN_DAG_ID,
        wait_for_completion=False,
        reset_dag_run=True
    )

    # ✅ Dummy task for "no drift"
    end_pipeline = DummyOperator(task_id="end_pipeline")

    # ✅ Reset flag after retraining
    reset_flag = PythonOperator(
        task_id="reset_drift_flag",
        python_callable=reset_drift_flag
    )

    # ✅ DAG flow
    decide >> [trigger_retrain, end_pipeline]
    trigger_retrain >> reset_flag
