# Import required modules from Airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Import the main model training function from your ML pipeline
from main import main

# Define the function that will run when the task is triggered
def retrain_model():
    # Print a message to indicate retraining has started
    print("Starting retraining process")
    
    # Call the main() function which includes preprocessing, training, MLflow logging, etc.
    main()
    
    # Print a message to indicate retraining has completed
    print("Retraining and logging complete")

# Define the DAG (Directed Acyclic Graph) for Airflow
with DAG(
    dag_id="retrain_model_dag",           # Unique identifier for the DAG
    start_date=datetime(2024, 1, 1),      # Start date of the DAG
    schedule_interval=None,               # Run manually or triggered by another DAG
    catchup=False                         # Do not backfill or run for past dates
) as dag:

    # Define the retraining task using a PythonOperator
    retrain = PythonOperator(
        task_id="retrain_best_model",     # Unique identifier for the task in the DAG
        python_callable=retrain_model     # Function to be executed
    )

    # Set the task in the DAG
    retrain
