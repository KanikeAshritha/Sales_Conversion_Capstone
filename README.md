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

**Local Development Workflow**
  
Environment Setup
1.	Clone the Repository
Clone the project from the GitHub repository to your local machine:
>git clone https://github.com/KanikeAshritha/Sales_Conversion_Capstone
>cd Final_Capstone
2.	Create a Python Virtual Environment
A virtual environment is recommended to isolate the dependencies:
>python -m venv venv
3.	Activate the Virtual Environment
o	On Windows:
venv\Scripts\activate
o	On macOS/Linux:
source venv/bin/activate
4.	Install Required Dependencies
Use the provided requirements.txt file to install all Python packages:
>pip install -r requirements.txt



**Executing the Project**
  
5.	Start the MLflow Tracking Server
In a separate terminal, launch the MLflow UI:
>mlflow ui
This starts the UI at http://localhost:5000, which is used to log experiments, metrics, parameters, and model artifacts.
6.	Run the Main Training Pipeline
The primary training pipeline can be executed using:
python main.py
This will load the dataset, preprocess the data, perform model training and evaluation, and log results to MLflow.
7.	Start the Prediction API Server
Once the model is trained and registered, launch the Flask-based API server for real-time inference:
python -m api.app
This exposes a /predict endpoint to send JSON payloads and receive predictions.
Running Apache Airflow Locally
8.	Initialize Airflow Metadata Database
airflow db init

9.	Start Airflow Using Docker Compose
Make sure Docker Desktop is installed and running, then execute:
docker compose up --build
This brings up both the Airflow Scheduler and Webserver. The web UI can be accessed via http://localhost:8080.


**Cloud Deployment Workflow (AWS)**
  
S3 to Redshift via AWS Glue
  
Step 1: Upload CSV File to Amazon S3
•	Place the .csv dataset (e.g., lead_conversion.csv) in a dedicated S3 bucket.
  
Step 2: IAM Role Setup
Create a new IAM role with the following policies:
•	AmazonS3FullAccess
•	AmazonRedshiftFullAccess
•	AWSGlueConsoleFullAccess, AWSGlueServiceRole
•	SecretsManagerReadWrite
•	AWSKeyManagementServicePowerUser
  
Step 3: Redshift Serverless Setup
•	Create a Redshift Workgroup and Namespace.
•	Attach the above IAM role and link it to the default VPC and a custom security group.
  
Step 4: Configure VPC Endpoints
Create the following endpoints:
Service	Type	Notes
S3	Gateway	Ensure routing table association
Redshift	Interface	Select all subnets + security group
Secrets Manager	Interface	Select all subnets + security group
KMS	Interface	Select all subnets + security group
STS (Security Token Service)	Interface	Required for role assumption
  
Step 5: Configure Security Groups
•	Inbound:
o	Allow All TCP from 0.0.0.0/0
o	Allow Redshift Port 5439
•	Outbound:
o	Allow all outbound traffic
•	Add self-referencing rule to allow Redshift inter-node communication.
  
Step 6: AWS Glue Connection + Crawler + Job
1.	Create Glue Connection to Redshift:
o	Use Workgroup, Database, Port, Username, and Password.
2.	Run Crawler to catalog the S3 table metadata.
3.	Glue ETL Script:
python

import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read from S3
datasource = glueContext.create_dynamic_frame.from_catalog(
    database="sales-db", 
    table_name="lead_conversion_10r_csv", 
    transformation_ctx="datasource"
)

# Write to Redshift
glueContext.write_dynamic_frame.from_options(
    frame=datasource,
    connection_type="redshift",
    connection_options={
        "redshiftTmpDir": "s3://aws-glue-assets-250260913801-ap-south-1/temporary/",
        "useConnectionProperties": "true",
        "dbtable": "public.leads_data",
        "connectionName": "Redshift connection"
    },
    transformation_ctx="redshift_output"
)

job.commit()

Redshift to SageMaker Integration
Python Script to Extract Redshift Table into SageMaker
Python

import psycopg2
import pandas as pd

REDSHIFT_HOST = "wg1.250260913801.ap-south-1.redshift-serverless.amazonaws.com"
REDSHIFT_PORT = 5439
REDSHIFT_DB = "dev"
REDSHIFT_USER = "admin"
REDSHIFT_PASS = "Admin-123"
TABLE_NAME = "leads_dataa"

conn = psycopg2.connect(
    host=REDSHIFT_HOST,
    port=REDSHIFT_PORT,
    dbname=REDSHIFT_DB,
    user=REDSHIFT_USER,
    password=REDSHIFT_PASS,
    sslmode='require'
)

query = f"SELECT * FROM {TABLE_NAME};"
df = pd.read_sql(query, conn)
conn.close()
df.to_csv(f"{TABLE_NAME}.csv", index=False)
print(df.head())
4.10.3 SageMaker Setup for Model Training & Tracking
Step-by-step:
1.	Create a SageMaker Domain and a new User Profile.
2.	Launch Jupyter Lab and MLflow Tracking Server from SageMaker Studio.
3.	Upload your entire codebase into SageMaker.
4.	Modify your MLflow tracking URI:
>mlflow.set_tracking_uri("https://<tracking-server-endpoint>")
5.	Execute your training script:
o	Model is trained and logged to S3.
o	Best model is auto-registered.




Ngrok + Flask API for Real-time Inference
  
Step 1: Setup Flask App
# Upload and unzip Flask app
>unzip flask.zip
>cd flask
>pip install -r requirements.txt
>python3 app.py
  
Step 2: Install and Run Ngrok
>wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
>tar -xvzf ngrok-v3-stable-linux-amd64.tgz
>chmod +x ngrok
>./ngrok config add-authtoken <your-token>
>./ngrok http 127.0.0.1:8080

Step 3: Test
•	Use the Ngrok public forwarding URL to open the UI and get predictions using the latest best model deployed from SageMaker.
4.10.5 MWAA (Managed Airflow)
1.	Create a new MWAA Environment and link:
o	S3 bucket containing dags/, requirements.txt
o	A custom VPC, subnets, security groups
2.	Assign IAM role with permissions for:
o	Glue, Redshift, SageMaker, S3, and KMS
3.	Access the Airflow UI:
o	Trigger data_ingestion_dag.py
o	Trigger drift_detection_dag.py
o	If drift is detected, retraining DAG is activated post monthly schedule


  
</pre>



