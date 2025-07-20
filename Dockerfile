# Base image used is Airflow with Python 3.9 support
FROM apache/airflow:2.8.1-python3.9

# Switch to root user to allow installation of system-level packages
USER root

# Updates the package list and installs libgomp1 required for XGBoost
# Cleans up cache to reduce image size
RUN apt-get update && \
    apt-get install -y libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copies Python requirements to container for dependency installation
COPY ./airflow/requirements.txt /requirements.txt

# Sets ownership of the requirements file to the airflow user
RUN chown airflow: /requirements.txt

# Creates the directory path for DAGs and a log file for drift detection
RUN mkdir -p /opt/airflow/dags/airflow/dags && \
    touch /opt/airflow/dags/airflow/dags/drift_detected_log.txt

# Switches back to airflow user for running airflow commands securely
USER airflow

# Installs Python libraries specified in requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
