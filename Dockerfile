FROM apache/airflow:2.8.1-python3.9

USER root

# Install libgomp1 to resolve OSError for libgomp.so.1
RUN apt-get update && \
    apt-get install -y libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./airflow/requirements.txt /requirements.txt

# Change ownership so the airflow user can access it
RUN chown airflow: /requirements.txt

# Ensure the log file path exists
RUN mkdir -p /opt/airflow/dags/airflow/dags && \
    touch /opt/airflow/dags/airflow/dags/drift_detected_log.txt

USER airflow

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt
