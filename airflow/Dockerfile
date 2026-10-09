FROM apache/airflow:3.0.6

USER root
COPY requirements.txt /tmp/requirements.txt

USER airflow
RUN pip install --no-cache-dir -r /tmp/requirements.txt