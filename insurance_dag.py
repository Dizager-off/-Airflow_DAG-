from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

from tasks.insurance_prep import download_data, preprocess_data
from tasks.insurance_train import train

with DAG(
    dag_id="insurance_regression_pipeline",
    start_date=datetime(2025, 3, 1),
    schedule=None,
    max_active_runs=1,
    catchup=False,
) as dag:

    download = PythonOperator(
        task_id="download_dataset",
        python_callable=download_data
    )

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train
    )

    download >> preprocess >> train_task