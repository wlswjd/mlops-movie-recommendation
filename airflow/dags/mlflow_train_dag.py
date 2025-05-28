from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

with DAG(
    dag_id='mlflow_train_pipeline',
    default_args=default_args,
    description='MLflow 실험 자동 실행 DAG',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@once',
    catchup=False,
    tags=['mlflow', 'train']
) as dag:

    run_mlflow_script = BashOperator(
        task_id='run_mlflow_test_script',
        bash_command='source /opt/airflow/venv/bin/activate && python /opt/airflow/mlflow/test_mlflow_run.py',
        env={
            'MLFLOW_S3_ENDPOINT_URL': 'http://minio:9000',
            'AWS_ACCESS_KEY_ID': 'minioadmin',
            'AWS_SECRET_ACCESS_KEY': 'minioadmin',
            'MLFLOW_TRACKING_URI': 'http://mlflow:5000'
        }
    )

    run_mlflow_script

