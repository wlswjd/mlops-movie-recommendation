from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def say_hello():
    print("Hello, 슈퍼진정맨님! DAG 실행 성공 🎉")

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

with DAG(
    dag_id='hello_superjinjeongman',
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval='@once',
    catchup=False,
    tags=['test']
) as dag:
    task1 = PythonOperator(
        task_id='say_hello_task',
        python_callable=say_hello
    )

    task1

