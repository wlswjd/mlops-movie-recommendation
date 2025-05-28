from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def run_logistic_experiment():
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://host.docker.internal:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
    mlflow.set_tracking_uri("http://host.docker.internal:5050")
    mlflow.set_experiment("imdb-logreg-dag")

    # 데이터 로드
    dataset = pd.read_csv("https://raw.githubusercontent.com/dair-ai/emotion_dataset/master/data/train.txt", sep=";", header=None)
    dataset.columns = ["text", "label"]

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"<br />", " ", text)
        text = re.sub(r"[^a-zA-Z]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    dataset["clean_text"] = dataset["text"].apply(clean_text)
    stop_words = stopwords.words("english")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=stop_words)

    X = vectorizer.fit_transform(dataset["clean_text"])
    y = dataset["label"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    with mlflow.start_run():
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

default_args = {
    "start_date": datetime(2023, 1, 1),
    "catchup": False
}

with DAG("imdb_logreg_dag", schedule_interval=None, default_args=default_args, tags=["ml"], description="IMDB Logistic Regression DAG") as dag:
    run_experiment = PythonOperator(
        task_id="run_logistic_experiment",
        python_callable=run_logistic_experiment
    )

