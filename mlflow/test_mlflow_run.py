
import os

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"


import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

# 데이터 생성
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# 모델 정의 및 학습
model = LinearRegression()
model.fit(X, y)

# 예측 및 평가
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)

# MLflow 실험 기록 시작
mlflow.set_tracking_uri("http://localhost:5050")
mlflow.set_experiment("movie-recommendation-test")

with mlflow.start_run():
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

    print(f"✅ 실험 완료! MSE: {mse}")

