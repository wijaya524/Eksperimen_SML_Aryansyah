import sys
sys.modules['torch'] = None
import mlflow
import pandas as pd
# 1. PERBAIKAN: Ganti Classifier menjadi Regressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Credit Scoring")


data = pd.read_csv('data_train.csv')

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("average_rating", axis=1),
    data['average_rating'],
    random_state=42,
    test_size=0.2
)

input_example = X_train[0:5]

n_estimator = 505
max_depth = 37


model = RandomForestRegressor(n_estimators=n_estimator, max_depth=max_depth, random_state=42)

mlflow.sklearn.autolog()

with mlflow.start_run():
  
    mlflow.log_param("n_estimator", n_estimator)
    mlflow.log_param("max_depth", max_depth)

 
    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

    r2_score = model.score(X_test, y_test)
    mlflow.log_metric("r2_score", r2_score)

print("Berjalan dengan sukses dan tercatat di MLflow!")