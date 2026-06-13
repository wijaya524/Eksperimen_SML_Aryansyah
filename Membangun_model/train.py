import os
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split


if os.environ.get("GITHUB_ACTIONS") == "true":
    print("Berjalan di GitHub Actions CI: Menggunakan local file store (mlruns)...")
   
else:
    print("Berjalan di Komputer Lokal: Menghubungkan ke server MLflow lokal...")
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")


mlflow.set_experiment("Credit Scoring")


csv_path = os.environ.get("CSV_URL", "modelling/data_train.csv")
target_var = os.environ.get("TARGET_VAR", "average_rating")

# Membaca dataset
data = pd.read_csv(csv_path)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(target_var, axis=1),
    data[target_var],
    random_state=42,
    test_size=0.2
)

input_example = X_train[0:5]


n_estimator = 505
max_depth = 37


model = RandomForestRegressor(n_estimators=n_estimator, max_depth=max_depth, random_state=42)


mlflow.sklearn.autolog()

with mlflow.start_run(nested=True):
  
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

print("Training selesai dengan sukses dan dieksport ke MLflow!")
