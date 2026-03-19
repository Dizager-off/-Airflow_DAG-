import pandas as pd
import numpy as np
import joblib
import mlflow

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CLEAN_DATA = "/home/alexfomin/airflow/data/insurance_clean.csv"
MODEL_PATH = "/home/alexfomin/airflow/data/insurance_model.pkl"
MLFLOW_URI = "file:///home/alexfomin/airflow/mlruns"


def train():
    print("🤖 Обучение модели регрессии...")
    df = pd.read_csv(CLEAN_DATA)

    X = df.drop(columns=["charges"])
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Подбор гиперпараметров ровно как в твоей оригинальной лабораторной
    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'l1_ratio': [0.001, 0.01, 0.1, 0.2],
        "penalty": ["l1", "l2", "elasticnet"],
        "loss": ['squared_error', 'huber']
    }

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("insurance_regression")

    with mlflow.start_run():
        model = SGDRegressor(random_state=42)
        grid = GridSearchCV(model, params, cv=3, n_jobs=1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        # Расчёт стандартных метрик регрессии
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"📊 Итоговые метрики:")
        print(f"RMSE: {rmse:.0f} евро")
        print(f"MAE:  {mae:.0f} евро")
        print(f"R2:   {r2:.3f}")

        # Логирование всего в MLflow
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        mlflow.sklearn.log_model(best_model, "model")

        # Сохранение модели локально
        joblib.dump(best_model, MODEL_PATH)

        print("✅ Обучение успешно завершено!")
        return True
