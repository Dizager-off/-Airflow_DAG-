import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Вечная прямая ссылка на датасет
DATASET_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"

RAW_DATA = "/home/alexfomin/airflow/data/insurance_raw.csv"
CLEAN_DATA = "/home/alexfomin/airflow/data/insurance_clean.csv"


def download_data():
    print("Скачивание датасета...")
    df = pd.read_csv(DATASET_URL)
    df.to_csv(RAW_DATA, index=False)
    print(f"Скачано. Строк: {len(df)}")
    return True


def preprocess_data():
    print("Предобработка и нормализация...")
    df = pd.read_csv(RAW_DATA)

    # Кодирование категориальных признаков
    df = pd.get_dummies(df, drop_first=True, dtype=int)

    # Нормализация всех признаков кроме целевого
    scaler = StandardScaler()
    features = df.drop(columns=["charges"])
    df[features.columns] = scaler.fit_transform(features)

    df.to_csv(CLEAN_DATA, index=False)
    print(f"✅ Предобработка завершена")
    return True
