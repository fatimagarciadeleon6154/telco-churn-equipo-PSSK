import os
import pandas as pd
import pytest

from src.data_loader import load_and_preprocess_data
from src.model_trainer import train_and_save_model
from src.predict import get_sample_customer, load_model, predict_churn


@pytest.fixture
def sample_raw_csv(tmp_path):

    data = {
        "customerID": [f"000{i}-TEST" for i in range(10)],
        "gender": ["Female", "Male"] * 5,
        "SeniorCitizen": [0, 1] * 5,
        "Partner": ["Yes", "No"] * 5,
        "Dependents": ["No", "Yes"] * 5,
        "tenure": [1, 24, 5, 60, 2, 48, 12, 36, 3, 72],
        "PhoneService": ["Yes"] * 10,
        "MultipleLines": ["No", "Yes", "No phone service", "Yes", "No"] * 2,
        "InternetService": ["Fiber optic", "DSL", "No", "Fiber optic", "DSL"] * 2,
        "OnlineSecurity": ["No", "Yes", "No internet service", "Yes", "No"] * 2,
        "OnlineBackup": ["Yes", "No", "No internet service", "Yes", "No"] * 2,
        "DeviceProtection": ["No", "Yes", "No internet service", "Yes", "No"] * 2,
        "TechSupport": ["No", "Yes", "No internet service", "Yes", "No"] * 2,
        "StreamingTV": ["Yes", "No", "No internet service", "Yes", "No"] * 2,
        "StreamingMovies": ["No", "Yes", "No internet service", "Yes", "No"] * 2,
        "Contract": ["Month-to-month", "One year", "Two year", "Two year", "Month-to-month"] * 2,
        "PaperlessBilling": ["Yes", "No"] * 5,
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
        ] * 2,
        "MonthlyCharges": [70.5, 55.0, 20.0, 95.0, 65.0, 80.0, 45.0, 90.0, 75.0, 100.0],
        "TotalCharges": [" ", "1320.0", "100.0", "5700.0", " ", "3840.0", "540.0", "3240.0", "225.0", "7200.0"],
        "Churn": ["Yes", "No", "No", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
    }
    csv_path = tmp_path / "mock_telco.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return str(csv_path)


def test_load_and_preprocess_data(sample_raw_csv):
    config = {
        "data": {
            "raw_path": sample_raw_csv,
            "test_size": 0.2,
            "random_state": 42,
        }
    }
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)

    assert not X_train.empty, "X_train no debe estar vacío"
    assert not X_test.empty, "X_test no debe estar vacío"
    assert "customerID" not in X_train.columns, "customerID debió ser eliminado"
    assert X_train.isnull().sum().sum() == 0, "No deben quedar valores nulos en X_train"
    assert set(y_train.unique()).issubset({0, 1}), "y_train debe estar codificado en 0 y 1"


def test_train_and_predict_end_to_end(sample_raw_csv, tmp_path):
    model_file = str(tmp_path / "test_model.pkl")
    config = {
        "data": {
            "raw_path": sample_raw_csv,
            "test_size": 0.2,
            "random_state": 42,
        },
        "model": {
            "name": "random_forest",
            "params": {"n_estimators": 10, "random_state": 42},
        },
        "paths": {
            "model_path": model_file,
        },
    }

    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)
    metrics = train_and_save_model(X_train, y_train, X_test, y_test, config)

    assert os.path.exists(model_file), "El archivo del modelo .pkl debe existir"
    assert "accuracy" in metrics and "recall" in metrics and "f1" in metrics

    resultado = predict_churn(X_test.iloc[[0]], model_path=model_file)
    assert resultado["prediction"] in [0, 1], "La predicción debe ser 0 o 1"


def test_sample_customer_schema():
    """Verifica que el cliente de ejemplo tenga las 30 columnas esperadas sin nulos."""
    df_sample = get_sample_customer()
    assert df_sample.shape == (1, 30), f"Se esperaban 30 columnas, se obtuvieron {df_sample.shape[1]}"
    assert df_sample.isnull().sum().sum() == 0


def test_load_model_missing_file():
    """Verifica que se lance FileNotFoundError claro si el modelo no existe."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_model("models/modelo_que_no_existe.pkl")

    assert "python -m src.main" in str(exc_info.value)