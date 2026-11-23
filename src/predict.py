import os
import sys
import joblib
import pandas as pd


def load_model(model_path: str = "models/model.pkl"):

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"\n[ERROR] no se encontró el archivo del modelo en '{model_path}'.\n"
            "asegúrate de haber ejecutado el pipeline de entrenamiento primero con:\n"
            "    python -m src.main"
        )
    try:
        return joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"[ERROR] el archivo '{model_path}' está corrupto o no se pudo cargar: {e}") from e


def get_sample_customer() -> pd.DataFrame:

    sample_data = {
        "SeniorCitizen": [0],
        "tenure": [2],
        "MonthlyCharges": [85.50],
        "TotalCharges": [171.00],
        "gender_Male": [1],
        "Partner_Yes": [0],
        "Dependents_Yes": [0],
        "PhoneService_Yes": [1],
        "MultipleLines_No phone service": [0],
        "MultipleLines_Yes": [1],
        "InternetService_Fiber optic": [1],
        "InternetService_No": [0],
        "OnlineSecurity_No internet service": [0],
        "OnlineSecurity_Yes": [0],
        "OnlineBackup_No internet service": [0],
        "OnlineBackup_Yes": [0],
        "DeviceProtection_No internet service": [0],
        "DeviceProtection_Yes": [0],
        "TechSupport_No internet service": [0],
        "TechSupport_Yes": [0],
        "StreamingTV_No internet service": [0],
        "StreamingTV_Yes": [1],
        "StreamingMovies_No internet service": [0],
        "StreamingMovies_Yes": [1],
        "Contract_One year": [0],
        "Contract_Two year": [0],  # Contrato mes a mes
        "PaperlessBilling_Yes": [1],
        "PaymentMethod_Credit card (automatic)": [0],
        "PaymentMethod_Electronic check": [1],
        "PaymentMethod_Mailed check": [0],
    }
    return pd.DataFrame(sample_data)


def predict_churn(customer_df: pd.DataFrame, model_path: str = "models/model.pkl") -> dict:

    model = load_model(model_path)

    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
        if list(customer_df.columns) != expected_cols:
            raise ValueError(
                "las columnas de entrada no coinciden con las esperadas por el modelo.\n"
                f"faltantes: {set(expected_cols) - set(customer_df.columns)}"
            )

    prediction = int(model.predict(customer_df)[0])
    result = {
        "prediction": prediction,
        "label": "Yes (Abandona)" if prediction == 1 else "No (Permanece)",
    }

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(customer_df)[0][1])
        result["churn_probability"] = round(proba, 4)

    return result


if __name__ == "__main__":
    MODEL_PATH = "models/model.pkl"
    try:
        print("cargando modelo y evaluando cliente de prueba...")
        cliente = get_sample_customer()
        resultado = predict_churn(cliente, model_path=MODEL_PATH)

        print("\n resultado de inferencia")
        print(f"predicción: {resultado['label']} (Clase: {resultado['prediction']})")
        if "churn_probability" in resultado:
            print(f"probabilidad de churn: {resultado['churn_probability'] * 100:.2f}%")

    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)