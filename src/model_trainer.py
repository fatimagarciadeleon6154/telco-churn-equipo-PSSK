import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_model(model_config: dict):
    """
    Fábrica de modelos: crea el modelo indicado en config["model"]["name"]
    con los hiperparámetros de config["model"]["params"].
    """
    name = model_config["name"]
    params = model_config.get("params") or {}

    if name == "random_forest":
        return RandomForestClassifier(**params)

    if name == "logistic_regression":
        # Este modelo funciona mejor con los datos escalados
        return make_pipeline(StandardScaler(), LogisticRegression(**params))

    if name == "svm":
        # La SVM también necesita los datos escalados
        return make_pipeline(StandardScaler(), SVC(**params))

    raise ValueError(
        f"Modelo '{name}' no soportado. "
        "Usa: random_forest, logistic_regression o svm"
    )


def train_and_save_model(X_train, y_train, X_test, y_test, config: dict) -> dict:
    """
    Entrena el modelo elegido en el config, calcula sus métricas
    y lo guarda con joblib.
    Retorna: diccionario con accuracy, recall y f1.
    """
    # 1. Crear y entrenar el modelo
    model = get_model(config["model"])
    model.fit(X_train, y_train)

    # 2. Evaluar con los datos de prueba
    y_pred = model.predict(X_test)
    metrics = {
        "model": config["model"]["name"],
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
    }

    # 3. Guardar el modelo en models/
    model_path = config["paths"]["model_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Modelo guardado en: {model_path}")

    return metrics


if __name__ == "__main__":
    # Usa la función de la Data Engineer y entrena cada modelo
    from src.data_loader import load_and_preprocess_data

    mock_config = {
        "data": {
            "raw_path": "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
            "test_size": 0.2,
            "random_state": 42,
        },
        "paths": {"model_path": "models/model.pkl"},
    }
    X_train, X_test, y_train, y_test = load_and_preprocess_data(mock_config)

    for name, params in [
        ("logistic_regression", {"max_iter": 1000}),
        ("svm", {"C": 1.0, "kernel": "rbf"}),
        ("random_forest", {"n_estimators": 100, "random_state": 42}),
    ]:
        mock_config["model"] = {"name": name, "params": params}
        print(train_and_save_model(X_train, y_train, X_test, y_test, mock_config))
