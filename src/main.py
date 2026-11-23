import os
import yaml
from pathlib import Path
from src.data_loader import load_and_preprocess_data
from src.model_trainer import train_and_save_model


def load_config(config_path: str = "config/params.yaml") -> dict:
    """Carga el archivo de configuración YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración en: {config_path}")
    
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def run_pipeline():
    """Orquesta la ejecución completa del pipeline de MLOps."""
    print("Iniciando pipeline de predicción de Churn...")

    # 1. Cargar configuración
    config = load_config()
    print(f"Configuración cargada. Modelo seleccionado: {config['model']['name']}")

    # Asegurar que el directorio de salida para el modelo exista
    model_output_dir = os.path.dirname(config["model"]["output_path"])
    if model_output_dir:
        os.makedirs(model_output_dir, exist_ok=True)

    # 2. Carga y preprocesamiento de datos (Rol 1: Data Engineer)
    print("Ejecutando extracción y limpieza de datos...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)
    print(f"Datos listos -> Train: {X_train.shape}, Test: {X_test.shape}")

    # 3. Entrenamiento y guardado del modelo (Rol 2: ML Engineer)
    print("Entrenando y evaluando el modelo...")
    metrics = train_and_save_model(X_train, y_train, X_test, y_test, config)

# 4. Reporte de métricas finales
    print("\n📈 Resultados del entrenamiento:")
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"   - {metric_name}: {value:.4f}")
        else:
            print(f"   - {metric_name}: {value}")

    print(f"\nModelo guardado exitosamente en: {config['model']['output_path']}")
    print("Pipeline finalizado sin errores.")


if __name__ == "__main__":
    run_pipeline()