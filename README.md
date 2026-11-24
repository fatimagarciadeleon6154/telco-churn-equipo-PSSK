# Predicción de Abandono de Clientes (Telco Customer Churn) — Arquitectura MLOps

Pipeline modular y reproducible de aprendizaje automático (*Machine Learning*) diseñado bajo principios de arquitectura MLOps para predecir la cancelación de servicio (*Churn*) en clientes de telecomunicaciones. El sistema desacopla la configuración, el procesamiento de datos, el entrenamiento, la serialización del modelo y la inferencia.

## Información del Proyecto

* **Institución:** Universidad de Guadalajara — Centro Universitario de Guadalajara (CUGDL)
* **Asignatura:** Arquitectura de Inteligencia Artificial
* **Repositorio:** [fatimagarciadeleon6154/telco-churn-equipo-PSSK](https://github.com/fatimagarciadeleon6154/telco-churn-equipo-PSSK)

### Integrantes — Equipo PSSK

* Kevin Uziel García Márquez
* Fátima Sofía García de León Torres
* Paola Lugo Avila
* Sophia Diaz de Leon Gleason

## Arquitectura del Repositorio

```text
telco-churn-equipo-PSSK/
├── config/
│   └── params.yaml                              # Rutas, semillas aleatorias e hiperparámetros del modelo
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv # Dataset original (7,043 registros y 21 columnas)
├── models/
│   └── model.pkl                                # Modelo entrenado y serializado listo para inferencia
├── src/
│   ├── __init__.py                              # Inicialización del paquete fuente
│   ├── data_loader.py                           # Carga, limpieza, codificación y partición estratificada
│   ├── model_trainer.py                         # Entrenamiento, cálculo de métricas y serialización
│   ├── main.py                                  # Orquestador principal del pipeline completo
│   └── predict.py                               # Módulo de inferencia sobre nuevos registros
├── test/
│   ├── __init__.py                              # Inicialización del paquete de pruebas
│   └── test_pipeline.py                         # Pruebas automatizadas del pipeline con pytest
├── .gitignore                                   # Exclusión de entorno virtual, cachés y archivos temporales
├── README.md                                    # Guía de arquitectura, instalación y ejecución
└── requirements.txt                             # Dependencias del proyecto
```

## Instalación y Configuración del Entorno

1. **Clonar el repositorio y acceder a la carpeta del proyecto:**

   ```bash
   git clone https://github.com/fatimagarciadeleon6154/telco-churn-equipo-PSSK.git
   cd telco-churn-equipo-PSSK
   ```

2. **Crear y activar el entorno virtual (`.venv`):**

   * **Windows (PowerShell):**

     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

   * **Linux / macOS:**

     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Instalar las dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

## Descarga y Ubicación del Dataset

Antes de ejecutar el pipeline, asegúrate de contar con el conjunto de datos **Telco Customer Churn**:

1. **Descargar el archivo CSV:**
   * Descarga el dataset público **Telco Customer Churn** (`WA_Fn-UseC_-Telco-Customer-Churn.csv`, 7,043 registros y 21 columnas) desde [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) o desde el repositorio de datos de muestra de IBM.

2. **Crear el directorio de datos crudos (si no existe en tu copia local):**

   * **Windows (PowerShell):**

     ```powershell
     New-Item -ItemType Directory -Force -Path data\raw
     ```

   * **Linux / macOS:**

     ```bash
     mkdir -p data/raw
     ```

3. **Colocar el dataset en la ruta esperada:**
   * Mueve o copia el archivo descargado dentro de la carpeta `data\raw\` verificando que conserve exactamente su nombre original:

     ```text
     telco-churn-equipo-PSSK\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv
     ```

## Configuración (`config/params.yaml`)

Todos los parámetros del experimento se gestionan de forma externalizada en `config/params.yaml` para garantizar reproducibilidad sin modificar el código fuente en `src/`:

* **`paths`:** Rutas relativas hacia el dataset crudo (`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`) y el archivo de salida del modelo (`models/model.pkl`).
* **`split` / `data`:** Proporción del conjunto de prueba (`test_size`) y semilla determinista (`random_state`).
* **`model`:** Hiperparámetros del algoritmo de clasificación supervisada.

## Instrucciones de Ejecución

> **Importante:** Ejecuta todos los comandos desde el directorio raíz `telco-churn-equipo-PSSK` con el entorno virtual `.venv` activado.

### 1. Ejecutar el Pipeline Completo (Entrenamiento y Evaluación)

Orquesta la lectura de `config/params.yaml`, la limpieza y partición de datos, el entrenamiento del clasificador, el despliegue de métricas en consola (`Accuracy`, `Precision`, `Recall`, `F1-Score`, `ROC-AUC`) y la exportación del artefacto `models/model.pkl`:

```bash
python -m src.main
```

### 2. Ejecutar Inferencia sobre Nuevos Datos

Carga el modelo serializado desde `models/model.pkl` y genera predicciones de clase (`0 = No Churn`, `1 = Churn`) y probabilidades estimadas sin reentrenar:

```bash
python -m src.predict
```

### 3. Ejecutar Pruebas Automatizadas

Verifica el funcionamiento de la carga de datos, el preprocesamiento, el entrenamiento, la serialización y la predicción mediante `pytest`:

```bash
python -m pytest -v
```

## Flujo del Pipeline MLOps

1. **Ingesta y Preprocesamiento (`src/data_loader.py`):** Lee el archivo CSV, elimina `customerID`, convierte `TotalCharges` a formato numérico gestionando valores vacíos, codifica las variables categóricas y divide los datos de manera estratificada.
2. **Entrenamiento y Evaluación (`src/model_trainer.py`):** Ajusta el modelo con el conjunto de entrenamiento, calcula las métricas de clasificación sobre el conjunto de prueba y serializa el modelo entrenado en `models/model.pkl`.
3. **Orquestación (`src/main.py`):** Coordina la ejecución secuencial de todas las etapas utilizando los parámetros definidos en `config/params.yaml`.
4. **Inferencia (`src/predict.py`):** Reutiliza `models/model.pkl` para emitir predicciones sobre nuevos perfiles de clientes.
5. **Validación (`test/test_pipeline.py`):** Asegura la integridad de cada etapa mediante pruebas automatizadas con `pytest`.