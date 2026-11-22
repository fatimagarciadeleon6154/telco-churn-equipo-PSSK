import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(config: dict):
    """
    Carga, limpia y divide el dataset de Telco Customer Churn.
    Retorna: X_train, X_test, y_train, y_test
    """
  
    raw_path = config["data"]["raw_path"]
    df = pd.read_csv(raw_path)

    
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    
    df = pd.get_dummies(df, drop_first=True, dtype=int)

    
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test



load_data = load_and_preprocess_data



if __name__ == "__main__":
    mock_config = {
        "data": {
            "raw_path": "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
            "test_size": 0.2,
            "random_state": 42
        }
    }
    X_train, X_test, y_train, y_test = load_and_preprocess_data(mock_config)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Valores nulos restantes: {X_train.isnull().sum().sum()}")
    print("¡Preprocesamiento exitoso!")