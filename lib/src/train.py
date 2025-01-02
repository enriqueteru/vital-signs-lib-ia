import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def train_and_save_model(
    data_path="data/raw/human_vital_signs_dataset_2024.csv",
    model_path="models/risk_model.pkl",
    n_estimators=100,
    test_size=0.2,
    random_state=42
):
    """
    Entrena un modelo Random Forest y guarda el archivo PKL en el sistema de archivos.

    :param data_path: Ruta al archivo CSV con los datos.
    :param model_path: Ruta donde se guardará el modelo entrenado.
    :param n_estimators: Número de árboles en el Random Forest.
    :param test_size: Proporción de los datos para el conjunto de prueba.
    :param random_state: Semilla para la reproducibilidad.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"El archivo de datos no se encontró en: {data_path}")

    # Cargar datos
    print("Cargando datos...")
    data = pd.read_csv(data_path)

    # Seleccionar columnas relevantes
    features = [
        "Heart Rate", "Respiratory Rate", "Body Temperature",
        "Oxygen Saturation", "Systolic Blood Pressure",
        "Diastolic Blood Pressure", "Age", "Gender",
        "Weight (kg)", "Height (m)"
    ]
    target = "Risk Category"

    # Preprocesamiento
    print("Preprocesando datos...")
    data["Gender"] = data["Gender"].map({"Male": 0, "Female": 1})
    data["Risk Category"] = data["Risk Category"].map({"Low Risk": 0, "High Risk": 1})
    data = data.dropna(subset=features + [target])

    X = data[features]
    y = data[target]

    # Dividir datos
    print("Dividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Entrenar modelo
    print("Entrenando modelo...")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # Evaluar modelo
    print("Evaluando modelo...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

    # Guardar modelo
    print(f"Guardando modelo en {model_path}...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print("Modelo guardado exitosamente.")
