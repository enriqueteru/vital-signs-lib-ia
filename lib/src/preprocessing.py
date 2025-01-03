import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(data_path, test_size=0.2, random_state=42):
    """
    Carga y preprocesa los datos desde un archivo CSV.
    :param data_path: Ruta al archivo CSV.
    :param test_size: Proporción del conjunto de prueba.
    :param random_state: Semilla para reproducibilidad.
    :return: X_train, X_test, y_train, y_test
    """
    # Cargar datos
    data = pd.read_csv(data_path)

    # Seleccionar características y etiquetas
    features = [
        "Heart Rate", "Respiratory Rate", "Body Temperature",
        "Oxygen Saturation", "Systolic Blood Pressure",
        "Diastolic Blood Pressure", "Age", "Gender",
        "Weight (kg)", "Height (m)"
    ]
    target = "Risk Category"

    # Preprocesamiento
    data["Gender"] = data["Gender"].map({"Male": 0, "Female": 1})
    data["Risk Category"] = data["Risk Category"].map({"Low Risk": 0, "High Risk": 1})
    data = data.dropna(subset=features + [target])

    X = data[features]
    y = data[target]

    # Dividir datos
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
