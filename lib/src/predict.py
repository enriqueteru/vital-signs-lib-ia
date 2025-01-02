from src.train import train_and_save_model
import os
import joblib
import pandas as pd

def predict(input_data, model_path="models/risk_model.pkl", data_path="data/raw/human_vital_signs_dataset_2024.csv"):
    """
    Realiza predicciones usando un modelo PKL.

    :param input_data: Diccionario con los datos necesarios para la predicción.
    :param model_path: Ruta al modelo entrenado (archivo PKL).
    :param data_path: Ruta al archivo CSV para entrenamiento si el modelo no existe.
    :return: Predicción (Low Risk o High Risk).
    """
    # Verificar si el modelo existe
    if not os.path.exists(model_path):
        print(f"Modelo no encontrado en {model_path}. Entrenando un nuevo modelo...")
        train_and_save_model(data_path=data_path, model_path=model_path)

    # Cargar el modelo
    model = joblib.load(model_path)

    # Crear un DataFrame con los datos de entrada
    features = [
        "Heart Rate", "Respiratory Rate", "Body Temperature",
        "Oxygen Saturation", "Systolic Blood Pressure",
        "Diastolic Blood Pressure", "Age", "Gender",
        "Weight (kg)", "Height (m)"
    ]
    input_df = pd.DataFrame([input_data], columns=features)

    # Realizar predicción
    prediction = model.predict(input_df)[0]
    return "High Risk" if prediction == 1 else "Low Risk"
