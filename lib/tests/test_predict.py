import os
from src.predict import predict
from src.train import train_and_save_model

def test_predict():
    print("Iniciando test de predicción...")  # Primer punto de impresión

    # Asegúrate de que el modelo esté entrenado
    data_path = "data/raw/human_vital_signs_dataset_2024.csv"
    model_path = "models/risk_model.pkl"
    if not os.path.exists(model_path):
        print("Modelo no encontrado. Entrenando modelo...")
        train_and_save_model(data_path=data_path, model_path=model_path)
        print("Modelo entrenado y guardado.")

    # Datos de prueba
    input_data = {
        "Heart Rate": 72,
        "Respiratory Rate": 16,
        "Body Temperature": 36.5,
        "Oxygen Saturation": 98.5,
        "Systolic Blood Pressure": 120,
        "Diastolic Blood Pressure": 80,
        "Age": 30,
        "Gender": 0,
        "Weight (kg)": 70,
        "Height (m)": 1.75
    }
    print("Realizando predicción...")
    prediction = predict(input_data, model_path=model_path)
    print(f"Resultado de la predicción: {prediction}")
    assert prediction in ["Low Risk", "High Risk"]

test_predict()
