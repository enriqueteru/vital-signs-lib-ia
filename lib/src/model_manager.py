import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from .preprocessing import preprocess_data


class ModelManager:
    def __init__(self, default_data_path="data/raw/human_vital_signs_dataset_2024.csv"):
        """
        Inicializa el gestor del modelo.
        :param default_data_path: Ruta por defecto al archivo CSV con los datos.
        """
        self.default_data_path = default_data_path
        self.model = None
        self.version = None  # Versión del modelo cargado o entrenado

    def train(self, data_path=None, test_size=0.2, random_state=42, n_estimators=100):
        """
        Entrena el modelo con los datos proporcionados.
        :param data_path: Ruta al archivo CSV con los datos (opcional).
        :param test_size: Proporción del conjunto de prueba.
        :param random_state: Semilla para reproducibilidad.
        :param n_estimators: Número de árboles en el Random Forest.
        :return: Reporte de métricas del entrenamiento.
        """
        data_path = data_path or self.default_data_path

        # Preprocesar datos
        print(f"Cargando y preprocesando datos desde {data_path}...")
        X_train, X_test, y_train, y_test = preprocess_data(data_path, test_size, random_state)

        # Entrenar modelo
        print("Entrenando modelo...")
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.model.fit(X_train, y_train)

        # Evaluar modelo
        print("Evaluando modelo...")
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"])
        print(report)

        return report

    def save(self, output_dir="models", version="v1"):
        """
        Guarda el modelo entrenado en la ubicación y con la versión indicados por el usuario.
        :param output_dir: Carpeta donde se guardará el modelo.
        :param version: Versión del modelo, se usará para nombrar el archivo.
        """
        if not self.model:
            raise ValueError("No hay un modelo entrenado para guardar.")

        model_path = os.path.join(output_dir, f"risk_model_{version}.pkl")
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.model, model_path)
        self.version = version
        print(f"Modelo guardado en: {model_path}")

    def load(self, model_path=None, version="v1"):
        """
        Carga un modelo desde un archivo. Usa un modelo por defecto si no se proporciona una ruta.
        :param model_path: Ruta al archivo del modelo (opcional).
        :param version: Versión del modelo a cargar (usada si no se proporciona model_path).
        """
        if not model_path:
            model_path = f"models/risk_model_{version}.pkl"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"El modelo no se encontró en {model_path}.")

        self.model = joblib.load(model_path)
        self.version = version
        print(f"Modelo cargado desde: {model_path} (versión: {version})")

    def test(self, input_data):
        """
        Realiza predicciones con el modelo entrenado.
        :param input_data: Diccionario con las características de entrada.
        :return: Predicción (Low Risk o High Risk).
        """
        if not self.model:
            raise ValueError("El modelo no está cargado. Usa load() para cargar un modelo entrenado.")

        # Crear un DataFrame con los datos de entrada
        features = [
            "Heart Rate", "Respiratory Rate", "Body Temperature",
            "Oxygen Saturation", "Systolic Blood Pressure",
            "Diastolic Blood Pressure", "Age", "Gender",
            "Weight (kg)", "Height (m)"
        ]
        input_df = pd.DataFrame([input_data], columns=features)

        # Realizar predicción
        prediction = self.model.predict(input_df)[0]
        return "High Risk" if prediction == 1 else "Low Risk"
