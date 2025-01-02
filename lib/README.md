# Librería de Signos Vitales

Librería para entrenar un modelo de aprendizaje automático que clasifica el riesgo basado en signos vitales. Usa un modelo de **Random Forest** para clasificar entre **"Low Risk"** y **"High Risk"**, y permite realizar predicciones desde un contenedor Docker.

---

## **Características**
- Entrenamiento del modelo basado en datos de signos vitales.
- Guarda los modelos entrenados en formato `.pkl` para su reutilización.
- Proporciona métodos para realizar predicciones con datos de entrada personalizados.
- Funciona dentro de un contenedor Docker, con volúmenes mapeados para guardar los modelos entrenados.

---

## **Requisitos Previos**

1. **Docker**: Asegúrate de tener Docker instalado. Puedes descargarlo desde [docker.com](https://www.docker.com/).
2. **Datos**: Coloca el archivo de datos en la carpeta `data/raw/` con el nombre `human_vital_signs_dataset_2024.csv`.

---

## **Estructura del Proyecto**

```plaintext
lib/
├── src/
│   ├── __init__.py                # Inicialización del paquete
│   ├── train.py                   # Entrenamiento del modelo
│   ├── predict.py                 # Predicciones
│   ├── utils.py                   # Funciones auxiliares
├── models/                        # Carpeta donde se guardarán los modelos PKL
├── data/
│   ├── raw/                       # Carpeta para datos sin procesar
│   │   └── human_vital_signs_dataset_2024.csv
├── tests/                         # Pruebas unitarias
├── setup.py                       # Configuración de la librería
├── requirements.txt               # Dependencias del proyecto
├── Dockerfile                     # Dockerfile para construir el entorno
└── README.md                      # Documentación
