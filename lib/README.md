# Librería de Signos Vitales

## Descripción

La librería está diseñada para gestionar modelos de aprendizaje automático (Machine Learning) enfocados en clasificar riesgos basados en signos vitales. Proporciona funcionalidades completas para:

- **Entrenar:** Preprocesar datos crudos, entrenar un modelo y devolver métricas de evaluación.
- **Guardar:** Almacenar modelos entrenados en archivos binarios con soporte para versionado.
- **Cargar:** Recuperar modelos entrenados desde archivos específicos.
- **Testear:** Realizar predicciones utilizando datos preprocesados.

Esta librería se implementa en Python y se distribuye como un paquete instalable que puede integrarse en sistemas más amplios, como APIs o aplicaciones frontend.

---

## **Requisitos Previos**

1. **Docker**: Asegúrate de tener Docker instalado. Puedes descargarlo desde [docker.com](https://www.docker.com/).
2. **Datos**: Coloca el archivo de datos en la carpeta `data/raw/` con el nombre `human_vital_signs_dataset_2024.csv`.

---

## **Estructura del Proyecto**

```plaintext
lib/
├── data/                              # Carpeta para datos
│   ├── raw/                           # Datos sin procesar
│   │   └── human_vital_signs_dataset_2024.csv
│   ├── processed/                     # Datos preprocesados (opcional)
│       └── training_data.csv
├── models/                            # Carpeta para guardar modelos entrenados
│   ├── risk_model_v1.pkl              # Modelo entrenado (ejemplo)
│   ├── risk_model_test.pkl            # Modelo de prueba
├── src/                               # Código fuente de la librería
│   ├── __init__.py                    # Inicialización del paquete
│   ├── model_manager.py               # Clase principal de la librería
│   ├── preprocessing.py               # Métodos auxiliares para preprocesamiento
├── tests/                             # Pruebas unitarias
│   ├── __init__.py                    # Inicialización del módulo de pruebas
│   ├── test_model_manager.py          # Pruebas para ModelManager
│   ├── test_preprocessing.py          # Pruebas para el módulo de preprocesamiento
├── Dockerfile                         # Dockerfile para ejecutar pruebas
├── docker-compose.yml                 # Orquestación de servicios para pruebas
├── requirements.txt                   # Dependencias del proyecto
├── setup.py                           # Configuración del paquete
└── README.md                          # Documentación del proyecto
