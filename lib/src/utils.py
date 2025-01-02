import pandas as pd

def validate_data(input_data):
    """
    Valida que los datos de entrada sean correctos.

    :param input_data: Diccionario con los datos necesarios.
    :return: True si los datos son válidos, lanza una excepción en caso contrario.
    """
    features = [
        "Heart Rate", "Respiratory Rate", "Body Temperature",
        "Oxygen Saturation", "Systolic Blood Pressure",
        "Diastolic Blood Pressure", "Age", "Gender",
        "Weight (kg)", "Height (m)"
    ]

    for feature in features:
        if feature not in input_data:
            raise ValueError(f"Falta el campo '{feature}' en los datos proporcionados.")
    return True
