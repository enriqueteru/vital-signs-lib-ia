import pytest
from src.preprocessing import preprocess_data

def test_preprocess_data():
    data_path = "data/raw/human_vital_signs_dataset_2024.csv"
    X_train, X_test, y_train, y_test = preprocess_data(data_path)

    # Validar que los datos están divididos correctamente
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
