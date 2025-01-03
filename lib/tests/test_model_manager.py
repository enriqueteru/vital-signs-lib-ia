import pytest
import os
from src.model_manager import ModelManager

def test_training():
    manager = ModelManager()
    report = manager.train()
    assert "Low Risk" in report and "High Risk" in report

def test_save_model():
    manager = ModelManager()
    manager.train()
    manager.save(version="test")
    model_path = "models/risk_model_test.pkl"
    assert os.path.exists(model_path)

def test_load_model():
    manager = ModelManager()
    model_path = "models/risk_model_test.pkl"
    manager.load(model_path=model_path, version="test")
    assert manager.model is not None
    assert manager.version == "test"

def test_prediction():
    manager = ModelManager()
    manager.load(model_path="models/risk_model_test.pkl", version="test")

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
    prediction = manager.test(input_data)
    assert prediction in ["Low Risk", "High Risk"]
