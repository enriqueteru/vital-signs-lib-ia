from src.train import train_and_save_model
import os

def test_train_and_save_model():
    data_path = "data/raw/human_vital_signs_dataset_2024.csv"
    model_path = "models/test_risk_model.pkl"
    train_and_save_model(data_path=data_path, model_path=model_path)
    assert os.path.exists(model_path)