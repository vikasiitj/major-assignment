# test_train.py
"""
Unit test to check if model.joblib exists and can make predictions.
"""

import os
import joblib
import numpy as np

def test_model_loading_and_prediction():
    assert os.path.exists("model.joblib"), "Model file not found!"

    model = joblib.load("model.joblib")
    dummy_input = np.random.rand(1, 8)  # 8 features for California Housing
    prediction = model.predict(dummy_input)

    assert prediction.shape == (1,), "Prediction shape mismatch!"
    print("✅ Model loaded successfully and made prediction:", prediction[0])

if __name__ == "__main__":
    test_model_loading_and_prediction()

