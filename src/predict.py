# predict.py
"""
Author: Vikash Kumar
Purpose: Load the trained model and make a prediction on random input.
"""

import joblib
import numpy as np

# Load the saved model
model = joblib.load("model.joblib")

# Create dummy input for prediction
X_dummy = np.random.rand(1, 8)  # California housing has 8 features
prediction = model.predict(X_dummy)

print("Dummy input:", X_dummy)
print("Predicted value:", prediction[0])

