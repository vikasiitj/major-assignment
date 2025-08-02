# quantize.py
"""
Author: Vikash Kumar
Purpose: Manually quantize a trained Linear Regression model by compressing coefficients.
"""

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
import os

# Load model and extract weights
model = joblib.load("model.joblib")
original_params = {
    "coef": model.coef_,
    "intercept": model.intercept_
}

# Save unquantized model weights
joblib.dump(original_params, "unquant_params.joblib")

# Quantize coefficients and intercept manually
min_val = np.min(original_params["coef"])
scale = 255 / (np.max(original_params["coef"]) - min_val)
quant_coef = np.round(scale * (original_params["coef"] - min_val)).astype(np.uint8)
quant_intercept = np.round(scale * (original_params["intercept"] - min_val)).astype(np.uint8)

# Save quantized weights
joblib.dump({"coef": quant_coef, "intercept": quant_intercept}, "quant_params.joblib")

# Dequantize to restore for testing
restored_coef = quant_coef / scale + min_val
restored_intercept = quant_intercept / scale + min_val

# Evaluate on full dataset
data = fetch_california_housing()
X, y = data.data, data.target

pred_original = np.dot(X, original_params["coef"]) + original_params["intercept"]
pred_quantized = np.dot(X, restored_coef) + restored_intercept

# Print comparison
print("Original R²:", r2_score(y, pred_original))
print("Quantized R²:", r2_score(y, pred_quantized))
print("Original model size (KB):", os.path.getsize("unquant_params.joblib") / 1024)
print("Quantized model size (KB):", os.path.getsize("quant_params.joblib") / 1024)

