# train.py
"""
Author: Vikash Kumar
Purpose: Train a Linear Regression model on California Housing data and save the model.
"""

from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate and print R² score
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Model R² Score: {r2:.4f}")

# Save the trained model
joblib.dump(model, "model.joblib")

