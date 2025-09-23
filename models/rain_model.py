# create_dummy_model.py
import joblib
from sklearn.linear_model import LinearRegression
import numpy as np

# Dummy training data: temperature, humidity, wind_speed, pressure
X = np.array([
    [25, 80, 5, 1010],
    [30, 70, 3, 1005],
    [28, 75, 4, 1008],
    [22, 90, 2, 1015],
    [35, 60, 6, 1000]
])

# Dummy target: rainfall in mm
y = np.array([10, 5, 8, 12, 2])

# Train dummy model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "rain_model.pkl")
print("Dummy rain_model.pkl created!")
