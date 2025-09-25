import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt

# Load dataset
file_path = "/dehradun_daily_2020_2024.csv"
df = pd.read_csv(file_path)

# Feature selection
features = ["tavg_C", "tmin_C", "tmax_C", "wind_speed_mps", "pressure_hPa", "sunshine_minutes"]
target = "rainfall_mm"

X = df[features]
y = df[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Plot actual vs predicted
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Rainfall (mm)")
plt.ylabel("Predicted Rainfall (mm)")
plt.title("Random Forest Regression - Rainfall Prediction")
plt.show()

# Save model as pickle
with open("dehradun_rainfall_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as dehradun_rainfall_model.pkl")

# Example: load the model again
loaded_model = pickle.load(open("dehradun_rainfall_model.pkl", "rb"))
sample_pred = loaded_model.predict([X_test.iloc[0]])
print("Sample prediction:", sample_pred)
