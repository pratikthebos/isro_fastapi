

# from fastapi import APIRouter
# from pydantic import BaseModel
# import joblib
# import numpy as np
# from datetime import datetime
# from app.core.config import MODEL_PATH

# router = APIRouter()

# # Load trained ML model once
# model = joblib.load(MODEL_PATH)

# # Input schema (date + weather params)
# class RainInput(BaseModel):
#     date: str  # e.g. "2025-09-25"
#     temperature: float
#     humidity: float
#     pressure: float
#     wind_speed: float

# @router.post("/predict_rain")
# def predict_rain(input: RainInput):
#     # Parse date to extract year & month
#     try:
#         parsed_date = datetime.strptime(input.date, "%Y-%m-%d")
#         year = parsed_date.year
#         month = parsed_date.month
#     except ValueError:
#         return {"error": "Invalid date format. Use YYYY-MM-DD"}

#     # Arrange features in the same order as model training
#     features = np.array([[ 
#         year,
#         month,
#         input.temperature,
#         input.humidity,
#         input.pressure,
#         input.wind_speed
#     ]])

#     # Make prediction
#     prediction = model.predict(features)[0]

#     return {
#         "date": input.date,
#         "prediction_mm": float(prediction),
#         "message": f"Predicted rainfall on {input.date} is approximately {prediction:.2f} mm."
#     }


# app/routes/predict.py

from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import numpy as np
from datetime import datetime
from app.core.config import MODEL_PATH

router = APIRouter()

# Load trained ML model once
model = joblib.load(MODEL_PATH)

# Input schema (only date)
class RainInput(BaseModel):
    date: str  # e.g. "2025-09-25"

@router.post("/predict_rain")
def predict_rain(input: RainInput):
    # Parse date to extract year & month
    try:
        parsed_date = datetime.strptime(input.date, "%Y-%m-%d")
        year = parsed_date.year
        month = parsed_date.month
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD"}

    # For demo: you can use placeholder/fixed values for temperature, humidity, etc.
    temperature = 30.0
    humidity = 70.0
    pressure = 1010.0
    wind_speed = 5.0

    # Arrange features in the same order as model training
    features = np.array([[ 
        year,
        month,
        temperature,
        humidity,
        pressure,
        wind_speed
    ]])

    # Make prediction
    prediction = model.predict(features)[0]

    return {
        "date": input.date,
        "prediction_mm": float(prediction),
        "message": f"Predicted rainfall on {input.date} is approximately {prediction:.2f} mm."
    }
