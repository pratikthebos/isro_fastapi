# from fastapi import APIRouter
# from pydantic import BaseModel
# import joblib
# import numpy as np
# from app.core.config import MODEL_PATH

# router = APIRouter()

# # Load model once
# model = joblib.load(MODEL_PATH)

# # Define expected parameters
# class RainInput(BaseModel):
#     temperature: float
#     humidity: float
#     wind_speed: float
#     pressure: float  # add as many as your model requires

# @router.post("/predict_rain")
# def predict_rain(input: RainInput):
#     # Convert input to array
#     features = np.array([[input.temperature, input.humidity, input.wind_speed, input.pressure]])
    
#     # Predict
#     prediction = model.predict(features)[0]  # assuming numeric output like mm of rain
    
#     return {"prediction": f"Today it will rain approximately {prediction} mm."}


# app/routes/predict.py
# app/routes/predict.py

from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import numpy as np
from app.core.config import MODEL_PATH

router = APIRouter()

# Load trained ML model
model = joblib.load(MODEL_PATH)

# Define input schema (6 features only)
class RainInput(BaseModel):
    year: int
    month: int
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float

@router.post("/predict_rain")
def predict_rain(input: RainInput):
    # Arrange features in same order as training
    features = np.array([[ 
        input.year,
        input.month,
        input.temperature,
        input.humidity,
        input.pressure,
        input.wind_speed
    ]])

    # Make prediction
    prediction = model.predict(features)[0]

    return {
        "prediction_mm": float(prediction),
        "message": f"Predicted rainfall is approximately {prediction:.2f} mm."
    }
