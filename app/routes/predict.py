from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import numpy as np
from app.core.config import MODEL_PATH

router = APIRouter()

# Load model once
model = joblib.load(MODEL_PATH)

# Define expected parameters
class RainInput(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float
    pressure: float  # add as many as your model requires

@router.post("/predict_rain")
def predict_rain(input: RainInput):
    # Convert input to array
    features = np.array([[input.temperature, input.humidity, input.wind_speed, input.pressure]])
    
    # Predict
    prediction = model.predict(features)[0]  # assuming numeric output like mm of rain
    
    return {"prediction": f"Today it will rain approximately {prediction} mm."}
