from fastapi import FastAPI
from app.routes import predict

app = FastAPI(title="Rain Prediction API")

app.include_router(predict.router, prefix="/api", tags=["Prediction"])

@app.get("/")
def root():
    return {"message": "Rain Prediction API is running! ISRO PROJECT requirement "}
