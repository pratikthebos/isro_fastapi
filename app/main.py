# from fastapi import FastAPI
# from app.routes import predict

# app = FastAPI(title="Rain Prediction API")

# app.include_router(predict.router, prefix="/api", tags=["Prediction"])

# @app.get("/")
# def root():
#     return {"message": "Rain Prediction API is running! ISRO PROJECT requirement "}
from app.routes import predict
from fastapi import FastAPI
import httpx
import asyncio

app = FastAPI(title="Rain Prediction API")
app.include_router(predict.router, prefix="/api", tags=["Prediction"])

URL = " https://isro-fastapi.onrender.com"
INTERVAL = 30  # seconds

async def ping_website():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(URL)
            if response.status_code == 200:
                print("Website reloaded successfully")
            else:
                print(f"Error: {response.status_code} {response.reason_phrase}")
        except Exception as e:
            print(f"Error: {e}")

async def periodic_ping():
    while True:
        await ping_website()
        await asyncio.sleep(INTERVAL)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_ping())

@app.get("/")
def root():
    return {"message": "Rain Prediction API is running! ISRO PROJECT requirement"}
