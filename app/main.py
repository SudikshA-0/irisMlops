"""FastAPI application.

Run:  uvicorn app.main:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.model_service import model_service
from app.schemas import IrisInput, PredictionResponse


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_service.load()
    yield


app = FastAPI(
    title="MLOps Iris Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def home():
    return {"message": "MLOps API is running"}


@app.get("/health")
def health():
    return {"status": "ok" if model_service.is_ready else "model_not_loaded"}


@app.get("/model-info")
def model_info():
    if not model_service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_service.metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: IrisInput):
    if not model_service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_service.predict(payload)
