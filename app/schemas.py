"""Request / response schemas."""
from pydantic import BaseModel, Field


class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, le=10, examples=[5.1])
    sepal_width: float = Field(..., gt=0, le=10, examples=[3.5])
    petal_length: float = Field(..., gt=0, le=10, examples=[1.4])
    petal_width: float = Field(..., gt=0, le=10, examples=[0.2])


class PredictionResponse(BaseModel):
    prediction: int
    flower_name: str
    confidence: float
