from pydantic import BaseModel, Field
from typing import Dict


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to classify")


class PredictionResponse(BaseModel):
    predicted_label: str
    predicted_class_id: int
    confidence: float
    class_probabilities: Dict[str, float]
