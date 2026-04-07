from fastapi import FastAPI
from app.schemas import PredictionRequest, PredictionResponse
from app.inference import predict_text

app = FastAPI(title="DistilBERT Binary Classifier API")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return a simple health status for the API."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Run DistilBERT inference on input text and return prediction details."""
    result = predict_text(request.text)
    return PredictionResponse(**result)
