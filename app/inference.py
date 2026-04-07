import sys 
from pathlib import Path 
sys.path.append(str(Path().resolve().parents[0]))
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import MODELS_DIR


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR / "distilbert_binary_classifier")
model = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR / "distilbert_binary_classifier")
model.to(DEVICE)
model.eval()

ID2LABEL = {
    0: "negative",
    1: "positive",
}


def predict_text(text: str) -> dict[str, Any]:
    """Convert raw user text into binary sentiment prediction with confidence scores"""
    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )

    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze(0)

    predicted_class_id = int(torch.argmax(probs).item())
    confidence = float(probs[predicted_class_id].item())

    class_probabilities = {
        ID2LABEL[i]: float(probs[i].item()) for i in range(len(probs))
    }

    return {
        "predicted_label": ID2LABEL[predicted_class_id],
        "predicted_class_id": predicted_class_id,
        "confidence": confidence,
        "class_probabilities": class_probabilities,
    }
