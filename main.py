from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, Any

app = FastAPI()

# Load model and tokenizer
MODEL_PATH = "saved_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

class TextRequest(BaseModel):
    text: str

def preprocess_text(text: str) -> Dict[str, torch.Tensor]:
    """Tokenize and preprocess the input text"""
    return tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

def predict_class(encoding: Dict[str, torch.Tensor]) -> int:
    """Run model prediction on preprocessed text"""
    with torch.no_grad():
        outputs = model(**encoding)
        _, preds = torch.max(outputs.logits, dim=1)
    return int(preds[0])

@app.post("/predict")
def predict(request: TextRequest):
    try:
        # Step 1: Preprocess the input text
        encoding = preprocess_text(request.text)
        
        # Step 2: Get prediction
        prediction = predict_class(encoding)
        
        # Step 3: Return response
        return {"prediction": prediction}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Example of adding another endpoint in the same style
class BatchPredictionRequest(BaseModel):
    texts: list[str]

def process_batch(texts: list[str]) -> list[int]:
    """Process multiple texts for batch prediction"""
    encodings = tokenizer(
        texts,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=1).tolist()
    
    return predictions

@app.post("/batch-predict")
def batch_predict(request: BatchPredictionRequest):
    try:
        if len(request.texts) > 100:
            return {"message": "Maximum batch size is 100 texts"}
        
        predictions = process_batch(request.texts)
        return {"predictions": predictions}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
