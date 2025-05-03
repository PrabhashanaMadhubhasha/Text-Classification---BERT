from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# Load model and tokenizer
MODEL_PATH = "saved_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: TextRequest):
    try:
        # Tokenize input
        encoding = tokenizer(
            request.text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**encoding)
            _, preds = torch.max(outputs.logits, dim=1)
        
        return {"prediction": int(preds[0])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}