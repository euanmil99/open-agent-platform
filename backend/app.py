# Swarm AI Backend - FastAPI Server with HuggingFace Inference API
# Free deployment ready - works with models that have Inference API access

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from huggingface_hub import InferenceClient

app = FastAPI(title="Swarm AI Backend", version="1.0.0")

# CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HuggingFace Inference Client
HF_TOKEN = os.getenv("HF_TOKEN", "hf_GzaY1XTEzCjMVWKxpE0BBbscL1JGgwmNKp")
client = InferenceClient(token=HF_TOKEN)

# Models with confirmed Inference API access (free tier)
MODELS = {
    "web": {
        "primary": "microsoft/Phi-3-mini-4k-instruct",
        "coordinator": "HuggingFaceH4/zephyr-7b-beta",
        "specialist": "google/flan-t5-large"
    },
    "mobile": {
        "primary": "microsoft/Phi-3-mini-4k-instruct",
        "fast": "google/flan-t5-base",
        "ultralight": "google/flan-t5-small"
    }
}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model_tier: str = "web"  # "web" or "mobile"
    model_name: str = "primary"
    max_tokens: int = 512

class ModelInfo(BaseModel):
    version: str
    url: str
    size_mb: Optional[int]
    tier: str

@app.get("/")
async def root():
    return {
        "service": "Swarm AI Backend",
        "status": "operational",
        "version": "1.0.0",
        "models": {
            "web": list(MODELS["web"].keys()),
            "mobile": list(MODELS["mobile"].keys())
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "hf_token_configured": bool(HF_TOKEN),
        "models_available": len(MODELS["web"]) + len(MODELS["mobile"])
    }

@app.get("/api/models/{tier}")
async def get_models(tier: str):
    """Get available models for a tier (web or mobile)"""
    if tier not in MODELS:
        raise HTTPException(status_code=404, detail="Tier not found")
    
    return {
        "tier": tier,
        "models": MODELS[tier]
    }

@app.get("/api/models/mobile/latest")
async def get_latest_mobile_models():
    """Get latest lightweight model versions for mobile"""
    return {
        "phi3_mini": ModelInfo(
            version="v1.0",
            url="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct",
            size_mb=2300,
            tier="mobile"
        ).dict(),
        "flan_t5_base": ModelInfo(
            version="v1.0",
            url="https://huggingface.co/google/flan-t5-base",
            size_mb=950,
            tier="mobile"
        ).dict()
    }

@app.post("/api/chat")
async def chat_completion(request: ChatRequest):
    """Generate chat completion using HuggingFace Inference API"""
    try:
        # Get model based on tier and name
        if request.model_tier not in MODELS:
            raise HTTPException(status_code=400, detail="Invalid model tier")
        
        if request.model_name not in MODELS[request.model_tier]:
            raise HTTPException(status_code=400, detail="Invalid model name")
        
        model_id = MODELS[request.model_tier][request.model_name]
        
        # Convert messages to prompt
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
        
        # Call HuggingFace Inference API
        response = client.text_generation(
            model=model_id,
            prompt=prompt,
            max_new_tokens=request.max_tokens
        )
        
        return {
            "model": model_id,
            "response": response,
            "tier": request.model_tier
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/training/feedback")
async def submit_training_data(data: dict):
    """Accept training feedback from mobile/web clients"""
    # In production, this would store data for model training
    return {
        "status": "accepted",
        "data_id": f"train_{data.get('device_id', 'unknown')}",
        "message": "Training data queued for processing"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
