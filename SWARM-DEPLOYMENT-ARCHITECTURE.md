# Swarm Deployment Architecture

## Overview

This document defines the complete architecture for deploying an agentic AI swarm system with perpetual self-training capabilities. The system is designed to run entirely on free-tier services and open-source models.

## Architecture Components

### 1. Tier-Based Model Strategy

#### Website (Heavy-Duty Models)
- **Primary Model**: Mistral-7B-Instruct-v0.3 (7B parameters)
  - Role: Main reasoning and task coordination
  - Strengths: Strong instruction following, multi-turn conversations
  - Deployment: HuggingFace Inference API

- **Coordinator Model**: Qwen2.5-7B-Instruct (7B parameters)
  - Role: Swarm coordination and decision making
  - Strengths: Excellent reasoning, multilingual support
  - Deployment: HuggingFace Inference API

- **Specialist Model**: Llama-3.1-8B-Instruct (8B parameters)
  - Role: Specialized tasks (coding, analysis)
  - Strengths: Versatile, strong performance
  - Deployment: HuggingFace Inference API

#### Android Mobile (Lightweight Models)
- **Primary Model**: Phi-3-mini-4k-instruct (3.8B parameters)
  - Role: Fast on-device inference
  - Strengths: Excellent quality-to-size ratio
  - Deployment: Quantized GGUF format for mobile

- **Fast Model**: Llama-3.2-1B-Instruct (1B parameters)
  - Role: Ultra-fast responses
  - Strengths: Smallest usable model, very fast
  - Deployment: Quantized for mobile

- **Ultra-Light Model**: TinyLlama-1.1B-Chat (1.1B parameters)
  - Role: Fallback for low-resource scenarios
  - Strengths: Minimal memory footprint
  - Deployment: Quantized for mobile

### 2. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    WEB BACKEND (Heavy-Duty)                 │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  Model Serving Layer                                    │ │
│ │  - Mistral-7B (Primary)                                 │ │
│ │  - Qwen2.5-7B (Coordinator)                             │ │
│ │  - Llama-3.1-8B (Specialist)                            │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  Training Pipeline                                      │ │
│ │  - Model Distillation Engine                            │ │
│ │  - Data Collection & Preprocessing                      │ │
│ │  - Fine-tuning via Google Colab (Free GPU)             │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  Model Registry (HuggingFace Hub)                       │ │
│ │  - Base Models (Teacher)                                │ │
│ │  - Fine-tuned Models (Student)                          │ │
│ │  - Version Control & Metadata                           │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ REST API
                            │ (Model Sync & Training Data)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  ANDROID MOBILE (Lightweight)               │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  Local Model Inference                                  │ │
│ │  - Phi-3-mini-4k (Primary)                              │ │
│ │  - Llama-3.2-1B (Fast)                                  │ │
│ │  - TinyLlama-1.1B (Ultra-light)                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  Feedback Collection                                    │ │
│ │  - User interactions                                    │ │
│ │  - Model predictions & corrections                      │ │
│ │  - Upload to web backend                                │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 3. Perpetual Training Pipeline

#### Training Cycle Flow

1. **Data Collection**
   - User interactions from mobile app
   - User interactions from web interface
   - Model responses and corrections
   - Quality ratings from users

2. **Data Preprocessing**
   - Filter low-quality interactions
   - Format into instruction-response pairs
   - Create balanced training datasets
   - Split: 80% train, 10% validation, 10% test

3. **Model Distillation**
   - Teacher: Heavy-duty models (7B-8B)
   - Student: Lightweight models (1B-3.8B)
   - Method: Knowledge distillation via soft targets
   - Framework: HuggingFace TRL + PEFT (LoRA)

4. **Training Execution**
   - Platform: Google Colab (Free T4 GPU)
   - Technique: LoRA fine-tuning (low-rank adaptation)
   - Batch size: 4-8 (depending on model size)
   - Training time: 2-6 hours per model

5. **Model Evaluation**
   - Automated metrics: Perplexity, BLEU, ROUGE
   - Human evaluation: Sample review
   - A/B testing: Compare with previous version

6. **Deployment**
   - Upload to HuggingFace Hub
   - Update model references in web backend
   - Push lightweight models to mobile via API
   - Mobile downloads updated models

#### Training Schedule

- **Continuous**: Data collection 24/7
- **Weekly**: Lightweight model retraining
- **Bi-weekly**: Heavy-duty model fine-tuning
- **Monthly**: Full model evaluation and architecture review

### 4. Free Tier Deployment

#### HuggingFace Inference API (Free Tier)

**Limits:**
- 30,000 tokens/hour per model
- Automatic scaling within limits
- No rate limit with API key

**Configuration:**
```bash
export HF_TOKEN="hf_GzaY1XTEzCjMVWKxpE0BBbscL1JGgwmNKp"
```

**Python Setup:**
```python
import os
from huggingface_hub import InferenceClient

client = InferenceClient(token=os.getenv("HF_TOKEN"))

# Heavy-duty models for web
models_web = {
    "primary": "mistralai/Mistral-7B-Instruct-v0.3",
    "coordinator": "Qwen/Qwen2.5-7B-Instruct",
    "specialist": "meta-llama/Llama-3.1-8B-Instruct"
}

# Lightweight models for mobile
models_mobile = {
    "primary": "microsoft/Phi-3-mini-4k-instruct",
    "fast": "meta-llama/Llama-3.2-1B-Instruct",
    "ultralight": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
}
```

#### Google Colab Training (Free Tier)

**Resources:**
- T4 GPU (15GB VRAM) - Free
- 12GB RAM
- 100GB disk space
- 12-hour session limit

**Training Script Template:**
```python
# Install dependencies
!pip install transformers trl peft accelerate bitsandbytes

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Load base model (teacher)
teacher_model = "mistralai/Mistral-7B-Instruct-v0.3"
student_model = "microsoft/Phi-3-mini-4k-instruct"

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training continues in training pipeline scripts
```

### 5. Web-Mobile API Communication

#### REST API Endpoints

**Backend (FastAPI/Flask):**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class ModelRequest(BaseModel):
    device_id: str
    current_version: str

class TrainingData(BaseModel):
    device_id: str
    interactions: list
    timestamp: str

@app.get("/api/models/mobile/latest")
async def get_latest_mobile_models(device_id: str):
    """Return latest lightweight model versions"""
    return {
        "phi3_mini": {
            "version": "v1.2",
            "url": "https://huggingface.co/user/phi3-mini-finetuned",
            "size_mb": 2300,
            "quantization": "q4_k_m"
        },
        "llama32_1b": {
            "version": "v1.1",
            "url": "https://huggingface.co/user/llama32-1b-finetuned",
            "size_mb": 650,
            "quantization": "q4_k_m"
        }
    }

@app.post("/api/training/feedback")
async def submit_training_data(data: TrainingData):
    """Accept training data from mobile devices"""
    # Store in database for later processing
    # Return success confirmation
    return {"status": "accepted", "data_id": "xyz123"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "models": "operational"}
```

#### Mobile Client (Android/Kotlin)

```kotlin
class ModelSyncService {
    private val apiBase = "https://your-backend.com/api"
    
    suspend fun checkForUpdates(): ModelUpdateInfo {
        val response = apiClient.get("$apiBase/models/mobile/latest")
        return response.body()
    }
    
    suspend fun downloadModel(url: String): File {
        // Download quantized GGUF model
        // Save to local storage
        return File("path/to/model")
    }
    
    suspend fun uploadTrainingData(interactions: List<Interaction>) {
        val data = TrainingData(
            deviceId = getDeviceId(),
            interactions = interactions,
            timestamp = System.currentTimeMillis()
        )
        apiClient.post("$apiBase/training/feedback", data)
    }
}
```

### 6. Implementation Roadmap

#### Phase 1: Foundation (Week 1-2)
- [ ] Set up HuggingFace account and API token
- [ ] Configure web backend with FastAPI
- [ ] Implement model serving endpoints
- [ ] Test heavy-duty models on web
- [ ] Create basic Android app structure
- [ ] Integrate llama.cpp for mobile inference

#### Phase 2: Model Integration (Week 3-4)
- [ ] Deploy Mistral-7B on web (primary)
- [ ] Deploy Qwen2.5-7B on web (coordinator)
- [ ] Deploy Llama-3.1-8B on web (specialist)
- [ ] Quantize Phi-3-mini for mobile
- [ ] Quantize Llama-3.2-1B for mobile
- [ ] Test model performance and latency

#### Phase 3: Training Pipeline (Week 5-6)
- [ ] Set up data collection infrastructure
- [ ] Create Google Colab training notebooks
- [ ] Implement LoRA fine-tuning pipeline
- [ ] Test knowledge distillation
- [ ] Automate model upload to HuggingFace
- [ ] Implement model versioning system

#### Phase 4: Mobile-Web Sync (Week 7-8)
- [ ] Implement REST API for model sync
- [ ] Create mobile model download service
- [ ] Implement training data upload from mobile
- [ ] Test end-to-end sync workflow
- [ ] Add error handling and retry logic
- [ ] Implement offline mode for mobile

#### Phase 5: Perpetual Learning (Week 9-10)
- [ ] Automate weekly training schedule
- [ ] Implement A/B testing framework
- [ ] Create model evaluation dashboard
- [ ] Set up monitoring and alerts
- [ ] Test full perpetual training cycle
- [ ] Document all processes

#### Phase 6: Optimization & Launch (Week 11-12)
- [ ] Performance optimization
- [ ] Security audit
- [ ] User testing
- [ ] Documentation finalization
- [ ] Beta launch
- [ ] Feedback collection

### 7. Deployment Steps

#### Web Backend Deployment

1. **Environment Setup**
```bash
# Clone repository
git clone https://github.com/euanmil99/open-agent-platform.git
cd open-agent-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install transformers huggingface-hub fastapi uvicorn
```

2. **Configure API Token**
```bash
# Set HuggingFace token
export HF_TOKEN="hf_GzaY1XTEzCjMVWKxpE0BBbscL1JGgwmNKp"

# Add to .env file
echo "HF_TOKEN=hf_GzaY1XTEzCjMVWKxpE0BBbscL1JGgwmNKp" > .env
```

3. **Run Backend**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Mobile App Deployment

1. **Prerequisites**
- Android Studio Arctic Fox or later
- NDK installed for native inference
- Minimum SDK: 26 (Android 8.0)

2. **Build Steps**
```bash
# Clone and open in Android Studio
git clone https://github.com/euanmil99/open-agent-platform.git
cd open-agent-platform/apps/mobile

# Download pre-quantized models
./scripts/download_models.sh

# Build APK
./gradlew assembleRelease
```

3. **Model Quantization**
```python
# Use llama.cpp for quantization
!git clone https://github.com/ggerganov/llama.cpp
!cd llama.cpp && make

# Quantize model
!./llama.cpp/quantize phi-3-mini-4k-instruct.gguf Q4_K_M
```

### 8. Monitoring & Maintenance

#### Key Metrics to Track

1. **Model Performance**
   - Response time (web: <2s, mobile: <500ms)
   - Token throughput
   - Model accuracy metrics
   - User satisfaction ratings

2. **Training Pipeline**
   - Data collection rate
   - Training success rate
   - Model improvement over time
   - Fine-tuning costs (should be $0)

3. **System Health**
   - API uptime
   - HuggingFace API quota usage
   - Mobile app crash rate
   - Sync success rate

#### Troubleshooting

**Issue: HuggingFace API rate limits**
- Solution: Implement request queuing and caching
- Use multiple model endpoints for load balancing

**Issue: Mobile models too large**
- Solution: Increase quantization (Q4 -> Q3)
- Remove less-used models from device

**Issue: Training taking too long**
- Solution: Reduce LoRA rank (r=16 -> r=8)
- Use smaller batch sizes
- Train on subset of data

**Issue: Poor model quality after distillation**
- Solution: Increase training data quality
- Adjust temperature during distillation
- Use ensemble teacher models

### 9. Cost Analysis

**Free Tier Resources:**
- HuggingFace Inference API: $0/month (30K tokens/hour)
- HuggingFace Model Storage: $0/month (unlimited public models)
- Google Colab T4 GPU: $0/month (with session limits)
- Vercel/Railway Backend: $0/month (hobby tier)
- Total Monthly Cost: **$0**

**Scalability Path:**
- When exceeding free limits, upgrade to:
  - HuggingFace Pro: $9/month (100K tokens/hour)
  - Colab Pro: $10/month (longer sessions)
  - Cloud hosting: $20-50/month

### 10. Next Steps

Refer to:
- `FREE-MODELS-ANALYSIS.md` for detailed model specifications
- `IMPROVEMENTS.md` for platform enhancement suggestions
- `AGENTS.md` for agent swarm coordination strategies

**Start Implementation:**
1. Set up HuggingFace token (already done!)
2. Test model inference with provided code
3. Build basic FastAPI backend
4. Create proof-of-concept mobile app
5. Implement first training cycle

---

**Document Version:** 1.0
**Last Updated:** November 25, 2025
**Status:** Ready for Implementation
