"""
Fake Product Review Detector - FastAPI Backend
==============================================
REST API for serving ML model predictions.
Provides endpoints for single review analysis and batch processing.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import joblib
import os
import pandas as pd
import numpy as np
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ===============================
# CONFIGURATION
# ===============================
app = FastAPI(
    title="Fake Review Detector API",
    description="ML-powered API for detecting fake product reviews",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ===============================
# MODELS
# ===============================
class ReviewRequest(BaseModel):
    text: str
    model_type: Optional[str] = "logistic_regression"
    return_details: Optional[bool] = True


class BatchReviewRequest(BaseModel):
    reviews: List[str]
    model_type: Optional[str] = "logistic_regression"


class ReviewResponse(BaseModel):
    prediction: str
    confidence: float
    fake_probability: float
    genuine_probability: float
    is_fake: bool
    red_flags: Optional[List[str]] = None
    severity: Optional[List[str]] = None
    features: Optional[dict] = None


class BatchReviewResponse(BaseModel):
    predictions: List[ReviewResponse]
    summary: dict


# ===============================
# HELPER FUNCTIONS
# ===============================
def load_models():
    """Load ML models"""
    try:
        vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
        lr_model = joblib.load(os.path.join(MODEL_DIR, "lr_model.pkl"))
        nb_model = joblib.load(os.path.join(MODEL_DIR, "nb_model.pkl"))
        
        return {
            'vectorizer': vectorizer,
            'lr_model': lr_model,
            'nb_model': nb_model,
            'loaded': True
        }
    except Exception as e:
        return {'loaded': False, 'error': str(e)}


def clean_text(text):
    """Clean text"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = ' '.join(text.split())
    return text


def extract_features(text):
    """Extract features from text"""
    features = {}
    
    features['word_count'] = len(text.split())
    features['char_count'] = len(text)
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    
    positive_words = ['love', 'great', 'excellent', 'amazing', 'perfect', 'best']
    negative_words = ['hate', 'terrible', 'worst', 'awful', 'horrible']
    
    features['positive_count'] = sum(1 for w in text.lower().split() if w in positive_words)
    features['negative_count'] = sum(1 for w in text.lower().split() if w in negative_words)
    
    promo_patterns = [r'buy now', r'must buy', r'limited offer', r'best ever', r'guaranteed']
    features['promo_count'] = sum(len(re.findall(p, text.lower())) for p in promo_patterns)
    features['is_short'] = 1 if len(text.split()) < 10 else 0
    
    return features


def detect_red_flags(text):
    """Detect red flags"""
    flags = []
    severity = []
    text_lower = text.lower()
    
    if text.count("!") >= 3:
        flags.append("Excessive exclamation marks")
        severity.append("medium")
    
    if text.isupper() and len(text) > 20:
        flags.append("All caps text")
        severity.append("high")
    
    promo_words = [
        ("buy now", "Urgent buying language"),
        ("must buy", "Pressure tactics"),
        ("limited offer", "Artificial urgency"),
        ("best ever", "Exaggerated praise"),
    ]
    
    for pattern, desc in promo_words:
        if pattern in text_lower:
            flags.append(f"Promotional: {desc}")
            severity.append("medium")
    
    if len(text.split()) < 6:
        flags.append("Unrealistically short")
        severity.append("high")
    
    if '★' in text or '☆' in text:
        flags.append("Star symbols detected")
        severity.append("low")
    
    return flags, severity


def predict_review(text, model_type="logistic_regression", return_details=True):
    """Predict if review is fake"""
    models = load_models()
    
    if not models['loaded']:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    clean = clean_text(text)
    vec = models['vectorizer'].transform([clean])
    
    if model_type == "naive_bayes":
        model = models['nb_model']
    else:
        model = models['lr_model']
    
    probs = model.predict_proba(vec)[0]
    classes = model.classes_
    
    fake_idx = list(classes).index(1)
    genuine_idx = list(classes).index(0)
    
    fake_prob = probs[fake_idx]
    genuine_prob = probs[genuine_idx]
    
    prediction = 1 if fake_prob > genuine_prob else 0
    confidence = max(fake_prob, genuine_prob)
    
    result = {
        "prediction": "FAKE" if prediction == 1 else "GENUINE",
        "confidence": float(confidence),
        "fake_probability": float(fake_prob),
        "genuine_probability": float(genuine_prob),
        "is_fake": bool(prediction == 1)
    }
    
    if return_details:
        flags, sev = detect_red_flags(text)
        result["red_flags"] = flags
        result["severity"] = sev
        result["features"] = extract_features(text)
    
    return ReviewResponse(**result)


# ===============================
# ENDPOINTS
# ===============================
@app.get("/")
def root():
    """Root endpoint"""
    return {
        "name": "Fake Review Detector API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "models": "/models"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    models = load_models()
    return {
        "status": "healthy" if models['loaded'] else "degraded",
        "models_loaded": models['loaded']
    }


@app.get("/models")
def get_models():
    """Get available models"""
    return {
        "models": [
            {"name": "logistic_regression", "description": "Logistic Regression classifier"},
            {"name": "naive_bayes", "description": "Multinomial Naive Bayes classifier"}
        ]
    }


@app.post("/predict", response_model=ReviewResponse)
def predict(request: ReviewRequest):
    """Predict if a single review is fake"""
    try:
        return predict_review(
            request.text,
            request.model_type,
            request.return_details
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchReviewResponse)
def predict_batch(request: BatchReviewRequest):
    """Predict multiple reviews"""
    try:
        predictions = []
        
        for text in request.reviews:
            pred = predict_review(
                text,
                request.model_type,
                return_details=True
            )
            predictions.append(pred)
        
        # Summary
        fake_count = sum(1 for p in predictions if p.is_fake)
        genuine_count = len(predictions) - fake_count
        avg_confidence = sum(p.confidence for p in predictions) / len(predictions)
        
        summary = {
            "total_reviews": len(predictions),
            "fake_count": fake_count,
            "genuine_count": genuine_count,
            "average_confidence": float(avg_confidence)
        }
        
        return BatchReviewResponse(
            predictions=predictions,
            summary=summary
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/csv", response_model=BatchReviewResponse)
async def predict_csv(file: UploadFile = File(...)):
    """Predict reviews from CSV file"""
    try:
        # Read CSV
        df = pd.read_csv(file.file)
        
        # Find review column
        review_col = None
        for col in ['review_body', 'review', 'text', 'review_text']:
            if col in df.columns:
                review_col = col
                break
        
        if review_col is None:
            raise HTTPException(
                status_code=400,
                detail="Could not find review column. Expected: review_body, review, text, or review_text"
            )
        
        # Get reviews
        reviews = df[review_col].fillna('').astype(str).tolist()
        
        # Predict
        request = BatchReviewRequest(reviews=reviews)
        return await predict_batch(request)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
