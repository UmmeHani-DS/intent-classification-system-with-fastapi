from fastapi import APIRouter, HTTPException
from .models import QueryRequest, BatchQueryRequest, ClassificationResponse, BatchClassificationResponse
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Import the global model state from main
from .main import model_state

def clean_text(text: str) -> str:
    """Clean text by removing punctuation, numbers, and normalizing whitespace"""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\d+', ' ', text)      # Remove numbers
    return ' '.join(text.split())         # Normalize whitespace

def process_text(text: str) -> str:
    """Process text using spaCy for lemmatization and stop word removal"""
    if model_state.nlp is None:
        raise HTTPException(status_code=503, detail="NLP model not loaded")
    
    doc = model_state.nlp(text)
    tokens = [token.lemma_ for token in doc 
              if not token.is_stop and not token.is_punct and len(token.text) > 1]
    return ' '.join(tokens)

def preprocess_text(text: str) -> str:
    """Complete text preprocessing pipeline"""
    cleaned = clean_text(text)
    processed = process_text(cleaned)
    return processed

def predict_intent(text: str):
    """Predict intent for given text using the loaded model"""
    if (model_state.model is None or 
        model_state.vectorizer is None or 
        model_state.label_encoder is None):
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Preprocess the input text
        cleaned_text = preprocess_text(text)
        
        # Generate TF-IDF features
        tfidf_features = model_state.vectorizer.transform([cleaned_text])

        # Generate keyword features
        keyword_vector = np.zeros((1, len(model_state.keywords)))
        text_lower = text.lower()
        for i, keyword in enumerate(model_state.keywords):
            if keyword in text_lower:
                keyword_vector[0, i] = 1

        # Combine TF-IDF and keyword features
        combined_features = hstack([tfidf_features, csr_matrix(keyword_vector)])

        # Make prediction
        prediction = model_state.model.predict(combined_features)[0]

        # Calculate confidence score
        if hasattr(model_state.model, 'predict_proba'):
            probabilities = model_state.model.predict_proba(combined_features)[0]
            confidence = float(np.max(probabilities))
        else:
            confidence = 0.95  # Fallback confidence score

        # Convert prediction back to intent label
        intent = model_state.label_encoder.inverse_transform([prediction])[0]
        return intent, confidence
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/classify", response_model=ClassificationResponse)
async def classify_single_query(request: QueryRequest):
    """Classify a single text query"""
    try:
        intent, confidence = predict_intent(request.text)
        return ClassificationResponse(
            text=request.text,
            intent=intent,
            confidence=confidence
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in single classification: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@router.post("/classify/batch", response_model=BatchClassificationResponse)
async def classify_batch_queries(request: BatchQueryRequest):
    """Classify multiple text queries in batch"""
    try:
        results = []
        for text in request.texts:
            intent, confidence = predict_intent(text)
            results.append({
                "text": text,
                "intent": intent,
                "confidence": confidence
            })
        return BatchClassificationResponse(results=results)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch classification: {e}")
        raise HTTPException(status_code=500, detail=f"Batch classification failed: {str(e)}")

@router.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if not model_state.model_info:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_state.model_info

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": model_state.model is not None,
        "nlp_loaded": model_state.nlp is not None,
        "vectorizer_loaded": model_state.vectorizer is not None,
        "label_encoder_loaded": model_state.label_encoder is not None,
        "api_version": "1.0.0"
    }