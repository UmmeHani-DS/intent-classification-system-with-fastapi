from pydantic import BaseModel, Field, validator
from typing import List, Optional #, Dict, Any

class QueryRequest(BaseModel):
    # Request model for single text classification
    text: str = Field(..., min_length=1, max_length=1000, description="Text to classify")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()

class BatchQueryRequest(BaseModel):
    # Request model for batch text classification
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to classify")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        
        # Validate each text in the list
        validated_texts = []
        for text in v:
            if not text or not text.strip():
                raise ValueError('Each text must be non-empty')
            validated_texts.append(text.strip())
        
        return validated_texts

class ClassificationResponse(BaseModel):
    # Response model for single text classification
    text: str = Field(..., description="Original input text")
    intent: str = Field(..., description="Predicted intent class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Send an email to John about the meeting",
                "intent": "send_email",
                "confidence": 0.95
            }
        }

class BatchClassificationResult(BaseModel):
    # Individual result in batch classification response
    text: str = Field(..., description="Original input text")
    intent: str = Field(..., description="Predicted intent class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")

class BatchClassificationResponse(BaseModel):
    # Response model for batch text classification
    results: List[BatchClassificationResult] = Field(..., description="List of classification results")
    
    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "text": "Send an email to John",
                        "intent": "send_email",
                        "confidence": 0.95
                    },
                    {
                        "text": "Schedule a meeting tomorrow",
                        "intent": "schedule_meeting",
                        "confidence": 0.87
                    }
                ]
            }
        }

class ModelInfo(BaseModel):
    # Response model for model information
    model_type: str = Field(..., description="Type of ML model used")
    classes: List[str] = Field(..., description="Available intent classes")
    feature_count: Optional[int] = Field(None, description="Number of features in the model")
    ngram_range: Optional[tuple] = Field(None, description="N-gram range used for text processing")
    status: str = Field(..., description="Model loading status")
    
    class Config:
        schema_extra = {
            "example": {
                "model_type": "Ensemble (Naive Bayes + Logistic Regression + SVM)",
                "classes": ["send_email", "schedule_meeting", "find_information", "general_query"],
                "feature_count": 5000,
                "ngram_range": [1, 2],
                "status": "loaded"
            }
        }

class HealthResponse(BaseModel):
    # Response model for health check
    status: str = Field(..., description="Overall API status")
    models_loaded: bool = Field(..., description="Whether ML models are loaded")
    nlp_loaded: bool = Field(..., description="Whether NLP model is loaded")
    vectorizer_loaded: bool = Field(..., description="Whether vectorizer is loaded")
    label_encoder_loaded: bool = Field(..., description="Whether label encoder is loaded")
    api_version: str = Field(..., description="API version")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "models_loaded": True,
                "nlp_loaded": True,
                "vectorizer_loaded": True,
                "label_encoder_loaded": True,
                "api_version": "1.0.0"
            }
        }

class ErrorResponse(BaseModel):
    # Response model for error cases
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code for programmatic handling")
    
    class Config:
        schema_extra = {
            "example": {
                "detail": "Classification failed: Model not loaded",
                "error_code": "MODEL_NOT_LOADED"
            }
        }