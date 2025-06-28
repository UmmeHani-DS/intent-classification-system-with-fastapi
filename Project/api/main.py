from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib
import logging
import spacy


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models (shared across the application)
class ModelState:
    model = None
    vectorizer = None
    label_encoder = None
    nlp = None
    model_info = {}
    keywords = ["email", "schedule", "find", "what", "how"]

# Create global state instance
model_state = ModelState()

async def load_models():
    # Load ML models and NLP components during startup
    try:
        logger.info("Loading ML models...")
        
        # Load pre-trained models
        model_state.model = joblib.load('ml/intent_classifier.joblib')
        model_state.vectorizer = joblib.load('ml/tfidf_vectorizer.joblib')
        model_state.label_encoder = joblib.load('ml/label_encoder.joblib')
        
        # Load spaCy NLP model
        model_state.nlp = spacy.load("en_core_web_sm")
        
        # Store model information for API responses
        model_state.model_info = {
            "model_type": "Ensemble (Naive Bayes + Logistic Regression + Random Forest)",
            "classes": model_state.label_encoder.classes_.tolist(),
            "feature_count": model_state.vectorizer.max_features,
            "ngram_range": model_state.vectorizer.ngram_range,
            "status": "loaded"
        }
        
        logger.info("Models loaded successfully")
        logger.info(f"Available intent classes: {len(model_state.model_info['classes'])}")
        
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        raise RuntimeError(f"Required model files are missing: {e}")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise RuntimeError(f"Failed to load models: {e}")

async def cleanup_models():
    # Cleanup resources during shutdown
    logger.info("Cleaning up resources...")
    
    # Clear model references
    model_state.model = None
    model_state.vectorizer = None
    model_state.label_encoder = None
    model_state.nlp = None
    model_state.model_info = {}
    
    logger.info("Resources cleaned up successfully")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manage application lifespan - startup and shutdown events
    # Startup
    logger.info("Starting up Intent Classification API...")
    try:
        await load_models()
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down Intent Classification API...")
    await cleanup_models()
    logger.info("Application shutdown completed")

# Initialize FastAPI app with lifespan management
app = FastAPI(
    title="Intent Classification API",
    description="API for classifying user intents using machine learning",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI (auto-generated)
    redoc_url="/redoc",  # ReDoc documentation
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers from endpoints.py
from api.endpoints import router as api_router
app.include_router(api_router, prefix="/api", tags=["classification"])

@app.get("/")
async def root():
    # Root endpoint with API information
    return {
        "message": "Intent Classification API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/api/health",
        "endpoints": {
            "classify_single": "/api/classify",
            "classify_batch": "/api/classify/batch",
            "model_info": "/api/model/info",
            "health": "/api/health"
        }
    }