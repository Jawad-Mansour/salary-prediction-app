"""
FastAPI Salary Prediction Application

GET /predict - Predict salary based on job details
GET /health  - Check if API is alive
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import uvicorn
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi_app.schemas import PredictionRequest, PredictionResponse, HealthResponse
from fastapi_app.utils import preprocess_input, load_freq_map
from fastapi_app.model_loader import model_loader

# Create FastAPI app
app = FastAPI(
    title="Salary Prediction API",
    description="Predicts data science salaries using a Decision Tree model",
    version="1.0.0"
)


# ============================================================================
# EXCEPTION HANDLER - FIXED VERSION
# ============================================================================

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """
    Handle Pydantic validation errors and return proper 422 response.
    
    This converts the ValidationError into a JSON-serializable format.
    """
    errors = []
    for error in exc.errors():
        errors.append({
            "loc": error.get("loc", []),
            "msg": error.get("msg", "Validation error"),
            "type": error.get("type", "value_error"),
            "input": error.get("input", None)
        })
    
    return JSONResponse(
        status_code=422,
        content={"detail": errors}
    )


# ============================================================================
# LOAD FREQUENCY MAP
# ============================================================================

# Load frequency map at startup
try:
    FREQ_MAP = load_freq_map("models/encoding_maps.json")
    print(f"✅ Loaded frequency map with {len(FREQ_MAP)} job titles")
except Exception as e:
    print(f"⚠️ Could not load frequency map: {e}")
    FREQ_MAP = {}


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint - confirms API is running and model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=model_loader.is_loaded,
        version="1.0.0"
    )


@app.get("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_salary(
    request: PredictionRequest = Depends()
):
    """
    Predict salary based on job details.
    
    All inputs are validated by Pydantic validators. 
    Invalid values return 422 error with clear message.
    
    Example:
        GET /predict?job_title=Data+Scientist&experience_level=SE&employment_type=FT&company_size=L&remote_ratio=100&work_year=2026&employee_residence=US&company_location=US
    """
    
    # Create input dictionary from validated request
    input_data = {
        "job_title": request.job_title,
        "experience_level": request.experience_level,
        "employment_type": request.employment_type,
        "company_size": request.company_size,
        "remote_ratio": request.remote_ratio,
        "work_year": request.work_year,
        "employee_residence": request.employee_residence,
        "company_location": request.company_location
    }
    
    # Check if model is loaded
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Preprocess input
    try:
        X = preprocess_input(input_data, FREQ_MAP)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")
    
    # Make prediction
    try:
        predicted_salary = model_loader.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    # Return response
    return PredictionResponse(
        job_title=request.job_title,
        experience_level=request.experience_level,
        employment_type=request.employment_type,
        company_size=request.company_size,
        remote_ratio=request.remote_ratio,
        work_year=request.work_year,
        employee_residence=request.employee_residence,
        company_location=request.company_location,
        predicted_salary_usd=round(predicted_salary, 2),
        status="success"
    )


@app.get("/", tags=["Root"])
async def root():
    """Welcome message with links to documentation."""
    return {
        "message": "Salary Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict?job_title=Data+Scientist&experience_level=SE&employment_type=FT&company_size=L&remote_ratio=100&work_year=2026&employee_residence=US&company_location=US"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )