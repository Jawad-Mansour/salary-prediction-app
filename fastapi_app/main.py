"""
FastAPI Salary Prediction Application

GET /predict - Predict salary based on job details
GET /health  - Check if API is alive
"""

from fastapi import FastAPI, HTTPException, Query
from typing import Optional
import uvicorn

from .schemas import PredictionResponse, HealthResponse
from .utils import preprocess_input, load_freq_map
from .model_loader import model_loader

# Create FastAPI app
app = FastAPI(
    title="Salary Prediction API",
    description="Predicts data science salaries using a Decision Tree model",
    version="1.0.0"
)

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
    """
    Health check endpoint - confirms API is running and model is loaded.
    
    Returns:
        - status: "ok" if API is running
        - model_loaded: True if model loaded successfully
        - version: API version
    """
    return HealthResponse(
        status="ok",
        model_loaded=model_loader.is_loaded,
        version="1.0.0"
    )


@app.get("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_salary(
    job_title: str = Query(..., description="Job title (e.g., Data Scientist)"),
    experience_level: str = Query(..., description="Experience level: EN, MI, SE, EX"),
    employment_type: str = Query(..., description="Employment type: FT, PT, CT, FL"),
    company_size: str = Query(..., description="Company size: S, M, L"),
    remote_ratio: int = Query(..., description="Remote ratio: 0, 50, 100"),
    work_year: int = Query(..., description="Work year: 2020-2026"),
    employee_residence: str = Query("US", description="Employee country code"),
    company_location: str = Query("US", description="Company country code"),
):
    """
    Predict salary based on job details.
    
    All inputs are validated. Invalid values return clear error messages.
    
    Example:
        GET /predict?job_title=Data+Scientist&experience_level=SE&employment_type=FT&company_size=L&remote_ratio=100&work_year=2026&employee_residence=US&company_location=US
    """
    
    # Step 1: Create input dictionary
    input_data = {
        "job_title": job_title,
        "experience_level": experience_level,
        "employment_type": employment_type,
        "company_size": company_size,
        "remote_ratio": remote_ratio,
        "work_year": work_year,
        "employee_residence": employee_residence,
        "company_location": company_location
    }
    
    # Step 2: Check if model is loaded
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Step 3: Preprocess input
    try:
        X = preprocess_input(input_data, FREQ_MAP)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")
    
    # Step 4: Make prediction
    try:
        predicted_salary = model_loader.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    # Step 5: Return response
    return PredictionResponse(
        job_title=job_title,
        experience_level=experience_level,
        employment_type=employment_type,
        company_size=company_size,
        remote_ratio=remote_ratio,
        work_year=work_year,
        employee_residence=employee_residence,
        company_location=company_location,
        predicted_salary_usd=round(predicted_salary, 2),
        status="success"
    )


# ============================================================================
# ROOT ENDPOINT (simple welcome)
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Welcome message with links to documentation."""
    return {
        "message": "Salary Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict?job_title=Data+Scientist&experience_level=SE&employment_type=FT&company_size=L&remote_ratio=100&work_year=2026&employee_residence=US&company_location=US"
    }


# ============================================================================
# RUN SERVER (for local testing)
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )