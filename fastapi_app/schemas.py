"""
Pydantic schemas for request validation and response formatting
"""

from pydantic import BaseModel, Field, validator
from typing import Optional


class PredictionRequest(BaseModel):
    """Query parameters for the /predict endpoint"""
    
    job_title: str = Field(..., description="Job title (e.g., Data Scientist)")
    experience_level: str = Field(..., description="Experience level: EN, MI, SE, EX")
    employment_type: str = Field(..., description="Employment type: FT, PT, CT, FL")
    company_size: str = Field(..., description="Company size: S, M, L")
    remote_ratio: int = Field(..., description="Remote ratio: 0, 50, 100")
    work_year: int = Field(..., description="Work year: 2020-2026")
    employee_residence: Optional[str] = Field("US", description="Employee country code")
    company_location: Optional[str] = Field("US", description="Company country code")
    
    # Validators - reject invalid values with clear messages
    @validator('experience_level')
    def validate_experience_level(cls, v):
        allowed = ['EN', 'MI', 'SE', 'EX']
        if v not in allowed:
            raise ValueError(f"experience_level must be one of: {allowed}")
        return v
    
    @validator('employment_type')
    def validate_employment_type(cls, v):
        allowed = ['FT', 'PT', 'CT', 'FL']
        if v not in allowed:
            raise ValueError(f"employment_type must be one of: {allowed}")
        return v
    
    @validator('company_size')
    def validate_company_size(cls, v):
        allowed = ['S', 'M', 'L']
        if v not in allowed:
            raise ValueError(f"company_size must be one of: {allowed}")
        return v
    
    @validator('remote_ratio')
    def validate_remote_ratio(cls, v):
        allowed = [0, 50, 100]
        if v not in allowed:
            raise ValueError(f"remote_ratio must be one of: {allowed}")
        return v
    
    @validator('work_year')
    def validate_work_year(cls, v):
        if v < 2020 or v > 2026:
            raise ValueError(f"work_year must be between 2020 and 2026")
        return v


class PredictionResponse(BaseModel):
    """Response format for the /predict endpoint"""
    
    job_title: str
    experience_level: str
    employment_type: str
    company_size: str
    remote_ratio: int
    work_year: int
    employee_residence: str
    company_location: str
    predicted_salary_usd: float
    status: str = "success"


class HealthResponse(BaseModel):
    """Response format for the /health endpoint"""
    
    status: str
    model_loaded: bool
    version: str