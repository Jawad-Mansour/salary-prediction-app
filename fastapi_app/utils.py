"""
Utility functions for preprocessing - self-contained for FastAPI
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path


# ============================================================================
# ENCODING MAPS (copied from preprocess.py - must match training)
# ============================================================================

ORDINAL_MAPS = {
    "experience_level": {"EN": 0, "MI": 1, "SE": 2, "EX": 3},
    "employment_type": {"FT": 0, "PT": 1, "CT": 2, "FL": 3},
    "company_size": {"S": 0, "M": 1, "L": 2}
}

COUNTRY_REGION_MAP = {
    "US": "NA", "CA": "NA", "GB": "WE", "DE": "WE", "FR": "WE",
    "ES": "WE", "IT": "WE", "NL": "WE", "CH": "WE", "BE": "WE",
    "AT": "WE", "IE": "WE", "PT": "WE", "SE": "WE", "DK": "WE",
    "FI": "WE", "NO": "WE", "AU": "AP", "NZ": "AP", "SG": "AP",
    "JP": "AP", "KR": "AP", "CN": "AP", "IN": "AP", "BR": "LA",
    "MX": "LA", "ZA": "AF", "AE": "ME", "remote": "REMOTE"
}

REGION_ENCODING = {
    "NA": 4, "WE": 3, "AP": 2, "ME": 2, "LA": 1, "AF": 0, "REMOTE": 2
}

# The exact feature order the model expects (19 features)
FEATURE_ORDER = [
    'region_encoded', 'region_x_exp', 'job_title_encoded', 'title_x_region',
    'dev_index', 'exp_x_size', 'work_year', 'remote_x_exp', 'work_year_squared',
    'remote_ratio', 'company_size_encoded', 'same_country',
    'experience_level_encoded', 'employment_type_encoded',
    'size_squared', 'size_x_region', 'size_x_title', 'is_large', 'is_small'
]

# Development index for countries
DEV_INDEX = {
    'US': 100, 'CA': 95, 'GB': 90, 'DE': 90, 'FR': 85, 'ES': 85,
    'IT': 85, 'NL': 95, 'SE': 95, 'NO': 95, 'AU': 95, 'SG': 100,
    'JP': 85, 'KR': 85, 'CN': 70, 'IN': 45, 'BR': 60, 'MX': 60,
    'ZA': 55, 'AE': 85, 'CH': 100
}


def preprocess_input(data: dict, freq_map: dict) -> pd.DataFrame:
    """
    Preprocess a single input dictionary into features.
    
    Args:
        data: Dictionary with raw input values
        freq_map: Job title frequency map from training
    
    Returns:
        DataFrame with 19 features in correct order
    """
    
    # Step 1: Create base DataFrame
    df = pd.DataFrame([data])
    
    # Step 2: Ordinal encoding
    df['experience_level_encoded'] = df['experience_level'].map(ORDINAL_MAPS['experience_level'])
    df['employment_type_encoded'] = df['employment_type'].map(ORDINAL_MAPS['employment_type'])
    df['company_size_encoded'] = df['company_size'].map(ORDINAL_MAPS['company_size'])
    
    # Step 3: Region encoding
    df['employee_region'] = df['employee_residence'].map(COUNTRY_REGION_MAP).fillna("REMOTE")
    df['region_encoded'] = df['employee_region'].map(REGION_ENCODING).fillna(2)
    
    # Step 4: Job title frequency encoding
    job_title = data['job_title']
    df['job_title_encoded'] = freq_map.get(job_title, 0.0)
    
    # Step 5: Same country flag
    df['same_country'] = (df['employee_residence'] == df['company_location']).astype(int)
    
    # Step 6: Development index
    df['dev_index'] = df['employee_residence'].map(DEV_INDEX).fillna(50)
    
    # Step 7: Engineered features
    df['exp_x_size'] = df['experience_level_encoded'] * df['company_size_encoded']
    df['remote_x_exp'] = (df['remote_ratio'] / 100) * df['experience_level_encoded']
    df['work_year_squared'] = df['work_year'] ** 2
    df['region_x_exp'] = df['region_encoded'] * df['experience_level_encoded']
    df['title_x_region'] = df['job_title_encoded'] * df['region_encoded']
    
    # Step 8: Company size features
    df['size_squared'] = df['company_size_encoded'] ** 2
    df['size_x_region'] = df['company_size_encoded'] * df['region_encoded']
    df['size_x_title'] = df['company_size_encoded'] * df['job_title_encoded']
    df['is_large'] = (df['company_size_encoded'] == 2).astype(int)
    df['is_small'] = (df['company_size_encoded'] == 0).astype(int)
    
    # Step 9: Select and order features
    X = df[FEATURE_ORDER]
    
    return X


def load_freq_map(filepath: str = "models/encoding_maps.json") -> dict:
    """Load job title frequency map from training artifacts."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Encoding maps not found at {filepath}")
    
    with open(path, 'r') as f:
        maps = json.load(f)
    
    return maps.get('job_title_freq_map', {})