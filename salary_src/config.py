"""
Configuration module — centralizes all path and environment settings
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Raw data file
RAW_DATA_PATH = RAW_DATA_DIR / "salaries.csv"

# Processed data cache
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "salaries_processed.csv"

# Models directory
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Kaggle ID
KAGGLE_DATASET_ID = "ruchi798/data-science-job-salaries"

# Column names
TARGET_COLUMN = "salary_in_usd"

# Categorical columns with their ordinal mappings
ORDINAL_MAPPINGS = {
    "experience_level": {
        "EN": 0,  # Entry Level
        "MI": 1,  # Mid Level
        "SE": 2,  # Senior Level
        "EX": 3,  # Executive Level
    },
    "employment_type": {
        "FT": 0,  # Full Time
        "PT": 1,  # Part Time
        "CT": 2,  # Contract
        "FL": 3,  # Freelance
    },
    "company_size": {
        "S": 0,   # Small
        "M": 1,   # Medium
        "L": 2,   # Large
    },
}

# Random seed for reproducibility
RANDOM_SEED = 42

# Train/test split ratio
TEST_SIZE = 0.2

# Decision Tree parameters
DECISION_TREE_PARAMS = {
    "max_depth": 10,
    "min_samples_split": 10,
    "random_state": RANDOM_SEED,
}