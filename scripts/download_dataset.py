#!/usr/bin/env python3
"""
Download Dataset Script

Run this ONCE to download the Kaggle dataset and save it locally.
This script should NOT be imported or called automatically.

Usage:
    python scripts/download_dataset.py

What it does:
    1. Downloads from KaggleHub
    2. Saves to data/raw/salaries_raw.csv
    3. Prints confirmation
"""

import kagglehub
import pandas as pd
from pathlib import Path


def download_and_cache():
    """Download dataset and save to local cache."""
    
    # Get project root (scripts/ is one level down)
    project_root = Path(__file__).parent.parent
    cache_path = project_root / "data" / "raw" / "salaries_raw.csv"
    
    # Check if already downloaded
    if cache_path.exists():
        print(f"⚠️  File already exists: {cache_path}")
        response = input("Download again? (y/N): ")
        if response.lower() != 'y':
            print("❌ Download cancelled")
            return
    
    print("🌐 Downloading from KaggleHub...")
    
    try:
        # Download the dataset
        path = kagglehub.dataset_download("arnabchaki/data-science-salaries-2023")
        
        # Find the CSV file
        csv_file = Path(path) / "ds_salaries.csv"
        if not csv_file.exists():
            csv_file = Path(path) / "salaries.csv"
        
        if not csv_file.exists():
            raise FileNotFoundError(f"No CSV found in {path}")
        
        # Read and save
        df = pd.read_csv(csv_file)
        
        # Create directory if needed
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to cache
        df.to_csv(cache_path, index=False)
        
        print(f"✅ Downloaded and saved: {cache_path}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise


if __name__ == "__main__":
    download_and_cache()