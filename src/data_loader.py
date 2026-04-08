"""
Data Loader Module

Responsibility: Load the dataset from local cache.
This module NEVER downloads data — that's the download script's job.

Usage in notebooks:
    from src.data_loader import load_salaries_dataset
    df = load_salaries_dataset()
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_salaries_dataset(cache_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the Data Science Salaries dataset from local cache.
    
    Args:
        cache_path: Path to the cached CSV. Defaults to data/raw/salaries_raw.csv
    
    Returns:
        DataFrame with the raw salary data
        
    Raises:
        FileNotFoundError: If the cached file doesn't exist.
                          Run scripts/download_dataset.py first.
    
    Example:
        >>> df = load_salaries_dataset()
        >>> print(df.shape)
        (3755, 11)
    """
    
    # Set default cache path if not provided
    if cache_path is None:
        project_root = Path(__file__).parent.parent
        cache_path = project_root / "data" / "raw" / "salaries_raw.csv"
    
    # Check if file exists
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {cache_path}\n"
            f"Please run: python scripts/download_dataset.py"
        )
    
    # Load the CSV
    df = pd.read_csv(cache_path)
    print(f"✅ Loaded {len(df)} rows from {cache_path}")
    
    return df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """Return basic info about the dataset for exploration."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }


# Quick test
if __name__ == "__main__":
    print("=" * 50)
    print("Testing data loader...")
    print("=" * 50)
    
    try:
        df = load_salaries_dataset()
        print(f"\n✅ Success! Loaded {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")