import kagglehub
import os
import pandas as pd
import shutil

# Download latest version
path = kagglehub.dataset_download("ruchi798/data-science-job-salaries")

print("Downloaded to:", path)

# Find the CSV file in the downloaded folder
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

if csv_files:
    csv_file = csv_files[0]
    source_path = os.path.join(path, csv_file)
    
    # Create data/raw directory if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)
    
    # Destination path
    destination_path = "data/raw/salaries.csv"
    
    # Copy or move the file
    shutil.copy(source_path, destination_path)
    
    print(f"Saved to: {destination_path}")
    
    # Optional: Show first few rows
    df = pd.read_csv(destination_path)
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
else:
    print("No CSV file found in downloaded dataset")
    
    
    """
COMMENT BLOCK - WHAT EACH PART DOES:

1. kagglehub.dataset_download()
   - Downloads dataset from Kaggle to local cache
   - Returns path where files are stored

2. os.listdir(path)
   - Lists all files in downloaded folder
   - Used to find CSV files

3. csv_files = [f for f in ... if f.endswith('.csv')]
   - Filters to only CSV files
   - Handles unknown filename patterns

4. os.path.join(path, csv_file)
   - Creates full file path (handles OS differences)
   - Windows uses '\', Mac/Linux uses '/'

5. os.makedirs("data/raw", exist_ok=True)
   - Creates data/raw folder structure
   - exist_ok=True prevents error if folder exists

6. shutil.copy(source_path, destination_path)
   - WHAT IS shutil: Shell utility for high-level file operations
   - USE: Copies file from source to destination
   - WHY HERE: Keeps original in cache, creates working copy in project
   - ALTERNATIVE: Would need manual read/write without shutil

7. pd.read_csv(destination_path)
   - Loads CSV data into pandas DataFrame
   - Enables data analysis operations

8. df.shape
   - Returns (rows, columns) tuple
   - Quick dataset size check

9. df.head()
   - Shows first 5 rows
   - Verifies data loaded correctly
"""