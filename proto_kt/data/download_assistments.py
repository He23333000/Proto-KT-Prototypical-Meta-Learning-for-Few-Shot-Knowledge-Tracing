"""
Download and extract ASSISTments datasets for Knowledge Tracing experiments.

ASSISTments is a popular educational platform dataset containing student interactions
with math problems. Each interaction includes:
    - user_id: Student identifier
    - problem_id: Question/skill identifier
    - correct: Whether student answered correctly (binary: 0 or 1)
    - order_id/timestamp: Temporal ordering of interactions

This module handles downloading and basic loading of the raw dataset files.
"""
import os
import urllib.request
import zipfile
import pandas as pd
from pathlib import Path


def download_assistments2009(data_dir='data/raw'):
    """
    Download ASSISTments 2009-2010 skill builder dataset.
    
    This is one of the most commonly used datasets for Knowledge Tracing research.
    It contains ~300K interactions from ~4K students on ~100 skills.
    
    Args:
        data_dir (str): Directory to save the raw CSV file
        
    Returns:
        str: Path to downloaded file, or None if download failed
        
    Dataset URL: https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data
    """
    # Create data directory if it doesn't exist
    # parents=True creates parent directories, exist_ok=True doesn't error if exists
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # URL to ASSISTments 2009-2010 skill builder data CSV
    url = "https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill_builder_data.csv"
    output_path = data_dir / "assistments2009_raw.csv"
    
    # Check if already downloaded to avoid re-downloading
    if output_path.exists():
        print(f"ASSISTments 2009 already downloaded at {output_path}")
        return str(output_path)
    
    # Attempt download
    print(f"Downloading ASSISTments 2009 from {url}...")
    try:
        # urllib.request.urlretrieve downloads file from URL and saves to disk
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded to {output_path}")
        return str(output_path)
    except Exception as e:
        # If download fails, provide manual download instructions
        print(f"Error downloading: {e}")
        print("Please manually download from: https://sites.google.com/site/assistmentsdata/")
        print(f"And place it at: {output_path}")
        return None


def download_assistments2015(data_dir='data/raw'):
    """
    Download ASSISTments 2015 dataset.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Note: This is a placeholder - actual download URL may vary
    print("ASSISTments 2015 dataset:")
    print("Please download from: https://sites.google.com/site/assistmentsdata/home/2015-assistments-skill-builder-data")
    print(f"And place it at: {data_dir / 'assistments2015_raw.csv'}")
    
    output_path = data_dir / "assistments2015_raw.csv"
    if output_path.exists():
        print(f"Found at {output_path}")
        return str(output_path)
    return None


def load_assistments_csv(filepath):
    """
    Load ASSISTments CSV and standardize column names.
    
    Different versions of ASSISTments datasets use slightly different column names
    (e.g., 'user_id' vs 'student_id', 'problem_id' vs 'question_id').
    This function handles these variations by mapping to standardized names.
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe with standardized column names:
            - user_id: Student identifier
            - problem_id: Question identifier
            - correct: Binary correctness (0/1)
            - order_id: Temporal ordering
    """
    # Load CSV (latin-1 encoding handles special characters in some datasets)
    df = pd.read_csv(filepath, encoding='latin-1')
    
    # Define required columns for Knowledge Tracing
    required_cols = ['user_id', 'problem_id', 'correct', 'order_id']
    missing = set(required_cols) - set(df.columns)
    
    # If any required columns are missing, try to find them under alternate names
    if missing:
        # Mapping of standard names to possible alternatives used in different dataset versions
        col_mapping = {
            'user_id': ['user_id', 'student_id', 'userId'],
            'problem_id': ['problem_id', 'question_id', 'item_id', 'problemId'],
            'correct': ['correct', 'correctness', 'score'],
            'order_id': ['order_id', 'timestamp', 'ms_first_response']
        }
        
        # Iterate through required columns and try to find alternatives
        for req_col, alternatives in col_mapping.items():
            if req_col not in df.columns:
                for alt in alternatives:
                    if alt in df.columns:
                        # Rename found alternative to standard name
                        df = df.rename(columns={alt: req_col})
                        break
    
    # Print basic statistics
    print(f"Loaded {len(df)} interactions from {len(df['user_id'].unique())} students")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


if __name__ == "__main__":
    # Download datasets
    path_2009 = download_assistments2009()
    path_2015 = download_assistments2015()
    
    if path_2009:
        df = load_assistments_csv(path_2009)
        print(f"\nASSISTments 2009 stats:")
        print(f"  Students: {df['user_id'].nunique()}")
        print(f"  Problems: {df['problem_id'].nunique()}")
        print(f"  Interactions: {len(df)}")

