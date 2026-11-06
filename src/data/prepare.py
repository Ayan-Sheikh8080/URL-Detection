import os
import json
import hashlib
import pandas as pd
from datetime import datetime
from pathlib import Path

# Configuration
DATA_RAW_PATH = Path("data/raw/sample_urls.csv")
PROCESSED_PATH = Path("data/processed/urls_processed.csv")
META_PATH = Path("data/metadata/dataset_meta.json")
DVC_FILE = Path("dvc.yaml")
PARAMS_FILE = Path("params.yaml")

def validate_data():
    """Validate the raw dataset for missing values, invalid URLs, and label issues."""
    print("=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)
    
    issues = []
    
    # Check if raw data exists
    if not DATA_RAW_PATH.exists():
        print(f"ERROR: Raw data not found at {DATA_RAW_PATH}")
        return False
    
    # Load data
    try:
        df = pd.read_csv(DATA_RAW_PATH)
    except Exception as e:
        print(f"ERROR: Failed to load CSV: {e}")
        return False
    
    # Check for missing values
    if df.isnull().any().any():
        missing_count = df.isnull().sum().sum()
        issues.append(f"Missing values found: {missing_count} total")
        print(f"⚠ WARNING: Missing values found: {missing_count} total")
    
    # Check for required columns
    required_cols = ['url', 'label', 'source']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        print(f"ERROR: Missing required columns: {missing_cols}")
        return False
    
    # Validate URLs
    import re
    url_pattern = re.compile(r'^https?://[^\s]+$')
    invalid_urls = []
    for url in df['url']:
        if not url_pattern.match(str(url)):
            invalid_urls.append(str(url))
    
    if invalid_urls:
        issues.append(f"Invalid URLs found: {len(invalid_urls)}")
        print(f"⚠ WARNING: Invalid URLs found: {len(invalid_urls)}")
    
    # Validate labels
    valid_labels = {0, 1}
    invalid_labels = df[~df['label'].isin(valid_labels)]['label'].unique().tolist()
    if invalid_labels:
        issues.append(f"Invalid labels found: {invalid_labels}")
        print(f"⚠ WARNING: Invalid labels found: {invalid_labels}")
    
    # Check for single class data
    if len(df['label'].unique()) < 2:
        issues.append("Single class data - only one label present")
        print(f"⚠ WARNING: Single class data - only label {df['label'].iloc[0]} present")
    
    # Report
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    print(f"Sources: {df['source'].unique().tolist()}")
    
    if issues:
        print(f"\nIssues found: {len(issues)}")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✓ Data validation passed")
        return True

def deduplicate_data():
    """Remove duplicate URLs from the dataset."""
    print("\n" + "=" * 60)
    print("DUPLICATE REMOVAL")
    print("=" * 60)
    
    df = pd.read_csv(DATA_RAW_PATH)
    original_count = len(df)
    
    # Remove duplicates based on URL
    df_deduped = df.drop_duplicates(subset=['url'], keep='first')
    dedup_count = original_count - len(df_deduped)
    
    print(f"Original rows: {original_count}")
    print(f"After dedup: {len(df_deduped)}")
    print(f"Removed: {dedup_count} duplicates")
    
    return df_deduped

def compute_dataset_hash(filepath):
    """Compute SHA-256 hash of dataset file for versioning."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdump()

def generate_metadata(df_deduped):
    """Generate dataset metadata for DVC tracking."""
    print("\n" + "=" * 60)
    print("GENERATING DATASET METADATA")
    print("=" * 60)
    
    # Compute hash
    file_hash = compute_dataset_hash(DATA_RAW_PATH)
    
    # Class distribution
    class_counts = df_deduped['label'].value_counts().to_dict()
    class_share = {str(k): v / len(df_deduped) for k, v in class_counts.items()}
    
    meta = {
        "row_count": int(len(df_deduped)),
        "column_names": list(df_deduped.columns),
        "class_distribution": class_counts,
        "class_share": class_share,
        "source": "sample_urls.csv",
        "dataset_hash": file_hash,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "version": "1.0.0"
    }
    
    # Ensure output directory exists
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Write metadata
    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"Metadata written to {META_PATH}")
    print(f"  Rows: {meta['row_count']}")
    print(f"  Class distribution: {meta['class_distribution']}")
    print(f"  Dataset hash: {file_hash[:16]}...")
    
    return meta

def main():
    """Main preparation pipeline step."""
    print("Phishing URL Data Preparation Pipeline")
    print("=" * 60)
    
    # Step 1: Validate
    if not validate_data():
        print("\n⚠ Validation had issues, but continuing...")
    
    # Step 2: Deduplicate
    df_deduped = deduplicate_data()
    
    # Step 3: Generate metadata
    meta = generate_metadata(df_deduped)
    
    # Step 4: Save processed data
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_deduped.to_csv(PROCESSED_PATH, index=False)
    print(f"\nProcessed data saved to {PROCESSED_PATH}")
    
    print("\n" + "=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()