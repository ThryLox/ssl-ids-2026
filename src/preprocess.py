import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler
import hashlib

def clean_data(df):
    """
    Handle NaNs, Infs, drop constant columns and non-numeric identifiers.
    """
    # Strip whitespace from columns
    df.columns = df.columns.str.strip()
    
    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop columns that are entirely NaN
    df = df.dropna(axis=1, how='all')
    
    # Impute remaining NaNs with median
    # Note: Only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Drop constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df = df.drop(columns=constant_cols)
    
    # Drop non-numeric identifiers or high-cardinality metadata
    cols_to_drop = ['Flow ID', 'Timestamp', 'Source IP', 'Destination IP', 'Source Port'] 
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    return df

def check_leakage(df):
    """
    Deduplicate flows by feature vector to ensure identical flows aren't in different splits.
    """
    # Exclude Label from deduplication check if needed, but usually we want unique flows
    initial_count = len(df)
    df = df.drop_duplicates()
    final_count = len(df)
    print(f"Deduplication: Removed {initial_count - final_count} duplicate flows.")
    return df

def load_and_preprocess_all(data_dir, samples_per_file=20000):
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    dfs = []
    
    for f in all_files:
        print(f"Loading {f}...")
        df = pd.read_csv(f)
        
        # Strip whitespace from columns
        df.columns = df.columns.str.strip()
        
        # Keep all attacks, sample BENIGN
        if 'Label' in df.columns:
            df_attacks = df[df['Label'] != 'BENIGN']
            df_benign = df[df['Label'] == 'BENIGN']
            # Sample benign if too many, but keep all attacks
            n_benign = min(len(df_benign), samples_per_file)
            df_benign = df_benign.sample(n_benign, random_state=42)
            df = pd.concat([df_attacks, df_benign])
            
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = clean_data(combined_df)
    combined_df = check_leakage(combined_df)
    
    return combined_df

if __name__ == "__main__":
    DATA_DIR = r"c:\Users\ekonk\OneDrive\Desktop\vibe\researchCyber\paper1\data\extracted"
    # Using smarter sampling
    df_cleaned = load_and_preprocess_all(DATA_DIR)
    print(f"Final feature count: {len(df_cleaned.columns)}")
    print(f"Final row count: {len(df_cleaned)}")
    print("Label distribution:\n", df_cleaned['Label'].value_counts())
    
    # Save for verification
    os.makedirs("results", exist_ok=True)
    df_cleaned.to_csv("results/cleaned_sample_full.csv", index=False)
