# =============================================================================
# DATA LOADING AND INITIAL EXPLORATION MODULE
# =============================================================================
"""
This module handles data loading and provides initial dataset overview
"""

import pandas as pd
import numpy as np

def load_and_explore_data(file_path):
    """
    Load dataset and provide comprehensive overview
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        
        # Print dataset overview
        print(f"📊 Dataset Overview:")
        print(f"   • Shape: {df.shape[0]} households × {df.shape[1]} features")
        print(f"   • Missing Values: {df.isnull().sum().sum()} total")
        print(f"   • Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # Data types summary
        dtype_counts = df.dtypes.value_counts().to_dict()
        print(f"   • Data Types: {dtype_counts}")
        
        # Basic statistics for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print(f"   • Numerical Columns: {len(numerical_cols)}")
            print(f"   • Categorical Columns: {df.shape[1] - len(numerical_cols)}")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"   ⚠️  Duplicate Rows: {duplicates}")
        
        # Missing value analysis
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            print(f"   • Columns with Missing Values: {len(missing_cols)}")
            for col in missing_cols:
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                print(f"     - {col}: {missing_pct:.1f}% missing")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None

def get_dataset_summary(df):
    """
    Generate comprehensive dataset summary statistics
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Summary statistics
    """
    summary = {
        'shape': df.shape,
        'missing_total': df.isnull().sum().sum(),
        'memory_usage_kb': df.memory_usage(deep=True).sum() / 1024,
        'numerical_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'duplicates': df.duplicated().sum()
    }
    
    return summary
