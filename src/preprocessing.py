# =============================================================================
# DATA PREPROCESSING MODULE
# =============================================================================
"""
This module implements a comprehensive data preprocessing pipeline
including missing value imputation, encoding, scaling, and validation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def handle_missing_values(df, strategy='knn', n_neighbors=5):
    """
    Handle missing values using various imputation strategies
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str): Imputation strategy ('knn', 'mean', 'median', 'mode')
        n_neighbors (int): Number of neighbors for KNN imputation
        
    Returns:
        tuple: (imputed_dataframe, imputation_info)
    """
    print(f"🔄 HANDLING MISSING VALUES ({strategy.upper()} strategy):")
    
    df_imputed = df.copy()
    imputation_info = {
        'strategy': strategy,
        'columns_imputed': [],
        'missing_before': df.isnull().sum().sum(),
        'missing_after': 0
    }
    
    # Identify columns with missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    
    if not missing_cols:
        print("   ✅ No missing values found!")
        return df_imputed, imputation_info
    
    print(f"   • Found missing values in {len(missing_cols)} columns")
    
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    missing_numerical = [col for col in missing_cols if col in numerical_cols]
    missing_categorical = [col for col in missing_cols if col in categorical_cols]
    
    # Handle numerical missing values
    if missing_numerical:
        print(f"   • Imputing {len(missing_numerical)} numerical columns")
        
        if strategy == 'knn':
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_imputed[missing_numerical] = imputer.fit_transform(df[missing_numerical])
        elif strategy == 'mean':
            imputer = SimpleImputer(strategy='mean')
            df_imputed[missing_numerical] = imputer.fit_transform(df[missing_numerical])
        elif strategy == 'median':
            imputer = SimpleImputer(strategy='median')
            df_imputed[missing_numerical] = imputer.fit_transform(df[missing_numerical])
        
        imputation_info['columns_imputed'].extend(missing_numerical)
        
        for col in missing_numerical:
            missing_count = df[col].isnull().sum()
            print(f"     - {col}: {missing_count} values imputed")
    
    # Handle categorical missing values
    if missing_categorical:
        print(f"   • Imputing {len(missing_categorical)} categorical columns")
        
        for col in missing_categorical:
            if strategy == 'mode' or strategy == 'knn':
                # Use mode for categorical variables
                mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                df_imputed[col] = df_imputed[col].fillna(mode_value)
            else:
                # Fill with 'Unknown' category
                df_imputed[col] = df_imputed[col].fillna('Unknown')
            
            missing_count = df[col].isnull().sum()
            imputation_info['columns_imputed'].append(col)
            print(f"     - {col}: {missing_count} values imputed")
    
    # Update missing count after imputation
    imputation_info['missing_after'] = df_imputed.isnull().sum().sum()
    
    print(f"   ✅ Missing values reduced from {imputation_info['missing_before']} to {imputation_info['missing_after']}")
    
    return df_imputed, imputation_info

def encode_categorical_variables(df, encoding_method='label'):
    """
    Encode categorical variables using various encoding methods
    
    Args:
        df (pd.DataFrame): Input dataframe
        encoding_method (str): Encoding method ('label', 'onehot', 'ordinal')
        
    Returns:
        tuple: (encoded_dataframe, encoding_info)
    """
    print(f"🔤 ENCODING CATEGORICAL VARIABLES ({encoding_method.upper()} encoding):")
    
    df_encoded = df.copy()
    encoding_info = {
        'method': encoding_method,
        'encoders': {},
        'columns_encoded': [],
        'new_columns': []
    }
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not categorical_cols:
        print("   ✅ No categorical variables found!")
        return df_encoded, encoding_info
    
    print(f"   • Found {len(categorical_cols)} categorical columns to encode")
    
    for col in categorical_cols:
        unique_values = df[col].nunique()
        print(f"   • {col}: {unique_values} unique values")
        
        if encoding_method == 'label':
            # Label encoding
            le = LabelEncoder()
            
            # Handle potential NaN values
            df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
            
            encoding_info['encoders'][col] = le
            encoding_info['columns_encoded'].append(col)
            encoding_info['new_columns'].append(f'{col}_encoded')
            
            print(f"     - Created: {col}_encoded")
        
        elif encoding_method == 'onehot' and unique_values <= 10:
            # One-hot encoding (only for columns with <= 10 categories)
            dummy_cols = pd.get_dummies(df_encoded[col], prefix=col, dummy_na=True)
            df_encoded = pd.concat([df_encoded, dummy_cols], axis=1)
            
            encoding_info['columns_encoded'].append(col)
            encoding_info['new_columns'].extend(dummy_cols.columns.tolist())
            
            print(f"     - Created: {len(dummy_cols.columns)} dummy variables")
        
        elif encoding_method == 'ordinal':
            # Ordinal encoding (manual mapping required for specific columns)
            if col == 'Raw_Material_Access':
                ordinal_mapping = {'Easy': 3, 'Moderate': 2, 'Difficult': 1}
                df_encoded[f'{col}_encoded'] = df_encoded[col].map(ordinal_mapping).fillna(2)
                
                encoding_info['encoders'][col] = ordinal_mapping
                encoding_info['columns_encoded'].append(col)
                encoding_info['new_columns'].append(f'{col}_encoded')
                
                print(f"     - Created: {col}_encoded (ordinal)")
            else:
                # Fall back to label encoding
                le = LabelEncoder()
                df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                encoding_info['encoders'][col] = le
                encoding_info['columns_encoded'].append(col)
                encoding_info['new_columns'].append(f'{col}_encoded')
    
    print(f"   ✅ Encoded {len(encoding_info['columns_encoded'])} categorical columns")
    
    return df_encoded, encoding_info

def scale_features(df, scaling_method='standard', feature_cols=None):
    """
    Scale numerical features using various scaling methods
    
    Args:
        df (pd.DataFrame): Input dataframe
        scaling_method (str): Scaling method ('standard', 'minmax', 'robust')
        feature_cols (list): Specific columns to scale (if None, scales all numerical)
        
    Returns:
        tuple: (scaled_dataframe, scaling_info)
    """
    print(f"⚖️  SCALING FEATURES ({scaling_method.upper()} scaling):")
    
    df_scaled = df.copy()
    scaling_info = {
        'method': scaling_method,
        'scaler': None,
        'columns_scaled': [],
        'scaling_stats': {}
    }
    
    # Determine columns to scale
    if feature_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude binary columns (0/1) from scaling
        feature_cols = []
        for col in numerical_cols:
            unique_vals = df[col].dropna().unique()
            if not (len(unique_vals) == 2 and set(unique_vals).issubset({0, 1})):
                feature_cols.append(col)
    
    # Filter to existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    if not feature_cols:
        print("   ✅ No numerical features to scale!")
        return df_scaled, scaling_info
    
    print(f"   • Scaling {len(feature_cols)} numerical features")
    
    # Select appropriate scaler
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_method == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()  # Default
    
    # Fit and transform
    try:
        df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
        
        scaling_info['scaler'] = scaler
        scaling_info['columns_scaled'] = feature_cols
        
        # Store scaling statistics
        for i, col in enumerate(feature_cols):
            original_mean = df[col].mean()
            original_std = df[col].std()
            scaled_mean = df_scaled[col].mean()
            scaled_std = df_scaled[col].std()
            
            scaling_info['scaling_stats'][col] = {
                'original_mean': original_mean,
                'original_std': original_std,
                'scaled_mean': scaled_mean,
                'scaled_std': scaled_std
            }
            
            print(f"     - {col}: μ={original_mean:.2f}→{scaled_mean:.2f}, σ={original_std:.2f}→{scaled_std:.2f}")
        
        print(f"   ✅ Successfully scaled {len(feature_cols)} features")
        
    except Exception as e:
        print(f"   ❌ Error during scaling: {str(e)}")
        scaling_info['error'] = str(e)
    
    return df_scaled, scaling_info

def detect_and_handle_outliers(df, method='iqr', action='cap'):
    """
    Detect and handle outliers in numerical columns
    
    Args:
        df (pd.DataFrame): Input dataframe
        method (str): Detection method ('iqr', 'zscore')
        action (str): Action to take ('cap', 'remove', 'flag')
        
    Returns:
        tuple: (processed_dataframe, outlier_info)
    """
    print(f"📊 OUTLIER DETECTION AND HANDLING ({method.upper()} method, {action} action):")
    
    df_processed = df.copy()
    outlier_info = {
        'method': method,
        'action': action,
        'outliers_detected': {},
        'total_outliers': 0,
        'columns_processed': []
    }
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numerical_cols:
        print("   ✅ No numerical columns for outlier detection!")
        return df_processed, outlier_info
    
    print(f"   • Processing {len(numerical_cols)} numerical columns")
    
    for col in numerical_cols:
        outliers_idx = []
        
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_idx = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers_idx = df[z_scores > 3].index.tolist()
            lower_bound = df[col].mean() - 3 * df[col].std()
            upper_bound = df[col].mean() + 3 * df[col].std()
        
        outlier_count = len(outliers_idx)
        
        if outlier_count > 0:
            outlier_info['outliers_detected'][col] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(df)) * 100,
                'indices': outliers_idx,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            # Take action on outliers
            if action == 'cap':
                df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
                df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound
                print(f"   • {col}: {outlier_count} outliers capped ({(outlier_count/len(df)*100):.1f}%)")
            
            elif action == 'remove':
                df_processed = df_processed.drop(outliers_idx)
                print(f"   • {col}: {outlier_count} outliers removed ({(outlier_count/len(df)*100):.1f}%)")
            
            elif action == 'flag':
                df_processed[f'{col}_outlier_flag'] = 0
                df_processed.loc[outliers_idx, f'{col}_outlier_flag'] = 1
                print(f"   • {col}: {outlier_count} outliers flagged ({(outlier_count/len(df)*100):.1f}%)")
            
            outlier_info['total_outliers'] += outlier_count
            outlier_info['columns_processed'].append(col)
        else:
            print(f"   • {col}: No outliers detected")
    
    print(f"   ✅ Processed outliers in {len(outlier_info['columns_processed'])} columns")
    
    return df_processed, outlier_info

def create_train_test_splits(df, target_cols=['Monthly_Income', 'Is_GI_Beneficiary'], 
                           test_size=0.2, random_state=42):
    """
    Create train-test splits for multiple target variables
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_cols (list): List of target column names
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing train-test splits
    """
    print(f"🎯 CREATING TRAIN-TEST SPLITS:")
    
    splits = {}
    
    # Identify feature columns (exclude targets)
    all_cols = df.columns.tolist()
    feature_cols = [col for col in all_cols if col not in target_cols]
    
    print(f"   • Features: {len(feature_cols)} columns")
    print(f"   • Targets: {len(target_cols)} columns")
    print(f"   • Test size: {test_size*100}%")
    
    # Prepare features
    X = df[feature_cols]
    
    # Handle missing values in features
    X_clean = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
    
    for target_col in target_cols:
        if target_col not in df.columns:
            print(f"   ⚠️  Warning: Target column '{target_col}' not found")
            continue
        
        # Prepare target
        y = df[target_col]
        
        # Remove rows where target is missing
        mask = ~y.isnull()
        X_target = X_clean[mask]
        y_target = y[mask]
        
        # Create split
        X_train, X_test, y_train, y_test = train_test_split(
            X_target, y_target, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_target if y_target.nunique() < 10 else None  # Stratify for classification
        )
        
        splits[target_col] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_cols,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        print(f"   • {target_col}: Train={len(X_train)}, Test={len(X_test)}")
    
    return splits

def validate_preprocessing(df_original, df_processed, preprocessing_info):
    """
    Validate preprocessing results and generate summary
    
    Args:
        df_original (pd.DataFrame): Original dataframe
        df_processed (pd.DataFrame): Processed dataframe
        preprocessing_info (dict): Information from preprocessing steps
        
    Returns:
        dict: Validation results
    """
    print(f"✅ VALIDATING PREPROCESSING RESULTS:")
    
    validation = {
        'shape_change': {
            'original': df_original.shape,
            'processed': df_processed.shape,
            'rows_changed': df_processed.shape[0] - df_original.shape[0],
            'cols_changed': df_processed.shape[1] - df_original.shape[1]
        },
        'missing_values': {
            'original': df_original.isnull().sum().sum(),
            'processed': df_processed.isnull().sum().sum(),
            'reduction': df_original.isnull().sum().sum() - df_processed.isnull().sum().sum()
        },
        'data_types': {
            'original_types': df_original.dtypes.value_counts().to_dict(),
            'processed_types': df_processed.dtypes.value_counts().to_dict()
        },
        'validation_passed': True,
        'issues': []
    }
    
    # Check for issues
    if df_processed.isnull().sum().sum() > 0:
        validation['issues'].append("Missing values still present after preprocessing")
    
    if df_processed.shape[0] < df_original.shape[0] * 0.95:  # Lost more than 5% of rows
        validation['issues'].append("Significant data loss during preprocessing")
        validation['validation_passed'] = False
    
    # Print validation summary
    print(f"   • Shape: {validation['shape_change']['original']} → {validation['shape_change']['processed']}")
    print(f"   • Missing values: {validation['missing_values']['original']} → {validation['missing_values']['processed']}")
    
    if validation['validation_passed']:
        print(f"   ✅ Validation PASSED")
    else:
        print(f"   ⚠️  Validation issues found:")
        for issue in validation['issues']:
            print(f"      - {issue}")
    
    return validation

def preprocess_data(df):
    """
    Main preprocessing pipeline function
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: (processed_dataframe, encoders_dict, scaler_object)
    """
    print(f"🔄 COMPREHENSIVE DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Store original dataframe for validation
    df_original = df.copy()
    
    # Step 1: Handle missing values
    df_processed, imputation_info = handle_missing_values(df, strategy='knn', n_neighbors=5)
    
    # Step 2: Detect and handle outliers
    df_processed, outlier_info = detect_and_handle_outliers(df_processed, method='iqr', action='cap')
    
    # Step 3: Encode categorical variables
    df_processed, encoding_info = encode_categorical_variables(df_processed, encoding_method='label')
    
    # Step 4: Scale numerical features
    scaling_features = ['Monthly_Income', 'Family_Size', 'Satisfaction_Score']
    df_processed, scaling_info = scale_features(df_processed, scaling_method='standard', 
                                              feature_cols=scaling_features)
    
    # Step 5: Validation
    preprocessing_summary = {
        'imputation': imputation_info,
        'outliers': outlier_info,
        'encoding': encoding_info,
        'scaling': scaling_info
    }
    
    validation_results = validate_preprocessing(df_original, df_processed, preprocessing_summary)
    
    # Print final summary
    print(f"\n📋 PREPROCESSING SUMMARY:")
    print(f"   • Original shape: {df_original.shape}")
    print(f"   • Processed shape: {df_processed.shape}")
    print(f"   • Missing values eliminated: {validation_results['missing_values']['reduction']}")
    print(f"   • New encoded columns: {len(encoding_info.get('new_columns', []))}")
    print(f"   • Scaled features: {len(scaling_info.get('columns_scaled', []))}")
    
    return df_processed, encoding_info.get('encoders', {}), scaling_info.get('scaler')

if __name__ == "__main__":
    print("🔄 DATA PREPROCESSING MODULE")
    print("This module provides comprehensive data preprocessing capabilities.")
    print("Run main.py to execute the complete analysis pipeline.")
