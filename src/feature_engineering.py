# =============================================================================
# FEATURE ENGINEERING MODULE
# =============================================================================
"""
This module implements sophisticated feature engineering techniques
demonstrating domain expertise and advanced data science skills
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def create_economic_indicators(df):
    """
    Create economic and financial indicators
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with economic indicators
    """
    df_econ = df.copy()
    
    # Per capita income calculations
    df_econ['Income_Per_Member'] = df_econ['Monthly_Income'] / df_econ['Family_Size']
    df_econ['Income_Per_Adult'] = df_econ['Monthly_Income'] / np.maximum(df_econ['Family_Size'] - 2, 1)  # Assuming 2 children max
    
    # Economic vulnerability indicators
    median_income = df_econ['Monthly_Income'].median()
    median_family_size = df_econ['Family_Size'].median()
    
    df_econ['Economic_Vulnerability'] = (
        (df_econ['Monthly_Income'] < median_income) & 
        (df_econ['Family_Size'] >= median_family_size)
    ).astype(int)
    
    # Income stability proxies
    df_econ['High_Income_Low_Satisfaction'] = (
        (df_econ['Monthly_Income'] > median_income) & 
        (df_econ['Satisfaction_Score'] <= 3)
    ).astype(int)
    
    # Economic efficiency metrics
    df_econ['Income_Efficiency'] = df_econ['Monthly_Income'] / (df_econ['Family_Size'] + 1)
    
    return df_econ

def create_resource_accessibility_features(df):
    """
    Create features related to resource accessibility and infrastructure
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with resource features
    """
    df_resource = df.copy()
    
    # Resource scoring system
    resource_mapping = {'Easy': 3, 'Moderate': 2, 'Difficult': 1}
    df_resource['Raw_Material_Score'] = df_resource['Raw_Material_Access'].map(resource_mapping).fillna(2)
    
    # Composite resource accessibility
    df_resource['Total_Resource_Score'] = (
        df_resource['Raw_Material_Score'] + 
        df_resource['Training_Access']
    )
    
    # Resource-training synergy
    df_resource['Resource_Training_Balance'] = (
        df_resource['Raw_Material_Score'] * df_resource['Training_Access']
    )
    
    # Resource adequacy indicators
    df_resource['Resource_Adequate'] = (df_resource['Raw_Material_Score'] >= 2).astype(int)
    df_resource['Full_Resource_Access'] = (df_resource['Total_Resource_Score'] >= 4).astype(int)
    
    return df_resource

def create_statistical_features(df):
    """
    Create statistical and distributional features
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with statistical features
    """
    df_stats = df.copy()
    
    # Z-scores for normalization
    for col in ['Monthly_Income', 'Family_Size', 'Satisfaction_Score']:
        if col in df_stats.columns:
            mean_val = df_stats[col].mean()
            std_val = df_stats[col].std()
            df_stats[f'{col}_Zscore'] = (df_stats[col] - mean_val) / std_val
    
    # Percentile ranks
    df_stats['Income_Percentile'] = df_stats['Monthly_Income'].rank(pct=True)
    df_stats['Satisfaction_Percentile'] = df_stats['Satisfaction_Score'].rank(pct=True)
    
    # Quartile classifications
    df_stats['Income_Quartile'] = pd.qcut(
        df_stats['Monthly_Income'], 
        q=4, 
        labels=['Q1_Low', 'Q2_Medium_Low', 'Q3_Medium_High', 'Q4_High'],
        duplicates='drop'
    )
    
    # Statistical boundaries
    income_q25 = df_stats['Monthly_Income'].quantile(0.25)
    income_q75 = df_stats['Monthly_Income'].quantile(0.75)
    
    df_stats['Income_Outlier_Low'] = (df_stats['Monthly_Income'] < income_q25).astype(int)
    df_stats['Income_Outlier_High'] = (df_stats['Monthly_Income'] > income_q75).astype(int)
    
    return df_stats

def create_interaction_features(df):
    """
    Create interaction and synergy features
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with interaction features
    """
    df_interact = df.copy()
    
    # Gender-based features
    df_interact['Female_Earner'] = (df_interact['Primary_Earner_Gender'] == 'Female').astype(int)
    df_interact['Male_Earner'] = (df_interact['Primary_Earner_Gender'] == 'Male').astype(int)
    
    # Key interaction terms
    df_interact['GI_Training_Synergy'] = (
        df_interact['Is_GI_Beneficiary'] * df_interact['Training_Access']
    )
    
    df_interact['Female_Training_Access'] = (
        df_interact['Female_Earner'] * df_interact['Training_Access']
    )
    
    df_interact['Female_GI_Beneficiary'] = (
        df_interact['Female_Earner'] * df_interact['Is_GI_Beneficiary']
    )
    
    # Size-income interactions
    df_interact['Large_Family_High_Income'] = (
        (df_interact['Family_Size'] >= df_interact['Family_Size'].median()) * 
        (df_interact['Monthly_Income'] >= df_interact['Monthly_Income'].median())
    ).astype(int)
    
    # Triple interactions
    df_interact['Female_GI_Training'] = (
        df_interact['Female_Earner'] * 
        df_interact['Is_GI_Beneficiary'] * 
        df_interact['Training_Access']
    )
    
    return df_interact

def create_categorical_binning_features(df):
    """
    Create categorical binning and classification features
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with binned features
    """
    df_binned = df.copy()
    
    # Income categorization
    income_bins = [0, 6000, 8000, 10000, float('inf')]
    income_labels = ['Low', 'Medium', 'High', 'Very_High']
    df_binned['Income_Category'] = pd.cut(
        df_binned['Monthly_Income'], 
        bins=income_bins, 
        labels=income_labels,
        include_lowest=True
    )
    
    # Family size categorization
    family_bins = [0, 4, 6, 8, float('inf')]
    family_labels = ['Small', 'Medium', 'Large', 'Very_Large']
    df_binned['Family_Size_Category'] = pd.cut(
        df_binned['Family_Size'], 
        bins=family_bins, 
        labels=family_labels,
        include_lowest=True
    )
    
    # Satisfaction categorization
    df_binned['Satisfaction_Level'] = df_binned['Satisfaction_Score'].map({
        1: 'Very_Low', 2: 'Low', 3: 'Medium', 4: 'High', 5: 'Very_High'
    })
    
    # Binary classifications
    df_binned['High_Satisfaction'] = (df_binned['Satisfaction_Score'] >= 4).astype(int)
    df_binned['Large_Family'] = (df_binned['Family_Size'] >= 7).astype(int)
    df_binned['Above_Median_Income'] = (
        df_binned['Monthly_Income'] > df_binned['Monthly_Income'].median()
    ).astype(int)
    
    return df_binned

def create_performance_indicators(df):
    """
    Create performance and success indicators
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with performance indicators
    """
    df_perf = df.copy()
    
    # Efficiency metrics
    df_perf['Satisfaction_Efficiency'] = (
        df_perf['Satisfaction_Score'] / (df_perf.get('Total_Resource_Score', 3) + 1)
    )
    
    df_perf['Resource_Utilization'] = (
        df_perf['Monthly_Income'] / (df_perf.get('Total_Resource_Score', 3) + 1)
    )
    
    # Success indices
    df_perf['Basic_Success_Index'] = (
        df_perf.get('Above_Median_Income', 0) + 
        df_perf.get('High_Satisfaction', 0) + 
        df_perf['Is_GI_Beneficiary'] + 
        df_perf['Training_Access']
    )
    
    # Advanced success index with weights
    df_perf['Artisan_Success_Index'] = (
        0.3 * df_perf.get('Above_Median_Income', 0) + 
        0.2 * df_perf.get('High_Satisfaction', 0) + 
        0.25 * df_perf['Is_GI_Beneficiary'] + 
        0.25 * df_perf['Training_Access']
    )
    
    # Performance categories
    success_percentile_75 = df_perf['Artisan_Success_Index'].quantile(0.75)
    success_percentile_25 = df_perf['Artisan_Success_Index'].quantile(0.25)
    
    df_perf['Performance_Category'] = pd.cut(
        df_perf['Artisan_Success_Index'],
        bins=[-np.inf, success_percentile_25, success_percentile_75, np.inf],
        labels=['Underperforming', 'Average', 'High_Performing']
    )
    
    return df_perf

def create_domain_specific_features(df):
    """
    Create domain-specific features for artisan analysis
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with domain features
    """
    df_domain = df.copy()
    
    # Artisan business maturity indicators
    df_domain['Established_Artisan'] = (
        (df_domain['Is_GI_Beneficiary'] == 1) & 
        (df_domain['Training_Access'] == 1) & 
        (df_domain['Monthly_Income'] > df_domain['Monthly_Income'].median())
    ).astype(int)
    
    # Growth potential indicators
    df_domain['Growth_Potential'] = (
        (df_domain['Training_Access'] == 1) & 
        (df_domain['Is_GI_Beneficiary'] == 0) & 
        (df_domain['Satisfaction_Score'] >= 4)
    ).astype(int)
    
    # Support requirement indicators
    df_domain['High_Support_Need'] = (
        (df_domain['Training_Access'] == 0) & 
        (df_domain['Monthly_Income'] < df_domain['Monthly_Income'].quantile(0.3))
    ).astype(int)
    
    # Market readiness score
    resource_score = df_domain.get('Raw_Material_Score', 2)
    df_domain['Market_Readiness'] = (
        0.4 * df_domain['Training_Access'] + 
        0.3 * df_domain['Is_GI_Beneficiary'] + 
        0.3 * (resource_score / 3)
    )
    
    return df_domain

def engineer_features(df):
    """
    Main feature engineering function that orchestrates all feature creation
    
    Args:
        df (pd.DataFrame): Original dataframe
        
    Returns:
        tuple: (engineered_dataframe, list_of_new_features)
    """
    print(f"\n🔧 ADVANCED FEATURE ENGINEERING:")
    print(f"   • Starting with {df.shape[1]} original features")
    
    # Store original columns
    original_cols = set(df.columns)
    
    # Apply feature engineering steps
    df_eng = df.copy()
    
    # Step 1: Economic indicators
    df_eng = create_economic_indicators(df_eng)
    print(f"   • Economic indicators created")
    
    # Step 2: Resource accessibility
    df_eng = create_resource_accessibility_features(df_eng)
    print(f"   • Resource accessibility features created")
    
    # Step 3: Statistical features
    df_eng = create_statistical_features(df_eng)
    print(f"   • Statistical features created")
    
    # Step 4: Interaction features
    df_eng = create_interaction_features(df_eng)
    print(f"   • Interaction features created")
    
    # Step 5: Categorical binning
    df_eng = create_categorical_binning_features(df_eng)
    print(f"   • Categorical binning completed")
    
    # Step 6: Performance indicators
    df_eng = create_performance_indicators(df_eng)
    print(f"   • Performance indicators created")
    
    # Step 7: Domain-specific features
    df_eng = create_domain_specific_features(df_eng)
    print(f"   • Domain-specific features created")
    
    # Identify new features
    new_features = list(set(df_eng.columns) - original_cols)
    
    print(f"   • Final dataset: {df_eng.shape[1]} features ({len(new_features)} new)")
    print(f"   • Key engineered features:")
    key_features = [
        'Income_Per_Member', 'GI_Training_Synergy', 'Artisan_Success_Index',
        'Economic_Vulnerability', 'Market_Readiness'
    ]
    for feature in key_features:
        if feature in new_features:
            print(f"     - {feature}")
    
    return df_eng, new_features

def get_feature_importance_groups():
    """
    Return feature groups for importance analysis
    
    Returns:
        dict: Feature groups organized by type
    """
    feature_groups = {
        'economic': [
            'Income_Per_Member', 'Economic_Vulnerability', 'Income_Efficiency',
            'Above_Median_Income', 'Income_Percentile'
        ],
        'resource': [
            'Raw_Material_Score', 'Total_Resource_Score', 'Resource_Training_Balance',
            'Full_Resource_Access', 'Market_Readiness'
        ],
        'interaction': [
            'GI_Training_Synergy', 'Female_Training_Access', 'Female_GI_Beneficiary',
            'Large_Family_High_Income'
        ],
        'performance': [
            'Artisan_Success_Index', 'Performance_Category', 'Satisfaction_Efficiency',
            'Growth_Potential', 'Established_Artisan'
        ],
        'demographic': [
            'Female_Earner', 'Large_Family', 'Family_Size_Category',
            'High_Satisfaction'
        ]
    }
    
    return feature_groups
