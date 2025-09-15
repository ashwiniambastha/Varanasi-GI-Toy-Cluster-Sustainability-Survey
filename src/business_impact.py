# =============================================================================
# BUSINESS IMPACT ANALYSIS MODULE
# =============================================================================
"""
This module quantifies business impact, calculates ROI, and provides
policy recommendations based on the analysis results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_training_impact(df):
    """
    Calculate the impact of training programs on income and satisfaction
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Training impact analysis results
    """
    print("🎓 TRAINING PROGRAM IMPACT ANALYSIS:")
    
    if 'Training_Access' not in df.columns or 'Monthly_Income' not in df.columns:
        print("   ❌ Required columns not found for training analysis")
        return {}
    
    # Separate trained and untrained groups
    trained = df[df['Training_Access'] == 1]
    untrained = df[df['Training_Access'] == 0]
    
    if len(trained) == 0 or len(untrained) == 0:
        print("   ❌ Insufficient data for training impact analysis")
        return {}
    
    # Calculate income differences
    trained_income_mean = trained['Monthly_Income'].mean()
    untrained_income_mean = untrained['Monthly_Income'].mean()
    income_difference = trained_income_mean - untrained_income_mean
    
    # Calculate satisfaction differences if available
    satisfaction_difference = 0
    if 'Satisfaction_Score' in df.columns:
        trained_satisfaction = trained['Satisfaction_Score'].mean()
        untrained_satisfaction = untrained['Satisfaction_Score'].mean()
        satisfaction_difference = trained_satisfaction - untrained_satisfaction
    
    # Statistical significance test
    t_stat, p_value = stats.ttest_ind(
        trained['Monthly_Income'].dropna(),
        untrained['Monthly_Income'].dropna()
    )
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((trained['Monthly_Income'].var() + untrained['Monthly_Income'].var()) / 2)
    cohens_d = income_difference / pooled_std if pooled_std > 0 else 0
    
    training_impact = {
        'trained_count': len(trained),
        'untrained_count': len(untrained),
        'trained_income_mean': trained_income_mean,
        'untrained_income_mean': untrained_income_mean,
        'income_difference': income_difference,
        'income_difference_percent': (income_difference / untrained_income_mean) * 100 if untrained_income_mean > 0 else 0,
        'satisfaction_difference': satisfaction_difference,
        'statistical_significance': {
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'cohens_d': cohens_d
        }
    }
    
    print(f"   • Trained artisans: {len(trained)} ({len(trained)/len(df)*100:.1f}%)")
    print(f"   • Untrained artisans: {len(untrained)} ({len(untrained)/len(df)*100:.1f}%)")
    print(f"   • Income impact: ₹{income_difference:.0f}/month ({training_impact['income_difference_percent']:.1f}% increase)")
    
    if training_impact['statistical_significance']['is_significant']:
        print(f"   • Statistical significance: YES (p={p_value:.3f})")
    else:
        print(f"   • Statistical significance: NO (p={p_value:.3f})")
    
    print(f"   • Effect size (Cohen's d): {cohens_d:.3f}")
    
    return training_impact

def calculate_gi_registration_impact(df):
    """
    Calculate the impact of GI registration on artisan outcomes
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: GI registration impact analysis
    """
    print(f"\n🏷️  GI REGISTRATION IMPACT ANALYSIS:")
    
    if 'Is_GI_Beneficiary' not in df.columns or 'Monthly_Income' not in df.columns:
        print("   ❌ Required columns not found for GI analysis")
        return {}
    
    # Separate GI beneficiaries and non-beneficiaries
    gi_beneficiaries = df[df['Is_GI_Beneficiary'] == 1]
    non_beneficiaries = df[df['Is_GI_Beneficiary'] == 0]
    
    if len(gi_beneficiaries) == 0 or len(non_beneficiaries) == 0:
        print("   ❌ Insufficient data for GI impact analysis")
        return {}
    
    # Calculate income differences
    gi_income_mean = gi_beneficiaries['Monthly_Income'].mean()
    non_gi_income_mean = non_beneficiaries['Monthly_Income'].mean()
    income_difference = gi_income_mean - non_gi_income_mean
    
    # Calculate satisfaction differences if available
    satisfaction_difference = 0
    if 'Satisfaction_Score' in df.columns:
        gi_satisfaction = gi_beneficiaries['Satisfaction_Score'].mean()
        non_gi_satisfaction = non_beneficiaries['Satisfaction_Score'].mean()
        satisfaction_difference = gi_satisfaction - non_gi_satisfaction
    
    # Statistical significance test
    t_stat, p_value = stats.ttest_ind(
        gi_beneficiaries['Monthly_Income'].dropna(),
        non_beneficiaries['Monthly_Income'].dropna()
    )
    
    # Effect size
    pooled_std = np.sqrt((gi_beneficiaries['Monthly_Income'].var() + non_beneficiaries['Monthly_Income'].var()) / 2)
    cohens_d = income_difference / pooled_std if pooled_std > 0 else 0
    
    gi_impact = {
        'gi_beneficiaries_count': len(gi_beneficiaries),
        'non_beneficiaries_count': len(non_beneficiaries),
        'gi_income_mean': gi_income_mean,
        'non_gi_income_mean': non_gi_income_mean,
        'income_difference': income_difference,
        'income_difference_percent': (income_difference / non_gi_income_mean) * 100 if non_gi_income_mean > 0 else 0,
        'satisfaction_difference': satisfaction_difference,
        'statistical_significance': {
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'cohens_d': cohens_d
        }
    }
    
    print(f"   • GI Beneficiaries: {len(gi_beneficiaries)} ({len(gi_beneficiaries)/len(df)*100:.1f}%)")
    print(f"   • Non-beneficiaries: {len(non_beneficiaries)} ({len(non_beneficiaries)/len(df)*100:.1f}%)")
    print(f"   • Income impact: ₹{income_difference:.0f}/month ({gi_impact['income_difference_percent']:.1f}% increase)")
    
    if gi_impact['statistical_significance']['is_significant']:
        print(f"   • Statistical significance: YES (p={p_value:.3f})")
    else:
        print(f"   • Statistical significance: NO (p={p_value:.3f})")
    
    return gi_impact

def calculate_gender_impact(df):
    """
    Analyze gender-based income and opportunity differences
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Gender impact analysis
    """
    print(f"\n👥 GENDER IMPACT ANALYSIS:")
    
    if 'Primary_Earner_Gender' not in df.columns or 'Monthly_Income' not in df.columns:
        print("   ❌ Required columns not found for gender analysis")
        return {}
    
    # Analyze by gender
    gender_analysis = df.groupby('Primary_Earner_Gender').agg({
        'Monthly_Income': ['mean', 'median', 'std', 'count'],
        'Training_Access': 'mean',
        'Is_GI_Beneficiary': 'mean
