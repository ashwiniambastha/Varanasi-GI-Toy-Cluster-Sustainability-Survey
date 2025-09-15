# =============================================================================
# UTILITY FUNCTIONS MODULE
# =============================================================================
"""
This module contains utility functions for project setup, 
file management, and result saving
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def setup_directories():
    """
    Create necessary project directories if they don't exist
    """
    directories = [
        'data',
        'src', 
        'results',
        'visualizations',
        'docs',
        'models'
    ]
    
    print("📁 SETTING UP PROJECT DIRECTORIES:")
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"   ✓ Created: {directory}/")
        else:
            print(f"   ✓ Exists: {directory}/")

def save_results(df_final, results_dict, output_dir='results'):
    """
    Save all analysis results to files
    
    Args:
        df_final (pd.DataFrame): Final processed dataframe
        results_dict (dict): Dictionary containing all analysis results
        output_dir (str): Output directory path
    """
    print(f"\n💾 SAVING ANALYSIS RESULTS:")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Save final dataset
        output_file = os.path.join(output_dir, 'processed_data.csv')
        df_final.to_csv(output_file, index=False)
        print(f"   ✓ Dataset saved: {output_file}")
        
        # Save ML model performance
        ml_results = results_dict.get('ml', {})
        if ml_results.get('regression'):
            reg_performance = []
            for model_name, metrics in ml_results['regression'].items():
                reg_performance.append({
                    'model': model_name,
                    'cv_mean': metrics['cv_mean'],
                    'cv_std': metrics['cv_std'],
                    'test_r2': metrics['test_r2'],
                    'test_rmse': metrics['test_rmse']
                })
            
            reg_df = pd.DataFrame(reg_performance)
            reg_file = os.path.join(output_dir, 'regression_performance.csv')
            reg_df.to_csv(reg_file, index=False)
            print(f"   ✓ Regression results saved: {reg_file}")
        
        # Save classification results
        if ml_results.get('classification'):
            clf_performance = []
            for model_name, metrics in ml_results['classification'].items():
                clf_performance.append({
                    'model': model_name,
                    'cv_mean': metrics['cv_mean'],
                    'cv_std': metrics['cv_std'],
                    'test_accuracy': metrics['test_acc']
                })
            
            clf_df = pd.DataFrame(clf_performance)
            clf_file = os.path.join(output_dir, 'classification_performance.csv')
            clf_df.to_csv(clf_file, index=False)
            print(f"   ✓ Classification results saved: {clf_file}")
        
        # Save feature importance
        if ml_results.get('feature_importance') is not None:
            feature_file = os.path.join(output_dir, 'feature_importance.csv')
            ml_results['feature_importance'].to_csv(feature_file, index=False)
            print(f"   ✓ Feature importance saved: {feature_file}")
        
        # Save comprehensive results as JSON
        # Convert numpy types to Python types for JSON serialization
        results_serializable = convert_numpy_types(results_dict)
        
        json_file = os.path.join(output_dir, 'comprehensive_results.json')
        with open(json_file, 'w') as f:
            json.dump(results_serializable, f, indent=2, default=str)
        print(f"   ✓ Comprehensive results saved: {json_file}")
        
    except Exception as e:
        print(f"   ❌ Error saving results: {str(e)}")

def convert_numpy_types(obj):
    """
    Convert numpy types to Python types for JSON serialization
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    else:
        return obj

def load_and_validate_data(file_path):
    """
    Load data with comprehensive validation
    
    Args:
        file_path (str): Path to the data file
        
    Returns:
        pd.DataFrame: Validated dataframe
    """
    try:
        df = pd.read_csv(file_path)
        
        # Basic validation
        if df.empty:
            raise ValueError("Dataset is empty")
        
        if df.shape[0] < 10:
            print(f"⚠️  Warning: Small dataset ({df.shape[0]} rows)")
        
        if df.duplicated().sum() > 0:
            print(f"⚠️  Warning: {df.duplicated().sum()} duplicate rows found")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None

def create_project_metadata():
    """
    Create project metadata for documentation
    
    Returns:
        dict: Project metadata
    """
    metadata = {
        'project_name': 'Varanasi GI Toy Cluster Survey Analysis',
        'version': '1.0.0',
        'created_date': datetime.now().isoformat(),
        'description': 'Comprehensive machine learning analysis of artisan households',
        'author': 'Data Science Team',
        'technologies': [
            'Python 3.8+',
            'pandas',
            'scikit-learn', 
            'XGBoost',
            'matplotlib',
            'seaborn'
        ],
        'objectives': [
            'Income prediction modeling',
            'GI beneficiary classification',
            'Artisan segmentation analysis',
            'Policy impact quantification'
        ]
    }
    
    return metadata

def format_currency(amount, currency='₹'):
    """
    Format currency amounts consistently
    
    Args:
        amount (float): Amount to format
        currency (str): Currency symbol
        
    Returns:
        str: Formatted currency string
    """
    if amount >= 100000:  # 1 Lakh or more
        return f"{currency}{amount/100000:.1f}L"
    elif amount >= 1000:  # 1 thousand or more
        return f"{currency}{amount/1000:.1f}K"
    else:
        return f"{currency}{amount:.0f}"

def calculate_summary_statistics(df, column):
    """
    Calculate comprehensive summary statistics
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name
        
    Returns:
        dict: Summary statistics
    """
    if column not in df.columns:
        return {}
    
    series = df[column].dropna()
    
    stats = {
        'count': len(series),
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'q25': series.quantile(0.25),
        'q75': series.quantile(0.75),
        'iqr': series.quantile(0.75) - series.quantile(0.25),
        'skewness': series.skew(),
        'kurtosis': series.kurtosis()
    }
    
    return stats

def create_feature_summary(df, engineered_features):
    """
    Create summary of engineered features
    
    Args:
        df (pd.DataFrame): Dataframe with engineered features
        engineered_features (list): List of engineered feature names
        
    Returns:
        dict: Feature summary
    """
    summary = {
        'total_features': df.shape[1],
        'original_features': df.shape[1] - len(engineered_features),
        'engineered_features': len(engineered_features),
        'feature_types': {},
        'missing_values': {},
        'feature_correlations': {}
    }
    
    # Analyze feature types
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            summary['feature_types'][col] = 'numerical'
        else:
            summary['feature_types'][col] = 'categorical'
    
    # Calculate missing values
    for col in df.columns:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        if missing_pct > 0:
            summary['missing_values'][col] = missing_pct
    
    # Calculate correlations with target variables
    target_vars = ['Monthly_Income', 'Is_GI_Beneficiary', 'Satisfaction_Score']
    for target in target_vars:
        if target in df.columns:
            correlations = {}
            for feature in engineered_features:
                if feature in df.columns and df[feature].dtype in ['int64', 'float64']:
                    try:
                        corr = df[target].corr(df[feature])
                        if not pd.isna(corr):
                            correlations[feature] = corr
                    except:
                        continue
            summary['feature_correlations'][target] = correlations
    
    return summary

def print_analysis_summary(results_dict):
    """
    Print formatted analysis summary
    
    Args:
        results_dict (dict): Complete analysis results
    """
    print(f"\n📊 ANALYSIS SUMMARY REPORT")
    print("=" * 50)
    
    # Dataset info
    feature_count = results_dict.get('feature_count', 0)
    print(f"📁 Dataset Information:")
    print(f"   • Total Households: 119")
    print(f"   • Engineered Features: {feature_count}")
    print(f"   • Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
    
    # ML Performance
    ml_results = results_dict.get('ml', {})
    if ml_results:
        print(f"\n🤖 Machine Learning Performance:")
        
        if ml_results.get('regression'):
            best_r2 = max([v['test_r2'] for v in ml_results['regression'].values()])
            best_reg_model = max(ml_results['regression'].items(), key=lambda x: x[1]['test_r2'])[0]
            print(f"   • Best Regression: {best_reg_model} (R² = {best_r2:.3f})")
        
        if ml_results.get('classification'):
            best_acc = max([v['test_acc'] for v in ml_results['classification'].values()])
            best_clf_model = max(ml_results['classification'].items(), key=lambda x: x[1]['test_acc'])[0]
            print(f"   • Best Classification: {best_clf_model} (Acc = {best_acc:.3f})")
    
    # Business Impact
    impact_results = results_dict.get('business_impact', {})
    if impact_results:
        print(f"\n💰 Business Impact:")
        training_impact = impact_results.get('training_impact', 0)
        gi_impact = impact_results.get('gi_impact', 0)
        economic_impact = impact_results.get('economic_impact_lakh', 0)
        
        print(f"   • Training Impact: {format_currency(training_impact)}/month")
        print(f"   • GI Impact: {format_currency(gi_impact)}/month")
        print(f"   • Economic Potential: {format_currency(economic_impact * 100000)}/year")
    
    # Clustering
    cluster_results = results_dict.get('clustering', {})
    if cluster_results:
        clusters = cluster_results.get('optimal_clusters', 4)
        print(f"\n🎯 Segmentation Analysis:")
        print(f"   • Artisan Segments: {clusters}")
        print(f"   • Clustering Algorithm: K-Means")

def export_portfolio_summary(results_dict, output_file='results/portfolio_summary.md'):
    """
    Export portfolio-ready summary to markdown file
    
    Args:
        results_dict (dict): Analysis results
        output_file (str): Output file path
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Extract key metrics
        ml_results = results_dict.get('ml', {})
        impact_results = results_dict.get('business_impact', {})
        feature_count = results_dict.get('feature_count', 0)
        
        best_r2 = 0
        best_acc = 0
        if ml_results.get('regression'):
            best_r2 = max([v['test_r2'] for v in ml_results['regression'].values()])
        if ml_results.get('classification'):
            best_acc = max([v['test_acc'] for v in ml_results['classification'].values()])
        
        # Create markdown content
        content = f"""# Varanasi GI Toy Survey Analysis - Portfolio Summary

## Project Highlights
- **Dataset**: 119 households, {feature_count}+ engineered features
- **ML Performance**: R² = {best_r2:.3f}, Accuracy = {best_acc:.3f}
- **Economic Impact**: ₹{impact_results.get('economic_impact_lakh', 0):.1f}L potential annual benefit
- **Technologies**: Python, XGBoost, scikit-learn, Advanced Analytics

## Technical Achievements
- Advanced feature engineering with domain expertise
- Comprehensive ML pipeline with 5+ algorithms
- Statistical analysis and business impact quantification
- Professional visualization and reporting

## Business Value
- Identified key factors influencing artisan income
- Quantified training program ROI
- Provided actionable policy recommendations
- Delivered stakeholder-ready insights

*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(output_file, 'w') as f:
            f.write(content)
        
        print(f"   ✓ Portfolio summary exported: {output_file}")
        
    except Exception as e:
        print(f"   ❌ Error exporting portfolio summary: {str(e)}")

def validate_results(results_dict):
    """
    Validate analysis results for completeness
    
    Args:
        results_dict (dict): Results dictionary
        
    Returns:
        bool: True if validation passes
    """
    required_keys = ['ml', 'business_impact', 'clustering', 'feature_count']
    
    print(f"\n✅ VALIDATING ANALYSIS RESULTS:")
    
    validation_passed = True
    
    for key in required_keys:
        if key in results_dict:
            print(f"   ✓ {key}: Present")
        else:
            print(f"   ❌ {key}: Missing")
            validation_passed = False
    
    # Validate ML results
    ml_results = results_dict.get('ml', {})
    if ml_results:
        if ml_results.get('regression'):
            print(f"   ✓ Regression models: {len(ml_results['regression'])} algorithms")
        if ml_results.get('classification'):
            print(f"   ✓ Classification models: {len(ml_results['classification'])} algorithms")
    
    if validation_passed:
        print(f"   🎯 All validations passed!")
    else:
        print(f"   ⚠️  Some validations failed!")
    
    return validation_passed

if __name__ == "__main__":
    # Test utility functions
    print("🔧 UTILITY FUNCTIONS MODULE")
    print("This module provides support functions for the analysis pipeline.")
    
    # Test directory setup
    setup_directories()
    
    # Test metadata creation
    metadata = create_project_metadata()
    print(f"\n📋 Project Metadata:")
    for key, value in metadata.items():
        if isinstance(value, list):
            print(f"   • {key}: {len(value)} items")
        else:
            print(f"   • {key}: {value}")
    
    print(f"\n✅ Utility functions tested successfully!")
