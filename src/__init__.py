# =============================================================================
# VARANASI TOY SURVEY ANALYSIS - PACKAGE INITIALIZATION
# =============================================================================
"""
Varanasi GI Toy Cluster Survey Analysis Package

This package provides comprehensive data science tools for analyzing
socio-economic factors affecting artisan households in Varanasi's
Geographic Indication (GI) toy manufacturing cluster.

Modules:
- data_loader: Data loading and validation utilities
- eda_analysis: Comprehensive exploratory data analysis
- feature_engineering: Advanced feature creation and transformation
- preprocessing: Data preprocessing and cleaning pipeline
- ml_models: Machine learning algorithms including XGBoost
- clustering_analysis: K-means clustering and segmentation
- business_impact: Economic impact analysis and ROI calculation
- visualizations: Advanced plotting and dashboard creation
- utils: Utility functions and project management tools
"""

__version__ = "1.0.0"
__author__ = "Data Science Team"
__email__ = "datascience@example.com"

# Import main modules for easy access
from .data_loader import load_and_explore_data
from .eda_analysis import perform_advanced_eda
from .feature_engineering import engineer_features
from .ml_models import compare_ml_algorithms
from .utils import setup_directories, save_results

# Package metadata
__all__ = [
    'load_and_explore_data',
    'perform_advanced_eda', 
    'engineer_features',
    'compare_ml_algorithms',
    'setup_directories',
    'save_results'
]

# Version info
version_info = (1, 0, 0)

# Package description
DESCRIPTION = """
A comprehensive machine learning analysis of artisan households in Varanasi's 
GI toy manufacturing cluster, featuring advanced feature engineering, 
multiple ML algorithms, and business impact quantification.
"""

# Key features
KEY_FEATURES = [
    "Advanced feature engineering (20+ engineered features)",
    "Multiple ML algorithms including XGBoost",
    "Statistical analysis and hypothesis testing", 
    "K-means clustering for artisan segmentation",
    "Economic impact quantification and ROI analysis",
    "Professional visualizations and dashboards",
    "Comprehensive documentation and reporting"
]

def get_package_info():
    """
    Get package information
    
    Returns:
        dict: Package information
    """
    return {
        'name': 'varanasi-toy-survey-analysis',
        'version': __version__,
        'author': __author__,
        'description': DESCRIPTION.strip(),
        'features': KEY_FEATURES
    }

if __name__ == "__main__":
    info = get_package_info()
    print(f"📦 Package: {info['name']}")
    print(f"📊 Version: {info['version']}")
    print(f"👨‍💻 Author: {info['author']}")
    print(f"\n📋 Description:")
    print(f"   {info['description']}")
    print(f"\n🎯 Key Features:")
    for i, feature in enumerate(info['features'], 1):
        print(f"   {i}. {feature}")
