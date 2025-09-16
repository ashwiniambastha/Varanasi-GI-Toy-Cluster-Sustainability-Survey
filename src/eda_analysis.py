# =============================================================================
# EXPLORATORY DATA ANALYSIS MODULE
# =============================================================================
"""
This module performs comprehensive exploratory data analysis
with statistical insights and advanced visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def perform_univariate_analysis(df):
    """
    Perform univariate analysis on key variables
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Univariate analysis results
    """
    print("📊 UNIVARIATE ANALYSIS:")
    
    results = {}
    
    # Analyze numerical variables
    numerical_vars = ['Monthly_Income', 'Family_Size', 'Satisfaction_Score']
    
    for var in numerical_vars:
        if var in df.columns:
            stats_summary = df[var].describe()
            
            results[var] = {
                'mean': stats_summary['mean'],
                'median': stats_summary['50%'],
                'std': stats_summary['std'],
                'min': stats_summary['min'],
                'max': stats_summary['max'],
                'skewness': df[var].skew(),
                'kurtosis': df[var].kurtosis()
            }
            
            print(f"   • {var}:")
            print(f"     - Mean: {stats_summary['mean']:.2f}")
            print(f"     - Median: {stats_summary['50%']:.2f}")
            print(f"     - Std Dev: {stats_summary['std']:.2f}")
            print(f"     - Skewness: {df[var].skew():.3f}")
    
    # Analyze categorical variables
    categorical_vars = ['Primary_Earner_Gender', 'Raw_Material_Access']
    
    for var in categorical_vars:
        if var in df.columns:
            value_counts = df[var].value_counts(normalize=True)
            results[var] = value_counts.to_dict()
            
            print(f"   • {var} Distribution:")
            for category, proportion in value_counts.items():
                print(f"     - {category}: {proportion:.1%}")
    
    return results

def perform_bivariate_analysis(df):
    """
    Perform bivariate analysis focusing on relationships with income
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Bivariate analysis results
    """
    print(f"\n🔗 BIVARIATE ANALYSIS:")
    
    results = {}
    
    if 'Monthly_Income' not in df.columns:
        print("   ❌ Monthly_Income column not found")
        return results
    
    # Correlation analysis
    numerical_vars = ['Family_Size', 'Satisfaction_Score', 'Training_Access']
    correlations = {}
    
    for var in numerical_vars:
        if var in df.columns:
            corr = df['Monthly_Income'].corr(df[var])
            correlations[var] = corr
            
            # Interpret correlation strength
            if abs(corr) > 0.7:
                strength = "Strong"
            elif abs(corr) > 0.3:
                strength = "Moderate"
            else:
                strength = "Weak"
            
            print(f"   • Income-{var} correlation: {corr:.3f} ({strength})")
    
    results['correlations'] = correlations
    
    # Categorical analysis
    categorical_vars = ['Primary_Earner_Gender', 'Is_GI_Beneficiary', 'Training_Access']
    
    for var in categorical_vars:
        if var in df.columns:
            group_analysis = df.groupby(var)['Monthly_Income'].agg(['mean', 'median', 'std', 'count'])
            results[f'{var}_income_analysis'] = group_analysis.to_dict('index')
            
            print(f"   • Income by {var}:")
            for category, stats in group_analysis.iterrows():
                print(f"     - {category}: Mean=₹{stats['mean']:.0f}, n={stats['count']}")
    
    # Statistical significance tests
    if 'Primary_Earner_Gender' in df.columns:
        male_income = df[df['Primary_Earner_Gender'] == 'Male']['Monthly_Income'].dropna()
        female_income = df[df['Primary_Earner_Gender'] == 'Female']['Monthly_Income'].dropna()
        
        if len(male_income) > 0 and len(female_income) > 0:
            t_stat, p_value = ttest_ind(male_income, female_income)
            results['gender_ttest'] = {'t_statistic': t_stat, 'p_value': p_value}
            
            significance = "Significant" if p_value < 0.05 else "Not significant"
            print(f"   • Gender income difference: {significance} (p={p_value:.3f})")
    
    return results

def analyze_missing_data(df):
    """
    Analyze missing data patterns
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Missing data analysis results
    """
    print(f"\n❓ MISSING DATA ANALYSIS:")
    
    missing_analysis = {}
    
    # Calculate missing percentages
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    missing_cols = missing_counts[missing_counts > 0]
    
    if len(missing_cols) == 0:
        print("   ✅ No missing data found!")
        return {'missing_columns': 0, 'total_missing': 0}
    
    print(f"   • Columns with missing data: {len(missing_cols)}")
    print(f"   • Total missing values: {missing_counts.sum()}")
    
    for col in missing_cols.index:
        count = missing_counts[col]
        percentage = missing_percentages[col]
        print(f"     - {col}: {count} values ({percentage:.1f}%)")
        
        missing_analysis[col] = {
            'count': count,
            'percentage': percentage
        }
    
    # Pattern analysis
    if len(missing_cols) > 1:
        # Check for patterns in missing data
        missing_patterns = df[missing_cols.index].isnull().value_counts()
        print(f"   • Missing data patterns: {len(missing_patterns)}")
    
    missing_analysis['total_columns_with_missing'] = len(missing_cols)
    missing_analysis['total_missing_values'] = missing_counts.sum()
    
    return missing_analysis

def detect_outliers(df):
    """
    Detect outliers using IQR method
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Outlier analysis results
    """
    print(f"\n📊 OUTLIER DETECTION:")
    
    outlier_analysis = {}
    numerical_vars = df.select_dtypes(include=[np.number]).columns
    
    for var in numerical_vars:
        Q1 = df[var].quantile(0.25)
        Q3 = df[var].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[var] < lower_bound) | (df[var] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(df)) * 100
        
        if outlier_count > 0:
            print(f"   • {var}: {outlier_count} outliers ({outlier_percentage:.1f}%)")
            
            outlier_analysis[var] = {
                'count': outlier_count,
                'percentage': outlier_percentage,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_values': outliers[var].tolist()
            }
        else:
            print(f"   • {var}: No outliers detected")
            outlier_analysis[var] = {'count': 0, 'percentage': 0}
    
    return outlier_analysis

def create_eda_visualizations(df, output_dir='visualizations'):
    """
    Create comprehensive EDA visualizations
    
    Args:
        df (pd.DataFrame): Input dataframe
        output_dir (str): Output directory for visualizations
        
    Returns:
        list: List of created visualization files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📊 CREATING EDA VISUALIZATIONS:")
    
    created_files = []
    
    # Create comprehensive EDA plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Exploratory Data Analysis', fontsize=16, fontweight='bold')
    
    # 1. Income distribution
    if 'Monthly_Income' in df.columns:
        df['Monthly_Income'].hist(bins=20, alpha=0.7, color='skyblue', edgecolor='black', ax=axes[0,0])
        axes[0,0].axvline(df['Monthly_Income'].mean(), color='red', linestyle='--', 
                         label=f'Mean: ₹{df["Monthly_Income"].mean():.0f}')
        axes[0,0].axvline(df['Monthly_Income'].median(), color='green', linestyle='--', 
                         label=f'Median: ₹{df["Monthly_Income"].median():.0f}')
        axes[0,0].set_title('Monthly Income Distribution')
        axes[0,0].set_xlabel('Monthly Income (₹)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].legend()
    
    # 2. Income vs Family Size scatter
    if all(col in df.columns for col in ['Monthly_Income', 'Family_Size']):
        # Clean data for plotting
        clean_data = df[['Monthly_Income', 'Family_Size']].dropna()
        if len(clean_data) > 1:
            axes[0,1].scatter(clean_data['Family_Size'], clean_data['Monthly_Income'], alpha=0.6, color='coral')
            # Add trend line
            try:
                z = np.polyfit(clean_data['Family_Size'], clean_data['Monthly_Income'], 1)
                p = np.poly1d(z)
                axes[0,1].plot(clean_data['Family_Size'], p(clean_data['Family_Size']), "r--", alpha=0.8)
            except:
                pass  # Skip trend line if it fails
            axes[0,1].set_title('Income vs Family Size')
            axes[0,1].set_xlabel('Family Size')
            axes[0,1].set_ylabel('Monthly Income (₹)')
    
    # 3. Training access vs Income boxplot
    if all(col in df.columns for col in ['Monthly_Income', 'Training_Access']):
        training_labels = {0: 'No Training', 1: 'Has Training'}
        df_plot = df.copy()
        df_plot['Training_Label'] = df_plot['Training_Access'].map(training_labels)
        df_plot = df_plot.dropna(subset=['Training_Label', 'Monthly_Income'])  # Clean data
        
        if len(df_plot) > 0:
            sns.boxplot(data=df_plot, x='Training_Label', y='Monthly_Income', ax=axes[0,2])
            axes[0,2].set_title('Income by Training Access')
            axes[0,2].set_ylabel('Monthly Income (₹)')
    
    # 4. Gender distribution pie chart
    if 'Primary_Earner_Gender' in df.columns:
        gender_counts = df['Primary_Earner_Gender'].value_counts()
        axes[1,0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
                     colors=['lightblue', 'lightpink'])
        axes[1,0].set_title('Primary Earner Gender Distribution')
    
    # 5. GI Beneficiary vs Satisfaction
    if all(col in df.columns for col in ['Is_GI_Beneficiary', 'Satisfaction_Score']):
        gi_labels = {0: 'Non-Beneficiary', 1: 'GI Beneficiary'}
        df_plot = df.copy()
        df_plot['GI_Label'] = df_plot['Is_GI_Beneficiary'].map(gi_labels)
        df_plot = df_plot.dropna(subset=['GI_Label', 'Satisfaction_Score'])  # Clean data
        
        if len(df_plot) > 0:
            sns.boxplot(data=df_plot, x='GI_Label', y='Satisfaction_Score', ax=axes[1,1])
            axes[1,1].set_title('Satisfaction by GI Beneficiary Status')
            axes[1,1].set_ylabel('Satisfaction Score')
    
    # 6. Correlation heatmap
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[1,2], fmt='.2f')
        axes[1,2].set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    
    # Save the plot
    eda_file = os.path.join(output_dir, 'comprehensive_eda_analysis.png')
    plt.savefig(eda_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    created_files.append(eda_file)
    print(f"   ✓ EDA visualization saved: {eda_file}")
    
    return created_files

def perform_advanced_eda(df):
    """
    Main function to perform comprehensive EDA
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Complete EDA results
    """
    print(f"📈 ADVANCED EXPLORATORY DATA ANALYSIS")
    print("=" * 50)
    
    # Perform all EDA components
    univariate_results = perform_univariate_analysis(df)
    bivariate_results = perform_bivariate_analysis(df)
    missing_data_results = analyze_missing_data(df)
    outlier_results = detect_outliers(df)
    
    # Create visualizations
    visualization_files = create_eda_visualizations(df)
    
    # Compile results
    eda_results = {
        'dataset_overview': {
            'shape': df.shape,
            'numerical_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns)
        },
        'univariate_analysis': univariate_results,
        'bivariate_analysis': bivariate_results,
        'missing_data_analysis': missing_data_results,
        'outlier_analysis': outlier_results,
        'visualization_files': visualization_files
    }
    
    # Print summary insights
    print(f"\n🎯 KEY EDA INSIGHTS:")
    
    if 'correlations' in bivariate_results:
        strongest_corr = max(bivariate_results['correlations'].items(), 
                           key=lambda x: abs(x[1]))
        print(f"   • Strongest income correlation: {strongest_corr[0]} (r={strongest_corr[1]:.3f})")
    
    if missing_data_results['total_missing_values'] > 0:
        print(f"   • Missing data requires attention: {missing_data_results['total_missing_values']} values")
    else:
        print(f"   • Data quality: Excellent (no missing values)")
    
    outlier_counts = [result['count'] for result in outlier_results.values()]
    total_outliers = sum(outlier_counts)
    if total_outliers > 0:
        print(f"   • Outliers detected: {total_outliers} across multiple variables")
    
    return eda_results

if __name__ == "__main__":
    print("📊 EXPLORATORY DATA ANALYSIS MODULE")
    print("This module performs comprehensive EDA with statistical insights.")
    print("Run main.py to execute the complete analysis pipeline.")
