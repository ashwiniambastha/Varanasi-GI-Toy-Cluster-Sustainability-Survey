# =============================================================================
# VISUALIZATIONS MODULE
# =============================================================================
"""
This module creates comprehensive visualizations and professional dashboards
for the Varanasi GI Toy Survey Analysis project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_executive_dashboard(df, ml_results, cluster_results, impact_results, output_dir='visualizations'):
    """
    Create comprehensive executive dashboard
    
    Args:
        df (pd.DataFrame): Processed dataframe
        ml_results (dict): ML analysis results
        cluster_results (dict): Clustering results
        impact_results (dict): Business impact results
        output_dir (str): Output directory
        
    Returns:
        str: Path to saved dashboard
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("📊 CREATING EXECUTIVE DASHBOARD:")
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('Varanasi GI Toy Cluster - Executive Dashboard', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # 1. Key Metrics Panel (Top row, spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    create_key_metrics_panel(ax1, df, ml_results, impact_results)
    
    # 2. Income Distribution (Top row, right)
    ax2 = fig.add_subplot(gs[0, 2:])
    create_income_distribution_plot(ax2, df)
    
    # 3. Model Performance Comparison (Second row, left)
    ax3 = fig.add_subplot(gs[1, :2])
    create_model_performance_plot(ax3, ml_results)
    
    # 4. Business Impact Chart (Second row, right)
    ax4 = fig.add_subplot(gs[1, 2:])
    create_business_impact_chart(ax4, impact_results)
    
    # 5. Cluster Analysis (Third row, left)
    ax5 = fig.add_subplot(gs[2, :2])
    create_cluster_overview_plot(ax5, df, cluster_results)
    
    # 6. Feature Importance (Third row, right)
    ax6 = fig.add_subplot(gs[2, 2:])
    create_feature_importance_plot(ax6, ml_results)
    
    # 7. Training vs GI Impact (Fourth row, left)
    ax7 = fig.add_subplot(gs[3, :2])
    create_intervention_comparison_plot(ax7, df)
    
    # 8. Geographic/Demographic Insights (Fourth row, right)
    ax8 = fig.add_subplot(gs[3, 2:])
    create_demographic_insights_plot(ax8, df)
    
    # Save dashboard
    dashboard_path = os.path.join(output_dir, 'executive_dashboard.png')
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"   ✅ Executive dashboard saved: {dashboard_path}")
    return dashboard_path

def create_key_metrics_panel(ax, df, ml_results, impact_results):
    """Create key metrics summary panel"""
    ax.axis('off')
    
    # Calculate key metrics
    total_households = len(df)
    avg_income = df['Monthly_Income'].mean()
    training_rate = df['Training_Access'].mean() * 100
    gi_rate = df['Is_GI_Beneficiary'].mean() * 100
    
    # Get best model performance
    best_r2 = 0
    if ml_results.get('regression'):
        best_r2 = max([v['test_r2'] for v in ml_results['regression'].values()])
    
    # Economic impact
    economic_impact = impact_results.get('economic_impact', {}).get('total_annual_impact_lakh', 0)
    
    # Create metrics boxes
    metrics = [
        ("Households\nSurveyed", f"{total_households}", "blue"),
        ("Average Monthly\nIncome", f"₹{avg_income:.0f}", "green"),
        ("Training\nCoverage", f"{training_rate:.0f}%", "orange"),
        ("GI Registration\nRate", f"{gi_rate:.0f}%", "purple"),
        ("Model\nAccuracy", f"R² = {best_r2:.3f}", "red"),
        ("Economic Impact\nPotential", f"₹{economic_impact:.1f}L", "gold")
    ]
    
    # Draw metrics boxes
    box_width = 1.5
    box_height = 0.8
    spacing = 0.1
    
    for i, (label, value, color) in enumerate(metrics):
        x = i * (box_width + spacing)
        
        # Background box
        rect = plt.Rectangle((x, 0), box_width, box_height, 
                           facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        
        # Value text
        ax.text(x + box_width/2, 0.55, value, ha='center', va='center',
               fontsize=14, fontweight='bold', color=color)
        
        # Label text
        ax.text(x + box_width/2, 0.25, label, ha='center', va='center',
               fontsize=10, wrap=True)
    
    ax.set_xlim(-0.2, len(metrics) * (box_width + spacing))
    ax.set_ylim(-0.1, 0.9)
    ax.set_title('Key Performance Indicators', fontsize=14, fontweight='bold', pad=20)

def create_income_distribution_plot(ax, df):
    """Create income distribution plot with annotations"""
    if 'Monthly_Income' not in df.columns:
        ax.text(0.5, 0.5, 'Income data not available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Create histogram
    n, bins, patches = ax.hist(df['Monthly_Income'], bins=20, alpha=0.7, 
                              color='skyblue', edgecolor='black')
    
    # Add mean and median lines
    mean_income = df['Monthly_Income'].mean()
    median_income = df['Monthly_Income'].median()
    
    ax.axvline(mean_income, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: ₹{mean_income:.0f}')
    ax.axvline(median_income, color='green', linestyle='--', linewidth=2, 
               label=f'Median: ₹{median_income:.0f}')
    
    # Color bars based on income levels
    for i, patch in enumerate(patches):
        if bins[i] < 6000:
            patch.set_facecolor('lightcoral')
        elif bins[i] > 10000:
            patch.set_facecolor('lightgreen')
    
    ax.set_title('Monthly Income Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Monthly Income (₹)')
    ax.set_ylabel('Number of Households')
    ax.legend()
    
    # Add statistics text box
    stats_text = f'μ = ₹{mean_income:.0f}\nσ = ₹{df["Monthly_Income"].std():.0f}\nRange: ₹{df["Monthly_Income"].min():.0f} - ₹{df["Monthly_Income"].max():.0f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def create_model_performance_plot(ax, ml_results):
    """Create ML model performance comparison"""
    reg_results = ml_results.get('regression', {})
    
    if not reg_results:
        ax.text(0.5, 0.5, 'ML results not available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Extract model names and R² scores
    models = list(reg_results.keys())
    r2_scores = [reg_results[model]['test_r2'] for model in models]
    cv_scores = [reg_results[model]['cv_mean'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, r2_scores, width, label='Test R²', 
                  color='lightblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, cv_scores, width, label='CV R²', 
                  color='lightcoral', alpha=0.8)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_title('Machine Learning Model Performance', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Score')
    ax.set_xlabel('Models')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    # Highlight best model
    best_idx = np.argmax(r2_scores)
    bars1[best_idx].set_color('gold')
    bars1[best_idx].set_edgecolor('orange')
    bars1[best_idx].set_linewidth(2)

def create_business_impact_chart(ax, impact_results):
    """Create business impact visualization"""
    training_impact = impact_results.get('training_impact', {}).get('income_difference', 0)
    gi_impact = impact_results.get('gi_impact', {}).get('income_difference', 0)
    economic_impact = impact_results.get('economic_impact', {}).get('total_annual_impact_lakh', 0)
    
    # Create impact comparison
    categories = ['Training\nProgram', 'GI\nRegistration']
    monthly_impacts = [training_impact, gi_impact]
    colors = ['lightgreen', 'lightblue']
    
    bars = ax.bar(categories, monthly_impacts, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar, value in zip(bars, monthly_impacts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
               f'₹{value:.0f}/month', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('Policy Intervention Impact', fontsize=12, fontweight='bold')
    ax.set_ylabel('Monthly Income Increase (₹)')
    
    # Add economic impact annotation
    ax.text(0.5, 0.95, f'Total Annual Potential: ₹{economic_impact:.1f} Lakh', 
           transform=ax.transAxes, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
           fontsize=11, fontweight='bold')

def create_cluster_overview_plot(ax, df, cluster_results):
    """Create cluster analysis overview"""
    if 'Cluster' not in df.columns:
        ax.text(0.5, 0.5, 'Cluster analysis not available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Cluster sizes
    cluster_counts = df['Cluster'].value_counts().sort_index()
    colors = sns.color_palette("Set3", len(cluster_counts))
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(cluster_counts.values, 
                                     labels=[f'Cluster {i}' for i in cluster_counts.index],
                                     colors=colors, autopct='%1.1f%%', startangle=90)
    
    ax.set_title('Artisan Segmentation', fontsize=12, fontweight='bold')
    
    # Add cluster insights if available
    insights = cluster_results.get('cluster_profiles', {}).get('insights', {})
    if insights:
        legend_elements = []
        for cluster_id, insight in insights.items():
            label = insight.get('cluster_label', f'Cluster {cluster_id}')
            legend_elements.append(Patch(facecolor=colors[cluster_id], label=label))
        
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

def create_feature_importance_plot(ax, ml_results):
    """Create feature importance visualization"""
    feature_importance = ml_results.get('feature_importance')
    
    if feature_importance is None or feature_importance.empty:
        ax.text(0.5, 0.5, 'Feature importance not available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Select top 10 features
    top_features = feature_importance.head(10)
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(top_features)), top_features['importance'], 
                  color='lightcoral', alpha=0.8)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance')
    ax.set_title('Top Feature Importance', fontsize=12, fontweight='bold')
    
    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
               f'{importance:.3f}', ha='left', va='center', fontsize=9)
    
    # Invert y-axis to show highest importance at top
    ax.invert_yaxis()

def create_intervention_comparison_plot(ax, df):
    """Create training vs GI intervention comparison"""
    if not all(col in df.columns for col in ['Training_Access', 'Is_GI_Beneficiary', 'Monthly_Income']):
        ax.text(0.5, 0.5, 'Intervention data not available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Create 2x2 comparison matrix
    categories = ['No Training\nNo GI', 'Training\nNo GI', 'No Training\nWith GI', 'Training\nWith GI']
    
    # Calculate mean income for each combination
    income_means = []
    counts = []
    
    for training in [0, 1]:
        for gi in [0, 1]:
            subset = df[(df['Training_Access'] == training) & (df['Is_GI_Beneficiary'] == gi)]
            if len(subset) > 0:
                income_means.append(subset['Monthly_Income'].mean())
                counts.append(len(subset))
            else:
                income_means.append(0)
                counts.append(0)
    
    # Create bar plot
    colors = ['lightcoral', 'lightblue', 'lightgreen', 'gold']
    bars = ax.bar(categories, income_means, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels and counts
    for bar, income, count in zip(bars, income_means, counts):
        if income > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                   f'₹{income:.0f}\n(n={count})', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
    
    ax.set_title('Training × GI Registration Impact', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Monthly Income (₹)')
    ax.tick_params(axis='x', rotation=45)

def create_demographic_insights_plot(ax, df):
    """Create demographic insights visualization"""
    if 'Primary_Earner_Gender' not in df.columns:
        ax.text(0.5, 0.5, 'Demographic data not available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Gender-based income analysis
    gender_income = df.groupby('Primary_Earner_Gender')['Monthly_Income'].agg(['mean', 'count'])
    
    # Create grouped bar plot
    x = np.arange(len(gender_income.index))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, gender_income['mean'], width, 
                  label='Avg Income', color='lightblue', alpha=0.8)
    
    # Create secondary y-axis for counts
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, gender_income['count'], width, 
                   label='Count', color='lightcoral', alpha=0.8)
    
    # Add value labels
    for bar, value in zip(bars1, gender_income['mean']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
               f'₹{value:.0f}', ha='center', va='bottom', fontsize=9)
    
    for bar, value in zip(bars2, gender_income['count']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{int(value)}', ha='center', va='bottom', fontsize=9)
    
    ax.set_title('Gender-based Analysis', fontsize=12, fontweight='bold')
    ax.set_xlabel('Primary Earner Gender')
    ax.set_ylabel('Average Monthly Income (₹)', color='blue')
    ax2.set_ylabel('Number of Households', color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(gender_income.index)
    
    # Add legends
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

def create_correlation_heatmap(df, output_dir='visualizations'):
    """Create correlation heatmap for key variables"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔥 Creating correlation heatmap...")
    
    # Select key numerical variables
    key_vars = [
        'Monthly_Income', 'Family_Size', 'Training_Access', 'Is_GI_Beneficiary',
        'Satisfaction_Score'
    ]
    
    # Add engineered features if available
    engineered_vars = [
        'Income_Per_Member', 'Total_Resource_Score', 'Artisan_Success_Index',
        'Economic_Vulnerability', 'GI_Training_Synergy'
    ]
    
    available_vars = [var for var in key_vars + engineered_vars if var in df.columns]
    
    if len(available_vars) < 3:
        print("   ⚠️  Insufficient variables for correlation analysis")
        return None
    
    # Create correlation matrix
    corr_matrix = df[available_vars].corr()
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
    
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save plot
    heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Correlation heatmap saved: {heatmap_path}")
    return heatmap_path

def create_trend_analysis_plots(df, output_dir='visualizations'):
    """Create trend analysis and distribution plots"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("📈 Creating trend analysis plots...")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Trend Analysis and Distributions', fontsize=16, fontweight='bold')
    
    # 1. Income vs Family Size with trend line
    if all(col in df.columns for col in ['Monthly_Income', 'Family_Size']):
        axes[0,0].scatter(df['Family_Size'], df['Monthly_Income'], alpha=0.6, color='blue')
        
        # Add trend line
        z = np.polyfit(df['Family_Size'].fillna(df['Family_Size'].mean()), 
                      df['Monthly_Income'].fillna(df['Monthly_Income'].mean()), 1)
        p = np.poly1d(z)
        axes[0,0].plot(df['Family_Size'], p(df['Family_Size']), "r--", alpha=0.8)
        
        # Calculate correlation
        correlation = df['Monthly_Income'].corr(df['Family_Size'])
        axes[0,0].set_title(f'Income vs Family Size (r={correlation:.3f})')
        axes[0,0].set_xlabel('Family Size')
        axes[0,0].set_ylabel('Monthly Income (₹)')
    
    # 2. Satisfaction distribution by training
    if all(col in df.columns for col in ['Satisfaction_Score', 'Training_Access']):
        training_labels = {0: 'No Training', 1: 'Has Training'}
        df_plot = df.copy()
        df_plot['Training_Label'] = df_plot['Training_Access'].map(training_labels)
        
        sns.boxplot(data=df_plot, x='Training_Label', y='Satisfaction_Score', ax=axes[0,1])
        axes[0,1].set_title('Satisfaction by Training Access')
        axes[0,1].set_xlabel('Training Status')
        axes[0,1].set_ylabel('Satisfaction Score')
    
    # 3. Income distribution by GI status
    if all(col in df.columns for col in ['Monthly_Income', 'Is_GI_Beneficiary']):
        gi_labels = {0: 'Non-Beneficiary', 1: 'GI Beneficiary'}
        df_plot = df.copy()
        df_plot['GI_Label'] = df_plot['Is_GI_Beneficiary'].map(gi_labels)
        
        sns.violinplot(data=df_plot, x='GI_Label', y='Monthly_Income', ax=axes[1,0])
        axes[1,0].set_title('Income Distribution by GI Status')
        axes[1,0].set_xlabel('GI Beneficiary Status')
        axes[1,0].set_ylabel('Monthly Income (₹)')
    
    # 4. Resource access impact
    if 'Raw_Material_Access' in df.columns and 'Monthly_Income' in df.columns:
        resource_order = ['Difficult', 'Moderate', 'Easy']
        df_resource = df[df['Raw_Material_Access'].isin(resource_order)]
        
        sns.barplot(data=df_resource, x='Raw_Material_Access', y='Monthly_Income', 
                   order=resource_order, ax=axes[1,1])
        axes[1,1].set_title('Income by Resource Access')
        axes[1,1].set_xlabel('Raw Material Access')
        axes[1,1].set_ylabel('Average Monthly Income (₹)')
        
        # Add value labels
        for i, bar in enumerate(axes[1,1].patches):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                          f'₹{bar.get_height():.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    trends_path = os.path.join(output_dir, 'trend_analysis_plots.png')
    plt.savefig(trends_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Trend analysis plots saved: {trends_path}")
    return trends_path

def create_roi_visualization(impact_results, output_dir='visualizations'):
    """Create ROI and economic impact visualization"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("💰 Creating ROI visualization...")
    
    roi_analysis = impact_results.get('roi_analysis', {})
    if not roi_analysis:
        print("   ⚠️  ROI analysis not available")
        return None
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Return on Investment Analysis', fontsize=16, fontweight='bold')
    
    # 1. Cost-Benefit Comparison
    costs = roi_analysis.get('costs', {})
    benefits = roi_analysis.get('benefits', {})
    
    categories = ['Training\nProgram', 'GI Registration\nDrive', 'Total\nBenefit']
    values = [
        costs.get('training_program_cost_lakh', 0),
        costs.get('gi_registration_cost_lakh', 0),
        benefits.get('total_benefit_lakh', 0)
    ]
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    
    bars = ax1.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'₹{value:.1f}L', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_title('Cost vs Benefit Analysis')
    ax1.set_ylabel('Amount (₹ Lakh)')
    
    # 2. ROI Metrics
    roi_metrics = roi_analysis.get('roi_metrics', {})
    roi_percentage = roi_metrics.get('roi_percentage', 0)
    payback_period = roi_metrics.get('payback_period_years', 0)
    
    # ROI gauge chart
    theta = np.linspace(0, np.pi, 100)
    roi_normalized = min(roi_percentage / 500, 1.0)  # Normalize to 0-1 for 0-500% ROI
    
    ax2.plot(theta, np.ones_like(theta), 'k-', linewidth=3)
    ax2.fill_between(theta[:int(roi_normalized*100)], 0, 1, alpha=0.7, color='green')
    ax2.fill_between(theta[int(roi_normalized*100):], 0, 1, alpha=0.3, color='red')
    
    ax2.set_xlim(0, np.pi)
    ax2.set_ylim(0, 1.2)
    ax2.set_title(f'ROI: {roi_percentage:.0f}%')
    ax2.axis('off')
    
    # Add ROI value text
    ax2.text(np.pi/2, 0.5, f'{roi_percentage:.0f}%\nROI', ha='center', va='center',
            fontsize=20, fontweight='bold', color='darkgreen' if roi_percentage > 0 else 'darkred')
    
    # 3. Payback Period
    years = np.arange(1, 6)
    cumulative_benefit = [benefits.get('total_benefit_lakh', 0) * year for year in years]
    total_cost = costs.get('total_cost_lakh', 0)
    
    ax3.plot(years, cumulative_benefit, 'g-', linewidth=3, marker='o', label='Cumulative Benefit')
    ax3.axhline(y=total_cost, color='red', linestyle='--', linewidth=2, label=f'Total Cost (₹{total_cost:.1f}L)')
    
    if payback_period < 5:
        ax3.axvline(x=payback_period, color='orange', linestyle=':', linewidth=2, 
                   label=f'Payback Period ({payback_period:.1f} years)')
    
    ax3.set_title('Payback Period Analysis')
    ax3.set_xlabel('Years')
    ax3.set_ylabel('Amount (₹ Lakh)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Impact Distribution
    training_benefit = benefits.get('training_benefit_lakh', 0)
    gi_benefit = benefits.get('gi_benefit_lakh', 0)
    
    if training_benefit > 0 or gi_benefit > 0:
        labels = ['Training\nPrograms', 'GI Registration\nDrive']
        sizes = [training_benefit, gi_benefit]
        colors = ['lightblue', 'lightcoral']
        
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax4.set_title('Benefit Distribution')
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    plt.tight_layout()
    
    # Save plot
    roi_path = os.path.join(output_dir, 'roi_analysis.png')
    plt.savefig(roi_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ ROI visualization saved: {roi_path}")
    return roi_path

def create_comprehensive_dashboard(df, ml_results, cluster_results, impact_results):
    """
    Main function to create all visualizations and dashboards
    
    Args:
        df (pd.DataFrame): Processed dataframe
        ml_results (dict): ML analysis results
        cluster_results (dict): Clustering results  
        impact_results (dict): Business impact results
        
    Returns:
        dict: Dictionary of created visualization files
    """
    print(f"📊 COMPREHENSIVE VISUALIZATION DASHBOARD CREATION")
    print("=" * 60)
    
    created_files = {}
    
    try:
        # 1. Executive Dashboard
        executive_dashboard = create_executive_dashboard(df, ml_results, cluster_results, impact_results)
        created_files['executive_dashboard'] = executive_dashboard
        
        # 2. Correlation Heatmap
        correlation_heatmap = create_correlation_heatmap(df)
        if correlation_heatmap:
            created_files['correlation_heatmap'] = correlation_heatmap
        
        # 3. Trend Analysis Plots
        trend_plots = create_trend_analysis_plots(df)
        if trend_plots:
            created_files['trend_analysis'] = trend_plots
        
        # 4. ROI Visualization
        roi_viz = create_roi_visualization(impact_results)
        if roi_viz:
            created_files['roi_analysis'] = roi_viz
        
        print(f"\n📊 VISUALIZATION SUMMARY:")
        print(f"   • Total visualizations created: {len(created_files)}")
        for viz_type, file_path in created_files.items():
            print(f"   • {viz_type}: {file_path}")
        
    except Exception as e:
        print(f"   ❌ Error creating visualizations: {str(e)}")
    
    return created_files

if __name__ == "__main__":
    print("📊 ADVANCED VISUALIZATIONS MODULE")
    print("This module creates professional dashboards and visualizations.")
    print("Run main.py to execute the complete analysis pipeline.")
