# =============================================================================
# CONCLUSIONS AND RECOMMENDATIONS MODULE
# =============================================================================
"""
This module draws final conclusions from the analysis and provides
actionable recommendations for stakeholders
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def analyze_model_performance(ml_results):
    """
    Analyze and conclude on model performance
    
    Args:
        ml_results (dict): ML analysis results
        
    Returns:
        dict: Model performance conclusions
    """
    print("🎯 MODEL PERFORMANCE CONCLUSIONS:")
    print("=" * 45)
    
    conclusions = {
        'best_performers': {},
        'insights': [],
        'recommendations': []
    }
    
    # Regression analysis
    reg_results = ml_results.get('regression', {})
    if reg_results:
        best_reg = max(reg_results.items(), key=lambda x: x[1]['test_r2'])
        worst_reg = min(reg_results.items(), key=lambda x: x[1]['test_r2'])
        
        conclusions['best_performers']['regression'] = {
            'model': best_reg[0],
            'r2_score': best_reg[1]['test_r2'],
            'rmse': best_reg[1]['test_rmse']
        }
        
        print(f"🏆 REGRESSION MODELS:")
        print(f"   • Best Performer: {best_reg[0]} (R² = {best_reg[1]['test_r2']:.3f})")
        print(f"   • RMSE: ₹{best_reg[1]['test_rmse']:.0f}")
        
        # Performance spread analysis
        r2_scores = [result['test_r2'] for result in reg_results.values()]
        performance_spread = max(r2_scores) - min(r2_scores)
        
        if performance_spread < 0.1:
            conclusions['insights'].append("Regression models show similar performance - feature engineering successful")
        else:
            conclusions['insights'].append(f"Significant performance variation ({performance_spread:.3f}) suggests model selection is crucial")
    
    # Classification analysis
    clf_results = ml_results.get('classification', {})
    if clf_results:
        best_clf = max(clf_results.items(), key=lambda x: x[1]['test_acc'])
        
        conclusions['best_performers']['classification'] = {
            'model': best_clf[0],
            'accuracy': best_clf[1]['test_acc']
        }
        
        print(f"\n🏆 CLASSIFICATION MODELS:")
        print(f"   • Best Performer: {best_clf[0]} (Accuracy = {best_clf[1]['test_acc']:.3f})")
        
        # XGBoost specific insights
        if 'XGBoost' in clf_results and 'XGBoost' in reg_results:
            xgb_reg_r2 = reg_results['XGBoost']['test_r2']
            xgb_clf_acc = clf_results['XGBoost']['test_acc']
            
            print(f"\n📊 XGBOOST PERFORMANCE INSIGHTS:")
            print(f"   • Regression R²: {xgb_reg_r2:.3f}")
            print(f"   • Classification Accuracy: {xgb_clf_acc:.3f}")
            
            if xgb_reg_r2 >= 0.7 or xgb_clf_acc >= 0.8:
                conclusions['insights'].append("XGBoost demonstrates strong performance across both tasks")
            
            conclusions['recommendations'].extend([
                "XGBoost shows promise for production deployment",
                "Consider ensemble methods for improved robustness"
            ])
    
    return conclusions

def draw_business_conclusions(df, impact_results, cluster_results):
    """
    Draw key business conclusions from the analysis
    
    Args:
        df (pd.DataFrame): Processed dataframe
        impact_results (dict): Business impact results
        cluster_results (dict): Clustering results
        
    Returns:
        dict: Business conclusions and recommendations
    """
    print(f"\n💼 BUSINESS CONCLUSIONS & STRATEGIC INSIGHTS:")
    print("=" * 50)
    
    business_conclusions = {
        'key_findings': [],
        'strategic_recommendations': [],
        'policy_implications': [],
        'implementation_priorities': []
    }
    
    # Training Impact Analysis
    training_impact = impact_results.get('training_impact', 0)
    gi_impact = impact_results.get('gi_impact', 0)
    economic_impact = impact_results.get('economic_impact_lakh', 0)
    
    print(f"🔍 KEY BUSINESS FINDINGS:")
    
    if training_impact > 0:
        business_conclusions['key_findings'].append(
            f"Training programs increase monthly income by ₹{training_impact:.0f} on average"
        )
        print(f"   • Training increases income by ₹{training_impact:.0f}/month")
    
    if gi_impact > 0:
        business_conclusions['key_findings'].append(
            f"GI registration provides ₹{gi_impact:.0f} monthly income benefit"
        )
        print(f"   • GI registration provides ₹{gi_impact:.0f}/month benefit")
    
    # Gender analysis
    female_earner_income = df[df['Primary_Earner_Gender'] == 'Female']['Monthly_Income'].mean()
    male_earner_income = df[df['Primary_Earner_Gender'] == 'Male']['Monthly_Income'].mean()
    gender_gap = abs(female_earner_income - male_earner_income)
    
    if gender_gap > 500:  # Significant gap
        business_conclusions['key_findings'].append(
            f"Gender income gap of ₹{gender_gap:.0f} requires targeted intervention"
        )
        print(f"   • Gender income gap: ₹{gender_gap:.0f}/month")
    
    # Resource accessibility impact
    resource_correlation = df['Raw_Material_Score'].corr(df['Monthly_Income'])
    if abs(resource_correlation) > 0.3:
        business_conclusions['key_findings'].append(
            f"Strong correlation ({resource_correlation:.3f}) between resource access and income"
        )
        print(f"   • Resource-income correlation: {resource_correlation:.3f}")
    
    # Strategic Recommendations
    print(f"\n📈 STRATEGIC RECOMMENDATIONS:")
    
    recommendations = [
        "Prioritize training program expansion to reach untrained artisans",
        "Streamline GI registration process to increase beneficiary coverage",
        "Address resource accessibility challenges through supply chain improvements",
        "Implement targeted support for underperforming artisan segments"
    ]
    
    if gender_gap > 500:
        recommendations.append("Develop gender-specific support programs to address income disparities")
    
    business_conclusions['strategic_recommendations'] = recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    # Policy Implications
    print(f"\n🏛️  POLICY IMPLICATIONS:")
    
    policy_implications = [
        f"Investment in training programs could generate ₹{economic_impact:.1f} Lakh annual economic impact",
        "GI certification process needs simplification and awareness campaigns",
        "Resource supply chain requires government intervention for sustainability",
        "Cluster-based interventions more effective than one-size-fits-all approaches"
    ]
    
    business_conclusions['policy_implications'] = policy_implications
    for i, policy in enumerate(policy_implications, 1):
        print(f"   {i}. {policy}")
    
    # Implementation Priorities
    print(f"\n🎯 IMPLEMENTATION PRIORITIES (Ranked by Impact):")
    
    priorities = [
        ("High", "Training Program Expansion", f"₹{training_impact:.0f}/month per artisan"),
        ("High", "GI Registration Drive", f"₹{gi_impact:.0f}/month per beneficiary"),
        ("Medium", "Resource Access Improvement", "Correlation-based impact"),
        ("Medium", "Targeted Cluster Interventions", "Segment-specific strategies")
    ]
    
    if gender_gap > 500:
        priorities.insert(2, ("High", "Gender Equity Programs", f"₹{gender_gap:.0f}/month potential"))
    
    business_conclusions['implementation_priorities'] = priorities
    for priority, action, impact in priorities:
        print(f"   • {priority} Priority: {action} ({impact})")
    
    return business_conclusions

def generate_research_limitations(df):
    """
    Identify and document research limitations
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        dict: Research limitations and future work suggestions
    """
    print(f"\n⚠️  RESEARCH LIMITATIONS & FUTURE WORK:")
    print("=" * 45)
    
    limitations = {
        'data_limitations': [],
        'methodological_limitations': [],
        'future_research': []
    }
    
    # Data limitations
    sample_size = len(df)
    missing_data_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    
    data_limits = [
        f"Small sample size (n={sample_size}) may limit generalizability",
        f"Cross-sectional data limits causal inference capabilities"
    ]
    
    if missing_data_pct > 10:
        data_limits.append(f"Significant missing data ({missing_data_pct:.1f}%) handled through imputation")
    
    limitations['data_limitations'] = data_limits
    
    print(f"📊 DATA LIMITATIONS:")
    for i, limit in enumerate(data_limits, 1):
        print(f"   {i}. {limit}")
    
    # Methodological limitations
    method_limits = [
        "KNN imputation may introduce bias in missing value handling",
        "Feature engineering based on domain assumptions may miss optimal combinations",
        "Model validation limited by small dataset size",
        "Clustering results sensitive to feature scaling and selection"
    ]
    
    limitations['methodological_limitations'] = method_limits
    
    print(f"\n🔬 METHODOLOGICAL LIMITATIONS:")
    for i, limit in enumerate(method_limits, 1):
        print(f"   {i}. {limit}")
    
    # Future research suggestions
    future_work = [
        "Longitudinal study to establish causal relationships",
        "Larger sample size for improved statistical power",
        "Deep learning approaches for complex pattern recognition",
        "Integration of external economic indicators",
        "Qualitative research to understand artisan motivations",
        "Geographic expansion to other GI clusters"
    ]
    
    limitations['future_research'] = future_work
    
    print(f"\n🔮 FUTURE RESEARCH DIRECTIONS:")
    for i, future in enumerate(future_work, 1):
        print(f"   {i}. {future}")
    
    return limitations

def create_executive_summary(df, results_dict):
    """
    Create executive summary for stakeholders
    
    Args:
        df (pd.DataFrame): Final dataset
        results_dict (dict): All analysis results
        
    Returns:
        dict: Executive summary
    """
    print(f"\n👔 EXECUTIVE SUMMARY FOR STAKEHOLDERS:")
    print("=" * 50)
    
    # Extract key metrics
    sample_size = len(df)
    ml_results = results_dict.get('ml', {})
    impact_results = results_dict.get('business_impact', {})
    
    best_r2 = 0
    if ml_results.get('regression'):
        best_r2 = max([v['test_r2'] for v in ml_results['regression'].values()])
    
    training_impact = impact_results.get('training_impact', 0)
    economic_impact = impact_results.get('economic_impact_lakh', 0)
    
    executive_summary = {
        'project_overview': {
            'objective': 'Analyze Varanasi GI toy cluster to identify growth opportunities',
            'scope': f'Survey of {sample_size} artisan households',
            'methodology': 'Advanced machine learning and statistical analysis'
        },
        'key_findings': {
            'predictive_accuracy': f'{best_r2*100:.1f}% variance explained in income prediction',
            'training_roi': f'₹{training_impact:.0f} monthly income increase per trained artisan',
            'economic_potential': f'₹{economic_impact:.1f} Lakh annual economic impact potential'
        },
        'recommendations': [
            'Immediate expansion of training programs',
            'Streamlined GI registration process',
            'Targeted interventions for underperforming segments'
        ],
        'next_steps': [
            'Pilot program implementation',
            'Stakeholder engagement sessions',
            'Monitoring and evaluation framework setup'
        ]
    }
    
    print(f"📋 PROJECT OVERVIEW:")
    print(f"   • Objective: {executive_summary['project_overview']['objective']}")
    print(f"   • Scope: {executive_summary['project_overview']['scope']}")
    print(f"   • Methodology: {executive_summary['project_overview']['methodology']}")
    
    print(f"\n🎯 KEY FINDINGS:")
    for key, value in executive_summary['key_findings'].items():
        print(f"   • {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n📋 IMMEDIATE RECOMMENDATIONS:")
    for i, rec in enumerate(executive_summary['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    print(f"\n➡️  NEXT STEPS:")
    for i, step in enumerate(executive_summary['next_steps'], 1):
        print(f"   {i}. {step}")
    
    return executive_summary

def generate_final_conclusions(df, results_dict):
    """
    Generate comprehensive final conclusions
    
    Args:
        df (pd.DataFrame): Final processed dataframe
        results_dict (dict): All analysis results
        
    Returns:
        dict: Complete conclusions and recommendations
    """
    print(f"\n🎯 COMPREHENSIVE FINAL CONCLUSIONS")
    print("=" * 60)
    
    # Analyze different aspects
    ml_conclusions = analyze_model_performance(results_dict.get('ml', {}))
    
    business_conclusions = draw_business_conclusions(
        df, 
        results_dict.get('business_impact', {}),
        results_dict.get('clustering', {})
    )
    
    limitations = generate_research_limitations(df)
    
    executive_summary = create_executive_summary(df, results_dict)
    
    # Compile final conclusions
    final_conclusions = {
        'model_performance': ml_conclusions,
        'business_insights': business_conclusions,
        'limitations': limitations,
        'executive_summary': executive_summary,
        'success_metrics': {
            'technical_success': f"Achieved R² = {max([v['test_r2'] for v in results_dict.get('ml', {}).get('regression', {}).values()], default=0):.3f}",
            'business_value': f"₹{results_dict.get('business_impact', {}).get('economic_impact_lakh', 0):.1f} Lakh potential impact",
            'actionable_insights': f"{len(business_conclusions.get('strategic_recommendations', []))} strategic recommendations"
        }
    }
    
    # Print final summary
    print(f"\n🏆 PROJECT SUCCESS METRICS:")
    for metric, value in final_conclusions['success_metrics'].items():
        print(f"   • {metric.replace('_', ' ').title()}: {value}")
    
    print(f"\n✅ ANALYSIS COMPLETED: Ready for implementation and deployment!")
    
    return final_conclusions

if __name__ == "__main__":
    print("📊 CONCLUSIONS AND RECOMMENDATIONS MODULE")
    print("This module draws final conclusions from the complete analysis.")
    print("Run main.py to execute the full pipeline and generate conclusions.")
