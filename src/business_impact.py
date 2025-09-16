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
from scipy.stats import ttest_ind
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
    t_stat, p_value = ttest_ind(
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
    t_stat, p_value = ttest_ind(
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
        'Is_GI_Beneficiary': 'mean'
    }).round(2)
    
    # Calculate gender gaps
    if 'Female' in gender_analysis.index and 'Male' in gender_analysis.index:
        female_income = gender_analysis.loc['Female', ('Monthly_Income', 'mean')]
        male_income = gender_analysis.loc['Male', ('Monthly_Income', 'mean')]
        income_gap = male_income - female_income
        income_gap_percent = (income_gap / female_income) * 100 if female_income > 0 else 0
        
        # Training access gap
        female_training = gender_analysis.loc['Female', ('Training_Access', 'mean')]
        male_training = gender_analysis.loc['Male', ('Training_Access', 'mean')]
        training_gap = male_training - female_training
        
        # GI registration gap
        female_gi = gender_analysis.loc['Female', ('Is_GI_Beneficiary', 'mean')]
        male_gi = gender_analysis.loc['Male', ('Is_GI_Beneficiary', 'mean')]
        gi_gap = male_gi - female_gi
    else:
        income_gap = income_gap_percent = training_gap = gi_gap = 0
        female_income = male_income = 0
    
    gender_impact = {
        'gender_breakdown': gender_analysis.to_dict(),
        'income_gap': {
            'absolute': income_gap,
            'percentage': income_gap_percent,
            'female_income': female_income,
            'male_income': male_income
        },
        'opportunity_gaps': {
            'training_access_gap': training_gap,
            'gi_registration_gap': gi_gap
        }
    }
    
    print(f"   • Female artisans: {gender_analysis.loc['Female', ('Monthly_Income', 'count')] if 'Female' in gender_analysis.index else 0}")
    print(f"   • Male artisans: {gender_analysis.loc['Male', ('Monthly_Income', 'count')] if 'Male' in gender_analysis.index else 0}")
    print(f"   • Income gap: ₹{income_gap:.0f}/month ({income_gap_percent:.1f}% {'higher for males' if income_gap > 0 else 'higher for females'})")
    print(f"   • Training access gap: {training_gap:.1%} {'higher for males' if training_gap > 0 else 'higher for females'}")
    
    return gender_impact

def calculate_resource_access_impact(df):
    """
    Analyze the impact of resource accessibility on outcomes
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Resource access impact analysis
    """
    print(f"\n🏭 RESOURCE ACCESS IMPACT ANALYSIS:")
    
    resource_impact = {}
    
    if 'Raw_Material_Access' in df.columns:
        # Analyze by resource access levels
        resource_analysis = df.groupby('Raw_Material_Access').agg({
            'Monthly_Income': ['mean', 'median', 'count'],
            'Satisfaction_Score': 'mean'
        }).round(2)
        
        resource_impact['raw_material_analysis'] = resource_analysis.to_dict()
        
        print(f"   • Resource Access Impact on Income:")
        for access_level in resource_analysis.index:
            income = resource_analysis.loc[access_level, ('Monthly_Income', 'mean')]
            count = resource_analysis.loc[access_level, ('Monthly_Income', 'count')]
            print(f"     - {access_level} access: ₹{income:.0f}/month (n={count})")
        
        # Calculate correlation if resource score exists
        if 'Raw_Material_Score' in df.columns:
            correlation = df['Monthly_Income'].corr(df['Raw_Material_Score'])
            resource_impact['income_correlation'] = correlation
            print(f"   • Resource-Income Correlation: {correlation:.3f}")
    
    return resource_impact

def quantify_economic_impact(df, training_impact, gi_impact, total_population=3000):
    """
    Quantify the potential economic impact of scaling interventions
    
    Args:
        df (pd.DataFrame): Input dataframe
        training_impact (dict): Training impact results
        gi_impact (dict): GI registration impact results
        total_population (int): Total artisan population in the cluster
        
    Returns:
        dict: Economic impact quantification
    """
    print(f"\n💰 ECONOMIC IMPACT QUANTIFICATION:")
    
    # Current sample statistics
    sample_size = len(df)
    sample_training_rate = df['Training_Access'].mean() if 'Training_Access' in df.columns else 0
    sample_gi_rate = df['Is_GI_Beneficiary'].mean() if 'Is_GI_Beneficiary' in df.columns else 0
    
    # Scale to total population
    current_trained = int(total_population * sample_training_rate)
    current_gi_registered = int(total_population * sample_gi_rate)
    
    untrained_population = total_population - current_trained
    unregistered_population = total_population - current_gi_registered
    
    # Calculate potential impacts
    monthly_training_impact = training_impact.get('income_difference', 0)
    monthly_gi_impact = gi_impact.get('income_difference', 0)
    
    # Annual impact calculations
    annual_training_impact = untrained_population * monthly_training_impact * 12
    annual_gi_impact = unregistered_population * monthly_gi_impact * 12
    
    # Combined impact (assuming no overlap for conservative estimate)
    total_annual_impact = annual_training_impact + annual_gi_impact
    
    # Convert to more readable format (Lakhs)
    annual_training_impact_lakh = annual_training_impact / 100000
    annual_gi_impact_lakh = annual_gi_impact / 100000
    total_annual_impact_lakh = total_annual_impact / 100000
    
    economic_impact = {
        'current_status': {
            'sample_size': sample_size,
            'total_population': total_population,
            'current_trained': current_trained,
            'current_gi_registered': current_gi_registered,
            'training_rate': sample_training_rate,
            'gi_rate': sample_gi_rate
        },
        'potential_beneficiaries': {
            'untrained_artisans': untrained_population,
            'unregistered_artisans': unregistered_population
        },
        'impact_per_artisan': {
            'monthly_training_benefit': monthly_training_impact,
            'monthly_gi_benefit': monthly_gi_impact,
            'annual_training_benefit': monthly_training_impact * 12,
            'annual_gi_benefit': monthly_gi_impact * 12
        },
        'total_economic_impact': {
            'annual_training_impact': annual_training_impact,
            'annual_gi_impact': annual_gi_impact,
            'total_annual_impact': total_annual_impact,
            'annual_training_impact_lakh': annual_training_impact_lakh,
            'annual_gi_impact_lakh': annual_gi_impact_lakh,
            'total_annual_impact_lakh': total_annual_impact_lakh
        }
    }
    
    print(f"   • Total Population: {total_population:,} artisans")
    print(f"   • Currently Trained: {current_trained:,} ({sample_training_rate:.1%})")
    print(f"   • Currently GI Registered: {current_gi_registered:,} ({sample_gi_rate:.1%})")
    print(f"   • Potential Training Beneficiaries: {untrained_population:,}")
    print(f"   • Potential GI Beneficiaries: {unregistered_population:,}")
    print(f"\n   💰 ANNUAL ECONOMIC IMPACT:")
    print(f"   • Training Programs: ₹{annual_training_impact_lakh:.1f} Lakh")
    print(f"   • GI Registration: ₹{annual_gi_impact_lakh:.1f} Lakh") 
    print(f"   • TOTAL POTENTIAL: ₹{total_annual_impact_lakh:.1f} Lakh")
    
    return economic_impact

def calculate_roi_analysis(economic_impact, program_costs=None):
    """
    Calculate Return on Investment for intervention programs
    
    Args:
        economic_impact (dict): Economic impact results
        program_costs (dict): Estimated program implementation costs
        
    Returns:
        dict: ROI analysis results
    """
    print(f"\n📊 RETURN ON INVESTMENT ANALYSIS:")
    
    # Default program costs (estimated in Lakhs)
    if program_costs is None:
        program_costs = {
            'training_program_cost_lakh': 2.0,  # ₹2 Lakh for training program
            'gi_registration_cost_lakh': 0.5,   # ₹0.5 Lakh for GI registration drive
            'administrative_overhead': 0.2      # 20% overhead
        }
    
    # Calculate costs
    training_cost = program_costs['training_program_cost_lakh']
    gi_cost = program_costs['gi_registration_cost_lakh']
    overhead_rate = program_costs.get('administrative_overhead', 0.2)
    
    total_direct_cost = training_cost + gi_cost
    total_cost_with_overhead = total_direct_cost * (1 + overhead_rate)
    
    # Calculate benefits
    training_benefit = economic_impact['total_economic_impact']['annual_training_impact_lakh']
    gi_benefit = economic_impact['total_economic_impact']['annual_gi_impact_lakh']
    total_benefit = training_benefit + gi_benefit
    
    # Calculate ROI metrics
    net_benefit = total_benefit - total_cost_with_overhead
    roi_ratio = total_benefit / total_cost_with_overhead if total_cost_with_overhead > 0 else 0
    roi_percentage = (roi_ratio - 1) * 100
    payback_period = total_cost_with_overhead / total_benefit if total_benefit > 0 else float('inf')
    
    roi_analysis = {
        'costs': {
            'training_program_cost_lakh': training_cost,
            'gi_registration_cost_lakh': gi_cost,
            'overhead_cost_lakh': total_direct_cost * overhead_rate,
            'total_cost_lakh': total_cost_with_overhead
        },
        'benefits': {
            'training_benefit_lakh': training_benefit,
            'gi_benefit_lakh': gi_benefit,
            'total_benefit_lakh': total_benefit
        },
        'roi_metrics': {
            'net_benefit_lakh': net_benefit,
            'roi_ratio': roi_ratio,
            'roi_percentage': roi_percentage,
            'payback_period_years': payback_period
        }
    }
    
    print(f"   💰 PROGRAM COSTS:")
    print(f"   • Training Program: ₹{training_cost:.1f} Lakh")
    print(f"   • GI Registration Drive: ₹{gi_cost:.1f} Lakh")
    print(f"   • Administrative Overhead: ₹{total_direct_cost * overhead_rate:.1f} Lakh")
    print(f"   • TOTAL COST: ₹{total_cost_with_overhead:.1f} Lakh")
    
    print(f"\n   📈 RETURNS:")
    print(f"   • Annual Economic Benefit: ₹{total_benefit:.1f} Lakh")
    print(f"   • Net Benefit: ₹{net_benefit:.1f} Lakh")
    print(f"   • ROI: {roi_percentage:.0f}% ({roi_ratio:.1f}x return)")
    print(f"   • Payback Period: {payback_period:.1f} years")
    
    return roi_analysis

def generate_policy_recommendations(training_impact, gi_impact, gender_impact, economic_impact):
    """
    Generate actionable policy recommendations based on analysis
    
    Args:
        training_impact (dict): Training impact results
        gi_impact (dict): GI impact results  
        gender_impact (dict): Gender impact results
        economic_impact (dict): Economic impact results
        
    Returns:
        dict: Policy recommendations
    """
    print(f"\n📋 POLICY RECOMMENDATIONS:")
    
    recommendations = {
        'immediate_actions': [],
        'medium_term_strategies': [],
        'long_term_initiatives': [],
        'priority_ranking': []
    }
    
    # Immediate actions (High impact, quick wins)
    if training_impact.get('income_difference', 0) > 500:
        recommendations['immediate_actions'].append({
            'action': 'Scale up training programs immediately',
            'rationale': f'Training increases income by ₹{training_impact["income_difference"]:.0f}/month',
            'target': f'{economic_impact["potential_beneficiaries"]["untrained_artisans"]} untrained artisans',
            'priority': 'HIGH'
        })
    
    if gi_impact.get('income_difference', 0) > 300:
        recommendations['immediate_actions'].append({
            'action': 'Launch GI registration awareness campaign',
            'rationale': f'GI registration increases income by ₹{gi_impact["income_difference"]:.0f}/month',
            'target': f'{economic_impact["potential_beneficiaries"]["unregistered_artisans"]} unregistered artisans',
            'priority': 'HIGH'
        })
    
    # Medium-term strategies
    if gender_impact.get('income_gap', {}).get('absolute', 0) > 500:
        recommendations['medium_term_strategies'].append({
            'action': 'Implement gender-specific support programs',
            'rationale': f'Address ₹{gender_impact["income_gap"]["absolute"]:.0f}/month gender income gap',
            'target': 'Female artisans',
            'priority': 'MEDIUM'
        })
    
    recommendations['medium_term_strategies'].append({
        'action': 'Establish artisan resource centers',
        'rationale': 'Improve raw material accessibility and reduce costs',
        'target': 'All artisans',
        'priority': 'MEDIUM'
    })
    
    # Long-term initiatives
    recommendations['long_term_initiatives'].extend([
        {
            'action': 'Develop market linkage programs',
            'rationale': 'Connect artisans directly with buyers and reduce intermediary costs',
            'target': 'All artisans',
            'priority': 'LOW'
        },
        {
            'action': 'Establish quality certification system',
            'rationale': 'Improve product quality and market positioning',
            'target': 'All artisans',
            'priority': 'LOW'
        }
    ])
    
    # Priority ranking based on impact and feasibility
    all_recommendations = (recommendations['immediate_actions'] + 
                          recommendations['medium_term_strategies'] + 
                          recommendations['long_term_initiatives'])
    
    # Sort by priority
    priority_order = {'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    recommendations['priority_ranking'] = sorted(all_recommendations, 
                                               key=lambda x: priority_order[x['priority']])
    
    # Print recommendations
    print(f"   🎯 IMMEDIATE ACTIONS (HIGH PRIORITY):")
    for i, action in enumerate(recommendations['immediate_actions'], 1):
        print(f"   {i}. {action['action']}")
        print(f"      • {action['rationale']}")
        print(f"      • Target: {action['target']}")
    
    print(f"\n   📈 MEDIUM-TERM STRATEGIES:")
    for i, strategy in enumerate(recommendations['medium_term_strategies'], 1):
        print(f"   {i}. {strategy['action']}")
        print(f"      • {strategy['rationale']}")
    
    print(f"\n   🔮 LONG-TERM INITIATIVES:")
    for i, initiative in enumerate(recommendations['long_term_initiatives'], 1):
        print(f"   {i}. {initiative['action']}")
    
    return recommendations

def analyze_business_impact(df):
    """
    Main function to perform comprehensive business impact analysis
    
    Args:
        df (pd.DataFrame): Input dataframe with engineered features
        
    Returns:
        dict: Complete business impact analysis results
    """
    print(f"💼 COMPREHENSIVE BUSINESS IMPACT ANALYSIS")
    print("=" * 60)
    
    # Perform individual impact analyses
    training_impact = calculate_training_impact(df)
    gi_impact = calculate_gi_registration_impact(df)
    gender_impact = calculate_gender_impact(df)
    resource_impact = calculate_resource_access_impact(df)
    
    # Quantify economic impact
    economic_impact = quantify_economic_impact(df, training_impact, gi_impact)
    
    # Calculate ROI
    roi_analysis = calculate_roi_analysis(economic_impact)
    
    # Generate policy recommendations
    policy_recommendations = generate_policy_recommendations(
        training_impact, gi_impact, gender_impact, economic_impact
    )
    
    # Compile complete results
    complete_impact_analysis = {
        'training_impact': training_impact,
        'gi_impact': gi_impact,
        'gender_impact': gender_impact,
        'resource_impact': resource_impact,
        'economic_impact': economic_impact,
        'roi_analysis': roi_analysis,
        'policy_recommendations': policy_recommendations
    }
    
    # Print summary
    print(f"\n💰 BUSINESS IMPACT SUMMARY:")
    print(f"   • Training Income Impact: ₹{training_impact.get('income_difference', 0):.0f}/month per artisan")
    print(f"   • GI Registration Impact: ₹{gi_impact.get('income_difference', 0):.0f}/month per artisan")
    print(f"   • Total Economic Potential: ₹{economic_impact['total_economic_impact']['total_annual_impact_lakh']:.1f} Lakh annually")
    print(f"   • ROI: {roi_analysis['roi_metrics']['roi_percentage']:.0f}% return on investment")
    print(f"   • Payback Period: {roi_analysis['roi_metrics']['payback_period_years']:.1f} years")
    
    return complete_impact_analysis

if __name__ == "__main__":
    print("💼 BUSINESS IMPACT ANALYSIS MODULE")
    print("This module quantifies business impact and calculates ROI.")
    print("Run main.py to execute the complete analysis pipeline.")
