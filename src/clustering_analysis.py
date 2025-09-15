# =============================================================================
# CLUSTERING ANALYSIS MODULE
# =============================================================================
"""
This module performs K-means clustering to identify distinct artisan segments
and provides comprehensive cluster profiling and analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

def determine_optimal_clusters(X, max_clusters=8, method='elbow'):
    """
    Determine optimal number of clusters using multiple methods
    
    Args:
        X (pd.DataFrame): Features for clustering
        max_clusters (int): Maximum number of clusters to test
        method (str): Method to use ('elbow', 'silhouette', 'both')
        
    Returns:
        dict: Optimal clustering results
    """
    print(f"🎯 DETERMINING OPTIMAL NUMBER OF CLUSTERS:")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test different numbers of clusters
    k_range = range(2, max_clusters + 1)
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
        calinski_scores.append(calinski_harabasz_score(X_scaled, cluster_labels))
    
    # Find optimal k using different methods
    results = {
        'k_range': list(k_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'calinski_scores': calinski_scores
    }
    
    # Elbow method (look for the "elbow" in inertia curve)
    # Simple approach: find the point where the decrease in inertia starts to level off
    differences = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
    elbow_k = k_range[differences.index(min(differences[1:]))] + 1
    
    # Silhouette method (highest silhouette score)
    silhouette_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    
    # Calinski-Harabasz method (highest score)
    calinski_k = k_range[calinski_scores.index(max(calinski_scores))]
    
    results.update({
        'elbow_k': elbow_k,
        'silhouette_k': silhouette_k,
        'calinski_k': calinski_k,
        'recommended_k': silhouette_k  # Use silhouette as primary recommendation
    })
    
    print(f"   • Elbow method suggests: k = {elbow_k}")
    print(f"   • Silhouette method suggests: k = {silhouette_k}")
    print(f"   • Calinski-Harabasz suggests: k = {calinski_k}")
    print(f"   • Recommended: k = {silhouette_k}")
    
    return results

def perform_kmeans_clustering(df, optimal_k, feature_cols=None):
    """
    Perform K-means clustering with the optimal number of clusters
    
    Args:
        df (pd.DataFrame): Input dataframe
        optimal_k (int): Optimal number of clusters
        feature_cols (list): List of feature columns to use for clustering
        
    Returns:
        dict: Clustering results
    """
    print(f"\n🎯 PERFORMING K-MEANS CLUSTERING (k={optimal_k}):")
    
    # Default feature selection if not provided
    if feature_cols is None:
        feature_cols = [
            'Income_Per_Member', 'Total_Resource_Score', 'Satisfaction_Score',
            'Family_Size', 'Training_Access', 'Is_GI_Beneficiary'
        ]
    
    # Select available features
    available_features = [col for col in feature_cols if col in df.columns]
    
    if len(available_features) < 3:
        print(f"   ❌ Insufficient features available for clustering")
        return {}
    
    print(f"   • Features used: {available_features}")
    
    # Prepare data
    X = df[available_features].fillna(df[available_features].median())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels
    
    # Calculate clustering metrics
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    calinski_score = calinski_harabasz_score(X_scaled, cluster_labels)
    inertia = kmeans.inertia_
    
    print(f"   • Silhouette Score: {silhouette_avg:.3f}")
    print(f"   • Calinski-Harabasz Score: {calinski_score:.2f}")
    print(f"   • Inertia: {inertia:.2f}")
    
    # Cluster centers (in original scale)
    cluster_centers_scaled = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers_df = pd.DataFrame(
        cluster_centers_scaled,
        columns=available_features,
        index=[f'Cluster_{i}' for i in range(optimal_k)]
    )
    
    results = {
        'df_clustered': df_clustered,
        'cluster_labels': cluster_labels,
        'cluster_centers': cluster_centers_df,
        'optimal_k': optimal_k,
        'features_used': available_features,
        'silhouette_score': silhouette_avg,
        'calinski_score': calinski_score,
        'inertia': inertia,
        'scaler': scaler,
        'kmeans_model': kmeans
    }
    
    return results

def profile_clusters(df_clustered, cluster_col='Cluster'):
    """
    Create comprehensive cluster profiles
    
    Args:
        df_clustered (pd.DataFrame): Dataframe with cluster assignments
        cluster_col (str): Name of cluster column
        
    Returns:
        dict: Cluster profiling results
    """
    print(f"\n👥 CLUSTER PROFILING:")
    
    if cluster_col not in df_clustered.columns:
        print(f"   ❌ Cluster column '{cluster_col}' not found")
        return {}
    
    profiles = {}
    unique_clusters = sorted(df_clustered[cluster_col].unique())
    
    # Key variables for profiling
    profile_vars = [
        'Monthly_Income', 'Family_Size', 'Training_Access', 
        'Is_GI_Beneficiary', 'Satisfaction_Score'
    ]
    
    # Add engineered features if available
    engineered_vars = [
        'Income_Per_Member', 'Artisan_Success_Index', 'Total_Resource_Score'
    ]
    
    available_vars = [var for var in profile_vars + engineered_vars 
                     if var in df_clustered.columns]
    
    print(f"   • Profiling {len(unique_clusters)} clusters using {len(available_vars)} variables")
    
    # Create cluster profiles
    for cluster_id in unique_clusters:
        cluster_data = df_clustered[df_clustered[cluster_col] == cluster_id]
        cluster_size = len(cluster_data)
        cluster_pct = (cluster_size / len(df_clustered)) * 100
        
        profile = {
            'size': cluster_size,
            'percentage': cluster_pct,
            'characteristics': {}
        }
        
        print(f"\n   📊 Cluster {cluster_id} (n={cluster_size}, {cluster_pct:.1f}%):")
        
        # Calculate statistics for each variable
        for var in available_vars:
            if var in cluster_data.columns:
                if cluster_data[var].dtype in ['int64', 'float64']:
                    # Numerical variable
                    mean_val = cluster_data[var].mean()
                    median_val = cluster_data[var].median()
                    std_val = cluster_data[var].std()
                    
                    profile['characteristics'][var] = {
                        'mean': mean_val,
                        'median': median_val,
                        'std': std_val
                    }
                    
                    if var == 'Monthly_Income':
                        print(f"     - Avg Income: ₹{mean_val:.0f}")
                    elif var == 'Satisfaction_Score':
                        print(f"     - Avg Satisfaction: {mean_val:.2f}")
                    elif var in ['Training_Access', 'Is_GI_Beneficiary']:
                        pct = mean_val * 100
                        print(f"     - {var}: {pct:.0f}%")
                else:
                    # Categorical variable
                    value_counts = cluster_data[var].value_counts(normalize=True)
                    profile['characteristics'][var] = value_counts.to_dict()
        
        profiles[cluster_id] = profile
    
    # Identify cluster characteristics
    cluster_insights = identify_cluster_insights(profiles, df_clustered)
    
    return {
        'profiles': profiles,
        'insights': cluster_insights,
        'cluster_count': len(unique_clusters)
    }

def identify_cluster_insights(profiles, df_clustered):
    """
    Generate insights and labels for each cluster
    
    Args:
        profiles (dict): Cluster profiles
        df_clustered (pd.DataFrame): Dataframe with clusters
        
    Returns:
        dict: Cluster insights and labels
    """
    insights = {}
    
    # Calculate overall means for comparison
    overall_income = df_clustered['Monthly_Income'].mean() if 'Monthly_Income' in df_clustered.columns else 0
    overall_satisfaction = df_clustered['Satisfaction_Score'].mean() if 'Satisfaction_Score' in df_clustered.columns else 0
    
    for cluster_id, profile in profiles.items():
        characteristics = profile['characteristics']
        
        # Determine cluster characteristics
        cluster_income = characteristics.get('Monthly_Income', {}).get('mean', 0)
        cluster_satisfaction = characteristics.get('Satisfaction_Score', {}).get('mean', 0)
        cluster_training = characteristics.get('Training_Access', {}).get('mean', 0)
        cluster_gi = characteristics.get('Is_GI_Beneficiary', {}).get('mean', 0)
        
        # Generate insights
        insight = {
            'income_level': 'High' if cluster_income > overall_income * 1.1 else 
                           'Low' if cluster_income < overall_income * 0.9 else 'Medium',
            'satisfaction_level': 'High' if cluster_satisfaction > overall_satisfaction * 1.1 else 
                                 'Low' if cluster_satisfaction < overall_satisfaction * 0.9 else 'Medium',
            'training_access': 'High' if cluster_training > 0.6 else 
                              'Low' if cluster_training < 0.3 else 'Medium',
            'gi_participation': 'High' if cluster_gi > 0.6 else 
                               'Low' if cluster_gi < 0.3 else 'Medium'
        }
        
        # Generate cluster label based on characteristics
        if insight['income_level'] == 'High' and insight['training_access'] == 'High':
            label = "High Performers"
        elif insight['training_access'] == 'High' and insight['income_level'] != 'High':
            label = "Growing Artisans"
        elif insight['income_level'] == 'Low' and insight['training_access'] == 'Low':
            label = "Support Needed"
        else:
            label = "Average Performers"
        
        insight['cluster_label'] = label
        insight['key_features'] = []
        
        # Add key descriptive features
        if insight['income_level'] == 'High':
            insight['key_features'].append('High income')
        if insight['training_access'] == 'High':
            insight['key_features'].append('Well trained')
        if insight['gi_participation'] == 'High':
            insight['key_features'].append('GI registered')
        if insight['satisfaction_level'] == 'High':
            insight['key_features'].append('Highly satisfied')
        
        insights[cluster_id] = insight
    
    return insights

def create_cluster_visualizations(df_clustered, cluster_results, output_dir='visualizations'):
    """
    Create visualizations for cluster analysis
    
    Args:
        df_clustered (pd.DataFrame): Dataframe with cluster assignments
        cluster_results (dict): Clustering results
        output_dir (str): Output directory
        
    Returns:
        list: List of created visualization files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📊 CREATING CLUSTER VISUALIZATIONS:")
    
    created_files = []
    
    # Create comprehensive clustering visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cluster Analysis Results', fontsize=16, fontweight='bold')
    
    # 1. Cluster size distribution
    cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
    axes[0,0].bar(cluster_counts.index, cluster_counts.values, color='skyblue', alpha=0.8)
    axes[0,0].set_title('Cluster Size Distribution')
    axes[0,0].set_xlabel('Cluster')
    axes[0,0].set_ylabel('Number of Artisans')
    for i, v in enumerate(cluster_counts.values):
        axes[0,0].text(i, v + 1, str(v), ha='center', fontweight='bold')
    
    # 2. Income vs Satisfaction by cluster
    if all(col in df_clustered.columns for col in ['Monthly_Income', 'Satisfaction_Score', 'Cluster']):
        scatter = axes[0,1].scatter(df_clustered['Satisfaction_Score'], df_clustered['Monthly_Income'], 
                                   c=df_clustered['Cluster'], cmap='viridis', alpha=0.7, s=50)
        axes[0,1].set_title('Income vs Satisfaction by Cluster')
        axes[0,1].set_xlabel('Satisfaction Score')
        axes[0,1].set_ylabel('Monthly Income (₹)')
        plt.colorbar(scatter, ax=axes[0,1], label='Cluster')
    
    # 3. Cluster characteristics heatmap
    if 'profiles' in cluster_results:
        cluster_data = []
        cluster_labels = []
        
        for cluster_id, profile in cluster_results['profiles'].items():
            char = profile['characteristics']
            row_data = []
            
            # Extract key metrics for heatmap
            metrics = ['Monthly_Income', 'Training_Access', 'Is_GI_Beneficiary', 'Satisfaction_Score']
            for metric in metrics:
                if metric in char:
                    if isinstance(char[metric], dict) and 'mean' in char[metric]:
                        row_data.append(char[metric]['mean'])
                    else:
                        row_data.append(0)
                else:
                    row_data.append(0)
            
            cluster_data.append(row_data)
            cluster_labels.append(f'Cluster {cluster_id}')
        
        if cluster_data:
            heatmap_df = pd.DataFrame(cluster_data, 
                                    index=cluster_labels,
                                    columns=['Income', 'Training', 'GI Beneficiary', 'Satisfaction'])
            
            # Normalize for better visualization
            heatmap_normalized = heatmap_df.div(heatmap_df.max())
            
            sns.heatmap(heatmap_normalized, annot=True, cmap='RdYlBu_r', 
                       ax=axes[1,0], fmt='.2f', cbar_kws={'label': 'Normalized Score'})
            axes[1,0].set_title('Cluster Characteristics (Normalized)')
    
    # 4. Training access vs GI beneficiary by cluster
    if all(col in df_clustered.columns for col in ['Training_Access', 'Is_GI_Beneficiary', 'Cluster']):
        cluster_training_gi = df_clustered.groupby('Cluster')[['Training_Access', 'Is_GI_Beneficiary']].mean()
        
        x = np.arange(len(cluster_training_gi.index))
        width = 0.35
        
        axes[1,1].bar(x - width/2, cluster_training_gi['Training_Access'], width, 
                     label='Training Access', alpha=0.8, color='lightblue')
        axes[1,1].bar(x + width/2, cluster_training_gi['Is_GI_Beneficiary'], width, 
                     label='GI Beneficiary', alpha=0.8, color='lightcoral')
        
        axes[1,1].set_title('Training & GI Access by Cluster')
        axes[1,1].set_xlabel('Cluster')
        axes[1,1].set_ylabel('Proportion')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels([f'Cluster {i}' for i in cluster_training_gi.index])
        axes[1,1].legend()
    
    plt.tight_layout()
    
    # Save the plot
    cluster_file = os.path.join(output_dir, 'cluster_analysis_results.png')
    plt.savefig(cluster_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    created_files.append(cluster_file)
    print(f"   ✓ Cluster visualization saved: {cluster_file}")
    
    return created_files

def perform_clustering(df):
    """
    Main function to perform complete clustering analysis
    
    Args:
        df (pd.DataFrame): Input dataframe with engineered features
        
    Returns:
        dict: Complete clustering analysis results
    """
    print(f"🎯 COMPREHENSIVE CLUSTERING ANALYSIS")
    print("=" * 50)
    
    # Select features for clustering
    clustering_features = [
        'Income_Per_Member', 'Total_Resource_Score', 'Satisfaction_Score',
        'Family_Size', 'Artisan_Success_Index', 'Training_Access'
    ]
    
    # Use available features
    available_features = [col for col in clustering_features if col in df.columns]
    
    if len(available_features) < 3:
        print("❌ Insufficient features for clustering analysis")
        return {}
    
    # Prepare clustering data
    X_cluster = df[available_features].fillna(df[available_features].median())
    
    # Step 1: Determine optimal number of clusters
    optimal_results = determine_optimal_clusters(X_cluster, max_clusters=8)
    optimal_k = optimal_results['recommended_k']
    
    # Step 2: Perform K-means clustering
    clustering_results = perform_kmeans_clustering(df, optimal_k, available_features)
    
    if not clustering_results:
        return {}
    
    # Step 3: Profile clusters
    profiling_results = profile_clusters(clustering_results['df_clustered'])
    
    # Step 4: Create visualizations
    visualization_files = create_cluster_visualizations(
        clustering_results['df_clustered'], 
        profiling_results
    )
    
    # Compile complete results
    complete_results = {
        'optimal_clusters': optimal_k,
        'optimization_results': optimal_results,
        'clustering_results': clustering_results,
        'cluster_profiles': profiling_results,
        'visualization_files': visualization_files,
        'features_used': available_features
    }
    
    # Print summary
    print(f"\n🎯 CLUSTERING SUMMARY:")
    print(f"   • Optimal clusters: {optimal_k}")
    print(f"   • Silhouette score: {clustering_results['silhouette_score']:.3f}")
    print(f"   • Features used: {len(available_features)}")
    
    if profiling_results and 'insights' in profiling_results:
        print(f"   • Cluster insights:")
        for cluster_id, insight in profiling_results['insights'].items():
            label = insight.get('cluster_label', f'Cluster {cluster_id}')
            size = profiling_results['profiles'][cluster_id]['size']
            print(f"     - Cluster {cluster_id}: {label} (n={size})")
    
    return complete_results

if __name__ == "__main__":
    print("🎯 CLUSTERING ANALYSIS MODULE")
    print("This module performs K-means clustering for artisan segmentation.")
    print("Run main.py to execute the complete analysis pipeline.")
