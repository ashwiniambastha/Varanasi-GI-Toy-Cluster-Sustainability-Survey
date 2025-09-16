# Methodology - Varanasi GI Toy Cluster Survey Analysis

## Table of Contents
1. [Research Design](#research-design)
2. [Data Collection & Preparation](#data-collection--preparation)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Feature Engineering](#feature-engineering)
5. [Data Preprocessing](#data-preprocessing)
6. [Machine Learning Pipeline](#machine-learning-pipeline)
7. [Clustering Analysis](#clustering-analysis)
8. [Business Impact Assessment](#business-impact-assessment)
9. [Validation & Evaluation](#validation--evaluation)
10. [Limitations](#limitations)

---

## Research Design

### Objective
To analyze socio-economic factors affecting artisan households in Varanasi's Geographic Indication (GI) toy manufacturing cluster and quantify the impact of policy interventions.

### Research Questions
1. **Primary**: What factors most significantly predict household income in the GI toy cluster?
2. **Secondary**: How do training programs and GI registration impact artisan outcomes?
3. **Tertiary**: Can we identify distinct artisan segments for targeted interventions?

### Approach
- **Cross-sectional survey analysis** of 119 artisan households
- **Mixed-methods approach** combining quantitative analysis with policy impact modeling
- **Machine learning-based predictive modeling** for income estimation
- **Clustering analysis** for market segmentation
- **Economic impact quantification** for policy recommendations

---

## Data Collection & Preparation

### Dataset Overview
- **Sample Size**: 119 households
- **Sampling Method**: Stratified random sampling across the GI toy cluster
- **Data Collection Period**: 2024 survey period
- **Geographic Scope**: Varanasi GI toy manufacturing cluster

### Variables Collected
| Variable | Type | Description | Values/Range |
|----------|------|-------------|--------------|
| `Monthly_Income` | Numerical | Household monthly income (₹) | 4,500 - 15,000 |
| `Family_Size` | Numerical | Number of family members | 2 - 12 |
| `Training_Access` | Binary | Access to training programs | 0 (No), 1 (Yes) |
| `Is_GI_Beneficiary` | Binary | GI registration status | 0 (No), 1 (Yes) |
| `Primary_Earner_Gender` | Categorical | Gender of primary earner | Male, Female |
| `Raw_Material_Access` | Ordinal | Ease of raw material access | Easy, Moderate, Difficult |
| `Satisfaction_Score` | Ordinal | Overall satisfaction rating | 1-5 scale |

### Data Quality Assessment
- **Missing Values**: Comprehensive assessment and documentation
- **Outlier Detection**: IQR and Z-score methods applied
- **Data Validation**: Cross-checking and consistency verification
- **Representativeness**: Comparison with known cluster demographics

---

## Exploratory Data Analysis

### Univariate Analysis
- **Distribution Analysis**: Histograms, density plots, and statistical summaries
- **Central Tendency**: Mean, median, mode calculations
- **Dispersion Measures**: Standard deviation, variance, range analysis
- **Shape Assessment**: Skewness and kurtosis evaluation

### Bivariate Analysis
- **Correlation Analysis**: Pearson correlation coefficients
- **Group Comparisons**: T-tests and ANOVA where appropriate
- **Cross-tabulation**: Categorical variable relationships
- **Statistical Significance**: P-value calculations at α = 0.05

### Visualization Strategy
- **Distribution Plots**: Income histograms with statistical overlays
- **Relationship Plots**: Scatter plots with trend lines
- **Comparison Charts**: Box plots and violin plots for group analysis
- **Correlation Heatmaps**: Feature relationship visualization

---

## Feature Engineering

### Economic Indicators
```python
# Per capita calculations
df['Income_Per_Member'] = df['Monthly_Income'] / df['Family_Size']

# Economic vulnerability index
df['Economic_Vulnerability'] = (
    (df['Monthly_Income'] < median_income) & 
    (df['Family_Size'] >= median_family_size)
).astype(int)
```

### Resource Accessibility Features
```python
# Resource scoring system
resource_mapping = {'Easy': 3, 'Moderate': 2, 'Difficult': 1}
df['Raw_Material_Score'] = df['Raw_Material_Access'].map(resource_mapping)

# Composite resource score
df['Total_Resource_Score'] = df['Raw_Material_Score'] + df['Training_Access']
```

### Interaction Terms
```python
# Program synergy effects
df['GI_Training_Synergy'] = df['Is_GI_Beneficiary'] * df['Training_Access']
df['Female_Training_Access'] = df['Female_Earner'] * df['Training_Access']
```

### Statistical Transformations
```python
# Z-score normalization
df['Income_Zscore'] = (df['Monthly_Income'] - μ) / σ

# Percentile ranks
df['Income_Percentile'] = df['Monthly_Income'].rank(pct=True)
```

### Success Indices
```python
# Weighted success index
df['Artisan_Success_Index'] = (
    0.3 * df['Above_Median_Income'] + 
    0.2 * df['High_Satisfaction'] + 
    0.25 * df['Is_GI_Beneficiary'] + 
    0.25 * df['Training_Access']
)
```

**Total Features Created**: 20+ engineered features across 7 categories

---

## Data Preprocessing

### Missing Value Treatment
#### Strategy Selection
- **KNN Imputation** (k=5) for numerical variables
- **Mode Imputation** for categorical variables
- **Justification**: Preserves relationships between variables better than mean/median imputation

#### Implementation
```python
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
```

### Outlier Handling
#### Detection Method
- **IQR Method**: Q1 - 1.5×IQR and Q3 + 1.5×IQR boundaries
- **Action**: Capping to boundary values (conservative approach)

#### Rationale
- Preserves sample size
- Reduces impact of extreme values
- Maintains data distribution shape

### Feature Scaling
#### Methods Applied
- **Standard Scaling**: For income and continuous variables
- **Min-Max Scaling**: For bounded variables when needed
- **Robust Scaling**: For variables with remaining outliers

#### Implementation
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])
```

### Categorical Encoding
- **Label Encoding**: For ordinal variables
- **One-Hot Encoding**: For nominal variables with <5 categories
- **Ordinal Encoding**: For Raw_Material_Access (Easy=3, Moderate=2, Difficult=1)

---

## Machine Learning Pipeline

### Model Selection Rationale
| Algorithm | Justification | Hyperparameters |
|-----------|---------------|-----------------|
| **XGBoost** | Handles mixed data types, feature importance | n_estimators=100, max_depth=6 |
| **Random Forest** | Ensemble robustness, interpretability | n_estimators=100, max_features='sqrt' |
| **Linear Regression** | Baseline, interpretable coefficients | Default |
| **SVM** | Non-linear relationships | RBF kernel, C=1.0 |
| **Decision Tree** | Simple interpretability | max_depth=10 |

### Training Strategy
#### Train-Test Split
- **Split Ratio**: 80% train, 20% test
- **Stratification**: Applied for classification tasks
- **Random Seed**: 42 for reproducibility

#### Cross-Validation
- **Method**: 5-fold stratified cross-validation
- **Metrics**: R² for regression, accuracy for classification
- **Validation**: Performed on training set only

### Hyperparameter Tuning
#### XGBoost Optimization
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}
```
- **Method**: GridSearchCV with 3-fold CV
- **Scoring**: R² for regression, accuracy for classification

### Evaluation Metrics
#### Regression (Income Prediction)
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error in ₹
- **MAE**: Mean Absolute Error

#### Classification (GI Beneficiary Prediction)
- **Accuracy**: Overall correct predictions
- **Precision/Recall**: Class-specific performance
- **F1-Score**: Harmonic mean of precision/recall

---

## Clustering Analysis

### Algorithm Selection
- **K-Means Clustering**: Chosen for interpretability and scalability
- **Distance Metric**: Euclidean distance on standardized features

### Optimal Cluster Determination
#### Methods Applied
1. **Elbow Method**: Inertia vs. number of clusters
2. **Silhouette Analysis**: Average silhouette score
3. **Calinski-Harabasz Index**: Cluster separation measure

#### Feature Selection for Clustering
```python
clustering_features = [
    'Income_Per_Member',
    'Total_Resource_Score', 
    'Satisfaction_Score',
    'Family_Size',
    'Artisan_Success_Index',
    'Training_Access'
]
```

### Cluster Validation
- **Silhouette Score**: Measure of cluster cohesion
- **Inertia Analysis**: Within-cluster sum of squares
- **Business Interpretability**: Meaningful segment characteristics

### Cluster Profiling
- **Demographic Characteristics**: Income, family size, gender distribution
- **Program Participation**: Training and GI registration rates
- **Performance Metrics**: Success indices and satisfaction scores

---

## Business Impact Assessment

### Economic Impact Modeling
#### Population Scaling
```python
# Scale sample to total population
total_population = 3000  # Estimated total artisan families
sample_rate = 119 / 3000  # Survey sample rate
```

#### Impact Calculation
```python
# Training impact
monthly_benefit = trained_mean_income - untrained_mean_income
annual_impact = untrained_population * monthly_benefit * 12
```

### Statistical Significance Testing
- **T-tests**: Compare means between intervention groups
- **Effect Size**: Cohen's d for practical significance
- **Confidence Intervals**: 95% CI for impact estimates

### ROI Analysis
#### Cost Estimation
- **Training Programs**: ₹2.0 Lakh estimated cost
- **GI Registration Drive**: ₹0.5 Lakh estimated cost  
- **Administrative Overhead**: 20% of direct costs

#### Benefit Calculation
- **Direct Benefits**: Income increases × affected population
- **Indirect Benefits**: Satisfaction and quality improvements
- **Time Horizon**: 5-year analysis period

### Policy Simulation
- **Scenario Modeling**: Different intervention coverage rates
- **Sensitivity Analysis**: Varying cost and benefit assumptions
- **Break-even Analysis**: Minimum impact required for positive ROI

---

## Validation & Evaluation

### Model Validation
#### Internal Validation
- **Cross-Validation**: 5-fold CV on training data
- **Learning Curves**: Training vs. validation performance
- **Feature Importance**: Consistency across models

#### External Validation
- **Hold-out Test Set**: 20% of data never used in training
- **Temporal Stability**: Model performance over time
- **Domain Expert Review**: Validation of findings with stakeholders

### Statistical Validation
- **Assumption Testing**: Normality, homoscedasticity checks
- **Multicollinearity**: VIF analysis for feature independence
- **Residual Analysis**: Model fit diagnostics

### Business Validation
- **Stakeholder Review**: Validation with domain experts
- **Practical Significance**: Beyond statistical significance
- **Implementation Feasibility**: Real-world applicability

---

## Limitations

### Data Limitations
1. **Sample Size**: 119 households may limit generalizability
2. **Cross-sectional Design**: Cannot establish causation
3. **Self-reported Data**: Potential reporting bias
4. **Geographic Scope**: Limited to one GI cluster

### Methodological Limitations
1. **Missing Data**: Despite imputation, information loss possible
2. **Feature Engineering**: Based on domain assumptions
3. **Model Selection**: Limited to available algorithms
4. **Temporal Factors**: Point-in-time analysis only

### External Validity
1. **Cluster Specificity**: May not apply to other GI clusters
2. **Economic Conditions**: Results tied to current economic context
3. **Policy Environment**: Assumes current regulatory framework
4. **Cultural Factors**: Local cultural context may influence results

---

## Quality Assurance

### Code Quality
- **Version Control**: Git-based development
- **Documentation**: Comprehensive docstrings and comments
- **Testing**: Unit tests for critical functions
- **Reproducibility**: Fixed random seeds and detailed methodology

### Data Quality
- **Validation Rules**: Automated data quality checks
- **Consistency Checks**: Cross-variable validation
- **Outlier Review**: Manual inspection of extreme values
- **Missing Data Patterns**: Analysis of missingness mechanisms

### Results Validation
- **Peer Review**: Independent validation of methodology
- **Sensitivity Analysis**: Robustness to assumption changes
- **Alternative Methods**: Comparison with different approaches
- **Domain Validation**: Expert review of findings

---

*This methodology follows best practices in data science and ensures reproducible, robust analysis suitable for policy decision-making.*
