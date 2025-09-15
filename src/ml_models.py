# =============================================================================
# MACHINE LEARNING MODELS COMPARISON MODULE (INCLUDING XGBOOST)
# =============================================================================
"""
This module implements and compares multiple machine learning algorithms
including XGBoost for both regression and classification tasks
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, accuracy_score, classification_report, mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def prepare_ml_data(df):
    """
    Prepare features and targets for machine learning
    
    Args:
        df (pd.DataFrame): Input dataframe with engineered features
        
    Returns:
        tuple: X, y_regression, y_classification
    """
    # Select optimal feature set
    feature_cols = [
        'Family_Size', 'Training_Access', 'Raw_Material_Score', 
        'Total_Resource_Score', 'Income_Per_Member', 'GI_Training_Synergy', 
        'Satisfaction_Score', 'Female_Training_Access', 'Artisan_Success_Index',
        'Economic_Vulnerability'
    ]
    
    # Handle missing values
    X = df[feature_cols].fillna(df[feature_cols].median())
    y_regression = df['Monthly_Income'].fillna(df['Monthly_Income'].median())
    y_classification = df['Is_GI_Beneficiary']
    
    return X, y_regression, y_classification

def compare_regression_models(X, y):
    """
    Compare multiple regression algorithms including XGBoost
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        
    Returns:
        dict: Model comparison results
    """
    print("📊 REGRESSION MODELS COMPARISON:")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define regression models
    regression_models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100, 
            random_state=42, 
            eval_metric='rmse'
        ),
        'Support Vector Regression': SVR(kernel='rbf', C=100, gamma=0.1)
    }
    
    regression_results = {}
    
    for name, model in regression_models.items():
        try:
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            # Fit and predict
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            regression_results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'model': model
            }
            
            print(f"   {name}:")
            print(f"     • CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"     • Test R²: {test_r2:.3f}")
            print(f"     • Test RMSE: ₹{test_rmse:.0f}")
            
        except Exception as e:
            print(f"   ❌ Error with {name}: {str(e)}")
            continue
    
    return regression_results

def compare_classification_models(X, y):
    """
    Compare multiple classification algorithms including XGBoost
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        
    Returns:
        dict: Model comparison results
    """
    print("\n🎯 CLASSIFICATION MODELS COMPARISON:")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define classification models
    classification_models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, 
            random_state=42, 
            eval_metric='logloss'
        ),
        'Support Vector Classifier': SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
    }
    
    classification_results = {}
    
    for name, model in classification_models.items():
        try:
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Fit and predict
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            
            # Predictions for detailed metrics
            y_pred = model.predict(X_test)
            
            classification_results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_acc': train_acc,
                'test_acc': test_acc,
                'model': model,
                'predictions': y_pred
            }
            
            print(f"   {name}:")
            print(f"     • CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"     • Test Accuracy: {test_acc:.3f}")
            
        except Exception as e:
            print(f"   ❌ Error with {name}: {str(e)}")
            continue
    
    return classification_results

def hyperparameter_tuning_xgboost(X, y, task='regression'):
    """
    Perform hyperparameter tuning for XGBoost
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        task (str): 'regression' or 'classification'
        
    Returns:
        dict: Best parameters and score
    """
    print(f"\n🔧 XGBOOST HYPERPARAMETER TUNING ({task.upper()}):")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    
    if task == 'regression':
        model = xgb.XGBRegressor(random_state=42, eval_metric='rmse')
        scoring = 'r2'
    else:
        model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        scoring = 'accuracy'
    
    try:
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring=scoring, 
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X, y)
        
        print(f"   • Best Parameters: {grid_search.best_params_}")
        print(f"   • Best CV Score: {grid_search.best_score_:.3f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }
        
    except Exception as e:
        print(f"   ❌ Error in hyperparameter tuning: {str(e)}")
        return None

def get_feature_importance(model, feature_names):
    """
    Extract feature importance from trained model
    
    Args:
        model: Trained sklearn/xgboost model
        feature_names (list): List of feature names
        
    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            return None
            
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
        
    except Exception as e:
        print(f"❌ Error extracting feature importance: {str(e)}")
        return None

def compare_ml_algorithms(df):
    """
    Main function to compare all machine learning algorithms
    
    Args:
        df (pd.DataFrame): Input dataframe with engineered features
        
    Returns:
        dict: Complete ML comparison results
    """
    # Prepare data
    X, y_reg, y_clf = prepare_ml_data(df)
    
    # Compare regression models
    reg_results = compare_regression_models(X, y_reg)
    
    # Compare classification models
    clf_results = compare_classification_models(X, y_clf)
    
    # XGBoost hyperparameter tuning
    xgb_reg_tuned = hyperparameter_tuning_xgboost(X, y_reg, 'regression')
    xgb_clf_tuned = hyperparameter_tuning_xgboost(X, y_clf, 'classification')
    
    # Get feature importance from best model
    best_reg_model = max(reg_results.items(), key=lambda x: x[1]['test_r2'])
    feature_importance = get_feature_importance(best_reg_model[1]['model'], X.columns.tolist())
    
    print(f"\n🏆 BEST MODELS:")
    print(f"   • Best Regression: {best_reg_model[0]} (R² = {best_reg_model[1]['test_r2']:.3f})")
    
    if clf_results:
        best_clf_model = max(clf_results.items(), key=lambda x: x[1]['test_acc'])
        print(f"   • Best Classification: {best_clf_model[0]} (Acc = {best_clf_model[1]['test_acc']:.3f})")
    
    return {
        'regression': reg_results,
        'classification': clf_results,
        'xgb_tuned_reg': xgb_reg_tuned,
        'xgb_tuned_clf': xgb_clf_tuned,
        'feature_importance': feature_importance,
        'best_regression': best_reg_model,
        'best_classification': best_clf_model if clf_results else None
    }
