import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance
import logging
import warnings
from typing import Dict, Any, Tuple, List, Optional, Union
import time
import os
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from config import Config
from logger import LogManager
import requests
import traceback
from contextlib import contextmanager

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class APIClient:
    """Centralized API client with multiple provider support"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.active_client = None
        self.client_type = None
        self._initialize_clients()
    
    def _initialize_clients(self) -> None:
        """Initialize available API clients in order of preference"""
        clients_to_try = [
            ('gemini', self._setup_gemini_client),
            ('fallback', self._setup_fallback_client)
        ]
        
        for client_type, setup_func in clients_to_try:
            try:
                client = setup_func()
                if client:
                    self.active_client = client
                    self.client_type = client_type
                    self.logger.info(f"Successfully initialized {client_type} API client")
                    break
            except Exception as e:
                self.logger.warning(f"Failed to initialize {client_type} client: {str(e)}")
        
        if not self.active_client:
            self.logger.warning("No API clients available, using rule-based explanations only")
    
    def _setup_gemini_client(self) -> Optional[Dict[str, Any]]:
        """Setup Google Gemini API client"""
        try:
            import google.generativeai as genai
            
            if not self.config.GEMINI_API_KEY:
                self.logger.warning("Gemini API key not provided")
                return None
            
            genai.configure(api_key=self.config.GEMINI_API_KEY)
            
            # Try different model names
            model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
            
            for model_name in model_names:
                try:
                    model = genai.GenerativeModel(model_name)
                    
                    # Test the connection with a simple prompt
                    test_response = model.generate_content("Hello, respond with 'API working'")
                    if test_response and test_response.text and 'API working' in test_response.text:
                        self.logger.info(f"Gemini API test successful with model: {model_name}")
                        return {
                            'type': 'gemini',
                            'client': model,
                            'model_name': model_name
                        }
                except Exception as e:
                    self.logger.debug(f"Model {model_name} failed: {str(e)}")
                    continue
            
            self.logger.warning("All Gemini models failed to initialize")
            return None
            
        except ImportError:
            self.logger.info("google-generativeai package not installed")
            return None
        except Exception as e:
            self.logger.error(f"Gemini client setup failed: {str(e)}")
            return None
    
    def _setup_fallback_client(self) -> Dict[str, Any]:
        """Setup fallback rule-based client"""
        return {
            'type': 'fallback',
            'available': True
        }
    
    def generate_explanation(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Generate explanation using available API client"""
        if not self.active_client:
            return self._generate_rule_based_explanation(prompt)
        
        for attempt in range(max_retries):
            try:
                if self.client_type == 'gemini':
                    return self._call_gemini(prompt)
                else:
                    return self._generate_rule_based_explanation(prompt)
                    
            except Exception as e:
                self.logger.warning(f"API call attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(self.config.BASE_RETRY_DELAY * (2 ** attempt))
        
        # Final fallback
        return self._generate_rule_based_explanation(prompt)
    
    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Call Google Gemini API"""
        try:
            response = self.active_client['client'].generate_content(prompt)
            if response and response.text:
                return response.text.strip()
            return None
        except Exception as e:
            self.logger.error(f"Gemini API call failed: {str(e)}")
            raise
    
    def _generate_rule_based_explanation(self, prompt: str) -> str:
        """Generate rule-based explanation as ultimate fallback"""
        if "regression" in prompt.lower():
            return """
**Model Analysis Summary**

This regression model predicts continuous numerical values based on input features. The analysis reveals key patterns in how different features influence the predictions.

**Key Insights:**
- The model uses multiple features with varying degrees of influence
- Feature importance analysis identifies the primary prediction drivers
- Some features show stronger correlation with the target variable

**Business Recommendations:**
1. Focus on monitoring the most influential features identified
2. Ensure data quality for high-importance features
3. Consider feature interactions when making decisions
4. Use insights to guide resource allocation and strategy

This analysis provides a foundation for understanding model behavior and making informed business decisions.
"""
        else:
            return """
**Model Analysis Summary**

This classification model categorizes inputs into different classes based on feature patterns. The analysis shows which characteristics are most decisive for classification.

**Key Insights:**
- The model examines multiple features to determine class membership
- Feature importance reveals key discriminating factors
- Classification patterns indicate underlying data relationships

**Business Recommendations:**
1. Prioritize data quality for the most important features
2. Monitor classification patterns for business insights
3. Use feature importance to guide process improvements
4. Consider model predictions in strategic decision-making

This analysis helps understand the classification logic and supports data-driven business decisions.
"""


class SimpleExplainer:
    """Enhanced explainer with improved error handling and API integration"""
    
    def __init__(self):
        self.config = Config()
        self.log_manager = LogManager(self.config)
        self.logger = self.log_manager.create_child_logger(__name__)
        self.api_client = APIClient(self.config, self.logger)
        
        # Performance tracking
        self.performance_metrics = {}
    
    @contextmanager
    def performance_tracker(self, operation: str):
        """Context manager to track operation performance"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.performance_metrics[operation] = duration
            self.log_manager.log_performance(operation, duration)
    
    def safe_predict(self, model, data: pd.DataFrame) -> np.ndarray:
        """Safely predict with comprehensive error handling"""
        try:
            with self.performance_tracker("model_prediction"):
                # Ensure data is numeric and clean
                if not isinstance(data, pd.DataFrame):
                    data = pd.DataFrame(data)
                
                # Handle different prediction methods
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(data)
                        if proba.shape[1] == 2:
                            return proba[:, 1]  # Return probability of positive class
                        else:
                            return proba  # Return all class probabilities
                    except Exception as e:
                        self.logger.warning(f"predict_proba failed: {e}, falling back to predict")
                        return model.predict(data)
                else:
                    return model.predict(data)
                    
        except Exception as e:
            self.logger.error(f"All prediction methods failed: {e}")
            # Return zeros as absolute fallback
            return np.zeros(len(data))
    
    def get_model_feature_names(self, model) -> Optional[List[str]]:
        """Extract feature names from model with multiple fallbacks"""
        try:
            # Try different attributes where feature names might be stored
            for attr in ['feature_names_in_', 'feature_names_', 'feature_name_', 
                        'booster.feature_names', '_feature_names']:
                try:
                    if '.' in attr:
                        # Handle nested attributes
                        obj = model
                        for part in attr.split('.'):
                            obj = getattr(obj, part)
                        feature_names = obj
                    else:
                        feature_names = getattr(model, attr)
                    
                    if feature_names is not None:
                        # Convert to list if needed
                        if hasattr(feature_names, 'tolist'):
                            return feature_names.tolist()
                        elif isinstance(feature_names, (list, tuple)):
                            return list(feature_names)
                        else:
                            return [str(name) for name in feature_names]
                            
                except AttributeError:
                    continue
                    
            self.logger.debug("No feature names found in model")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to extract feature names: {e}")
            return None

    def align_features(self, df: pd.DataFrame, model_feature_names: List[str]) -> pd.DataFrame:
        """Align dataset features with model's expected features"""
        if model_feature_names is None:
            return df
        
        try:
            df_aligned = df.copy()
            
            # Find missing and extra features
            current_features = set(df_aligned.columns)
            expected_features = set(model_feature_names)
            
            missing_features = expected_features - current_features
            extra_features = current_features - expected_features
            
            if missing_features:
                self.logger.warning(f"Missing features: {missing_features}")
                # Add missing features with zeros
                for feature in missing_features:
                    df_aligned[feature] = 0.0
            
            if extra_features:
                self.logger.warning(f"Extra features (will be ignored): {extra_features}")
            
            # Reorder columns to match model expectations
            try:
                df_aligned = df_aligned[model_feature_names]
                self.logger.info(f"Successfully aligned {len(model_feature_names)} features")
                return df_aligned
            except KeyError as e:
                self.logger.error(f"Feature alignment failed: {e}")
                return df
                
        except Exception as e:
            self.logger.error(f"Feature alignment error: {e}")
            return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data cleaning with improved handling"""
        try:
            with self.performance_tracker("data_cleaning"):
                df_clean = df.copy()
                original_shape = df_clean.shape
                
                # Handle missing values
                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
                
                # Fill numeric missing values
                for col in numeric_cols:
                    if df_clean[col].isnull().any():
                        # Use median for numeric, fallback to mean, then 0
                        fill_value = df_clean[col].median()
                        if pd.isna(fill_value):
                            fill_value = df_clean[col].mean()
                        if pd.isna(fill_value):
                            fill_value = 0.0
                        df_clean[col].fillna(fill_value, inplace=True)
                
                # Handle categorical columns
                for col in categorical_cols:
                    if df_clean[col].isnull().any():
                        mode_values = df_clean[col].mode()
                        fill_value = mode_values[0] if len(mode_values) > 0 else 'Unknown'
                        df_clean[col].fillna(fill_value, inplace=True)
                    
                    # Encode categorical variables
                    unique_values = df_clean[col].unique()
                    if len(unique_values) > self.config.MAX_CATEGORICAL_CARDINALITY:
                        # Keep only top categories for high cardinality features
                        top_categories = df_clean[col].value_counts().head(
                            self.config.TOP_CATEGORIES_LIMIT
                        ).index.tolist()
                        df_clean[col] = df_clean[col].apply(
                            lambda x: x if x in top_categories else 'Other'
                        )
                        unique_values = df_clean[col].unique()
                    
                    # Create mapping for categorical encoding
                    mapping = {val: i for i, val in enumerate(sorted(unique_values))}
                    df_clean[col] = df_clean[col].map(mapping)
                
                # Handle infinite and extreme values
                df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
                df_clean = df_clean.fillna(0)
                
                # Ensure all columns are numeric
                df_clean = df_clean.select_dtypes(include=[np.number])
                
                self.logger.info(f"Data cleaning completed: {original_shape} -> {df_clean.shape}")
                return df_clean
                
        except Exception as e:
            self.logger.error(f"Data cleaning failed: {e}")
            # Return original dataframe as fallback
            return df.select_dtypes(include=[np.number]).fillna(0)
    
    def detect_model_type(self, model, predictions: np.ndarray) -> str:
        """Detect if model is classification or regression"""
        try:
            # Check for classification-specific methods
            if hasattr(model, 'predict_proba') or hasattr(model, 'predict_log_proba'):
                return 'classification'
            
            # Analyze predictions
            unique_preds = np.unique(predictions)
            
            # If few unique values and all integers, likely classification
            if (len(unique_preds) <= 20 and 
                all(isinstance(x, (int, np.integer)) or x.is_integer() 
                    for x in unique_preds[:min(10, len(unique_preds))])):
                return 'classification'
            
            return 'regression'
            
        except Exception as e:
            self.logger.warning(f"Model type detection failed: {e}, assuming regression")
            return 'regression'
    
    def generate_shap_explanation(self, model, df_sample: pd.DataFrame) -> Dict[str, Any]:
        """Generate SHAP explanation with multiple explainer fallbacks"""
        try:
            with self.performance_tracker("shap_explanation"):
                shap_explainer = None
                
                # Try TreeExplainer for tree-based models
                if any(tree_indicator in str(type(model)).lower() 
                      for tree_indicator in ['tree', 'forest', 'xgb', 'lgb', 'catboost']):
                    try:
                        shap_explainer = shap.TreeExplainer(model)
                        self.logger.info("Using SHAP TreeExplainer")
                    except Exception as e:
                        self.logger.debug(f"TreeExplainer failed: {e}")
                
                # Fallback to KernelExplainer
                if shap_explainer is None:
                    try:
                        background_size = min(self.config.BACKGROUND_SIZE_KERNEL, len(df_sample))
                        background_data = df_sample.iloc[:background_size]
                        shap_explainer = shap.KernelExplainer(
                            lambda x: self.safe_predict(model, pd.DataFrame(x, columns=df_sample.columns)), 
                            background_data
                        )
                        self.logger.info("Using SHAP KernelExplainer")
                    except Exception as e:
                        self.logger.debug(f"KernelExplainer failed: {e}")
                
                # Final fallback to generic Explainer
                if shap_explainer is None:
                    background_size = min(50, len(df_sample))
                    shap_explainer = shap.Explainer(
                        lambda x: self.safe_predict(model, pd.DataFrame(x, columns=df_sample.columns)),
                        df_sample.iloc[:background_size]
                    )
                    self.logger.info("Using SHAP generic Explainer")
                
                # Generate SHAP values
                sample_size = min(self.config.MAX_SAMPLES_SHAP, len(df_sample))
                shap_values = shap_explainer.shap_values(df_sample.iloc[:sample_size])
                
                # Handle different SHAP value formats
                if isinstance(shap_values, list):
                    # Multi-class classification or multiple outputs
                    shap_values = shap_values[-1] if len(shap_values) > 1 else shap_values[0]
                
                return {
                    "global_feature_importance": np.abs(shap_values).mean(axis=0).tolist(),
                    "local_explanations": shap_values[:self.config.LOCAL_EXPLANATION_SAMPLES].tolist(),
                    "feature_names": df_sample.columns.tolist(),
                    "status": "success",
                    "explainer_type": type(shap_explainer).__name__
                }
                
        except Exception as e:
            self.log_manager.log_exception(e, "SHAP explanation generation")
            return {"status": "failed", "error": str(e)}
    
    def generate_lime_explanation(self, model, df_sample: pd.DataFrame, model_type: str) -> Dict[str, Any]:
        """Generate LIME explanation with improved error handling"""
        try:
            with self.performance_tracker("lime_explanation"):
                lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=df_sample.values,
                    feature_names=df_sample.columns.tolist(),
                    mode=model_type,
                    discretize_continuous=True,
                    random_state=42
                )
                
                lime_explanations = []
                sample_size = min(self.config.MAX_SAMPLES_LIME, len(df_sample))
                
                for i in range(sample_size):
                    try:
                        instance = df_sample.iloc[i].values
                        exp = lime_explainer.explain_instance(
                            instance,
                            lambda x: self.safe_predict(
                                model, 
                                pd.DataFrame(x, columns=df_sample.columns)
                            ) if len(x.shape) > 1 else 
                            self.safe_predict(
                                model, 
                                pd.DataFrame(x.reshape(1, -1), columns=df_sample.columns)
                            )[0],
                            num_features=min(self.config.TOP_FEATURES_DISPLAY, len(df_sample.columns))
                        )
                        lime_explanations.append(dict(exp.as_list()))
                    except Exception as e:
                        self.logger.warning(f"LIME explanation failed for instance {i}: {e}")
                        lime_explanations.append({"error": f"Instance {i} failed: {str(e)}"})
                
                return {
                    "explanations": lime_explanations,
                    "feature_names": df_sample.columns.tolist(),
                    "status": "success"
                }
                
        except Exception as e:
            self.log_manager.log_exception(e, "LIME explanation generation")
            return {"status": "failed", "error": str(e)}
    
    def generate_permutation_importance(self, model, df_sample: pd.DataFrame, 
                                       predictions: np.ndarray, model_type: str) -> Dict[str, Any]:
        """Generate permutation importance with proper scoring"""
        try:
            with self.performance_tracker("permutation_importance"):
                sample_size = min(self.config.MAX_SAMPLES_PERMUTATION, len(df_sample))
                
                # Select appropriate scoring method
                scoring_method = 'neg_mean_squared_error' if model_type == 'regression' else 'accuracy'
                
                perm_importance = permutation_importance(
                    model,
                    df_sample.iloc[:sample_size],
                    predictions[:sample_size],
                    n_repeats=3,
                    random_state=42,
                    scoring=scoring_method
                )
                
                # Create ranked feature list
                feature_importance_pairs = list(zip(
                    df_sample.columns.tolist(),
                    perm_importance.importances_mean.tolist(),
                    perm_importance.importances_std.tolist()
                ))
                feature_importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
                
                return {
                    "global_feature_importance": perm_importance.importances_mean.tolist(),
                    "std": perm_importance.importances_std.tolist(),
                    "feature_names": df_sample.columns.tolist(),
                    "ranked_features": [
                        {"feature": name, "importance": imp, "std": std}
                        for name, imp, std in feature_importance_pairs
                    ],
                    "status": "success",
                    "scoring_method": scoring_method
                }
                
        except Exception as e:
            self.log_manager.log_exception(e, "Permutation importance generation")
            return {"status": "failed", "error": str(e)}
    
    def generate_feature_interactions(self, explanations: Dict[str, Any], 
                                    df_sample: pd.DataFrame) -> Dict[str, Any]:
        """Generate feature interaction analysis from SHAP values"""
        try:
            shap_data = explanations.get('shap', {})
            if shap_data.get('status') != 'success':
                return {"status": "failed", "error": "Requires successful SHAP explanation"}
            
            shap_values = np.array(shap_data['local_explanations'])
            if len(shap_values) == 0 or shap_values.shape[1] < 2:
                return {"status": "failed", "error": "Insufficient SHAP values for interaction analysis"}
            
            feature_interactions = []
            n_features = min(self.config.TOP_FEATURES_DISPLAY, shap_values.shape[1])
            feature_names = df_sample.columns.tolist()
            
            # Calculate pairwise interactions
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    # Interaction strength as correlation between SHAP values
                    interaction_strength = np.abs(np.corrcoef(shap_values[:, i], shap_values[:, j])[0, 1])
                    
                    if not np.isnan(interaction_strength):
                        feature_interactions.append({
                            'feature_1': feature_names[i],
                            'feature_2': feature_names[j],
                            'interaction_strength': round(float(interaction_strength), 6)
                        })
            
            # Sort by interaction strength
            feature_interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
            
            return {
                "top_interactions": feature_interactions[:self.config.MAX_FEATURE_INTERACTIONS],
                "description": "Feature pairs with strongest interactions based on SHAP value correlations",
                "status": "success"
            }
            
        except Exception as e:
            self.log_manager.log_exception(e, "Feature interaction analysis")
            return {"status": "failed", "error": str(e)}
    
    def calculate_fidelity_scores(self, explanations: Dict[str, Any], 
                                 predictions: np.ndarray) -> Dict[str, Union[float, str, None]]:
        """Calculate fidelity scores for different explanation methods"""
        fidelity_scores = {}
        
        try:
            for method_name in ['shap', 'lime', 'permutation_importance']:
                method_data = explanations.get(method_name, {})
                
                if method_data.get('status') != 'success':
                    fidelity_scores[method_name] = None
                    continue
                
                if method_name == 'shap' and 'local_explanations' in method_data:
                    # Calculate SHAP fidelity
                    try:
                        local_values = np.array(method_data['local_explanations'])
                        if len(local_values) > 0:
                            predicted_sum = local_values.sum(axis=1)
                            actual_preds = predictions[:len(predicted_sum)]
                            
                            if len(actual_preds) > 0 and np.std(actual_preds) > 1e-6:
                                fidelity = 1 - (np.mean(np.abs(predicted_sum - actual_preds)) / 
                                              (np.std(actual_preds) + 1e-6))
                                fidelity_scores[method_name] = max(0, min(1, float(fidelity)))
                            else:
                                fidelity_scores[method_name] = 0.0
                        else:
                            fidelity_scores[method_name] = None
                    except Exception as e:
                        self.logger.warning(f"SHAP fidelity calculation failed: {e}")
                        fidelity_scores[method_name] = None
                
                elif method_name == 'lime':
                    # LIME fidelity is harder to calculate, use a proxy
                    fidelity_scores[method_name] = "Based on local model fit"
                
                elif method_name == 'permutation_importance':
                    # Calculate consistency as fidelity proxy
                    try:
                        importance_values = method_data.get('global_feature_importance', [])
                        std_values = method_data.get('std', [])
                        
                        if importance_values and std_values:
                            mean_importance = np.mean(np.abs(importance_values))
                            mean_std = np.mean(std_values)
                            
                            if mean_importance > 1e-6:
                                consistency = 1 - (mean_std / (mean_importance + 1e-6))
                                fidelity_scores[method_name] = max(0, min(1, float(consistency)))
                            else:
                                fidelity_scores[method_name] = 0.0
                        else:
                            fidelity_scores[method_name] = None
                    except Exception as e:
                        self.logger.warning(f"Permutation importance fidelity calculation failed: {e}")
                        fidelity_scores[method_name] = None
        
        except Exception as e:
            self.log_manager.log_exception(e, "Fidelity score calculation")
            fidelity_scores["error"] = str(e)
        
        return fidelity_scores
    
    def generate_explanations(self, model, df: pd.DataFrame, 
                            predictions: np.ndarray) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        """Generate comprehensive model explanations"""
        explanations = {}
        
        try:
            with self.performance_tracker("total_explanation_generation"):
                # Get model feature names and align data
                model_feature_names = self.get_model_feature_names(model)
                df_clean = self.clean_data(df)
                
                # Align features if model has expected feature names
                if model_feature_names is not None:
                    df_clean = self.align_features(df_clean, model_feature_names)
                
                # Detect model type
                model_type = self.detect_model_type(model, predictions)
                self.logger.info(f"Detected model type: {model_type}")
                
                # Limit samples for performance
                max_samples = min(self.config.MAX_SAMPLES_PROCESSING, len(df_clean))
                df_sample = df_clean.iloc[:max_samples].copy()
                predictions_sample = predictions[:max_samples]
                
                self.logger.info(f"Processing {max_samples} samples with {len(df_sample.columns)} features")
                
                # Generate SHAP explanation
                explanations['shap'] = self.generate_shap_explanation(model, df_sample)
                
                # Generate LIME explanation
                explanations['lime'] = self.generate_lime_explanation(model, df_sample, model_type)
                
                # Generate Permutation Importance
                explanations['permutation_importance'] = self.generate_permutation_importance(
                    model, df_sample, predictions_sample, model_type
                )
                
                # Generate Feature Interactions
                explanations['feature_interaction'] = self.generate_feature_interactions(
                    explanations, df_sample
                )
                
                # Calculate fidelity scores
                fidelity_scores = self.calculate_fidelity_scores(explanations, predictions_sample)
                
                # Generate business explanation
                business_explanation = self._generate_business_explanation(
                    explanations, model_type, predictions_sample, df_sample, fidelity_scores
                )
                
                # Log performance metrics
                self.logger.info(f"Explanation generation completed. Performance metrics: {self.performance_metrics}")
                
                return explanations, business_explanation, fidelity_scores
                
        except Exception as e:
            self.log_manager.log_exception(e, "Explanation generation")
            return {}, f"Explanation generation failed: {str(e)}", {}

    def _generate_business_explanation(self, explanations: Dict[str, Any], model_type: str,
                                     predictions: np.ndarray, df_sample: pd.DataFrame, 
                                     fidelity_scores: Dict[str, Any]) -> str:
        """Generate business-friendly explanation using AI or fallback"""
        try:
            # Create summary for AI prompt
            summary_parts = [
                f"Model Type: {model_type.capitalize()}",
                f"Samples Analyzed: {len(predictions)}",
                f"Features: {len(df_sample.columns)}"
            ]
            
            # Add top features from SHAP if available
            shap_data = explanations.get('shap', {})
            if shap_data.get('status') == 'success':
                importance = shap_data.get('global_feature_importance', [])
                feature_names = shap_data.get('feature_names', [])
                
                if importance and feature_names:
                    # Get top 3 features
                    top_indices = np.argsort(np.abs(importance))[-3:][::-1]
                    top_features = [
                        f"{feature_names[i]} ({importance[i]:.3f})" 
                        for i in top_indices if i < len(feature_names)
                    ]
                    summary_parts.append(f"Top Features: {', '.join(top_features)}")
            
            # Add prediction statistics
            if len(predictions) > 0:
                if model_type == 'classification':
                    if len(np.unique(predictions)) == 2:
                        pos_rate = np.mean(predictions)
                        summary_parts.append(f"Positive Rate: {pos_rate:.1%}")
                    else:
                        unique_classes = len(np.unique(predictions))
                        summary_parts.append(f"Classes: {unique_classes}")
                else:
                    pred_mean = np.mean(predictions)
                    pred_std = np.std(predictions)
                    summary_parts.append(f"Prediction Range: {pred_mean:.2f} ± {pred_std:.2f}")
            
            summary_text = " | ".join(summary_parts)
            
            # Create AI prompt
            prompt = f"""Explain this machine learning model analysis in clear business terms:

{summary_text}

Provide a concise explanation covering:
1. What the model predicts and key business insights
2. Which features are most important and why they matter
3. Practical business recommendations

Focus on actionable insights for business decision-making. Keep it under 300 words."""
            
            # Try AI explanation first
            ai_explanation = self.api_client.generate_explanation(prompt)
            
            if ai_explanation and len(ai_explanation.strip()) > 50:
                return ai_explanation
            else:
                # Fallback to rule-based explanation
                return self._generate_simple_explanation(
                    explanations, model_type, predictions, df_sample
                )
                
        except Exception as e:
            self.log_manager.log_exception(e, "Business explanation generation")
            return self._generate_simple_explanation(
                explanations, model_type, predictions, df_sample
            )
    
    def _generate_simple_explanation(self, explanations: Dict[str, Any], model_type: str,
                                   predictions: np.ndarray, df_sample: pd.DataFrame) -> str:
        """Generate simple rule-based explanation"""
        try:
            # Basic model description
            if model_type == 'classification':
                unique_preds = np.unique(predictions)
                if len(unique_preds) == 2:
                    positive_rate = np.mean(predictions)
                    explanation = f"""
**Model Overview**
This binary classification model predicts outcomes with a {positive_rate:.1%} positive rate across the analyzed samples.

**Key Insights**
The model analyzes {len(df_sample.columns)} features to make predictions. """
                else:
                    explanation = f"""
**Model Overview**
This multi-class classification model categorizes data into {len(unique_preds)} different classes.

**Key Insights**
The model evaluates {len(df_sample.columns)} features for classification decisions. """
            else:
                pred_mean = np.mean(predictions)
                pred_std = np.std(predictions)
                explanation = f"""
**Model Overview**
This regression model predicts continuous values with an average of {pred_mean:.2f} and standard deviation of {pred_std:.2f}.

**Key Insights**
The model uses {len(df_sample.columns)} features to generate numerical predictions. """
            
            # Add top features if available
            shap_data = explanations.get('shap', {})
            if shap_data.get('status') == 'success':
                importance = shap_data.get('global_feature_importance', [])
                feature_names = shap_data.get('feature_names', [])
                
                if importance and feature_names:
                    top_indices = np.argsort(np.abs(importance))[-3:][::-1]
                    top_features = [(feature_names[i], importance[i]) for i in top_indices if i < len(feature_names)]
                    
                    explanation += "\n\n**Most Important Features:**\n"
                    for name, imp in top_features:
                        explanation += f"- {name}: {abs(imp):.3f}\n"
            
            explanation += "\n\n**Business Recommendations:**\n"
            explanation += "1. Focus on the most influential features for decision-making\n"
            explanation += "2. Monitor data quality for high-importance features\n"
            explanation += "3. Use these insights to guide strategic planning\n"
            explanation += "4. Consider feature interactions when making business decisions"
            
            return explanation.strip()
            
        except Exception as e:
            self.logger.error(f"Simple explanation generation failed: {e}")
            return "A business explanation could not be generated due to processing issues. Please refer to the technical analysis results above."


def create_visualization_charts(explanations: Dict[str, Any], df: pd.DataFrame) -> Dict[str, go.Figure]:
    """Create interactive visualizations for explanations"""
    charts = {}
    
    try:
        # SHAP Global Importance Chart
        shap_data = explanations.get('shap', {})
        if shap_data.get('status') == 'success':
            importance = shap_data.get('global_feature_importance', [])
            feature_names = shap_data.get('feature_names', [])
            
            if importance and feature_names:
                shap_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': np.abs(importance)
                }).sort_values('Importance', ascending=False).head(10)
                
                charts['shap_global'] = px.bar(
                    shap_df, x='Importance', y='Feature', orientation='h',
                    title="SHAP Global Feature Importance",
                    labels={'Importance': 'Absolute SHAP Value', 'Feature': 'Features'}
                )
                charts['shap_global'].update_layout(height=400)
        
        # Permutation Importance Chart
        perm_data = explanations.get('permutation_importance', {})
        if perm_data.get('status') == 'success' and 'ranked_features' in perm_data:
            perm_df = pd.DataFrame(perm_data['ranked_features']).head(10)
            
            charts['permutation'] = px.bar(
                perm_df, x='importance', y='feature', orientation='h',
                error_x='std',
                title="Permutation Feature Importance",
                labels={'importance': 'Importance Score', 'feature': 'Features'}
            )
            charts['permutation'].update_layout(height=400)
        
        # Feature Interactions Chart
        interaction_data = explanations.get('feature_interaction', {})
        if interaction_data.get('status') == 'success' and 'top_interactions' in interaction_data:
            interactions = interaction_data['top_interactions'][:5]
            
            if interactions:
                interaction_df = pd.DataFrame(interactions)
                interaction_df['Pair'] = (interaction_df['feature_1'] + 
                                        " ↔ " + interaction_df['feature_2'])
                
                charts['interactions'] = px.bar(
                    interaction_df, x='interaction_strength', y='Pair', orientation='h',
                    title="Top Feature Interactions",
                    labels={'interaction_strength': 'Interaction Strength', 'Pair': 'Feature Pairs'}
                )
                charts['interactions'].update_layout(height=300)
        
        # Dataset Overview Chart
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
        if len(numeric_cols) > 0:
            sample_data = df[numeric_cols].head(100)
            charts['data_overview'] = px.box(
                sample_data, 
                title="Dataset Feature Distributions (Sample)",
                labels={'value': 'Feature Values', 'variable': 'Features'}
            )
            charts['data_overview'].update_layout(height=400)
    
    except Exception as e:
        logging.error(f"Chart creation failed: {e}")
    
    return charts


def create_app():
    """Create and configure the Streamlit application"""
    try:
        st.set_page_config(
            page_title="Enhanced Explainable AI Dashboard",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize explainer
        explainer = SimpleExplainer()
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        .metric-container {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }
        .status-success {
            color: #28a745;
            font-weight: bold;
        }
        .status-error {
            color: #dc3545;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🤖 Enhanced Explainable AI Dashboard</h1>
            <p>Upload your model and data to generate comprehensive AI explanations</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # API Status
            st.subheader("API Status")
            if explainer.api_client.active_client:
                st.success(f"✅ {explainer.api_client.client_type.title()} API Active")
            else:
                st.warning("⚠️ Using Fallback Explanations")
            
            # Performance Settings
            st.subheader("Performance Settings")
            max_samples = st.slider(
                "Max Samples to Process", 
                min_value=100, 
                max_value=2000, 
                value=explainer.config.MAX_SAMPLES_PROCESSING
            )
            explainer.config.MAX_SAMPLES_PROCESSING = max_samples
            
            # Model Support Info
            st.subheader("Supported Formats")
            st.write("**Models:** .pkl, .joblib")
            st.write("**Data:** .csv")
            st.write("**Types:** sklearn, xgboost, lightgbm, catboost")
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Main Dashboard", 
            "📈 Visualizations", 
            "📝 Analysis Logs", 
            "ℹ️ System Info"
        ])
        
        with tab1:
            # File upload section
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📁 Upload Model")
                model_file = st.file_uploader(
                    "Choose model file", 
                    type=["pkl", "joblib"],
                    help="Upload a trained scikit-learn compatible model"
                )
                
                if model_file:
                    st.success(f"✅ Model uploaded: {model_file.name}")
            
            with col2:
                st.subheader("📁 Upload Dataset")
                data_file = st.file_uploader(
                    "Choose CSV file", 
                    type=["csv"],
                    help="Upload the dataset used for predictions"
                )
                
                if data_file:
                    st.success(f"✅ Data uploaded: {data_file.name}")
            
            # Process files if both are uploaded
            if model_file and data_file:
                try:
                    # Load files with progress indication
                    with st.spinner("Loading model and dataset..."):
                        model = joblib.load(model_file)
                        df = pd.read_csv(data_file)
                    
                    # Validation
                    if df.empty:
                        st.error("❌ The uploaded dataset is empty.")
                        st.stop()
                    
                    if not hasattr(model, 'predict'):
                        st.error("❌ Invalid model: must have a 'predict' method.")
                        st.stop()
                    
                    # Dataset info
                    st.subheader("📋 Dataset Overview")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Samples", len(df))
                    with col2:
                        st.metric("Features", len(df.columns))
                    with col3:
                        st.metric("Missing Values", df.isnull().sum().sum())
                    with col4:
                        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                    
                    # Show dataset preview
                    with st.expander("🔍 Dataset Preview", expanded=False):
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Data Types:**")
                            st.write(df.dtypes.value_counts())
                        with col2:
                            st.write("**Missing Values by Column:**")
                            missing_data = df.isnull().sum()
                            missing_data = missing_data[missing_data > 0]
                            if not missing_data.empty:
                                st.write(missing_data)
                            else:
                                st.write("No missing values found")
                    
                    # Generate predictions
                    st.subheader("🔮 Generating Predictions")
                    with st.spinner("Making predictions..."):
                        # Prepare data for prediction
                        model_feature_names = explainer.get_model_feature_names(model)
                        df_for_prediction = explainer.clean_data(df)
                        
                        if model_feature_names:
                            df_for_prediction = explainer.align_features(df_for_prediction, model_feature_names)
                        
                        predictions = explainer.safe_predict(model, df_for_prediction)
                        model_type = explainer.detect_model_type(model, predictions)
                    
                    # Prediction summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model Type", model_type.title())
                    with col2:
                        st.metric("Predictions Generated", len(predictions))
                    with col3:
                        if model_type == 'classification':
                            unique_preds = len(np.unique(predictions))
                            st.metric("Unique Classes", unique_preds)
                        else:
                            pred_range = np.max(predictions) - np.min(predictions)
                            st.metric("Prediction Range", f"{pred_range:.3f}")
                    
                    # Show sample predictions
                    with st.expander("📊 Sample Predictions", expanded=False):
                        pred_df = pd.DataFrame({
                            'Index': range(min(10, len(predictions))),
                            'Prediction': predictions[:10]
                        })
                        st.dataframe(pred_df, use_container_width=True)
                        
                        if model_type == 'classification':
                            st.write("**Class Distribution:**")
                            class_counts = pd.Series(predictions).value_counts().sort_index()
                            st.bar_chart(class_counts)
                        else:
                            st.write("**Prediction Distribution:**")
                            fig = px.histogram(predictions, title="Prediction Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Generate explanations
                    st.subheader("🔍 Generating Explanations")
                    
                    with st.spinner("Generating comprehensive explanations... This may take a few minutes."):
                        explanations, business_explanation, fidelity_scores = explainer.generate_explanations(
                            model, df, predictions
                        )
                    
                    # Display explanations
                    st.subheader("📊 Model Explanations")
                    
                    # Explanation method tabs
                    exp_tab1, exp_tab2, exp_tab3, exp_tab4 = st.tabs([
                        "🎯 SHAP Analysis", 
                        "🔬 LIME Analysis", 
                        "📈 Permutation Importance", 
                        "🔗 Feature Interactions"
                    ])
                    
                    with exp_tab1:
                        shap_data = explanations.get('shap', {})
                        if shap_data.get('status') == 'success':
                            st.success("✅ SHAP Analysis Completed")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Global Feature Importance**")
                                importance = shap_data.get('global_feature_importance', [])
                                feature_names = shap_data.get('feature_names', [])
                                
                                if importance and feature_names:
                                    shap_df = pd.DataFrame({
                                        'Feature': feature_names,
                                        'Importance': np.abs(importance)
                                    }).sort_values('Importance', ascending=False).head(10)
                                    
                                    fig = px.bar(shap_df, x='Importance', y='Feature', 
                                               orientation='h', title="SHAP Feature Importance")
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.write("**Local Explanations Sample**")
                                local_explanations = shap_data.get('local_explanations', [])
                                if local_explanations:
                                    local_df = pd.DataFrame(
                                        local_explanations[:3], 
                                        columns=feature_names
                                    )
                                    st.dataframe(local_df, use_container_width=True)
                            
                            # Additional SHAP info
                            explainer_type = shap_data.get('explainer_type', 'Unknown')
                            st.info(f"Used explainer: {explainer_type}")
                        
                        else:
                            st.error(f"❌ SHAP Analysis Failed: {shap_data.get('error', 'Unknown error')}")
                    
                    with exp_tab2:
                        lime_data = explanations.get('lime', {})
                        if lime_data.get('status') == 'success':
                            st.success("✅ LIME Analysis Completed")
                            
                            lime_explanations = lime_data.get('explanations', [])
                            if lime_explanations and len(lime_explanations) > 0:
                                first_exp = lime_explanations[0]
                                if isinstance(first_exp, dict) and 'error' not in first_exp:
                                    lime_df = pd.DataFrame(
                                        list(first_exp.items()),
                                        columns=['Feature', 'Contribution']
                                    ).sort_values('Contribution', key=abs, ascending=False)
                                    
                                    fig = px.bar(lime_df.head(10), x='Contribution', y='Feature',
                                               orientation='h', title="LIME Feature Contributions (First Instance)")
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    st.write("**All Instance Explanations:**")
                                    for i, exp in enumerate(lime_explanations[:3]):
                                        if isinstance(exp, dict) and 'error' not in exp:
                                            st.write(f"Instance {i+1}: {len(exp)} features explained")
                                else:
                                    st.warning("LIME explanation contains errors")
                        else:
                            st.error(f"❌ LIME Analysis Failed: {lime_data.get('error', 'Unknown error')}")
                    
                    with exp_tab3:
                        perm_data = explanations.get('permutation_importance', {})
                        if perm_data.get('status') == 'success':
                            st.success("✅ Permutation Importance Completed")
                            
                            ranked_features = perm_data.get('ranked_features', [])
                            if ranked_features:
                                perm_df = pd.DataFrame(ranked_features).head(10)
                                
                                fig = px.bar(perm_df, x='importance', y='feature',
                                           orientation='h', error_x='std',
                                           title="Permutation Feature Importance")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.write("**Detailed Rankings:**")
                                st.dataframe(perm_df[['feature', 'importance', 'std']], 
                                           use_container_width=True)
                                
                                scoring_method = perm_data.get('scoring_method', 'Unknown')
                                st.info(f"Scoring method: {scoring_method}")
                        else:
                            st.error(f"❌ Permutation Importance Failed: {perm_data.get('error', 'Unknown error')}")
                    
                    with exp_tab4:
                        interaction_data = explanations.get('feature_interaction', {})
                        if interaction_data.get('status') == 'success':
                            st.success("✅ Feature Interaction Analysis Completed")
                            
                            top_interactions = interaction_data.get('top_interactions', [])
                            if top_interactions:
                                interaction_df = pd.DataFrame(top_interactions)
                                interaction_df['Pair'] = (interaction_df['feature_1'] + 
                                                        " ↔ " + interaction_df['feature_2'])
                                
                                fig = px.bar(interaction_df.head(5), x='interaction_strength', y='Pair',
                                           orientation='h', title="Top Feature Interactions")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.write("**Interaction Details:**")
                                st.dataframe(interaction_df[['Pair', 'interaction_strength']].head(10),
                                           use_container_width=True)
                            
                            description = interaction_data.get('description', '')
                            if description:
                                st.info(description)
                        else:
                            st.error(f"❌ Feature Interaction Analysis Failed: {interaction_data.get('error', 'Unknown error')}")
                    
                    # Fidelity Scores
                    st.subheader("🎯 Explanation Fidelity Scores")
                    if fidelity_scores and 'error' not in fidelity_scores:
                        cols = st.columns(len([k for k, v in fidelity_scores.items() if v is not None]))
                        
                        col_idx = 0
                        for method, score in fidelity_scores.items():
                            if score is not None:
                                with cols[col_idx]:
                                    if isinstance(score, (int, float)):
                                        st.metric(
                                            label=method.replace('_', ' ').title(),
                                            value=f"{score:.3f}",
                                            help="Higher scores indicate better explanation fidelity"
                                        )
                                    else:
                                        st.metric(
                                            label=method.replace('_', ' ').title(),
                                            value="Qualitative",
                                            help=str(score)
                                        )
                                    col_idx += 1
                    else:
                        st.error("Fidelity score calculation failed")
                    
                    # Business Explanation
                    st.subheader("💼 Business-Friendly Explanation")
                    if business_explanation:
                        st.markdown(business_explanation)
                    else:
                        st.warning("Business explanation not available")
                    
                    # Performance Metrics
                    if explainer.performance_metrics:
                        with st.expander("⏱️ Performance Metrics", expanded=False):
                            perf_df = pd.DataFrame([
                                {"Operation": k, "Time (seconds)": f"{v:.3f}"} 
                                for k, v in explainer.performance_metrics.items()
                            ])
                            st.dataframe(perf_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    explainer.log_manager.log_exception(e, "Main application execution")
                    
                    with st.expander("🔧 Troubleshooting", expanded=False):
                        st.write("**Common issues and solutions:**")
                        st.write("1. **Model compatibility**: Ensure your model has a 'predict' method")
                        st.write("2. **Data format**: CSV files with proper column headers work best")
                        st.write("3. **Feature alignment**: Model and data features should match")
                        st.write("4. **Memory issues**: Try reducing the dataset size or max samples")
        
        with tab2:
            st.header("📈 Interactive Visualizations")
            
            if 'explanations' in locals() and explanations:
                charts = create_visualization_charts(explanations, df)
                
                if charts:
                    for chart_name, chart in charts.items():
                        st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("Upload model and data to generate visualizations")
            else:
                st.info("Upload model and data first to see visualizations")
        
        with tab3:
            st.header("📝 Analysis Logs")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                log_level = st.selectbox("Filter by Level", ["All", "DEBUG", "INFO", "WARNING", "ERROR"])
            
            with col2:
                log_limit = st.number_input("Max Entries", min_value=10, max_value=500, value=50)
            
            with col3:
                if st.button("🔄 Refresh Logs"):
                    st.rerun()
            
            # Get and display logs
            filter_level = None if log_level == "All" else log_level
            logs = explainer.log_manager.get_logs(level=filter_level, limit=int(log_limit))
            
            if logs:
                st.write(f"Showing {len(logs)} log entries")
                
                # Log statistics
                log_stats = explainer.log_manager.get_log_statistics()
                if log_stats and 'error' not in log_stats:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Logs", log_stats.get('total_logs', 0))
                    with col2:
                        st.metric("Log Level", log_stats.get('log_level', 'Unknown'))
                    with col3:
                        st.metric("Initialized", "Yes" if log_stats.get('initialized') else "No")
                    with col4:
                        st.metric("File Logging", "Yes" if log_stats.get('file_logging_enabled') else "No")
                
                # Display recent logs
                for log in reversed(logs[-20:]):
                    level_color = {
                        'ERROR': '🔴',
                        'WARNING': '🟡', 
                        'INFO': '🔵',
                        'DEBUG': '⚪'
                    }.get(log['level'], '⚪')
                    
                    with st.expander(f"{level_color} {log['timestamp']} - {log['level']} - {log['logger']}", 
                                   expanded=False):
                        st.code(log['message'])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Module:** {log.get('module', 'Unknown')}")
                            st.write(f"**Function:** {log.get('function', 'Unknown')}")
                        with col2:
                            st.write(f"**Line:** {log.get('line', 'Unknown')}")
                            if 'exception' in log:
                                st.write("**Exception:**")
                                st.code(log['exception'])
                
                # Download logs
                if st.button("📥 Download Logs"):
                    logs_json = json.dumps(logs, indent=2)
                    st.download_button(
                        label="Download as JSON",
                        data=logs_json,
                        file_name=f"explainer_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                # Clear logs
                if st.button("🗑️ Clear All Logs"):
                    if explainer.log_manager.clear_logs():
                        st.success("Logs cleared successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to clear logs")
            
            else:
                st.info("No logs available. Upload a model and dataset to generate analysis logs.")
        
        with tab4:
            st.header("ℹ️ System Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Configuration")
                config_data = {
                    "Max Processing Samples": explainer.config.MAX_SAMPLES_PROCESSING,
                    "Max SHAP Samples": explainer.config.MAX_SAMPLES_SHAP,
                    "Max LIME Samples": explainer.config.MAX_SAMPLES_LIME,
                    "Max Permutation Samples": explainer.config.MAX_SAMPLES_PERMUTATION,
                    "Background Size (Kernel)": explainer.config.BACKGROUND_SIZE_KERNEL,
                    "Max Retries": explainer.config.MAX_RETRIES,
                    "Retry Delay": f"{explainer.config.BASE_RETRY_DELAY}s",
                    "Log Level": explainer.config.LOG_LEVEL
                }
                
                for key, value in config_data.items():
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.write(f"**{key}:**")
                    with col_b:
                        st.write(str(value))
            
            with col2:
                st.subheader("API Status")
                if explainer.api_client.active_client:
                    st.success(f"✅ Active: {explainer.api_client.client_type.title()}")
                    if explainer.api_client.client_type == 'gemini':
                        model_name = explainer.api_client.active_client.get('model_name', 'Unknown')
                        st.info(f"Model: {model_name}")
                else:
                    st.warning("⚠️ No API clients available")
                
                st.subheader("Supported Formats")
                st.write("**Model Types:**")
                for fmt in explainer.config.SUPPORTED_MODEL_TYPES:
                    st.write(f"• {fmt}")
                
                st.write("**File Formats:**")
                for fmt in explainer.config.SUPPORTED_FILE_FORMATS:
                    st.write(f"• {fmt}")
                
                st.write("**Data Formats:**")
                for fmt in explainer.config.SUPPORTED_DATA_FORMATS:
                    st.write(f"• {fmt}")
            
            # System performance
            st.subheader("Performance Information")
            if hasattr(explainer, 'performance_metrics') and explainer.performance_metrics:
                perf_cols = st.columns(len(explainer.performance_metrics))
                for i, (operation, duration) in enumerate(explainer.performance_metrics.items()):
                    with perf_cols[i]:
                        st.metric(
                            operation.replace('_', ' ').title(),
                            f"{duration:.2f}s"
                        )
            else:
                st.info("Performance metrics will appear after processing data")
    
    except Exception as e:
        st.error(f"Application initialization failed: {str(e)}")
        st.write("Please check your environment and dependencies.")


if __name__ == "__main__":
    create_app()