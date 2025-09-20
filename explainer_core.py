import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance
import logging
import warnings
from typing import Dict, Any, Tuple, List, Optional, Union
import time
from contextlib import contextmanager

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SimpleExplainer:
    """Simplified explainer without Streamlit dependencies for API deployment"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
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
            self.logger.info(f"{operation} took {duration:.3f}s")
    
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
                    if len(unique_values) > 100:  # MAX_CATEGORICAL_CARDINALITY
                        # Keep only top categories for high cardinality features
                        top_categories = df_clean[col].value_counts().head(99).index.tolist()
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
                        background_size = min(100, len(df_sample))
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
                sample_size = min(100, len(df_sample))
                shap_values = shap_explainer.shap_values(df_sample.iloc[:sample_size])
                
                # Handle different SHAP value formats
                if isinstance(shap_values, list):
                    # Multi-class classification or multiple outputs
                    shap_values = shap_values[-1] if len(shap_values) > 1 else shap_values[0]
                
                return {
                    "global_feature_importance": np.abs(shap_values).mean(axis=0).tolist(),
                    "local_explanations": shap_values[:5].tolist(),
                    "feature_names": df_sample.columns.tolist(),
                    "status": "success",
                    "explainer_type": type(shap_explainer).__name__
                }
                
        except Exception as e:
            self.logger.error(f"SHAP explanation generation failed: {e}")
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
                sample_size = min(5, len(df_sample))
                
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
                            num_features=min(10, len(df_sample.columns))
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
            self.logger.error(f"LIME explanation generation failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def generate_permutation_importance(self, model, df_sample: pd.DataFrame, 
                                       predictions: np.ndarray, model_type: str) -> Dict[str, Any]:
        """Generate permutation importance with proper scoring"""
        try:
            with self.performance_tracker("permutation_importance"):
                sample_size = min(500, len(df_sample))
                
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
            self.logger.error(f"Permutation importance generation failed: {e}")
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
            n_features = min(10, shap_values.shape[1])
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
                "top_interactions": feature_interactions[:10],
                "description": "Feature pairs with strongest interactions based on SHAP value correlations",
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Feature interaction analysis failed: {e}")
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
            self.logger.error(f"Fidelity score calculation failed: {e}")
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
                max_samples = min(1000, len(df_clean))
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
                    explanations, model_type, predictions_sample, df_sample
                )
                
                # Log performance metrics
                self.logger.info(f"Explanation generation completed. Performance metrics: {self.performance_metrics}")
                
                return explanations, business_explanation, fidelity_scores
                
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            return {}, f"Explanation generation failed: {str(e)}", {}

    def _generate_business_explanation(self, explanations: Dict[str, Any], model_type: str,
                                     predictions: np.ndarray, df_sample: pd.DataFrame) -> str:
        """Generate business-friendly explanation using rule-based approach"""
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
            self.logger.error(f"Business explanation generation failed: {e}")
            return "A business explanation could not be generated due to processing issues. Please refer to the technical analysis results."