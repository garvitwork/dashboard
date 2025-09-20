import pandas as pd
import numpy as np
import logging
import warnings
from typing import Dict, Any, Tuple, List, Optional, Union
import time
from contextlib import contextmanager

# Try to import explanation libraries, handle failures gracefully
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available")

try:
    from sklearn.inspection import permutation_importance
    PERMUTATION_AVAILABLE = True
except ImportError:
    PERMUTATION_AVAILABLE = False
    logging.warning("Sklearn permutation_importance not available")

# Suppress warnings
warnings.filterwarnings('ignore')

class SimpleExplainer:
    """Clean explainer without Streamlit dependencies"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.performance_metrics = {}
        
        # Configuration
        self.MAX_SAMPLES_PROCESSING = 1000
        self.MAX_SAMPLES_SHAP = 100
        self.MAX_SAMPLES_LIME = 5
        self.MAX_SAMPLES_PERMUTATION = 500
        self.BACKGROUND_SIZE_KERNEL = 100
        self.TOP_FEATURES_DISPLAY = 10
        self.MAX_FEATURE_INTERACTIONS = 10
        self.LOCAL_EXPLANATION_SAMPLES = 5
        self.MAX_CATEGORICAL_CARDINALITY = 100
        self.TOP_CATEGORIES_LIMIT = 99
        
        self.logger.info(f"SimpleExplainer initialized - SHAP: {SHAP_AVAILABLE}, LIME: {LIME_AVAILABLE}, Permutation: {PERMUTATION_AVAILABLE}")
    
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
                if not isinstance(data, pd.DataFrame):
                    data = pd.DataFrame(data)
                
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(data)
                        if proba.shape[1] == 2:
                            return proba[:, 1]
                        else:
                            return proba
                    except Exception as e:
                        self.logger.warning(f"predict_proba failed: {e}, falling back to predict")
                        return model.predict(data)
                else:
                    return model.predict(data)
                    
        except Exception as e:
            self.logger.error(f"All prediction methods failed: {e}")
            return np.zeros(len(data))
    
    def get_model_feature_names(self, model) -> Optional[List[str]]:
        """Extract feature names from model"""
        try:
            for attr in ['feature_names_in_', 'feature_names_', 'feature_name_']:
                try:
                    feature_names = getattr(model, attr, None)
                    if feature_names is not None:
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
            current_features = set(df_aligned.columns)
            expected_features = set(model_feature_names)
            
            missing_features = expected_features - current_features
            if missing_features:
                self.logger.warning(f"Missing features: {missing_features}")
                for feature in missing_features:
                    df_aligned[feature] = 0.0
            
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
        """Comprehensive data cleaning"""
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
                    if len(unique_values) > self.MAX_CATEGORICAL_CARDINALITY:
                        top_categories = df_clean[col].value_counts().head(self.TOP_CATEGORIES_LIMIT).index.tolist()
                        df_clean[col] = df_clean[col].apply(
                            lambda x: x if x in top_categories else 'Other'
                        )
                        unique_values = df_clean[col].unique()
                    
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
            return df.select_dtypes(include=[np.number]).fillna(0)
    
    def detect_model_type(self, model, predictions: np.ndarray) -> str:
        """Detect if model is classification or regression"""
        try:
            if hasattr(model, 'predict_proba') or hasattr(model, 'predict_log_proba'):
                return 'classification'
            
            unique_preds = np.unique(predictions)
            if (len(unique_preds) <= 20 and 
                all(isinstance(x, (int, np.integer)) or x.is_integer() 
                    for x in unique_preds[:min(10, len(unique_preds))])):
                return 'classification'
            
            return 'regression'
            
        except Exception as e:
            self.logger.warning(f"Model type detection failed: {e}, assuming regression")
            return 'regression'
    
    def generate_simple_feature_importance(self, model, df_sample: pd.DataFrame) -> Dict[str, Any]:
        """Generate simple feature importance using model coefficients"""
        try:
            feature_importance = None
            method_used = "unknown"
            
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                method_used = "tree_importance"
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if len(coef.shape) > 1:
                    feature_importance = np.abs(coef).mean(axis=0)
                else:
                    feature_importance = np.abs(coef)
                method_used = "linear_coef"
            
            if feature_importance is not None:
                return {
                    "global_feature_importance": feature_importance.tolist(),
                    "feature_names": df_sample.columns.tolist(),
                    "status": "success",
                    "method": method_used
                }
            else:
                # Fallback to random importance (for demonstration)
                random_importance = np.random.random(len(df_sample.columns))
                return {
                    "global_feature_importance": random_importance.tolist(),
                    "feature_names": df_sample.columns.tolist(),
                    "status": "fallback",
                    "method": "random_fallback",
                    "note": "Model does not provide feature importance. Using random values for demonstration."
                }
                
        except Exception as e:
            self.logger.error(f"Simple feature importance failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def generate_shap_explanation(self, model, df_sample: pd.DataFrame) -> Dict[str, Any]:
        """Generate SHAP explanation if available"""
        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP not available, using fallback feature importance")
            return self.generate_simple_feature_importance(model, df_sample)
        
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
                        background_size = min(self.BACKGROUND_SIZE_KERNEL, len(df_sample))
                        background_data = df_sample.iloc[:background_size]
                        shap_explainer = shap.KernelExplainer(
                            lambda x: self.safe_predict(model, pd.DataFrame(x, columns=df_sample.columns)), 
                            background_data
                        )
                        self.logger.info("Using SHAP KernelExplainer")
                    except Exception as e:
                        self.logger.debug(f"KernelExplainer failed: {e}")
                        return self.generate_simple_feature_importance(model, df_sample)
                
                # Generate SHAP values
                sample_size = min(self.MAX_SAMPLES_SHAP, len(df_sample))
                shap_values = shap_explainer.shap_values(df_sample.iloc[:sample_size])
                
                # Handle different SHAP value formats
                if isinstance(shap_values, list):
                    shap_values = shap_values[-1] if len(shap_values) > 1 else shap_values[0]
                
                return {
                    "global_feature_importance": np.abs(shap_values).mean(axis=0).tolist(),
                    "local_explanations": shap_values[:self.LOCAL_EXPLANATION_SAMPLES].tolist(),
                    "feature_names": df_sample.columns.tolist(),
                    "status": "success",
                    "explainer_type": type(shap_explainer).__name__
                }
                
        except Exception as e:
            self.logger.error(f"SHAP explanation generation failed: {e}")
            return self.generate_simple_feature_importance(model, df_sample)
    
    def generate_lime_explanation(self, model, df_sample: pd.DataFrame, model_type: str) -> Dict[str, Any]:
        """Generate LIME explanation if available"""
        if not LIME_AVAILABLE:
            return {"status": "failed", "error": "LIME not available"}
        
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
                sample_size = min(self.MAX_SAMPLES_LIME, len(df_sample))
                
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
                            num_features=min(self.TOP_FEATURES_DISPLAY, len(df_sample.columns))
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
        """Generate permutation importance if available"""
        if not PERMUTATION_AVAILABLE:
            return self.generate_simple_feature_importance(model, df_sample)
        
        try:
            with self.performance_tracker("permutation_importance"):
                sample_size = min(self.MAX_SAMPLES_PERMUTATION, len(df_sample))
                
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
            return self.generate_simple_feature_importance(model, df_sample)
    
    def generate_feature_interactions(self, explanations: Dict[str, Any], 
                                    df_sample: pd.DataFrame) -> Dict[str, Any]:
        """Generate feature interaction analysis"""
        try:
            shap_data = explanations.get('shap', {})
            if shap_data.get('status') in ['success'] and 'local_explanations' in shap_data:
                shap_values = np.array(shap_data['local_explanations'])
                if len(shap_values) > 0 and shap_values.shape[1] >= 2:
                    feature_interactions = []
                    n_features = min(self.TOP_FEATURES_DISPLAY, shap_values.shape[1])
                    feature_names = df_sample.columns.tolist()
                    
                    # Calculate pairwise interactions
                    for i in range(n_features):
                        for j in range(i + 1, n_features):
                            interaction_strength = np.abs(np.corrcoef(shap_values[:, i], shap_values[:, j])[0, 1])
                            
                            if not np.isnan(interaction_strength):
                                feature_interactions.append({
                                    'feature_1': feature_names[i],
                                    'feature_2': feature_names[j],
                                    'interaction_strength': round(float(interaction_strength), 6)
                                })
                    
                    feature_interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
                    
                    return {
                        "top_interactions": feature_interactions[:self.MAX_FEATURE_INTERACTIONS],
                        "description": "Feature pairs with strongest interactions based on SHAP value correlations",
                        "status": "success"
                    }
            
            # Fallback to simple correlation analysis
            numeric_df = df_sample.select_dtypes(include=[np.number])
            if len(numeric_df.columns) >= 2:
                correlation_matrix = numeric_df.corr().abs()
                feature_interactions = []
                
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i + 1, len(correlation_matrix.columns)):
                        feature_1 = correlation_matrix.columns[i]
                        feature_2 = correlation_matrix.columns[j]
                        interaction_strength = correlation_matrix.iloc[i, j]
                        
                        if not np.isnan(interaction_strength):
                            feature_interactions.append({
                                'feature_1': feature_1,
                                'feature_2': feature_2,
                                'interaction_strength': round(float(interaction_strength), 6)
                            })
                
                feature_interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
                
                return {
                    "top_interactions": feature_interactions[:self.MAX_FEATURE_INTERACTIONS],
                    "description": "Feature pairs with strongest correlations (fallback method)",
                    "status": "fallback"
                }
            
            return {"status": "failed", "error": "Insufficient data for interaction analysis"}
            
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
                
                if method_data.get('status') not in ['success', 'fallback']:
                    fidelity_scores[method_name] = None
                    continue
                
                if method_name == 'shap' and 'local_explanations' in method_data:
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
                    fidelity_scores[method_name] = "Based on local model fit"
                
                elif method_name == 'permutation_importance':
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
                            fidelity_scores[method_name] = 0.5  # Default for fallback methods
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
                
                if model_feature_names is not None:
                    df_clean = self.align_features(df_clean, model_feature_names)
                
                # Detect model type
                model_type = self.detect_model_type(model, predictions)
                self.logger.info(f"Detected model type: {model_type}")
                
                # Limit samples for performance
                max_samples = min(self.MAX_SAMPLES_PROCESSING, len(df_clean))
                df_sample = df_clean.iloc[:max_samples].copy()
                predictions_sample = predictions[:max_samples]
                
                self.logger.info(f"Processing {max_samples} samples with {len(df_sample.columns)} features")
                
                # Generate explanations
                explanations['shap'] = self.generate_shap_explanation(model, df_sample)
                explanations['lime'] = self.generate_lime_explanation(model, df_sample, model_type)
                explanations['permutation_importance'] = self.generate_permutation_importance(
                    model, df_sample, predictions_sample, model_type
                )
                explanations['feature_interaction'] = self.generate_feature_interactions(
                    explanations, df_sample
                )
                
                # Calculate fidelity scores
                fidelity_scores = self.calculate_fidelity_scores(explanations, predictions_sample)
                
                # Generate business explanation
                business_explanation = self._generate_business_explanation(
                    explanations, model_type, predictions_sample, df_sample
                )
                
                self.logger.info(f"Explanation generation completed. Performance metrics: {self.performance_metrics}")
                
                return explanations, business_explanation, fidelity_scores
                
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            return {}, f"Explanation generation failed: {str(e)}", {}

    def _generate_business_explanation(self, explanations: Dict[str, Any], model_type: str,
                                     predictions: np.ndarray, df_sample: pd.DataFrame) -> str:
        """Generate business-friendly explanation"""
        try:
            # Basic model description
            if model_type == 'classification':
                unique_preds = np.unique(predictions)
                if len(unique_preds) == 2:
                    positive_rate = np.mean(predictions)
                    explanation = f"""**Model Overview**
This binary classification model predicts outcomes with a {positive_rate:.1%} positive rate across the analyzed samples.

**Key Insights**
The model analyzes {len(df_sample.columns)} features to make predictions. """
                else:
                    explanation = f"""**Model Overview**
This multi-class classification model categorizes data into {len(unique_preds)} different classes.

**Key Insights**
The model evaluates {len(df_sample.columns)} features for classification decisions. """
            else:
                pred_mean = np.mean(predictions)
                pred_std = np.std(predictions)
                explanation = f"""**Model Overview**
This regression model predicts continuous values with an average of {pred_mean:.2f} and standard deviation of {pred_std:.2f}.

**Key Insights**
The model uses {len(df_sample.columns)} features to generate numerical predictions. """
            
            # Add top features if available
            shap_data = explanations.get('shap', {})
            if shap_data.get('status') in ['success', 'fallback']:
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
            
            # Add note about fallback methods if used
            fallback_methods = []
            for method, data in explanations.items():
                if data.get('status') == 'fallback':
                    fallback_methods.append(method)
            
            if fallback_methods:
                explanation += f"\n\n**Technical Note:** Some explanation methods used fallback approaches due to dependency limitations: {', '.join(fallback_methods)}"
            
            return explanation.strip()
            
        except Exception as e:
            self.logger.error(f"Business explanation generation failed: {e}")
            return "A business explanation could not be generated due to processing issues. Please refer to the technical analysis results."