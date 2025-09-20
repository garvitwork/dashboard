import os
from dataclasses import dataclass
from typing import Optional, Tuple
import logging


@dataclass
class Config:
    """Configuration class for the API application"""
    
    # API Configuration
    GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
    
    # Processing Configuration
    MAX_SAMPLES_PROCESSING: int = 1000
    MAX_SAMPLES_SHAP: int = 100
    MAX_SAMPLES_LIME: int = 5
    MAX_SAMPLES_PERMUTATION: int = 500
    BACKGROUND_SIZE_KERNEL: int = 100
    
    # Retry Configuration
    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY: float = 1.0
    BACKOFF_FACTOR: float = 2.0
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE: Optional[str] = None
    MAX_LOG_ENTRIES: int = 1000
    
    # Model Support Configuration
    SUPPORTED_MODEL_TYPES: Tuple[str, ...] = ('sklearn', 'xgboost', 'lightgbm', 'catboost')
    SUPPORTED_FILE_FORMATS: Tuple[str, ...] = ('.pkl', '.joblib', '.model')
    SUPPORTED_DATA_FORMATS: Tuple[str, ...] = ('.csv', '.xlsx', '.json', '.parquet')
    
    # Feature Engineering Configuration
    MAX_CATEGORICAL_CARDINALITY: int = 100
    TOP_CATEGORIES_LIMIT: int = 99
    MISSING_VALUE_THRESHOLD: float = 0.5
    
    # Explanation Configuration
    TOP_FEATURES_DISPLAY: int = 10
    MAX_FEATURE_INTERACTIONS: int = 10
    LOCAL_EXPLANATION_SAMPLES: int = 5
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()
    
    def validate(self) -> bool:
        """Validate configuration values"""
        try:
            # Validate positive integers
            positive_int_fields = [
                'MAX_SAMPLES_PROCESSING', 'MAX_SAMPLES_SHAP', 'MAX_SAMPLES_LIME',
                'MAX_SAMPLES_PERMUTATION', 'BACKGROUND_SIZE_KERNEL', 'MAX_RETRIES',
                'MAX_LOG_ENTRIES', 'MAX_CATEGORICAL_CARDINALITY', 'TOP_CATEGORIES_LIMIT',
                'TOP_FEATURES_DISPLAY', 'MAX_FEATURE_INTERACTIONS', 'LOCAL_EXPLANATION_SAMPLES'
            ]
            
            for field in positive_int_fields:
                value = getattr(self, field)
                if not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{field} must be a positive integer, got {value}")
            
            # Validate positive floats
            positive_float_fields = ['BASE_RETRY_DELAY', 'BACKOFF_FACTOR']
            for field in positive_float_fields:
                value = getattr(self, field)
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError(f"{field} must be a positive number, got {value}")
            
            # Validate threshold
            if not 0 <= self.MISSING_VALUE_THRESHOLD <= 1:
                raise ValueError("MISSING_VALUE_THRESHOLD must be between 0 and 1")
            
            # Validate log level
            valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if self.LOG_LEVEL.upper() not in valid_log_levels:
                raise ValueError(f"LOG_LEVEL must be one of {valid_log_levels}")
            
            # Warn about missing API keys
            if not self.GEMINI_API_KEY:
                logging.warning("GEMINI_API_KEY not set. Will use fallback explanation methods.")
            
            return True
            
        except Exception as e:
            logging.error(f"Configuration validation failed: {e}")
            raise
    
    def get_api_timeout(self) -> int:
        """Calculate API timeout based on retry configuration"""
        return int(self.BASE_RETRY_DELAY * (self.BACKOFF_FACTOR ** self.MAX_RETRIES))
    
    def get_max_samples_for_method(self, method: str) -> int:
        """Get maximum samples for a specific explanation method"""
        method_limits = {
            'shap': self.MAX_SAMPLES_SHAP,
            'lime': self.MAX_SAMPLES_LIME,
            'permutation': self.MAX_SAMPLES_PERMUTATION,
            'general': self.MAX_SAMPLES_PROCESSING
        }
        return method_limits.get(method.lower(), self.MAX_SAMPLES_PROCESSING)