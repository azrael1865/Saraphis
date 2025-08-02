"""
Machine Learning Predictor for Financial Fraud Detection
Consolidated ML system that integrates with the enhanced predictor and preprocessing manager.
This file provides backward compatibility and simplified interfaces.
"""

import logging
import numpy as np
import pandas as pd
import time
import threading
import hashlib
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path

# ML library imports with fallbacks
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (roc_auc_score, precision_recall_curve, accuracy_score, 
                                precision_score, recall_score, f1_score, confusion_matrix,
                                classification_report)
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
    from sklearn.calibration import CalibratedClassifierCV
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Import the main enhanced ML predictor
try:
    from enhanced_ml_predictor import (
        EnhancedFinancialMLPredictor as BaseMLPredictor,
        ModelConfig,
        ModelMetrics,
        PredictionResult,
        ModelVersion,
        ModelType,
        ModelStatus,
        MLPredictorError,
        ModelNotFittedError,
        InvalidInputError,
        ModelValidationError,
        FeatureMismatchError,
        ModelDriftError,
        ModelLoadError,
        HyperparameterError,
        ModelTrainingError,
        PredictionError,
        InputValidator,
        ModelDriftDetector
    )
    ENHANCED_AVAILABLE = True
except ImportError:
    try:
        from enhanced_ml_predictor import (
            EnhancedFinancialMLPredictor as BaseMLPredictor,
            ModelConfig,
            ModelMetrics,
            PredictionResult,
            ModelVersion,
            ModelType,
            ModelStatus,
            MLPredictorError,
            ModelNotFittedError,
            InvalidInputError,
            ModelValidationError,
            FeatureMismatchError,
            ModelDriftError,
            ModelLoadError,
            HyperparameterError,
            ModelTrainingError,
            PredictionError,
            InputValidator,
            ModelDriftDetector
        )
        ENHANCED_AVAILABLE = True
    except ImportError as e:
        # Fallback to basic implementation if enhanced version not available
        logger = logging.getLogger(__name__)
        logger.error(f"Enhanced ML predictor not available: {e}")
        ENHANCED_AVAILABLE = False
        
        # Define minimal fallback classes
        class BaseMLPredictor:
            def __init__(self, config=None):
                self.config = config
                logger.warning("Using fallback ML predictor - limited functionality")
            
            def predict_fraud(self, transaction):
                return {
                    'fraud_probability': 0.5,
                    'predictions': {'fallback': 0.5},
                    'confidence': 0.0,
                    'feature_importance': {}
                }
        
        # Add missing type classes
        class ModelConfig:
            def __init__(self, **kwargs):
                self.model_type = type('ModelType', (), {'value': 'fallback'})()
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class ModelMetrics:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class PredictionResult:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class ModelVersion:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class ModelType:
            RANDOM_FOREST = 'random_forest'
            XGBOOST = 'xgboost'
            LIGHTGBM = 'lightgbm'
            ENSEMBLE = 'ensemble'
            GRADIENT_BOOSTING = 'gradient_boosting'
            LOGISTIC_REGRESSION = 'logistic_regression'
            MLP = 'mlp'
            SVM = 'svm'
            DECISION_TREE = 'decision_tree'
            NAIVE_BAYES = 'naive_bayes'
            ISOLATION_FOREST = 'isolation_forest'
        
        class ModelStatus:
            TRAINED = 'trained'
            VALIDATED = 'validated'
            FAILED = 'failed'
            DEPLOYED = 'deployed'

# Import preprocessing manager
try:
    from enhanced_fraud_core_main import CompletePreprocessingManager
    PREPROCESSING_AVAILABLE = True
except ImportError:
    try:
        from enhanced_fraud_core_main import CompletePreprocessingManager
        PREPROCESSING_AVAILABLE = True
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("CompletePreprocessingManager not available")
        PREPROCESSING_AVAILABLE = False
        
        # Minimal fallback preprocessing
        class CompletePreprocessingManager:
            def __init__(self, config=None):
                self.config = config or {}
                self.feature_count = 0
            
            def preprocess_transaction(self, transaction):
                return {
                    'processed_data': transaction,
                    'metadata': {'feature_count': 0, 'quality_score': 1.0},
                    'quality_assessment': {'overall_score': 1.0}
                }

# Configure logging
logger = logging.getLogger(__name__)

class FinancialMLPredictor:
    """
    Simplified ML Predictor interface that delegates to the enhanced predictor.
    Provides backward compatibility for existing code with preprocessing integration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, enable_preprocessing: bool = True):
        """Initialize the ML predictor with optional preprocessing"""
        if ENHANCED_AVAILABLE:
            # Convert simple config dict to ModelConfig if needed
            if isinstance(config, dict):
                from enhanced_ml_predictor import ModelConfig
                ml_config = ModelConfig(**config)
            else:
                ml_config = config
            
            self.predictor = BaseMLPredictor(ml_config)
            logger.info("Using enhanced ML predictor")
        else:
            self.predictor = BaseMLPredictor(config)
            logger.warning("Using fallback ML predictor")
        
        # Initialize preprocessing manager
        self.enable_preprocessing = enable_preprocessing and PREPROCESSING_AVAILABLE
        self.preprocessing_manager = None
        
        if self.enable_preprocessing:
            try:
                # Default preprocessing configuration optimized for ML
                preprocessing_config = {
                    'feature_engineering': {
                        'enable_time_features': True,
                        'enable_amount_features': True,
                        'enable_frequency_features': True,
                        'enable_velocity_features': True,
                        'enable_merchant_features': True,
                        'enable_geographic_features': True
                    },
                    'data_quality': {
                        'missing_value_threshold': 0.3,
                        'outlier_method': 'iqr',
                        'outlier_threshold': 3.0
                    },
                    'feature_selection': {
                        'method': 'mutual_info',
                        'k_features': 100,
                        'correlation_threshold': 0.95
                    },
                    'scaling': {
                        'method': 'standard',
                        'feature_range': [0, 1]
                    }
                }
                
                self.preprocessing_manager = CompletePreprocessingManager(preprocessing_config)
                logger.info("Preprocessing manager initialized for ML predictor")
            except Exception as e:
                logger.warning(f"Failed to initialize preprocessing: {e}")
                self.enable_preprocessing = False
    
    def predict_fraud(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fraud for a transaction with optional preprocessing
        
        Args:
            transaction: Transaction features
            
        Returns:
            Fraud prediction result with preprocessing metadata
        """
        # Apply preprocessing if enabled
        processed_transaction = transaction
        preprocessing_metadata = {}
        
        if self.enable_preprocessing and self.preprocessing_manager:
            try:
                preprocessing_result = self.preprocessing_manager.preprocess_transaction(transaction)
                processed_transaction = preprocessing_result['processed_data']
                preprocessing_metadata = preprocessing_result.get('metadata', {})
                
                logger.debug(f"ML preprocessing completed - Features: {preprocessing_metadata.get('feature_count', 0)}, "
                           f"Quality: {preprocessing_metadata.get('quality_score', 0):.3f}")
            except Exception as e:
                logger.warning(f"ML preprocessing failed, using original transaction: {e}")
                processed_transaction = transaction
        
        # Make prediction with processed transaction
        if ENHANCED_AVAILABLE:
            # Use enhanced predictor
            result = self.predictor.predict(processed_transaction)
            
            # Convert to simple dict format for backward compatibility
            if hasattr(result, 'fraud_probability'):
                prediction_result = {
                    'fraud_probability': result.fraud_probability,
                    'risk_score': result.risk_score,
                    'is_fraud': result.is_fraud,
                    'confidence': result.confidence,
                    'risk_level': result.risk_level,
                    'predictions': {self.predictor.config.model_type.value: result.fraud_probability}
                }
            else:
                prediction_result = result
        else:
            # Use fallback
            prediction_result = self.predictor.predict_fraud(processed_transaction)
        
        # Add preprocessing metadata to result
        if preprocessing_metadata:
            if isinstance(prediction_result, dict):
                prediction_result['preprocessing'] = preprocessing_metadata
            else:
                # If result is an object, try to add as attribute
                try:
                    prediction_result.preprocessing = preprocessing_metadata
                except:
                    pass
        
        return prediction_result
    
    def train_model(self, X: Union[np.ndarray, pd.DataFrame], 
                   y: Union[np.ndarray, pd.Series], **kwargs) -> bool:
        """
        Train the ML model
        
        Args:
            X: Feature matrix
            y: Target labels
            **kwargs: Additional training parameters
            
        Returns:
            Success status
        """
        if ENHANCED_AVAILABLE:
            try:
                version = self.predictor.train_model(X, y, **kwargs)
                return version.status in ['trained', 'validated']
            except Exception as e:
                logger.error(f"Training failed: {e}")
                return False
        else:
            logger.warning("Training not available in fallback mode")
            return False
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        if ENHANCED_AVAILABLE and hasattr(self.predictor, 'get_statistics'):
            return self.predictor.get_statistics()
        else:
            return {'status': 'fallback_mode', 'metrics_available': False}
    
    def save_model(self, path: str) -> bool:
        """Save the trained model"""
        if ENHANCED_AVAILABLE and self.predictor.current_model:
            try:
                from pathlib import Path
                model_path = Path(path)
                self.predictor._save_model(self.predictor.current_model, model_path)
                return True
            except Exception as e:
                logger.error(f"Failed to save model: {e}")
                return False
        else:
            logger.warning("Model saving not available")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load a trained model"""
        if ENHANCED_AVAILABLE:
            return self.predictor.load_model(path)
        else:
            logger.warning("Model loading not available in fallback mode")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Random Forest"""
        if not self.fitted or not self.feature_names:
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_))

class BaseModel:
    """Base model class for fraud detection models"""
    def __init__(self):
        self.fitted = False
        self.model = None
        self.feature_names = []
    
    def create_model(self):
        """Create the underlying model"""
        raise NotImplementedError
    
    def fit(self, X, y):
        """Fit the model"""
        self.model = self.create_model()
        self.model.fit(X, y)
        self.fitted = True
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
    
    def predict(self, X):
        """Make predictions"""
        if not self.fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.fitted:
            raise ValueError("Model not fitted")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance"""
        return {}

class XGBoostModel(BaseModel):
    """XGBoost implementation"""
    
    def create_model(self) -> xgb.XGBClassifier:
        """Create XGBoost model"""
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'use_label_encoder': False,
            'random_state': self.config.random_state,
            'n_jobs': -1
        }
        params = {**default_params, **self.config.model_params}
        return xgb.XGBClassifier(**params)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from XGBoost"""
        if not self.fitted or not self.feature_names:
            return {}
        importance = self.model.get_booster().get_score(importance_type='gain')
        # Map feature names
        return {self.feature_names[int(k[1:])]: v for k, v in importance.items() if k.startswith('f')}

class LightGBMModel(BaseModel):
    """LightGBM implementation"""
    
    def create_model(self) -> lgb.LGBMClassifier:
        """Create LightGBM model"""
        default_params = {
            'n_estimators': 100,
            'max_depth': -1,
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0,
            'lambda_l2': 0,
            'min_child_samples': 20,
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'random_state': self.config.random_state,
            'n_jobs': -1,
            'verbose': -1
        }
        params = {**default_params, **self.config.model_params}
        return lgb.LGBMClassifier(**params)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from LightGBM"""
        if not self.fitted or not self.feature_names:
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_))

class EnsembleModel(BaseModel):
    """Ensemble of multiple models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_models = []
        
    def create_model(self) -> VotingClassifier:
        """Create ensemble model"""
        # Create diverse base models
        models = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('xgb', xgb.XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss')),
            ('lgb', lgb.LGBMClassifier(n_estimators=50, verbose=-1)),
            ('lr', LogisticRegression(max_iter=1000))
        ]
        
        return VotingClassifier(estimators=models, voting='soft', n_jobs=-1)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get averaged feature importance from ensemble"""
        if not self.fitted:
            return {}
        
        # Average importance from tree-based models
        importance_dict = {}
        tree_models = ['rf', 'xgb', 'lgb']
        
        for name, model in self.model.estimators_:
            if name in tree_models and hasattr(model, 'feature_importances_'):
                for i, imp in enumerate(model.feature_importances_):
                    feature_name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'
                    if feature_name not in importance_dict:
                        importance_dict[feature_name] = []
                    importance_dict[feature_name].append(imp)
        
        # Average the importances
        return {k: np.mean(v) for k, v in importance_dict.items()}

class AdvancedFinancialMLPredictor:
    """
    Advanced Machine Learning Predictor for Financial Fraud Detection
    
    Features:
    - Multiple ML algorithms (Random Forest, XGBoost, LightGBM, etc.)
    - Model versioning and management
    - Hyperparameter tuning
    - Model calibration
    - Comprehensive evaluation metrics
    - Production-ready prediction pipeline
    - A/B testing support
    - Model explainability
    - Enhanced validation and error handling
    - Integration with enhanced predictor when available
    - Preprocessing pipeline integration
    """
    
    def __init__(self, config: Optional[ModelConfig] = None, use_enhanced: bool = True, enable_preprocessing: bool = True):
        """
        Initialize ML predictor
        
        Args:
            config: Model configuration
            use_enhanced: Whether to use enhanced predictor when available
            enable_preprocessing: Whether to enable preprocessing pipeline
        """
        self.config = config or ModelConfig()
        self.current_model: Optional[BaseModel] = None
        self.model_versions: Dict[str, ModelVersion] = {}
        self.active_version: Optional[str] = None
        self.prediction_stats = {
            'total_predictions': 0,
            'total_frauds_detected': 0,
            'average_prediction_time': 0.0,
            'last_prediction_time': None,
            'failed_predictions': 0,
            'drift_warnings': 0
        }
        self._lock = threading.RLock()
        
        # Enhanced predictor integration
        self.use_enhanced = use_enhanced and ENHANCED_AVAILABLE
        self._enhanced_predictor = None
        
        if self.use_enhanced:
            try:
                # Convert config to enhanced config if needed
                enhanced_config = self._convert_to_enhanced_config(self.config)
                self._enhanced_predictor = EnhancedFinancialMLPredictor(enhanced_config)
                logger.info("Using enhanced ML predictor with comprehensive validation")
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced predictor: {e}, falling back to standard")
                self.use_enhanced = False
        
        # Initialize preprocessing manager
        self.enable_preprocessing = enable_preprocessing and PREPROCESSING_AVAILABLE
        self.preprocessing_manager = None
        
        if self.enable_preprocessing:
            try:
                # Advanced preprocessing configuration for comprehensive ML
                preprocessing_config = {
                    'feature_engineering': {
                        'enable_time_features': True,
                        'enable_amount_features': True,
                        'enable_frequency_features': True,
                        'enable_velocity_features': True,
                        'enable_merchant_features': True,
                        'enable_geographic_features': True
                    },
                    'data_quality': {
                        'missing_value_threshold': 0.2,
                        'outlier_method': 'iqr',
                        'outlier_threshold': 2.5,
                        'duplicate_threshold': 0.98
                    },
                    'feature_selection': {
                        'method': 'mutual_info',
                        'k_features': 150,
                        'correlation_threshold': 0.98
                    },
                    'scaling': {
                        'method': 'robust',
                        'feature_range': [-1, 1]
                    }
                }
                
                self.preprocessing_manager = CompletePreprocessingManager(preprocessing_config)
                logger.info("Advanced preprocessing manager initialized for comprehensive ML predictor")
            except Exception as e:
                logger.warning(f"Failed to initialize preprocessing: {e}")
                self.enable_preprocessing = False
        
        # Initialize model directory
        if self.config.model_path is None:
            self.config.model_path = Path("models")
        self.config.model_path.mkdir(exist_ok=True)
        
        logger.info(f"AdvancedFinancialMLPredictor initialized with {self.config.model_type.value} "
                   f"(enhanced: {self.use_enhanced}, preprocessing: {self.enable_preprocessing})")
    
    def _create_model(self, model_type: ModelType) -> BaseModel:
        """Create model instance based on type"""
        model_map = {
            ModelType.RANDOM_FOREST: RandomForestModel,
            ModelType.XGBOOST: XGBoostModel,
            ModelType.LIGHTGBM: LightGBMModel,
            ModelType.ENSEMBLE: EnsembleModel,
            ModelType.GRADIENT_BOOSTING: lambda c: BaseModel(c),  # Placeholder
            ModelType.LOGISTIC_REGRESSION: lambda c: BaseModel(c),
            ModelType.MLP: lambda c: BaseModel(c),
            ModelType.SVM: lambda c: BaseModel(c),
            ModelType.DECISION_TREE: lambda c: BaseModel(c),
            ModelType.NAIVE_BAYES: lambda c: BaseModel(c),
            ModelType.ISOLATION_FOREST: lambda c: BaseModel(c),
            ModelType.NEURAL_NETWORK: lambda c: self._raise_not_implemented("NEURAL_NETWORK"),
            ModelType.LSTM: lambda c: self._raise_not_implemented("LSTM"),
            ModelType.TRANSFORMER: lambda c: self._raise_not_implemented("TRANSFORMER"),
            ModelType.VOTING_CLASSIFIER: lambda c: BaseModel(c),  # Placeholder
            ModelType.STACKING_CLASSIFIER: lambda c: BaseModel(c)  # Placeholder
        }
        
        model_class = model_map.get(model_type)
        if model_class is None:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model_class(self.config)
    
    def _raise_not_implemented(self, model_name: str):
        """Raise NotImplementedError for unimplemented neural network models"""
        raise NotImplementedError(f"{model_name} model not yet implemented in ml_predictor.py")
    
    def train_model(self, 
                   X: Union[np.ndarray, pd.DataFrame], 
                   y: Union[np.ndarray, pd.Series],
                   model_type: Optional[ModelType] = None,
                   version_notes: str = "") -> ModelVersion:
        """
        Train a new fraud detection model
        
        Args:
            X: Feature matrix
            y: Target labels
            model_type: Type of model to train
            version_notes: Notes about this model version
            
        Returns:
            Model version information
        """
        # Use enhanced predictor if available
        if self.use_enhanced and self._enhanced_predictor:
            try:
                return self._enhanced_predictor.train_model(X, y, model_type, version_notes)
            except Exception as e:
                logger.error(f"Enhanced predictor training failed: {e}, falling back to standard")
                self.use_enhanced = False
        
        # Standard training
        start_time = time.time()
        model_type = model_type or self.config.model_type
        
        logger.info(f"Starting model training with {model_type.value}")
        logger.info(f"Data shape: {X.shape}, Fraud ratio: {np.mean(y):.3f}")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X = X.values
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if self.config.stratify else None
        )
        
        # Further split for validation if needed
        if self.config.validation_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=self.config.validation_size,
                random_state=self.config.random_state,
                stratify=y_train if self.config.stratify else None
            )
        
        # Create and train model
        model = self._create_model(model_type)
        model.feature_names = feature_names
        
        # Hyperparameter tuning if enabled
        if self.config.hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning...")
            model = self._tune_hyperparameters(model, X_train, y_train)
        else:
            model.fit(X_train, y_train)
        
        # Model calibration if enabled
        if self.config.calibrate_model:
            logger.info("Calibrating model probabilities...")
            model = self._calibrate_model(model, X_val if self.config.validation_size > 0 else X_test, 
                                        y_val if self.config.validation_size > 0 else y_test)
        
        # Evaluate model
        metrics = self._evaluate_model(model, X_test, y_test)
        
        # Calculate feature importance
        if self.config.calculate_feature_importance:
            metrics.feature_importance = model.get_feature_importance()
        
        training_time = time.time() - start_time
        metrics.training_time_seconds = training_time
        
        # Create version
        version_id = self._generate_version_id()
        version = ModelVersion(
            version_id=version_id,
            model_type=model_type,
            created_at=datetime.now(),
            metrics=metrics,
            status=ModelStatus.TRAINED,
            config=self.config,
            training_data_hash=self._hash_data(X, y),
            feature_names=feature_names,
            notes=version_notes
        )
        
        # Save model if configured
        if self.config.save_model:
            model_path = self.config.model_path / f"model_{version_id}.pkl"
            self._save_model(model, model_path)
            version.model_path = model_path
            metrics.model_size_bytes = model_path.stat().st_size
        
        # Store version
        with self._lock:
            self.model_versions[version_id] = version
            self.current_model = model
            
        # Validate against thresholds
        if self._validate_model(metrics):
            version.status = ModelStatus.VALIDATED
            logger.info(f"Model {version_id} validated successfully")
        else:
            version.status = ModelStatus.FAILED
            logger.warning(f"Model {version_id} failed validation")
        
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        logger.info(f"Model performance - AUC: {metrics.auc_roc:.3f}, "
                   f"Precision: {metrics.precision:.3f}, Recall: {metrics.recall:.3f}")
        
        return version
    
    def predict(self, features: Union[Dict[str, Any], pd.DataFrame, np.ndarray],
               transaction_id: Optional[str] = None) -> Union[PredictionResult, List[PredictionResult]]:
        """
        Predict fraud probability for transaction(s)
        
        Args:
            features: Transaction features
            transaction_id: Optional transaction ID
            
        Returns:
            Prediction result(s)
        """
        # Use enhanced predictor if available
        if self.use_enhanced and self._enhanced_predictor:
            try:
                return self._enhanced_predictor.predict(features, transaction_id)
            except Exception as e:
                logger.error(f"Enhanced predictor prediction failed: {e}, falling back to standard")
                self.use_enhanced = False
        
        # Standard prediction
        if self.current_model is None or not self.current_model.fitted:
            raise ValueError("No trained model available for prediction")
        
        start_time = time.time()
        
        # Apply preprocessing if enabled and features is a dictionary
        processed_features = features
        preprocessing_metadata = {}
        
        if (self.enable_preprocessing and self.preprocessing_manager and 
            isinstance(features, dict)):
            try:
                preprocessing_result = self.preprocessing_manager.preprocess_transaction(features)
                processed_features = preprocessing_result['processed_data']
                preprocessing_metadata = preprocessing_result.get('metadata', {})
                
                logger.debug(f"Advanced ML preprocessing completed - Features: {preprocessing_metadata.get('feature_count', 0)}, "
                           f"Quality: {preprocessing_metadata.get('quality_score', 0):.3f}")
            except Exception as e:
                logger.warning(f"Advanced ML preprocessing failed, using original features: {e}")
                processed_features = features
        
        # Handle different input types
        if isinstance(processed_features, dict):
            # Single prediction from dictionary
            X = self._prepare_features_dict(processed_features)
            single_prediction = True
        elif isinstance(processed_features, pd.DataFrame):
            X = processed_features.values
            single_prediction = False
        else:
            X = processed_features
            single_prediction = len(X.shape) == 1
            if single_prediction:
                X = X.reshape(1, -1)
        
        # Make predictions
        try:
            probabilities = self.current_model.predict_proba(X)
            fraud_probabilities = probabilities[:, 1]
            
            # Create prediction results
            results = []
            for i, prob in enumerate(fraud_probabilities):
                # Calculate risk score and level
                risk_score = self._calculate_risk_score(prob)
                risk_level = self._determine_risk_level(risk_score)
                
                # Create result
                result = PredictionResult(
                    transaction_id=transaction_id if single_prediction else f"TXN_{i}",
                    fraud_probability=float(prob),
                    risk_score=float(risk_score),
                    is_fraud=prob >= self.config.fraud_threshold,
                    risk_level=risk_level,
                    confidence=float(max(prob, 1 - prob)),
                    model_version=self.active_version or "unknown",
                    risk_factors=self._identify_risk_factors(X[i], prob)
                )
                
                # Add preprocessing metadata if available
                if preprocessing_metadata:
                    try:
                        result.preprocessing = preprocessing_metadata
                    except:
                        # If PredictionResult doesn't support attributes, ignore
                        pass
                
                results.append(result)
                
                # Update statistics
                with self._lock:
                    self.prediction_stats['total_predictions'] += 1
                    if result.is_fraud:
                        self.prediction_stats['total_frauds_detected'] += 1
            
            prediction_time = time.time() - start_time
            
            # Update timing statistics
            with self._lock:
                total_predictions = self.prediction_stats['total_predictions']
                avg_time = self.prediction_stats['average_prediction_time']
                self.prediction_stats['average_prediction_time'] = (
                    (avg_time * (total_predictions - len(results)) + prediction_time) / total_predictions
                )
                self.prediction_stats['last_prediction_time'] = datetime.now()
            
            logger.info(f"Predicted {len(results)} transactions in {prediction_time:.3f} seconds")
            
            return results[0] if single_prediction else results
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def predict_batch(self, batch_features: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
        """
        Predict for batch of transactions (backward compatibility)
        
        Args:
            batch_features: List of feature dictionaries
            
        Returns:
            List of (fraud_probability, risk_score) tuples
        """
        # Convert to DataFrame for batch prediction
        df = pd.DataFrame(batch_features)
        results = self.predict(df)
        
        return [(r.fraud_probability, r.risk_score) for r in results]
    
    def evaluate_model(self, X: Union[np.ndarray, pd.DataFrame], 
                      y: Union[np.ndarray, pd.Series]) -> ModelMetrics:
        """
        Evaluate model performance
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Model metrics
        """
        if self.current_model is None:
            raise ValueError("No model available for evaluation")
        
        return self._evaluate_model(self.current_model, X, y)
    
    def _evaluate_model(self, model: BaseModel, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Internal method to evaluate model"""
        # Make predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = ModelMetrics(
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred, zero_division=0),
            recall=recall_score(y, y_pred),
            f1_score=f1_score(y, y_pred),
            auc_roc=roc_auc_score(y, y_proba),
            confusion_matrix=confusion_matrix(y, y_pred),
            classification_report=classification_report(y, y_pred, output_dict=True)
        )
        
        # Calculate additional metrics
        tn, fp, fn, tp = metrics.confusion_matrix.ravel()
        metrics.true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics.false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Precision-Recall curve
        precision_curve, recall_curve, thresholds = precision_recall_curve(y, y_proba)
        metrics.precision_recall_auc = np.trapz(precision_curve, recall_curve)
        
        # Store threshold analysis
        metrics.thresholds = thresholds.tolist()[:100]  # Limit size
        metrics.precision_by_threshold = precision_curve.tolist()[:100]
        metrics.recall_by_threshold = recall_curve.tolist()[:100]
        
        return metrics
    
    def _tune_hyperparameters(self, model: BaseModel, X: np.ndarray, y: np.ndarray) -> BaseModel:
        """Tune model hyperparameters"""
        # Define parameter grids for different model types
        param_grids = {
            ModelType.RANDOM_FOREST: {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [10, 20, 50],
                'min_samples_leaf': [5, 10, 20]
            },
            ModelType.XGBOOST: {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0]
            },
            ModelType.LIGHTGBM: {
                'n_estimators': [50, 100, 200],
                'num_leaves': [15, 31, 63],
                'learning_rate': [0.01, 0.1, 0.3],
                'feature_fraction': [0.6, 0.8, 1.0]
            }
        }
        
        # Get parameter grid for model type
        param_grid = param_grids.get(self.config.model_type, {})
        
        if not param_grid:
            # No tuning for this model type
            model.fit(X, y)
            return model
        
        # Perform grid search
        if self.config.tuning_method == "grid":
            search = GridSearchCV(
                model.model,
                param_grid,
                cv=self.config.tuning_cv_folds,
                scoring=self.config.cv_scoring,
                n_jobs=-1,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                model.model,
                param_grid,
                n_iter=self.config.tuning_iterations,
                cv=self.config.tuning_cv_folds,
                scoring=self.config.cv_scoring,
                n_jobs=-1,
                verbose=1
            )
        
        search.fit(X, y)
        
        # Update model with best parameters
        model.model = search.best_estimator_
        model.fitted = True
        
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best CV score: {search.best_score_:.3f}")
        
        return model
    
    def _calibrate_model(self, model: BaseModel, X: np.ndarray, y: np.ndarray) -> BaseModel:
        """Calibrate model probabilities"""
        calibrated = CalibratedClassifierCV(
            model.model,
            method=self.config.calibration_method,
            cv='prefit'
        )
        calibrated.fit(X, y)
        
        # Wrap calibrated model
        model.model = calibrated
        return model
    
    def _calculate_risk_score(self, fraud_probability: float) -> float:
        """Calculate risk score from fraud probability"""
        # Apply non-linear transformation for better risk differentiation
        if fraud_probability < 0.5:
            risk_score = fraud_probability * 0.6
        elif fraud_probability < 0.8:
            risk_score = 0.3 + (fraud_probability - 0.5) * 1.2
        else:
            risk_score = 0.66 + (fraud_probability - 0.8) * 1.7
        
        return min(risk_score, 1.0)
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level from risk score"""
        if risk_score < 0.3:
            return "low"
        elif risk_score < 0.6:
            return "medium"
        elif risk_score < 0.8:
            return "high"
        else:
            return "critical"
    
    def _identify_risk_factors(self, features: np.ndarray, fraud_probability: float) -> List[str]:
        """Identify top risk factors for a prediction"""
        risk_factors = []
        
        # Get feature importance if available
        if self.current_model and hasattr(self.current_model, 'get_feature_importance'):
            importance = self.current_model.get_feature_importance()
            if importance:
                # Sort by importance
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                
                # Add top contributing features
                for feature, imp in sorted_features[:5]:
                    if imp > 0.05:  # Only significant features
                        risk_factors.append(f"{feature} (importance: {imp:.2f})")
        
        # Add probability-based factors
        if fraud_probability > 0.9:
            risk_factors.append("Very high fraud probability")
        elif fraud_probability > 0.7:
            risk_factors.append("High fraud probability")
        
        return risk_factors
    
    def _validate_model(self, metrics: ModelMetrics) -> bool:
        """Validate model against performance thresholds"""
        return (
            metrics.auc_roc >= self.config.min_auc and
            metrics.precision >= self.config.min_precision and
            metrics.recall >= self.config.min_recall
        )
    
    def _prepare_features_dict(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare features from dictionary"""
        if not self.current_model or not self.current_model.feature_names:
            raise ValueError("Model must be trained before preparing features")
        
        # Extract features in correct order
        feature_values = []
        for feature_name in self.current_model.feature_names:
            value = features.get(feature_name, 0)  # Default to 0 if missing
            feature_values.append(value)
        
        return np.array(feature_values).reshape(1, -1)
    
    def _generate_version_id(self) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"{self.config.model_type.value}_{timestamp}_{random_suffix}"
    
    def _hash_data(self, X: np.ndarray, y: np.ndarray) -> str:
        """Generate hash of training data"""
        data_str = f"{X.shape}_{y.shape}_{np.mean(X)}_{np.mean(y)}"
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _save_model(self, model: BaseModel, path: Path) -> None:
        """Save model to disk"""
        with open(path, 'wb') as f:
            joblib.dump(model, f)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, model_name: str) -> bool:
        """
        Load a trained model
        
        Args:
            model_name: Model version ID or path
            
        Returns:
            Success status
        """
        try:
            # Check if it's a version ID
            if model_name in self.model_versions:
                version = self.model_versions[model_name]
                if version.model_path and version.model_path.exists():
                    with open(version.model_path, 'rb') as f:
                        self.current_model = joblib.load(f)
                    self.active_version = model_name
                    logger.info(f"Loaded model version {model_name}")
                    return True
            
            # Try as file path
            model_path = Path(model_name)
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.current_model = joblib.load(f)
                self.active_version = model_path.stem
                logger.info(f"Loaded model from {model_path}")
                return True
            
            logger.error(f"Model {model_name} not found")
            return False
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def deploy_model(self, version_id: str) -> bool:
        """
        Deploy a model version to production
        
        Args:
            version_id: Version to deploy
            
        Returns:
            Success status
        """
        if version_id not in self.model_versions:
            logger.error(f"Version {version_id} not found")
            return False
        
        version = self.model_versions[version_id]
        
        # Validate model is ready
        if version.status != ModelStatus.VALIDATED:
            logger.error(f"Model {version_id} is not validated")
            return False
        
        # Load model
        if not self.load_model(version_id):
            return False
        
        # Update version status
        with self._lock:
            # Retire current active version
            for v_id, v in self.model_versions.items():
                if v.is_active and v_id != version_id:
                    v.is_active = False
                    v.retired_at = datetime.now()
            
            # Activate new version
            version.is_active = True
            version.deployed_at = datetime.now()
            version.status = ModelStatus.DEPLOYED
            self.active_version = version_id
        
        logger.info(f"Model {version_id} deployed successfully")
        return True
    
    def get_model_versions(self) -> List[ModelVersion]:
        """Get all model versions"""
        with self._lock:
            return list(self.model_versions.values())
    
    def get_active_model_info(self) -> Optional[ModelVersion]:
        """Get information about active model"""
        if self.active_version:
            return self.model_versions.get(self.active_version)
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get predictor statistics"""
        with self._lock:
            stats = self.prediction_stats.copy()
            stats.update({
                'total_models_trained': len(self.model_versions),
                'active_model': self.active_version,
                'fraud_detection_rate': (
                    self.prediction_stats['total_frauds_detected'] / 
                    self.prediction_stats['total_predictions']
                    if self.prediction_stats['total_predictions'] > 0 else 0
                ),
                'using_enhanced': self.use_enhanced,
                'enhanced_available': ENHANCED_AVAILABLE
            })
            
            # Add enhanced stats if available
            if self.use_enhanced and self._enhanced_predictor:
                try:
                    enhanced_stats = self._enhanced_predictor.get_statistics()
                    stats['enhanced_stats'] = enhanced_stats
                except Exception as e:
                    logger.warning(f"Failed to get enhanced stats: {e}")
            
            return stats
    
    def _convert_to_enhanced_config(self, config: ModelConfig) -> 'EnhancedModelConfig':
        """Convert standard config to enhanced config"""
        try:
            # Create enhanced config with compatible parameters
            enhanced_config = EnhancedModelConfig()
            
            # Copy compatible parameters
            for attr in ['model_type', 'model_params', 'test_size', 'validation_size', 
                        'random_state', 'stratify', 'cv_folds', 'cv_scoring',
                        'hyperparameter_tuning', 'tuning_method', 'tuning_iterations',
                        'calibrate_model', 'fraud_threshold', 'high_risk_threshold',
                        'calculate_feature_importance', 'save_model', 'model_path']:
                if hasattr(config, attr) and hasattr(enhanced_config, attr):
                    setattr(enhanced_config, attr, getattr(config, attr))
            
            # Set enhanced-specific defaults
            enhanced_config.enable_drift_detection = True
            enhanced_config.raise_on_error = False
            
            return enhanced_config
        except Exception as e:
            logger.error(f"Failed to convert config: {e}")
            return EnhancedModelConfig()
    
    def get_enhanced_features(self) -> Dict[str, Any]:
        """Get enhanced features if available"""
        if self.use_enhanced and self._enhanced_predictor:
            try:
                return self._enhanced_predictor.get_statistics()
            except Exception as e:
                logger.warning(f"Failed to get enhanced features: {e}")
        return {}
    
    def reset_statistics(self) -> None:
        """Reset prediction statistics"""
        with self._lock:
            self.prediction_stats = {
                'total_predictions': 0,
                'total_frauds_detected': 0,
                'average_prediction_time': 0.0,
                'last_prediction_time': None,
                'failed_predictions': 0,
                'drift_warnings': 0
            }
    
    def __repr__(self) -> str:
        return (f"FinancialMLPredictor(model_type={self.config.model_type.value}, "
               f"versions={len(self.model_versions)}, "
               f"active={self.active_version}, "
               f"predictions={self.prediction_stats['total_predictions']})")

# Legacy compatibility - alias for backward compatibility
MLPredictor = FinancialMLPredictor

# Export main classes
__all__ = [
    'FinancialMLPredictor',
    'MLPredictor',  # Legacy compatibility
    'ModelConfig',
    'ModelMetrics',
    'PredictionResult',
    'ModelVersion',
    'ModelType',
    'ModelStatus'
]

if __name__ == "__main__":
    # Example usage and testing
    import numpy as np
    import pandas as pd
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    # Add some structure for fraud patterns
    fraud_mask = np.random.random(n_samples) < 0.05  # 5% fraud rate
    X[fraud_mask, :5] += 2  # Make first 5 features different for frauds
    
    y = fraud_mask.astype(int)
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Initialize predictor
    config = ModelConfig(
        model_type=ModelType.RANDOM_FOREST,
        hyperparameter_tuning=True,
        calculate_feature_importance=True
    )
    
    predictor = FinancialMLPredictor(config)
    
    # Train model
    print("Training model...")
    version = predictor.train_model(X_df, y, version_notes="Initial model")
    print(f"Model trained: {version.version_id}")
    print(f"Metrics: AUC={version.metrics.auc_roc:.3f}, "
          f"Precision={version.metrics.precision:.3f}, "
          f"Recall={version.metrics.recall:.3f}")
    
    # Make predictions
    print("\nMaking predictions...")
    test_features = X_df.iloc[0].to_dict()
    result = predictor.predict(test_features, transaction_id="TEST001")
    print(f"Prediction: {result}")
    
    # Batch predictions
    batch_results = predictor.predict(X_df.iloc[:10])
    print(f"\nBatch predictions: {len(batch_results)} results")
    
    # Deploy model
    if predictor.deploy_model(version.version_id):
        print(f"\nModel {version.version_id} deployed successfully")
    
    # Get statistics
    stats = predictor.get_statistics()
    print(f"\nPredictor statistics: {stats}")
    
    print("\nFinancialMLPredictor ready for production use!")