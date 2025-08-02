"""
Enhanced ML Predictor - Chunk 2: Individual Model Implementations with Error Handling
Comprehensive ML model implementations with advanced error handling, validation,
and monitoring for financial fraud detection systems.
"""

import logging
import numpy as np
import pandas as pd
import warnings
import time
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod

# ML library imports with fallback handling
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

# PyTorch imports for neural networks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import MinMaxScaler
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None

# Define LSTMFinancialNetwork directly (adapted from universal_ai_core)
LSTM_FINANCIAL_AVAILABLE = PYTORCH_AVAILABLE  # Available if PyTorch is available

if PYTORCH_AVAILABLE:
    class LSTMFinancialNetwork(nn.Module):
        """
        LSTM-based neural network for financial time series prediction.
        Adapted from molecular neural networks with financial-specific optimizations.
        """
        
        def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, 
                     output_dim: int = 1, dropout_rate: float = 0.2, use_attention: bool = True):
            super(LSTMFinancialNetwork, self).__init__()
            
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.output_dim = output_dim
            self.use_attention = use_attention
            
            # LSTM layers
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout_rate if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )
            
            # Attention mechanism
            if self.use_attention:
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=8,
                    dropout=dropout_rate,
                    batch_first=True
                )
                self.attention_norm = nn.LayerNorm(hidden_dim)
            
            # Output layers
            self.dropout = nn.Dropout(dropout_rate)
            self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
            
            # Initialize weights
            self._initialize_weights()
        
        def _initialize_weights(self):
            """Initialize network weights"""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.LSTM):
                    for name, param in module.named_parameters():
                        if 'weight' in name:
                            nn.init.xavier_uniform_(param)
                        elif 'bias' in name:
                            nn.init.constant_(param, 0)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through the network"""
            # LSTM forward pass
            lstm_out, (hidden, cell) = self.lstm(x)
            
            # Apply attention if enabled
            if self.use_attention:
                attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                lstm_out = self.attention_norm(attended_out + lstm_out)
            
            # Use the last time step output
            last_output = lstm_out[:, -1, :]
            
            # Apply dropout
            last_output = self.dropout(last_output)
            
            # Main prediction path
            x = torch.relu(self.fc1(last_output))
            x = self.dropout(x)
            output = self.fc2(x)
            
            return output
else:
    LSTMFinancialNetwork = None

# Sklearn imports
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, BaggingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

# Import enhanced framework components
try:
    from financial_fraud_domain.enhanced_ml_framework import (
        BaseModel, ModelConfig, ModelType, ValidationLevel,
        MLPredictorError, ModelNotFittedError, InvalidInputError,
        ModelValidationError, ModelTrainingError, PredictionError,
        HyperparameterError, InputValidator,
        validate_input, handle_errors, monitor_performance
    )
    FRAMEWORK_AVAILABLE = True
except ImportError as e:
    FRAMEWORK_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Enhanced ML framework not available: {e}")

# Configure logging
logger = logging.getLogger(__name__)

# Import IEEE data loader
try:
    from ieee_fraud_data_loader import IEEEFraudDataLoader, load_ieee_fraud_data
    IEEE_LOADER_AVAILABLE = True
except ImportError as e:
    IEEE_LOADER_AVAILABLE = False
    logger.warning(f"IEEE fraud data loader not available: {e}")

# ======================== ENHANCED RANDOM FOREST MODEL ========================

class EnhancedRandomForestModel(BaseModel):
    """Enhanced Random Forest implementation with comprehensive validation and monitoring"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.oob_score_value = None
        self.feature_importances_std = None
        
    def create_model(self):
        """Create Random Forest model with enhanced parameter validation"""
        try:
            # Set default parameters optimized for fraud detection
            default_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "bootstrap": True,
                "oob_score": True,
                "n_jobs": self.config.n_jobs,
                "random_state": self.config.random_state,
                "class_weight": "balanced",  # Handle class imbalance
                "warm_start": False,
                "ccp_alpha": 0.0
            }
            
            # Merge with user parameters
            params = {**default_params, **self.config.model_params}
            
            # Enhanced parameter validation
            self._validate_rf_params(params)
            
            # Create model
            model = RandomForestClassifier(**params)
            
            logger.info(f"Random Forest created with {params['n_estimators']} estimators")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create Random Forest model: {str(e)}")
            raise ModelTrainingError(f"Random Forest creation failed: {str(e)}", 
                                   training_stage="model_creation")
    
    def _validate_rf_params(self, params: Dict[str, Any]):
        """Validate Random Forest specific parameters"""
        # Check n_estimators
        if params["n_estimators"] <= 0:
            raise HyperparameterError("n_estimators must be positive")
        if params["n_estimators"] > 1000:
            logger.warning(f"Large n_estimators ({params['n_estimators']}) may be slow")
        
        # Check max_depth
        if params["max_depth"] is not None and params["max_depth"] <= 0:
            raise HyperparameterError("max_depth must be positive or None")
        
        # Check min_samples_split
        if params["min_samples_split"] < 2:
            raise HyperparameterError("min_samples_split must be at least 2")
        
        # Check min_samples_leaf
        if params["min_samples_leaf"] < 1:
            raise HyperparameterError("min_samples_leaf must be at least 1")
        
        # Check max_features
        valid_max_features = ["auto", "sqrt", "log2", None]
        if isinstance(params["max_features"], str) and params["max_features"] not in valid_max_features:
            raise HyperparameterError(f"max_features must be one of {valid_max_features}")
        
        # Logical consistency checks
        if params["max_depth"] is not None and params["max_depth"] < 3:
            logger.warning("Very shallow max_depth may lead to underfitting")
        
        if params["min_samples_split"] > params["min_samples_leaf"] * 2:
            logger.warning("min_samples_split should be at least 2 * min_samples_leaf")
    
    @monitor_performance('rf_fit')
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """Enhanced fit method with OOB score tracking"""
        result = super().fit(X, y)
        
        # Store OOB score if available
        if hasattr(self.model, 'oob_score_') and self.model.oob_score_:
            self.oob_score_value = self.model.oob_score_
            logger.info(f"Random Forest OOB score: {self.oob_score_value:.3f}")
        
        # Calculate feature importance standard deviation
        if hasattr(self.model, 'estimators_'):
            importances = np.array([tree.feature_importances_ for tree in self.model.estimators_])
            self.feature_importances_std = np.std(importances, axis=0)
        
        return result
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Random Forest with enhanced metrics"""
        if not self.fitted or not self.feature_names:
            return {}
            
        try:
            importance_dict = {}
            
            # Basic feature importance
            if hasattr(self.model, 'feature_importances_'):
                for i, importance in enumerate(self.model.feature_importances_):
                    if i < len(self.feature_names):
                        importance_dict[self.feature_names[i]] = float(importance)
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Failed to get Random Forest feature importance: {str(e)}")
            return {}
    
    def get_oob_score(self) -> Optional[float]:
        """Get out-of-bag score if available"""
        return self.oob_score_value
    
    def get_feature_importance_std(self) -> Dict[str, float]:
        """Get feature importance standard deviation across trees"""
        if not self.fitted or not self.feature_names or self.feature_importances_std is None:
            return {}
            
        try:
            return {
                self.feature_names[i]: float(std)
                for i, std in enumerate(self.feature_importances_std)
                if i < len(self.feature_names)
            }
        except Exception as e:
            logger.error(f"Failed to get feature importance std: {str(e)}")
            return {}

# ======================== ENHANCED XGBOOST MODEL ========================

class EnhancedXGBoostModel(BaseModel):
    """Enhanced XGBoost implementation with comprehensive validation and monitoring"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.booster = None
        self.evals_result = {}
        
    def create_model(self):
        """Create XGBoost model with enhanced parameter validation"""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, falling back to Gradient Boosting")
            return self._create_gradient_boosting_fallback()
        
        try:
            # Set default parameters optimized for fraud detection
            default_params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "gamma": 0,
                "reg_alpha": 0,
                "reg_lambda": 1,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "use_label_encoder": False,
                "random_state": self.config.random_state,
                "n_jobs": self.config.n_jobs,
                "verbosity": 0,
                "scale_pos_weight": 1  # Will be adjusted for class imbalance
            }
            
            # Merge with user parameters
            params = {**default_params, **self.config.model_params}
            
            # Enhanced parameter validation
            self._validate_xgb_params(params)
            
            # Create model
            model = xgb.XGBClassifier(**params)
            
            logger.info(f"XGBoost created with {params['n_estimators']} estimators")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create XGBoost model: {str(e)}")
            logger.info("Falling back to Gradient Boosting")
            return self._create_gradient_boosting_fallback()
    
    def _create_gradient_boosting_fallback(self):
        """Create Gradient Boosting fallback when XGBoost is not available"""
        try:
            fallback_params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "random_state": self.config.random_state,
                "validation_fraction": 0.1,
                "n_iter_no_change": 10,
                "tol": 1e-4
            }
            
            # Map XGBoost params to GradientBoosting params
            xgb_params = self.config.model_params
            if "max_depth" in xgb_params:
                fallback_params["max_depth"] = xgb_params["max_depth"]
            if "learning_rate" in xgb_params:
                fallback_params["learning_rate"] = xgb_params["learning_rate"]
            if "n_estimators" in xgb_params:
                fallback_params["n_estimators"] = xgb_params["n_estimators"]
            if "subsample" in xgb_params:
                fallback_params["subsample"] = xgb_params["subsample"]
            
            return GradientBoostingClassifier(**fallback_params)
            
        except Exception as e:
            logger.error(f"Fallback model creation failed: {str(e)}")
            raise ModelTrainingError(f"All gradient boosting models failed: {str(e)}")
    
    def _validate_xgb_params(self, params: Dict[str, Any]):
        """Validate XGBoost specific parameters"""
        # Check n_estimators
        if params["n_estimators"] <= 0:
            raise HyperparameterError("n_estimators must be positive")
        if params["n_estimators"] > 1000:
            logger.warning(f"Large n_estimators ({params['n_estimators']}) may be slow")
        
        # Check max_depth
        if params["max_depth"] <= 0:
            raise HyperparameterError("max_depth must be positive")
        if params["max_depth"] > 20:
            logger.warning(f"Very deep trees ({params['max_depth']}) may overfit")
        
        # Check learning_rate
        if not 0 < params["learning_rate"] <= 1:
            raise HyperparameterError("learning_rate must be between 0 and 1")
        
        # Check subsample
        if not 0 < params["subsample"] <= 1:
            raise HyperparameterError("subsample must be between 0 and 1")
        
        # Check colsample_bytree
        if not 0 < params["colsample_bytree"] <= 1:
            raise HyperparameterError("colsample_bytree must be between 0 and 1")
        
        # Check regularization parameters
        if params["gamma"] < 0:
            raise HyperparameterError("gamma must be non-negative")
        if params["reg_alpha"] < 0:
            raise HyperparameterError("reg_alpha must be non-negative")
        if params["reg_lambda"] < 0:
            raise HyperparameterError("reg_lambda must be non-negative")
        
        # Logical consistency checks
        if params["learning_rate"] < 0.01:
            logger.warning("Very low learning_rate may require more estimators")
        if params["subsample"] < 0.5:
            logger.warning("Very low subsample may hurt performance")
    
    @monitor_performance('xgb_fit')
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """Enhanced fit method with evaluation tracking"""
        # Adjust scale_pos_weight for class imbalance
        if hasattr(self.model, 'scale_pos_weight'):
            y_values = y.values if isinstance(y, pd.Series) else y
            pos_count = np.sum(y_values == 1)
            neg_count = np.sum(y_values == 0)
            if pos_count > 0 and neg_count > 0:
                scale_weight = neg_count / pos_count
                self.model.set_params(scale_pos_weight=scale_weight)
                logger.info(f"Adjusted scale_pos_weight to {scale_weight:.2f}")
        
        result = super().fit(X, y)
        
        # Store evaluation results if available
        if hasattr(self.model, 'evals_result_'):
            self.evals_result = self.model.evals_result_
        
        return result
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from XGBoost with enhanced metrics"""
        if not self.fitted or not self.feature_names:
            return {}
            
        try:
            importance_dict = {}
            
            # Try to get importance from booster
            if XGBOOST_AVAILABLE and hasattr(self.model, 'get_booster'):
                booster = self.model.get_booster()
                importance = booster.get_score(importance_type='gain')
                
                # Map feature indices to names
                for feature_idx, imp_value in importance.items():
                    if feature_idx.startswith('f') and feature_idx[1:].isdigit():
                        idx = int(feature_idx[1:])
                        if idx < len(self.feature_names):
                            importance_dict[self.feature_names[idx]] = float(imp_value)
                    elif feature_idx in self.feature_names:
                        importance_dict[feature_idx] = float(imp_value)
            
            # Fallback to feature_importances_ if available
            elif hasattr(self.model, 'feature_importances_'):
                for i, importance in enumerate(self.model.feature_importances_):
                    if i < len(self.feature_names):
                        importance_dict[self.feature_names[i]] = float(importance)
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Failed to get XGBoost feature importance: {str(e)}")
            return {}
    
    def get_eval_results(self) -> Dict[str, Any]:
        """Get evaluation results from training"""
        return self.evals_result

# ======================== ENHANCED LIGHTGBM MODEL ========================

class EnhancedLightGBMModel(BaseModel):
    """Enhanced LightGBM implementation with comprehensive validation and monitoring"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.booster = None
        self.evals_result = {}
        
    def create_model(self):
        """Create LightGBM model with enhanced parameter validation"""
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available, falling back to Random Forest")
            return self._create_random_forest_fallback()
        
        try:
            # Set default parameters optimized for fraud detection
            default_params = {
                "n_estimators": 100,
                "num_leaves": 31,
                "max_depth": -1,
                "learning_rate": 0.1,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "lambda_l1": 0,
                "lambda_l2": 0,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "random_state": self.config.random_state,
                "n_jobs": self.config.n_jobs,
                "verbose": -1,
                "class_weight": "balanced"
            }
            
            # Merge with user parameters
            params = {**default_params, **self.config.model_params}
            
            # Enhanced parameter validation
            self._validate_lgb_params(params)
            
            # Create model
            model = lgb.LGBMClassifier(**params)
            
            logger.info(f"LightGBM created with {params['n_estimators']} estimators")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create LightGBM model: {str(e)}")
            logger.info("Falling back to Random Forest")
            return self._create_random_forest_fallback()
    
    def _create_random_forest_fallback(self):
        """Create Random Forest fallback when LightGBM is not available"""
        try:
            fallback_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "bootstrap": True,
                "oob_score": True,
                "n_jobs": self.config.n_jobs,
                "random_state": self.config.random_state,
                "class_weight": "balanced"
            }
            
            # Map LightGBM params to RandomForest params
            lgb_params = self.config.model_params
            if "n_estimators" in lgb_params:
                fallback_params["n_estimators"] = lgb_params["n_estimators"]
            if "max_depth" in lgb_params and lgb_params["max_depth"] > 0:
                fallback_params["max_depth"] = lgb_params["max_depth"]
            if "min_child_samples" in lgb_params:
                fallback_params["min_samples_leaf"] = lgb_params["min_child_samples"]
            
            return RandomForestClassifier(**fallback_params)
            
        except Exception as e:
            logger.error(f"Fallback model creation failed: {str(e)}")
            raise ModelTrainingError(f"All tree-based models failed: {str(e)}")
    
    def _validate_lgb_params(self, params: Dict[str, Any]):
        """Validate LightGBM specific parameters"""
        # Check n_estimators
        if params["n_estimators"] <= 0:
            raise HyperparameterError("n_estimators must be positive")
        if params["n_estimators"] > 1000:
            logger.warning(f"Large n_estimators ({params['n_estimators']}) may be slow")
        
        # Check num_leaves
        if params["num_leaves"] < 2:
            raise HyperparameterError("num_leaves must be at least 2")
        if params["num_leaves"] > 1000:
            logger.warning(f"Large num_leaves ({params['num_leaves']}) may overfit")
        
        # Check learning_rate
        if not 0 < params["learning_rate"] <= 1:
            raise HyperparameterError("learning_rate must be between 0 and 1")
        
        # Check feature_fraction
        if not 0 < params["feature_fraction"] <= 1:
            raise HyperparameterError("feature_fraction must be between 0 and 1")
        
        # Check bagging_fraction
        if not 0 < params["bagging_fraction"] <= 1:
            raise HyperparameterError("bagging_fraction must be between 0 and 1")
        
        # Check regularization parameters
        if params["lambda_l1"] < 0:
            raise HyperparameterError("lambda_l1 must be non-negative")
        if params["lambda_l2"] < 0:
            raise HyperparameterError("lambda_l2 must be non-negative")
        
        # Check min_child_samples
        if params["min_child_samples"] < 1:
            raise HyperparameterError("min_child_samples must be at least 1")
        
        # Logical consistency checks
        if params["max_depth"] > 0 and params["max_depth"] < 3:
            logger.warning("Very shallow max_depth may lead to underfitting")
        if params["num_leaves"] >= 2 ** params["max_depth"] if params["max_depth"] > 0 else False:
            logger.warning("num_leaves should be less than 2^max_depth for best performance")
    
    @monitor_performance('lgb_fit')
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """Enhanced fit method with evaluation tracking"""
        result = super().fit(X, y)
        
        # Store evaluation results if available
        if hasattr(self.model, 'evals_result_'):
            self.evals_result = self.model.evals_result_
        
        return result
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from LightGBM with enhanced metrics"""
        if not self.fitted or not self.feature_names:
            return {}
            
        try:
            importance_dict = {}
            
            # Get importance from feature_importances_
            if hasattr(self.model, 'feature_importances_'):
                for i, importance in enumerate(self.model.feature_importances_):
                    if i < len(self.feature_names):
                        importance_dict[self.feature_names[i]] = float(importance)
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Failed to get LightGBM feature importance: {str(e)}")
            return {}
    
    def get_eval_results(self) -> Dict[str, Any]:
        """Get evaluation results from training"""
        return self.evals_result

# ======================== ENHANCED LOGISTIC REGRESSION MODEL ========================

class EnhancedLogisticRegressionModel(BaseModel):
    """Enhanced Logistic Regression implementation with comprehensive validation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.coefficients = None
        self.intercept = None
        self.scaler = None
        
    def create_model(self):
        """Create Logistic Regression model with enhanced parameter validation"""
        try:
            # Set default parameters optimized for fraud detection
            default_params = {
                "C": 1.0,
                "penalty": "l2",
                "solver": "liblinear",
                "max_iter": 1000,
                "tol": 1e-4,
                "random_state": self.config.random_state,
                "class_weight": "balanced",
                "warm_start": False,
                "fit_intercept": True,
                "intercept_scaling": 1.0
            }
            
            # Merge with user parameters
            params = {**default_params, **self.config.model_params}
            
            # Enhanced parameter validation
            self._validate_lr_params(params)
            
            # Create model
            model = LogisticRegression(**params)
            
            logger.info(f"Logistic Regression created with C={params['C']}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create Logistic Regression model: {str(e)}")
            raise ModelTrainingError(f"Logistic Regression creation failed: {str(e)}", 
                                   training_stage="model_creation")
    
    def _validate_lr_params(self, params: Dict[str, Any]):
        """Validate Logistic Regression specific parameters"""
        # Check C parameter
        if params["C"] <= 0:
            raise HyperparameterError("C must be positive")
        
        # Check penalty
        valid_penalties = ["l1", "l2", "elasticnet", "none"]
        if params["penalty"] not in valid_penalties:
            raise HyperparameterError(f"penalty must be one of {valid_penalties}")
        
        # Check solver
        valid_solvers = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
        if params["solver"] not in valid_solvers:
            raise HyperparameterError(f"solver must be one of {valid_solvers}")
        
        # Check max_iter
        if params["max_iter"] <= 0:
            raise HyperparameterError("max_iter must be positive")
        
        # Check tol
        if params["tol"] <= 0:
            raise HyperparameterError("tol must be positive")
        
        # Check solver-penalty combinations
        if params["penalty"] == "l1" and params["solver"] not in ["liblinear", "saga"]:
            raise HyperparameterError("L1 penalty requires liblinear or saga solver")
        
        if params["penalty"] == "elasticnet" and params["solver"] != "saga":
            raise HyperparameterError("elasticnet penalty requires saga solver")
        
        # Performance warnings
        if params["C"] > 100:
            logger.warning("Large C value may lead to overfitting")
        if params["C"] < 0.01:
            logger.warning("Small C value may lead to underfitting")
        if params["max_iter"] > 5000:
            logger.warning("Large max_iter may indicate convergence issues")
    
    @monitor_performance('lr_fit')
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """Enhanced fit method with feature scaling"""
        # Scale features for better convergence
        if self.config.validation_level != ValidationLevel.NONE:
            self.scaler = StandardScaler()
            if isinstance(X, pd.DataFrame):
                X_scaled = self.scaler.fit_transform(X)
                X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            else:
                X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        result = super().fit(X_scaled, y)
        
        # Store coefficients and intercept
        if hasattr(self.model, 'coef_'):
            self.coefficients = self.model.coef_[0] if self.model.coef_.ndim > 1 else self.model.coef_
        if hasattr(self.model, 'intercept_'):
            self.intercept = self.model.intercept_[0] if isinstance(self.model.intercept_, np.ndarray) else self.model.intercept_
        
        return result
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Enhanced predict method with feature scaling"""
        if self.scaler is not None:
            if isinstance(X, pd.DataFrame):
                X_scaled = self.scaler.transform(X)
                X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            else:
                X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return super().predict(X_scaled)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Enhanced predict_proba method with feature scaling"""
        if self.scaler is not None:
            if isinstance(X, pd.DataFrame):
                X_scaled = self.scaler.transform(X)
                X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            else:
                X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return super().predict_proba(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Logistic Regression coefficients"""
        if not self.fitted or not self.feature_names or self.coefficients is None:
            return {}
            
        try:
            # Use absolute values of coefficients as importance
            importance_dict = {}
            abs_coefficients = np.abs(self.coefficients)
            
            for i, importance in enumerate(abs_coefficients):
                if i < len(self.feature_names):
                    importance_dict[self.feature_names[i]] = float(importance)
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Failed to get Logistic Regression feature importance: {str(e)}")
            return {}
    
    def get_coefficients(self) -> Dict[str, float]:
        """Get model coefficients"""
        if not self.fitted or not self.feature_names or self.coefficients is None:
            return {}
            
        try:
            coef_dict = {}
            for i, coef in enumerate(self.coefficients):
                if i < len(self.feature_names):
                    coef_dict[self.feature_names[i]] = float(coef)
            
            return coef_dict
            
        except Exception as e:
            logger.error(f"Failed to get coefficients: {str(e)}")
            return {}

# ======================== ENHANCED ENSEMBLE MODEL ========================

class EnhancedEnsembleModel(BaseModel):
    """Enhanced Ensemble model with comprehensive validation and monitoring"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_models = []
        self.model_weights = []
        self.ensemble_type = "voting"  # or "stacking"
        
    def create_model(self):
        """Create ensemble model with enhanced error handling"""
        try:
            # Define base models with error handling
            base_models = []
            
            # Add Random Forest
            try:
                rf_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                    class_weight="balanced"
                )
                base_models.append(('rf', rf_model))
                logger.info("Added Random Forest to ensemble")
            except Exception as e:
                logger.warning(f"Failed to add Random Forest to ensemble: {e}")
            
            # Add XGBoost if available
            if XGBOOST_AVAILABLE:
                try:
                    xgb_model = xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=self.config.random_state,
                        n_jobs=self.config.n_jobs,
                        use_label_encoder=False,
                        eval_metric='logloss'
                    )
                    base_models.append(('xgb', xgb_model))
                    logger.info("Added XGBoost to ensemble")
                except Exception as e:
                    logger.warning(f"Failed to add XGBoost to ensemble: {e}")
            
            # Add LightGBM if available
            if LIGHTGBM_AVAILABLE:
                try:
                    lgb_model = lgb.LGBMClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=self.config.random_state,
                        n_jobs=self.config.n_jobs,
                        verbose=-1,
                        class_weight="balanced"
                    )
                    base_models.append(('lgb', lgb_model))
                    logger.info("Added LightGBM to ensemble")
                except Exception as e:
                    logger.warning(f"Failed to add LightGBM to ensemble: {e}")
            
            # Add Logistic Regression
            try:
                lr_model = LogisticRegression(
                    max_iter=1000,
                    random_state=self.config.random_state,
                    class_weight="balanced"
                )
                base_models.append(('lr', lr_model))
                logger.info("Added Logistic Regression to ensemble")
            except Exception as e:
                logger.warning(f"Failed to add Logistic Regression to ensemble: {e}")
            
            # Check if we have enough models
            if len(base_models) < 2:
                raise ModelTrainingError("Ensemble requires at least 2 base models")
            
            # Create voting classifier
            voting_type = self.config.model_params.get('voting', 'soft')
            if voting_type not in ['hard', 'soft']:
                voting_type = 'soft'
            
            model = VotingClassifier(
                estimators=base_models,
                voting=voting_type,
                n_jobs=self.config.n_jobs
            )
            
            self.base_models = base_models
            logger.info(f"Ensemble created with {len(base_models)} base models")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create ensemble model: {str(e)}")
            raise ModelTrainingError(f"Ensemble creation failed: {str(e)}", 
                                   training_stage="model_creation")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get averaged feature importance from ensemble"""
        if not self.fitted or not self.feature_names:
            return {}
            
        try:
            importance_dict = {}
            importance_counts = {}
            
            # Get importance from each base model
            for name, model in self.model.estimators_:
                model_importance = {}
                
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models
                    for i, imp in enumerate(model.feature_importances_):
                        if i < len(self.feature_names):
                            feature_name = self.feature_names[i]
                            model_importance[feature_name] = float(imp)
                            
                elif hasattr(model, 'coef_'):
                    # Linear models
                    coefficients = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                    for i, coef in enumerate(np.abs(coefficients)):
                        if i < len(self.feature_names):
                            feature_name = self.feature_names[i]
                            model_importance[feature_name] = float(coef)
                
                # Aggregate importance
                for feature, importance in model_importance.items():
                    if feature not in importance_dict:
                        importance_dict[feature] = 0
                        importance_counts[feature] = 0
                    importance_dict[feature] += importance
                    importance_counts[feature] += 1
            
            # Average the importances
            for feature in importance_dict:
                if importance_counts[feature] > 0:
                    importance_dict[feature] /= importance_counts[feature]
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Failed to get ensemble feature importance: {str(e)}")
            return {}
    
    def get_base_model_predictions(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """Get predictions from each base model"""
        if not self.fitted:
            raise ModelNotFittedError("Ensemble must be fitted before getting base predictions")
        
        predictions = {}
        
        try:
            for name, model in self.model.estimators_:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)[:, 1]  # Get positive class probability
                else:
                    pred = model.predict(X)
                predictions[name] = pred
                
        except Exception as e:
            logger.error(f"Failed to get base model predictions: {str(e)}")
            
        return predictions

# ======================== NEURAL NETWORK MODELS ========================

class EnhancedLSTMModel(BaseModel):
    """Enhanced LSTM model wrapper integrating LSTMFinancialNetwork with BaseModel interface"""
    
    def __init__(self, config: ModelConfig):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTM models but is not available")
        if not LSTM_FINANCIAL_AVAILABLE:
            raise ImportError("LSTMFinancialNetwork is required but could not be imported")
        
        super().__init__(config)
        self.sequence_length = config.model_params.get('sequence_length', 60)
        self.scaler = MinMaxScaler()
        self.fitted_scaler = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.create_model()
    
    def create_model(self):
        """Create the LSTM model"""
        try:
            # Model will be fully initialized during fit when we know input dimensions
            self.model = None
            logger.info("LSTM model placeholder created - will initialize during fit")
        except Exception as e:
            logger.error(f"Failed to create LSTM model: {str(e)}")
            raise ModelCreationError(f"LSTM model creation failed: {str(e)}")
    
    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray = None) -> tuple:
        """Prepare time series sequences for LSTM training using vectorized operations"""
        try:
            # Scale the features
            if not self.fitted_scaler:
                X_scaled = self.scaler.fit_transform(X)
                self.fitted_scaler = True
            else:
                X_scaled = self.scaler.transform(X)
            
            num_samples = len(X_scaled)
            num_features = X_scaled.shape[1]
            num_sequences = num_samples - self.sequence_length
            
            if num_sequences <= 0:
                logger.warning(f"Not enough samples ({num_samples}) for sequence_length ({self.sequence_length})")
                return np.array([]), np.array([]) if y is not None else None
            
            logger.info(f"Creating {num_sequences} sequences using vectorized operations...")
            
            # Pre-allocate arrays for efficiency
            sequences = np.zeros((num_sequences, self.sequence_length, num_features), dtype=np.float32)
            
            # Vectorized sequence creation using array slicing
            for i in range(self.sequence_length):
                sequences[:, i, :] = X_scaled[i:i + num_sequences]
            
            if y is not None:
                targets = y[self.sequence_length:].astype(np.float32)
                return sequences, targets
            else:
                return sequences, None
                
        except Exception as e:
            logger.error(f"Failed to prepare LSTM sequences: {str(e)}")
            raise DataPreprocessingError(f"LSTM sequence preparation failed: {str(e)}")
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """Fit the LSTM model"""
        start_time = time.time()
        
        try:
            # Convert to numpy arrays
            if isinstance(X, pd.DataFrame):
                X_np = X.values
                self.feature_names = list(X.columns)
            else:
                X_np = X
                self.feature_names = [f"feature_{i}" for i in range(X_np.shape[1])]
            
            if isinstance(y, pd.Series):
                y_np = y.values
            else:
                y_np = y
            
            logger.info(f"Starting LSTM training with {X_np.shape[0]} samples, {X_np.shape[1]} features")
            
            # Prepare sequences
            X_sequences, y_sequences = self._prepare_sequences(X_np, y_np)
            
            if len(X_sequences) == 0:
                raise ValueError(f"Insufficient data for sequence creation. Need at least {self.sequence_length + 1} samples")
            
            logger.info(f"Created {len(X_sequences)} sequences of length {self.sequence_length}")
            
            # Initialize model now that we know dimensions
            input_dim = X_sequences.shape[2]
            hidden_dim = self.config.model_params.get('hidden_dim', 128)
            num_layers = self.config.model_params.get('num_layers', 2)
            dropout_rate = self.config.model_params.get('dropout_rate', 0.2)
            
            self.model = LSTMFinancialNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=1,  # Binary classification
                dropout_rate=dropout_rate,
                use_attention=self.config.model_params.get('use_attention', True)
            ).to(self.device)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_sequences).to(self.device)
            y_tensor = torch.FloatTensor(y_sequences.reshape(-1, 1)).to(self.device)
            
            # Create data loader
            dataset = TensorDataset(X_tensor, y_tensor)
            batch_size = self.config.model_params.get('batch_size', 32)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Setup optimizer and loss function
            learning_rate = self.config.model_params.get('learning_rate', 0.001)
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
            criterion = nn.BCEWithLogitsLoss()
            
            # Training loop
            epochs = self.config.model_params.get('epochs', 100)
            self.model.train()
            total_batches = len(dataloader)
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                    # Log progress every 1000 batches or at the end of epoch
                    if (batch_idx + 1) % 1000 == 0 or (batch_idx + 1) == total_batches:
                        current_avg_loss = epoch_loss / (batch_idx + 1)
                        progress_pct = ((batch_idx + 1) / total_batches) * 100
                        logger.info(f"Epoch {epoch}/{epochs}, Batch {batch_idx + 1}/{total_batches} ({progress_pct:.1f}%), Running Loss: {current_avg_loss:.4f}")
                
                avg_loss = epoch_loss / len(dataloader)
                logger.info(f"Epoch {epoch}/{epochs} COMPLETED, Final Average Loss: {avg_loss:.4f}")
            
            self.fitted = True
            self.training_time = time.time() - start_time
            
            logger.info(f"LSTM model training completed in {self.training_time:.2f} seconds")
            
            return {'training_time': self.training_time, 'epochs': epochs, 'final_loss': avg_loss}
            
        except Exception as e:
            logger.error(f"LSTM model training failed: {str(e)}")
            raise ModelTrainingError(f"LSTM training failed: {str(e)}")
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions using the trained LSTM model with batch-wise processing"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        start_time = time.time()
        
        try:
            # Convert to numpy
            if isinstance(X, pd.DataFrame):
                X_np = X.values
            else:
                X_np = X
            
            # Prepare sequences
            X_sequences, _ = self._prepare_sequences(X_np)
            
            if len(X_sequences) == 0:
                raise ValueError(f"Insufficient data for prediction. Need at least {self.sequence_length} samples")
            
            # MEMORY FIX: Process in batches instead of all at once
            self.model.eval()
            prediction_batch_size = min(20000, len(X_sequences))  # Conservative batch size
            all_predictions = []
            
            logger.info(f"Processing {len(X_sequences)} sequences in batches of {prediction_batch_size}")
            
            with torch.no_grad():
                for i in range(0, len(X_sequences), prediction_batch_size):
                    # Clear memory before each batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Get batch
                    end_idx = min(i + prediction_batch_size, len(X_sequences))
                    batch_sequences = X_sequences[i:end_idx]
                    
                    # Convert batch to tensor
                    X_batch_tensor = torch.FloatTensor(batch_sequences).to(self.device)
                    
                    # Process batch
                    batch_outputs = self.model(X_batch_tensor)
                    batch_predictions = torch.sigmoid(batch_outputs).cpu().numpy()
                    all_predictions.append(batch_predictions)
                    
                    # Clean up batch tensors
                    del X_batch_tensor, batch_outputs
                    
                    logger.info(f"Processed batch {i//prediction_batch_size + 1}/{(len(X_sequences) + prediction_batch_size - 1)//prediction_batch_size}")
            
            # Combine all predictions
            predictions = np.concatenate(all_predictions, axis=0)
            
            # Convert to binary predictions
            binary_predictions = (predictions > 0.5).astype(int).flatten()
            
            # Pad predictions to match input length
            full_predictions = np.zeros(len(X_np))
            full_predictions[self.sequence_length:] = binary_predictions
            # Use majority class for first sequence_length predictions
            majority_class = int(binary_predictions.mean() > 0.5)
            full_predictions[:self.sequence_length] = majority_class
            
            self.prediction_time = time.time() - start_time
            logger.info(f"Prediction completed in {self.prediction_time:.2f} seconds")
            
            return full_predictions
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {str(e)}")
            raise PredictionError(f"LSTM prediction failed: {str(e)}")
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities using the trained LSTM model with batch-wise processing"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        start_time = time.time()
        
        try:
            # Convert to numpy
            if isinstance(X, pd.DataFrame):
                X_np = X.values
            else:
                X_np = X
            
            # Prepare sequences
            X_sequences, _ = self._prepare_sequences(X_np)
            
            if len(X_sequences) == 0:
                raise ValueError(f"Insufficient data for prediction. Need at least {self.sequence_length} samples")
            
            # MEMORY FIX: Process in batches instead of all at once
            self.model.eval()
            prediction_batch_size = min(20000, len(X_sequences))  # Conservative batch size
            all_probabilities = []
            
            logger.info(f"Processing {len(X_sequences)} sequences for probabilities in batches of {prediction_batch_size}")
            
            with torch.no_grad():
                for i in range(0, len(X_sequences), prediction_batch_size):
                    # Clear memory before each batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Get batch
                    end_idx = min(i + prediction_batch_size, len(X_sequences))
                    batch_sequences = X_sequences[i:end_idx]
                    
                    # Convert batch to tensor
                    X_batch_tensor = torch.FloatTensor(batch_sequences).to(self.device)
                    
                    # Process batch
                    batch_outputs = self.model(X_batch_tensor)
                    batch_probabilities = torch.sigmoid(batch_outputs).cpu().numpy().flatten()
                    all_probabilities.append(batch_probabilities)
                    
                    # Clean up batch tensors
                    del X_batch_tensor, batch_outputs
            
            # Combine all probabilities
            probabilities = np.concatenate(all_probabilities, axis=0)
            
            # Create probability matrix for binary classification
            prob_matrix = np.zeros((len(X_np), 2))
            
            # Fill in probabilities for sequence predictions
            prob_matrix[self.sequence_length:, 1] = probabilities
            prob_matrix[self.sequence_length:, 0] = 1 - probabilities
            
            # Use average probability for first sequence_length predictions
            avg_prob = probabilities.mean()
            prob_matrix[:self.sequence_length, 1] = avg_prob
            prob_matrix[:self.sequence_length, 0] = 1 - avg_prob
            
            self.prediction_time = time.time() - start_time
            
            return prob_matrix
            
        except Exception as e:
            logger.error(f"LSTM probability prediction failed: {str(e)}")
            raise PredictionError(f"LSTM probability prediction failed: {str(e)}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance for LSTM model (approximated using attention weights)"""
        try:
            if not self.fitted or self.model is None:
                return {}
            
            # For LSTM, feature importance is challenging to compute directly
            # We'll return uniform importance for now, but could implement attention analysis
            if not self.feature_names:
                return {}
            
            # Simple uniform importance for now
            n_features = len(self.feature_names)
            uniform_importance = 1.0 / n_features
            
            importance_dict = {
                feature: uniform_importance for feature in self.feature_names
            }
            
            logger.info("LSTM feature importance computed (uniform distribution)")
            return importance_dict
            
        except Exception as e:
            logger.error(f"Failed to compute LSTM feature importance: {str(e)}")
            return {}

# ======================== MODEL FACTORY ========================

class ModelFactory:
    """Factory for creating ML models with enhanced error handling"""
    
    @staticmethod
    def create_model(model_type: ModelType, config: ModelConfig) -> BaseModel:
        """Create model instance based on type with comprehensive error handling"""
        try:
            if model_type == ModelType.RANDOM_FOREST:
                return EnhancedRandomForestModel(config)
            elif model_type == ModelType.XGBOOST:
                return EnhancedXGBoostModel(config)
            elif model_type == ModelType.LIGHTGBM:
                return EnhancedLightGBMModel(config)
            elif model_type == ModelType.LOGISTIC_REGRESSION:
                return EnhancedLogisticRegressionModel(config)
            elif model_type == ModelType.ENSEMBLE:
                return EnhancedEnsembleModel(config)
            elif model_type == ModelType.NEURAL_NETWORK:
                raise NotImplementedError(f"NEURAL_NETWORK model not yet implemented - requires neural network class")
            elif model_type == ModelType.LSTM:
                return EnhancedLSTMModel(config)
            elif model_type == ModelType.TRANSFORMER:
                raise NotImplementedError(f"TRANSFORMER model not yet implemented - requires transformer neural network class")
            else:
                logger.warning(f"Unsupported model type: {model_type}, falling back to Random Forest")
                return EnhancedRandomForestModel(config)
                
        except Exception as e:
            logger.error(f"Failed to create model {model_type}: {str(e)}")
            logger.info("Falling back to Random Forest")
            return EnhancedRandomForestModel(config)
    
    @staticmethod
    def get_available_models() -> List[ModelType]:
        """Get list of available model types"""
        available = [ModelType.RANDOM_FOREST, ModelType.LOGISTIC_REGRESSION, ModelType.ENSEMBLE]
        
        if XGBOOST_AVAILABLE:
            available.append(ModelType.XGBOOST)
        if LIGHTGBM_AVAILABLE:
            available.append(ModelType.LIGHTGBM)
        if PYTORCH_AVAILABLE and LSTM_FINANCIAL_AVAILABLE:
            available.append(ModelType.LSTM)
            
        return available

# Export model implementations
__all__ = [
    # Model implementations
    'EnhancedRandomForestModel',
    'EnhancedXGBoostModel', 
    'EnhancedLightGBMModel',
    'EnhancedLogisticRegressionModel',
    'EnhancedEnsembleModel',
    'EnhancedLSTMModel',
    
    # Factory
    'ModelFactory',
    
    # Availability flags
    'XGBOOST_AVAILABLE',
    'LIGHTGBM_AVAILABLE',
    'PYTORCH_AVAILABLE',
    'LSTM_FINANCIAL_AVAILABLE'
]

if __name__ == "__main__":
    print("Enhanced ML Models - Chunk 2: Individual model implementations with error handling loaded")
    
    # Test model creation
    try:
        from financial_fraud_domain.enhanced_ml_framework import ModelConfig, ModelType
        
        config = ModelConfig(model_type=ModelType.RANDOM_FOREST)
        model = ModelFactory.create_model(ModelType.RANDOM_FOREST, config)
        print(f"✓ Created {model.__class__.__name__}")
        
        available_models = ModelFactory.get_available_models()
        print(f"✓ Available models: {[m.value for m in available_models]}")
        
        print("✓ Enhanced ML Models Chunk 2 loaded successfully")
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")