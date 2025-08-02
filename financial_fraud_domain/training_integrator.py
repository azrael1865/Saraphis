"""
Training Integrator for Financial Fraud Detection
Enhanced integrator with IEEE fraud dataset support and Brain system integration
"""

# DISABLE ALL FALLBACKS - Force real components to identify failure points
DISABLE_FALLBACKS = True  # FORCE ALL REAL COMPONENTS - NO FALLBACKS

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Import IEEE data loader
try:
    from financial_fraud_domain.ieee_fraud_data_loader import IEEEFraudDataLoader, load_ieee_fraud_data
    IEEE_LOADER_AVAILABLE = True
except ImportError as e:
    if DISABLE_FALLBACKS:
        raise ImportError(f"FALLBACK DISABLED: IEEE fraud data loader import failed: {e}")
    IEEE_LOADER_AVAILABLE = False
    logger.warning(f"IEEE fraud data loader not available: {e}")

# Import ML components
try:
    from financial_fraud_domain.enhanced_ml_framework import ModelConfig, ModelType, ValidationLevel
    from financial_fraud_domain.enhanced_ml_models import EnhancedRandomForestModel
    ML_FRAMEWORK_AVAILABLE = True
except ImportError as e:
    if DISABLE_FALLBACKS:
        raise ImportError(f"FALLBACK DISABLED: ML framework import failed: {e}")
    ML_FRAMEWORK_AVAILABLE = False
    logger.warning(f"ML framework not available: {e}")

# Import domain components
try:
    from financial_fraud_domain.enhanced_fraud_core_exceptions import FraudTrainingError, FraudDataError
    DOMAIN_COMPONENTS_AVAILABLE = True
except ImportError as e:
    DOMAIN_COMPONENTS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Domain components not available: {e}")

# Configure logging
logger = logging.getLogger(__name__)


class IEEEFraudTrainingIntegrator:
    """Enhanced training integrator with IEEE fraud dataset support"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize training integrator with IEEE dataset support
        
        Args:
            config: Configuration dictionary containing:
                - data_dir: Path to IEEE fraud dataset
                - model_config: ML model configuration
                - training_params: Training parameters
                - brain_integration: Brain system integration settings
        """
        self.config = config or {}
        self.data_dir = self.config.get('data_dir', 'training_data/ieee-fraud-detection')
        self.brain_integration_enabled = self.config.get('brain_integration', {}).get('enabled', False)
        
        # Initialize IEEE data loader
        if IEEE_LOADER_AVAILABLE:
            self.data_loader = IEEEFraudDataLoader(
                data_dir=self.data_dir,
                fraud_domain_config=self.config.get('fraud_domain_config', {})
            )
        else:
            self.data_loader = None
            logger.warning("IEEE data loader not available - training will use fallback data")
        
        # Training state
        self.active_training_sessions = {}
        self.training_history = []
        
        logger.info("IEEEFraudTrainingIntegrator initialized")
    
    def start_ieee_training(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start training with IEEE fraud dataset
        
        Args:
            training_config: Training configuration containing:
                - model_type: Type of model to train
                - training_params: Training parameters (epochs, batch_size, etc.)
                - validation_split: Validation data split ratio
                - use_cache: Whether to use cached processed data
                
        Returns:
            Training session information
        """
        print(f"DEBUG: start_ieee_training called with config keys: {list(training_config.keys())}")
        try:
            print(f"DEBUG: Checking IEEE_LOADER_AVAILABLE: {IEEE_LOADER_AVAILABLE}")
            if not IEEE_LOADER_AVAILABLE:
                print(f"DEBUG: IEEE loader not available, raising FraudTrainingError")
                raise FraudTrainingError("IEEE data loader not available")
            
            print(f"DEBUG: Generating session ID")
            session_id = f"ieee_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"DEBUG: Generated session ID: {session_id}")
            
            logger.info(f"Starting IEEE fraud training session: {session_id}")
            
            # Load IEEE dataset
            print(f"DEBUG: Getting validation_split and use_cache from config")
            validation_split = training_config.get('validation_split', 0.2)
            use_cache = training_config.get('use_cache', True)
            print(f"DEBUG: validation_split={validation_split}, use_cache={use_cache}")
            
            print(f"DEBUG: About to call self.data_loader.load_data")
            X_train, y_train, val_data = self.data_loader.load_data(
                dataset_type="train",
                validation_split=validation_split,
                use_cache=use_cache
            )
            print(f"DEBUG: data_loader.load_data completed")
            print(f"DEBUG: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            
            # Prepare model configuration
            print(f"DEBUG: About to call _prepare_model_config")
            model_config = self._prepare_model_config(training_config, X_train.shape[1])
            print(f"DEBUG: _prepare_model_config completed")
            
            # Initialize training session
            print(f"DEBUG: Creating training session dict")
            training_session = {
                'session_id': session_id,
                'status': 'initializing',
                'start_time': datetime.now(),
                'config': training_config,
                'model_config': model_config,
                'data_info': {
                    'train_samples': X_train.shape[0],
                    'features': X_train.shape[1],
                    'fraud_rate': y_train.mean(),
                    'validation_samples': val_data[0].shape[0] if val_data else 0
                },
                'progress': 0.0,
                'current_epoch': 0,
                'metrics': {},
                'brain_integration': self.brain_integration_enabled
            }
            
            # Store training session
            print(f"DEBUG: Storing training session {session_id}")
            self.active_training_sessions[session_id] = training_session
            print(f"DEBUG: Training session stored successfully")
            
            # Execute training based on model type
            print(f"DEBUG: Checking ML_FRAMEWORK_AVAILABLE: {ML_FRAMEWORK_AVAILABLE}")
            if ML_FRAMEWORK_AVAILABLE:
                print(f"DEBUG: About to call _execute_ml_training")
                print(f"DEBUG: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
                if isinstance(model_config, dict):
                    print(f"DEBUG: model_config keys: {list(model_config.keys())}")
                else:
                    print(f"DEBUG: model_config is ModelConfig object: {type(model_config)}")
                training_result = self._execute_ml_training(
                    session_id, X_train, y_train, val_data, model_config
                )
                print(f"DEBUG: _execute_ml_training completed")
                print(f"DEBUG: Training result: {training_result}")
            else:
                print(f"DEBUG: ML framework not available, raising error")
                # FALLBACK DISABLED - FORCE REAL ML FRAMEWORK
                raise RuntimeError("ML framework fallback disabled - must use real ML framework training")
            
            # Update session with results
            print(f"DEBUG: Updating session with training results")
            training_session.update(training_result)
            training_session['status'] = 'completed' if training_result.get('success', False) else 'failed'
            print(f"DEBUG: Session status set to: {training_session['status']}")
            training_session['end_time'] = datetime.now()
            training_session['duration'] = (training_session['end_time'] - training_session['start_time']).total_seconds()
            print(f"DEBUG: Training duration: {training_session['duration']} seconds")
            
            # Move to history
            print(f"DEBUG: Moving session to history")
            self.training_history.append(training_session)
            if session_id in self.active_training_sessions:
                del self.active_training_sessions[session_id]
            print(f"DEBUG: Session moved to history successfully")
            
            logger.info(f"IEEE training session {session_id} completed: {training_session['status']}")
            print(f"DEBUG: Returning training session with status: {training_session['status']}")
            
            return training_session
            
        except Exception as e:
            print(f"DEBUG: Exception in start_ieee_training: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            logger.error(f"IEEE training failed: {e}")
            if DOMAIN_COMPONENTS_AVAILABLE:
                print(f"DEBUG: DOMAIN_COMPONENTS_AVAILABLE=True, raising FraudTrainingError")
                raise FraudTrainingError(f"IEEE training failed: {e}")
            else:
                print(f"DEBUG: DOMAIN_COMPONENTS_AVAILABLE=False, raising generic Exception")
                raise Exception(f"IEEE training failed: {e}")
    
    def _prepare_model_config(self, training_config: Dict[str, Any], n_features: int) -> Union[ModelConfig, Dict[str, Any]]:
        """Prepare model configuration for IEEE dataset"""
        model_type = training_config.get('model_type', 'lstm')  # Default to LSTM for neural network training
        
        if ML_FRAMEWORK_AVAILABLE:
            # Use enhanced ML framework configuration with dynamic model type selection
            if model_type in ['neural_network', 'lstm', 'transformer']:
                selected_model_type = getattr(ModelType, model_type.upper())
            elif model_type == 'random_forest':
                selected_model_type = ModelType.RANDOM_FOREST  
            elif model_type == 'gradient_boosting':
                selected_model_type = ModelType.GRADIENT_BOOSTING
            else:
                # Default to neural network for fraud detection
                selected_model_type = ModelType.NEURAL_NETWORK
            
            # Set appropriate model parameters based on model type
            model_params = training_config.get('model_params', {})
            if selected_model_type == ModelType.LSTM and not model_params:
                # Set memory-efficient LSTM defaults for fraud detection
                model_params = {
                    'sequence_length': 10,  # Reduced from 60 to prevent memory explosion
                    'hidden_dim': 64,       # Smaller than default 128
                    'num_layers': 2,
                    'dropout_rate': 0.2,
                    'batch_size': 32,
                    'epochs': 50,
                    'learning_rate': 0.001,
                    'use_attention': True
                }
                logger.info(f"Using memory-efficient LSTM defaults for fraud detection: {model_params}")
            
            model_config = ModelConfig(
                model_type=selected_model_type,
                validation_level=ValidationLevel.COMPREHENSIVE,
                model_params=model_params,
                preprocessing_config=training_config.get('preprocessing_config', {}),
                random_state=training_config.get('random_state', 42)
            )
            return model_config  # Return the ModelConfig object directly, not __dict__
        else:
            # Fallback configuration
            return {
                'model_type': model_type,
                'n_features': n_features,
                'model_params': training_config.get('model_params', {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                })
            }
    
    def _execute_ml_training(self, session_id: str, X_train: np.ndarray, y_train: np.ndarray,
                           val_data: Optional[Tuple[np.ndarray, np.ndarray]], 
                           model_config: Union[ModelConfig, Dict[str, Any]]) -> Dict[str, Any]:
        """Execute training using enhanced ML framework"""
        print(f"DEBUG: _execute_ml_training called for session {session_id}")
        try:
            # Handle model configuration object
            print(f"DEBUG: Processing model_config")
            print(f"DEBUG: model_config type: {type(model_config)}")
            
            if isinstance(model_config, ModelConfig):
                print(f"DEBUG: model_config is already a ModelConfig object")
                config = model_config
            else:
                print(f"DEBUG: Creating ModelConfig from dict")
                print(f"DEBUG: model_config: {model_config}")
                config = ModelConfig(**model_config)
                print(f"DEBUG: ModelConfig created from dict")
                
            print(f"DEBUG: Final config type: {type(config)}")
            print(f"DEBUG: config isinstance ModelConfig: {isinstance(config, ModelConfig)}")
            
            # Initialize model using ModelFactory for dynamic model selection
            from financial_fraud_domain.enhanced_ml_models import ModelFactory
            print(f"DEBUG: Creating model using ModelFactory for type: {config.model_type}")
            print(f"DEBUG: About to call ModelFactory.create_model({config.model_type}, config)")
            model = ModelFactory.create_model(config.model_type, config)
            print(f"DEBUG: Model initialized successfully: {type(model).__name__}")
            
            # Train model
            print(f"DEBUG: Starting model training")
            print(f"DEBUG: CONFIRMING REAL TRAINING - X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
            print(f"DEBUG: Total samples being trained on: {X_train.shape[0]:,}")
            print(f"DEBUG: Memory usage of X_train: {X_train.nbytes / (1024*1024):.1f} MB")
            print(f"DEBUG: Data type: {X_train.dtype}")
            logger.info(f"Training {config.model_type} model with {X_train.shape[0]} samples")
            print(f"DEBUG: About to call model.fit() on REAL DATA")
            
            import time
            start_time = time.time()
            training_result = model.fit(X_train, y_train)
            training_time = time.time() - start_time
            print(f"DEBUG: model.fit() completed in {training_time:.2f} seconds")
            print(f"DEBUG: model.fit() completed")
            print(f"DEBUG: Training result from model.fit(): {training_result}")
            
            # Evaluate on validation set if available
            print(f"DEBUG: Checking validation data")
            val_metrics = {}
            if val_data:
                print(f"DEBUG: Validation data available, evaluating model")
                X_val, y_val = val_data
                print(f"DEBUG: Validation data shapes - X_val: {X_val.shape}, y_val: {y_val.shape}")
                
                print(f"DEBUG: About to call model.predict()")
                val_predictions = model.predict(X_val)
                print(f"DEBUG: model.predict() completed")
                
                print(f"DEBUG: About to call model.predict_proba()")
                val_probabilities = model.predict_proba(X_val)
                print(f"DEBUG: model.predict_proba() completed")
                
                # Calculate validation metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
                
                print(f"DEBUG: Calculating validation metrics")
                val_metrics = {
                    'val_accuracy': accuracy_score(y_val, val_predictions),
                    'val_precision': precision_score(y_val, val_predictions),
                    'val_recall': recall_score(y_val, val_predictions),
                    'val_auc': roc_auc_score(y_val, val_probabilities[:, 1])
                }
                print(f"DEBUG: Validation metrics calculated: {val_metrics}")
            else:
                print(f"DEBUG: No validation data provided")
            
            print(f"DEBUG: Preparing return result for _execute_ml_training")
            result = {
                'success': True,
                'model': model,
                'training_metrics': training_result,
                'validation_metrics': val_metrics,
                'feature_importances': model.get_feature_importance() if hasattr(model, 'get_feature_importance') else None
            }
            print(f"DEBUG: _execute_ml_training returning success=True")
            return result
            
        except Exception as e:
            print(f"DEBUG: Exception in _execute_ml_training: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            logger.error(f"ML training failed: {e}")
            logger.error(f"Exception type: {type(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            result = {
                'success': False,
                'error': str(e),
                'error_type': 'ml_training_error'
            }
            print(f"DEBUG: _execute_ml_training returning success=False with error: {e}")
            return result
    
    def _execute_fallback_training(self, session_id: str, X_train: np.ndarray, y_train: np.ndarray,
                                 val_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Execute fallback training using basic sklearn"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, classification_report
            
            # Initialize basic model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            # Train model
            logger.info(f"Training fallback RandomForest with {X_train.shape[0]} samples")
            model.fit(X_train, y_train)
            
            # Get training accuracy
            train_predictions = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_predictions)
            
            # Evaluate on validation set if available
            val_metrics = {}
            if val_data:
                X_val, y_val = val_data
                val_predictions = model.predict(X_val)
                val_accuracy = accuracy_score(y_val, val_predictions)
                val_metrics = {
                    'val_accuracy': val_accuracy,
                    'val_classification_report': classification_report(y_val, val_predictions, output_dict=True)
                }
            
            return {
                'success': True,
                'model': model,
                'training_metrics': {
                    'train_accuracy': train_accuracy,
                    'model_type': 'RandomForestClassifier'
                },
                'validation_metrics': val_metrics,
                'feature_importances': model.feature_importances_.tolist()
            }
            
        except Exception as e:
            logger.error(f"Fallback training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'fallback_training_error'
            }
    
    def monitor_training(self, session_id: str) -> Dict[str, Any]:
        """Monitor active training session"""
        if session_id in self.active_training_sessions:
            session = self.active_training_sessions[session_id]
            return {
                'session_id': session_id,
                'status': session['status'],
                'progress': session['progress'],
                'current_epoch': session['current_epoch'],
                'metrics': session['metrics'],
                'data_info': session['data_info']
            }
        else:
            # Check training history
            for session in self.training_history:
                if session['session_id'] == session_id:
                    return {
                        'session_id': session_id,
                        'status': session['status'],
                        'duration': session.get('duration', 0),
                        'final_metrics': session.get('validation_metrics', {}),
                        'data_info': session['data_info']
                    }
            
            return {
                'session_id': session_id,
                'status': 'not_found',
                'error': 'Training session not found'
            }
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get complete training history"""
        return self.training_history
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Get data quality report from IEEE dataset"""
        if self.data_loader:
            return self.data_loader.get_data_quality_report()
        else:
            return {'error': 'IEEE data loader not available'}
    
    def clear_training_cache(self) -> bool:
        """Clear IEEE dataset cache"""
        if self.data_loader:
            return self.data_loader.clear_cache()
        return False


# Legacy compatibility class
class TrainingIntegrator(IEEEFraudTrainingIntegrator):
    """Legacy training integrator for backward compatibility"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        logger.info("Legacy TrainingIntegrator initialized (using IEEE implementation)")
    
    def start_training(self, training_config: Dict[str, Any]) -> bool:
        """Start model training (legacy interface)"""
        try:
            result = self.start_ieee_training(training_config)
            return result.get('success', False)
        except Exception as e:
            logger.error(f"Legacy training failed: {e}")
            return False
    
    def complete_training(self, training_id: str) -> bool:
        """Complete model training (legacy interface)"""
        session_info = self.monitor_training(training_id)
        return session_info.get('status') == 'completed'

if __name__ == "__main__":
    integrator = TrainingIntegrator()
    print("Training integrator initialized")