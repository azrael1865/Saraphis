"""
Enhanced Fraud Detection Core - System Integration
Main system integration file that connects the enhanced fraud detection core
to the existing Saraphis financial fraud detection system
"""

import logging
import os
import sys
import threading
import time
import uuid
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Add the current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

# Import enhanced fraud core components
from enhanced_fraud_core_main import (
    EnhancedFraudDetectionCore, EnhancedFraudCoreConfig, FraudDetectionResult,
    create_default_fraud_core, create_production_fraud_core
)
from enhanced_fraud_core_integration import (
    IntegrationManager, IntegrationConfig, setup_complete_fraud_detection_system,
    run_system_health_check
)
from enhanced_fraud_core_exceptions import (
    EnhancedFraudException, DetectionError, ValidationError,
    DetectionStrategy, ValidationLevel, SecurityLevel
)
from enhanced_fraud_core_security import SecurityContext, SecurityManager

# Import existing system components (with fallback handling)
try:
    from enhanced_data_validator import EnhancedFinancialDataValidator
    ENHANCED_VALIDATOR_AVAILABLE = True
except ImportError:
    ENHANCED_VALIDATOR_AVAILABLE = False

try:
    from enhanced_transaction_validator import EnhancedTransactionFieldValidator
    ENHANCED_TRANSACTION_VALIDATOR_AVAILABLE = True
except ImportError:
    ENHANCED_TRANSACTION_VALIDATOR_AVAILABLE = False

try:
    from enhanced_data_loader import EnhancedFinancialDataLoader
    ENHANCED_DATA_LOADER_AVAILABLE = True
except ImportError:
    ENHANCED_DATA_LOADER_AVAILABLE = False

try:
    from enhanced_preprocessing_integration import PreprocessingIntegrationManager
    PREPROCESSING_INTEGRATION_AVAILABLE = True
except ImportError:
    PREPROCESSING_INTEGRATION_AVAILABLE = False

try:
    from enhanced_ml_integration import MLIntegrationManager
    ML_INTEGRATION_AVAILABLE = True
except ImportError:
    ML_INTEGRATION_AVAILABLE = False

try:
    from enhanced_proof_verifier import FinancialProofVerifier
    PROOF_VERIFIER_AVAILABLE = True
except ImportError:
    PROOF_VERIFIER_AVAILABLE = False

try:
    from proof_verifier import ProofType, ProofStatus, ProofLevel
    BASIC_PROOF_VERIFIER_AVAILABLE = True
except ImportError:
    BASIC_PROOF_VERIFIER_AVAILABLE = False

# Try to import Brain system components
try:
    from independent_core.brain import Brain, BrainSystemConfig
    from independent_core.domain_registry import DomainRegistry, DomainConfig, DomainStatus, DomainType
    from independent_core.domain_router import DomainRouter, RoutingStrategy
    from independent_core.domain_state import DomainStateManager
    BRAIN_SYSTEM_AVAILABLE = True
except ImportError:
    BRAIN_SYSTEM_AVAILABLE = False
    logger.warning("Brain system components not available")

# Try to import Brain fraud support
try:
    from brain_fraud_support import BrainFraudExtensions, create_fraud_enabled_brain
    BRAIN_FRAUD_SUPPORT_AVAILABLE = True
except ImportError:
    BRAIN_FRAUD_SUPPORT_AVAILABLE = False

# Try to import Brain fraud connector
try:
    from brain_fraud_connector import BrainFraudSystemConnector, BrainFraudConnectorConfig
    BRAIN_FRAUD_CONNECTOR_AVAILABLE = True
except ImportError:
    BRAIN_FRAUD_CONNECTOR_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================== SYSTEM INTEGRATION CLASS ========================

class SaraphisEnhancedFraudSystem:
    """
    Main system integration class that orchestrates all components
    of the enhanced fraud detection system within Saraphis
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the Saraphis enhanced fraud detection system"""
        self.config_file = config_file
        self.integration_manager = None
        self.fraud_core = None
        self.existing_components = {}
        self.system_initialized = False
        
        # Component availability flags
        self.component_availability = {
            'enhanced_validator': ENHANCED_VALIDATOR_AVAILABLE,
            'enhanced_transaction_validator': ENHANCED_TRANSACTION_VALIDATOR_AVAILABLE,
            'enhanced_data_loader': ENHANCED_DATA_LOADER_AVAILABLE,
            'preprocessing_integration': PREPROCESSING_INTEGRATION_AVAILABLE,
            'ml_integration': ML_INTEGRATION_AVAILABLE,
            'proof_verifier': PROOF_VERIFIER_AVAILABLE,
            'basic_proof_verifier': BASIC_PROOF_VERIFIER_AVAILABLE,
            'brain_system': BRAIN_SYSTEM_AVAILABLE,
            'brain_fraud_support': BRAIN_FRAUD_SUPPORT_AVAILABLE,
            'brain_fraud_connector': BRAIN_FRAUD_CONNECTOR_AVAILABLE
        }
        
        # Brain integration components
        self.brain = None
        self.brain_fraud_connector = None
        self.brain_integration_enabled = False
        self.brain_performance_metrics = {
            'total_detections': 0,
            'brain_used_count': 0,
            'direct_used_count': 0,
            'average_latency_ms': 0.0,
            'success_rate': 1.0,
            'brain_usage_rate': 0.0
        }
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self) -> None:
        """Initialize the complete fraud detection system"""
        try:
            logger.info("Initializing Saraphis Enhanced Fraud Detection System...")
            
            # Initialize integration manager
            if self.config_file:
                self.integration_manager = setup_complete_fraud_detection_system(self.config_file)
            else:
                self.integration_manager = setup_complete_fraud_detection_system()
            
            # Get fraud core reference
            self.fraud_core = self.integration_manager.fraud_core
            
            # Initialize existing system components
            self._initialize_existing_components()
            
            # Setup integrations
            self._setup_component_integrations()
            
            # Initialize Brain integration
            self._initialize_brain_integration()
            
            self.system_initialized = True
            logger.info("Saraphis Enhanced Fraud Detection System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Saraphis Enhanced Fraud Detection System: {e}")
            raise
    
    def _initialize_existing_components(self) -> None:
        """Initialize existing system components"""
        
        # Enhanced Data Validator
        if self.component_availability['enhanced_validator']:
            try:
                self.existing_components['data_validator'] = EnhancedFinancialDataValidator()
                logger.info("Enhanced Data Validator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Enhanced Data Validator: {e}")
        
        # Enhanced Transaction Validator
        if self.component_availability['enhanced_transaction_validator']:
            try:
                self.existing_components['transaction_validator'] = EnhancedTransactionFieldValidator()
                logger.info("Enhanced Transaction Validator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Enhanced Transaction Validator: {e}")
        
        # Enhanced Data Loader
        if self.component_availability['enhanced_data_loader']:
            try:
                self.existing_components['data_loader'] = EnhancedFinancialDataLoader()
                logger.info("Enhanced Data Loader initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Enhanced Data Loader: {e}")
        
        # Preprocessing Integration Manager
        if self.component_availability['preprocessing_integration']:
            try:
                self.existing_components['preprocessing_manager'] = PreprocessingIntegrationManager()
                logger.info("Preprocessing Integration Manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Preprocessing Integration Manager: {e}")
        
        # ML Integration Manager
        if self.component_availability['ml_integration']:
            try:
                self.existing_components['ml_manager'] = MLIntegrationManager()
                logger.info("ML Integration Manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ML Integration Manager: {e}")
        
        # Proof Verifier
        if self.component_availability['proof_verifier']:
            try:
                self.existing_components['proof_verifier'] = FinancialProofVerifier()
                logger.info("Enhanced Proof Verifier initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Enhanced Proof Verifier: {e}")
        
        logger.info(f"Initialized {len(self.existing_components)} existing system components")
    
    def _initialize_brain_integration(self) -> None:
        """Initialize Brain system integration"""
        if not self.component_availability['brain_system']:
            logger.info("Brain system not available, running in standalone mode")
            return
        
        try:
            logger.info("Initializing Brain system integration...")
            
            # Initialize Brain with fraud support
            if self.component_availability['brain_fraud_support']:
                self.brain = create_fraud_enabled_brain()
                logger.info("Brain initialized with fraud support")
            else:
                # Initialize basic Brain and add fraud support manually
                brain_config = BrainSystemConfig(
                    base_path=Path.cwd() / ".brain_fraud",
                    enable_monitoring=True,
                    enable_adaptation=True,
                    max_domains=50
                )
                self.brain = Brain(brain_config)
                
                # Manually add fraud capabilities if available
                if self.component_availability['brain_fraud_support']:
                    from brain_fraud_support import BrainFraudExtensions
                    BrainFraudExtensions.enhance_brain_for_fraud(self.brain)
                
                logger.info("Brain initialized with manual fraud support")
            
            # Initialize Brain fraud connector if available
            if self.component_availability['brain_fraud_connector']:
                try:
                    connector_config = BrainFraudConnectorConfig(
                        fraud_domain_name="financial_fraud",
                        enable_auto_registration=True,
                        enable_state_sync=True,
                        enable_performance_monitoring=True,
                        cache_predictions=True,
                        cache_ttl_seconds=3600
                    )
                    
                    self.brain_fraud_connector = BrainFraudSystemConnector(
                        brain=self.brain,
                        fraud_system=self,
                        config=connector_config
                    )
                    
                    # Start connector
                    self.brain_fraud_connector.start()
                    
                    logger.info("Brain fraud connector initialized and started")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize Brain fraud connector: {e}")
                    self.brain_fraud_connector = None
            
            self.brain_integration_enabled = True
            logger.info("Brain integration completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Brain integration: {e}")
            self.brain_integration_enabled = False
            self.brain = None
            self.brain_fraud_connector = None
    
    def _setup_component_integrations(self) -> None:
        """Setup integrations between enhanced core and existing components"""
        
        # Integrate with data validation
        if 'data_validator' in self.existing_components:
            self._integrate_data_validator()
        
        # Integrate with transaction validation
        if 'transaction_validator' in self.existing_components:
            self._integrate_transaction_validator()
        
        # Integrate with data loading
        if 'data_loader' in self.existing_components:
            self._integrate_data_loader()
        
        # Integrate with preprocessing
        if 'preprocessing_manager' in self.existing_components:
            self._integrate_preprocessing()
        
        # Integrate with ML
        if 'ml_manager' in self.existing_components:
            self._integrate_ml()
        
        # Integrate with proof verification
        if 'proof_verifier' in self.existing_components:
            self._integrate_proof_verifier()
    
    def _integrate_data_validator(self) -> None:
        """Integrate with enhanced data validator"""
        try:
            data_validator = self.existing_components['data_validator']
            
            # Add data validator to fraud core's preprocessing pipeline
            original_preprocess = self.integration_manager._preprocess_transaction
            
            def enhanced_preprocess(transaction: Dict[str, Any]) -> Dict[str, Any]:
                # Run original preprocessing
                processed_transaction = original_preprocess(transaction)
                
                # Add data validation
                try:
                    validation_result = data_validator.validate_transaction_data(processed_transaction)
                    processed_transaction['_data_validation_result'] = validation_result
                except Exception as e:
                    logger.warning(f"Data validation failed: {e}")
                    processed_transaction['_data_validation_error'] = str(e)
                
                return processed_transaction
            
            self.integration_manager._preprocess_transaction = enhanced_preprocess
            logger.info("Data validator integration completed")
            
        except Exception as e:
            logger.error(f"Failed to integrate data validator: {e}")
    
    def _integrate_transaction_validator(self) -> None:
        """Integrate with enhanced transaction validator"""
        try:
            transaction_validator = self.existing_components['transaction_validator']
            
            # Add transaction validator to validation framework
            original_validate = self.fraud_core.validation_framework.validate_all
            
            def enhanced_validate(data: Dict[str, Any], context=None) -> Dict[str, Any]:
                # Run original validation
                validation_result = original_validate(data, context)
                
                # Add transaction field validation
                if 'transaction' in data:
                    try:
                        field_validation = transaction_validator.validate_fields(data['transaction'])
                        validation_result['transaction_field_validation'] = field_validation
                        
                        # Merge validation results
                        if not field_validation.get('is_valid', True):
                            validation_result['is_valid'] = False
                            validation_result['errors'].extend(field_validation.get('errors', []))
                        
                    except Exception as e:
                        logger.warning(f"Transaction field validation failed: {e}")
                        validation_result['warnings'].append(f"Transaction field validation error: {e}")
                
                return validation_result
            
            self.fraud_core.validation_framework.validate_all = enhanced_validate
            logger.info("Transaction validator integration completed")
            
        except Exception as e:
            logger.error(f"Failed to integrate transaction validator: {e}")
    
    def _integrate_data_loader(self) -> None:
        """Integrate with enhanced data loader"""
        try:
            data_loader = self.existing_components['data_loader']
            
            # Add data enrichment capability
            original_preprocess = self.integration_manager._preprocess_transaction
            
            def enriched_preprocess(transaction: Dict[str, Any]) -> Dict[str, Any]:
                # Run original preprocessing
                processed_transaction = original_preprocess(transaction)
                
                # Add data enrichment
                try:
                    enriched_data = data_loader.enrich_transaction_data(processed_transaction)
                    processed_transaction.update(enriched_data)
                except Exception as e:
                    logger.warning(f"Data enrichment failed: {e}")
                    processed_transaction['_enrichment_error'] = str(e)
                
                return processed_transaction
            
            self.integration_manager._preprocess_transaction = enriched_preprocess
            logger.info("Data loader integration completed")
            
        except Exception as e:
            logger.error(f"Failed to integrate data loader: {e}")
    
    def _integrate_preprocessing(self) -> None:
        """Integrate with preprocessing manager"""
        try:
            preprocessing_manager = self.existing_components['preprocessing_manager']
            
            # Add preprocessing to fraud detection pipeline
            original_preprocess = self.integration_manager._preprocess_transaction
            
            def comprehensive_preprocess(transaction: Dict[str, Any]) -> Dict[str, Any]:
                # Run original preprocessing
                processed_transaction = original_preprocess(transaction)
                
                # Add comprehensive preprocessing
                try:
                    preprocessed_data = preprocessing_manager.process_transaction(processed_transaction)
                    processed_transaction.update(preprocessed_data)
                except Exception as e:
                    logger.warning(f"Comprehensive preprocessing failed: {e}")
                    processed_transaction['_preprocessing_error'] = str(e)
                
                return processed_transaction
            
            self.integration_manager._preprocess_transaction = comprehensive_preprocess
            logger.info("Preprocessing integration completed")
            
        except Exception as e:
            logger.error(f"Failed to integrate preprocessing: {e}")
    
    def _integrate_ml(self) -> None:
        """Integrate with ML manager"""
        try:
            ml_manager = self.existing_components['ml_manager']
            
            # Enhance ML prediction capabilities
            if hasattr(self.fraud_core, 'ml_predictor') and self.fraud_core.ml_predictor:
                original_predict = self.fraud_core.ml_predictor.predict_fraud
                
                def enhanced_predict(transaction: Dict[str, Any]) -> Dict[str, Any]:
                    # Run original prediction
                    original_result = original_predict(transaction)
                    
                    # Add ML manager predictions
                    try:
                        ml_result = ml_manager.predict_fraud(transaction)
                        # Merge results
                        enhanced_result = original_result.copy()
                        enhanced_result['ml_manager_prediction'] = ml_result
                        
                        # Combine fraud probabilities (weighted average)
                        original_prob = original_result.get('fraud_probability', 0.0)
                        ml_prob = ml_result.get('fraud_probability', 0.0)
                        enhanced_result['fraud_probability'] = (original_prob * 0.6 + ml_prob * 0.4)
                        
                        return enhanced_result
                    except Exception as e:
                        logger.warning(f"ML manager prediction failed: {e}")
                        return original_result
                
                self.fraud_core.ml_predictor.predict_fraud = enhanced_predict
                logger.info("ML integration completed")
            
        except Exception as e:
            logger.error(f"Failed to integrate ML: {e}")
    
    def _integrate_proof_verifier(self) -> None:
        """Integrate with proof verifier"""
        try:
            proof_verifier = self.existing_components['proof_verifier']
            
            # Add proof verification to post-processing
            original_postprocess = self.integration_manager._postprocess_result
            
            def verified_postprocess(result: FraudDetectionResult, 
                                   transaction: Dict[str, Any]) -> FraudDetectionResult:
                # Run original post-processing
                processed_result = original_postprocess(result, transaction)
                
                # Add proof verification
                try:
                    if result.fraud_detected:
                        proof_result = proof_verifier.verify_fraud_claim(
                            transaction, result.fraud_probability
                        )
                        processed_result.additional_metadata['proof_verification'] = proof_result
                        
                        # Adjust confidence based on proof verification
                        if proof_result.get('verified', False):
                            processed_result.confidence = min(processed_result.confidence + 0.1, 1.0)
                        
                except Exception as e:
                    logger.warning(f"Proof verification failed: {e}")
                    processed_result.additional_metadata['proof_verification_error'] = str(e)
                
                return processed_result
            
            self.integration_manager._postprocess_result = verified_postprocess
            logger.info("Proof verifier integration completed")
            
        except Exception as e:
            logger.error(f"Failed to integrate proof verifier: {e}")
    
    def detect_fraud(self, transaction: Dict[str, Any], 
                    user_context: Optional[Dict[str, Any]] = None,
                    use_brain: bool = True) -> FraudDetectionResult:
        """
        Main fraud detection method that integrates all system components
        
        Args:
            transaction: Transaction data for fraud detection
            user_context: Optional user context for security
            use_brain: Whether to use Brain system if available (default: True)
            
        Returns:
            FraudDetectionResult with comprehensive fraud analysis
        """
        if not self.system_initialized:
            raise RuntimeError("System not initialized")
        
        try:
            start_time = time.time()
            
            # Create security context if user context provided
            security_context = None
            if user_context:
                security_context = SecurityContext(
                    user_id=user_context.get('user_id', 'unknown'),
                    session_id=user_context.get('session_id', 'unknown'),
                    ip_address=user_context.get('ip_address', '127.0.0.1'),
                    user_agent=user_context.get('user_agent', 'unknown'),
                    permissions=user_context.get('permissions', ['read']),
                    authentication_method=user_context.get('auth_method', 'unknown'),
                    timestamp=datetime.now(),
                    expires_at=datetime.now()
                )
            
            # Determine detection method
            result = None
            used_brain = False
            
            # Try Brain-based detection first if enabled and available
            if use_brain and self.brain_integration_enabled and self.brain:
                try:
                    if hasattr(self.brain, 'detect_fraud'):
                        result = self.brain.detect_fraud(transaction, user_context)
                        used_brain = True
                        logger.debug("Used Brain system for fraud detection")
                    elif self.brain_fraud_connector:
                        result = self.brain_fraud_connector.detect_fraud(
                            transaction, user_context, use_brain=True
                        )
                        used_brain = True
                        logger.debug("Used Brain fraud connector for detection")
                except Exception as e:
                    logger.warning(f"Brain fraud detection failed, falling back to direct: {e}")
                    result = None
            
            # Fall back to direct fraud detection if Brain failed or not available
            if result is None:
                result = self.integration_manager.detect_fraud_with_integration(
                    transaction, security_context
                )
                used_brain = False
                logger.debug("Used direct fraud detection")
            
            # Update performance metrics
            detection_time = (time.time() - start_time) * 1000  # Convert to ms
            self._update_performance_metrics(detection_time, used_brain, True)
            
            return result
            
        except Exception as e:
            logger.error(f"Integrated fraud detection failed: {e}")
            # Update performance metrics for failed detection
            detection_time = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0.0
            self._update_performance_metrics(detection_time, False, False)
            raise
    
    def batch_detect_fraud(self, transactions: List[Dict[str, Any]], 
                          user_context: Optional[Dict[str, Any]] = None,
                          use_brain: bool = True,
                          max_workers: int = 10) -> List[FraudDetectionResult]:
        """
        Batch fraud detection with full integration
        
        Args:
            transactions: List of transactions to analyze
            user_context: Optional user context for security
            use_brain: Whether to use Brain system if available
            max_workers: Maximum number of worker threads for parallel processing
            
        Returns:
            List of FraudDetectionResult objects
        """
        if not self.system_initialized:
            raise RuntimeError("System not initialized")
        
        if not transactions:
            return []
        
        # For small batches, process sequentially
        if len(transactions) <= 5:
            results = []
            for transaction in transactions:
                try:
                    result = self.detect_fraud(transaction, user_context, use_brain)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch fraud detection failed for transaction {transaction.get('transaction_id', 'unknown')}: {e}")
                    fallback_result = self._create_fallback_result(transaction, str(e))
                    results.append(fallback_result)
            return results
        
        # For larger batches, use parallel processing
        results = [None] * len(transactions)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all transactions
            future_to_index = {
                executor.submit(
                    self.detect_fraud, 
                    transaction, 
                    user_context, 
                    use_brain
                ): index
                for index, transaction in enumerate(transactions)
            }
            
            # Collect results
            for future in as_completed(future_to_index, timeout=60):
                index = future_to_index[future]
                transaction = transactions[index]
                
                try:
                    result = future.result(timeout=30)
                    results[index] = result
                except Exception as e:
                    logger.error(f"Batch fraud detection failed for transaction {transaction.get('transaction_id', 'unknown')}: {e}")
                    fallback_result = self._create_fallback_result(transaction, str(e))
                    results[index] = fallback_result
        
        return results
    
    def _create_fallback_result(self, transaction: Dict[str, Any], error_message: str) -> FraudDetectionResult:
        """Create a fallback result for failed detections"""
        return FraudDetectionResult(
            transaction_id=transaction.get('transaction_id', 'unknown'),
            fraud_detected=False,
            fraud_probability=0.0,
            risk_score=0.0,
            confidence=0.0,
            detection_strategy=DetectionStrategy.HYBRID,
            detection_time=0.0,
            timestamp=datetime.now(),
            explanation=f"Detection failed: {error_message}",
            validation_passed=False,
            validation_errors=[f"Detection error: {error_message}"]
        )
    
    def _update_performance_metrics(self, detection_time_ms: float, used_brain: bool, success: bool) -> None:
        """Update performance metrics"""
        try:
            self.brain_performance_metrics['total_detections'] += 1
            
            if used_brain:
                self.brain_performance_metrics['brain_used_count'] += 1
            else:
                self.brain_performance_metrics['direct_used_count'] += 1
            
            # Update average latency (running average)
            total = self.brain_performance_metrics['total_detections']
            current_avg = self.brain_performance_metrics['average_latency_ms']
            self.brain_performance_metrics['average_latency_ms'] = (
                (current_avg * (total - 1) + detection_time_ms) / total
            )
            
            # Update success rate
            if not success:
                success_count = self.brain_performance_metrics['success_rate'] * (total - 1)
                self.brain_performance_metrics['success_rate'] = success_count / total
            
            # Update brain usage rate
            brain_count = self.brain_performance_metrics['brain_used_count']
            self.brain_performance_metrics['brain_usage_rate'] = brain_count / total
            
        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")
    
    def monitor_fraud_performance(self) -> Dict[str, Any]:
        """Monitor fraud detection performance with Brain integration metrics"""
        try:
            # Get base performance from integration manager
            base_performance = self.integration_manager.monitor_fraud_performance()
            
            # Add Brain-specific metrics
            enhanced_performance = base_performance.copy()
            enhanced_performance['brain_integration'] = {
                'enabled': self.brain_integration_enabled,
                'metrics': self.brain_performance_metrics.copy()
            }
            
            # Add Brain system status if available
            if self.brain:
                try:
                    if hasattr(self.brain, 'monitor_fraud_performance'):
                        brain_metrics = self.brain.monitor_fraud_performance()
                        enhanced_performance['brain_system_metrics'] = brain_metrics
                except Exception as e:
                    enhanced_performance['brain_system_metrics_error'] = str(e)
            
            # Add connector metrics if available
            if self.brain_fraud_connector:
                try:
                    connector_metrics = self.brain_fraud_connector.get_performance_metrics()
                    enhanced_performance['brain_connector_metrics'] = connector_metrics
                except Exception as e:
                    enhanced_performance['brain_connector_metrics_error'] = str(e)
            
            return enhanced_performance
            
        except Exception as e:
            logger.error(f"Performance monitoring failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def configure_fraud_detection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure fraud detection settings across all systems"""
        results = {
            'success': True,
            'updates': [],
            'errors': []
        }
        
        try:
            # Configure integration manager
            if self.integration_manager:
                try:
                    integration_result = self.integration_manager.configure_fraud_detection(config)
                    results['integration_manager'] = integration_result
                    if integration_result.get('success'):
                        results['updates'].extend(integration_result.get('updated_settings', []))
                except Exception as e:
                    results['errors'].append(f"Integration manager configuration failed: {e}")
            
            # Configure Brain if available
            if self.brain and hasattr(self.brain, 'configure_fraud_detection'):
                try:
                    brain_result = self.brain.configure_fraud_detection(config)
                    results['brain_configuration'] = brain_result
                    if brain_result.get('success'):
                        results['updates'].extend(brain_result.get('updates', []))
                except Exception as e:
                    results['errors'].append(f"Brain configuration failed: {e}")
            
            # Configure Brain fraud connector if available
            if self.brain_fraud_connector:
                try:
                    connector_result = self.brain_fraud_connector.configure_fraud_detection(config)
                    results['connector_configuration'] = connector_result
                    if connector_result.get('success'):
                        results['updates'].extend(connector_result.get('updated_settings', []))
                except Exception as e:
                    results['errors'].append(f"Connector configuration failed: {e}")
            
            # Update local configuration
            if 'performance_monitoring' in config:
                # This would update local performance monitoring settings
                results['updates'].append('performance_monitoring')
            
            if results['errors']:
                results['success'] = False
            
        except Exception as e:
            results['success'] = False
            results['errors'].append(str(e))
            logger.error(f"Configuration failed: {e}")
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        if not self.system_initialized:
            return {'status': 'not_initialized'}
        
        status = {
            'system_initialized': self.system_initialized,
            'component_availability': self.component_availability,
            'active_components': list(self.existing_components.keys()),
            'integration_status': self.integration_manager.get_integration_status(),
            'brain_integration': {
                'enabled': self.brain_integration_enabled,
                'brain_available': self.brain is not None,
                'connector_available': self.brain_fraud_connector is not None,
                'integrated': self.brain_integration_enabled and self.brain is not None
            },
            'performance_metrics': self.brain_performance_metrics.copy(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add fraud core status
        if self.fraud_core:
            status['fraud_core_status'] = self.fraud_core.get_system_status()
        
        # Add Brain status if available
        if self.brain:
            try:
                if hasattr(self.brain, 'get_fraud_system_status'):
                    status['brain_fraud_status'] = self.brain.get_fraud_system_status()
                elif hasattr(self.brain, 'get_system_status'):
                    status['brain_status'] = self.brain.get_system_status()
            except Exception as e:
                status['brain_status_error'] = str(e)
        
        # Add Brain fraud connector status if available
        if self.brain_fraud_connector:
            try:
                status['brain_connector_status'] = self.brain_fraud_connector.get_status()
            except Exception as e:
                status['brain_connector_status_error'] = str(e)
        
        return status
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive system health check"""
        health_check = run_system_health_check(self.integration_manager)
        
        # Add Brain integration health check
        if self.brain_integration_enabled:
            brain_health = {
                'brain_connected': self.brain is not None,
                'brain_responsive': False,
                'domain_registered': False,
                'connector_healthy': False,
                'performance_acceptable': True
            }
            
            # Test Brain responsiveness
            if self.brain:
                try:
                    if hasattr(self.brain, 'get_fraud_system_status'):
                        brain_status = self.brain.get_fraud_system_status()
                        brain_health['brain_responsive'] = True
                        brain_health['domain_registered'] = brain_status.get('fraud_domain_registered', False)
                    elif hasattr(self.brain, 'get_system_status'):
                        self.brain.get_system_status()
                        brain_health['brain_responsive'] = True
                except Exception as e:
                    logger.warning(f"Brain health check failed: {e}")
            
            # Test connector health
            if self.brain_fraud_connector:
                try:
                    connector_status = self.brain_fraud_connector.get_status()
                    brain_health['connector_healthy'] = connector_status.get('initialized', False)
                except Exception as e:
                    logger.warning(f"Brain connector health check failed: {e}")
            
            # Check performance metrics
            if self.brain_performance_metrics['total_detections'] > 0:
                avg_latency = self.brain_performance_metrics['average_latency_ms']
                success_rate = self.brain_performance_metrics['success_rate']
                
                brain_health['performance_acceptable'] = (
                    avg_latency < 5000 and  # Less than 5 seconds
                    success_rate > 0.95     # Greater than 95% success rate
                )
            
            health_check['brain_integration_health'] = brain_health
        
        return health_check
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get information about all system components"""
        return {
            'enhanced_fraud_core': {
                'available': self.fraud_core is not None,
                'version': '1.0.0',
                'features': [
                    'Advanced validation framework',
                    'Error recovery and circuit breakers',
                    'Comprehensive monitoring',
                    'Security and audit system',
                    'Multiple detection strategies'
                ]
            },
            'existing_components': {
                name: {
                    'available': name in self.existing_components,
                    'integrated': name in self.existing_components
                }
                for name in [
                    'data_validator', 'transaction_validator', 'data_loader',
                    'preprocessing_manager', 'ml_manager', 'proof_verifier'
                ]
            },
            'integration_manager': {
                'available': self.integration_manager is not None,
                'initialized': self.integration_manager.initialized if self.integration_manager else False
            },
            'brain_system': {
                'available': self.brain is not None,
                'integration_enabled': self.brain_integration_enabled,
                'fraud_support': hasattr(self.brain, 'detect_fraud') if self.brain else False,
                'connector_available': self.brain_fraud_connector is not None,
                'performance_metrics': self.brain_performance_metrics.copy()
            }
        }
    
    def shutdown(self) -> None:
        """Shutdown the entire system"""
        logger.info("Shutting down Saraphis Enhanced Fraud Detection System...")
        
        # Shutdown Brain integration
        if self.brain_fraud_connector:
            try:
                self.brain_fraud_connector.shutdown()
                logger.info("Brain fraud connector shutdown")
            except Exception as e:
                logger.error(f"Failed to shutdown Brain fraud connector: {e}")
        
        if self.brain:
            try:
                if hasattr(self.brain, 'shutdown'):
                    self.brain.shutdown()
                logger.info("Brain system shutdown")
            except Exception as e:
                logger.error(f"Failed to shutdown Brain system: {e}")
        
        # Shutdown integration manager
        if self.integration_manager:
            self.integration_manager.shutdown()
        
        # Shutdown existing components
        for name, component in self.existing_components.items():
            try:
                if hasattr(component, 'shutdown'):
                    component.shutdown()
                logger.info(f"Component '{name}' shutdown")
            except Exception as e:
                logger.error(f"Failed to shutdown component '{name}': {e}")
        
        self.system_initialized = False
        logger.info("Saraphis Enhanced Fraud Detection System shutdown complete")

# ======================== UTILITY FUNCTIONS ========================

def create_saraphis_fraud_system(config_file: Optional[str] = None) -> SaraphisEnhancedFraudSystem:
    """Create and initialize Saraphis fraud detection system"""
    return SaraphisEnhancedFraudSystem(config_file)

def create_production_fraud_system(config_file: Optional[str] = None) -> SaraphisEnhancedFraudSystem:
    """Create production-ready fraud detection system with full Brain integration"""
    return create_saraphis_fraud_system(config_file)

def create_test_fraud_system() -> SaraphisEnhancedFraudSystem:
    """Create fraud detection system for testing purposes"""
    return create_saraphis_fraud_system()

def get_system_capabilities() -> Dict[str, Any]:
    """Get capabilities of the fraud detection system"""
    return {
        'detection_strategies': [strategy.value for strategy in DetectionStrategy],
        'validation_levels': [level.value for level in ValidationLevel],
        'security_levels': [level.value for level in SecurityLevel],
        'features': [
            'Real-time fraud detection',
            'Batch processing',
            'Multiple detection strategies',
            'Advanced validation',
            'Error recovery',
            'Comprehensive monitoring',
            'Security and audit',
            'Proof verification',
            'ML integration',
            'Data enrichment',
            'Performance optimization',
            'Brain system integration',
            'Dual access patterns',
            'Advanced routing',
            'Circuit breaker protection',
            'State synchronization',
            'Performance monitoring',
            'Fallback mechanisms'
        ],
        'integration_capabilities': [
            'Database storage',
            'Message queue notifications',
            'API integrations',
            'Webhook alerts',
            'External service enrichment',
            'Monitoring backends',
            'Audit logging',
            'Brain system integration',
            'Domain registration',
            'Routing and load balancing',
            'State management',
            'Performance analytics',
            'Auto-scaling support'
        ]
    }

def run_system_demonstration() -> Dict[str, Any]:
    """Run a demonstration of the fraud detection system"""
    logger.info("Starting Saraphis Enhanced Fraud Detection System demonstration...")
    
    # Initialize system
    system = create_saraphis_fraud_system()
    
    try:
        # Sample transactions for demonstration
        demo_transactions = [
            {
                'transaction_id': 'demo_normal_001',
                'user_id': 'demo_user_123',
                'amount': 85.50,
                'merchant_id': 'coffee_shop_abc',
                'timestamp': datetime.now().isoformat(),
                'type': 'purchase',
                'currency': 'USD',
                'description': 'Coffee and pastry'
            },
            {
                'transaction_id': 'demo_suspicious_001',
                'user_id': 'demo_user_456',
                'amount': 12500.00,
                'merchant_id': 'suspicious_merchant_xyz',
                'timestamp': datetime.now().isoformat(),
                'type': 'purchase',
                'currency': 'USD',
                'description': 'High-value electronics'
            },
            {
                'transaction_id': 'demo_midnight_001',
                'user_id': 'demo_user_789',
                'amount': 500.00,
                'merchant_id': 'atm_location_123',
                'timestamp': datetime.now().replace(hour=2, minute=30).isoformat(),
                'type': 'withdrawal',
                'currency': 'USD',
                'description': 'ATM withdrawal'
            }
        ]
        
        # Run fraud detection with both Brain and direct methods
        results = []
        brain_results = []
        direct_results = []
        
        # Test with Brain system if available
        if system.brain_integration_enabled:
            for transaction in demo_transactions:
                try:
                    brain_result = system.detect_fraud(transaction, use_brain=True)
                    brain_results.append(brain_result)
                except Exception as e:
                    logger.warning(f"Brain detection failed for {transaction['transaction_id']}: {e}")
        
        # Test with direct system
        for transaction in demo_transactions:
            try:
                direct_result = system.detect_fraud(transaction, use_brain=False)
                direct_results.append(direct_result)
            except Exception as e:
                logger.error(f"Direct detection failed for {transaction['transaction_id']}: {e}")
        
        # Use brain results if available, otherwise use direct results
        results = brain_results if brain_results else direct_results
        
        # Get system status
        system_status = system.get_system_status()
        
        # Get component info
        component_info = system.get_component_info()
        
        # Run health check
        health_check = system.run_health_check()
        
        demonstration_results = {
            'demonstration_completed': True,
            'transactions_processed': len(results),
            'fraud_detected_count': sum(1 for r in results if r.fraud_detected),
            'detection_results': [
                {
                    'transaction_id': r.transaction_id,
                    'fraud_detected': r.fraud_detected,
                    'fraud_probability': r.fraud_probability,
                    'risk_score': r.risk_score,
                    'confidence': r.confidence,
                    'detection_strategy': r.detection_strategy.value,
                    'detection_time': r.detection_time,
                    'explanation': r.explanation
                }
                for r in results
            ],
            'system_status': system_status,
            'component_info': component_info,
            'health_check': health_check,
            'demonstration_timestamp': datetime.now().isoformat()
        }
        
        logger.info("Demonstration completed successfully")
        return demonstration_results
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        return {
            'demonstration_completed': False,
            'error': str(e),
            'demonstration_timestamp': datetime.now().isoformat()
        }
    finally:
        # Cleanup
        system.shutdown()

# ======================== MAIN EXECUTION ========================

if __name__ == '__main__':
    print("Saraphis Enhanced Fraud Detection System")
    print("=" * 50)
    
    # Get system capabilities
    capabilities = get_system_capabilities()
    print("\nSystem Capabilities:")
    for category, items in capabilities.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for item in items:
            print(f"  - {item}")
    
    print("\n" + "=" * 50)
    print("Running System Demonstration...")
    print("=" * 50)
    
    # Run demonstration
    demo_results = run_system_demonstration()
    
    if demo_results['demonstration_completed']:
        print(f"\nDemonstration Results:")
        print(f"Transactions processed: {demo_results['transactions_processed']}")
        print(f"Fraud detected: {demo_results['fraud_detected_count']}")
        
        print("\nDetection Results:")
        for result in demo_results['detection_results']:
            print(f"  Transaction {result['transaction_id']}:")
            print(f"    Fraud detected: {result['fraud_detected']}")
            print(f"    Fraud probability: {result['fraud_probability']:.3f}")
            print(f"    Risk score: {result['risk_score']:.3f}")
            print(f"    Confidence: {result['confidence']:.3f}")
            print(f"    Detection time: {result['detection_time']:.3f}s")
            print(f"    Strategy: {result['detection_strategy']}")
            print(f"    Explanation: {result['explanation']}")
            print()
        
        print("System Status:")
        print(f"  System initialized: {demo_results['system_status']['system_initialized']}")
        print(f"  Active components: {len(demo_results['system_status']['active_components'])}")
        print(f"  Integration healthy: {demo_results['health_check']['overall_health']}")
        
    else:
        print(f"Demonstration failed: {demo_results['error']}")
    
    print("\n" + "=" * 50)
    print("Enhanced Fraud Detection System Ready for Production!")
    print("=" * 50)