"""
Enhanced Fraud Detection Core - Chunk 6: Main Core Class with Detection Logic
The main fraud detection class that orchestrates all components for comprehensive fraud detection
"""

import logging
import time
import threading
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import uuid
from functools import wraps
import traceback
from contextlib import contextmanager

# Import all core components
try:
    from enhanced_fraud_core_exceptions import (
        EnhancedFraudException, DetectionError, ValidationError, SecurityError,
        PerformanceError, ResourceError, DetectionStrategy, ValidationLevel,
        SecurityLevel, ErrorContext, create_error_context
    )
    from enhanced_fraud_core_validators import (
        ValidationFramework, ValidationConfig, TransactionValidator,
        StrategyValidator, ComponentValidator, PerformanceValidator,
        SecurityValidator
    )
    from enhanced_fraud_core_recovery import (
        ErrorRecoveryManager, ErrorRecoveryConfig, CircuitBreaker,
        CircuitBreakerConfig, with_recovery, with_circuit_breaker
    )
    from enhanced_fraud_core_monitoring import (
        MonitoringManager, MonitoringConfig, MetricsCollector,
        PerformanceProfiler, CacheManager, monitor_performance, with_caching
    )
    from enhanced_fraud_core_security import (
        SecurityManager, SecurityConfig, SecurityContext, ThreatDetector,
        AuditLogger
    )
except ImportError:
    # Fallback to absolute imports for standalone execution
    try:
        from enhanced_fraud_core_exceptions import (
            EnhancedFraudException, DetectionError, ValidationError, SecurityError,
            PerformanceError, ResourceError, DetectionStrategy, ValidationLevel,
            SecurityLevel, ErrorContext, create_error_context
        )
        from enhanced_fraud_core_validators import (
            ValidationFramework, ValidationConfig, TransactionValidator,
            StrategyValidator, ComponentValidator, PerformanceValidator,
            SecurityValidator
        )
        from enhanced_fraud_core_recovery import (
            ErrorRecoveryManager, ErrorRecoveryConfig, CircuitBreaker,
            CircuitBreakerConfig, with_recovery, with_circuit_breaker
        )
        from enhanced_fraud_core_monitoring import (
            MonitoringManager, MonitoringConfig, MetricsCollector,
            PerformanceProfiler, CacheManager, monitor_performance, with_caching
        )
        from enhanced_fraud_core_security import (
            SecurityManager, SecurityConfig, SecurityContext, ThreatDetector,
            AuditLogger
        )
    except ImportError as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Core components not available: {e}, using minimal fallbacks")
        # Create minimal fallback classes
        class EnhancedFraudException(Exception): pass
        class DetectionError(EnhancedFraudException): pass
        class ValidationError(EnhancedFraudException): pass
        class SecurityError(EnhancedFraudException): pass
        class PerformanceError(EnhancedFraudException): pass
        class ResourceError(EnhancedFraudException): pass
        
        class DetectionStrategy:
            AGGRESSIVE = "aggressive"
            BALANCED = "balanced"
            CONSERVATIVE = "conservative"
            HYBRID = "hybrid"
            RULES_ONLY = "rules_only"
            ML_ONLY = "ml_only"
            ENSEMBLE = "ensemble"
        
        class ValidationLevel:
            BASIC = "basic"
            STANDARD = "standard"
            COMPREHENSIVE = "comprehensive"
        
        class SecurityLevel:
            LOW = "low"
            MEDIUM = "medium"
            HIGH = "high"
        
        def create_error_context(operation, component=None, **kwargs):
            return {"operation": operation, "component": component, **kwargs}
        
        # Add missing config classes
        class ValidationConfig:
            def __init__(self): pass
        
        class ErrorRecoveryConfig:
            def __init__(self): pass
        
        class MonitoringConfig:
            def __init__(self): pass
        
        class SecurityConfig:
            def __init__(self): pass
        
        # Add minimal fallback manager classes
        class ValidationFramework:
            def __init__(self, config): 
                self.config = config
            def validate_all(self, data, context):
                return {'is_valid': True, 'errors': [], 'warnings': []}
            def get_performance_metrics(self):
                return {'validation_count': 0}
            def shutdown(self): pass
        
        class ErrorRecoveryManager:
            def __init__(self, config): 
                self.config = config
            def recover(self, fallback_func, error, context, *args):
                return fallback_func(*args)
            def get_recovery_metrics(self):
                return {'recovery_count': 0}
        
        class MonitoringManager:
            def __init__(self, config): 
                self.config = config
            def get_system_status(self):
                return {'status': 'active'}
            def shutdown(self): pass
        
        class SecurityManager:
            def __init__(self, config): 
                self.config = config
                self.threat_detector = type('ThreatDetector', (), {'analyze_request': lambda *args: []})()
                self.audit_logger = type('AuditLogger', (), {'log_event': lambda *args, **kwargs: None})()
            def get_security_dashboard(self):
                return {'threats': 0}
        
        # Add minimal security context
        class SecurityContext:
            def __init__(self, user_id=None, ip_address=None):
                self.user_id = user_id
                self.ip_address = ip_address
        
        # Add decorators as no-ops
        def monitor_performance(func):
            return func
        
        def with_recovery(func):
            return func

# Configure logging
logger = logging.getLogger(__name__)

# ======================== CORE CONFIGURATION ========================

@dataclass
class EnhancedFraudCoreConfig:
    """Comprehensive configuration for enhanced fraud detection core"""
    
    # Core detection settings
    detection_strategy: DetectionStrategy = DetectionStrategy.HYBRID
    enable_async_processing: bool = True
    max_worker_threads: int = 4
    max_worker_processes: int = 2
    batch_processing_size: int = 100
    
    # Validation settings
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    
    # Error recovery settings
    recovery_config: ErrorRecoveryConfig = field(default_factory=ErrorRecoveryConfig)
    
    # Monitoring settings
    monitoring_config: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Security settings
    security_config: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Performance settings
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600
    
    # Fraud detection thresholds
    fraud_probability_threshold: float = 0.7
    risk_score_threshold: float = 0.8
    confidence_threshold: float = 0.6
    
    # Integration settings
    enable_ml_integration: bool = True
    enable_rule_engine: bool = True
    enable_behavioral_analysis: bool = True
    enable_preprocessing: bool = True
    
    # Preprocessing settings
    preprocessing_config: dict = field(default_factory=lambda: {
        'feature_engineering': {
            'enable_time_features': True,
            'enable_amount_features': True,
            'enable_frequency_features': True,
            'enable_velocity_features': True,
            'enable_merchant_features': True,
            'enable_geographic_features': True
        },
        'data_quality': {
            'missing_value_threshold': 0.5,
            'outlier_method': 'iqr',
            'outlier_threshold': 3.0,
            'duplicate_threshold': 0.95
        },
        'feature_selection': {
            'method': 'mutual_info',
            'k_features': 50,
            'correlation_threshold': 0.9
        },
        'scaling': {
            'method': 'standard',
            'feature_range': [0, 1]
        }
    })
    
    # Audit and compliance
    enable_audit_logging: bool = True
    enable_compliance_checks: bool = True
    
    # Timeout settings
    detection_timeout: float = 30.0
    total_operation_timeout: float = 300.0

# ======================== DETECTION RESULT ========================

@dataclass
class FraudDetectionResult:
    """Comprehensive fraud detection result"""
    
    # Core detection results
    transaction_id: str
    fraud_detected: bool
    fraud_probability: float
    risk_score: float
    confidence: float
    
    # Detection details
    detection_strategy: DetectionStrategy
    detection_time: float
    timestamp: datetime
    
    # Rule-based results
    violated_rules: List[str] = field(default_factory=list)
    rule_scores: Dict[str, float] = field(default_factory=dict)
    
    # ML-based results
    ml_predictions: Dict[str, float] = field(default_factory=dict)
    model_confidence: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Behavioral analysis
    behavioral_anomalies: List[str] = field(default_factory=list)
    behavioral_score: float = 0.0
    
    # Explanation and reasoning
    explanation: str = ""
    reasoning_chain: List[str] = field(default_factory=list)
    
    # Metadata
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    additional_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Validation results
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

# ======================== MAIN FRAUD DETECTION CORE ========================

class EnhancedFraudDetectionCore:
    """Main enhanced fraud detection core class"""
    
    def __init__(self, config: EnhancedFraudCoreConfig):
        self.config = config
        self.initialized = False
        self.initialization_lock = threading.Lock()
        
        # Initialize core components
        self.validation_framework = ValidationFramework(config.validation_config)
        self.error_recovery_manager = ErrorRecoveryManager(config.recovery_config)
        self.monitoring_manager = MonitoringManager(config.monitoring_config)
        self.security_manager = SecurityManager(config.security_config)
        
        # Detection components
        self.rule_engine = None
        self.ml_predictor = None
        self.behavioral_analyzer = None
        self.preprocessing_manager = None
        
        # Thread pool for concurrent processing
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_worker_threads)
        
        # Initialize component integrations
        self._initialize_integrations()
        
        # Initialize preprocessing manager
        if self.config.enable_preprocessing:
            self._initialize_preprocessing()
        
        # Mark as initialized
        self.initialized = True
        logger.info("Enhanced fraud detection core initialized successfully")
    
    def _initialize_integrations(self) -> None:
        """Initialize integration with existing system components"""
        try:
            # Try to import and initialize existing components
            if self.config.enable_ml_integration:
                self._initialize_ml_integration()
            
            if self.config.enable_rule_engine:
                self._initialize_rule_engine()
            
            if self.config.enable_behavioral_analysis:
                self._initialize_behavioral_analysis()
                
        except Exception as e:
            logger.warning(f"Some integrations failed to initialize: {e}")
    
    def _initialize_preprocessing(self) -> None:
        """Initialize preprocessing manager"""
        try:
            self.preprocessing_manager = CompletePreprocessingManager(
                config=self.config.preprocessing_config
            )
            logger.info("Complete preprocessing manager initialized")
        except Exception as e:
            logger.warning(f"Preprocessing initialization failed: {e}")
    
    def _initialize_ml_integration(self) -> None:
        """Initialize ML integration"""
        try:
            # Try to import existing enhanced ML predictor
            try:
                from enhanced_ml_predictor import EnhancedMLPredictor
                self.ml_predictor = EnhancedMLPredictor()
                logger.info("Enhanced ML predictor initialized")
            except ImportError:
                # Fall back to production ML predictor
                self.ml_predictor = ProductionMLPredictor()
                logger.info("Production ML predictor initialized")
        except Exception as e:
            logger.warning(f"ML integration failed: {e}")
    
    def _initialize_rule_engine(self) -> None:
        """Initialize rule engine"""
        try:
            # Initialize production rule engine
            self.rule_engine = ProductionRuleEngine()
            logger.info("Production rule engine initialized")
        except Exception as e:
            logger.warning(f"Rule engine initialization failed: {e}")
    
    def _initialize_behavioral_analysis(self) -> None:
        """Initialize behavioral analysis"""
        try:
            # Initialize production behavioral analyzer
            self.behavioral_analyzer = ProductionBehavioralAnalyzer()
            logger.info("Production behavioral analysis initialized")
        except Exception as e:
            logger.warning(f"Behavioral analysis initialization failed: {e}")
    
    @monitor_performance
    @with_recovery
    def detect_fraud(self, transaction: Dict[str, Any], 
                    security_context: Optional[SecurityContext] = None) -> FraudDetectionResult:
        """Main fraud detection method"""
        
        if not self.initialized:
            raise DetectionError("Fraud detection core not initialized")
        
        start_time = time.time()
        correlation_id = str(uuid.uuid4())
        
        # Create error context
        error_context = create_error_context(
            correlation_id=correlation_id,
            component='EnhancedFraudDetectionCore',
            operation='detect_fraud',
            user_id=security_context.user_id if security_context else None
        )
        
        try:
            # Security validation
            if security_context:
                threats = self.security_manager.threat_detector.analyze_request(
                    security_context.ip_address,
                    security_context.user_id,
                    transaction
                )
                
                if threats:
                    logger.warning(f"Security threats detected: {len(threats)} threats")
                    for threat in threats:
                        severity = getattr(threat, 'severity', 'low')
                        if hasattr(severity, 'value'):
                            severity = severity.value
                        if severity in ['error', 'critical']:
                            description = getattr(threat, 'description', 'Unknown threat')
                            raise SecurityError(f"Security threat detected: {description}")
            
            # Validate transaction
            validation_result = self.validation_framework.validate_all(
                {'transaction': transaction},
                error_context
            )
            
            if not validation_result['is_valid']:
                raise ValidationError(
                    f"Transaction validation failed: {validation_result['errors']}",
                    context=error_context
                )
            
            # Preprocess transaction if preprocessing is enabled
            processed_transaction = transaction
            preprocessing_metadata = {}
            
            if self.config.enable_preprocessing and self.preprocessing_manager:
                try:
                    preprocessing_result = self.preprocessing_manager.preprocess_transaction(transaction)
                    processed_transaction = preprocessing_result['processed_data']
                    preprocessing_metadata = preprocessing_result.get('metadata', {})
                    
                    # Log preprocessing quality metrics
                    quality_score = preprocessing_metadata.get('quality_score', 1.0)
                    logger.debug(f"Transaction preprocessing completed - Quality Score: {quality_score:.3f}")
                    
                except Exception as e:
                    logger.warning(f"Preprocessing failed, using original transaction: {e}")
                    processed_transaction = transaction
            
            # Detect fraud based on strategy
            detection_result = self._execute_detection_strategy(
                processed_transaction, self.config.detection_strategy, error_context
            )
            
            # Add preprocessing metadata to result
            if preprocessing_metadata:
                detection_result.additional_metadata['preprocessing'] = preprocessing_metadata
            
            # Post-processing
            detection_result = self._post_process_result(detection_result, transaction)
            
            # Add validation results
            detection_result.validation_passed = validation_result['is_valid']
            detection_result.validation_errors = validation_result['errors']
            detection_result.validation_warnings = validation_result['warnings']
            
            # Record metrics
            detection_time = time.time() - start_time
            detection_result.detection_time = detection_time
            
            # Audit logging
            if self.config.enable_audit_logging:
                self.security_manager.audit_logger.log_event(
                    event_type='fraud_detection',
                    user_id=security_context.user_id if security_context else None,
                    resource='transaction',
                    action='detect_fraud',
                    result='success' if not detection_result.fraud_detected else 'fraud_detected',
                    ip_address=security_context.ip_address if security_context else None,
                    additional_data={
                        'transaction_id': transaction.get('transaction_id'),
                        'fraud_probability': detection_result.fraud_probability,
                        'detection_time': detection_time,
                        'correlation_id': correlation_id
                    }
                )
            
            return detection_result
            
        except Exception as e:
            # Log error
            logger.error(f"Fraud detection failed: {e}", exc_info=True)
            
            # Audit error
            if self.config.enable_audit_logging:
                self.security_manager.audit_logger.log_event(
                    event_type='fraud_detection_error',
                    user_id=security_context.user_id if security_context else None,
                    resource='transaction',
                    action='detect_fraud',
                    result=f'error: {str(e)}',
                    ip_address=security_context.ip_address if security_context else None,
                    additional_data={
                        'transaction_id': transaction.get('transaction_id'),
                        'error_type': type(e).__name__,
                        'correlation_id': correlation_id
                    }
                )
            
            # Use error recovery
            return self.error_recovery_manager.recover(
                self._create_fallback_result,
                e,
                error_context,
                transaction.get('transaction_id', 'unknown'),
                str(e)
            )
    
    def _execute_detection_strategy(self, transaction: Dict[str, Any],
                                   strategy: DetectionStrategy,
                                   context: ErrorContext) -> FraudDetectionResult:
        """Execute fraud detection based on strategy"""
        
        if strategy == DetectionStrategy.RULES_ONLY:
            return self._rules_only_detection(transaction, context)
        elif strategy == DetectionStrategy.ML_ONLY:
            return self._ml_only_detection(transaction, context)
        elif strategy == DetectionStrategy.HYBRID:
            return self._hybrid_detection(transaction, context)
        elif strategy == DetectionStrategy.ENSEMBLE:
            return self._ensemble_detection(transaction, context)
        else:
            raise DetectionError(f"Unsupported detection strategy: {strategy}")
    
    def _rules_only_detection(self, transaction: Dict[str, Any],
                             context: ErrorContext) -> FraudDetectionResult:
        """Rules-only fraud detection"""
        
        result = FraudDetectionResult(
            transaction_id=transaction.get('transaction_id', 'unknown'),
            fraud_detected=False,
            fraud_probability=0.0,
            risk_score=0.0,
            confidence=0.0,
            detection_strategy=DetectionStrategy.RULES_ONLY,
            detection_time=0.0,
            timestamp=datetime.now()
        )
        
        if self.rule_engine:
            # Execute rules with user history support
            rule_results = self.rule_engine.evaluate_transaction(transaction, user_history=None)
            
            # Process rule results
            result.violated_rules = rule_results.get('violated_rules', [])
            result.rule_scores = rule_results.get('rule_scores', {})
            
            # Calculate overall scores
            if result.violated_rules:
                result.fraud_detected = True
                result.fraud_probability = min(rule_results.get('total_score', 0.0), 1.0)
                result.risk_score = result.fraud_probability
                result.confidence = 0.8  # High confidence for rule-based detection
                
                result.explanation = f"Rules violated: {', '.join(result.violated_rules)}"
                result.reasoning_chain = [f"Rule '{rule}' violated with score {score}" 
                                        for rule, score in result.rule_scores.items()]
                
                # Add rule details to metadata
                result.additional_metadata['rule_details'] = rule_results.get('rule_details', {})
        
        return result
    
    def _ml_only_detection(self, transaction: Dict[str, Any],
                          context: ErrorContext) -> FraudDetectionResult:
        """ML-only fraud detection"""
        
        result = FraudDetectionResult(
            transaction_id=transaction.get('transaction_id', 'unknown'),
            fraud_detected=False,
            fraud_probability=0.0,
            risk_score=0.0,
            confidence=0.0,
            detection_strategy=DetectionStrategy.ML_ONLY,
            detection_time=0.0,
            timestamp=datetime.now()
        )
        
        if self.ml_predictor:
            # Make ML prediction
            prediction = self.ml_predictor.predict_fraud(transaction)
            
            # Process ML results
            result.ml_predictions = prediction.get('predictions', {})
            result.model_confidence = prediction.get('confidence', 0.0)
            result.feature_importance = prediction.get('feature_importance', {})
            
            # Calculate overall scores
            fraud_prob = prediction.get('fraud_probability', 0.0)
            result.fraud_probability = fraud_prob
            result.risk_score = fraud_prob
            result.confidence = result.model_confidence
            
            result.fraud_detected = fraud_prob > self.config.fraud_probability_threshold
            
            if result.fraud_detected:
                result.explanation = f"ML model predicted fraud with probability {fraud_prob:.3f}"
                result.reasoning_chain = [f"Feature '{feature}' importance: {importance:.3f}" 
                                        for feature, importance in result.feature_importance.items()]
        
        return result
    
    def _hybrid_detection(self, transaction: Dict[str, Any],
                         context: ErrorContext) -> FraudDetectionResult:
        """Hybrid fraud detection combining rules and ML"""
        
        # Get results from both approaches
        rule_result = self._rules_only_detection(transaction, context)
        ml_result = self._ml_only_detection(transaction, context)
        
        # Combine results
        combined_result = FraudDetectionResult(
            transaction_id=transaction.get('transaction_id', 'unknown'),
            fraud_detected=False,
            fraud_probability=0.0,
            risk_score=0.0,
            confidence=0.0,
            detection_strategy=DetectionStrategy.HYBRID,
            detection_time=0.0,
            timestamp=datetime.now()
        )
        
        # Combine rule and ML results
        combined_result.violated_rules = rule_result.violated_rules
        combined_result.rule_scores = rule_result.rule_scores
        combined_result.ml_predictions = ml_result.ml_predictions
        combined_result.model_confidence = ml_result.model_confidence
        combined_result.feature_importance = ml_result.feature_importance
        
        # Weighted combination (rules: 40%, ML: 60%)
        rule_weight = 0.4
        ml_weight = 0.6
        
        combined_fraud_prob = (rule_result.fraud_probability * rule_weight + 
                             ml_result.fraud_probability * ml_weight)
        
        combined_result.fraud_probability = combined_fraud_prob
        combined_result.risk_score = combined_fraud_prob
        combined_result.confidence = (rule_result.confidence * rule_weight + 
                                    ml_result.confidence * ml_weight)
        
        combined_result.fraud_detected = combined_fraud_prob > self.config.fraud_probability_threshold
        
        # Combine explanations
        explanations = []
        if rule_result.explanation:
            explanations.append(f"Rules: {rule_result.explanation}")
        if ml_result.explanation:
            explanations.append(f"ML: {ml_result.explanation}")
        
        combined_result.explanation = "; ".join(explanations)
        combined_result.reasoning_chain = rule_result.reasoning_chain + ml_result.reasoning_chain
        
        return combined_result
    
    def _ensemble_detection(self, transaction: Dict[str, Any],
                           context: ErrorContext) -> FraudDetectionResult:
        """Ensemble fraud detection with multiple models"""
        
        # For now, use hybrid as base for ensemble
        base_result = self._hybrid_detection(transaction, context)
        
        # Add behavioral analysis if available
        if self.behavioral_analyzer:
            behavioral_result = self.behavioral_analyzer.analyze_transaction(transaction, user_history=None)
            
            base_result.behavioral_anomalies = behavioral_result.get('anomalies', [])
            base_result.behavioral_score = behavioral_result.get('anomaly_score', 0.0)
            
            # Adjust fraud probability based on behavioral analysis
            if base_result.behavioral_anomalies:
                base_result.fraud_probability = min(
                    base_result.fraud_probability + (base_result.behavioral_score * 0.2),
                    1.0
                )
                
                base_result.explanation += f"; Behavioral anomalies: {', '.join(base_result.behavioral_anomalies)}"
                
            # Add behavioral details to metadata
            base_result.additional_metadata['behavioral_details'] = {
                'user_profile_size': behavioral_result.get('user_profile_size', 0),
                'profile_age_days': behavioral_result.get('profile_age_days', 0)
            }
        
        base_result.detection_strategy = DetectionStrategy.ENSEMBLE
        return base_result
    
    def _post_process_result(self, result: FraudDetectionResult,
                            transaction: Dict[str, Any]) -> FraudDetectionResult:
        """Post-process detection result"""
        
        # Add compliance checks
        if self.config.enable_compliance_checks:
            result = self._add_compliance_checks(result, transaction)
        
        # Add additional metadata (preserve existing metadata)
        additional_data = {
            'transaction_amount': transaction.get('amount'),
            'transaction_type': transaction.get('type'),
            'merchant_id': transaction.get('merchant_id'),
            'user_id': transaction.get('user_id'),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        # Merge with existing metadata
        result.additional_metadata.update(additional_data)
        
        return result
    
    def _add_compliance_checks(self, result: FraudDetectionResult,
                              transaction: Dict[str, Any]) -> FraudDetectionResult:
        """Add compliance-related checks"""
        
        # Check for high-value transactions
        amount = transaction.get('amount', 0)
        if amount > 10000:  # Example threshold
            result.reasoning_chain.append(f"High-value transaction: ${amount}")
            result.risk_score = min(result.risk_score + 0.1, 1.0)
        
        # Check for unusual times
        transaction_time = transaction.get('timestamp')
        if transaction_time:
            try:
                tx_datetime = datetime.fromisoformat(transaction_time.replace('Z', '+00:00'))
                if tx_datetime.hour < 6 or tx_datetime.hour > 22:  # Outside business hours
                    result.reasoning_chain.append("Transaction outside business hours")
                    result.risk_score = min(result.risk_score + 0.05, 1.0)
            except Exception:
                pass
        
        return result
    
    def _create_fallback_result(self, transaction_id: str, error_message: str) -> FraudDetectionResult:
        """Create fallback result when detection fails"""
        
        return FraudDetectionResult(
            transaction_id=transaction_id,
            fraud_detected=False,
            fraud_probability=0.0,
            risk_score=0.0,
            confidence=0.0,
            detection_strategy=DetectionStrategy.HYBRID,
            detection_time=0.0,
            timestamp=datetime.now(),
            explanation=f"Fallback result due to error: {error_message}",
            validation_passed=False,
            validation_errors=[f"Detection failed: {error_message}"]
        )
    
    def batch_detect_fraud(self, transactions: List[Dict[str, Any]],
                          security_context: Optional[SecurityContext] = None) -> List[FraudDetectionResult]:
        """Batch fraud detection for multiple transactions"""
        
        if not transactions:
            return []
        
        results = []
        
        if self.config.enable_async_processing:
            # Process in parallel
            futures = []
            for transaction in transactions:
                future = self.thread_pool.submit(self.detect_fraud, transaction, security_context)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=self.config.detection_timeout)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch detection failed for transaction: {e}")
                    results.append(self._create_fallback_result('unknown', str(e)))
        else:
            # Process sequentially
            for transaction in transactions:
                try:
                    result = self.detect_fraud(transaction, security_context)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch detection failed for transaction: {e}")
                    results.append(self._create_fallback_result(
                        transaction.get('transaction_id', 'unknown'), str(e)
                    ))
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            'core_status': {
                'initialized': self.initialized,
                'detection_strategy': self.config.detection_strategy,
                'async_processing': self.config.enable_async_processing
            },
            'validation_status': {
                'performance_metrics': self.validation_framework.get_performance_metrics()
            },
            'monitoring_status': self.monitoring_manager.get_system_status(),
            'security_status': self.security_manager.get_security_dashboard(),
            'recovery_status': self.error_recovery_manager.get_recovery_metrics(),
            'component_health': {
                'rule_engine': self.rule_engine is not None,
                'ml_predictor': self.ml_predictor is not None,
                'behavioral_analyzer': self.behavioral_analyzer is not None,
                'preprocessing_manager': self.preprocessing_manager is not None
            },
            'preprocessing_status': {
                'enabled': self.config.enable_preprocessing,
                'manager_ready': self.preprocessing_manager is not None,
                'feature_count': getattr(self.preprocessing_manager, 'feature_count', 0) if self.preprocessing_manager else 0
            }
        }
    
    def shutdown(self) -> None:
        """Shutdown fraud detection core"""
        
        logger.info("Shutting down enhanced fraud detection core...")
        
        # Shutdown components
        self.validation_framework.shutdown()
        self.monitoring_manager.shutdown()
        self.thread_pool.shutdown(wait=True)
        
        self.initialized = False
        logger.info("Enhanced fraud detection core shutdown complete")

# ======================== COMPLETE PREPROCESSING MANAGER ========================

class CompletePreprocessingManager:
    """Complete preprocessing system for financial fraud detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_engineering_config = config.get('feature_engineering', {})
        self.data_quality_config = config.get('data_quality', {})
        self.feature_selection_config = config.get('feature_selection', {})
        self.scaling_config = config.get('scaling', {})
        
        # Initialize scalers
        self.scalers = {}
        self._initialize_scalers()
        
        # Feature selection
        self.feature_selector = None
        self.selected_features = []
        
        # Quality monitoring
        self.quality_metrics = defaultdict(list)
        self.feature_count = 0
        
        logger.info("Complete preprocessing manager initialized successfully")
    
    def _initialize_scalers(self):
        """Initialize scaling components"""
        scaling_method = self.scaling_config.get('method', 'standard')
        
        if scaling_method == 'standard':
            self.scalers['standard'] = StandardScaler()
        elif scaling_method == 'minmax':
            feature_range = self.scaling_config.get('feature_range', [0, 1])
            self.scalers['minmax'] = MinMaxScaler(feature_range=feature_range)
        elif scaling_method == 'robust':
            self.scalers['robust'] = RobustScaler()
        else:
            self.scalers['standard'] = StandardScaler()
    
    def preprocess_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Complete preprocessing pipeline for a single transaction"""
        
        try:
            # Step 1: Data Quality Assessment
            quality_result = self._assess_data_quality(transaction)
            
            # Step 2: Feature Engineering
            engineered_features = self._engineer_features(transaction)
            
            # Step 3: Data Quality Enhancement
            enhanced_features = self._enhance_data_quality(engineered_features)
            
            # Step 4: Feature Selection (if configured)
            selected_features = self._apply_feature_selection(enhanced_features)
            
            # Step 5: Feature Scaling
            scaled_features = self._scale_features(selected_features)
            
            # Step 6: Final Quality Check
            final_quality = self._final_quality_check(scaled_features)
            
            # Combine with original transaction data
            processed_transaction = {**transaction, **scaled_features}
            
            metadata = {
                'quality_score': quality_result['overall_score'],
                'feature_count': len(scaled_features),
                'quality_issues': quality_result.get('issues', []),
                'processing_timestamp': datetime.now().isoformat(),
                'final_quality': final_quality
            }
            
            return {
                'processed_data': processed_transaction,
                'metadata': metadata,
                'quality_assessment': quality_result
            }
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return {
                'processed_data': transaction,
                'metadata': {'error': str(e), 'quality_score': 0.0},
                'quality_assessment': {'overall_score': 0.0, 'issues': [str(e)]}
            }
    
    def _assess_data_quality(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data quality of transaction"""
        
        issues = []
        scores = []
        
        # Check for missing required fields
        required_fields = ['transaction_id', 'amount', 'user_id', 'timestamp']
        missing_fields = [field for field in required_fields if not transaction.get(field)]
        
        if missing_fields:
            issues.append(f"Missing required fields: {missing_fields}")
            scores.append(0.5)
        else:
            scores.append(1.0)
        
        # Check amount validity
        amount = transaction.get('amount', 0)
        if amount <= 0:
            issues.append("Invalid amount: must be positive")
            scores.append(0.3)
        elif amount > 1000000:  # Extreme amount
            issues.append("Extreme amount detected")
            scores.append(0.7)
        else:
            scores.append(1.0)
        
        # Check timestamp format
        timestamp = transaction.get('timestamp', '')
        if timestamp:
            try:
                datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                scores.append(1.0)
            except:
                issues.append("Invalid timestamp format")
                scores.append(0.6)
        else:
            scores.append(0.8)
        
        # Check for data completeness
        total_fields = len(transaction)
        non_empty_fields = sum(1 for v in transaction.values() if v is not None and v != '')
        completeness = non_empty_fields / total_fields if total_fields > 0 else 0
        scores.append(completeness)
        
        overall_score = np.mean(scores) if scores else 0.0
        
        return {
            'overall_score': overall_score,
            'completeness': completeness,
            'issues': issues,
            'field_count': total_fields,
            'non_empty_fields': non_empty_fields
        }
    
    def _engineer_features(self, transaction: Dict[str, Any]) -> Dict[str, float]:
        """Engineer features from transaction data"""
        
        features = {}
        config = self.feature_engineering_config
        
        # Amount features
        if config.get('enable_amount_features', True):
            features.update(self._create_amount_features(transaction))
        
        # Time features
        if config.get('enable_time_features', True):
            features.update(self._create_time_features(transaction))
        
        # Merchant features
        if config.get('enable_merchant_features', True):
            features.update(self._create_merchant_features(transaction))
        
        # Geographic features
        if config.get('enable_geographic_features', True):
            features.update(self._create_geographic_features(transaction))
        
        # Frequency features (simplified for single transaction)
        if config.get('enable_frequency_features', True):
            features.update(self._create_frequency_features(transaction))
        
        # Velocity features (simplified for single transaction)
        if config.get('enable_velocity_features', True):
            features.update(self._create_velocity_features(transaction))
        
        self.feature_count = len(features)
        return features
    
    def _create_amount_features(self, transaction: Dict[str, Any]) -> Dict[str, float]:
        """Create amount-based features"""
        features = {}
        amount = float(transaction.get('amount', 0))
        
        features['amount'] = amount
        features['amount_log'] = np.log1p(amount)
        features['amount_sqrt'] = np.sqrt(amount)
        features['amount_squared'] = amount ** 2
        features['is_round_amount'] = 1.0 if amount > 0 and amount % 100 == 0 else 0.0
        features['is_small_amount'] = 1.0 if amount < 10 else 0.0
        features['is_medium_amount'] = 1.0 if 10 <= amount <= 1000 else 0.0
        features['is_large_amount'] = 1.0 if 1000 < amount <= 10000 else 0.0
        features['is_very_large_amount'] = 1.0 if amount > 10000 else 0.0
        
        # Amount bins
        if amount == 0:
            features['amount_bin'] = 0
        elif amount <= 10:
            features['amount_bin'] = 1
        elif amount <= 100:
            features['amount_bin'] = 2
        elif amount <= 1000:
            features['amount_bin'] = 3
        elif amount <= 10000:
            features['amount_bin'] = 4
        else:
            features['amount_bin'] = 5
        
        return features
    
    def _create_time_features(self, transaction: Dict[str, Any]) -> Dict[str, float]:
        """Create time-based features"""
        features = {}
        timestamp = transaction.get('timestamp', '')
        
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                # Basic time features
                features['hour'] = float(dt.hour)
                features['day_of_week'] = float(dt.weekday())
                features['day_of_month'] = float(dt.day)
                features['month'] = float(dt.month)
                features['year'] = float(dt.year)
                
                # Cyclical time features
                features['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
                features['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
                features['day_sin'] = np.sin(2 * np.pi * dt.weekday() / 7)
                features['day_cos'] = np.cos(2 * np.pi * dt.weekday() / 7)
                features['month_sin'] = np.sin(2 * np.pi * dt.month / 12)
                features['month_cos'] = np.cos(2 * np.pi * dt.month / 12)
                
                # Time categories
                features['is_weekend'] = 1.0 if dt.weekday() >= 5 else 0.0
                features['is_night'] = 1.0 if dt.hour < 6 or dt.hour > 22 else 0.0
                features['is_business_hours'] = 1.0 if 9 <= dt.hour <= 17 else 0.0
                features['is_early_morning'] = 1.0 if 6 <= dt.hour < 9 else 0.0
                features['is_evening'] = 1.0 if 18 <= dt.hour <= 22 else 0.0
                features['is_lunch_time'] = 1.0 if 11 <= dt.hour <= 14 else 0.0
                
                # Quarter and season
                quarter = (dt.month - 1) // 3 + 1
                features['quarter'] = float(quarter)
                features['is_month_end'] = 1.0 if dt.day >= 28 else 0.0
                features['is_month_start'] = 1.0 if dt.day <= 3 else 0.0
                
            except Exception:
                # Default values if timestamp parsing fails
                features.update({
                    'hour': 12.0, 'day_of_week': 1.0, 'day_of_month': 15.0,
                    'month': 6.0, 'year': 2024.0, 'hour_sin': 0.0, 'hour_cos': 1.0,
                    'day_sin': 0.0, 'day_cos': 1.0, 'month_sin': 0.0, 'month_cos': 1.0,
                    'is_weekend': 0.0, 'is_night': 0.0, 'is_business_hours': 1.0,
                    'is_early_morning': 0.0, 'is_evening': 0.0, 'is_lunch_time': 0.0,
                    'quarter': 2.0, 'is_month_end': 0.0, 'is_month_start': 0.0
                })
        
        return features
    
    def _create_merchant_features(self, transaction: Dict[str, Any]) -> Dict[str, float]:
        """Create merchant-based features"""
        features = {}
        merchant_id = str(transaction.get('merchant_id', ''))
        
        features['merchant_id_length'] = float(len(merchant_id))
        features['has_merchant_id'] = 1.0 if merchant_id else 0.0
        features['merchant_id_hash'] = float(hash(merchant_id) % 10000) if merchant_id else 0.0
        
        # Merchant category (if available)
        merchant_category = str(transaction.get('merchant_category', ''))
        features['has_merchant_category'] = 1.0 if merchant_category else 0.0
        
        # Merchant risk indicators (based on name patterns)
        if merchant_id:
            merchant_lower = merchant_id.lower()
            features['merchant_has_numbers'] = 1.0 if any(c.isdigit() for c in merchant_id) else 0.0
            features['merchant_has_special_chars'] = 1.0 if any(not c.isalnum() for c in merchant_id) else 0.0
            features['merchant_length_suspicious'] = 1.0 if len(merchant_id) < 3 or len(merchant_id) > 50 else 0.0
        
        return features
    
    def _create_geographic_features(self, transaction: Dict[str, Any]) -> Dict[str, float]:
        """Create geographic features"""
        features = {}
        
        # Country features
        country = str(transaction.get('country', '')).upper()
        features['has_country'] = 1.0 if country else 0.0
        features['is_domestic'] = 1.0 if country in ['US', 'USA'] else 0.0
        features['is_high_risk_country'] = 1.0 if country in ['XX', 'YY', 'ZZ'] else 0.0  # Example
        
        # User country vs transaction country
        user_country = str(transaction.get('user_country', 'US')).upper()
        features['is_cross_border'] = 1.0 if country and user_country and country != user_country else 0.0
        
        # City/region features
        city = str(transaction.get('city', ''))
        features['has_city'] = 1.0 if city else 0.0
        
        # IP-based features (if available)
        ip_country = str(transaction.get('ip_country', '')).upper()
        features['has_ip_country'] = 1.0 if ip_country else 0.0
        features['ip_country_mismatch'] = 1.0 if (ip_country and country and 
                                                ip_country != country) else 0.0
        
        return features
    
    def _create_frequency_features(self, transaction: Dict[str, Any]) -> Dict[str, float]:
        """Create frequency-based features (simplified for single transaction)"""
        features = {}
        
        # These would typically require historical data
        # For now, we'll create placeholder features
        features['user_transaction_count_estimate'] = 1.0  # Would be calculated from history
        features['merchant_transaction_count_estimate'] = 1.0
        features['daily_transaction_estimate'] = 1.0
        features['weekly_transaction_estimate'] = 7.0
        
        return features
    
    def _create_velocity_features(self, transaction: Dict[str, Any]) -> Dict[str, float]:
        """Create velocity-based features (simplified for single transaction)"""
        features = {}
        
        # These would typically require historical data
        # For now, we'll create placeholder features
        features['time_since_last_transaction'] = 3600.0  # Placeholder: 1 hour
        features['amount_velocity_1h'] = 0.0  # Would be calculated from recent history
        features['transaction_velocity_1h'] = 0.0
        features['amount_velocity_24h'] = 0.0
        features['transaction_velocity_24h'] = 0.0
        
        return features
    
    def _enhance_data_quality(self, features: Dict[str, float]) -> Dict[str, float]:
        """Enhance data quality through imputation and outlier handling"""
        
        enhanced_features = features.copy()
        
        # Handle missing values
        for key, value in enhanced_features.items():
            if value is None or np.isnan(value) or np.isinf(value):
                enhanced_features[key] = 0.0  # Simple imputation
        
        # Outlier detection and handling
        outlier_method = self.data_quality_config.get('outlier_method', 'iqr')
        outlier_threshold = self.data_quality_config.get('outlier_threshold', 3.0)
        
        if outlier_method == 'iqr':
            enhanced_features = self._handle_outliers_iqr(enhanced_features)
        elif outlier_method == 'zscore':
            enhanced_features = self._handle_outliers_zscore(enhanced_features, outlier_threshold)
        
        return enhanced_features
    
    def _handle_outliers_iqr(self, features: Dict[str, float]) -> Dict[str, float]:
        """Handle outliers using IQR method"""
        # For single transaction, we can't calculate IQR
        # In production, this would use historical statistics
        return features
    
    def _handle_outliers_zscore(self, features: Dict[str, float], threshold: float) -> Dict[str, float]:
        """Handle outliers using Z-score method"""
        # For single transaction, we can't calculate Z-score
        # In production, this would use historical statistics
        return features
    
    def _apply_feature_selection(self, features: Dict[str, float]) -> Dict[str, float]:
        """Apply feature selection if configured"""
        
        # For single transaction preprocessing, we'll return all features
        # In production, this would use pre-trained feature selectors
        
        correlation_threshold = self.feature_selection_config.get('correlation_threshold', 0.9)
        
        # Simple correlation-based feature removal (placeholder)
        # In practice, this would use historical correlation analysis
        selected_features = features.copy()
        
        return selected_features
    
    def _scale_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Scale features using configured scaling method"""
        
        # For single transaction, we can't fit scalers
        # In production, this would use pre-fitted scalers
        
        scaling_method = self.scaling_config.get('method', 'standard')
        scaled_features = features.copy()
        
        # Apply simple normalization for demonstration
        if scaling_method == 'minmax':
            feature_range = self.scaling_config.get('feature_range', [0, 1])
            min_val, max_val = feature_range
            
            # Simple min-max scaling for demo (would use fitted scaler in production)
            for key, value in scaled_features.items():
                if key.startswith('amount') and not key.endswith('_bin'):
                    # Scale amount features to [0, 1] range
                    if value > 0:
                        scaled_features[key] = min(value / 100000, 1.0)  # Arbitrary max for demo
        
        return scaled_features
    
    def _final_quality_check(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Perform final quality check on processed features"""
        
        quality_issues = []
        
        # Check for infinite or NaN values
        invalid_features = [k for k, v in features.items() 
                          if v is None or np.isnan(v) or np.isinf(v)]
        
        if invalid_features:
            quality_issues.append(f"Invalid values in features: {invalid_features}")
        
        # Check feature ranges
        extreme_features = [k for k, v in features.items() 
                          if abs(v) > 1000000]  # Arbitrary threshold
        
        if extreme_features:
            quality_issues.append(f"Extreme values in features: {extreme_features}")
        
        quality_score = 1.0 - (len(quality_issues) * 0.2)
        quality_score = max(0.0, quality_score)
        
        return {
            'quality_score': quality_score,
            'issues': quality_issues,
            'feature_count': len(features),
            'valid_features': len(features) - len(invalid_features)
        }

# ======================== PRODUCTION COMPONENTS ========================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from collections import defaultdict, deque
import hashlib
import pickle
import os
import re
from datetime import timedelta

class ProductionMLPredictor:
    """Production-ready ML predictor with ensemble models"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_fitted = False
        self.prediction_cache = {}
        self.model_weights = {
            'random_forest': 0.4,
            'gradient_boosting': 0.3,
            'logistic_regression': 0.3
        }
        
        # Initialize models
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        self.models['logistic_regression'] = LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        )
        
        # Try to load pre-trained models
        self._load_pretrained_models()
    
    def _load_pretrained_models(self):
        """Load pre-trained models if available"""
        try:
            # Look for model files in current directory
            model_files = {
                'random_forest': 'fraud_rf_model.pkl',
                'gradient_boosting': 'fraud_gb_model.pkl',
                'logistic_regression': 'fraud_lr_model.pkl',
                'scaler': 'fraud_scaler.pkl'
            }
            
            models_loaded = 0
            for model_name, filename in model_files.items():
                if os.path.exists(filename):
                    try:
                        with open(filename, 'rb') as f:
                            if model_name == 'scaler':
                                self.scaler = pickle.load(f)
                            else:
                                self.models[model_name] = pickle.load(f)
                        models_loaded += 1
                    except Exception as e:
                        logger.warning(f"Failed to load {model_name}: {e}")
            
            if models_loaded > 0:
                self.is_fitted = True
                logger.info(f"Loaded {models_loaded} pre-trained models")
                
        except Exception as e:
            logger.warning(f"Model loading failed: {e}")
    
    def predict_fraud(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Predict fraud probability for transaction"""
        
        try:
            # Extract features
            features = self._extract_features(transaction)
            
            # Check cache
            cache_key = self._get_cache_key(features)
            if cache_key in self.prediction_cache:
                return self.prediction_cache[cache_key]
            
            if not self.is_fitted:
                # Use heuristic prediction if models not trained
                return self._heuristic_prediction(transaction, features)
            
            # Scale features
            feature_df = pd.DataFrame([features])
            feature_df = feature_df.reindex(columns=self.feature_columns, fill_value=0)
            features_scaled = self.scaler.transform(feature_df)
            
            # Get predictions from all models
            predictions = {}
            for model_name, model in self.models.items():
                try:
                    pred_proba = model.predict_proba(features_scaled)[0, 1]
                    predictions[model_name] = pred_proba
                except Exception as e:
                    logger.warning(f"Model {model_name} prediction failed: {e}")
                    predictions[model_name] = 0.5
            
            # Ensemble prediction
            fraud_probability = sum(
                predictions[model] * self.model_weights.get(model, 0.33)
                for model in predictions
            )
            
            # Get feature importance
            feature_importance = self._get_feature_importance()
            
            result = {
                'fraud_probability': fraud_probability,
                'predictions': predictions,
                'confidence': self._calculate_confidence(list(predictions.values())),
                'feature_importance': feature_importance
            }
            
            # Cache result
            self.prediction_cache[cache_key] = result
            self._cleanup_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return self._fallback_prediction(transaction)
    
    def _extract_features(self, transaction: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from transaction"""
        features = {}
        
        # Amount features
        amount = float(transaction.get('amount', 0))
        features['amount'] = amount
        features['amount_log'] = np.log1p(amount)
        features['amount_sqrt'] = np.sqrt(amount)
        features['is_round_amount'] = 1.0 if amount > 0 and amount % 100 == 0 else 0.0
        features['is_high_amount'] = 1.0 if amount > 1000 else 0.0
        features['is_very_high_amount'] = 1.0 if amount > 5000 else 0.0
        
        # Time features
        timestamp = transaction.get('timestamp', '')
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                features['hour'] = dt.hour
                features['day_of_week'] = dt.weekday()
                features['is_weekend'] = 1.0 if dt.weekday() >= 5 else 0.0
                features['is_night'] = 1.0 if dt.hour < 6 or dt.hour > 22 else 0.0
                features['is_business_hours'] = 1.0 if 9 <= dt.hour <= 17 else 0.0
            except Exception:
                features.update({
                    'hour': 12, 'day_of_week': 1, 'is_weekend': 0.0,
                    'is_night': 0.0, 'is_business_hours': 1.0
                })
        
        # Merchant features
        merchant_id = str(transaction.get('merchant_id', ''))
        features['merchant_id_length'] = len(merchant_id)
        features['has_merchant_id'] = 1.0 if merchant_id else 0.0
        
        # User features
        user_id = str(transaction.get('user_id', ''))
        features['user_id_length'] = len(user_id)
        features['has_user_id'] = 1.0 if user_id else 0.0
        
        # Transaction type features
        tx_type = str(transaction.get('type', 'unknown')).lower()
        features['is_purchase'] = 1.0 if 'purchase' in tx_type else 0.0
        features['is_transfer'] = 1.0 if 'transfer' in tx_type else 0.0
        features['is_withdrawal'] = 1.0 if 'withdrawal' in tx_type else 0.0
        
        # Location features
        country = str(transaction.get('country', '')).upper()
        features['is_domestic'] = 1.0 if country in ['US', 'USA'] else 0.0
        features['has_country'] = 1.0 if country else 0.0
        
        return features
    
    def _heuristic_prediction(self, transaction: Dict[str, Any], features: Dict[str, float]) -> Dict[str, Any]:
        """Heuristic prediction when models are not available"""
        
        score = 0.0
        
        # High amount risk
        if features['amount'] > 5000:
            score += 0.3
        if features['amount'] > 10000:
            score += 0.4
        
        # Time-based risk
        if features['is_night']:
            score += 0.2
        if features['is_weekend']:
            score += 0.1
        
        # Round amount risk
        if features['is_round_amount'] and features['amount'] > 1000:
            score += 0.2
        
        # Missing information risk
        if not features['has_merchant_id']:
            score += 0.1
        if not features['has_user_id']:
            score += 0.3
        
        fraud_probability = min(score, 1.0)
        
        return {
            'fraud_probability': fraud_probability,
            'predictions': {'heuristic': fraud_probability},
            'confidence': 0.5,
            'feature_importance': {
                'amount': 0.4,
                'is_night': 0.2,
                'is_round_amount': 0.2,
                'has_user_id': 0.2
            }
        }
    
    def _fallback_prediction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback prediction when all else fails"""
        return {
            'fraud_probability': 0.5,
            'predictions': {'fallback': 0.5},
            'confidence': 0.0,
            'feature_importance': {}
        }
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance"""
        importance = defaultdict(float)
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_') and self.feature_columns:
                for i, feature in enumerate(self.feature_columns):
                    if i < len(model.feature_importances_):
                        importance[feature] += model.feature_importances_[i] * self.model_weights.get(model_name, 0.33)
        
        # Return top 10 features
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:10])
    
    def _calculate_confidence(self, predictions: List[float]) -> float:
        """Calculate prediction confidence"""
        if not predictions:
            return 0.0
        
        # Confidence based on agreement between models
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # High agreement = high confidence
        confidence = max(0.0, 1.0 - (std_pred * 2))
        return confidence
    
    def _get_cache_key(self, features: Dict[str, float]) -> str:
        """Generate cache key for features"""
        feature_str = json.dumps(features, sort_keys=True)
        return hashlib.md5(feature_str.encode()).hexdigest()
    
    def _cleanup_cache(self):
        """Clean up prediction cache"""
        if len(self.prediction_cache) > 1000:
            # Remove oldest 20% of entries
            keys_to_remove = list(self.prediction_cache.keys())[:200]
            for key in keys_to_remove:
                del self.prediction_cache[key]

class ProductionRuleEngine:
    """Production-ready rule engine with comprehensive fraud rules"""
    
    def __init__(self):
        self.rules = {}
        self.rule_stats = defaultdict(lambda: {'triggered': 0, 'total': 0})
        self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize comprehensive fraud detection rules"""
        
        # Amount-based rules
        self.rules['high_amount'] = {
            'threshold': 5000,
            'score': 0.3,
            'description': 'Transaction amount exceeds high threshold'
        }
        
        self.rules['very_high_amount'] = {
            'threshold': 10000,
            'score': 0.6,
            'description': 'Transaction amount exceeds very high threshold'
        }
        
        self.rules['suspicious_round_amount'] = {
            'min_amount': 1000,
            'score': 0.2,
            'description': 'Suspicious round amount transaction'
        }
        
        # Time-based rules
        self.rules['unusual_time'] = {
            'start_hour': 2,
            'end_hour': 5,
            'score': 0.3,
            'description': 'Transaction during unusual hours'
        }
        
        self.rules['weekend_high_value'] = {
            'amount_threshold': 2000,
            'score': 0.2,
            'description': 'High value transaction on weekend'
        }
        
        # Merchant rules
        self.rules['suspicious_merchant'] = {
            'blacklist': [
                'suspicious_merchant_1', 'suspicious_merchant_2',
                'fraud_merchant', 'scam_shop', 'fake_store'
            ],
            'score': 0.8,
            'description': 'Transaction with blacklisted merchant'
        }
        
        self.rules['new_merchant_high_value'] = {
            'amount_threshold': 1000,
            'score': 0.4,
            'description': 'High value transaction with new merchant'
        }
        
        # Velocity rules
        self.rules['rapid_transactions'] = {
            'time_window': 300,  # 5 minutes
            'max_count': 5,
            'score': 0.7,
            'description': 'Too many transactions in short time'
        }
        
        # Pattern rules
        self.rules['card_testing_pattern'] = {
            'small_amount_threshold': 10,
            'large_amount_threshold': 1000,
            'score': 0.9,
            'description': 'Card testing pattern detected'
        }
        
        # Geographic rules
        self.rules['high_risk_country'] = {
            'risk_countries': ['XX', 'YY', 'ZZ'],  # Example country codes
            'score': 0.5,
            'description': 'Transaction from high-risk country'
        }
        
        self.rules['cross_border_high_value'] = {
            'amount_threshold': 3000,
            'score': 0.3,
            'description': 'High value cross-border transaction'
        }
    
    def evaluate_transaction(self, transaction: Dict[str, Any], 
                           user_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Evaluate transaction against all fraud rules"""
        
        violated_rules = []
        rule_scores = {}
        rule_details = {}
        
        # Amount rules
        amount = transaction.get('amount', 0)
        
        if amount > self.rules['high_amount']['threshold']:
            rule_name = 'high_amount'
            violated_rules.append(rule_name)
            rule_scores[rule_name] = self.rules[rule_name]['score']
            rule_details[rule_name] = {
                'amount': amount,
                'threshold': self.rules[rule_name]['threshold']
            }
            self.rule_stats[rule_name]['triggered'] += 1
        
        if amount > self.rules['very_high_amount']['threshold']:
            rule_name = 'very_high_amount'
            violated_rules.append(rule_name)
            rule_scores[rule_name] = self.rules[rule_name]['score']
            rule_details[rule_name] = {
                'amount': amount,
                'threshold': self.rules[rule_name]['threshold']
            }
            self.rule_stats[rule_name]['triggered'] += 1
        
        # Round amount rule
        if (amount >= self.rules['suspicious_round_amount']['min_amount'] and 
            amount % 100 == 0):
            rule_name = 'suspicious_round_amount'
            violated_rules.append(rule_name)
            rule_scores[rule_name] = self.rules[rule_name]['score']
            rule_details[rule_name] = {'amount': amount}
            self.rule_stats[rule_name]['triggered'] += 1
        
        # Time-based rules
        timestamp = transaction.get('timestamp', '')
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                # Unusual time rule
                if (self.rules['unusual_time']['start_hour'] <= dt.hour <= 
                    self.rules['unusual_time']['end_hour']):
                    rule_name = 'unusual_time'
                    violated_rules.append(rule_name)
                    rule_scores[rule_name] = self.rules[rule_name]['score']
                    rule_details[rule_name] = {'hour': dt.hour}
                    self.rule_stats[rule_name]['triggered'] += 1
                
                # Weekend high value rule
                if (dt.weekday() >= 5 and 
                    amount > self.rules['weekend_high_value']['amount_threshold']):
                    rule_name = 'weekend_high_value'
                    violated_rules.append(rule_name)
                    rule_scores[rule_name] = self.rules[rule_name]['score']
                    rule_details[rule_name] = {
                        'amount': amount,
                        'day': dt.strftime('%A')
                    }
                    self.rule_stats[rule_name]['triggered'] += 1
                    
            except Exception:
                pass
        
        # Merchant rules
        merchant_id = str(transaction.get('merchant_id', ''))
        
        if merchant_id in self.rules['suspicious_merchant']['blacklist']:
            rule_name = 'suspicious_merchant'
            violated_rules.append(rule_name)
            rule_scores[rule_name] = self.rules[rule_name]['score']
            rule_details[rule_name] = {'merchant_id': merchant_id}
            self.rule_stats[rule_name]['triggered'] += 1
        
        # Geographic rules
        country = str(transaction.get('country', '')).upper()
        user_country = str(transaction.get('user_country', 'US')).upper()
        
        if country in self.rules['high_risk_country']['risk_countries']:
            rule_name = 'high_risk_country'
            violated_rules.append(rule_name)
            rule_scores[rule_name] = self.rules[rule_name]['score']
            rule_details[rule_name] = {'country': country}
            self.rule_stats[rule_name]['triggered'] += 1
        
        if (country != user_country and 
            amount > self.rules['cross_border_high_value']['amount_threshold']):
            rule_name = 'cross_border_high_value'
            violated_rules.append(rule_name)
            rule_scores[rule_name] = self.rules[rule_name]['score']
            rule_details[rule_name] = {
                'amount': amount,
                'country': country,
                'user_country': user_country
            }
            self.rule_stats[rule_name]['triggered'] += 1
        
        # Update total counts for all rules
        for rule_name in self.rules:
            self.rule_stats[rule_name]['total'] += 1
        
        return {
            'violated_rules': violated_rules,
            'rule_scores': rule_scores,
            'rule_details': rule_details,
            'total_score': sum(rule_scores.values())
        }
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get rule performance statistics"""
        stats = {}
        for rule_name, counts in self.rule_stats.items():
            if counts['total'] > 0:
                stats[rule_name] = {
                    'triggered': counts['triggered'],
                    'total': counts['total'],
                    'trigger_rate': counts['triggered'] / counts['total']
                }
        return stats

class ProductionBehavioralAnalyzer:
    """Production-ready behavioral analyzer with user profiling"""
    
    def __init__(self):
        self.user_profiles = {}
        self.transaction_history = defaultdict(lambda: deque(maxlen=100))
        self.anomaly_threshold = 2.0  # Standard deviations
        self.profile_lock = threading.Lock()
    
    def analyze_transaction(self, transaction: Dict[str, Any], 
                          user_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Analyze transaction for behavioral anomalies"""
        
        user_id = transaction.get('user_id', 'unknown')
        amount = transaction.get('amount', 0)
        timestamp = transaction.get('timestamp', '')
        
        anomalies = []
        anomaly_score = 0.0
        
        with self.profile_lock:
            # Get or create user profile
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = self._create_user_profile(user_id, user_history)
            
            profile = self.user_profiles[user_id]
            
            # Amount anomaly detection
            amount_anomaly = self._detect_amount_anomaly(amount, profile)
            if amount_anomaly['is_anomaly']:
                anomalies.append(f"Unusual amount: {amount_anomaly['reason']}")
                anomaly_score += amount_anomaly['score']
            
            # Time pattern anomaly
            time_anomaly = self._detect_time_anomaly(timestamp, profile)
            if time_anomaly['is_anomaly']:
                anomalies.append(f"Unusual time: {time_anomaly['reason']}")
                anomaly_score += time_anomaly['score']
            
            # Frequency anomaly
            frequency_anomaly = self._detect_frequency_anomaly(user_id, timestamp, profile)
            if frequency_anomaly['is_anomaly']:
                anomalies.append(f"Unusual frequency: {frequency_anomaly['reason']}")
                anomaly_score += frequency_anomaly['score']
            
            # Merchant anomaly
            merchant_anomaly = self._detect_merchant_anomaly(transaction, profile)
            if merchant_anomaly['is_anomaly']:
                anomalies.append(f"Unusual merchant: {merchant_anomaly['reason']}")
                anomaly_score += merchant_anomaly['score']
            
            # Update profile
            self._update_user_profile(user_id, transaction, profile)
        
        return {
            'anomalies': anomalies,
            'anomaly_score': min(anomaly_score, 1.0),
            'user_profile_size': len(self.transaction_history[user_id]),
            'profile_age_days': profile.get('profile_age_days', 0)
        }
    
    def _create_user_profile(self, user_id: str, user_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Create user profile from history"""
        
        profile = {
            'user_id': user_id,
            'transaction_count': 0,
            'total_amount': 0.0,
            'avg_amount': 0.0,
            'std_amount': 0.0,
            'min_amount': float('inf'),
            'max_amount': 0.0,
            'common_merchants': defaultdict(int),
            'common_hours': defaultdict(int),
            'avg_daily_transactions': 0.0,
            'last_transaction_time': None,
            'profile_created': datetime.now(),
            'profile_age_days': 0
        }
        
        if user_history:
            amounts = []
            for tx in user_history:
                amount = tx.get('amount', 0)
                amounts.append(amount)
                
                # Update merchants
                merchant_id = tx.get('merchant_id', 'unknown')
                profile['common_merchants'][merchant_id] += 1
                
                # Update time patterns
                timestamp = tx.get('timestamp', '')
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        profile['common_hours'][dt.hour] += 1
                        profile['last_transaction_time'] = dt
                    except Exception:
                        pass
            
            if amounts:
                profile['transaction_count'] = len(amounts)
                profile['total_amount'] = sum(amounts)
                profile['avg_amount'] = np.mean(amounts)
                profile['std_amount'] = np.std(amounts)
                profile['min_amount'] = min(amounts)
                profile['max_amount'] = max(amounts)
                
                # Calculate daily transaction rate
                if profile['last_transaction_time']:
                    days = (datetime.now() - profile['last_transaction_time']).days
                    if days > 0:
                        profile['avg_daily_transactions'] = len(amounts) / days
                        profile['profile_age_days'] = days
        
        return profile
    
    def _detect_amount_anomaly(self, amount: float, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Detect amount-based anomalies"""
        
        if profile['transaction_count'] < 5:
            return {'is_anomaly': False, 'score': 0.0, 'reason': 'Insufficient history'}
        
        # Z-score based detection
        avg_amount = profile['avg_amount']
        std_amount = profile['std_amount']
        
        if std_amount > 0:
            z_score = abs((amount - avg_amount) / std_amount)
            
            if z_score > self.anomaly_threshold:
                return {
                    'is_anomaly': True,
                    'score': min(z_score / 5.0, 0.5),
                    'reason': f'Amount ${amount:.2f} is {z_score:.1f} std devs from average ${avg_amount:.2f}'
                }
        
        # Check if amount is unusually high compared to max
        if amount > profile['max_amount'] * 2:
            return {
                'is_anomaly': True,
                'score': 0.4,
                'reason': f'Amount ${amount:.2f} is more than double previous max ${profile["max_amount"]:.2f}'
            }
        
        return {'is_anomaly': False, 'score': 0.0, 'reason': ''}
    
    def _detect_time_anomaly(self, timestamp: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Detect time-based anomalies"""
        
        if not timestamp or not profile['common_hours']:
            return {'is_anomaly': False, 'score': 0.0, 'reason': 'No time history'}
        
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            hour = dt.hour
            
            # Check if this hour is rare for the user
            total_transactions = sum(profile['common_hours'].values())
            hour_frequency = profile['common_hours'].get(hour, 0) / total_transactions
            
            if hour_frequency < 0.05 and total_transactions > 20:  # Less than 5% of transactions
                most_common_hours = sorted(profile['common_hours'].items(), 
                                         key=lambda x: x[1], reverse=True)[:3]
                common_hours_str = ', '.join([f"{h[0]}:00" for h in most_common_hours])
                
                return {
                    'is_anomaly': True,
                    'score': 0.3,
                    'reason': f'Transaction at {hour}:00 (user typically transacts at {common_hours_str})'
                }
            
            # Check time since last transaction
            if profile['last_transaction_time']:
                time_diff = (dt - profile['last_transaction_time']).total_seconds()
                if time_diff < 60:  # Less than 1 minute
                    return {
                        'is_anomaly': True,
                        'score': 0.4,
                        'reason': f'Only {time_diff:.0f} seconds since last transaction'
                    }
            
        except Exception:
            pass
        
        return {'is_anomaly': False, 'score': 0.0, 'reason': ''}
    
    def _detect_frequency_anomaly(self, user_id: str, timestamp: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Detect transaction frequency anomalies"""
        
        if not timestamp:
            return {'is_anomaly': False, 'score': 0.0, 'reason': 'No timestamp'}
        
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            # Check transactions in last hour
            recent_transactions = []
            for tx_time in self.transaction_history[user_id]:
                if isinstance(tx_time, str):
                    try:
                        tx_dt = datetime.fromisoformat(tx_time.replace('Z', '+00:00'))
                        if (dt - tx_dt).total_seconds() < 3600:  # 1 hour
                            recent_transactions.append(tx_dt)
                    except Exception:
                        continue
            
            if len(recent_transactions) >= 5:  # 5+ transactions in an hour
                return {
                    'is_anomaly': True,
                    'score': 0.5,
                    'reason': f'{len(recent_transactions)} transactions in the last hour'
                }
            
            # Check against average daily rate
            if profile['avg_daily_transactions'] > 0:
                hourly_rate = profile['avg_daily_transactions'] / 24
                if len(recent_transactions) > hourly_rate * 3:  # 3x normal rate
                    return {
                        'is_anomaly': True,
                        'score': 0.3,
                        'reason': f'Transaction rate {len(recent_transactions)}/hour exceeds normal rate'
                    }
            
        except Exception:
            pass
        
        return {'is_anomaly': False, 'score': 0.0, 'reason': ''}
    
    def _detect_merchant_anomaly(self, transaction: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        """Detect merchant-based anomalies"""
        
        merchant_id = transaction.get('merchant_id', 'unknown')
        amount = transaction.get('amount', 0)
        
        if not profile['common_merchants']:
            return {'is_anomaly': False, 'score': 0.0, 'reason': 'No merchant history'}
        
        # Check if new merchant with high amount
        if merchant_id not in profile['common_merchants']:
            if amount > profile['avg_amount'] * 2:
                return {
                    'is_anomaly': True,
                    'score': 0.4,
                    'reason': f'First transaction with new merchant for amount ${amount:.2f}'
                }
        
        return {'is_anomaly': False, 'score': 0.0, 'reason': ''}
    
    def _update_user_profile(self, user_id: str, transaction: Dict[str, Any], profile: Dict[str, Any]) -> None:
        """Update user profile with new transaction"""
        
        # Add transaction timestamp to history
        timestamp = transaction.get('timestamp', '')
        if timestamp:
            self.transaction_history[user_id].append(timestamp)
        
        # Update merchant count
        merchant_id = transaction.get('merchant_id', 'unknown')
        profile['common_merchants'][merchant_id] += 1
        
        # Update time patterns
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                profile['common_hours'][dt.hour] += 1
                profile['last_transaction_time'] = dt
            except Exception:
                pass
        
        # Update amount statistics
        amount = transaction.get('amount', 0)
        profile['transaction_count'] += 1
        profile['total_amount'] += amount
        profile['avg_amount'] = profile['total_amount'] / profile['transaction_count']
        
        if amount < profile['min_amount']:
            profile['min_amount'] = amount
        if amount > profile['max_amount']:
            profile['max_amount'] = amount
        
        # Recalculate standard deviation (simplified)
        if profile['transaction_count'] > 1:
            # This is a simplified update - in production, you'd want a more efficient method
            amounts = [amount]  # Would need to store recent amounts for accurate calculation
            profile['std_amount'] = np.std(amounts) if len(amounts) > 1 else 0
        
        # Update profile age
        profile['profile_age_days'] = (datetime.now() - profile['profile_created']).days

# ======================== UTILITY FUNCTIONS ========================

def create_default_fraud_core() -> EnhancedFraudDetectionCore:
    """Create enhanced fraud detection core with default configuration"""
    config = EnhancedFraudCoreConfig()
    return EnhancedFraudDetectionCore(config)

def create_production_fraud_core() -> EnhancedFraudDetectionCore:
    """Create enhanced fraud detection core optimized for production"""
    config = EnhancedFraudCoreConfig(
        detection_strategy=DetectionStrategy.ENSEMBLE,
        enable_async_processing=True,
        max_worker_threads=8,
        enable_caching=True,
        cache_size=5000,
        fraud_probability_threshold=0.6,
        enable_audit_logging=True,
        enable_compliance_checks=True
    )
    return EnhancedFraudDetectionCore(config)

def validate_transaction_format(transaction: Dict[str, Any]) -> bool:
    """Validate transaction format"""
    required_fields = ['transaction_id', 'amount', 'user_id', 'timestamp']
    return all(field in transaction for field in required_fields)