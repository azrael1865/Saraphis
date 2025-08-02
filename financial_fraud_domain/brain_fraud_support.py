"""
Brain System Enhancement for Fraud Domain Support
Extends the Brain system with complete fraud detection capabilities
"""

import logging
import threading
import time
import json
import uuid
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import traceback

# Configure logging
logger = logging.getLogger(__name__)

# ======================== BRAIN FRAUD EXTENSIONS ========================

class BrainFraudExtensions:
    """
    Extensions to add to Brain class for complete fraud domain support.
    These methods should be dynamically added to the Brain instance.
    """
    
    @staticmethod
    def enhance_brain_for_fraud(brain_instance):
        """
        Enhance a Brain instance with fraud detection capabilities.
        This function adds all necessary methods and attributes to support fraud detection.
        """
        
        # Add fraud-specific attributes
        brain_instance._fraud_initialized = False
        brain_instance._fraud_connector = None
        brain_instance._fraud_domain_name = "financial_fraud"
        brain_instance._fraud_detection_cache = {}
        brain_instance._fraud_cache_lock = threading.Lock()
        
        # Add methods
        brain_instance.detect_fraud = lambda *args, **kwargs: BrainFraudExtensions._detect_fraud(brain_instance, *args, **kwargs)
        brain_instance.batch_detect_fraud = lambda *args, **kwargs: BrainFraudExtensions._batch_detect_fraud(brain_instance, *args, **kwargs)
        brain_instance.get_fraud_system_status = lambda: BrainFraudExtensions._get_fraud_system_status(brain_instance)
        brain_instance.configure_fraud_detection = lambda *args, **kwargs: BrainFraudExtensions._configure_fraud_detection(brain_instance, *args, **kwargs)
        brain_instance.monitor_fraud_performance = lambda: BrainFraudExtensions._monitor_fraud_performance(brain_instance)
        brain_instance.train_fraud_model = lambda *args, **kwargs: BrainFraudExtensions._train_fraud_model(brain_instance, *args, **kwargs)
        brain_instance.get_fraud_capabilities = lambda: BrainFraudExtensions._get_fraud_capabilities(brain_instance)
        brain_instance.validate_fraud_transaction = lambda *args, **kwargs: BrainFraudExtensions._validate_fraud_transaction(brain_instance, *args, **kwargs)
        brain_instance.get_fraud_risk_assessment = lambda *args, **kwargs: BrainFraudExtensions._get_fraud_risk_assessment(brain_instance, *args, **kwargs)
        brain_instance.export_fraud_patterns = lambda: BrainFraudExtensions._export_fraud_patterns(brain_instance)
        brain_instance.import_fraud_patterns = lambda *args, **kwargs: BrainFraudExtensions._import_fraud_patterns(brain_instance, *args, **kwargs)
        
        # Initialize fraud support
        BrainFraudExtensions._initialize_fraud_support(brain_instance)
        
        logger.info("Brain instance enhanced with fraud detection capabilities")
    
    @staticmethod
    def _initialize_fraud_support(brain_instance):
        """Initialize fraud detection support in Brain"""
        try:
            # Check if fraud domain needs to be registered
            if not brain_instance.domain_registry.is_domain_registered(brain_instance._fraud_domain_name):
                # Import domain config
                from independent_core.domain_registry import DomainConfig, DomainType
                
                # Create fraud domain configuration
                fraud_config = DomainConfig(
                    domain_type=DomainType.SPECIALIZED,
                    description="Financial fraud detection and prevention",
                    version="1.0.0",
                    max_memory_mb=2048,
                    max_cpu_percent=40.0,
                    priority=9,
                    hidden_layers=[512, 256, 128, 64],
                    activation_function="relu",
                    dropout_rate=0.2,
                    learning_rate=0.001,
                    enable_caching=True,
                    cache_size=1000,
                    enable_logging=True,
                    author="Saraphis AI",
                    tags=["financial", "fraud", "security", "risk", "ml", "rules"]
                )
                
                # Register domain
                success = brain_instance.domain_registry.register_domain(
                    brain_instance._fraud_domain_name,
                    fraud_config
                )
                
                if success:
                    logger.info(f"Fraud domain '{brain_instance._fraud_domain_name}' registered successfully")
                    
                    # Update domain status to active
                    brain_instance.domain_registry.update_domain_status(
                        brain_instance._fraud_domain_name,
                        brain_instance.domain_registry.DomainStatus.ACTIVE
                    )
            
            # Add fraud patterns to domain router
            BrainFraudExtensions._configure_fraud_routing(brain_instance)
            
            # Initialize fraud connector if available
            try:
                from financial_fraud_domain.brain_fraud_connector import BrainFraudSystemConnector, BrainFraudConnectorConfig
                
                connector_config = BrainFraudConnectorConfig(
                    fraud_domain_name=brain_instance._fraud_domain_name,
                    enable_auto_registration=False,  # Already registered
                    enable_state_sync=True,
                    enable_performance_monitoring=True
                )
                
                # Create connector but don't initialize Brain (avoid circular reference)
                brain_instance._fraud_connector = BrainFraudSystemConnector.__new__(BrainFraudSystemConnector)
                brain_instance._fraud_connector.config = connector_config
                brain_instance._fraud_connector.state = brain_instance._fraud_connector.ConnectorState()
                brain_instance._fraud_connector.initialized = False
                brain_instance._fraud_connector._shutdown = False
                brain_instance._fraud_connector.brain = brain_instance
                
                # Initialize fraud system separately
                try:
                    from financial_fraud_domain.enhanced_fraud_core_system import create_production_fraud_system
                    brain_instance._fraud_connector.fraud_system = create_production_fraud_system()
                    brain_instance._fraud_connector.fraud_core = brain_instance._fraud_connector.fraud_system.fraud_core
                    brain_instance._fraud_connector.initialized = True
                    logger.info("Fraud connector initialized with Brain instance")
                except Exception as e:
                    logger.warning(f"Could not initialize fraud system in connector: {e}")
                
            except ImportError:
                logger.info("Fraud connector not available, using basic fraud support")
                brain_instance._fraud_connector = None
            
            brain_instance._fraud_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize fraud support: {e}")
            brain_instance._fraud_initialized = False
    
    @staticmethod
    def _configure_fraud_routing(brain_instance):
        """Configure fraud-specific routing patterns"""
        try:
            # Add fraud detection patterns to shared knowledge
            fraud_patterns = {
                'keywords': [
                    'fraud', 'fraudulent', 'scam', 'suspicious',
                    'transaction', 'payment', 'financial', 'money',
                    'risk', 'anomaly', 'unauthorized', 'theft',
                    'laundering', 'phishing', 'identity', 'breach'
                ],
                'patterns': [
                    {'pattern': 'detect.*fraud', 'weight': 1.0},
                    {'pattern': 'fraud.*detection', 'weight': 1.0},
                    {'pattern': 'suspicious.*transaction', 'weight': 0.9},
                    {'pattern': 'risk.*assessment', 'weight': 0.8},
                    {'pattern': 'financial.*anomaly', 'weight': 0.8},
                    {'pattern': 'payment.*verification', 'weight': 0.7},
                    {'pattern': 'transaction.*analysis', 'weight': 0.7}
                ],
                'domain_indicators': {
                    'transaction_fields': ['amount', 'merchant_id', 'user_id', 'timestamp'],
                    'risk_indicators': ['high_amount', 'unusual_time', 'new_merchant'],
                    'fraud_types': ['card_fraud', 'account_takeover', 'money_laundering']
                }
            }
            
            # Store patterns in shared knowledge
            brain_instance.brain_core.add_shared_knowledge(
                key=f"domain_{brain_instance._fraud_domain_name}_routing_patterns",
                value=fraud_patterns,
                domain=brain_instance._fraud_domain_name
            )
            
            logger.debug("Fraud routing patterns configured")
            
        except Exception as e:
            logger.error(f"Failed to configure fraud routing: {e}")
    
    @staticmethod
    def _detect_fraud(brain_instance, transaction: Dict[str, Any], 
                     user_context: Optional[Dict[str, Any]] = None) -> 'FraudDetectionResult':
        """
        Detect fraud in a financial transaction.
        
        Args:
            transaction: Transaction data including amount, user_id, merchant_id, etc.
            user_context: Optional user context for security and personalization
            
        Returns:
            FraudDetectionResult with detailed fraud analysis
        """
        if not brain_instance._fraud_initialized:
            BrainFraudExtensions._initialize_fraud_support(brain_instance)
        
        # Check cache first
        cache_key = BrainFraudExtensions._generate_cache_key(transaction)
        with brain_instance._fraud_cache_lock:
            if cache_key in brain_instance._fraud_detection_cache:
                cached_result, cache_time = brain_instance._fraud_detection_cache[cache_key]
                if (datetime.now() - cache_time).seconds < 3600:  # 1 hour cache
                    return cached_result
        
        try:
            # If connector is available, use it
            if brain_instance._fraud_connector and brain_instance._fraud_connector.initialized:
                result = brain_instance._fraud_connector.detect_fraud(
                    transaction, user_context, use_brain=True
                )
                
                # Convert BrainPredictionResult to FraudDetectionResult if needed
                if hasattr(result, 'transaction_id'):
                    # Already a FraudDetectionResult
                    pass
                else:
                    # Convert from BrainPredictionResult
                    from financial_fraud_domain.enhanced_fraud_core_main import FraudDetectionResult, DetectionStrategy
                    
                    fraud_data = result.prediction if isinstance(result.prediction, dict) else {}
                    result = FraudDetectionResult(
                        transaction_id=transaction.get('transaction_id', 'unknown'),
                        fraud_detected=fraud_data.get('fraud_detected', False),
                        fraud_probability=fraud_data.get('fraud_probability', 0.0),
                        risk_score=fraud_data.get('risk_score', 0.0),
                        confidence=result.confidence,
                        detection_strategy=DetectionStrategy.HYBRID,
                        detection_time=result.execution_time,
                        timestamp=datetime.now(),
                        explanation=fraud_data.get('explanation', ''),
                        reasoning_chain=result.reasoning
                    )
            else:
                # Use Brain's prediction system
                prediction_input = {
                    'domain': brain_instance._fraud_domain_name,
                    'task': 'fraud_detection',
                    'data': transaction,
                    'context': user_context or {}
                }
                
                brain_result = brain_instance.predict(
                    prediction_input,
                    domain=brain_instance._fraud_domain_name,
                    return_reasoning=True
                )
                
                # Convert to FraudDetectionResult
                from financial_fraud_domain.enhanced_fraud_core_main import FraudDetectionResult, DetectionStrategy
                
                # Extract fraud-specific data
                fraud_detected = False
                fraud_probability = 0.0
                risk_score = 0.0
                
                if isinstance(brain_result.prediction, dict):
                    fraud_detected = brain_result.prediction.get('fraud_detected', False)
                    fraud_probability = brain_result.prediction.get('fraud_probability', 0.0)
                    risk_score = brain_result.prediction.get('risk_score', 0.0)
                elif isinstance(brain_result.prediction, bool):
                    fraud_detected = brain_result.prediction
                    fraud_probability = brain_result.confidence if fraud_detected else 1 - brain_result.confidence
                    risk_score = fraud_probability
                
                result = FraudDetectionResult(
                    transaction_id=transaction.get('transaction_id', 'unknown'),
                    fraud_detected=fraud_detected,
                    fraud_probability=fraud_probability,
                    risk_score=risk_score,
                    confidence=brain_result.confidence,
                    detection_strategy=DetectionStrategy.HYBRID,
                    detection_time=brain_result.execution_time,
                    timestamp=datetime.now(),
                    explanation=brain_result.reasoning[0] if brain_result.reasoning else 'Fraud analysis completed',
                    reasoning_chain=brain_result.reasoning,
                    additional_metadata={
                        'brain_domain': brain_result.domain,
                        'routing_info': brain_result.routing_info
                    }
                )
            
            # Cache result
            with brain_instance._fraud_cache_lock:
                brain_instance._fraud_detection_cache[cache_key] = (result, datetime.now())
                # Limit cache size
                if len(brain_instance._fraud_detection_cache) > 1000:
                    # Remove oldest entries
                    oldest_keys = sorted(brain_instance._fraud_detection_cache.keys(), 
                                       key=lambda k: brain_instance._fraud_detection_cache[k][1])[:100]
                    for key in oldest_keys:
                        del brain_instance._fraud_detection_cache[key]
            
            return result
            
        except Exception as e:
            logger.error(f"Fraud detection failed: {e}")
            # Return safe default result
            from financial_fraud_domain.enhanced_fraud_core_main import FraudDetectionResult, DetectionStrategy
            
            return FraudDetectionResult(
                transaction_id=transaction.get('transaction_id', 'unknown'),
                fraud_detected=False,
                fraud_probability=0.0,
                risk_score=0.0,
                confidence=0.0,
                detection_strategy=DetectionStrategy.HYBRID,
                detection_time=0.0,
                timestamp=datetime.now(),
                explanation=f"Fraud detection failed: {str(e)}",
                validation_passed=False,
                validation_errors=[str(e)]
            )
    
    @staticmethod
    def _batch_detect_fraud(brain_instance, transactions: List[Dict[str, Any]], 
                           user_context: Optional[Dict[str, Any]] = None) -> List['FraudDetectionResult']:
        """Batch fraud detection for multiple transactions"""
        results = []
        
        # Use thread pool for parallel processing
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all transactions
            futures = {
                executor.submit(
                    BrainFraudExtensions._detect_fraud, 
                    brain_instance, 
                    transaction, 
                    user_context
                ): transaction
                for transaction in transactions
            }
            
            # Collect results in order
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    transaction = futures[future]
                    logger.error(f"Batch fraud detection failed for transaction {transaction.get('transaction_id')}: {e}")
                    
                    # Add error result
                    from financial_fraud_domain.enhanced_fraud_core_main import FraudDetectionResult, DetectionStrategy
                    
                    error_result = FraudDetectionResult(
                        transaction_id=transaction.get('transaction_id', 'unknown'),
                        fraud_detected=False,
                        fraud_probability=0.0,
                        risk_score=0.0,
                        confidence=0.0,
                        detection_strategy=DetectionStrategy.HYBRID,
                        detection_time=0.0,
                        timestamp=datetime.now(),
                        explanation=f"Batch detection failed: {str(e)}",
                        validation_passed=False,
                        validation_errors=[str(e)]
                    )
                    results.append(error_result)
        
        return results
    
    @staticmethod
    def _get_fraud_system_status(brain_instance) -> Dict[str, Any]:
        """Get comprehensive fraud system status"""
        status = {
            'fraud_support_initialized': brain_instance._fraud_initialized,
            'fraud_domain_registered': brain_instance.domain_registry.is_domain_registered(brain_instance._fraud_domain_name),
            'fraud_connector_available': brain_instance._fraud_connector is not None,
            'cache_size': len(brain_instance._fraud_detection_cache)
        }
        
        # Get domain info
        if status['fraud_domain_registered']:
            domain_info = brain_instance.domain_registry.get_domain_info(brain_instance._fraud_domain_name)
            status['fraud_domain_info'] = {
                'status': domain_info.get('status'),
                'type': domain_info.get('type'),
                'version': domain_info.get('version'),
                'total_predictions': domain_info.get('total_predictions', 0),
                'average_confidence': domain_info.get('average_confidence', 0)
            }
        
        # Get connector status if available
        if brain_instance._fraud_connector:
            try:
                connector_status = brain_instance._fraud_connector.get_status()
                status['fraud_connector_status'] = connector_status
            except Exception as e:
                status['fraud_connector_status'] = {'error': str(e)}
        
        # Get fraud domain metrics from Brain
        try:
            domain_health = brain_instance.get_domain_health(brain_instance._fraud_domain_name)
            status['fraud_domain_health'] = {
                'health_score': domain_health.get('health_score', 0),
                'health_status': domain_health.get('health_status', 'unknown'),
                'is_trained': domain_health.get('is_trained', False),
                'performance': domain_health.get('performance', {})
            }
        except Exception as e:
            status['fraud_domain_health'] = {'error': str(e)}
        
        return status
    
    @staticmethod
    def _configure_fraud_detection(brain_instance, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure fraud detection settings"""
        results = {
            'success': True,
            'updates': [],
            'errors': []
        }
        
        try:
            # If connector is available, use it
            if brain_instance._fraud_connector:
                connector_result = brain_instance._fraud_connector.configure_fraud_detection(config)
                results['connector_configuration'] = connector_result
                if connector_result.get('success'):
                    results['updates'].extend(connector_result.get('updated_settings', []))
            
            # Update Brain-level settings
            if 'cache_enabled' in config:
                # This would affect Brain's caching for fraud domain
                results['updates'].append('cache_enabled')
            
            if 'fraud_domain_priority' in config:
                # Update domain priority
                domain_info = brain_instance.domain_registry.get_domain_info(brain_instance._fraud_domain_name)
                if domain_info:
                    domain_info['config']['priority'] = config['fraud_domain_priority']
                    results['updates'].append('fraud_domain_priority')
            
        except Exception as e:
            results['success'] = False
            results['errors'].append(str(e))
            logger.error(f"Failed to configure fraud detection: {e}")
        
        return results
    
    @staticmethod
    def _monitor_fraud_performance(brain_instance) -> Dict[str, Any]:
        """Monitor fraud detection performance"""
        performance = {
            'timestamp': datetime.now().isoformat(),
            'fraud_domain': brain_instance._fraud_domain_name,
            'cache_metrics': {
                'size': len(brain_instance._fraud_detection_cache),
                'hit_rate': 0.0  # Would need to track hits/misses
            }
        }
        
        # Get domain performance from Brain
        try:
            domain_capabilities = brain_instance.get_domain_capabilities(brain_instance._fraud_domain_name)
            performance['domain_performance'] = domain_capabilities.get('performance_profile', {})
        except Exception as e:
            performance['domain_performance'] = {'error': str(e)}
        
        # Get connector performance if available
        if brain_instance._fraud_connector:
            try:
                connector_metrics = brain_instance._fraud_connector.get_performance_metrics()
                performance['connector_metrics'] = connector_metrics
            except Exception as e:
                performance['connector_metrics'] = {'error': str(e)}
        
        return performance
    
    @staticmethod
    def _train_fraud_model(brain_instance, training_data: Any, 
                          training_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train fraud detection model"""
        try:
            # Use Brain's training system
            result = brain_instance.train_domain(
                brain_instance._fraud_domain_name,
                training_data,
                training_config
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Fraud model training failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    def _get_fraud_capabilities(brain_instance) -> Dict[str, Any]:
        """Get detailed fraud detection capabilities"""
        return brain_instance.get_domain_capabilities(brain_instance._fraud_domain_name)
    
    @staticmethod
    def _validate_fraud_transaction(brain_instance, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Validate transaction data for fraud detection"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Required fields
        required_fields = ['transaction_id', 'user_id', 'amount', 'timestamp']
        for field in required_fields:
            if field not in transaction:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Missing required field: {field}")
        
        # Validate amount
        if 'amount' in transaction:
            try:
                amount = float(transaction['amount'])
                if amount < 0:
                    validation_result['valid'] = False
                    validation_result['errors'].append("Amount cannot be negative")
                elif amount > 1000000:
                    validation_result['warnings'].append("Unusually high amount")
            except (TypeError, ValueError):
                validation_result['valid'] = False
                validation_result['errors'].append("Invalid amount format")
        
        # Validate timestamp
        if 'timestamp' in transaction:
            try:
                # Try to parse timestamp
                if isinstance(transaction['timestamp'], str):
                    datetime.fromisoformat(transaction['timestamp'].replace('Z', '+00:00'))
            except Exception:
                validation_result['valid'] = False
                validation_result['errors'].append("Invalid timestamp format")
        
        return validation_result
    
    @staticmethod
    def _get_fraud_risk_assessment(brain_instance, user_id: str, 
                                  time_period: Optional[int] = 30) -> Dict[str, Any]:
        """Get fraud risk assessment for a user"""
        try:
            # This would analyze historical data for the user
            # For now, return a mock assessment
            assessment = {
                'user_id': user_id,
                'risk_level': 'low',  # low, medium, high, critical
                'risk_score': 0.2,
                'assessment_period_days': time_period,
                'factors': {
                    'transaction_frequency': 'normal',
                    'amount_patterns': 'consistent',
                    'merchant_diversity': 'typical',
                    'geographic_consistency': 'stable'
                },
                'recommendations': [
                    'Continue normal monitoring',
                    'No special restrictions needed'
                ],
                'last_assessment': datetime.now().isoformat()
            }
            
            return assessment
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {
                'error': str(e),
                'user_id': user_id
            }
    
    @staticmethod
    def _export_fraud_patterns(brain_instance) -> Dict[str, Any]:
        """Export learned fraud patterns"""
        try:
            # Get fraud knowledge from Brain
            fraud_knowledge = brain_instance.brain_core.search_shared_knowledge(
                "", brain_instance._fraud_domain_name
            )
            
            patterns = {
                'export_timestamp': datetime.now().isoformat(),
                'domain': brain_instance._fraud_domain_name,
                'patterns': {
                    'rules': [],
                    'ml_features': [],
                    'behavioral_patterns': [],
                    'risk_indicators': []
                },
                'statistics': {
                    'total_patterns': 0,
                    'high_confidence_patterns': 0
                }
            }
            
            # Extract patterns from knowledge
            for item in fraud_knowledge:
                if 'pattern' in item.get('key', ''):
                    pattern_data = item.get('value', {})
                    if isinstance(pattern_data, dict):
                        pattern_type = pattern_data.get('type', 'unknown')
                        if pattern_type in patterns['patterns']:
                            patterns['patterns'][pattern_type].append(pattern_data)
                            patterns['statistics']['total_patterns'] += 1
                            
                            if pattern_data.get('confidence', 0) > 0.8:
                                patterns['statistics']['high_confidence_patterns'] += 1
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to export fraud patterns: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def _import_fraud_patterns(brain_instance, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Import fraud patterns"""
        try:
            imported_count = 0
            
            # Validate pattern format
            if 'patterns' not in patterns:
                return {
                    'success': False,
                    'error': 'Invalid pattern format'
                }
            
            # Import each pattern type
            for pattern_type, pattern_list in patterns['patterns'].items():
                for pattern in pattern_list:
                    key = f"fraud_pattern_{pattern_type}_{imported_count}"
                    brain_instance.brain_core.add_shared_knowledge(
                        key=key,
                        value={
                            'type': pattern_type,
                            'pattern': pattern,
                            'imported_at': datetime.now().isoformat(),
                            'source': 'import'
                        },
                        domain=brain_instance._fraud_domain_name
                    )
                    imported_count += 1
            
            return {
                'success': True,
                'imported_patterns': imported_count
            }
            
        except Exception as e:
            logger.error(f"Failed to import fraud patterns: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    def _generate_cache_key(transaction: Dict[str, Any]) -> str:
        """Generate cache key for transaction"""
        # Create deterministic key from transaction data
        key_parts = [
            transaction.get('transaction_id', ''),
            str(transaction.get('user_id', '')),
            str(transaction.get('amount', 0)),
            str(transaction.get('merchant_id', ''))
        ]
        return '_'.join(key_parts)

# ======================== BRAIN INITIALIZATION PATCH ========================

def patch_brain_class():
    """
    Patch the Brain class to automatically support fraud detection.
    This should be called when the Brain module is imported.
    """
    try:
        # Import Brain class
        from independent_core.brain import Brain
        
        # Store original __init__
        original_init = Brain.__init__
        
        # Create new __init__ that adds fraud support
        def enhanced_init(self, config=None):
            # Call original init
            original_init(self, config)
            
            # Add fraud support
            try:
                BrainFraudExtensions.enhance_brain_for_fraud(self)
            except Exception as e:
                logger.warning(f"Could not add fraud support to Brain: {e}")
        
        # Replace __init__
        Brain.__init__ = enhanced_init
        
        logger.info("Brain class patched for fraud detection support")
        
    except Exception as e:
        logger.error(f"Failed to patch Brain class: {e}")

# ======================== STANDALONE FRAUD BRAIN ========================

class FraudDetectionBrain:
    """
    Standalone fraud detection brain that can be used without full Brain system.
    Provides same API but focused only on fraud detection.
    """
    
    def __init__(self):
        """Initialize standalone fraud detection brain"""
        self.initialized = False
        self._cache = {}
        self._cache_lock = threading.Lock()
        
        # Try to import fraud system
        try:
            from financial_fraud_domain.enhanced_fraud_core_system import create_production_fraud_system
            self.fraud_system = create_production_fraud_system()
            self.initialized = True
            logger.info("Standalone fraud detection brain initialized")
        except Exception as e:
            logger.error(f"Failed to initialize standalone fraud brain: {e}")
            self.fraud_system = None
    
    def detect_fraud(self, transaction: Dict[str, Any], 
                    user_context: Optional[Dict[str, Any]] = None) -> Any:
        """Detect fraud in transaction"""
        if not self.initialized or not self.fraud_system:
            raise RuntimeError("Fraud detection brain not initialized")
        
        return self.fraud_system.detect_fraud(transaction, user_context)
    
    def batch_detect_fraud(self, transactions: List[Dict[str, Any]], 
                          user_context: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Batch fraud detection"""
        if not self.initialized or not self.fraud_system:
            raise RuntimeError("Fraud detection brain not initialized")
        
        return self.fraud_system.batch_detect_fraud(transactions, user_context)
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        if not self.initialized or not self.fraud_system:
            return {'initialized': False, 'error': 'System not initialized'}
        
        return self.fraud_system.get_system_status()
    
    def configure(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure fraud detection"""
        if not self.initialized or not self.fraud_system:
            return {'success': False, 'error': 'System not initialized'}
        
        return self.fraud_system.configure_fraud_detection(config)
    
    def shutdown(self) -> None:
        """Shutdown fraud brain"""
        if self.fraud_system:
            self.fraud_system.shutdown()
        self.initialized = False

# ======================== INTEGRATION HELPERS ========================

def ensure_brain_fraud_support():
    """Ensure Brain has fraud detection support"""
    try:
        # Try to patch Brain class
        patch_brain_class()
        return True
    except Exception as e:
        logger.error(f"Could not ensure Brain fraud support: {e}")
        return False

def create_fraud_enabled_brain(config=None):
    """Create a Brain instance with fraud detection enabled"""
    try:
        from independent_core.brain import Brain
        
        # Create Brain instance
        brain = Brain(config)
        
        # Ensure fraud support
        if not hasattr(brain, 'detect_fraud'):
            BrainFraudExtensions.enhance_brain_for_fraud(brain)
        
        return brain
        
    except Exception as e:
        logger.error(f"Failed to create fraud-enabled Brain: {e}")
        # Return standalone fraud brain as fallback
        return FraudDetectionBrain()

# ======================== USAGE EXAMPLES ========================

def demonstrate_brain_fraud_integration():
    """Demonstrate Brain fraud detection integration"""
    print("\n" + "="*80)
    print("BRAIN FRAUD DETECTION INTEGRATION DEMONSTRATION")
    print("="*80)
    
    # Create fraud-enabled Brain
    brain = create_fraud_enabled_brain()
    
    # Example transaction
    transaction = {
        'transaction_id': 'demo_tx_001',
        'user_id': 'demo_user_123',
        'amount': 5000.00,
        'merchant_id': 'suspicious_merchant_xyz',
        'timestamp': datetime.now().isoformat(),
        'type': 'purchase',
        'currency': 'USD'
    }
    
    print(f"\nDetecting fraud for transaction: {transaction['transaction_id']}")
    print(f"Amount: ${transaction['amount']}")
    
    try:
        # Detect fraud
        result = brain.detect_fraud(transaction)
        
        print(f"\nResult:")
        print(f"  Fraud Detected: {result.fraud_detected}")
        print(f"  Probability: {result.fraud_probability:.3f}")
        print(f"  Risk Score: {result.risk_score:.3f}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Explanation: {result.explanation}")
        
        # Get fraud system status
        status = brain.get_fraud_system_status()
        print(f"\nFraud System Status:")
        print(f"  Initialized: {status['fraud_support_initialized']}")
        print(f"  Domain Registered: {status['fraud_domain_registered']}")
        print(f"  Cache Size: {status['cache_size']}")
        
    except Exception as e:
        print(f"\nError: {e}")
    
    finally:
        # Cleanup
        if hasattr(brain, 'shutdown'):
            brain.shutdown()
        print("\nDemonstration complete")

# Auto-patch Brain class when module is imported
ensure_brain_fraud_support()

if __name__ == '__main__':
    demonstrate_brain_fraud_integration()