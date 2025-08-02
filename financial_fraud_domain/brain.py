"""
Brain System - Compatible with Fraud Detection Integration
Main Brain system that integrates with fraud detection domain
"""

import logging
import threading
import json
import time
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import uuid
import traceback
import psutil

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class BrainSystemConfig:
    """Brain system configuration"""
    base_path: Path = field(default_factory=lambda: Path.cwd() / ".brain")
    enable_persistence: bool = True
    enable_monitoring: bool = True
    enable_adaptation: bool = True
    max_domains: int = 50
    max_memory_gb: float = 8.0
    max_cpu_percent: float = 70.0
    enable_parallel_predictions: bool = True
    max_prediction_threads: int = 4
    prediction_cache_size: int = 1000
    auto_save_interval: int = 300  # seconds
    max_concurrent_training: int = 2
    default_routing_strategy: str = "hybrid"
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Ensure base_path is a Path object
        if isinstance(self.base_path, str):
            self.base_path = Path(self.base_path)
        
        # Create directory structure
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Setup subdirectories
        self.knowledge_path = self.base_path / "knowledge"
        self.models_path = self.base_path / "models"
        self.training_path = self.base_path / "training"
        self.logs_path = self.base_path / "logs"
        
        # Create subdirectories
        for path in [self.knowledge_path, self.models_path, self.training_path, self.logs_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Brain core configuration
        self.brain_config = {
            'max_memory_mb': int(self.max_memory_gb * 1024),
            'max_cpu_percent': self.max_cpu_percent,
            'cache_size': self.prediction_cache_size,
            'enable_parallel': self.enable_parallel_predictions
        }


@dataclass
class BrainPredictionResult:
    """Brain prediction result"""
    success: bool
    prediction: Any
    confidence: float
    domain: str
    reasoning: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'success': self.success,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'domain': self.domain,
            'reasoning': self.reasoning,
            'metadata': self.metadata,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp.isoformat()
        }


class DomainRegistry:
    """Basic domain registry for Brain system"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        self.domains = {}
        self.domain_configs = {}
        self.domain_handlers = {}
        self._lock = threading.RLock()
        
        # Load existing domains if storage path provided
        if self.storage_path and self.storage_path.exists():
            self._load_domains()
    
    def register_domain(self, domain_name: str, config: Any) -> bool:
        """Register a new domain"""
        with self._lock:
            try:
                self.domains[domain_name] = {
                    'name': domain_name,
                    'status': 'registered',
                    'registered_at': datetime.now(),
                    'type': getattr(config, 'domain_type', 'unknown')
                }
                self.domain_configs[domain_name] = config
                
                logger.info(f"Domain '{domain_name}' registered successfully")
                self._save_domains()
                return True
                
            except Exception as e:
                logger.error(f"Failed to register domain '{domain_name}': {e}")
                return False
    
    def is_domain_registered(self, domain_name: str) -> bool:
        """Check if domain is registered"""
        return domain_name in self.domains
    
    def list_domains(self) -> List[Dict[str, Any]]:
        """List all registered domains"""
        with self._lock:
            return list(self.domains.values())
    
    def get_domain_config(self, domain_name: str) -> Optional[Any]:
        """Get domain configuration"""
        return self.domain_configs.get(domain_name)
    
    def _save_domains(self):
        """Save domains to storage"""
        if not self.storage_path:
            return
        
        try:
            data = {
                'domains': {name: asdict(domain) if hasattr(domain, '__dict__') else domain 
                           for name, domain in self.domains.items()},
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save domains: {e}")
    
    def _load_domains(self):
        """Load domains from storage"""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self.domains = data.get('domains', {})
            logger.info(f"Loaded {len(self.domains)} domains from storage")
            
        except Exception as e:
            logger.error(f"Failed to load domains: {e}")


class BrainCore:
    """Core Brain processing engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.shared_knowledge = {}
        self.prediction_cache = {}
        self.processing_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_processing_time': 0.0
        }
        self._lock = threading.RLock()
    
    def add_shared_knowledge(self, key: str, value: Any, domain: Optional[str] = None):
        """Add shared knowledge item"""
        with self._lock:
            if domain:
                key = f"{domain}_{key}"
            self.shared_knowledge[key] = {
                'value': value,
                'added_at': datetime.now(),
                'domain': domain
            }
    
    def get_shared_knowledge(self, key: str, domain: Optional[str] = None) -> Any:
        """Get shared knowledge item"""
        if domain:
            key = f"{domain}_{key}"
        
        knowledge = self.shared_knowledge.get(key)
        return knowledge['value'] if knowledge else None
    
    def process_prediction(self, data: Dict[str, Any], domain: str) -> BrainPredictionResult:
        """Process prediction request"""
        start_time = time.time()
        
        with self._lock:
            self.processing_stats['total_predictions'] += 1
        
        try:
            # Simple prediction logic - can be enhanced
            if domain == 'financial_fraud':
                prediction = self._process_fraud_prediction(data)
            else:
                prediction = self._process_general_prediction(data)
            
            processing_time = time.time() - start_time
            
            result = BrainPredictionResult(
                success=True,
                prediction=prediction,
                confidence=0.8,  # Default confidence
                domain=domain,
                reasoning=[f"Processed by Brain core for domain: {domain}"],
                processing_time=processing_time
            )
            
            with self._lock:
                self.processing_stats['successful_predictions'] += 1
                # Update average processing time
                total = self.processing_stats['total_predictions']
                current_avg = self.processing_stats['average_processing_time']
                self.processing_stats['average_processing_time'] = (
                    (current_avg * (total - 1) + processing_time) / total
                )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            with self._lock:
                self.processing_stats['failed_predictions'] += 1
            
            return BrainPredictionResult(
                success=False,
                prediction=None,
                confidence=0.0,
                domain=domain,
                reasoning=[f"Prediction failed: {str(e)}"],
                processing_time=processing_time,
                metadata={'error': str(e)}
            )
    
    def _process_fraud_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process fraud detection prediction"""
        # Simple fraud detection logic
        amount = data.get('amount', 0)
        
        # Basic rules
        fraud_score = 0.0
        indicators = []
        
        if amount > 10000:
            fraud_score += 0.3
            indicators.append('high_amount')
        
        if amount < 0:
            fraud_score += 0.9
            indicators.append('negative_amount')
        
        # Check for suspicious patterns
        timestamp = data.get('timestamp', '')
        if '02:' in timestamp or '03:' in timestamp or '04:' in timestamp:
            fraud_score += 0.2
            indicators.append('unusual_hour')
        
        merchant = data.get('merchant_id', '')
        if 'suspicious' in merchant.lower():
            fraud_score += 0.4
            indicators.append('suspicious_merchant')
        
        return {
            'fraud_probability': min(fraud_score, 1.0),
            'fraud_score': fraud_score,
            'fraud_indicators': indicators,
            'recommendation': 'block' if fraud_score > 0.7 else 'approve'
        }
    
    def _process_general_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process general prediction"""
        return {
            'prediction': 'processed',
            'confidence': 0.5,
            'data_received': len(data)
        }


class Brain:
    """Main Brain system class"""
    
    def __init__(self, config: Optional[Union[BrainSystemConfig, Dict[str, Any]]] = None):
        """Initialize Brain system"""
        # Handle configuration
        if config is None:
            self.config = BrainSystemConfig()
        elif isinstance(config, dict):
            self.config = BrainSystemConfig(**config)
        elif isinstance(config, BrainSystemConfig):
            self.config = config
        else:
            raise TypeError(f"Invalid config type: {type(config)}")
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self._initialized = False
        self._shutdown = False
        self._lock = threading.RLock()
        
        # Core components
        self.brain_core = BrainCore(self.config.brain_config)
        self.domain_registry = DomainRegistry(self.config.knowledge_path / "domains.json")
        
        # Fraud-specific components
        self.fraud_handlers = {}
        self.fraud_metrics = {
            'total_fraud_requests': 0,
            'fraud_detected_count': 0,
            'average_detection_time': 0.0
        }
        
        # Performance monitoring
        self.performance_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'average_latency': 0.0
        }
        
        # Initialize default domains
        self._initialize_default_domains()
        
        # Register fraud domain
        self._register_fraud_domain()
        
        self._initialized = True
        logger.info("Brain system initialized successfully")
    
    def _setup_logging(self):
        """Setup Brain logging"""
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Create logs directory
        log_file = self.config.logs_path / "brain.log"
        
        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def _initialize_default_domains(self):
        """Initialize default domains"""
        # Simple domain config class
        class DomainConfig:
            def __init__(self, domain_type='general', description='', **kwargs):
                self.domain_type = domain_type
                self.description = description
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        default_domains = [
            ('general', DomainConfig(domain_type='general', description='General purpose domain')),
            ('mathematics', DomainConfig(domain_type='specialized', description='Mathematical reasoning')),
            ('language', DomainConfig(domain_type='specialized', description='Language processing'))
        ]
        
        for domain_name, config in default_domains:
            if not self.domain_registry.is_domain_registered(domain_name):
                self.domain_registry.register_domain(domain_name, config)
    
    def _register_fraud_domain(self):
        """Register financial fraud domain"""
        domain_name = 'financial_fraud'
        
        if not self.domain_registry.is_domain_registered(domain_name):
            # Simple fraud domain config
            class FraudDomainConfig:
                def __init__(self):
                    self.domain_type = 'specialized'
                    self.description = 'Financial fraud detection domain'
                    self.version = '1.0.0'
                    self.capabilities = [
                        'fraud_detection', 'risk_assessment', 'transaction_analysis'
                    ]
            
            config = FraudDomainConfig()
            success = self.domain_registry.register_domain(domain_name, config)
            
            if success:
                # Initialize fraud handler
                self.fraud_handlers[domain_name] = self._create_fraud_handler()
                logger.info(f"Fraud domain '{domain_name}' registered and handler created")
            else:
                logger.error(f"Failed to register fraud domain '{domain_name}'")
    
    def _create_fraud_handler(self):
        """Create fraud detection handler"""
        class FraudHandler:
            def __init__(self, brain_instance):
                self.brain = brain_instance
                self.metrics = {
                    'total_requests': 0,
                    'fraud_detected': 0,
                    'processing_time': 0.0
                }
            
            def detect_fraud(self, data):
                start_time = time.time()
                self.metrics['total_requests'] += 1
                
                try:
                    # Use brain core for prediction
                    result = self.brain.brain_core.process_prediction(data, 'financial_fraud')
                    
                    if result.success and result.prediction:
                        fraud_prob = result.prediction.get('fraud_probability', 0.0)
                        if fraud_prob > 0.5:
                            self.metrics['fraud_detected'] += 1
                    
                    processing_time = time.time() - start_time
                    self.metrics['processing_time'] = (
                        (self.metrics['processing_time'] * (self.metrics['total_requests'] - 1) + 
                         processing_time) / self.metrics['total_requests']
                    )
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Fraud detection failed: {e}")
                    return BrainPredictionResult(
                        success=False,
                        prediction=None,
                        confidence=0.0,
                        domain='financial_fraud',
                        reasoning=[f"Error: {str(e)}"]
                    )
        
        return FraudHandler(self)
    
    def add_domain(self, domain_name: str, config: Any, initialize_model: bool = True) -> Dict[str, Any]:
        """Add a new domain to the Brain"""
        try:
            success = self.domain_registry.register_domain(domain_name, config)
            
            if success:
                return {
                    'success': True,
                    'domain_name': domain_name,
                    'domain_type': getattr(config, 'domain_type', 'unknown'),
                    'initialized': initialize_model
                }
            else:
                return {
                    'success': False,
                    'error': f'Failed to register domain: {domain_name}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def list_available_domains(self) -> List[Dict[str, Any]]:
        """List all available domains"""
        domains = self.domain_registry.list_domains()
        
        # Add status information
        for domain in domains:
            domain['status'] = 'active' if domain['name'] in self.fraud_handlers or domain['name'] in ['general', 'mathematics', 'language'] else 'registered'
        
        return domains
    
    def get_domain_capabilities(self, domain_name: str) -> Dict[str, Any]:
        """Get capabilities for a specific domain"""
        config = self.domain_registry.get_domain_config(domain_name)
        
        if not config:
            return {'error': f'Domain {domain_name} not found'}
        
        capabilities = getattr(config, 'capabilities', [])
        
        return {
            'domain_name': domain_name,
            'status': 'active',
            'capabilities': [
                {'name': cap, 'description': f'{cap} capability'} 
                for cap in capabilities
            ] if isinstance(capabilities, list) else [
                {'name': 'general', 'description': 'General processing capability'}
            ]
        }
    
    def predict(self, data: Any, domain: Optional[str] = None, return_reasoning: bool = False) -> BrainPredictionResult:
        """Make a prediction using the Brain system"""
        start_time = time.time()
        
        try:
            # Determine domain
            if not domain:
                # Simple domain routing
                if isinstance(data, dict):
                    if any(key in data for key in ['transaction_id', 'amount', 'fraud']):
                        domain = 'financial_fraud'
                    else:
                        domain = 'general'
                else:
                    domain = 'general'
            
            # Process prediction
            if domain == 'financial_fraud' and domain in self.fraud_handlers:
                result = self.fraud_handlers[domain].detect_fraud(data)
            else:
                result = self.brain_core.process_prediction(data, domain)
            
            # Update metrics
            with self._lock:
                self.performance_metrics['total_operations'] += 1
                if result.success:
                    self.performance_metrics['successful_operations'] += 1
                else:
                    self.performance_metrics['failed_operations'] += 1
                
                # Update average latency
                latency = time.time() - start_time
                total_ops = self.performance_metrics['total_operations']
                current_avg = self.performance_metrics['average_latency']
                self.performance_metrics['average_latency'] = (
                    (current_avg * (total_ops - 1) + latency) / total_ops
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return BrainPredictionResult(
                success=False,
                prediction=None,
                confidence=0.0,
                domain=domain or 'unknown',
                reasoning=[f"Prediction error: {str(e)}"],
                metadata={'error': str(e)}
            )
    
    def detect_fraud(self, transaction: Dict[str, Any], user_context: Optional[Dict[str, Any]] = None):
        """Detect fraud in a financial transaction"""
        # Import fraud detection result class
        try:
            from brain_fraud_support import FraudDetectionResult
        except ImportError:
            # Create a simple fraud detection result
            class FraudDetectionResult:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
        
        start_time = time.time()
        self.fraud_metrics['total_fraud_requests'] += 1
        
        try:
            # Use fraud handler if available
            fraud_domain = 'financial_fraud'
            if fraud_domain in self.fraud_handlers:
                result = self.fraud_handlers[fraud_domain].detect_fraud(transaction)
                
                if result.success and result.prediction:
                    fraud_prob = result.prediction.get('fraud_probability', 0.0)
                    fraud_detected = fraud_prob > 0.5
                    
                    if fraud_detected:
                        self.fraud_metrics['fraud_detected_count'] += 1
                    
                    detection_time = (time.time() - start_time) * 1000
                    
                    # Update average detection time
                    current_avg = self.fraud_metrics['average_detection_time']
                    count = self.fraud_metrics['total_fraud_requests']
                    self.fraud_metrics['average_detection_time'] = (
                        (current_avg * (count - 1) + detection_time) / count
                    )
                    
                    return FraudDetectionResult(
                        transaction_id=transaction.get('transaction_id', 'unknown'),
                        fraud_detected=fraud_detected,
                        fraud_probability=fraud_prob,
                        risk_score=result.prediction.get('fraud_score', 0.0),
                        confidence=result.confidence,
                        detection_strategy="brain_integrated",
                        detection_time=detection_time,
                        timestamp=datetime.now(),
                        explanation=f"Brain fraud detection: {result.reasoning[0] if result.reasoning else 'No reasoning'}",
                        validation_passed=True,
                        indicators=result.prediction.get('fraud_indicators', [])
                    )
            
            # Fallback fraud detection
            return self._fallback_fraud_detection(transaction)
            
        except Exception as e:
            logger.error(f"Fraud detection error: {e}")
            return FraudDetectionResult(
                transaction_id=transaction.get('transaction_id', 'unknown'),
                fraud_detected=False,
                fraud_probability=0.0,
                risk_score=0.0,
                confidence=0.0,
                detection_strategy="error_fallback",
                detection_time=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                explanation=f"Detection failed: {str(e)}",
                validation_passed=False,
                validation_errors=[str(e)]
            )
    
    def _fallback_fraud_detection(self, transaction: Dict[str, Any]):
        """Fallback fraud detection method"""
        try:
            from brain_fraud_support import FraudDetectionResult
        except ImportError:
            class FraudDetectionResult:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
        
        # Simple fallback logic
        amount = transaction.get('amount', 0)
        fraud_score = 0.1  # Low default score
        
        if amount > 10000:
            fraud_score = 0.6
        elif amount < 0:
            fraud_score = 0.9
        
        return FraudDetectionResult(
            transaction_id=transaction.get('transaction_id', 'fallback'),
            fraud_detected=fraud_score > 0.5,
            fraud_probability=fraud_score,
            risk_score=fraud_score,
            confidence=0.7,
            detection_strategy="fallback_rules",
            detection_time=10.0,
            timestamp=datetime.now(),
            explanation="Fallback fraud detection using simple rules",
            validation_passed=True
        )
    
    def get_domain_health(self, domain_name: str) -> Dict[str, Any]:
        """Get domain health information"""
        if not self.domain_registry.is_domain_registered(domain_name):
            return {
                'health_score': 0,
                'health_status': 'not_found',
                'is_trained': False
            }
        
        # Basic health metrics
        base_score = 70
        
        # Boost score for active domains
        if domain_name in self.fraud_handlers:
            base_score += 15
        
        # Boost score for default domains
        if domain_name in ['general', 'mathematics', 'language']:
            base_score += 10
        
        return {
            'health_score': min(base_score, 100),
            'health_status': 'good' if base_score >= 70 else 'poor',
            'is_trained': domain_name in self.fraud_handlers or domain_name in ['general', 'mathematics', 'language']
        }
    
    def get_brain_status(self) -> Dict[str, Any]:
        """Get comprehensive Brain status"""
        # Resource usage
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            cpu_percent = process.cpu_percent()
        except:
            memory_mb = 0
            cpu_percent = 0
        
        return {
            'initialized': self._initialized,
            'total_domains': len(self.domain_registry.domains),
            'active_domains': len([d for d in self.list_available_domains() if d['status'] == 'active']),
            'memory_usage_mb': memory_mb,
            'memory_usage_percent': (memory_mb / (self.config.max_memory_gb * 1024)) * 100,
            'cpu_usage_percent': cpu_percent,
            'performance_metrics': self.performance_metrics.copy(),
            'fraud_metrics': self.fraud_metrics.copy(),
            'brain_core_stats': self.brain_core.processing_stats.copy(),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_fraud_system_status(self) -> Dict[str, Any]:
        """Get fraud system status"""
        return {
            'fraud_domain_registered': self.domain_registry.is_domain_registered('financial_fraud'),
            'fraud_handlers_active': len(self.fraud_handlers),
            'fraud_metrics': self.fraud_metrics.copy(),
            'fraud_detection_available': 'financial_fraud' in self.fraud_handlers,
            'timestamp': datetime.now().isoformat()
        }
    
    def shutdown(self):
        """Shutdown Brain system"""
        if self._shutdown:
            return
        
        logger.info("Shutting down Brain system...")
        self._shutdown = True
        
        # Save domain state
        try:
            self.domain_registry._save_domains()
        except Exception as e:
            logger.error(f"Failed to save domains during shutdown: {e}")
        
        logger.info("Brain system shutdown complete")