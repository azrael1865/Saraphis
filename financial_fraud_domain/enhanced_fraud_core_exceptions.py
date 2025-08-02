"""
Enhanced Fraud Detection Core - Chunk 1: Core Exceptions and Enums
Comprehensive exception hierarchy and enums for the enhanced fraud detection system
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field

# Configure logging
logger = logging.getLogger(__name__)

# ======================== CORE ENUMS ========================

class DetectionStrategy(Enum):
    """Fraud detection strategy types"""
    RULES_ONLY = "rules_only"
    ML_ONLY = "ml_only"
    SYMBOLIC_ONLY = "symbolic_only"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"

class ValidationLevel(Enum):
    """Validation strictness levels"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"

class SecurityLevel(Enum):
    """Security levels for fraud detection"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class RecoveryStrategy(Enum):
    """Error recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    FAIL_FAST = "fail_fast"
    GRACEFUL_DEGRADATION = "graceful_degradation"

class CacheEvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"
    ADAPTIVE = "adaptive"

class MonitoringLevel(Enum):
    """Monitoring detail levels"""
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    DEBUG = "debug"

class AuditLevel(Enum):
    """Audit logging levels"""
    NONE = "none"
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"

# ======================== CORE EXCEPTIONS ========================

@dataclass
class ErrorContext:
    """Context information for errors"""
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

class EnhancedFraudException(Exception):
    """Base exception for enhanced fraud detection system"""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None,
                 recovery_suggestions: Optional[List[str]] = None,
                 error_code: Optional[str] = None):
        super().__init__(message)
        self.context = context or ErrorContext()
        self.recovery_suggestions = recovery_suggestions or []
        self.error_code = error_code
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization"""
        return {
            'exception_type': self.__class__.__name__,
            'message': str(self),
            'error_code': self.error_code,
            'timestamp': self.timestamp.isoformat(),
            'context': {
                'timestamp': self.context.timestamp.isoformat(),
                'correlation_id': self.context.correlation_id,
                'user_id': self.context.user_id,
                'session_id': self.context.session_id,
                'request_id': self.context.request_id,
                'component': self.context.component,
                'operation': self.context.operation,
                'additional_data': self.context.additional_data
            },
            'recovery_suggestions': self.recovery_suggestions
        }

# ======================== PRIMARY FRAUD CORE ERROR ========================

class FraudCoreError(EnhancedFraudException):
    """
    Primary base exception for fraud detection core functionality.
    This is the main exception that other modules import.
    """
    pass

# ======================== CORE EXCEPTION CATEGORIES ========================

class ValidationError(FraudCoreError):
    """Exception raised when validation fails"""
    pass

class ConfigurationError(FraudCoreError):
    """Exception raised when configuration is invalid"""
    pass

class DetectionError(FraudCoreError):
    """Exception raised during fraud detection process"""
    pass

class SecurityError(FraudCoreError):
    """Exception raised for security-related issues"""
    pass

class PerformanceError(FraudCoreError):
    """Exception raised for performance-related issues"""
    pass

class ResourceError(FraudCoreError):
    """Exception raised when resource limits are exceeded"""
    pass

class IntegrationError(FraudCoreError):
    """Exception raised during system integration"""
    pass

class DataQualityError(FraudCoreError):
    """Exception raised when data quality issues are detected"""
    pass

class ModelError(FraudCoreError):
    """Exception raised for ML model-related issues"""
    pass

class ProcessingError(FraudCoreError):
    """Exception raised during data processing operations"""
    
    def __init__(self, message: str, processing_stage: Optional[str] = None,
                 input_data: Optional[Any] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.processing_stage = processing_stage
        self.input_data = input_data
        if self.context:
            self.context.additional_data.update({
                'processing_stage': processing_stage,
                'input_data_type': type(input_data).__name__ if input_data else None
            })

class DataError(FraudCoreError):
    """Exception raised for data-related issues"""
    
    def __init__(self, message: str, data_source: Optional[str] = None,
                 data_type: Optional[str] = None, record_count: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.data_source = data_source
        self.data_type = data_type
        self.record_count = record_count
        if self.context:
            self.context.additional_data.update({
                'data_source': data_source,
                'data_type': data_type,
                'record_count': record_count
            })

class AuditError(FraudCoreError):
    """Exception raised for audit-related issues"""
    pass

class CacheError(FraudCoreError):
    """Exception raised for cache-related issues"""
    pass

class MonitoringError(FraudCoreError):
    """Exception raised for monitoring-related issues"""
    pass

class CircuitBreakerError(FraudCoreError):
    """Exception raised when circuit breaker is open"""
    pass

# ======================== ANALYTICS-SPECIFIC EXCEPTIONS ========================

class AnalyticsError(FraudCoreError):
    """Base exception for analytics-related errors"""
    pass

class ReportGenerationError(AnalyticsError):
    """Exception raised when report generation fails"""
    
    def __init__(self, message: str, report_type: Optional[str] = None,
                 report_id: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.report_type = report_type
        self.report_id = report_id

class VisualizationError(AnalyticsError):
    """Exception raised when visualization creation fails"""
    
    def __init__(self, message: str, visualization_type: Optional[str] = None,
                 chart_id: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.visualization_type = visualization_type
        self.chart_id = chart_id

class ExportError(AnalyticsError):
    """Exception raised when data export fails"""
    
    def __init__(self, message: str, export_format: Optional[str] = None,
                 destination: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.export_format = export_format
        self.destination = destination

class MetricsCalculationError(AnalyticsError):
    """Exception raised when metrics calculation fails"""
    
    def __init__(self, message: str, metric_type: Optional[str] = None,
                 calculation_method: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.metric_type = metric_type
        self.calculation_method = calculation_method

# ======================== IEEE DATASET SPECIFIC EXCEPTIONS ========================

class FraudDataError(DataError):
    """Exception raised for IEEE fraud dataset data issues"""
    
    def __init__(self, message: str, dataset_type: Optional[str] = None,
                 file_path: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.dataset_type = dataset_type
        self.file_path = file_path

class FraudValidationError(ValidationError):
    """Exception raised when IEEE fraud dataset validation fails"""
    
    def __init__(self, message: str, validation_type: Optional[str] = None,
                 validation_failures: Optional[List[str]] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.validation_type = validation_type
        self.validation_failures = validation_failures or []

class FraudProcessingError(ProcessingError):
    """Exception raised during IEEE fraud dataset processing"""
    
    def __init__(self, message: str, processing_stage: Optional[str] = None,
                 dataset_size: Optional[int] = None, **kwargs):
        super().__init__(message, processing_stage=processing_stage, **kwargs)
        self.dataset_size = dataset_size

class FraudTrainingError(ModelError):
    """Exception raised during fraud model training with IEEE dataset"""
    
    def __init__(self, message: str, training_stage: Optional[str] = None,
                 model_type: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.training_stage = training_stage
        self.model_type = model_type

# ======================== SPECIALIZED EXCEPTIONS ========================

class TransactionValidationError(ValidationError):
    """Exception raised when transaction validation fails"""
    
    def __init__(self, message: str, transaction_id: Optional[str] = None,
                 validation_failures: Optional[List[str]] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.transaction_id = transaction_id
        self.validation_failures = validation_failures or []

class StrategyValidationError(ValidationError):
    """Exception raised when strategy validation fails"""
    
    def __init__(self, message: str, strategy: Optional[DetectionStrategy] = None,
                 validation_failures: Optional[List[str]] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.strategy = strategy
        self.validation_failures = validation_failures or []

class ComponentValidationError(ValidationError):
    """Exception raised when component validation fails"""
    
    def __init__(self, message: str, component_name: Optional[str] = None,
                 validation_failures: Optional[List[str]] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.component_name = component_name
        self.validation_failures = validation_failures or []

class PerformanceValidationError(ValidationError):
    """Exception raised when performance validation fails"""
    
    def __init__(self, message: str, metric_name: Optional[str] = None,
                 threshold: Optional[float] = None, actual_value: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.metric_name = metric_name
        self.threshold = threshold
        self.actual_value = actual_value

class SecurityValidationError(ValidationError):
    """Exception raised when security validation fails"""
    
    def __init__(self, message: str, security_level: Optional[SecurityLevel] = None,
                 threat_type: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.security_level = security_level
        self.threat_type = threat_type

# ======================== UTILITY FUNCTIONS ========================

def create_error_context(correlation_id: Optional[str] = None,
                        user_id: Optional[str] = None,
                        component: Optional[str] = None,
                        operation: Optional[str] = None,
                        **additional_data) -> ErrorContext:
    """Create standardized error context"""
    return ErrorContext(
        correlation_id=correlation_id,
        user_id=user_id,
        component=component,
        operation=operation,
        additional_data=additional_data
    )

def log_exception(exception: Union[FraudCoreError, EnhancedFraudException], 
                 logger_instance: logging.Logger = None) -> None:
    """Log exception with structured information"""
    if logger_instance is None:
        logger_instance = logger
    
    logger_instance.error(
        f"Fraud Core Exception: {exception.__class__.__name__}",
        extra={
            'exception_data': exception.to_dict(),
            'correlation_id': exception.context.correlation_id,
            'component': exception.context.component,
            'operation': exception.context.operation
        }
    )

def handle_exception(exception: Exception, context: Optional[ErrorContext] = None,
                    recovery_suggestions: Optional[List[str]] = None) -> FraudCoreError:
    """Convert standard exception to fraud core exception"""
    if isinstance(exception, FraudCoreError):
        return exception
    
    if isinstance(exception, EnhancedFraudException):
        # Convert EnhancedFraudException to FraudCoreError
        return FraudCoreError(
            str(exception),
            context=exception.context,
            recovery_suggestions=exception.recovery_suggestions,
            error_code=exception.error_code
        )
    
    # Map common exceptions to specific fraud core exceptions
    exception_mapping = {
        ValueError: ValidationError,
        KeyError: ConfigurationError,
        TimeoutError: PerformanceError,
        MemoryError: ResourceError,
        IOError: DataError,
        OSError: ResourceError,
        RuntimeError: ProcessingError,
        AttributeError: ConfigurationError,
        TypeError: ValidationError,
        ImportError: IntegrationError,
        ModuleNotFoundError: IntegrationError
    }
    
    for exception_type, fraud_exception_class in exception_mapping.items():
        if isinstance(exception, exception_type):
            return fraud_exception_class(str(exception), context, recovery_suggestions)
    
    # Default to FraudCoreError for unknown exceptions
    return FraudCoreError(str(exception), context, recovery_suggestions)

# ======================== EXCEPTION REGISTRY ========================

EXCEPTION_REGISTRY = {
    'fraud_core': FraudCoreError,
    'validation': ValidationError,
    'configuration': ConfigurationError,
    'detection': DetectionError,
    'security': SecurityError,
    'performance': PerformanceError,
    'resource': ResourceError,
    'integration': IntegrationError,
    'data_quality': DataQualityError,
    'data': DataError,
    'processing': ProcessingError,
    'model': ModelError,
    'audit': AuditError,
    'cache': CacheError,
    'monitoring': MonitoringError,
    'circuit_breaker': CircuitBreakerError,
    'fraud_data': FraudDataError,
    'fraud_validation': FraudValidationError,
    'fraud_processing': FraudProcessingError,
    'fraud_training': FraudTrainingError,
    'transaction_validation': TransactionValidationError,
    'strategy_validation': StrategyValidationError,
    'component_validation': ComponentValidationError,
    'performance_validation': PerformanceValidationError,
    'security_validation': SecurityValidationError,
    'analytics': AnalyticsError,
    'report_generation': ReportGenerationError,
    'visualization': VisualizationError,
    'export': ExportError,
    'metrics_calculation': MetricsCalculationError
}

def get_exception_class(exception_type: str) -> type:
    """Get exception class by type name"""
    return EXCEPTION_REGISTRY.get(exception_type, FraudCoreError)

def create_exception(exception_type: str, message: str, **kwargs) -> FraudCoreError:
    """Create exception instance by type name"""
    exception_class = get_exception_class(exception_type)
    return exception_class(message, **kwargs)

# ======================== EXCEPTION BUILDER ========================

class ExceptionBuilder:
    """Builder pattern for creating complex exceptions with context"""
    
    def __init__(self, exception_class: type = FraudCoreError):
        self.exception_class = exception_class
        self.message = ""
        self.context = ErrorContext()
        self.recovery_suggestions = []
        self.error_code = None
        self.additional_params = {}
    
    def with_message(self, message: str) -> 'ExceptionBuilder':
        """Set exception message"""
        self.message = message
        return self
    
    def with_correlation_id(self, correlation_id: str) -> 'ExceptionBuilder':
        """Set correlation ID"""
        self.context.correlation_id = correlation_id
        return self
    
    def with_user_id(self, user_id: str) -> 'ExceptionBuilder':
        """Set user ID"""
        self.context.user_id = user_id
        return self
    
    def with_component(self, component: str) -> 'ExceptionBuilder':
        """Set component name"""
        self.context.component = component
        return self
    
    def with_operation(self, operation: str) -> 'ExceptionBuilder':
        """Set operation name"""
        self.context.operation = operation
        return self
    
    def with_error_code(self, error_code: str) -> 'ExceptionBuilder':
        """Set error code"""
        self.error_code = error_code
        return self
    
    def with_recovery_suggestion(self, suggestion: str) -> 'ExceptionBuilder':
        """Add recovery suggestion"""
        self.recovery_suggestions.append(suggestion)
        return self
    
    def with_additional_data(self, **kwargs) -> 'ExceptionBuilder':
        """Add additional context data"""
        self.context.additional_data.update(kwargs)
        return self
    
    def with_param(self, **kwargs) -> 'ExceptionBuilder':
        """Add exception-specific parameters"""
        self.additional_params.update(kwargs)
        return self
    
    def build(self) -> FraudCoreError:
        """Build the exception instance"""
        return self.exception_class(
            self.message,
            context=self.context,
            recovery_suggestions=self.recovery_suggestions,
            error_code=self.error_code,
            **self.additional_params
        )

# ======================== EXCEPTION DECORATORS ========================

def handle_fraud_exceptions(default_return=None, log_errors=True, 
                           reraise=False, exception_type=FraudCoreError):
    """
    Decorator to handle fraud detection exceptions
    
    Args:
        default_return: Default value to return on exception
        log_errors: Whether to log exceptions
        reraise: Whether to reraise exceptions after handling
        exception_type: Type of exception to raise if reraising
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FraudCoreError as e:
                if log_errors:
                    log_exception(e)
                if reraise:
                    raise
                return default_return
            except Exception as e:
                fraud_exception = handle_exception(e)
                if log_errors:
                    log_exception(fraud_exception)
                if reraise:
                    raise fraud_exception
                return default_return
        return wrapper
    return decorator