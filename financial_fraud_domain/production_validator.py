"""
Production Validator - Chunk 3: Production Validator with Monitoring
Comprehensive production-ready validator with monitoring, diagnostics,
health checks, and integration with existing validation infrastructure.
"""

import logging
import pandas as pd
import numpy as np
import time
import asyncio
import threading
import psutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import json
import hashlib
import gc

# Import enhanced components
try:
    from enhanced_validator_framework import (
        TransactionValidationError, BusinessRuleValidationError, 
        enhanced_input_validator, enhanced_error_handler, enhanced_performance_monitor,
        validation_context_manager, ValidationMetricsCollector
    )
    from enhanced_transaction_validator import (
        EnhancedTransactionFieldValidator, EnhancedBusinessRuleValidator,
        ValidationContext, RiskLevel
    )
    from data_validator import (
        ValidationLevel, ValidationSeverity, ComplianceStandard,
        ValidationIssue, ValidationResult, FinancialDataValidator
    )
    from enhanced_data_validator import (
        EnhancedFinancialDataValidator, ValidationConfig,
        ValidationMode, SecurityLevel
    )
    ENHANCED_COMPONENTS = True
except ImportError as e:
    ENHANCED_COMPONENTS = False
    logger.warning(f"Enhanced components not available: {e}")

# Configure logging
logger = logging.getLogger(__name__)

# ======================== PRODUCTION MONITORING CLASSES ========================

@dataclass
class SystemHealth:
    """System health status tracking"""
    status: str  # 'healthy', 'degraded', 'unhealthy'
    cpu_usage: float
    memory_usage: float
    memory_limit: float
    active_validations: int
    error_rate: float
    response_time: float
    uptime_seconds: float
    last_check: datetime = field(default_factory=datetime.now)
    issues: List[str] = field(default_factory=list)

@dataclass
class ValidationDiagnostics:
    """Comprehensive validation diagnostics"""
    validation_id: str
    validator_version: str
    validation_level: ValidationLevel
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    records_processed: int = 0
    issues_found: int = 0
    success: bool = False
    error_message: Optional[str] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    total_validations: int = 0
    successful_validations: int = 0
    failed_validations: int = 0
    average_duration: float = 0.0
    peak_memory_usage: float = 0.0
    total_records_processed: int = 0
    throughput_records_per_second: float = 0.0
    error_rate_percentage: float = 0.0
    cache_effectiveness: float = 0.0
    last_reset: datetime = field(default_factory=datetime.now)

class ProductionMonitor:
    """Production monitoring and alerting system"""
    
    def __init__(self, alert_thresholds: Optional[Dict[str, float]] = None):
        self.alert_thresholds = alert_thresholds or {
            'cpu_threshold': 80.0,  # CPU usage percentage
            'memory_threshold': 85.0,  # Memory usage percentage
            'error_rate_threshold': 10.0,  # Error rate percentage
            'response_time_threshold': 30.0,  # Response time in seconds
            'queue_size_threshold': 100  # Validation queue size
        }
        
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 metrics
        self.health_checks = deque(maxlen=100)  # Keep last 100 health checks
        self.alerts = deque(maxlen=500)  # Keep last 500 alerts
        
        self._monitoring_active = True
        self._monitor_thread = None
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start background monitoring thread"""
        def monitor_loop():
            while self._monitoring_active:
                try:
                    health = self._check_system_health()
                    self.health_checks.append(health)
                    
                    # Check for alerts
                    self._check_alerts(health)
                    
                    time.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def _check_system_health(self) -> SystemHealth:
        """Check current system health"""
        cpu_usage = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        memory_limit = memory_info.total / 1024 / 1024 / 1024  # GB
        
        # Calculate error rate from recent metrics
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 validations
        error_rate = 0.0
        if recent_metrics:
            failed_count = sum(1 for m in recent_metrics if not m.get('success', True))
            error_rate = failed_count / len(recent_metrics) * 100
        
        # Calculate average response time
        response_time = 0.0
        if recent_metrics:
            response_times = [m.get('duration', 0) for m in recent_metrics]
            response_time = sum(response_times) / len(response_times)
        
        # Determine status
        issues = []
        if cpu_usage > self.alert_thresholds['cpu_threshold']:
            issues.append(f"High CPU usage: {cpu_usage:.1f}%")
        if memory_usage > self.alert_thresholds['memory_threshold']:
            issues.append(f"High memory usage: {memory_usage:.1f}%")
        if error_rate > self.alert_thresholds['error_rate_threshold']:
            issues.append(f"High error rate: {error_rate:.1f}%")
        if response_time > self.alert_thresholds['response_time_threshold']:
            issues.append(f"Slow response time: {response_time:.1f}s")
        
        status = 'healthy'
        if len(issues) >= 3:
            status = 'unhealthy'
        elif len(issues) >= 1:
            status = 'degraded'
        
        return SystemHealth(
            status=status,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            memory_limit=memory_limit,
            active_validations=0,  # Would be set by validator
            error_rate=error_rate,
            response_time=response_time,
            uptime_seconds=time.time() - self._start_time if hasattr(self, '_start_time') else 0,
            issues=issues
        )
    
    def _check_alerts(self, health: SystemHealth):
        """Check for alert conditions"""
        if health.status in ['degraded', 'unhealthy']:
            alert = {
                'timestamp': datetime.now(),
                'severity': 'WARNING' if health.status == 'degraded' else 'CRITICAL',
                'message': f"System health {health.status}: {', '.join(health.issues)}",
                'health_data': asdict(health)
            }
            self.alerts.append(alert)
            
            # Log alert
            if health.status == 'unhealthy':
                logger.critical(f"CRITICAL ALERT: {alert['message']}")
            else:
                logger.warning(f"WARNING ALERT: {alert['message']}")
    
    def record_validation_metrics(self, diagnostics: ValidationDiagnostics):
        """Record validation metrics"""
        metrics = {
            'timestamp': datetime.now(),
            'validation_id': diagnostics.validation_id,
            'duration': diagnostics.duration_seconds,
            'records_processed': diagnostics.records_processed,
            'issues_found': diagnostics.issues_found,
            'success': diagnostics.success,
            'memory_usage': diagnostics.memory_usage_mb,
            'cpu_usage': diagnostics.cpu_usage_percent
        }
        self.metrics_history.append(metrics)
    
    def get_performance_summary(self) -> PerformanceMetrics:
        """Get performance summary from recent metrics"""
        recent_metrics = list(self.metrics_history)
        if not recent_metrics:
            return PerformanceMetrics()
        
        total_validations = len(recent_metrics)
        successful_validations = sum(1 for m in recent_metrics if m.get('success', False))
        failed_validations = total_validations - successful_validations
        
        durations = [m.get('duration', 0) for m in recent_metrics if m.get('duration')]
        average_duration = sum(durations) / len(durations) if durations else 0.0
        
        memory_usages = [m.get('memory_usage', 0) for m in recent_metrics]
        peak_memory_usage = max(memory_usages) if memory_usages else 0.0
        
        total_records = sum(m.get('records_processed', 0) for m in recent_metrics)
        total_time = sum(durations)
        throughput = total_records / total_time if total_time > 0 else 0.0
        
        error_rate = failed_validations / total_validations * 100 if total_validations > 0 else 0.0
        
        return PerformanceMetrics(
            total_validations=total_validations,
            successful_validations=successful_validations,
            failed_validations=failed_validations,
            average_duration=average_duration,
            peak_memory_usage=peak_memory_usage,
            total_records_processed=total_records,
            throughput_records_per_second=throughput,
            error_rate_percentage=error_rate,
            cache_effectiveness=0.0  # Would be calculated from cache metrics
        )
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        return list(self.alerts)[-count:]
    
    def shutdown(self):
        """Shutdown monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)

# ======================== PRODUCTION VALIDATOR ========================

class ProductionFinancialValidator:
    """
    Production-ready financial data validator with comprehensive monitoring,
    error handling, and integration with existing validation components.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 enable_monitoring: bool = True,
                 enable_caching: bool = True,
                 max_workers: int = 4):
        
        # Initialize configuration
        self.config = self._create_config(config)
        self.enable_monitoring = enable_monitoring
        self.enable_caching = enable_caching
        self.max_workers = max_workers
        
        # Initialize validators
        self._init_validators()
        
        # Initialize monitoring
        if enable_monitoring:
            self.monitor = ProductionMonitor()
            self.monitor._start_time = time.time()
        else:
            self.monitor = None
        
        # Initialize metrics collector
        self.metrics_collector = ValidationMetricsCollector()
        
        # Performance tracking
        self._performance_metrics = {}
        self._validation_cache = {} if enable_caching else None
        self._active_validations = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Diagnostics
        self._diagnostics_history = deque(maxlen=1000)
        
        logger.info(f"Production Financial Validator initialized with monitoring={enable_monitoring}, caching={enable_caching}")
    
    def _create_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create validator configuration"""
        default_config = {
            'validation_level': ValidationLevel.COMPREHENSIVE,
            'max_memory_mb': 4096,
            'timeout_seconds': 300,
            'enable_parallel_processing': True,
            'enable_detailed_logging': False,  # Disable in production
            'enable_recovery': True,
            'cache_ttl_seconds': 3600,
            'max_cache_size': 10000,
            'performance_monitoring': True,
            'security_validation': True,
            'compliance_standards': [ComplianceStandard.PCI_DSS, ComplianceStandard.SOX]
        }
        
        if config:
            default_config.update(config)
        
        return default_config
    
    def _init_validators(self):
        """Initialize validation components"""
        try:
            # Enhanced field validator
            self.field_validator = EnhancedTransactionFieldValidator()
            
            # Enhanced business rule validator  
            self.business_rule_validator = EnhancedBusinessRuleValidator()
            
            # Initialize existing validators if available
            if ENHANCED_COMPONENTS:
                try:
                    # Try to use enhanced data validator
                    validation_config = ValidationConfig(
                        validation_level=self.config['validation_level'],
                        context=ValidationContext.PRODUCTION if self.config.get('production_mode', False) else ValidationContext.DEVELOPMENT,
                        mode=ValidationMode.FULL,
                        security_level=SecurityLevel.CONFIDENTIAL,
                        timeout_seconds=self.config['timeout_seconds'],
                        max_memory_mb=self.config['max_memory_mb']
                    )
                    self.enhanced_validator = EnhancedFinancialDataValidator(validation_config)
                except Exception as e:
                    logger.warning(f"Could not initialize enhanced validator: {e}")
                    self.enhanced_validator = None
                
                try:
                    # Fallback to basic validator
                    self.basic_validator = FinancialDataValidator()
                except Exception as e:
                    logger.warning(f"Could not initialize basic validator: {e}")
                    self.basic_validator = None
            
            logger.info("Validation components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize validators: {e}")
            raise
    
    @enhanced_input_validator(
        required_fields=['transaction_id', 'amount', 'timestamp'],
        max_size_mb=1024,
        min_records=1
    )
    @enhanced_error_handler(recovery_strategy='partial', max_retries=2)
    @enhanced_performance_monitor('validate_financial_data')
    @validation_context_manager(context_type="production_validation")
    def validate_financial_data(self, 
                               data: pd.DataFrame,
                               validation_context: Optional[ValidationContext] = None,
                               validation_level: Optional[ValidationLevel] = None,
                               enable_parallel: Optional[bool] = None) -> ValidationResult:
        """
        Comprehensive production validation of financial data
        
        Args:
            data: Financial transaction data to validate
            validation_context: Additional validation context
            validation_level: Override default validation level
            enable_parallel: Enable parallel processing
            
        Returns:
            Comprehensive validation result with diagnostics
        """
        
        validation_id = self._generate_validation_id(data)
        start_time = datetime.now()
        
        # Initialize diagnostics
        diagnostics = ValidationDiagnostics(
            validation_id=validation_id,
            validator_version="3.0.0",
            validation_level=validation_level or self.config['validation_level'],
            start_time=start_time,
            records_processed=len(data)
        )
        
        try:
            with self._lock:
                self._active_validations[validation_id] = diagnostics
            
            # Check cache first
            cache_key = None
            if self.enable_caching:
                cache_key = self._generate_cache_key(data, validation_level)
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    logger.debug(f"Using cached validation result for {validation_id}")
                    return cached_result
            
            # Record metrics
            self.metrics_collector.record_validation(True, len(data), 0, 0.0, "financial_data")
            
            # Perform validation
            issues = []
            
            # Field validation
            field_issues = self.field_validator.validate_transaction_fields(data)
            issues.extend(field_issues)
            
            # Business rule validation
            if validation_context:
                business_issues = self.business_rule_validator.validate_velocity_limits(data, validation_context)
                issues.extend(business_issues)
                
                fraud_issues = self.business_rule_validator.detect_fraud_patterns(data)
                issues.extend(fraud_issues)
            
            # Enhanced validation if available
            if self.enhanced_validator and ENHANCED_COMPONENTS:
                try:
                    enhanced_result = self.enhanced_validator.validate(
                        data, 
                        validation_level=validation_level or self.config['validation_level'],
                        required_compliance=self.config['compliance_standards']
                    )
                    issues.extend(enhanced_result.issues)
                except Exception as e:
                    logger.warning(f"Enhanced validation failed: {e}")
            
            # Calculate validation metrics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Determine validation success
            critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
            error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
            is_valid = len(critical_issues) == 0 and len(error_issues) == 0
            
            # Calculate quality score
            total_weight = len(data) * 100  # Base weight
            issue_weight = sum(self._get_issue_weight(issue) for issue in issues)
            quality_score = max(0, (total_weight - issue_weight) / total_weight) if total_weight > 0 else 0
            
            # Create result
            result = ValidationResult(
                is_valid=is_valid,
                issues=issues,
                data_quality_score=quality_score,
                validation_level=validation_level or self.config['validation_level'],
                compliance_passed=len(critical_issues) == 0,
                timestamp=end_time,
                processing_time=duration,
                validator_version="3.0.0",
                metadata={
                    'validation_id': validation_id,
                    'records_processed': len(data),
                    'field_validation_issues': len(field_issues),
                    'business_rule_issues': len([i for i in issues if 'business' in i.category]),
                    'fraud_indicators': len([i for i in issues if 'fraud' in i.category]),
                    'validation_context': asdict(validation_context) if validation_context else None,
                    'performance_metrics': self._get_performance_metrics(validation_id)
                }
            )
            
            # Update diagnostics
            diagnostics.end_time = end_time
            diagnostics.duration_seconds = duration
            diagnostics.issues_found = len(issues)
            diagnostics.success = is_valid
            diagnostics.memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
            diagnostics.cpu_usage_percent = psutil.cpu_percent()
            
            # Cache result if successful
            if self.enable_caching and cache_key and is_valid:
                self._cache_result(cache_key, result)
            
            # Record monitoring metrics
            if self.monitor:
                self.monitor.record_validation_metrics(diagnostics)
            
            # Update metrics collector
            self.metrics_collector.record_validation(is_valid, len(data), len(issues), duration)
            
            logger.info(f"Validation {validation_id} completed: valid={is_valid}, issues={len(issues)}, duration={duration:.3f}s")
            
            return result
            
        except Exception as e:
            # Update diagnostics with error
            diagnostics.end_time = datetime.now()
            diagnostics.duration_seconds = (diagnostics.end_time - start_time).total_seconds()
            diagnostics.success = False
            diagnostics.error_message = str(e)
            
            # Record error metrics
            self.metrics_collector.record_error(type(e).__name__, recoverable=True)
            
            logger.error(f"Validation {validation_id} failed: {e}")
            raise
            
        finally:
            # Store diagnostics
            self._diagnostics_history.append(diagnostics)
            
            # Remove from active validations
            with self._lock:
                self._active_validations.pop(validation_id, None)
    
    def _generate_validation_id(self, data: pd.DataFrame) -> str:
        """Generate unique validation ID"""
        content = f"{len(data)}_{data.columns.tolist()}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _generate_cache_key(self, data: pd.DataFrame, validation_level: Optional[ValidationLevel]) -> str:
        """Generate cache key for validation"""
        data_hash = hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()
        level_str = validation_level.value if validation_level else "default"
        return f"validation_{data_hash}_{level_str}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[ValidationResult]:
        """Get cached validation result"""
        if not self._validation_cache or cache_key not in self._validation_cache:
            return None
        
        cached_entry = self._validation_cache[cache_key]
        
        # Check TTL
        if time.time() - cached_entry['timestamp'] > self.config['cache_ttl_seconds']:
            del self._validation_cache[cache_key]
            return None
        
        return cached_entry['result']
    
    def _cache_result(self, cache_key: str, result: ValidationResult):
        """Cache validation result"""
        if not self._validation_cache:
            return
        
        # Implement LRU eviction
        if len(self._validation_cache) >= self.config['max_cache_size']:
            # Remove oldest entry
            oldest_key = min(self._validation_cache.keys(), 
                           key=lambda k: self._validation_cache[k]['timestamp'])
            del self._validation_cache[oldest_key]
        
        self._validation_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
    
    def _get_issue_weight(self, issue: ValidationIssue) -> float:
        """Calculate weight for validation issue"""
        weights = {
            ValidationSeverity.CRITICAL: 50,
            ValidationSeverity.ERROR: 20,
            ValidationSeverity.WARNING: 5,
            ValidationSeverity.INFO: 1
        }
        return weights.get(issue.severity, 1)
    
    def _get_performance_metrics(self, validation_id: str) -> Dict[str, Any]:
        """Get performance metrics for validation"""
        return self._performance_metrics.get(validation_id, {})
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health status"""
        if self.monitor:
            return self.monitor._check_system_health()
        else:
            # Basic health check without monitoring
            return SystemHealth(
                status='healthy',
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                memory_limit=psutil.virtual_memory().total / 1024 / 1024 / 1024,
                active_validations=len(self._active_validations),
                error_rate=0.0,
                response_time=0.0,
                uptime_seconds=0.0
            )
    
    def get_performance_summary(self) -> PerformanceMetrics:
        """Get comprehensive performance summary"""
        if self.monitor:
            return self.monitor.get_performance_summary()
        else:
            return self.metrics_collector.get_summary()
    
    def get_diagnostics(self, validation_id: Optional[str] = None) -> Union[ValidationDiagnostics, List[ValidationDiagnostics]]:
        """Get validation diagnostics"""
        if validation_id:
            # Return specific validation diagnostics
            for diag in self._diagnostics_history:
                if diag.validation_id == validation_id:
                    return diag
            return None
        else:
            # Return all recent diagnostics
            return list(self._diagnostics_history)
    
    def get_active_validations(self) -> Dict[str, ValidationDiagnostics]:
        """Get currently active validations"""
        with self._lock:
            return self._active_validations.copy()
    
    def clear_cache(self) -> bool:
        """Clear validation cache"""
        if self._validation_cache:
            self._validation_cache.clear()
            logger.info("Validation cache cleared")
            return True
        return False
    
    def optimize_performance(self):
        """Perform performance optimization"""
        try:
            # Clear old cache entries
            if self._validation_cache:
                current_time = time.time()
                ttl = self.config['cache_ttl_seconds']
                expired_keys = [
                    key for key, entry in self._validation_cache.items()
                    if current_time - entry['timestamp'] > ttl
                ]
                for key in expired_keys:
                    del self._validation_cache[key]
            
            # Garbage collection
            gc.collect()
            
            # Log optimization
            logger.info("Performance optimization completed")
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
    
    def shutdown(self):
        """Graceful shutdown of validator"""
        logger.info("Shutting down Production Financial Validator")
        
        try:
            # Shutdown monitoring
            if self.monitor:
                self.monitor.shutdown()
            
            # Shutdown thread pool
            self._thread_pool.shutdown(wait=True, timeout=30)
            
            # Final metrics
            final_metrics = self.get_performance_summary()
            logger.info(f"Final validation metrics: {asdict(final_metrics)}")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()

# ======================== CONVENIENCE FUNCTIONS ========================

def create_production_validator(config: Optional[Dict[str, Any]] = None,
                               monitoring: bool = True,
                               caching: bool = True) -> ProductionFinancialValidator:
    """Create production-ready financial validator"""
    return ProductionFinancialValidator(
        config=config,
        enable_monitoring=monitoring,
        enable_caching=caching
    )

def validate_financial_transactions(data: pd.DataFrame,
                                  config: Optional[Dict[str, Any]] = None,
                                  validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE) -> ValidationResult:
    """Quick validation function for financial transactions"""
    with create_production_validator(config) as validator:
        return validator.validate_financial_data(data, validation_level=validation_level)

# Export production components
__all__ = [
    # Classes
    'SystemHealth',
    'ValidationDiagnostics', 
    'PerformanceMetrics',
    'ProductionMonitor',
    'ProductionFinancialValidator',
    
    # Functions
    'create_production_validator',
    'validate_financial_transactions'
]

if __name__ == "__main__":
    print("Production Validator - Chunk 3: Production Validator with Monitoring loaded")