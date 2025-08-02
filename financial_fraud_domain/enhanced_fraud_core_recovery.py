"""
Enhanced Fraud Detection Core - Chunk 3: Error Recovery and Circuit Breaker System
Advanced error recovery mechanisms and circuit breaker patterns for resilient fraud detection
"""

import logging
import time
import threading
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
import traceback
import random
import json
from functools import wraps
from collections import deque, defaultdict
import weakref

# Import core exceptions and enums
from enhanced_fraud_core_exceptions import (
    EnhancedFraudException, ValidationError, ConfigurationError, SecurityError,
    PerformanceError, ResourceError, DetectionError, CircuitBreakerError,
    RecoveryStrategy, AlertSeverity, ErrorContext, create_error_context, log_exception
)

# Configure logging
logger = logging.getLogger(__name__)

# ======================== CIRCUIT BREAKER CONFIGURATION ========================

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    expected_exception: type = Exception
    success_threshold: int = 3  # for half-open state
    timeout: float = 30.0  # operation timeout
    
    # Monitoring
    enable_monitoring: bool = True
    alert_on_open: bool = True
    alert_on_recovery: bool = True
    
    # Fallback
    fallback_function: Optional[Callable] = None
    enable_fallback: bool = True

# ======================== ERROR RECOVERY CONFIGURATION ========================

@dataclass
class ErrorRecoveryConfig:
    """Configuration for error recovery"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    
    # Recovery strategies
    default_strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    strategy_mapping: Dict[type, RecoveryStrategy] = field(default_factory=dict)
    
    # Timeouts
    operation_timeout: float = 30.0
    total_timeout: float = 300.0  # 5 minutes
    
    # Circuit breaker integration
    enable_circuit_breaker: bool = True
    circuit_breaker_config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    
    # Monitoring
    enable_recovery_metrics: bool = True
    log_recovery_attempts: bool = True

# ======================== CIRCUIT BREAKER IMPLEMENTATION ========================

class CircuitBreaker:
    """Circuit breaker implementation with monitoring and fallback"""
    
    def __init__(self, config: CircuitBreakerConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.lock = threading.Lock()
        
        # Monitoring
        self.metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'circuit_open_count': 0,
            'recovery_attempts': 0,
            'fallback_calls': 0
        }
        self.failure_history = deque(maxlen=100)
        
        # Callbacks
        self.on_state_change_callbacks = []
        self.on_failure_callbacks = []
        self.on_success_callbacks = []
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self.lock:
            self.metrics['total_calls'] += 1
            
            # Check circuit state
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._set_state(CircuitBreakerState.HALF_OPEN)
                else:
                    return self._handle_open_circuit(func, *args, **kwargs)
            
            # Execute function
            try:
                start_time = time.time()
                result = self._execute_with_timeout(func, *args, **kwargs)
                execution_time = time.time() - start_time
                
                self._handle_success(execution_time)
                return result
                
            except Exception as e:
                self._handle_failure(e)
                raise
    
    def _execute_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with timeout"""
        if self.config.timeout <= 0:
            return func(*args, **kwargs)
        
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(func, *args, **kwargs)
        
        try:
            return future.result(timeout=self.config.timeout)
        except TimeoutError:
            future.cancel()
            raise PerformanceError(f"Operation timed out after {self.config.timeout} seconds")
        finally:
            executor.shutdown(wait=False)
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset"""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.recovery_timeout
    
    def _handle_success(self, execution_time: float) -> None:
        """Handle successful execution"""
        self.metrics['successful_calls'] += 1
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._set_state(CircuitBreakerState.CLOSED)
                self.failure_count = 0
                self.success_count = 0
        
        # Call success callbacks
        for callback in self.on_success_callbacks:
            try:
                callback(execution_time)
            except Exception as e:
                logger.error(f"Error in success callback: {e}")
    
    def _handle_failure(self, exception: Exception) -> None:
        """Handle failed execution"""
        self.metrics['failed_calls'] += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        # Record failure
        self.failure_history.append({
            'timestamp': datetime.now().isoformat(),
            'exception': str(exception),
            'exception_type': type(exception).__name__
        })
        
        # Check if circuit should open
        if (self.state == CircuitBreakerState.CLOSED and 
            self.failure_count >= self.config.failure_threshold):
            self._set_state(CircuitBreakerState.OPEN)
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self._set_state(CircuitBreakerState.OPEN)
            self.success_count = 0
        
        # Call failure callbacks
        for callback in self.on_failure_callbacks:
            try:
                callback(exception)
            except Exception as e:
                logger.error(f"Error in failure callback: {e}")
    
    def _handle_open_circuit(self, func: Callable, *args, **kwargs) -> Any:
        """Handle call when circuit is open"""
        if self.config.enable_fallback and self.config.fallback_function:
            self.metrics['fallback_calls'] += 1
            logger.warning(f"Circuit breaker '{self.name}' is open, using fallback")
            return self.config.fallback_function(*args, **kwargs)
        else:
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is open",
                context=create_error_context(
                    component='CircuitBreaker',
                    operation='call',
                    additional_data={'circuit_name': self.name}
                )
            )
    
    def _set_state(self, new_state: CircuitBreakerState) -> None:
        """Set circuit breaker state"""
        old_state = self.state
        self.state = new_state
        
        if old_state != new_state:
            logger.info(f"Circuit breaker '{self.name}' state changed: {old_state.value} -> {new_state.value}")
            
            if new_state == CircuitBreakerState.OPEN:
                self.metrics['circuit_open_count'] += 1
                if self.config.alert_on_open:
                    self._send_alert(AlertSeverity.ERROR, f"Circuit breaker '{self.name}' opened")
            elif new_state == CircuitBreakerState.CLOSED and old_state == CircuitBreakerState.HALF_OPEN:
                if self.config.alert_on_recovery:
                    self._send_alert(AlertSeverity.INFO, f"Circuit breaker '{self.name}' recovered")
            
            # Call state change callbacks
            for callback in self.on_state_change_callbacks:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    logger.error(f"Error in state change callback: {e}")
    
    def _send_alert(self, severity: AlertSeverity, message: str) -> None:
        """Send alert for circuit breaker events"""
        logger.log(
            logging.ERROR if severity == AlertSeverity.ERROR else logging.INFO,
            f"Circuit Breaker Alert: {message}",
            extra={
                'circuit_name': self.name,
                'circuit_state': self.state.value,
                'failure_count': self.failure_count,
                'metrics': self.metrics
            }
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        with self.lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'metrics': self.metrics.copy(),
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'recovery_timeout': self.config.recovery_timeout,
                    'success_threshold': self.config.success_threshold
                }
            }
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state"""
        with self.lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            logger.info(f"Circuit breaker '{self.name}' manually reset")
    
    def add_callback(self, callback_type: str, callback: Callable) -> None:
        """Add callback for circuit breaker events"""
        if callback_type == 'state_change':
            self.on_state_change_callbacks.append(callback)
        elif callback_type == 'failure':
            self.on_failure_callbacks.append(callback)
        elif callback_type == 'success':
            self.on_success_callbacks.append(callback)
        else:
            raise ValueError(f"Invalid callback type: {callback_type}")

# ======================== ERROR RECOVERY MANAGER ========================

class ErrorRecoveryManager:
    """Comprehensive error recovery manager"""
    
    def __init__(self, config: ErrorRecoveryConfig):
        self.config = config
        self.circuit_breakers = {}
        self.recovery_metrics = defaultdict(lambda: {
            'total_attempts': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'average_recovery_time': 0.0,
            'strategy_usage': defaultdict(int)
        })
        self.active_recoveries = {}
        self.lock = threading.Lock()
        
        # Recovery strategies
        self.strategy_handlers = {
            RecoveryStrategy.RETRY: self._handle_retry,
            RecoveryStrategy.FALLBACK: self._handle_fallback,
            RecoveryStrategy.CIRCUIT_BREAKER: self._handle_circuit_breaker,
            RecoveryStrategy.FAIL_FAST: self._handle_fail_fast,
            RecoveryStrategy.GRACEFUL_DEGRADATION: self._handle_graceful_degradation
        }
    
    def recover(self, func: Callable, exception: Exception, context: Optional[ErrorContext] = None, 
                *args, **kwargs) -> Any:
        """Recover from error using appropriate strategy"""
        operation_id = f"{func.__name__}_{id(exception)}"
        
        with self.lock:
            if operation_id in self.active_recoveries:
                logger.warning(f"Recovery already in progress for {operation_id}")
                return self.active_recoveries[operation_id]
        
        start_time = time.time()
        strategy = self._determine_strategy(exception)
        
        try:
            self._record_recovery_attempt(func.__name__, strategy)
            
            # Execute recovery strategy
            result = self.strategy_handlers[strategy](
                func, exception, context, *args, **kwargs
            )
            
            # Record success
            recovery_time = time.time() - start_time
            self._record_recovery_success(func.__name__, strategy, recovery_time)
            
            return result
            
        except Exception as recovery_exception:
            recovery_time = time.time() - start_time
            self._record_recovery_failure(func.__name__, strategy, recovery_time, recovery_exception)
            raise
        finally:
            with self.lock:
                self.active_recoveries.pop(operation_id, None)
    
    def _determine_strategy(self, exception: Exception) -> RecoveryStrategy:
        """Determine appropriate recovery strategy"""
        # Check explicit mapping
        for exception_type, strategy in self.config.strategy_mapping.items():
            if isinstance(exception, exception_type):
                return strategy
        
        # Default strategy based on exception type
        if isinstance(exception, (TimeoutError, PerformanceError)):
            return RecoveryStrategy.RETRY
        elif isinstance(exception, (SecurityError, ValidationError)):
            return RecoveryStrategy.FAIL_FAST
        elif isinstance(exception, ResourceError):
            return RecoveryStrategy.GRACEFUL_DEGRADATION
        elif isinstance(exception, CircuitBreakerError):
            return RecoveryStrategy.FALLBACK
        else:
            return self.config.default_strategy
    
    def _handle_retry(self, func: Callable, exception: Exception, context: Optional[ErrorContext] = None,
                     *args, **kwargs) -> Any:
        """Handle retry recovery strategy"""
        last_exception = exception
        
        for attempt in range(self.config.max_retries + 1):
            if attempt > 0:
                delay = self._calculate_delay(attempt)
                logger.info(f"Retry attempt {attempt} after {delay:.2f}s delay")
                time.sleep(delay)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt == self.config.max_retries:
                    logger.error(f"Retry failed after {self.config.max_retries} attempts")
                    break
                else:
                    logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
        
        # All retries failed
        raise DetectionError(
            f"Operation failed after {self.config.max_retries} retry attempts",
            context=context or create_error_context(
                component='ErrorRecoveryManager',
                operation='retry',
                additional_data={'original_exception': str(last_exception)}
            )
        )
    
    def _handle_fallback(self, func: Callable, exception: Exception, context: Optional[ErrorContext] = None,
                        *args, **kwargs) -> Any:
        """Handle fallback recovery strategy"""
        # Try to find a fallback function
        fallback_func = getattr(func, '_fallback_function', None)
        if fallback_func:
            logger.info(f"Using fallback function for {func.__name__}")
            return fallback_func(*args, **kwargs)
        
        # Default fallback behavior
        logger.warning(f"No fallback function available for {func.__name__}, returning default result")
        return self._get_default_result(func, exception)
    
    def _handle_circuit_breaker(self, func: Callable, exception: Exception, context: Optional[ErrorContext] = None,
                               *args, **kwargs) -> Any:
        """Handle circuit breaker recovery strategy"""
        circuit_name = f"{func.__name__}_circuit"
        
        if circuit_name not in self.circuit_breakers:
            self.circuit_breakers[circuit_name] = CircuitBreaker(
                self.config.circuit_breaker_config,
                circuit_name
            )
        
        circuit = self.circuit_breakers[circuit_name]
        return circuit.call(func, *args, **kwargs)
    
    def _handle_fail_fast(self, func: Callable, exception: Exception, context: Optional[ErrorContext] = None,
                         *args, **kwargs) -> Any:
        """Handle fail fast recovery strategy"""
        logger.error(f"Fail fast strategy triggered for {func.__name__}: {exception}")
        raise exception
    
    def _handle_graceful_degradation(self, func: Callable, exception: Exception, context: Optional[ErrorContext] = None,
                                   *args, **kwargs) -> Any:
        """Handle graceful degradation recovery strategy"""
        logger.warning(f"Graceful degradation for {func.__name__}: {exception}")
        
        # Try to provide a degraded version of the service
        degraded_result = self._get_degraded_result(func, exception)
        
        # Log degradation
        logger.info(f"Returning degraded result for {func.__name__}")
        return degraded_result
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry with exponential backoff"""
        delay = min(
            self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1)),
            self.config.max_delay
        )
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    def _get_default_result(self, func: Callable, exception: Exception) -> Any:
        """Get default result for fallback"""
        # This would be customized based on the function type
        if 'detect' in func.__name__.lower():
            return {
                'fraud_detected': False,
                'confidence': 0.0,
                'reason': 'Fallback result due to error',
                'error': str(exception)
            }
        else:
            return None
    
    def _get_degraded_result(self, func: Callable, exception: Exception) -> Any:
        """Get degraded result for graceful degradation"""
        # This would be customized based on the function type
        if 'detect' in func.__name__.lower():
            return {
                'fraud_detected': False,
                'confidence': 0.0,
                'reason': 'Degraded service due to error',
                'error': str(exception),
                'degraded': True
            }
        else:
            return {'degraded': True, 'error': str(exception)}
    
    def _record_recovery_attempt(self, func_name: str, strategy: RecoveryStrategy) -> None:
        """Record recovery attempt"""
        if not self.config.enable_recovery_metrics:
            return
        
        with self.lock:
            metrics = self.recovery_metrics[func_name]
            metrics['total_attempts'] += 1
            metrics['strategy_usage'][strategy.value] += 1
        
        if self.config.log_recovery_attempts:
            logger.info(f"Recovery attempt for {func_name} using {strategy.value} strategy")
    
    def _record_recovery_success(self, func_name: str, strategy: RecoveryStrategy, recovery_time: float) -> None:
        """Record successful recovery"""
        if not self.config.enable_recovery_metrics:
            return
        
        with self.lock:
            metrics = self.recovery_metrics[func_name]
            metrics['successful_recoveries'] += 1
            
            # Update average recovery time
            total_successes = metrics['successful_recoveries']
            current_avg = metrics['average_recovery_time']
            metrics['average_recovery_time'] = ((current_avg * (total_successes - 1)) + recovery_time) / total_successes
        
        logger.info(f"Successful recovery for {func_name} using {strategy.value} in {recovery_time:.2f}s")
    
    def _record_recovery_failure(self, func_name: str, strategy: RecoveryStrategy, recovery_time: float, 
                                exception: Exception) -> None:
        """Record failed recovery"""
        if not self.config.enable_recovery_metrics:
            return
        
        with self.lock:
            metrics = self.recovery_metrics[func_name]
            metrics['failed_recoveries'] += 1
        
        logger.error(f"Failed recovery for {func_name} using {strategy.value} after {recovery_time:.2f}s: {exception}")
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)
    
    def get_recovery_metrics(self) -> Dict[str, Any]:
        """Get recovery metrics"""
        with self.lock:
            return {
                'recovery_metrics': dict(self.recovery_metrics),
                'circuit_breakers': {
                    name: cb.get_metrics() for name, cb in self.circuit_breakers.items()
                },
                'config': {
                    'max_retries': self.config.max_retries,
                    'base_delay': self.config.base_delay,
                    'max_delay': self.config.max_delay,
                    'default_strategy': self.config.default_strategy.value
                }
            }
    
    def reset_all_circuit_breakers(self) -> None:
        """Reset all circuit breakers"""
        for circuit in self.circuit_breakers.values():
            circuit.reset()
        logger.info("All circuit breakers reset")
    
    def add_strategy_mapping(self, exception_type: type, strategy: RecoveryStrategy) -> None:
        """Add custom strategy mapping"""
        self.config.strategy_mapping[exception_type] = strategy
        logger.info(f"Added strategy mapping: {exception_type.__name__} -> {strategy.value}")

# ======================== RECOVERY DECORATORS ========================

def with_recovery(recovery_manager: ErrorRecoveryManager, context: Optional[ErrorContext] = None):
    """Decorator to add error recovery to functions"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return recovery_manager.recover(func, e, context, *args, **kwargs)
        return wrapper
    return decorator

def with_circuit_breaker(config: CircuitBreakerConfig = None, name: str = None):
    """Decorator to add circuit breaker to functions"""
    def decorator(func: Callable) -> Callable:
        circuit_name = name or f"{func.__name__}_circuit"
        circuit_config = config or CircuitBreakerConfig()
        circuit = CircuitBreaker(circuit_config, circuit_name)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return circuit.call(func, *args, **kwargs)
        
        # Add circuit breaker as attribute for external access
        wrapper._circuit_breaker = circuit
        return wrapper
    return decorator

def with_retry(max_retries: int = 3, base_delay: float = 1.0, backoff_multiplier: float = 2.0):
    """Decorator to add retry logic to functions"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    delay = base_delay * (backoff_multiplier ** attempt)
                    time.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator

# ======================== RECOVERY UTILITIES ========================

def create_default_recovery_manager() -> ErrorRecoveryManager:
    """Create default error recovery manager"""
    config = ErrorRecoveryConfig()
    return ErrorRecoveryManager(config)

def create_circuit_breaker_for_function(func: Callable, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """Create circuit breaker for specific function"""
    circuit_config = config or CircuitBreakerConfig()
    circuit_name = f"{func.__name__}_circuit"
    return CircuitBreaker(circuit_config, circuit_name)

def get_recovery_statistics(recovery_manager: ErrorRecoveryManager) -> Dict[str, Any]:
    """Get comprehensive recovery statistics"""
    metrics = recovery_manager.get_recovery_metrics()
    
    # Calculate aggregate statistics
    total_attempts = sum(m['total_attempts'] for m in metrics['recovery_metrics'].values())
    total_successes = sum(m['successful_recoveries'] for m in metrics['recovery_metrics'].values())
    total_failures = sum(m['failed_recoveries'] for m in metrics['recovery_metrics'].values())
    
    success_rate = (total_successes / total_attempts) if total_attempts > 0 else 0
    
    return {
        'aggregate_stats': {
            'total_attempts': total_attempts,
            'total_successes': total_successes,
            'total_failures': total_failures,
            'success_rate': success_rate
        },
        'detailed_metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }