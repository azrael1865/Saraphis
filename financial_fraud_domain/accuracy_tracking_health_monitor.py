"""
Accuracy Tracking Health Monitor - Phase 5C-1A
Basic health monitoring system for accuracy tracking components
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning" 
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class ComponentType(Enum):
    """Types of components to monitor"""
    DATASET_MANAGER = "dataset_manager"
    DATABASE = "database"
    EVALUATION_ENGINE = "evaluation_engine"
    MONITORING = "monitoring"
    ORCHESTRATOR = "orchestrator"
    API = "api"

@dataclass
class HealthMetrics:
    """Health metrics for a component"""
    status: HealthStatus
    response_time_ms: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    last_check: datetime
    uptime_seconds: float
    error_count: int = 0
    warning_count: int = 0

@dataclass
class HealthAlert:
    """Health alert information"""
    component_name: str
    component_type: ComponentType
    severity: HealthStatus
    message: str
    timestamp: datetime
    resolved: bool = False

class AccuracyTrackingHealthMonitor:
    """
    Basic health monitoring system for accuracy tracking components
    Provides real-time status information and health metrics collection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize health monitor
        
        Args:
            config: Configuration dictionary with monitoring settings
        """
        self.config = config or self._get_default_config()
        self.components: Dict[str, Dict[str, Any]] = {}
        self.health_metrics: Dict[str, HealthMetrics] = {}
        self.alerts: List[HealthAlert] = []
        self.change_callbacks: List[Callable[[str, HealthStatus], None]] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Monitoring thread
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        # Configuration
        self.check_interval = self.config.get('check_interval_seconds', 30)
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'response_time_ms': 5000,
            'error_rate': 0.05,
            'cpu_usage': 80.0,
            'memory_usage': 85.0
        })
        
        logger.info("AccuracyTrackingHealthMonitor initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'check_interval_seconds': 30,
            'alert_thresholds': {
                'response_time_ms': 5000,
                'error_rate': 0.05,
                'cpu_usage': 80.0,
                'memory_usage': 85.0
            },
            'enable_alerts': True,
            'max_alerts': 1000,
            'health_history_retention_hours': 24
        }
    
    def register_component(self, 
                          name: str, 
                          component_type: ComponentType,
                          health_check_func: Optional[Callable[[], Dict[str, Any]]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a component for health monitoring
        
        Args:
            name: Component name
            component_type: Type of component
            health_check_func: Function to check component health
            metadata: Additional component metadata
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                self.components[name] = {
                    'type': component_type,
                    'health_check_func': health_check_func,
                    'metadata': metadata or {},
                    'registered_at': datetime.now(),
                    'enabled': True
                }
                
                # Initialize health metrics
                self.health_metrics[name] = HealthMetrics(
                    status=HealthStatus.UNKNOWN,
                    response_time_ms=0.0,
                    error_rate=0.0,
                    cpu_usage=0.0,
                    memory_usage=0.0,
                    last_check=datetime.now(),
                    uptime_seconds=0.0
                )
                
                logger.info(f"Registered component for health monitoring: {name} ({component_type.value})")
                return True
                
            except Exception as e:
                logger.error(f"Failed to register component {name}: {e}")
                return False
    
    def unregister_component(self, name: str) -> bool:
        """
        Unregister a component from health monitoring
        
        Args:
            name: Component name
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                if name in self.components:
                    del self.components[name]
                    if name in self.health_metrics:
                        del self.health_metrics[name]
                    logger.info(f"Unregistered component: {name}")
                    return True
                return False
            except Exception as e:
                logger.error(f"Failed to unregister component {name}: {e}")
                return False
    
    def check_component_health(self, name: str) -> Optional[HealthMetrics]:
        """
        Check health of a specific component
        
        Args:
            name: Component name
            
        Returns:
            HealthMetrics if successful, None otherwise
        """
        if name not in self.components:
            logger.warning(f"Component not registered: {name}")
            return None
        
        component = self.components[name]
        if not component.get('enabled', True):
            return self.health_metrics.get(name)
        
        try:
            start_time = time.time()
            
            # Use custom health check function if available
            if component.get('health_check_func'):
                health_data = component['health_check_func']()
            else:
                # Default health check
                health_data = self._default_health_check(name, component)
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Update metrics
            metrics = HealthMetrics(
                status=self._determine_health_status(health_data, response_time),
                response_time_ms=response_time,
                error_rate=health_data.get('error_rate', 0.0),
                cpu_usage=health_data.get('cpu_usage', 0.0),
                memory_usage=health_data.get('memory_usage', 0.0),
                last_check=datetime.now(),
                uptime_seconds=health_data.get('uptime_seconds', 0.0),
                error_count=health_data.get('error_count', 0),
                warning_count=health_data.get('warning_count', 0)
            )
            
            # Store metrics
            with self._lock:
                old_status = self.health_metrics.get(name, metrics).status if name in self.health_metrics else HealthStatus.UNKNOWN
                self.health_metrics[name] = metrics
                
                # Check for status changes and generate alerts
                if old_status != metrics.status:
                    self._handle_status_change(name, old_status, metrics.status)
                
                # Check thresholds and generate alerts
                self._check_alert_thresholds(name, metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Health check failed for component {name}: {e}")
            
            # Update with error status
            error_metrics = HealthMetrics(
                status=HealthStatus.CRITICAL,
                response_time_ms=0.0,
                error_rate=1.0,
                cpu_usage=0.0,
                memory_usage=0.0,
                last_check=datetime.now(),
                uptime_seconds=0.0,
                error_count=1
            )
            
            with self._lock:
                self.health_metrics[name] = error_metrics
            
            return error_metrics
    
    def _default_health_check(self, name: str, component: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default health check implementation
        
        Args:
            name: Component name
            component: Component configuration
            
        Returns:
            Health check results
        """
        # Basic health check - just return defaults
        # In a real implementation, this would check actual component status
        return {
            'status': 'healthy',
            'error_rate': 0.0,
            'cpu_usage': 10.0,
            'memory_usage': 25.0,
            'uptime_seconds': time.time() - component.get('registered_at', datetime.now()).timestamp(),
            'error_count': 0,
            'warning_count': 0
        }
    
    def _determine_health_status(self, health_data: Dict[str, Any], response_time: float) -> HealthStatus:
        """
        Determine health status based on health data
        
        Args:
            health_data: Health check results
            response_time: Response time in milliseconds
            
        Returns:
            HealthStatus
        """
        # Check critical conditions
        if (health_data.get('error_rate', 0.0) > self.alert_thresholds['error_rate'] or
            response_time > self.alert_thresholds['response_time_ms'] or
            health_data.get('cpu_usage', 0.0) > self.alert_thresholds['cpu_usage'] or
            health_data.get('memory_usage', 0.0) > self.alert_thresholds['memory_usage']):
            return HealthStatus.CRITICAL
        
        # Check warning conditions
        if (health_data.get('error_rate', 0.0) > self.alert_thresholds['error_rate'] * 0.5 or
            response_time > self.alert_thresholds['response_time_ms'] * 0.7 or
            health_data.get('cpu_usage', 0.0) > self.alert_thresholds['cpu_usage'] * 0.8 or
            health_data.get('memory_usage', 0.0) > self.alert_thresholds['memory_usage'] * 0.8):
            return HealthStatus.WARNING
        
        return HealthStatus.HEALTHY
    
    def _handle_status_change(self, name: str, old_status: HealthStatus, new_status: HealthStatus):
        """Handle component status changes"""
        logger.info(f"Component {name} status changed: {old_status.value} -> {new_status.value}")
        
        # Notify callbacks
        for callback in self.change_callbacks:
            try:
                callback(name, new_status)
            except Exception as e:
                logger.error(f"Error in status change callback: {e}")
    
    def _check_alert_thresholds(self, name: str, metrics: HealthMetrics):
        """Check if metrics exceed alert thresholds"""
        if not self.config.get('enable_alerts', True):
            return
        
        alerts_to_create = []
        
        if metrics.response_time_ms > self.alert_thresholds['response_time_ms']:
            alerts_to_create.append(f"High response time: {metrics.response_time_ms:.2f}ms")
        
        if metrics.error_rate > self.alert_thresholds['error_rate']:
            alerts_to_create.append(f"High error rate: {metrics.error_rate:.2%}")
        
        if metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
            alerts_to_create.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
        
        if metrics.memory_usage > self.alert_thresholds['memory_usage']:
            alerts_to_create.append(f"High memory usage: {metrics.memory_usage:.1f}%")
        
        # Create alerts
        component_type = self.components[name]['type']
        for alert_message in alerts_to_create:
            self._create_alert(name, component_type, metrics.status, alert_message)
    
    def _create_alert(self, component_name: str, component_type: ComponentType, 
                     severity: HealthStatus, message: str):
        """Create a health alert"""
        alert = HealthAlert(
            component_name=component_name,
            component_type=component_type,
            severity=severity,
            message=message,
            timestamp=datetime.now()
        )
        
        with self._lock:
            self.alerts.append(alert)
            
            # Limit alert history
            max_alerts = self.config.get('max_alerts', 1000)
            if len(self.alerts) > max_alerts:
                self.alerts = self.alerts[-max_alerts:]
        
        logger.warning(f"Health alert created: {component_name} - {message}")
    
    def get_component_health(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get health information for a component
        
        Args:
            name: Component name
            
        Returns:
            Health information dictionary or None
        """
        if name not in self.health_metrics:
            return None
        
        metrics = self.health_metrics[name]
        component = self.components[name]
        
        return {
            'name': name,
            'type': component['type'].value,
            'status': metrics.status.value,
            'metrics': asdict(metrics),
            'metadata': component.get('metadata', {}),
            'enabled': component.get('enabled', True)
        }
    
    def get_overall_health(self) -> Dict[str, Any]:
        """
        Get overall system health summary
        
        Returns:
            Overall health information
        """
        with self._lock:
            if not self.health_metrics:
                return {
                    'status': HealthStatus.UNKNOWN.value,
                    'component_count': 0,
                    'healthy_components': 0,
                    'warning_components': 0,
                    'critical_components': 0,
                    'last_updated': datetime.now().isoformat()
                }
            
            status_counts = {
                HealthStatus.HEALTHY: 0,
                HealthStatus.WARNING: 0,
                HealthStatus.CRITICAL: 0,
                HealthStatus.UNKNOWN: 0
            }
            
            for metrics in self.health_metrics.values():
                status_counts[metrics.status] += 1
            
            # Determine overall status
            if status_counts[HealthStatus.CRITICAL] > 0:
                overall_status = HealthStatus.CRITICAL
            elif status_counts[HealthStatus.WARNING] > 0:
                overall_status = HealthStatus.WARNING
            elif status_counts[HealthStatus.UNKNOWN] > 0:
                overall_status = HealthStatus.UNKNOWN
            else:
                overall_status = HealthStatus.HEALTHY
            
            return {
                'status': overall_status.value,
                'component_count': len(self.health_metrics),
                'healthy_components': status_counts[HealthStatus.HEALTHY],
                'warning_components': status_counts[HealthStatus.WARNING],
                'critical_components': status_counts[HealthStatus.CRITICAL],
                'unknown_components': status_counts[HealthStatus.UNKNOWN],
                'active_alerts': len([a for a in self.alerts if not a.resolved]),
                'last_updated': datetime.now().isoformat()
            }
    
    def get_all_component_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Get health information for all components
        
        Returns:
            Dictionary mapping component names to health information
        """
        result = {}
        for name in self.components:
            health_info = self.get_component_health(name)
            if health_info:
                result[name] = health_info
        return result
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        Get all active (unresolved) alerts
        
        Returns:
            List of active alerts
        """
        with self._lock:
            return [asdict(alert) for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_index: int) -> bool:
        """
        Resolve an alert by index
        
        Args:
            alert_index: Index of alert to resolve
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                if 0 <= alert_index < len(self.alerts):
                    self.alerts[alert_index].resolved = True
                    logger.info(f"Resolved alert: {self.alerts[alert_index].message}")
                    return True
                return False
            except Exception as e:
                logger.error(f"Failed to resolve alert {alert_index}: {e}")
                return False
    
    def add_status_change_callback(self, callback: Callable[[str, HealthStatus], None]):
        """Add callback for component status changes"""
        self.change_callbacks.append(callback)
    
    def remove_status_change_callback(self, callback: Callable[[str, HealthStatus], None]):
        """Remove status change callback"""
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)
    
    def start_monitoring(self):
        """Start continuous health monitoring"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Monitoring already started")
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("Started continuous health monitoring")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring"""
        if self._monitoring_thread:
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5)
            logger.info("Stopped continuous health monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self._stop_monitoring.wait(self.check_interval):
            try:
                # Check all registered components
                for name in list(self.components.keys()):
                    if self._stop_monitoring.is_set():
                        break
                    self.check_component_health(name)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def export_health_data(self) -> str:
        """
        Export health data as JSON
        
        Returns:
            JSON string containing all health data
        """
        export_data = {
            'overall_health': self.get_overall_health(),
            'components': self.get_all_component_health(),
            'active_alerts': self.get_active_alerts(),
            'exported_at': datetime.now().isoformat()
        }
        return json.dumps(export_data, indent=2, default=str)
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_monitoring()
        logger.info("AccuracyTrackingHealthMonitor cleanup completed")

def create_health_monitor(config: Optional[Dict[str, Any]] = None) -> AccuracyTrackingHealthMonitor:
    """
    Create health monitor with default configuration
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured AccuracyTrackingHealthMonitor instance
    """
    return AccuracyTrackingHealthMonitor(config)

if __name__ == "__main__":
    # Example usage
    monitor = create_health_monitor()
    
    # Register some example components
    monitor.register_component("dataset_manager", ComponentType.DATASET_MANAGER)
    monitor.register_component("fraud_database", ComponentType.DATABASE)
    monitor.register_component("orchestrator", ComponentType.ORCHESTRATOR)
    
    # Start monitoring
    monitor.start_monitoring()
    
    print("Health monitor started. Checking component health...")
    
    # Check health manually
    for name in ["dataset_manager", "fraud_database", "orchestrator"]:
        health = monitor.check_component_health(name)
        if health:
            print(f"{name}: {health.status.value}")
    
    # Get overall health
    overall = monitor.get_overall_health()
    print(f"\nOverall health: {overall['status']}")
    print(f"Components: {overall['component_count']}")
    
    # Cleanup
    monitor.cleanup()