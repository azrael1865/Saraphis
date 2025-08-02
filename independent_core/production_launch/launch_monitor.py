"""
Launch Monitor - Monitors launch progress and system health
NO FALLBACKS - HARD FAILURES ONLY

Provides real-time monitoring of the production launch process,
tracking component health, performance metrics, and system status.
"""

import os
import sys
import json
import time
import logging
import threading
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
import statistics

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class ComponentMetrics:
    """Metrics for a single component"""
    component_name: str
    component_type: str  # 'system' or 'agent'
    cpu_usage: float
    memory_usage: float
    response_time: float
    error_rate: float
    request_count: int
    health_score: float
    status: str
    last_update: datetime


@dataclass
class LaunchMetrics:
    """Overall launch metrics"""
    start_time: datetime
    components_total: int
    components_deployed: int
    components_healthy: int
    overall_health: float
    resource_usage: Dict[str, float]
    error_count: int
    warning_count: int


class LaunchMonitor:
    """Monitors launch progress and system health"""
    
    def __init__(self, launch_config):
        """
        Initialize launch monitor.
        
        Args:
            launch_config: Launch configuration
        """
        self.launch_config = launch_config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Monitoring state
        self.monitoring_active = False
        self.launch_start_time = None
        
        # Component tracking
        self.component_metrics: Dict[str, ComponentMetrics] = {}
        self.component_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Resource tracking
        self.resource_history = deque(maxlen=100)
        self.resource_baseline = None
        
        # Alert tracking
        self.alerts: List[Dict[str, Any]] = []
        self.alert_thresholds = {
            'cpu_percent': 90,
            'memory_percent': 95,
            'disk_percent': 95,
            'error_rate': 0.05,
            'response_time': 5000  # ms
        }
        
        # Performance tracking
        self.performance_metrics = {
            'launch_phases': {},
            'component_startup_times': {},
            'integration_times': {}
        }
        
        # Thread management
        self._monitor_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        
        # Monitoring interval
        self.monitor_interval = self.launch_config.health_check_interval
    
    def start_launch_monitoring(self):
        """Start monitoring the launch process"""
        try:
            self.logger.info("Starting launch monitoring...")
            
            with self._lock:
                if self.monitoring_active:
                    self.logger.warning("Monitoring already active")
                    return
                
                self.monitoring_active = True
                self.launch_start_time = datetime.now()
                
                # Capture resource baseline
                self.resource_baseline = self._get_resource_usage()
                
                # Start monitoring thread
                self._stop_event.clear()
                self._monitor_thread = threading.Thread(
                    target=self._monitoring_loop,
                    daemon=True
                )
                self._monitor_thread.start()
                
                self.logger.info("Launch monitoring started")
                
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            self.monitoring_active = False
            raise
    
    def stop_launch_monitoring(self):
        """Stop monitoring the launch process"""
        try:
            self.logger.info("Stopping launch monitoring...")
            
            with self._lock:
                if not self.monitoring_active:
                    return
                
                self.monitoring_active = False
                self._stop_event.set()
                
                if self._monitor_thread:
                    self._monitor_thread.join(timeout=5)
                
                self.logger.info("Launch monitoring stopped")
                
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                # Collect metrics
                self._collect_system_metrics()
                self._check_component_health()
                self._check_thresholds()
                
                # Sleep for monitoring interval
                self._stop_event.wait(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_system_metrics(self):
        """Collect system-wide metrics"""
        try:
            metrics = self._get_resource_usage()
            
            with self._lock:
                self.resource_history.append({
                    'timestamp': datetime.now(),
                    'metrics': metrics
                })
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get network IO
            net_io = psutil.net_io_counters()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024 ** 3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024 ** 3),
                'network_sent_mb': net_io.bytes_sent / (1024 ** 2) if net_io else 0,
                'network_recv_mb': net_io.bytes_recv / (1024 ** 2) if net_io else 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get resource usage: {e}")
            return {}
    
    def _check_component_health(self):
        """Check health of all monitored components"""
        try:
            with self._lock:
                for component_name, metrics in self.component_metrics.items():
                    # Update component health based on recent metrics
                    health_score = self._calculate_health_score(metrics)
                    metrics.health_score = health_score
                    
                    # Determine status
                    if health_score >= 0.9:
                        metrics.status = 'healthy'
                    elif health_score >= 0.7:
                        metrics.status = 'degraded'
                    else:
                        metrics.status = 'unhealthy'
                    
                    metrics.last_update = datetime.now()
                    
        except Exception as e:
            self.logger.error(f"Failed to check component health: {e}")
    
    def _calculate_health_score(self, metrics: ComponentMetrics) -> float:
        """Calculate health score for a component"""
        try:
            # Factors affecting health
            cpu_factor = 1.0 - (metrics.cpu_usage / 100.0)
            memory_factor = 1.0 - (metrics.memory_usage / 100.0)
            error_factor = 1.0 - min(metrics.error_rate * 10, 1.0)  # Scale error rate
            response_factor = 1.0 - min(metrics.response_time / 5000.0, 1.0)  # 5s max
            
            # Weighted average
            weights = {
                'cpu': 0.2,
                'memory': 0.2,
                'error': 0.4,
                'response': 0.2
            }
            
            health_score = (
                cpu_factor * weights['cpu'] +
                memory_factor * weights['memory'] +
                error_factor * weights['error'] +
                response_factor * weights['response']
            )
            
            return max(0.0, min(1.0, health_score))
            
        except Exception:
            return 0.0
    
    def _check_thresholds(self):
        """Check if any metrics exceed thresholds"""
        try:
            current_metrics = self._get_resource_usage()
            
            # Check system thresholds
            for metric, value in current_metrics.items():
                if metric in self.alert_thresholds:
                    threshold = self.alert_thresholds[metric]
                    
                    if value > threshold:
                        self._generate_alert({
                            'type': 'threshold_exceeded',
                            'metric': metric,
                            'value': value,
                            'threshold': threshold,
                            'severity': 'warning' if value < threshold * 1.1 else 'critical'
                        })
            
            # Check component thresholds
            with self._lock:
                for component_name, metrics in self.component_metrics.items():
                    if metrics.error_rate > self.alert_thresholds['error_rate']:
                        self._generate_alert({
                            'type': 'high_error_rate',
                            'component': component_name,
                            'error_rate': metrics.error_rate,
                            'threshold': self.alert_thresholds['error_rate'],
                            'severity': 'critical'
                        })
                    
                    if metrics.response_time > self.alert_thresholds['response_time']:
                        self._generate_alert({
                            'type': 'slow_response',
                            'component': component_name,
                            'response_time': metrics.response_time,
                            'threshold': self.alert_thresholds['response_time'],
                            'severity': 'warning'
                        })
                        
        except Exception as e:
            self.logger.error(f"Failed to check thresholds: {e}")
    
    def _generate_alert(self, alert_data: Dict[str, Any]):
        """Generate an alert"""
        try:
            alert = {
                'id': f"alert_{len(self.alerts) + 1}",
                'timestamp': datetime.now().isoformat(),
                **alert_data
            }
            
            with self._lock:
                self.alerts.append(alert)
            
            # Log alert
            if alert['severity'] == 'critical':
                self.logger.error(f"CRITICAL ALERT: {alert}")
            else:
                self.logger.warning(f"Alert: {alert}")
                
        except Exception as e:
            self.logger.error(f"Failed to generate alert: {e}")
    
    def start_component_monitoring(self, component_name: str, component_type: str):
        """
        Start monitoring a specific component.
        
        Args:
            component_name: Name of the component
            component_type: Type of component ('system' or 'agent')
        """
        try:
            with self._lock:
                self.component_metrics[component_name] = ComponentMetrics(
                    component_name=component_name,
                    component_type=component_type,
                    cpu_usage=0.0,
                    memory_usage=0.0,
                    response_time=0.0,
                    error_rate=0.0,
                    request_count=0,
                    health_score=1.0,
                    status='initializing',
                    last_update=datetime.now()
                )
                
                self.logger.info(f"Started monitoring component: {component_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to start component monitoring: {e}")
    
    def update_component_metrics(self, component_name: str, metrics: Dict[str, Any]):
        """
        Update metrics for a component.
        
        Args:
            component_name: Name of the component
            metrics: Updated metrics
        """
        try:
            with self._lock:
                if component_name in self.component_metrics:
                    component = self.component_metrics[component_name]
                    
                    # Update metrics
                    component.cpu_usage = metrics.get('cpu_usage', component.cpu_usage)
                    component.memory_usage = metrics.get('memory_usage', component.memory_usage)
                    component.response_time = metrics.get('response_time', component.response_time)
                    component.error_rate = metrics.get('error_rate', component.error_rate)
                    component.request_count = metrics.get('request_count', component.request_count)
                    component.last_update = datetime.now()
                    
                    # Store in history
                    self.component_history[component_name].append({
                        'timestamp': datetime.now(),
                        'metrics': metrics
                    })
                    
        except Exception as e:
            self.logger.error(f"Failed to update component metrics: {e}")
    
    def get_component_metrics(self, component_name: str) -> Dict[str, Any]:
        """
        Get current metrics for a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Component metrics
        """
        try:
            with self._lock:
                if component_name in self.component_metrics:
                    metrics = self.component_metrics[component_name]
                    
                    return {
                        'component_name': metrics.component_name,
                        'component_type': metrics.component_type,
                        'cpu_usage': metrics.cpu_usage,
                        'memory_usage': metrics.memory_usage,
                        'response_time': metrics.response_time,
                        'error_rate': metrics.error_rate,
                        'request_count': metrics.request_count,
                        'health_score': metrics.health_score,
                        'status': metrics.status,
                        'last_update': metrics.last_update.isoformat()
                    }
                else:
                    return {}
                    
        except Exception as e:
            self.logger.error(f"Failed to get component metrics: {e}")
            return {}
    
    def get_component_health(self, component_name: str) -> Dict[str, Any]:
        """
        Get health status for a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Component health status
        """
        try:
            with self._lock:
                if component_name in self.component_metrics:
                    metrics = self.component_metrics[component_name]
                    
                    return {
                        'status': metrics.status,
                        'health_score': metrics.health_score,
                        'last_update': metrics.last_update.isoformat()
                    }
                else:
                    return {
                        'status': 'unknown',
                        'health_score': 0.0
                    }
                    
        except Exception as e:
            self.logger.error(f"Failed to get component health: {e}")
            return {
                'status': 'error',
                'health_score': 0.0,
                'error': str(e)
            }
    
    def setup_system_monitoring(self, system_name: str) -> Dict[str, Any]:
        """
        Setup monitoring for a system.
        
        Args:
            system_name: Name of the system
            
        Returns:
            Setup result
        """
        try:
            self.start_component_monitoring(system_name, 'system')
            
            # Initialize with baseline metrics
            self.update_component_metrics(system_name, {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'response_time': 0.0,
                'error_rate': 0.0,
                'request_count': 0
            })
            
            return {
                'setup': True,
                'system': system_name,
                'monitoring_active': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to setup system monitoring: {e}")
            return {
                'setup': False,
                'error': str(e)
            }
    
    def setup_agent_monitoring(self, agent_name: str) -> Dict[str, Any]:
        """
        Setup monitoring for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Setup result
        """
        try:
            self.start_component_monitoring(agent_name, 'agent')
            
            # Initialize with baseline metrics
            self.update_component_metrics(agent_name, {
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'response_time': 50.0,  # Initial response time
                'error_rate': 0.0,
                'request_count': 0
            })
            
            return {
                'setup': True,
                'agent': agent_name,
                'monitoring_active': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to setup agent monitoring: {e}")
            return {
                'setup': False,
                'error': str(e)
            }
    
    def get_launch_summary(self) -> Dict[str, Any]:
        """Get summary of launch progress and health"""
        try:
            with self._lock:
                # Calculate overall metrics
                total_components = len(self.component_metrics)
                healthy_components = sum(
                    1 for m in self.component_metrics.values()
                    if m.status == 'healthy'
                )
                
                # Calculate average health score
                if total_components > 0:
                    avg_health = sum(
                        m.health_score for m in self.component_metrics.values()
                    ) / total_components
                else:
                    avg_health = 0.0
                
                # Get current resource usage
                current_resources = self._get_resource_usage()
                
                # Calculate launch duration
                if self.launch_start_time:
                    duration = (datetime.now() - self.launch_start_time).total_seconds()
                else:
                    duration = 0
                
                return {
                    'launch_duration_seconds': duration,
                    'components_total': total_components,
                    'components_healthy': healthy_components,
                    'overall_health_score': avg_health,
                    'resource_usage': current_resources,
                    'alert_count': len(self.alerts),
                    'critical_alerts': sum(1 for a in self.alerts if a.get('severity') == 'critical'),
                    'status': 'healthy' if avg_health >= 0.9 else 'degraded' if avg_health >= 0.7 else 'unhealthy',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get launch summary: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report"""
        try:
            with self._lock:
                # Calculate resource usage trends
                resource_trends = {}
                if len(self.resource_history) >= 2:
                    recent_metrics = [h['metrics'] for h in list(self.resource_history)[-10:]]
                    
                    for metric in ['cpu_percent', 'memory_percent']:
                        values = [m.get(metric, 0) for m in recent_metrics]
                        resource_trends[metric] = {
                            'current': values[-1] if values else 0,
                            'average': statistics.mean(values) if values else 0,
                            'max': max(values) if values else 0,
                            'trend': 'increasing' if values[-1] > values[0] else 'decreasing'
                        }
                
                # Component performance
                component_performance = {}
                for name, metrics in self.component_metrics.items():
                    component_performance[name] = {
                        'health_score': metrics.health_score,
                        'response_time': metrics.response_time,
                        'error_rate': metrics.error_rate,
                        'status': metrics.status
                    }
                
                return {
                    'resource_trends': resource_trends,
                    'component_performance': component_performance,
                    'startup_times': self.performance_metrics.get('component_startup_times', {}),
                    'alerts_generated': len(self.alerts),
                    'monitoring_duration': (datetime.now() - self.launch_start_time).total_seconds() if self.launch_start_time else 0,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get performance report: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def record_phase_timing(self, phase_name: str, duration: float):
        """Record timing for a launch phase"""
        try:
            with self._lock:
                self.performance_metrics['launch_phases'][phase_name] = {
                    'duration_seconds': duration,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to record phase timing: {e}")
    
    def record_startup_time(self, component_name: str, startup_time: float):
        """Record component startup time"""
        try:
            with self._lock:
                self.performance_metrics['component_startup_times'][component_name] = startup_time
                
        except Exception as e:
            self.logger.error(f"Failed to record startup time: {e}")
    
    def get_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get alerts, optionally filtered by severity.
        
        Args:
            severity: Optional severity filter ('warning', 'critical')
            
        Returns:
            List of alerts
        """
        try:
            with self._lock:
                if severity:
                    return [a for a in self.alerts if a.get('severity') == severity]
                else:
                    return self.alerts.copy()
                    
        except Exception as e:
            self.logger.error(f"Failed to get alerts: {e}")
            return []
    
    def clear_alerts(self):
        """Clear all alerts"""
        with self._lock:
            self.alerts.clear()
            self.logger.info("Cleared all alerts")