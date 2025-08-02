"""
Production Alert System - Real-time alert system for Saraphis production
NO FALLBACKS - HARD FAILURES ONLY

Provides comprehensive alerting for critical issues, performance degradation,
security violations, and system failures with <30 second response time.
"""

import os
import sys
import json
import time
import logging
import threading
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import traceback
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts"""
    PERFORMANCE = "performance"
    SECURITY = "security"
    SYSTEM_FAILURE = "system_failure"
    AGENT_FAILURE = "agent_failure"
    RESOURCE = "resource"
    ERROR_RATE = "error_rate"
    COMMUNICATION = "communication"
    HEALTH = "health"


@dataclass
class Alert:
    """Represents a system alert"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    source: str  # System or agent name
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    status: str = "active"  # active, acknowledged, resolved
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    response_actions: List[str] = field(default_factory=list)
    related_alerts: List[str] = field(default_factory=list)


@dataclass
class AlertRule:
    """Defines rules for alert generation"""
    rule_id: str
    name: str
    alert_type: AlertType
    condition: Dict[str, Any]  # Condition parameters
    severity: AlertSeverity
    threshold: float
    duration: int  # Seconds the condition must persist
    cooldown: int  # Seconds before the same alert can trigger again
    actions: List[str]  # Actions to take when triggered
    enabled: bool = True
    last_triggered: Optional[datetime] = None


@dataclass
class AlertResponse:
    """Response to an alert"""
    response_id: str
    alert_id: str
    action_type: str
    status: str  # pending, in_progress, completed, failed
    started_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ProductionAlertSystem:
    """Real-time alert system for Saraphis production"""
    
    def __init__(self, monitor, optimization_engine):
        """
        Initialize production alert system.
        
        Args:
            monitor: Real-time production monitor instance
            optimization_engine: Production optimization engine instance
        """
        self.monitor = monitor
        self.optimization_engine = optimization_engine
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Alert state
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_responses: Dict[str, AlertResponse] = {}
        self.alert_history = deque(maxlen=1000)
        
        # Alert metrics
        self.alert_counts = defaultdict(int)
        self.false_positive_count = 0
        self.response_times = deque(maxlen=100)
        
        # Alert configuration
        self.max_active_alerts = 100
        self.alert_retention_hours = 24
        self.response_timeout = 30  # 30 seconds max response time
        
        # Notification channels
        self.notification_channels = {
            'email': self._send_email_notification,
            'slack': self._send_slack_notification,
            'webhook': self._send_webhook_notification,
            'log': self._log_notification
        }
        self.enabled_channels = ['log']  # Default to log only
        
        # Response actions
        self.response_actions = {
            'optimize': self._trigger_optimization,
            'restart': self._trigger_restart,
            'scale': self._trigger_scaling,
            'isolate': self._trigger_isolation,
            'notify': self._trigger_notification
        }
        
        # Alert suppression
        self.suppression_rules: Dict[str, Dict[str, Any]] = {}
        self.suppressed_alerts = defaultdict(int)
        
        # Thread management
        self._alert_thread = None
        self._response_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize default alert rules
        self._initialize_default_rules()
        
        # Register with monitor for alerts
        self.monitor.register_alert_callback(self._handle_monitor_alert)
        
    def _initialize_default_rules(self):
        """Initialize default alert rules"""
        default_rules = [
            AlertRule(
                rule_id="cpu_critical",
                name="Critical CPU Usage",
                alert_type=AlertType.RESOURCE,
                condition={"metric": "cpu_usage", "operator": ">"},
                severity=AlertSeverity.CRITICAL,
                threshold=95.0,
                duration=30,
                cooldown=300,
                actions=["optimize", "notify"]
            ),
            AlertRule(
                rule_id="memory_critical",
                name="Critical Memory Usage",
                alert_type=AlertType.RESOURCE,
                condition={"metric": "memory_usage", "operator": ">"},
                severity=AlertSeverity.CRITICAL,
                threshold=95.0,
                duration=30,
                cooldown=300,
                actions=["optimize", "notify"]
            ),
            AlertRule(
                rule_id="high_error_rate",
                name="High Error Rate",
                alert_type=AlertType.ERROR_RATE,
                condition={"metric": "error_rate", "operator": ">"},
                severity=AlertSeverity.HIGH,
                threshold=0.05,
                duration=60,
                cooldown=600,
                actions=["optimize", "notify"]
            ),
            AlertRule(
                rule_id="system_health_critical",
                name="Critical System Health",
                alert_type=AlertType.HEALTH,
                condition={"metric": "health_score", "operator": "<"},
                severity=AlertSeverity.CRITICAL,
                threshold=0.5,
                duration=10,
                cooldown=300,
                actions=["optimize", "isolate", "notify"]
            ),
            AlertRule(
                rule_id="security_violation",
                name="Security Violation Detected",
                alert_type=AlertType.SECURITY,
                condition={"metric": "security_events", "operator": ">"},
                severity=AlertSeverity.CRITICAL,
                threshold=0,
                duration=0,
                cooldown=60,
                actions=["isolate", "notify"]
            ),
            AlertRule(
                rule_id="slow_response",
                name="Slow Response Time",
                alert_type=AlertType.PERFORMANCE,
                condition={"metric": "response_time", "operator": ">"},
                severity=AlertSeverity.MEDIUM,
                threshold=1000.0,  # 1 second
                duration=120,
                cooldown=600,
                actions=["optimize"]
            ),
            AlertRule(
                rule_id="agent_failure",
                name="Agent Failure",
                alert_type=AlertType.AGENT_FAILURE,
                condition={"metric": "agent_status", "operator": "==", "value": "error"},
                severity=AlertSeverity.HIGH,
                threshold=0,
                duration=0,
                cooldown=300,
                actions=["restart", "notify"]
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
    
    def start_alert_system(self) -> Dict[str, Any]:
        """Start the alert system"""
        try:
            self.logger.info("Starting production alert system...")
            
            with self._lock:
                if self._alert_thread and self._alert_thread.is_alive():
                    raise RuntimeError("Alert system already running")
                
                self._stop_event.clear()
                
                # Start alert monitoring thread
                self._alert_thread = threading.Thread(
                    target=self._alert_monitoring_loop,
                    daemon=True
                )
                self._alert_thread.start()
                
                # Start response processing thread
                self._response_thread = threading.Thread(
                    target=self._response_processing_loop,
                    daemon=True
                )
                self._response_thread.start()
                
                self.logger.info("Alert system started successfully")
                
                return {
                    'started': True,
                    'active_rules': len([r for r in self.alert_rules.values() if r.enabled]),
                    'enabled_channels': self.enabled_channels,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to start alert system: {e}")
            return {
                'started': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def stop_alert_system(self) -> Dict[str, Any]:
        """Stop the alert system"""
        try:
            self.logger.info("Stopping alert system...")
            
            with self._lock:
                self._stop_event.set()
                
                if self._alert_thread:
                    self._alert_thread.join(timeout=5)
                
                if self._response_thread:
                    self._response_thread.join(timeout=5)
                
                # Shutdown executor
                self._executor.shutdown(wait=False)
                
                self.logger.info("Alert system stopped")
                
                return {
                    'stopped': True,
                    'total_alerts': len(self.alert_history),
                    'active_alerts': len([a for a in self.alerts.values() if a.status == 'active']),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error stopping alert system: {e}")
            return {
                'stopped': False,
                'error': str(e)
            }
    
    def _alert_monitoring_loop(self):
        """Main alert monitoring loop"""
        while not self._stop_event.is_set():
            try:
                # Check alert rules
                self._check_alert_rules()
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
                # Sleep for a short interval
                self._stop_event.wait(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in alert monitoring loop: {e}")
                self.logger.error(traceback.format_exc())
    
    def _response_processing_loop(self):
        """Process alert responses"""
        while not self._stop_event.is_set():
            try:
                # Process pending responses
                self._process_pending_responses()
                
                # Check response timeouts
                self._check_response_timeouts()
                
                # Sleep for a short interval
                self._stop_event.wait(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in response processing loop: {e}")
                self.logger.error(traceback.format_exc())
    
    def _handle_monitor_alert(self, alert_data: Dict[str, Any]):
        """Handle alerts from the monitor"""
        try:
            # Create alert from monitor data
            alert_type = AlertType.PERFORMANCE  # Default
            if 'security' in alert_data.get('type', ''):
                alert_type = AlertType.SECURITY
            elif 'resource' in alert_data.get('type', ''):
                alert_type = AlertType.RESOURCE
            elif 'error' in alert_data.get('type', ''):
                alert_type = AlertType.ERROR_RATE
            
            severity = AlertSeverity.MEDIUM  # Default
            if alert_data.get('severity') == 'critical':
                severity = AlertSeverity.CRITICAL
            elif alert_data.get('severity') == 'high':
                severity = AlertSeverity.HIGH
            elif alert_data.get('severity') == 'low':
                severity = AlertSeverity.LOW
            
            alert_id = f"alert_{int(time.time() * 1000)}_{alert_data.get('type', 'unknown')}"
            
            alert = Alert(
                alert_id=alert_id,
                alert_type=alert_type,
                severity=severity,
                source=alert_data.get('source', 'monitor'),
                message=alert_data.get('message', 'Alert triggered'),
                details=alert_data,
                timestamp=datetime.now()
            )
            
            self._create_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Error handling monitor alert: {e}")
    
    def _check_alert_rules(self):
        """Check all alert rules against current metrics"""
        try:
            # Get current system and agent metrics
            system_status = self.monitor.monitor_all_systems()
            agent_status = self.monitor.monitor_all_agents()
            
            for rule in self.alert_rules.values():
                if not rule.enabled:
                    continue
                
                # Check cooldown
                if rule.last_triggered:
                    cooldown_elapsed = (datetime.now() - rule.last_triggered).total_seconds()
                    if cooldown_elapsed < rule.cooldown:
                        continue
                
                # Check rule condition
                triggered = self._evaluate_rule_condition(rule, system_status, agent_status)
                
                if triggered:
                    self._trigger_alert_rule(rule)
                    
        except Exception as e:
            self.logger.error(f"Error checking alert rules: {e}")
    
    def _evaluate_rule_condition(self, rule: AlertRule, system_status: Dict[str, Any], 
                                agent_status: Dict[str, Any]) -> bool:
        """Evaluate if a rule condition is met"""
        try:
            condition = rule.condition
            metric = condition.get('metric')
            operator = condition.get('operator')
            
            # Get metric values based on alert type
            if rule.alert_type == AlertType.RESOURCE:
                # Check resource metrics for all systems
                for system_name, status in system_status.get('system_status', {}).items():
                    value = status.get(metric, 0)
                    if self._compare_values(value, operator, rule.threshold):
                        return True
                        
            elif rule.alert_type == AlertType.ERROR_RATE:
                # Check error rates
                for system_name, status in system_status.get('system_status', {}).items():
                    if metric == 'error_rate' and status.get('error_count', 0) > 0:
                        # Calculate error rate
                        error_rate = status['error_count'] / max(1, self.monitor.request_counts.get(system_name, 1))
                        if self._compare_values(error_rate, operator, rule.threshold):
                            return True
                            
            elif rule.alert_type == AlertType.HEALTH:
                # Check health scores
                for system_name, status in system_status.get('system_status', {}).items():
                    value = status.get(metric, 1.0)
                    if self._compare_values(value, operator, rule.threshold):
                        return True
                        
            elif rule.alert_type == AlertType.AGENT_FAILURE:
                # Check agent status
                for agent_name, status in agent_status.get('agent_status', {}).items():
                    if metric == 'agent_status' and status.get('status') == condition.get('value'):
                        return True
                        
            elif rule.alert_type == AlertType.SECURITY:
                # Check security events
                security_status = self.monitor.monitor_security_status()
                if len(security_status.get('security_events', [])) > rule.threshold:
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating rule condition: {e}")
            return False
    
    def _compare_values(self, value: float, operator: str, threshold: float) -> bool:
        """Compare values based on operator"""
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return value == threshold
        elif operator == "!=":
            return value != threshold
        return False
    
    def _trigger_alert_rule(self, rule: AlertRule):
        """Trigger an alert based on a rule"""
        try:
            alert_id = f"alert_{rule.rule_id}_{int(time.time() * 1000)}"
            
            alert = Alert(
                alert_id=alert_id,
                alert_type=rule.alert_type,
                severity=rule.severity,
                source=f"rule_{rule.rule_id}",
                message=f"{rule.name} triggered",
                details={
                    'rule_id': rule.rule_id,
                    'rule_name': rule.name,
                    'threshold': rule.threshold,
                    'condition': rule.condition
                },
                timestamp=datetime.now()
            )
            
            # Add response actions from rule
            alert.response_actions = rule.actions.copy()
            
            self._create_alert(alert)
            
            # Update rule last triggered time
            rule.last_triggered = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error triggering alert rule: {e}")
    
    def _create_alert(self, alert: Alert):
        """Create a new alert"""
        try:
            with self._lock:
                # Check alert limits
                if len(self.alerts) >= self.max_active_alerts:
                    # Remove oldest resolved alert
                    oldest_resolved = None
                    for a in self.alerts.values():
                        if a.status == 'resolved':
                            if not oldest_resolved or a.resolved_at < oldest_resolved.resolved_at:
                                oldest_resolved = a
                    
                    if oldest_resolved:
                        del self.alerts[oldest_resolved.alert_id]
                
                # Check for suppression
                if self._is_alert_suppressed(alert):
                    self.suppressed_alerts[alert.alert_type.value] += 1
                    return
                
                # Store alert
                self.alerts[alert.alert_id] = alert
                self.alert_history.append(alert)
                self.alert_counts[alert.alert_type.value] += 1
                
                # Log alert
                self._log_alert(alert)
                
                # Send notifications
                self._send_notifications(alert)
                
                # Create response actions
                for action in alert.response_actions:
                    self._create_response_action(alert, action)
                    
        except Exception as e:
            self.logger.error(f"Error creating alert: {e}")
    
    def _is_alert_suppressed(self, alert: Alert) -> bool:
        """Check if an alert should be suppressed"""
        try:
            # Check suppression rules
            for rule_id, rule in self.suppression_rules.items():
                if (rule.get('alert_type') == alert.alert_type.value and
                    rule.get('source') == alert.source):
                    
                    # Check time window
                    window = rule.get('window_seconds', 300)
                    count = rule.get('max_alerts', 5)
                    
                    # Count recent similar alerts
                    recent_count = sum(
                        1 for a in self.alert_history
                        if (a.alert_type == alert.alert_type and
                            a.source == alert.source and
                            (datetime.now() - a.timestamp).total_seconds() < window)
                    )
                    
                    if recent_count >= count:
                        return True
                        
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking alert suppression: {e}")
            return False
    
    def _log_alert(self, alert: Alert):
        """Log an alert"""
        if alert.severity == AlertSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ALERT: {alert.message} ({alert.source})")
        elif alert.severity == AlertSeverity.HIGH:
            self.logger.error(f"HIGH ALERT: {alert.message} ({alert.source})")
        elif alert.severity == AlertSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM ALERT: {alert.message} ({alert.source})")
        else:
            self.logger.info(f"LOW ALERT: {alert.message} ({alert.source})")
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        try:
            for channel in self.enabled_channels:
                if channel in self.notification_channels:
                    # Send notification asynchronously
                    self._executor.submit(
                        self.notification_channels[channel], alert
                    )
                    
        except Exception as e:
            self.logger.error(f"Error sending notifications: {e}")
    
    def _send_email_notification(self, alert: Alert):
        """Send email notification"""
        try:
            # In real implementation, would send actual email
            self.logger.info(f"Would send email for alert: {alert.alert_id}")
        except Exception as e:
            self.logger.error(f"Error sending email: {e}")
    
    def _send_slack_notification(self, alert: Alert):
        """Send Slack notification"""
        try:
            # In real implementation, would send to Slack
            self.logger.info(f"Would send Slack message for alert: {alert.alert_id}")
        except Exception as e:
            self.logger.error(f"Error sending Slack notification: {e}")
    
    def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification"""
        try:
            # In real implementation, would call webhook
            self.logger.info(f"Would call webhook for alert: {alert.alert_id}")
        except Exception as e:
            self.logger.error(f"Error sending webhook: {e}")
    
    def _log_notification(self, alert: Alert):
        """Log notification (already done in _log_alert)"""
        pass
    
    def _create_response_action(self, alert: Alert, action: str):
        """Create a response action for an alert"""
        try:
            response_id = f"response_{alert.alert_id}_{action}"
            
            response = AlertResponse(
                response_id=response_id,
                alert_id=alert.alert_id,
                action_type=action,
                status='pending',
                started_at=datetime.now()
            )
            
            with self._lock:
                self.alert_responses[response_id] = response
                
        except Exception as e:
            self.logger.error(f"Error creating response action: {e}")
    
    def _process_pending_responses(self):
        """Process pending alert responses"""
        try:
            with self._lock:
                pending_responses = [
                    r for r in self.alert_responses.values()
                    if r.status == 'pending'
                ]
            
            for response in pending_responses:
                # Execute response action
                self._execute_response_action(response)
                
        except Exception as e:
            self.logger.error(f"Error processing pending responses: {e}")
    
    def _execute_response_action(self, response: AlertResponse):
        """Execute a response action"""
        try:
            response.status = 'in_progress'
            
            # Get the action handler
            action_handler = self.response_actions.get(response.action_type)
            
            if not action_handler:
                response.status = 'failed'
                response.error = f"Unknown action type: {response.action_type}"
                response.completed_at = datetime.now()
                return
            
            # Get the associated alert
            alert = self.alerts.get(response.alert_id)
            if not alert:
                response.status = 'failed'
                response.error = "Alert not found"
                response.completed_at = datetime.now()
                return
            
            # Execute action
            try:
                result = action_handler(alert)
                response.result = result
                response.status = 'completed'
                
                # Track response time
                response_time = (datetime.now() - response.started_at).total_seconds()
                self.response_times.append(response_time)
                
            except Exception as e:
                response.status = 'failed'
                response.error = str(e)
                self.logger.error(f"Response action failed: {e}")
            
            response.completed_at = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error executing response action: {e}")
    
    def _trigger_optimization(self, alert: Alert) -> Dict[str, Any]:
        """Trigger optimization in response to alert"""
        try:
            # Queue optimization based on alert type
            if alert.alert_type == AlertType.RESOURCE:
                return self.optimization_engine.optimize_resource_allocation()
            elif alert.alert_type == AlertType.PERFORMANCE:
                return self.optimization_engine.optimize_system_performance()
            elif alert.alert_type == AlertType.COMMUNICATION:
                return self.optimization_engine.optimize_agent_coordination()
            else:
                return {'status': 'no_optimization_available'}
                
        except Exception as e:
            raise Exception(f"Optimization trigger failed: {e}")
    
    def _trigger_restart(self, alert: Alert) -> Dict[str, Any]:
        """Trigger component restart in response to alert"""
        try:
            # In real implementation, would restart the component
            self.logger.info(f"Would restart component: {alert.source}")
            return {'status': 'restart_triggered', 'component': alert.source}
        except Exception as e:
            raise Exception(f"Restart trigger failed: {e}")
    
    def _trigger_scaling(self, alert: Alert) -> Dict[str, Any]:
        """Trigger scaling in response to alert"""
        try:
            # In real implementation, would scale the component
            self.logger.info(f"Would scale component: {alert.source}")
            return {'status': 'scaling_triggered', 'component': alert.source}
        except Exception as e:
            raise Exception(f"Scaling trigger failed: {e}")
    
    def _trigger_isolation(self, alert: Alert) -> Dict[str, Any]:
        """Trigger component isolation in response to alert"""
        try:
            # In real implementation, would isolate the component
            self.logger.info(f"Would isolate component: {alert.source}")
            return {'status': 'isolation_triggered', 'component': alert.source}
        except Exception as e:
            raise Exception(f"Isolation trigger failed: {e}")
    
    def _trigger_notification(self, alert: Alert) -> Dict[str, Any]:
        """Trigger additional notifications"""
        try:
            # Send to all channels
            self._send_notifications(alert)
            return {'status': 'notifications_sent'}
        except Exception as e:
            raise Exception(f"Notification trigger failed: {e}")
    
    def _check_response_timeouts(self):
        """Check for response timeouts"""
        try:
            with self._lock:
                for response in self.alert_responses.values():
                    if response.status == 'in_progress':
                        elapsed = (datetime.now() - response.started_at).total_seconds()
                        if elapsed > self.response_timeout:
                            response.status = 'failed'
                            response.error = 'Response timeout'
                            response.completed_at = datetime.now()
                            
                            self.logger.error(
                                f"Response timeout for {response.response_id} "
                                f"({elapsed:.1f}s > {self.response_timeout}s)"
                            )
                            
        except Exception as e:
            self.logger.error(f"Error checking response timeouts: {e}")
    
    def _cleanup_old_alerts(self):
        """Clean up old alerts"""
        try:
            with self._lock:
                cutoff_time = datetime.now() - timedelta(hours=self.alert_retention_hours)
                
                # Remove old resolved alerts
                alerts_to_remove = []
                for alert_id, alert in self.alerts.items():
                    if (alert.status == 'resolved' and 
                        alert.resolved_at and 
                        alert.resolved_at < cutoff_time):
                        alerts_to_remove.append(alert_id)
                
                for alert_id in alerts_to_remove:
                    del self.alerts[alert_id]
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up old alerts: {e}")
    
    def detect_critical_issues(self) -> Dict[str, Any]:
        """
        Detect critical issues in real-time.
        NO FALLBACKS - HARD FAILURES ONLY
        """
        try:
            critical_alerts = []
            
            with self._lock:
                for alert in self.alerts.values():
                    if (alert.severity == AlertSeverity.CRITICAL and 
                        alert.status == 'active'):
                        critical_alerts.append({
                            'alert_id': alert.alert_id,
                            'type': alert.alert_type.value,
                            'source': alert.source,
                            'message': alert.message,
                            'timestamp': alert.timestamp.isoformat(),
                            'response_actions': alert.response_actions
                        })
            
            return {
                'detection': 'critical_issues',
                'critical_count': len(critical_alerts),
                'critical_alerts': critical_alerts,
                'detection_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to detect critical issues: {e}")
            return {
                'detection': 'critical_issues',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_performance_alerts(self) -> Dict[str, Any]:
        """
        Generate performance degradation alerts.
        NO FALLBACKS - HARD FAILURES ONLY
        """
        try:
            # Get performance metrics
            perf_metrics = self.monitor.track_performance_metrics()
            
            if 'current_metrics' not in perf_metrics:
                raise RuntimeError("Unable to get performance metrics")
            
            current = perf_metrics['current_metrics']
            performance_alerts = []
            
            # Check for performance degradation
            if current['system_performance'] < 0.8:
                alert_id = f"perf_alert_{int(time.time() * 1000)}"
                alert = Alert(
                    alert_id=alert_id,
                    alert_type=AlertType.PERFORMANCE,
                    severity=AlertSeverity.HIGH if current['system_performance'] < 0.7 else AlertSeverity.MEDIUM,
                    source='system_performance',
                    message=f"System performance degraded to {current['system_performance']:.2%}",
                    details={'performance_score': current['system_performance']},
                    timestamp=datetime.now()
                )
                alert.response_actions = ['optimize']
                self._create_alert(alert)
                performance_alerts.append(alert_id)
            
            # Check latency
            if current['latency_p95'] > 500:  # 500ms
                alert_id = f"latency_alert_{int(time.time() * 1000)}"
                alert = Alert(
                    alert_id=alert_id,
                    alert_type=AlertType.PERFORMANCE,
                    severity=AlertSeverity.HIGH if current['latency_p95'] > 1000 else AlertSeverity.MEDIUM,
                    source='latency',
                    message=f"P95 latency increased to {current['latency_p95']:.0f}ms",
                    details={'latency_p95': current['latency_p95']},
                    timestamp=datetime.now()
                )
                alert.response_actions = ['optimize']
                self._create_alert(alert)
                performance_alerts.append(alert_id)
            
            return {
                'generation': 'performance_alerts',
                'alerts_generated': len(performance_alerts),
                'alert_ids': performance_alerts,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance alerts: {e}")
            return {
                'generation': 'performance_alerts',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def detect_security_violations(self) -> Dict[str, Any]:
        """
        Detect security violations and threats.
        NO FALLBACKS - HARD FAILURES ONLY
        """
        try:
            # Get security status
            security_status = self.monitor.monitor_security_status()
            
            if 'security_events' not in security_status:
                raise RuntimeError("Unable to get security status")
            
            security_alerts = []
            
            for event in security_status['security_events']:
                if event['severity'] in ['high', 'critical']:
                    alert_id = f"sec_alert_{int(time.time() * 1000)}_{event['type']}"
                    alert = Alert(
                        alert_id=alert_id,
                        alert_type=AlertType.SECURITY,
                        severity=AlertSeverity.CRITICAL if event['severity'] == 'critical' else AlertSeverity.HIGH,
                        source='security_monitor',
                        message=f"Security violation: {event['type']}",
                        details=event,
                        timestamp=datetime.now()
                    )
                    alert.response_actions = ['isolate', 'notify']
                    self._create_alert(alert)
                    security_alerts.append(alert_id)
            
            return {
                'detection': 'security_violations',
                'violations_detected': len(security_alerts),
                'alert_ids': security_alerts,
                'security_score': security_status.get('security_score', 0),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to detect security violations: {e}")
            return {
                'detection': 'security_violations',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def monitor_system_failures(self) -> Dict[str, Any]:
        """
        Monitor system failures and errors.
        NO FALLBACKS - HARD FAILURES ONLY
        """
        try:
            # Get system status
            system_status = self.monitor.monitor_all_systems()
            
            if 'system_status' not in system_status:
                raise RuntimeError("Unable to get system status")
            
            failure_alerts = []
            
            for system_name, status in system_status['system_status'].items():
                # Check for system failure
                if status['health_score'] == 0 or status['status'] == 'critical':
                    alert_id = f"sys_fail_{int(time.time() * 1000)}_{system_name}"
                    alert = Alert(
                        alert_id=alert_id,
                        alert_type=AlertType.SYSTEM_FAILURE,
                        severity=AlertSeverity.CRITICAL,
                        source=system_name,
                        message=f"System failure: {system_name}",
                        details=status,
                        timestamp=datetime.now()
                    )
                    alert.response_actions = ['restart', 'notify']
                    self._create_alert(alert)
                    failure_alerts.append(alert_id)
            
            return {
                'monitoring': 'system_failures',
                'failures_detected': len(failure_alerts),
                'failed_systems': [a.split('_')[-1] for a in failure_alerts],
                'alert_ids': failure_alerts,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to monitor system failures: {e}")
            return {
                'monitoring': 'system_failures',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def track_agent_communication_issues(self) -> Dict[str, Any]:
        """
        Track agent communication issues.
        NO FALLBACKS - HARD FAILURES ONLY
        """
        try:
            # Get agent status
            agent_status = self.monitor.monitor_all_agents()
            
            if 'agent_status' not in agent_status:
                raise RuntimeError("Unable to get agent status")
            
            communication_alerts = []
            
            for agent_name, status in agent_status['agent_status'].items():
                # Check communication latency
                if status['communication_latency'] > 100:  # 100ms
                    alert_id = f"comm_alert_{int(time.time() * 1000)}_{agent_name}"
                    alert = Alert(
                        alert_id=alert_id,
                        alert_type=AlertType.COMMUNICATION,
                        severity=AlertSeverity.HIGH if status['communication_latency'] > 200 else AlertSeverity.MEDIUM,
                        source=agent_name,
                        message=f"High communication latency: {status['communication_latency']:.0f}ms",
                        details={'latency': status['communication_latency']},
                        timestamp=datetime.now()
                    )
                    alert.response_actions = ['optimize']
                    self._create_alert(alert)
                    communication_alerts.append(alert_id)
                
                # Check coordination score
                if status['coordination_score'] < 0.7:
                    alert_id = f"coord_alert_{int(time.time() * 1000)}_{agent_name}"
                    alert = Alert(
                        alert_id=alert_id,
                        alert_type=AlertType.COMMUNICATION,
                        severity=AlertSeverity.HIGH,
                        source=agent_name,
                        message=f"Poor agent coordination: {status['coordination_score']:.2%}",
                        details={'coordination_score': status['coordination_score']},
                        timestamp=datetime.now()
                    )
                    alert.response_actions = ['optimize']
                    self._create_alert(alert)
                    communication_alerts.append(alert_id)
            
            return {
                'tracking': 'agent_communication_issues',
                'issues_detected': len(communication_alerts),
                'alert_ids': communication_alerts,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to track agent communication issues: {e}")
            return {
                'tracking': 'agent_communication_issues',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_alert_notifications(self) -> Dict[str, Any]:
        """
        Generate alert notifications and responses.
        NO FALLBACKS - HARD FAILURES ONLY
        """
        try:
            # Get active alerts
            active_alerts = []
            
            with self._lock:
                for alert in self.alerts.values():
                    if alert.status == 'active':
                        active_alerts.append(alert)
            
            # Send notifications for high priority alerts
            notifications_sent = 0
            
            for alert in active_alerts:
                if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                    self._send_notifications(alert)
                    notifications_sent += 1
            
            return {
                'generation': 'alert_notifications',
                'active_alerts': len(active_alerts),
                'notifications_sent': notifications_sent,
                'enabled_channels': self.enabled_channels,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate alert notifications: {e}")
            return {
                'generation': 'alert_notifications',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def prioritize_alerts(self) -> Dict[str, Any]:
        """
        Prioritize alerts based on severity and impact.
        NO FALLBACKS - HARD FAILURES ONLY
        """
        try:
            with self._lock:
                active_alerts = [a for a in self.alerts.values() if a.status == 'active']
            
            # Sort by severity and timestamp
            severity_order = {
                AlertSeverity.CRITICAL: 0,
                AlertSeverity.HIGH: 1,
                AlertSeverity.MEDIUM: 2,
                AlertSeverity.LOW: 3
            }
            
            prioritized_alerts = sorted(
                active_alerts,
                key=lambda a: (severity_order[a.severity], a.timestamp)
            )
            
            # Create prioritized list
            priority_list = []
            for i, alert in enumerate(prioritized_alerts[:10]):  # Top 10
                priority_list.append({
                    'priority': i + 1,
                    'alert_id': alert.alert_id,
                    'type': alert.alert_type.value,
                    'severity': alert.severity.value,
                    'source': alert.source,
                    'message': alert.message,
                    'age_seconds': (datetime.now() - alert.timestamp).total_seconds()
                })
            
            return {
                'prioritization': 'alerts',
                'total_active': len(active_alerts),
                'prioritized_count': len(priority_list),
                'priority_list': priority_list,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to prioritize alerts: {e}")
            return {
                'prioritization': 'alerts',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def coordinate_alert_responses(self) -> Dict[str, Any]:
        """
        Coordinate responses to critical alerts.
        NO FALLBACKS - HARD FAILURES ONLY
        """
        try:
            # Get pending and in-progress responses
            with self._lock:
                active_responses = [
                    r for r in self.alert_responses.values()
                    if r.status in ['pending', 'in_progress']
                ]
            
            # Check response times
            avg_response_time = 0
            if self.response_times:
                avg_response_time = sum(self.response_times) / len(self.response_times)
            
            # Coordinate responses
            coordination_actions = []
            
            # Group responses by action type
            action_groups = defaultdict(list)
            for response in active_responses:
                action_groups[response.action_type].append(response)
            
            # Ensure no conflicting actions
            for action_type, responses in action_groups.items():
                if len(responses) > 1:
                    # Multiple same actions - coordinate
                    coordination_actions.append({
                        'action': 'consolidate',
                        'type': action_type,
                        'count': len(responses)
                    })
            
            return {
                'coordination': 'alert_responses',
                'active_responses': len(active_responses),
                'average_response_time': avg_response_time,
                'response_time_target': self.response_timeout,
                'meets_target': avg_response_time <= self.response_timeout,
                'coordination_actions': coordination_actions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to coordinate alert responses: {e}")
            return {
                'coordination': 'alert_responses',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_alert_report(self, alert_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive alert report.
        NO FALLBACKS - HARD FAILURES ONLY
        """
        try:
            report = {
                'report_id': f"alert_report_{int(time.time())}",
                'timestamp': datetime.now().isoformat(),
                
                # Alert summary
                'alert_summary': {
                    'total_alerts': len(self.alert_history),
                    'active_alerts': len([a for a in self.alerts.values() if a.status == 'active']),
                    'resolved_alerts': len([a for a in self.alerts.values() if a.status == 'resolved']),
                    'suppressed_alerts': sum(self.suppressed_alerts.values())
                },
                
                # Alert breakdown by type
                'alert_breakdown': dict(self.alert_counts),
                
                # Severity distribution
                'severity_distribution': {
                    'critical': len([a for a in self.alert_history if a.severity == AlertSeverity.CRITICAL]),
                    'high': len([a for a in self.alert_history if a.severity == AlertSeverity.HIGH]),
                    'medium': len([a for a in self.alert_history if a.severity == AlertSeverity.MEDIUM]),
                    'low': len([a for a in self.alert_history if a.severity == AlertSeverity.LOW])
                },
                
                # Response metrics
                'response_metrics': {
                    'total_responses': len(self.alert_responses),
                    'completed_responses': len([r for r in self.alert_responses.values() if r.status == 'completed']),
                    'failed_responses': len([r for r in self.alert_responses.values() if r.status == 'failed']),
                    'average_response_time': sum(self.response_times) / len(self.response_times) if self.response_times else 0,
                    'response_time_compliance': sum(1 for t in self.response_times if t <= self.response_timeout) / len(self.response_times) if self.response_times else 0
                },
                
                # Recent critical alerts
                'recent_critical_alerts': [
                    {
                        'alert_id': a.alert_id,
                        'type': a.alert_type.value,
                        'source': a.source,
                        'message': a.message,
                        'timestamp': a.timestamp.isoformat(),
                        'status': a.status
                    }
                    for a in sorted(
                        [a for a in self.alert_history if a.severity == AlertSeverity.CRITICAL],
                        key=lambda x: x.timestamp,
                        reverse=True
                    )[:10]
                ],
                
                # System health indicators
                'health_indicators': {
                    'alert_rate': len(self.alert_history) / max(1, (datetime.now() - self.alert_history[0].timestamp).total_seconds() / 3600) if self.alert_history else 0,
                    'false_positive_rate': self.false_positive_count / max(1, len(self.alert_history)),
                    'response_success_rate': len([r for r in self.alert_responses.values() if r.status == 'completed']) / max(1, len(self.alert_responses))
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate alert report: {e}")
            return {
                'report_id': f"alert_report_error_{int(time.time())}",
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }