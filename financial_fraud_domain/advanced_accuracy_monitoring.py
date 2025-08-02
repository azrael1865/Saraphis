"""
Advanced Accuracy Monitoring for Financial Fraud Detection
Production-ready alerting system, automated responses, and dashboard management.
Integrates with RealTimeAccuracyMonitor for comprehensive monitoring capabilities.
"""

import logging
import time
import threading
import queue
import json
import pickle
import smtplib
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque, Counter
from contextlib import contextmanager
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency, entropy
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import yaml
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import statistics

# Import from existing modules
try:
    from real_time_accuracy_monitor import (
        RealTimeAccuracyMonitor, DriftAlert, DriftType, DriftSeverity,
        MonitoringStatus, AccuracyWindow
    )
    from accuracy_tracking_db import (
        AccuracyTrackingDatabase, AccuracyMetric, ModelVersion,
        MetricType, DataType, ModelStatus
    )
    from enhanced_fraud_core_monitoring import (
        MonitoringManager, MetricsCollector, PerformanceMetrics,
        CacheManager, monitor_performance, AlertSeverity
    )
    from enhanced_fraud_core_exceptions import (
        EnhancedFraudException, MonitoringError, ValidationError,
        ErrorContext, create_error_context
    )
except ImportError:
    # Fallback for standalone development
    from real_time_accuracy_monitor import (
        RealTimeAccuracyMonitor, DriftAlert, DriftType, DriftSeverity,
        MonitoringStatus, AccuracyWindow
    )
    from accuracy_tracking_db import (
        AccuracyTrackingDatabase, AccuracyMetric, ModelVersion,
        MetricType, DataType, ModelStatus
    )
    from enhanced_fraud_core_monitoring import (
        MonitoringManager, MetricsCollector, PerformanceMetrics,
        CacheManager, monitor_performance, AlertSeverity
    )
    from enhanced_fraud_core_exceptions import (
        EnhancedFraudException, MonitoringError, ValidationError,
        ErrorContext, create_error_context
    )

# Configure logging
logger = logging.getLogger(__name__)

# ======================== CUSTOM EXCEPTIONS ========================

class AlertingError(MonitoringError):
    """Exception raised during alerting operations"""
    pass

class DashboardError(MonitoringError):
    """Exception raised during dashboard operations"""
    pass

class RemediationError(MonitoringError):
    """Exception raised during automated remediation"""
    pass

# ======================== ENUMS AND CONSTANTS ========================

class AlertChannel(Enum):
    """Alert notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    SMS = "sms"
    LOG = "log"

class ResponseAction(Enum):
    """Automated response actions"""
    ROLLBACK_MODEL = "rollback_model"
    ADJUST_THRESHOLD = "adjust_threshold"
    INCREASE_MONITORING = "increase_monitoring"
    NOTIFY_TEAM = "notify_team"
    DISABLE_MODEL = "disable_model"
    ENABLE_FALLBACK = "enable_fallback"
    COLLECT_DIAGNOSTICS = "collect_diagnostics"

class DriftDetectionMethod(Enum):
    """Advanced drift detection methods"""
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    CHI_SQUARE = "chi_square"
    POPULATION_STABILITY_INDEX = "psi"
    KULLBACK_LEIBLER = "kl_divergence"
    WASSERSTEIN = "wasserstein"
    JENSEN_SHANNON = "jensen_shannon"

class DashboardWidgetType(Enum):
    """Dashboard widget types"""
    METRIC_CARD = "metric_card"
    TIME_SERIES = "time_series"
    GAUGE = "gauge"
    HEATMAP = "heatmap"
    DISTRIBUTION = "distribution"
    ALERT_LIST = "alert_list"
    MODEL_COMPARISON = "model_comparison"

# Default configurations
DEFAULT_ALERT_CONFIG = {
    'alert_cooldown_minutes': 15,
    'max_alerts_per_hour': 10,
    'escalation_timeout_minutes': 30,
    'retry_attempts': 3,
    'retry_delay_seconds': 60
}

DEFAULT_RESPONSE_CONFIG = {
    'auto_rollback_threshold': 0.15,  # 15% accuracy drop
    'threshold_adjustment_step': 0.05,
    'max_threshold_adjustment': 0.2,
    'diagnostic_collection_timeout': 300  # 5 minutes
}

DEFAULT_DASHBOARD_CONFIG = {
    'refresh_interval_seconds': 30,
    'max_data_points': 1000,
    'default_time_range_hours': 24,
    'widget_height': 400,
    'widget_width': 600
}

# ======================== DATA STRUCTURES ========================

@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    condition: Dict[str, Any]
    severity: DriftSeverity
    channels: List[AlertChannel]
    enabled: bool = True
    cooldown_minutes: int = 15
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlertNotification:
    """Alert notification record"""
    alert_id: str
    rule_id: str
    timestamp: datetime
    channel: AlertChannel
    recipient: str
    status: str  # sent, failed, pending
    attempts: int = 0
    error_message: Optional[str] = None

@dataclass
class ResponseRule:
    """Automated response rule"""
    rule_id: str
    trigger_condition: Dict[str, Any]
    actions: List[ResponseAction]
    safety_limits: Dict[str, Any]
    enabled: bool = True
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RemediationAction:
    """Remediation action record"""
    action_id: str
    response_rule_id: str
    action_type: ResponseAction
    timestamp: datetime
    status: str  # pending, executing, completed, failed
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    widget_type: DashboardWidgetType
    title: str
    data_source: str
    config: Dict[str, Any]
    position: Dict[str, int]  # x, y, width, height
    refresh_interval: Optional[int] = None

# ======================== ALERTING SYSTEM ========================

class AlertingSystem:
    """
    Comprehensive alerting system with multi-channel notifications,
    escalation procedures, and automated responses.
    """
    
    def __init__(
        self,
        accuracy_monitor: RealTimeAccuracyMonitor,
        alert_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize AlertingSystem with RealTimeAccuracyMonitor integration.
        
        Args:
            accuracy_monitor: RealTimeAccuracyMonitor instance
            alert_config: Alert configuration
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        if not isinstance(accuracy_monitor, RealTimeAccuracyMonitor):
            raise ValidationError(
                "accuracy_monitor must be a RealTimeAccuracyMonitor instance",
                context=create_error_context(
                    component="AlertingSystem",
                    operation="init"
                )
            )
        
        self.accuracy_monitor = accuracy_monitor
        self.config = self._load_config(alert_config)
        
        # Alert management
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.notification_queue = queue.Queue()
        
        # Response management
        self.response_rules = {}
        self.active_remediations = {}
        self.remediation_history = deque(maxlen=500)
        
        # Channel handlers
        self.channel_handlers = {
            AlertChannel.EMAIL: self._send_email_alert,
            AlertChannel.SLACK: self._send_slack_alert,
            AlertChannel.PAGERDUTY: self._send_pagerduty_alert,
            AlertChannel.WEBHOOK: self._send_webhook_alert,
            AlertChannel.SMS: self._send_sms_alert,
            AlertChannel.LOG: self._send_log_alert
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background workers
        self.notification_worker = None
        self.escalation_worker = None
        self.stop_workers = threading.Event()
        
        # Statistics
        self.alert_stats = defaultdict(int)
        
        # Start workers
        self._start_workers()
        
        # Register with accuracy monitor
        self._register_alert_callback()
        
        self.logger.info("AlertingSystem initialized successfully")
    
    def _load_config(self, alert_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load and validate alert configuration"""
        config = DEFAULT_ALERT_CONFIG.copy()
        
        if alert_config:
            config.update(alert_config)
        
        # Add channel-specific configurations
        config['channels'] = {
            'email': {
                'smtp_server': alert_config.get('smtp_server', 'localhost') if alert_config else 'localhost',
                'smtp_port': alert_config.get('smtp_port', 587) if alert_config else 587,
                'smtp_user': alert_config.get('smtp_user', '') if alert_config else '',
                'smtp_password': alert_config.get('smtp_password', '') if alert_config else '',
                'from_address': alert_config.get('from_address', 'alerts@frauddetection.com') if alert_config else 'alerts@frauddetection.com'
            },
            'slack': {
                'webhook_url': alert_config.get('slack_webhook_url', '') if alert_config else '',
                'default_channel': alert_config.get('slack_channel', '#alerts') if alert_config else '#alerts'
            },
            'pagerduty': {
                'integration_key': alert_config.get('pagerduty_key', '') if alert_config else '',
                'api_endpoint': 'https://events.pagerduty.com/v2/enqueue'
            },
            'webhook': {
                'default_url': alert_config.get('webhook_url', '') if alert_config else '',
                'timeout': alert_config.get('webhook_timeout', 30) if alert_config else 30
            }
        }
        
        return config
    
    def _start_workers(self) -> None:
        """Start background worker threads"""
        # Notification worker
        self.notification_worker = threading.Thread(
            target=self._notification_worker_loop,
            daemon=True
        )
        self.notification_worker.start()
        
        # Escalation worker
        self.escalation_worker = threading.Thread(
            target=self._escalation_worker_loop,
            daemon=True
        )
        self.escalation_worker.start()
        
        self.logger.info("Alert workers started")
    
    def _register_alert_callback(self) -> None:
        """Register alert callback with accuracy monitor"""
        def alert_callback(drift_alert: DriftAlert):
            self.process_drift_alert(drift_alert)
        
        # This would integrate with the monitoring manager's alert system
        if hasattr(self.accuracy_monitor, 'monitoring_manager'):
            self.accuracy_monitor.monitoring_manager.add_alert_callback(
                lambda severity, message: self._handle_monitoring_alert(severity, message)
            )
    
    # ======================== ALERT CONFIGURATION ========================
    
    def configure_alert_rule(
        self,
        rule_name: str,
        condition: Dict[str, Any],
        severity: DriftSeverity,
        channels: List[AlertChannel],
        cooldown_minutes: Optional[int] = None,
        enabled: bool = True
    ) -> AlertRule:
        """
        Configure an alert rule.
        
        Args:
            rule_name: Name of the alert rule
            condition: Alert condition configuration
            severity: Alert severity level
            channels: Notification channels
            cooldown_minutes: Cooldown period between alerts
            enabled: Whether rule is enabled
            
        Returns:
            Created AlertRule
        """
        rule_id = self._generate_rule_id()
        
        rule = AlertRule(
            rule_id=rule_id,
            name=rule_name,
            condition=condition,
            severity=severity,
            channels=channels,
            enabled=enabled,
            cooldown_minutes=cooldown_minutes or self.config['alert_cooldown_minutes']
        )
        
        with self._lock:
            self.alert_rules[rule_id] = rule
        
        self.logger.info(f"Configured alert rule: {rule_name} (ID: {rule_id})")
        
        return rule
    
    def update_alert_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing alert rule"""
        with self._lock:
            if rule_id not in self.alert_rules:
                self.logger.error(f"Alert rule {rule_id} not found")
                return False
            
            rule = self.alert_rules[rule_id]
            
            # Update allowed fields
            for field in ['condition', 'severity', 'channels', 'cooldown_minutes', 'enabled']:
                if field in updates:
                    setattr(rule, field, updates[field])
            
            self.logger.info(f"Updated alert rule {rule_id}")
            return True
    
    def delete_alert_rule(self, rule_id: str) -> bool:
        """Delete an alert rule"""
        with self._lock:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                self.logger.info(f"Deleted alert rule {rule_id}")
                return True
            return False
    
    # ======================== ALERT PROCESSING ========================
    
    def process_drift_alert(self, drift_alert: DriftAlert) -> None:
        """Process drift alert from accuracy monitor"""
        try:
            # Check applicable rules
            applicable_rules = self._find_applicable_rules(drift_alert)
            
            for rule in applicable_rules:
                if self._should_trigger_alert(rule, drift_alert):
                    self.trigger_accuracy_alert(
                        alert_type=drift_alert.drift_type.value,
                        model_id=drift_alert.model_id,
                        severity=drift_alert.severity,
                        alert_data={
                            'drift_alert': asdict(drift_alert),
                            'rule_id': rule.rule_id
                        }
                    )
            
        except Exception as e:
            self.logger.error(f"Error processing drift alert: {e}")
    
    def trigger_accuracy_alert(
        self,
        alert_type: str,
        model_id: str,
        severity: DriftSeverity,
        alert_data: Dict[str, Any]
    ) -> str:
        """
        Trigger an accuracy alert.
        
        Args:
            alert_type: Type of alert
            model_id: Model identifier
            severity: Alert severity
            alert_data: Additional alert data
            
        Returns:
            Alert ID
        """
        alert_id = self._generate_alert_id()
        
        alert_record = {
            'alert_id': alert_id,
            'alert_type': alert_type,
            'model_id': model_id,
            'severity': severity,
            'timestamp': datetime.now(),
            'data': alert_data,
            'status': 'active',
            'notifications_sent': []
        }
        
        with self._lock:
            self.active_alerts[alert_id] = alert_record
            self.alert_history.append(alert_record)
            self.alert_stats[alert_type] += 1
        
        # Queue notifications
        rule_id = alert_data.get('rule_id')
        if rule_id and rule_id in self.alert_rules:
            rule = self.alert_rules[rule_id]
            for channel in rule.channels:
                notification = AlertNotification(
                    alert_id=alert_id,
                    rule_id=rule_id,
                    timestamp=datetime.now(),
                    channel=channel,
                    recipient=self._get_recipient(channel),
                    status='pending'
                )
                self.notification_queue.put(notification)
        
        self.logger.info(
            f"Triggered alert {alert_id}: {alert_type} for model {model_id} "
            f"with severity {severity.value}"
        )
        
        # Check for automated responses
        self._check_automated_responses(alert_record)
        
        return alert_id
    
    def _find_applicable_rules(self, drift_alert: DriftAlert) -> List[AlertRule]:
        """Find alert rules applicable to a drift alert"""
        applicable_rules = []
        
        with self._lock:
            for rule in self.alert_rules.values():
                if not rule.enabled:
                    continue
                
                # Check if rule conditions match
                if self._evaluate_rule_condition(rule.condition, drift_alert):
                    applicable_rules.append(rule)
        
        return applicable_rules
    
    def _evaluate_rule_condition(
        self,
        condition: Dict[str, Any],
        drift_alert: DriftAlert
    ) -> bool:
        """Evaluate if rule condition matches drift alert"""
        # Check drift type
        if 'drift_types' in condition:
            if drift_alert.drift_type.value not in condition['drift_types']:
                return False
        
        # Check severity
        if 'min_severity' in condition:
            severity_order = [
                DriftSeverity.LOW,
                DriftSeverity.MEDIUM,
                DriftSeverity.HIGH,
                DriftSeverity.CRITICAL
            ]
            min_severity = DriftSeverity(condition['min_severity'])
            if severity_order.index(drift_alert.severity) < severity_order.index(min_severity):
                return False
        
        # Check threshold
        if 'threshold' in condition:
            if abs(drift_alert.current_value - drift_alert.baseline_value) < condition['threshold']:
                return False
        
        # Check model filter
        if 'model_ids' in condition:
            if drift_alert.model_id not in condition['model_ids']:
                return False
        
        return True
    
    def _should_trigger_alert(self, rule: AlertRule, drift_alert: DriftAlert) -> bool:
        """Check if alert should be triggered based on cooldown"""
        # Check cooldown period
        cooldown_key = f"{rule.rule_id}:{drift_alert.model_id}:{drift_alert.drift_type.value}"
        
        with self._lock:
            # Check recent alerts
            recent_alerts = [
                a for a in self.alert_history
                if (a.get('data', {}).get('rule_id') == rule.rule_id and
                    a['model_id'] == drift_alert.model_id and
                    (datetime.now() - a['timestamp']).total_seconds() < rule.cooldown_minutes * 60)
            ]
            
            if recent_alerts:
                self.logger.debug(
                    f"Alert suppressed due to cooldown: {rule.name} for {drift_alert.model_id}"
                )
                return False
        
        return True
    
    # ======================== NOTIFICATION HANDLING ========================
    
    def _notification_worker_loop(self) -> None:
        """Background worker for sending notifications"""
        while not self.stop_workers.is_set():
            try:
                # Get notification from queue (timeout to check stop condition)
                try:
                    notification = self.notification_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Send notification
                self._send_notification(notification)
                
            except Exception as e:
                self.logger.error(f"Error in notification worker: {e}")
    
    def _send_notification(self, notification: AlertNotification) -> None:
        """Send a notification through specified channel"""
        try:
            # Get alert data
            with self._lock:
                if notification.alert_id not in self.active_alerts:
                    return
                alert_data = self.active_alerts[notification.alert_id]
            
            # Get handler for channel
            handler = self.channel_handlers.get(notification.channel)
            if not handler:
                self.logger.error(f"No handler for channel: {notification.channel}")
                notification.status = 'failed'
                notification.error_message = "Channel handler not found"
                return
            
            # Send notification
            success = handler(alert_data, notification)
            
            if success:
                notification.status = 'sent'
                notification.attempts += 1
                
                # Update alert record
                with self._lock:
                    alert_data['notifications_sent'].append({
                        'channel': notification.channel.value,
                        'timestamp': datetime.now().isoformat(),
                        'recipient': notification.recipient
                    })
            else:
                notification.status = 'failed'
                notification.attempts += 1
                
                # Retry if within limits
                if notification.attempts < self.config['retry_attempts']:
                    time.sleep(self.config['retry_delay_seconds'])
                    self.notification_queue.put(notification)
            
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
            notification.status = 'failed'
            notification.error_message = str(e)
    
    def _send_email_alert(
        self,
        alert_data: Dict[str, Any],
        notification: AlertNotification
    ) -> bool:
        """Send email alert"""
        try:
            config = self.config['channels']['email']
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert_data['severity'].value.upper()}] Fraud Detection Alert: {alert_data['alert_type']}"
            msg['From'] = config['from_address']
            msg['To'] = notification.recipient
            
            # Create body
            text_body = self._format_alert_text(alert_data)
            html_body = self._format_alert_html(alert_data)
            
            msg.attach(MIMEText(text_body, 'plain'))
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                if config['smtp_user'] and config['smtp_password']:
                    server.starttls()
                    server.login(config['smtp_user'], config['smtp_password'])
                
                server.send_message(msg)
            
            self.logger.info(f"Email alert sent to {notification.recipient}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _send_slack_alert(
        self,
        alert_data: Dict[str, Any],
        notification: AlertNotification
    ) -> bool:
        """Send Slack alert"""
        try:
            config = self.config['channels']['slack']
            
            if not config['webhook_url']:
                self.logger.error("Slack webhook URL not configured")
                return False
            
            # Format message
            severity_emoji = {
                DriftSeverity.LOW: "ℹ️",
                DriftSeverity.MEDIUM: "⚠️",
                DriftSeverity.HIGH: "🚨",
                DriftSeverity.CRITICAL: "🔴"
            }
            
            emoji = severity_emoji.get(alert_data['severity'], "📢")
            
            payload = {
                'text': f"{emoji} *Fraud Detection Alert*",
                'attachments': [{
                    'color': self._get_severity_color(alert_data['severity']),
                    'fields': [
                        {
                            'title': 'Alert Type',
                            'value': alert_data['alert_type'],
                            'short': True
                        },
                        {
                            'title': 'Model',
                            'value': alert_data['model_id'],
                            'short': True
                        },
                        {
                            'title': 'Severity',
                            'value': alert_data['severity'].value,
                            'short': True
                        },
                        {
                            'title': 'Time',
                            'value': alert_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                            'short': True
                        }
                    ],
                    'footer': 'Fraud Detection System',
                    'ts': int(alert_data['timestamp'].timestamp())
                }]
            }
            
            # Add details if available
            if 'data' in alert_data and 'drift_alert' in alert_data['data']:
                drift_alert = alert_data['data']['drift_alert']
                payload['attachments'][0]['fields'].extend([
                    {
                        'title': 'Current Value',
                        'value': f"{drift_alert.get('current_value', 'N/A'):.3f}",
                        'short': True
                    },
                    {
                        'title': 'Baseline Value',
                        'value': f"{drift_alert.get('baseline_value', 'N/A'):.3f}",
                        'short': True
                    }
                ])
            
            # Send to Slack
            response = requests.post(
                config['webhook_url'],
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                self.logger.info("Slack alert sent successfully")
                return True
            else:
                self.logger.error(f"Slack API error: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def _send_pagerduty_alert(
        self,
        alert_data: Dict[str, Any],
        notification: AlertNotification
    ) -> bool:
        """Send PagerDuty alert"""
        try:
            config = self.config['channels']['pagerduty']
            
            if not config['integration_key']:
                self.logger.error("PagerDuty integration key not configured")
                return False
            
            # Map severity
            pd_severity = {
                DriftSeverity.LOW: "info",
                DriftSeverity.MEDIUM: "warning",
                DriftSeverity.HIGH: "error",
                DriftSeverity.CRITICAL: "critical"
            }
            
            payload = {
                'routing_key': config['integration_key'],
                'event_action': 'trigger',
                'dedup_key': alert_data['alert_id'],
                'payload': {
                    'summary': f"Fraud Detection Alert: {alert_data['alert_type']}",
                    'source': alert_data['model_id'],
                    'severity': pd_severity.get(alert_data['severity'], 'error'),
                    'timestamp': alert_data['timestamp'].isoformat(),
                    'custom_details': {
                        'alert_type': alert_data['alert_type'],
                        'model_id': alert_data['model_id'],
                        'data': alert_data.get('data', {})
                    }
                }
            }
            
            response = requests.post(
                config['api_endpoint'],
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 202:
                self.logger.info("PagerDuty alert sent successfully")
                return True
            else:
                self.logger.error(f"PagerDuty API error: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to send PagerDuty alert: {e}")
            return False
    
    def _send_webhook_alert(
        self,
        alert_data: Dict[str, Any],
        notification: AlertNotification
    ) -> bool:
        """Send webhook alert"""
        try:
            config = self.config['channels']['webhook']
            webhook_url = notification.recipient or config['default_url']
            
            if not webhook_url:
                self.logger.error("Webhook URL not configured")
                return False
            
            # Prepare payload
            payload = {
                'alert_id': alert_data['alert_id'],
                'alert_type': alert_data['alert_type'],
                'model_id': alert_data['model_id'],
                'severity': alert_data['severity'].value,
                'timestamp': alert_data['timestamp'].isoformat(),
                'data': alert_data.get('data', {})
            }
            
            response = requests.post(
                webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=config['timeout']
            )
            
            if response.status_code in [200, 201, 202]:
                self.logger.info(f"Webhook alert sent to {webhook_url}")
                return True
            else:
                self.logger.error(f"Webhook error: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    def _send_sms_alert(
        self,
        alert_data: Dict[str, Any],
        notification: AlertNotification
    ) -> bool:
        """Send SMS alert (placeholder - implement with SMS provider)"""
        self.logger.warning("SMS alerts not implemented")
        return False
    
    def _send_log_alert(
        self,
        alert_data: Dict[str, Any],
        notification: AlertNotification
    ) -> bool:
        """Log alert to system log"""
        severity_map = {
            DriftSeverity.LOW: logging.INFO,
            DriftSeverity.MEDIUM: logging.WARNING,
            DriftSeverity.HIGH: logging.ERROR,
            DriftSeverity.CRITICAL: logging.CRITICAL
        }
        
        log_level = severity_map.get(alert_data['severity'], logging.ERROR)
        
        self.logger.log(
            log_level,
            f"ALERT: {alert_data['alert_type']} for model {alert_data['model_id']} - "
            f"Severity: {alert_data['severity'].value} - "
            f"Data: {json.dumps(alert_data.get('data', {}), default=str)}"
        )
        
        return True
    
    # ======================== AUTOMATED RESPONSE SYSTEM ========================
    
    def configure_automated_responses(
        self,
        response_rules: Dict[str, Dict[str, Any]],
        safety_limits: Dict[str, Any]
    ) -> None:
        """
        Configure automated response rules.
        
        Args:
            response_rules: Dictionary of response rule configurations
            safety_limits: Safety limits for automated actions
        """
        self.safety_limits = safety_limits
        
        for rule_name, rule_config in response_rules.items():
            rule_id = self._generate_rule_id()
            
            rule = ResponseRule(
                rule_id=rule_id,
                trigger_condition=rule_config.get('trigger_condition', {}),
                actions=[ResponseAction(a) for a in rule_config.get('actions', [])],
                safety_limits=rule_config.get('safety_limits', {}),
                enabled=rule_config.get('enabled', True),
                priority=rule_config.get('priority', 1)
            )
            
            with self._lock:
                self.response_rules[rule_id] = rule
            
            self.logger.info(f"Configured response rule: {rule_name} (ID: {rule_id})")
    
    def _check_automated_responses(self, alert_record: Dict[str, Any]) -> None:
        """Check and execute automated responses for an alert"""
        applicable_rules = []
        
        with self._lock:
            for rule in self.response_rules.values():
                if not rule.enabled:
                    continue
                
                if self._evaluate_response_condition(rule.trigger_condition, alert_record):
                    applicable_rules.append(rule)
        
        # Sort by priority
        applicable_rules.sort(key=lambda r: r.priority)
        
        # Execute responses
        for rule in applicable_rules:
            self._execute_response_rule(rule, alert_record)
    
    def _evaluate_response_condition(
        self,
        condition: Dict[str, Any],
        alert_record: Dict[str, Any]
    ) -> bool:
        """Evaluate if response condition matches alert"""
        # Check alert type
        if 'alert_types' in condition:
            if alert_record['alert_type'] not in condition['alert_types']:
                return False
        
        # Check severity threshold
        if 'min_severity' in condition:
            severity_order = [
                DriftSeverity.LOW,
                DriftSeverity.MEDIUM,
                DriftSeverity.HIGH,
                DriftSeverity.CRITICAL
            ]
            min_severity = DriftSeverity(condition['min_severity'])
            if severity_order.index(alert_record['severity']) < severity_order.index(min_severity):
                return False
        
        # Check metric threshold
        if 'metric_threshold' in condition:
            drift_data = alert_record.get('data', {}).get('drift_alert', {})
            if 'current_value' in drift_data:
                if drift_data['current_value'] > condition['metric_threshold']:
                    return False
        
        return True
    
    def _execute_response_rule(
        self,
        rule: ResponseRule,
        alert_record: Dict[str, Any]
    ) -> None:
        """Execute automated response rule"""
        for action in rule.actions:
            try:
                # Check safety limits
                if not self._check_safety_limits(action, rule.safety_limits):
                    self.logger.warning(
                        f"Safety limit exceeded for action {action.value}"
                    )
                    continue
                
                # Create remediation record
                action_id = self._generate_action_id()
                remediation = RemediationAction(
                    action_id=action_id,
                    response_rule_id=rule.rule_id,
                    action_type=action,
                    timestamp=datetime.now(),
                    status='pending'
                )
                
                with self._lock:
                    self.active_remediations[action_id] = remediation
                
                # Execute action
                self._execute_remediation_action(remediation, alert_record)
                
            except Exception as e:
                self.logger.error(f"Failed to execute response action {action.value}: {e}")
    
    def _check_safety_limits(
        self,
        action: ResponseAction,
        limits: Dict[str, Any]
    ) -> bool:
        """Check if action is within safety limits"""
        # Check global limits
        if hasattr(self, 'safety_limits') and 'max_alerts_per_hour' in self.safety_limits:
            recent_alerts = sum(
                1 for a in self.alert_history
                if (datetime.now() - a['timestamp']).total_seconds() < 3600
            )
            if recent_alerts > self.safety_limits['max_alerts_per_hour']:
                return False
        
        # Check action-specific limits
        if action == ResponseAction.ROLLBACK_MODEL:
            # Check rollback frequency
            recent_rollbacks = sum(
                1 for r in self.remediation_history
                if (r.action_type == ResponseAction.ROLLBACK_MODEL and
                    (datetime.now() - r.timestamp).total_seconds() < 3600)
            )
            if recent_rollbacks >= limits.get('max_rollbacks_per_hour', 1):
                return False
        
        return True
    
    def execute_automated_remediation(
        self,
        remediation_config: Dict[str, Any],
        validation_rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute automated remediation action.
        
        Args:
            remediation_config: Remediation configuration
            validation_rules: Validation rules for remediation
            
        Returns:
            Remediation result
        """
        action_type = ResponseAction(remediation_config['action_type'])
        model_id = remediation_config.get('model_id')
        
        result = {
            'action_type': action_type.value,
            'model_id': model_id,
            'timestamp': datetime.now(),
            'status': 'pending',
            'details': {}
        }
        
        try:
            if action_type == ResponseAction.ROLLBACK_MODEL:
                result['details'] = self._rollback_model(model_id, validation_rules)
                
            elif action_type == ResponseAction.ADJUST_THRESHOLD:
                result['details'] = self._adjust_threshold(
                    model_id,
                    remediation_config.get('adjustment', 0.05)
                )
                
            elif action_type == ResponseAction.INCREASE_MONITORING:
                result['details'] = self._increase_monitoring(model_id)
                
            elif action_type == ResponseAction.COLLECT_DIAGNOSTICS:
                result['details'] = self._collect_diagnostics(model_id)
                
            elif action_type == ResponseAction.DISABLE_MODEL:
                result['details'] = self._disable_model(model_id)
                
            elif action_type == ResponseAction.ENABLE_FALLBACK:
                result['details'] = self._enable_fallback(model_id)
            
            result['status'] = 'completed'
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            self.logger.error(f"Remediation failed: {e}")
        
        return result
    
    def _execute_remediation_action(
        self,
        remediation: RemediationAction,
        alert_record: Dict[str, Any]
    ) -> None:
        """Execute a specific remediation action"""
        model_id = alert_record['model_id']
        
        try:
            if remediation.action_type == ResponseAction.ROLLBACK_MODEL:
                result = self._rollback_model(model_id, {})
                
            elif remediation.action_type == ResponseAction.ADJUST_THRESHOLD:
                result = self._adjust_threshold(model_id, 0.05)
                
            elif remediation.action_type == ResponseAction.INCREASE_MONITORING:
                result = self._increase_monitoring(model_id)
                
            elif remediation.action_type == ResponseAction.NOTIFY_TEAM:
                result = self._notify_team(alert_record)
                
            elif remediation.action_type == ResponseAction.COLLECT_DIAGNOSTICS:
                result = self._collect_diagnostics(model_id)
                
            elif remediation.action_type == ResponseAction.DISABLE_MODEL:
                result = self._disable_model(model_id)
                
            elif remediation.action_type == ResponseAction.ENABLE_FALLBACK:
                result = self._enable_fallback(model_id)
            
            else:
                raise ValueError(f"Unknown action type: {remediation.action_type}")
            
            remediation.status = 'completed'
            remediation.result = result
            
        except Exception as e:
            remediation.status = 'failed'
            remediation.error_message = str(e)
            raise
        
        finally:
            with self._lock:
                self.remediation_history.append(remediation)
    
    def _rollback_model(
        self,
        model_id: str,
        validation_rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Rollback to previous model version"""
        # This would integrate with model versioning system
        self.logger.info(f"Rolling back model {model_id}")
        
        # Placeholder implementation
        return {
            'action': 'rollback',
            'model_id': model_id,
            'previous_version': 'v1.0',
            'rollback_time': datetime.now().isoformat()
        }
    
    def _adjust_threshold(
        self,
        model_id: str,
        adjustment: float
    ) -> Dict[str, Any]:
        """Adjust model decision threshold"""
        self.logger.info(f"Adjusting threshold for model {model_id} by {adjustment}")
        
        # This would integrate with model configuration
        return {
            'action': 'adjust_threshold',
            'model_id': model_id,
            'adjustment': adjustment,
            'new_threshold': 0.5 + adjustment  # Placeholder
        }
    
    def _increase_monitoring(self, model_id: str) -> Dict[str, Any]:
        """Increase monitoring frequency for model"""
        self.logger.info(f"Increasing monitoring for model {model_id}")
        
        # Update monitoring configuration
        if hasattr(self.accuracy_monitor, 'config'):
            self.accuracy_monitor.config['update_interval_seconds'] = 15  # Faster updates
        
        return {
            'action': 'increase_monitoring',
            'model_id': model_id,
            'new_interval': 15
        }
    
    def _notify_team(self, alert_record: Dict[str, Any]) -> Dict[str, Any]:
        """Send additional team notifications"""
        # Trigger high-priority notifications
        for channel in [AlertChannel.EMAIL, AlertChannel.SLACK]:
            notification = AlertNotification(
                alert_id=alert_record['alert_id'],
                rule_id='team_notification',
                timestamp=datetime.now(),
                channel=channel,
                recipient=self._get_team_recipient(channel),
                status='pending'
            )
            self.notification_queue.put(notification)
        
        return {
            'action': 'notify_team',
            'channels': ['email', 'slack']
        }
    
    def _collect_diagnostics(self, model_id: str) -> Dict[str, Any]:
        """Collect diagnostic information"""
        diagnostics = {
            'action': 'collect_diagnostics',
            'model_id': model_id,
            'timestamp': datetime.now().isoformat(),
            'data': {}
        }
        
        # Collect model state
        if hasattr(self.accuracy_monitor, 'model_states') and model_id in self.accuracy_monitor.model_states:
            state = self.accuracy_monitor.model_states[model_id]
            diagnostics['data']['model_state'] = {
                'status': state.status.value,
                'current_accuracy': state.current_accuracy,
                'total_predictions': state.total_predictions,
                'drift_alerts': len(state.drift_alerts)
            }
        
        # Collect recent metrics
        diagnostics['data']['recent_metrics'] = self.accuracy_monitor.get_current_accuracy_metrics(model_id)
        
        return diagnostics
    
    def _disable_model(self, model_id: str) -> Dict[str, Any]:
        """Disable model from production"""
        self.logger.warning(f"Disabling model {model_id}")
        
        # Stop monitoring
        self.accuracy_monitor.stop_monitoring([model_id])
        
        return {
            'action': 'disable_model',
            'model_id': model_id,
            'disabled_at': datetime.now().isoformat()
        }
    
    def _enable_fallback(self, model_id: str) -> Dict[str, Any]:
        """Enable fallback model"""
        self.logger.info(f"Enabling fallback for model {model_id}")
        
        # This would integrate with model serving infrastructure
        return {
            'action': 'enable_fallback',
            'model_id': model_id,
            'fallback_model': 'baseline_model_v1',
            'enabled_at': datetime.now().isoformat()
        }
    
    # ======================== ESCALATION MANAGEMENT ========================
    
    def manage_alert_escalation(
        self,
        alert_id: str,
        escalation_rules: Dict[str, Any]
    ) -> None:
        """
        Manage alert escalation procedures.
        
        Args:
            alert_id: Alert identifier
            escalation_rules: Escalation rules configuration
        """
        with self._lock:
            if alert_id not in self.active_alerts:
                self.logger.error(f"Alert {alert_id} not found")
                return
            
            alert_record = self.active_alerts[alert_id]
        
        # Check escalation conditions
        alert_age = (datetime.now() - alert_record['timestamp']).total_seconds() / 60
        
        if alert_age > escalation_rules.get('escalation_minutes', 30):
            # Escalate severity
            if alert_record['severity'] != DriftSeverity.CRITICAL:
                self._escalate_alert(alert_id, escalation_rules)
    
    def _escalation_worker_loop(self) -> None:
        """Background worker for alert escalation"""
        while not self.stop_workers.is_set():
            try:
                with self._lock:
                    active_alert_ids = list(self.active_alerts.keys())
                
                for alert_id in active_alert_ids:
                    # Check default escalation rules
                    self.manage_alert_escalation(
                        alert_id,
                        {'escalation_minutes': self.config['escalation_timeout_minutes']}
                    )
                
                # Sleep before next check
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in escalation worker: {e}")
    
    def _escalate_alert(
        self,
        alert_id: str,
        escalation_rules: Dict[str, Any]
    ) -> None:
        """Escalate an alert"""
        with self._lock:
            alert_record = self.active_alerts[alert_id]
            
            # Update severity
            old_severity = alert_record['severity']
            severity_order = [
                DriftSeverity.LOW,
                DriftSeverity.MEDIUM,
                DriftSeverity.HIGH,
                DriftSeverity.CRITICAL
            ]
            
            current_index = severity_order.index(old_severity)
            if current_index < len(severity_order) - 1:
                alert_record['severity'] = severity_order[current_index + 1]
                
                self.logger.warning(
                    f"Escalated alert {alert_id} from {old_severity.value} to "
                    f"{alert_record['severity'].value}"
                )
                
                # Send escalation notifications
                self._send_escalation_notifications(alert_record)
    
    def _send_escalation_notifications(self, alert_record: Dict[str, Any]) -> None:
        """Send notifications for escalated alert"""
        # Send to high-priority channels
        for channel in [AlertChannel.PAGERDUTY, AlertChannel.SMS]:
            notification = AlertNotification(
                alert_id=alert_record['alert_id'],
                rule_id='escalation',
                timestamp=datetime.now(),
                channel=channel,
                recipient=self._get_escalation_recipient(channel),
                status='pending'
            )
            self.notification_queue.put(notification)
    
    def track_alert_resolution(
        self,
        alert_id: str,
        resolution_actions: List[Dict[str, Any]]
    ) -> bool:
        """
        Track resolution of an alert.
        
        Args:
            alert_id: Alert identifier
            resolution_actions: List of resolution actions taken
            
        Returns:
            Success status
        """
        with self._lock:
            if alert_id not in self.active_alerts:
                self.logger.error(f"Alert {alert_id} not found")
                return False
            
            alert_record = self.active_alerts[alert_id]
            alert_record['status'] = 'resolved'
            alert_record['resolved_at'] = datetime.now()
            alert_record['resolution_actions'] = resolution_actions
            
            # Move to history
            del self.active_alerts[alert_id]
            
        self.logger.info(f"Alert {alert_id} resolved with {len(resolution_actions)} actions")
        return True
    
    # ======================== HELPER METHODS ========================
    
    def _format_alert_text(self, alert_data: Dict[str, Any]) -> str:
        """Format alert as plain text"""
        text = f"""
Fraud Detection Alert

Type: {alert_data['alert_type']}
Model: {alert_data['model_id']}
Severity: {alert_data['severity'].value}
Time: {alert_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        if 'data' in alert_data and 'drift_alert' in alert_data['data']:
            drift = alert_data['data']['drift_alert']
            text += f"""
Drift Details:
- Current Value: {drift.get('current_value', 'N/A')}
- Baseline Value: {drift.get('baseline_value', 'N/A')}
- Threshold: {drift.get('threshold', 'N/A')}

Recommendations:
"""
            for rec in drift.get('recommendations', []):
                text += f"- {rec}\n"
        
        return text
    
    def _format_alert_html(self, alert_data: Dict[str, Any]) -> str:
        """Format alert as HTML"""
        severity_colors = {
            DriftSeverity.LOW: '#17a2b8',
            DriftSeverity.MEDIUM: '#ffc107',
            DriftSeverity.HIGH: '#fd7e14',
            DriftSeverity.CRITICAL: '#dc3545'
        }
        
        color = severity_colors.get(alert_data['severity'], '#6c757d')
        
        html = f"""
<html>
<body style="font-family: Arial, sans-serif;">
    <div style="border: 2px solid {color}; padding: 20px; border-radius: 5px;">
        <h2 style="color: {color};">Fraud Detection Alert</h2>
        
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 5px;"><strong>Type:</strong></td>
                <td style="padding: 5px;">{alert_data['alert_type']}</td>
            </tr>
            <tr>
                <td style="padding: 5px;"><strong>Model:</strong></td>
                <td style="padding: 5px;">{alert_data['model_id']}</td>
            </tr>
            <tr>
                <td style="padding: 5px;"><strong>Severity:</strong></td>
                <td style="padding: 5px;">{alert_data['severity'].value}</td>
            </tr>
            <tr>
                <td style="padding: 5px;"><strong>Time:</strong></td>
                <td style="padding: 5px;">{alert_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</td>
            </tr>
        </table>
"""
        
        if 'data' in alert_data and 'drift_alert' in alert_data['data']:
            drift = alert_data['data']['drift_alert']
            html += f"""
        <h3>Drift Details</h3>
        <ul>
            <li>Current Value: {drift.get('current_value', 'N/A'):.3f}</li>
            <li>Baseline Value: {drift.get('baseline_value', 'N/A'):.3f}</li>
            <li>Threshold: {drift.get('threshold', 'N/A'):.3f}</li>
        </ul>
        
        <h3>Recommendations</h3>
        <ul>
"""
            for rec in drift.get('recommendations', []):
                html += f"            <li>{rec}</li>\n"
            html += "        </ul>"
        
        html += """
    </div>
</body>
</html>
"""
        return html
    
    def _get_severity_color(self, severity: DriftSeverity) -> str:
        """Get color for severity level"""
        colors = {
            DriftSeverity.LOW: 'good',
            DriftSeverity.MEDIUM: 'warning',
            DriftSeverity.HIGH: 'danger',
            DriftSeverity.CRITICAL: '#FF0000'
        }
        return colors.get(severity, '#808080')
    
    def _get_recipient(self, channel: AlertChannel) -> str:
        """Get default recipient for channel"""
        # This would be configured per deployment
        recipients = {
            AlertChannel.EMAIL: 'alerts@example.com',
            AlertChannel.SLACK: '#fraud-alerts',
            AlertChannel.PAGERDUTY: 'fraud-detection-service',
            AlertChannel.WEBHOOK: self.config['channels']['webhook']['default_url'],
            AlertChannel.SMS: '+1234567890'
        }
        return recipients.get(channel, '')
    
    def _get_team_recipient(self, channel: AlertChannel) -> str:
        """Get team recipient for channel"""
        recipients = {
            AlertChannel.EMAIL: 'fraud-team@example.com',
            AlertChannel.SLACK: '#fraud-team-urgent'
        }
        return recipients.get(channel, self._get_recipient(channel))
    
    def _get_escalation_recipient(self, channel: AlertChannel) -> str:
        """Get escalation recipient for channel"""
        recipients = {
            AlertChannel.PAGERDUTY: 'fraud-detection-oncall',
            AlertChannel.SMS: '+1234567890'  # On-call phone
        }
        return recipients.get(channel, self._get_recipient(channel))
    
    def _generate_rule_id(self) -> str:
        """Generate unique rule ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        return f"rule_{timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        return f"alert_{timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
    
    def _generate_action_id(self) -> str:
        """Generate unique action ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        return f"action_{timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
    
    def _handle_monitoring_alert(self, severity: Any, message: str) -> None:
        """Handle alert from monitoring manager"""
        # Convert monitoring alert to accuracy alert
        alert_data = {
            'alert_type': 'monitoring_alert',
            'model_id': 'system',
            'severity': DriftSeverity.HIGH if hasattr(severity, 'value') and severity.value == 'ERROR' else DriftSeverity.MEDIUM,
            'timestamp': datetime.now(),
            'data': {'message': message}
        }
        
        # Process as regular alert
        self.trigger_accuracy_alert(
            alert_type='monitoring_alert',
            model_id='system',
            severity=alert_data['severity'],
            alert_data={'message': message}
        )
    
    # ======================== PUBLIC API ========================
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert system status"""
        with self._lock:
            return {
                'active_alerts': len(self.active_alerts),
                'total_alerts': len(self.alert_history),
                'active_rules': sum(1 for r in self.alert_rules.values() if r.enabled),
                'response_rules': sum(1 for r in self.response_rules.values() if r.enabled),
                'active_remediations': len(self.active_remediations),
                'alert_statistics': dict(self.alert_stats),
                'recent_alerts': [
                    {
                        'alert_id': a['alert_id'],
                        'type': a['alert_type'],
                        'severity': a['severity'].value,
                        'timestamp': a['timestamp'].isoformat()
                    }
                    for a in list(self.alert_history)[-10:]
                ]
            }
    
    def get_alert_history(
        self,
        hours: int = 24,
        model_id: Optional[str] = None,
        severity: Optional[DriftSeverity] = None
    ) -> List[Dict[str, Any]]:
        """Get alert history with filters"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            alerts = []
            for alert in self.alert_history:
                if alert['timestamp'] < cutoff_time:
                    continue
                    
                if model_id and alert['model_id'] != model_id:
                    continue
                    
                if severity and alert['severity'] != severity:
                    continue
                
                alerts.append({
                    'alert_id': alert['alert_id'],
                    'alert_type': alert['alert_type'],
                    'model_id': alert['model_id'],
                    'severity': alert['severity'].value,
                    'timestamp': alert['timestamp'].isoformat(),
                    'status': alert.get('status', 'unknown')
                })
        
        return alerts
    
    def shutdown(self) -> None:
        """Shutdown alerting system"""
        self.logger.info("Shutting down AlertingSystem")
        
        # Stop workers
        self.stop_workers.set()
        
        if self.notification_worker:
            self.notification_worker.join(timeout=5)
        
        if self.escalation_worker:
            self.escalation_worker.join(timeout=5)
        
        self.logger.info("AlertingSystem shutdown complete")


# ======================== DASHBOARD MANAGER ========================

class DashboardManager:
    """
    Comprehensive dashboard management system for real-time monitoring
    and visualization of accuracy metrics.
    """
    
    def __init__(
        self,
        accuracy_monitor: RealTimeAccuracyMonitor,
        dashboard_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize DashboardManager with monitoring integration.
        
        Args:
            accuracy_monitor: RealTimeAccuracyMonitor instance
            dashboard_config: Dashboard configuration
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        if not isinstance(accuracy_monitor, RealTimeAccuracyMonitor):
            raise ValidationError(
                "accuracy_monitor must be a RealTimeAccuracyMonitor instance",
                context=create_error_context(
                    component="DashboardManager",
                    operation="init"
                )
            )
        
        self.accuracy_monitor = accuracy_monitor
        self.config = self._load_config(dashboard_config)
        
        # Dashboard management
        self.dashboards = {}
        self.widgets = {}
        self.data_sources = {}
        
        # Real-time update management
        self.update_subscriptions = defaultdict(list)
        self.update_thread = None
        self.stop_updates = threading.Event()
        
        # Visualization cache
        self.visualization_cache = {}
        self.cache_lock = threading.RLock()
        
        # Initialize data sources
        self._init_data_sources()
        
        # Start update thread
        self._start_update_thread()
        
        self.logger.info("DashboardManager initialized successfully")
    
    def _load_config(self, dashboard_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load and validate dashboard configuration"""
        config = DEFAULT_DASHBOARD_CONFIG.copy()
        
        if dashboard_config:
            config.update(dashboard_config)
        
        # Add visualization settings
        config['visualization'] = {
            'theme': dashboard_config.get('theme', 'plotly_white') if dashboard_config else 'plotly_white',
            'color_scheme': dashboard_config.get('color_scheme', 'viridis') if dashboard_config else 'viridis',
            'font_family': dashboard_config.get('font_family', 'Arial') if dashboard_config else 'Arial',
            'export_formats': dashboard_config.get('export_formats', ['html', 'png', 'json']) if dashboard_config else ['html', 'png', 'json']
        }
        
        return config
    
    def _init_data_sources(self) -> None:
        """Initialize data sources for dashboard"""
        # Accuracy metrics source
        self.data_sources['accuracy_metrics'] = lambda model_id: (
            self.accuracy_monitor.get_current_accuracy_metrics(model_id)
        )
        
        # Drift detection source
        self.data_sources['drift_detection'] = lambda model_id: (
            self.accuracy_monitor.detect_basic_accuracy_drift(model_id)
        )
        
        # Confidence trends source
        self.data_sources['confidence_trends'] = lambda model_id: (
            self.accuracy_monitor.monitor_prediction_confidence_drift(model_id)
        )
        
        # Accuracy trends source
        self.data_sources['accuracy_trends'] = lambda model_id: (
            self.accuracy_monitor.track_accuracy_trend_changes(model_id)
        )
        
        # Stability metrics source
        self.data_sources['stability_metrics'] = lambda model_id: (
            self.accuracy_monitor.calculate_accuracy_stability_metrics(model_id)
        )
    
    def _start_update_thread(self) -> None:
        """Start real-time update thread"""
        def update_loop():
            while not self.stop_updates.is_set():
                try:
                    self._process_updates()
                    time.sleep(self.config['refresh_interval_seconds'])
                except Exception as e:
                    self.logger.error(f"Error in update loop: {e}")
                    time.sleep(self.config['refresh_interval_seconds'])
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
    
    # ======================== DASHBOARD CREATION ========================
    
    def create_accuracy_dashboard(
        self,
        dashboard_config: Dict[str, Any],
        model_filters: List[str]
    ) -> Dict[str, Any]:
        """
        Create comprehensive accuracy monitoring dashboard.
        
        Args:
            dashboard_config: Dashboard configuration
            model_filters: List of model IDs to include
            
        Returns:
            Dashboard configuration with widgets
        """
        dashboard_id = self._generate_dashboard_id()
        
        # Create dashboard structure
        dashboard = {
            'dashboard_id': dashboard_id,
            'title': dashboard_config.get('title', 'Accuracy Monitoring Dashboard'),
            'created_at': datetime.now(),
            'config': dashboard_config,
            'model_filters': model_filters,
            'widgets': [],
            'layout': {
                'grid': dashboard_config.get('grid', {'columns': 12, 'rows': 8}),
                'responsive': dashboard_config.get('responsive', True)
            }
        }
        
        # Add requested widgets
        requested_widgets = dashboard_config.get('widgets', [
            'accuracy_trend', 'drift_detection', 'alert_status',
            'confidence_distribution', 'model_comparison', 'stability_gauge'
        ])
        
        widget_positions = self._calculate_widget_positions(requested_widgets)
        
        for i, widget_type in enumerate(requested_widgets):
            if widget_type == 'accuracy_trend':
                widget = self._create_accuracy_trend_widget(model_filters, widget_positions[i])
            elif widget_type == 'drift_detection':
                widget = self._create_drift_detection_widget(model_filters, widget_positions[i])
            elif widget_type == 'alert_status':
                widget = self._create_alert_status_widget(model_filters, widget_positions[i])
            elif widget_type == 'confidence_distribution':
                widget = self._create_confidence_distribution_widget(model_filters, widget_positions[i])
            elif widget_type == 'model_comparison':
                widget = self._create_model_comparison_widget(model_filters, widget_positions[i])
            elif widget_type == 'stability_gauge':
                widget = self._create_stability_gauge_widget(model_filters, widget_positions[i])
            else:
                continue
            
            dashboard['widgets'].append(widget)
            self.widgets[widget.widget_id] = widget
        
        # Store dashboard
        self.dashboards[dashboard_id] = dashboard
        
        # Setup real-time updates
        self.setup_real_time_updates(
            {'dashboard_id': dashboard_id},
            dashboard_config.get('refresh_interval', self.config['refresh_interval_seconds'])
        )
        
        self.logger.info(f"Created dashboard {dashboard_id} with {len(dashboard['widgets'])} widgets")
        
        return dashboard
    
    def generate_monitoring_widgets(
        self,
        widget_config: Dict[str, Any],
        data_sources: List[str]
    ) -> List[DashboardWidget]:
        """
        Generate monitoring widgets based on configuration.
        
        Args:
            widget_config: Widget configuration
            data_sources: List of data sources to use
            
        Returns:
            List of generated widgets
        """
        widgets = []
        
        for source in data_sources:
            if source == 'accuracy_metrics':
                widget = self._create_metric_card_widget(
                    title="Current Accuracy",
                    data_source=source,
                    config=widget_config
                )
                widgets.append(widget)
                
            elif source == 'drift_detection':
                widget = self._create_drift_heatmap_widget(
                    title="Drift Detection Status",
                    data_source=source,
                    config=widget_config
                )
                widgets.append(widget)
                
            elif source == 'confidence_trends':
                widget = self._create_time_series_widget(
                    title="Confidence Trends",
                    data_source=source,
                    config=widget_config
                )
                widgets.append(widget)
        
        return widgets
    
    def create_trend_visualizations(
        self,
        visualization_config: Dict[str, Any],
        historical_data: Dict[str, Any]
    ) -> Dict[str, go.Figure]:
        """
        Create trend visualizations from historical data.
        
        Args:
            visualization_config: Visualization configuration
            historical_data: Historical metrics data
            
        Returns:
            Dictionary of Plotly figures
        """
        figures = {}
        
        # Accuracy trend over time
        if 'accuracy_history' in historical_data:
            figures['accuracy_trend'] = self._create_accuracy_trend_figure(
                historical_data['accuracy_history'],
                visualization_config
            )
        
        # Drift detection timeline
        if 'drift_history' in historical_data:
            figures['drift_timeline'] = self._create_drift_timeline_figure(
                historical_data['drift_history'],
                visualization_config
            )
        
        # Model comparison
        if 'model_comparisons' in historical_data:
            figures['model_comparison'] = self._create_model_comparison_figure(
                historical_data['model_comparisons'],
                visualization_config
            )
        
        # Feature importance stability
        if 'feature_stability' in historical_data:
            figures['feature_stability'] = self._create_feature_stability_figure(
                historical_data['feature_stability'],
                visualization_config
            )
        
        return figures
    
    # ======================== WIDGET CREATION ========================
    
    def _create_accuracy_trend_widget(
        self,
        model_filters: List[str],
        position: Dict[str, int]
    ) -> DashboardWidget:
        """Create accuracy trend widget"""
        widget = DashboardWidget(
            widget_id=self._generate_widget_id(),
            widget_type=DashboardWidgetType.TIME_SERIES,
            title="Accuracy Trends",
            data_source="accuracy_trends",
            config={
                'models': model_filters,
                'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
                'time_range': self.config['default_time_range_hours'],
                'aggregation': 'mean',
                'show_confidence_bands': True
            },
            position=position
        )
        return widget
    
    def _create_drift_detection_widget(
        self,
        model_filters: List[str],
        position: Dict[str, int]
    ) -> DashboardWidget:
        """Create drift detection widget"""
        widget = DashboardWidget(
            widget_id=self._generate_widget_id(),
            widget_type=DashboardWidgetType.HEATMAP,
            title="Drift Detection Status",
            data_source="drift_detection",
            config={
                'models': model_filters,
                'drift_types': ['accuracy_drift', 'confidence_drift', 'feature_drift'],
                'severity_levels': ['low', 'medium', 'high', 'critical'],
                'color_scale': 'RdYlGn_r'
            },
            position=position
        )
        return widget
    
    def _create_alert_status_widget(
        self,
        model_filters: List[str],
        position: Dict[str, int]
    ) -> DashboardWidget:
        """Create alert status widget"""
        widget = DashboardWidget(
            widget_id=self._generate_widget_id(),
            widget_type=DashboardWidgetType.ALERT_LIST,
            title="Active Alerts",
            data_source="alert_status",
            config={
                'models': model_filters,
                'max_alerts': 10,
                'sort_by': 'severity',
                'show_resolution_status': True
            },
            position=position
        )
        return widget
    
    def _create_confidence_distribution_widget(
        self,
        model_filters: List[str],
        position: Dict[str, int]
    ) -> DashboardWidget:
        """Create confidence distribution widget"""
        widget = DashboardWidget(
            widget_id=self._generate_widget_id(),
            widget_type=DashboardWidgetType.DISTRIBUTION,
            title="Prediction Confidence Distribution",
            data_source="confidence_trends",
            config={
                'models': model_filters,
                'bins': 20,
                'show_statistics': True,
                'overlay_threshold': True
            },
            position=position
        )
        return widget
    
    def _create_model_comparison_widget(
        self,
        model_filters: List[str],
        position: Dict[str, int]
    ) -> DashboardWidget:
        """Create model comparison widget"""
        widget = DashboardWidget(
            widget_id=self._generate_widget_id(),
            widget_type=DashboardWidgetType.MODEL_COMPARISON,
            title="Model Performance Comparison",
            data_source="model_comparison",
            config={
                'models': model_filters,
                'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
                'comparison_type': 'radar',
                'show_baseline': True
            },
            position=position
        )
        return widget
    
    def _create_stability_gauge_widget(
        self,
        model_filters: List[str],
        position: Dict[str, int]
    ) -> DashboardWidget:
        """Create stability gauge widget"""
        widget = DashboardWidget(
            widget_id=self._generate_widget_id(),
            widget_type=DashboardWidgetType.GAUGE,
            title="Model Stability",
            data_source="stability_metrics",
            config={
                'models': model_filters,
                'gauge_ranges': [
                    {'range': [0, 0.3], 'color': 'red'},
                    {'range': [0.3, 0.7], 'color': 'yellow'},
                    {'range': [0.7, 1.0], 'color': 'green'}
                ],
                'show_trend': True
            },
            position=position
        )
        return widget
    
    def _create_metric_card_widget(
        self,
        title: str,
        data_source: str,
        config: Dict[str, Any]
    ) -> DashboardWidget:
        """Create metric card widget"""
        widget = DashboardWidget(
            widget_id=self._generate_widget_id(),
            widget_type=DashboardWidgetType.METRIC_CARD,
            title=title,
            data_source=data_source,
            config=config,
            position={'x': 0, 'y': 0, 'width': 3, 'height': 2}
        )
        return widget
    
    def _create_drift_heatmap_widget(
        self,
        title: str,
        data_source: str,
        config: Dict[str, Any]
    ) -> DashboardWidget:
        """Create drift heatmap widget"""
        widget = DashboardWidget(
            widget_id=self._generate_widget_id(),
            widget_type=DashboardWidgetType.HEATMAP,
            title=title,
            data_source=data_source,
            config=config,
            position={'x': 0, 'y': 0, 'width': 6, 'height': 4}
        )
        return widget
    
    def _create_time_series_widget(
        self,
        title: str,
        data_source: str,
        config: Dict[str, Any]
    ) -> DashboardWidget:
        """Create time series widget"""
        widget = DashboardWidget(
            widget_id=self._generate_widget_id(),
            widget_type=DashboardWidgetType.TIME_SERIES,
            title=title,
            data_source=data_source,
            config=config,
            position={'x': 0, 'y': 0, 'width': 6, 'height': 3}
        )
        return widget
    
    # ======================== VISUALIZATION CREATION ========================
    
    def _create_accuracy_trend_figure(
        self,
        accuracy_history: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> go.Figure:
        """Create accuracy trend visualization"""
        fig = go.Figure()
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(accuracy_history)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
            # Add traces for each metric
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                if metric in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df[metric],
                        mode='lines+markers',
                        name=metric.replace('_', ' ').title(),
                        line=dict(width=2)
                    ))
        
        # Update layout
        fig.update_layout(
            title="Model Accuracy Trends",
            xaxis_title="Time",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            hovermode='x unified',
            template=self.config['visualization']['theme']
        )
        
        return fig
    
    def _create_drift_timeline_figure(
        self,
        drift_history: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> go.Figure:
        """Create drift timeline visualization"""
        fig = go.Figure()
        
        # Group by drift type
        drift_types = defaultdict(list)
        for drift in drift_history:
            drift_types[drift['drift_type']].append(drift)
        
        # Create timeline for each drift type
        colors = px.colors.qualitative.Set3
        
        for i, (drift_type, events) in enumerate(drift_types.items()):
            timestamps = [e['timestamp'] for e in events]
            severities = [e['severity'] for e in events]
            
            # Map severity to numeric values
            severity_values = {
                'low': 1, 'medium': 2, 'high': 3, 'critical': 4
            }
            y_values = [severity_values.get(s, 1) for s in severities]
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=y_values,
                mode='markers',
                name=drift_type,
                marker=dict(
                    size=10,
                    color=colors[i % len(colors)],
                    symbol='circle'
                ),
                text=[f"{e['model_id']}<br>{e['severity']}" for e in events],
                hovertemplate='%{text}<br>%{x}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title="Drift Detection Timeline",
            xaxis_title="Time",
            yaxis_title="Severity",
            yaxis=dict(
                tickmode='array',
                tickvals=[1, 2, 3, 4],
                ticktext=['Low', 'Medium', 'High', 'Critical']
            ),
            template=self.config['visualization']['theme']
        )
        
        return fig
    
    def _create_model_comparison_figure(
        self,
        model_comparisons: Dict[str, Dict[str, float]],
        config: Dict[str, Any]
    ) -> go.Figure:
        """Create model comparison radar chart"""
        fig = go.Figure()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for model_id, scores in model_comparisons.items():
            values = [scores.get(m, 0) for m in metrics]
            values.append(values[0])  # Close the radar
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model_id
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Model Performance Comparison",
            template=self.config['visualization']['theme']
        )
        
        return fig
    
    def _create_feature_stability_figure(
        self,
        feature_stability: Dict[str, Dict[str, float]],
        config: Dict[str, Any]
    ) -> go.Figure:
        """Create feature stability heatmap"""
        # Prepare data for heatmap
        features = list(feature_stability.keys())
        models = list(set(m for f in feature_stability.values() for m in f.keys()))
        
        z_values = []
        for feature in features:
            row = []
            for model in models:
                value = feature_stability.get(feature, {}).get(model, 0)
                row.append(value)
            z_values.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=models,
            y=features,
            colorscale='RdYlGn',
            zmid=0.5,
            text=[[f"{v:.2f}" for v in row] for row in z_values],
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Feature Importance Stability",
            xaxis_title="Models",
            yaxis_title="Features",
            template=self.config['visualization']['theme']
        )
        
        return fig
    
    # ======================== DATA EXPORT ========================
    
    def export_monitoring_data(
        self,
        export_config: Dict[str, Any],
        data_filters: Dict[str, Any]
    ) -> Union[str, bytes]:
        """
        Export monitoring data in specified format.
        
        Args:
            export_config: Export configuration
            data_filters: Filters for data selection
            
        Returns:
            Exported data
        """
        # Collect data based on filters
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'filters': data_filters,
            'data': {}
        }
        
        # Get model IDs
        model_ids = data_filters.get('model_ids', list(self.accuracy_monitor.active_models))
        
        # Collect metrics for each model
        for model_id in model_ids:
            model_data = {}
            
            # Current metrics
            model_data['current_metrics'] = self.accuracy_monitor.get_current_accuracy_metrics(model_id)
            
            # Accuracy history
            if data_filters.get('include_history', True):
                if hasattr(self.accuracy_monitor, 'model_states'):
                    model_state = self.accuracy_monitor.model_states.get(model_id)
                    if model_state:
                        model_data['accuracy_history'] = list(model_state.accuracy_history)[-100:]
                        model_data['confidence_history'] = list(model_state.confidence_history)[-100:]
            
            # Drift alerts
            if data_filters.get('include_alerts', True):
                if hasattr(self.accuracy_monitor, 'model_states'):
                    model_state = self.accuracy_monitor.model_states.get(model_id)
                    if model_state:
                        model_data['drift_alerts'] = [
                            asdict(alert) for alert in model_state.drift_alerts[-50:]
                        ]
            
            export_data['data'][model_id] = model_data
        
        # Export in requested format
        export_format = export_config.get('format', 'json')
        
        if export_format == 'json':
            return json.dumps(export_data, indent=2, default=str)
            
        elif export_format == 'csv':
            # Flatten data for CSV
            rows = []
            for model_id, model_data in export_data['data'].items():
                metrics = model_data.get('current_metrics', {})
                row = {'model_id': model_id}
                row.update(metrics)
                rows.append(row)
            
            df = pd.DataFrame(rows)
            return df.to_csv(index=False)
            
        elif export_format == 'html':
            # Create HTML report
            html = self._generate_html_report(export_data)
            return html
            
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    def _generate_html_report(self, data: Dict[str, Any]) -> str:
        """Generate HTML report from data"""
        html = f"""
<html>
<head>
    <title>Accuracy Monitoring Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric-card {{ 
            border: 1px solid #ddd; 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 5px;
            background-color: #f9f9f9;
        }}
    </style>
</head>
<body>
    <h1>Accuracy Monitoring Report</h1>
    <p>Generated: {data['export_timestamp']}</p>
    
    <h2>Model Performance Summary</h2>
"""
        
        for model_id, model_data in data['data'].items():
            metrics = model_data.get('current_metrics', {})
            
            html += f"""
    <div class="metric-card">
        <h3>{model_id}</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
"""
            
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    html += f"""
            <tr>
                <td>{key.replace('_', ' ').title()}</td>
                <td>{value:.4f}</td>
            </tr>
"""
            
            html += """
        </table>
    </div>
"""
        
        html += """
</body>
</html>
"""
        return html
    
    # ======================== REAL-TIME UPDATES ========================
    
    def setup_real_time_updates(
        self,
        update_config: Dict[str, Any],
        refresh_intervals: int
    ) -> None:
        """
        Setup real-time dashboard updates.
        
        Args:
            update_config: Update configuration
            refresh_intervals: Refresh interval in seconds
        """
        dashboard_id = update_config.get('dashboard_id')
        if not dashboard_id or dashboard_id not in self.dashboards:
            raise ValueError(f"Invalid dashboard ID: {dashboard_id}")
        
        # Register update subscription
        subscription = {
            'dashboard_id': dashboard_id,
            'refresh_interval': refresh_intervals,
            'last_update': datetime.now(),
            'update_config': update_config
        }
        
        self.update_subscriptions[dashboard_id].append(subscription)
        
        self.logger.info(f"Setup real-time updates for dashboard {dashboard_id}")
    
    def _process_updates(self) -> None:
        """Process real-time updates for all dashboards"""
        current_time = datetime.now()
        
        for dashboard_id, subscriptions in self.update_subscriptions.items():
            for subscription in subscriptions:
                # Check if update is needed
                time_since_update = (current_time - subscription['last_update']).total_seconds()
                
                if time_since_update >= subscription['refresh_interval']:
                    # Update dashboard
                    self._update_dashboard(dashboard_id)
                    subscription['last_update'] = current_time
    
    def _update_dashboard(self, dashboard_id: str) -> None:
        """Update dashboard with latest data"""
        if dashboard_id not in self.dashboards:
            return
        
        dashboard = self.dashboards[dashboard_id]
        
        # Update each widget
        for widget in dashboard['widgets']:
            try:
                # Get data from source
                data_source_func = self.data_sources.get(widget.data_source)
                if not data_source_func:
                    continue
                
                # Get data for each model
                widget_data = {}
                for model_id in dashboard['model_filters']:
                    if model_id in self.accuracy_monitor.active_models:
                        widget_data[model_id] = data_source_func(model_id)
                
                # Update widget data
                widget.config['latest_data'] = widget_data
                widget.config['last_update'] = datetime.now()
                
                # Clear visualization cache for this widget
                with self.cache_lock:
                    cache_key = f"{dashboard_id}:{widget.widget_id}"
                    if cache_key in self.visualization_cache:
                        del self.visualization_cache[cache_key]
                
            except Exception as e:
                self.logger.error(f"Error updating widget {widget.widget_id}: {e}")
    
    # ======================== ADVANCED DRIFT DETECTION ========================
    
    def detect_statistical_drift(
        self,
        model_id: str,
        statistical_tests: List[str],
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detect drift using statistical tests.
        
        Args:
            model_id: Model identifier
            statistical_tests: List of statistical tests to perform
            significance_level: Significance level for tests
            
        Returns:
            Statistical drift detection results
        """
        results = {
            'model_id': model_id,
            'timestamp': datetime.now(),
            'significance_level': significance_level,
            'tests': {},
            'drift_detected': False
        }
        
        # Get model state
        if not hasattr(self.accuracy_monitor, 'model_states') or model_id not in self.accuracy_monitor.model_states:
            return {'error': f'Model {model_id} not being monitored'}
        
        model_state = self.accuracy_monitor.model_states[model_id]
        
        # Get recent predictions
        recent_predictions = list(self.accuracy_monitor.prediction_cache.get(model_id, []))
        if len(recent_predictions) < 100:
            return {'error': 'Insufficient data for statistical tests'}
        
        # Split into reference and current windows
        split_point = len(recent_predictions) // 2
        reference_data = recent_predictions[:split_point]
        current_data = recent_predictions[split_point:]
        
        # Perform requested tests
        for test_name in statistical_tests:
            if test_name == 'kolmogorov_smirnov':
                result = self._perform_ks_test(reference_data, current_data)
            elif test_name == 'chi_square':
                result = self._perform_chi_square_test(reference_data, current_data)
            elif test_name == 'population_stability_index':
                result = self._calculate_psi(reference_data, current_data)
            else:
                continue
            
            results['tests'][test_name] = result
            
            if result.get('p_value', 1) < significance_level:
                results['drift_detected'] = True
        
        return results
    
    def monitor_data_quality_impact(
        self,
        model_id: str,
        data_quality_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Monitor impact of data quality on model accuracy.
        
        Args:
            model_id: Model identifier
            data_quality_metrics: Data quality metrics to monitor
            
        Returns:
            Data quality impact analysis
        """
        results = {
            'model_id': model_id,
            'timestamp': datetime.now(),
            'quality_issues': [],
            'impact_on_accuracy': {},
            'recommendations': []
        }
        
        # Get recent predictions with features
        predictions = list(self.accuracy_monitor.prediction_cache.get(model_id, []))
        if not predictions:
            return results
        
        # Analyze data quality issues
        for metric_name, threshold in data_quality_metrics.items():
            if metric_name == 'missing_values':
                issue = self._check_missing_values(predictions, threshold)
                if issue:
                    results['quality_issues'].append(issue)
                    
            elif metric_name == 'outliers':
                issue = self._check_outliers(predictions, threshold)
                if issue:
                    results['quality_issues'].append(issue)
                    
            elif metric_name == 'data_drift':
                issue = self._check_data_drift(predictions, threshold)
                if issue:
                    results['quality_issues'].append(issue)
        
        # Analyze impact on accuracy
        if results['quality_issues']:
            # Group predictions by quality issues
            clean_predictions = [p for p in predictions if not self._has_quality_issues(p)]
            problematic_predictions = [p for p in predictions if self._has_quality_issues(p)]
            
            if clean_predictions and problematic_predictions:
                # Calculate accuracy for each group
                clean_accuracy = self._calculate_accuracy_for_predictions(clean_predictions)
                problematic_accuracy = self._calculate_accuracy_for_predictions(problematic_predictions)
                
                results['impact_on_accuracy'] = {
                    'clean_data_accuracy': clean_accuracy,
                    'problematic_data_accuracy': problematic_accuracy,
                    'accuracy_difference': clean_accuracy - problematic_accuracy
                }
                
                # Generate recommendations
                if results['impact_on_accuracy']['accuracy_difference'] > 0.05:
                    results['recommendations'].append(
                        "Significant accuracy degradation due to data quality issues. "
                        "Consider implementing data validation and cleaning pipelines."
                    )
        
        return results
    
    def analyze_feature_drift_impact(
        self,
        model_id: str,
        feature_importance_data: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analyze impact of feature drift on model performance.
        
        Args:
            model_id: Model identifier
            feature_importance_data: Feature importance scores
            
        Returns:
            Feature drift impact analysis
        """
        results = {
            'model_id': model_id,
            'timestamp': datetime.now(),
            'drifted_features': [],
            'impact_scores': {},
            'high_risk_features': []
        }
        
        # Get predictions with features
        predictions = list(self.accuracy_monitor.prediction_cache.get(model_id, []))
        if len(predictions) < 200:
            return results
        
        # Split data
        split_point = len(predictions) // 2
        reference_features = [p.features for p in predictions[:split_point]]
        current_features = [p.features for p in predictions[split_point:]]
        
        # Analyze each feature
        for feature_name, importance in feature_importance_data.items():
            if not reference_features or feature_name not in reference_features[0]:
                continue
            
            # Extract feature values
            ref_values = [f.get(feature_name, 0) for f in reference_features]
            curr_values = [f.get(feature_name, 0) for f in current_features]
            
            # Perform KS test
            ks_stat, p_value = ks_2samp(ref_values, curr_values)
            
            if p_value < 0.05:  # Significant drift
                drift_info = {
                    'feature': feature_name,
                    'importance': importance,
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'impact_score': importance * ks_stat  # Simple impact metric
                }
                
                results['drifted_features'].append(drift_info)
                results['impact_scores'][feature_name] = drift_info['impact_score']
                
                # Flag high-risk features
                if importance > 0.1 and ks_stat > 0.2:
                    results['high_risk_features'].append(feature_name)
        
        # Sort by impact
        results['drifted_features'].sort(key=lambda x: x['impact_score'], reverse=True)
        
        return results
    
    def calculate_population_stability_index(
        self,
        model_id: str,
        reference_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Calculate Population Stability Index (PSI) for model inputs.
        
        Args:
            model_id: Model identifier
            reference_data: Reference dataset (if None, uses historical data)
            
        Returns:
            PSI calculation results
        """
        results = {
            'model_id': model_id,
            'timestamp': datetime.now(),
            'psi_scores': {},
            'overall_psi': 0,
            'stability_status': 'stable'
        }
        
        # Get current predictions
        current_predictions = list(self.accuracy_monitor.prediction_cache.get(model_id, []))
        if len(current_predictions) < 100:
            return {'error': 'Insufficient data for PSI calculation'}
        
        # Extract features
        current_features = pd.DataFrame([p.features for p in current_predictions])
        
        # Get reference data
        if reference_data is None:
            # Use first half as reference
            split_point = len(current_predictions) // 2
            reference_features = pd.DataFrame([p.features for p in current_predictions[:split_point]])
        else:
            reference_features = reference_data
        
        # Calculate PSI for each feature
        for feature in current_features.columns:
            if feature not in reference_features.columns:
                continue
            
            psi = self._calculate_feature_psi(
                reference_features[feature],
                current_features[feature]
            )
            
            results['psi_scores'][feature] = psi
        
        # Calculate overall PSI
        if results['psi_scores']:
            results['overall_psi'] = np.mean(list(results['psi_scores'].values()))
            
            # Determine stability status
            if results['overall_psi'] < 0.1:
                results['stability_status'] = 'stable'
            elif results['overall_psi'] < 0.2:
                results['stability_status'] = 'moderate_shift'
            else:
                results['stability_status'] = 'significant_shift'
        
        return results
    
    # ======================== HELPER METHODS ========================
    
    def _calculate_widget_positions(
        self,
        widget_types: List[str]
    ) -> List[Dict[str, int]]:
        """Calculate widget positions in grid layout"""
        positions = []
        
        # Simple grid layout
        widgets_per_row = 2
        default_width = 6
        default_height = 4
        
        for i, widget_type in enumerate(widget_types):
            row = i // widgets_per_row
            col = i % widgets_per_row
            
            position = {
                'x': col * default_width,
                'y': row * default_height,
                'width': default_width,
                'height': default_height
            }
            
            # Adjust for specific widget types
            if widget_type == 'alert_status':
                position['height'] = 3
            elif widget_type == 'model_comparison':
                position['width'] = 12
                position['x'] = 0
            
            positions.append(position)
        
        return positions
    
    def _perform_ks_test(
        self,
        reference_data: List[Any],
        current_data: List[Any]
    ) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test"""
        # Extract numeric values for comparison
        ref_values = [p.confidence for p in reference_data if hasattr(p, 'confidence')]
        curr_values = [p.confidence for p in current_data if hasattr(p, 'confidence')]
        
        if not ref_values or not curr_values:
            return {'error': 'No data for comparison'}
        
        ks_stat, p_value = ks_2samp(ref_values, curr_values)
        
        return {
            'test_name': 'kolmogorov_smirnov',
            'ks_statistic': float(ks_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
    
    def _perform_chi_square_test(
        self,
        reference_data: List[Any],
        current_data: List[Any]
    ) -> Dict[str, Any]:
        """Perform Chi-square test"""
        # Get prediction distributions
        ref_preds = [p.prediction.get('prediction', '') for p in reference_data]
        curr_preds = [p.prediction.get('prediction', '') for p in current_data]
        
        # Create contingency table
        ref_counts = Counter(ref_preds)
        curr_counts = Counter(curr_preds)
        
        categories = list(set(ref_counts.keys()) | set(curr_counts.keys()))
        
        if len(categories) < 2:
            return {'error': 'Insufficient categories for chi-square test'}
        
        observed = np.array([
            [ref_counts.get(cat, 0) for cat in categories],
            [curr_counts.get(cat, 0) for cat in categories]
        ])
        
        chi2, p_value, dof, expected = chi2_contingency(observed)
        
        return {
            'test_name': 'chi_square',
            'chi2_statistic': float(chi2),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'significant': p_value < 0.05
        }
    
    def _calculate_psi(
        self,
        reference_data: List[Any],
        current_data: List[Any]
    ) -> Dict[str, Any]:
        """Calculate Population Stability Index"""
        # Use confidence scores for PSI
        ref_values = np.array([p.confidence for p in reference_data if hasattr(p, 'confidence')])
        curr_values = np.array([p.confidence for p in current_data if hasattr(p, 'confidence')])
        
        if len(ref_values) == 0 or len(curr_values) == 0:
            return {'error': 'No data for PSI calculation'}
        
        # Create bins
        bins = np.linspace(0, 1, 11)
        
        # Calculate distributions
        ref_hist, _ = np.histogram(ref_values, bins=bins)
        curr_hist, _ = np.histogram(curr_values, bins=bins)
        
        # Normalize
        ref_hist = ref_hist / len(ref_values)
        curr_hist = curr_hist / len(curr_values)
        
        # Calculate PSI
        psi = 0
        for i in range(len(ref_hist)):
            if ref_hist[i] > 0 and curr_hist[i] > 0:
                psi += (curr_hist[i] - ref_hist[i]) * np.log(curr_hist[i] / ref_hist[i])
        
        return {
            'test_name': 'population_stability_index',
            'psi_value': float(psi),
            'stability_status': 'stable' if psi < 0.1 else 'moderate' if psi < 0.2 else 'significant',
            'significant': psi > 0.1
        }
    
    def _calculate_feature_psi(
        self,
        reference_values: pd.Series,
        current_values: pd.Series,
        n_bins: int = 10
    ) -> float:
        """Calculate PSI for a single feature"""
        # Handle numeric features
        if pd.api.types.is_numeric_dtype(reference_values):
            # Create bins based on reference data
            _, bins = pd.qcut(reference_values, q=n_bins, retbins=True, duplicates='drop')
            
            # Calculate distributions
            ref_counts = pd.cut(reference_values, bins=bins).value_counts()
            curr_counts = pd.cut(current_values, bins=bins).value_counts()
        else:
            # Categorical features
            ref_counts = reference_values.value_counts()
            curr_counts = current_values.value_counts()
            
            # Align categories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            ref_counts = ref_counts.reindex(all_categories, fill_value=0)
            curr_counts = curr_counts.reindex(all_categories, fill_value=0)
        
        # Normalize
        ref_probs = (ref_counts + 1) / (len(reference_values) + len(ref_counts))
        curr_probs = (curr_counts + 1) / (len(current_values) + len(curr_counts))
        
        # Calculate PSI
        psi = np.sum((curr_probs - ref_probs) * np.log(curr_probs / ref_probs))
        
        return float(psi)
    
    def _check_missing_values(
        self,
        predictions: List[Any],
        threshold: float
    ) -> Optional[Dict[str, Any]]:
        """Check for missing values in features"""
        missing_counts = defaultdict(int)
        total_count = len(predictions)
        
        for pred in predictions:
            for feature, value in pred.features.items():
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    missing_counts[feature] += 1
        
        # Check if any feature exceeds threshold
        issues = []
        for feature, count in missing_counts.items():
            missing_rate = count / total_count
            if missing_rate > threshold:
                issues.append({
                    'feature': feature,
                    'missing_rate': missing_rate,
                    'count': count
                })
        
        if issues:
            return {
                'type': 'missing_values',
                'severity': 'high' if max(i['missing_rate'] for i in issues) > 0.1 else 'medium',
                'affected_features': issues
            }
        
        return None
    
    def _check_outliers(
        self,
        predictions: List[Any],
        threshold: float
    ) -> Optional[Dict[str, Any]]:
        """Check for outliers in features"""
        # Simple outlier detection using IQR
        outlier_features = []
        
        # Get all features
        if not predictions or not hasattr(predictions[0], 'features'):
            return None
        
        feature_names = list(predictions[0].features.keys())
        
        for feature in feature_names:
            values = [p.features.get(feature, 0) for p in predictions if isinstance(p.features.get(feature), (int, float))]
            
            if len(values) < 10:
                continue
            
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            outliers = [v for v in values if v < lower_bound or v > upper_bound]
            
            if len(outliers) / len(values) > 0.05:  # More than 5% outliers
                outlier_features.append({
                    'feature': feature,
                    'outlier_rate': len(outliers) / len(values),
                    'outlier_count': len(outliers)
                })
        
        if outlier_features:
            return {
                'type': 'outliers',
                'severity': 'medium',
                'affected_features': outlier_features
            }
        
        return None
    
    def _check_data_drift(
        self,
        predictions: List[Any],
        threshold: float
    ) -> Optional[Dict[str, Any]]:
        """Check for data drift"""
        # Simple drift check using distribution comparison
        if len(predictions) < 200:
            return None
        
        # Split data
        split_point = len(predictions) // 2
        first_half = predictions[:split_point]
        second_half = predictions[split_point:]
        
        drifted_features = []
        
        # Check confidence drift
        conf1 = [p.confidence for p in first_half]
        conf2 = [p.confidence for p in second_half]
        
        ks_stat, p_value = ks_2samp(conf1, conf2)
        
        if p_value < threshold:
            drifted_features.append({
                'feature': 'prediction_confidence',
                'ks_statistic': ks_stat,
                'p_value': p_value
            })
        
        if drifted_features:
            return {
                'type': 'data_drift',
                'severity': 'high' if p_value < 0.01 else 'medium',
                'drifted_features': drifted_features
            }
        
        return None
    
    def _has_quality_issues(self, prediction: Any) -> bool:
        """Check if prediction has quality issues"""
        # Simple check for missing values
        for value in prediction.features.values():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return True
        return False
    
    def _calculate_accuracy_for_predictions(self, predictions: List[Any]) -> float:
        """Calculate accuracy for a list of predictions"""
        correct = sum(
            1 for p in predictions
            if (p.actual and 
                p.prediction.get('prediction') == ('fraud' if p.actual.get('is_fraud') else 'legitimate'))
        )
        total = sum(1 for p in predictions if p.actual is not None)
        
        return correct / total if total > 0 else 0
    
    def _generate_dashboard_id(self) -> str:
        """Generate unique dashboard ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        return f"dashboard_{timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
    
    def _generate_widget_id(self) -> str:
        """Generate unique widget ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        return f"widget_{timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
    
    # ======================== PUBLIC API ========================
    
    def get_dashboard_status(self, dashboard_id: str) -> Dict[str, Any]:
        """Get status of a dashboard"""
        if dashboard_id not in self.dashboards:
            return {'error': f'Dashboard {dashboard_id} not found'}
        
        dashboard = self.dashboards[dashboard_id]
        
        return {
            'dashboard_id': dashboard_id,
            'title': dashboard['title'],
            'created_at': dashboard['created_at'].isoformat(),
            'widget_count': len(dashboard['widgets']),
            'model_filters': dashboard['model_filters'],
            'last_update': max(
                (w.config.get('last_update', datetime.min) for w in dashboard['widgets']),
                default=datetime.min
            ).isoformat()
        }
    
    def get_widget_data(
        self,
        widget_id: str,
        format: str = 'json'
    ) -> Union[Dict[str, Any], go.Figure]:
        """Get data or visualization for a widget"""
        if widget_id not in self.widgets:
            return {'error': f'Widget {widget_id} not found'}
        
        widget = self.widgets[widget_id]
        
        if format == 'json':
            # Return raw data
            return {
                'widget_id': widget_id,
                'title': widget.title,
                'type': widget.widget_type.value,
                'data': widget.config.get('latest_data', {}),
                'last_update': widget.config.get('last_update', datetime.min).isoformat()
            }
        elif format == 'figure':
            # Generate visualization
            return self._generate_widget_visualization(widget)
        else:
            return {'error': f'Unsupported format: {format}'}
    
    def _generate_widget_visualization(self, widget: DashboardWidget) -> go.Figure:
        """Generate visualization for a widget"""
        # This would create appropriate visualizations based on widget type
        # Placeholder implementation
        fig = go.Figure()
        fig.update_layout(title=widget.title)
        return fig
    
    def shutdown(self) -> None:
        """Shutdown dashboard manager"""
        self.logger.info("Shutting down DashboardManager")
        
        # Stop update thread
        self.stop_updates.set()
        if self.update_thread:
            self.update_thread.join(timeout=5)
        
        # Clear caches
        self.visualization_cache.clear()
        
        self.logger.info("DashboardManager shutdown complete")


# ======================== USAGE EXAMPLES ========================

if __name__ == "__main__":
    # Example usage demonstrating the advanced monitoring features
    from datetime import datetime, timedelta
    import numpy as np
    
    print("=== Advanced Accuracy Monitoring Examples ===\n")
    
    # Mock components for demonstration
    class MockAccuracyMonitor(RealTimeAccuracyMonitor):
        def __init__(self):
            self.model_states = {}
            self.active_models = {'fraud_model_v1'}
            self.monitoring_manager = type('obj', (object,), {
                'add_alert_callback': lambda x: None
            })
            self.prediction_cache = {
                'fraud_model_v1': []
            }
            
        def get_current_accuracy_metrics(self, model_id):
            return {
                'model_id': model_id,
                'current_accuracy': 0.92,
                'trend': {'trend': 'stable'}
            }
    
    # Initialize components
    accuracy_monitor = MockAccuracyMonitor()
    
    # Example 1: AlertingSystem
    print("1. Configuring AlertingSystem...")
    
    alert_config = {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'from_address': 'alerts@frauddetection.com',
        'slack_webhook_url': 'https://hooks.slack.com/services/XXX',
        'alert_cooldown_minutes': 15
    }
    
    alerting_system = AlertingSystem(accuracy_monitor, alert_config)
    
    # Configure alert rules
    rule = alerting_system.configure_alert_rule(
        rule_name="High Accuracy Drop",
        condition={
            'drift_types': ['accuracy_drift'],
            'min_severity': 'high',
            'threshold': 0.05
        },
        severity=DriftSeverity.HIGH,
        channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
        cooldown_minutes=30
    )
    
    print(f"   Created alert rule: {rule.name} (ID: {rule.rule_id})")
    
    # Configure automated responses
    response_rules = {
        "accuracy_drop": {
            "trigger_condition": {
                "alert_types": ["accuracy_drift"],
                "min_severity": "high",
                "metric_threshold": 0.85
            },
            "actions": ["notify_team", "increase_monitoring"],
            "safety_limits": {"max_alerts_per_hour": 5}
        },
        "severe_drift": {
            "trigger_condition": {
                "min_severity": "critical"
            },
            "actions": ["disable_model", "enable_fallback"],
            "safety_limits": {"max_rollbacks_per_hour": 1}
        }
    }
    
    alerting_system.configure_automated_responses(
        response_rules,
        safety_limits={'max_alerts_per_hour': 10}
    )
    
    print("   Configured 2 automated response rules\n")
    
    # Example 2: Trigger an alert
    print("2. Triggering test alert...")
    
    alert_id = alerting_system.trigger_accuracy_alert(
        alert_type="accuracy_drift",
        model_id="fraud_model_v1",
        severity=DriftSeverity.HIGH,
        alert_data={
            'current_accuracy': 0.82,
            'baseline_accuracy': 0.92,
            'drift_magnitude': 0.10
        }
    )
    
    print(f"   Triggered alert: {alert_id}")
    print(f"   Alert summary: {alerting_system.get_alert_summary()}\n")
    
    # Example 3: DashboardManager
    print("3. Creating monitoring dashboard...")
    
    dashboard_config = {
        'title': 'Fraud Detection Accuracy Dashboard',
        'widgets': ['accuracy_trend', 'drift_detection', 'alert_status'],
        'refresh_interval': 30,
        'time_range': '24h'
    }
    
    dashboard_manager = DashboardManager(accuracy_monitor, dashboard_config)
    
    dashboard = dashboard_manager.create_accuracy_dashboard(
        dashboard_config,
        model_filters=['fraud_model_v1']
    )
    
    print(f"   Created dashboard: {dashboard['dashboard_id']}")
    print(f"   Widgets: {len(dashboard['widgets'])}")
    for widget in dashboard['widgets']:
        print(f"     - {widget.title} ({widget.widget_type.value})")
    print()
    
    # Example 4: Statistical drift detection
    print("4. Performing statistical drift detection...")
    
    drift_results = dashboard_manager.detect_statistical_drift(
        model_id='fraud_model_v1',
        statistical_tests=['kolmogorov_smirnov', 'chi_square', 'population_stability_index'],
        significance_level=0.05
    )
    
    if 'error' not in drift_results:
        print(f"   Drift detected: {drift_results.get('drift_detected', False)}")
        for test_name, result in drift_results.get('tests', {}).items():
            print(f"   {test_name}: p-value={result.get('p_value', 'N/A'):.4f}")
    else:
        print(f"   {drift_results['error']}")
    print()
    
    # Example 5: Data quality monitoring
    print("5. Monitoring data quality impact...")
    
    quality_metrics = {
        'missing_values': 0.05,  # 5% threshold
        'outliers': 1.5,        # 1.5 IQR
        'data_drift': 0.05      # p-value threshold
    }
    
    quality_results = dashboard_manager.monitor_data_quality_impact(
        model_id='fraud_model_v1',
        data_quality_metrics=quality_metrics
    )
    
    print(f"   Quality issues found: {len(quality_results['quality_issues'])}")
    for issue in quality_results['quality_issues']:
        print(f"     - {issue.get('type', 'unknown')} (severity: {issue.get('severity', 'unknown')})")
    print()
    
    # Example 6: PSI calculation
    print("6. Calculating Population Stability Index...")
    
    psi_results = dashboard_manager.calculate_population_stability_index(
        model_id='fraud_model_v1'
    )
    
    if 'error' not in psi_results:
        print(f"   Overall PSI: {psi_results.get('overall_psi', 0):.3f}")
        print(f"   Stability status: {psi_results.get('stability_status', 'unknown')}")
    else:
        print(f"   {psi_results['error']}")
    print()
    
    # Example 7: Export monitoring data
    print("7. Exporting monitoring data...")
    
    export_data = dashboard_manager.export_monitoring_data(
        export_config={'format': 'json'},
        data_filters={
            'model_ids': ['fraud_model_v1'],
            'include_history': True,
            'include_alerts': True
        }
    )
    
    print(f"   Exported data size: {len(export_data)} characters")
    print()
    
    # Example 8: Alert resolution
    print("8. Resolving alert...")
    
    resolution_actions = [
        {
            'action': 'investigated',
            'timestamp': datetime.now().isoformat(),
            'notes': 'Temporary data quality issue resolved'
        },
        {
            'action': 'model_updated',
            'timestamp': datetime.now().isoformat(),
            'notes': 'Retrained model with recent data'
        }
    ]
    
    resolved = alerting_system.track_alert_resolution(alert_id, resolution_actions)
    print(f"   Alert resolved: {resolved}")
    
    # Cleanup
    alerting_system.shutdown()
    dashboard_manager.shutdown()
    
    print("\n=== All examples completed successfully! ===")