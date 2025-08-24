"""
Comprehensive test suite for ProductionAlertSystem
Tests all aspects of alert generation, notification, response handling, and monitoring
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, patch, call
from collections import defaultdict, deque
import logging

from production_monitoring.alert_system import (
    AlertSeverity,
    AlertType,
    Alert,
    AlertRule,
    AlertResponse,
    ProductionAlertSystem
)


class TestAlertDataClasses:
    """Test Alert, AlertRule, and AlertResponse dataclasses"""
    
    def test_alert_creation(self):
        """Test Alert dataclass creation"""
        alert = Alert(
            alert_id="test_alert_1",
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.HIGH,
            source="test_system",
            message="Test alert message",
            details={"metric": "cpu", "value": 95},
            timestamp=datetime.now()
        )
        
        assert alert.alert_id == "test_alert_1"
        assert alert.alert_type == AlertType.PERFORMANCE
        assert alert.severity == AlertSeverity.HIGH
        assert alert.source == "test_system"
        assert alert.message == "Test alert message"
        assert alert.details["metric"] == "cpu"
        assert alert.status == "active"  # Default
        assert alert.acknowledged_by is None
        assert alert.acknowledged_at is None
        assert alert.resolved_at is None
        assert alert.response_actions == []
        assert alert.related_alerts == []
    
    def test_alert_rule_creation(self):
        """Test AlertRule dataclass creation"""
        rule = AlertRule(
            rule_id="cpu_rule",
            name="CPU High Usage",
            alert_type=AlertType.RESOURCE,
            condition={"metric": "cpu_usage", "operator": ">"},
            severity=AlertSeverity.HIGH,
            threshold=90.0,
            duration=30,
            cooldown=300,
            actions=["optimize", "notify"]
        )
        
        assert rule.rule_id == "cpu_rule"
        assert rule.name == "CPU High Usage"
        assert rule.alert_type == AlertType.RESOURCE
        assert rule.condition["metric"] == "cpu_usage"
        assert rule.condition["operator"] == ">"
        assert rule.severity == AlertSeverity.HIGH
        assert rule.threshold == 90.0
        assert rule.duration == 30
        assert rule.cooldown == 300
        assert rule.actions == ["optimize", "notify"]
        assert rule.enabled == True  # Default
        assert rule.last_triggered is None
    
    def test_alert_response_creation(self):
        """Test AlertResponse dataclass creation"""
        response = AlertResponse(
            response_id="resp_1",
            alert_id="alert_1",
            action_type="optimize",
            status="pending",
            started_at=datetime.now()
        )
        
        assert response.response_id == "resp_1"
        assert response.alert_id == "alert_1"
        assert response.action_type == "optimize"
        assert response.status == "pending"
        assert response.started_at is not None
        assert response.completed_at is None
        assert response.result is None
        assert response.error is None


class TestProductionAlertSystemInitialization:
    """Test ProductionAlertSystem initialization"""
    
    def test_initialization(self):
        """Test system initialization"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        assert system.monitor == mock_monitor
        assert system.optimization_engine == mock_optimization
        assert system.alerts == {}
        assert system.alert_rules != {}  # Should have default rules
        assert system.alert_responses == {}
        assert len(system.alert_history) == 0
        assert system.max_active_alerts == 100
        assert system.alert_retention_hours == 24
        assert system.response_timeout == 30
        assert system.enabled_channels == ['log']
        
        # Verify monitor callback registration
        mock_monitor.register_alert_callback.assert_called_once()
    
    def test_default_rules_initialization(self):
        """Test that default alert rules are initialized"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        # Check for specific default rules
        assert "cpu_critical" in system.alert_rules
        assert "memory_critical" in system.alert_rules
        assert "high_error_rate" in system.alert_rules
        assert "system_health_critical" in system.alert_rules
        assert "security_violation" in system.alert_rules
        assert "slow_response" in system.alert_rules
        assert "agent_failure" in system.alert_rules
        
        # Verify rule properties
        cpu_rule = system.alert_rules["cpu_critical"]
        assert cpu_rule.alert_type == AlertType.RESOURCE
        assert cpu_rule.severity == AlertSeverity.CRITICAL
        assert cpu_rule.threshold == 95.0
        assert cpu_rule.actions == ["optimize", "notify"]


class TestAlertSystemStartStop:
    """Test starting and stopping the alert system"""
    
    def test_start_alert_system(self):
        """Test starting the alert system"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        result = system.start_alert_system()
        
        assert result['started'] == True
        assert 'active_rules' in result
        assert result['enabled_channels'] == ['log']
        assert 'timestamp' in result
        
        # Wait a bit for threads to start
        time.sleep(0.1)
        
        # Verify threads are running
        assert system._alert_thread is not None
        assert system._alert_thread.is_alive()
        assert system._response_thread is not None
        assert system._response_thread.is_alive()
        
        # Stop the system
        system.stop_alert_system()
    
    def test_start_already_running(self):
        """Test starting when already running"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        # Start once
        system.start_alert_system()
        time.sleep(0.1)
        
        # Try to start again
        result = system.start_alert_system()
        
        assert result['started'] == False
        assert 'error' in result
        
        # Stop the system
        system.stop_alert_system()
    
    def test_stop_alert_system(self):
        """Test stopping the alert system"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        # Start the system
        system.start_alert_system()
        time.sleep(0.1)
        
        # Stop the system
        result = system.stop_alert_system()
        
        assert result['stopped'] == True
        assert 'total_alerts' in result
        assert 'active_alerts' in result
        assert 'timestamp' in result
        
        # Verify threads are stopped
        assert system._stop_event.is_set()


class TestAlertCreation:
    """Test alert creation and management"""
    
    def test_create_alert(self):
        """Test creating a new alert"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        alert = Alert(
            alert_id="test_1",
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.HIGH,
            source="test",
            message="Test alert",
            details={},
            timestamp=datetime.now()
        )
        
        system._create_alert(alert)
        
        assert "test_1" in system.alerts
        assert system.alerts["test_1"] == alert
        assert len(system.alert_history) == 1
        assert system.alert_counts[AlertType.PERFORMANCE.value] == 1
    
    def test_alert_limit_enforcement(self):
        """Test that alert limit is enforced"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        system.max_active_alerts = 5  # Set low limit for testing
        
        # Create more alerts than the limit
        for i in range(10):
            alert = Alert(
                alert_id=f"test_{i}",
                alert_type=AlertType.PERFORMANCE,
                severity=AlertSeverity.LOW,
                source="test",
                message=f"Alert {i}",
                details={},
                timestamp=datetime.now()
            )
            
            # Mark some as resolved for cleanup
            if i < 5:
                alert.status = 'resolved'
                alert.resolved_at = datetime.now()
            
            system._create_alert(alert)
        
        # Should not exceed max_active_alerts
        assert len(system.alerts) <= system.max_active_alerts
    
    def test_alert_suppression(self):
        """Test alert suppression rules"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        # Add suppression rule
        system.suppression_rules['test_rule'] = {
            'alert_type': AlertType.PERFORMANCE.value,
            'source': 'test_source',
            'window_seconds': 60,
            'max_alerts': 2
        }
        
        # Create alerts that should trigger suppression
        for i in range(5):
            alert = Alert(
                alert_id=f"test_{i}",
                alert_type=AlertType.PERFORMANCE,
                severity=AlertSeverity.MEDIUM,
                source="test_source",
                message=f"Alert {i}",
                details={},
                timestamp=datetime.now()
            )
            system._create_alert(alert)
        
        # Only first 2 alerts should be created, rest suppressed
        assert len([a for a in system.alerts.values() if a.source == "test_source"]) <= 2
        assert system.suppressed_alerts[AlertType.PERFORMANCE.value] >= 3


class TestAlertRuleEvaluation:
    """Test alert rule evaluation and triggering"""
    
    def test_evaluate_resource_rule(self):
        """Test evaluating resource-based rules"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        rule = AlertRule(
            rule_id="test_cpu",
            name="Test CPU",
            alert_type=AlertType.RESOURCE,
            condition={"metric": "cpu_usage", "operator": ">"},
            severity=AlertSeverity.HIGH,
            threshold=80.0,
            duration=0,
            cooldown=0,
            actions=[]
        )
        
        system_status = {
            'system_status': {
                'system1': {'cpu_usage': 85.0},  # Above threshold
                'system2': {'cpu_usage': 75.0}   # Below threshold
            }
        }
        
        agent_status = {'agent_status': {}}
        
        # Should trigger for system1
        result = system._evaluate_rule_condition(rule, system_status, agent_status)
        assert result == True
    
    def test_evaluate_error_rate_rule(self):
        """Test evaluating error rate rules"""
        mock_monitor = Mock()
        mock_monitor.request_counts = defaultdict(int)
        mock_monitor.request_counts['system1'] = 100
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        rule = AlertRule(
            rule_id="test_errors",
            name="Test Errors",
            alert_type=AlertType.ERROR_RATE,
            condition={"metric": "error_rate", "operator": ">"},
            severity=AlertSeverity.HIGH,
            threshold=0.05,  # 5% error rate
            duration=0,
            cooldown=0,
            actions=[]
        )
        
        system_status = {
            'system_status': {
                'system1': {'error_count': 10}  # 10/100 = 10% error rate
            }
        }
        
        agent_status = {'agent_status': {}}
        
        result = system._evaluate_rule_condition(rule, system_status, agent_status)
        assert result == True
    
    def test_evaluate_agent_failure_rule(self):
        """Test evaluating agent failure rules"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        rule = AlertRule(
            rule_id="test_agent",
            name="Test Agent",
            alert_type=AlertType.AGENT_FAILURE,
            condition={"metric": "agent_status", "operator": "==", "value": "error"},
            severity=AlertSeverity.HIGH,
            threshold=0,
            duration=0,
            cooldown=0,
            actions=[]
        )
        
        system_status = {'system_status': {}}
        agent_status = {
            'agent_status': {
                'agent1': {'status': 'error'},  # Matches condition
                'agent2': {'status': 'running'}  # Doesn't match
            }
        }
        
        result = system._evaluate_rule_condition(rule, system_status, agent_status)
        assert result == True
    
    def test_rule_cooldown(self):
        """Test rule cooldown period"""
        mock_monitor = Mock()
        mock_monitor.monitor_all_systems.return_value = {
            'system_status': {
                'system1': {'cpu_usage': 96.0}
            }
        }
        mock_monitor.monitor_all_agents.return_value = {'agent_status': {}}
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        # Get the CPU critical rule
        rule = system.alert_rules['cpu_critical']
        rule.cooldown = 1  # Set short cooldown for testing
        
        # First trigger
        system._check_alert_rules()
        first_trigger_time = rule.last_triggered
        assert first_trigger_time is not None
        
        # Immediate second check - should not trigger due to cooldown
        initial_alert_count = len(system.alerts)
        system._check_alert_rules()
        assert len(system.alerts) == initial_alert_count  # No new alert
        
        # Wait for cooldown to expire
        time.sleep(1.1)
        
        # Should trigger again after cooldown
        system._check_alert_rules()
        assert rule.last_triggered > first_trigger_time


class TestValueComparison:
    """Test value comparison operators"""
    
    def test_compare_values(self):
        """Test all comparison operators"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        # Test greater than
        assert system._compare_values(10, ">", 5) == True
        assert system._compare_values(5, ">", 10) == False
        
        # Test less than
        assert system._compare_values(5, "<", 10) == True
        assert system._compare_values(10, "<", 5) == False
        
        # Test greater than or equal
        assert system._compare_values(10, ">=", 10) == True
        assert system._compare_values(11, ">=", 10) == True
        assert system._compare_values(9, ">=", 10) == False
        
        # Test less than or equal
        assert system._compare_values(10, "<=", 10) == True
        assert system._compare_values(9, "<=", 10) == True
        assert system._compare_values(11, "<=", 10) == False
        
        # Test equality
        assert system._compare_values(10, "==", 10) == True
        assert system._compare_values(9, "==", 10) == False
        
        # Test inequality
        assert system._compare_values(9, "!=", 10) == True
        assert system._compare_values(10, "!=", 10) == False
        
        # Test unknown operator
        assert system._compare_values(10, "unknown", 5) == False


class TestAlertNotifications:
    """Test alert notification system"""
    
    def test_send_notifications(self):
        """Test sending notifications through enabled channels"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        system.enabled_channels = ['email', 'slack']
        
        alert = Alert(
            alert_id="test_1",
            alert_type=AlertType.SYSTEM_FAILURE,  # Fixed: CRITICAL is not a valid AlertType
            severity=AlertSeverity.CRITICAL,
            source="test",
            message="Critical alert",
            details={},
            timestamp=datetime.now()
        )
        
        with patch.object(system._executor, 'submit') as mock_submit:
            system._send_notifications(alert)
            
            # Should submit notification tasks for enabled channels
            assert mock_submit.call_count == 2
    
    def test_log_alert_severity_levels(self):
        """Test alert logging at different severity levels"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        with patch.object(system.logger, 'critical') as mock_critical, \
             patch.object(system.logger, 'error') as mock_error, \
             patch.object(system.logger, 'warning') as mock_warning, \
             patch.object(system.logger, 'info') as mock_info:
            
            # Critical alert
            alert_critical = Alert(
                alert_id="crit",
                alert_type=AlertType.SYSTEM_FAILURE,
                severity=AlertSeverity.CRITICAL,
                source="test",
                message="Critical",
                details={},
                timestamp=datetime.now()
            )
            system._log_alert(alert_critical)
            mock_critical.assert_called_once()
            
            # High alert
            alert_high = Alert(
                alert_id="high",
                alert_type=AlertType.PERFORMANCE,
                severity=AlertSeverity.HIGH,
                source="test",
                message="High",
                details={},
                timestamp=datetime.now()
            )
            system._log_alert(alert_high)
            mock_error.assert_called_once()
            
            # Medium alert
            alert_medium = Alert(
                alert_id="med",
                alert_type=AlertType.PERFORMANCE,
                severity=AlertSeverity.MEDIUM,
                source="test",
                message="Medium",
                details={},
                timestamp=datetime.now()
            )
            system._log_alert(alert_medium)
            mock_warning.assert_called_once()
            
            # Low alert
            alert_low = Alert(
                alert_id="low",
                alert_type=AlertType.PERFORMANCE,
                severity=AlertSeverity.LOW,
                source="test",
                message="Low",
                details={},
                timestamp=datetime.now()
            )
            system._log_alert(alert_low)
            mock_info.assert_called_once()
    
    def test_log_notification_not_implemented(self):
        """Test that _log_notification raises NotImplementedError"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        alert = Alert(
            alert_id="test",
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.LOW,
            source="test",
            message="Test",
            details={},
            timestamp=datetime.now()
        )
        
        with pytest.raises(NotImplementedError):
            system._log_notification(alert)


class TestResponseActions:
    """Test alert response actions"""
    
    def test_create_response_action(self):
        """Test creating response actions"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        alert = Alert(
            alert_id="test_1",
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.HIGH,
            source="test",
            message="Test",
            details={},
            timestamp=datetime.now()
        )
        
        system._create_response_action(alert, "optimize")
        
        response_id = f"response_test_1_optimize"
        assert response_id in system.alert_responses
        
        response = system.alert_responses[response_id]
        assert response.alert_id == "test_1"
        assert response.action_type == "optimize"
        assert response.status == "pending"
    
    def test_execute_response_action_optimize(self):
        """Test executing optimization response action"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        mock_optimization.optimize_resource_allocation.return_value = {'status': 'optimized'}
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        alert = Alert(
            alert_id="test_1",
            alert_type=AlertType.RESOURCE,
            severity=AlertSeverity.HIGH,
            source="test",
            message="Test",
            details={},
            timestamp=datetime.now()
        )
        system.alerts["test_1"] = alert
        
        response = AlertResponse(
            response_id="resp_1",
            alert_id="test_1",
            action_type="optimize",
            status="pending",
            started_at=datetime.now()
        )
        
        system._execute_response_action(response)
        
        assert response.status == "completed"
        assert response.result == {'status': 'optimized'}
        assert response.completed_at is not None
        
        # Check response time tracking
        assert len(system.response_times) == 1
    
    def test_execute_response_action_failure(self):
        """Test response action failure handling"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        mock_optimization.optimize_resource_allocation.side_effect = Exception("Optimization failed")
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        alert = Alert(
            alert_id="test_1",
            alert_type=AlertType.RESOURCE,
            severity=AlertSeverity.HIGH,
            source="test",
            message="Test",
            details={},
            timestamp=datetime.now()
        )
        system.alerts["test_1"] = alert
        
        response = AlertResponse(
            response_id="resp_1",
            alert_id="test_1",
            action_type="optimize",
            status="pending",
            started_at=datetime.now()
        )
        
        system._execute_response_action(response)
        
        assert response.status == "failed"
        assert "Optimization trigger failed" in response.error
        assert response.completed_at is not None
    
    def test_execute_unknown_action(self):
        """Test executing unknown action type"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        response = AlertResponse(
            response_id="resp_1",
            alert_id="test_1",
            action_type="unknown_action",
            status="pending",
            started_at=datetime.now()
        )
        
        system._execute_response_action(response)
        
        assert response.status == "failed"
        assert "Unknown action type" in response.error
    
    def test_response_timeout_check(self):
        """Test response timeout checking"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        system.response_timeout = 0.1  # Very short timeout for testing
        
        # Create an in-progress response that will timeout
        response = AlertResponse(
            response_id="resp_1",
            alert_id="test_1",
            action_type="optimize",
            status="in_progress",
            started_at=datetime.now() - timedelta(seconds=1)  # Started 1 second ago
        )
        system.alert_responses["resp_1"] = response
        
        system._check_response_timeouts()
        
        assert response.status == "failed"
        assert "Response timeout" in response.error
        assert response.completed_at is not None


class TestMonitorIntegration:
    """Test integration with monitor callbacks"""
    
    def test_handle_monitor_alert(self):
        """Test handling alerts from monitor"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        # Test security alert
        alert_data = {
            'type': 'security_breach',
            'severity': 'critical',
            'source': 'firewall',
            'message': 'Unauthorized access detected'
        }
        
        system._handle_monitor_alert(alert_data)
        
        # Should create an alert
        assert len(system.alerts) == 1
        created_alert = list(system.alerts.values())[0]
        assert created_alert.alert_type == AlertType.SECURITY
        assert created_alert.severity == AlertSeverity.CRITICAL
        assert created_alert.source == 'firewall'
    
    def test_handle_monitor_alert_type_mapping(self):
        """Test alert type mapping from monitor data"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        test_cases = [
            ('security_issue', AlertType.SECURITY),
            ('resource_exhaustion', AlertType.RESOURCE),
            ('error_spike', AlertType.ERROR_RATE),
            ('unknown_type', AlertType.PERFORMANCE)  # Default
        ]
        
        for monitor_type, expected_alert_type in test_cases:
            alert_data = {
                'type': monitor_type,
                'severity': 'medium',
                'source': 'test'
            }
            
            system._handle_monitor_alert(alert_data)
            
            # Get the latest alert
            latest_alert = list(system.alerts.values())[-1]
            assert latest_alert.alert_type == expected_alert_type


class TestAlertCleanup:
    """Test alert cleanup and maintenance"""
    
    def test_cleanup_old_alerts(self):
        """Test cleaning up old resolved alerts"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        system.alert_retention_hours = 1  # Short retention for testing
        
        # Create old resolved alert
        old_alert = Alert(
            alert_id="old_1",
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.LOW,
            source="test",
            message="Old alert",
            details={},
            timestamp=datetime.now() - timedelta(hours=2)
        )
        old_alert.status = 'resolved'
        old_alert.resolved_at = datetime.now() - timedelta(hours=2)
        system.alerts["old_1"] = old_alert
        
        # Create recent resolved alert
        recent_alert = Alert(
            alert_id="recent_1",
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.LOW,
            source="test",
            message="Recent alert",
            details={},
            timestamp=datetime.now()
        )
        recent_alert.status = 'resolved'
        recent_alert.resolved_at = datetime.now()
        system.alerts["recent_1"] = recent_alert
        
        # Create active alert
        active_alert = Alert(
            alert_id="active_1",
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.LOW,
            source="test",
            message="Active alert",
            details={},
            timestamp=datetime.now() - timedelta(hours=2)
        )
        system.alerts["active_1"] = active_alert
        
        system._cleanup_old_alerts()
        
        # Old resolved alert should be removed
        assert "old_1" not in system.alerts
        # Recent resolved alert should remain
        assert "recent_1" in system.alerts
        # Active alert should remain regardless of age
        assert "active_1" in system.alerts


class TestCriticalIssueDetection:
    """Test critical issue detection methods"""
    
    def test_detect_critical_issues(self):
        """Test detecting critical issues"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        # Create critical alerts
        critical_alert = Alert(
            alert_id="crit_1",
            alert_type=AlertType.SYSTEM_FAILURE,
            severity=AlertSeverity.CRITICAL,
            source="database",
            message="Database failure",
            details={},
            timestamp=datetime.now()
        )
        critical_alert.response_actions = ['restart', 'notify']
        system.alerts["crit_1"] = critical_alert
        
        # Create non-critical alert
        normal_alert = Alert(
            alert_id="norm_1",
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.MEDIUM,
            source="api",
            message="Slow response",
            details={},
            timestamp=datetime.now()
        )
        system.alerts["norm_1"] = normal_alert
        
        result = system.detect_critical_issues()
        
        assert result['detection'] == 'critical_issues'
        assert result['critical_count'] == 1
        assert len(result['critical_alerts']) == 1
        assert result['critical_alerts'][0]['alert_id'] == 'crit_1'


class TestPerformanceAlerts:
    """Test performance alert generation"""
    
    def test_generate_performance_alerts(self):
        """Test generating performance degradation alerts"""
        mock_monitor = Mock()
        mock_monitor.track_performance_metrics.return_value = {
            'current_metrics': {
                'system_performance': 0.75,  # Below 0.8 threshold
                'latency_p95': 600  # Above 500ms threshold
            }
        }
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        result = system.generate_performance_alerts()
        
        assert result['generation'] == 'performance_alerts'
        assert result['alerts_generated'] == 2
        assert len(result['alert_ids']) == 2
        
        # Check that alerts were created
        assert len(system.alerts) == 2
        
        # Verify alert properties
        for alert in system.alerts.values():
            assert alert.alert_type == AlertType.PERFORMANCE
            assert 'optimize' in alert.response_actions
    
    def test_generate_performance_alerts_no_metrics(self):
        """Test handling when metrics are unavailable"""
        mock_monitor = Mock()
        mock_monitor.track_performance_metrics.return_value = {}
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        result = system.generate_performance_alerts()
        
        assert result['generation'] == 'performance_alerts'
        assert 'error' in result


class TestSecurityViolationDetection:
    """Test security violation detection"""
    
    def test_detect_security_violations(self):
        """Test detecting security violations"""
        mock_monitor = Mock()
        mock_monitor.monitor_security_status.return_value = {
            'security_events': [
                {
                    'type': 'unauthorized_access',
                    'severity': 'critical',
                    'source': '192.168.1.100'
                },
                {
                    'type': 'port_scan',
                    'severity': 'high',
                    'source': '10.0.0.50'
                },
                {
                    'type': 'failed_login',
                    'severity': 'low',
                    'source': 'user123'
                }
            ],
            'security_score': 0.4
        }
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        result = system.detect_security_violations()
        
        assert result['detection'] == 'security_violations'
        assert result['violations_detected'] == 2  # Only high and critical
        assert len(result['alert_ids']) == 2
        assert result['security_score'] == 0.4
        
        # Verify alerts were created with correct properties
        for alert in system.alerts.values():
            assert alert.alert_type == AlertType.SECURITY
            assert alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]
            assert 'isolate' in alert.response_actions
            assert 'notify' in alert.response_actions


class TestSystemFailureMonitoring:
    """Test system failure monitoring"""
    
    def test_monitor_system_failures(self):
        """Test monitoring system failures"""
        mock_monitor = Mock()
        mock_monitor.monitor_all_systems.return_value = {
            'system_status': {
                'database': {
                    'health_score': 0,  # Failed
                    'status': 'critical'
                },
                'api_server': {
                    'health_score': 0.9,
                    'status': 'healthy'
                },
                'cache': {
                    'health_score': 0.3,
                    'status': 'critical'  # Critical status
                }
            }
        }
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        result = system.monitor_system_failures()
        
        assert result['monitoring'] == 'system_failures'
        assert result['failures_detected'] == 2  # database and cache
        assert 'database' in result['failed_systems']
        assert 'cache' in result['failed_systems']
        
        # Verify failure alerts
        for alert in system.alerts.values():
            assert alert.alert_type == AlertType.SYSTEM_FAILURE
            assert alert.severity == AlertSeverity.CRITICAL
            assert 'restart' in alert.response_actions
            assert 'notify' in alert.response_actions


class TestAgentCommunicationTracking:
    """Test agent communication issue tracking"""
    
    def test_track_agent_communication_issues(self):
        """Test tracking agent communication issues"""
        mock_monitor = Mock()
        mock_monitor.monitor_all_agents.return_value = {
            'agent_status': {
                'agent1': {
                    'communication_latency': 150,  # Above 100ms threshold
                    'coordination_score': 0.8
                },
                'agent2': {
                    'communication_latency': 50,
                    'coordination_score': 0.6  # Below 0.7 threshold
                },
                'agent3': {
                    'communication_latency': 250,  # High latency
                    'coordination_score': 0.5  # Poor coordination
                }
            }
        }
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        result = system.track_agent_communication_issues()
        
        assert result['tracking'] == 'agent_communication_issues'
        assert result['issues_detected'] == 4  # 2 latency + 2 coordination issues
        
        # Verify communication alerts
        comm_alerts = [a for a in system.alerts.values() if a.alert_type == AlertType.COMMUNICATION]
        assert len(comm_alerts) == 4
        
        for alert in comm_alerts:
            assert 'optimize' in alert.response_actions


class TestAlertPrioritization:
    """Test alert prioritization"""
    
    def test_prioritize_alerts(self):
        """Test prioritizing alerts by severity"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        # Create alerts with different severities
        alerts = [
            ("low_1", AlertSeverity.LOW, datetime.now() - timedelta(minutes=10)),
            ("med_1", AlertSeverity.MEDIUM, datetime.now() - timedelta(minutes=5)),
            ("high_1", AlertSeverity.HIGH, datetime.now() - timedelta(minutes=3)),
            ("crit_1", AlertSeverity.CRITICAL, datetime.now() - timedelta(minutes=1)),
            ("crit_2", AlertSeverity.CRITICAL, datetime.now() - timedelta(minutes=2)),
        ]
        
        for alert_id, severity, timestamp in alerts:
            alert = Alert(
                alert_id=alert_id,
                alert_type=AlertType.PERFORMANCE,
                severity=severity,
                source="test",
                message=f"Alert {alert_id}",
                details={},
                timestamp=timestamp
            )
            system.alerts[alert_id] = alert
        
        result = system.prioritize_alerts()
        
        assert result['prioritization'] == 'alerts'
        assert result['total_active'] == 5
        assert result['prioritized_count'] == 5
        
        # Check priority order - critical alerts should be first
        priority_list = result['priority_list']
        assert priority_list[0]['alert_id'] == 'crit_2'  # Older critical
        assert priority_list[1]['alert_id'] == 'crit_1'  # Newer critical
        assert priority_list[2]['alert_id'] == 'high_1'
        assert priority_list[3]['alert_id'] == 'med_1'
        assert priority_list[4]['alert_id'] == 'low_1'


class TestResponseCoordination:
    """Test alert response coordination"""
    
    def test_coordinate_alert_responses(self):
        """Test coordinating alert responses"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        # Add response times for metrics
        system.response_times.extend([10, 15, 20, 25, 30])
        
        # Create multiple responses
        responses = [
            AlertResponse("resp_1", "alert_1", "optimize", "pending", datetime.now()),
            AlertResponse("resp_2", "alert_2", "optimize", "in_progress", datetime.now()),
            AlertResponse("resp_3", "alert_3", "restart", "pending", datetime.now()),
            AlertResponse("resp_4", "alert_4", "optimize", "completed", datetime.now()),
        ]
        
        for response in responses:
            system.alert_responses[response.response_id] = response
        
        result = system.coordinate_alert_responses()
        
        assert result['coordination'] == 'alert_responses'
        assert result['active_responses'] == 3  # pending and in_progress only
        assert result['average_response_time'] == 20  # (10+15+20+25+30)/5
        assert result['response_time_target'] == 30
        assert result['meets_target'] == True
        
        # Should detect multiple optimize actions
        assert len(result['coordination_actions']) == 1
        assert result['coordination_actions'][0]['action'] == 'consolidate'
        assert result['coordination_actions'][0]['type'] == 'optimize'
        assert result['coordination_actions'][0]['count'] == 2  # 2 active optimize actions


class TestAlertReport:
    """Test alert report generation"""
    
    def test_generate_alert_report(self):
        """Test generating comprehensive alert report"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        # Create sample alerts and responses
        for i in range(5):
            alert = Alert(
                alert_id=f"alert_{i}",
                alert_type=AlertType.PERFORMANCE if i % 2 == 0 else AlertType.RESOURCE,
                severity=AlertSeverity.CRITICAL if i == 0 else AlertSeverity.HIGH,
                source=f"system_{i}",
                message=f"Alert {i}",
                details={},
                timestamp=datetime.now() - timedelta(minutes=i)
            )
            
            if i < 2:
                alert.status = 'resolved'
                alert.resolved_at = datetime.now()
            
            system.alerts[alert.alert_id] = alert
            system.alert_history.append(alert)
            system.alert_counts[alert.alert_type.value] += 1
            
            # Add response
            if i < 3:
                # Completed responses
                response = AlertResponse(
                    response_id=f"resp_{i}",
                    alert_id=f"alert_{i}",
                    action_type="optimize",
                    status="completed",
                    started_at=datetime.now() - timedelta(seconds=30),
                    completed_at=datetime.now()
                )
            elif i == 3:
                # Failed response
                response = AlertResponse(
                    response_id=f"resp_{i}",
                    alert_id=f"alert_{i}",
                    action_type="optimize",
                    status="failed",
                    started_at=datetime.now() - timedelta(seconds=30),
                    completed_at=datetime.now()
                )
            else:
                # In-progress response (i == 4)
                response = AlertResponse(
                    response_id=f"resp_{i}",
                    alert_id=f"alert_{i}",
                    action_type="optimize",
                    status="in_progress",
                    started_at=datetime.now() - timedelta(seconds=30),
                    completed_at=None
                )
            system.alert_responses[response.response_id] = response
            
            if i < 3:
                system.response_times.append(25)
        
        # Add false positives for metrics
        system.false_positive_count = 1
        
        report = system.generate_alert_report({})
        
        assert 'report_id' in report
        assert 'timestamp' in report
        
        # Check alert summary
        assert report['alert_summary']['total_alerts'] == 5
        assert report['alert_summary']['active_alerts'] == 3
        assert report['alert_summary']['resolved_alerts'] == 2
        
        # Check breakdown
        assert report['alert_breakdown']['performance'] == 3
        assert report['alert_breakdown']['resource'] == 2
        
        # Check severity distribution
        assert report['severity_distribution']['critical'] == 1
        assert report['severity_distribution']['high'] == 4
        
        # Check response metrics
        assert report['response_metrics']['total_responses'] == 5
        assert report['response_metrics']['completed_responses'] == 3
        assert report['response_metrics']['failed_responses'] == 1  # One is still in progress
        assert report['response_metrics']['average_response_time'] == 25
        
        # Check health indicators
        assert report['health_indicators']['false_positive_rate'] == 0.2  # 1/5
        assert report['health_indicators']['response_success_rate'] == 0.6  # 3/5


class TestThreadSafety:
    """Test thread safety of alert operations"""
    
    def test_concurrent_alert_creation(self):
        """Test creating alerts from multiple threads"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        def create_alerts(thread_id):
            for i in range(10):
                alert = Alert(
                    alert_id=f"thread_{thread_id}_alert_{i}",
                    alert_type=AlertType.PERFORMANCE,
                    severity=AlertSeverity.MEDIUM,
                    source=f"thread_{thread_id}",
                    message=f"Alert from thread {thread_id}",
                    details={},
                    timestamp=datetime.now()
                )
                system._create_alert(alert)
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_alerts, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All alerts should be created without race conditions
        assert len(system.alerts) <= system.max_active_alerts
        assert len(system.alert_history) == 50  # 5 threads * 10 alerts


class TestGenerateAlertNotifications:
    """Test alert notification generation"""
    
    def test_generate_alert_notifications(self):
        """Test generating alert notifications"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        # Create alerts with different severities
        alerts = [
            (AlertSeverity.LOW, "active"),
            (AlertSeverity.MEDIUM, "active"), 
            (AlertSeverity.HIGH, "active"),    # Should get notification
            (AlertSeverity.CRITICAL, "active"), # Should get notification
            (AlertSeverity.HIGH, "resolved")   # Should not get notification (not active)
        ]
        
        for i, (severity, status) in enumerate(alerts):
            alert = Alert(
                alert_id=f"test_{i}",
                alert_type=AlertType.PERFORMANCE,
                severity=severity,
                source="test",
                message=f"Test alert {i}",
                details={},
                timestamp=datetime.now()
            )
            alert.status = status
            system.alerts[alert.alert_id] = alert
        
        # Mock the notification sending
        with patch.object(system, '_send_notifications') as mock_send:
            result = system.generate_alert_notifications()
            
            assert result['generation'] == 'alert_notifications'
            assert result['active_alerts'] == 4  # 4 active alerts
            assert result['notifications_sent'] == 2  # Only HIGH and CRITICAL
            assert result['enabled_channels'] == ['log']
            
            # Verify _send_notifications was called for high priority alerts
            assert mock_send.call_count == 2
            
            # Verify it was called with HIGH and CRITICAL alerts
            sent_alert_severities = [call[0][0].severity for call in mock_send.call_args_list]
            assert AlertSeverity.HIGH in sent_alert_severities
            assert AlertSeverity.CRITICAL in sent_alert_severities
    
    def test_generate_alert_notifications_no_active_alerts(self):
        """Test notification generation with no active alerts"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        # Create only resolved alerts
        alert = Alert(
            alert_id="resolved_1",
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.CRITICAL,
            source="test",
            message="Resolved alert",
            details={},
            timestamp=datetime.now()
        )
        alert.status = 'resolved'
        system.alerts[alert.alert_id] = alert
        
        result = system.generate_alert_notifications()
        
        assert result['generation'] == 'alert_notifications'
        assert result['active_alerts'] == 0
        assert result['notifications_sent'] == 0


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_monitor_data(self):
        """Test handling empty monitor data"""
        mock_monitor = Mock()
        mock_monitor.monitor_all_systems.return_value = {}
        mock_monitor.monitor_all_agents.return_value = {}
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        # Should not crash when checking rules
        system._check_alert_rules()
        
        # No alerts should be created
        assert len(system.alerts) == 0
    
    def test_monitor_exception_handling(self):
        """Test handling monitor exceptions with hard failures"""
        mock_monitor = Mock()
        mock_monitor.monitor_all_systems.side_effect = Exception("Monitor error")
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        # Should now fail hard instead of handling gracefully
        with pytest.raises(RuntimeError) as exc_info:
            system._check_alert_rules()
        
        assert "HARD FAILURE: Alert rule checking failed" in str(exc_info.value)
        assert "Monitor error" in str(exc_info.value)
    
    def test_invalid_alert_data(self):
        """Test handling invalid alert data from monitor"""
        mock_monitor = Mock()
        mock_optimization = Mock()
        
        system = ProductionAlertSystem(mock_monitor, mock_optimization)
        
        # Invalid alert data
        invalid_data = {
            'type': None,
            'severity': 'invalid_severity',
            'source': ''
        }
        
        # Should handle gracefully
        system._handle_monitor_alert(invalid_data)
        
        # Should still create an alert with defaults
        assert len(system.alerts) == 1
        alert = list(system.alerts.values())[0]
        assert alert.alert_type == AlertType.PERFORMANCE  # Default
        assert alert.severity == AlertSeverity.MEDIUM  # Default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])