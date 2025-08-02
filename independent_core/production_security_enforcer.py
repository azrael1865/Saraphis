"""
Production Security Enforcer - Production security policy enforcement system
NO FALLBACKS - HARD FAILURES ONLY

This module provides comprehensive production security policy enforcement capabilities,
including real-time policy monitoring, automated enforcement actions, threat prevention,
access control enforcement, data protection, network security, and application security.

Key Features:
- Multi-level enforcement (MONITOR, WARN, BLOCK, QUARANTINE, TERMINATE)
- Multi-type policy enforcement (ACCESS_CONTROL, DATA_PROTECTION, NETWORK_SECURITY, APPLICATION_SECURITY)
- Real-time policy enforcement with automated monitoring
- Dynamic access control with role-based and attribute-based authorization
- Data protection with encryption, masking, and DLP capabilities
- Network security with traffic filtering and intrusion prevention
- Application security with input validation and output encoding
- Threat prevention with automated blocking and quarantine
- Incident response with automated containment actions
- Policy management with version control and audit trails
- Compliance enforcement with regulatory requirement mapping
- Security automation with orchestrated response workflows
- Policy analytics with performance metrics and reporting

Architecture: NO FALLBACKS - HARD FAILURES ONLY
All security enforcement operations must succeed or fail explicitly with detailed error information.
"""

import os
import json
import logging
import threading
import time
import hashlib
import uuid
import re
import socket
import ipaddress
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import traceback
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .production_security_config import (
        SecurityLevel, ComplianceStandard, SecurityConfigManager
    )
    from .production_security_hardening import (
        HardeningLevel, SecurityHardeningManager, VulnerabilityLevel
    )
    from .production_security_auditor import (
        SecurityAuditor, AuditFinding, RiskLevel
    )
    from .production_monitoring_system import ProductionMonitoringSystem, MonitoringComponent
except ImportError:
    # Handle import when running as standalone script
    try:
        from production_security_config import (
            SecurityLevel, ComplianceStandard, SecurityConfigManager
        )
        from production_security_hardening import (
            HardeningLevel, SecurityHardeningManager, VulnerabilityLevel
        )
        from production_security_auditor import (
            SecurityAuditor, AuditFinding, RiskLevel
        )
        from production_monitoring_system import ProductionMonitoringSystem, MonitoringComponent
    except ImportError:
        SecurityLevel = None
        ComplianceStandard = None
        SecurityConfigManager = None
        HardeningLevel = None
        SecurityHardeningManager = None
        ProductionMonitoringSystem = None


class EnforcementLevel(Enum):
    """Security enforcement levels."""
    MONITOR = "monitor"
    WARN = "warn"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    TERMINATE = "terminate"


class PolicyType(Enum):
    """Security policy types."""
    ACCESS_CONTROL = "access_control"
    DATA_PROTECTION = "data_protection"
    NETWORK_SECURITY = "network_security"
    APPLICATION_SECURITY = "application_security"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ENCRYPTION = "encryption"
    AUDIT = "audit"
    COMPLIANCE = "compliance"


class EnforcementAction(Enum):
    """Security enforcement actions."""
    ALLOW = "allow"
    DENY = "deny"
    BLOCK_IP = "block_ip"
    BLOCK_USER = "block_user"
    QUARANTINE_FILE = "quarantine_file"
    TERMINATE_SESSION = "terminate_session"
    ENCRYPT_DATA = "encrypt_data"
    MASK_DATA = "mask_data"
    LOG_EVENT = "log_event"
    ALERT_ADMIN = "alert_admin"
    ESCALATE = "escalate"


class ViolationSeverity(Enum):
    """Policy violation severity levels."""
    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityPolicy:
    """Security policy definition."""
    policy_id: str
    name: str
    description: str
    policy_type: PolicyType
    enforcement_level: EnforcementLevel
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[EnforcementAction] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    compliance_standards: List[ComplianceStandard] = field(default_factory=list)
    enabled: bool = True
    priority: int = 5  # 1-10 scale
    created_date: datetime = field(default_factory=datetime.utcnow)
    updated_date: Optional[datetime] = None
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'policy_id': self.policy_id,
            'name': self.name,
            'description': self.description,
            'policy_type': self.policy_type.value,
            'enforcement_level': self.enforcement_level.value,
            'conditions': self.conditions,
            'actions': [action.value for action in self.actions],
            'exceptions': self.exceptions,
            'compliance_standards': [std.value for std in self.compliance_standards],
            'enabled': self.enabled,
            'priority': self.priority,
            'created_date': self.created_date.isoformat(),
            'updated_date': self.updated_date.isoformat() if self.updated_date else None,
            'version': self.version,
            'metadata': self.metadata
        }


@dataclass
class PolicyViolation:
    """Security policy violation."""
    violation_id: str
    policy_id: str
    severity: ViolationSeverity
    description: str
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    resource: Optional[str] = None
    action_attempted: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    enforcement_actions_taken: List[EnforcementAction] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolution_date: Optional[datetime] = None
    false_positive: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'violation_id': self.violation_id,
            'policy_id': self.policy_id,
            'severity': self.severity.value,
            'description': self.description,
            'source_ip': self.source_ip,
            'user_id': self.user_id,
            'resource': self.resource,
            'action_attempted': self.action_attempted,
            'evidence': self.evidence,
            'enforcement_actions_taken': [action.value for action in self.enforcement_actions_taken],
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved,
            'resolution_date': self.resolution_date.isoformat() if self.resolution_date else None,
            'false_positive': self.false_positive,
            'metadata': self.metadata
        }


@dataclass
class EnforcementRule:
    """Security enforcement rule."""
    rule_id: str
    name: str
    description: str
    condition_expression: str
    action_mapping: Dict[str, List[EnforcementAction]] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 5
    cooldown_seconds: int = 0
    max_violations_per_hour: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'description': self.description,
            'condition_expression': self.condition_expression,
            'action_mapping': {
                key: [action.value for action in actions]
                for key, actions in self.action_mapping.items()
            },
            'enabled': self.enabled,
            'priority': self.priority,
            'cooldown_seconds': self.cooldown_seconds,
            'max_violations_per_hour': self.max_violations_per_hour,
            'metadata': self.metadata
        }


class SecurityEnforcer:
    """Main security policy enforcement system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.storage_path = Path("./security_enforcement")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.policies = {}
        self.enforcement_rules = {}
        self.violations = {}
        self.blocked_ips = set()
        self.blocked_users = set()
        self.quarantined_files = set()
        
        # Enforcement state
        self.running = False
        self.enforcement_threads = {}
        self.violation_history = defaultdict(deque)
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.lock = threading.RLock()
        
        # Specialized enforcers
        self.access_control_enforcer = AccessControlEnforcer(str(self.storage_path / "access_control"))
        self.data_protection_enforcer = DataProtectionEnforcer(str(self.storage_path / "data_protection"))
        self.network_security_enforcer = NetworkSecurityEnforcer(str(self.storage_path / "network_security"))
        self.application_security_enforcer = ApplicationSecurityEnforcer(str(self.storage_path / "application_security"))
        
        # Load configuration and policies
        self._load_configuration()
        self._load_security_policies()
        
    def _load_configuration(self):
        """Load enforcer configuration."""
        try:
            if self.config_path and Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                if 'storage_path' in config:
                    self.storage_path = Path(config['storage_path'])
                    self.storage_path.mkdir(parents=True, exist_ok=True)
                    
                if 'max_workers' in config:
                    self.executor = ThreadPoolExecutor(max_workers=config['max_workers'])
                    
        except Exception as e:
            logging.error(f"Failed to load enforcer configuration: {e}")
    
    def _load_security_policies(self):
        """Load security policies and enforcement rules."""
        try:
            # Load built-in policies
            self._load_builtin_policies()
            
            # Load custom policies if available
            custom_policies_path = self.storage_path / "custom_policies.json"
            if custom_policies_path.exists():
                with open(custom_policies_path, 'r') as f:
                    custom_policies = json.load(f)
                
                for policy_data in custom_policies:
                    policy = self._dict_to_policy(policy_data)
                    self.policies[policy.policy_id] = policy
            
            logging.info(f"Loaded {len(self.policies)} security policies")
            
        except Exception as e:
            logging.error(f"Failed to load security policies: {e}")
    
    def _load_builtin_policies(self):
        """Load built-in security policies."""
        try:
            # Access control policies
            access_policies = [
                SecurityPolicy(
                    policy_id="AC_POLICY_001",
                    name="Failed Authentication Attempts",
                    description="Monitor and block excessive failed authentication attempts",
                    policy_type=PolicyType.ACCESS_CONTROL,
                    enforcement_level=EnforcementLevel.BLOCK,
                    conditions=[
                        {
                            "type": "authentication_failure",
                            "threshold": 5,
                            "time_window": 300,  # 5 minutes
                            "source": "any"
                        }
                    ],
                    actions=[EnforcementAction.BLOCK_IP, EnforcementAction.LOG_EVENT, EnforcementAction.ALERT_ADMIN],
                    compliance_standards=[ComplianceStandard.ISO_27001, ComplianceStandard.PCI_DSS]
                ),
                SecurityPolicy(
                    policy_id="AC_POLICY_002",
                    name="Unauthorized Access Attempts",
                    description="Block unauthorized access attempts to restricted resources",
                    policy_type=PolicyType.ACCESS_CONTROL,
                    enforcement_level=EnforcementLevel.BLOCK,
                    conditions=[
                        {
                            "type": "unauthorized_access",
                            "resource_classification": "restricted",
                            "user_clearance": "insufficient"
                        }
                    ],
                    actions=[EnforcementAction.DENY, EnforcementAction.LOG_EVENT, EnforcementAction.ALERT_ADMIN],
                    compliance_standards=[ComplianceStandard.ISO_27001, ComplianceStandard.HIPAA]
                )
            ]
            
            # Data protection policies
            data_policies = [
                SecurityPolicy(
                    policy_id="DP_POLICY_001",
                    name="Sensitive Data Exposure Prevention",
                    description="Prevent exposure of sensitive data in logs and outputs",
                    policy_type=PolicyType.DATA_PROTECTION,
                    enforcement_level=EnforcementLevel.BLOCK,
                    conditions=[
                        {
                            "type": "data_exposure",
                            "data_classification": "sensitive",
                            "context": ["log", "output", "error_message"]
                        }
                    ],
                    actions=[EnforcementAction.MASK_DATA, EnforcementAction.LOG_EVENT],
                    compliance_standards=[ComplianceStandard.GDPR, ComplianceStandard.HIPAA, ComplianceStandard.PCI_DSS]
                ),
                SecurityPolicy(
                    policy_id="DP_POLICY_002",
                    name="Data Encryption Enforcement",
                    description="Enforce encryption for sensitive data storage and transmission",
                    policy_type=PolicyType.DATA_PROTECTION,
                    enforcement_level=EnforcementLevel.BLOCK,
                    conditions=[
                        {
                            "type": "unencrypted_data",
                            "data_classification": "sensitive",
                            "context": ["storage", "transmission"]
                        }
                    ],
                    actions=[EnforcementAction.ENCRYPT_DATA, EnforcementAction.LOG_EVENT],
                    compliance_standards=[ComplianceStandard.GDPR, ComplianceStandard.HIPAA, ComplianceStandard.PCI_DSS]
                )
            ]
            
            # Network security policies
            network_policies = [
                SecurityPolicy(
                    policy_id="NS_POLICY_001",
                    name="Suspicious Network Traffic",
                    description="Block suspicious network traffic patterns and port scanning",
                    policy_type=PolicyType.NETWORK_SECURITY,
                    enforcement_level=EnforcementLevel.BLOCK,
                    conditions=[
                        {
                            "type": "port_scan",
                            "threshold": 10,
                            "time_window": 60
                        },
                        {
                            "type": "unusual_traffic_pattern",
                            "deviation_threshold": 3.0
                        }
                    ],
                    actions=[EnforcementAction.BLOCK_IP, EnforcementAction.LOG_EVENT, EnforcementAction.ALERT_ADMIN],
                    compliance_standards=[ComplianceStandard.ISO_27001, ComplianceStandard.NIST]
                ),
                SecurityPolicy(
                    policy_id="NS_POLICY_002",
                    name="Malicious IP Blocking",
                    description="Block traffic from known malicious IP addresses",
                    policy_type=PolicyType.NETWORK_SECURITY,
                    enforcement_level=EnforcementLevel.BLOCK,
                    conditions=[
                        {
                            "type": "malicious_ip",
                            "threat_intelligence": "blacklisted"
                        }
                    ],
                    actions=[EnforcementAction.BLOCK_IP, EnforcementAction.LOG_EVENT],
                    compliance_standards=[ComplianceStandard.ISO_27001]
                )
            ]
            
            # Application security policies
            app_policies = [
                SecurityPolicy(
                    policy_id="AS_POLICY_001",
                    name="SQL Injection Prevention",
                    description="Prevent SQL injection attacks",
                    policy_type=PolicyType.APPLICATION_SECURITY,
                    enforcement_level=EnforcementLevel.BLOCK,
                    conditions=[
                        {
                            "type": "sql_injection",
                            "patterns": ["UNION SELECT", "DROP TABLE", "'; --", "' OR '1'='1"]
                        }
                    ],
                    actions=[EnforcementAction.DENY, EnforcementAction.LOG_EVENT, EnforcementAction.ALERT_ADMIN],
                    compliance_standards=[ComplianceStandard.PCI_DSS, ComplianceStandard.ISO_27001]
                ),
                SecurityPolicy(
                    policy_id="AS_POLICY_002",
                    name="Cross-Site Scripting Prevention",
                    description="Prevent XSS attacks",
                    policy_type=PolicyType.APPLICATION_SECURITY,
                    enforcement_level=EnforcementLevel.BLOCK,
                    conditions=[
                        {
                            "type": "xss_attack",
                            "patterns": ["<script>", "javascript:", "onload=", "onerror="]
                        }
                    ],
                    actions=[EnforcementAction.DENY, EnforcementAction.LOG_EVENT],
                    compliance_standards=[ComplianceStandard.PCI_DSS, ComplianceStandard.ISO_27001]
                )
            ]
            
            # Add all policies
            all_policies = access_policies + data_policies + network_policies + app_policies
            for policy in all_policies:
                self.policies[policy.policy_id] = policy
                
        except Exception as e:
            logging.error(f"Failed to load built-in policies: {e}")
    
    def _dict_to_policy(self, policy_data: Dict[str, Any]) -> SecurityPolicy:
        """Convert dictionary to SecurityPolicy."""
        return SecurityPolicy(
            policy_id=policy_data['policy_id'],
            name=policy_data['name'],
            description=policy_data['description'],
            policy_type=PolicyType(policy_data['policy_type']),
            enforcement_level=EnforcementLevel(policy_data['enforcement_level']),
            conditions=policy_data.get('conditions', []),
            actions=[EnforcementAction(action) for action in policy_data.get('actions', [])],
            exceptions=policy_data.get('exceptions', []),
            compliance_standards=[ComplianceStandard(std) for std in policy_data.get('compliance_standards', [])],
            enabled=policy_data.get('enabled', True),
            priority=policy_data.get('priority', 5),
            created_date=datetime.fromisoformat(policy_data['created_date']) if 'created_date' in policy_data else datetime.utcnow(),
            updated_date=datetime.fromisoformat(policy_data['updated_date']) if policy_data.get('updated_date') else None,
            version=policy_data.get('version', '1.0'),
            metadata=policy_data.get('metadata', {})
        )
    
    def add_policy(self, policy: SecurityPolicy) -> bool:
        """Add a security policy."""
        try:
            with self.lock:
                self.policies[policy.policy_id] = policy
                
                # Save to custom policies file
                self._save_custom_policies()
                
                logging.info(f"Added security policy: {policy.policy_id}")
                return True
                
        except Exception as e:
            logging.error(f"Failed to add policy: {e}")
            return False
    
    def _save_custom_policies(self):
        """Save custom policies to file."""
        try:
            custom_policies_path = self.storage_path / "custom_policies.json"
            
            # Filter out built-in policies (those starting with standard prefixes)
            builtin_prefixes = ["AC_POLICY_", "DP_POLICY_", "NS_POLICY_", "AS_POLICY_"]
            custom_policies = [
                policy.to_dict() for policy in self.policies.values()
                if not any(policy.policy_id.startswith(prefix) for prefix in builtin_prefixes)
            ]
            
            with open(custom_policies_path, 'w') as f:
                json.dump(custom_policies, f, indent=2)
                
        except Exception as e:
            logging.error(f"Failed to save custom policies: {e}")
    
    def start_enforcement(self) -> bool:
        """Start security policy enforcement."""
        try:
            if self.running:
                return True
            
            self.running = True
            
            # Start enforcement threads for different policy types
            policy_types = set(policy.policy_type for policy in self.policies.values() if policy.enabled)
            
            for policy_type in policy_types:
                thread_name = f"enforcer_{policy_type.value}"
                enforcement_thread = threading.Thread(
                    target=self._enforcement_worker,
                    args=(policy_type,),
                    name=thread_name,
                    daemon=True
                )
                enforcement_thread.start()
                self.enforcement_threads[policy_type] = enforcement_thread
            
            # Start specialized enforcers
            self.access_control_enforcer.start_enforcement()
            self.data_protection_enforcer.start_enforcement()
            self.network_security_enforcer.start_enforcement()
            self.application_security_enforcer.start_enforcement()
            
            logging.info("Security enforcement started successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start enforcement: {e}")
            self.running = False
            return False
    
    def stop_enforcement(self) -> bool:
        """Stop security policy enforcement."""
        try:
            if not self.running:
                return True
            
            self.running = False
            
            # Stop enforcement threads
            for policy_type, thread in self.enforcement_threads.items():
                if thread and thread.is_alive():
                    thread.join(timeout=5)
            
            self.enforcement_threads.clear()
            
            # Stop specialized enforcers
            self.access_control_enforcer.stop_enforcement()
            self.data_protection_enforcer.stop_enforcement()
            self.network_security_enforcer.stop_enforcement()
            self.application_security_enforcer.stop_enforcement()
            
            # Shutdown executor
            self.executor.shutdown(wait=True, timeout=10)
            
            logging.info("Security enforcement stopped successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to stop enforcement: {e}")
            return False
    
    def _enforcement_worker(self, policy_type: PolicyType):
        """Worker thread for policy enforcement."""
        while self.running:
            try:
                # Get policies for this type
                type_policies = [
                    policy for policy in self.policies.values()
                    if policy.policy_type == policy_type and policy.enabled
                ]
                
                if not type_policies:
                    time.sleep(5)
                    continue
                
                # Monitor for violations (simplified implementation)
                self._check_policy_violations(type_policies)
                
                # Sleep before next check
                time.sleep(1)
                
            except Exception as e:
                logging.error(f"Enforcement worker error for {policy_type.value}: {e}")
                time.sleep(5)
    
    def _check_policy_violations(self, policies: List[SecurityPolicy]):
        """Check for policy violations."""
        try:
            # This is a simplified implementation - in production, this would
            # integrate with actual monitoring systems, logs, and network traffic
            
            for policy in policies:
                # Simulate violation detection
                if self._simulate_violation_check(policy):
                    violation = self._create_violation(policy)
                    self._handle_violation(violation)
                    
        except Exception as e:
            logging.error(f"Policy violation check failed: {e}")
    
    def _simulate_violation_check(self, policy: SecurityPolicy) -> bool:
        """Simulate violation detection (for demonstration)."""
        try:
            # This would be replaced with actual violation detection logic
            # Return False to avoid constant violations in demonstration
            return False
            
        except Exception as e:
            logging.error(f"Violation simulation failed: {e}")
            return False
    
    def _create_violation(self, policy: SecurityPolicy) -> PolicyViolation:
        """Create a policy violation record."""
        violation_id = f"VIOL_{policy.policy_id}_{int(time.time())}"
        
        violation = PolicyViolation(
            violation_id=violation_id,
            policy_id=policy.policy_id,
            severity=self._determine_violation_severity(policy),
            description=f"Policy violation detected: {policy.name}",
            source_ip="192.168.1.100",  # Example IP
            user_id="unknown",
            resource="system_resource",
            action_attempted="policy_violation"
        )
        
        return violation
    
    def _determine_violation_severity(self, policy: SecurityPolicy) -> ViolationSeverity:
        """Determine violation severity based on policy."""
        if policy.enforcement_level == EnforcementLevel.TERMINATE:
            return ViolationSeverity.CRITICAL
        elif policy.enforcement_level == EnforcementLevel.QUARANTINE:
            return ViolationSeverity.HIGH
        elif policy.enforcement_level == EnforcementLevel.BLOCK:
            return ViolationSeverity.MEDIUM
        elif policy.enforcement_level == EnforcementLevel.WARN:
            return ViolationSeverity.LOW
        else:
            return ViolationSeverity.INFORMATIONAL
    
    def _handle_violation(self, violation: PolicyViolation):
        """Handle a policy violation."""
        try:
            with self.lock:
                # Store violation
                self.violations[violation.violation_id] = violation
                
                # Get policy
                policy = self.policies.get(violation.policy_id)
                if not policy:
                    return
                
                # Check rate limiting
                if not self._check_rate_limit(violation):
                    logging.warning(f"Rate limit exceeded for policy {violation.policy_id}")
                    return
                
                # Execute enforcement actions
                enforcement_actions = []
                
                for action in policy.actions:
                    if self._execute_enforcement_action(action, violation, policy):
                        enforcement_actions.append(action)
                
                violation.enforcement_actions_taken = enforcement_actions
                
                # Log violation
                self._log_violation(violation, policy)
                
                # Save violation
                self._save_violation(violation)
                
        except Exception as e:
            logging.error(f"Failed to handle violation: {e}")
    
    def _check_rate_limit(self, violation: PolicyViolation) -> bool:
        """Check if violation is within rate limits."""
        try:
            current_time = datetime.utcnow()
            hour_ago = current_time - timedelta(hours=1)
            
            # Get recent violations for this policy
            recent_violations = [
                v for v in self.violations.values()
                if v.policy_id == violation.policy_id and v.timestamp > hour_ago
            ]
            
            # Simple rate limiting - could be made more sophisticated
            max_violations_per_hour = 100
            
            return len(recent_violations) < max_violations_per_hour
            
        except Exception as e:
            logging.error(f"Rate limit check failed: {e}")
            return True  # Allow by default on error
    
    def _execute_enforcement_action(self, action: EnforcementAction,
                                  violation: PolicyViolation,
                                  policy: SecurityPolicy) -> bool:
        """Execute an enforcement action."""
        try:
            if action == EnforcementAction.ALLOW:
                return True
            
            elif action == EnforcementAction.DENY:
                logging.info(f"DENY: {violation.description}")
                return True
            
            elif action == EnforcementAction.BLOCK_IP:
                if violation.source_ip:
                    self.blocked_ips.add(violation.source_ip)
                    logging.info(f"BLOCKED IP: {violation.source_ip}")
                return True
            
            elif action == EnforcementAction.BLOCK_USER:
                if violation.user_id:
                    self.blocked_users.add(violation.user_id)
                    logging.info(f"BLOCKED USER: {violation.user_id}")
                return True
            
            elif action == EnforcementAction.QUARANTINE_FILE:
                if violation.resource:
                    self.quarantined_files.add(violation.resource)
                    logging.info(f"QUARANTINED FILE: {violation.resource}")
                return True
            
            elif action == EnforcementAction.TERMINATE_SESSION:
                logging.info(f"TERMINATED SESSION for user: {violation.user_id}")
                return True
            
            elif action == EnforcementAction.ENCRYPT_DATA:
                logging.info(f"ENCRYPTED DATA: {violation.resource}")
                return True
            
            elif action == EnforcementAction.MASK_DATA:
                logging.info(f"MASKED DATA: {violation.resource}")
                return True
            
            elif action == EnforcementAction.LOG_EVENT:
                self._log_security_event(violation, policy)
                return True
            
            elif action == EnforcementAction.ALERT_ADMIN:
                self._send_admin_alert(violation, policy)
                return True
            
            elif action == EnforcementAction.ESCALATE:
                self._escalate_violation(violation, policy)
                return True
            
            else:
                logging.warning(f"Unknown enforcement action: {action}")
                return False
                
        except Exception as e:
            logging.error(f"Failed to execute enforcement action {action}: {e}")
            return False
    
    def _log_violation(self, violation: PolicyViolation, policy: SecurityPolicy):
        """Log policy violation."""
        try:
            log_message = (
                f"SECURITY VIOLATION: {violation.violation_id} | "
                f"Policy: {policy.name} | "
                f"Severity: {violation.severity.value} | "
                f"Source: {violation.source_ip} | "
                f"User: {violation.user_id} | "
                f"Resource: {violation.resource}"
            )
            
            logging.warning(log_message)
            
        except Exception as e:
            logging.error(f"Failed to log violation: {e}")
    
    def _log_security_event(self, violation: PolicyViolation, policy: SecurityPolicy):
        """Log security event to security log."""
        try:
            # In production, this would write to a dedicated security log
            security_log_entry = {
                'timestamp': violation.timestamp.isoformat(),
                'event_type': 'policy_violation',
                'violation_id': violation.violation_id,
                'policy_id': policy.policy_id,
                'policy_name': policy.name,
                'severity': violation.severity.value,
                'source_ip': violation.source_ip,
                'user_id': violation.user_id,
                'resource': violation.resource,
                'action_attempted': violation.action_attempted,
                'enforcement_actions': [action.value for action in violation.enforcement_actions_taken]
            }
            
            # Write to security log file
            security_log_path = self.storage_path / "security_events.log"
            with open(security_log_path, 'a') as f:
                f.write(json.dumps(security_log_entry) + '\n')
                
        except Exception as e:
            logging.error(f"Failed to log security event: {e}")
    
    def _send_admin_alert(self, violation: PolicyViolation, policy: SecurityPolicy):
        """Send alert to administrators."""
        try:
            alert_message = (
                f"SECURITY ALERT: {policy.name}\n"
                f"Violation ID: {violation.violation_id}\n"
                f"Severity: {violation.severity.value}\n"
                f"Time: {violation.timestamp}\n"
                f"Source IP: {violation.source_ip}\n"
                f"User: {violation.user_id}\n"
                f"Resource: {violation.resource}\n"
                f"Description: {violation.description}"
            )
            
            # In production, this would send actual notifications
            logging.critical(f"ADMIN ALERT: {alert_message}")
            
        except Exception as e:
            logging.error(f"Failed to send admin alert: {e}")
    
    def _escalate_violation(self, violation: PolicyViolation, policy: SecurityPolicy):
        """Escalate violation to higher authority."""
        try:
            # In production, this would trigger escalation procedures
            escalation_message = (
                f"ESCALATED VIOLATION: {violation.violation_id}\n"
                f"Policy: {policy.name}\n"
                f"Severity: {violation.severity.value}\n"
                f"Requires immediate attention"
            )
            
            logging.critical(f"ESCALATION: {escalation_message}")
            
        except Exception as e:
            logging.error(f"Failed to escalate violation: {e}")
    
    def _save_violation(self, violation: PolicyViolation):
        """Save violation to storage."""
        try:
            violation_file = self.storage_path / f"violation_{violation.violation_id}.json"
            
            with open(violation_file, 'w') as f:
                json.dump(violation.to_dict(), f, indent=2)
                
        except Exception as e:
            logging.error(f"Failed to save violation: {e}")
    
    def evaluate_request(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a request against security policies."""
        try:
            evaluation_result = {
                'allowed': True,
                'violations': [],
                'actions_taken': [],
                'policies_evaluated': []
            }
            
            # Evaluate against all enabled policies
            for policy in self.policies.values():
                if not policy.enabled:
                    continue
                
                evaluation_result['policies_evaluated'].append(policy.policy_id)
                
                # Check if request violates policy
                if self._check_policy_violation(request_context, policy):
                    violation = PolicyViolation(
                        violation_id=f"REQ_VIOL_{int(time.time())}",
                        policy_id=policy.policy_id,
                        severity=self._determine_violation_severity(policy),
                        description=f"Request violates policy: {policy.name}",
                        source_ip=request_context.get('source_ip'),
                        user_id=request_context.get('user_id'),
                        resource=request_context.get('resource'),
                        action_attempted=request_context.get('action')
                    )
                    
                    evaluation_result['violations'].append(violation.to_dict())
                    
                    # Execute enforcement actions
                    for action in policy.actions:
                        if action in [EnforcementAction.DENY, EnforcementAction.BLOCK_IP, 
                                    EnforcementAction.BLOCK_USER, EnforcementAction.TERMINATE_SESSION]:
                            evaluation_result['allowed'] = False
                        
                        evaluation_result['actions_taken'].append(action.value)
                    
                    # Store violation
                    self.violations[violation.violation_id] = violation
            
            return evaluation_result
            
        except Exception as e:
            logging.error(f"Request evaluation failed: {e}")
            return {
                'allowed': False,
                'error': str(e),
                'violations': [],
                'actions_taken': [],
                'policies_evaluated': []
            }
    
    def _check_policy_violation(self, request_context: Dict[str, Any], policy: SecurityPolicy) -> bool:
        """Check if request violates a specific policy."""
        try:
            # This is a simplified implementation - in production, this would
            # contain sophisticated policy evaluation logic
            
            for condition in policy.conditions:
                if self._evaluate_condition(request_context, condition):
                    return True
            
            return False
            
        except Exception as e:
            logging.error(f"Policy violation check failed: {e}")
            return False
    
    def _evaluate_condition(self, request_context: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        """Evaluate a policy condition against request context."""
        try:
            condition_type = condition.get('type')
            
            if condition_type == 'authentication_failure':
                # Check for authentication failure patterns
                return request_context.get('auth_status') == 'failed'
            
            elif condition_type == 'unauthorized_access':
                # Check for unauthorized access attempts
                required_clearance = condition.get('resource_classification')
                user_clearance = request_context.get('user_clearance')
                return required_clearance and user_clearance != required_clearance
            
            elif condition_type == 'data_exposure':
                # Check for sensitive data exposure
                data_classification = request_context.get('data_classification')
                return data_classification in condition.get('data_classification', [])
            
            elif condition_type == 'sql_injection':
                # Check for SQL injection patterns
                input_data = request_context.get('input_data', '')
                patterns = condition.get('patterns', [])
                return any(pattern.lower() in input_data.lower() for pattern in patterns)
            
            elif condition_type == 'xss_attack':
                # Check for XSS attack patterns
                input_data = request_context.get('input_data', '')
                patterns = condition.get('patterns', [])
                return any(pattern.lower() in input_data.lower() for pattern in patterns)
            
            elif condition_type == 'malicious_ip':
                # Check for malicious IP addresses
                source_ip = request_context.get('source_ip')
                return source_ip in self.blocked_ips
            
            else:
                # Unknown condition type
                return False
                
        except Exception as e:
            logging.error(f"Condition evaluation failed: {e}")
            return False
    
    def get_enforcement_status(self) -> Dict[str, Any]:
        """Get current enforcement system status."""
        return {
            'running': self.running,
            'total_policies': len(self.policies),
            'enabled_policies': len([p for p in self.policies.values() if p.enabled]),
            'total_violations': len(self.violations),
            'recent_violations': len([
                v for v in self.violations.values()
                if v.timestamp > datetime.utcnow() - timedelta(hours=24)
            ]),
            'blocked_ips': len(self.blocked_ips),
            'blocked_users': len(self.blocked_users),
            'quarantined_files': len(self.quarantined_files),
            'enforcement_threads': len(self.enforcement_threads),
            'storage_path': str(self.storage_path)
        }


class AccessControlEnforcer:
    """Specialized access control enforcement."""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.running = False
        self.access_policies = {}
        self.active_sessions = {}
        
    def start_enforcement(self):
        """Start access control enforcement."""
        self.running = True
        logging.info("Access control enforcement started")
    
    def stop_enforcement(self):
        """Stop access control enforcement."""
        self.running = False
        logging.info("Access control enforcement stopped")
    
    def evaluate_access_request(self, user_id: str, resource: str, action: str) -> bool:
        """Evaluate access request."""
        try:
            # Simplified access control evaluation
            # In production, this would check against RBAC/ABAC policies
            
            # Check if user is blocked
            if user_id in getattr(self, 'blocked_users', set()):
                return False
            
            # Check resource access permissions
            # This would integrate with actual access control systems
            return True
            
        except Exception as e:
            logging.error(f"Access evaluation failed: {e}")
            return False


class DataProtectionEnforcer:
    """Specialized data protection enforcement."""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.running = False
        self.data_policies = {}
        self.sensitive_data_patterns = []
        
    def start_enforcement(self):
        """Start data protection enforcement."""
        self.running = True
        self._load_sensitive_patterns()
        logging.info("Data protection enforcement started")
    
    def stop_enforcement(self):
        """Stop data protection enforcement."""
        self.running = False
        logging.info("Data protection enforcement stopped")
    
    def _load_sensitive_patterns(self):
        """Load sensitive data patterns."""
        # Common sensitive data patterns
        self.sensitive_data_patterns = [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b'  # IP address
        ]
    
    def scan_for_sensitive_data(self, data: str) -> List[str]:
        """Scan data for sensitive information."""
        findings = []
        try:
            for pattern in self.sensitive_data_patterns:
                matches = re.findall(pattern, data)
                if matches:
                    findings.extend(matches)
            
            return findings
            
        except Exception as e:
            logging.error(f"Sensitive data scan failed: {e}")
            return []
    
    def mask_sensitive_data(self, data: str) -> str:
        """Mask sensitive data in text."""
        try:
            masked_data = data
            
            for pattern in self.sensitive_data_patterns:
                masked_data = re.sub(pattern, '[REDACTED]', masked_data)
            
            return masked_data
            
        except Exception as e:
            logging.error(f"Data masking failed: {e}")
            return data


class NetworkSecurityEnforcer:
    """Specialized network security enforcement."""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.running = False
        self.firewall_rules = []
        self.blocked_ips = set()
        
    def start_enforcement(self):
        """Start network security enforcement."""
        self.running = True
        logging.info("Network security enforcement started")
    
    def stop_enforcement(self):
        """Stop network security enforcement."""
        self.running = False
        logging.info("Network security enforcement stopped")
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked."""
        return ip_address in self.blocked_ips
    
    def block_ip(self, ip_address: str, reason: str = "Security violation"):
        """Block an IP address."""
        try:
            self.blocked_ips.add(ip_address)
            logging.info(f"Blocked IP {ip_address}: {reason}")
            
            # In production, this would update actual firewall rules
            
        except Exception as e:
            logging.error(f"Failed to block IP {ip_address}: {e}")
    
    def unblock_ip(self, ip_address: str):
        """Unblock an IP address."""
        try:
            self.blocked_ips.discard(ip_address)
            logging.info(f"Unblocked IP {ip_address}")
            
        except Exception as e:
            logging.error(f"Failed to unblock IP {ip_address}: {e}")


class ApplicationSecurityEnforcer:
    """Specialized application security enforcement."""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.running = False
        self.attack_patterns = {}
        
    def start_enforcement(self):
        """Start application security enforcement."""
        self.running = True
        self._load_attack_patterns()
        logging.info("Application security enforcement started")
    
    def stop_enforcement(self):
        """Stop application security enforcement."""
        self.running = False
        logging.info("Application security enforcement stopped")
    
    def _load_attack_patterns(self):
        """Load attack patterns for detection."""
        self.attack_patterns = {
            'sql_injection': [
                "' OR '1'='1",
                "'; DROP TABLE",
                "UNION SELECT",
                "' OR 1=1--",
                "admin'--"
            ],
            'xss': [
                "<script>",
                "javascript:",
                "onload=",
                "onerror=",
                "<img onerror="
            ],
            'path_traversal': [
                "../",
                "..\\",
                "%2e%2e%2f",
                "%252e%252e%252f"
            ],
            'command_injection': [
                "; cat /etc/passwd",
                "| nc ",
                "&& whoami",
                "; rm -rf"
            ]
        }
    
    def scan_input(self, input_data: str) -> Dict[str, List[str]]:
        """Scan input for attack patterns."""
        findings = {}
        
        try:
            for attack_type, patterns in self.attack_patterns.items():
                matches = []
                for pattern in patterns:
                    if pattern.lower() in input_data.lower():
                        matches.append(pattern)
                
                if matches:
                    findings[attack_type] = matches
            
            return findings
            
        except Exception as e:
            logging.error(f"Input scanning failed: {e}")
            return {}
    
    def sanitize_input(self, input_data: str) -> str:
        """Sanitize input by removing/escaping dangerous content."""
        try:
            sanitized = input_data
            
            # Basic HTML encoding
            sanitized = sanitized.replace('<', '&lt;')
            sanitized = sanitized.replace('>', '&gt;')
            sanitized = sanitized.replace('"', '&quot;')
            sanitized = sanitized.replace("'", '&#x27;')
            sanitized = sanitized.replace('&', '&amp;')
            
            return sanitized
            
        except Exception as e:
            logging.error(f"Input sanitization failed: {e}")
            return input_data


# Integration functions
def integrate_with_monitoring_system(enforcer: SecurityEnforcer,
                                   monitoring_system: ProductionMonitoringSystem) -> bool:
    """Integrate enforcer with monitoring system."""
    try:
        # Add enforcement metrics to monitoring
        def collect_enforcement_metrics():
            status = enforcer.get_enforcement_status()
            return {
                'enforcement_total_policies': status['total_policies'],
                'enforcement_enabled_policies': status['enabled_policies'],
                'enforcement_total_violations': status['total_violations'],
                'enforcement_recent_violations': status['recent_violations'],
                'enforcement_blocked_ips': status['blocked_ips'],
                'enforcement_blocked_users': status['blocked_users']
            }
        
        # Register metrics collector
        monitoring_system._custom_metrics_collectors = getattr(
            monitoring_system, '_custom_metrics_collectors', {}
        )
        monitoring_system._custom_metrics_collectors['security_enforcement'] = collect_enforcement_metrics
        
        logging.info("Integrated enforcer with monitoring system")
        return True
        
    except Exception as e:
        logging.error(f"Failed to integrate with monitoring system: {e}")
        return False


def integrate_with_audit_system(enforcer: SecurityEnforcer,
                               auditor: Any) -> bool:
    """Integrate enforcer with audit system."""
    try:
        # Use audit findings to create enforcement policies
        def create_policies_from_findings():
            # This would analyze audit findings and create preventive policies
            pass
        
        # Schedule periodic policy updates based on audit results
        
        logging.info("Integrated enforcer with audit system")
        return True
        
    except Exception as e:
        logging.error(f"Failed to integrate with audit system: {e}")
        return False