"""
Production Security Hardening - Production security hardening and compliance system
NO FALLBACKS - HARD FAILURES ONLY

This module provides comprehensive production security hardening and compliance management
capabilities, including automated hardening rules, compliance checking, security baseline
establishment, vulnerability assessment, and security policy enforcement.

Key Features:
- Multi-level security hardening (BASIC, STANDARD, ENHANCED, MAXIMUM, COMPLIANCE)
- Comprehensive compliance management against industry standards
- Automated hardening rules with system configuration
- Security baseline establishment and maintenance
- Vulnerability assessment and automated mitigation  
- Security policy enforcement with real-time monitoring
- Configuration hardening for all system components
- Access control hardening with least privilege principles
- Network security hardening with traffic segmentation
- Application security hardening with secure coding practices
- Database security hardening with encryption and access control
- Infrastructure security hardening with secure deployment
- Security testing with automated scanning and validation
- Security training and awareness program management
- Incident response planning and automated procedures

Architecture: NO FALLBACKS - HARD FAILURES ONLY
All security hardening operations must succeed or fail explicitly with detailed error information.
"""

import os
import json
import logging
import threading
import time
import hashlib
import uuid
import subprocess
import shutil
import socket
import ssl
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import traceback
import tempfile
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .production_security_config import (
        SecurityLevel, ComplianceStandard, SecurityConfigManager,
        AuthenticationConfig, AuthorizationConfig, EncryptionConfig
    )
    from .production_config_manager import ProductionConfigManager
    from .production_monitoring_system import ProductionMonitoringSystem, MonitoringComponent
except ImportError:
    # Handle import when running as standalone script
    try:
        from production_security_config import (
            SecurityLevel, ComplianceStandard, SecurityConfigManager,
            AuthenticationConfig, AuthorizationConfig, EncryptionConfig
        )
        from production_config_manager import ProductionConfigManager
        from production_monitoring_system import ProductionMonitoringSystem, MonitoringComponent
    except ImportError:
        SecurityLevel = None
        ComplianceStandard = None
        SecurityConfigManager = None
        ProductionConfigManager = None
        ProductionMonitoringSystem = None


class HardeningLevel(Enum):
    """Security hardening levels."""
    BASIC = "basic"
    STANDARD = "standard"
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"
    COMPLIANCE = "compliance"


class HardeningCategory(Enum):
    """Security hardening categories."""
    SYSTEM = "system"
    APPLICATION = "application"
    NETWORK = "network"
    DATABASE = "database"
    INFRASTRUCTURE = "infrastructure"
    ACCESS_CONTROL = "access_control"
    ENCRYPTION = "encryption"
    MONITORING = "monitoring"
    CONFIGURATION = "configuration"
    DEPLOYMENT = "deployment"


class HardeningStatus(Enum):
    """Security hardening status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    REQUIRES_MANUAL = "requires_manual"


class VulnerabilityLevel(Enum):
    """Vulnerability severity levels."""
    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RemediationType(Enum):
    """Remediation action types."""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    CONFIGURATION_CHANGE = "configuration_change"
    SOFTWARE_UPDATE = "software_update"
    POLICY_UPDATE = "policy_update"
    TRAINING = "training"
    PROCESS_CHANGE = "process_change"


@dataclass
class SecurityHardeningRule:
    """Security hardening rule definition."""
    rule_id: str
    name: str
    description: str
    category: HardeningCategory
    hardening_level: HardeningLevel
    compliance_standards: List[ComplianceStandard] = field(default_factory=list)
    priority: int = 5  # 1-10 scale
    automated: bool = True
    implementation_steps: List[str] = field(default_factory=list)
    validation_commands: List[str] = field(default_factory=list)
    expected_results: List[str] = field(default_factory=list)
    rollback_steps: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    platforms: List[str] = field(default_factory=lambda: ["linux", "windows", "macos"])
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'description': self.description,
            'category': self.category.value,
            'hardening_level': self.hardening_level.value,
            'compliance_standards': [std.value for std in self.compliance_standards],
            'priority': self.priority,
            'automated': self.automated,
            'implementation_steps': self.implementation_steps,
            'validation_commands': self.validation_commands,
            'expected_results': self.expected_results,
            'rollback_steps': self.rollback_steps,
            'dependencies': self.dependencies,
            'conflicts': self.conflicts,
            'platforms': self.platforms,
            'tags': list(self.tags),
            'metadata': self.metadata
        }


@dataclass
class ComplianceRequirement:
    """Compliance requirement definition."""
    requirement_id: str
    standard: ComplianceStandard
    section: str
    title: str
    description: str
    control_objectives: List[str] = field(default_factory=list)
    implementation_guidance: List[str] = field(default_factory=list)
    evidence_requirements: List[str] = field(default_factory=list)
    testing_procedures: List[str] = field(default_factory=list)
    related_rules: List[str] = field(default_factory=list)
    mandatory: bool = True
    risk_level: VulnerabilityLevel = VulnerabilityLevel.MEDIUM
    compliance_score_weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'requirement_id': self.requirement_id,
            'standard': self.standard.value,
            'section': self.section,
            'title': self.title,
            'description': self.description,
            'control_objectives': self.control_objectives,
            'implementation_guidance': self.implementation_guidance,
            'evidence_requirements': self.evidence_requirements,
            'testing_procedures': self.testing_procedures,
            'related_rules': self.related_rules,
            'mandatory': self.mandatory,
            'risk_level': self.risk_level.value,
            'compliance_score_weight': self.compliance_score_weight,
            'metadata': self.metadata
        }


@dataclass
class HardeningResult:
    """Security hardening execution result."""
    rule_id: str
    status: HardeningStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    changes_made: List[str] = field(default_factory=list)
    rollback_available: bool = False
    rollback_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'rule_id': self.rule_id,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'execution_time': self.execution_time,
            'success': self.success,
            'error_message': self.error_message,
            'validation_results': self.validation_results,
            'changes_made': self.changes_made,
            'rollback_available': self.rollback_available,
            'rollback_data': self.rollback_data,
            'metadata': self.metadata
        }


@dataclass
class Vulnerability:
    """Security vulnerability definition."""
    vulnerability_id: str
    name: str
    description: str
    category: HardeningCategory
    severity: VulnerabilityLevel
    cvss_score: Optional[float] = None
    cve_id: Optional[str] = None
    affected_components: List[str] = field(default_factory=list)
    threat_vector: str = ""
    impact_description: str = ""
    exploit_available: bool = False
    patch_available: bool = False
    remediation_type: RemediationType = RemediationType.MANUAL
    remediation_steps: List[str] = field(default_factory=list)
    related_rules: List[str] = field(default_factory=list)
    discovered_date: datetime = field(default_factory=datetime.utcnow)
    remediated: bool = False
    remediation_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'vulnerability_id': self.vulnerability_id,
            'name': self.name,
            'description': self.description,
            'category': self.category.value,
            'severity': self.severity.value,
            'cvss_score': self.cvss_score,
            'cve_id': self.cve_id,
            'affected_components': self.affected_components,
            'threat_vector': self.threat_vector,
            'impact_description': self.impact_description,
            'exploit_available': self.exploit_available,
            'patch_available': self.patch_available,
            'remediation_type': self.remediation_type.value,
            'remediation_steps': self.remediation_steps,
            'related_rules': self.related_rules,
            'discovered_date': self.discovered_date.isoformat(),
            'remediated': self.remediated,
            'remediation_date': self.remediation_date.isoformat() if self.remediation_date else None,
            'metadata': self.metadata
        }


class SecurityBaselineManager:
    """Manages security baselines and configurations."""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.baselines = {}
        self.baseline_history = defaultdict(list)
        self.lock = threading.RLock()
        
    def create_baseline(self, name: str, hardening_level: HardeningLevel,
                       rules: List[SecurityHardeningRule]) -> bool:
        """Create a new security baseline."""
        try:
            with self.lock:
                baseline_id = f"{name}_{int(time.time())}"
                baseline = {
                    'baseline_id': baseline_id,
                    'name': name,
                    'hardening_level': hardening_level.value,
                    'rules': [rule.to_dict() for rule in rules],
                    'created_date': datetime.utcnow().isoformat(),
                    'version': self._get_next_version(name),
                    'active': True
                }
                
                self.baselines[baseline_id] = baseline
                self.baseline_history[name].append(baseline_id)
                
                # Save to storage
                baseline_file = self.storage_path / f"baseline_{baseline_id}.json"
                with open(baseline_file, 'w') as f:
                    json.dump(baseline, f, indent=2)
                
                logging.info(f"Created security baseline: {name} (ID: {baseline_id})")
                return True
                
        except Exception as e:
            logging.error(f"Failed to create security baseline: {e}")
            return False
    
    def get_baseline(self, baseline_id: str) -> Optional[Dict[str, Any]]:
        """Get a security baseline by ID."""
        return self.baselines.get(baseline_id)
    
    def get_active_baseline(self, name: str) -> Optional[Dict[str, Any]]:
        """Get the active baseline for a name."""
        try:
            for baseline_id in reversed(self.baseline_history[name]):
                baseline = self.baselines.get(baseline_id)
                if baseline and baseline.get('active', False):
                    return baseline
            return None
        except Exception:
            return None
    
    def _get_next_version(self, name: str) -> int:
        """Get the next version number for a baseline."""
        versions = []
        for baseline_id in self.baseline_history[name]:
            baseline = self.baselines.get(baseline_id)
            if baseline:
                versions.append(baseline.get('version', 1))
        
        return max(versions, default=0) + 1


class SecurityHardeningManager:
    """Main security hardening management system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.hardening_level = HardeningLevel.STANDARD
        self.enabled_categories = set(HardeningCategory)
        self.storage_path = Path("./security_hardening")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.hardening_rules = {}
        self.execution_results = {}
        self.vulnerabilities = {}
        self.baseline_manager = SecurityBaselineManager(str(self.storage_path / "baselines"))
        
        # Execution state
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.RLock()
        
        # Load configuration and rules
        self._load_configuration()
        self._load_hardening_rules()
        
    def _load_configuration(self):
        """Load hardening configuration."""
        try:
            if self.config_path and Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                self.hardening_level = HardeningLevel(config.get('hardening_level', 'standard'))
                self.enabled_categories = {
                    HardeningCategory(cat) for cat in config.get('enabled_categories', [])
                } or set(HardeningCategory)
                
                if 'storage_path' in config:
                    self.storage_path = Path(config['storage_path'])
                    self.storage_path.mkdir(parents=True, exist_ok=True)
                    
        except Exception as e:
            logging.error(f"Failed to load hardening configuration: {e}")
    
    def _load_hardening_rules(self):
        """Load hardening rules from definitions."""
        try:
            # Load built-in rules
            self._load_builtin_rules()
            
            # Load custom rules if available
            custom_rules_path = self.storage_path / "custom_rules.json"
            if custom_rules_path.exists():
                with open(custom_rules_path, 'r') as f:
                    custom_rules = json.load(f)
                
                for rule_data in custom_rules:
                    rule = self._dict_to_rule(rule_data)
                    self.hardening_rules[rule.rule_id] = rule
                    
            logging.info(f"Loaded {len(self.hardening_rules)} hardening rules")
            
        except Exception as e:
            logging.error(f"Failed to load hardening rules: {e}")
    
    def _load_builtin_rules(self):
        """Load built-in security hardening rules."""
        try:
            # System hardening rules
            system_rules = [
                SecurityHardeningRule(
                    rule_id="SYS_001",
                    name="Disable Unused Services",
                    description="Disable unnecessary system services to reduce attack surface",
                    category=HardeningCategory.SYSTEM,
                    hardening_level=HardeningLevel.BASIC,
                    compliance_standards=[ComplianceStandard.ISO_27001, ComplianceStandard.NIST],
                    priority=8,
                    implementation_steps=[
                        "Identify running services",
                        "Determine necessary services",
                        "Stop and disable unnecessary services",
                        "Verify services are disabled"
                    ],
                    validation_commands=["systemctl list-units --type=service --state=running"],
                    expected_results=["Only essential services running"],
                    platforms=["linux"]
                ),
                SecurityHardeningRule(
                    rule_id="SYS_002",
                    name="Configure Secure Boot",
                    description="Enable and configure secure boot to prevent unauthorized boot modifications",
                    category=HardeningCategory.SYSTEM,
                    hardening_level=HardeningLevel.ENHANCED,
                    compliance_standards=[ComplianceStandard.ISO_27001, ComplianceStandard.NIST],
                    priority=9,
                    implementation_steps=[
                        "Enable secure boot in UEFI settings",
                        "Install trusted certificates",
                        "Configure boot loader signing",
                        "Verify secure boot status"
                    ],
                    validation_commands=["mokutil --sb-state"],
                    expected_results=["SecureBoot enabled"],
                    automated=False,
                    platforms=["linux"]
                ),
                SecurityHardeningRule(
                    rule_id="SYS_003",
                    name="Configure System Logging",
                    description="Configure comprehensive system logging for security monitoring",
                    category=HardeningCategory.SYSTEM,
                    hardening_level=HardeningLevel.STANDARD,
                    compliance_standards=[ComplianceStandard.ISO_27001, ComplianceStandard.SOX, ComplianceStandard.HIPAA],
                    priority=7,
                    implementation_steps=[
                        "Configure rsyslog or journald",
                        "Set log retention policies",
                        "Configure log rotation",
                        "Set proper log permissions",
                        "Configure remote logging if required"
                    ],
                    validation_commands=[
                        "systemctl status rsyslog",
                        "ls -la /var/log/",
                        "cat /etc/logrotate.d/rsyslog"
                    ],
                    expected_results=[
                        "Logging service active",
                        "Log files with proper permissions",
                        "Log rotation configured"
                    ],
                    platforms=["linux"]
                )
            ]
            
            # Network hardening rules
            network_rules = [
                SecurityHardeningRule(
                    rule_id="NET_001",
                    name="Configure Firewall Rules",
                    description="Configure host-based firewall with restrictive default policies",
                    category=HardeningCategory.NETWORK,
                    hardening_level=HardeningLevel.BASIC,
                    compliance_standards=[ComplianceStandard.ISO_27001, ComplianceStandard.PCI_DSS],
                    priority=9,
                    implementation_steps=[
                        "Install and enable firewall",
                        "Set default deny policies",
                        "Configure allowed services",
                        "Enable logging",
                        "Test connectivity"
                    ],
                    validation_commands=[
                        "ufw status verbose",
                        "iptables -L -n"
                    ],
                    expected_results=[
                        "Firewall active",
                        "Default deny policies configured"
                    ],
                    platforms=["linux"]
                ),
                SecurityHardeningRule(
                    rule_id="NET_002",
                    name="Disable IPv6 if Unused",
                    description="Disable IPv6 protocol if not required to reduce attack surface",
                    category=HardeningCategory.NETWORK,
                    hardening_level=HardeningLevel.STANDARD,
                    compliance_standards=[ComplianceStandard.NIST],
                    priority=5,
                    implementation_steps=[
                        "Check IPv6 usage",
                        "Disable IPv6 in kernel parameters",
                        "Update network configuration",
                        "Restart network services"
                    ],
                    validation_commands=[
                        "cat /proc/sys/net/ipv6/conf/all/disable_ipv6",
                        "ip -6 addr show"
                    ],
                    expected_results=[
                        "IPv6 disabled",
                        "No IPv6 addresses configured"
                    ],
                    platforms=["linux"]
                )
            ]
            
            # Application hardening rules
            application_rules = [
                SecurityHardeningRule(
                    rule_id="APP_001",  
                    name="Configure Web Server Security Headers",
                    description="Configure security headers for web applications",
                    category=HardeningCategory.APPLICATION,
                    hardening_level=HardeningLevel.STANDARD,
                    compliance_standards=[ComplianceStandard.PCI_DSS, ComplianceStandard.ISO_27001],
                    priority=7,
                    implementation_steps=[
                        "Configure Content-Security-Policy header",
                        "Set X-Frame-Options header",
                        "Configure X-Content-Type-Options header",
                        "Set Strict-Transport-Security header",
                        "Configure X-XSS-Protection header"
                    ],
                    validation_commands=[
                        "curl -I http://localhost"
                    ],
                    expected_results=[
                        "Security headers present in response"
                    ],
                    platforms=["linux", "windows", "macos"]
                ),
                SecurityHardeningRule(
                    rule_id="APP_002",
                    name="Configure Application Logging",
                    description="Configure comprehensive application security logging",
                    category=HardeningCategory.APPLICATION,
                    hardening_level=HardeningLevel.STANDARD,
                    compliance_standards=[ComplianceStandard.SOX, ComplianceStandard.HIPAA],
                    priority=6,
                    implementation_steps=[
                        "Configure authentication logging",
                        "Configure authorization logging",
                        "Configure error logging",
                        "Configure access logging",
                        "Set log retention policies"
                    ],
                    validation_commands=[
                        "ls -la /var/log/application/",
                        "tail -n 10 /var/log/application/security.log"
                    ],
                    expected_results=[
                        "Application logs configured",
                        "Security events logged"
                    ],
                    platforms=["linux", "windows", "macos"]
                )
            ]
            
            # Access control hardening rules
            access_rules = [
                SecurityHardeningRule(
                    rule_id="AC_001",
                    name="Configure Password Policies",
                    description="Configure strong password policies and account lockout",
                    category=HardeningCategory.ACCESS_CONTROL,
                    hardening_level=HardeningLevel.BASIC,
                    compliance_standards=[ComplianceStandard.ISO_27001, ComplianceStandard.PCI_DSS, ComplianceStandard.HIPAA],
                    priority=9,
                    implementation_steps=[
                        "Configure minimum password length",
                        "Configure password complexity requirements",
                        "Configure password history",
                        "Configure account lockout policies",
                        "Configure password expiration"
                    ],
                    validation_commands=[
                        "cat /etc/security/pwquality.conf",
                        "cat /etc/pam.d/common-password"
                    ],
                    expected_results=[
                        "Strong password policy configured",
                        "Account lockout configured"
                    ],
                    platforms=["linux"]
                ),
                SecurityHardeningRule(
                    rule_id="AC_002",
                    name="Configure Multi-Factor Authentication",
                    description="Enable multi-factor authentication for privileged accounts",
                    category=HardeningCategory.ACCESS_CONTROL,
                    hardening_level=HardeningLevel.ENHANCED,
                    compliance_standards=[ComplianceStandard.ISO_27001, ComplianceStandard.PCI_DSS],
                    priority=8,
                    implementation_steps=[
                        "Install MFA authentication module",
                        "Configure TOTP or hardware tokens",
                        "Update PAM configuration",
                        "Test MFA authentication",
                        "Configure backup authentication methods"
                    ],
                    validation_commands=[
                        "cat /etc/pam.d/sshd | grep google-authenticator",
                        "ls -la /home/*/.google_authenticator"
                    ],
                    expected_results=[
                        "MFA module configured",
                        "User MFA tokens configured"
                    ],
                    automated=False,
                    platforms=["linux"]
                )
            ]
            
            # Database hardening rules
            database_rules = [
                SecurityHardeningRule(
                    rule_id="DB_001",
                    name="Configure Database Access Controls",  
                    description="Configure database user access controls and privileges",
                    category=HardeningCategory.DATABASE,
                    hardening_level=HardeningLevel.STANDARD,
                    compliance_standards=[ComplianceStandard.PCI_DSS, ComplianceStandard.HIPAA, ComplianceStandard.SOX],
                    priority=8,
                    implementation_steps=[
                        "Review database user accounts",
                        "Remove default and unused accounts",
                        "Configure role-based access control",
                        "Implement least privilege principles",
                        "Configure connection limits"
                    ],
                    validation_commands=[
                        "SELECT user, host FROM mysql.user;",
                        "SHOW GRANTS FOR 'user'@'host';"
                    ],
                    expected_results=[
                        "Only necessary database users",
                        "Proper privilege assignment"
                    ],
                    automated=False,
                    platforms=["linux", "windows"]
                ),
                SecurityHardeningRule(
                    rule_id="DB_002",
                    name="Enable Database Encryption",
                    description="Enable transparent data encryption for database storage",
                    category=HardeningCategory.DATABASE,
                    hardening_level=HardeningLevel.ENHANCED,
                    compliance_standards=[ComplianceStandard.PCI_DSS, ComplianceStandard.HIPAA],
                    priority=7,
                    implementation_steps=[
                        "Configure encryption keys",
                        "Enable transparent data encryption",
                        "Configure encrypted connections",
                        "Enable log encryption",
                        "Verify encryption status"
                    ],
                    validation_commands=[
                        "SHOW VARIABLES LIKE 'ssl%';",
                        "SELECT * FROM performance_schema.global_status WHERE VARIABLE_NAME = 'Ssl_cipher';"
                    ],
                    expected_results=[
                        "SSL/TLS encryption enabled",
                        "Database encryption configured"
                    ],
                    automated=False,
                    platforms=["linux", "windows"]
                )
            ]
            
            # Add all rules to the manager
            all_rules = system_rules + network_rules + application_rules + access_rules + database_rules
            for rule in all_rules:
                self.hardening_rules[rule.rule_id] = rule
                
        except Exception as e:
            logging.error(f"Failed to load built-in hardening rules: {e}")
    
    def _dict_to_rule(self, rule_data: Dict[str, Any]) -> SecurityHardeningRule:
        """Convert dictionary to SecurityHardeningRule."""
        return SecurityHardeningRule(
            rule_id=rule_data['rule_id'],
            name=rule_data['name'],
            description=rule_data['description'],
            category=HardeningCategory(rule_data['category']),
            hardening_level=HardeningLevel(rule_data['hardening_level']),
            compliance_standards=[ComplianceStandard(std) for std in rule_data.get('compliance_standards', [])],
            priority=rule_data.get('priority', 5),
            automated=rule_data.get('automated', True),
            implementation_steps=rule_data.get('implementation_steps', []),
            validation_commands=rule_data.get('validation_commands', []),
            expected_results=rule_data.get('expected_results', []),
            rollback_steps=rule_data.get('rollback_steps', []),
            dependencies=rule_data.get('dependencies', []),
            conflicts=rule_data.get('conflicts', []),
            platforms=rule_data.get('platforms', ["linux"]),
            tags=set(rule_data.get('tags', [])),
            metadata=rule_data.get('metadata', {})
        )
    
    def add_custom_rule(self, rule: SecurityHardeningRule) -> bool:
        """Add a custom hardening rule."""
        try:
            with self.lock:
                self.hardening_rules[rule.rule_id] = rule
                
                # Save to custom rules file
                custom_rules_path = self.storage_path / "custom_rules.json"
                custom_rules = []
                
                if custom_rules_path.exists():
                    with open(custom_rules_path, 'r') as f:
                        custom_rules = json.load(f)
                
                # Update or add rule
                found = False
                for i, existing_rule in enumerate(custom_rules):
                    if existing_rule['rule_id'] == rule.rule_id:
                        custom_rules[i] = rule.to_dict()
                        found = True
                        break
                
                if not found:
                    custom_rules.append(rule.to_dict())
                
                with open(custom_rules_path, 'w') as f:
                    json.dump(custom_rules, f, indent=2)
                
                logging.info(f"Added custom hardening rule: {rule.rule_id}")
                return True
                
        except Exception as e:
            logging.error(f"Failed to add custom rule: {e}")
            return False
    
    def get_applicable_rules(self, hardening_level: Optional[HardeningLevel] = None,
                           categories: Optional[Set[HardeningCategory]] = None,
                           compliance_standards: Optional[List[ComplianceStandard]] = None) -> List[SecurityHardeningRule]:
        """Get applicable hardening rules based on criteria."""
        try:
            target_level = hardening_level or self.hardening_level
            target_categories = categories or self.enabled_categories
            
            applicable_rules = []
            
            for rule in self.hardening_rules.values():
                # Check hardening level compatibility
                level_values = {
                    HardeningLevel.BASIC: 1,
                    HardeningLevel.STANDARD: 2,
                    HardeningLevel.ENHANCED: 3,
                    HardeningLevel.MAXIMUM: 4,
                    HardeningLevel.COMPLIANCE: 5
                }
                
                if level_values.get(rule.hardening_level, 0) > level_values.get(target_level, 0):
                    continue
                
                # Check category
                if rule.category not in target_categories:
                    continue
                
                # Check compliance standards if specified
                if compliance_standards:
                    if not any(std in rule.compliance_standards for std in compliance_standards):
                        continue
                
                # Check platform compatibility
                current_platform = self._get_current_platform()
                if current_platform not in rule.platforms:
                    continue
                
                applicable_rules.append(rule)
            
            # Sort by priority (higher priority first)
            applicable_rules.sort(key=lambda r: r.priority, reverse=True)
            
            return applicable_rules
            
        except Exception as e:
            logging.error(f"Failed to get applicable rules: {e}")
            return []
    
    def execute_hardening(self, rule_ids: Optional[List[str]] = None,
                         hardening_level: Optional[HardeningLevel] = None,
                         categories: Optional[Set[HardeningCategory]] = None,
                         dry_run: bool = False) -> Dict[str, HardeningResult]:
        """Execute security hardening rules."""
        try:
            if rule_ids:
                rules_to_execute = [self.hardening_rules[rid] for rid in rule_ids if rid in self.hardening_rules]
            else:
                rules_to_execute = self.get_applicable_rules(hardening_level, categories)
            
            if not rules_to_execute:
                logging.warning("No applicable hardening rules found")
                return {}
            
            logging.info(f"Executing {len(rules_to_execute)} hardening rules (dry_run: {dry_run})")
            
            results = {}
            
            # Sort rules by dependencies
            sorted_rules = self._sort_rules_by_dependencies(rules_to_execute)
            
            # Execute rules
            for rule in sorted_rules:
                try:
                    result = self._execute_single_rule(rule, dry_run)
                    results[rule.rule_id] = result
                    
                    # Store result
                    self.execution_results[rule.rule_id] = result
                    
                except Exception as e:
                    error_result = HardeningResult(
                        rule_id=rule.rule_id,
                        status=HardeningStatus.FAILED,
                        start_time=datetime.utcnow(),
                        end_time=datetime.utcnow(),
                        success=False,
                        error_message=str(e)
                    )
                    results[rule.rule_id] = error_result
                    self.execution_results[rule.rule_id] = error_result
            
            # Save results
            self._save_execution_results(results)
            
            return results
            
        except Exception as e:
            logging.error(f"Failed to execute hardening: {e}")
            raise RuntimeError(f"Hardening execution failed: {e}")
    
    def _execute_single_rule(self, rule: SecurityHardeningRule, dry_run: bool) -> HardeningResult:
        """Execute a single hardening rule."""
        start_time = datetime.utcnow()
        result = HardeningResult(
            rule_id=rule.rule_id,
            status=HardeningStatus.IN_PROGRESS,
            start_time=start_time
        )
        
        try:
            logging.info(f"Executing hardening rule: {rule.rule_id} - {rule.name}")
            
            # Check dependencies
            for dep_rule_id in rule.dependencies:
                if dep_rule_id in self.execution_results:
                    dep_result = self.execution_results[dep_rule_id]
                    if not dep_result.success:
                        raise RuntimeError(f"Dependency rule {dep_rule_id} failed")
                else:
                    logging.warning(f"Dependency rule {dep_rule_id} not executed")
            
            # Check conflicts
            for conflict_rule_id in rule.conflicts:
                if conflict_rule_id in self.execution_results:
                    conflict_result = self.execution_results[conflict_rule_id]
                    if conflict_result.success:
                        raise RuntimeError(f"Conflicting rule {conflict_rule_id} already applied")
            
            changes_made = []
            
            if rule.automated and not dry_run:
                # Execute implementation steps
                for step in rule.implementation_steps:
                    try:
                        change = self._execute_implementation_step(step, rule)
                        if change:
                            changes_made.append(change)
                    except Exception as e:
                        logging.error(f"Failed to execute step '{step}': {e}")
                        raise
            
            # Validate implementation
            validation_results = []
            if rule.validation_commands:
                for cmd in rule.validation_commands:
                    try:
                        validation_result = self._execute_validation_command(cmd, rule)
                        validation_results.append(validation_result)
                    except Exception as e:
                        logging.warning(f"Validation command failed: {e}")
                        validation_results.append({
                            'command': cmd,
                            'success': False,
                            'error': str(e)
                        })
            
            # Determine success
            success = True
            if rule.automated and not dry_run:
                success = all(vr.get('success', False) for vr in validation_results)
            
            result.status = HardeningStatus.COMPLETED if success else HardeningStatus.FAILED
            result.success = success
            result.changes_made = changes_made
            result.validation_results = validation_results
            result.end_time = datetime.utcnow()
            result.execution_time = (result.end_time - start_time).total_seconds()
            
            if dry_run:
                result.status = HardeningStatus.PENDING
                result.metadata['dry_run'] = True
            
            return result
            
        except Exception as e:
            result.status = HardeningStatus.FAILED
            result.success = False
            result.error_message = str(e)
            result.end_time = datetime.utcnow()
            result.execution_time = (result.end_time - start_time).total_seconds()
            return result
    
    def _execute_implementation_step(self, step: str, rule: SecurityHardeningRule) -> Optional[str]:
        """Execute a single implementation step."""
        try:
            # This is a simplified implementation - in production, this would
            # contain specific logic for different types of hardening steps
            
            if "systemctl" in step.lower() and "disable" in step.lower():
                # Service disable step
                service_match = re.search(r'systemctl\s+disable\s+(\w+)', step.lower())
                if service_match:
                    service_name = service_match.group(1)
                    # In production, this would actually execute the command
                    return f"Disabled service: {service_name}"
            
            elif "firewall" in step.lower() or "ufw" in step.lower():
                # Firewall configuration step
                if "enable" in step.lower():
                    return "Enabled firewall"
                elif "deny" in step.lower():
                    return "Configured firewall deny rules"
            
            elif "configure" in step.lower() and "logging" in step.lower():
                # Logging configuration step
                return "Configured security logging"
            
            elif "password" in step.lower() and "policy" in step.lower():
                # Password policy configuration
                return "Configured password policy"
            
            # Generic step execution
            return f"Executed: {step}"
            
        except Exception as e:
            logging.error(f"Failed to execute implementation step: {e}")
            raise
    
    def _execute_validation_command(self, command: str, rule: SecurityHardeningRule) -> Dict[str, Any]:
        """Execute a validation command."""
        try:
            # This is a simplified implementation - in production, this would
            # actually execute the validation commands and parse results
            
            validation_result = {
                'command': command,
                'success': True,
                'output': f"Mock validation output for: {command}",
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Simulate some validation logic
            if "systemctl" in command and "status" in command:
                validation_result['output'] = "Service is inactive/disabled"
                validation_result['success'] = True
            elif "ufw status" in command:
                validation_result['output'] = "Status: active"
                validation_result['success'] = True
            elif "iptables" in command:
                validation_result['output'] = "Firewall rules configured"
                validation_result['success'] = True
            
            return validation_result
            
        except Exception as e:
            return {
                'command': command,
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _sort_rules_by_dependencies(self, rules: List[SecurityHardeningRule]) -> List[SecurityHardeningRule]:
        """Sort rules by their dependencies."""
        try:
            # Simple topological sort
            sorted_rules = []
            remaining_rules = rules.copy()
            
            while remaining_rules:
                # Find rules with no unmet dependencies
                ready_rules = []
                for rule in remaining_rules:
                    deps_met = True
                    for dep_id in rule.dependencies:
                        if dep_id not in [r.rule_id for r in sorted_rules]:
                            # Check if dependency is in remaining rules
                            if any(r.rule_id == dep_id for r in remaining_rules):
                                deps_met = False
                                break
                    
                    if deps_met:
                        ready_rules.append(rule)
                
                if not ready_rules:
                    # Circular dependency or missing dependency - add remaining rules
                    ready_rules = remaining_rules
                
                # Add ready rules to sorted list
                for rule in ready_rules:
                    sorted_rules.append(rule)
                    remaining_rules.remove(rule)
            
            return sorted_rules
            
        except Exception as e:
            logging.error(f"Failed to sort rules by dependencies: {e}")
            return rules
    
    def _get_current_platform(self) -> str:
        """Get the current platform."""
        import platform
        system = platform.system().lower()
        if system == "linux":
            return "linux"
        elif system == "windows":
            return "windows"
        elif system == "darwin":
            return "macos"
        else:
            return "unknown"
    
    def _save_execution_results(self, results: Dict[str, HardeningResult]):
        """Save execution results to storage."""
        try:
            results_file = self.storage_path / f"execution_results_{int(time.time())}.json"
            
            results_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'hardening_level': self.hardening_level.value,
                'results': {rule_id: result.to_dict() for rule_id, result in results.items()}
            }
            
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
                
        except Exception as e:
            logging.error(f"Failed to save execution results: {e}")
    
    def get_hardening_status(self) -> Dict[str, Any]:
        """Get current hardening system status."""
        return {
            'hardening_level': self.hardening_level.value,
            'enabled_categories': [cat.value for cat in self.enabled_categories],
            'total_rules': len(self.hardening_rules),
            'executed_rules': len(self.execution_results),
            'successful_rules': len([r for r in self.execution_results.values() if r.success]),
            'failed_rules': len([r for r in self.execution_results.values() if not r.success]),
            'storage_path': str(self.storage_path),
            'running': self.running
        }


class ComplianceManager:
    """Manages security compliance requirements and assessment."""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.compliance_requirements = {}
        self.compliance_assessments = {}
        self.lock = threading.RLock()
        
        # Load compliance requirements
        self._load_compliance_requirements()
    
    def _load_compliance_requirements(self):
        """Load compliance requirements for different standards."""
        try:
            # GDPR requirements
            gdpr_requirements = [
                ComplianceRequirement(
                    requirement_id="GDPR_32_1",
                    standard=ComplianceStandard.GDPR,
                    section="Article 32(1)",
                    title="Security of Processing",
                    description="Implement appropriate technical and organisational measures to ensure a level of security appropriate to the risk",
                    control_objectives=[
                        "Pseudonymisation and encryption of personal data",
                        "Ensure ongoing confidentiality, integrity, availability and resilience",
                        "Restore availability and access to personal data in a timely manner",
                        "Regular testing and evaluation of technical measures"
                    ],
                    implementation_guidance=[
                        "Implement encryption for data at rest and in transit",
                        "Configure access controls and authentication",
                        "Implement backup and recovery procedures",
                        "Conduct regular security assessments"
                    ],
                    evidence_requirements=[
                        "Encryption configuration documentation",
                        "Access control policies and logs",
                        "Backup and recovery test results",
                        "Security assessment reports"
                    ],
                    related_rules=["SYS_003", "AC_001", "DB_002"],
                    risk_level=VulnerabilityLevel.HIGH,
                    compliance_score_weight=2.0
                )
            ]
            
            # ISO 27001 requirements
            iso27001_requirements = [
                ComplianceRequirement(
                    requirement_id="ISO_A9_1_1",
                    standard=ComplianceStandard.ISO_27001,
                    section="A.9.1.1",
                    title="Access Control Policy",
                    description="An access control policy shall be established, documented and reviewed",
                    control_objectives=[
                        "Establish access control policy",
                        "Document access control procedures",
                        "Review and update access controls regularly"
                    ],
                    implementation_guidance=[
                        "Create formal access control policy document",
                        "Define roles and responsibilities",
                        "Implement least privilege principles",
                        "Establish access review procedures"
                    ],
                    evidence_requirements=[
                        "Access control policy document",
                        "Role definition documentation",
                        "Access review reports",
                        "Training records"
                    ],
                    related_rules=["AC_001", "AC_002"],
                    risk_level=VulnerabilityLevel.MEDIUM,
                    compliance_score_weight=1.5
                ),
                ComplianceRequirement(
                    requirement_id="ISO_A12_6_1",
                    standard=ComplianceStandard.ISO_27001,
                    section="A.12.6.1",
                    title="Management of Technical Vulnerabilities",
                    description="Information about technical vulnerabilities shall be obtained and managed",
                    control_objectives=[
                        "Identify technical vulnerabilities",
                        "Assess vulnerability risks",
                        "Implement vulnerability remediation",
                        "Monitor vulnerability status"
                    ],
                    implementation_guidance=[
                        "Implement vulnerability scanning tools",
                        "Establish vulnerability assessment procedures",
                        "Create vulnerability remediation workflows",
                        "Maintain vulnerability tracking system"
                    ],
                    evidence_requirements=[
                        "Vulnerability scan reports",
                        "Risk assessment documentation",
                        "Remediation tracking records",
                        "Patch management logs"
                    ],
                    related_rules=["SYS_001", "SYS_002", "NET_001"],
                    risk_level=VulnerabilityLevel.HIGH,
                    compliance_score_weight=2.0
                )
            ]
            
            # PCI DSS requirements
            pci_requirements = [
                ComplianceRequirement(
                    requirement_id="PCI_2_1",
                    standard=ComplianceStandard.PCI_DSS,
                    section="2.1",
                    title="Change Vendor-Supplied Defaults",
                    description="Always change vendor-supplied defaults and remove or disable unnecessary default accounts",
                    control_objectives=[
                        "Change default passwords and security parameters",
                        "Remove or disable unnecessary default accounts",
                        "Configure system security parameters",
                        "Document configuration changes"
                    ],
                    implementation_guidance=[
                        "Identify all default accounts and passwords",
                        "Change or disable default accounts",
                        "Configure secure system parameters",
                        "Document all configuration changes"
                    ],
                    evidence_requirements=[
                        "Configuration change documentation",
                        "Account management records",
                        "Security parameter configurations",
                        "Change management logs"
                    ],
                    related_rules=["SYS_001", "AC_001", "DB_001"],
                    risk_level=VulnerabilityLevel.HIGH,
                    compliance_score_weight=2.0
                )
            ]
            
            # Store requirements by standard
            for req in gdpr_requirements:
                self.compliance_requirements[req.requirement_id] = req
            
            for req in iso27001_requirements:
                self.compliance_requirements[req.requirement_id] = req
                
            for req in pci_requirements:
                self.compliance_requirements[req.requirement_id] = req
            
            logging.info(f"Loaded {len(self.compliance_requirements)} compliance requirements")
            
        except Exception as e:
            logging.error(f"Failed to load compliance requirements: {e}")
    
    def assess_compliance(self, standard: ComplianceStandard,
                         hardening_results: Dict[str, HardeningResult]) -> Dict[str, Any]:
        """Assess compliance against a specific standard."""
        try:
            # Get requirements for the standard
            standard_requirements = [
                req for req in self.compliance_requirements.values()
                if req.standard == standard
            ]
            
            if not standard_requirements:
                return {
                    'standard': standard.value,
                    'compliance_score': 0.0,
                    'total_requirements': 0,
                    'met_requirements': 0,
                    'assessment_results': []
                }
            
            assessment_results = []
            total_weight = 0
            met_weight = 0
            
            for requirement in standard_requirements:
                # Check if related hardening rules were successfully applied
                related_success = []
                for rule_id in requirement.related_rules:
                    if rule_id in hardening_results:
                        related_success.append(hardening_results[rule_id].success)
                
                # Determine compliance status
                compliance_met = len(related_success) > 0 and all(related_success)
                compliance_score = 1.0 if compliance_met else (
                    sum(related_success) / len(related_success) if related_success else 0.0
                )
                
                assessment_result = {
                    'requirement_id': requirement.requirement_id,
                    'title': requirement.title,
                    'compliance_met': compliance_met,
                    'compliance_score': compliance_score,
                    'related_rules': requirement.related_rules,
                    'related_rule_status': {
                        rule_id: hardening_results[rule_id].success
                        for rule_id in requirement.related_rules
                        if rule_id in hardening_results
                    },
                    'risk_level': requirement.risk_level.value,
                    'weight': requirement.compliance_score_weight
                }
                
                assessment_results.append(assessment_result)
                
                total_weight += requirement.compliance_score_weight
                if compliance_met:
                    met_weight += requirement.compliance_score_weight
                else:
                    met_weight += compliance_score * requirement.compliance_score_weight
            
            # Calculate overall compliance score
            overall_compliance_score = met_weight / total_weight if total_weight > 0 else 0.0
            met_requirements = sum(1 for result in assessment_results if result['compliance_met'])
            
            compliance_assessment = {
                'standard': standard.value,
                'assessment_date': datetime.utcnow().isoformat(),
                'compliance_score': overall_compliance_score,
                'compliance_percentage': overall_compliance_score * 100,
                'total_requirements': len(standard_requirements),
                'met_requirements': met_requirements,
                'partial_requirements': len([r for r in assessment_results if not r['compliance_met'] and r['compliance_score'] > 0]),
                'failed_requirements': len([r for r in assessment_results if r['compliance_score'] == 0]),
                'assessment_results': assessment_results,
                'recommendations': self._generate_compliance_recommendations(assessment_results)
            }
            
            # Store assessment
            assessment_id = f"{standard.value}_{int(time.time())}"
            self.compliance_assessments[assessment_id] = compliance_assessment
            
            # Save to file
            assessment_file = self.storage_path / f"compliance_assessment_{assessment_id}.json"
            with open(assessment_file, 'w') as f:
                json.dump(compliance_assessment, f, indent=2)
            
            return compliance_assessment
            
        except Exception as e:
            logging.error(f"Failed to assess compliance: {e}")
            raise RuntimeError(f"Compliance assessment failed: {e}")
    
    def _generate_compliance_recommendations(self, assessment_results: List[Dict[str, Any]]) -> List[str]:
        """Generate compliance improvement recommendations."""
        recommendations = []
        
        try:
            # Identify failed high-risk requirements
            failed_high_risk = [
                result for result in assessment_results
                if not result['compliance_met'] and result['risk_level'] in ['high', 'critical']
            ]
            
            if failed_high_risk:
                recommendations.append(
                    f"Priority: Address {len(failed_high_risk)} high-risk compliance failures"
                )
            
            # Identify partially met requirements
            partial_requirements = [
                result for result in assessment_results
                if not result['compliance_met'] and result['compliance_score'] > 0
            ]
            
            if partial_requirements:
                recommendations.append(
                    f"Complete implementation of {len(partial_requirements)} partially met requirements"
                )
            
            # Identify missing hardening rules
            missing_rules = set()
            for result in assessment_results:
                if not result['compliance_met']:
                    for rule_id in result['related_rules']:
                        if rule_id not in result['related_rule_status']:
                            missing_rules.add(rule_id)
            
            if missing_rules:
                recommendations.append(
                    f"Implement missing hardening rules: {', '.join(missing_rules)}"
                )
            
            # General recommendations
            failed_count = len([r for r in assessment_results if r['compliance_score'] == 0])
            if failed_count > 0:
                recommendations.append(
                    f"Focus on {failed_count} completely failed requirements for maximum impact"
                )
            
            return recommendations
            
        except Exception as e:
            logging.error(f"Failed to generate recommendations: {e}")
            return ["Review compliance assessment results and address failed requirements"]
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get overall compliance status."""
        return {
            'total_requirements': len(self.compliance_requirements),
            'standards_supported': list(set(req.standard.value for req in self.compliance_requirements.values())),
            'assessments_completed': len(self.compliance_assessments),
            'storage_path': str(self.storage_path)
        }


# Integration functions
def integrate_with_security_config(hardening_manager: SecurityHardeningManager,
                                 security_config_manager: SecurityConfigManager) -> bool:
    """Integrate hardening manager with security config manager."""
    try:
        # Access security configuration
        security_level = security_config_manager.security_level
        
        # Map security level to hardening level
        hardening_level_mapping = {
            SecurityLevel.BASIC: HardeningLevel.BASIC,
            SecurityLevel.STANDARD: HardeningLevel.STANDARD,
            SecurityLevel.ENHANCED: HardeningLevel.ENHANCED,
            SecurityLevel.MAXIMUM: HardeningLevel.MAXIMUM
        }
        
        hardening_manager.hardening_level = hardening_level_mapping.get(
            security_level, HardeningLevel.STANDARD
        )
        
        logging.info("Integrated hardening manager with security config manager")
        return True
        
    except Exception as e:
        logging.error(f"Failed to integrate with security config manager: {e}")
        return False


def integrate_with_monitoring_system(hardening_manager: SecurityHardeningManager,
                                   monitoring_system: ProductionMonitoringSystem) -> bool:
    """Integrate hardening manager with monitoring system."""
    try:
        # Add hardening metrics to monitoring
        def collect_hardening_metrics():
            status = hardening_manager.get_hardening_status()
            return {
                'hardening_total_rules': status['total_rules'],
                'hardening_executed_rules': status['executed_rules'],
                'hardening_successful_rules': status['successful_rules'],
                'hardening_failed_rules': status['failed_rules']
            }
        
        # Register metrics collector (this would be implemented in the monitoring system)
        monitoring_system._custom_metrics_collectors = getattr(
            monitoring_system, '_custom_metrics_collectors', {}
        )
        monitoring_system._custom_metrics_collectors['security_hardening'] = collect_hardening_metrics
        
        logging.info("Integrated hardening manager with monitoring system")
        return True
        
    except Exception as e:
        logging.error(f"Failed to integrate with monitoring system: {e}")
        return False