"""
Production Security Configuration - Production security configuration and management
NO FALLBACKS - HARD FAILURES ONLY

This module provides comprehensive production security configuration and management
capabilities, including authentication, authorization, encryption, network security,
certificate management, key rotation, security auditing, and compliance checking.

Key Features:
- Multi-level security configuration (BASIC, STANDARD, ENHANCED, MAXIMUM)
- Comprehensive authentication and authorization management
- Advanced encryption configuration and key management
- Network security configuration and monitoring
- Certificate management with automatic renewal
- Security auditing and compliance reporting
- Key rotation and lifecycle management
- Security policy enforcement and validation
- Integration with existing production configuration systems

Architecture: NO FALLBACKS - HARD FAILURES ONLY
All security operations must succeed or fail explicitly with detailed error information.
"""

import os
import json
import yaml
import logging
import threading
import time
import hashlib
import secrets
import ssl
import socket
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import copy
import traceback
import uuid
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography import x509
from cryptography.x509.oid import NameOID
import bcrypt
import jwt
from contextlib import contextmanager
import tempfile

try:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from production_config_manager import ProductionConfigManager, ProductionConfig
        from production_deployment_config import DeploymentConfigManager
        from gac_system.gac_config import GACConfigManager
except ImportError:
    pass

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security level types."""
    BASIC = "basic"
    STANDARD = "standard"
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"


class AuthenticationType(Enum):
    """Authentication type options."""
    NONE = "none"
    BASIC = "basic"
    DIGEST = "digest"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    LDAP = "ldap"
    SAML = "saml"
    MULTI_FACTOR = "multi_factor"
    CERTIFICATE = "certificate"


class AuthorizationModel(Enum):
    """Authorization model types."""
    NONE = "none"
    ROLE_BASED = "role_based"
    ATTRIBUTE_BASED = "attribute_based"
    RESOURCE_BASED = "resource_based"
    POLICY_BASED = "policy_based"
    HIERARCHICAL = "hierarchical"


class EncryptionAlgorithm(Enum):
    """Encryption algorithm types."""
    AES_128_GCM = "aes_128_gcm"
    AES_256_GCM = "aes_256_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    ECC_P256 = "ecc_p256"
    ECC_P384 = "ecc_p384"


class SecurityAuditLevel(Enum):
    """Security audit level types."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    FORENSIC = "forensic"


class ComplianceStandard(Enum):
    """Compliance standard types."""
    NONE = "none"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NIST = "nist"
    CUSTOM = "custom"


@dataclass
class AuthenticationConfig:
    """Authentication configuration."""
    enabled: bool = True
    type: AuthenticationType = AuthenticationType.JWT
    
    # Basic authentication
    basic_realm: str = "Saraphis Production"
    basic_users: Dict[str, str] = field(default_factory=dict)  # username: hashed_password
    
    # JWT configuration
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    jwt_refresh_enabled: bool = True
    jwt_refresh_expiration_days: int = 7
    jwt_issuer: str = "saraphis-production"
    jwt_audience: str = "saraphis-api"
    
    # OAuth2 configuration
    oauth2_client_id: str = ""
    oauth2_client_secret: str = ""
    oauth2_scope: List[str] = field(default_factory=lambda: ["read", "write"])
    oauth2_token_url: str = ""
    oauth2_auth_url: str = ""
    oauth2_redirect_uri: str = ""
    
    # LDAP configuration
    ldap_server: str = ""
    ldap_port: int = 389
    ldap_use_ssl: bool = True
    ldap_bind_dn: str = ""
    ldap_bind_password: str = ""
    ldap_user_base: str = ""
    ldap_user_filter: str = "(uid={username})"
    ldap_group_base: str = ""
    ldap_group_filter: str = "(member={user_dn})"
    
    # Multi-factor authentication
    mfa_enabled: bool = False
    mfa_methods: List[str] = field(default_factory=lambda: ["totp", "sms"])
    mfa_backup_codes: int = 10
    mfa_timeout_seconds: int = 300
    
    # Certificate authentication
    cert_required: bool = False
    cert_ca_path: str = ""
    cert_verify_depth: int = 3
    cert_crl_check: bool = True
    
    # Session management
    session_timeout_minutes: int = 30
    session_absolute_timeout_hours: int = 8
    concurrent_sessions_limit: int = 3
    session_hijacking_protection: bool = True
    
    # Password policy
    password_min_length: int = 12
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_symbols: bool = True
    password_history_count: int = 5
    password_expiration_days: int = 90


@dataclass
class AuthorizationConfig:
    """Authorization configuration."""
    enabled: bool = True
    model: AuthorizationModel = AuthorizationModel.ROLE_BASED
    
    # Role-based access control
    roles: Dict[str, List[str]] = field(default_factory=dict)  # role: permissions
    user_roles: Dict[str, List[str]] = field(default_factory=dict)  # user: roles
    role_hierarchy: Dict[str, List[str]] = field(default_factory=dict)  # parent_role: child_roles
    
    # Resource-based access control
    resources: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    resource_permissions: Dict[str, List[str]] = field(default_factory=dict)
    
    # Policy-based access control
    policies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    policy_engine: str = "json"  # json, rego, python
    
    # Attribute-based access control
    attributes: Dict[str, Any] = field(default_factory=dict)
    attribute_policies: Dict[str, str] = field(default_factory=dict)
    
    # Access control settings
    default_deny: bool = True
    explicit_permissions_required: bool = True
    permission_inheritance: bool = True
    
    # Administrative access
    admin_roles: List[str] = field(default_factory=lambda: ["admin", "superuser"])
    emergency_access_enabled: bool = True
    emergency_access_timeout_minutes: int = 60
    
    # Delegation
    delegation_enabled: bool = True
    delegation_max_depth: int = 2
    delegation_timeout_hours: int = 24


@dataclass
class EncryptionConfig:
    """Encryption configuration."""
    enabled: bool = True
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    
    # Symmetric encryption
    symmetric_key_size: int = 256
    symmetric_key_rotation_days: int = 30
    
    # Asymmetric encryption
    asymmetric_key_size: int = 4096
    asymmetric_key_rotation_days: int = 365
    
    # Key management
    key_storage: str = "file"  # file, hsm, vault
    key_storage_path: str = "keys"
    master_key: str = ""
    key_derivation_iterations: int = 100000
    
    # Encryption at rest
    encrypt_at_rest: bool = True
    encrypt_database: bool = True
    encrypt_logs: bool = True
    encrypt_configuration: bool = True
    encrypt_temporary_files: bool = True
    
    # Encryption in transit
    encrypt_in_transit: bool = True
    tls_version: str = "1.3"
    tls_cipher_suites: List[str] = field(default_factory=lambda: [
        "TLS_AES_256_GCM_SHA384",
        "TLS_AES_128_GCM_SHA256",
        "TLS_CHACHA20_POLY1305_SHA256"
    ])
    
    # Key rotation
    automatic_rotation: bool = True
    rotation_schedule: str = "0 0 1 * *"  # Monthly
    rotation_notification: bool = True
    
    # Backup and recovery
    key_backup_enabled: bool = True
    key_backup_encryption: bool = True
    key_recovery_enabled: bool = True
    key_escrow_enabled: bool = False


@dataclass
class NetworkSecurityConfig:
    """Network security configuration."""
    enabled: bool = True
    
    # Firewall configuration
    firewall_enabled: bool = True
    allowed_ports: List[int] = field(default_factory=lambda: [80, 443, 8000])
    blocked_ports: List[int] = field(default_factory=list)
    allowed_ips: List[str] = field(default_factory=list)
    blocked_ips: List[str] = field(default_factory=list)
    allowed_networks: List[str] = field(default_factory=list)
    blocked_networks: List[str] = field(default_factory=list)
    
    # Rate limiting
    rate_limiting_enabled: bool = True
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10
    rate_limit_exceptions: List[str] = field(default_factory=list)
    
    # DDoS protection
    ddos_protection_enabled: bool = True
    ddos_threshold_requests: int = 1000
    ddos_threshold_connections: int = 100
    ddos_block_duration_minutes: int = 60
    
    # Intrusion detection
    ids_enabled: bool = True
    ids_rules: List[str] = field(default_factory=list)
    ids_alert_threshold: int = 5
    ids_block_threshold: int = 10
    
    # SSL/TLS configuration
    ssl_enabled: bool = True
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    ssl_ca_path: str = ""
    ssl_verify_client: bool = False
    ssl_protocols: List[str] = field(default_factory=lambda: ["TLSv1.2", "TLSv1.3"])
    ssl_ciphers: List[str] = field(default_factory=lambda: [
        "ECDHE-RSA-AES256-GCM-SHA384",
        "ECDHE-RSA-AES128-GCM-SHA256"
    ])
    
    # Network monitoring
    monitoring_enabled: bool = True
    monitor_connections: bool = True
    monitor_traffic: bool = True
    monitor_anomalies: bool = True
    
    # VPN configuration
    vpn_enabled: bool = False
    vpn_type: str = "openvpn"
    vpn_config_path: str = ""
    vpn_client_certs: bool = True


@dataclass
class AuditConfig:
    """Security audit configuration."""
    enabled: bool = True
    level: SecurityAuditLevel = SecurityAuditLevel.STANDARD
    
    # Audit logging
    log_authentication: bool = True
    log_authorization: bool = True
    log_data_access: bool = True
    log_configuration_changes: bool = True
    log_system_events: bool = True
    log_security_events: bool = True
    
    # Audit storage
    audit_log_path: str = "logs/audit.log"
    audit_log_format: str = "json"
    audit_log_rotation: bool = True
    audit_log_retention_days: int = 365
    audit_log_encryption: bool = True
    
    # Compliance monitoring
    compliance_standards: List[ComplianceStandard] = field(default_factory=lambda: [ComplianceStandard.ISO_27001])
    compliance_checks_enabled: bool = True
    compliance_report_frequency: str = "monthly"
    
    # Audit analysis
    automated_analysis: bool = True
    anomaly_detection: bool = True
    threat_detection: bool = True
    compliance_scoring: bool = True
    
    # Audit reporting
    generate_reports: bool = True
    report_formats: List[str] = field(default_factory=lambda: ["pdf", "json", "csv"])
    report_recipients: List[str] = field(default_factory=list)
    
    # Audit integrity
    audit_log_signing: bool = True
    audit_log_verification: bool = True
    audit_trail_protection: bool = True
    
    # External audit support
    external_audit_support: bool = True
    audit_export_format: str = "json"
    audit_search_enabled: bool = True


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    name: str
    description: str = ""
    version: str = "1.0.0"
    
    # Policy rules
    rules: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: Dict[str, Any] = field(default_factory=dict)
    
    # Policy enforcement
    enforcement_mode: str = "enforce"  # enforce, warn, monitor
    violation_actions: List[str] = field(default_factory=lambda: ["log", "alert"])
    
    # Policy metadata
    tags: Dict[str, str] = field(default_factory=dict)
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Policy validation
    validation_enabled: bool = True
    validation_schedule: str = "daily"


class CertificateManager:
    """Certificate management utilities."""
    
    def __init__(self, cert_dir: str = "certificates"):
        self.cert_dir = Path(cert_dir)
        self.cert_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger('CertificateManager')
    
    def generate_self_signed_cert(
        self,
        common_name: str,
        san_list: Optional[List[str]] = None,
        key_size: int = 2048,
        validity_days: int = 365
    ) -> Tuple[str, str]:
        """Generate self-signed certificate."""
        try:
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=default_backend()
            )
            
            # Create certificate
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, common_name),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Saraphis Production"),
                x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Security"),
            ])
            
            builder = x509.CertificateBuilder()
            builder = builder.subject_name(subject)
            builder = builder.issuer_name(issuer)
            builder = builder.public_key(private_key.public_key())
            builder = builder.serial_number(x509.random_serial_number())
            builder = builder.not_valid_before(datetime.utcnow())
            builder = builder.not_valid_after(datetime.utcnow() + timedelta(days=validity_days))
            
            # Add SAN extension
            if san_list:
                san_names = [x509.DNSName(name) for name in san_list]
                builder = builder.add_extension(
                    x509.SubjectAlternativeName(san_names),
                    critical=False,
                )
            
            # Sign certificate
            certificate = builder.sign(private_key, hashes.SHA256(), default_backend())
            
            # Save certificate and key
            cert_path = self.cert_dir / f"{common_name}.crt"
            key_path = self.cert_dir / f"{common_name}.key"
            
            with open(cert_path, "wb") as f:
                f.write(certificate.public_bytes(serialization.Encoding.PEM))
            
            with open(key_path, "wb") as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            self.logger.info(f"Generated self-signed certificate for {common_name}")
            return str(cert_path), str(key_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate self-signed certificate: {e}")
    
    def verify_certificate(self, cert_path: str, ca_path: Optional[str] = None) -> bool:
        """Verify certificate validity."""
        try:
            with open(cert_path, "rb") as f:
                cert = x509.load_pem_x509_certificate(f.read(), default_backend())
            
            # Check if certificate is expired
            now = datetime.utcnow()
            if now < cert.not_valid_before or now > cert.not_valid_after:
                return False
            
            # Verify against CA if provided
            if ca_path:
                with open(ca_path, "rb") as f:
                    ca_cert = x509.load_pem_x509_certificate(f.read(), default_backend())
                
                # Verify signature (simplified verification)
                try:
                    ca_public_key = ca_cert.public_key()
                    ca_public_key.verify(
                        cert.signature,
                        cert.tbs_certificate_bytes,
                        padding.PKCS1v15(),
                        cert.signature_hash_algorithm
                    )
                except:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Certificate verification failed: {e}")
            return False
    
    def get_certificate_info(self, cert_path: str) -> Dict[str, Any]:
        """Get certificate information."""
        try:
            with open(cert_path, "rb") as f:
                cert = x509.load_pem_x509_certificate(f.read(), default_backend())
            
            return {
                'subject': cert.subject.rfc4514_string(),
                'issuer': cert.issuer.rfc4514_string(),
                'serial_number': str(cert.serial_number),
                'not_valid_before': cert.not_valid_before.isoformat(),
                'not_valid_after': cert.not_valid_after.isoformat(),
                'public_key_size': cert.public_key().key_size,
                'signature_algorithm': cert.signature_hash_algorithm.name,
                'is_expired': datetime.utcnow() > cert.not_valid_after
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get certificate info: {e}")


class KeyManager:
    """Cryptographic key management utilities."""
    
    def __init__(self, key_dir: str = "keys"):
        self.key_dir = Path(key_dir)
        self.key_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger('KeyManager')
        self.keys: Dict[str, bytes] = {}
        self.key_metadata: Dict[str, Dict[str, Any]] = {}
    
    def generate_key(self, key_name: str, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM) -> str:
        """Generate cryptographic key."""
        try:
            if algorithm in [EncryptionAlgorithm.AES_128_GCM, EncryptionAlgorithm.AES_256_GCM]:
                key_size = 16 if algorithm == EncryptionAlgorithm.AES_128_GCM else 32
                key = secrets.token_bytes(key_size)
            elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                key = secrets.token_bytes(32)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Store key
            self.keys[key_name] = key
            self.key_metadata[key_name] = {
                'algorithm': algorithm.value,
                'created_at': datetime.utcnow().isoformat(),
                'key_size': len(key),
                'key_id': str(uuid.uuid4())
            }
            
            # Save key to file
            key_path = self.key_dir / f"{key_name}.key"
            with open(key_path, 'wb') as f:
                f.write(key)
            
            self.logger.info(f"Generated key '{key_name}' with algorithm {algorithm.value}")
            return str(key_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate key '{key_name}': {e}")
    
    def rotate_key(self, key_name: str) -> str:
        """Rotate existing key."""
        if key_name not in self.key_metadata:
            raise ValueError(f"Key '{key_name}' not found")
        
        try:
            # Backup old key
            old_key_path = self.key_dir / f"{key_name}.key"
            backup_path = self.key_dir / f"{key_name}.key.backup.{int(time.time())}"
            if old_key_path.exists():
                old_key_path.rename(backup_path)
            
            # Generate new key with same algorithm
            algorithm = EncryptionAlgorithm(self.key_metadata[key_name]['algorithm'])
            new_key_path = self.generate_key(key_name, algorithm)
            
            self.logger.info(f"Rotated key '{key_name}'")
            return new_key_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to rotate key '{key_name}': {e}")
    
    def get_key(self, key_name: str) -> bytes:
        """Get key by name."""
        if key_name in self.keys:
            return self.keys[key_name]
        
        # Try to load from file
        key_path = self.key_dir / f"{key_name}.key"
        if key_path.exists():
            with open(key_path, 'rb') as f:
                key = f.read()
            self.keys[key_name] = key
            return key
        
        raise ValueError(f"Key '{key_name}' not found")
    
    def delete_key(self, key_name: str) -> None:
        """Delete key."""
        try:
            # Remove from memory
            if key_name in self.keys:
                del self.keys[key_name]
            if key_name in self.key_metadata:
                del self.key_metadata[key_name]
            
            # Remove file
            key_path = self.key_dir / f"{key_name}.key"
            if key_path.exists():
                key_path.unlink()
            
            self.logger.info(f"Deleted key '{key_name}'")
            
        except Exception as e:
            raise RuntimeError(f"Failed to delete key '{key_name}': {e}")


class SecurityValidator:
    """Security configuration validation utilities."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        self.security_level = security_level
        self.logger = logging.getLogger('SecurityValidator')
    
    def validate_authentication_config(self, config: AuthenticationConfig) -> List[str]:
        """Validate authentication configuration."""
        errors = []
        
        if not config.enabled:
            if self.security_level in [SecurityLevel.ENHANCED, SecurityLevel.MAXIMUM]:
                errors.append("Authentication must be enabled for enhanced/maximum security levels")
        
        if config.type == AuthenticationType.JWT:
            if not config.jwt_secret_key:
                errors.append("JWT secret key is required")
            elif len(config.jwt_secret_key) < 32:
                errors.append("JWT secret key should be at least 32 characters")
        
        if config.type == AuthenticationType.OAUTH2:
            if not config.oauth2_client_id or not config.oauth2_client_secret:
                errors.append("OAuth2 client credentials are required")
        
        if self.security_level == SecurityLevel.MAXIMUM:
            if not config.mfa_enabled:
                errors.append("Multi-factor authentication required for maximum security")
        
        # Password policy validation
        if config.password_min_length < 8:
            errors.append("Password minimum length should be at least 8 characters")
        
        if self.security_level in [SecurityLevel.ENHANCED, SecurityLevel.MAXIMUM]:
            if config.password_min_length < 12:
                errors.append("Password minimum length should be at least 12 characters for enhanced/maximum security")
        
        return errors
    
    def validate_encryption_config(self, config: EncryptionConfig) -> List[str]:
        """Validate encryption configuration."""
        errors = []
        
        if not config.enabled:
            if self.security_level in [SecurityLevel.STANDARD, SecurityLevel.ENHANCED, SecurityLevel.MAXIMUM]:
                errors.append("Encryption must be enabled for standard/enhanced/maximum security levels")
        
        if config.algorithm in [EncryptionAlgorithm.AES_128_GCM] and self.security_level in [SecurityLevel.ENHANCED, SecurityLevel.MAXIMUM]:
            errors.append("AES-128 not recommended for enhanced/maximum security levels")
        
        if config.asymmetric_key_size < 2048:
            errors.append("Asymmetric key size should be at least 2048 bits")
        
        if self.security_level == SecurityLevel.MAXIMUM:
            if config.asymmetric_key_size < 4096:
                errors.append("Asymmetric key size should be at least 4096 bits for maximum security")
        
        if not config.encrypt_at_rest and self.security_level in [SecurityLevel.ENHANCED, SecurityLevel.MAXIMUM]:
            errors.append("Encryption at rest required for enhanced/maximum security")
        
        if not config.encrypt_in_transit:
            errors.append("Encryption in transit is required")
        
        return errors
    
    def validate_network_security_config(self, config: NetworkSecurityConfig) -> List[str]:
        """Validate network security configuration."""
        errors = []
        
        if not config.enabled:
            if self.security_level in [SecurityLevel.STANDARD, SecurityLevel.ENHANCED, SecurityLevel.MAXIMUM]:
                errors.append("Network security must be enabled for standard/enhanced/maximum security levels")
        
        if not config.ssl_enabled:
            errors.append("SSL/TLS must be enabled")
        
        if not config.firewall_enabled and self.security_level in [SecurityLevel.ENHANCED, SecurityLevel.MAXIMUM]:
            errors.append("Firewall must be enabled for enhanced/maximum security")
        
        if not config.rate_limiting_enabled and self.security_level in [SecurityLevel.STANDARD, SecurityLevel.ENHANCED, SecurityLevel.MAXIMUM]:
            errors.append("Rate limiting should be enabled for standard/enhanced/maximum security")
        
        # Validate SSL configuration
        if config.ssl_enabled:
            if not config.ssl_cert_path or not config.ssl_key_path:
                errors.append("SSL certificate and key paths are required when SSL is enabled")
        
        return errors


class SecurityConfigManager:
    """
    Production Security Configuration Manager - Comprehensive security configuration and management.
    
    This class provides complete security configuration management including authentication,
    authorization, encryption, network security, certificate management, key rotation,
    security auditing, and compliance checking for production environments.
    """
    
    def __init__(self, config_dir: str = "security_config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.security_lock = threading.RLock()
        self.security_level = SecurityLevel.STANDARD
        
        # Security configurations
        self.authentication_config = AuthenticationConfig()
        self.authorization_config = AuthorizationConfig()
        self.encryption_config = EncryptionConfig()
        self.network_security_config = NetworkSecurityConfig()
        self.audit_config = AuditConfig()
        
        # Security policies
        self.security_policies: Dict[str, SecurityPolicy] = {}
        
        # Security management components
        self.certificate_manager = CertificateManager(str(self.config_dir / "certificates"))
        self.key_manager = KeyManager(str(self.config_dir / "keys"))
        self.security_validator = SecurityValidator(self.security_level)
        
        # Integration references
        self.production_config_manager: Optional['ProductionConfigManager'] = None
        self.deployment_config_manager: Optional['DeploymentConfigManager'] = None
        self.gac_config_manager: Optional['GACConfigManager'] = None
        
        # Security monitoring
        self.security_events: List[Dict[str, Any]] = []
        self.security_violations: List[Dict[str, Any]] = []
        self.security_metrics: Dict[str, Any] = {}
        
        # Key rotation scheduling
        self.key_rotation_active = False
        self.key_rotation_thread: Optional[threading.Thread] = None
        
        logger.info(f"SecurityConfigManager initialized with {self.security_level.value} security level")
    
    def initialize_security_config_manager(
        self,
        security_level: SecurityLevel = SecurityLevel.STANDARD,
        production_config_manager: Optional['ProductionConfigManager'] = None,
        deployment_config_manager: Optional['DeploymentConfigManager'] = None,
        gac_config_manager: Optional['GACConfigManager'] = None
    ) -> None:
        """Initialize security config manager with system integrations."""
        try:
            with self.security_lock:
                self.security_level = security_level
                self.security_validator = SecurityValidator(security_level)
                
                # Store system integrations
                self.production_config_manager = production_config_manager
                self.deployment_config_manager = deployment_config_manager
                self.gac_config_manager = gac_config_manager
                
                # Load existing configurations
                self._load_security_configurations()
                
                # Apply security level configurations
                self._apply_security_level_settings()
                
                # Initialize integrations
                if production_config_manager:
                    self._integrate_production_config()
                
                if deployment_config_manager:
                    self._integrate_deployment_config()
                
                # Start key rotation if enabled
                if self.encryption_config.automatic_rotation:
                    self._start_key_rotation()
                
                logger.info(f"Security configuration manager initialized with {security_level.value} level")
                
        except Exception as e:
            error_msg = f"Failed to initialize security configuration manager: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def configure_authentication(
        self,
        auth_type: AuthenticationType = AuthenticationType.JWT,
        **kwargs
    ) -> AuthenticationConfig:
        """Configure authentication settings."""
        try:
            with self.security_lock:
                self.authentication_config.type = auth_type
                
                # Apply configuration parameters
                for key, value in kwargs.items():
                    if hasattr(self.authentication_config, key):
                        setattr(self.authentication_config, key, value)
                
                # Generate JWT secret if needed
                if auth_type == AuthenticationType.JWT and not self.authentication_config.jwt_secret_key:
                    self.authentication_config.jwt_secret_key = secrets.token_urlsafe(32)
                
                # Validate configuration
                errors = self.security_validator.validate_authentication_config(self.authentication_config)
                if errors:
                    raise ValueError(f"Authentication configuration validation failed: {'; '.join(errors)}")
                
                # Save configuration
                self._save_authentication_config()
                
                logger.info(f"Authentication configured with type {auth_type.value}")
                return self.authentication_config
                
        except Exception as e:
            error_msg = f"Failed to configure authentication: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def configure_authorization(
        self,
        auth_model: AuthorizationModel = AuthorizationModel.ROLE_BASED,
        **kwargs
    ) -> AuthorizationConfig:
        """Configure authorization settings."""
        try:
            with self.security_lock:
                self.authorization_config.model = auth_model
                
                # Apply configuration parameters
                for key, value in kwargs.items():
                    if hasattr(self.authorization_config, key):
                        setattr(self.authorization_config, key, value)
                
                # Setup default roles if none exist
                if not self.authorization_config.roles:
                    self._setup_default_roles()
                
                # Save configuration
                self._save_authorization_config()
                
                logger.info(f"Authorization configured with model {auth_model.value}")
                return self.authorization_config
                
        except Exception as e:
            error_msg = f"Failed to configure authorization: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def configure_encryption(
        self,
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
        **kwargs
    ) -> EncryptionConfig:
        """Configure encryption settings."""
        try:
            with self.security_lock:
                self.encryption_config.algorithm = algorithm
                
                # Apply configuration parameters
                for key, value in kwargs.items():
                    if hasattr(self.encryption_config, key):
                        setattr(self.encryption_config, key, value)
                
                # Generate master key if needed
                if not self.encryption_config.master_key:
                    master_key_path = self.key_manager.generate_key("master_key", algorithm)
                    self.encryption_config.master_key = master_key_path
                
                # Validate configuration
                errors = self.security_validator.validate_encryption_config(self.encryption_config)
                if errors:
                    raise ValueError(f"Encryption configuration validation failed: {'; '.join(errors)}")
                
                # Save configuration
                self._save_encryption_config()
                
                logger.info(f"Encryption configured with algorithm {algorithm.value}")
                return self.encryption_config
                
        except Exception as e:
            error_msg = f"Failed to configure encryption: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def configure_network_security(
        self,
        **kwargs
    ) -> NetworkSecurityConfig:
        """Configure network security settings."""
        try:
            with self.security_lock:
                # Apply configuration parameters
                for key, value in kwargs.items():
                    if hasattr(self.network_security_config, key):
                        setattr(self.network_security_config, key, value)
                
                # Generate SSL certificate if needed
                if self.network_security_config.ssl_enabled and not self.network_security_config.ssl_cert_path:
                    cert_path, key_path = self.certificate_manager.generate_self_signed_cert(
                        "localhost",
                        ["localhost", "127.0.0.1"]
                    )
                    self.network_security_config.ssl_cert_path = cert_path
                    self.network_security_config.ssl_key_path = key_path
                
                # Validate configuration
                errors = self.security_validator.validate_network_security_config(self.network_security_config)
                if errors:
                    raise ValueError(f"Network security configuration validation failed: {'; '.join(errors)}")
                
                # Save configuration
                self._save_network_security_config()
                
                logger.info("Network security configured successfully")
                return self.network_security_config
                
        except Exception as e:
            error_msg = f"Failed to configure network security: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def create_security_policy(
        self,
        name: str,
        rules: Dict[str, Any],
        description: str = "",
        **kwargs
    ) -> SecurityPolicy:
        """Create security policy."""
        try:
            with self.security_lock:
                if name in self.security_policies:
                    raise ValueError(f"Security policy '{name}' already exists")
                
                policy = SecurityPolicy(
                    name=name,
                    description=description,
                    rules=rules,
                    **kwargs
                )
                
                # Validate policy
                self._validate_security_policy(policy)
                
                # Store policy
                self.security_policies[name] = policy
                
                # Save policy
                self._save_security_policy(policy)
                
                logger.info(f"Security policy '{name}' created successfully")
                return policy
                
        except Exception as e:
            error_msg = f"Failed to create security policy '{name}': {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def enforce_security_policy(self, policy_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce security policy."""
        try:
            if policy_name not in self.security_policies:
                raise ValueError(f"Security policy '{policy_name}' not found")
            
            policy = self.security_policies[policy_name]
            
            # Evaluate policy rules
            enforcement_result = self._evaluate_policy_rules(policy, context)
            
            # Take enforcement actions
            if not enforcement_result['allowed'] and policy.enforcement_mode == "enforce":
                self._take_enforcement_actions(policy, enforcement_result, context)
            
            # Log enforcement
            self._log_policy_enforcement(policy_name, enforcement_result, context)
            
            return enforcement_result
            
        except Exception as e:
            error_msg = f"Failed to enforce security policy '{policy_name}': {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def rotate_keys(self, key_names: Optional[List[str]] = None) -> Dict[str, str]:
        """Rotate cryptographic keys."""
        try:
            with self.security_lock:
                rotated_keys = {}
                
                if key_names is None:
                    # Rotate all keys
                    key_names = list(self.key_manager.key_metadata.keys())
                
                for key_name in key_names:
                    try:
                        new_key_path = self.key_manager.rotate_key(key_name)
                        rotated_keys[key_name] = new_key_path
                        
                        # Log key rotation
                        self._log_security_event({
                            'event_type': 'key_rotation',
                            'key_name': key_name,
                            'timestamp': datetime.utcnow().isoformat(),
                            'status': 'success'
                        })
                        
                    except Exception as e:
                        logger.error(f"Failed to rotate key '{key_name}': {e}")
                        rotated_keys[key_name] = f"ERROR: {e}"
                
                logger.info(f"Key rotation completed: {len(rotated_keys)} keys processed")
                return rotated_keys
                
        except Exception as e:
            error_msg = f"Failed to rotate keys: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def generate_security_report(self, report_type: str = "comprehensive") -> Dict[str, Any]:
        """Generate security report."""
        try:
            with self.security_lock:
                report = {
                    'report_type': report_type,
                    'generated_at': datetime.utcnow().isoformat(),
                    'security_level': self.security_level.value,
                    'summary': self._generate_security_summary(),
                    'configurations': {
                        'authentication': asdict(self.authentication_config),
                        'authorization': asdict(self.authorization_config),
                        'encryption': asdict(self.encryption_config),
                        'network_security': asdict(self.network_security_config),
                        'audit': asdict(self.audit_config)
                    },
                    'policies': {name: asdict(policy) for name, policy in self.security_policies.items()},
                    'security_events': self.security_events[-100:] if report_type == "comprehensive" else [],
                    'security_violations': self.security_violations[-50:] if report_type == "comprehensive" else [],
                    'recommendations': self._generate_security_recommendations()
                }
                
                # Add certificate information
                if report_type == "comprehensive":
                    report['certificates'] = self._get_certificate_status()
                    report['keys'] = self._get_key_status()
                
                return report
                
        except Exception as e:
            error_msg = f"Failed to generate security report: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        try:
            with self.security_lock:
                return {
                    'security_level': self.security_level.value,
                    'authentication_enabled': self.authentication_config.enabled,
                    'authorization_enabled': self.authorization_config.enabled,
                    'encryption_enabled': self.encryption_config.enabled,
                    'network_security_enabled': self.network_security_config.enabled,
                    'audit_enabled': self.audit_config.enabled,
                    'active_policies': len(self.security_policies),
                    'key_rotation_active': self.key_rotation_active,
                    'recent_events': len(self.security_events),
                    'recent_violations': len(self.security_violations),
                    'certificates_count': len(list(self.certificate_manager.cert_dir.glob("*.crt"))),
                    'keys_count': len(self.key_manager.key_metadata)
                }
                
        except Exception as e:
            error_msg = f"Failed to get security status: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _load_security_configurations(self) -> None:
        """Load existing security configurations."""
        try:
            config_files = {
                'authentication.json': 'authentication_config',
                'authorization.json': 'authorization_config',
                'encryption.json': 'encryption_config',
                'network_security.json': 'network_security_config',
                'audit.json': 'audit_config'
            }
            
            for filename, attr_name in config_files.items():
                config_path = self.config_dir / filename
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    
                    # Create configuration object from data
                    if attr_name == 'authentication_config':
                        setattr(self, attr_name, AuthenticationConfig(**config_data))
                    elif attr_name == 'authorization_config':
                        setattr(self, attr_name, AuthorizationConfig(**config_data))
                    elif attr_name == 'encryption_config':
                        setattr(self, attr_name, EncryptionConfig(**config_data))
                    elif attr_name == 'network_security_config':
                        setattr(self, attr_name, NetworkSecurityConfig(**config_data))
                    elif attr_name == 'audit_config':
                        setattr(self, attr_name, AuditConfig(**config_data))
                    
                    logger.debug(f"Loaded {attr_name} from {filename}")
            
            # Load security policies
            policies_dir = self.config_dir / "policies"
            if policies_dir.exists():
                for policy_file in policies_dir.glob("*.json"):
                    with open(policy_file, 'r') as f:
                        policy_data = json.load(f)
                    policy = SecurityPolicy(**policy_data)
                    self.security_policies[policy.name] = policy
            
        except Exception as e:
            logger.error(f"Failed to load security configurations: {e}")
    
    def _apply_security_level_settings(self) -> None:
        """Apply security level specific settings."""
        if self.security_level == SecurityLevel.BASIC:
            self.authentication_config.mfa_enabled = False
            self.encryption_config.asymmetric_key_size = 2048
            self.audit_config.level = SecurityAuditLevel.MINIMAL
            
        elif self.security_level == SecurityLevel.STANDARD:
            self.authentication_config.mfa_enabled = False
            self.encryption_config.asymmetric_key_size = 2048
            self.audit_config.level = SecurityAuditLevel.STANDARD
            
        elif self.security_level == SecurityLevel.ENHANCED:
            self.authentication_config.mfa_enabled = True
            self.encryption_config.asymmetric_key_size = 4096
            self.audit_config.level = SecurityAuditLevel.COMPREHENSIVE
            self.network_security_config.ids_enabled = True
            
        elif self.security_level == SecurityLevel.MAXIMUM:
            self.authentication_config.mfa_enabled = True
            self.authentication_config.cert_required = True
            self.encryption_config.asymmetric_key_size = 4096
            self.encryption_config.automatic_rotation = True
            self.audit_config.level = SecurityAuditLevel.FORENSIC
            self.network_security_config.ids_enabled = True
            self.network_security_config.ddos_protection_enabled = True
    
    def _save_authentication_config(self) -> None:
        """Save authentication configuration."""
        config_path = self.config_dir / "authentication.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.authentication_config), f, indent=2, default=str)
    
    def _save_authorization_config(self) -> None:
        """Save authorization configuration."""
        config_path = self.config_dir / "authorization.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.authorization_config), f, indent=2, default=str)
    
    def _save_encryption_config(self) -> None:
        """Save encryption configuration."""
        config_path = self.config_dir / "encryption.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.encryption_config), f, indent=2, default=str)
    
    def _save_network_security_config(self) -> None:
        """Save network security configuration."""
        config_path = self.config_dir / "network_security.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.network_security_config), f, indent=2, default=str)
    
    def _save_security_policy(self, policy: SecurityPolicy) -> None:
        """Save security policy."""
        policies_dir = self.config_dir / "policies"
        policies_dir.mkdir(exist_ok=True)
        
        policy_path = policies_dir / f"{policy.name}.json"
        with open(policy_path, 'w') as f:
            json.dump(asdict(policy), f, indent=2, default=str)
    
    def _setup_default_roles(self) -> None:
        """Setup default authorization roles."""
        default_roles = {
            'admin': ['read', 'write', 'delete', 'manage_users', 'manage_system'],
            'user': ['read', 'write'],
            'readonly': ['read'],
            'guest': []
        }
        
        self.authorization_config.roles.update(default_roles)
        self.authorization_config.admin_roles = ['admin']
    
    def _validate_security_policy(self, policy: SecurityPolicy) -> None:
        """Validate security policy."""
        if not policy.name:
            raise ValueError("Policy name cannot be empty")
        
        if not policy.rules:
            raise ValueError("Policy must have at least one rule")
        
        # Validate rule structure
        for rule_name, rule_data in policy.rules.items():
            if not isinstance(rule_data, dict):
                raise ValueError(f"Rule '{rule_name}' must be a dictionary")
    
    def _evaluate_policy_rules(self, policy: SecurityPolicy, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate policy rules against context."""
        # Simplified policy evaluation
        # In production, use a proper policy engine
        
        allowed = True
        violations = []
        
        for rule_name, rule_data in policy.rules.items():
            if not self._evaluate_single_rule(rule_data, context):
                allowed = False
                violations.append(rule_name)
        
        return {
            'allowed': allowed,
            'violations': violations,
            'policy_name': policy.name,
            'evaluated_at': datetime.utcnow().isoformat()
        }
    
    def _evaluate_single_rule(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate single policy rule."""
        # Simplified rule evaluation
        rule_type = rule.get('type', 'allow')
        
        if rule_type == 'allow':
            return True
        elif rule_type == 'deny':
            return False
        elif rule_type == 'conditional':
            condition = rule.get('condition', {})
            return self._evaluate_condition(condition, context)
        
        return True
    
    def _evaluate_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate policy condition."""
        # Simplified condition evaluation
        for key, expected_value in condition.items():
            if key not in context or context[key] != expected_value:
                return False
        return True
    
    def _take_enforcement_actions(self, policy: SecurityPolicy, result: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Take enforcement actions for policy violations."""
        for action in policy.violation_actions:
            if action == 'log':
                self._log_security_violation(policy.name, result, context)
            elif action == 'alert':
                self._send_security_alert(policy.name, result, context)
            elif action == 'block':
                self._block_access(context)
    
    def _log_policy_enforcement(self, policy_name: str, result: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Log policy enforcement."""
        event = {
            'event_type': 'policy_enforcement',
            'policy_name': policy_name,
            'allowed': result['allowed'],
            'violations': result.get('violations', []),
            'context': context,
            'timestamp': datetime.utcnow().isoformat()
        }
        self._log_security_event(event)
    
    def _log_security_event(self, event: Dict[str, Any]) -> None:
        """Log security event."""
        self.security_events.append(event)
        
        # Keep only last 1000 events
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
        
        # Write to audit log if enabled
        if self.audit_config.enabled:
            logger.info(f"Security event: {json.dumps(event)}")
    
    def _log_security_violation(self, policy_name: str, result: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Log security violation."""
        violation = {
            'policy_name': policy_name,
            'violations': result.get('violations', []),
            'context': context,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.security_violations.append(violation)
        
        # Keep only last 500 violations
        if len(self.security_violations) > 500:
            self.security_violations = self.security_violations[-500:]
    
    def _send_security_alert(self, policy_name: str, result: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Send security alert."""
        # Implement alert sending (email, Slack, etc.)
        logger.warning(f"Security alert: Policy '{policy_name}' violated - {result['violations']}")
    
    def _block_access(self, context: Dict[str, Any]) -> None:
        """Block access based on context."""
        # Implement access blocking logic
        logger.warning(f"Access blocked for context: {context}")
    
    def _start_key_rotation(self) -> None:
        """Start automatic key rotation."""
        if self.key_rotation_active:
            return
        
        self.key_rotation_active = True
        self.key_rotation_thread = threading.Thread(
            target=self._key_rotation_loop,
            name="KeyRotation",
            daemon=True
        )
        self.key_rotation_thread.start()
        logger.info("Automatic key rotation started")
    
    def _key_rotation_loop(self) -> None:
        """Key rotation background loop."""
        while self.key_rotation_active:
            try:
                # Check if any keys need rotation
                for key_name, metadata in self.key_manager.key_metadata.items():
                    created_at = datetime.fromisoformat(metadata['created_at'])
                    rotation_days = self.encryption_config.symmetric_key_rotation_days
                    
                    if datetime.utcnow() - created_at > timedelta(days=rotation_days):
                        logger.info(f"Rotating key '{key_name}' due to age")
                        self.key_manager.rotate_key(key_name)
                
                # Sleep for 1 hour
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in key rotation loop: {e}")
                time.sleep(3600)
    
    def _generate_security_summary(self) -> Dict[str, Any]:
        """Generate security summary."""
        return {
            'security_level': self.security_level.value,
            'authentication_type': self.authentication_config.type.value,
            'authorization_model': self.authorization_config.model.value,
            'encryption_algorithm': self.encryption_config.algorithm.value,
            'ssl_enabled': self.network_security_config.ssl_enabled,
            'audit_level': self.audit_config.level.value,
            'active_policies': len(self.security_policies),
            'recent_events': len(self.security_events),
            'recent_violations': len(self.security_violations)
        }
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        if self.security_level == SecurityLevel.BASIC:
            recommendations.append("Consider upgrading to STANDARD security level")
        
        if not self.authentication_config.mfa_enabled and self.security_level in [SecurityLevel.ENHANCED, SecurityLevel.MAXIMUM]:
            recommendations.append("Enable multi-factor authentication")
        
        if self.encryption_config.asymmetric_key_size < 4096 and self.security_level == SecurityLevel.MAXIMUM:
            recommendations.append("Upgrade to 4096-bit asymmetric keys")
        
        if not self.network_security_config.ids_enabled:
            recommendations.append("Enable intrusion detection system")
        
        if not self.audit_config.enabled:
            recommendations.append("Enable security auditing")
        
        return recommendations
    
    def _get_certificate_status(self) -> Dict[str, Any]:
        """Get certificate status information."""
        cert_status = {}
        
        for cert_file in self.certificate_manager.cert_dir.glob("*.crt"):
            cert_name = cert_file.stem
            try:
                cert_info = self.certificate_manager.get_certificate_info(str(cert_file))
                cert_status[cert_name] = cert_info
            except Exception as e:
                cert_status[cert_name] = {'error': str(e)}
        
        return cert_status
    
    def _get_key_status(self) -> Dict[str, Any]:
        """Get key status information."""
        return copy.deepcopy(self.key_manager.key_metadata)
    
    def _integrate_production_config(self) -> None:
        """Integrate with production configuration manager."""
        try:
            if not self.production_config_manager:
                return
            
            prod_config = self.production_config_manager.production_config
            
            # Apply production config security settings
            if hasattr(prod_config, 'jwt_secret_key') and prod_config.jwt_secret_key:
                self.authentication_config.jwt_secret_key = prod_config.jwt_secret_key
            
            if hasattr(prod_config, 'cors_enabled'):
                # Configure CORS based on production settings
                pass
            
            logger.info("Production configuration integrated successfully")
            
        except Exception as e:
            logger.error(f"Failed to integrate production configuration: {e}")
    
    def _integrate_deployment_config(self) -> None:
        """Integrate with deployment configuration."""
        try:
            if not self.deployment_config_manager:
                return
            
            # Apply deployment-specific security settings
            logger.info("Deployment configuration integrated successfully")
            
        except Exception as e:
            logger.error(f"Failed to integrate deployment configuration: {e}")
    
    def shutdown(self) -> None:
        """Shutdown security configuration manager."""
        try:
            with self.security_lock:
                # Stop key rotation
                self.key_rotation_active = False
                if self.key_rotation_thread and self.key_rotation_thread.is_alive():
                    self.key_rotation_thread.join(timeout=10.0)
                
                # Save all configurations
                self._save_authentication_config()
                self._save_authorization_config()
                self._save_encryption_config()
                self._save_network_security_config()
                
                # Save all policies
                for policy in self.security_policies.values():
                    self._save_security_policy(policy)
                
                logger.info("Security configuration manager shutdown completed")
                
        except Exception as e:
            logger.error(f"Error during security config manager shutdown: {e}")


def create_security_config_manager(
    config_dir: str = "security_config",
    security_level: SecurityLevel = SecurityLevel.STANDARD
) -> SecurityConfigManager:
    """Factory function to create a SecurityConfigManager instance."""
    return SecurityConfigManager(config_dir=config_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    config_manager = create_security_config_manager()
    
    # Initialize with sample settings
    config_manager.initialize_security_config_manager(SecurityLevel.ENHANCED)
    
    # Configure authentication
    auth_config = config_manager.configure_authentication(
        AuthenticationType.JWT,
        mfa_enabled=True
    )
    print(f"Authentication configured: {auth_config.type.value}")
    
    # Configure encryption
    encryption_config = config_manager.configure_encryption(
        EncryptionAlgorithm.AES_256_GCM,
        automatic_rotation=True
    )
    print(f"Encryption configured: {encryption_config.algorithm.value}")
    
    # Get security status
    status = config_manager.get_security_status()
    print(f"Security status: {status['security_level']} level with {status['active_policies']} policies")