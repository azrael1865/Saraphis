"""
Enhanced Fraud Detection Core - Chunk 5: Security and Audit System
Comprehensive security framework and audit logging for fraud detection system
"""

import logging
import time
import threading
import hashlib
import secrets
import json
import sqlite3
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import hmac
import base64
import pickle
import gzip
from pathlib import Path
import uuid
from collections import defaultdict, deque
import ipaddress
import re
import traceback
from contextlib import contextmanager
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt

# Import core exceptions and enums
from enhanced_fraud_core_exceptions import (
    EnhancedFraudException, SecurityError, ValidationError, AuditError,
    SecurityLevel, AlertSeverity, AuditLevel, ErrorContext, create_error_context
)

# Configure logging
logger = logging.getLogger(__name__)

# ======================== SECURITY CONFIGURATION ========================

@dataclass
class SecurityConfig:
    """Configuration for security system"""
    # Authentication
    session_timeout: int = 3600  # 1 hour
    max_login_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes
    password_min_length: int = 12
    require_mfa: bool = True
    
    # Authorization
    default_permissions: List[str] = field(default_factory=lambda: ['read'])
    admin_permissions: List[str] = field(default_factory=lambda: ['read', 'write', 'admin'])
    
    # Encryption
    encryption_key: Optional[str] = None
    encryption_algorithm: str = 'AES-256'
    hash_algorithm: str = 'SHA-256'
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # 1 minute
    rate_limit_burst: int = 10
    
    # Threat detection
    enable_threat_detection: bool = True
    max_failed_attempts: int = 10
    suspicious_activity_threshold: int = 5
    
    # Audit settings
    audit_level: AuditLevel = AuditLevel.DETAILED
    audit_retention_days: int = 90
    audit_encryption: bool = True
    
    # Security headers
    security_headers: Dict[str, str] = field(default_factory=lambda: {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
    })

# ======================== SECURITY MODELS ========================

@dataclass
class SecurityContext:
    """Security context for operations"""
    user_id: str
    session_id: str
    ip_address: str
    user_agent: str
    permissions: List[str]
    authentication_method: str
    timestamp: datetime
    expires_at: datetime
    additional_claims: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ThreatEvent:
    """Threat detection event"""
    event_id: str
    event_type: str
    severity: AlertSeverity
    source_ip: str
    user_id: Optional[str]
    description: str
    timestamp: datetime
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AuditEvent:
    """Audit log event"""
    event_id: str
    event_type: str
    user_id: Optional[str]
    resource: str
    action: str
    result: str
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

# ======================== AUTHENTICATION MANAGER ========================

class AuthenticationManager:
    """Handles user authentication and session management"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.active_sessions = {}
        self.failed_attempts = defaultdict(list)
        self.locked_accounts = {}
        self.lock = threading.Lock()
        
        # Initialize encryption
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        
        # JWT settings
        self.jwt_secret = secrets.token_urlsafe(32)
        self.jwt_algorithm = 'HS256'
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key"""
        if self.config.encryption_key:
            # Derive key from provided password
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'salt_',  # In production, use a proper salt
                iterations=100000,
            )
            return base64.urlsafe_b64encode(kdf.derive(self.config.encryption_key.encode()))
        else:
            return Fernet.generate_key()
    
    def authenticate(self, user_id: str, password: str, ip_address: str, 
                    user_agent: str, mfa_token: Optional[str] = None) -> Optional[SecurityContext]:
        """Authenticate user and create session"""
        
        # Check if account is locked
        if self._is_account_locked(user_id):
            raise SecurityError(
                f"Account {user_id} is locked due to too many failed attempts",
                context=create_error_context(
                    component='AuthenticationManager',
                    operation='authenticate',
                    user_id=user_id
                )
            )
        
        # Validate credentials (placeholder - would integrate with actual auth system)
        if not self._validate_credentials(user_id, password):
            self._record_failed_attempt(user_id, ip_address)
            raise SecurityError(
                "Invalid credentials",
                context=create_error_context(
                    component='AuthenticationManager',
                    operation='authenticate',
                    user_id=user_id
                )
            )
        
        # Check MFA if required
        if self.config.require_mfa and not self._validate_mfa_token(user_id, mfa_token):
            raise SecurityError(
                "MFA token required or invalid",
                context=create_error_context(
                    component='AuthenticationManager',
                    operation='authenticate',
                    user_id=user_id
                )
            )
        
        # Create session
        session_context = self._create_session(user_id, ip_address, user_agent)
        
        # Clear failed attempts
        self._clear_failed_attempts(user_id)
        
        return session_context
    
    def _validate_credentials(self, user_id: str, password: str) -> bool:
        """Validate user credentials (placeholder)"""
        # In a real implementation, this would check against a database
        # For now, we'll use a simple check
        return len(password) >= self.config.password_min_length
    
    def _validate_mfa_token(self, user_id: str, mfa_token: Optional[str]) -> bool:
        """Validate MFA token (placeholder)"""
        # In a real implementation, this would validate TOTP or other MFA
        return mfa_token is not None and len(mfa_token) == 6
    
    def _create_session(self, user_id: str, ip_address: str, user_agent: str) -> SecurityContext:
        """Create authenticated session"""
        session_id = str(uuid.uuid4())
        now = datetime.now()
        expires_at = now + timedelta(seconds=self.config.session_timeout)
        
        # Get user permissions (placeholder)
        permissions = self._get_user_permissions(user_id)
        
        context = SecurityContext(
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            permissions=permissions,
            authentication_method='password_mfa' if self.config.require_mfa else 'password',
            timestamp=now,
            expires_at=expires_at
        )
        
        # Store session
        with self.lock:
            self.active_sessions[session_id] = context
        
        logger.info(f"Session created for user {user_id}")
        return context
    
    def _get_user_permissions(self, user_id: str) -> List[str]:
        """Get user permissions (placeholder)"""
        # In a real implementation, this would query a database
        if user_id == 'admin':
            return self.config.admin_permissions
        else:
            return self.config.default_permissions
    
    def _record_failed_attempt(self, user_id: str, ip_address: str) -> None:
        """Record failed authentication attempt"""
        with self.lock:
            self.failed_attempts[user_id].append({
                'ip_address': ip_address,
                'timestamp': datetime.now()
            })
            
            # Clean old attempts
            cutoff_time = datetime.now() - timedelta(seconds=self.config.lockout_duration)
            self.failed_attempts[user_id] = [
                attempt for attempt in self.failed_attempts[user_id]
                if attempt['timestamp'] > cutoff_time
            ]
            
            # Check if should lock account
            if len(self.failed_attempts[user_id]) >= self.config.max_login_attempts:
                self.locked_accounts[user_id] = datetime.now()
                logger.warning(f"Account {user_id} locked due to too many failed attempts")
    
    def _is_account_locked(self, user_id: str) -> bool:
        """Check if account is locked"""
        with self.lock:
            if user_id not in self.locked_accounts:
                return False
            
            lock_time = self.locked_accounts[user_id]
            if datetime.now() - lock_time > timedelta(seconds=self.config.lockout_duration):
                del self.locked_accounts[user_id]
                return False
            
            return True
    
    def _clear_failed_attempts(self, user_id: str) -> None:
        """Clear failed attempts for user"""
        with self.lock:
            self.failed_attempts[user_id] = []
            self.locked_accounts.pop(user_id, None)
    
    def validate_session(self, session_id: str) -> Optional[SecurityContext]:
        """Validate session and return context"""
        with self.lock:
            if session_id not in self.active_sessions:
                return None
            
            context = self.active_sessions[session_id]
            
            # Check if expired
            if datetime.now() > context.expires_at:
                del self.active_sessions[session_id]
                return None
            
            return context
    
    def logout(self, session_id: str) -> None:
        """Logout user and invalidate session"""
        with self.lock:
            if session_id in self.active_sessions:
                user_id = self.active_sessions[session_id].user_id
                del self.active_sessions[session_id]
                logger.info(f"User {user_id} logged out")
    
    def create_jwt_token(self, context: SecurityContext) -> str:
        """Create JWT token from security context"""
        payload = {
            'user_id': context.user_id,
            'session_id': context.session_id,
            'permissions': context.permissions,
            'exp': int(context.expires_at.timestamp()),
            'iat': int(context.timestamp.timestamp())
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def validate_jwt_token(self, token: str) -> Optional[SecurityContext]:
        """Validate JWT token and return context"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Check if session still exists
            context = self.validate_session(payload['session_id'])
            return context
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None

# ======================== AUTHORIZATION MANAGER ========================

class AuthorizationManager:
    """Handles authorization and access control"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.role_permissions = {
            'admin': ['read', 'write', 'delete', 'admin'],
            'analyst': ['read', 'write'],
            'viewer': ['read']
        }
        
        # Resource-based permissions
        self.resource_permissions = {
            'transactions': ['read', 'write'],
            'reports': ['read', 'write'],
            'settings': ['admin'],
            'users': ['admin']
        }
    
    def check_permission(self, context: SecurityContext, resource: str, action: str) -> bool:
        """Check if user has permission for resource action"""
        
        # Check basic permission
        if action not in context.permissions:
            return False
        
        # Check resource-specific permissions
        if resource in self.resource_permissions:
            required_permissions = self.resource_permissions[resource]
            if not any(perm in context.permissions for perm in required_permissions):
                return False
        
        return True
    
    def require_permission(self, resource: str, action: str):
        """Decorator to require permission for function"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Get security context from arguments or thread-local storage
                context = self._get_security_context_from_args(args, kwargs)
                
                if not context:
                    raise SecurityError(
                        "No security context available",
                        context=create_error_context(
                            component='AuthorizationManager',
                            operation='require_permission'
                        )
                    )
                
                if not self.check_permission(context, resource, action):
                    raise SecurityError(
                        f"Insufficient permissions for {action} on {resource}",
                        context=create_error_context(
                            component='AuthorizationManager',
                            operation='require_permission',
                            user_id=context.user_id,
                            additional_data={
                                'resource': resource,
                                'action': action,
                                'user_permissions': context.permissions
                            }
                        )
                    )
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def _get_security_context_from_args(self, args: tuple, kwargs: dict) -> Optional[SecurityContext]:
        """Extract security context from function arguments"""
        # Check for context in kwargs
        if 'security_context' in kwargs:
            return kwargs['security_context']
        
        # Check for context in args
        for arg in args:
            if isinstance(arg, SecurityContext):
                return arg
        
        return None

# ======================== THREAT DETECTION ========================

class ThreatDetector:
    """Detects and responds to security threats"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.threat_events = deque(maxlen=10000)
        self.ip_activity = defaultdict(list)
        self.user_activity = defaultdict(list)
        self.blocked_ips = set()
        self.lock = threading.Lock()
        
        # Threat patterns
        self.threat_patterns = {
            'sql_injection': re.compile(r'(union|select|drop|delete|insert|update|exec|script)', re.IGNORECASE),
            'xss': re.compile(r'(<script|javascript:|vbscript:|onload|onerror)', re.IGNORECASE),
            'path_traversal': re.compile(r'(\.\./|\.\.\\|/etc/|/var/|/proc/)', re.IGNORECASE)
        }
    
    def analyze_request(self, ip_address: str, user_id: Optional[str], 
                       request_data: Dict[str, Any]) -> List[ThreatEvent]:
        """Analyze request for threats"""
        threats = []
        
        # Check for blocked IPs
        if ip_address in self.blocked_ips:
            threats.append(self._create_threat_event(
                'blocked_ip',
                AlertSeverity.ERROR,
                ip_address,
                user_id,
                f"Request from blocked IP: {ip_address}"
            ))
        
        # Check for suspicious patterns
        threats.extend(self._check_injection_patterns(ip_address, user_id, request_data))
        
        # Check rate limiting
        if self._is_rate_limited(ip_address):
            threats.append(self._create_threat_event(
                'rate_limit_exceeded',
                AlertSeverity.WARNING,
                ip_address,
                user_id,
                f"Rate limit exceeded for IP: {ip_address}"
            ))
        
        # Check for anomalous behavior
        threats.extend(self._check_behavioral_anomalies(ip_address, user_id))
        
        # Record threats
        for threat in threats:
            self._record_threat_event(threat)
        
        return threats
    
    def _create_threat_event(self, event_type: str, severity: AlertSeverity,
                           ip_address: str, user_id: Optional[str], description: str) -> ThreatEvent:
        """Create threat event"""
        return ThreatEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            severity=severity,
            source_ip=ip_address,
            user_id=user_id,
            description=description,
            timestamp=datetime.now()
        )
    
    def _check_injection_patterns(self, ip_address: str, user_id: Optional[str],
                                 request_data: Dict[str, Any]) -> List[ThreatEvent]:
        """Check for injection attack patterns"""
        threats = []
        
        # Check all string values in request
        for key, value in request_data.items():
            if isinstance(value, str):
                for pattern_name, pattern in self.threat_patterns.items():
                    if pattern.search(value):
                        threats.append(self._create_threat_event(
                            pattern_name,
                            AlertSeverity.ERROR,
                            ip_address,
                            user_id,
                            f"{pattern_name} pattern detected in field {key}: {value[:100]}"
                        ))
        
        return threats
    
    def _is_rate_limited(self, ip_address: str) -> bool:
        """Check if IP is rate limited"""
        with self.lock:
            now = datetime.now()
            
            # Clean old activity
            cutoff_time = now - timedelta(seconds=self.config.rate_limit_window)
            self.ip_activity[ip_address] = [
                timestamp for timestamp in self.ip_activity[ip_address]
                if timestamp > cutoff_time
            ]
            
            # Check rate limit
            if len(self.ip_activity[ip_address]) >= self.config.rate_limit_requests:
                return True
            
            # Record current request
            self.ip_activity[ip_address].append(now)
            return False
    
    def _check_behavioral_anomalies(self, ip_address: str, user_id: Optional[str]) -> List[ThreatEvent]:
        """Check for behavioral anomalies"""
        threats = []
        
        # Check for multiple failed attempts
        if user_id:
            with self.lock:
                user_events = [
                    event for event in self.threat_events
                    if event.user_id == user_id and 
                    (datetime.now() - event.timestamp).total_seconds() < 300  # 5 minutes
                ]
                
                if len(user_events) >= self.config.suspicious_activity_threshold:
                    threats.append(self._create_threat_event(
                        'suspicious_activity',
                        AlertSeverity.WARNING,
                        ip_address,
                        user_id,
                        f"Suspicious activity detected for user {user_id}"
                    ))
        
        return threats
    
    def _record_threat_event(self, threat: ThreatEvent) -> None:
        """Record threat event"""
        with self.lock:
            self.threat_events.append(threat)
            
            # Auto-block IPs with high-severity threats
            if threat.severity == AlertSeverity.ERROR:
                self.blocked_ips.add(threat.source_ip)
                logger.warning(f"IP {threat.source_ip} blocked due to threat: {threat.description}")
    
    def get_threat_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get threat summary for time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_threats = [
                threat for threat in self.threat_events
                if threat.timestamp >= cutoff_time
            ]
        
        # Count by type and severity
        threat_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for threat in recent_threats:
            threat_counts[threat.event_type] += 1
            severity_counts[threat.severity.value] += 1
        
        return {
            'total_threats': len(recent_threats),
            'threat_types': dict(threat_counts),
            'severity_distribution': dict(severity_counts),
            'blocked_ips': list(self.blocked_ips),
            'time_period_hours': hours
        }

# ======================== AUDIT LOGGER ========================

class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.audit_db_path = "audit_logs.db"
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
        # Initialize database
        self._init_audit_database()
        
        # Background cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_logs)
        self.cleanup_thread.daemon = True
        self.cleanup_thread.start()
    
    def _init_audit_database(self) -> None:
        """Initialize audit database"""
        try:
            conn = sqlite3.connect(self.audit_db_path)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    event_type TEXT NOT NULL,
                    user_id TEXT,
                    resource TEXT NOT NULL,
                    action TEXT NOT NULL,
                    result TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    additional_data TEXT,
                    checksum TEXT NOT NULL
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to initialize audit database: {e}")
    
    def log_event(self, event_type: str, user_id: Optional[str], resource: str,
                 action: str, result: str, ip_address: Optional[str] = None,
                 user_agent: Optional[str] = None, additional_data: Optional[Dict[str, Any]] = None) -> None:
        """Log audit event"""
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            result=result,
            timestamp=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent,
            additional_data=additional_data or {}
        )
        
        self._store_audit_event(event)
    
    def _store_audit_event(self, event: AuditEvent) -> None:
        """Store audit event in database"""
        try:
            conn = sqlite3.connect(self.audit_db_path)
            
            # Serialize additional data
            additional_data_json = json.dumps(event.additional_data)
            
            # Encrypt if configured
            if self.config.audit_encryption:
                additional_data_json = self.cipher.encrypt(additional_data_json.encode()).decode()
            
            # Create integrity checksum
            checksum = self._create_checksum(event, additional_data_json)
            
            conn.execute('''
                INSERT INTO audit_events 
                (event_id, event_type, user_id, resource, action, result, timestamp, 
                 ip_address, user_agent, additional_data, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id,
                event.event_type,
                event.user_id,
                event.resource,
                event.action,
                event.result,
                event.timestamp.isoformat(),
                event.ip_address,
                event.user_agent,
                additional_data_json,
                checksum
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store audit event: {e}")
    
    def _create_checksum(self, event: AuditEvent, additional_data: str) -> str:
        """Create integrity checksum for audit event"""
        data = f"{event.event_id}{event.event_type}{event.user_id}{event.resource}{event.action}{event.result}{event.timestamp.isoformat()}{additional_data}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _cleanup_old_logs(self) -> None:
        """Background cleanup of old audit logs"""
        while True:
            try:
                cutoff_date = datetime.now() - timedelta(days=self.config.audit_retention_days)
                
                conn = sqlite3.connect(self.audit_db_path)
                conn.execute(
                    'DELETE FROM audit_events WHERE timestamp < ?',
                    (cutoff_date.isoformat(),)
                )
                conn.commit()
                conn.close()
                
                # Sleep for 24 hours
                time.sleep(86400)
                
            except Exception as e:
                logger.error(f"Error in audit cleanup: {e}")
                time.sleep(3600)  # Sleep for 1 hour on error
    
    def search_events(self, user_id: Optional[str] = None, event_type: Optional[str] = None,
                     resource: Optional[str] = None, start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None, limit: int = 1000) -> List[AuditEvent]:
        """Search audit events"""
        try:
            conn = sqlite3.connect(self.audit_db_path)
            
            query = 'SELECT * FROM audit_events WHERE 1=1'
            params = []
            
            if user_id:
                query += ' AND user_id = ?'
                params.append(user_id)
            
            if event_type:
                query += ' AND event_type = ?'
                params.append(event_type)
            
            if resource:
                query += ' AND resource = ?'
                params.append(resource)
            
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time.isoformat())
            
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time.isoformat())
            
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to AuditEvent objects
            events = []
            for row in rows:
                additional_data = row[9]  # additional_data column
                
                # Decrypt if necessary
                if self.config.audit_encryption and additional_data:
                    try:
                        additional_data = self.cipher.decrypt(additional_data.encode()).decode()
                    except Exception:
                        logger.warning(f"Failed to decrypt audit data for event {row[1]}")
                        additional_data = '{}'
                
                events.append(AuditEvent(
                    event_id=row[1],
                    event_type=row[2],
                    user_id=row[3],
                    resource=row[4],
                    action=row[5],
                    result=row[6],
                    timestamp=datetime.fromisoformat(row[7]),
                    ip_address=row[8],
                    user_agent=row[9],
                    additional_data=json.loads(additional_data) if additional_data else {}
                ))
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to search audit events: {e}")
            return []

# ======================== SECURITY MANAGER ========================

class SecurityManager:
    """Comprehensive security management system"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.auth_manager = AuthenticationManager(config)
        self.authz_manager = AuthorizationManager(config)
        self.threat_detector = ThreatDetector(config)
        self.audit_logger = AuditLogger(config)
        
        # Security decorators
        self.require_auth = self._create_auth_decorator()
        self.require_permission = self.authz_manager.require_permission
        self.audit_action = self._create_audit_decorator()
    
    def _create_auth_decorator(self):
        """Create authentication decorator"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract security context
                context = self.authz_manager._get_security_context_from_args(args, kwargs)
                
                if not context:
                    raise SecurityError(
                        "Authentication required",
                        context=create_error_context(
                            component='SecurityManager',
                            operation='require_auth'
                        )
                    )
                
                # Validate session
                if not self.auth_manager.validate_session(context.session_id):
                    raise SecurityError(
                        "Invalid or expired session",
                        context=create_error_context(
                            component='SecurityManager',
                            operation='require_auth',
                            user_id=context.user_id
                        )
                    )
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def _create_audit_decorator(self):
        """Create audit logging decorator"""
        def decorator(resource: str, action: str = None):
            def inner_decorator(func: Callable) -> Callable:
                @wraps(func)
                def wrapper(*args, **kwargs):
                    # Extract security context
                    context = self.authz_manager._get_security_context_from_args(args, kwargs)
                    
                    start_time = time.time()
                    result = "success"
                    
                    try:
                        return_value = func(*args, **kwargs)
                        return return_value
                    except Exception as e:
                        result = f"error: {str(e)}"
                        raise
                    finally:
                        # Log audit event
                        self.audit_logger.log_event(
                            event_type='function_call',
                            user_id=context.user_id if context else None,
                            resource=resource,
                            action=action or func.__name__,
                            result=result,
                            ip_address=context.ip_address if context else None,
                            user_agent=context.user_agent if context else None,
                            additional_data={
                                'function': func.__name__,
                                'execution_time': time.time() - start_time
                            }
                        )
                
                return wrapper
            return inner_decorator
        return decorator
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard data"""
        return {
            'active_sessions': len(self.auth_manager.active_sessions),
            'locked_accounts': len(self.auth_manager.locked_accounts),
            'blocked_ips': len(self.threat_detector.blocked_ips),
            'threat_summary': self.threat_detector.get_threat_summary(),
            'recent_events': len(self.audit_logger.search_events(limit=100)),
            'security_config': {
                'mfa_required': self.config.require_mfa,
                'session_timeout': self.config.session_timeout,
                'audit_level': self.config.audit_level.value
            }
        }

# ======================== SECURITY UTILITIES ========================

def create_default_security_manager() -> SecurityManager:
    """Create default security manager"""
    config = SecurityConfig()
    return SecurityManager(config)

def hash_password(password: str) -> str:
    """Hash password securely"""
    salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return f"{salt}:{password_hash.hex()}"

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    try:
        salt, password_hash = hashed.split(':')
        computed_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return computed_hash.hex() == password_hash
    except ValueError:
        return False

def generate_secure_token(length: int = 32) -> str:
    """Generate secure random token"""
    return secrets.token_urlsafe(length)

def validate_ip_address(ip: str) -> bool:
    """Validate IP address format"""
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False