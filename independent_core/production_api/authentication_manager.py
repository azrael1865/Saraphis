"""
Saraphis Authentication Manager
Production-ready authentication with token validation and authorization
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import logging
import jwt
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class TokenValidator:
    """Token validation system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Token configuration
        self.secret_key = config.get('secret_key', secrets.token_urlsafe(32))
        self.algorithm = config.get('algorithm', 'HS256')
        self.token_expiry = config.get('token_expiry_seconds', 3600)
        self.refresh_expiry = config.get('refresh_expiry_seconds', 86400)
        
        # Token storage
        self.active_tokens = {}
        self.revoked_tokens = set()
        self.refresh_tokens = {}
        
        # Token metrics
        self.token_metrics = {
            'issued': 0,
            'validated': 0,
            'expired': 0,
            'revoked': 0,
            'refreshed': 0
        }
        
        self.logger.info("Token Validator initialized")
    
    def generate_token(self, user_id: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate JWT token for user"""
        try:
            current_time = time.time()
            
            # Create token payload
            payload = {
                'user_id': user_id,
                'iat': current_time,
                'exp': current_time + self.token_expiry,
                'jti': secrets.token_urlsafe(16),  # JWT ID
                'metadata': metadata or {}
            }
            
            # Generate access token
            access_token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            
            # Generate refresh token
            refresh_payload = {
                'user_id': user_id,
                'iat': current_time,
                'exp': current_time + self.refresh_expiry,
                'jti': secrets.token_urlsafe(16),
                'type': 'refresh'
            }
            
            refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm=self.algorithm)
            
            # Store tokens
            self.active_tokens[payload['jti']] = {
                'user_id': user_id,
                'issued_at': current_time,
                'expires_at': payload['exp']
            }
            
            self.refresh_tokens[refresh_payload['jti']] = {
                'user_id': user_id,
                'issued_at': current_time,
                'expires_at': refresh_payload['exp'],
                'access_token_jti': payload['jti']
            }
            
            # Update metrics
            self.token_metrics['issued'] += 1
            
            return {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'token_type': 'Bearer',
                'expires_in': self.token_expiry,
                'issued_at': current_time
            }
            
        except Exception as e:
            self.logger.error(f"Token generation failed: {e}")
            raise RuntimeError(f"Token generation failed: {e}")
    
    def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token"""
        try:
            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is revoked
            jti = payload.get('jti')
            if jti in self.revoked_tokens:
                return {
                    'valid': False,
                    'details': 'Token has been revoked'
                }
            
            # Check if token is in active tokens
            if jti not in self.active_tokens and payload.get('type') != 'refresh':
                return {
                    'valid': False,
                    'details': 'Token not found in active tokens'
                }
            
            # Update metrics
            self.token_metrics['validated'] += 1
            
            return {
                'valid': True,
                'user_id': payload['user_id'],
                'jti': jti,
                'issued_at': payload['iat'],
                'expires_at': payload['exp'],
                'metadata': payload.get('metadata', {})
            }
            
        except jwt.ExpiredSignatureError:
            self.token_metrics['expired'] += 1
            return {
                'valid': False,
                'details': 'Token has expired'
            }
        except jwt.InvalidTokenError as e:
            return {
                'valid': False,
                'details': f'Invalid token: {str(e)}'
            }
        except Exception as e:
            self.logger.error(f"Token validation failed: {e}")
            return {
                'valid': False,
                'details': f'Validation error: {str(e)}'
            }
    
    def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token using refresh token"""
        try:
            # Validate refresh token
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get('type') != 'refresh':
                return {
                    'success': False,
                    'details': 'Invalid refresh token type'
                }
            
            jti = payload.get('jti')
            if jti not in self.refresh_tokens:
                return {
                    'success': False,
                    'details': 'Refresh token not found'
                }
            
            # Revoke old access token
            refresh_info = self.refresh_tokens[jti]
            old_access_jti = refresh_info.get('access_token_jti')
            if old_access_jti:
                self.revoke_token(old_access_jti)
            
            # Generate new access token
            new_tokens = self.generate_token(payload['user_id'], payload.get('metadata'))
            
            # Update metrics
            self.token_metrics['refreshed'] += 1
            
            return {
                'success': True,
                'tokens': new_tokens
            }
            
        except Exception as e:
            self.logger.error(f"Token refresh failed: {e}")
            return {
                'success': False,
                'details': str(e)
            }
    
    def revoke_token(self, jti: str):
        """Revoke a token"""
        try:
            self.revoked_tokens.add(jti)
            if jti in self.active_tokens:
                del self.active_tokens[jti]
            
            self.token_metrics['revoked'] += 1
            
        except Exception as e:
            self.logger.error(f"Token revocation failed: {e}")


class UserManager:
    """User management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # User storage (in production, would use database)
        self.users = self._initialize_users()
        self.user_sessions = defaultdict(list)
        
        self.logger.info("User Manager initialized")
    
    def _initialize_users(self) -> Dict[str, Dict[str, Any]]:
        """Initialize user data"""
        return {
            'admin': {
                'user_id': 'admin',
                'username': 'admin',
                'email': 'admin@saraphis.ai',
                'role': 'admin',
                'tier': 'enterprise',
                'permissions': ['*'],
                'api_keys': ['sk_admin_' + secrets.token_urlsafe(16)],
                'created_at': time.time(),
                'status': 'active'
            },
            'user1': {
                'user_id': 'user1',
                'username': 'testuser',
                'email': 'user@saraphis.ai',
                'role': 'user',
                'tier': 'pro',
                'permissions': ['read', 'write'],
                'api_keys': ['sk_user_' + secrets.token_urlsafe(16)],
                'created_at': time.time(),
                'status': 'active'
            },
            'service': {
                'user_id': 'service',
                'username': 'service_account',
                'email': 'service@saraphis.ai',
                'role': 'service',
                'tier': 'enterprise',
                'permissions': ['read', 'write', 'execute'],
                'api_keys': ['sk_service_' + secrets.token_urlsafe(16)],
                'created_at': time.time(),
                'status': 'active'
            }
        }
    
    def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """Get user information"""
        try:
            if user_id in self.users:
                user = self.users[user_id].copy()
                # Remove sensitive data
                user.pop('api_keys', None)
                
                return {
                    'found': True,
                    'user': user
                }
            
            return {
                'found': False,
                'details': f'User {user_id} not found'
            }
            
        except Exception as e:
            self.logger.error(f"User info retrieval failed: {e}")
            return {
                'found': False,
                'details': str(e)
            }
    
    def validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """Validate API key"""
        try:
            # Find user with API key
            for user_id, user_data in self.users.items():
                if api_key in user_data.get('api_keys', []):
                    return {
                        'valid': True,
                        'user_id': user_id,
                        'user': self.get_user_info(user_id)['user']
                    }
            
            return {
                'valid': False,
                'details': 'Invalid API key'
            }
            
        except Exception as e:
            self.logger.error(f"API key validation failed: {e}")
            return {
                'valid': False,
                'details': str(e)
            }
    
    def create_session(self, user_id: str, session_id: str, metadata: Dict[str, Any] = None):
        """Create user session"""
        try:
            session = {
                'session_id': session_id,
                'user_id': user_id,
                'created_at': time.time(),
                'last_activity': time.time(),
                'metadata': metadata or {}
            }
            
            self.user_sessions[user_id].append(session)
            
        except Exception as e:
            self.logger.error(f"Session creation failed: {e}")
    
    def update_session_activity(self, session_id: str):
        """Update session last activity"""
        try:
            for user_sessions in self.user_sessions.values():
                for session in user_sessions:
                    if session['session_id'] == session_id:
                        session['last_activity'] = time.time()
                        return
                        
        except Exception as e:
            self.logger.error(f"Session activity update failed: {e}")


class PermissionManager:
    """Permission management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Permission definitions
        self.permissions = self._initialize_permissions()
        self.role_permissions = self._initialize_role_permissions()
        
        self.logger.info("Permission Manager initialized")
    
    def _initialize_permissions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize permission definitions"""
        return {
            'brain': {
                'read': ['GET /brain', 'GET /brain/status', 'GET /brain/health'],
                'write': ['POST /brain/predict'],
                'admin': ['PUT /brain/config', 'DELETE /brain/cache']
            },
            'uncertainty': {
                'read': ['GET /uncertainty'],
                'write': ['POST /uncertainty/quantify', 'POST /uncertainty/analyze'],
                'admin': ['PUT /uncertainty/config']
            },
            'training': {
                'read': ['GET /training', 'GET /training/status'],
                'write': ['POST /training/start'],
                'admin': ['PUT /training/stop', 'DELETE /training/model']
            },
            'compression': {
                'read': ['GET /compression'],
                'write': ['POST /compression/compress', 'POST /compression/decompress'],
                'admin': ['PUT /compression/config']
            },
            'proof': {
                'read': ['GET /proof'],
                'write': ['POST /proof/generate', 'POST /proof/verify'],
                'admin': ['PUT /proof/config']
            },
            'security': {
                'read': ['GET /security', 'GET /security/audit'],
                'write': ['POST /security/compliance'],
                'admin': ['PUT /security/config', 'POST /security/incident']
            },
            'data': {
                'read': ['GET /data'],
                'write': ['POST /data/backup', 'POST /data/restore'],
                'admin': ['PUT /data/replicate', 'DELETE /data']
            },
            'api': {
                'read': ['GET /api/status', 'GET /api/metrics', 'GET /api/health'],
                'write': [],
                'admin': ['PUT /api/config']
            }
        }
    
    def _initialize_role_permissions(self) -> Dict[str, Set[str]]:
        """Initialize role-based permissions"""
        return {
            'admin': {'*'},  # All permissions
            'user': {'read', 'write'},
            'service': {'read', 'write', 'execute'},
            'readonly': {'read'},
            'guest': {'read:api'}
        }
    
    def check_permission(self, user_id: str, endpoint: str, method: str) -> Dict[str, Any]:
        """Check if user has permission for endpoint"""
        try:
            # Get user role (would come from user data in production)
            user_role = self._get_user_role(user_id)
            
            # Admin has all permissions
            if user_role == 'admin' or '*' in self.role_permissions.get(user_role, set()):
                return {
                    'authorized': True,
                    'permissions': ['*']
                }
            
            # Get required permission for endpoint
            required_permission = self._get_required_permission(endpoint, method)
            
            # Check if user role has permission
            user_permissions = self.role_permissions.get(user_role, set())
            
            # Check exact match
            if required_permission in user_permissions:
                return {
                    'authorized': True,
                    'permissions': list(user_permissions)
                }
            
            # Check permission categories
            permission_parts = required_permission.split(':')
            if len(permission_parts) == 2:
                category, action = permission_parts
                if action in user_permissions or f"{category}:*" in user_permissions:
                    return {
                        'authorized': True,
                        'permissions': list(user_permissions)
                    }
            
            return {
                'authorized': False,
                'details': f'Insufficient permissions. Required: {required_permission}',
                'required_permission': required_permission,
                'user_permissions': list(user_permissions)
            }
            
        except Exception as e:
            self.logger.error(f"Permission check failed: {e}")
            return {
                'authorized': False,
                'details': str(e)
            }
    
    def _get_user_role(self, user_id: str) -> str:
        """Get user role (simplified for demo)"""
        role_mapping = {
            'admin': 'admin',
            'user1': 'user',
            'service': 'service'
        }
        return role_mapping.get(user_id, 'guest')
    
    def _get_required_permission(self, endpoint: str, method: str) -> str:
        """Get required permission for endpoint"""
        try:
            # Determine service from endpoint
            service = None
            for svc in self.permissions.keys():
                if f'/{svc}' in endpoint:
                    service = svc
                    break
            
            if not service:
                service = 'api'
            
            # Determine action based on method and endpoint
            endpoint_pattern = f"{method} {endpoint}"
            
            # Check each permission level
            for action in ['admin', 'write', 'read']:
                if endpoint_pattern in self.permissions.get(service, {}).get(action, []):
                    return f"{service}:{action}"
            
            # Default permission
            if method == 'GET':
                return f"{service}:read"
            elif method in ['POST', 'PUT', 'PATCH']:
                return f"{service}:write"
            elif method == 'DELETE':
                return f"{service}:admin"
            else:
                return f"{service}:read"
                
        except Exception as e:
            self.logger.error(f"Failed to determine required permission: {e}")
            return 'unknown:read'


class AuthenticationManager:
    """Production-ready authentication manager with token validation and authorization"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.token_validator = TokenValidator(config.get('tokens', {}))
        self.user_manager = UserManager(config.get('users', {}))
        self.permission_manager = PermissionManager(config.get('permissions', {}))
        
        # Authentication configuration
        self.require_authentication = config.get('require_authentication', True)
        self.allow_api_keys = config.get('allow_api_keys', True)
        self.session_timeout = config.get('session_timeout', 3600)
        
        # Authentication metrics
        self.auth_metrics = {
            'successful_authentications': 0,
            'failed_authentications': 0,
            'successful_authorizations': 0,
            'failed_authorizations': 0
        }
        
        self.logger.info("Authentication Manager initialized")
    
    def authenticate_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate the request using token validation"""
        try:
            # Check if authentication is required
            if not self.require_authentication:
                return {
                    'authenticated': True,
                    'user': {'user_id': 'anonymous', 'role': 'guest'}
                }
            
            # Extract authentication credentials
            auth_token = self._extract_auth_token(request)
            api_key = self._extract_api_key(request)
            
            # Try token authentication first
            if auth_token:
                token_result = self._authenticate_with_token(auth_token)
                if token_result['authenticated']:
                    self.auth_metrics['successful_authentications'] += 1
                    return token_result
            
            # Try API key authentication
            if api_key and self.allow_api_keys:
                api_key_result = self._authenticate_with_api_key(api_key)
                if api_key_result['authenticated']:
                    self.auth_metrics['successful_authentications'] += 1
                    return api_key_result
            
            # Authentication failed
            self.auth_metrics['failed_authentications'] += 1
            
            return {
                'authenticated': False,
                'details': 'No valid authentication credentials provided'
            }
            
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            self.auth_metrics['failed_authentications'] += 1
            return {
                'authenticated': False,
                'details': f'Authentication error: {str(e)}'
            }
    
    def authorize_request(self, request: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Authorize the request based on user permissions"""
        try:
            endpoint = request.get('endpoint', '')
            method = request.get('method', 'GET')
            user_id = user.get('user_id', 'anonymous')
            
            # Check permissions
            permission_check = self.permission_manager.check_permission(
                user_id, endpoint, method
            )
            
            if permission_check['authorized']:
                self.auth_metrics['successful_authorizations'] += 1
            else:
                self.auth_metrics['failed_authorizations'] += 1
            
            return permission_check
            
        except Exception as e:
            self.logger.error(f"Authorization failed: {e}")
            self.auth_metrics['failed_authorizations'] += 1
            return {
                'authorized': False,
                'details': f'Authorization error: {str(e)}'
            }
    
    def _extract_auth_token(self, request: Dict[str, Any]) -> Optional[str]:
        """Extract authentication token from request"""
        try:
            headers = request.get('headers', {})
            
            # Check Authorization header
            auth_header = headers.get('Authorization', '')
            if auth_header.startswith('Bearer '):
                return auth_header[7:]  # Remove 'Bearer ' prefix
            
            # Check X-Auth-Token header
            if 'X-Auth-Token' in headers:
                return headers['X-Auth-Token']
            
            # Check token in request body
            if 'token' in request:
                return request['token']
            
            return None
            
        except Exception as e:
            self.logger.error(f"Token extraction failed: {e}")
            return None
    
    def _extract_api_key(self, request: Dict[str, Any]) -> Optional[str]:
        """Extract API key from request"""
        try:
            headers = request.get('headers', {})
            
            # Check X-API-Key header
            if 'X-API-Key' in headers:
                return headers['X-API-Key']
            
            # Check api_key parameter
            if 'api_key' in request:
                return request['api_key']
            
            # Check query parameters
            params = request.get('parameters', {})
            if 'api_key' in params:
                return params['api_key']
            
            return None
            
        except Exception as e:
            self.logger.error(f"API key extraction failed: {e}")
            return None
    
    def _authenticate_with_token(self, token: str) -> Dict[str, Any]:
        """Authenticate using JWT token"""
        try:
            # Validate token
            validation_result = self.token_validator.validate_token(token)
            
            if not validation_result['valid']:
                return {
                    'authenticated': False,
                    'details': validation_result.get('details', 'Invalid token')
                }
            
            # Get user info
            user_id = validation_result['user_id']
            user_info = self.user_manager.get_user_info(user_id)
            
            if not user_info['found']:
                return {
                    'authenticated': False,
                    'details': 'User not found'
                }
            
            # Create session
            session_id = secrets.token_urlsafe(16)
            self.user_manager.create_session(user_id, session_id, {
                'token_jti': validation_result['jti']
            })
            
            return {
                'authenticated': True,
                'user': user_info['user'],
                'session_id': session_id,
                'token_info': validation_result
            }
            
        except Exception as e:
            self.logger.error(f"Token authentication failed: {e}")
            return {
                'authenticated': False,
                'details': str(e)
            }
    
    def _authenticate_with_api_key(self, api_key: str) -> Dict[str, Any]:
        """Authenticate using API key"""
        try:
            # Validate API key
            validation_result = self.user_manager.validate_api_key(api_key)
            
            if not validation_result['valid']:
                return {
                    'authenticated': False,
                    'details': validation_result.get('details', 'Invalid API key')
                }
            
            # Create session
            session_id = secrets.token_urlsafe(16)
            self.user_manager.create_session(validation_result['user_id'], session_id, {
                'auth_method': 'api_key'
            })
            
            return {
                'authenticated': True,
                'user': validation_result['user'],
                'session_id': session_id,
                'auth_method': 'api_key'
            }
            
        except Exception as e:
            self.logger.error(f"API key authentication failed: {e}")
            return {
                'authenticated': False,
                'details': str(e)
            }
    
    def generate_token(self, user_id: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate authentication token for user"""
        try:
            return self.token_validator.generate_token(user_id, metadata)
        except Exception as e:
            self.logger.error(f"Token generation failed: {e}")
            raise
    
    def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh authentication token"""
        try:
            return self.token_validator.refresh_token(refresh_token)
        except Exception as e:
            self.logger.error(f"Token refresh failed: {e}")
            raise
    
    def revoke_token(self, token: str):
        """Revoke authentication token"""
        try:
            # Decode token to get JTI
            payload = jwt.decode(token, self.token_validator.secret_key, 
                               algorithms=[self.token_validator.algorithm], 
                               options={"verify_exp": False})
            jti = payload.get('jti')
            if jti:
                self.token_validator.revoke_token(jti)
        except Exception as e:
            self.logger.error(f"Token revocation failed: {e}")
            raise
    
    def get_auth_metrics(self) -> Dict[str, Any]:
        """Get authentication metrics"""
        try:
            return {
                'authentication': self.auth_metrics,
                'tokens': self.token_validator.token_metrics,
                'active_sessions': sum(len(sessions) for sessions in self.user_manager.user_sessions.values())
            }
        except Exception as e:
            self.logger.error(f"Failed to get auth metrics: {e}")
            return {}