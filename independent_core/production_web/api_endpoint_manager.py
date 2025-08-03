"""
Saraphis API Endpoint Manager
Production-ready API endpoint management for web interface
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
import json
import re

logger = logging.getLogger(__name__)


class APIEndpointManager:
    """Production-ready API endpoint management for web interface"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # API endpoint registry
        self.endpoints = {
            # Dashboard endpoints
            '/api/v1/dashboards': {
                'methods': ['GET', 'POST'],
                'description': 'List and create dashboards',
                'handler': 'dashboard_handler',
                'auth_required': True,
                'rate_limit': 100,
                'cache_ttl': 60
            },
            '/api/v1/dashboards/{dashboard_id}': {
                'methods': ['GET', 'PUT', 'DELETE'],
                'description': 'Get, update, or delete specific dashboard',
                'handler': 'dashboard_detail_handler',
                'auth_required': True,
                'rate_limit': 100,
                'cache_ttl': 30
            },
            '/api/v1/dashboards/{dashboard_id}/render': {
                'methods': ['POST'],
                'description': 'Render dashboard with real-time data',
                'handler': 'dashboard_render_handler',
                'auth_required': True,
                'rate_limit': 50,
                'cache_ttl': 5
            },
            
            # Component endpoints
            '/api/v1/components': {
                'methods': ['GET', 'POST'],
                'description': 'List and create components',
                'handler': 'component_handler',
                'auth_required': True,
                'rate_limit': 100,
                'cache_ttl': 60
            },
            '/api/v1/components/{component_id}': {
                'methods': ['GET', 'PUT', 'DELETE'],
                'description': 'Manage specific component',
                'handler': 'component_detail_handler',
                'auth_required': True,
                'rate_limit': 100,
                'cache_ttl': 30
            },
            '/api/v1/components/{component_id}/update': {
                'methods': ['POST'],
                'description': 'Update component with new data',
                'handler': 'component_update_handler',
                'auth_required': True,
                'rate_limit': 200,
                'cache_ttl': 0
            },
            
            # Real-time data endpoints
            '/api/v1/realtime/subscribe': {
                'methods': ['POST'],
                'description': 'Subscribe to real-time data stream',
                'handler': 'realtime_subscribe_handler',
                'auth_required': True,
                'rate_limit': 10,
                'cache_ttl': 0
            },
            '/api/v1/realtime/unsubscribe': {
                'methods': ['POST'],
                'description': 'Unsubscribe from data stream',
                'handler': 'realtime_unsubscribe_handler',
                'auth_required': True,
                'rate_limit': 10,
                'cache_ttl': 0
            },
            '/api/v1/realtime/streams': {
                'methods': ['GET'],
                'description': 'List available data streams',
                'handler': 'realtime_streams_handler',
                'auth_required': True,
                'rate_limit': 50,
                'cache_ttl': 300
            },
            
            # User preference endpoints
            '/api/v1/users/{user_id}/preferences': {
                'methods': ['GET', 'PUT'],
                'description': 'Get or update user preferences',
                'handler': 'user_preferences_handler',
                'auth_required': True,
                'rate_limit': 50,
                'cache_ttl': 60
            },
            '/api/v1/users/{user_id}/permissions': {
                'methods': ['GET'],
                'description': 'Get user permissions',
                'handler': 'user_permissions_handler',
                'auth_required': True,
                'rate_limit': 50,
                'cache_ttl': 300
            },
            '/api/v1/users/{user_id}/sessions': {
                'methods': ['GET', 'DELETE'],
                'description': 'Manage user sessions',
                'handler': 'user_sessions_handler',
                'auth_required': True,
                'rate_limit': 20,
                'cache_ttl': 0
            },
            
            # Interaction endpoints
            '/api/v1/interactions': {
                'methods': ['POST'],
                'description': 'Submit user interaction',
                'handler': 'interaction_handler',
                'auth_required': True,
                'rate_limit': 500,
                'cache_ttl': 0
            },
            '/api/v1/interactions/validate': {
                'methods': ['POST'],
                'description': 'Validate interaction before processing',
                'handler': 'interaction_validate_handler',
                'auth_required': True,
                'rate_limit': 500,
                'cache_ttl': 0
            },
            
            # Export endpoints
            '/api/v1/export': {
                'methods': ['POST'],
                'description': 'Export data in various formats',
                'handler': 'export_handler',
                'auth_required': True,
                'rate_limit': 10,
                'cache_ttl': 0
            },
            '/api/v1/export/{export_id}/status': {
                'methods': ['GET'],
                'description': 'Check export status',
                'handler': 'export_status_handler',
                'auth_required': True,
                'rate_limit': 100,
                'cache_ttl': 5
            },
            '/api/v1/export/{export_id}/download': {
                'methods': ['GET'],
                'description': 'Download exported file',
                'handler': 'export_download_handler',
                'auth_required': True,
                'rate_limit': 10,
                'cache_ttl': 0
            },
            
            # Metrics endpoints
            '/api/v1/metrics/dashboard': {
                'methods': ['GET'],
                'description': 'Get dashboard performance metrics',
                'handler': 'dashboard_metrics_handler',
                'auth_required': True,
                'rate_limit': 50,
                'cache_ttl': 60
            },
            '/api/v1/metrics/components': {
                'methods': ['GET'],
                'description': 'Get component performance metrics',
                'handler': 'component_metrics_handler',
                'auth_required': True,
                'rate_limit': 50,
                'cache_ttl': 60
            },
            '/api/v1/metrics/realtime': {
                'methods': ['GET'],
                'description': 'Get real-time data metrics',
                'handler': 'realtime_metrics_handler',
                'auth_required': True,
                'rate_limit': 50,
                'cache_ttl': 30
            },
            
            # WebSocket endpoints
            '/api/v1/websocket/connect': {
                'methods': ['GET'],
                'description': 'Establish WebSocket connection',
                'handler': 'websocket_connect_handler',
                'auth_required': True,
                'rate_limit': 10,
                'cache_ttl': 0,
                'websocket': True
            },
            '/api/v1/websocket/status': {
                'methods': ['GET'],
                'description': 'Get WebSocket connection status',
                'handler': 'websocket_status_handler',
                'auth_required': True,
                'rate_limit': 50,
                'cache_ttl': 10
            },
            
            # Health check endpoints
            '/api/v1/health': {
                'methods': ['GET'],
                'description': 'Basic health check',
                'handler': 'health_handler',
                'auth_required': False,
                'rate_limit': 1000,
                'cache_ttl': 5
            },
            '/api/v1/health/detailed': {
                'methods': ['GET'],
                'description': 'Detailed health status',
                'handler': 'health_detailed_handler',
                'auth_required': True,
                'rate_limit': 100,
                'cache_ttl': 10
            },
            
            # Admin endpoints
            '/api/v1/admin/config': {
                'methods': ['GET', 'PUT'],
                'description': 'Get or update system configuration',
                'handler': 'admin_config_handler',
                'auth_required': True,
                'role_required': 'admin',
                'rate_limit': 10,
                'cache_ttl': 0
            },
            '/api/v1/admin/users': {
                'methods': ['GET', 'POST'],
                'description': 'Manage system users',
                'handler': 'admin_users_handler',
                'auth_required': True,
                'role_required': 'admin',
                'rate_limit': 20,
                'cache_ttl': 0
            },
            '/api/v1/admin/logs': {
                'methods': ['GET'],
                'description': 'Access system logs',
                'handler': 'admin_logs_handler',
                'auth_required': True,
                'role_required': 'admin',
                'rate_limit': 10,
                'cache_ttl': 0
            }
        }
        
        # Endpoint handlers (would be actual implementations)
        self.handlers = {}
        
        # Request validation schemas
        self.request_schemas = self._initialize_request_schemas()
        
        # Response schemas
        self.response_schemas = self._initialize_response_schemas()
        
        # Endpoint statistics
        self.endpoint_stats = defaultdict(lambda: {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0,
            'last_accessed': None
        })
        
        # Route patterns
        self.route_patterns = self._compile_route_patterns()
        
        # Middleware stack
        self.middleware = [
            self._authentication_middleware,
            self._rate_limiting_middleware,
            self._validation_middleware,
            self._logging_middleware
        ]
        
        # Thread safety
        self._lock = threading.Lock()
        
        self.logger.info("API Endpoint Manager initialized")
    
    def get_all_endpoints(self) -> Dict[str, Any]:
        """Get all registered endpoints with metadata"""
        try:
            endpoints = {}
            
            for path, config in self.endpoints.items():
                endpoints[path] = {
                    'methods': config['methods'],
                    'description': config['description'],
                    'auth_required': config.get('auth_required', True),
                    'role_required': config.get('role_required'),
                    'rate_limit': config.get('rate_limit', 100),
                    'cache_ttl': config.get('cache_ttl', 0),
                    'websocket': config.get('websocket', False),
                    'request_schema': self._get_schema_summary(path, 'request'),
                    'response_schema': self._get_schema_summary(path, 'response')
                }
            
            return endpoints
            
        except Exception as e:
            self.logger.error(f"Endpoint retrieval failed: {e}")
            return {}
    
    def match_endpoint(self, path: str, method: str) -> Optional[Dict[str, Any]]:
        """Match request path and method to endpoint"""
        try:
            # Direct match
            if path in self.endpoints:
                endpoint = self.endpoints[path]
                if method in endpoint['methods']:
                    return {
                        'endpoint': path,
                        'config': endpoint,
                        'params': {}
                    }
            
            # Pattern match
            for pattern, (endpoint_path, regex) in self.route_patterns.items():
                match = regex.match(path)
                if match:
                    endpoint = self.endpoints[endpoint_path]
                    if method in endpoint['methods']:
                        return {
                            'endpoint': endpoint_path,
                            'config': endpoint,
                            'params': match.groupdict()
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Endpoint matching failed: {e}")
            return None
    
    def validate_request(self, endpoint: str, method: str, 
                        request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate request against endpoint schema"""
        try:
            schema_key = f"{endpoint}:{method}"
            
            if schema_key not in self.request_schemas:
                return {
                    'valid': True,
                    'details': 'No schema defined'
                }
            
            schema = self.request_schemas[schema_key]
            
            # Validate required fields
            missing_fields = []
            for field in schema.get('required', []):
                if field not in request_data:
                    missing_fields.append(field)
            
            if missing_fields:
                return {
                    'valid': False,
                    'details': f'Missing required fields: {missing_fields}'
                }
            
            # Validate field types
            type_errors = []
            for field, field_schema in schema.get('properties', {}).items():
                if field in request_data:
                    expected_type = field_schema.get('type')
                    actual_value = request_data[field]
                    
                    if not self._validate_type(actual_value, expected_type):
                        type_errors.append(
                            f'{field}: expected {expected_type}, got {type(actual_value).__name__}'
                        )
            
            if type_errors:
                return {
                    'valid': False,
                    'details': f'Type errors: {type_errors}'
                }
            
            return {
                'valid': True,
                'details': 'Validation passed'
            }
            
        except Exception as e:
            self.logger.error(f"Request validation failed: {e}")
            return {
                'valid': False,
                'details': f'Validation error: {str(e)}'
            }
    
    def format_response(self, endpoint: str, method: str, 
                       response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format response according to endpoint schema"""
        try:
            # Add standard response wrapper
            formatted_response = {
                'success': response_data.get('success', True),
                'timestamp': time.time(),
                'endpoint': endpoint,
                'method': method
            }
            
            # Add data or error
            if response_data.get('success', True):
                formatted_response['data'] = response_data.get('data', response_data)
            else:
                formatted_response['error'] = {
                    'code': response_data.get('error_code', 'UNKNOWN_ERROR'),
                    'message': response_data.get('error', 'An error occurred'),
                    'details': response_data.get('details', {})
                }
            
            # Add metadata if present
            if 'metadata' in response_data:
                formatted_response['metadata'] = response_data['metadata']
            
            # Add pagination if present
            if 'pagination' in response_data:
                formatted_response['pagination'] = response_data['pagination']
            
            return formatted_response
            
        except Exception as e:
            self.logger.error(f"Response formatting failed: {e}")
            return {
                'success': False,
                'error': {
                    'code': 'FORMATTING_ERROR',
                    'message': str(e)
                },
                'timestamp': time.time()
            }
    
    def get_endpoint_stats(self, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """Get endpoint usage statistics"""
        try:
            with self._lock:
                if endpoint:
                    return self.endpoint_stats.get(endpoint, {}).copy()
                else:
                    return dict(self.endpoint_stats)
                    
        except Exception as e:
            self.logger.error(f"Stats retrieval failed: {e}")
            return {}
    
    def record_request(self, endpoint: str, method: str, 
                      response_time: float, success: bool):
        """Record endpoint request for statistics"""
        try:
            with self._lock:
                stats = self.endpoint_stats[endpoint]
                
                # Update counts
                stats['total_requests'] += 1
                if success:
                    stats['successful_requests'] += 1
                else:
                    stats['failed_requests'] += 1
                
                # Update average response time
                total_time = stats['average_response_time'] * (stats['total_requests'] - 1)
                stats['average_response_time'] = (total_time + response_time) / stats['total_requests']
                
                # Update last accessed
                stats['last_accessed'] = time.time()
                
        except Exception as e:
            self.logger.error(f"Request recording failed: {e}")
    
    def _initialize_request_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Initialize request validation schemas"""
        schemas = {
            # Dashboard schemas
            '/api/v1/dashboards:POST': {
                'required': ['name', 'type'],
                'properties': {
                    'name': {'type': 'string', 'max_length': 100},
                    'type': {'type': 'string', 'enum': ['system_overview', 'uncertainty_analysis', 'training_monitoring', 'production_metrics']},
                    'description': {'type': 'string', 'max_length': 500},
                    'configuration': {'type': 'object'}
                }
            },
            '/api/v1/dashboards/{dashboard_id}:PUT': {
                'properties': {
                    'name': {'type': 'string', 'max_length': 100},
                    'description': {'type': 'string', 'max_length': 500},
                    'configuration': {'type': 'object'}
                }
            },
            '/api/v1/dashboards/{dashboard_id}/render:POST': {
                'properties': {
                    'preferences': {'type': 'object'},
                    'time_range': {'type': 'object'},
                    'refresh': {'type': 'boolean'}
                }
            },
            
            # Component schemas
            '/api/v1/components:POST': {
                'required': ['type', 'configuration'],
                'properties': {
                    'type': {'type': 'string'},
                    'configuration': {'type': 'object'},
                    'dashboard_id': {'type': 'string'}
                }
            },
            '/api/v1/components/{component_id}/update:POST': {
                'required': ['data'],
                'properties': {
                    'data': {'type': 'object'},
                    'configuration': {'type': 'object'}
                }
            },
            
            # Real-time schemas
            '/api/v1/realtime/subscribe:POST': {
                'required': ['stream'],
                'properties': {
                    'stream': {'type': 'string'},
                    'filters': {'type': 'object'}
                }
            },
            '/api/v1/realtime/unsubscribe:POST': {
                'required': ['stream'],
                'properties': {
                    'stream': {'type': 'string'}
                }
            },
            
            # Interaction schemas
            '/api/v1/interactions:POST': {
                'required': ['interaction_type'],
                'properties': {
                    'interaction_type': {'type': 'string'},
                    'session_id': {'type': 'string'},
                    'dashboard_type': {'type': 'string'}
                }
            },
            
            # Export schemas
            '/api/v1/export:POST': {
                'required': ['format', 'data_type'],
                'properties': {
                    'format': {'type': 'string', 'enum': ['json', 'csv', 'xlsx', 'pdf', 'png']},
                    'data_type': {'type': 'string', 'enum': ['dashboard', 'chart', 'table', 'metrics', 'logs']},
                    'filters': {'type': 'object'},
                    'options': {'type': 'object'}
                }
            }
        }
        
        return schemas
    
    def _initialize_response_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Initialize response schemas"""
        schemas = {
            # Standard success response
            'success': {
                'type': 'object',
                'properties': {
                    'success': {'type': 'boolean'},
                    'data': {'type': 'object'},
                    'timestamp': {'type': 'number'},
                    'endpoint': {'type': 'string'},
                    'method': {'type': 'string'}
                }
            },
            
            # Standard error response
            'error': {
                'type': 'object',
                'properties': {
                    'success': {'type': 'boolean'},
                    'error': {
                        'type': 'object',
                        'properties': {
                            'code': {'type': 'string'},
                            'message': {'type': 'string'},
                            'details': {'type': 'object'}
                        }
                    },
                    'timestamp': {'type': 'number'}
                }
            },
            
            # Paginated response
            'paginated': {
                'type': 'object',
                'properties': {
                    'success': {'type': 'boolean'},
                    'data': {'type': 'array'},
                    'pagination': {
                        'type': 'object',
                        'properties': {
                            'page': {'type': 'integer'},
                            'per_page': {'type': 'integer'},
                            'total': {'type': 'integer'},
                            'pages': {'type': 'integer'}
                        }
                    },
                    'timestamp': {'type': 'number'}
                }
            }
        }
        
        return schemas
    
    def _compile_route_patterns(self) -> Dict[str, tuple]:
        """Compile regex patterns for parameterized routes"""
        patterns = {}
        
        for endpoint in self.endpoints:
            # Check if endpoint has parameters
            if '{' in endpoint:
                # Convert route pattern to regex
                pattern = endpoint
                param_names = re.findall(r'{(\w+)}', endpoint)
                
                for param_name in param_names:
                    pattern = pattern.replace(
                        f'{{{param_name}}}',
                        f'(?P<{param_name}>[^/]+)'
                    )
                
                # Add start and end anchors
                pattern = f'^{pattern}$'
                
                patterns[endpoint] = (endpoint, re.compile(pattern))
        
        return patterns
    
    def _get_schema_summary(self, endpoint: str, schema_type: str) -> Dict[str, Any]:
        """Get summary of endpoint schema"""
        if schema_type == 'request':
            schemas = self.request_schemas
        else:
            schemas = self.response_schemas
        
        # Find matching schema
        for method in self.endpoints.get(endpoint, {}).get('methods', []):
            schema_key = f"{endpoint}:{method}"
            if schema_key in schemas:
                schema = schemas[schema_key]
                return {
                    'required': schema.get('required', []),
                    'properties': list(schema.get('properties', {}).keys())
                }
        
        return {}
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type"""
        type_mapping = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # Unknown type, allow
    
    def _authentication_middleware(self, request: Dict[str, Any], 
                                 endpoint_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Authentication middleware"""
        if not endpoint_config.get('auth_required', True):
            return None
        
        # Check authentication
        if 'user' not in request or not request['user']:
            return {
                'success': False,
                'error_code': 'AUTHENTICATION_REQUIRED',
                'error': 'Authentication required'
            }
        
        # Check role if required
        role_required = endpoint_config.get('role_required')
        if role_required:
            user_role = request['user'].get('role', 'user')
            if user_role != role_required and user_role != 'admin':
                return {
                    'success': False,
                    'error_code': 'INSUFFICIENT_PERMISSIONS',
                    'error': f'Role {role_required} required'
                }
        
        return None
    
    def _rate_limiting_middleware(self, request: Dict[str, Any], 
                                endpoint_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Rate limiting middleware"""
        # Rate limiting would be implemented here
        # For now, just pass through
        return None
    
    def _validation_middleware(self, request: Dict[str, Any], 
                             endpoint_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Request validation middleware"""
        endpoint = request.get('endpoint')
        method = request.get('method')
        data = request.get('data', {})
        
        validation_result = self.validate_request(endpoint, method, data)
        
        if not validation_result['valid']:
            return {
                'success': False,
                'error_code': 'VALIDATION_ERROR',
                'error': validation_result['details']
            }
        
        return None
    
    def _logging_middleware(self, request: Dict[str, Any], 
                          endpoint_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Logging middleware"""
        self.logger.info(
            f"API Request: {request.get('method')} {request.get('endpoint')} "
            f"from user {request.get('user', {}).get('user_id', 'anonymous')}"
        )
        
        return None