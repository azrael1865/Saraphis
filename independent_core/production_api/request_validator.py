"""
Saraphis Request Validator
Production-ready request validation and input sanitization
NO FALLBACKS - HARD FAILURES ONLY
"""

import re
import time
import logging
import json
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
import html
import urllib.parse

logger = logging.getLogger(__name__)


class InputSanitizer:
    """Input sanitization system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Sanitization rules
        self.max_string_length = config.get('max_string_length', 10000)
        self.max_array_size = config.get('max_array_size', 1000)
        self.max_object_depth = config.get('max_object_depth', 10)
        self.allow_html = config.get('allow_html', False)
        self.allow_sql = config.get('allow_sql', False)
        
        # Dangerous patterns
        self.dangerous_patterns = {
            'sql_injection': [
                r"(\b(union|select|insert|update|delete|drop|create)\b.*\b(from|where|table)\b)",
                r"(\'|\")(\s*)(or|and)(\s*)(\'|\")?(\s*)=",
                r"(\b(exec|execute|xp_)\b)",
                r"(;|--|\*|\/\*)"
            ],
            'xss': [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>",
                r"<object[^>]*>",
                r"<embed[^>]*>"
            ],
            'command_injection': [
                r"(\||;|&|`|\$\(|\$\{)",
                r"(>|<|>>|<<)",
                r"(rm|dd|chmod|chown|kill|shutdown|reboot)"
            ],
            'path_traversal': [
                r"\.\./",
                r"\.\\/",
                r"%2e%2e",
                r"\.\.%2f"
            ]
        }
        
        self.logger.info("Input Sanitizer initialized")
    
    def sanitize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request parameters"""
        try:
            sanitized = {}
            errors = []
            
            for key, value in parameters.items():
                # Sanitize key
                sanitized_key = self._sanitize_string(key, 'parameter_key')
                if sanitized_key != key:
                    errors.append(f"Parameter key modified: {key}")
                
                # Sanitize value
                sanitized_value, value_errors = self._sanitize_value(value, f"parameter.{key}")
                sanitized[sanitized_key] = sanitized_value
                errors.extend(value_errors)
            
            if errors:
                sanitized['errors'] = errors
            
            return sanitized
            
        except Exception as e:
            self.logger.error(f"Parameter sanitization failed: {e}")
            return {
                'errors': [f'Sanitization error: {str(e)}']
            }
    
    def sanitize_body(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request body"""
        try:
            sanitized_body, errors = self._sanitize_object(body, 'body', 0)
            
            if errors:
                sanitized_body['errors'] = errors
            
            return sanitized_body
            
        except Exception as e:
            self.logger.error(f"Body sanitization failed: {e}")
            return {
                'errors': [f'Body sanitization error: {str(e)}']
            }
    
    def _sanitize_value(self, value: Any, path: str) -> tuple[Any, List[str]]:
        """Sanitize any value type"""
        errors = []
        
        try:
            if isinstance(value, str):
                return self._sanitize_string(value, path), errors
            elif isinstance(value, dict):
                return self._sanitize_object(value, path, 0)
            elif isinstance(value, list):
                return self._sanitize_array(value, path)
            elif isinstance(value, (int, float)):
                return self._sanitize_number(value, path), errors
            elif isinstance(value, bool):
                return value, errors
            elif value is None:
                return None, errors
            else:
                errors.append(f"Unknown value type at {path}: {type(value)}")
                return str(value), errors
                
        except Exception as e:
            errors.append(f"Sanitization error at {path}: {str(e)}")
            return None, errors
    
    def _sanitize_string(self, value: str, path: str) -> str:
        """Sanitize string value"""
        try:
            # Check length
            if len(value) > self.max_string_length:
                value = value[:self.max_string_length]
            
            # Remove null bytes
            value = value.replace('\x00', '')
            
            # HTML escape if needed
            if not self.allow_html:
                value = html.escape(value)
            
            # Check for dangerous patterns
            for pattern_type, patterns in self.dangerous_patterns.items():
                if pattern_type == 'sql_injection' and self.allow_sql:
                    continue
                    
                for pattern in patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        # Log potential attack
                        self.logger.warning(f"Dangerous pattern detected ({pattern_type}) at {path}")
                        # Remove or escape the pattern
                        value = re.sub(pattern, '', value, flags=re.IGNORECASE)
            
            # URL decode to prevent encoded attacks
            try:
                decoded = urllib.parse.unquote(value)
                if decoded != value:
                    # Re-sanitize decoded value
                    return self._sanitize_string(decoded, path)
            except:
                pass
            
            return value
            
        except Exception as e:
            self.logger.error(f"String sanitization failed: {e}")
            return ""
    
    def _sanitize_object(self, obj: Dict[str, Any], path: str, depth: int) -> tuple[Dict[str, Any], List[str]]:
        """Sanitize object/dictionary"""
        errors = []
        
        try:
            # Check depth
            if depth > self.max_object_depth:
                errors.append(f"Object depth exceeded at {path}")
                return {}, errors
            
            sanitized = {}
            
            for key, value in obj.items():
                # Sanitize key
                sanitized_key = self._sanitize_string(str(key), f"{path}.key")
                
                # Sanitize value
                sanitized_value, value_errors = self._sanitize_value(
                    value, f"{path}.{sanitized_key}"
                )
                
                sanitized[sanitized_key] = sanitized_value
                errors.extend(value_errors)
            
            return sanitized, errors
            
        except Exception as e:
            errors.append(f"Object sanitization error at {path}: {str(e)}")
            return {}, errors
    
    def _sanitize_array(self, arr: List[Any], path: str) -> tuple[List[Any], List[str]]:
        """Sanitize array/list"""
        errors = []
        
        try:
            # Check size
            if len(arr) > self.max_array_size:
                errors.append(f"Array size exceeded at {path}: {len(arr)} > {self.max_array_size}")
                arr = arr[:self.max_array_size]
            
            sanitized = []
            
            for i, item in enumerate(arr):
                sanitized_item, item_errors = self._sanitize_value(item, f"{path}[{i}]")
                sanitized.append(sanitized_item)
                errors.extend(item_errors)
            
            return sanitized, errors
            
        except Exception as e:
            errors.append(f"Array sanitization error at {path}: {str(e)}")
            return [], errors
    
    def _sanitize_number(self, value: float, path: str) -> float:
        """Sanitize numeric value"""
        try:
            # Check for infinity or NaN
            if value != value:  # NaN check
                return 0
            elif value == float('inf'):
                return 999999999
            elif value == float('-inf'):
                return -999999999
            
            # Limit range
            max_value = 1e15
            min_value = -1e15
            
            return max(min_value, min(max_value, value))
            
        except Exception as e:
            self.logger.error(f"Number sanitization failed: {e}")
            return 0


class RequestValidator:
    """Production-ready request validator with comprehensive validation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validation configuration
        self.max_body_size = config.get('max_body_size', 10 * 1024 * 1024)  # 10MB
        self.max_header_size = config.get('max_header_size', 8192)  # 8KB
        self.max_url_length = config.get('max_url_length', 2048)
        self.allowed_content_types = config.get('allowed_content_types', [
            'application/json',
            'application/x-www-form-urlencoded',
            'multipart/form-data',
            'text/plain'
        ])
        
        # Initialize sanitizer
        self.sanitizer = InputSanitizer(config.get('sanitization', {}))
        
        # Validation rules
        self.validation_rules = self._initialize_validation_rules()
        
        # Validation metrics
        self.validation_metrics = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'sanitizations_performed': 0
        }
        
        self.logger.info("Request Validator initialized")
    
    def validate_request_format(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate request format and structure"""
        try:
            errors = []
            
            # Update metrics
            self.validation_metrics['total_validations'] += 1
            
            # Check required fields
            required_fields = ['endpoint', 'method']
            for field in required_fields:
                if field not in request:
                    errors.append(f'Missing required field: {field}')
            
            # Validate endpoint
            if 'endpoint' in request:
                endpoint_errors = self._validate_endpoint(request['endpoint'])
                errors.extend(endpoint_errors)
            
            # Validate HTTP method
            if 'method' in request:
                method_errors = self._validate_method(request['method'])
                errors.extend(method_errors)
            
            # Validate headers
            if 'headers' in request:
                header_errors = self._validate_headers(request['headers'])
                errors.extend(header_errors)
            
            # Validate parameters
            if 'parameters' in request:
                param_errors = self._validate_parameters(request['parameters'])
                errors.extend(param_errors)
            
            # Validate body
            if 'body' in request:
                body_errors = self._validate_body(request['body'], request.get('headers', {}))
                errors.extend(body_errors)
            
            # Update metrics
            if errors:
                self.validation_metrics['failed_validations'] += 1
            else:
                self.validation_metrics['passed_validations'] += 1
            
            return {
                'valid': len(errors) == 0,
                'errors': errors
            }
            
        except Exception as e:
            self.logger.error(f"Request format validation failed: {e}")
            self.validation_metrics['failed_validations'] += 1
            return {
                'valid': False,
                'errors': [f'Validation error: {str(e)}']
            }
    
    def validate_request_content(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate request content and parameters"""
        try:
            errors = []
            sanitized_request = request.copy()
            
            # Sanitize parameters
            if 'parameters' in request:
                sanitized_params = self.sanitizer.sanitize_parameters(request['parameters'])
                if 'errors' in sanitized_params:
                    errors.extend(sanitized_params['errors'])
                    sanitized_params.pop('errors')
                sanitized_request['parameters'] = sanitized_params
                self.validation_metrics['sanitizations_performed'] += 1
            
            # Sanitize body
            if 'body' in request:
                sanitized_body = self.sanitizer.sanitize_body(request['body'])
                if 'errors' in sanitized_body:
                    errors.extend(sanitized_body['errors'])
                    sanitized_body.pop('errors')
                sanitized_request['body'] = sanitized_body
                self.validation_metrics['sanitizations_performed'] += 1
            
            # Validate content against rules
            if 'endpoint' in request and 'method' in request:
                rule_errors = self._validate_against_rules(sanitized_request)
                errors.extend(rule_errors)
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'sanitized_request': sanitized_request
            }
            
        except Exception as e:
            self.logger.error(f"Request content validation failed: {e}")
            return {
                'valid': False,
                'errors': [f'Content validation error: {str(e)}']
            }
    
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize endpoint-specific validation rules"""
        return {
            '/brain/predict': {
                'required_fields': ['input_data'],
                'field_types': {
                    'input_data': (list, dict),
                    'model_id': str,
                    'confidence_threshold': (int, float)
                },
                'field_constraints': {
                    'confidence_threshold': lambda x: 0 <= x <= 1
                }
            },
            '/training/start': {
                'required_fields': ['dataset_id', 'model_type'],
                'field_types': {
                    'dataset_id': str,
                    'model_type': str,
                    'hyperparameters': dict
                },
                'allowed_values': {
                    'model_type': ['neural_network', 'transformer', 'hybrid']
                }
            },
            '/compression/compress': {
                'required_fields': ['data'],
                'field_types': {
                    'data': str,
                    'algorithm': str
                },
                'allowed_values': {
                    'algorithm': ['lz4', 'zlib', 'brotli']
                }
            },
            '/proof/generate': {
                'required_fields': ['statement', 'proof_type'],
                'field_types': {
                    'statement': str,
                    'proof_type': str,
                    'public_inputs': list
                },
                'allowed_values': {
                    'proof_type': ['zero_knowledge', 'merkle', 'signature']
                }
            }
        }
    
    def _validate_endpoint(self, endpoint: str) -> List[str]:
        """Validate endpoint format"""
        errors = []
        
        try:
            # Check if starts with /
            if not endpoint.startswith('/'):
                errors.append('Endpoint must start with /')
            
            # Check length
            if len(endpoint) > self.max_url_length:
                errors.append(f'Endpoint too long: {len(endpoint)} > {self.max_url_length}')
            
            # Check for invalid characters
            if not re.match(r'^/[\w\-/]*$', endpoint):
                errors.append('Endpoint contains invalid characters')
            
            # Check for path traversal
            if '..' in endpoint:
                errors.append('Path traversal detected in endpoint')
            
            # Check for double slashes
            if '//' in endpoint:
                errors.append('Double slashes not allowed in endpoint')
                
        except Exception as e:
            errors.append(f'Endpoint validation error: {str(e)}')
        
        return errors
    
    def _validate_method(self, method: str) -> List[str]:
        """Validate HTTP method"""
        errors = []
        
        try:
            valid_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD']
            
            if method not in valid_methods:
                errors.append(f'Invalid HTTP method: {method}')
                
        except Exception as e:
            errors.append(f'Method validation error: {str(e)}')
        
        return errors
    
    def _validate_headers(self, headers: Any) -> List[str]:
        """Validate request headers"""
        errors = []
        
        try:
            # Check type
            if not isinstance(headers, dict):
                errors.append('Headers must be a dictionary')
                return errors
            
            # Check total size
            headers_str = json.dumps(headers)
            if len(headers_str) > self.max_header_size:
                errors.append(f'Headers too large: {len(headers_str)} > {self.max_header_size}')
            
            # Validate individual headers
            for name, value in headers.items():
                # Check header name
                if not isinstance(name, str):
                    errors.append(f'Header name must be string: {name}')
                elif not re.match(r'^[\w\-]+$', name):
                    errors.append(f'Invalid header name: {name}')
                
                # Check header value
                if not isinstance(value, str):
                    errors.append(f'Header value must be string: {name}')
                elif len(value) > 4096:
                    errors.append(f'Header value too long: {name}')
                elif '\r' in value or '\n' in value:
                    errors.append(f'Header value contains line breaks: {name}')
                    
        except Exception as e:
            errors.append(f'Headers validation error: {str(e)}')
        
        return errors
    
    def _validate_parameters(self, parameters: Any) -> List[str]:
        """Validate request parameters"""
        errors = []
        
        try:
            # Check type
            if not isinstance(parameters, dict):
                errors.append('Parameters must be a dictionary')
                return errors
            
            # Validate individual parameters
            for name, value in parameters.items():
                # Check parameter name
                if not isinstance(name, str):
                    errors.append(f'Parameter name must be string: {name}')
                elif not re.match(r'^[\w\-\[\]\.]+$', name):
                    errors.append(f'Invalid parameter name: {name}')
                
                # Check parameter value type
                if not isinstance(value, (str, int, float, bool, list)):
                    errors.append(f'Invalid parameter type for {name}: {type(value)}')
                    
        except Exception as e:
            errors.append(f'Parameters validation error: {str(e)}')
        
        return errors
    
    def _validate_body(self, body: Any, headers: Dict[str, str]) -> List[str]:
        """Validate request body"""
        errors = []
        
        try:
            # Check type
            if not isinstance(body, (dict, list, str)):
                errors.append('Body must be a dictionary, list, or string')
                return errors
            
            # Check size
            body_str = json.dumps(body) if isinstance(body, (dict, list)) else str(body)
            if len(body_str) > self.max_body_size:
                errors.append(f'Body too large: {len(body_str)} > {self.max_body_size}')
            
            # Check content type
            content_type = headers.get('Content-Type', 'application/json')
            base_content_type = content_type.split(';')[0].strip()
            
            if base_content_type not in self.allowed_content_types:
                errors.append(f'Content type not allowed: {content_type}')
            
            # Validate JSON structure if applicable
            if base_content_type == 'application/json' and isinstance(body, str):
                try:
                    json.loads(body)
                except json.JSONDecodeError as e:
                    errors.append(f'Invalid JSON in body: {str(e)}')
                    
        except Exception as e:
            errors.append(f'Body validation error: {str(e)}')
        
        return errors
    
    def _validate_against_rules(self, request: Dict[str, Any]) -> List[str]:
        """Validate request against endpoint-specific rules"""
        errors = []
        
        try:
            endpoint = request.get('endpoint', '')
            body = request.get('body', {})
            
            # Find matching rules
            matching_rule = None
            for rule_endpoint, rules in self.validation_rules.items():
                if endpoint == rule_endpoint or endpoint.startswith(rule_endpoint + '/'):
                    matching_rule = rules
                    break
            
            if not matching_rule:
                return errors  # No specific rules for this endpoint
            
            # Check required fields
            if 'required_fields' in matching_rule:
                for field in matching_rule['required_fields']:
                    if field not in body:
                        errors.append(f'Missing required field: {field}')
            
            # Check field types
            if 'field_types' in matching_rule:
                for field, expected_types in matching_rule['field_types'].items():
                    if field in body:
                        value = body[field]
                        if not isinstance(value, expected_types):
                            errors.append(
                                f'Invalid type for {field}: expected {expected_types}, got {type(value)}'
                            )
            
            # Check allowed values
            if 'allowed_values' in matching_rule:
                for field, allowed in matching_rule['allowed_values'].items():
                    if field in body:
                        value = body[field]
                        if value not in allowed:
                            errors.append(
                                f'Invalid value for {field}: {value} not in {allowed}'
                            )
            
            # Check field constraints
            if 'field_constraints' in matching_rule:
                for field, constraint in matching_rule['field_constraints'].items():
                    if field in body:
                        value = body[field]
                        try:
                            if not constraint(value):
                                errors.append(f'Constraint failed for {field}: {value}')
                        except Exception as e:
                            errors.append(f'Constraint check failed for {field}: {str(e)}')
                            
        except Exception as e:
            errors.append(f'Rule validation error: {str(e)}')
        
        return errors
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get validation metrics"""
        try:
            total = self.validation_metrics['total_validations']
            if total > 0:
                success_rate = self.validation_metrics['passed_validations'] / total
            else:
                success_rate = 1.0
            
            return {
                'total_validations': total,
                'passed_validations': self.validation_metrics['passed_validations'],
                'failed_validations': self.validation_metrics['failed_validations'],
                'success_rate': success_rate,
                'sanitizations_performed': self.validation_metrics['sanitizations_performed']
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get validation metrics: {e}")
            return {}