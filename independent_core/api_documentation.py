"""
API Documentation System for Saraphis Independent Core

Comprehensive API documentation generation with support for REST, GraphQL, RPC,
WebSocket, and Internal API documentation across all 186 Python files.

Advanced endpoint discovery, parameter analysis, and interactive documentation generation.
"""

import os
import ast
import json
import inspect
import re
import asyncio
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Union, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import importlib.util
import sys
from urllib.parse import urlparse
import logging

class APIType(Enum):
    """Types of APIs supported"""
    REST = "rest"
    GRAPHQL = "graphql"
    RPC = "rpc"
    WEBSOCKET = "websocket"
    INTERNAL = "internal"
    ASYNC_API = "async_api"

class HTTPMethod(Enum):
    """HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"

class ParameterType(Enum):
    """Parameter types"""
    QUERY = "query"
    PATH = "path"
    BODY = "body"
    HEADER = "header"
    FORM = "form"

@dataclass
class APIParameter:
    """API parameter definition"""
    name: str
    param_type: ParameterType
    data_type: str
    required: bool = True
    description: str = ""
    default_value: Any = None
    validation_rules: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

@dataclass
class APIResponse:
    """API response definition"""
    status_code: int
    description: str
    content_type: str = "application/json"
    schema: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)

@dataclass
class APIEndpoint:
    """Complete API endpoint definition"""
    path: str
    method: HTTPMethod
    function_name: str
    description: str
    parameters: List[APIParameter] = field(default_factory=list)
    responses: List[APIResponse] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    authentication_required: bool = False
    rate_limited: bool = False
    deprecated: bool = False
    version: str = "1.0"
    module_name: str = ""
    file_path: str = ""
    examples: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class APIDocumentation:
    """Complete API documentation structure"""
    title: str
    description: str
    version: str
    base_url: str
    endpoints: List[APIEndpoint] = field(default_factory=list)
    authentication: Dict[str, Any] = field(default_factory=dict)
    rate_limiting: Dict[str, Any] = field(default_factory=dict)
    error_codes: Dict[str, str] = field(default_factory=dict)
    data_models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    examples: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    changelog: List[Dict[str, Any]] = field(default_factory=list)

class APIDiscovery:
    """Advanced API endpoint discovery system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_patterns = self._compile_api_patterns()
        self.decorator_patterns = self._compile_decorator_patterns()
    
    def _compile_api_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for API detection"""
        return {
            'flask_route': re.compile(r'@app\.route\([\'"]([^\'"]+)[\'"](?:,\s*methods\s*=\s*\[([^\]]+)\])?'),
            'fastapi_route': re.compile(r'@app\.(get|post|put|delete|patch)\([\'"]([^\'"]+)[\'"]'),
            'django_url': re.compile(r'path\([\'"]([^\'"]+)[\'"],\s*(\w+)'),
            'async_def': re.compile(r'async\s+def\s+(\w+)'),
            'websocket': re.compile(r'@app\.websocket\([\'"]([^\'"]+)[\'"]'),
            'rpc_method': re.compile(r'def\s+(\w+).*@rpc\.method'),
            'api_endpoint': re.compile(r'@(api_endpoint|endpoint)\([\'"]([^\'"]+)[\'"]'),
            'route_decorator': re.compile(r'@route\([\'"]([^\'"]+)[\'"]')
        }
    
    def _compile_decorator_patterns(self) -> Dict[str, re.Pattern]:
        """Compile patterns for decorator analysis"""
        return {
            'auth_required': re.compile(r'@(login_required|auth_required|requires_auth)'),
            'rate_limit': re.compile(r'@(rate_limit|limiter)'),
            'deprecated': re.compile(r'@deprecated'),
            'validate': re.compile(r'@validate\(([^)]+)\)'),
            'permission': re.compile(r'@(permission_required|requires_permission)\([\'"]([^\'"]+)[\'"]'),
            'cache': re.compile(r'@cache\(([^)]+)\)')
        }
    
    def discover_apis_in_file(self, file_path: str) -> List[APIEndpoint]:
        """Discover all API endpoints in a Python file"""
        endpoints = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST for detailed analysis
            tree = ast.parse(content)
            
            # Analyze each function
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    endpoint = self._analyze_function_for_api(node, content, file_path)
                    if endpoint:
                        endpoints.append(endpoint)
            
            # Additional pattern-based discovery
            pattern_endpoints = self._discover_with_patterns(content, file_path)
            endpoints.extend(pattern_endpoints)
            
        except Exception as e:
            self.logger.error(f"Error discovering APIs in {file_path}: {e}")
        
        return endpoints
    
    def _analyze_function_for_api(self, node: ast.FunctionDef, content: str, file_path: str) -> Optional[APIEndpoint]:
        """Analyze a function node for API endpoint characteristics"""
        
        # Extract function source
        function_lines = content.split('\n')[node.lineno-1:node.end_lineno]
        function_source = '\n'.join(function_lines)
        
        # Check for API decorators
        api_info = self._extract_api_info_from_decorators(node.decorator_list, function_source)
        
        if not api_info:
            # Check if function name suggests it's an API
            if self._is_likely_api_function(node.name, function_source):
                api_info = self._infer_api_info(node, function_source)
        
        if not api_info:
            return None
        
        # Create endpoint
        endpoint = APIEndpoint(
            path=api_info.get('path', f'/{node.name}'),
            method=HTTPMethod(api_info.get('method', 'GET')),
            function_name=node.name,
            description=ast.get_docstring(node) or f"API endpoint: {node.name}",
            module_name=self._extract_module_name(file_path),
            file_path=file_path
        )
        
        # Extract parameters
        endpoint.parameters = self._extract_api_parameters(node, function_source)
        
        # Extract responses
        endpoint.responses = self._extract_api_responses(node, function_source)
        
        # Extract metadata
        endpoint.authentication_required = self._has_auth_requirement(function_source)
        endpoint.rate_limited = self._has_rate_limiting(function_source)
        endpoint.deprecated = self._is_deprecated(function_source)
        endpoint.tags = self._extract_tags(function_source)
        
        # Generate examples
        endpoint.examples = self._generate_api_examples(endpoint)
        
        return endpoint
    
    def _extract_api_info_from_decorators(self, decorators: List[ast.expr], source: str) -> Optional[Dict[str, str]]:
        """Extract API information from function decorators"""
        
        for decorator in decorators:
            if isinstance(decorator, ast.Call):
                if hasattr(decorator.func, 'attr'):
                    decorator_name = decorator.func.attr
                elif hasattr(decorator.func, 'id'):
                    decorator_name = decorator.func.id
                else:
                    continue
                
                # Check for Flask routes
                if decorator_name == 'route':
                    return self._parse_flask_route(decorator)
                
                # Check for FastAPI routes
                elif decorator_name in ['get', 'post', 'put', 'delete', 'patch']:
                    return self._parse_fastapi_route(decorator, decorator_name)
                
                # Check for WebSocket endpoints
                elif decorator_name == 'websocket':
                    return self._parse_websocket_route(decorator)
        
        return None
    
    def _parse_flask_route(self, decorator: ast.Call) -> Dict[str, str]:
        """Parse Flask route decorator"""
        info = {'method': 'GET'}
        
        # Extract path
        if decorator.args:
            if isinstance(decorator.args[0], ast.Str):
                info['path'] = decorator.args[0].s
            elif isinstance(decorator.args[0], ast.Constant):
                info['path'] = decorator.args[0].value
        
        # Extract methods
        for keyword in decorator.keywords:
            if keyword.arg == 'methods':
                if isinstance(keyword.value, ast.List):
                    methods = []
                    for elt in keyword.value.elts:
                        if isinstance(elt, ast.Str):
                            methods.append(elt.s)
                        elif isinstance(elt, ast.Constant):
                            methods.append(elt.value)
                    if methods:
                        info['method'] = methods[0]  # Take first method
        
        return info
    
    def _parse_fastapi_route(self, decorator: ast.Call, method: str) -> Dict[str, str]:
        """Parse FastAPI route decorator"""
        info = {'method': method.upper()}
        
        # Extract path
        if decorator.args:
            if isinstance(decorator.args[0], ast.Str):
                info['path'] = decorator.args[0].s
            elif isinstance(decorator.args[0], ast.Constant):
                info['path'] = decorator.args[0].value
        
        return info
    
    def _parse_websocket_route(self, decorator: ast.Call) -> Dict[str, str]:
        """Parse WebSocket route decorator"""
        info = {'method': 'WEBSOCKET'}
        
        # Extract path
        if decorator.args:
            if isinstance(decorator.args[0], ast.Str):
                info['path'] = decorator.args[0].s
            elif isinstance(decorator.args[0], ast.Constant):
                info['path'] = decorator.args[0].value
        
        return info
    
    def _is_likely_api_function(self, func_name: str, source: str) -> bool:
        """Determine if function is likely an API endpoint"""
        
        api_indicators = [
            'api_', 'endpoint_', 'handle_', 'process_', 'get_', 'post_', 
            'put_', 'delete_', 'patch_', 'create_', 'update_', 'remove_'
        ]
        
        # Check function name
        if any(func_name.startswith(indicator) for indicator in api_indicators):
            return True
        
        # Check for HTTP method patterns in source
        http_patterns = ['request', 'response', 'json', 'status_code', 'abort', 'redirect']
        return any(pattern in source.lower() for pattern in http_patterns)
    
    def _infer_api_info(self, node: ast.FunctionDef, source: str) -> Dict[str, str]:
        """Infer API information from function characteristics"""
        
        func_name = node.name
        
        # Infer HTTP method from function name
        if func_name.startswith(('get_', 'fetch_', 'retrieve_', 'find_')):
            method = 'GET'
        elif func_name.startswith(('post_', 'create_', 'add_', 'insert_')):
            method = 'POST'
        elif func_name.startswith(('put_', 'update_', 'modify_', 'edit_')):
            method = 'PUT'
        elif func_name.startswith(('delete_', 'remove_', 'destroy_')):
            method = 'DELETE'
        elif func_name.startswith(('patch_', 'partial_')):
            method = 'PATCH'
        else:
            method = 'POST'  # Default for unclear cases
        
        # Generate path from function name
        path = '/' + func_name.replace('_', '/')
        
        return {'method': method, 'path': path}
    
    def _extract_api_parameters(self, node: ast.FunctionDef, source: str) -> List[APIParameter]:
        """Extract API parameters from function signature and docstring"""
        parameters = []
        
        # Extract from function arguments
        for arg in node.args.args:
            if arg.arg in ['self', 'cls']:
                continue
            
            param_type = self._infer_parameter_type(arg.arg, source)
            data_type = self._infer_data_type(arg, source)
            
            param = APIParameter(
                name=arg.arg,
                param_type=param_type,
                data_type=data_type,
                description=f"Parameter: {arg.arg}"
            )
            
            parameters.append(param)
        
        # Extract from docstring
        docstring_params = self._extract_params_from_docstring(ast.get_docstring(node))
        
        # Merge information
        for param in parameters:
            for doc_param in docstring_params:
                if doc_param['name'] == param.name:
                    param.description = doc_param.get('description', param.description)
                    param.data_type = doc_param.get('type', param.data_type)
                    break
        
        return parameters
    
    def _extract_api_responses(self, node: ast.FunctionDef, source: str) -> List[APIResponse]:
        """Extract API response information"""
        responses = []
        
        # Default success response
        responses.append(APIResponse(
            status_code=200,
            description="Successful response",
            content_type="application/json"
        ))
        
        # Extract from docstring
        docstring = ast.get_docstring(node)
        if docstring:
            response_info = self._extract_responses_from_docstring(docstring)
            responses.extend(response_info)
        
        # Analyze function body for return statements
        return_analysis = self._analyze_return_statements(node)
        if return_analysis:
            responses[0].schema = return_analysis
        
        return responses
    
    def _infer_parameter_type(self, param_name: str, source: str) -> ParameterType:
        """Infer parameter type from name and context"""
        
        if param_name in ['id', 'user_id', 'item_id']:
            return ParameterType.PATH
        elif param_name in ['data', 'payload', 'body']:
            return ParameterType.BODY
        elif 'request.headers' in source and param_name in source:
            return ParameterType.HEADER
        else:
            return ParameterType.QUERY
    
    def _infer_data_type(self, arg: ast.arg, source: str) -> str:
        """Infer data type from annotation or context"""
        
        if arg.annotation:
            if hasattr(arg.annotation, 'id'):
                return arg.annotation.id
            elif hasattr(arg.annotation, 'attr'):
                return arg.annotation.attr
        
        # Infer from parameter name
        if arg.arg.endswith('_id'):
            return 'integer'
        elif arg.arg.endswith('_count'):
            return 'integer'
        elif arg.arg.endswith('_flag'):
            return 'boolean'
        else:
            return 'string'
    
    def _extract_params_from_docstring(self, docstring: Optional[str]) -> List[Dict[str, str]]:
        """Extract parameter information from docstring"""
        params = []
        
        if not docstring:
            return params
        
        # Look for various docstring formats
        param_patterns = [
            re.compile(r'Args?:\s*\n(.+?)(?:\n\s*\n|\n\s*Returns?:|$)', re.DOTALL | re.IGNORECASE),
            re.compile(r'Parameters?:\s*\n(.+?)(?:\n\s*\n|\n\s*Returns?:|$)', re.DOTALL | re.IGNORECASE),
            re.compile(r':param\s+(\w+):\s*(.+)', re.IGNORECASE),
            re.compile(r'@param\s+(\w+)\s+(.+)', re.IGNORECASE)
        ]
        
        for pattern in param_patterns:
            matches = pattern.findall(docstring)
            for match in matches:
                if isinstance(match, tuple) and len(match) == 2:
                    name, description = match
                    params.append({'name': name.strip(), 'description': description.strip()})
                elif isinstance(match, str):
                    # Parse Args/Parameters section
                    for line in match.split('\n'):
                        line = line.strip()
                        if ':' in line:
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                name = parts[0].strip()
                                description = parts[1].strip()
                                params.append({'name': name, 'description': description})
        
        return params
    
    def _extract_responses_from_docstring(self, docstring: str) -> List[APIResponse]:
        """Extract response information from docstring"""
        responses = []
        
        # Look for Returns section
        returns_pattern = re.compile(r'Returns?:\s*\n(.+?)(?:\n\s*\n|$)', re.DOTALL | re.IGNORECASE)
        match = returns_pattern.search(docstring)
        
        if match:
            returns_text = match.group(1)
            
            # Look for status codes
            status_pattern = re.compile(r'(\d{3})\s*:?\s*(.+)')
            status_matches = status_pattern.findall(returns_text)
            
            for status_code, description in status_matches:
                responses.append(APIResponse(
                    status_code=int(status_code),
                    description=description.strip()
                ))
        
        return responses
    
    def _analyze_return_statements(self, node: ast.FunctionDef) -> Optional[Dict[str, Any]]:
        """Analyze return statements to infer response schema"""
        
        return_info = {}
        
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value:
                if isinstance(child.value, ast.Dict):
                    # Dictionary return
                    return_info['type'] = 'object'
                    return_info['properties'] = self._analyze_dict_structure(child.value)
                elif isinstance(child.value, ast.List):
                    # List return
                    return_info['type'] = 'array'
                elif isinstance(child.value, ast.Str):
                    # String return
                    return_info['type'] = 'string'
                elif isinstance(child.value, ast.Num):
                    # Number return
                    return_info['type'] = 'number'
        
        return return_info if return_info else None
    
    def _analyze_dict_structure(self, dict_node: ast.Dict) -> Dict[str, Any]:
        """Analyze dictionary structure for schema generation"""
        properties = {}
        
        for key, value in zip(dict_node.keys, dict_node.values):
            if isinstance(key, ast.Str):
                key_name = key.s
            elif isinstance(key, ast.Constant):
                key_name = str(key.value)
            else:
                continue
            
            if isinstance(value, ast.Str):
                properties[key_name] = {'type': 'string'}
            elif isinstance(value, ast.Num):
                properties[key_name] = {'type': 'number'}
            elif isinstance(value, ast.List):
                properties[key_name] = {'type': 'array'}
            elif isinstance(value, ast.Dict):
                properties[key_name] = {
                    'type': 'object',
                    'properties': self._analyze_dict_structure(value)
                }
            else:
                properties[key_name] = {'type': 'unknown'}
        
        return properties
    
    def _has_auth_requirement(self, source: str) -> bool:
        """Check if endpoint requires authentication"""
        auth_patterns = ['@login_required', '@auth_required', '@requires_auth', 'token', 'authenticated']
        return any(pattern in source for pattern in auth_patterns)
    
    def _has_rate_limiting(self, source: str) -> bool:
        """Check if endpoint has rate limiting"""
        rate_patterns = ['@rate_limit', '@limiter', 'rate_limit', 'throttle']
        return any(pattern in source for pattern in rate_patterns)
    
    def _is_deprecated(self, source: str) -> bool:
        """Check if endpoint is deprecated"""
        deprecated_patterns = ['@deprecated', 'deprecated', 'DEPRECATED']
        return any(pattern in source for pattern in deprecated_patterns)
    
    def _extract_tags(self, source: str) -> List[str]:
        """Extract tags from source code"""
        tags = []
        
        # Look for tag decorators or comments
        tag_patterns = [
            r'@tag\([\'"]([^\'"]+)[\'"]',
            r'#\s*tag:\s*([^\n]+)',
            r'tags\s*=\s*\[([^\]]+)\]'
        ]
        
        for pattern in tag_patterns:
            matches = re.findall(pattern, source, re.IGNORECASE)
            for match in matches:
                if isinstance(match, str):
                    tags.extend([tag.strip().strip('"\'') for tag in match.split(',')])
        
        return tags
    
    def _generate_api_examples(self, endpoint: APIEndpoint) -> List[Dict[str, Any]]:
        """Generate example requests and responses for endpoint"""
        examples = []
        
        # Generate curl example
        curl_example = self._generate_curl_example(endpoint)
        if curl_example:
            examples.append(curl_example)
        
        # Generate Python requests example
        python_example = self._generate_python_example(endpoint)
        if python_example:
            examples.append(python_example)
        
        return examples
    
    def _generate_curl_example(self, endpoint: APIEndpoint) -> Dict[str, Any]:
        """Generate curl example for endpoint"""
        
        curl_parts = ['curl', '-X', endpoint.method.value]
        
        # Add headers if needed
        if endpoint.authentication_required:
            curl_parts.extend(['-H', '"Authorization: Bearer YOUR_TOKEN"'])
        
        curl_parts.extend(['-H', '"Content-Type: application/json"'])
        
        # Add body for POST/PUT/PATCH
        if endpoint.method in [HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.PATCH]:
            body_params = [p for p in endpoint.parameters if p.param_type == ParameterType.BODY]
            if body_params:
                example_body = {param.name: f"example_{param.name}" for param in body_params}
                curl_parts.extend(['-d', f"'{json.dumps(example_body)}'"])
        
        # Add URL with path parameters
        url = endpoint.path
        path_params = [p for p in endpoint.parameters if p.param_type == ParameterType.PATH]
        for param in path_params:
            url = url.replace(f'{{{param.name}}}', f'example_{param.name}')
        
        # Add query parameters
        query_params = [p for p in endpoint.parameters if p.param_type == ParameterType.QUERY]
        if query_params:
            query_string = '&'.join([f'{p.name}=example_{p.name}' for p in query_params])
            url += f'?{query_string}'
        
        curl_parts.append(f'"http://localhost:8000{url}"')
        
        return {
            'language': 'bash',
            'title': 'cURL Example',
            'code': ' '.join(curl_parts)
        }
    
    def _generate_python_example(self, endpoint: APIEndpoint) -> Dict[str, Any]:
        """Generate Python requests example for endpoint"""
        
        code_lines = [
            'import requests',
            '',
            f'url = "http://localhost:8000{endpoint.path}"'
        ]
        
        # Add headers
        headers = ['headers = {', '    "Content-Type": "application/json"']
        if endpoint.authentication_required:
            headers.append('    "Authorization": "Bearer YOUR_TOKEN"')
        headers.append('}')
        code_lines.extend(headers)
        
        # Add parameters
        if endpoint.parameters:
            body_params = [p for p in endpoint.parameters if p.param_type == ParameterType.BODY]
            query_params = [p for p in endpoint.parameters if p.param_type == ParameterType.QUERY]
            
            if body_params:
                code_lines.extend([
                    '',
                    'data = {',
                    *[f'    "{param.name}": "example_{param.name}",' for param in body_params],
                    '}'
                ])
            
            if query_params:
                code_lines.extend([
                    '',
                    'params = {',
                    *[f'    "{param.name}": "example_{param.name}",' for param in query_params],
                    '}'
                ])
        
        # Add request
        request_args = ['url', 'headers=headers']
        
        if endpoint.method in [HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.PATCH]:
            request_args.append('json=data')
        
        if [p for p in endpoint.parameters if p.param_type == ParameterType.QUERY]:
            request_args.append('params=params')
        
        code_lines.extend([
            '',
            f'response = requests.{endpoint.method.value.lower()}({", ".join(request_args)})',
            'print(response.json())'
        ])
        
        return {
            'language': 'python',
            'title': 'Python Example',
            'code': '\n'.join(code_lines)
        }
    
    def _discover_with_patterns(self, content: str, file_path: str) -> List[APIEndpoint]:
        """Discover endpoints using regex patterns"""
        endpoints = []
        
        for pattern_name, pattern in self.api_patterns.items():
            matches = pattern.findall(content)
            
            for match in matches:
                if pattern_name == 'flask_route':
                    path, methods_str = match if isinstance(match, tuple) else (match, None)
                    method = self._parse_methods_string(methods_str) or 'GET'
                    
                    endpoint = APIEndpoint(
                        path=path,
                        method=HTTPMethod(method),
                        function_name=f"handler_{path.replace('/', '_').strip('_')}",
                        description=f"Flask route: {path}",
                        module_name=self._extract_module_name(file_path),
                        file_path=file_path
                    )
                    endpoints.append(endpoint)
                
                elif pattern_name == 'fastapi_route':
                    method, path = match if isinstance(match, tuple) else ('get', match)
                    
                    endpoint = APIEndpoint(
                        path=path,
                        method=HTTPMethod(method.upper()),
                        function_name=f"handler_{path.replace('/', '_').strip('_')}",
                        description=f"FastAPI route: {path}",
                        module_name=self._extract_module_name(file_path),
                        file_path=file_path
                    )
                    endpoints.append(endpoint)
        
        return endpoints
    
    def _parse_methods_string(self, methods_str: Optional[str]) -> Optional[str]:
        """Parse methods string from route decorator"""
        if not methods_str:
            return None
        
        # Remove quotes and brackets, split by comma
        methods = methods_str.strip('[]').replace('"', '').replace("'", "")
        method_list = [m.strip() for m in methods.split(',')]
        
        return method_list[0] if method_list else None
    
    def _extract_module_name(self, file_path: str) -> str:
        """Extract module name from file path"""
        path = Path(file_path)
        if 'independent_core' in path.parts:
            start_idx = path.parts.index('independent_core')
            module_parts = path.parts[start_idx:]
            return '.'.join(module_parts).replace('.py', '')
        return path.stem

class OpenAPIGenerator:
    """Generate OpenAPI/Swagger documentation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_openapi_spec(self, api_doc: APIDocumentation) -> Dict[str, Any]:
        """Generate complete OpenAPI 3.0 specification"""
        
        spec = {
            'openapi': '3.0.0',
            'info': {
                'title': api_doc.title,
                'description': api_doc.description,
                'version': api_doc.version
            },
            'servers': [
                {'url': api_doc.base_url, 'description': 'Development server'}
            ],
            'paths': {},
            'components': {
                'schemas': api_doc.data_models,
                'securitySchemes': self._generate_security_schemes(api_doc.authentication)
            }
        }
        
        # Generate paths
        for endpoint in api_doc.endpoints:
            path_spec = self._generate_path_spec(endpoint)
            
            if endpoint.path not in spec['paths']:
                spec['paths'][endpoint.path] = {}
            
            spec['paths'][endpoint.path][endpoint.method.value.lower()] = path_spec
        
        return spec
    
    def _generate_path_spec(self, endpoint: APIEndpoint) -> Dict[str, Any]:
        """Generate OpenAPI path specification for endpoint"""
        
        spec = {
            'summary': endpoint.description,
            'description': endpoint.description,
            'operationId': endpoint.function_name,
            'tags': endpoint.tags or ['default'],
            'parameters': self._generate_parameters_spec(endpoint.parameters),
            'responses': self._generate_responses_spec(endpoint.responses)
        }
        
        # Add request body for POST/PUT/PATCH
        if endpoint.method in [HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.PATCH]:
            body_params = [p for p in endpoint.parameters if p.param_type == ParameterType.BODY]
            if body_params:
                spec['requestBody'] = self._generate_request_body_spec(body_params)
        
        # Add security if required
        if endpoint.authentication_required:
            spec['security'] = [{'bearerAuth': []}]
        
        # Add deprecation warning
        if endpoint.deprecated:
            spec['deprecated'] = True
        
        return spec
    
    def _generate_parameters_spec(self, parameters: List[APIParameter]) -> List[Dict[str, Any]]:
        """Generate OpenAPI parameters specification"""
        
        specs = []
        
        for param in parameters:
            if param.param_type == ParameterType.BODY:
                continue  # Body parameters handled separately
            
            param_spec = {
                'name': param.name,
                'in': param.param_type.value,
                'description': param.description,
                'required': param.required,
                'schema': {
                    'type': param.data_type
                }
            }
            
            if param.default_value is not None:
                param_spec['schema']['default'] = param.default_value
            
            if param.examples:
                param_spec['examples'] = {
                    f'example_{i}': {'value': example} 
                    for i, example in enumerate(param.examples, 1)
                }
            
            specs.append(param_spec)
        
        return specs
    
    def _generate_request_body_spec(self, body_params: List[APIParameter]) -> Dict[str, Any]:
        """Generate OpenAPI request body specification"""
        
        properties = {}
        required = []
        
        for param in body_params:
            properties[param.name] = {
                'type': param.data_type,
                'description': param.description
            }
            
            if param.required:
                required.append(param.name)
        
        return {
            'required': True,
            'content': {
                'application/json': {
                    'schema': {
                        'type': 'object',
                        'properties': properties,
                        'required': required
                    }
                }
            }
        }
    
    def _generate_responses_spec(self, responses: List[APIResponse]) -> Dict[str, Any]:
        """Generate OpenAPI responses specification"""
        
        specs = {}
        
        for response in responses:
            response_spec = {
                'description': response.description
            }
            
            if response.schema:
                response_spec['content'] = {
                    response.content_type: {
                        'schema': response.schema
                    }
                }
            
            if response.examples:
                if 'content' not in response_spec:
                    response_spec['content'] = {response.content_type: {}}
                
                response_spec['content'][response.content_type]['examples'] = {
                    f'example_{i}': {'value': example}
                    for i, example in enumerate(response.examples, 1)
                }
            
            specs[str(response.status_code)] = response_spec
        
        return specs
    
    def _generate_security_schemes(self, auth_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate OpenAPI security schemes"""
        
        schemes = {}
        
        # Default bearer token authentication
        schemes['bearerAuth'] = {
            'type': 'http',
            'scheme': 'bearer',
            'bearerFormat': 'JWT'
        }
        
        # Add custom authentication schemes from config
        for scheme_name, scheme_config in auth_config.items():
            schemes[scheme_name] = scheme_config
        
        return schemes

class APIDocumentationGenerator:
    """Main API documentation generator"""
    
    def __init__(self):
        self.discovery = APIDiscovery()
        self.openapi_generator = OpenAPIGenerator()
        self.logger = logging.getLogger(__name__)
    
    async def generate_api_documentation(self, target: Any) -> str:
        """Generate comprehensive API documentation for a target"""
        
        try:
            # Discover API endpoints
            endpoints = self.discovery.discover_apis_in_file(target.file_path)
            
            if not endpoints:
                return f"No API endpoints found in {target.module_name}"
            
            # Create API documentation structure
            api_doc = APIDocumentation(
                title=f"API Documentation: {target.module_name}",
                description=f"API endpoints for {target.module_name}",
                version="1.0.0",
                base_url="http://localhost:8000",
                endpoints=endpoints
            )
            
            # Generate comprehensive documentation
            documentation = self._generate_comprehensive_api_docs(api_doc)
            
            return documentation
            
        except Exception as e:
            self.logger.error(f"Failed to generate API documentation for {target.module_name}: {e}")
            return f"Error generating API documentation: {str(e)}"
    
    def _generate_comprehensive_api_docs(self, api_doc: APIDocumentation) -> str:
        """Generate comprehensive API documentation"""
        
        doc_parts = [
            f"# {api_doc.title}",
            "",
            api_doc.description,
            "",
            f"**Version:** {api_doc.version}",
            f"**Base URL:** {api_doc.base_url}",
            "",
            "## Endpoints",
            ""
        ]
        
        # Group endpoints by path
        endpoints_by_path = {}
        for endpoint in api_doc.endpoints:
            if endpoint.path not in endpoints_by_path:
                endpoints_by_path[endpoint.path] = []
            endpoints_by_path[endpoint.path].append(endpoint)
        
        # Generate documentation for each path
        for path in sorted(endpoints_by_path.keys()):
            doc_parts.extend([f"### {path}", ""])
            
            for endpoint in endpoints_by_path[path]:
                doc_parts.extend(self._generate_endpoint_docs(endpoint))
                doc_parts.append("")
        
        # Add OpenAPI specification
        openapi_spec = self.openapi_generator.generate_openapi_spec(api_doc)
        doc_parts.extend([
            "## OpenAPI Specification",
            "",
            "```json",
            json.dumps(openapi_spec, indent=2),
            "```"
        ])
        
        return "\n".join(doc_parts)
    
    def _generate_endpoint_docs(self, endpoint: APIEndpoint) -> List[str]:
        """Generate documentation for a single endpoint"""
        
        docs = [
            f"#### {endpoint.method.value} {endpoint.path}",
            "",
            endpoint.description,
            ""
        ]
        
        # Add metadata
        metadata = []
        if endpoint.authentication_required:
            metadata.append("🔒 Authentication required")
        if endpoint.rate_limited:
            metadata.append("⏱️ Rate limited")
        if endpoint.deprecated:
            metadata.append("⚠️ Deprecated")
        
        if metadata:
            docs.extend(metadata)
            docs.append("")
        
        # Add parameters
        if endpoint.parameters:
            docs.extend(["**Parameters:**", ""])
            
            for param in endpoint.parameters:
                param_doc = f"- **{param.name}** ({param.param_type.value})"
                if param.required:
                    param_doc += " *required*"
                param_doc += f": {param.data_type} - {param.description}"
                docs.append(param_doc)
            
            docs.append("")
        
        # Add responses
        if endpoint.responses:
            docs.extend(["**Responses:**", ""])
            
            for response in endpoint.responses:
                docs.append(f"- **{response.status_code}**: {response.description}")
            
            docs.append("")
        
        # Add examples
        if endpoint.examples:
            docs.extend(["**Examples:**", ""])
            
            for example in endpoint.examples:
                docs.extend([
                    f"##### {example['title']}",
                    "",
                    f"```{example['language']}",
                    example['code'],
                    "```",
                    ""
                ])
        
        return docs
    
    async def discover_all_apis(self, root_path: str) -> APIDocumentation:
        """Discover all APIs in the codebase"""
        
        all_endpoints = []
        
        # Walk through all Python files
        for py_file in Path(root_path).rglob("*.py"):
            if py_file.name.startswith("__"):
                continue
            
            try:
                endpoints = self.discovery.discover_apis_in_file(str(py_file))
                all_endpoints.extend(endpoints)
            except Exception as e:
                self.logger.error(f"Error discovering APIs in {py_file}: {e}")
        
        # Create comprehensive API documentation
        api_doc = APIDocumentation(
            title="Saraphis Independent Core API",
            description="Comprehensive API documentation for all endpoints in Saraphis Independent Core",
            version="1.0.0",
            base_url="http://localhost:8000",
            endpoints=all_endpoints
        )
        
        return api_doc
    
    async def generate_interactive_docs(self, api_doc: APIDocumentation, output_path: str):
        """Generate interactive API documentation (Swagger UI style)"""
        
        openapi_spec = self.openapi_generator.generate_openapi_spec(api_doc)
        
        # Generate HTML with Swagger UI
        swagger_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{api_doc.title}</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@3.25.0/swagger-ui.css" />
    <style>
        html {{ box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }}
        *, *:before, *:after {{ box-sizing: inherit; }}
        body {{ margin: 0; background: #fafafa; }}
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@3.25.0/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@3.25.0/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {{
            const ui = SwaggerUIBundle({{
                url: 'data:application/json;base64,' + btoa(JSON.stringify({json.dumps(openapi_spec)})),
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout"
            }});
        }};
    </script>
</body>
</html>
"""
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(swagger_html)

if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Test API documentation generator
        generator = APIDocumentationGenerator()
        
        # Test with a sample file
        test_file = "/home/will-casterlin/Desktop/Saraphis/independent_core/brain.py"
        
        class MockTarget:
            def __init__(self, file_path: str, module_name: str):
                self.file_path = file_path
                self.module_name = module_name
        
        target = MockTarget(test_file, "independent_core.brain")
        
        # Generate API documentation
        api_docs = await generator.generate_api_documentation(target)
        print("API Documentation Generated:")
        print(api_docs[:1000] + "..." if len(api_docs) > 1000 else api_docs)
        
        # Discover all APIs
        api_doc = await generator.discover_all_apis("/home/will-casterlin/Desktop/Saraphis/independent_core")
        print(f"\nDiscovered {len(api_doc.endpoints)} API endpoints")
        
        # Generate interactive documentation
        await generator.generate_interactive_docs(api_doc, "docs/api.html")
        print("Interactive API documentation generated at docs/api.html")
    
    asyncio.run(main())