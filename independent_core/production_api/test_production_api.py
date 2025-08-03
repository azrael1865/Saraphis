#!/usr/bin/env python3
"""
Test script for Saraphis Production API Gateway & Load Balancer System
"""

import logging
import time
import json
import random
from datetime import datetime
import concurrent.futures
from typing import Dict, List, Any

# Import all production API components
from production_api import (
    APIGatewayManager,
    LoadBalancer,
    HealthChecker,
    RateLimiter,
    AuthenticationManager,
    RequestValidator,
    ResponseFormatter,
    APIMetricsCollector
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_api_gateway():
    """Test API Gateway functionality"""
    print("\n" + "="*50)
    print("Testing API Gateway")
    print("="*50)
    
    config = {
        'api_version': '1.0',
        'timeout_seconds': 30,
        'max_retries': 3,
        'circuit_breaker_threshold': 5,
        'load_balancer': {
            'algorithm': 'weighted_round_robin'
        },
        'rate_limiter': {
            'algorithm': 'token_bucket'
        },
        'authentication': {
            'require_authentication': True
        },
        'validation': {},
        'formatting': {
            'compression_enabled': True
        },
        'metrics': {}
    }
    
    gateway = APIGatewayManager(config)
    
    # Get authentication token
    auth_manager = gateway.auth_manager
    token_result = auth_manager.generate_token('user1', {'tier': 'pro'})
    access_token = token_result['access_token']
    
    print(f"\n1. Generated access token: {access_token[:20]}...")
    
    # Test successful request
    print("\n2. Testing successful request...")
    request = {
        'endpoint': '/brain/status',
        'method': 'GET',
        'headers': {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        },
        'client_ip': '192.168.1.100'
    }
    
    response = gateway.route_request(request)
    print(f"Response: {json.dumps(response, indent=2)}")
    
    # Test rate limiting
    print("\n3. Testing rate limiting...")
    for i in range(5):
        response = gateway.route_request(request)
        print(f"Request {i+1}: {'Success' if response.get('success') else 'Rate limited'}")
        time.sleep(0.1)
    
    # Test unauthorized request
    print("\n4. Testing unauthorized request...")
    unauthorized_request = request.copy()
    unauthorized_request['headers'] = {}
    response = gateway.route_request(unauthorized_request)
    print(f"Unauthorized response: {response.get('error', {}).get('message')}")
    
    # Test invalid endpoint
    print("\n5. Testing invalid endpoint...")
    invalid_request = request.copy()
    invalid_request['endpoint'] = '/invalid/endpoint'
    response = gateway.route_request(invalid_request)
    print(f"Invalid endpoint response: {response.get('error', {}).get('message')}")
    
    # Get gateway status
    print("\n6. Gateway status...")
    status = gateway.get_gateway_status()
    print(f"Status: {json.dumps(status, indent=2)}")


def test_load_balancer():
    """Test Load Balancer functionality"""
    print("\n" + "="*50)
    print("Testing Load Balancer")
    print("="*50)
    
    config = {
        'algorithm': 'weighted_round_robin',
        'sticky_sessions': True,
        'session_timeout': 3600,
        'health_checks': {
            'interval_seconds': 30,
            'timeout_seconds': 5,
            'unhealthy_threshold': 3,
            'healthy_threshold': 2
        }
    }
    
    load_balancer = LoadBalancer(config)
    
    # Test request distribution
    print("\n1. Testing request distribution...")
    service = 'brain_service'
    distribution = {}
    
    for i in range(20):
        request = {
            'client_ip': f'192.168.1.{i % 5}',
            'session_id': f'session_{i % 3}'
        }
        
        result = load_balancer.distribute_request(request, service)
        if result['success']:
            endpoint = result['endpoint']
            distribution[endpoint] = distribution.get(endpoint, 0) + 1
    
    print("Request distribution:")
    for endpoint, count in distribution.items():
        print(f"  {endpoint}: {count} requests")
    
    # Test health checking
    print("\n2. Testing health checks...")
    health_checker = load_balancer.health_checker
    
    for endpoint in list(distribution.keys())[:2]:
        health = health_checker.check_endpoint_health(endpoint)
        print(f"\nEndpoint: {endpoint}")
        print(f"  Healthy: {health['healthy']}")
        print(f"  Response time: {health['response_time']:.3f}s")
        print(f"  Error rate: {health['error_rate']:.2%}")
    
    # Test different algorithms
    print("\n3. Testing load balancing algorithms...")
    algorithms = ['round_robin', 'weighted_round_robin', 'least_connections', 'ip_hash']
    
    for algo in algorithms:
        load_balancer.algorithm = algo
        result = load_balancer.distribute_request({'client_ip': '192.168.1.1'}, service)
        if result['success']:
            print(f"{algo}: {result['endpoint']}")
    
    # Get load balancer status
    print("\n4. Load balancer status...")
    status = load_balancer.get_load_balancer_status()
    print(f"Algorithm: {status['algorithm']}")
    print(f"Services: {len(status['services'])}")
    for service_name, service_status in list(status['services'].items())[:1]:
        print(f"\n{service_name}:")
        print(f"  Total endpoints: {service_status['total_endpoints']}")
        print(f"  Available endpoints: {service_status['available_endpoints']}")


def test_rate_limiter():
    """Test Rate Limiter functionality"""
    print("\n" + "="*50)
    print("Testing Rate Limiter")
    print("="*50)
    
    config = {
        'algorithm': 'token_bucket',
        'global_rate_limit': 10000,
        'burst_multiplier': 2.0
    }
    
    rate_limiter = RateLimiter(config)
    
    # Test basic rate limiting
    print("\n1. Testing token bucket rate limiting...")
    request = {
        'endpoint': '/brain/predict',
        'method': 'POST',
        'user': {'user_id': 'test_user', 'tier': 'free'},
        'client_ip': '192.168.1.100'
    }
    
    allowed_count = 0
    for i in range(150):
        result = rate_limiter.check_rate_limits(request)
        if result['allowed']:
            allowed_count += 1
        if i % 50 == 49:
            print(f"After {i+1} requests: {allowed_count} allowed")
    
    # Test different algorithms
    print("\n2. Testing different rate limiting algorithms...")
    algorithms = ['token_bucket', 'leaky_bucket', 'fixed_window', 'sliding_window']
    
    for algo in algorithms:
        rate_limiter.algorithm = algo
        
        # Make 10 rapid requests
        allowed = 0
        for _ in range(10):
            result = rate_limiter.check_rate_limits(request)
            if result['allowed']:
                allowed += 1
        
        print(f"{algo}: {allowed}/10 requests allowed")
    
    # Test tier-based limits
    print("\n3. Testing tier-based rate limits...")
    tiers = ['free', 'basic', 'pro', 'enterprise']
    
    for tier in tiers:
        tier_request = request.copy()
        tier_request['user'] = {'user_id': f'{tier}_user', 'tier': tier}
        
        allowed = 0
        for _ in range(100):
            result = rate_limiter.check_rate_limits(tier_request)
            if result['allowed']:
                allowed += 1
        
        print(f"{tier} tier: {allowed}/100 requests allowed")
    
    # Get rate limiter status
    print("\n4. Rate limiter status...")
    status = rate_limiter.get_rate_limiter_status()
    print(f"Algorithm: {status['algorithm']}")
    print(f"Active clients: {status['active_clients']}")
    print(f"Throttled clients: {status['throttled_clients']}")


def test_authentication():
    """Test Authentication system"""
    print("\n" + "="*50)
    print("Testing Authentication System")
    print("="*50)
    
    config = {
        'require_authentication': True,
        'allow_api_keys': True,
        'session_timeout': 3600,
        'tokens': {
            'token_expiry_seconds': 3600,
            'refresh_expiry_seconds': 86400
        }
    }
    
    auth_manager = AuthenticationManager(config)
    
    # Test token generation
    print("\n1. Testing token generation...")
    token_result = auth_manager.generate_token('user1', {'source': 'test'})
    print(f"Access token: {token_result['access_token'][:20]}...")
    print(f"Refresh token: {token_result['refresh_token'][:20]}...")
    print(f"Expires in: {token_result['expires_in']} seconds")
    
    # Test token authentication
    print("\n2. Testing token authentication...")
    request = {
        'headers': {
            'Authorization': f"Bearer {token_result['access_token']}"
        }
    }
    
    auth_result = auth_manager.authenticate_request(request)
    print(f"Authenticated: {auth_result['authenticated']}")
    print(f"User: {auth_result.get('user', {}).get('user_id')}")
    
    # Test API key authentication
    print("\n3. Testing API key authentication...")
    api_key_request = {
        'headers': {
            'X-API-Key': 'sk_user_' + 'a' * 22  # Simulated API key
        }
    }
    
    auth_result = auth_manager.authenticate_request(api_key_request)
    print(f"API key authenticated: {auth_result['authenticated']}")
    
    # Test authorization
    print("\n4. Testing authorization...")
    endpoints = [
        ('/brain/status', 'GET'),
        ('/brain/predict', 'POST'),
        ('/training/start', 'POST'),
        ('/security/config', 'PUT')
    ]
    
    user = {'user_id': 'user1', 'role': 'user'}
    
    for endpoint, method in endpoints:
        request = {'endpoint': endpoint, 'method': method}
        authz_result = auth_manager.authorize_request(request, user)
        print(f"{method} {endpoint}: {'Authorized' if authz_result['authorized'] else 'Denied'}")
    
    # Test token refresh
    print("\n5. Testing token refresh...")
    refresh_result = auth_manager.refresh_token(token_result['refresh_token'])
    print(f"Refresh successful: {refresh_result.get('success', False)}")
    
    # Get authentication metrics
    print("\n6. Authentication metrics...")
    metrics = auth_manager.get_auth_metrics()
    print(f"Metrics: {json.dumps(metrics, indent=2)}")


def test_request_validation():
    """Test Request Validation"""
    print("\n" + "="*50)
    print("Testing Request Validation")
    print("="*50)
    
    config = {
        'max_body_size': 10 * 1024 * 1024,
        'max_header_size': 8192,
        'max_url_length': 2048,
        'sanitization': {
            'max_string_length': 10000,
            'allow_html': False,
            'allow_sql': False
        }
    }
    
    validator = RequestValidator(config)
    
    # Test valid request
    print("\n1. Testing valid request...")
    valid_request = {
        'endpoint': '/brain/predict',
        'method': 'POST',
        'headers': {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer token123'
        },
        'body': {
            'input_data': [1, 2, 3, 4, 5],
            'model_id': 'model_v1',
            'confidence_threshold': 0.9
        }
    }
    
    result = validator.validate_request_format(valid_request)
    print(f"Valid: {result['valid']}")
    if not result['valid']:
        print(f"Errors: {result['errors']}")
    
    # Test invalid requests
    print("\n2. Testing invalid requests...")
    invalid_requests = [
        {
            'name': 'Missing endpoint',
            'request': {'method': 'GET'}
        },
        {
            'name': 'Invalid method',
            'request': {'endpoint': '/test', 'method': 'INVALID'}
        },
        {
            'name': 'Path traversal',
            'request': {'endpoint': '/../etc/passwd', 'method': 'GET'}
        }
    ]
    
    for test in invalid_requests:
        result = validator.validate_request_format(test['request'])
        print(f"\n{test['name']}:")
        print(f"  Valid: {result['valid']}")
        print(f"  Errors: {result['errors'][0] if result['errors'] else 'None'}")
    
    # Test input sanitization
    print("\n3. Testing input sanitization...")
    malicious_request = {
        'endpoint': '/test',
        'method': 'POST',
        'parameters': {
            'sql': "'; DROP TABLE users; --",
            'xss': '<script>alert("XSS")</script>',
            'cmd': 'ls -la; rm -rf /'
        },
        'body': {
            'data': 'normal data',
            'nested': {
                'script': '<script>evil()</script>'
            }
        }
    }
    
    result = validator.validate_request_content(malicious_request)
    print(f"Sanitization performed: {result['valid']}")
    if 'sanitized_request' in result:
        print("Sanitized parameters:")
        for key, value in result['sanitized_request'].get('parameters', {}).items():
            print(f"  {key}: {value}")
    
    # Get validation metrics
    print("\n4. Validation metrics...")
    metrics = validator.get_validation_metrics()
    print(f"Total validations: {metrics['total_validations']}")
    print(f"Success rate: {metrics['success_rate']:.2%}")


def test_response_formatting():
    """Test Response Formatting"""
    print("\n" + "="*50)
    print("Testing Response Formatting")
    print("="*50)
    
    config = {
        'api_version': '1.0',
        'compression_enabled': True,
        'compression_threshold': 100,
        'cache_enabled': True,
        'pretty_print': False
    }
    
    formatter = ResponseFormatter(config)
    
    # Test basic formatting
    print("\n1. Testing basic response formatting...")
    data = {
        'status': 'success',
        'result': {
            'prediction': 0.95,
            'confidence': 0.87,
            'model': 'v2.1'
        }
    }
    
    formatted = formatter.format_response(data)
    print(f"Success: {formatted['success']}")
    print(f"API Version: {formatted['api_version']}")
    print(f"Has metadata: {'_metadata' in formatted}")
    
    # Test compression
    print("\n2. Testing response compression...")
    large_data = {
        'data': 'x' * 1000,  # 1KB of data
        'items': list(range(100))
    }
    
    compressed_response = formatter.format_response(large_data)
    if '_compressed' in compressed_response and compressed_response['_compressed']:
        comp_info = compressed_response['_compression']
        print(f"Compressed: Yes")
        print(f"Algorithm: {comp_info['algorithm']}")
        print(f"Original size: {comp_info['original_size']} bytes")
        print(f"Compressed size: {comp_info['compressed_size']} bytes")
        print(f"Compression ratio: {comp_info['ratio']:.2f}")
    else:
        print("Not compressed (below threshold)")
    
    # Test error formatting
    print("\n3. Testing error response formatting...")
    error_response = formatter.format_error(
        status_code=404,
        message='Resource not found',
        details={'resource': 'model_v3', 'request_id': 'req_12345'}
    )
    
    print(f"Error code: {error_response['error']['code']}")
    print(f"Error message: {error_response['error']['message']}")
    
    # Test cache headers
    print("\n4. Testing cache header generation...")
    cacheable_data = {
        'cacheable': True,
        'data': 'static content'
    }
    
    cached_response = formatter.format_response(cacheable_data)
    if '_cache_headers' in cached_response:
        print("Cache headers:")
        for header, value in cached_response['_cache_headers'].items():
            print(f"  {header}: {value}")
    
    # Get formatting metrics
    print("\n5. Formatting metrics...")
    metrics = formatter.get_formatting_metrics()
    print(f"Total responses: {metrics['total_responses']}")
    print(f"Compression rate: {metrics['compression_rate']:.2%}")
    print(f"Average compression ratio: {metrics['average_compression_ratio']:.2f}")


def test_api_metrics():
    """Test API Metrics Collection"""
    print("\n" + "="*50)
    print("Testing API Metrics Collection")
    print("="*50)
    
    config = {
        'retention_period_hours': 24,
        'aggregation_intervals': [1, 5, 15, 60],
        'percentiles': [50, 90, 95, 99]
    }
    
    metrics_collector = APIMetricsCollector(config)
    
    # Simulate API traffic
    print("\n1. Simulating API traffic...")
    endpoints = ['/brain/predict', '/training/status', '/compression/compress', '/api/health']
    users = ['user1', 'user2', 'user3', 'admin']
    
    for i in range(100):
        request = {
            'endpoint': random.choice(endpoints),
            'method': random.choice(['GET', 'POST']),
            'user': {'user_id': random.choice(users)},
            'client_ip': f'192.168.1.{random.randint(1, 100)}'
        }
        
        response = {
            'success': random.random() > 0.05,  # 95% success rate
            'data': {'result': 'test'}
        }
        
        processing_time = random.uniform(0.01, 0.5)
        
        metrics_collector.track_request_metrics(
            request_id=f'req_{i}',
            request=request,
            response=response,
            processing_time=processing_time
        )
    
    # Wait for metrics to be processed
    time.sleep(1)
    
    # Get comprehensive metrics
    print("\n2. API metrics summary...")
    metrics = metrics_collector.get_api_metrics()
    
    summary = metrics.get('summary', {})
    print(f"Total requests: {summary.get('total_requests', 0)}")
    print(f"Error rate: {summary.get('error_rate', 0):.2%}")
    print(f"Average response time: {summary.get('average_response_time', 0):.3f}s")
    
    # Get endpoint metrics
    print("\n3. Top endpoints by request count...")
    endpoint_metrics = metrics.get('endpoint_metrics', {})
    for endpoint, stats in list(endpoint_metrics.items())[:3]:
        print(f"\n{endpoint}:")
        print(f"  Requests: {stats['requests']}")
        print(f"  Average time: {stats['average_time']:.3f}s")
    
    # Get performance metrics
    print("\n4. Performance metrics...")
    perf_metrics = metrics.get('performance_metrics', {})
    if 'response_time_percentiles' in perf_metrics:
        print("Response time percentiles:")
        for percentile, value in perf_metrics['response_time_percentiles'].items():
            print(f"  {percentile}: {value:.3f}s")
    
    # Check for alerts
    print("\n5. Active alerts...")
    alerts = metrics.get('alerts', [])
    if alerts:
        for alert in alerts:
            print(f"- {alert['type']}: {alert.get('details', {})}")
    else:
        print("No active alerts")


def test_load_testing():
    """Perform load testing on the API Gateway"""
    print("\n" + "="*50)
    print("Load Testing API Gateway")
    print("="*50)
    
    # Initialize gateway
    gateway_config = {
        'api_version': '1.0',
        'authentication': {
            'require_authentication': False  # Disable for load test
        }
    }
    
    gateway = APIGatewayManager(gateway_config)
    
    print("\n1. Performing concurrent load test...")
    
    def make_request(i):
        """Make a single request"""
        request = {
            'endpoint': random.choice(['/brain/predict', '/training/status', '/api/health']),
            'method': 'GET',
            'client_ip': f'192.168.1.{i % 100}'
        }
        
        start_time = time.time()
        response = gateway.route_request(request)
        duration = time.time() - start_time
        
        return {
            'success': response.get('success', False),
            'duration': duration
        }
    
    # Run concurrent requests
    num_requests = 100
    num_workers = 10
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(make_request, i) for i in range(num_requests)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful = sum(1 for r in results if r['success'])
    durations = [r['duration'] for r in results]
    
    print(f"\nLoad test results:")
    print(f"Total requests: {num_requests}")
    print(f"Successful: {successful} ({successful/num_requests*100:.1f}%)")
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests per second: {num_requests/total_time:.1f}")
    print(f"Average response time: {sum(durations)/len(durations):.3f}s")
    print(f"Min response time: {min(durations):.3f}s")
    print(f"Max response time: {max(durations):.3f}s")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("SARAPHIS PRODUCTION API GATEWAY & LOAD BALANCER SYSTEM TEST")
    print("="*70)
    
    try:
        test_api_gateway()
        test_load_balancer()
        test_rate_limiter()
        test_authentication()
        test_request_validation()
        test_response_formatting()
        test_api_metrics()
        test_load_testing()
        
        print("\n" + "="*70)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\nERROR: Test failed - {e}")


if __name__ == "__main__":
    main()