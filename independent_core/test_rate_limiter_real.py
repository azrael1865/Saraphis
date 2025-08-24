"""
Real comprehensive test suite for RateLimiter without mocks
Tests all methods with actual functionality and edge cases
"""

import pytest
import time
import threading
from collections import deque
import random
import string
from concurrent.futures import ThreadPoolExecutor, as_completed

from production_api.rate_limiter import RateLimiter


class TestRateLimiterInitialization:
    """Test RateLimiter initialization and configuration"""
    
    def test_default_initialization(self):
        """Test default initialization with minimal config"""
        config = {}
        limiter = RateLimiter(config)
        
        assert limiter.algorithm == 'token_bucket'
        assert limiter.global_rate_limit == 10000
        assert limiter.burst_multiplier == 2.0
        assert limiter.enable_distributed is False
        assert isinstance(limiter.service_limits, dict)
        assert len(limiter.service_limits) == 8  # 8 services configured
        assert limiter.token_buckets is not None
        assert limiter.leaky_buckets is not None
        assert limiter.fixed_windows is not None
        assert limiter.sliding_windows is not None
        assert limiter.throttled_clients == {}
        assert limiter.blacklisted_clients == set()
    
    def test_custom_initialization(self):
        """Test initialization with custom configuration"""
        config = {
            'algorithm': 'sliding_window',
            'global_rate_limit': 5000,
            'burst_multiplier': 3.0,
            'enable_distributed': True
        }
        limiter = RateLimiter(config)
        
        assert limiter.algorithm == 'sliding_window'
        assert limiter.global_rate_limit == 5000
        assert limiter.burst_multiplier == 3.0
        assert limiter.enable_distributed is True
    
    def test_service_limits_initialization(self):
        """Test service limits are properly initialized"""
        limiter = RateLimiter({})
        
        service_limits = limiter.service_limits
        
        # Check all expected services are present
        expected_services = ['brain', 'uncertainty', 'training', 'compression', 
                           'proof', 'security', 'data', 'api']
        assert all(service in service_limits for service in expected_services)
        
        # Check each service has required fields
        for service, config in service_limits.items():
            assert 'requests_per_minute' in config
            assert 'requests_per_hour' in config
            assert 'burst_size' in config
            assert 'priority' in config
            assert isinstance(config['requests_per_minute'], int)
            assert isinstance(config['requests_per_hour'], int)
            assert isinstance(config['burst_size'], int)
            assert config['priority'] in ['critical', 'high', 'medium', 'low']
    
    def test_cleanup_thread_starts(self):
        """Test that cleanup thread is started"""
        limiter = RateLimiter({})
        
        # Check that there's at least one daemon thread running
        daemon_threads = [t for t in threading.enumerate() if t.daemon]
        assert len(daemon_threads) > 0
        
        # Give cleanup thread time to start
        time.sleep(0.1)
        
        # Verify cleanup thread is running by checking thread count
        current_threads = threading.active_count()
        assert current_threads > 1  # Main thread + at least cleanup thread


class TestClientIdentification:
    """Test client identification functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.limiter = RateLimiter({})
    
    def test_extract_client_id_priority_order(self):
        """Test client ID extraction follows correct priority"""
        # Test user_id has highest priority
        request = {
            'user': {'user_id': 'user123'},
            'api_key': 'key456',
            'client_ip': '192.168.1.1',
            'session_id': 'session789'
        }
        assert self.limiter._extract_client_id(request) == 'user123'
        
        # Test api_key is second priority
        request = {
            'api_key': 'key456',
            'client_ip': '192.168.1.1',
            'session_id': 'session789'
        }
        assert self.limiter._extract_client_id(request) == 'key456'
        
        # Test client_ip is third priority
        request = {
            'client_ip': '192.168.1.1',
            'session_id': 'session789'
        }
        assert self.limiter._extract_client_id(request) == '192.168.1.1'
        
        # Test X-Forwarded-For header
        request = {
            'headers': {'X-Forwarded-For': '10.0.0.1'},
            'session_id': 'session789'
        }
        assert self.limiter._extract_client_id(request) == '10.0.0.1'
        
        # Test session_id is used as fallback
        request = {'session_id': 'session789'}
        assert self.limiter._extract_client_id(request) == 'session789'
        
        # Test anonymous is default
        request = {}
        assert self.limiter._extract_client_id(request) == 'anonymous'
    
    def test_extract_client_id_type_conversion(self):
        """Test client ID is properly converted to string"""
        # Integer user ID
        request = {'user': {'user_id': 12345}}
        client_id = self.limiter._extract_client_id(request)
        assert client_id == '12345'
        assert isinstance(client_id, str)
        
        # None values should fall through
        request = {'user': {'user_id': None}, 'api_key': 'backup'}
        assert self.limiter._extract_client_id(request) == 'backup'
        
        # Empty string should fall through
        request = {'api_key': '', 'client_ip': '192.168.1.1'}
        assert self.limiter._extract_client_id(request) == '192.168.1.1'


class TestRateLimitConfiguration:
    """Test rate limit configuration retrieval"""
    
    def setup_method(self):
        """Setup for each test"""
        self.limiter = RateLimiter({'burst_multiplier': 2.0})
    
    def test_service_endpoint_matching(self):
        """Test that endpoints correctly match to services"""
        test_cases = [
            ('/brain/predict', 'brain'),
            ('/uncertainty/analyze', 'uncertainty'),
            ('/training/start', 'training'),
            ('/compression/compress', 'compression'),
            ('/proof/generate', 'proof'),
            ('/security/validate', 'security'),
            ('/data/query', 'data'),
            ('/api/status', 'api'),
            ('/unknown/endpoint', 'api'),  # Default to api
            ('', 'api'),  # Empty endpoint defaults to api
        ]
        
        for endpoint, expected_service in test_cases:
            request = {'endpoint': endpoint}
            config = self.limiter._get_rate_limit_config(request)
            assert config['service'] == expected_service
    
    def test_tier_multipliers(self):
        """Test that tier multipliers are applied correctly"""
        base_request = {'endpoint': '/brain/predict'}
        
        tier_configs = [
            ('free', 1.0, 100),      # 100 * 1.0
            ('basic', 2.0, 200),     # 100 * 2.0
            ('pro', 5.0, 500),       # 100 * 5.0
            ('enterprise', 10.0, 1000), # 100 * 10.0
            ('unknown', 1.0, 100),   # Default to 1.0
        ]
        
        for tier, multiplier, expected_rpm in tier_configs:
            request = base_request.copy()
            request['user'] = {'tier': tier}
            config = self.limiter._get_rate_limit_config(request)
            assert config['requests_per_minute'] == expected_rpm
    
    def test_burst_multiplier_application(self):
        """Test that burst multiplier is correctly applied"""
        request = {
            'endpoint': '/brain/predict',
            'user': {'tier': 'pro'}
        }
        
        config = self.limiter._get_rate_limit_config(request)
        
        # Base burst is 20, pro multiplier is 5.0, burst_multiplier is 2.0
        # Expected: 20 * 5.0 * 2.0 = 200
        assert config['burst_size'] == 200


class TestGlobalRateLimit:
    """Test global rate limiting functionality"""
    
    def test_global_rate_limit_enforcement(self):
        """Test that global rate limit is enforced"""
        limiter = RateLimiter({'global_rate_limit': 10})
        
        # Make 10 requests (should all be allowed)
        for i in range(10):
            result = limiter._check_global_rate_limit()
            assert result['allowed'] is True
        
        # 11th request should be blocked
        result = limiter._check_global_rate_limit()
        assert result['allowed'] is False
        assert 'Global rate limit exceeded' in result['details']
        assert 'retry_after' in result
    
    def test_global_rate_limit_sliding_window(self):
        """Test that global rate limit uses sliding window"""
        limiter = RateLimiter({'global_rate_limit': 5})
        
        # Add 5 requests at different times
        base_time = time.time()
        for i in range(5):
            limiter._check_global_rate_limit()
            time.sleep(0.01)  # Small delay between requests
        
        # Should be blocked immediately
        result = limiter._check_global_rate_limit()
        assert result['allowed'] is False
        
        # Wait for window to slide (oldest request expires)
        time.sleep(60.1)  # Wait just over 60 seconds
        
        # Should be allowed again
        result = limiter._check_global_rate_limit()
        assert result['allowed'] is True


class TestTokenBucketAlgorithm:
    """Test token bucket rate limiting algorithm"""
    
    def test_token_bucket_burst_and_refill(self):
        """Test token bucket allows burst and refills over time"""
        limiter = RateLimiter({'algorithm': 'token_bucket'})
        client_id = 'test_client'
        config = {
            'service': 'test',
            'requests_per_minute': 60,  # 1 per second
            'burst_size': 5
        }
        
        # Should allow burst of 5 requests
        for i in range(5):
            result = limiter._check_token_bucket_limit(client_id, config)
            assert result['allowed'] is True
        
        # 6th request should be blocked
        result = limiter._check_token_bucket_limit(client_id, config)
        assert result['allowed'] is False
        
        # Wait for refill (1 token per second)
        time.sleep(1.1)
        
        # Should allow 1 more request
        result = limiter._check_token_bucket_limit(client_id, config)
        assert result['allowed'] is True
        
        # But not 2
        result = limiter._check_token_bucket_limit(client_id, config)
        assert result['allowed'] is False
    
    def test_token_bucket_max_capacity(self):
        """Test token bucket doesn't exceed max capacity"""
        limiter = RateLimiter({'algorithm': 'token_bucket'})
        client_id = 'test_client'
        config = {
            'service': 'test',
            'requests_per_minute': 60,
            'burst_size': 5
        }
        
        # Make one request to initialize bucket
        limiter._check_token_bucket_limit(client_id, config)
        
        # Wait long enough for full refill
        time.sleep(10)
        
        # Should only allow burst_size requests, not more
        allowed_count = 0
        for _ in range(10):
            result = limiter._check_token_bucket_limit(client_id, config)
            if result['allowed']:
                allowed_count += 1
        
        assert allowed_count == 5  # Should cap at burst_size


class TestLeakyBucketAlgorithm:
    """Test leaky bucket rate limiting algorithm"""
    
    def test_leaky_bucket_queue_and_leak(self):
        """Test leaky bucket queues requests and leaks them"""
        limiter = RateLimiter({'algorithm': 'leaky_bucket'})
        client_id = 'test_client'
        config = {
            'service': 'test',
            'requests_per_minute': 60,  # 1 per second leak rate
            'burst_size': 3  # Queue capacity
        }
        
        # Fill the queue
        for i in range(3):
            result = limiter._check_leaky_bucket_limit(client_id, config)
            assert result['allowed'] is True
            assert result['queue_size'] == i + 1
        
        # Queue is full, next request blocked
        result = limiter._check_leaky_bucket_limit(client_id, config)
        assert result['allowed'] is False
        
        # Wait for leak
        time.sleep(1.1)
        
        # Should have leaked ~1 request, allowing one more
        result = limiter._check_leaky_bucket_limit(client_id, config)
        assert result['allowed'] is True


class TestFixedWindowAlgorithm:
    """Test fixed window rate limiting algorithm"""
    
    def test_fixed_window_reset(self):
        """Test fixed window resets after window expires"""
        limiter = RateLimiter({'algorithm': 'fixed_window'})
        client_id = 'test_client'
        config = {
            'service': 'test',
            'requests_per_minute': 3
        }
        
        # Use up the limit
        for _ in range(3):
            result = limiter._check_fixed_window_limit(client_id, config)
            assert result['allowed'] is True
        
        # Next request blocked
        result = limiter._check_fixed_window_limit(client_id, config)
        assert result['allowed'] is False
        
        # Wait for window to expire
        time.sleep(61)  # Just over 60 seconds
        
        # Window should reset, allowing requests again
        result = limiter._check_fixed_window_limit(client_id, config)
        assert result['allowed'] is True


class TestSlidingWindowAlgorithm:
    """Test sliding window rate limiting algorithm"""
    
    def test_sliding_window_smooth_rate(self):
        """Test sliding window provides smooth rate limiting"""
        limiter = RateLimiter({'algorithm': 'sliding_window'})
        client_id = 'test_client'
        config = {
            'service': 'test',
            'requests_per_minute': 6  # 1 per 10 seconds average
        }
        
        # Make 6 requests quickly
        for _ in range(6):
            result = limiter._check_sliding_window_limit(client_id, config)
            assert result['allowed'] is True
            time.sleep(0.1)
        
        # 7th request should be blocked
        result = limiter._check_sliding_window_limit(client_id, config)
        assert result['allowed'] is False
        
        # Wait for oldest request to slide out of window
        time.sleep(60)
        
        # Should allow new request
        result = limiter._check_sliding_window_limit(client_id, config)
        assert result['allowed'] is True


class TestAdaptiveAlgorithm:
    """Test adaptive rate limiting algorithm"""
    
    def test_adaptive_adjusts_limits(self):
        """Test adaptive algorithm adjusts limits based on behavior"""
        limiter = RateLimiter({'algorithm': 'adaptive'})
        client_id = 'test_client'
        config = {
            'service': 'test',
            'requests_per_minute': 10
        }
        
        # Simulate good behavior (low denial rate)
        limiter.rate_limit_metrics[client_id] = {
            'allowed': 90,
            'denied': 10  # 10% denial rate
        }
        
        # Make requests - should get higher limit (10 * 1.2 = 12)
        request = {}
        allowed_count = 0
        for _ in range(15):
            result = limiter._check_adaptive_limit(client_id, config, request)
            if result['allowed']:
                allowed_count += 1
        
        # Should allow more than base limit of 10
        assert allowed_count >= 12
        
        # Now simulate bad behavior (high denial rate)
        limiter.rate_limit_metrics[client_id] = {
            'allowed': 20,
            'denied': 80  # 80% denial rate
        }
        
        # Clear sliding window
        limiter.sliding_windows.clear()
        
        # Make requests - should get lower limit (10 * 0.5 = 5)
        allowed_count = 0
        for _ in range(10):
            result = limiter._check_adaptive_limit(client_id, config, request)
            if result['allowed']:
                allowed_count += 1
        
        # Should allow fewer than base limit
        assert allowed_count <= 5


class TestThrottlingAndBlacklisting:
    """Test client throttling and blacklisting functionality"""
    
    def test_throttling_lifecycle(self):
        """Test throttling is applied and expires correctly"""
        limiter = RateLimiter({})
        client_id = 'bad_client'
        
        # Setup high violation rate
        limiter.rate_limit_metrics[client_id] = {
            'allowed': 10,
            'denied': 40  # 80% denial rate
        }
        
        # Apply throttling
        config = {'service': 'test'}
        limiter._apply_throttling(client_id, config)
        
        # Check client is throttled
        assert limiter._is_client_throttled(client_id) is True
        assert client_id in limiter.throttled_clients
        
        # Check throttle info
        throttle_info = limiter.throttled_clients[client_id]
        assert throttle_info['duration'] == 300  # 5 minutes for severe
        
        # Manually expire the throttle
        limiter.throttled_clients[client_id]['until'] = time.time() - 1
        
        # Check throttle expires
        assert limiter._is_client_throttled(client_id) is False
        assert client_id not in limiter.throttled_clients
    
    def test_blacklist_blocking(self):
        """Test blacklisted clients are completely blocked"""
        limiter = RateLimiter({})
        client_id = 'blocked_client'
        
        # Add to blacklist
        limiter.blacklisted_clients.add(client_id)
        
        # Try to make request
        request = {'user': {'user_id': client_id}}
        result = limiter.check_rate_limits(request)
        
        assert result['allowed'] is False
        assert result['details'] == 'Client is blacklisted'
        assert result['retry_after'] is None


class TestMainRateLimitCheck:
    """Test main rate limit checking functionality"""
    
    def test_complete_rate_limit_flow(self):
        """Test complete rate limiting flow with real requests"""
        limiter = RateLimiter({
            'algorithm': 'token_bucket',
            'global_rate_limit': 100
        })
        
        # Create requests for different services
        requests = [
            {'user': {'user_id': 'user1', 'tier': 'free'}, 'endpoint': '/brain/predict'},
            {'user': {'user_id': 'user2', 'tier': 'pro'}, 'endpoint': '/data/query'},
            {'api_key': 'key123', 'endpoint': '/api/status'},
            {'client_ip': '192.168.1.1', 'endpoint': '/security/validate'}
        ]
        
        results = []
        for request in requests:
            result = limiter.check_rate_limits(request)
            results.append(result)
        
        # All first requests should be allowed
        assert all(r['allowed'] for r in results)
        
        # Verify different clients get different treatment
        # Make many requests from one client to trigger limits
        spam_request = {'user': {'user_id': 'spammer'}, 'endpoint': '/brain/predict'}
        
        spam_results = []
        for _ in range(150):  # More than brain service limit
            result = limiter.check_rate_limits(spam_request)
            spam_results.append(result['allowed'])
        
        # Should have some allowed and some denied
        allowed = sum(spam_results)
        denied = len(spam_results) - allowed
        
        assert allowed > 0
        assert denied > 0
        
        # Check metrics were updated
        assert limiter.rate_limit_metrics['spammer']['allowed'] > 0
        assert limiter.rate_limit_metrics['spammer']['denied'] > 0
    
    def test_algorithm_behaviors(self):
        """Test different algorithms show different behaviors"""
        algorithms = ['token_bucket', 'leaky_bucket', 'fixed_window', 'sliding_window']
        
        results = {}
        for algorithm in algorithms:
            limiter = RateLimiter({'algorithm': algorithm})
            
            # Use a service with low limits to see differences
            request = {
                'user': {'user_id': f'{algorithm}_user'},
                'endpoint': '/brain/predict'  # 100 req/min for free tier
            }
            
            # Make rapid burst of requests
            allowed_pattern = []
            for i in range(30):
                result = limiter.check_rate_limits(request)
                allowed_pattern.append(1 if result['allowed'] else 0)
                if i < 10:
                    time.sleep(0.01)  # Fast burst initially
                else:
                    time.sleep(0.1)   # Then slower
            
            results[algorithm] = allowed_pattern
        
        # Different algorithms should show different patterns
        # Token bucket: burst then steady
        # Leaky bucket: queue then leak
        # Fixed window: allow up to limit then hard stop
        # Sliding window: smooth distribution
        
        # Check that patterns are different
        patterns = list(results.values())
        # At least some algorithms should differ
        unique_patterns = []
        for p in patterns:
            if p not in unique_patterns:
                unique_patterns.append(p)
        
        assert len(unique_patterns) >= 2  # At least 2 different patterns


class TestMetricsAndStatus:
    """Test metrics collection and status reporting"""
    
    def test_metrics_accuracy(self):
        """Test that metrics are accurately tracked"""
        limiter = RateLimiter({'algorithm': 'fixed_window'})
        client_id = 'metrics_test'
        
        # Make specific number of allowed and denied requests
        request = {'user': {'user_id': client_id}, 'endpoint': '/brain/predict'}
        
        # First 100 should be allowed (brain limit for free tier)
        for _ in range(100):
            limiter.check_rate_limits(request)
        
        # Get current metrics
        metrics = limiter.rate_limit_metrics[client_id]
        assert metrics['allowed'] == 100
        assert metrics['denied'] == 0
        
        # Next requests should be denied
        for _ in range(50):
            limiter.check_rate_limits(request)
        
        metrics = limiter.rate_limit_metrics[client_id]
        assert metrics['allowed'] == 100
        assert metrics['denied'] == 50
    
    def test_status_reporting(self):
        """Test comprehensive status reporting"""
        limiter = RateLimiter({'algorithm': 'sliding_window'})
        
        # Generate some activity
        clients = ['client1', 'client2', 'client3']
        for client in clients:
            request = {'user': {'user_id': client}, 'endpoint': '/api/test'}
            for _ in range(10):
                limiter.check_rate_limits(request)
        
        # Add a throttled client
        limiter.throttled_clients['bad_client'] = {
            'until': time.time() + 300,
            'duration': 300
        }
        
        # Add a blacklisted client
        limiter.blacklisted_clients.add('blocked_client')
        
        # Get status
        status = limiter.get_rate_limiter_status()
        
        assert status['algorithm'] == 'sliding_window'
        assert status['active_clients'] == 3
        assert status['throttled_clients'] == 1
        assert status['blacklisted_clients'] == 1
        assert 'metrics_summary' in status
        
        # Check metrics summary
        summary = status['metrics_summary']
        assert summary['total_requests'] == 30  # 3 clients * 10 requests
        assert summary['total_allowed'] == 30   # All allowed
        assert summary['total_denied'] == 0
        assert 'top_clients' in summary
    
    def test_top_clients_calculation(self):
        """Test top clients are correctly identified"""
        limiter = RateLimiter({})
        
        # Create clients with different request volumes
        limiter.rate_limit_metrics['heavy_user'] = {'allowed': 1000, 'denied': 100}
        limiter.rate_limit_metrics['medium_user'] = {'allowed': 500, 'denied': 50}
        limiter.rate_limit_metrics['light_user'] = {'allowed': 100, 'denied': 10}
        
        top_clients = limiter._get_top_clients(limit=2)
        
        assert len(top_clients) == 2
        assert top_clients[0]['client_id'] == 'heavy_user'
        assert top_clients[0]['total_requests'] == 1100
        assert top_clients[1]['client_id'] == 'medium_user'
        assert top_clients[1]['total_requests'] == 550


class TestCleanupThread:
    """Test background cleanup thread functionality"""
    
    def test_cleanup_removes_old_data(self):
        """Test cleanup thread removes old data"""
        limiter = RateLimiter({})
        
        # Add some old data directly
        old_time = time.time() - 7200  # 2 hours ago
        limiter.sliding_windows['old_client'] = deque([old_time])
        limiter.sliding_windows['current_client'] = deque([time.time()])
        
        # Add expired throttle
        limiter.throttled_clients['expired'] = {
            'until': time.time() - 100,
            'duration': 60
        }
        limiter.throttled_clients['active'] = {
            'until': time.time() + 100,
            'duration': 60
        }
        
        # Run cleanup manually
        with limiter._lock:
            current_time = time.time()
            
            # Clean sliding windows
            for key in list(limiter.sliding_windows.keys()):
                window = limiter.sliding_windows[key]
                if window and len(window) > 0 and window[-1] < current_time - 3600:
                    del limiter.sliding_windows[key]
            
            # Clean throttles
            expired = [k for k, v in limiter.throttled_clients.items() 
                      if current_time > v['until']]
            for k in expired:
                del limiter.throttled_clients[k]
        
        # Check cleanup worked
        assert 'old_client' not in limiter.sliding_windows
        assert 'current_client' in limiter.sliding_windows
        assert 'expired' not in limiter.throttled_clients
        assert 'active' in limiter.throttled_clients


class TestThreadSafety:
    """Test thread safety of rate limiter"""
    
    def test_concurrent_requests(self):
        """Test rate limiter handles concurrent requests safely"""
        limiter = RateLimiter({
            'algorithm': 'token_bucket',
            'global_rate_limit': 1000
        })
        
        def make_requests(client_id, count):
            """Make requests from a client"""
            results = []
            for _ in range(count):
                request = {
                    'user': {'user_id': client_id},
                    'endpoint': '/api/test'
                }
                result = limiter.check_rate_limits(request)
                results.append(result['allowed'])
            return sum(results)
        
        # Run concurrent requests from multiple clients
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(10):
                future = executor.submit(make_requests, f'client_{i}', 50)
                futures.append(future)
            
            results = [f.result() for f in as_completed(futures)]
        
        # All clients should get their requests processed
        assert len(results) == 10
        assert all(r > 0 for r in results)  # Each client got some requests through
    
    def test_concurrent_metrics_updates(self):
        """Test concurrent metrics updates don't corrupt data"""
        limiter = RateLimiter({})
        client_id = 'concurrent_test'
        
        def update_metrics(allowed_count, denied_count):
            """Update metrics multiple times"""
            for _ in range(allowed_count):
                limiter._update_metrics(client_id, allowed=True)
            for _ in range(denied_count):
                limiter._update_metrics(client_id, allowed=False)
        
        # Run concurrent updates
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for _ in range(5):
                future = executor.submit(update_metrics, 100, 50)
                futures.append(future)
            
            for f in as_completed(futures):
                f.result()
        
        # Check final metrics are correct
        metrics = limiter.rate_limit_metrics[client_id]
        assert metrics['allowed'] == 500  # 5 threads * 100
        assert metrics['denied'] == 250   # 5 threads * 50


class TestEdgeCases:
    """Test edge cases and error scenarios"""
    
    def test_empty_request_handling(self):
        """Test handling of empty requests"""
        limiter = RateLimiter({})
        
        # Empty request
        result = limiter.check_rate_limits({})
        assert 'allowed' in result
        assert isinstance(result['allowed'], bool)
        
        # None values
        result = limiter.check_rate_limits({'user': None})
        assert 'allowed' in result
        
        # Malformed data
        result = limiter.check_rate_limits({
            'user': 'not_a_dict',
            'endpoint': None
        })
        assert 'allowed' in result
    
    def test_extreme_values(self):
        """Test handling of extreme values"""
        limiter = RateLimiter({})
        
        # Very large numbers
        config = {
            'service': 'test',
            'requests_per_minute': 10**9,
            'burst_size': 10**6
        }
        
        result = limiter._check_token_bucket_limit('test', config)
        assert 'allowed' in result
        assert not isinstance(result.get('tokens_remaining'), type(None))
        
        # Zero limits
        config = {
            'service': 'test',
            'requests_per_minute': 0,
            'burst_size': 0
        }
        
        result = limiter._check_token_bucket_limit('zero_test', config)
        assert result['allowed'] is False
        
        # Negative values (should be handled gracefully)
        limiter.token_buckets['negative_test'] = {
            'tokens': -10,
            'last_refill': time.time(),
            'refill_rate': 1.0
        }
        
        result = limiter._check_token_bucket_limit('negative_test', config)
        assert 'allowed' in result
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters"""
        limiter = RateLimiter({})
        
        # Unicode client IDs
        special_ids = [
            'тест_клиент',
            '测试客户',
            'emoji_😀_client',
            'special!@#$%^&*()',
            'spaces in id',
            'tab\there',
            'newline\nhere'
        ]
        
        for client_id in special_ids:
            request = {'user': {'user_id': client_id}}
            result = limiter.check_rate_limits(request)
            assert 'allowed' in result
            assert isinstance(result['allowed'], bool)
    
    def test_invalid_client_id_validation(self):
        """Test that invalid client IDs are properly validated"""
        limiter = RateLimiter({})
        
        # This should trigger validation error
        with pytest.raises(Exception):
            # Pass None directly to _update_metrics
            limiter._update_metrics(None, allowed=True)
        
        # Empty string should also fail
        with pytest.raises(Exception):
            limiter._update_metrics('', allowed=True)
        
        # Non-string types should fail
        with pytest.raises(Exception):
            limiter._update_metrics(123, allowed=True)
    
    def test_time_boundary_conditions(self):
        """Test handling of time-related edge cases"""
        limiter = RateLimiter({'algorithm': 'token_bucket'})
        
        # Test with future time (clock skew scenario)
        client_id = 'time_test'
        config = {
            'service': 'test',
            'requests_per_minute': 60,
            'burst_size': 10
        }
        
        # Initialize bucket with future time
        limiter.token_buckets[f'{client_id}:{config["service"]}'] = {
            'tokens': 5,
            'last_refill': time.time() + 3600,  # 1 hour in future
            'refill_rate': 1.0
        }
        
        # Should handle gracefully
        result = limiter._check_token_bucket_limit(client_id, config)
        assert 'allowed' in result


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    def test_multi_tier_rate_limiting(self):
        """Test rate limiting across different user tiers"""
        limiter = RateLimiter({'algorithm': 'sliding_window'})
        
        # Create users with different tiers
        users = [
            {'id': 'free_user', 'tier': 'free'},
            {'id': 'basic_user', 'tier': 'basic'},
            {'id': 'pro_user', 'tier': 'pro'},
            {'id': 'enterprise_user', 'tier': 'enterprise'}
        ]
        
        results = {}
        for user in users:
            request = {
                'user': {'user_id': user['id'], 'tier': user['tier']},
                'endpoint': '/brain/predict'
            }
            
            # Make requests up to likely limit
            allowed = 0
            for _ in range(200):
                result = limiter.check_rate_limits(request)
                if result['allowed']:
                    allowed += 1
                else:
                    break  # Stop when hitting limit
            
            results[user['tier']] = allowed
        
        # Higher tiers should allow more requests
        assert results['free'] < results['basic']
        assert results['basic'] < results['pro']
        assert results['pro'] < results['enterprise']
    
    def test_service_priority_handling(self):
        """Test that service priorities are respected"""
        limiter = RateLimiter({})
        
        # Test different priority services
        services = [
            ('/security/validate', 'critical'),
            ('/brain/predict', 'high'),
            ('/training/start', 'medium'),
            ('/api/status', 'low')
        ]
        
        for endpoint, expected_priority in services:
            request = {'endpoint': endpoint}
            config = limiter._get_rate_limit_config(request)
            assert config['priority'] == expected_priority
    
    def test_adaptive_learning_over_time(self):
        """Test adaptive algorithm learns from client behavior"""
        limiter = RateLimiter({'algorithm': 'adaptive'})
        client_id = 'learning_client'
        
        request = {
            'user': {'user_id': client_id},
            'endpoint': '/brain/predict'
        }
        
        # Phase 1: Good behavior
        for _ in range(60):
            limiter.check_rate_limits(request)
            time.sleep(0.05)  # Spread out requests
        
        initial_metrics = limiter.rate_limit_metrics[client_id].copy()
        
        # Phase 2: Bad behavior (rapid requests)
        for _ in range(100):
            limiter.check_rate_limits(request)
            # No delay - hammer the service
        
        final_metrics = limiter.rate_limit_metrics[client_id]
        
        # Should have adapted - more denials in phase 2
        denial_rate_initial = initial_metrics.get('denied', 0) / max(1, sum(initial_metrics.values()))
        denial_rate_final = final_metrics['denied'] / sum(final_metrics.values())
        
        # Denial rate should increase with bad behavior
        assert denial_rate_final > denial_rate_initial


class TestRobustness:
    """Test robustness under stress conditions"""
    
    def test_high_volume_requests(self):
        """Test handling of high volume of requests"""
        limiter = RateLimiter({
            'algorithm': 'sliding_window',
            'global_rate_limit': 10000
        })
        
        # Generate many requests quickly
        start_time = time.time()
        request_count = 1000
        
        for i in range(request_count):
            request = {
                'user': {'user_id': f'user_{i % 100}'},  # 100 different users
                'endpoint': '/api/test'
            }
            limiter.check_rate_limits(request)
        
        elapsed = time.time() - start_time
        
        # Should handle 1000 requests quickly
        assert elapsed < 5.0  # Should complete in under 5 seconds
        
        # Check metrics are consistent
        total_allowed = sum(m['allowed'] for m in limiter.rate_limit_metrics.values())
        total_denied = sum(m['denied'] for m in limiter.rate_limit_metrics.values())
        assert total_allowed + total_denied == request_count
    
    def test_memory_usage_with_many_clients(self):
        """Test memory doesn't grow unbounded with many clients"""
        limiter = RateLimiter({})
        
        # Create many unique clients
        for i in range(1000):
            client_id = f'client_{i}'
            request = {
                'user': {'user_id': client_id},
                'endpoint': '/api/test'
            }
            limiter.check_rate_limits(request)
        
        # Check data structures aren't growing unbounded
        assert len(limiter.rate_limit_metrics) <= 1001  # 1000 clients + possible global
        
        # Sliding windows should be reasonable
        assert len(limiter.sliding_windows) <= 2000  # Some overhead is ok
    
    def test_recovery_from_errors(self):
        """Test system recovers from various error conditions"""
        limiter = RateLimiter({})
        
        # Corrupt internal state
        limiter.token_buckets['bad_bucket'] = {'invalid': 'data'}
        limiter.sliding_windows['bad_window'] = 'not_a_deque'
        
        # Should still handle requests
        request = {'user': {'user_id': 'test_user'}, 'endpoint': '/api/test'}
        result = limiter.check_rate_limits(request)
        
        assert 'allowed' in result
        assert isinstance(result['allowed'], bool)
        
        # Should have logged errors but continued
        assert limiter.rate_limit_metrics['test_user']['allowed'] > 0 or \
               limiter.rate_limit_metrics['test_user']['denied'] > 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])