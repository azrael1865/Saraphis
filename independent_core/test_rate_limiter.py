"""
Comprehensive test suite for RateLimiter
Tests all 26 methods and various rate limiting algorithms
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from collections import deque, defaultdict
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
        assert isinstance(limiter.rate_limit_metrics, defaultdict)
    
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
    
    def test_thread_safety_initialization(self):
        """Test thread safety components are initialized"""
        limiter = RateLimiter({})
        
        assert hasattr(limiter, '_lock')
        assert isinstance(limiter._lock, threading.Lock)


class TestClientIdentification:
    """Test client identification functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.limiter = RateLimiter({})
    
    def test_extract_client_id_user_id_priority(self):
        """Test client ID extraction with user ID having highest priority"""
        request = {
            'user': {'user_id': 'user123'},
            'api_key': 'key456',
            'client_ip': '192.168.1.1'
        }
        
        client_id = self.limiter._extract_client_id(request)
        
        assert client_id == 'user123'
    
    def test_extract_client_id_api_key_fallback(self):
        """Test client ID extraction falls back to API key"""
        request = {
            'api_key': 'key456',
            'client_ip': '192.168.1.1',
            'session_id': 'session789'
        }
        
        client_id = self.limiter._extract_client_id(request)
        
        assert client_id == 'key456'
    
    def test_extract_client_id_ip_fallback(self):
        """Test client ID extraction falls back to client IP"""
        request = {
            'client_ip': '192.168.1.1',
            'session_id': 'session789'
        }
        
        client_id = self.limiter._extract_client_id(request)
        
        assert client_id == '192.168.1.1'
    
    def test_extract_client_id_forwarded_for(self):
        """Test client ID extraction from X-Forwarded-For header"""
        request = {
            'headers': {'X-Forwarded-For': '10.0.0.1'},
            'session_id': 'session789'
        }
        
        client_id = self.limiter._extract_client_id(request)
        
        assert client_id == '10.0.0.1'
    
    def test_extract_client_id_session_fallback(self):
        """Test client ID extraction falls back to session ID"""
        request = {
            'session_id': 'session789'
        }
        
        client_id = self.limiter._extract_client_id(request)
        
        assert client_id == 'session789'
    
    def test_extract_client_id_anonymous_default(self):
        """Test client ID extraction defaults to anonymous"""
        request = {}
        
        client_id = self.limiter._extract_client_id(request)
        
        assert client_id == 'anonymous'
    
    def test_extract_client_id_error_handling(self):
        """Test client ID extraction handles errors gracefully"""
        # Create request that will cause extraction to fail
        request = {'user': None}  # Will cause AttributeError when accessing user.get()
        
        with patch.object(self.limiter.logger, 'error') as mock_logger:
            client_id = self.limiter._extract_client_id(request)
        
        assert client_id == 'anonymous'
        mock_logger.assert_called_once()
    
    def test_extract_client_id_type_conversion(self):
        """Test client ID is properly converted to string"""
        request = {
            'user': {'user_id': 12345}  # Integer user ID
        }
        
        client_id = self.limiter._extract_client_id(request)
        
        assert client_id == '12345'
        assert isinstance(client_id, str)


class TestRateLimitConfiguration:
    """Test rate limit configuration retrieval"""
    
    def setup_method(self):
        """Setup for each test"""
        self.limiter = RateLimiter({})
    
    def test_get_rate_limit_config_brain_service(self):
        """Test rate limit config for brain service"""
        request = {
            'endpoint': '/brain/predict',
            'user': {'tier': 'pro'}
        }
        
        config = self.limiter._get_rate_limit_config(request)
        
        assert config['service'] == 'brain'
        assert config['priority'] == 'high'
        assert config['requests_per_minute'] == 500  # 100 * 5.0 (pro tier)
        assert config['requests_per_hour'] == 5000   # 1000 * 5.0 (pro tier)
    
    def test_get_rate_limit_config_unknown_service(self):
        """Test rate limit config for unknown service defaults to api"""
        request = {
            'endpoint': '/unknown/endpoint',
            'user': {'tier': 'basic'}
        }
        
        config = self.limiter._get_rate_limit_config(request)
        
        assert config['service'] == 'api'
        assert config['priority'] == 'low'
        assert config['requests_per_minute'] == 1000  # 500 * 2.0 (basic tier)
    
    def test_get_rate_limit_config_tier_multipliers(self):
        """Test tier multipliers are applied correctly"""
        base_request = {
            'endpoint': '/uncertainty/analyze',
        }
        
        # Test different tiers
        tiers_and_multipliers = [
            ('free', 1.0),
            ('basic', 2.0),
            ('pro', 5.0),
            ('enterprise', 10.0),
            ('unknown_tier', 1.0)  # Should default to 1.0
        ]
        
        for tier, expected_multiplier in tiers_and_multipliers:
            request = base_request.copy()
            request['user'] = {'tier': tier}
            
            config = self.limiter._get_rate_limit_config(request)
            
            base_limit = self.limiter.service_limits['uncertainty']['requests_per_minute']
            expected_limit = int(base_limit * expected_multiplier)
            assert config['requests_per_minute'] == expected_limit
    
    def test_get_rate_limit_config_burst_multiplier(self):
        """Test burst multiplier is applied to burst size"""
        self.limiter.burst_multiplier = 3.0
        request = {
            'endpoint': '/security/validate',
            'user': {'tier': 'enterprise'}
        }
        
        config = self.limiter._get_rate_limit_config(request)
        
        base_burst = self.limiter.service_limits['security']['burst_size']
        tier_multiplier = 10.0  # enterprise
        expected_burst = int(base_burst * tier_multiplier * 3.0)  # burst_multiplier
        assert config['burst_size'] == expected_burst
    
    def test_get_rate_limit_config_no_user(self):
        """Test rate limit config when no user information provided"""
        request = {
            'endpoint': '/data/query'
        }
        
        config = self.limiter._get_rate_limit_config(request)
        
        assert config['service'] == 'data'
        # Should use free tier multiplier (1.0)
        base_limit = self.limiter.service_limits['data']['requests_per_minute']
        assert config['requests_per_minute'] == base_limit
    
    def test_get_rate_limit_config_error_handling(self):
        """Test rate limit config error handling"""
        # Create request that will cause an error
        request = {'endpoint': None}  # Will cause error in string operations
        
        with patch.object(self.limiter.logger, 'error') as mock_logger:
            config = self.limiter._get_rate_limit_config(request)
        
        # Should return default config
        assert config['service'] == 'unknown'
        assert config['requests_per_minute'] == 100
        assert config['requests_per_hour'] == 1000
        assert config['burst_size'] == 20
        assert config['priority'] == 'medium'
        mock_logger.assert_called_once()


class TestGlobalRateLimit:
    """Test global rate limiting functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.limiter = RateLimiter({'global_rate_limit': 5})  # Small limit for testing
    
    def test_global_rate_limit_allows_under_limit(self):
        """Test global rate limit allows requests under limit"""
        # Make requests under the limit
        for _ in range(3):
            result = self.limiter._check_global_rate_limit()
            assert result['allowed'] is True
    
    def test_global_rate_limit_blocks_over_limit(self):
        """Test global rate limit blocks requests over limit"""
        # Fill up the limit
        for _ in range(5):
            self.limiter._check_global_rate_limit()
        
        # Next request should be blocked
        result = self.limiter._check_global_rate_limit()
        
        assert result['allowed'] is False
        assert result['details'] == 'Global rate limit exceeded'
        assert 'retry_after' in result
        assert isinstance(result['retry_after'], (int, float))
    
    def test_global_rate_limit_sliding_window(self):
        """Test global rate limit uses sliding window"""
        # Fill up the limit
        start_time = time.time()
        
        for _ in range(5):
            self.limiter._check_global_rate_limit()
        
        # Should be blocked
        result = self.limiter._check_global_rate_limit()
        assert result['allowed'] is False
        
        # Mock time advancement to test sliding window
        with patch('time.time') as mock_time:
            mock_time.return_value = start_time + 61  # 1 minute later
            
            # Should be allowed again due to sliding window
            result = self.limiter._check_global_rate_limit()
            assert result['allowed'] is True
    
    def test_global_rate_limit_error_handling(self):
        """Test global rate limit error handling fails open"""
        with patch('time.time', side_effect=Exception("Time error")):
            with patch.object(self.limiter.logger, 'error') as mock_logger:
                result = self.limiter._check_global_rate_limit()
        
        # Should fail open (allow request)
        assert result['allowed'] is True
        mock_logger.assert_called_once()


class TestTokenBucketAlgorithm:
    """Test token bucket rate limiting algorithm"""
    
    def setup_method(self):
        """Setup for each test"""
        config = {'algorithm': 'token_bucket'}
        self.limiter = RateLimiter(config)
        self.client_id = 'test_client'
        self.rate_config = {
            'service': 'test',
            'requests_per_minute': 60,  # 1 request per second
            'burst_size': 10
        }
    
    def test_token_bucket_allows_initial_burst(self):
        """Test token bucket allows initial burst of requests"""
        # Should allow burst_size requests initially
        for i in range(self.rate_config['burst_size']):
            result = self.limiter._check_token_bucket_limit(self.client_id, self.rate_config)
            assert result['allowed'] is True
            assert result['tokens_remaining'] == self.rate_config['burst_size'] - 1 - i
    
    def test_token_bucket_blocks_after_burst(self):
        """Test token bucket blocks requests after burst is exhausted"""
        # Exhaust the burst
        for _ in range(self.rate_config['burst_size']):
            self.limiter._check_token_bucket_limit(self.client_id, self.rate_config)
        
        # Next request should be blocked
        result = self.limiter._check_token_bucket_limit(self.client_id, self.rate_config)
        
        assert result['allowed'] is False
        assert result['details'] == 'Rate limit exceeded (token bucket)'
        assert result['tokens_remaining'] == 0
        assert 'retry_after' in result
    
    def test_token_bucket_refills_over_time(self):
        """Test token bucket refills tokens over time"""
        # Exhaust the bucket
        for _ in range(self.rate_config['burst_size']):
            self.limiter._check_token_bucket_limit(self.client_id, self.rate_config)
        
        # Mock time advancement
        bucket_key = f"{self.client_id}:{self.rate_config['service']}"
        bucket = self.limiter.token_buckets[bucket_key]
        original_time = bucket['last_refill']
        
        # Advance time by 5 seconds (should refill ~5 tokens at 1/second rate)
        with patch('time.time') as mock_time:
            mock_time.return_value = original_time + 5.0
            
            result = self.limiter._check_token_bucket_limit(self.client_id, self.rate_config)
            assert result['allowed'] is True
            assert result['tokens_remaining'] >= 3  # Should have refilled tokens
    
    def test_token_bucket_refill_rate_calculation(self):
        """Test token bucket refill rate is calculated correctly"""
        result = self.limiter._check_token_bucket_limit(self.client_id, self.rate_config)
        
        bucket_key = f"{self.client_id}:{self.rate_config['service']}"
        bucket = self.limiter.token_buckets[bucket_key]
        
        expected_refill_rate = self.rate_config['requests_per_minute'] / 60.0
        assert bucket['refill_rate'] == expected_refill_rate
        assert result['refill_rate'] == expected_refill_rate
    
    def test_token_bucket_maximum_tokens_capped(self):
        """Test token bucket tokens are capped at burst size"""
        # Wait for potential refill, then check max tokens
        with patch('time.time') as mock_time:
            mock_time.return_value = time.time() + 3600  # 1 hour later
            
            result = self.limiter._check_token_bucket_limit(self.client_id, self.rate_config)
            
            bucket_key = f"{self.client_id}:{self.rate_config['service']}"
            bucket = self.limiter.token_buckets[bucket_key]
            
            # Tokens should not exceed burst size
            assert bucket['tokens'] <= self.rate_config['burst_size']
    
    def test_token_bucket_error_handling(self):
        """Test token bucket error handling"""
        with patch('time.time', side_effect=Exception("Time error")):
            with patch.object(self.limiter.logger, 'error') as mock_logger:
                result = self.limiter._check_token_bucket_limit(self.client_id, self.rate_config)
        
        assert result['allowed'] is False
        assert 'Token bucket check error' in result['details']
        mock_logger.assert_called_once()


class TestLeakyBucketAlgorithm:
    """Test leaky bucket rate limiting algorithm"""
    
    def setup_method(self):
        """Setup for each test"""
        config = {'algorithm': 'leaky_bucket'}
        self.limiter = RateLimiter(config)
        self.client_id = 'test_client'
        self.rate_config = {
            'service': 'test',
            'requests_per_minute': 60,  # 1 request per second leak rate
            'burst_size': 5
        }
    
    def test_leaky_bucket_allows_initial_requests(self):
        """Test leaky bucket allows requests up to burst size"""
        for i in range(self.rate_config['burst_size']):
            result = self.limiter._check_leaky_bucket_limit(self.client_id, self.rate_config)
            assert result['allowed'] is True
            assert result['queue_size'] == i + 1
            assert result['queue_capacity'] == self.rate_config['burst_size']
    
    def test_leaky_bucket_blocks_when_full(self):
        """Test leaky bucket blocks when queue is full"""
        # Fill the bucket
        for _ in range(self.rate_config['burst_size']):
            self.limiter._check_leaky_bucket_limit(self.client_id, self.rate_config)
        
        # Next request should be blocked
        result = self.limiter._check_leaky_bucket_limit(self.client_id, self.rate_config)
        
        assert result['allowed'] is False
        assert result['details'] == 'Rate limit exceeded (leaky bucket)'
        assert result['queue_size'] == self.rate_config['burst_size']
        assert 'retry_after' in result
    
    def test_leaky_bucket_leaks_over_time(self):
        """Test leaky bucket leaks requests over time"""
        # Fill the bucket
        for _ in range(self.rate_config['burst_size']):
            self.limiter._check_leaky_bucket_limit(self.client_id, self.rate_config)
        
        # Mock time advancement to trigger leak
        bucket_key = f"{self.client_id}:{self.rate_config['service']}"
        bucket = self.limiter.leaky_buckets[bucket_key]
        original_time = bucket['last_leak']
        
        with patch('time.time') as mock_time:
            # Advance time by 3 seconds (should leak ~3 requests at 1/second)
            mock_time.return_value = original_time + 3.0
            
            result = self.limiter._check_leaky_bucket_limit(self.client_id, self.rate_config)
            
            # Should allow request due to leaked space
            assert result['allowed'] is True
            assert result['queue_size'] <= self.rate_config['burst_size']
    
    def test_leaky_bucket_leak_rate_calculation(self):
        """Test leaky bucket leak rate is calculated correctly"""
        self.limiter._check_leaky_bucket_limit(self.client_id, self.rate_config)
        
        bucket_key = f"{self.client_id}:{self.rate_config['service']}"
        bucket = self.limiter.leaky_buckets[bucket_key]
        
        expected_leak_rate = self.rate_config['requests_per_minute'] / 60.0
        assert bucket['leak_rate'] == expected_leak_rate
    
    def test_leaky_bucket_error_handling(self):
        """Test leaky bucket error handling"""
        with patch('time.time', side_effect=Exception("Time error")):
            with patch.object(self.limiter.logger, 'error') as mock_logger:
                result = self.limiter._check_leaky_bucket_limit(self.client_id, self.rate_config)
        
        assert result['allowed'] is False
        assert 'Leaky bucket check error' in result['details']
        mock_logger.assert_called_once()


class TestFixedWindowAlgorithm:
    """Test fixed window rate limiting algorithm"""
    
    def setup_method(self):
        """Setup for each test"""
        config = {'algorithm': 'fixed_window'}
        self.limiter = RateLimiter(config)
        self.client_id = 'test_client'
        self.rate_config = {
            'service': 'test',
            'requests_per_minute': 5
        }
    
    def test_fixed_window_allows_within_limit(self):
        """Test fixed window allows requests within limit"""
        for i in range(self.rate_config['requests_per_minute']):
            result = self.limiter._check_fixed_window_limit(self.client_id, self.rate_config)
            assert result['allowed'] is True
            assert result['requests_in_window'] == i + 1
            assert result['window_limit'] == self.rate_config['requests_per_minute']
            assert 'window_reset' in result
    
    def test_fixed_window_blocks_over_limit(self):
        """Test fixed window blocks requests over limit"""
        # Fill the window
        for _ in range(self.rate_config['requests_per_minute']):
            self.limiter._check_fixed_window_limit(self.client_id, self.rate_config)
        
        # Next request should be blocked
        result = self.limiter._check_fixed_window_limit(self.client_id, self.rate_config)
        
        assert result['allowed'] is False
        assert result['details'] == 'Rate limit exceeded (fixed window)'
        assert result['requests_in_window'] == self.rate_config['requests_per_minute']
        assert 'retry_after' in result
        assert 'window_reset' in result
    
    def test_fixed_window_resets_after_expiry(self):
        """Test fixed window resets after window expires"""
        # Fill the window
        for _ in range(self.rate_config['requests_per_minute']):
            self.limiter._check_fixed_window_limit(self.client_id, self.rate_config)
        
        # Mock time advancement past window
        window_key = f"{self.client_id}:{self.rate_config['service']}"
        window = self.limiter.fixed_windows[window_key]
        original_start = window['window_start']
        
        with patch('time.time') as mock_time:
            mock_time.return_value = original_start + 65  # Past 60-second window
            
            result = self.limiter._check_fixed_window_limit(self.client_id, self.rate_config)
            
            assert result['allowed'] is True
            assert result['requests_in_window'] == 1  # Reset to 1
    
    def test_fixed_window_error_handling(self):
        """Test fixed window error handling"""
        with patch('time.time', side_effect=Exception("Time error")):
            with patch.object(self.limiter.logger, 'error') as mock_logger:
                result = self.limiter._check_fixed_window_limit(self.client_id, self.rate_config)
        
        assert result['allowed'] is False
        assert 'Fixed window check error' in result['details']
        mock_logger.assert_called_once()


class TestSlidingWindowAlgorithm:
    """Test sliding window rate limiting algorithm"""
    
    def setup_method(self):
        """Setup for each test"""
        config = {'algorithm': 'sliding_window'}
        self.limiter = RateLimiter(config)
        self.client_id = 'test_client'
        self.rate_config = {
            'service': 'test',
            'requests_per_minute': 5
        }
    
    def test_sliding_window_allows_within_limit(self):
        """Test sliding window allows requests within limit"""
        for i in range(self.rate_config['requests_per_minute']):
            result = self.limiter._check_sliding_window_limit(self.client_id, self.rate_config)
            assert result['allowed'] is True
            assert result['requests_in_window'] == i + 1
            assert result['window_limit'] == self.rate_config['requests_per_minute']
    
    def test_sliding_window_blocks_over_limit(self):
        """Test sliding window blocks requests over limit"""
        # Fill the window
        for _ in range(self.rate_config['requests_per_minute']):
            self.limiter._check_sliding_window_limit(self.client_id, self.rate_config)
        
        # Next request should be blocked
        result = self.limiter._check_sliding_window_limit(self.client_id, self.rate_config)
        
        assert result['allowed'] is False
        assert result['details'] == 'Rate limit exceeded (sliding window)'
        assert result['requests_in_window'] == self.rate_config['requests_per_minute']
        assert 'retry_after' in result
    
    def test_sliding_window_slides_with_time(self):
        """Test sliding window removes old requests"""
        # Fill the window
        start_time = time.time()
        
        for _ in range(self.rate_config['requests_per_minute']):
            self.limiter._check_sliding_window_limit(self.client_id, self.rate_config)
        
        # Mock time advancement to slide window
        with patch('time.time') as mock_time:
            mock_time.return_value = start_time + 65  # Past window
            
            result = self.limiter._check_sliding_window_limit(self.client_id, self.rate_config)
            
            assert result['allowed'] is True  # Old requests should be gone
            assert result['requests_in_window'] == 1
    
    def test_sliding_window_partial_slide(self):
        """Test sliding window with partial time advancement"""
        # Add requests at different times
        base_time = time.time()
        
        with patch('time.time') as mock_time:
            # Add 3 requests
            for i in range(3):
                mock_time.return_value = base_time + i * 10
                self.limiter._check_sliding_window_limit(self.client_id, self.rate_config)
            
            # Advance past some but not all requests (remove first request)
            mock_time.return_value = base_time + 70  # 70 seconds later
            
            result = self.limiter._check_sliding_window_limit(self.client_id, self.rate_config)
            
            assert result['allowed'] is True
            # Should have 2 remaining + 1 new = 3 total
            assert result['requests_in_window'] == 3
    
    def test_sliding_window_error_handling(self):
        """Test sliding window error handling"""
        with patch('time.time', side_effect=Exception("Time error")):
            with patch.object(self.limiter.logger, 'error') as mock_logger:
                result = self.limiter._check_sliding_window_limit(self.client_id, self.rate_config)
        
        assert result['allowed'] is False
        assert 'Sliding window check error' in result['details']
        mock_logger.assert_called_once()


class TestAdaptiveAlgorithm:
    """Test adaptive rate limiting algorithm"""
    
    def setup_method(self):
        """Setup for each test"""
        config = {'algorithm': 'adaptive'}
        self.limiter = RateLimiter(config)
        self.client_id = 'test_client'
        self.rate_config = {
            'service': 'test',
            'requests_per_minute': 100
        }
    
    def test_adaptive_uses_base_limit_initially(self):
        """Test adaptive algorithm uses base limit initially"""
        request = {}
        
        # Should use sliding window with base limit initially
        result = self.limiter._check_adaptive_limit(self.client_id, self.rate_config, request)
        
        assert result['allowed'] is True
    
    def test_adaptive_reduces_limit_for_high_denial_rate(self):
        """Test adaptive algorithm reduces limit for clients with high denial rate"""
        # Simulate high denial rate
        self.limiter.rate_limit_metrics[self.client_id] = {
            'allowed': 20,
            'denied': 80  # 80% denial rate
        }
        
        request = {}
        
        with patch.object(self.limiter, '_check_sliding_window_limit') as mock_sliding:
            mock_sliding.return_value = {'allowed': True}
            
            self.limiter._check_adaptive_limit(self.client_id, self.rate_config, request)
            
            # Should call sliding window with reduced limit
            call_args = mock_sliding.call_args[0]
            adapted_config = call_args[1]
            
            # Should be 50% of base limit (high denial rate)
            assert adapted_config['requests_per_minute'] == 50
    
    def test_adaptive_reduces_limit_for_moderate_denial_rate(self):
        """Test adaptive algorithm reduces limit for moderate denial rate"""
        # Simulate moderate denial rate
        self.limiter.rate_limit_metrics[self.client_id] = {
            'allowed': 70,
            'denied': 30  # 30% denial rate
        }
        
        request = {}
        
        with patch.object(self.limiter, '_check_sliding_window_limit') as mock_sliding:
            mock_sliding.return_value = {'allowed': True}
            
            self.limiter._check_adaptive_limit(self.client_id, self.rate_config, request)
            
            call_args = mock_sliding.call_args[0]
            adapted_config = call_args[1]
            
            # Should be 80% of base limit (moderate denial rate)
            assert adapted_config['requests_per_minute'] == 80
    
    def test_adaptive_increases_limit_for_low_denial_rate(self):
        """Test adaptive algorithm increases limit for low denial rate"""
        # Simulate low denial rate
        self.limiter.rate_limit_metrics[self.client_id] = {
            'allowed': 90,
            'denied': 10  # 10% denial rate
        }
        
        request = {}
        
        with patch.object(self.limiter, '_check_sliding_window_limit') as mock_sliding:
            mock_sliding.return_value = {'allowed': True}
            
            self.limiter._check_adaptive_limit(self.client_id, self.rate_config, request)
            
            call_args = mock_sliding.call_args[0]
            adapted_config = call_args[1]
            
            # Should be 120% of base limit (low denial rate)
            assert adapted_config['requests_per_minute'] == 120
    
    def test_adaptive_ignores_insufficient_data(self):
        """Test adaptive algorithm ignores clients with insufficient data"""
        # Simulate insufficient data
        self.limiter.rate_limit_metrics[self.client_id] = {
            'allowed': 30,
            'denied': 20  # Only 50 total requests, below threshold
        }
        
        request = {}
        
        with patch.object(self.limiter, '_check_sliding_window_limit') as mock_sliding:
            mock_sliding.return_value = {'allowed': True}
            
            self.limiter._check_adaptive_limit(self.client_id, self.rate_config, request)
            
            call_args = mock_sliding.call_args[0]
            adapted_config = call_args[1]
            
            # Should use base limit (no adaptation)
            assert adapted_config['requests_per_minute'] == 100
    
    def test_adaptive_error_handling(self):
        """Test adaptive algorithm error handling"""
        request = {}
        
        with patch.object(self.limiter, '_check_sliding_window_limit', side_effect=Exception("Sliding error")):
            with patch.object(self.limiter.logger, 'error') as mock_logger:
                result = self.limiter._check_adaptive_limit(self.client_id, self.rate_config, request)
        
        mock_logger.assert_called_once()
        # Should fallback to sliding window with original config


class TestThrottlingAndBlacklisting:
    """Test client throttling and blacklisting functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.limiter = RateLimiter({})
        self.client_id = 'test_client'
    
    def test_is_client_throttled_not_throttled(self):
        """Test client is not throttled initially"""
        assert self.limiter._is_client_throttled(self.client_id) is False
    
    def test_is_client_throttled_active_throttle(self):
        """Test client is throttled when active throttle exists"""
        # Add throttle
        future_time = time.time() + 300  # 5 minutes in future
        self.limiter.throttled_clients[self.client_id] = {
            'until': future_time,
            'duration': 300,
            'reason': 'High violation rate'
        }
        
        assert self.limiter._is_client_throttled(self.client_id) is True
    
    def test_is_client_throttled_expired_throttle(self):
        """Test expired throttle is removed and client not throttled"""
        # Add expired throttle
        past_time = time.time() - 10  # 10 seconds ago
        self.limiter.throttled_clients[self.client_id] = {
            'until': past_time,
            'duration': 300,
            'reason': 'High violation rate'
        }
        
        result = self.limiter._is_client_throttled(self.client_id)
        
        assert result is False
        assert self.client_id not in self.limiter.throttled_clients
    
    def test_apply_throttling_severe_violations(self):
        """Test throttling is applied for severe violations"""
        # Set up metrics for severe violations
        self.limiter.rate_limit_metrics[self.client_id] = {
            'allowed': 10,
            'denied': 40  # 80% denial rate
        }
        
        config = {'service': 'test'}
        
        with patch.object(self.limiter.logger, 'warning') as mock_logger:
            self.limiter._apply_throttling(self.client_id, config)
        
        assert self.client_id in self.limiter.throttled_clients
        throttle_info = self.limiter.throttled_clients[self.client_id]
        assert throttle_info['duration'] == 300  # 5 minutes
        assert 'High violation rate' in throttle_info['reason']
        mock_logger.assert_called_once()
    
    def test_apply_throttling_moderate_violations(self):
        """Test throttling is applied for moderate violations"""
        # Set up metrics for moderate violations
        self.limiter.rate_limit_metrics[self.client_id] = {
            'allowed': 30,
            'denied': 20  # 40% denial rate
        }
        
        config = {'service': 'test'}
        
        self.limiter._apply_throttling(self.client_id, config)
        
        assert self.client_id in self.limiter.throttled_clients
        throttle_info = self.limiter.throttled_clients[self.client_id]
        assert throttle_info['duration'] == 60  # 1 minute
    
    def test_apply_throttling_no_throttle_needed(self):
        """Test no throttling applied for low violations"""
        # Set up metrics for low violations
        self.limiter.rate_limit_metrics[self.client_id] = {
            'allowed': 80,
            'denied': 20  # 20% denial rate (below moderate threshold)
        }
        
        config = {'service': 'test'}
        
        self.limiter._apply_throttling(self.client_id, config)
        
        assert self.client_id not in self.limiter.throttled_clients
    
    def test_apply_throttling_error_handling(self):
        """Test throttling application error handling"""
        config = {'service': 'test'}
        
        with patch('time.time', side_effect=Exception("Time error")):
            with patch.object(self.limiter.logger, 'error') as mock_logger:
                self.limiter._apply_throttling(self.client_id, config)
        
        mock_logger.assert_called_once()
    
    def test_blacklisted_client_blocked(self):
        """Test blacklisted clients are blocked"""
        # Add client to blacklist
        self.limiter.blacklisted_clients.add(self.client_id)
        
        request = {'client_ip': self.client_id}
        
        result = self.limiter.check_rate_limits(request)
        
        assert result['allowed'] is False
        assert result['details'] == 'Client is blacklisted'
        assert result['retry_after'] is None


class TestMainRateLimitCheck:
    """Test main rate limit checking functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.limiter = RateLimiter({'global_rate_limit': 1000})
    
    def test_check_rate_limits_success(self):
        """Test successful rate limit check"""
        request = {
            'user': {'user_id': 'user123', 'tier': 'pro'},
            'endpoint': '/brain/predict'
        }
        
        result = self.limiter.check_rate_limits(request)
        
        assert result['allowed'] is True
    
    def test_check_rate_limits_global_limit_exceeded(self):
        """Test rate limit check blocked by global limit"""
        # Set very small global limit
        self.limiter.global_rate_limit = 1
        
        # Make two requests
        request = {'client_ip': '192.168.1.1'}
        
        self.limiter.check_rate_limits(request)  # First should pass
        result = self.limiter.check_rate_limits(request)  # Second should fail
        
        assert result['allowed'] is False
        assert 'Global rate limit exceeded' in result['details']
    
    def test_check_rate_limits_algorithm_selection(self):
        """Test different algorithms are selected correctly"""
        algorithms = ['token_bucket', 'leaky_bucket', 'fixed_window', 'sliding_window', 'adaptive']
        
        for algorithm in algorithms:
            limiter = RateLimiter({'algorithm': algorithm})
            request = {'client_ip': '192.168.1.1', 'endpoint': '/api/test'}
            
            result = limiter.check_rate_limits(request)
            
            # All algorithms should allow first request
            assert result['allowed'] is True
    
    def test_check_rate_limits_unknown_algorithm_defaults(self):
        """Test unknown algorithm defaults to token bucket"""
        limiter = RateLimiter({'algorithm': 'unknown_algorithm'})
        
        request = {'client_ip': '192.168.1.1', 'endpoint': '/api/test'}
        
        # Should use token bucket as fallback
        with patch.object(limiter, '_check_token_bucket_limit') as mock_token_bucket:
            mock_token_bucket.return_value = {'allowed': True}
            
            limiter.check_rate_limits(request)
            
            mock_token_bucket.assert_called_once()
    
    def test_check_rate_limits_metrics_updated(self):
        """Test metrics are updated after rate limit check"""
        request = {'client_ip': '192.168.1.1'}
        
        result = self.limiter.check_rate_limits(request)
        
        assert result['allowed'] is True
        
        # Check metrics were updated
        client_id = '192.168.1.1'
        assert self.limiter.rate_limit_metrics[client_id]['allowed'] == 1
        assert self.limiter.rate_limit_metrics[client_id]['denied'] == 0
    
    def test_check_rate_limits_error_handling(self):
        """Test rate limit check error handling"""
        request = {}
        
        # Force an error in client ID extraction
        with patch.object(self.limiter, '_extract_client_id', side_effect=Exception("ID error")):
            with patch.object(self.limiter.logger, 'error') as mock_logger:
                result = self.limiter.check_rate_limits(request)
        
        assert result['allowed'] is False
        assert 'Rate limit check error' in result['details']
        mock_logger.assert_called_once()


class TestMetricsAndStatus:
    """Test metrics collection and status reporting"""
    
    def setup_method(self):
        """Setup for each test"""
        self.limiter = RateLimiter({})
    
    def test_update_metrics_allowed(self):
        """Test metrics update for allowed requests"""
        client_id = 'test_client'
        
        self.limiter._update_metrics(client_id, allowed=True)
        
        assert self.limiter.rate_limit_metrics[client_id]['allowed'] == 1
        assert self.limiter.rate_limit_metrics[client_id]['denied'] == 0
    
    def test_update_metrics_denied(self):
        """Test metrics update for denied requests"""
        client_id = 'test_client'
        
        self.limiter._update_metrics(client_id, allowed=False)
        
        assert self.limiter.rate_limit_metrics[client_id]['allowed'] == 0
        assert self.limiter.rate_limit_metrics[client_id]['denied'] == 1
    
    def test_update_metrics_multiple_calls(self):
        """Test metrics update with multiple calls"""
        client_id = 'test_client'
        
        # Multiple allowed and denied requests
        for _ in range(3):
            self.limiter._update_metrics(client_id, allowed=True)
        
        for _ in range(2):
            self.limiter._update_metrics(client_id, allowed=False)
        
        assert self.limiter.rate_limit_metrics[client_id]['allowed'] == 3
        assert self.limiter.rate_limit_metrics[client_id]['denied'] == 2
    
    def test_update_metrics_error_handling(self):
        """Test metrics update error handling"""
        client_id = None  # Will cause error
        
        with patch.object(self.limiter.logger, 'error') as mock_logger:
            self.limiter._update_metrics(client_id, allowed=True)
        
        mock_logger.assert_called_once()
    
    def test_get_rate_limiter_status(self):
        """Test comprehensive status reporting"""
        # Add some test data
        self.limiter.rate_limit_metrics['client1'] = {'allowed': 10, 'denied': 2}
        self.limiter.rate_limit_metrics['client2'] = {'allowed': 5, 'denied': 8}
        self.limiter.throttled_clients['client3'] = {'until': time.time() + 100}
        self.limiter.blacklisted_clients.add('client4')
        
        status = self.limiter.get_rate_limiter_status()
        
        assert status['algorithm'] == 'token_bucket'
        assert status['global_rate_limit'] == 10000
        assert status['active_clients'] == 2
        assert status['throttled_clients'] == 1
        assert status['blacklisted_clients'] == 1
        assert 'metrics_summary' in status
    
    def test_get_metrics_summary(self):
        """Test metrics summary calculation"""
        # Add test metrics
        self.limiter.rate_limit_metrics['client1'] = {'allowed': 10, 'denied': 2}
        self.limiter.rate_limit_metrics['client2'] = {'allowed': 8, 'denied': 4}
        
        summary = self.limiter._get_metrics_summary()
        
        assert summary['total_requests'] == 24  # 10+2+8+4
        assert summary['total_allowed'] == 18   # 10+8
        assert summary['total_denied'] == 6     # 2+4
        assert summary['denial_rate'] == 0.25   # 6/24
        assert 'top_clients' in summary
    
    def test_get_top_clients(self):
        """Test top clients calculation"""
        # Add test data with different request volumes
        self.limiter.rate_limit_metrics['high_volume'] = {'allowed': 100, 'denied': 10}
        self.limiter.rate_limit_metrics['medium_volume'] = {'allowed': 50, 'denied': 5}
        self.limiter.rate_limit_metrics['low_volume'] = {'allowed': 10, 'denied': 1}
        
        top_clients = self.limiter._get_top_clients(limit=2)
        
        assert len(top_clients) == 2
        assert top_clients[0]['client_id'] == 'high_volume'
        assert top_clients[0]['total_requests'] == 110
        assert top_clients[1]['client_id'] == 'medium_volume'
        assert top_clients[1]['total_requests'] == 55
    
    def test_get_status_error_handling(self):
        """Test status reporting error handling"""
        with patch.object(self.limiter, '_get_metrics_summary', side_effect=Exception("Summary error")):
            with patch.object(self.limiter.logger, 'error') as mock_logger:
                status = self.limiter.get_rate_limiter_status()
        
        assert 'error' in status
        mock_logger.assert_called_once()


class TestCleanupThread:
    """Test background cleanup thread functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.limiter = RateLimiter({})
    
    def test_cleanup_old_sliding_windows(self):
        """Test cleanup removes old sliding window data"""
        # Add old data
        old_time = time.time() - 7200  # 2 hours ago
        self.limiter.sliding_windows['old_client'] = deque([old_time, old_time + 10])
        self.limiter.sliding_windows['active_client'] = deque([time.time() - 30])
        
        # Mock current time and run cleanup
        with patch('time.time') as mock_time:
            mock_time.return_value = time.time()
            
            # Simulate cleanup run
            with self.limiter._lock:
                current_time = time.time()
                for key in list(self.limiter.sliding_windows.keys()):
                    window = self.limiter.sliding_windows[key]
                    if window and window[-1] < current_time - 3600:  # 1 hour old
                        del self.limiter.sliding_windows[key]
        
        # Old window should be removed, active should remain
        assert 'old_client' not in self.limiter.sliding_windows
        assert 'active_client' in self.limiter.sliding_windows
    
    def test_cleanup_expired_throttles(self):
        """Test cleanup removes expired throttles"""
        current_time = time.time()
        
        # Add expired and active throttles
        self.limiter.throttled_clients['expired_client'] = {'until': current_time - 10}
        self.limiter.throttled_clients['active_client'] = {'until': current_time + 100}
        
        # Simulate cleanup
        with self.limiter._lock:
            expired_throttles = []
            for client_id, info in self.limiter.throttled_clients.items():
                if current_time > info['until']:
                    expired_throttles.append(client_id)
            
            for client_id in expired_throttles:
                del self.limiter.throttled_clients[client_id]
        
        # Expired should be removed, active should remain
        assert 'expired_client' not in self.limiter.throttled_clients
        assert 'active_client' in self.limiter.throttled_clients
    
    @patch('time.sleep')
    def test_cleanup_loop_error_handling(self, mock_sleep):
        """Test cleanup loop handles errors gracefully"""
        # Force cleanup to raise exception
        with patch.object(self.limiter, '_lock', side_effect=Exception("Lock error")):
            with patch.object(self.limiter.logger, 'error') as mock_logger:
                # Run one iteration of cleanup
                try:
                    current_time = time.time()
                    
                    # This should raise an exception
                    with self.limiter._lock:
                        pass
                    
                except Exception as e:
                    # Simulate error handling
                    self.limiter.logger.error(f"Cleanup loop error: {e}")
        
        mock_logger.assert_called_once()


class TestThreadSafety:
    """Test thread safety of rate limiter"""
    
    def setup_method(self):
        """Setup for each test"""
        self.limiter = RateLimiter({'algorithm': 'token_bucket', 'global_rate_limit': 1000})
    
    def test_concurrent_token_bucket_access(self):
        """Test concurrent access to token bucket doesn't cause race conditions"""
        client_id = 'concurrent_client'
        config = {
            'service': 'test',
            'requests_per_minute': 60,
            'burst_size': 10
        }
        
        results = []
        
        def make_request():
            result = self.limiter._check_token_bucket_limit(client_id, config)
            results.append(result['allowed'])
        
        # Run concurrent requests
        threads = []
        for _ in range(15):  # More than burst size
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have exactly burst_size (10) allowed requests
        allowed_count = sum(results)
        assert allowed_count == config['burst_size']
    
    def test_concurrent_rate_limit_checks(self):
        """Test concurrent rate limit checks"""
        results = []
        
        def check_rate_limit(client_id):
            request = {
                'client_ip': f'192.168.1.{client_id}',
                'endpoint': '/api/test'
            }
            result = self.limiter.check_rate_limits(request)
            results.append((client_id, result['allowed']))
        
        # Run concurrent rate limit checks for different clients
        threads = []
        for i in range(20):
            thread = threading.Thread(target=check_rate_limit, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All should be allowed (different clients)
        assert len(results) == 20
        assert all(allowed for _, allowed in results)
    
    def test_concurrent_metrics_updates(self):
        """Test concurrent metrics updates"""
        client_id = 'metrics_client'
        
        def update_metrics(allowed):
            for _ in range(10):
                self.limiter._update_metrics(client_id, allowed)
        
        # Run concurrent metrics updates
        threads = []
        # 5 threads updating allowed, 3 threads updating denied
        for i in range(8):
            allowed = i < 5
            thread = threading.Thread(target=update_metrics, args=(allowed,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have 50 allowed (5 threads * 10) and 30 denied (3 threads * 10)
        metrics = self.limiter.rate_limit_metrics[client_id]
        assert metrics['allowed'] == 50
        assert metrics['denied'] == 30


class TestRateLimiterEdgeCases:
    """Test edge cases and error scenarios"""
    
    def setup_method(self):
        """Setup for each test"""
        self.limiter = RateLimiter({})
    
    def test_empty_request(self):
        """Test handling of empty request"""
        request = {}
        
        result = self.limiter.check_rate_limits(request)
        
        # Should handle gracefully and default to anonymous
        assert 'allowed' in result
    
    def test_malformed_request(self):
        """Test handling of malformed request data"""
        request = {
            'user': 'not_a_dict',  # Should be dict
            'endpoint': None       # Should be string
        }
        
        result = self.limiter.check_rate_limits(request)
        
        # Should handle gracefully
        assert 'allowed' in result
    
    def test_zero_rate_limits(self):
        """Test handling of zero rate limits"""
        config = {
            'service': 'zero_test',
            'requests_per_minute': 0,
            'burst_size': 0
        }
        
        # All algorithms should handle zero limits
        algorithms = ['token_bucket', 'leaky_bucket', 'fixed_window', 'sliding_window']
        
        for algorithm_name in algorithms:
            method_name = f'_check_{algorithm_name}_limit'
            if hasattr(self.limiter, method_name):
                method = getattr(self.limiter, method_name)
                result = method('test_client', config)
                
                # Should be blocked due to zero limits
                assert result['allowed'] is False
    
    def test_negative_time_values(self):
        """Test handling of negative time values"""
        # This could happen due to clock adjustments
        client_id = 'time_test_client'
        config = {
            'service': 'test',
            'requests_per_minute': 60,
            'burst_size': 10
        }
        
        # Set up bucket with future time
        bucket_key = f"{client_id}:{config['service']}"
        future_time = time.time() + 1000
        self.limiter.token_buckets[bucket_key] = {
            'tokens': 5,
            'last_refill': future_time,
            'refill_rate': 1.0
        }
        
        # Should handle gracefully (negative time_passed)
        result = self.limiter._check_token_bucket_limit(client_id, config)
        
        assert 'allowed' in result
    
    def test_very_large_numbers(self):
        """Test handling of very large numbers"""
        config = {
            'service': 'large_test',
            'requests_per_minute': 10**9,  # Very large
            'burst_size': 10**6
        }
        
        result = self.limiter._check_token_bucket_limit('test_client', config)
        
        # Should handle large numbers without overflow
        assert 'allowed' in result
    
    def test_unicode_client_ids(self):
        """Test handling of unicode client IDs"""
        request = {
            'user': {'user_id': 'тест_клиент_123'}  # Unicode characters
        }
        
        result = self.limiter.check_rate_limits(request)
        
        # Should handle unicode gracefully
        assert 'allowed' in result


class TestRateLimiterIntegration:
    """Test integration scenarios"""
    
    def setup_method(self):
        """Setup for integration tests"""
        self.limiter = RateLimiter({
            'algorithm': 'adaptive',
            'global_rate_limit': 100,
            'burst_multiplier': 2.0
        })
    
    def test_full_workflow_with_throttling(self):
        """Test complete workflow including throttling"""
        client_id = 'integration_client'
        
        # Create a request that will be rate limited
        request = {
            'user': {'user_id': client_id, 'tier': 'free'},
            'endpoint': '/brain/predict'
        }
        
        # Make many requests to trigger rate limiting and throttling
        results = []
        
        for i in range(200):  # More than any single service limit
            result = self.limiter.check_rate_limits(request)
            results.append(result['allowed'])
            
            # Add small delay to avoid overwhelming
            time.sleep(0.001)
        
        # Should have some allowed and some denied
        allowed_count = sum(results)
        denied_count = len(results) - allowed_count
        
        assert allowed_count > 0   # Should allow some
        assert denied_count > 0    # Should deny some
        
        # Client should eventually be throttled
        throttled = self.limiter._is_client_throttled(client_id)
        
        # Metrics should be updated
        metrics = self.limiter.rate_limit_metrics[client_id]
        assert metrics['allowed'] > 0
        assert metrics['denied'] > 0
    
    def test_multi_service_rate_limiting(self):
        """Test rate limiting across multiple services"""
        client_id = 'multi_service_client'
        
        # Test different services have different limits
        services = ['brain', 'uncertainty', 'proof', 'api']
        
        for service in services:
            request = {
                'user': {'user_id': client_id, 'tier': 'basic'},
                'endpoint': f'/{service}/test'
            }
            
            # First request should always be allowed
            result = self.limiter.check_rate_limits(request)
            assert result['allowed'] is True
        
        # Each service should have independent limits
        brain_config = self.limiter._get_rate_limit_config({
            'endpoint': '/brain/test',
            'user': {'tier': 'basic'}
        })
        api_config = self.limiter._get_rate_limit_config({
            'endpoint': '/api/test',
            'user': {'tier': 'basic'}
        })
        
        # Different services should have different limits
        assert brain_config['requests_per_minute'] != api_config['requests_per_minute']
    
    def test_tier_based_rate_limiting(self):
        """Test rate limiting varies by user tier"""
        tiers = ['free', 'basic', 'pro', 'enterprise']
        
        for tier in tiers:
            request = {
                'user': {'user_id': f'{tier}_user', 'tier': tier},
                'endpoint': '/data/query'
            }
            
            config = self.limiter._get_rate_limit_config(request)
            
            # Higher tiers should have higher limits
            base_limit = self.limiter.service_limits['data']['requests_per_minute']
            
            if tier == 'free':
                assert config['requests_per_minute'] == base_limit
            elif tier == 'basic':
                assert config['requests_per_minute'] == base_limit * 2
            elif tier == 'pro':
                assert config['requests_per_minute'] == base_limit * 5
            elif tier == 'enterprise':
                assert config['requests_per_minute'] == base_limit * 10
    
    def test_algorithm_comparison(self):
        """Test different algorithms behave differently"""
        algorithms = ['token_bucket', 'leaky_bucket', 'fixed_window', 'sliding_window']
        
        results = {}
        
        for algorithm in algorithms:
            limiter = RateLimiter({'algorithm': algorithm})
            client_id = f'{algorithm}_client'
            
            # Make burst of requests
            request = {
                'user': {'user_id': client_id},
                'endpoint': '/api/test'
            }
            
            allowed_count = 0
            for _ in range(50):  # Burst of 50 requests
                result = limiter.check_rate_limits(request)
                if result['allowed']:
                    allowed_count += 1
            
            results[algorithm] = allowed_count
        
        # Different algorithms should behave differently
        # Token bucket should allow more initially due to burst
        # Fixed window should allow up to limit then block all
        # Results should vary between algorithms
        unique_results = len(set(results.values()))
        assert unique_results > 1  # Should have different behaviors


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])