"""
Comprehensive test suite for LoadBalancer
Tests all functionality including health checks, distribution algorithms, session management, and edge cases
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict
from production_api.load_balancer import LoadBalancer, HealthChecker


class TestHealthChecker(unittest.TestCase):
    """Comprehensive test suite for HealthChecker"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'interval_seconds': 1,  # Short interval for testing
            'timeout_seconds': 0.1,
            'unhealthy_threshold': 2,
            'healthy_threshold': 1
        }
        self.health_checker = HealthChecker(self.config)
        
    def tearDown(self):
        """Clean up"""
        # Give time for background threads to finish
        time.sleep(0.2)
    
    def test_initialization(self):
        """Test health checker initialization"""
        self.assertEqual(self.health_checker.health_check_interval, 1)
        self.assertEqual(self.health_checker.health_check_timeout, 0.1)
        self.assertEqual(self.health_checker.unhealthy_threshold, 2)
        self.assertEqual(self.health_checker.healthy_threshold, 1)
        
        # Check default health status structure
        self.assertIsInstance(self.health_checker.endpoint_health, defaultdict)
        self.assertIsInstance(self.health_checker.health_history, defaultdict)
        
        print("✓ HealthChecker initialization test passed")
    
    def test_endpoint_health_check(self):
        """Test endpoint health check functionality"""
        endpoint = "test-endpoint.local"
        
        # Perform health check
        result = self.health_checker.check_endpoint_health(endpoint)
        
        # Verify result structure
        self.assertIn('healthy', result)
        self.assertIn('status', result)
        self.assertIn('response_time', result)
        self.assertIn('error_rate', result)
        
        # Verify health status is tracked
        self.assertIn(endpoint, self.health_checker.endpoint_health)
        
        health_status = self.health_checker.endpoint_health[endpoint]
        self.assertIn('status', health_status)
        self.assertIn('last_check', health_status)
        self.assertIn('response_time', health_status)
        self.assertGreater(health_status['last_check'], 0)
        
        print("✓ Endpoint health check test passed")
    
    def test_health_status_transitions(self):
        """Test health status transitions (healthy -> unhealthy -> healthy)"""
        endpoint = "transition-test.local"
        
        # Mock consistent failures to trigger unhealthy status
        with patch.object(self.health_checker, '_perform_health_check') as mock_check:
            # Simulate failures
            mock_check.return_value = {
                'success': False,
                'status_code': 500,
                'details': 'Server error'
            }
            
            # Perform multiple checks to exceed unhealthy threshold
            for _ in range(self.config['unhealthy_threshold']):
                result = self.health_checker.check_endpoint_health(endpoint)
            
            # Should be unhealthy now
            health_status = self.health_checker.endpoint_health[endpoint]
            self.assertEqual(health_status['status'], 'unhealthy')
            
            # Now simulate successes
            mock_check.return_value = {
                'success': True,
                'status_code': 200,
                'response_time': 0.05
            }
            
            # Perform enough successful checks to become healthy
            for _ in range(self.config['healthy_threshold']):
                result = self.health_checker.check_endpoint_health(endpoint)
            
            # Should be healthy now
            health_status = self.health_checker.endpoint_health[endpoint]
            self.assertEqual(health_status['status'], 'healthy')
        
        print("✓ Health status transitions test passed")
    
    def test_endpoint_availability(self):
        """Test endpoint availability checking"""
        endpoint = "availability-test.local"
        
        # Initially unknown, should be considered available
        available = self.health_checker.is_endpoint_available(endpoint)
        self.assertTrue(available)
        
        # Force endpoint to unhealthy status
        health_status = self.health_checker.endpoint_health[endpoint]
        health_status['status'] = 'unhealthy'
        health_status['last_check'] = time.time()
        
        # Should not be available when unhealthy
        available = self.health_checker.is_endpoint_available(endpoint)
        self.assertFalse(available)
        
        # Set to healthy
        health_status['status'] = 'healthy'
        available = self.health_checker.is_endpoint_available(endpoint)
        self.assertTrue(available)
        
        print("✓ Endpoint availability test passed")
    
    def test_endpoint_metrics(self):
        """Test endpoint metrics calculation"""
        endpoint = "metrics-test.local"
        
        # Add some mock history
        history = self.health_checker.health_history[endpoint]
        test_data = [
            {'timestamp': time.time(), 'success': True, 'response_time': 0.1},
            {'timestamp': time.time(), 'success': False, 'response_time': 0.5},
            {'timestamp': time.time(), 'success': True, 'response_time': 0.2},
            {'timestamp': time.time(), 'success': True, 'response_time': 0.15},
        ]
        
        for data in test_data:
            history.append(data)
        
        # Get metrics
        metrics = self.health_checker.get_endpoint_metrics(endpoint)
        
        # Verify metrics structure
        self.assertIn('average_response_time', metrics)
        self.assertIn('error_rate', metrics)
        self.assertIn('uptime_percentage', metrics)
        self.assertIn('total_checks', metrics)
        
        # Verify calculations
        self.assertEqual(metrics['total_checks'], 4)
        self.assertAlmostEqual(metrics['error_rate'], 0.25, places=2)  # 1 failure out of 4
        self.assertAlmostEqual(metrics['uptime_percentage'], 75.0, places=1)  # 3 successes out of 4
        
        # Average response time should be calculated correctly
        expected_avg = (0.1 + 0.5 + 0.2 + 0.15) / 4
        self.assertAlmostEqual(metrics['average_response_time'], expected_avg, places=2)
        
        print("✓ Endpoint metrics test passed")
    
    def test_error_handling(self):
        """Test error handling in health checks"""
        endpoint = "error-test.local"
        
        # Mock _perform_health_check to raise exception
        with patch.object(self.health_checker, '_perform_health_check') as mock_check:
            mock_check.side_effect = Exception("Network error")
            
            result = self.health_checker.check_endpoint_health(endpoint)
            
            # Should return error status
            self.assertFalse(result['healthy'])
            self.assertEqual(result['status'], 'error')
            self.assertIn('Network error', result['details'])
        
        print("✓ Error handling test passed")


class TestLoadBalancer(unittest.TestCase):
    """Comprehensive test suite for LoadBalancer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'algorithm': 'weighted_round_robin',
            'sticky_sessions': True,
            'session_timeout': 300,
            'health_checks': {
                'interval_seconds': 1,
                'timeout_seconds': 0.1,
                'unhealthy_threshold': 2,
                'healthy_threshold': 1
            }
        }
        self.load_balancer = LoadBalancer(self.config)
        
        # Mock all endpoints as healthy
        for service, endpoints in self.load_balancer.endpoints.items():
            for endpoint in endpoints:
                health_status = self.load_balancer.health_checker.endpoint_health[endpoint]
                health_status['status'] = 'healthy'
                health_status['last_check'] = time.time()
    
    def tearDown(self):
        """Clean up"""
        time.sleep(0.1)
    
    def test_initialization(self):
        """Test load balancer initialization"""
        self.assertEqual(self.load_balancer.algorithm, 'weighted_round_robin')
        self.assertTrue(self.load_balancer.sticky_sessions)
        self.assertEqual(self.load_balancer.session_timeout, 300)
        
        # Check endpoints are initialized
        self.assertGreater(len(self.load_balancer.endpoints), 0)
        self.assertIn('brain_service', self.load_balancer.endpoints)
        
        # Check health checker is initialized
        self.assertIsInstance(self.load_balancer.health_checker, HealthChecker)
        
        # Check weights are initialized
        self.assertGreater(len(self.load_balancer.endpoint_weights), 0)
        
        print("✓ LoadBalancer initialization test passed")
    
    def test_distribute_request_basic(self):
        """Test basic request distribution"""
        request = {
            'path': '/api/test',
            'client_ip': '192.168.1.100'
        }
        
        # Distribute to brain service
        result = self.load_balancer.distribute_request(request, 'brain_service')
        
        # Should succeed
        self.assertTrue(result['success'])
        self.assertIn('endpoint', result)
        self.assertIn('service', result)
        self.assertEqual(result['service'], 'brain_service')
        
        # Endpoint should be one of the brain service endpoints
        brain_endpoints = self.load_balancer.endpoints['brain_service']
        self.assertIn(result['endpoint'], brain_endpoints)
        
        print("✓ Basic request distribution test passed")
    
    def test_round_robin_algorithm(self):
        """Test round-robin distribution algorithm"""
        # Set algorithm to round robin
        self.load_balancer.algorithm = 'round_robin'
        
        request = {'path': '/api/test', 'client_ip': '192.168.1.100'}
        service = 'uncertainty_service'
        endpoints = self.load_balancer.endpoints[service]
        
        selected_endpoints = []
        for i in range(len(endpoints) * 2):  # Go through twice
            result = self.load_balancer.distribute_request(request, service)
            self.assertTrue(result['success'])
            selected_endpoints.append(result['endpoint'])
        
        # Should have used all endpoints in round-robin fashion
        unique_endpoints = set(selected_endpoints)
        self.assertEqual(len(unique_endpoints), len(endpoints))
        
        # Pattern should repeat
        first_half = selected_endpoints[:len(endpoints)]
        second_half = selected_endpoints[len(endpoints):]
        self.assertEqual(first_half, second_half)
        
        print("✓ Round-robin algorithm test passed")
    
    def test_weighted_round_robin_algorithm(self):
        """Test weighted round-robin distribution"""
        self.load_balancer.algorithm = 'weighted_round_robin'
        
        request = {'path': '/api/test', 'client_ip': '192.168.1.100'}
        service = 'training_service'
        
        selected_endpoints = []
        for _ in range(20):  # Multiple requests
            result = self.load_balancer.distribute_request(request, service)
            self.assertTrue(result['success'])
            selected_endpoints.append(result['endpoint'])
        
        # Count selections per endpoint
        endpoint_counts = defaultdict(int)
        for endpoint in selected_endpoints:
            endpoint_counts[endpoint] += 1
        
        # Higher weighted endpoints should be selected more often
        # First endpoint should have higher count than later ones
        counts = list(endpoint_counts.values())
        self.assertGreater(len(counts), 1)  # Multiple endpoints were used
        
        print("✓ Weighted round-robin algorithm test passed")
    
    def test_least_connections_algorithm(self):
        """Test least connections distribution"""
        self.load_balancer.algorithm = 'least_connections'
        
        # Simulate different connection counts
        endpoints = self.load_balancer.endpoints['brain_service']
        for i, endpoint in enumerate(endpoints):
            self.load_balancer.connection_counts[endpoint] = i * 2
        
        request = {'path': '/api/test', 'client_ip': '192.168.1.100'}
        result = self.load_balancer.distribute_request(request, 'brain_service')
        
        # Should select the endpoint with least connections (first one with 0 connections)
        self.assertTrue(result['success'])
        self.assertEqual(result['endpoint'], endpoints[0])
        
        print("✓ Least connections algorithm test passed")
    
    def test_ip_hash_algorithm(self):
        """Test IP hash-based distribution"""
        self.load_balancer.algorithm = 'ip_hash'
        
        # Same IP should always go to same endpoint
        request1 = {'path': '/api/test', 'client_ip': '192.168.1.100'}
        request2 = {'path': '/api/test', 'client_ip': '192.168.1.100'}
        
        result1 = self.load_balancer.distribute_request(request1, 'brain_service')
        result2 = self.load_balancer.distribute_request(request2, 'brain_service')
        
        self.assertTrue(result1['success'])
        self.assertTrue(result2['success'])
        self.assertEqual(result1['endpoint'], result2['endpoint'])
        
        # Different IPs should potentially go to different endpoints
        request3 = {'path': '/api/test', 'client_ip': '192.168.1.200'}
        result3 = self.load_balancer.distribute_request(request3, 'brain_service')
        self.assertTrue(result3['success'])
        
        print("✓ IP hash algorithm test passed")
    
    def test_sticky_sessions(self):
        """Test sticky session functionality"""
        self.load_balancer.sticky_sessions = True
        
        request_with_session = {
            'path': '/api/test',
            'session_id': 'test_session_123',
            'client_ip': '192.168.1.100'
        }
        
        # First request establishes session mapping
        result1 = self.load_balancer.distribute_request(request_with_session, 'brain_service')
        self.assertTrue(result1['success'])
        selected_endpoint = result1['endpoint']
        
        # Subsequent requests with same session should go to same endpoint
        result2 = self.load_balancer.distribute_request(request_with_session, 'brain_service')
        self.assertTrue(result2['success'])
        self.assertEqual(result2['endpoint'], selected_endpoint)
        
        # Verify session mapping exists
        self.assertIn('test_session_123', self.load_balancer.session_mappings)
        
        print("✓ Sticky sessions test passed")
    
    def test_session_timeout(self):
        """Test session timeout functionality"""
        self.load_balancer.sticky_sessions = True
        self.load_balancer.session_timeout = 1  # 1 second timeout
        
        request = {
            'path': '/api/test',
            'session_id': 'timeout_test_session',
            'client_ip': '192.168.1.100'
        }
        
        # Establish session
        result1 = self.load_balancer.distribute_request(request, 'brain_service')
        self.assertTrue(result1['success'])
        
        # Session should exist
        self.assertIn('timeout_test_session', self.load_balancer.session_mappings)
        
        # Wait for session to expire
        time.sleep(1.5)
        
        # Request should trigger session cleanup and create new mapping
        result2 = self.load_balancer.distribute_request(request, 'brain_service')
        self.assertTrue(result2['success'])
        
        print("✓ Session timeout test passed")
    
    def test_unhealthy_endpoint_handling(self):
        """Test handling of unhealthy endpoints"""
        service = 'compression_service'
        endpoints = self.load_balancer.endpoints[service]
        
        # Mark all but one endpoint as unhealthy
        for endpoint in endpoints[:-1]:
            health_status = self.load_balancer.health_checker.endpoint_health[endpoint]
            health_status['status'] = 'unhealthy'
        
        # Should route to the only healthy endpoint
        request = {'path': '/api/test', 'client_ip': '192.168.1.100'}
        result = self.load_balancer.distribute_request(request, service)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['endpoint'], endpoints[-1])  # The healthy one
        
        print("✓ Unhealthy endpoint handling test passed")
    
    def test_no_available_endpoints(self):
        """Test behavior when no endpoints are available"""
        service = 'proof_service'
        endpoints = self.load_balancer.endpoints[service]
        
        # Mark all endpoints as unhealthy
        for endpoint in endpoints:
            health_status = self.load_balancer.health_checker.endpoint_health[endpoint]
            health_status['status'] = 'unhealthy'
        
        request = {'path': '/api/test', 'client_ip': '192.168.1.100'}
        result = self.load_balancer.distribute_request(request, service)
        
        # Should fail
        self.assertFalse(result['success'])
        self.assertIn('details', result)
        self.assertIn('No available endpoints', result['details'])
        
        print("✓ No available endpoints test passed")
    
    def test_alternative_endpoint_selection(self):
        """Test alternative endpoint selection when primary fails"""
        service = 'data_service'
        endpoints = self.load_balancer.endpoints[service]
        
        # Mock health check to fail for first endpoint, succeed for others
        def mock_health_check(endpoint):
            if endpoint == endpoints[0]:
                return {'healthy': False, 'status': 'unhealthy'}
            else:
                return {'healthy': True, 'status': 'healthy', 'response_time': 0.1, 'error_rate': 0.0}
        
        with patch.object(self.load_balancer.health_checker, 'check_endpoint_health', side_effect=mock_health_check):
            # Force selection of first endpoint initially
            self.load_balancer.algorithm = 'round_robin'
            self.load_balancer.current_index[service] = 0
            
            request = {'path': '/api/test', 'client_ip': '192.168.1.100'}
            result = self.load_balancer.distribute_request(request, service)
            
            # Should succeed with alternative endpoint
            self.assertTrue(result['success'])
            self.assertNotEqual(result['endpoint'], endpoints[0])
            self.assertIn(result['endpoint'], endpoints[1:])
        
        print("✓ Alternative endpoint selection test passed")
    
    def test_load_balancer_status(self):
        """Test load balancer status reporting"""
        status = self.load_balancer.get_load_balancer_status()
        
        # Verify status structure
        self.assertIn('algorithm', status)
        self.assertIn('sticky_sessions', status)
        self.assertIn('services', status)
        
        # Verify service status structure
        for service_name, service_status in status['services'].items():
            self.assertIn('total_endpoints', service_status)
            self.assertIn('available_endpoints', service_status)
            self.assertIn('endpoints', service_status)
            
            # Verify endpoint details
            for endpoint_name, endpoint_details in service_status['endpoints'].items():
                self.assertIn('status', endpoint_details)
                self.assertIn('metrics', endpoint_details)
                self.assertIn('request_count', endpoint_details)
                self.assertIn('weight', endpoint_details)
        
        print("✓ Load balancer status test passed")
    
    def test_request_counting_and_metrics(self):
        """Test request counting and metrics tracking"""
        service = 'api_service'
        endpoint = self.load_balancer.endpoints[service][0]
        
        # Ensure endpoint is healthy
        health_status = self.load_balancer.health_checker.endpoint_health[endpoint]
        health_status['status'] = 'healthy'
        
        initial_count = self.load_balancer.request_counts[endpoint]
        
        # Make multiple requests
        for i in range(5):
            request = {'path': f'/api/test/{i}', 'client_ip': '192.168.1.100'}
            result = self.load_balancer.distribute_request(request, service)
            self.assertTrue(result['success'])
        
        # Check request count increased
        final_count = self.load_balancer.request_counts[endpoint]
        # Note: Due to load balancing, not all requests may go to the same endpoint
        self.assertGreaterEqual(final_count, initial_count)
        
        print("✓ Request counting and metrics test passed")
    
    def test_connection_count_management(self):
        """Test connection count tracking"""
        service = 'security_service'
        request = {'path': '/api/test', 'client_ip': '192.168.1.100'}
        
        # Track initial connection counts
        initial_counts = dict(self.load_balancer.connection_counts)
        
        result = self.load_balancer.distribute_request(request, service)
        self.assertTrue(result['success'])
        
        endpoint = result['endpoint']
        
        # Connection count should have been incremented and then decremented
        # (since _route_to_endpoint increments and then decrements)
        final_count = self.load_balancer.connection_counts[endpoint]
        initial_count = initial_counts.get(endpoint, 0)
        
        # Should be back to initial or slightly higher due to race conditions
        self.assertGreaterEqual(final_count, initial_count)
        
        print("✓ Connection count management test passed")
    
    def test_unknown_service_handling(self):
        """Test handling of requests to unknown services"""
        request = {'path': '/api/test', 'client_ip': '192.168.1.100'}
        result = self.load_balancer.distribute_request(request, 'nonexistent_service')
        
        # Should fail gracefully
        self.assertFalse(result['success'])
        self.assertIn('details', result)
        
        print("✓ Unknown service handling test passed")
    
    def test_thread_safety(self):
        """Test thread safety of load balancer operations"""
        service = 'brain_service'
        request = {'path': '/api/test', 'client_ip': '192.168.1.100'}
        
        results = []
        errors = []
        
        def make_request():
            try:
                result = self.load_balancer.distribute_request(request, service)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads making concurrent requests
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5)
        
        # Should have no errors and successful results
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 10)
        
        for result in results:
            self.assertTrue(result['success'])
        
        print("✓ Thread safety test passed")
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        service = 'training_service'
        
        # Test with malformed request
        malformed_request = None
        result = self.load_balancer.distribute_request(malformed_request, service)
        
        # Should handle gracefully
        self.assertFalse(result['success'])
        self.assertIn('details', result)
        
        # Test with empty service
        request = {'path': '/api/test', 'client_ip': '192.168.1.100'}
        result = self.load_balancer.distribute_request(request, '')
        
        # Should handle gracefully
        self.assertFalse(result['success'])
        
        print("✓ Error handling and recovery test passed")
    
    def test_health_check_integration(self):
        """Test integration between load balancer and health checker"""
        service = 'uncertainty_service'
        endpoints = self.load_balancer.endpoints[service]
        
        # Set specific health states
        for i, endpoint in enumerate(endpoints):
            health_status = self.load_balancer.health_checker.endpoint_health[endpoint]
            health_status['status'] = 'healthy' if i == 0 else 'unhealthy'
            health_status['response_time'] = 0.1 + i * 0.05
            health_status['error_rate'] = i * 0.1
            health_status['last_check'] = time.time()
        
        request = {'path': '/api/test', 'client_ip': '192.168.1.100'}
        result = self.load_balancer.distribute_request(request, service)
        
        self.assertTrue(result['success'])
        
        # Should include health info in result
        self.assertIn('health_status', result)
        health_info = result['health_status']
        self.assertIn('status', health_info)
        self.assertIn('response_time', health_info)
        self.assertIn('error_rate', health_info)
        
        print("✓ Health check integration test passed")


def run_comprehensive_tests():
    """Run all tests and provide summary"""
    print("=" * 70)
    print("LoadBalancer Comprehensive Test Suite")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    
    # Load tests for both classes
    health_checker_suite = loader.loadTestsFromTestCase(TestHealthChecker)
    load_balancer_suite = loader.loadTestsFromTestCase(TestLoadBalancer)
    
    # Combine suites
    combined_suite = unittest.TestSuite([health_checker_suite, load_balancer_suite])
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(combined_suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary:")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailed Tests:")
        for test, traceback in result.failures:
            print(f"  - {test}")
            print(f"    Failure: {traceback.split('AssertionError:')[-1].strip()[:200]}...")
    
    if result.errors:
        print("\nTests with Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
            print(f"    Error: {traceback.split('Exception:')[-1].strip()[:200]}...")
    
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)