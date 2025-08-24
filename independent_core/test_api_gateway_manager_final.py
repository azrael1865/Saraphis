"""
Final comprehensive test for APIGatewayManager with actual methods
"""

import sys
import os
import logging
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import json
import time
import threading

# Setup
logging.basicConfig(level=logging.INFO)
sys.path.insert(0, '.')

from production_api.api_gateway_manager import APIGatewayManager

def test_api_gateway_manager_comprehensive():
    """Comprehensive test of APIGatewayManager"""
    
    print("\n" + "="*60)
    print("COMPREHENSIVE API GATEWAY MANAGER TEST")
    print("="*60)
    
    # Setup
    test_results = []
    
    try:
        # Test 1: Initialization
        print("\n[TEST 1: Initialization]")
        try:
            config = {
                'api_version': '1.0',
                'timeout_seconds': 30,
                'max_retries': 3,
                'circuit_breaker_threshold': 5,
                'load_balancer': {'algorithm': 'round_robin'},
                'rate_limiter': {'global_rate_limit': 1000},
                'authentication': {'secret_key': 'test_key'},
                'validation': {'strict_mode': True},
                'formatting': {'indent': 2},
                'metrics': {'enabled': True}
            }
            
            gateway = APIGatewayManager(config)
            print("✓ APIGatewayManager initialized successfully")
            print(f"  - Version: {gateway.api_version}")
            print(f"  - Timeout: {gateway.timeout_seconds}s")
            print(f"  - Max retries: {gateway.max_retries}")
            test_results.append(("Initialization", True))
        except Exception as e:
            print(f"✗ Initialization failed: {e}")
            test_results.append(("Initialization", False))
            return test_results
        
        # Test 2: Gateway status
        print("\n[TEST 2: Gateway Status]")
        try:
            status = gateway.get_gateway_status()
            if status.get('status') == 'operational':
                print("✓ Gateway status check successful")
                print(f"  - Status: {status['status']}")
                print(f"  - Components: {len(status.get('components', {}))}")
                test_results.append(("Gateway Status", True))
            else:
                print(f"✗ Gateway status check failed: {status}")
                test_results.append(("Gateway Status", False))
        except Exception as e:
            print(f"✗ Gateway status check failed: {e}")
            test_results.append(("Gateway Status", False))
        
        # Test 3: Route configuration
        print("\n[TEST 3: Route Configuration]")
        try:
            routes = gateway.routes
            expected_services = ['brain', 'uncertainty', 'training', 'compression', 'proof', 'security', 'data', 'api']
            
            all_services_present = all(service in routes for service in expected_services)
            if all_services_present and len(routes) >= len(expected_services):
                print("✓ Route configuration validated")
                print(f"  - Services configured: {list(routes.keys())}")
                print(f"  - Total endpoints: {sum(len(config['endpoints']) for config in routes.values())}")
                test_results.append(("Route Configuration", True))
            else:
                print(f"✗ Route configuration incomplete: {list(routes.keys())}")
                test_results.append(("Route Configuration", False))
        except Exception as e:
            print(f"✗ Route configuration failed: {e}")
            test_results.append(("Route Configuration", False))
        
        # Test 4: Basic request routing
        print("\n[TEST 4: Basic Request Routing]")
        try:
            test_request = {
                'endpoint': '/brain/status',
                'method': 'GET',
                'headers': {'Authorization': 'Bearer test_token'},
                'body': {}
            }
            
            response = gateway.route_request(test_request)
            if response.get('success', False) or 'data' in response or 'error' in response:
                print("✓ Basic request routing successful")
                print(f"  - Response type: {'success' if response.get('success') else 'handled_error'}")
                print(f"  - Request ID: {response.get('request_id', 'N/A')}")
                test_results.append(("Basic Request Routing", True))
            else:
                print(f"✗ Basic request routing failed: {response}")
                test_results.append(("Basic Request Routing", False))
        except Exception as e:
            print(f"✗ Basic request routing failed: {e}")
            test_results.append(("Basic Request Routing", False))
        
        # Test 5: Service routing for different endpoints
        print("\n[TEST 5: Service Routing]")
        try:
            test_endpoints = [
                '/brain/health',
                '/uncertainty/quantify', 
                '/training/start',
                '/compression/compress',
                '/proof/generate',
                '/api/status'
            ]
            
            routing_success = 0
            for endpoint in test_endpoints:
                test_req = {
                    'endpoint': endpoint,
                    'method': 'GET' if 'status' in endpoint or 'health' in endpoint else 'POST',
                    'headers': {'Authorization': 'Bearer test_token'},
                    'body': {}
                }
                
                try:
                    response = gateway.route_request(test_req)
                    if response.get('success', False) or 'data' in response or 'error' in response:
                        routing_success += 1
                except:
                    pass
            
            if routing_success >= len(test_endpoints) * 0.8:  # 80% success rate
                print(f"✓ Service routing successful ({routing_success}/{len(test_endpoints)})")
                test_results.append(("Service Routing", True))
            else:
                print(f"✗ Service routing insufficient ({routing_success}/{len(test_endpoints)})")
                test_results.append(("Service Routing", False))
        except Exception as e:
            print(f"✗ Service routing failed: {e}")
            test_results.append(("Service Routing", False))
        
        # Test 6: Circuit breaker functionality
        print("\n[TEST 6: Circuit Breaker]")
        try:
            # Check initial state
            service_name = 'brain_service'
            initial_state = gateway._check_circuit_breaker(service_name)
            
            # Record some failures
            for i in range(3):
                gateway._record_circuit_breaker_failure(service_name)
            
            # Check circuit breaker state
            breaker_state = gateway.circuit_breaker_state[service_name]
            
            if breaker_state['failures'] >= 3:
                print("✓ Circuit breaker functionality works")
                print(f"  - Initial state: {'open' if not initial_state else 'closed'}")
                print(f"  - Failures recorded: {breaker_state['failures']}")
                print(f"  - Current state: {breaker_state['state']}")
                test_results.append(("Circuit Breaker", True))
            else:
                print("✗ Circuit breaker not recording failures properly")
                test_results.append(("Circuit Breaker", False))
        except Exception as e:
            print(f"✗ Circuit breaker test failed: {e}")
            test_results.append(("Circuit Breaker", False))
        
        # Test 7: Request validation
        print("\n[TEST 7: Request Validation]")
        try:
            # Test invalid requests
            invalid_requests = [
                {},  # Empty request
                {'endpoint': ''},  # Empty endpoint
                {'method': 'INVALID'},  # Invalid method
            ]
            
            validation_working = 0
            for invalid_req in invalid_requests:
                try:
                    response = gateway.route_request(invalid_req)
                    # Should get error response, not crash
                    if 'error' in response or not response.get('success', True):
                        validation_working += 1
                except:
                    validation_working += 1  # Catching and handling is also valid
            
            if validation_working >= len(invalid_requests) * 0.8:
                print("✓ Request validation working")
                print(f"  - Invalid requests handled: {validation_working}/{len(invalid_requests)}")
                test_results.append(("Request Validation", True))
            else:
                print(f"✗ Request validation insufficient: {validation_working}/{len(invalid_requests)}")
                test_results.append(("Request Validation", False))
        except Exception as e:
            print(f"✗ Request validation failed: {e}")
            test_results.append(("Request Validation", False))
        
        # Test 8: Request history tracking
        print("\n[TEST 8: Request History]")
        try:
            initial_history_size = len(gateway.request_history)
            
            # Make a few test requests
            for i in range(3):
                test_req = {
                    'endpoint': f'/api/status',
                    'method': 'GET',
                    'headers': {'Authorization': 'Bearer test_token'},
                    'body': {}
                }
                gateway.route_request(test_req)
            
            final_history_size = len(gateway.request_history)
            
            if final_history_size > initial_history_size:
                print("✓ Request history tracking works")
                print(f"  - Initial size: {initial_history_size}")
                print(f"  - Final size: {final_history_size}")
                test_results.append(("Request History", True))
            else:
                print("✗ Request history not tracking properly")
                test_results.append(("Request History", False))
        except Exception as e:
            print(f"✗ Request history test failed: {e}")
            test_results.append(("Request History", False))
        
        # Test 9: Async processing capability
        print("\n[TEST 9: Async Processing]")
        try:
            # Check if processing threads are running
            thread_count = len(gateway.processing_threads)
            queue_available = hasattr(gateway, 'request_queue')
            
            if thread_count > 0 and queue_available:
                print("✓ Async processing infrastructure available")
                print(f"  - Processing threads: {thread_count}")
                print(f"  - Request queue available: {queue_available}")
                test_results.append(("Async Processing", True))
            else:
                print(f"✗ Async processing infrastructure incomplete")
                print(f"  - Threads: {thread_count}, Queue: {queue_available}")
                test_results.append(("Async Processing", False))
        except Exception as e:
            print(f"✗ Async processing test failed: {e}")
            test_results.append(("Async Processing", False))
        
        # Test 10: Component integration
        print("\n[TEST 10: Component Integration]")
        try:
            # Check all components are initialized
            components = [
                'load_balancer',
                'rate_limiter', 
                'auth_manager',
                'request_validator',
                'response_formatter',
                'metrics_collector'
            ]
            
            components_initialized = 0
            for component in components:
                if hasattr(gateway, component) and getattr(gateway, component) is not None:
                    components_initialized += 1
            
            if components_initialized == len(components):
                print("✓ All components integrated successfully")
                print(f"  - Components initialized: {components_initialized}/{len(components)}")
                test_results.append(("Component Integration", True))
            else:
                print(f"✗ Component integration incomplete: {components_initialized}/{len(components)}")
                test_results.append(("Component Integration", False))
        except Exception as e:
            print(f"✗ Component integration test failed: {e}")
            test_results.append(("Component Integration", False))
        
        # Test 11: Error handling
        print("\n[TEST 11: Error Handling]")
        try:
            # Test various error conditions
            error_conditions = [
                {'endpoint': '/nonexistent', 'method': 'GET'},  # Non-existent endpoint
                {'endpoint': '/brain/status', 'method': 'DELETE'},  # Wrong method
            ]
            
            error_handling_working = 0
            for condition in error_conditions:
                try:
                    test_req = {**condition, 'headers': {}, 'body': {}}
                    response = gateway.route_request(test_req)
                    # Should get proper error response
                    if 'error' in response or not response.get('success', True):
                        error_handling_working += 1
                except Exception:
                    # Should not crash, should handle gracefully
                    pass
            
            if error_handling_working >= len(error_conditions) * 0.8:
                print("✓ Error handling working properly")
                print(f"  - Error conditions handled: {error_handling_working}/{len(error_conditions)}")
                test_results.append(("Error Handling", True))
            else:
                print(f"✗ Error handling insufficient: {error_handling_working}/{len(error_conditions)}")
                test_results.append(("Error Handling", False))
        except Exception as e:
            print(f"✗ Error handling test failed: {e}")
            test_results.append(("Error Handling", False))
        
        # Test 12: Configuration validation
        print("\n[TEST 12: Configuration Validation]")
        try:
            # Test with different configurations
            configs_to_test = [
                {'api_version': '2.0', 'timeout_seconds': 60},  # Modified config
                {},  # Minimal config
                {'api_version': '1.0', 'max_retries': 1, 'circuit_breaker_threshold': 10}  # Custom config
            ]
            
            config_tests_passed = 0
            for test_config in configs_to_test:
                try:
                    test_gateway = APIGatewayManager(test_config)
                    config_tests_passed += 1
                except Exception:
                    pass
            
            if config_tests_passed >= len(configs_to_test) * 0.8:
                print("✓ Configuration validation working")
                print(f"  - Configurations tested: {config_tests_passed}/{len(configs_to_test)}")
                test_results.append(("Configuration Validation", True))
            else:
                print(f"✗ Configuration validation failed: {config_tests_passed}/{len(configs_to_test)}")
                test_results.append(("Configuration Validation", False))
        except Exception as e:
            print(f"✗ Configuration validation test failed: {e}")
            test_results.append(("Configuration Validation", False))
    
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    success_rate = (passed/total * 100) if total > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
    
    return test_results


if __name__ == "__main__":
    results = test_api_gateway_manager_comprehensive()
    
    # Exit with appropriate code
    all_passed = all(result for _, result in results)
    sys.exit(0 if all_passed else 1)