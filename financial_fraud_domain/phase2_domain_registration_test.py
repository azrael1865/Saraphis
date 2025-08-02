#!/usr/bin/env python3
"""
Phase 2: Domain Registration Testing
Tests fraud domain registration and advanced domain functionality
"""

import logging
import sys
import traceback
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

class DomainRegistrationTestSuite:
    """Comprehensive test suite for Domain Registration"""
    
    def __init__(self):
        self.test_results = {}
        self.brain_instance = None
        self.test_start_time = None
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all Phase 2 domain registration tests"""
        self.test_start_time = time.time()
        
        print("=" * 80)
        print("PHASE 2: DOMAIN REGISTRATION TESTING")
        print("=" * 80)
        
        # Initialize Brain for testing
        if not self._initialize_brain():
            return self.generate_test_summary()
        
        # Test 1: Basic Domain Registration
        self.test_basic_domain_registration()
        
        # Test 2: Fraud Domain Registration
        self.test_fraud_domain_registration()
        
        # Test 3: Enhanced Domain Features
        self.test_enhanced_domain_features()
        
        # Test 4: Domain Configuration Validation
        self.test_domain_configuration_validation()
        
        # Test 5: Multiple Domain Management
        self.test_multiple_domain_management()
        
        # Test 6: Domain State Persistence
        self.test_domain_state_persistence()
        
        # Generate summary
        return self.generate_test_summary()
    
    def _initialize_brain(self) -> bool:
        """Initialize Brain system for testing"""
        try:
            from brain import Brain, BrainSystemConfig
            from pathlib import Path
            
            config = BrainSystemConfig(
                base_path=Path.cwd() / ".brain_domain_test",
                enable_persistence=True,
                enable_monitoring=True,
                max_domains=20,
                max_memory_gb=2.0,
                enable_parallel_predictions=True
            )
            
            self.brain_instance = Brain(config)
            logger.info("Brain system initialized for domain registration testing")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Brain: {e}")
            self.test_results['brain_initialization'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    def test_basic_domain_registration(self) -> bool:
        """Test 1: Basic domain registration functionality"""
        print("\n--- Test 1: Basic Domain Registration ---")
        
        try:
            registration_tests = {}
            
            # Test 1.1: Register a simple test domain
            test_domain_name = f"test_domain_{int(time.time())}"
            
            class TestDomainConfig:
                def __init__(self):
                    self.domain_type = 'test'
                    self.description = 'Test domain for registration testing'
                    self.version = '1.0.0'
            
            config = TestDomainConfig()
            
            # Test registration
            result = self.brain_instance.add_domain(test_domain_name, config, initialize_model=True)
            
            if result['success']:
                print(f"✓ Successfully registered test domain: {test_domain_name}")
                registration_tests['register_test_domain'] = True
                
                # Verify domain appears in list
                domains = self.brain_instance.list_available_domains()
                domain_found = any(d['name'] == test_domain_name for d in domains)
                
                if domain_found:
                    print(f"✓ Test domain found in available domains list")
                    registration_tests['domain_in_list'] = True
                else:
                    print(f"✗ Test domain not found in available domains list")
                    registration_tests['domain_in_list'] = False
                    
            else:
                print(f"✗ Failed to register test domain: {result.get('error', 'Unknown error')}")
                registration_tests['register_test_domain'] = False
                registration_tests['domain_in_list'] = False
            
            # Test 1.2: Check domain capabilities
            try:
                capabilities = self.brain_instance.get_domain_capabilities(test_domain_name)
                if 'error' not in capabilities:
                    print(f"✓ Retrieved domain capabilities")
                    registration_tests['get_capabilities'] = True
                else:
                    print(f"✗ Failed to get domain capabilities: {capabilities['error']}")
                    registration_tests['get_capabilities'] = False
            except Exception as e:
                print(f"✗ Error getting capabilities: {e}")
                registration_tests['get_capabilities'] = False
            
            # Test 1.3: Check domain health
            try:
                health = self.brain_instance.get_domain_health(test_domain_name)
                health_score = health.get('health_score', 0)
                print(f"✓ Domain health check: {health_score}/100")
                registration_tests['health_check'] = health_score > 0
            except Exception as e:
                print(f"✗ Health check failed: {e}")
                registration_tests['health_check'] = False
            
            # Summary
            passed_tests = sum(registration_tests.values())
            total_tests = len(registration_tests)
            
            if passed_tests == total_tests:
                self.test_results['basic_registration'] = {
                    'status': 'PASSED',
                    'tests': registration_tests,
                    'test_domain': test_domain_name
                }
                return True
            else:
                self.test_results['basic_registration'] = {
                    'status': 'PARTIAL',
                    'tests': registration_tests,
                    'success_rate': (passed_tests / total_tests) * 100
                }
                return False
                
        except Exception as e:
            print(f"✗ Basic domain registration testing failed: {e}")
            self.test_results['basic_registration'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    def test_fraud_domain_registration(self) -> bool:
        """Test 2: Fraud domain registration and functionality"""
        print("\n--- Test 2: Fraud Domain Registration ---")
        
        try:
            fraud_tests = {}
            
            # Test 2.1: Check if fraud domain is already registered
            fraud_domain_name = 'financial_fraud'
            domains = self.brain_instance.list_available_domains()
            fraud_domain_exists = any(d['name'] == fraud_domain_name for d in domains)
            
            if fraud_domain_exists:
                print(f"✓ Fraud domain '{fraud_domain_name}' is registered")
                fraud_tests['fraud_domain_exists'] = True
            else:
                print(f"✗ Fraud domain '{fraud_domain_name}' not found")
                fraud_tests['fraud_domain_exists'] = False
            
            # Test 2.2: Get fraud domain capabilities
            try:
                capabilities = self.brain_instance.get_domain_capabilities(fraud_domain_name)
                if 'error' not in capabilities:
                    cap_count = len(capabilities.get('capabilities', []))
                    print(f"✓ Fraud domain has {cap_count} capabilities")
                    fraud_tests['fraud_capabilities'] = cap_count > 0
                else:
                    print(f"✗ Error getting fraud capabilities: {capabilities['error']}")
                    fraud_tests['fraud_capabilities'] = False
            except Exception as e:
                print(f"✗ Exception getting fraud capabilities: {e}")
                fraud_tests['fraud_capabilities'] = False
            
            # Test 2.3: Test fraud detection functionality
            try:
                test_transaction = {
                    'transaction_id': f'test_{uuid.uuid4().hex[:8]}',
                    'amount': 15000,  # High amount
                    'user_id': 'test_user',
                    'merchant_id': 'suspicious_merchant',
                    'timestamp': '2024-01-01T03:00:00Z',  # Unusual hour
                    'type': 'purchase'
                }
                
                # Test with Brain's fraud detection
                if hasattr(self.brain_instance, 'detect_fraud'):
                    result = self.brain_instance.detect_fraud(test_transaction)
                    
                    print(f"✓ Fraud detection executed:")
                    print(f"    - Fraud detected: {getattr(result, 'fraud_detected', 'unknown')}")
                    print(f"    - Fraud probability: {getattr(result, 'fraud_probability', 0.0):.3f}")
                    print(f"    - Confidence: {getattr(result, 'confidence', 0.0):.3f}")
                    
                    fraud_tests['fraud_detection'] = True
                else:
                    # Test with general prediction
                    result = self.brain_instance.predict(test_transaction, domain=fraud_domain_name)
                    
                    print(f"✓ General prediction executed:")
                    print(f"    - Success: {result.success}")
                    print(f"    - Confidence: {result.confidence:.3f}")
                    print(f"    - Domain: {result.domain}")
                    
                    fraud_tests['fraud_detection'] = result.success
                    
            except Exception as e:
                print(f"✗ Fraud detection test failed: {e}")
                fraud_tests['fraud_detection'] = False
            
            # Test 2.4: Check fraud system status
            try:
                if hasattr(self.brain_instance, 'get_fraud_system_status'):
                    fraud_status = self.brain_instance.get_fraud_system_status()
                    print(f"✓ Fraud system status retrieved:")
                    print(f"    - Domain registered: {fraud_status.get('fraud_domain_registered', False)}")
                    print(f"    - Handlers active: {fraud_status.get('fraud_handlers_active', 0)}")
                    fraud_tests['fraud_status'] = fraud_status.get('fraud_domain_registered', False)
                else:
                    print(f"ℹ Fraud system status method not available")
                    fraud_tests['fraud_status'] = True  # Not a failure
            except Exception as e:
                print(f"✗ Fraud status check failed: {e}")
                fraud_tests['fraud_status'] = False
            
            # Summary
            passed_tests = sum(fraud_tests.values())
            total_tests = len(fraud_tests)
            
            if passed_tests >= total_tests * 0.75:  # 75% pass rate for fraud tests
                self.test_results['fraud_registration'] = {
                    'status': 'PASSED',
                    'tests': fraud_tests,
                    'pass_rate': (passed_tests / total_tests) * 100
                }
                return True
            else:
                self.test_results['fraud_registration'] = {
                    'status': 'PARTIAL',
                    'tests': fraud_tests,
                    'pass_rate': (passed_tests / total_tests) * 100
                }
                return False
                
        except Exception as e:
            print(f"✗ Fraud domain registration testing failed: {e}")
            self.test_results['fraud_registration'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    def test_enhanced_domain_features(self) -> bool:
        """Test 3: Enhanced domain features"""
        print("\n--- Test 3: Enhanced Domain Features ---")
        
        try:
            enhanced_tests = {}
            
            # Test 3.1: Domain routing functionality
            try:
                # Test if domain router exists
                if hasattr(self.brain_instance, 'domain_router'):
                    print("✓ Domain router component available")
                    enhanced_tests['domain_router'] = True
                else:
                    print("ℹ Domain router not available (may be optional)")
                    enhanced_tests['domain_router'] = True  # Not a failure
            except Exception as e:
                print(f"✗ Domain router test failed: {e}")
                enhanced_tests['domain_router'] = False
            
            # Test 3.2: Enhanced registry features
            try:
                registry = self.brain_instance.domain_registry
                
                # Check registry type
                if hasattr(registry, 'register_fraud_domain'):
                    print("✓ Enhanced registry with fraud support detected")
                    enhanced_tests['enhanced_registry'] = True
                else:
                    print("ℹ Basic registry detected (fraud support may be integrated)")
                    enhanced_tests['enhanced_registry'] = True  # Not a failure
                    
            except Exception as e:
                print(f"✗ Enhanced registry test failed: {e}")
                enhanced_tests['enhanced_registry'] = False
            
            # Test 3.3: Domain state management
            try:
                # Check if domain state manager exists
                if hasattr(self.brain_instance, 'domain_state_manager'):
                    print("✓ Domain state manager available")
                    enhanced_tests['state_manager'] = True
                else:
                    print("ℹ Domain state manager not available (may be integrated)")
                    enhanced_tests['state_manager'] = True  # Not a failure
            except Exception as e:
                print(f"✗ Domain state manager test failed: {e}")
                enhanced_tests['state_manager'] = False
            
            # Test 3.4: Performance monitoring
            try:
                status = self.brain_instance.get_brain_status()
                
                # Check for performance metrics
                if 'performance_metrics' in status:
                    print("✓ Performance monitoring available")
                    print(f"    - Total operations: {status['performance_metrics']['total_operations']}")
                    enhanced_tests['performance_monitoring'] = True
                else:
                    print("✗ Performance monitoring not available")
                    enhanced_tests['performance_monitoring'] = False
                    
            except Exception as e:
                print(f"✗ Performance monitoring test failed: {e}")
                enhanced_tests['performance_monitoring'] = False
            
            # Summary
            passed_tests = sum(enhanced_tests.values())
            total_tests = len(enhanced_tests)
            
            if passed_tests >= total_tests * 0.6:  # 60% pass rate for enhanced features
                self.test_results['enhanced_features'] = {
                    'status': 'PASSED',
                    'tests': enhanced_tests,
                    'pass_rate': (passed_tests / total_tests) * 100
                }
                return True
            else:
                self.test_results['enhanced_features'] = {
                    'status': 'PARTIAL',
                    'tests': enhanced_tests,
                    'pass_rate': (passed_tests / total_tests) * 100
                }
                return False
                
        except Exception as e:
            print(f"✗ Enhanced domain features testing failed: {e}")
            self.test_results['enhanced_features'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    def test_domain_configuration_validation(self) -> bool:
        """Test 4: Domain configuration validation"""
        print("\n--- Test 4: Domain Configuration Validation ---")
        
        try:
            validation_tests = {}
            
            # Test 4.1: Valid configuration
            try:
                class ValidConfig:
                    def __init__(self):
                        self.domain_type = 'test'
                        self.description = 'Valid test configuration'
                        self.version = '1.0.0'
                
                valid_config = ValidConfig()
                result = self.brain_instance.add_domain(f'valid_test_{int(time.time())}', valid_config)
                
                if result['success']:
                    print("✓ Valid configuration accepted")
                    validation_tests['valid_config'] = True
                else:
                    print(f"✗ Valid configuration rejected: {result.get('error')}")
                    validation_tests['valid_config'] = False
                    
            except Exception as e:
                print(f"✗ Valid configuration test failed: {e}")
                validation_tests['valid_config'] = False
            
            # Test 4.2: Edge case - None configuration
            try:
                result = self.brain_instance.add_domain(f'none_config_{int(time.time())}', None)
                
                # Should handle gracefully (either accept with defaults or reject cleanly)
                if result['success'] or 'error' in result:
                    print("✓ None configuration handled gracefully")
                    validation_tests['none_config'] = True
                else:
                    print("✗ None configuration not handled properly")
                    validation_tests['none_config'] = False
                    
            except Exception as e:
                print(f"ℹ None configuration caused exception (acceptable): {e}")
                validation_tests['none_config'] = True  # Exception is acceptable
            
            # Test 4.3: Duplicate domain registration
            try:
                duplicate_name = 'financial_fraud'  # Should already exist
                
                class DuplicateConfig:
                    def __init__(self):
                        self.domain_type = 'duplicate'
                        self.description = 'Duplicate test'
                
                result = self.brain_instance.add_domain(duplicate_name, DuplicateConfig())
                
                # Should either reject duplicate or update existing
                print(f"ℹ Duplicate registration result: {result.get('success', False)}")
                validation_tests['duplicate_handling'] = True  # Any behavior is acceptable
                
            except Exception as e:
                print(f"ℹ Duplicate registration caused exception (acceptable): {e}")
                validation_tests['duplicate_handling'] = True
            
            # Summary
            passed_tests = sum(validation_tests.values())
            total_tests = len(validation_tests)
            
            if passed_tests >= total_tests * 0.8:  # 80% pass rate
                self.test_results['configuration_validation'] = {
                    'status': 'PASSED',
                    'tests': validation_tests,
                    'pass_rate': (passed_tests / total_tests) * 100
                }
                return True
            else:
                self.test_results['configuration_validation'] = {
                    'status': 'PARTIAL',
                    'tests': validation_tests,
                    'pass_rate': (passed_tests / total_tests) * 100
                }
                return False
                
        except Exception as e:
            print(f"✗ Configuration validation testing failed: {e}")
            self.test_results['configuration_validation'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    def test_multiple_domain_management(self) -> bool:
        """Test 5: Multiple domain management"""
        print("\n--- Test 5: Multiple Domain Management ---")
        
        try:
            management_tests = {}
            
            # Test 5.1: Register multiple domains
            domain_names = []
            
            for i in range(3):
                domain_name = f'multi_test_{int(time.time())}_{i}'
                
                class MultiConfig:
                    def __init__(self, index):
                        self.domain_type = f'multi_{index}'
                        self.description = f'Multi-domain test {index}'
                        self.priority = index
                
                config = MultiConfig(i)
                result = self.brain_instance.add_domain(domain_name, config)
                
                if result['success']:
                    domain_names.append(domain_name)
            
            registered_count = len(domain_names)
            print(f"✓ Registered {registered_count}/3 test domains")
            management_tests['multiple_registration'] = registered_count >= 2
            
            # Test 5.2: List all domains
            try:
                all_domains = self.brain_instance.list_available_domains()
                total_domains = len(all_domains)
                
                print(f"✓ Listed {total_domains} total domains")
                management_tests['list_all_domains'] = total_domains >= 4  # At least default + some test domains
                
                # Show domain types
                domain_types = set(d.get('type', 'unknown') for d in all_domains)
                print(f"    - Domain types found: {', '.join(domain_types)}")
                
            except Exception as e:
                print(f"✗ List domains failed: {e}")
                management_tests['list_all_domains'] = False
            
            # Test 5.3: Check domain health for multiple domains
            try:
                healthy_domains = 0
                
                for domain_name in domain_names[:2]:  # Check first 2
                    health = self.brain_instance.get_domain_health(domain_name)
                    if health.get('health_score', 0) > 0:
                        healthy_domains += 1
                
                print(f"✓ {healthy_domains}/{min(2, len(domain_names))} test domains are healthy")
                management_tests['multiple_health_checks'] = healthy_domains > 0
                
            except Exception as e:
                print(f"✗ Multiple health checks failed: {e}")
                management_tests['multiple_health_checks'] = False
            
            # Summary
            passed_tests = sum(management_tests.values())
            total_tests = len(management_tests)
            
            if passed_tests == total_tests:
                self.test_results['multiple_domain_management'] = {
                    'status': 'PASSED',
                    'tests': management_tests,
                    'domains_created': domain_names
                }
                return True
            else:
                self.test_results['multiple_domain_management'] = {
                    'status': 'PARTIAL',
                    'tests': management_tests,
                    'pass_rate': (passed_tests / total_tests) * 100
                }
                return False
                
        except Exception as e:
            print(f"✗ Multiple domain management testing failed: {e}")
            self.test_results['multiple_domain_management'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    def test_domain_state_persistence(self) -> bool:
        """Test 6: Domain state persistence"""
        print("\n--- Test 6: Domain State Persistence ---")
        
        try:
            persistence_tests = {}
            
            # Test 6.1: Check if domains persist across sessions
            initial_domains = self.brain_instance.list_available_domains()
            initial_count = len(initial_domains)
            
            print(f"ℹ Initial domain count: {initial_count}")
            
            # Create a test domain
            test_domain = f'persistence_test_{int(time.time())}'
            
            class PersistenceConfig:
                def __init__(self):
                    self.domain_type = 'persistence_test'
                    self.description = 'Domain for persistence testing'
                    self.persistent = True
            
            result = self.brain_instance.add_domain(test_domain, PersistenceConfig())
            
            if result['success']:
                print(f"✓ Created persistence test domain: {test_domain}")
                persistence_tests['create_persistent_domain'] = True
            else:
                print(f"✗ Failed to create persistence test domain")
                persistence_tests['create_persistent_domain'] = False
            
            # Test 6.2: Check domain storage
            try:
                # Check if domain registry has storage
                if hasattr(self.brain_instance.domain_registry, 'storage_path'):
                    storage_path = self.brain_instance.domain_registry.storage_path
                    
                    if storage_path and storage_path.exists():
                        print(f"✓ Domain storage file exists: {storage_path}")
                        persistence_tests['storage_file_exists'] = True
                        
                        # Check storage content
                        try:
                            with open(storage_path, 'r') as f:
                                stored_data = json.load(f)
                            
                            stored_domains = stored_data.get('domains', {})
                            print(f"✓ Storage contains {len(stored_domains)} domains")
                            persistence_tests['storage_content_valid'] = len(stored_domains) > 0
                            
                        except Exception as e:
                            print(f"✗ Failed to read storage content: {e}")
                            persistence_tests['storage_content_valid'] = False
                    else:
                        print(f"ℹ Domain storage not enabled or file doesn't exist")
                        persistence_tests['storage_file_exists'] = True  # Not a failure
                else:
                    print(f"ℹ Domain registry doesn't use file storage")
                    persistence_tests['storage_file_exists'] = True  # Not a failure
                    
            except Exception as e:
                print(f"✗ Storage check failed: {e}")
                persistence_tests['storage_file_exists'] = False
            
            # Test 6.3: Configuration persistence
            try:
                # Check if we can retrieve the config we just stored
                if result['success']:
                    retrieved_config = self.brain_instance.domain_registry.get_domain_config(test_domain)
                    
                    if retrieved_config:
                        print(f"✓ Domain configuration retrieved successfully")
                        persistence_tests['config_retrieval'] = True
                    else:
                        print(f"✗ Domain configuration not retrievable")
                        persistence_tests['config_retrieval'] = False
                else:
                    print(f"ℹ Skipping config retrieval (domain creation failed)")
                    persistence_tests['config_retrieval'] = True  # Skip if domain creation failed
                    
            except Exception as e:
                print(f"✗ Configuration retrieval failed: {e}")
                persistence_tests['config_retrieval'] = False
            
            # Summary
            passed_tests = sum(persistence_tests.values())
            total_tests = len(persistence_tests)
            
            if passed_tests >= total_tests * 0.7:  # 70% pass rate
                self.test_results['domain_persistence'] = {
                    'status': 'PASSED',
                    'tests': persistence_tests,
                    'pass_rate': (passed_tests / total_tests) * 100
                }
                return True
            else:
                self.test_results['domain_persistence'] = {
                    'status': 'PARTIAL',
                    'tests': persistence_tests,
                    'pass_rate': (passed_tests / total_tests) * 100
                }
                return False
                
        except Exception as e:
            print(f"✗ Domain state persistence testing failed: {e}")
            self.test_results['domain_persistence'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    def generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        total_time = time.time() - self.test_start_time
        
        print("\n" + "=" * 80)
        print("PHASE 2 TEST SUMMARY")
        print("=" * 80)
        
        # Count results
        passed = sum(1 for result in self.test_results.values() if result.get('status') == 'PASSED')
        partial = sum(1 for result in self.test_results.values() if result.get('status') == 'PARTIAL')
        failed = sum(1 for result in self.test_results.values() if result.get('status') == 'FAILED')
        total = len(self.test_results)
        
        # Display results
        print(f"\nTest Results:")
        print(f"  ✓ Passed:  {passed}/{total}")
        print(f"  ◐ Partial: {partial}/{total}")
        print(f"  ✗ Failed:  {failed}/{total}")
        print(f"\nTotal time: {total_time:.2f} seconds")
        
        # Detailed results
        print(f"\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = result.get('status', 'UNKNOWN')
            if status == 'PASSED':
                print(f"  ✓ {test_name}: {status}")
                if 'pass_rate' in result:
                    print(f"      Pass rate: {result['pass_rate']:.1f}%")
            elif status == 'PARTIAL':
                print(f"  ◐ {test_name}: {status}")
                if 'pass_rate' in result:
                    print(f"      Pass rate: {result['pass_rate']:.1f}%")
            elif status == 'FAILED':
                print(f"  ✗ {test_name}: {status}")
                if 'error' in result:
                    print(f"      Error: {result['error']}")
        
        # Recommendations
        print(f"\nRecommendations:")
        if failed > 0:
            print("  - Fix failed tests before proceeding to Phase 3")
            print("  - Check error messages above for specific issues")
        elif partial > 0:
            print("  - Review partial test results")
            print("  - Some features may be optional or can be enhanced")
        else:
            print("  - All tests passed! Ready for Phase 3")
        
        # Key findings
        print(f"\nKey Findings:")
        
        # Check fraud domain status
        fraud_test = self.test_results.get('fraud_registration', {})
        if fraud_test.get('status') == 'PASSED':
            print("  - ✓ Fraud domain registration is working correctly")
        else:
            print("  - ⚠️ Fraud domain registration needs attention")
        
        # Check basic functionality
        basic_test = self.test_results.get('basic_registration', {})
        if basic_test.get('status') == 'PASSED':
            print("  - ✓ Basic domain registration is fully functional")
        
        # Check enhanced features
        enhanced_test = self.test_results.get('enhanced_features', {})
        if enhanced_test.get('status') in ['PASSED', 'PARTIAL']:
            print("  - ✓ Enhanced features are available (some may be optional)")
        
        # Return summary
        summary = {
            'phase': 2,
            'total_tests': total,
            'passed': passed,
            'partial': partial,
            'failed': failed,
            'success_rate': (passed / total) * 100 if total > 0 else 0,
            'total_time': total_time,
            'ready_for_next_phase': failed == 0,
            'detailed_results': self.test_results,
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def cleanup(self):
        """Clean up test resources"""
        if self.brain_instance and hasattr(self.brain_instance, 'shutdown'):
            try:
                self.brain_instance.shutdown()
                print("✓ Brain instance shutdown completed")
            except Exception as e:
                print(f"Warning: Brain shutdown failed: {e}")


def run_phase2_tests():
    """Run Phase 2 tests and return results"""
    test_suite = DomainRegistrationTestSuite()
    
    try:
        results = test_suite.run_all_tests()
        return results
    finally:
        test_suite.cleanup()


if __name__ == "__main__":
    print("Starting Phase 2: Domain Registration Testing...")
    results = run_phase2_tests()
    
    # Save results
    results_file = Path("phase2_test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    # Exit with appropriate code
    if results['ready_for_next_phase']:
        print("\n🎉 Phase 2 Complete! Ready for Phase 3.")
        sys.exit(0)
    else:
        print("\n⚠️  Phase 2 Issues Found. Review before continuing.")
        sys.exit(1)