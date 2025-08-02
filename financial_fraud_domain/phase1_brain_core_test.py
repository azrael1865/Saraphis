#!/usr/bin/env python3
"""
Phase 1: Core Brain System Testing
Tests the fundamental Brain system components before fraud domain integration
"""

import logging
import sys
import traceback
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

class BrainCoreTestSuite:
    """Comprehensive test suite for Core Brain system"""
    
    def __init__(self):
        self.test_results = {}
        self.brain_instance = None
        self.test_start_time = None
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all Phase 1 core Brain tests"""
        self.test_start_time = time.time()
        
        print("=" * 80)
        print("PHASE 1: CORE BRAIN SYSTEM TESTING")
        print("=" * 80)
        
        # Test 1: Brain Import and Availability
        self.test_brain_imports()
        
        # Test 2: Brain Configuration Creation
        self.test_brain_configuration()
        
        # Test 3: Brain Initialization
        self.test_brain_initialization()
        
        # Test 4: Basic Brain Operations
        self.test_basic_operations()
        
        # Test 5: Domain Registry Functionality
        self.test_domain_registry()
        
        # Test 6: Brain Status and Health
        self.test_brain_status()
        
        # Generate summary
        return self.generate_test_summary()
    
    def test_brain_imports(self) -> bool:
        """Test 1: Verify all Brain imports are working"""
        print("\n--- Test 1: Brain Import and Availability ---")
        
        try:
            # Test core Brain imports
            from brain import Brain, BrainSystemConfig
            print("✓ Core Brain imports successful")
            
            # Test enhanced components
            from enhanced_fraud_core_main import EnhancedFraudCoreConfig
            print("✓ Enhanced fraud core imports successful")
            
            # Test integration components  
            from ml_predictor import FinancialMLPredictor
            print("✓ ML predictor imports successful")
            
            # Test validation components
            from validation_integration import get_integrated_validator
            print("✓ Validation integration imports successful")
            
            # Test data loading components
            from data_loading_integration import get_integrated_data_loader
            print("✓ Data loading integration imports successful")
            
            # Test preprocessing components
            from preprocessing_integration import get_integrated_preprocessor
            print("✓ Preprocessing integration imports successful")
            
            self.test_results['imports'] = {'status': 'PASSED', 'details': 'All imports successful'}
            return True
            
        except ImportError as e:
            print(f"✗ Import failed: {e}")
            self.test_results['imports'] = {'status': 'FAILED', 'error': str(e)}
            return False
        except Exception as e:
            print(f"✗ Unexpected error during imports: {e}")
            self.test_results['imports'] = {'status': 'ERROR', 'error': str(e)}
            return False
    
    def test_brain_configuration(self) -> bool:
        """Test 2: Create and validate Brain configuration"""
        print("\n--- Test 2: Brain Configuration Creation ---")
        
        try:
            from brain import BrainSystemConfig
            from pathlib import Path
            
            # Create test configuration
            config = BrainSystemConfig(
                base_path=Path.cwd() / ".brain_test",
                enable_persistence=True,
                enable_monitoring=True,
                enable_adaptation=True,
                max_domains=20,
                max_memory_gb=4.0,
                max_cpu_percent=50.0,
                enable_parallel_predictions=True,
                max_prediction_threads=4,
                prediction_cache_size=1000
            )
            
            print(f"✓ Brain configuration created successfully")
            print(f"  - Base path: {config.base_path}")
            print(f"  - Max domains: {config.max_domains}")
            print(f"  - Memory limit: {config.max_memory_gb}GB")
            print(f"  - CPU limit: {config.max_cpu_percent}%")
            print(f"  - Parallel predictions: {config.enable_parallel_predictions}")
            
            # Validate configuration paths
            print(f"  - Knowledge path: {config.knowledge_path}")
            print(f"  - Models path: {config.models_path}")
            print(f"  - Training path: {config.training_path}")
            
            self.test_results['configuration'] = {
                'status': 'PASSED', 
                'config': {
                    'base_path': str(config.base_path),
                    'max_domains': config.max_domains,
                    'memory_gb': config.max_memory_gb,
                    'cpu_percent': config.max_cpu_percent
                }
            }
            return True
            
        except Exception as e:
            print(f"✗ Configuration creation failed: {e}")
            self.test_results['configuration'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    def test_brain_initialization(self) -> bool:
        """Test 3: Initialize Brain system"""
        print("\n--- Test 3: Brain Initialization ---")
        
        try:
            from brain import Brain, BrainSystemConfig
            from pathlib import Path
            
            # Create configuration
            config = BrainSystemConfig(
                base_path=Path.cwd() / ".brain_test",
                enable_persistence=True,
                enable_monitoring=True,
                max_domains=10,
                max_memory_gb=2.0,
                enable_parallel_predictions=True,
                max_prediction_threads=2
            )
            
            print("Initializing Brain system...")
            start_time = time.time()
            
            # Initialize Brain
            self.brain_instance = Brain(config)
            
            init_time = time.time() - start_time
            print(f"✓ Brain initialized successfully in {init_time:.2f} seconds")
            
            # Check initialization state
            if hasattr(self.brain_instance, '_initialized'):
                print(f"  - Initialization flag: {self.brain_instance._initialized}")
            
            # Check for required components
            required_components = ['domain_registry', 'brain_core']
            missing_components = []
            
            for component in required_components:
                if hasattr(self.brain_instance, component):
                    print(f"  ✓ Component '{component}' present")
                else:
                    print(f"  ✗ Component '{component}' missing")
                    missing_components.append(component)
            
            if missing_components:
                self.test_results['initialization'] = {
                    'status': 'PARTIAL',
                    'missing_components': missing_components,
                    'init_time': init_time
                }
                return False
            else:
                self.test_results['initialization'] = {
                    'status': 'PASSED',
                    'init_time': init_time,
                    'components_found': required_components
                }
                return True
                
        except Exception as e:
            print(f"✗ Brain initialization failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            self.test_results['initialization'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    def test_basic_operations(self) -> bool:
        """Test 4: Basic Brain operations"""
        print("\n--- Test 4: Basic Brain Operations ---")
        
        if not self.brain_instance:
            print("✗ No Brain instance available for testing")
            self.test_results['basic_operations'] = {'status': 'SKIPPED', 'reason': 'No Brain instance'}
            return False
        
        try:
            operations_tested = {}
            
            # Test 1: Get Brain status
            try:
                status = self.brain_instance.get_brain_status()
                print(f"✓ Brain status retrieved: {status.get('initialized', 'unknown')}")
                operations_tested['get_status'] = True
            except Exception as e:
                print(f"✗ Brain status failed: {e}")
                operations_tested['get_status'] = False
            
            # Test 2: List available domains
            try:
                domains = self.brain_instance.list_available_domains()
                print(f"✓ Available domains listed: {len(domains)} domains")
                for domain in domains[:3]:  # Show first 3
                    print(f"    - {domain.get('name', 'unknown')}: {domain.get('status', 'unknown')}")
                operations_tested['list_domains'] = True
            except Exception as e:
                print(f"✗ List domains failed: {e}")
                operations_tested['list_domains'] = False
            
            # Test 3: Check if fraud domain can be added
            try:
                # Don't actually add yet, just check the method exists
                if hasattr(self.brain_instance, 'add_domain'):
                    print("✓ Add domain method available")
                    operations_tested['add_domain_method'] = True
                else:
                    print("✗ Add domain method not found")
                    operations_tested['add_domain_method'] = False
            except Exception as e:
                print(f"✗ Add domain check failed: {e}")
                operations_tested['add_domain_method'] = False
            
            # Test 4: Memory usage check
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                print(f"✓ Memory usage: {memory_mb:.1f} MB")
                operations_tested['memory_check'] = True
            except Exception as e:
                print(f"✗ Memory check failed: {e}")
                operations_tested['memory_check'] = False
            
            # Summary
            passed_operations = sum(operations_tested.values())
            total_operations = len(operations_tested)
            
            if passed_operations == total_operations:
                self.test_results['basic_operations'] = {
                    'status': 'PASSED',
                    'operations': operations_tested,
                    'success_rate': 100.0
                }
                return True
            else:
                self.test_results['basic_operations'] = {
                    'status': 'PARTIAL',
                    'operations': operations_tested,
                    'success_rate': (passed_operations / total_operations) * 100
                }
                return False
                
        except Exception as e:
            print(f"✗ Basic operations testing failed: {e}")
            self.test_results['basic_operations'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    def test_domain_registry(self) -> bool:
        """Test 5: Domain registry functionality"""
        print("\n--- Test 5: Domain Registry Functionality ---")
        
        if not self.brain_instance:
            print("✗ No Brain instance available for testing")
            self.test_results['domain_registry'] = {'status': 'SKIPPED', 'reason': 'No Brain instance'}
            return False
        
        try:
            registry_tests = {}
            
            # Test 1: Access domain registry
            try:
                if hasattr(self.brain_instance, 'domain_registry'):
                    registry = self.brain_instance.domain_registry
                    print("✓ Domain registry accessible")
                    registry_tests['registry_access'] = True
                    
                    # Test registry methods
                    if hasattr(registry, 'list_domains'):
                        domains = registry.list_domains()
                        print(f"✓ Registry lists {len(domains)} domains")
                        registry_tests['list_domains'] = True
                    else:
                        print("✗ Registry missing list_domains method")
                        registry_tests['list_domains'] = False
                        
                else:
                    print("✗ Domain registry not found")
                    registry_tests['registry_access'] = False
                    
            except Exception as e:
                print(f"✗ Domain registry access failed: {e}")
                registry_tests['registry_access'] = False
            
            # Test 2: Check for enhanced registry features
            try:
                if hasattr(self.brain_instance, 'domain_registry'):
                    registry = self.brain_instance.domain_registry
                    
                    # Check for fraud-specific methods
                    fraud_methods = ['register_fraud_domain', 'get_fraud_domain_status']
                    found_methods = []
                    
                    for method in fraud_methods:
                        if hasattr(registry, method):
                            found_methods.append(method)
                    
                    print(f"✓ Found {len(found_methods)} fraud-specific methods: {found_methods}")
                    registry_tests['fraud_methods'] = len(found_methods) > 0
                else:
                    registry_tests['fraud_methods'] = False
                    
            except Exception as e:
                print(f"✗ Enhanced registry check failed: {e}")
                registry_tests['fraud_methods'] = False
            
            # Test 3: Domain registration capability
            try:
                if hasattr(self.brain_instance, 'add_domain'):
                    print("✓ Domain registration method available")
                    registry_tests['can_register'] = True
                else:
                    print("✗ Domain registration method not found")
                    registry_tests['can_register'] = False
            except Exception as e:
                print(f"✗ Domain registration check failed: {e}")
                registry_tests['can_register'] = False
            
            # Summary
            passed_tests = sum(registry_tests.values())
            total_tests = len(registry_tests)
            
            if passed_tests >= total_tests * 0.8:  # 80% pass rate
                self.test_results['domain_registry'] = {
                    'status': 'PASSED',
                    'tests': registry_tests,
                    'pass_rate': (passed_tests / total_tests) * 100
                }
                return True
            else:
                self.test_results['domain_registry'] = {
                    'status': 'PARTIAL',
                    'tests': registry_tests,
                    'pass_rate': (passed_tests / total_tests) * 100
                }
                return False
                
        except Exception as e:
            print(f"✗ Domain registry testing failed: {e}")
            self.test_results['domain_registry'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    def test_brain_status(self) -> bool:
        """Test 6: Brain status and health monitoring"""
        print("\n--- Test 6: Brain Status and Health ---")
        
        if not self.brain_instance:
            print("✗ No Brain instance available for testing")
            self.test_results['brain_status'] = {'status': 'SKIPPED', 'reason': 'No Brain instance'}
            return False
        
        try:
            status_tests = {}
            
            # Test 1: Basic status
            try:
                status = self.brain_instance.get_brain_status()
                print(f"✓ Brain status: {status}")
                
                # Check important status fields
                important_fields = ['initialized', 'total_domains', 'memory_usage_percent']
                found_fields = []
                
                for field in important_fields:
                    if field in status:
                        found_fields.append(field)
                        print(f"    - {field}: {status[field]}")
                
                status_tests['basic_status'] = len(found_fields) > 0
                
            except Exception as e:
                print(f"✗ Basic status failed: {e}")
                status_tests['basic_status'] = False
            
            # Test 2: Resource monitoring
            try:
                import psutil
                process = psutil.Process()
                
                cpu_percent = process.cpu_percent(interval=1)
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                print(f"✓ Resource monitoring:")
                print(f"    - CPU: {cpu_percent:.1f}%")
                print(f"    - Memory: {memory_mb:.1f} MB")
                print(f"    - Threads: {process.num_threads()}")
                
                status_tests['resource_monitoring'] = True
                
            except Exception as e:
                print(f"✗ Resource monitoring failed: {e}")
                status_tests['resource_monitoring'] = False
            
            # Test 3: Check for monitoring features
            try:
                monitoring_features = []
                
                if hasattr(self.brain_instance, 'get_fraud_system_status'):
                    monitoring_features.append('fraud_system_status')
                
                if hasattr(self.brain_instance, 'monitor_fraud_performance'):
                    monitoring_features.append('fraud_performance_monitoring')
                
                print(f"✓ Found monitoring features: {monitoring_features}")
                status_tests['monitoring_features'] = len(monitoring_features) > 0
                
            except Exception as e:
                print(f"✗ Monitoring features check failed: {e}")
                status_tests['monitoring_features'] = False
            
            # Summary
            passed_tests = sum(status_tests.values())
            total_tests = len(status_tests)
            
            if passed_tests >= total_tests * 0.7:  # 70% pass rate
                self.test_results['brain_status'] = {
                    'status': 'PASSED',
                    'tests': status_tests,
                    'pass_rate': (passed_tests / total_tests) * 100
                }
                return True
            else:
                self.test_results['brain_status'] = {
                    'status': 'PARTIAL',
                    'tests': status_tests,
                    'pass_rate': (passed_tests / total_tests) * 100
                }
                return False
                
        except Exception as e:
            print(f"✗ Brain status testing failed: {e}")
            self.test_results['brain_status'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    def generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        total_time = time.time() - self.test_start_time
        
        print("\n" + "=" * 80)
        print("PHASE 1 TEST SUMMARY")
        print("=" * 80)
        
        # Count results
        passed = sum(1 for result in self.test_results.values() if result.get('status') == 'PASSED')
        partial = sum(1 for result in self.test_results.values() if result.get('status') == 'PARTIAL')
        failed = sum(1 for result in self.test_results.values() if result.get('status') == 'FAILED')
        skipped = sum(1 for result in self.test_results.values() if result.get('status') == 'SKIPPED')
        total = len(self.test_results)
        
        # Display results
        print(f"\nTest Results:")
        print(f"  ✓ Passed:  {passed}/{total}")
        print(f"  ◐ Partial: {partial}/{total}")
        print(f"  ✗ Failed:  {failed}/{total}")
        print(f"  ○ Skipped: {skipped}/{total}")
        print(f"\nTotal time: {total_time:.2f} seconds")
        
        # Detailed results
        print(f"\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = result.get('status', 'UNKNOWN')
            if status == 'PASSED':
                print(f"  ✓ {test_name}: {status}")
            elif status == 'PARTIAL':
                print(f"  ◐ {test_name}: {status}")
            elif status == 'FAILED':
                print(f"  ✗ {test_name}: {status}")
                if 'error' in result:
                    print(f"      Error: {result['error']}")
            else:
                print(f"  ○ {test_name}: {status}")
        
        # Recommendations
        print(f"\nRecommendations:")
        if failed > 0:
            print("  - Fix failed tests before proceeding to Phase 2")
            print("  - Check error messages above for specific issues")
        elif partial > 0:
            print("  - Review partial test results")
            print("  - Consider if missing features are required")
        else:
            print("  - All tests passed! Ready for Phase 2")
        
        # Return summary
        summary = {
            'phase': 1,
            'total_tests': total,
            'passed': passed,
            'partial': partial,
            'failed': failed,
            'skipped': skipped,
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


def run_phase1_tests():
    """Run Phase 1 tests and return results"""
    test_suite = BrainCoreTestSuite()
    
    try:
        results = test_suite.run_all_tests()
        return results
    finally:
        test_suite.cleanup()


if __name__ == "__main__":
    print("Starting Phase 1: Core Brain System Testing...")
    results = run_phase1_tests()
    
    # Save results
    results_file = Path("phase1_test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    # Exit with appropriate code
    if results['ready_for_next_phase']:
        print("\n🎉 Phase 1 Complete! Ready for Phase 2.")
        sys.exit(0)
    else:
        print("\n⚠️  Phase 1 Issues Found. Review before continuing.")
        sys.exit(1)