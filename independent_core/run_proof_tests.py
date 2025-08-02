#!/usr/bin/env python3
"""
Test Runner for Proof System Integration
Executes proof system tests and generates reports compatible with Saraphis structure
"""

import sys
import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('proof_system_tests.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ProofSystemTestRunner:
    """Test runner for proof system integration tests"""
    
    def __init__(self):
        """Initialize test runner"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    def run_proof_system_tests(self, test_type: str = 'all') -> Dict[str, Any]:
        """Run proof system tests"""
        self.logger.info("="*80)
        self.logger.info("PROOF SYSTEM INTEGRATION TESTS")
        self.logger.info("="*80)
        self.start_time = time.time()
        
        try:
            if test_type in ['all', 'unit']:
                self.logger.info("\nRunning unit tests...")
                self.results['unit_tests'] = self._run_unit_tests()
                
            if test_type in ['all', 'integration']:
                self.logger.info("\nRunning integration tests...")
                self.results['integration_tests'] = self._run_integration_tests()
                
            if test_type in ['all', 'comprehensive']:
                self.logger.info("\nRunning comprehensive tests...")
                self.results['comprehensive_tests'] = self._run_comprehensive_tests()
                
            self.end_time = time.time()
            self.results['summary'] = self._generate_summary()
            
            # Save results
            self._save_results()
            
            # Display report
            self._display_report()
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Test execution failed: {str(e)}")
            raise
            
    def _run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests for proof components"""
        try:
            # Import and run unit tests
            from test_proof_components import ProofComponentUnitTests
            
            unit_test_runner = ProofComponentUnitTests({}, [])
            unit_results = unit_test_runner.run_all_unit_tests()
            
            return {
                'status': 'completed',
                'total_tests': unit_results.get('summary', {}).get('total_tests', 0),
                'passed_tests': unit_results.get('summary', {}).get('passed_tests', 0),
                'failed_tests': unit_results.get('summary', {}).get('failed_tests', 0),
                'success_rate': unit_results.get('summary', {}).get('success_rate', 0),
                'detailed_results': unit_results
            }
            
        except Exception as e:
            self.logger.error(f"Unit tests failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
            
    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run basic integration tests"""
        try:
            # Import and run integration tests 
            from test_brain_proof_integration import run_basic_integration_tests
            
            integration_results = run_basic_integration_tests()
            
            return {
                'status': 'completed',
                'results': integration_results
            }
            
        except ImportError:
            # Fallback: basic integration test
            return self._run_basic_integration_fallback()
        except Exception as e:
            self.logger.error(f"Integration tests failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
            
    def _run_basic_integration_fallback(self) -> Dict[str, Any]:
        """Fallback integration test"""
        try:
            from independent_core.proof_system.proof_integration_manager import ProofIntegrationManager
            from independent_core.proof_system.rule_based_engine import RuleBasedProofEngine
            
            # Basic integration test
            manager = ProofIntegrationManager()
            manager.register_engine('rule_based', RuleBasedProofEngine())
            
            # Test proof generation
            test_transaction = {
                'transaction_id': 'integration_test_001',
                'amount': 1000,
                'risk_score': 0.7
            }
            
            proof = manager.generate_comprehensive_proof(
                transaction=test_transaction,
                model_state={'iteration': 1}
            )
            
            success = 'rule_based' in proof and 'confidence' in proof
            
            return {
                'status': 'completed',
                'basic_integration_passed': success,
                'proof_generated': proof is not None,
                'engines_working': len(manager.get_registered_engines()) > 0
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
            
    def _run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        try:
            from test_proof_system_comprehensive import ProofSystemTestSuite
            
            test_suite = ProofSystemTestSuite()
            comprehensive_results = test_suite.run_all_tests()
            
            return {
                'status': 'completed',
                'results': comprehensive_results
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive tests failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
            
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test execution summary"""
        total_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Count tests across all categories
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for category, results in self.results.items():
            if category != 'summary' and isinstance(results, dict):
                if results.get('status') == 'completed':
                    category_total = results.get('total_tests', 0)
                    category_passed = results.get('passed_tests', 0)
                    category_failed = results.get('failed_tests', 0)
                    
                    total_tests += category_total
                    passed_tests += category_passed
                    failed_tests += category_failed
                    
        return {
            'execution_time_seconds': total_time,
            'timestamp': datetime.now().isoformat(),
            'overall_metrics': {
                'total_tests_executed': total_tests,
                'total_tests_passed': passed_tests,
                'total_tests_failed': failed_tests,
                'overall_success_rate': passed_tests / total_tests if total_tests > 0 else 0
            },
            'categories_tested': list(self.results.keys()),
            'ready_for_production': passed_tests / total_tests >= 0.8 if total_tests > 0 else False
        }
        
    def _save_results(self):
        """Save test results to file"""
        filename = f"proof_system_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            self.logger.info(f"\nResults saved to: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            
    def _display_report(self):
        """Display test execution report"""
        summary = self.results.get('summary', {})
        overall = summary.get('overall_metrics', {})
        
        self.logger.info("\n" + "="*80)
        self.logger.info("PROOF SYSTEM TEST EXECUTION SUMMARY")
        self.logger.info("="*80)
        
        # Overall results
        self.logger.info(f"\nOVERALL RESULTS:")
        self.logger.info(f"  Total Tests: {overall.get('total_tests_executed', 0)}")
        self.logger.info(f"  Passed: {overall.get('total_tests_passed', 0)}")
        self.logger.info(f"  Failed: {overall.get('total_tests_failed', 0)}")
        self.logger.info(f"  Success Rate: {overall.get('overall_success_rate', 0):.1%}")
        
        # Category breakdown
        self.logger.info(f"\nCATEGORY BREAKDOWN:")
        for category, results in self.results.items():
            if category != 'summary' and isinstance(results, dict):
                status = results.get('status', 'unknown')
                self.logger.info(f"\n  {category.upper()}:")
                self.logger.info(f"    Status: {status}")
                
                if status == 'completed':
                    total = results.get('total_tests', 0)
                    passed = results.get('passed_tests', 0)
                    if total > 0:
                        self.logger.info(f"    Tests: {passed}/{total} passed")
                elif status == 'failed':
                    error = results.get('error', 'Unknown error')
                    self.logger.info(f"    Error: {error}")
                    
        # Execution time
        exec_time = summary.get('execution_time_seconds', 0)
        self.logger.info(f"\nEXECUTION TIME: {exec_time:.2f} seconds")
        
        # Production readiness
        ready = summary.get('ready_for_production', False)
        self.logger.info(f"\nPRODUCTION READY: {'YES' if ready else 'NO'}")
        
        if ready:
            self.logger.info("\n✅ Proof system tests completed successfully!")
            self.logger.info("   System is ready for production deployment")
        else:
            self.logger.info("\n❌ Some tests failed or below threshold")
            self.logger.info("   Please review failed tests before production deployment")
            
        self.logger.info("="*80)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run Proof System Integration Tests'
    )
    parser.add_argument(
        '--type',
        choices=['all', 'unit', 'integration', 'comprehensive'],
        default='all',
        help='Type of tests to run'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Run tests
    runner = ProofSystemTestRunner()
    
    try:
        results = runner.run_proof_system_tests(test_type=args.type)
        
        # Exit code based on success
        summary = results.get('summary', {})
        ready = summary.get('ready_for_production', False)
        
        sys.exit(0 if ready else 1)
        
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        sys.exit(2)


if __name__ == "__main__":
    main()