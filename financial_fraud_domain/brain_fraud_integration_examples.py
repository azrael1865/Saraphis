"""
Brain-Fraud System Integration Examples and Testing Framework
Complete production-ready examples demonstrating the unified system
"""

import logging
import time
import json
import uuid
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================== INTEGRATION EXAMPLES ========================

class BrainFraudIntegrationExamples:
    """Complete examples demonstrating Brain-Fraud system integration"""
    
    @staticmethod
    def example_1_basic_brain_integration():
        """Example 1: Basic fraud detection through Brain system"""
        print("\n" + "="*80)
        print("EXAMPLE 1: Basic Fraud Detection through Brain System")
        print("="*80)
        
        try:
            # Import Brain system
            from independent_core.brain import Brain
            
            # Initialize Brain with default configuration
            brain = Brain()
            
            # The fraud domain is automatically registered by the connector
            print("\nAvailable domains:")
            domains = brain.list_available_domains()
            for domain in domains:
                print(f"  - {domain['name']}: {domain['description']}")
            
            # Sample transaction
            transaction = {
                'transaction_id': 'tx_example_001',
                'user_id': 'user_123',
                'amount': 5500.00,
                'merchant_id': 'suspicious_merchant_xyz',
                'timestamp': datetime.now().isoformat(),
                'type': 'purchase',
                'currency': 'USD',
                'description': 'High-value electronics purchase'
            }
            
            print(f"\nDetecting fraud for transaction: {transaction['transaction_id']}")
            print(f"Amount: ${transaction['amount']}")
            print(f"Merchant: {transaction['merchant_id']}")
            
            # Detect fraud through Brain
            result = brain.detect_fraud(transaction)
            
            print(f"\nFraud Detection Result:")
            print(f"  Fraud Detected: {result.fraud_detected}")
            print(f"  Fraud Probability: {result.fraud_probability:.3f}")
            print(f"  Risk Score: {result.risk_score:.3f}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Detection Strategy: {result.detection_strategy.value}")
            print(f"  Explanation: {result.explanation}")
            
            # Get fraud system status through Brain
            fraud_status = brain.get_fraud_system_status()
            print(f"\nFraud System Status:")
            print(f"  Integrated: {fraud_status['integrated']}")
            print(f"  Total Predictions: {fraud_status['connector_state']['performance']['total_predictions']}")
            
            # Cleanup
            brain.shutdown()
            print("\nBrain system shutdown complete")
            
        except Exception as e:
            print(f"\nError in Example 1: {e}")
            import traceback
            traceback.print_exc()
    
    @staticmethod
    def example_2_direct_fraud_system():
        """Example 2: Direct fraud system access"""
        print("\n" + "="*80)
        print("EXAMPLE 2: Direct Fraud System Access")
        print("="*80)
        
        try:
            # Import fraud system
            from financial_fraud_domain.enhanced_fraud_core_system import (
                SaraphisEnhancedFraudSystem, create_production_fraud_system
            )
            
            # Initialize fraud system
            fraud_system = create_production_fraud_system()
            
            # Get system info
            component_info = fraud_system.get_component_info()
            print("\nFraud System Components:")
            for component, info in component_info.items():
                if isinstance(info, dict) and info.get('available'):
                    print(f"  ✓ {component}")
            
            # Sample transaction
            transaction = {
                'transaction_id': 'tx_example_002',
                'user_id': 'user_456',
                'amount': 125.50,
                'merchant_id': 'coffee_shop_abc',
                'timestamp': datetime.now().isoformat(),
                'type': 'purchase',
                'currency': 'USD',
                'description': 'Morning coffee and breakfast'
            }
            
            # User context for security
            user_context = {
                'user_id': 'user_456',
                'session_id': str(uuid.uuid4()),
                'ip_address': '192.168.1.100',
                'user_agent': 'Mozilla/5.0',
                'permissions': ['read', 'detect_fraud'],
                'auth_method': 'oauth2'
            }
            
            print(f"\nDetecting fraud for transaction: {transaction['transaction_id']}")
            
            # Direct fraud detection
            result = fraud_system.detect_fraud(transaction, user_context)
            
            print(f"\nFraud Detection Result:")
            print(f"  Fraud Detected: {result.fraud_detected}")
            print(f"  Fraud Probability: {result.fraud_probability:.3f}")
            print(f"  Risk Score: {result.risk_score:.3f}")
            print(f"  Validation Passed: {result.validation_passed}")
            
            # Get performance metrics
            performance = fraud_system.monitor_fraud_performance()
            print(f"\nPerformance Metrics:")
            print(f"  Total Detections: {performance['system_metrics']['total_detections']}")
            print(f"  Success Rate: {performance['detection_performance']['success_rate']:.2%}")
            print(f"  Average Latency: {performance['detection_performance']['average_latency_ms']:.2f}ms")
            
            # Cleanup
            fraud_system.shutdown()
            print("\nFraud system shutdown complete")
            
        except Exception as e:
            print(f"\nError in Example 2: {e}")
            import traceback
            traceback.print_exc()
    
    @staticmethod
    def example_3_batch_processing():
        """Example 3: Batch fraud detection with both systems"""
        print("\n" + "="*80)
        print("EXAMPLE 3: Batch Fraud Detection")
        print("="*80)
        
        try:
            from independent_core.brain import Brain
            from financial_fraud_domain.enhanced_fraud_core_system import create_production_fraud_system
            
            # Initialize both systems
            brain = Brain()
            fraud_system = create_production_fraud_system()
            
            # Generate batch of transactions
            transactions = [
                {
                    'transaction_id': f'tx_batch_{i:03d}',
                    'user_id': f'user_{i % 10}',
                    'amount': 100.0 * (i + 1),
                    'merchant_id': 'suspicious_merchant_xyz' if i % 5 == 0 else f'merchant_{i % 3}',
                    'timestamp': datetime.now().isoformat(),
                    'type': 'purchase',
                    'currency': 'USD'
                }
                for i in range(10)
            ]
            
            print(f"Processing {len(transactions)} transactions...")
            
            # Process through Brain
            print("\nProcessing through Brain system:")
            start_time = time.time()
            
            brain_results = []
            for tx in transactions:
                result = brain.detect_fraud(tx)
                brain_results.append(result)
            
            brain_time = time.time() - start_time
            brain_fraud_count = sum(1 for r in brain_results if r.fraud_detected)
            
            print(f"  Time: {brain_time:.3f}s")
            print(f"  Fraudulent: {brain_fraud_count}/{len(transactions)}")
            print(f"  Avg time per transaction: {brain_time/len(transactions)*1000:.2f}ms")
            
            # Process through direct fraud system
            print("\nProcessing through direct fraud system:")
            start_time = time.time()
            
            direct_results = fraud_system.batch_detect_fraud(transactions)
            
            direct_time = time.time() - start_time
            direct_fraud_count = sum(1 for r in direct_results if r.fraud_detected)
            
            print(f"  Time: {direct_time:.3f}s")
            print(f"  Fraudulent: {direct_fraud_count}/{len(transactions)}")
            print(f"  Avg time per transaction: {direct_time/len(transactions)*1000:.2f}ms")
            
            # Compare results
            print("\nResults Comparison:")
            print(f"  Speed improvement: {((brain_time - direct_time) / brain_time * 100):.1f}%")
            print(f"  Consistency: {brain_fraud_count == direct_fraud_count}")
            
            # Cleanup
            brain.shutdown()
            fraud_system.shutdown()
            
        except Exception as e:
            print(f"\nError in Example 3: {e}")
            import traceback
            traceback.print_exc()
    
    @staticmethod
    def example_4_configuration_management():
        """Example 4: Configuration management across systems"""
        print("\n" + "="*80)
        print("EXAMPLE 4: Configuration Management")
        print("="*80)
        
        try:
            from independent_core.brain import Brain
            
            # Initialize Brain
            brain = Brain()
            
            # Current configuration
            current_status = brain.get_fraud_system_status()
            print("Current Fraud Detection Configuration:")
            print(f"  Domain: {current_status['configuration']['fraud_domain_name']}")
            print(f"  Auto-registration: {current_status['configuration']['auto_registration']}")
            
            # Configure fraud detection
            new_config = {
                'detection_strategy': 'ensemble',
                'fraud_probability_threshold': 0.65,
                'risk_score_threshold': 0.75,
                'confidence_threshold': 0.55,
                'cache_predictions': True,
                'cache_ttl_seconds': 1800
            }
            
            print("\nApplying new configuration...")
            config_result = brain.configure_fraud_detection(new_config)
            
            print(f"Configuration Result:")
            print(f"  Success: {config_result['success']}")
            print(f"  Updated settings: {config_result['updates']}")
            
            # Test with new configuration
            test_transaction = {
                'transaction_id': 'tx_config_test',
                'user_id': 'user_999',
                'amount': 7500.00,
                'merchant_id': 'test_merchant',
                'timestamp': datetime.now().isoformat(),
                'type': 'purchase'
            }
            
            result = brain.detect_fraud(test_transaction)
            print(f"\nTest Detection with New Config:")
            print(f"  Strategy Used: {result.detection_strategy.value}")
            print(f"  Fraud Detected: {result.fraud_detected}")
            
            # Cleanup
            brain.shutdown()
            
        except Exception as e:
            print(f"\nError in Example 4: {e}")
            import traceback
            traceback.print_exc()
    
    @staticmethod
    def example_5_monitoring_and_health():
        """Example 5: System monitoring and health checks"""
        print("\n" + "="*80)
        print("EXAMPLE 5: System Monitoring and Health")
        print("="*80)
        
        try:
            from financial_fraud_domain.enhanced_fraud_core_system import create_production_fraud_system
            
            # Initialize system
            fraud_system = create_production_fraud_system()
            
            # Run some transactions to generate metrics
            print("Generating activity...")
            test_transactions = [
                {
                    'transaction_id': f'tx_monitor_{i}',
                    'user_id': f'user_{i}',
                    'amount': 100.0 + (i * 50),
                    'merchant_id': f'merchant_{i % 5}',
                    'timestamp': datetime.now().isoformat(),
                    'type': 'purchase'
                }
                for i in range(20)
            ]
            
            results = fraud_system.batch_detect_fraud(test_transactions)
            
            # Get comprehensive status
            print("\nSystem Status:")
            status = fraud_system.get_system_status()
            print(f"  System Initialized: {status['system_initialized']}")
            print(f"  Active Components: {len(status['active_components'])}")
            print(f"  Brain Integrated: {status['brain_integration']['integrated']}")
            
            # Performance monitoring
            print("\nPerformance Monitoring:")
            performance = fraud_system.monitor_fraud_performance()
            print(f"  Total Detections: {performance['system_metrics']['total_detections']}")
            print(f"  Success Rate: {performance['detection_performance']['success_rate']:.2%}")
            print(f"  Average Latency: {performance['detection_performance']['average_latency_ms']:.2f}ms")
            print(f"  Brain Usage Rate: {performance['detection_performance']['brain_usage_rate']:.2%}")
            
            # Health check
            print("\nHealth Check:")
            health = fraud_system.run_health_check()
            print(f"  Overall Health: {health['overall_health']}")
            
            if 'brain_integration_health' in health:
                brain_health = health['brain_integration_health']
                print(f"  Brain Connected: {brain_health.get('brain_connected', False)}")
                print(f"  Domain Registered: {brain_health.get('domain_registered', False)}")
                print(f"  Circuit Breaker: {'OPEN' if brain_health.get('circuit_breaker_open', False) else 'CLOSED'}")
                print(f"  Cache Hit Rate: {brain_health.get('cache_hit_rate', 0):.2%}")
            
            # Cleanup
            fraud_system.shutdown()
            
        except Exception as e:
            print(f"\nError in Example 5: {e}")
            import traceback
            traceback.print_exc()

# ======================== INTEGRATION TESTING ========================

class BrainFraudIntegrationTests(unittest.TestCase):
    """Comprehensive integration tests for Brain-Fraud system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_base_path = Path.cwd() / ".brain_fraud_test"
        cls.test_base_path.mkdir(exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        import shutil
        if cls.test_base_path.exists():
            shutil.rmtree(cls.test_base_path)
    
    def setUp(self):
        """Set up each test"""
        self.brain = None
        self.fraud_system = None
        self.connector = None
    
    def tearDown(self):
        """Clean up after each test"""
        if self.brain:
            self.brain.shutdown()
        if self.fraud_system:
            self.fraud_system.shutdown()
        if self.connector:
            self.connector.shutdown()
    
    def test_brain_fraud_domain_registration(self):
        """Test automatic fraud domain registration in Brain"""
        from independent_core.brain import Brain
        
        # Initialize Brain
        self.brain = Brain()
        
        # Check if fraud domain is registered
        domains = self.brain.list_available_domains()
        domain_names = [d['name'] for d in domains]
        
        self.assertIn('financial_fraud', domain_names)
        
        # Get fraud domain info
        fraud_domain = next((d for d in domains if d['name'] == 'financial_fraud'), None)
        self.assertIsNotNone(fraud_domain)
        self.assertEqual(fraud_domain['type'], 'specialized')
        self.assertTrue(fraud_domain['is_trained'])
    
    def test_brain_detect_fraud_method(self):
        """Test Brain.detect_fraud() method"""
        from independent_core.brain import Brain
        
        self.brain = Brain()
        
        # Test transaction
        transaction = {
            'transaction_id': 'test_001',
            'user_id': 'test_user',
            'amount': 1000.00,
            'merchant_id': 'test_merchant',
            'timestamp': datetime.now().isoformat(),
            'type': 'purchase'
        }
        
        # Detect fraud through Brain
        result = self.brain.detect_fraud(transaction)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result.fraud_detected, bool)
        self.assertIsInstance(result.fraud_probability, float)
        self.assertGreaterEqual(result.fraud_probability, 0.0)
        self.assertLessEqual(result.fraud_probability, 1.0)
    
    def test_direct_fraud_system_access(self):
        """Test direct fraud system access"""
        from financial_fraud_domain.enhanced_fraud_core_system import create_test_fraud_system
        
        self.fraud_system = create_test_fraud_system()
        
        # Test transaction
        transaction = {
            'transaction_id': 'test_002',
            'user_id': 'test_user',
            'amount': 500.00,
            'merchant_id': 'test_merchant',
            'timestamp': datetime.now().isoformat(),
            'type': 'purchase'
        }
        
        # Direct fraud detection
        result = self.fraud_system.detect_fraud(transaction)
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'fraud_detected'))
        self.assertTrue(hasattr(result, 'detection_strategy'))
    
    def test_brain_fraud_consistency(self):
        """Test consistency between Brain and direct access"""
        from independent_core.brain import Brain
        from financial_fraud_domain.enhanced_fraud_core_system import create_test_fraud_system
        
        self.brain = Brain()
        self.fraud_system = create_test_fraud_system()
        
        # Test with same transaction
        transaction = {
            'transaction_id': 'test_consistency',
            'user_id': 'test_user',
            'amount': 15000.00,  # High amount to trigger fraud
            'merchant_id': 'suspicious_merchant_xyz',
            'timestamp': datetime.now().isoformat(),
            'type': 'purchase'
        }
        
        # Get results from both systems
        brain_result = self.brain.detect_fraud(transaction)
        direct_result = self.fraud_system.detect_fraud(transaction, use_brain=False)
        
        # Both should detect fraud for high-risk transaction
        self.assertEqual(brain_result.fraud_detected, direct_result.fraud_detected)
        
        # Probabilities should be similar (within 10%)
        prob_diff = abs(brain_result.fraud_probability - direct_result.fraud_probability)
        self.assertLess(prob_diff, 0.1)
    
    def test_configuration_synchronization(self):
        """Test configuration synchronization between systems"""
        from independent_core.brain import Brain
        
        self.brain = Brain()
        
        # Configure through Brain
        config = {
            'detection_strategy': 'ml_only',
            'fraud_probability_threshold': 0.55,
            'cache_predictions': False
        }
        
        result = self.brain.configure_fraud_detection(config)
        self.assertTrue(result['success'])
        
        # Verify configuration was applied
        test_transaction = {
            'transaction_id': 'test_config',
            'user_id': 'test_user',
            'amount': 1000.00,
            'merchant_id': 'test_merchant',
            'timestamp': datetime.now().isoformat(),
            'type': 'purchase'
        }
        
        result = self.brain.detect_fraud(test_transaction)
        self.assertEqual(result.detection_strategy.value, 'ml_only')
    
    def test_batch_processing_performance(self):
        """Test batch processing performance"""
        from financial_fraud_domain.enhanced_fraud_core_system import create_test_fraud_system
        
        self.fraud_system = create_test_fraud_system()
        
        # Generate batch
        batch_size = 100
        transactions = [
            {
                'transaction_id': f'batch_{i}',
                'user_id': f'user_{i % 10}',
                'amount': 100.0 + (i * 10),
                'merchant_id': f'merchant_{i % 5}',
                'timestamp': datetime.now().isoformat(),
                'type': 'purchase'
            }
            for i in range(batch_size)
        ]
        
        # Process batch
        start_time = time.time()
        results = self.fraud_system.batch_detect_fraud(transactions)
        batch_time = time.time() - start_time
        
        self.assertEqual(len(results), batch_size)
        self.assertLess(batch_time, 10.0)  # Should complete within 10 seconds
        
        # Check all results are valid
        for result in results:
            self.assertIsInstance(result.fraud_detected, bool)
            self.assertIsInstance(result.fraud_probability, float)
    
    def test_error_recovery(self):
        """Test error recovery mechanisms"""
        from financial_fraud_domain.enhanced_fraud_core_system import create_test_fraud_system
        
        self.fraud_system = create_test_fraud_system()
        
        # Invalid transaction (missing required fields)
        invalid_transaction = {
            'transaction_id': 'invalid_001',
            # Missing user_id, amount, timestamp
        }
        
        # Should not crash, should return error result
        result = self.fraud_system.detect_fraud(invalid_transaction)
        
        self.assertIsNotNone(result)
        self.assertFalse(result.validation_passed)
        self.assertTrue(len(result.validation_errors) > 0)
    
    def test_security_context_handling(self):
        """Test security context handling"""
        from financial_fraud_domain.enhanced_fraud_core_system import create_production_fraud_system
        
        self.fraud_system = create_production_fraud_system()
        
        # Transaction
        transaction = {
            'transaction_id': 'security_test',
            'user_id': 'test_user',
            'amount': 500.00,
            'merchant_id': 'test_merchant',
            'timestamp': datetime.now().isoformat(),
            'type': 'purchase'
        }
        
        # User context with security info
        user_context = {
            'user_id': 'test_user',
            'session_id': str(uuid.uuid4()),
            'ip_address': '10.0.0.1',
            'permissions': ['read', 'detect_fraud'],
            'auth_method': 'api_key'
        }
        
        # Should work with security context
        result = self.fraud_system.detect_fraud(transaction, user_context)
        self.assertIsNotNone(result)
        self.assertTrue(result.validation_passed)
    
    def test_monitoring_and_metrics(self):
        """Test monitoring and metrics collection"""
        from financial_fraud_domain.enhanced_fraud_core_system import create_test_fraud_system
        
        self.fraud_system = create_test_fraud_system()
        
        # Generate some activity
        for i in range(10):
            transaction = {
                'transaction_id': f'metrics_{i}',
                'user_id': f'user_{i}',
                'amount': 100.0 * (i + 1),
                'merchant_id': f'merchant_{i % 3}',
                'timestamp': datetime.now().isoformat(),
                'type': 'purchase'
            }
            self.fraud_system.detect_fraud(transaction)
        
        # Get performance metrics
        metrics = self.fraud_system.monitor_fraud_performance()
        
        self.assertEqual(metrics['system_metrics']['total_detections'], 10)
        self.assertGreater(metrics['detection_performance']['success_rate'], 0.9)
        self.assertIsInstance(metrics['detection_performance']['average_latency_ms'], float)

# ======================== PERFORMANCE BENCHMARKS ========================

class BrainFraudPerformanceBenchmarks:
    """Performance benchmarks for Brain-Fraud integration"""
    
    @staticmethod
    def benchmark_latency():
        """Benchmark detection latency"""
        print("\n" + "="*80)
        print("LATENCY BENCHMARK")
        print("="*80)
        
        from independent_core.brain import Brain
        from financial_fraud_domain.enhanced_fraud_core_system import create_production_fraud_system
        
        brain = Brain()
        fraud_system = create_production_fraud_system()
        
        # Test transaction
        transaction = {
            'transaction_id': 'bench_latency',
            'user_id': 'bench_user',
            'amount': 1000.00,
            'merchant_id': 'bench_merchant',
            'timestamp': datetime.now().isoformat(),
            'type': 'purchase'
        }
        
        # Warm up
        for _ in range(5):
            brain.detect_fraud(transaction)
            fraud_system.detect_fraud(transaction, use_brain=False)
        
        # Benchmark Brain detection
        brain_times = []
        for i in range(100):
            transaction['transaction_id'] = f'bench_brain_{i}'
            start = time.time()
            brain.detect_fraud(transaction)
            brain_times.append((time.time() - start) * 1000)
        
        # Benchmark direct detection
        direct_times = []
        for i in range(100):
            transaction['transaction_id'] = f'bench_direct_{i}'
            start = time.time()
            fraud_system.detect_fraud(transaction, use_brain=False)
            direct_times.append((time.time() - start) * 1000)
        
        # Results
        print(f"\nBrain Detection Latency:")
        print(f"  Average: {sum(brain_times)/len(brain_times):.2f}ms")
        print(f"  Min: {min(brain_times):.2f}ms")
        print(f"  Max: {max(brain_times):.2f}ms")
        print(f"  P95: {sorted(brain_times)[95]:.2f}ms")
        
        print(f"\nDirect Detection Latency:")
        print(f"  Average: {sum(direct_times)/len(direct_times):.2f}ms")
        print(f"  Min: {min(direct_times):.2f}ms")
        print(f"  Max: {max(direct_times):.2f}ms")
        print(f"  P95: {sorted(direct_times)[95]:.2f}ms")
        
        brain.shutdown()
        fraud_system.shutdown()
    
    @staticmethod
    def benchmark_throughput():
        """Benchmark detection throughput"""
        print("\n" + "="*80)
        print("THROUGHPUT BENCHMARK")
        print("="*80)
        
        from financial_fraud_domain.enhanced_fraud_core_system import create_production_fraud_system
        
        fraud_system = create_production_fraud_system()
        
        # Generate transactions
        num_transactions = 1000
        transactions = [
            {
                'transaction_id': f'throughput_{i}',
                'user_id': f'user_{i % 100}',
                'amount': 100.0 + (i % 1000),
                'merchant_id': f'merchant_{i % 50}',
                'timestamp': datetime.now().isoformat(),
                'type': 'purchase'
            }
            for i in range(num_transactions)
        ]
        
        # Benchmark batch processing
        print(f"Processing {num_transactions} transactions in batch...")
        start = time.time()
        results = fraud_system.batch_detect_fraud(transactions)
        batch_time = time.time() - start
        
        print(f"\nBatch Processing Results:")
        print(f"  Total time: {batch_time:.2f}s")
        print(f"  Transactions per second: {num_transactions/batch_time:.2f}")
        print(f"  Average time per transaction: {batch_time/num_transactions*1000:.2f}ms")
        
        fraud_detected = sum(1 for r in results if r.fraud_detected)
        print(f"  Fraudulent transactions: {fraud_detected}/{num_transactions} ({fraud_detected/num_transactions*100:.1f}%)")
        
        fraud_system.shutdown()
    
    @staticmethod
    def benchmark_memory_usage():
        """Benchmark memory usage"""
        print("\n" + "="*80)
        print("MEMORY USAGE BENCHMARK")
        print("="*80)
        
        try:
            import psutil
            import os
            
            from financial_fraud_domain.enhanced_fraud_core_system import create_production_fraud_system
            
            process = psutil.Process(os.getpid())
            
            # Baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024
            print(f"Baseline memory: {baseline_memory:.2f} MB")
            
            # Initialize system
            fraud_system = create_production_fraud_system()
            initialized_memory = process.memory_info().rss / 1024 / 1024
            print(f"After initialization: {initialized_memory:.2f} MB (+{initialized_memory - baseline_memory:.2f} MB)")
            
            # Process transactions
            for i in range(1000):
                transaction = {
                    'transaction_id': f'mem_{i}',
                    'user_id': f'user_{i % 100}',
                    'amount': 100.0 + i,
                    'merchant_id': f'merchant_{i % 50}',
                    'timestamp': datetime.now().isoformat(),
                    'type': 'purchase'
                }
                fraud_system.detect_fraud(transaction)
                
                if i % 100 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    print(f"After {i} transactions: {current_memory:.2f} MB")
            
            final_memory = process.memory_info().rss / 1024 / 1024
            print(f"\nFinal memory: {final_memory:.2f} MB")
            print(f"Total increase: {final_memory - baseline_memory:.2f} MB")
            
            fraud_system.shutdown()
            
        except ImportError:
            print("psutil not available, skipping memory benchmark")

# ======================== MAIN EXECUTION ========================

def run_all_examples():
    """Run all integration examples"""
    examples = BrainFraudIntegrationExamples()
    
    examples.example_1_basic_brain_integration()
    time.sleep(1)
    
    examples.example_2_direct_fraud_system()
    time.sleep(1)
    
    examples.example_3_batch_processing()
    time.sleep(1)
    
    examples.example_4_configuration_management()
    time.sleep(1)
    
    examples.example_5_monitoring_and_health()

def run_all_benchmarks():
    """Run all performance benchmarks"""
    benchmarks = BrainFraudPerformanceBenchmarks()
    
    benchmarks.benchmark_latency()
    time.sleep(1)
    
    benchmarks.benchmark_throughput()
    time.sleep(1)
    
    benchmarks.benchmark_memory_usage()

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("BRAIN-FRAUD SYSTEM INTEGRATION")
    print("Complete Production-Ready Implementation")
    print("="*80)
    
    # Run examples
    print("\n\nRUNNING INTEGRATION EXAMPLES...")
    run_all_examples()
    
    # Run tests
    print("\n\nRUNNING INTEGRATION TESTS...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run benchmarks
    print("\n\nRUNNING PERFORMANCE BENCHMARKS...")
    run_all_benchmarks()
    
    print("\n\n" + "="*80)
    print("INTEGRATION DEMONSTRATION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()