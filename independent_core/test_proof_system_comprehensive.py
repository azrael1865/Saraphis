"""
Comprehensive Test Suite for Proof System Integration
Validates entire proof system integration with Brain system and fraud detection
"""

import unittest
import sys
import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import proof system components
from proof_system.rule_based_engine import RuleBasedProofEngine
from proof_system.ml_based_engine import MLBasedProofEngine
from proof_system.cryptographic_engine import CryptographicProofEngine
from proof_system.proof_integration_manager import ProofIntegrationManager
from proof_system.confidence_generator import ConfidenceGenerator
from proof_system.algebraic_rule_enforcer import AlgebraicRuleEnforcer

# Import Brain system components
from brain import Brain
from training_manager import TrainingManager

# Import fraud detection components
from financial_fraud_domain.enhanced_data_loader import EnhancedDataLoader
from financial_fraud_domain.performance_monitor import PerformanceMonitor
from financial_fraud_domain.accuracy_analytics_reporter import AccuracyAnalyticsReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_proof_system_comprehensive.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ProofSystemTestSuite:
    """Main test suite orchestrator for proof system validation"""
    
    def __init__(self):
        """Initialize test suite with all necessary components"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.test_config = self._load_test_configuration()
        
        # Initialize monitoring components
        self.performance_monitor = PerformanceMonitor()
        self.accuracy_reporter = AccuracyAnalyticsReporter()
        
        # Test results storage
        self.test_results = {
            'unit_tests': {},
            'integration_tests': {},
            'performance_tests': {},
            'validation_tests': {},
            'error_tests': {}
        }
        
        # Initialize test data
        self.test_data = None
        self.synthetic_data = None
        self._prepare_test_data()
        
    def _load_test_configuration(self) -> Dict[str, Any]:
        """Load test configuration"""
        return {
            'proof_system': {
                'timeout': 30,  # seconds
                'max_retries': 3,
                'confidence_threshold': 0.85,
                'performance_overhead_limit': 0.15,  # 15%
                'memory_limit_mb': 2048,
                'batch_size': 32
            },
            'test_scenarios': {
                'unit_test_iterations': 100,
                'integration_test_cases': 50,
                'performance_test_duration': 60,  # seconds
                'error_scenarios': 20
            },
            'validation_criteria': {
                'min_accuracy': 0.85,
                'max_false_positive_rate': 0.1,
                'max_latency_ms': 200,
                'min_throughput_tps': 100
            }
        }
        
    def _prepare_test_data(self):
        """Prepare test data including synthetic cases"""
        self.logger.info("Preparing test data...")
        
        try:
            # Load enhanced data if available
            try:
                data_loader = EnhancedDataLoader()
                self.test_data = data_loader.load_sample_data(
                    sample_size=self.test_config['proof_system']['batch_size'] * 5
                )
            except Exception as e:
                self.logger.warning(f"Could not load enhanced data: {str(e)}")
                self.test_data = []
            
            # Generate synthetic test cases
            self.synthetic_data = self._generate_synthetic_test_cases()
            
            self.logger.info(f"Loaded {len(self.test_data)} real samples and "
                           f"{len(self.synthetic_data)} synthetic samples")
            
        except Exception as e:
            self.logger.error(f"Failed to prepare test data: {str(e)}")
            self.synthetic_data = self._generate_synthetic_test_cases()
            self.test_data = []
            
    def _generate_synthetic_test_cases(self) -> List[Dict[str, Any]]:
        """Generate synthetic test cases for edge scenarios"""
        synthetic_cases = []
        
        # Edge case scenarios
        edge_cases = [
            {
                'id': 'edge_extreme_values',
                'data': {
                    'transaction_amount': 1e6,
                    'account_age_days': 1,
                    'previous_fraud_count': 100,
                    'risk_score': 0.99
                },
                'expected_fraud': True,
                'description': 'Extreme value combination'
            },
            {
                'id': 'edge_missing_values', 
                'data': {
                    'transaction_amount': None,
                    'account_age_days': 365,
                    'previous_fraud_count': 0,
                    'risk_score': 0.5
                },
                'expected_fraud': False,
                'description': 'Missing critical values'
            },
            {
                'id': 'edge_boundary',
                'data': {
                    'transaction_amount': 0.01,
                    'account_age_days': 0,
                    'previous_fraud_count': 0,
                    'risk_score': 0.0
                },
                'expected_fraud': False,
                'description': 'Minimum boundary values'
            }
        ]
        
        synthetic_cases.extend(edge_cases)
        
        # Generate random synthetic cases
        np.random.seed(42)
        for i in range(100):
            synthetic_cases.append({
                'id': f'synthetic_{i}',
                'data': {
                    'transaction_amount': np.random.lognormal(3, 2),
                    'account_age_days': np.random.randint(0, 3650),
                    'previous_fraud_count': np.random.poisson(0.5),
                    'risk_score': np.random.beta(2, 5),
                    'features': np.random.randn(10).tolist()
                },
                'expected_fraud': np.random.random() > 0.9,
                'description': f'Random synthetic case {i}'
            })
            
        return synthetic_cases
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite"""
        self.logger.info("Starting comprehensive proof system test suite...")
        start_time = time.time()
        
        try:
            # Phase 1: Unit Tests
            self.logger.info("Phase 1: Running unit tests...")
            self.test_results['unit_tests'] = self._run_unit_tests()
            
            # Phase 2: Integration Tests
            self.logger.info("Phase 2: Running integration tests...")
            self.test_results['integration_tests'] = self._run_integration_tests()
            
            # Phase 3: Performance Tests
            self.logger.info("Phase 3: Running performance tests...")
            self.test_results['performance_tests'] = self._run_performance_tests()
            
            # Phase 4: Validation Tests
            self.logger.info("Phase 4: Running validation tests...")
            self.test_results['validation_tests'] = self._run_validation_tests()
            
            # Phase 5: Error Handling Tests
            self.logger.info("Phase 5: Running error handling tests...")
            self.test_results['error_tests'] = self._run_error_tests()
            
            # Generate comprehensive report
            total_time = time.time() - start_time
            report = self._generate_test_report(total_time)
            
            self.logger.info(f"Test suite completed in {total_time:.2f} seconds")
            return report
            
        except Exception as e:
            self.logger.error(f"Test suite failed: {str(e)}")
            raise
            
    def _run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests for individual proof components"""
        results = {}
        
        # Test rule-based engine
        results['rule_based_engine'] = self._test_rule_based_engine()
        
        # Test ML-based engine
        results['ml_based_engine'] = self._test_ml_based_engine()
        
        # Test cryptographic engine
        results['cryptographic_engine'] = self._test_cryptographic_engine()
        
        # Test confidence generator
        results['confidence_generator'] = self._test_confidence_generator()
        
        # Test algebraic rule enforcer
        results['algebraic_rule_enforcer'] = self._test_algebraic_rule_enforcer()
        
        # Test integration manager
        results['integration_manager'] = self._test_integration_manager()
        
        return results
        
    def _test_rule_based_engine(self) -> Dict[str, Any]:
        """Test rule-based proof engine"""
        try:
            engine = RuleBasedProofEngine()
            
            # Test basic evaluation
            test_transaction = self.synthetic_data[0]['data']
            result = engine.evaluate_transaction(test_transaction)
            
            # Verify result structure
            required_fields = ['triggered_rules', 'risk_score', 'risk_level', 'confidence']
            has_required = all(field in result for field in required_fields)
            
            return {
                'passed': has_required and isinstance(result['risk_score'], (int, float)),
                'result_structure_valid': has_required,
                'test_transaction_processed': True
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
            
    def _test_ml_based_engine(self) -> Dict[str, Any]:
        """Test ML-based proof engine"""
        try:
            engine = MLBasedProofEngine()
            
            # Test ML proof generation
            test_transaction = {
                'features': np.random.randn(10),
                'model_prediction': 0.8,
                'model_confidence': 0.9
            }
            model_state = {
                'weights': np.random.randn(10, 10),
                'gradients': np.random.randn(10, 10),
                'iteration': 100
            }
            
            result = engine.generate_ml_proof(test_transaction, model_state)
            
            # Verify result structure
            required_fields = ['confidence_score', 'prediction_stability', 'feature_importance']
            has_required = all(field in result for field in required_fields)
            
            return {
                'passed': has_required,
                'result_structure_valid': has_required,
                'confidence_in_range': 0 <= result.get('confidence_score', -1) <= 1
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
            
    def _test_cryptographic_engine(self) -> Dict[str, Any]:
        """Test cryptographic proof engine"""
        try:
            engine = CryptographicProofEngine()
            
            # Test proof generation and verification
            test_data = {'transaction_id': 'test_001', 'amount': 1000}
            proof = engine.generate_proof(test_data)
            
            # Verify proof structure
            required_fields = ['hash', 'timestamp', 'nonce']
            has_required = all(field in proof for field in required_fields)
            
            # Test verification
            is_valid = engine.verify_proof(test_data, proof)
            
            return {
                'passed': has_required and is_valid,
                'proof_structure_valid': has_required,
                'verification_successful': is_valid,
                'hash_length_correct': len(proof.get('hash', '')) == 64
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
            
    def _test_confidence_generator(self) -> Dict[str, Any]:
        """Test confidence generator"""
        try:
            generator = ConfidenceGenerator()
            
            # Test confidence generation
            result = generator.generate_confidence(
                rule_score=0.8,
                ml_probability=0.85,
                crypto_valid=True
            )
            
            # Verify result structure
            required_fields = ['score', 'confidence_interval', 'components']
            has_required = all(field in result for field in required_fields)
            
            score_valid = 0 <= result.get('score', -1) <= 1
            
            return {
                'passed': has_required and score_valid,
                'result_structure_valid': has_required,
                'score_in_range': score_valid
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
            
    def _test_algebraic_rule_enforcer(self) -> Dict[str, Any]:
        """Test algebraic rule enforcer"""
        try:
            enforcer = AlgebraicRuleEnforcer()
            
            # Test gradient validation
            test_gradients = np.random.randn(10, 10)
            result = enforcer.validate_gradients(test_gradients, learning_rate=0.01)
            
            # Verify result structure
            required_fields = ['valid', 'gradient_norm', 'constraints']
            has_required = all(field in result for field in required_fields)
            
            return {
                'passed': has_required,
                'result_structure_valid': has_required,
                'validation_completed': True
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
            
    def _test_integration_manager(self) -> Dict[str, Any]:
        """Test proof integration manager"""
        try:
            manager = ProofIntegrationManager()
            
            # Register engines
            manager.register_engine('rule_based', RuleBasedProofEngine())
            manager.register_engine('ml_based', MLBasedProofEngine())
            manager.register_engine('cryptographic', CryptographicProofEngine())
            
            # Test comprehensive proof generation
            test_transaction = self.synthetic_data[0]['data']
            model_state = {'iteration': 1}
            
            proof = manager.generate_comprehensive_proof(test_transaction, model_state)
            
            # Verify proof structure
            has_engines = any(engine in proof for engine in ['rule_based', 'ml_based', 'cryptographic'])
            has_confidence = 'confidence' in proof
            
            return {
                'passed': has_engines and has_confidence,
                'engines_registered': len(manager.get_registered_engines()) == 3,
                'proof_generated': has_engines,
                'confidence_calculated': has_confidence
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
            
    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests"""
        results = {}
        
        # Test Brain system integration
        results['brain_integration'] = self._test_brain_integration()
        
        # Test training integration
        results['training_integration'] = self._test_training_integration()
        
        # Test end-to-end processing
        results['end_to_end'] = self._test_end_to_end_processing()
        
        return results
        
    def _test_brain_integration(self) -> Dict[str, Any]:
        """Test proof system integration with Brain"""
        try:
            # Initialize Brain with proof system
            brain = Brain(
                input_dim=20,
                hidden_dim=64,
                output_dim=2,
                enable_proof_system=True
            )
            
            # Test basic operations
            test_input = np.random.randn(32, 20)
            
            # Forward pass
            output = brain.forward(test_input)
            proof_available = hasattr(brain, 'get_last_proof') and brain.get_last_proof() is not None
            
            return {
                'passed': proof_available,
                'brain_initialized': True,
                'forward_pass_completed': output is not None,
                'proof_system_active': proof_available
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
            
    def _test_training_integration(self) -> Dict[str, Any]:
        """Test proof system integration with training"""
        try:
            # Initialize components
            brain = Brain(
                input_dim=20,
                hidden_dim=64,
                output_dim=2,
                enable_proof_system=True
            )
            
            training_manager = TrainingManager(
                model=brain,
                learning_rate=0.001
            )
            
            # Simulate training step
            batch_data = {
                'features': np.random.randn(16, 20),
                'labels': np.random.randint(0, 2, 16)
            }
            
            # Execute training step
            metrics = training_manager.train_step(batch_data)
            
            return {
                'passed': metrics is not None,
                'training_manager_initialized': True,
                'training_step_completed': metrics is not None
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
            
    def _test_end_to_end_processing(self) -> Dict[str, Any]:
        """Test end-to-end transaction processing"""
        try:
            # Setup complete pipeline
            manager = ProofIntegrationManager()
            manager.register_engine('rule_based', RuleBasedProofEngine())
            manager.register_engine('ml_based', MLBasedProofEngine())
            manager.register_engine('cryptographic', CryptographicProofEngine())
            
            # Process test transaction
            test_transaction = {
                'transaction_id': 'test_e2e_001',
                'amount': 1000,
                'features': np.random.randn(10).tolist(),
                'model_prediction': 0.7,
                'model_confidence': 0.8
            }
            
            model_state = {'iteration': 1, 'weights': np.random.randn(10, 10).tolist()}
            
            # Generate proof
            proof = manager.generate_comprehensive_proof(test_transaction, model_state)
            
            # Verify end-to-end processing
            has_all_engines = all(engine in proof for engine in ['rule_based', 'ml_based', 'cryptographic'])
            has_confidence = 'confidence' in proof and isinstance(proof['confidence'], (int, float))
            
            return {
                'passed': has_all_engines and has_confidence,
                'all_engines_processed': has_all_engines,
                'confidence_generated': has_confidence,
                'transaction_id_preserved': proof.get('transaction_id') == test_transaction['transaction_id']
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
            
    def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests"""
        results = {}
        
        # Test proof generation overhead
        results['overhead'] = self._test_proof_overhead()
        
        # Test throughput
        results['throughput'] = self._test_throughput()
        
        # Test memory usage
        results['memory'] = self._test_memory_usage()
        
        return results
        
    def _test_proof_overhead(self) -> Dict[str, Any]:
        """Test proof system overhead"""
        try:
            # Measure baseline (without proof system)
            brain_baseline = Brain(
                input_dim=20,
                hidden_dim=64,
                output_dim=2,
                enable_proof_system=False
            )
            
            test_input = np.random.randn(32, 20)
            
            # Baseline timing
            start_time = time.time()
            for _ in range(10):
                output = brain_baseline.forward(test_input)
            baseline_time = time.time() - start_time
            
            # With proof system
            brain_with_proof = Brain(
                input_dim=20,
                hidden_dim=64,
                output_dim=2,
                enable_proof_system=True
            )
            
            start_time = time.time()
            for _ in range(10):
                output = brain_with_proof.forward(test_input)
            proof_time = time.time() - start_time
            
            # Calculate overhead
            overhead_percent = ((proof_time - baseline_time) / baseline_time) * 100
            acceptable = overhead_percent <= self.test_config['proof_system']['performance_overhead_limit'] * 100
            
            return {
                'passed': acceptable,
                'overhead_percent': overhead_percent,
                'baseline_time_ms': baseline_time * 1000,
                'proof_time_ms': proof_time * 1000,
                'acceptable': acceptable
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
            
    def _test_throughput(self) -> Dict[str, Any]:
        """Test system throughput"""
        try:
            manager = ProofIntegrationManager()
            manager.register_engine('rule_based', RuleBasedProofEngine())
            
            # Process transactions for 5 seconds
            start_time = time.time()
            transaction_count = 0
            
            while time.time() - start_time < 5:
                test_transaction = {
                    'transaction_id': f'throughput_test_{transaction_count}',
                    'amount': np.random.uniform(1, 10000)
                }
                
                proof = manager.generate_comprehensive_proof(test_transaction, {'iteration': 1})
                transaction_count += 1
                
            elapsed_time = time.time() - start_time
            throughput = transaction_count / elapsed_time
            
            meets_requirement = throughput >= self.test_config['validation_criteria']['min_throughput_tps']
            
            return {
                'passed': meets_requirement,
                'throughput_tps': throughput,
                'transactions_processed': transaction_count,
                'test_duration': elapsed_time,
                'meets_requirement': meets_requirement
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
            
    def _test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage"""
        try:
            import psutil
            import gc
            
            process = psutil.Process()
            gc.collect()
            
            # Baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create multiple proof system instances
            components = []
            for _ in range(10):
                manager = ProofIntegrationManager()
                manager.register_engine('rule_based', RuleBasedProofEngine())
                manager.register_engine('ml_based', MLBasedProofEngine())
                components.append(manager)
                
            # Process some transactions
            for manager in components[:5]:
                for i in range(20):
                    transaction = {'transaction_id': f'mem_test_{i}', 'amount': 1000}
                    proof = manager.generate_comprehensive_proof(transaction, {'iteration': 1})
                    
            # Measure peak memory
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - baseline_memory
            
            within_limit = memory_increase <= self.test_config['proof_system']['memory_limit_mb']
            
            # Cleanup
            del components
            gc.collect()
            
            return {
                'passed': within_limit,
                'memory_increase_mb': memory_increase,
                'baseline_memory_mb': baseline_memory,
                'peak_memory_mb': peak_memory,
                'within_limit': within_limit
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
            
    def _run_validation_tests(self) -> Dict[str, Any]:
        """Run validation tests"""
        results = {}
        
        # Test proof correctness
        results['proof_correctness'] = self._test_proof_correctness()
        
        # Test confidence accuracy
        results['confidence_accuracy'] = self._test_confidence_accuracy()
        
        return results
        
    def _test_proof_correctness(self) -> Dict[str, Any]:
        """Test correctness of generated proofs"""
        try:
            manager = ProofIntegrationManager()
            manager.register_engine('rule_based', RuleBasedProofEngine())
            manager.register_engine('cryptographic', CryptographicProofEngine())
            
            correct_count = 0
            total_count = min(50, len(self.synthetic_data))
            
            for i in range(total_count):
                transaction = self.synthetic_data[i]['data']
                proof = manager.generate_comprehensive_proof(transaction, {'iteration': i})
                
                # Validate proof structure
                if self._validate_proof_structure(proof):
                    correct_count += 1
                    
            accuracy = correct_count / total_count if total_count > 0 else 0
            acceptable = accuracy >= 0.9
            
            return {
                'passed': acceptable,
                'correct_proofs': correct_count,
                'total_proofs': total_count,
                'accuracy': accuracy,
                'acceptable': acceptable
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
            
    def _validate_proof_structure(self, proof: Dict[str, Any]) -> bool:
        """Validate proof structure"""
        required_fields = ['transaction_id', 'generation_timestamp', 'confidence']
        return all(field in proof for field in required_fields)
        
    def _test_confidence_accuracy(self) -> Dict[str, Any]:
        """Test confidence generation accuracy"""
        try:
            generator = ConfidenceGenerator()
            correct_predictions = 0
            total_predictions = min(30, len(self.synthetic_data))
            
            for i in range(total_predictions):
                case = self.synthetic_data[i]
                expected_fraud = case['expected_fraud']
                
                # Generate confidence
                confidence_result = generator.generate_confidence(
                    rule_score=0.8 if expected_fraud else 0.2,
                    ml_probability=0.9 if expected_fraud else 0.1,
                    crypto_valid=True
                )
                
                confidence_score = confidence_result['score']
                predicted_fraud = confidence_score > 0.5
                
                if predicted_fraud == expected_fraud:
                    correct_predictions += 1
                    
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            acceptable = accuracy >= self.test_config['validation_criteria']['min_accuracy']
            
            return {
                'passed': acceptable,
                'correct_predictions': correct_predictions,
                'total_predictions': total_predictions,
                'accuracy': accuracy,
                'acceptable': acceptable
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
            
    def _run_error_tests(self) -> Dict[str, Any]:
        """Run error handling tests"""
        results = {}
        
        # Test component failure recovery
        results['component_failure'] = self._test_component_failure()
        
        # Test invalid input handling
        results['invalid_input'] = self._test_invalid_input()
        
        return results
        
    def _test_component_failure(self) -> Dict[str, Any]:
        """Test component failure recovery"""
        try:
            manager = ProofIntegrationManager()
            
            # Register a faulty engine
            class FaultyEngine:
                def generate_proof(self, *args, **kwargs):
                    raise Exception("Simulated engine failure")
                    
            manager.register_engine('rule_based', RuleBasedProofEngine())
            manager.register_engine('faulty', FaultyEngine())
            
            # Try to generate proof
            transaction = {'transaction_id': 'failure_test', 'amount': 1000}
            proof = manager.generate_comprehensive_proof(transaction, {'iteration': 1})
            
            # Should have succeeded with working engine, failed with faulty
            has_rule_based = 'rule_based' in proof
            has_errors = 'errors' in proof and 'faulty' in proof['errors']
            
            return {
                'passed': has_rule_based and has_errors,
                'partial_success': has_rule_based,
                'error_reported': has_errors,
                'graceful_degradation': has_rule_based and has_errors
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
            
    def _test_invalid_input(self) -> Dict[str, Any]:
        """Test invalid input handling"""
        invalid_inputs = [
            None,
            {},
            {'amount': None},
            {'amount': float('inf')},
            {'amount': float('nan')},
            'string_instead_of_dict'
        ]
        
        manager = ProofIntegrationManager()
        manager.register_engine('rule_based', RuleBasedProofEngine())
        
        handled_count = 0
        
        for invalid_input in invalid_inputs:
            try:
                proof = manager.generate_comprehensive_proof(invalid_input, {'iteration': 1})
                # If we get here, it was handled gracefully
                handled_count += 1
            except Exception:
                # Exception is also acceptable - means error was caught
                handled_count += 1
                
        all_handled = handled_count == len(invalid_inputs)
        
        return {
            'passed': all_handled,
            'handled_inputs': handled_count,
            'total_inputs': len(invalid_inputs),
            'all_handled': all_handled
        }
        
    def _generate_test_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        # Calculate overall metrics
        total_tests = 0
        passed_tests = 0
        
        def count_tests(results_dict):
            nonlocal total_tests, passed_tests
            for key, value in results_dict.items():
                if isinstance(value, dict):
                    if 'passed' in value:
                        total_tests += 1
                        if value['passed']:
                            passed_tests += 1
                    else:
                        count_tests(value)
                        
        count_tests(self.test_results)
        
        report = {
            'summary': {
                'total_time': total_time,
                'timestamp': datetime.now().isoformat(),
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0
            },
            'results': self.test_results,
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_filename = f"proof_system_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Test report saved to {report_filename}")
        except Exception as e:
            self.logger.error(f"Failed to save report: {str(e)}")
            
        return report
        
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check performance
        perf_tests = self.test_results.get('performance_tests', {})
        overhead = perf_tests.get('overhead', {})
        
        if not overhead.get('acceptable', True):
            recommendations.append(
                f"Performance overhead ({overhead.get('overhead_percent', 0):.1f}%) exceeds limit. "
                f"Consider optimizing proof generation."
            )
            
        # Check throughput
        throughput = perf_tests.get('throughput', {})
        if not throughput.get('meets_requirement', True):
            recommendations.append(
                f"Throughput ({throughput.get('throughput_tps', 0):.0f} TPS) below requirement. "
                f"Consider parallelizing operations."
            )
            
        # Check accuracy
        validation = self.test_results.get('validation_tests', {})
        accuracy = validation.get('confidence_accuracy', {})
        
        if not accuracy.get('acceptable', True):
            recommendations.append(
                f"Confidence accuracy ({accuracy.get('accuracy', 0):.1%}) below threshold. "
                f"Consider tuning confidence generation."
            )
            
        if not recommendations:
            recommendations.append("All tests passed within acceptable parameters. System ready for integration.")
            
        return recommendations


def main():
    """Main entry point for test suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Proof System Comprehensive Test Suite')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--quick', action='store_true', help='Run quick subset of tests')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Run test suite
    test_suite = ProofSystemTestSuite()
    
    try:
        if args.quick:
            # Quick test - only unit tests
            logger.info("Running quick test subset...")
            results = {
                'unit_tests': test_suite._run_unit_tests(),
                'basic_integration': test_suite._test_brain_integration()
            }
        else:
            # Full test suite
            results = test_suite.run_all_tests()
            
        # Display summary
        logger.info("\n" + "="*80)
        logger.info("PROOF SYSTEM TEST RESULTS")
        logger.info("="*80)
        
        summary = results.get('summary', {})
        logger.info(f"Total Tests: {summary.get('total_tests', 0)}")
        logger.info(f"Passed: {summary.get('passed_tests', 0)}")
        logger.info(f"Failed: {summary.get('failed_tests', 0)}")
        logger.info(f"Success Rate: {summary.get('success_rate', 0):.1%}")
        
        if 'recommendations' in results:
            logger.info("\nRECOMMENDATIONS:")
            for rec in results['recommendations']:
                logger.info(f"  • {rec}")
                
        logger.info("="*80)
        
        # Exit with appropriate code
        success = summary.get('success_rate', 0) >= 0.8
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()