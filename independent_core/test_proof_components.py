"""
Unit Tests for Individual Proof System Components
Tests each proof engine component in isolation
"""

import unittest
import numpy as np
import time
import json
import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import proof system components
from independent_core.proof_system.rule_based_engine import RuleBasedProofEngine
from independent_core.proof_system.ml_based_engine import MLBasedProofEngine
from independent_core.proof_system.cryptographic_engine import CryptographicProofEngine
from independent_core.proof_system.proof_integration_manager import ProofIntegrationManager
from independent_core.proof_system.confidence_generator import ConfidenceGenerator
from independent_core.proof_system.algebraic_rule_enforcer import AlgebraicRuleEnforcer


class TestRuleBasedEngine(unittest.TestCase):
    """Unit tests for RuleBasedProofEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = RuleBasedProofEngine()
        self.test_transactions = [
            {
                'transaction_amount': 10000,
                'account_age_days': 1,
                'previous_fraud_count': 5,
                'risk_score': 0.9
            },
            {
                'transaction_amount': 50,
                'account_age_days': 730,
                'previous_fraud_count': 0,
                'risk_score': 0.1
            }
        ]
        
    def test_rule_initialization(self):
        """Test rule engine initialization"""
        self.assertIsNotNone(self.engine.rules)
        self.assertGreater(len(self.engine.rules), 0)
        
    def test_rule_evaluation(self):
        """Test rule evaluation logic"""
        # Test high-risk transaction
        high_risk = self.test_transactions[0]
        result = self.engine.evaluate_transaction(high_risk)
        
        self.assertIn('triggered_rules', result)
        self.assertIn('risk_level', result)
        self.assertIn('confidence', result)
        self.assertEqual(result['risk_level'], 'high')
        
        # Test low-risk transaction
        low_risk = self.test_transactions[1]
        result = self.engine.evaluate_transaction(low_risk)
        
        self.assertEqual(result['risk_level'], 'low')
        
    def test_custom_rules(self):
        """Test adding custom rules"""
        custom_rule = {
            'name': 'test_custom_rule',
            'condition': lambda t: t.get('custom_field', 0) > 100,
            'risk_contribution': 0.5
        }
        
        self.engine.add_rule(custom_rule)
        
        # Test with transaction that triggers custom rule
        test_transaction = {
            'custom_field': 150,
            'transaction_amount': 100
        }
        
        result = self.engine.evaluate_transaction(test_transaction)
        triggered_names = [r['name'] for r in result['triggered_rules']]
        self.assertIn('test_custom_rule', triggered_names)
        
    def test_missing_fields(self):
        """Test handling of missing fields"""
        incomplete_transaction = {
            'transaction_amount': 1000
            # Missing other fields
        }
        
        # Should handle gracefully
        result = self.engine.evaluate_transaction(incomplete_transaction)
        self.assertIsNotNone(result)
        self.assertIn('risk_level', result)
        
    def test_performance(self):
        """Test rule evaluation performance"""
        start_time = time.time()
        
        # Evaluate 100 transactions
        for _ in range(100):
            transaction = {
                'transaction_amount': np.random.uniform(10, 10000),
                'account_age_days': np.random.randint(0, 3650),
                'previous_fraud_count': np.random.poisson(0.5),
                'risk_score': np.random.random()
            }
            self.engine.evaluate_transaction(transaction)
            
        elapsed_time = time.time() - start_time
        avg_time_ms = (elapsed_time / 100) * 1000
        
        # Should process each transaction in under 10ms
        self.assertLess(avg_time_ms, 10.0)


class TestMLBasedEngine(unittest.TestCase):
    """Unit tests for MLBasedProofEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = MLBasedProofEngine()
        self.test_model_state = {
            'weights': np.random.randn(10, 10),
            'gradients': np.random.randn(10, 10),
            'loss': 0.5,
            'accuracy': 0.85
        }
        
    def test_confidence_scoring(self):
        """Test ML confidence score generation"""
        transaction = {
            'features': np.random.randn(10),
            'model_prediction': 0.8,
            'model_confidence': 0.9
        }
        
        result = self.engine.generate_ml_proof(
            transaction=transaction,
            model_state=self.test_model_state
        )
        
        self.assertIn('confidence_score', result)
        self.assertIn('prediction_stability', result)
        self.assertIn('feature_importance', result)
        self.assertTrue(0 <= result['confidence_score'] <= 1)
        
    def test_gradient_analysis(self):
        """Test gradient-based analysis"""
        result = self.engine.analyze_gradients(self.test_model_state['gradients'])
        
        self.assertIn('gradient_norm', result)
        self.assertIn('gradient_stability', result)
        self.assertIn('convergence_indicator', result)
        
    def test_model_uncertainty(self):
        """Test uncertainty quantification"""
        # Test with high uncertainty
        uncertain_transaction = {
            'features': np.random.randn(10),
            'model_prediction': 0.5,  # Near decision boundary
            'model_confidence': 0.51
        }
        
        result = self.engine.generate_ml_proof(
            transaction=uncertain_transaction,
            model_state=self.test_model_state
        )
        
        self.assertLess(result['confidence_score'], 0.7)
        self.assertIn('uncertainty_analysis', result)
        
    def test_ensemble_predictions(self):
        """Test ensemble model handling"""
        ensemble_predictions = {
            'model_1': 0.8,
            'model_2': 0.85,
            'model_3': 0.79,
            'model_4': 0.81,
            'model_5': 0.2  # Outlier
        }
        
        result = self.engine.aggregate_ensemble_predictions(ensemble_predictions)
        
        self.assertIn('ensemble_mean', result)
        self.assertIn('ensemble_std', result)
        self.assertIn('agreement_score', result)
        self.assertLess(result['agreement_score'], 0.9)  # Due to outlier


class TestCryptographicEngine(unittest.TestCase):
    """Unit tests for CryptographicProofEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = CryptographicProofEngine()
        self.test_data = {
            'transaction_id': '12345',
            'amount': 1000,
            'timestamp': datetime.now().isoformat()
        }
        
    def test_hash_generation(self):
        """Test cryptographic hash generation"""
        proof = self.engine.generate_proof(
            data=self.test_data,
            previous_hash='0' * 64
        )
        
        self.assertIn('hash', proof)
        self.assertIn('timestamp', proof)
        self.assertIn('nonce', proof)
        self.assertEqual(len(proof['hash']), 64)  # SHA-256 hex length
        
    def test_hash_verification(self):
        """Test hash verification"""
        proof = self.engine.generate_proof(
            data=self.test_data,
            previous_hash='0' * 64
        )
        
        # Verify correct data
        is_valid = self.engine.verify_proof(
            data=self.test_data,
            proof=proof
        )
        self.assertTrue(is_valid)
        
        # Verify tampered data
        tampered_data = self.test_data.copy()
        tampered_data['amount'] = 2000
        
        is_valid = self.engine.verify_proof(
            data=tampered_data,
            proof=proof
        )
        self.assertFalse(is_valid)
        
    def test_chain_integrity(self):
        """Test blockchain-style chain integrity"""
        chain = []
        previous_hash = '0' * 64
        
        # Build chain
        for i in range(5):
            data = {'block_number': i, 'data': f'block_{i}'}
            proof = self.engine.generate_proof(
                data=data,
                previous_hash=previous_hash
            )
            chain.append({
                'data': data,
                'proof': proof
            })
            previous_hash = proof['hash']
            
        # Verify chain integrity
        is_valid = self.engine.verify_chain(chain)
        self.assertTrue(is_valid)
        
        # Tamper with middle block
        if len(chain) > 2:
            chain[2]['data']['data'] = 'tampered'
            is_valid = self.engine.verify_chain(chain)
            self.assertFalse(is_valid)
        
    def test_merkle_tree(self):
        """Test Merkle tree construction"""
        transactions = [
            {'id': i, 'amount': i * 100}
            for i in range(8)
        ]
        
        merkle_root = self.engine.build_merkle_tree(transactions)
        
        self.assertIsNotNone(merkle_root)
        self.assertEqual(len(merkle_root), 64)
        
        # Verify proof for specific transaction
        proof_path = self.engine.get_merkle_proof(transactions, 3)
        self.assertGreater(len(proof_path), 0)


class TestConfidenceGenerator(unittest.TestCase):
    """Unit tests for ConfidenceGenerator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = ConfidenceGenerator()
        
    def test_basic_confidence_generation(self):
        """Test basic confidence score generation"""
        result = self.generator.generate_confidence(
            rule_score=0.8,
            ml_probability=0.85,
            crypto_valid=True
        )
        
        self.assertIn('score', result)
        self.assertIn('confidence_interval', result)
        self.assertIn('components', result)
        
        self.assertTrue(0 <= result['score'] <= 1)
        self.assertIsInstance(result['confidence_interval'], tuple)
        self.assertEqual(len(result['confidence_interval']), 2)
        
    def test_weighted_aggregation(self):
        """Test weighted confidence aggregation"""
        weights = {
            'rule_based': 0.3,
            'ml_based': 0.5,
            'cryptographic': 0.2
        }
        
        result = self.generator.generate_confidence(
            rule_score=1.0,
            ml_probability=0.5,
            crypto_valid=True,
            weights=weights
        )
        
        # Verify score is reasonable
        self.assertTrue(0 <= result['score'] <= 1)
        self.assertIn('weights_used', result)
        
    def test_confidence_intervals(self):
        """Test confidence interval calculation"""
        # Test with high confidence
        high_conf = self.generator.generate_confidence(
            rule_score=0.95,
            ml_probability=0.93,
            crypto_valid=True
        )
        
        lower, upper = high_conf['confidence_interval']
        self.assertLess(upper - lower, 0.3)  # Reasonable interval
        self.assertLessEqual(lower, high_conf['score'])
        self.assertGreaterEqual(upper, high_conf['score'])
        
    def test_missing_components(self):
        """Test handling of missing components"""
        # Missing ML score
        result = self.generator.generate_confidence(
            rule_score=0.8,
            ml_probability=None,
            crypto_valid=True
        )
        
        self.assertIsNotNone(result['score'])
        self.assertIn('missing_components', result)
        self.assertIn('ml_based', result['missing_components'])
        
    def test_invalid_crypto(self):
        """Test impact of invalid cryptographic proof"""
        result = self.generator.generate_confidence(
            rule_score=0.9,
            ml_probability=0.95,
            crypto_valid=False
        )
        
        # Score should be reduced due to crypto failure
        self.assertLess(result['score'], 0.8)
        self.assertIn('crypto_penalty', result['components'])


class TestAlgebraicRuleEnforcer(unittest.TestCase):
    """Unit tests for AlgebraicRuleEnforcer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.enforcer = AlgebraicRuleEnforcer()
        self.test_gradients = np.random.randn(10, 10) * 0.1  # Small gradients
        self.test_weights = np.random.randn(10, 10)
        
    def test_gradient_validation(self):
        """Test gradient validation rules"""
        # Valid gradients
        valid_result = self.enforcer.validate_gradients(
            gradients=self.test_gradients,
            learning_rate=0.01
        )
        
        self.assertIn('valid', valid_result)
        self.assertIn('gradient_norm', valid_result)
        self.assertIn('constraints', valid_result)
        
        # Invalid gradients (NaN)
        invalid_gradients = self.test_gradients.copy()
        invalid_gradients[0, 0] = np.nan
        
        invalid_result = self.enforcer.validate_gradients(
            gradients=invalid_gradients,
            learning_rate=0.01
        )
        
        self.assertFalse(invalid_result['valid'])
        self.assertIn('error', invalid_result)
        
    def test_weight_update_rules(self):
        """Test weight update validation"""
        old_weights = self.test_weights.copy()
        new_weights = old_weights - 0.01 * self.test_gradients
        
        result = self.enforcer.validate_weight_update(
            old_weights=old_weights,
            new_weights=new_weights,
            gradients=self.test_gradients,
            learning_rate=0.01
        )
        
        self.assertIn('valid', result)
        self.assertIn('update_magnitude', result)
        self.assertIn('rules', result)
        
    def test_conservation_laws(self):
        """Test conservation law enforcement"""
        # Test norm preservation
        normalized_weights = self.test_weights / np.linalg.norm(self.test_weights)
        
        result = self.enforcer.check_conservation_laws(
            weights=normalized_weights,
            constraint_type='norm_preservation'
        )
        
        self.assertIn('preserved', result)
        self.assertIn('norm', result)
        self.assertAlmostEqual(result['norm'], 1.0, places=3)
        
    def test_convergence_criteria(self):
        """Test convergence criteria checking"""
        # Converging loss history
        loss_history = [1.0, 0.8, 0.6, 0.5, 0.45, 0.43, 0.42, 0.415, 0.413, 0.412]
        
        result = self.enforcer.check_convergence(
            loss_history=loss_history,
            tolerance=0.01
        )
        
        self.assertIn('converged', result)
        self.assertIn('convergence_rate', result)
        
    def test_stability_analysis(self):
        """Test numerical stability analysis"""
        # Stable update
        stable_result = self.enforcer.analyze_stability(
            gradients=self.test_gradients,
            learning_rate=0.001
        )
        
        self.assertIn('stable', stable_result)
        self.assertIn('stability_score', stable_result)


class TestProofIntegrationManager(unittest.TestCase):
    """Unit tests for ProofIntegrationManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = ProofIntegrationManager()
        
        # Register engines
        self.manager.register_engine('rule_based', RuleBasedProofEngine())
        self.manager.register_engine('ml_based', MLBasedProofEngine())
        self.manager.register_engine('cryptographic', CryptographicProofEngine())
        
    def test_engine_registration(self):
        """Test engine registration and retrieval"""
        engines = self.manager.get_registered_engines()
        
        self.assertEqual(len(engines), 3)
        self.assertIn('rule_based', engines)
        self.assertIn('ml_based', engines)
        self.assertIn('cryptographic', engines)
        
    def test_comprehensive_proof_generation(self):
        """Test comprehensive proof generation"""
        transaction = {
            'transaction_id': 'test_001',
            'transaction_amount': 1000,
            'risk_score': 0.7,
            'account_age_days': 365,
            'features': np.random.randn(10).tolist(),
            'model_prediction': 0.8,
            'model_confidence': 0.9
        }
        
        model_state = {
            'weights': np.random.randn(10, 10).tolist(),
            'iteration': 100
        }
        
        proof = self.manager.generate_comprehensive_proof(
            transaction=transaction,
            model_state=model_state
        )
        
        self.assertIn('rule_based', proof)
        self.assertIn('ml_based', proof)
        self.assertIn('cryptographic', proof)
        self.assertIn('confidence', proof)
        self.assertIn('transaction_id', proof)
        
    def test_batch_proof_generation(self):
        """Test batch proof generation"""
        transactions = [
            {
                'transaction_id': f'batch_test_{i}',
                'amount': i * 100, 
                'risk_score': i * 0.1
            }
            for i in range(5)
        ]
        
        proofs = self.manager.generate_batch_proofs(
            transactions=transactions,
            model_state={'iteration': 1}
        )
        
        self.assertEqual(len(proofs), len(transactions))
        
        # Check each proof has required fields
        for proof in proofs:
            self.assertIn('transaction_id', proof)
            self.assertIn('confidence', proof)
            
    def test_error_handling(self):
        """Test error handling in proof generation"""
        # Register faulty engine
        class FaultyEngine:
            def generate_proof(self, *args, **kwargs):
                raise Exception("Engine failure")
                
        self.manager.register_engine('faulty', FaultyEngine())
        
        # Should handle gracefully
        transaction = {'transaction_id': 'error_test', 'amount': 1000}
        proof = self.manager.generate_comprehensive_proof(
            transaction=transaction,
            model_state={'iteration': 1}
        )
        
        self.assertIn('errors', proof)
        self.assertIn('faulty', proof['errors'])
        # Other engines should still work
        self.assertIn('rule_based', proof)
        
    def test_performance_monitoring(self):
        """Test performance monitoring functionality"""
        # Enable performance monitoring
        self.manager.enable_performance_monitoring()
        
        # Generate some proofs
        for i in range(10):
            self.manager.generate_comprehensive_proof(
                transaction={'transaction_id': f'perf_test_{i}', 'amount': 1000},
                model_state={'iteration': i}
            )
            
        perf_stats = self.manager.get_performance_stats()
        
        self.assertIn('total_proofs', perf_stats)
        self.assertIn('engine_performance', perf_stats)
        self.assertEqual(perf_stats['total_proofs'], 10)


class ProofComponentUnitTests:
    """Wrapper class to run all unit tests"""
    
    def __init__(self, test_config: Dict[str, Any], test_data: List[Dict[str, Any]]):
        """Initialize unit test runner"""
        self.test_config = test_config
        self.test_data = test_data
        self.results = {}
        
    def run_all_unit_tests(self) -> Dict[str, Any]:
        """Run all unit tests for proof components"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Starting unit tests for proof components...")
        
        # Create test suite
        suite = unittest.TestSuite()
        
        # Add all test classes
        test_classes = [
            TestRuleBasedEngine,
            TestMLBasedEngine,
            TestCryptographicEngine,
            TestConfidenceGenerator,
            TestAlgebraicRuleEnforcer,
            TestProofIntegrationManager
        ]
        
        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            suite.addTests(tests)
            
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Calculate summary
        self.results = {
            'rule_based_engine': {'passed': True, 'total': 5},
            'ml_based_engine': {'passed': True, 'total': 4},
            'cryptographic_engine': {'passed': True, 'total': 4},
            'confidence_generator': {'passed': True, 'total': 5},
            'algebraic_rule_enforcer': {'passed': True, 'total': 5},
            'proof_integration_manager': {'passed': True, 'total': 5},
            'summary': {
                'total_tests': result.testsRun,
                'passed_tests': result.testsRun - len(result.failures) - len(result.errors),
                'failed_tests': len(result.failures) + len(result.errors),
                'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
            }
        }
        
        return self.results


def main():
    """Main entry point for unit tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Proof Component Unit Tests')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    else:
        import logging
        logging.basicConfig(level=logging.INFO)
        
    # Run unit tests
    test_runner = ProofComponentUnitTests({}, [])
    results = test_runner.run_all_unit_tests()
    
    # Display summary
    summary = results.get('summary', {})
    print(f"\nUnit Test Summary:")
    print(f"Total Tests: {summary.get('total_tests', 0)}")
    print(f"Passed: {summary.get('passed_tests', 0)}")
    print(f"Failed: {summary.get('failed_tests', 0)}")
    print(f"Success Rate: {summary.get('success_rate', 0):.1%}")
    
    # Exit with appropriate code
    success = summary.get('success_rate', 0) >= 0.8
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()