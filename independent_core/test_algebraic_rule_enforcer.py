"""
Test file for AlgebraicRuleEnforcer
Tests all methods and validates mathematical operations
"""

import unittest
import numpy as np
import time
from datetime import datetime
from typing import Dict, Any, List

from proof_system.algebraic_rule_enforcer import AlgebraicRuleEnforcer


class TestAlgebraicRuleEnforcer(unittest.TestCase):
    """Test suite for AlgebraicRuleEnforcer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.enforcer = AlgebraicRuleEnforcer()
        
    def test_initialization(self):
        """Test proper initialization"""
        self.assertIsNotNone(self.enforcer)
        self.assertEqual(self.enforcer.tolerance, 1e-6)
        self.assertIsInstance(self.enforcer.convergence_history, list)
        
    def test_validate_gradients_normal(self):
        """Test gradient validation with normal gradients"""
        gradients = np.array([0.1, -0.2, 0.15, -0.05])
        learning_rate = 0.01
        
        result = self.enforcer.validate_gradients(gradients, learning_rate)
        
        self.assertTrue(result['valid'])
        self.assertIn('gradient_norm', result)
        self.assertIn('gradient_mean', result)
        self.assertIn('gradient_std', result)
        self.assertIn('constraints', result)
        
    def test_validate_gradients_vanishing(self):
        """Test detection of vanishing gradients"""
        gradients = np.array([1e-11, 1e-12, 1e-11, 1e-12])
        learning_rate = 0.01
        
        result = self.enforcer.validate_gradients(gradients, learning_rate)
        
        self.assertFalse(result['valid'])
        constraints = result['constraints']
        self.assertTrue(any(
            c['constraint'] == 'gradient_magnitude' and not c['satisfied'] 
            for c in constraints
        ))
        
    def test_validate_gradients_exploding(self):
        """Test detection of exploding gradients"""
        gradients = np.array([100, 200, -150, 300])
        learning_rate = 0.01
        
        result = self.enforcer.validate_gradients(gradients, learning_rate)
        
        self.assertFalse(result['valid'])
        constraints = result['constraints']
        self.assertTrue(any(
            c['constraint'] == 'gradient_magnitude' and not c['satisfied']
            for c in constraints
        ))
        
    def test_validate_gradients_nan(self):
        """Test handling of NaN gradients"""
        gradients = np.array([0.1, np.nan, 0.15, -0.05])
        learning_rate = 0.01
        
        result = self.enforcer.validate_gradients(gradients, learning_rate)
        
        self.assertFalse(result['valid'])
        self.assertIn('error', result)
        self.assertIn('NaN', result['error'])
        
    def test_validate_gradients_inf(self):
        """Test handling of infinite gradients"""
        gradients = np.array([0.1, np.inf, 0.15, -0.05])
        learning_rate = 0.01
        
        result = self.enforcer.validate_gradients(gradients, learning_rate)
        
        self.assertFalse(result['valid'])
        self.assertIn('error', result)
        self.assertIn('infinite', result['error'])
        
    def test_validate_weight_update_correct(self):
        """Test validation of correct weight update"""
        old_weights = np.array([1.0, 2.0, 3.0, 4.0])
        gradients = np.array([0.1, 0.2, 0.15, 0.05])
        learning_rate = 0.01
        new_weights = old_weights - learning_rate * gradients
        
        result = self.enforcer.validate_weight_update(
            old_weights, new_weights, gradients, learning_rate
        )
        
        self.assertTrue(result['valid'])
        self.assertIn('update_magnitude', result)
        self.assertIn('relative_update', result)
        self.assertIn('rules', result)
        
    def test_validate_weight_update_wrong_direction(self):
        """Test detection of incorrect update direction"""
        old_weights = np.array([1.0, 2.0, 3.0, 4.0])
        gradients = np.array([0.1, 0.2, 0.15, 0.05])
        learning_rate = 0.01
        # Incorrectly add gradients instead of subtracting
        new_weights = old_weights + learning_rate * gradients
        
        result = self.enforcer.validate_weight_update(
            old_weights, new_weights, gradients, learning_rate
        )
        
        self.assertFalse(result['valid'])
        rules = result['rules']
        self.assertTrue(any(
            r['rule'] == 'gradient_descent_direction' and not r['satisfied']
            for r in rules
        ))
        
    def test_validate_weight_update_dimension_mismatch(self):
        """Test handling of dimension mismatch"""
        old_weights = np.array([1.0, 2.0, 3.0])
        gradients = np.array([0.1, 0.2, 0.15, 0.05])
        learning_rate = 0.01
        new_weights = np.array([0.99, 1.98, 2.985])
        
        result = self.enforcer.validate_weight_update(
            old_weights, new_weights, gradients, learning_rate
        )
        
        self.assertFalse(result['valid'])
        self.assertIn('error', result)
        self.assertIn('Dimension mismatch', result['error'])
        
    def test_check_conservation_laws_norm(self):
        """Test norm preservation check"""
        # Unit norm weights
        weights = np.array([0.5, 0.5, 0.5, 0.5])
        weights = weights / np.linalg.norm(weights)
        
        result = self.enforcer.check_conservation_laws(weights, 'norm_preservation')
        
        self.assertTrue(result['preserved'])
        self.assertIn('norm', result)
        self.assertAlmostEqual(result['norm'], 1.0, places=5)
        
    def test_check_conservation_laws_sum(self):
        """Test sum preservation check"""
        weights = np.array([0.25, 0.25, -0.25, -0.25])
        
        result = self.enforcer.check_conservation_laws(weights, 'sum_preservation')
        
        self.assertTrue(result['preserved'])
        self.assertIn('sum', result)
        self.assertAlmostEqual(result['sum'], 0.0, places=5)
        
    def test_check_conservation_laws_orthogonality(self):
        """Test orthogonality preservation check"""
        # Create an orthogonal matrix
        weights = np.array([[1, 0], [0, 1]])
        
        result = self.enforcer.check_conservation_laws(weights, 'orthogonality')
        
        self.assertTrue(result['preserved'])
        self.assertIn('orthogonality_error', result)
        
    def test_check_conservation_laws_symmetry(self):
        """Test symmetry preservation check"""
        # Create a symmetric matrix
        weights = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        
        result = self.enforcer.check_conservation_laws(weights, 'symmetry')
        
        self.assertTrue(result['preserved'])
        self.assertIn('symmetry_error', result)
        
    def test_check_convergence_converged(self):
        """Test convergence detection"""
        # Loss history showing convergence
        loss_history = [1.0, 0.5, 0.3, 0.2, 0.15, 0.14, 0.135, 0.133, 0.132, 0.131]
        
        result = self.enforcer.check_convergence(loss_history, tolerance=0.01, patience=5)
        
        self.assertTrue(result['converged'])
        self.assertIn('convergence_rate', result)
        self.assertIn('relative_change', result)
        
    def test_check_convergence_not_converged(self):
        """Test non-convergence detection"""
        # Loss history showing no convergence
        loss_history = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        
        result = self.enforcer.check_convergence(loss_history, tolerance=0.01, patience=5)
        
        self.assertFalse(result['converged'])
        self.assertIn('relative_change', result)
        
    def test_check_convergence_insufficient_history(self):
        """Test handling of insufficient history"""
        loss_history = [1.0, 0.5]
        
        result = self.enforcer.check_convergence(loss_history, tolerance=0.01, patience=5)
        
        self.assertFalse(result['converged'])
        self.assertIn('reason', result)
        self.assertIn('Insufficient history', result['reason'])
        
    def test_analyze_stability_stable(self):
        """Test stability analysis with stable gradients"""
        gradients = np.array([0.1, -0.2, 0.15, -0.05])
        learning_rate = 0.01
        
        result = self.enforcer.analyze_stability(gradients, learning_rate)
        
        self.assertTrue(result['stable'])
        self.assertIn('stability_score', result)
        self.assertIn('stability_checks', result)
        self.assertGreater(result['stability_score'], 0.8)
        
    def test_analyze_stability_unstable(self):
        """Test stability analysis with unstable gradients"""
        gradients = np.array([100, 200, -150, 300])
        learning_rate = 0.1
        
        result = self.enforcer.analyze_stability(gradients, learning_rate)
        
        self.assertFalse(result['stable'])
        self.assertIn('instability_reason', result)
        self.assertIn('all_instability_reasons', result)
        
    def test_check_constraints_satisfied(self):
        """Test constraint checking with satisfied constraints"""
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        constraints = [
            {'type': 'max_norm', 'value': 1.0},
            {'type': 'non_negative', 'value': None},
            {'type': 'bounded', 'value': (0, 1)}
        ]
        
        result = self.enforcer.check_constraints(weights, constraints)
        
        self.assertTrue(result['all_satisfied'])
        self.assertIn('satisfied_constraints', result)
        self.assertIn('violated_constraints', result)
        self.assertEqual(len(result['violated_constraints']), 0)
        
    def test_check_constraints_violated(self):
        """Test constraint checking with violated constraints"""
        weights = np.array([-0.1, 0.2, 0.3, 1.5])
        constraints = [
            {'type': 'max_norm', 'value': 1.0},
            {'type': 'non_negative', 'value': None},
            {'type': 'bounded', 'value': (0, 1)}
        ]
        
        result = self.enforcer.check_constraints(weights, constraints)
        
        self.assertFalse(result['all_satisfied'])
        self.assertIn('violated_constraints', result)
        self.assertGreater(len(result['violated_constraints']), 0)
        
    def test_validate_sheaf_compression(self):
        """Test sheaf compression validation"""
        sheaf_data = {
            'sections': {
                'cell_1': [0.1, 0.2, 0.3],
                'cell_2': [0.4, 0.5, 0.6]
            },
            'restriction_maps': {
                'map_1': np.array([[1, 0], [0, 1]])
            },
            'topology': {
                'open_sets': ['set_1', 'set_2']
            }
        }
        compressed_data = {'compressed': True, 'size': 100}
        
        result = self.enforcer.validate_sheaf_compression(sheaf_data, compressed_data)
        
        self.assertTrue(result['valid'])
        self.assertIn('sheaf_properties_preserved', result)
        self.assertIn('violations', result)
        
    def test_validate_sheaf_cohomology(self):
        """Test sheaf cohomology validation"""
        sheaf_data = {
            'sections': {'cell_1': [0.1, 0.2]},
            'restriction_maps': {'map_1': np.array([[1, 0], [0, 1]])},
            'topology': {'open_sets': ['set_1', 'set_2']}
        }
        computed_cohomology = {
            'cohomology_groups': [[0.1, 0.2], [0.3]],
            'betti_numbers': [2, 1]
        }
        
        result = self.enforcer.validate_sheaf_cohomology(sheaf_data, computed_cohomology, degree=0)
        
        self.assertTrue(result['valid'])
        self.assertIn('cohomology_properties', result)
        self.assertEqual(result['degree'], 0)
        
    def test_validate_sheaf_morphism(self):
        """Test sheaf morphism validation"""
        source_sheaf = {
            'sections': {'cell_1': [0.1, 0.2], 'cell_2': [0.3, 0.4]},
            'restriction_maps': {'map_1': np.array([[1, 0], [0, 1]])},
            'topology': {'open_sets': ['set_1', 'set_2']}
        }
        target_sheaf = {
            'sections': {'cell_1': [0.2, 0.3], 'cell_2': [0.4, 0.5]},
            'restriction_maps': {'map_1': np.array([[1, 0], [0, 1]])},
            'topology': {'open_sets': ['set_1', 'set_2']}
        }
        morphism = {'type': 'identity'}
        
        result = self.enforcer.validate_sheaf_morphism(source_sheaf, target_sheaf, morphism)
        
        self.assertTrue(result['valid'])
        self.assertIn('morphism_properties', result)
        
    def test_integrate_with_proof_system(self):
        """Test integration with proof system"""
        # Mock proof system
        class MockProofSystem:
            pass
        
        proof_system = MockProofSystem()
        
        result = self.enforcer.integrate_with_proof_system(proof_system)
        
        self.assertIsNotNone(result)
        self.assertIn('algebraic_enforcer', result)
        self.assertIn('proof_system', result)
        self.assertIn('enabled_rules', result)
        self.assertIn('thresholds', result)
        self.assertIn('sheaf_validation_features', result)
        
    def test_integrate_with_proof_system_none(self):
        """Test integration with None proof system"""
        result = self.enforcer.integrate_with_proof_system(None)
        
        self.assertIsNone(result)
        
    def test_performance_gradient_validation(self):
        """Test performance of gradient validation"""
        gradients = np.random.randn(1000)
        learning_rate = 0.01
        
        start_time = time.time()
        result = self.enforcer.validate_gradients(gradients, learning_rate)
        end_time = time.time()
        
        self.assertIsNotNone(result)
        self.assertIn('validation_time_ms', result)
        # Should complete in reasonable time (< 100ms)
        self.assertLess((end_time - start_time) * 1000, 100)
        
    def test_edge_case_empty_arrays(self):
        """Test handling of empty arrays"""
        gradients = np.array([])
        learning_rate = 0.01
        
        result = self.enforcer.validate_gradients(gradients, learning_rate)
        
        self.assertFalse(result['valid'])
        
    def test_edge_case_list_input(self):
        """Test handling of list inputs instead of numpy arrays"""
        gradients = [0.1, -0.2, 0.15, -0.05]
        learning_rate = 0.01
        
        result = self.enforcer.validate_gradients(gradients, learning_rate)
        
        self.assertTrue(result['valid'])
        self.assertIn('gradient_norm', result)


class TestAlgebraicRuleEnforcerIntegration(unittest.TestCase):
    """Integration tests for AlgebraicRuleEnforcer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.enforcer = AlgebraicRuleEnforcer()
        
    def test_full_training_cycle_validation(self):
        """Test validation through a simulated training cycle"""
        # Simulate a training cycle
        weights = np.random.randn(10)
        learning_rate = 0.01
        loss_history = []
        
        for iteration in range(20):
            # Simulate gradients (decreasing magnitude over time)
            gradients = np.random.randn(10) * (0.5 / (iteration + 1))
            
            # Validate gradients
            grad_result = self.enforcer.validate_gradients(gradients, learning_rate)
            self.assertIsNotNone(grad_result)
            
            # Update weights
            new_weights = weights - learning_rate * gradients
            
            # Validate update
            update_result = self.enforcer.validate_weight_update(
                weights, new_weights, gradients, learning_rate
            )
            self.assertIsNotNone(update_result)
            
            # Update weights
            weights = new_weights
            
            # Simulate loss (decreasing)
            loss = 1.0 / (iteration + 1)
            loss_history.append(loss)
            
            # Check convergence periodically
            if len(loss_history) >= 10:
                conv_result = self.enforcer.check_convergence(loss_history)
                self.assertIsNotNone(conv_result)
                
            # Analyze stability
            stability_result = self.enforcer.analyze_stability(gradients, learning_rate)
            self.assertIsNotNone(stability_result)
            
    def test_constraint_enforcement_workflow(self):
        """Test complete constraint enforcement workflow"""
        weights = np.random.randn(5)
        
        # Define multiple constraint types
        constraints = [
            {'type': 'max_norm', 'value': 2.0},
            {'type': 'bounded', 'value': (-1, 1)}
        ]
        
        # Check initial constraints
        result = self.enforcer.check_constraints(weights, constraints)
        self.assertIsNotNone(result)
        
        # If violated, project weights to satisfy constraints
        if not result['all_satisfied']:
            # Clip to bounds
            weights = np.clip(weights, -1, 1)
            # Normalize if needed
            if np.linalg.norm(weights) > 2.0:
                weights = weights * 2.0 / np.linalg.norm(weights)
                
            # Re-check constraints
            result = self.enforcer.check_constraints(weights, constraints)
            self.assertTrue(result['all_satisfied'])
            
    def test_sheaf_validation_workflow(self):
        """Test complete sheaf validation workflow"""
        # Create sheaf data
        sheaf_data = {
            'sections': {f'cell_{i}': np.random.randn(3).tolist() for i in range(5)},
            'restriction_maps': {
                f'map_{i}': np.random.randn(2, 2).tolist() for i in range(3)
            },
            'cohomology_groups': [np.random.randn(2).tolist() for _ in range(3)],
            'topology': {
                'open_sets': [f'set_{i}' for i in range(4)]
            }
        }
        
        # Simulate compression
        compressed_data = {'data': 'compressed', 'size': 50}
        
        # Validate compression
        compression_result = self.enforcer.validate_sheaf_compression(
            sheaf_data, compressed_data
        )
        self.assertIsNotNone(compression_result)
        
        # Validate cohomology
        computed_cohomology = {
            'cohomology_groups': sheaf_data['cohomology_groups'],
            'betti_numbers': [2, 1, 0]
        }
        
        cohomology_result = self.enforcer.validate_sheaf_cohomology(
            sheaf_data, computed_cohomology, degree=0
        )
        self.assertIsNotNone(cohomology_result)
        
        # Create and validate morphism
        target_sheaf = {
            'sections': {f'cell_{i}': np.random.randn(3).tolist() for i in range(5)},
            'restriction_maps': {
                f'map_{i}': np.random.randn(2, 2).tolist() for i in range(3)
            },
            'topology': {
                'open_sets': [f'set_{i}' for i in range(4)]
            }
        }
        
        morphism = {'type': 'linear', 'matrix': np.random.randn(3, 3).tolist()}
        
        morphism_result = self.enforcer.validate_sheaf_morphism(
            sheaf_data, target_sheaf, morphism
        )
        self.assertIsNotNone(morphism_result)


def run_tests():
    """Run all tests and return results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestAlgebraicRuleEnforcer))
    suite.addTests(loader.loadTestsFromTestCase(TestAlgebraicRuleEnforcerIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return summary
    return {
        'total_tests': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'success': result.wasSuccessful()
    }


if __name__ == '__main__':
    # Run tests
    results = run_tests()
    
    # Print summary
    print("\n" + "="*60)
    print("ALGEBRAIC RULE ENFORCER TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {results['total_tests']}")
    print(f"Failures: {results['failures']}")
    print(f"Errors: {results['errors']}")
    print(f"Success: {'✅ PASSED' if results['success'] else '❌ FAILED'}")
    print("="*60)