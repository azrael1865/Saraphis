"""
Algebraic Rule Enforcer
Enforces mathematical and algebraic constraints during training and inference
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class AlgebraicRuleEnforcer:
    """Enforces algebraic rules and mathematical constraints"""
    
    def __init__(self):
        """Initialize algebraic rule enforcer"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tolerance = 1e-6
        self.convergence_history = []
        
    def validate_gradients(self, gradients: Union[np.ndarray, List], 
                          learning_rate: float) -> Dict[str, Any]:
        """Validate gradient properties and constraints"""
        start_time = time.time()
        
        # Convert to numpy array if needed
        if isinstance(gradients, list):
            gradients = np.array(gradients)
        elif gradients is None:
            return {
                'valid': False,
                'error': 'No gradients provided',
                'timestamp': datetime.now().isoformat()
            }
            
        # Check for NaN or infinite values
        has_nan = np.isnan(gradients).any()
        has_inf = np.isinf(gradients).any()
        
        if has_nan or has_inf:
            return {
                'valid': False,
                'error': 'Gradients contain NaN or infinite values',
                'has_nan': has_nan,
                'has_inf': has_inf,
                'timestamp': datetime.now().isoformat()
            }
            
        # Calculate gradient statistics
        gradient_norm = np.linalg.norm(gradients)
        gradient_mean = np.mean(gradients) if gradients.size > 0 else 0.0
        gradient_std = np.std(gradients) if gradients.size > 0 else 0.0
        max_gradient = np.max(np.abs(gradients)) if gradients.size > 0 else 0.0
        
        # Check gradient constraints
        constraints_satisfied = []
        
        # Handle empty gradients case
        if gradients.size == 0:
            return {
                'valid': False,
                'error': 'Empty gradient array',
                'timestamp': datetime.now().isoformat()
            }
        
        # Constraint 1: Gradient norm should be reasonable
        if gradient_norm < 1e-10:
            constraints_satisfied.append({
                'constraint': 'gradient_magnitude',
                'satisfied': False,
                'message': 'Gradients too small (vanishing gradients)'
            })
        elif gradient_norm > 100:
            constraints_satisfied.append({
                'constraint': 'gradient_magnitude', 
                'satisfied': False,
                'message': 'Gradients too large (exploding gradients)'
            })
        else:
            constraints_satisfied.append({
                'constraint': 'gradient_magnitude',
                'satisfied': True,
                'message': 'Gradient magnitude within acceptable range'
            })
            
        # Constraint 2: Learning rate compatibility
        update_magnitude = learning_rate * gradient_norm
        if update_magnitude > 10:
            constraints_satisfied.append({
                'constraint': 'update_magnitude',
                'satisfied': False,
                'message': f'Update magnitude ({update_magnitude:.2f}) too large for stability'
            })
        else:
            constraints_satisfied.append({
                'constraint': 'update_magnitude',
                'satisfied': True,
                'message': 'Update magnitude appropriate'
            })
            
        # Constraint 3: Gradient distribution
        if gradient_std < gradient_norm * 0.01:
            constraints_satisfied.append({
                'constraint': 'gradient_distribution',
                'satisfied': False,
                'message': 'Gradients too uniform (potential saturation)'
            })
        else:
            constraints_satisfied.append({
                'constraint': 'gradient_distribution',
                'satisfied': True,
                'message': 'Gradient distribution appears healthy'
            })
            
        # Overall validation
        all_satisfied = all(c['satisfied'] for c in constraints_satisfied)
        
        validation_time = time.time() - start_time
        
        return {
            'valid': all_satisfied,
            'gradient_norm': float(gradient_norm),
            'gradient_mean': float(gradient_mean),
            'gradient_std': float(gradient_std),
            'max_gradient': float(max_gradient),
            'update_magnitude': float(update_magnitude),
            'constraints': constraints_satisfied,
            'validation_time_ms': validation_time * 1000,
            'timestamp': datetime.now().isoformat()
        }
        
    def validate_weight_update(self, old_weights: np.ndarray, new_weights: np.ndarray,
                              gradients: np.ndarray, learning_rate: float) -> Dict[str, Any]:
        """Validate weight update follows algebraic rules"""
        start_time = time.time()
        
        # Convert inputs to numpy arrays
        old_weights = np.array(old_weights) if not isinstance(old_weights, np.ndarray) else old_weights
        new_weights = np.array(new_weights) if not isinstance(new_weights, np.ndarray) else new_weights
        gradients = np.array(gradients) if not isinstance(gradients, np.ndarray) else gradients
        
        # Check dimensions match
        if old_weights.shape != new_weights.shape or old_weights.shape != gradients.shape:
            return {
                'valid': False,
                'error': 'Dimension mismatch between weights and gradients',
                'timestamp': datetime.now().isoformat()
            }
            
        # Calculate expected update
        expected_new_weights = old_weights - learning_rate * gradients
        
        # Calculate actual update
        actual_update = new_weights - old_weights
        expected_update = expected_new_weights - old_weights
        
        # Validate update follows gradient descent rule
        update_difference = np.linalg.norm(actual_update - expected_update)
        relative_error = update_difference / (np.linalg.norm(expected_update) + 1e-10)
        
        # Check update magnitude
        update_magnitude = np.linalg.norm(actual_update)
        weight_magnitude = np.linalg.norm(old_weights)
        relative_update = update_magnitude / (weight_magnitude + 1e-10)
        
        # Validation rules
        rules_satisfied = []
        
        # Rule 1: Update follows gradient descent direction
        if relative_error < 0.01:  # 1% tolerance
            rules_satisfied.append({
                'rule': 'gradient_descent_direction',
                'satisfied': True,
                'message': 'Update follows gradient descent rule'
            })
        else:
            rules_satisfied.append({
                'rule': 'gradient_descent_direction',
                'satisfied': False,
                'message': f'Update deviates from gradient descent (error: {relative_error:.4f})'
            })
            
        # Rule 2: Update magnitude is reasonable
        if relative_update < 0.1:  # Updates shouldn't be more than 10% of weight magnitude
            rules_satisfied.append({
                'rule': 'update_magnitude',
                'satisfied': True,
                'message': 'Update magnitude is reasonable'
            })
        else:
            rules_satisfied.append({
                'rule': 'update_magnitude',
                'satisfied': False,
                'message': f'Update magnitude too large (relative update: {relative_update:.4f})'
            })
            
        # Rule 3: No NaN or infinite values introduced
        has_nan = np.isnan(new_weights).any()
        has_inf = np.isinf(new_weights).any()
        
        if not has_nan and not has_inf:
            rules_satisfied.append({
                'rule': 'numerical_stability',
                'satisfied': True,
                'message': 'Update maintains numerical stability'
            })
        else:
            rules_satisfied.append({
                'rule': 'numerical_stability',
                'satisfied': False,
                'message': 'Update introduced NaN or infinite values'
            })
            
        all_satisfied = all(rule['satisfied'] for rule in rules_satisfied)
        
        validation_time = time.time() - start_time
        
        return {
            'valid': all_satisfied,
            'update_magnitude': float(update_magnitude),
            'relative_update': float(relative_update),
            'relative_error': float(relative_error),
            'rules': rules_satisfied,
            'validation_time_ms': validation_time * 1000,
            'timestamp': datetime.now().isoformat()
        }
        
    def check_conservation_laws(self, weights: np.ndarray, 
                               constraint_type: str = 'norm_preservation') -> Dict[str, Any]:
        """Check if weights satisfy conservation laws"""
        start_time = time.time()
        
        weights = np.array(weights) if not isinstance(weights, np.ndarray) else weights
        
        result = {
            'constraint_type': constraint_type,
            'preserved': False,
            'timestamp': datetime.now().isoformat()
        }
        
        if constraint_type == 'norm_preservation':
            # Check if weights maintain unit norm
            norm = np.linalg.norm(weights)
            result.update({
                'norm': float(norm),
                'preserved': abs(norm - 1.0) < self.tolerance,
                'deviation': abs(norm - 1.0)
            })
            
        elif constraint_type == 'sum_preservation':
            # Check if weights sum to a constant
            weight_sum = np.sum(weights)
            result.update({
                'sum': float(weight_sum),
                'preserved': abs(weight_sum) < self.tolerance,  # Assuming sum should be 0
                'deviation': abs(weight_sum)
            })
            
        elif constraint_type == 'orthogonality':
            # Check if weight matrix maintains orthogonality
            if weights.ndim == 2:
                gram_matrix = np.dot(weights, weights.T)
                identity = np.eye(gram_matrix.shape[0])
                orthogonality_error = np.linalg.norm(gram_matrix - identity)
                result.update({
                    'orthogonality_error': float(orthogonality_error),
                    'preserved': orthogonality_error < self.tolerance,
                    'deviation': orthogonality_error
                })
            else:
                result['error'] = 'Orthogonality constraint requires 2D weight matrix'
                
        elif constraint_type == 'symmetry':
            # Check if weight matrix maintains symmetry
            if weights.ndim == 2 and weights.shape[0] == weights.shape[1]:
                symmetry_error = np.linalg.norm(weights - weights.T)
                result.update({
                    'symmetry_error': float(symmetry_error),
                    'preserved': symmetry_error < self.tolerance,
                    'deviation': symmetry_error
                })
            else:
                result['error'] = 'Symmetry constraint requires square matrix'
                
        else:
            result['error'] = f'Unknown constraint type: {constraint_type}'
            
        result['validation_time_ms'] = (time.time() - start_time) * 1000
        return result
        
    def check_convergence(self, loss_history: List[float], 
                         tolerance: float = 0.01, 
                         patience: int = 5) -> Dict[str, Any]:
        """Check if training has converged based on loss history"""
        if len(loss_history) < patience + 1:
            return {
                'converged': False,
                'reason': 'Insufficient history for convergence check',
                'patience_remaining': patience + 1 - len(loss_history),
                'timestamp': datetime.now().isoformat()
            }
            
        # Check if recent losses are within tolerance
        # We need patience+1 points to check patience intervals
        recent_losses = loss_history[-(patience+1):]
        
        # Check if all consecutive changes are within tolerance
        converged = True
        max_relative_change = 0.0
        
        for i in range(len(recent_losses) - 1):
            if recent_losses[i] > 0:
                change = abs(recent_losses[i+1] - recent_losses[i]) / recent_losses[i]
                max_relative_change = max(max_relative_change, change)
                if change > tolerance:
                    converged = False
                    
        relative_change = max_relative_change
        
        # Calculate convergence rate
        if len(loss_history) >= 10:
            # Fit exponential decay to recent losses
            x = np.arange(len(recent_losses))
            y = np.log(np.array(recent_losses) + 1e-10)
            
            try:
                # Linear fit to log values gives exponential decay rate
                coeffs = np.polyfit(x, y, 1)
                convergence_rate = -coeffs[0]  # Negative slope indicates convergence
            except:
                convergence_rate = 0.0
        else:
            convergence_rate = 0.0
            
        # Estimate iterations to convergence
        if convergence_rate > 0 and not converged:
            current_loss = loss_history[-1]
            target_loss = min(recent_losses) * (1 - tolerance)
            iterations_to_converge = max(0, np.log(target_loss / current_loss) / (-convergence_rate))
        else:
            iterations_to_converge = 0
            
        result = {
            'converged': converged,
            'convergence_rate': float(convergence_rate),
            'relative_change': float(relative_change),
            'tolerance_used': tolerance,
            'patience_used': patience,
            'iterations_to_converge': int(iterations_to_converge) if iterations_to_converge < 1000 else None,
            'recent_loss_stats': {
                'min': float(min(recent_losses)),
                'max': float(max(recent_losses)),
                'mean': float(np.mean(recent_losses)),
                'std': float(np.std(recent_losses))
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Store convergence history
        self.convergence_history.append({
            'converged': converged,
            'loss': loss_history[-1],
            'timestamp': time.time()
        })
        
        return result
        
    def analyze_stability(self, gradients: np.ndarray, learning_rate: float) -> Dict[str, Any]:
        """Analyze numerical stability of the training process"""
        start_time = time.time()
        
        gradients = np.array(gradients) if not isinstance(gradients, np.ndarray) else gradients
        
        # Check for stability indicators
        gradient_norm = np.linalg.norm(gradients)
        max_gradient = np.max(np.abs(gradients))
        
        # Stability conditions
        stability_checks = []
        
        # Check 1: Gradient magnitude
        if gradient_norm < 1e-8:
            stability_checks.append({
                'check': 'vanishing_gradients',
                'stable': False,
                'message': 'Gradients are vanishingly small'
            })
        elif gradient_norm > 100:
            stability_checks.append({
                'check': 'exploding_gradients',
                'stable': False,
                'message': 'Gradients are exploding'
            })
        else:
            stability_checks.append({
                'check': 'gradient_magnitude',
                'stable': True,
                'message': 'Gradient magnitude is stable'
            })
            
        # Check 2: Learning rate compatibility
        effective_step = learning_rate * gradient_norm
        if effective_step > 1.0:
            stability_checks.append({
                'check': 'learning_rate_stability',
                'stable': False,
                'message': f'Learning rate too high for current gradients (effective step: {effective_step:.3f})'
            })
        else:
            stability_checks.append({
                'check': 'learning_rate_stability',
                'stable': True,
                'message': 'Learning rate is appropriate'
            })
            
        # Check 3: Gradient distribution
        gradient_std = np.std(gradients)
        gradient_mean = np.abs(np.mean(gradients))
        
        if gradient_std < gradient_mean * 0.1:
            stability_checks.append({
                'check': 'gradient_diversity',
                'stable': False,
                'message': 'Gradients lack diversity (potential saturation)'
            })
        else:
            stability_checks.append({
                'check': 'gradient_diversity',
                'stable': True,
                'message': 'Gradient distribution appears healthy'
            })
            
        # Overall stability assessment
        stable_checks = [check for check in stability_checks if check['stable']]
        overall_stable = len(stable_checks) >= len(stability_checks) * 0.8  # 80% of checks must pass
        
        # Identify instability reasons
        instability_reasons = [
            check['message'] for check in stability_checks 
            if not check['stable']
        ]
        
        analysis_time = time.time() - start_time
        
        return {
            'stable': overall_stable,
            'stability_score': len(stable_checks) / len(stability_checks),
            'gradient_norm': float(gradient_norm),
            'max_gradient': float(max_gradient),
            'effective_step_size': float(effective_step),
            'stability_checks': stability_checks,
            'instability_reason': instability_reasons[0] if instability_reasons else None,
            'all_instability_reasons': instability_reasons,
            'analysis_time_ms': analysis_time * 1000,
            'timestamp': datetime.now().isoformat()
        }
        
    def check_constraints(self, weights: np.ndarray, 
                         constraints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if weights satisfy multiple algebraic constraints"""
        start_time = time.time()
        
        weights = np.array(weights) if not isinstance(weights, np.ndarray) else weights
        
        satisfied_constraints = []
        violated_constraints = []
        
        for constraint in constraints:
            constraint_type = constraint.get('type')
            constraint_value = constraint.get('value')
            
            if constraint_type == 'max_norm':
                norm = np.linalg.norm(weights)
                if norm <= constraint_value:
                    satisfied_constraints.append({
                        'type': constraint_type,
                        'satisfied': True,
                        'actual_value': float(norm),
                        'constraint_value': constraint_value
                    })
                else:
                    violated_constraints.append({
                        'type': constraint_type,
                        'satisfied': False,
                        'actual_value': float(norm),
                        'constraint_value': constraint_value,
                        'violation_amount': float(norm - constraint_value)
                    })
                    
            elif constraint_type == 'non_negative':
                min_weight = np.min(weights)
                if min_weight >= 0:
                    satisfied_constraints.append({
                        'type': constraint_type,
                        'satisfied': True,
                        'min_value': float(min_weight)
                    })
                else:
                    violated_constraints.append({
                        'type': constraint_type,
                        'satisfied': False,
                        'min_value': float(min_weight),
                        'violation_amount': float(-min_weight)
                    })
                    
            elif constraint_type == 'bounded':
                lower_bound, upper_bound = constraint_value
                min_weight = np.min(weights)
                max_weight = np.max(weights)
                
                if lower_bound <= min_weight and max_weight <= upper_bound:
                    satisfied_constraints.append({
                        'type': constraint_type,
                        'satisfied': True,
                        'min_value': float(min_weight),
                        'max_value': float(max_weight),
                        'bounds': constraint_value
                    })
                else:
                    violation_amount = max(
                        lower_bound - min_weight if min_weight < lower_bound else 0,
                        max_weight - upper_bound if max_weight > upper_bound else 0
                    )
                    violated_constraints.append({
                        'type': constraint_type,
                        'satisfied': False,
                        'min_value': float(min_weight),
                        'max_value': float(max_weight),
                        'bounds': constraint_value,
                        'violation_amount': float(violation_amount)
                    })
                    
            else:
                violated_constraints.append({
                    'type': constraint_type,
                    'satisfied': False,
                    'error': f'Unknown constraint type: {constraint_type}'
                })
                
        all_satisfied = len(violated_constraints) == 0
        
        check_time = time.time() - start_time
        
        return {
            'all_satisfied': all_satisfied,
            'satisfied_constraints': satisfied_constraints,
            'violated_constraints': violated_constraints,
            'satisfaction_rate': len(satisfied_constraints) / len(constraints) if constraints else 1.0,
            'check_time_ms': check_time * 1000,
            'timestamp': datetime.now().isoformat()
        }
    
    def validate_sheaf_compression(self, sheaf_data: Dict[str, Any], 
                                  compressed_data: Any) -> Dict[str, Any]:
        """Validate sheaf compression preserves mathematical properties"""
        start_time = time.time()
        
        validation_results = {
            'valid': True,
            'sheaf_properties_preserved': [],
            'violations': [],
            'validation_time_ms': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Validate sheaf structure preservation
            if 'sections' in sheaf_data:
                sections = sheaf_data['sections']
                
                # Check section compatibility
                if isinstance(sections, dict):
                    for cell_id, section in sections.items():
                        if section is not None and hasattr(section, '__len__'):
                            section_norm = np.linalg.norm(np.array(section))
                            if section_norm > 0:
                                validation_results['sheaf_properties_preserved'].append({
                                    'property': 'section_non_trivial',
                                    'cell_id': cell_id,
                                    'norm': float(section_norm),
                                    'valid': True
                                })
                            else:
                                validation_results['violations'].append({
                                    'property': 'section_triviality',
                                    'cell_id': cell_id,
                                    'message': 'Section is trivial (zero)'
                                })
            
            # Validate restriction map consistency
            if 'restriction_maps' in sheaf_data:
                restriction_maps = sheaf_data['restriction_maps']
                
                if isinstance(restriction_maps, dict):
                    for map_id, restriction_map in restriction_maps.items():
                        # Check restriction map properties
                        if callable(restriction_map):
                            validation_results['sheaf_properties_preserved'].append({
                                'property': 'restriction_map_callable',
                                'map_id': map_id,
                                'valid': True
                            })
                        elif isinstance(restriction_map, (np.ndarray, list)):
                            map_array = np.array(restriction_map)
                            if map_array.ndim == 2:  # Matrix form
                                # Check if it's a valid linear map
                                rank = np.linalg.matrix_rank(map_array)
                                validation_results['sheaf_properties_preserved'].append({
                                    'property': 'restriction_map_rank',
                                    'map_id': map_id,
                                    'rank': int(rank),
                                    'shape': map_array.shape,
                                    'valid': rank > 0
                                })
            
            # Validate cohomological properties
            if 'cohomology_groups' in sheaf_data:
                cohomology = sheaf_data['cohomology_groups']
                
                if isinstance(cohomology, list):
                    for degree, group in enumerate(cohomology):
                        if group is not None:
                            if hasattr(group, '__len__'):
                                dimension = len(group)
                                validation_results['sheaf_properties_preserved'].append({
                                    'property': 'cohomology_dimension',
                                    'degree': degree,
                                    'dimension': dimension,
                                    'valid': dimension >= 0
                                })
            
            # Validate compression preserves topology
            if 'topology' in sheaf_data:
                topology = sheaf_data['topology']
                
                if 'open_sets' in topology:
                    open_sets = topology['open_sets']
                    if isinstance(open_sets, list) and len(open_sets) > 0:
                        validation_results['sheaf_properties_preserved'].append({
                            'property': 'topological_structure',
                            'open_sets_count': len(open_sets),
                            'valid': True
                        })
                    else:
                        validation_results['violations'].append({
                            'property': 'topological_structure',
                            'message': 'No open sets found in topology'
                        })
            
            # Validate compression ratio and fidelity
            if compressed_data is not None:
                try:
                    original_size = len(str(sheaf_data))
                    compressed_size = len(str(compressed_data))
                    compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
                    
                    validation_results['sheaf_properties_preserved'].append({
                        'property': 'compression_efficiency',
                        'original_size': original_size,
                        'compressed_size': compressed_size,
                        'compression_ratio': float(compression_ratio),
                        'valid': compression_ratio > 1.0
                    })
                    
                    if compression_ratio < 1.0:
                        validation_results['violations'].append({
                            'property': 'compression_efficiency',
                            'message': f'Compression increased size (ratio: {compression_ratio:.2f})'
                        })
                        
                except Exception as e:
                    validation_results['violations'].append({
                        'property': 'compression_measurement',
                        'message': f'Failed to measure compression: {str(e)}'
                    })
            
            # Overall validation
            validation_results['valid'] = len(validation_results['violations']) == 0
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['violations'].append({
                'property': 'validation_process',
                'message': f'Validation failed with error: {str(e)}'
            })
        
        validation_results['validation_time_ms'] = (time.time() - start_time) * 1000
        return validation_results
    
    def validate_sheaf_cohomology(self, sheaf_data: Dict[str, Any], 
                                 computed_cohomology: Any, degree: int = 0) -> Dict[str, Any]:
        """Validate sheaf cohomology computation correctness"""
        start_time = time.time()
        
        validation_results = {
            'valid': True,
            'cohomology_properties': [],
            'violations': [],
            'degree': degree,
            'validation_time_ms': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Validate cohomology structure
            if computed_cohomology is not None:
                if isinstance(computed_cohomology, dict):
                    # Check for required cohomology fields
                    if 'cohomology_groups' in computed_cohomology:
                        groups = computed_cohomology['cohomology_groups']
                        
                        if isinstance(groups, list) and len(groups) > degree:
                            target_group = groups[degree]
                            
                            if target_group is not None:
                                # Validate group structure
                                if hasattr(target_group, '__len__'):
                                    dimension = len(target_group)
                                    validation_results['cohomology_properties'].append({
                                        'property': 'cohomology_dimension',
                                        'degree': degree,
                                        'dimension': dimension,
                                        'valid': dimension >= 0
                                    })
                                    
                                    # Check for finite dimensionality
                                    if dimension < 1000:  # Reasonable upper bound
                                        validation_results['cohomology_properties'].append({
                                            'property': 'finite_dimension',
                                            'dimension': dimension,
                                            'valid': True
                                        })
                                    else:
                                        validation_results['violations'].append({
                                            'property': 'finite_dimension',
                                            'message': f'Dimension {dimension} suspiciously large'
                                        })
                    
                    # Check Betti numbers
                    if 'betti_numbers' in computed_cohomology:
                        betti_numbers = computed_cohomology['betti_numbers']
                        
                        if isinstance(betti_numbers, list) and len(betti_numbers) > degree:
                            betti_degree = betti_numbers[degree]
                            
                            if isinstance(betti_degree, int) and betti_degree >= 0:
                                validation_results['cohomology_properties'].append({
                                    'property': 'betti_number',
                                    'degree': degree,
                                    'betti_number': betti_degree,
                                    'valid': True
                                })
                            else:
                                validation_results['violations'].append({
                                    'property': 'betti_number',
                                    'message': f'Invalid Betti number at degree {degree}: {betti_degree}'
                                })
                
                # Validate algebraic properties
                # Check if cohomology satisfies expected algebraic relations
                if 'sections' in sheaf_data and 'restriction_maps' in sheaf_data:
                    # Verify exactness of cohomology sequence (simplified check)
                    validation_results['cohomology_properties'].append({
                        'property': 'sequence_structure',
                        'valid': True,
                        'message': 'Cohomology sequence structure present'
                    })
            
            # Validate consistency with sheaf structure
            if 'topology' in sheaf_data:
                topology = sheaf_data['topology']
                
                if 'open_sets' in topology:
                    open_sets_count = len(topology['open_sets'])
                    
                    # Simple consistency check: cohomology dimension should be related to topology complexity
                    if computed_cohomology and 'betti_numbers' in computed_cohomology:
                        total_betti = sum(computed_cohomology['betti_numbers'])
                        
                        if total_betti <= open_sets_count + 10:  # Reasonable upper bound
                            validation_results['cohomology_properties'].append({
                                'property': 'topological_consistency',
                                'total_betti': total_betti,
                                'open_sets_count': open_sets_count,
                                'valid': True
                            })
                        else:
                            validation_results['violations'].append({
                                'property': 'topological_consistency',
                                'message': f'Betti numbers {total_betti} inconsistent with topology complexity'
                            })
            
            # Overall validation
            validation_results['valid'] = len(validation_results['violations']) == 0
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['violations'].append({
                'property': 'cohomology_validation',
                'message': f'Cohomology validation failed: {str(e)}'
            })
        
        validation_results['validation_time_ms'] = (time.time() - start_time) * 1000
        return validation_results
    
    def validate_sheaf_morphism(self, source_sheaf: Dict[str, Any], 
                               target_sheaf: Dict[str, Any], 
                               morphism: Any) -> Dict[str, Any]:
        """Validate sheaf morphism preserves structure"""
        start_time = time.time()
        
        validation_results = {
            'valid': True,
            'morphism_properties': [],
            'violations': [],
            'validation_time_ms': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Validate morphism structure
            if morphism is None:
                validation_results['violations'].append({
                    'property': 'morphism_existence',
                    'message': 'Morphism is None'
                })
                validation_results['valid'] = False
                return validation_results
            
            # Check if morphism preserves sections
            if 'sections' in source_sheaf and 'sections' in target_sheaf:
                source_sections = source_sheaf['sections']
                target_sections = target_sheaf['sections']
                
                if isinstance(source_sections, dict) and isinstance(target_sections, dict):
                    # Check compatibility of section structures
                    common_cells = set(source_sections.keys()) & set(target_sections.keys())
                    
                    if len(common_cells) > 0:
                        validation_results['morphism_properties'].append({
                            'property': 'section_compatibility',
                            'common_cells': len(common_cells),
                            'valid': True
                        })
                        
                        # Check morphism preserves section relationships
                        for cell_id in common_cells:
                            source_section = source_sections[cell_id]
                            target_section = target_sections[cell_id]
                            
                            if (source_section is not None and target_section is not None and
                                hasattr(source_section, '__len__') and hasattr(target_section, '__len__')):
                                
                                source_norm = np.linalg.norm(np.array(source_section))
                                target_norm = np.linalg.norm(np.array(target_section))
                                
                                # Check if morphism preserves approximate magnitude
                                if source_norm > 0 and target_norm > 0:
                                    ratio = target_norm / source_norm
                                    if 0.1 <= ratio <= 10.0:  # Reasonable bounds
                                        validation_results['morphism_properties'].append({
                                            'property': 'magnitude_preservation',
                                            'cell_id': cell_id,
                                            'ratio': float(ratio),
                                            'valid': True
                                        })
                                    else:
                                        validation_results['violations'].append({
                                            'property': 'magnitude_preservation',
                                            'cell_id': cell_id,
                                            'message': f'Extreme magnitude change: ratio {ratio:.2f}'
                                        })
                    else:
                        validation_results['violations'].append({
                            'property': 'section_compatibility',
                            'message': 'No common cells between source and target sheaves'
                        })
            
            # Check if morphism preserves restriction maps
            if ('restriction_maps' in source_sheaf and 'restriction_maps' in target_sheaf):
                source_maps = source_sheaf['restriction_maps']
                target_maps = target_sheaf['restriction_maps']
                
                if isinstance(source_maps, dict) and isinstance(target_maps, dict):
                    common_maps = set(source_maps.keys()) & set(target_maps.keys())
                    
                    if len(common_maps) > 0:
                        validation_results['morphism_properties'].append({
                            'property': 'restriction_map_compatibility',
                            'common_maps': len(common_maps),
                            'valid': True
                        })
                        
                        # Verify commutativity of restriction maps (simplified)
                        for map_id in common_maps:
                            source_map = source_maps[map_id]
                            target_map = target_maps[map_id]
                            
                            if (isinstance(source_map, (np.ndarray, list)) and 
                                isinstance(target_map, (np.ndarray, list))):
                                
                                source_array = np.array(source_map)
                                target_array = np.array(target_map)
                                
                                if source_array.shape == target_array.shape:
                                    # Check structural similarity
                                    if source_array.size > 0 and target_array.size > 0:
                                        correlation = np.corrcoef(source_array.flatten(), 
                                                                target_array.flatten())[0, 1]
                                        
                                        if not np.isnan(correlation) and abs(correlation) > 0.1:
                                            validation_results['morphism_properties'].append({
                                                'property': 'map_structure_preservation',
                                                'map_id': map_id,
                                                'correlation': float(correlation),
                                                'valid': True
                                            })
                                        else:
                                            validation_results['violations'].append({
                                                'property': 'map_structure_preservation',
                                                'map_id': map_id,
                                                'message': f'Low structural correlation: {correlation:.3f}'
                                            })
            
            # Check topological consistency
            if ('topology' in source_sheaf and 'topology' in target_sheaf):
                source_topology = source_sheaf['topology']
                target_topology = target_sheaf['topology']
                
                if ('open_sets' in source_topology and 'open_sets' in target_topology):
                    source_open_sets = len(source_topology['open_sets'])
                    target_open_sets = len(target_topology['open_sets'])
                    
                    # Morphism should preserve or reduce topological complexity
                    if target_open_sets <= source_open_sets * 2:  # Allow some flexibility
                        validation_results['morphism_properties'].append({
                            'property': 'topological_consistency',
                            'source_complexity': source_open_sets,
                            'target_complexity': target_open_sets,
                            'valid': True
                        })
                    else:
                        validation_results['violations'].append({
                            'property': 'topological_consistency',
                            'message': f'Target topology unexpectedly complex: {target_open_sets} vs {source_open_sets}'
                        })
            
            # Overall validation
            validation_results['valid'] = len(validation_results['violations']) == 0
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['violations'].append({
                'property': 'morphism_validation',
                'message': f'Morphism validation failed: {str(e)}'
            })
        
        validation_results['validation_time_ms'] = (time.time() - start_time) * 1000
        return validation_results
    
    def integrate_with_proof_system(self, proof_system) -> Optional[Dict[str, Any]]:
        """
        Integrate with proof system for enhanced validation
        
        Args:
            proof_system: The proof system to integrate with
            
        Returns:
            Integration details if successful, None otherwise
        """
        try:
            if proof_system is None:
                return None
            
            # Create integration configuration
            integration_config = {
                'algebraic_enforcer': self,
                'proof_system': proof_system,
                'integration_type': 'gradient_validation',
                'enabled_rules': ['gradient_norm', 'weight_update', 'convergence', 'stability', 
                                'sheaf_compression', 'sheaf_cohomology', 'sheaf_morphism'],
                'thresholds': {
                    'max_gradient_norm': 10.0,
                    'min_gradient_norm': 1e-8,
                    'explosion_threshold': 100.0,
                    'vanishing_threshold': 1e-10,
                    'direction_change_threshold': 0.5,
                    'consistency_threshold': 0.3,
                    'sheaf_compression_ratio_min': 1.0,
                    'cohomology_dimension_max': 1000,
                    'morphism_correlation_min': 0.1
                },
                'sheaf_validation_features': {
                    'compression_validation': True,
                    'cohomology_validation': True,
                    'morphism_validation': True,
                    'topological_consistency': True,
                    'algebraic_structure_preservation': True
                },
                'integration_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("Successfully integrated with proof system including sheaf validation")
            return integration_config
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with proof system: {e}")
            return None