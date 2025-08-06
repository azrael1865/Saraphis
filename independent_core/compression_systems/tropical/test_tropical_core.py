"""
Test suite for tropical semiring operations.
Validates all core functionality and edge cases.
"""

import torch
import pytest
from tropical_core import (
    TROPICAL_ZERO,
    TropicalNumber,
    TropicalValidation,
    TropicalMathematicalOperations,
    TropicalGradientTracker,
    is_tropical_zero,
    to_tropical_safe,
    from_tropical_safe,
    tropical_distance
)


def test_tropical_number_creation():
    """Test TropicalNumber creation and validation"""
    # Valid creation
    t1 = TropicalNumber(5.0)
    assert t1.value == 5.0
    
    t2 = TropicalNumber(-10.0)
    assert t2.value == -10.0
    
    # Tropical zero
    t_zero = TropicalNumber(TROPICAL_ZERO)
    assert t_zero.is_zero()
    
    # Very negative values get clamped to TROPICAL_ZERO
    t_clamped = TropicalNumber(-1e40)
    assert t_clamped.is_zero()
    assert t_clamped.value == TROPICAL_ZERO
    
    # Invalid values
    with pytest.raises(TypeError):
        TropicalNumber("not a number")
    
    with pytest.raises(ValueError):
        TropicalNumber(float('nan'))
    
    with pytest.raises(ValueError):
        TropicalNumber(float('inf'))


def test_tropical_number_operations():
    """Test TropicalNumber arithmetic operations"""
    t1 = TropicalNumber(3.0)
    t2 = TropicalNumber(5.0)
    t_zero = TropicalNumber(TROPICAL_ZERO)
    
    # Tropical addition (max)
    result = t1 | t2
    assert result.value == 5.0
    
    result = t2 | t1
    assert result.value == 5.0
    
    # Addition with zero
    result = t1 | t_zero
    assert result.value == 3.0
    
    result = t_zero | t1
    assert result.value == 3.0
    
    # Tropical multiplication (addition)
    result = t1 + t2
    assert result.value == 8.0
    
    # Multiplication with zero
    result = t1 + t_zero
    assert result.is_zero()
    
    # Scalar multiplication
    result = 2 * t1
    assert result.value == 6.0
    
    result = t1 * 3
    assert result.value == 9.0
    
    # Scalar multiplication of zero
    result = 5 * t_zero
    assert result.is_zero()


def test_tropical_number_comparisons():
    """Test TropicalNumber comparison operations"""
    t1 = TropicalNumber(3.0)
    t2 = TropicalNumber(5.0)
    t3 = TropicalNumber(3.0)
    t_zero1 = TropicalNumber(TROPICAL_ZERO)
    t_zero2 = TropicalNumber(-1e50)  # Also tropical zero
    
    # Equality
    assert t1 == t3
    assert t_zero1 == t_zero2
    assert t1 != t2
    
    # Ordering
    assert t1 < t2
    assert t2 > t1
    assert t1 <= t3
    assert t1 >= t3
    assert t_zero1 < t1
    assert t1 > t_zero1


def test_tropical_validation():
    """Test TropicalValidation methods"""
    # Valid values
    TropicalValidation.validate_tropical_value(5.0)
    TropicalValidation.validate_tropical_value(-10.0)
    TropicalValidation.validate_tropical_value(TROPICAL_ZERO)
    
    # Invalid values
    with pytest.raises(TypeError):
        TropicalValidation.validate_tropical_value("not a number")
    
    with pytest.raises(ValueError):
        TropicalValidation.validate_tropical_value(float('nan'))
    
    with pytest.raises(ValueError):
        TropicalValidation.validate_tropical_value(float('inf'))
    
    with pytest.raises(ValueError):
        TropicalValidation.validate_tropical_value(1e39)
    
    # Valid tensors
    valid_tensor = torch.tensor([1.0, 2.0, 3.0])
    TropicalValidation.validate_tropical_tensor(valid_tensor)
    
    # Invalid tensors
    with pytest.raises(ValueError):
        TropicalValidation.validate_tropical_tensor(torch.tensor([]))
    
    with pytest.raises(ValueError):
        TropicalValidation.validate_tropical_tensor(torch.tensor([1.0, float('nan')]))
    
    with pytest.raises(ValueError):
        TropicalValidation.validate_tropical_tensor(torch.tensor([1.0, float('inf')]))
    
    with pytest.raises(TypeError):
        TropicalValidation.validate_tropical_tensor(torch.tensor([1, 2], dtype=torch.int32))


def test_tropical_mathematical_operations():
    """Test TropicalMathematicalOperations class"""
    ops = TropicalMathematicalOperations()
    
    # Scalar operations
    assert ops.tropical_add(3.0, 5.0) == 5.0
    assert ops.tropical_add(5.0, 3.0) == 5.0
    assert ops.tropical_add(3.0, TROPICAL_ZERO) == 3.0
    assert ops.tropical_add(TROPICAL_ZERO, 3.0) == 3.0
    
    assert ops.tropical_multiply(3.0, 5.0) == 8.0
    assert ops.tropical_multiply(3.0, TROPICAL_ZERO) == TROPICAL_ZERO
    assert ops.tropical_multiply(TROPICAL_ZERO, 3.0) == TROPICAL_ZERO
    
    # Tensor operations
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 1.0, 5.0])
    
    result = ops.tropical_add(a, b)
    expected = torch.tensor([4.0, 2.0, 5.0])
    assert torch.allclose(result, expected)
    
    result = ops.tropical_multiply(a, b)
    expected = torch.tensor([5.0, 3.0, 8.0])
    assert torch.allclose(result, expected)
    
    # Test with tropical zeros
    a_with_zero = torch.tensor([1.0, TROPICAL_ZERO, 3.0])
    b_normal = torch.tensor([2.0, 4.0, 1.0])
    
    result = ops.tropical_add(a_with_zero, b_normal)
    expected = torch.tensor([2.0, 4.0, 3.0])
    assert torch.allclose(result, expected)
    
    result = ops.tropical_multiply(a_with_zero, b_normal)
    assert result[1] == TROPICAL_ZERO


def test_tropical_power():
    """Test tropical power operation"""
    ops = TropicalMathematicalOperations()
    
    # Scalar power
    assert ops.tropical_power(3.0, 2) == 6.0
    assert ops.tropical_power(5.0, 3) == 15.0
    assert ops.tropical_power(TROPICAL_ZERO, 5) == TROPICAL_ZERO
    
    # Tensor power
    a = torch.tensor([2.0, 3.0, 4.0])
    result = ops.tropical_power(a, 3)
    expected = torch.tensor([6.0, 9.0, 12.0])
    assert torch.allclose(result, expected)
    
    # With tropical zeros
    a_with_zero = torch.tensor([2.0, TROPICAL_ZERO, 4.0])
    result = ops.tropical_power(a_with_zero, 2)
    assert result[0] == 4.0
    assert result[1] == TROPICAL_ZERO
    assert result[2] == 8.0


def test_tropical_sum_and_product():
    """Test tropical sum (max) and product (sum) operations"""
    ops = TropicalMathematicalOperations()
    
    # List operations
    values = [1.0, 5.0, 3.0, 2.0]
    assert ops.tropical_sum(values) == 5.0
    assert ops.tropical_product(values) == 11.0
    
    # With tropical zeros
    values_with_zero = [1.0, TROPICAL_ZERO, 3.0, 2.0]
    assert ops.tropical_sum(values_with_zero) == 3.0
    assert ops.tropical_product(values_with_zero) == TROPICAL_ZERO
    
    # Tensor operations
    tensor_values = torch.tensor([1.0, 5.0, 3.0, 2.0])
    assert ops.tropical_sum(tensor_values) == 5.0
    assert ops.tropical_product(tensor_values) == 11.0
    
    # Empty list/tensor errors
    with pytest.raises(ValueError):
        ops.tropical_sum([])
    
    with pytest.raises(ValueError):
        ops.tropical_product([])
    
    with pytest.raises(ValueError):
        ops.tropical_sum(torch.tensor([]))


def test_tropical_matrix_multiply():
    """Test tropical matrix multiplication"""
    ops = TropicalMathematicalOperations()
    
    # Simple 2x2 matrices
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    
    # (A ⊗ B)_ij = max_k(A_ik + B_kj)
    # Expected:
    # (0,0) = max(1+5, 2+7) = max(6, 9) = 9
    # (0,1) = max(1+6, 2+8) = max(7, 10) = 10
    # (1,0) = max(3+5, 4+7) = max(8, 11) = 11
    # (1,1) = max(3+6, 4+8) = max(9, 12) = 12
    
    result = ops.tropical_matrix_multiply(A, B)
    expected = torch.tensor([[9.0, 10.0], [11.0, 12.0]])
    assert torch.allclose(result, expected)
    
    # Test with tropical zeros
    A_with_zero = torch.tensor([[1.0, TROPICAL_ZERO], [3.0, 4.0]])
    B_with_zero = torch.tensor([[5.0, 6.0], [TROPICAL_ZERO, 8.0]])
    
    result = ops.tropical_matrix_multiply(A_with_zero, B_with_zero)
    # (0,0) = max(1+5, ZERO+ZERO) = 6
    # (0,1) = max(1+6, ZERO+8) = 7
    # (1,0) = max(3+5, 4+ZERO) = 8
    # (1,1) = max(3+6, 4+8) = 12
    expected = torch.tensor([[6.0, 7.0], [8.0, 12.0]])
    assert torch.allclose(result, expected)


def test_utility_functions():
    """Test utility functions"""
    # is_tropical_zero
    assert is_tropical_zero(TROPICAL_ZERO)
    assert is_tropical_zero(-1e40)
    assert not is_tropical_zero(0.0)
    assert not is_tropical_zero(5.0)
    
    t_zero = TropicalNumber(TROPICAL_ZERO)
    t_normal = TropicalNumber(5.0)
    assert is_tropical_zero(t_zero)
    assert not is_tropical_zero(t_normal)
    
    tensor = torch.tensor([1.0, TROPICAL_ZERO, 3.0])
    zero_mask = is_tropical_zero(tensor)
    assert zero_mask[1] and not zero_mask[0] and not zero_mask[2]
    
    # to_tropical_safe
    t = to_tropical_safe(5.0)
    assert isinstance(t, TropicalNumber)
    assert t.value == 5.0
    
    t = to_tropical_safe(-1e40)
    assert t.is_zero()
    
    tensor = torch.tensor([1.0, 2.0, -1e40])
    t_tensor = to_tropical_safe(tensor)
    assert t_tensor[2] == TROPICAL_ZERO
    
    # from_tropical_safe
    t = TropicalNumber(5.0)
    assert from_tropical_safe(t) == 5.0
    
    tensor = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(from_tropical_safe(tensor), tensor)
    
    # tropical_distance
    assert tropical_distance(3.0, 5.0) == 2.0
    assert tropical_distance(5.0, 3.0) == 2.0
    assert tropical_distance(3.0, 3.0) == 0.0
    assert tropical_distance(TROPICAL_ZERO, TROPICAL_ZERO) == 0.0
    assert tropical_distance(3.0, TROPICAL_ZERO) == float('inf')
    assert tropical_distance(TROPICAL_ZERO, 3.0) == float('inf')


def test_tropical_gradient_tracker():
    """Test gradient tracking for tropical operations"""
    tracker = TropicalGradientTracker()
    
    # Test tropical add (max) backward
    a = torch.tensor([1.0, 5.0, 3.0], requires_grad=True)
    b = torch.tensor([4.0, 2.0, 3.0], requires_grad=True)
    grad_output = torch.tensor([1.0, 1.0, 1.0])
    
    grad_a, grad_b = tracker.tropical_add_backward(grad_output, a, b)
    
    # Expected: gradient flows to the larger value
    # a[0] < b[0], so grad_a[0] = 0, grad_b[0] = 1
    # a[1] > b[1], so grad_a[1] = 1, grad_b[1] = 0
    # a[2] = b[2], so grad_a[2] = 0.5, grad_b[2] = 0.5
    assert grad_a[0] == 0.0 and grad_b[0] == 1.0
    assert grad_a[1] == 1.0 and grad_b[1] == 0.0
    assert grad_a[2] == 0.5 and grad_b[2] == 0.5
    
    # Test tropical multiply (add) backward
    grad_a, grad_b = tracker.tropical_multiply_backward(grad_output, a, b)
    
    # Expected: gradient passes through equally (it's just addition)
    assert torch.allclose(grad_a, grad_output)
    assert torch.allclose(grad_b, grad_output)
    
    # Test with tropical zeros
    a_with_zero = torch.tensor([1.0, TROPICAL_ZERO, 3.0])
    b_normal = torch.tensor([2.0, 4.0, 1.0])
    
    grad_a, grad_b = tracker.tropical_multiply_backward(grad_output, a_with_zero, b_normal)
    
    # Gradient should be zero where input is tropical zero
    assert grad_a[1] == 0.0 and grad_b[1] == 0.0


def test_gpu_compatibility():
    """Test GPU compatibility if available"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        ops = TropicalMathematicalOperations(device=device)
        
        # Create GPU tensors
        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        b = torch.tensor([4.0, 1.0, 5.0], device=device)
        
        # Test operations
        result = ops.tropical_add(a, b)
        assert result.device == device
        expected = torch.tensor([4.0, 2.0, 5.0], device=device)
        assert torch.allclose(result, expected)
        
        result = ops.tropical_multiply(a, b)
        assert result.device == device
        expected = torch.tensor([5.0, 3.0, 8.0], device=device)
        assert torch.allclose(result, expected)
        
        # Test gradient tracking
        tracker = TropicalGradientTracker(device=device)
        grad_output = torch.ones(3, device=device)
        grad_a, grad_b = tracker.tropical_add_backward(grad_output, a, b)
        assert grad_a.device == device
        assert grad_b.device == device


def test_overflow_protection():
    """Test overflow protection in operations"""
    ops = TropicalMathematicalOperations()
    
    # Test scalar overflow
    with pytest.raises(OverflowError):
        ops.tropical_multiply(1e38, 1e38)
    
    with pytest.raises(OverflowError):
        ops.tropical_power(1e20, 1e20)
    
    # Test tensor overflow
    large_tensor = torch.tensor([1e38, 1e37])
    with pytest.raises(OverflowError):
        ops.tropical_multiply(large_tensor, large_tensor)
    
    # Test list overflow
    with pytest.raises(OverflowError):
        ops.tropical_product([1e38, 1e38])


def test_edge_cases():
    """Test various edge cases"""
    ops = TropicalMathematicalOperations()
    
    # Single element operations
    assert ops.tropical_sum([5.0]) == 5.0
    assert ops.tropical_product([5.0]) == 5.0
    
    single_tensor = torch.tensor([5.0])
    assert ops.tropical_sum(single_tensor) == 5.0
    assert ops.tropical_product(single_tensor) == 5.0
    
    # All tropical zeros
    all_zeros = [TROPICAL_ZERO, TROPICAL_ZERO, TROPICAL_ZERO]
    assert ops.tropical_sum(all_zeros) == TROPICAL_ZERO
    assert ops.tropical_product(all_zeros) == TROPICAL_ZERO
    
    # Very small differences (numerical stability)
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([1.0 + 1e-12, 2.0 - 1e-12])
    result = ops.tropical_add(a, b)
    assert torch.allclose(result, b, atol=1e-10)


if __name__ == "__main__":
    # Run all tests
    test_tropical_number_creation()
    test_tropical_number_operations()
    test_tropical_number_comparisons()
    test_tropical_validation()
    test_tropical_mathematical_operations()
    test_tropical_power()
    test_tropical_sum_and_product()
    test_tropical_matrix_multiply()
    test_utility_functions()
    test_tropical_gradient_tracker()
    test_gpu_compatibility()
    test_overflow_protection()
    test_edge_cases()
    
    print("All tests passed!")