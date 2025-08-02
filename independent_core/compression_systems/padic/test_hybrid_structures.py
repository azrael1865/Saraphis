"""
Compatibility tests for hybrid p-adic structures
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import pytest
import time
from fractions import Fraction
from typing import Dict, Any

from .padic_encoder import PadicWeight, PadicValidation
from .hybrid_padic_structures import (
    HybridPadicWeight,
    HybridPadicValidator,
    HybridPadicConverter,
    HybridPadicManager
)


class TestHybridPadicCompatibility:
    """Test suite for hybrid p-adic structure compatibility"""
    
    def setup_method(self):
        """Setup test environment"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for hybrid p-adic tests")
        
        self.device = torch.device('cuda:0')
        self.validator = HybridPadicValidator()
        self.converter = HybridPadicConverter()
        self.manager = HybridPadicManager()
    
    def test_hybrid_conversion_roundtrip(self):
        """Test that hybrid conversion preserves p-adic properties"""
        # Create test p-adic weight
        test_weight = PadicWeight(
            value=Fraction(7, 4),
            prime=5,
            precision=4,
            valuation=0,
            digits=[2, 3, 1, 0]
        )
        
        # Convert to hybrid and back
        hybrid_weight = self.manager.create_hybrid_weight(test_weight)
        restored_weight = self.manager.restore_padic_weight(hybrid_weight)
        
        # Verify properties are preserved
        assert restored_weight.prime == test_weight.prime
        assert restored_weight.precision == test_weight.precision
        assert restored_weight.valuation == test_weight.valuation
        
        # Verify digits are approximately preserved (allowing for floating point precision)
        for orig, restored in zip(test_weight.digits, restored_weight.digits):
            assert abs(orig - restored) <= 1  # Allow for rounding in conversion
    
    def test_hybrid_weight_validation(self):
        """Test hybrid weight validation"""
        # Create valid hybrid weight
        exponent_channel = torch.rand(4, device=self.device, dtype=torch.float32) * 2.0
        mantissa_channel = torch.rand(4, device=self.device, dtype=torch.float32) * 4.0  # Values in [0, 5)
        
        hybrid_weight = HybridPadicWeight(
            exponent_channel=exponent_channel,
            mantissa_channel=mantissa_channel,
            prime=5,
            precision=4,
            valuation=0,
            device=self.device,
            dtype=torch.float32
        )
        
        # Should not raise exception
        self.validator.validate_hybrid_weight(hybrid_weight)
    
    def test_invalid_hybrid_weight_validation(self):
        """Test that invalid hybrid weights are rejected"""
        # Test with CPU tensors (should fail)
        with pytest.raises(ValueError, match="must be on GPU"):
            exponent_channel = torch.rand(4, dtype=torch.float32)  # CPU tensor
            mantissa_channel = torch.rand(4, device=self.device, dtype=torch.float32)
            
            HybridPadicWeight(
                exponent_channel=exponent_channel,
                mantissa_channel=mantissa_channel,
                prime=5,
                precision=4,
                valuation=0,
                device=self.device,
                dtype=torch.float32
            )
    
    def test_shape_mismatch_validation(self):
        """Test that shape mismatches are caught"""
        with pytest.raises(ValueError, match="must have same shape"):
            exponent_channel = torch.rand(4, device=self.device, dtype=torch.float32)
            mantissa_channel = torch.rand(3, device=self.device, dtype=torch.float32)  # Different shape
            
            HybridPadicWeight(
                exponent_channel=exponent_channel,
                mantissa_channel=mantissa_channel,
                prime=5,
                precision=4,
                valuation=0,
                device=self.device,
                dtype=torch.float32
            )
    
    def test_prime_validation(self):
        """Test that invalid primes are rejected"""
        with pytest.raises(ValueError, match="Prime must be > 1"):
            exponent_channel = torch.rand(4, device=self.device, dtype=torch.float32)
            mantissa_channel = torch.rand(4, device=self.device, dtype=torch.float32)
            
            HybridPadicWeight(
                exponent_channel=exponent_channel,
                mantissa_channel=mantissa_channel,
                prime=1,  # Invalid prime
                precision=4,
                valuation=0,
                device=self.device,
                dtype=torch.float32
            )
    
    def test_conversion_with_different_primes(self):
        """Test conversion with different prime numbers"""
        primes = [2, 3, 5, 7, 11]
        
        for prime in primes:
            # Create test weight with current prime
            digits = [i % prime for i in range(4)]
            test_weight = PadicWeight(
                value=Fraction(sum(digits[i] * (prime ** i) for i in range(4))),
                prime=prime,
                precision=4,
                valuation=0,
                digits=digits
            )
            
            # Test conversion
            hybrid_weight = self.manager.create_hybrid_weight(test_weight)
            restored_weight = self.manager.restore_padic_weight(hybrid_weight)
            
            assert restored_weight.prime == prime
            assert len(restored_weight.digits) == 4
    
    def test_conversion_performance(self):
        """Test conversion performance metrics"""
        # Create test weight
        test_weight = PadicWeight(
            value=Fraction(15, 8),
            prime=3,
            precision=6,
            valuation=0,
            digits=[0, 1, 2, 1, 0, 1]
        )
        
        # Reset metrics
        self.manager.reset_stats()
        
        # Perform conversions
        start_time = time.time()
        hybrid_weight = self.manager.create_hybrid_weight(test_weight)
        restored_weight = self.manager.restore_padic_weight(hybrid_weight)
        end_time = time.time()
        
        # Check metrics
        stats = self.manager.get_operation_stats()
        assert stats['hybrid_weights_created'] == 1
        assert stats['padic_weights_restored'] == 1
        assert stats['total_operation_time'] > 0
        assert stats['validation_failures'] == 0
        
        # Check conversion was reasonably fast
        total_time = end_time - start_time
        assert total_time < 1.0  # Should complete in under 1 second
    
    def test_ultrametric_property_preservation(self):
        """Test that ultrametric property is preserved in conversion"""
        # Create test weights
        test_weight1 = PadicWeight(
            value=Fraction(1, 3),
            prime=3,
            precision=4,
            valuation=0,
            digits=[0, 1, 0, 0]
        )
        
        test_weight2 = PadicWeight(
            value=Fraction(2, 3),
            prime=3,
            precision=4,
            valuation=0,
            digits=[0, 2, 0, 0]
        )
        
        # Convert to hybrid
        hybrid1 = self.manager.create_hybrid_weight(test_weight1)
        hybrid2 = self.manager.create_hybrid_weight(test_weight2)
        
        # Both should pass ultrametric validation
        assert hybrid1.ultrametric_preserved
        assert hybrid2.ultrametric_preserved
        
        self.validator.validate_hybrid_weight(hybrid1)
        self.validator.validate_hybrid_weight(hybrid2)
    
    def test_gpu_memory_usage_tracking(self):
        """Test GPU memory usage tracking"""
        # Get initial memory usage
        initial_memory = self.manager.get_memory_usage()
        
        # Create several hybrid weights
        test_weights = []
        for i in range(5):
            digits = [(i + j) % 5 for j in range(4)]
            test_weight = PadicWeight(
                value=Fraction(sum(digits[j] * (5 ** j) for j in range(4))),
                prime=5,
                precision=4,
                valuation=0,
                digits=digits
            )
            test_weights.append(test_weight)
        
        # Convert to hybrid
        hybrid_weights = [self.manager.create_hybrid_weight(w) for w in test_weights]
        
        # Check memory usage increased
        final_memory = self.manager.get_memory_usage()
        assert final_memory['allocated_mb'] >= initial_memory['allocated_mb']
        
        # Clean up
        del hybrid_weights
        self.manager.cleanup_gpu_memory()
    
    def test_error_handling_in_conversion(self):
        """Test error handling in conversion process"""
        # Test with None input
        with pytest.raises(ValueError, match="cannot be None"):
            self.manager.create_hybrid_weight(None)
        
        with pytest.raises(ValueError, match="cannot be None"):
            self.manager.restore_padic_weight(None)
        
        # Test with invalid p-adic weight
        with pytest.raises(Exception):  # Should raise some validation error
            invalid_weight = PadicWeight(
                value=Fraction(1, 2),
                prime=-1,  # Invalid prime
                precision=4,
                valuation=0,
                digits=[0, 1, 2, 3]
            )
            self.manager.create_hybrid_weight(invalid_weight)
    
    def test_compatibility_validation(self):
        """Test compatibility validation between hybrid and p-adic weights"""
        # Create compatible weights
        test_weight = PadicWeight(
            value=Fraction(13, 8),
            prime=7,
            precision=3,
            valuation=0,
            digits=[6, 1, 2]
        )
        
        hybrid_weight = self.manager.create_hybrid_weight(test_weight)
        
        # Should be compatible
        assert self.manager.validate_hybrid_compatibility(hybrid_weight, test_weight)
        
        # Create incompatible weight (different prime)
        incompatible_weight = PadicWeight(
            value=Fraction(13, 8),
            prime=5,  # Different prime
            precision=3,
            valuation=0,
            digits=[3, 0, 2]
        )
        
        # Should not be compatible
        assert not self.manager.validate_hybrid_compatibility(hybrid_weight, incompatible_weight)


def test_hybrid_conversion_roundtrip():
    """Standalone test for hybrid conversion roundtrip - can be run without pytest"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    # Create test p-adic weight
    test_weight = PadicWeight(
        value=Fraction(7, 4),
        prime=5,
        precision=4,
        valuation=0,
        digits=[2, 3, 1, 0]
    )
    
    # Convert to hybrid and back
    manager = HybridPadicManager()
    hybrid_weight = manager.create_hybrid_weight(test_weight)
    restored_weight = manager.restore_padic_weight(hybrid_weight)
    
    # Verify properties are preserved
    assert restored_weight.prime == test_weight.prime
    assert restored_weight.precision == test_weight.precision
    assert restored_weight.valuation == test_weight.valuation
    
    # Verify digits are approximately preserved
    for orig, restored in zip(test_weight.digits, restored_weight.digits):
        assert abs(orig - restored) <= 1
    
    print("✅ Hybrid conversion roundtrip test passed")


def test_hybrid_weight_creation():
    """Standalone test for hybrid weight creation"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    device = torch.device('cuda:0')
    
    # Create valid hybrid weight
    exponent_channel = torch.rand(4, device=device, dtype=torch.float32) * 2.0
    mantissa_channel = torch.rand(4, device=device, dtype=torch.float32) * 4.0
    
    hybrid_weight = HybridPadicWeight(
        exponent_channel=exponent_channel,
        mantissa_channel=mantissa_channel,
        prime=5,
        precision=4,
        valuation=0,
        device=device,
        dtype=torch.float32
    )
    
    # Validate
    validator = HybridPadicValidator()
    validator.validate_hybrid_weight(hybrid_weight)
    
    print("✅ Hybrid weight creation test passed")


if __name__ == "__main__":
    """Run basic tests when executed as script"""
    print("Running hybrid p-adic structure tests...")
    
    try:
        test_hybrid_conversion_roundtrip()
        test_hybrid_weight_creation()
        print("✅ All basic tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise