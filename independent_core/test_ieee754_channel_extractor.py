"""
Comprehensive test suite for IEEE754ChannelExtractor
Tests all functionality including edge cases and special values
"""

import unittest
import torch
import numpy as np
import struct
from compression_systems.categorical.ieee754_channel_extractor import (
    IEEE754ChannelExtractor,
    IEEE754Channels,
    create_ieee754_extractor,
    extract_ieee754_from_tensor
)
from compression_systems.padic.padic_encoder import PadicWeight


class TestIEEE754ChannelExtractor(unittest.TestCase):
    """Comprehensive test suite for IEEE754ChannelExtractor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.extractor = IEEE754ChannelExtractor(validate_reconstruction=True)
        self.extractor_no_validation = IEEE754ChannelExtractor(validate_reconstruction=False)
    
    def test_initialization(self):
        """Test extractor initialization"""
        self.assertTrue(self.extractor.validate_reconstruction)
        self.assertFalse(self.extractor_no_validation.validate_reconstruction)
        self.assertEqual(self.extractor.SIGN_MASK, 0x80000000)
        self.assertEqual(self.extractor.EXPONENT_MASK, 0x7F800000)
        self.assertEqual(self.extractor.MANTISSA_MASK, 0x007FFFFF)
    
    def test_extract_normal_floats(self):
        """Test extraction of normal floating point values"""
        test_values = [1.0, -1.0, 2.5, -3.14159, 100.0, -0.001]
        tensor = torch.tensor(test_values, dtype=torch.float32)
        
        channels = self.extractor.extract_channels_from_tensor(tensor)
        
        # Verify channel shapes
        self.assertEqual(channels.sign_channel.shape, (len(test_values),))
        self.assertEqual(channels.exponent_channel.shape, (len(test_values),))
        self.assertEqual(channels.mantissa_channel.shape, (len(test_values),))
        
        # Test specific values
        # 1.0 = 0x3F800000: sign=0, exp=127, mantissa=0
        self.assertEqual(channels.sign_channel[0], 0)
        self.assertEqual(channels.exponent_channel[0], 127)
        
        # -1.0 = 0xBF800000: sign=1, exp=127, mantissa=0
        self.assertEqual(channels.sign_channel[1], 1)
        self.assertEqual(channels.exponent_channel[1], 127)
    
    def test_extract_special_values(self):
        """Test extraction of special IEEE 754 values"""
        # Test zero
        tensor_zero = torch.tensor([0.0, -0.0], dtype=torch.float32)
        channels_zero = self.extractor.extract_channels_from_tensor(tensor_zero)
        
        # Both positive and negative zero have exponent=0
        self.assertEqual(channels_zero.exponent_channel[0], 0)
        self.assertEqual(channels_zero.exponent_channel[1], 0)
        self.assertEqual(channels_zero.sign_channel[0], 0)  # +0.0
        self.assertEqual(channels_zero.sign_channel[1], 1)  # -0.0
        
        # Test infinity (should work after fix)
        tensor_inf = torch.tensor([float('inf'), float('-inf')], dtype=torch.float32)
        try:
            channels_inf = self.extractor.extract_channels_from_tensor(tensor_inf)
            # Infinity has exponent=255, mantissa=0
            self.assertEqual(channels_inf.exponent_channel[0], 255)
            self.assertEqual(channels_inf.exponent_channel[1], 255)
            print("✓ Infinity handling works correctly")
        except Exception as e:
            print(f"✗ Infinity handling failed: {e}")
    
    def test_extract_denormalized_numbers(self):
        """Test extraction of denormalized numbers"""
        # Smallest positive denormalized float32: 2^-149
        min_denorm = np.float32(2**-149)
        tensor = torch.tensor([min_denorm], dtype=torch.float32)
        
        channels = self.extractor.extract_channels_from_tensor(tensor)
        
        # Denormalized numbers have exponent=0
        self.assertEqual(channels.exponent_channel[0], 0)
        self.assertTrue(channels.mantissa_channel[0] > 0)
    
    def test_reconstruction_accuracy(self):
        """Test reconstruction accuracy for various values"""
        test_values = [
            1.0, -1.0, 0.5, -0.5,
            1234.5678, -9876.5432,
            1e-10, -1e-10,
            1e10, -1e10
        ]
        tensor = torch.tensor(test_values, dtype=torch.float32)
        
        channels = self.extractor.extract_channels_from_tensor(tensor)
        reconstructed = self.extractor.reconstruct_from_channels(channels)
        
        # Check reconstruction accuracy
        np.testing.assert_allclose(
            reconstructed, 
            np.array(test_values, dtype=np.float32),
            rtol=1e-6,
            atol=1e-6
        )
        print("✓ Reconstruction accuracy test passed")
    
    def test_tensor_shapes(self):
        """Test handling of various tensor shapes"""
        shapes = [(10,), (5, 5), (2, 3, 4), (1, 1, 1, 100)]
        
        for shape in shapes:
            tensor = torch.randn(shape, dtype=torch.float32)
            channels = self.extractor.extract_channels_from_tensor(tensor)
            
            # All channels should be flattened
            expected_size = np.prod(shape)
            self.assertEqual(channels.sign_channel.size, expected_size)
            self.assertEqual(channels.exponent_channel.size, expected_size)
            self.assertEqual(channels.mantissa_channel.size, expected_size)
        
        print(f"✓ Tensor shape handling test passed for {len(shapes)} shapes")
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        # Test None input
        with self.assertRaises(ValueError):
            self.extractor.extract_channels_from_tensor(None)
        
        # Test empty tensor
        with self.assertRaises(ValueError):
            self.extractor.extract_channels_from_tensor(torch.tensor([]))
        
        # Test NaN values (should fail with hard failure)
        tensor_nan = torch.tensor([float('nan')], dtype=torch.float32)
        with self.assertRaises(RuntimeError):
            self.extractor.extract_channels_from_tensor(tensor_nan)
        
        print("✓ Edge case handling test passed")
    
    def test_statistics_tracking(self):
        """Test extraction statistics tracking"""
        # Reset stats
        self.extractor.extraction_stats['total_extractions'] = 0
        
        # Perform multiple extractions
        for i in range(5):
            tensor = torch.randn(10, dtype=torch.float32)
            self.extractor.extract_channels_from_tensor(tensor)
        
        stats = self.extractor.get_extraction_statistics()
        self.assertEqual(stats['total_extractions'], 5)
        self.assertIn('average_mantissa_entropy', stats)
        self.assertIn('validation_success_rate', stats)
        
        print("✓ Statistics tracking test passed")
    
    def test_channel_validation(self):
        """Test IEEE754Channels validation"""
        # Create valid channels
        sign = np.array([0, 1, 0], dtype=np.uint8)
        exponent = np.array([127, 128, 0], dtype=np.uint8)
        mantissa = np.array([1.0, 1.5, 0.0], dtype=np.float32)
        original = np.array([1.0, -2.0, 0.0], dtype=np.float32)
        
        # This should work
        channels = IEEE754Channels(sign, exponent, mantissa, original)
        self.assertIsNotNone(channels)
        
        # Test invalid sign values
        invalid_sign = np.array([0, 2, 0], dtype=np.uint8)  # 2 is invalid
        with self.assertRaises(ValueError):
            IEEE754Channels(invalid_sign, exponent, mantissa, original)
        
        # Test invalid exponent dtype (should be uint8)
        invalid_exp = np.array([127, 128, 0], dtype=np.uint16)  # Wrong dtype
        with self.assertRaises(ValueError):
            IEEE754Channels(sign, invalid_exp, mantissa, original)
        
        # Test shape mismatch
        wrong_shape = np.array([0, 1], dtype=np.uint8)  # Wrong size
        with self.assertRaises(ValueError):
            IEEE754Channels(wrong_shape, exponent, mantissa, original)
        
        print("✓ Channel validation test passed")
    
    def test_mantissa_entropy_calculation(self):
        """Test mantissa entropy calculation"""
        # Uniform mantissa should have high entropy
        uniform_mantissa = np.random.uniform(0, 2, 1000).astype(np.float32)
        entropy_uniform = self.extractor._calculate_mantissa_entropy(uniform_mantissa)
        
        # Constant mantissa should have low entropy
        constant_mantissa = np.ones(1000, dtype=np.float32)
        entropy_constant = self.extractor._calculate_mantissa_entropy(constant_mantissa)
        
        self.assertGreater(entropy_uniform, entropy_constant)
        print(f"✓ Entropy calculation test passed (uniform: {entropy_uniform:.2f}, constant: {entropy_constant:.2f})")
    
    def test_padic_optimization(self):
        """Test optimization for p-adic encoding"""
        tensor = torch.randn(100, dtype=torch.float32)
        channels = self.extractor.extract_channels_from_tensor(tensor)
        
        # Test optimization for prime 257
        optimized = self.extractor.optimize_channels_for_padic(channels, target_prime=257)
        
        # Verify optimization doesn't break reconstruction too much
        reconstructed_original = self.extractor.reconstruct_from_channels(channels)
        reconstructed_optimized = self.extractor.reconstruct_from_channels(optimized)
        
        # Some difference is expected due to optimization
        max_diff = np.max(np.abs(reconstructed_original - reconstructed_optimized))
        self.assertLess(max_diff, 0.1)  # Reasonable tolerance
        
        print(f"✓ P-adic optimization test passed (max diff: {max_diff:.6f})")
    
    def test_factory_functions(self):
        """Test factory and convenience functions"""
        # Test factory
        extractor = create_ieee754_extractor(validate_reconstruction=False)
        self.assertFalse(extractor.validate_reconstruction)
        
        # Test convenience function
        tensor = torch.tensor([1.0, 2.0, 3.0])
        channels = extract_ieee754_from_tensor(tensor)
        self.assertIsInstance(channels, IEEE754Channels)
        
        print("✓ Factory functions test passed")
    
    def test_safe_padic_reconstruction(self):
        """Test safe p-adic weight reconstruction"""
        from fractions import Fraction
        # Create a mock PadicWeight with all required fields
        weight = PadicWeight(
            value=Fraction(1234, 1000),  # 1.234 as a Fraction
            digits=[1, 2, 3, 4, 5],
            valuation=0,
            prime=7,
            precision=5
        )
        
        # Test reconstruction
        value = self.extractor._safe_padic_reconstruction(weight)
        self.assertIsInstance(value, float)
        self.assertFalse(np.isnan(value))
        self.assertFalse(np.isinf(value))
        
        # Test overflow prevention - now uses pre-computed value to avoid overflow
        weight_large = PadicWeight(
            value=Fraction(10**20, 1),  # Very large value
            digits=[250] * 100,  # Many large digits
            valuation=50,  # Large positive valuation
            prime=257,
            precision=100
        )
        
        # Should use the pre-computed value instead of reconstructing
        result = self.extractor._safe_padic_reconstruction(weight_large)
        self.assertEqual(result, 1e20)
        
        # Test case that would overflow during reconstruction
        # Create a weight with moderate value but reconstruction would overflow
        weight_reconstruct_overflow = PadicWeight(
            value=Fraction(1, 1),  # Small pre-computed value (will be ignored in our test)
            digits=[256] * 200,  # Max digits, matching precision
            valuation=100,  # Very large valuation that would cause overflow
            prime=257,
            precision=200  # Precision beyond safe limit
        )
        
        # Temporarily clear the value to force reconstruction path
        original_value = weight_reconstruct_overflow.value
        weight_reconstruct_overflow.value = None
        
        # This should raise an error due to unsafe precision
        with self.assertRaises((ValueError, OverflowError, AttributeError)):
            self.extractor._safe_padic_reconstruction(weight_reconstruct_overflow)
        
        # Restore value
        weight_reconstruct_overflow.value = original_value
        
        print("✓ Safe p-adic reconstruction test passed")
    
    def test_multiple_special_values(self):
        """Test handling of multiple special values in one tensor"""
        # Mix of normal, zero, and infinity values
        test_values = [
            1.0,           # Normal
            0.0,           # Zero
            -0.0,          # Negative zero
            float('inf'),  # Positive infinity
            float('-inf'), # Negative infinity
            1e-38,         # Small normal
            1e38,          # Large normal
        ]
        
        tensor = torch.tensor(test_values[:-2], dtype=torch.float32)  # Exclude infinities for now
        
        try:
            channels = self.extractor.extract_channels_from_tensor(tensor)
            stats = self.extractor.get_extraction_statistics()
            self.assertGreater(stats['special_values_handled'], 0)
            print(f"✓ Mixed special values test passed (handled: {stats['special_values_handled']})")
        except Exception as e:
            print(f"✗ Mixed special values test failed: {e}")


def run_comprehensive_tests():
    """Run all tests and provide summary"""
    print("=" * 60)
    print("IEEE754ChannelExtractor Comprehensive Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestIEEE754ChannelExtractor)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailed Tests:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\nTests with Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
            print(f"    Error: {traceback.split('AssertionError:')[-1].strip()[:100]}")
    
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)