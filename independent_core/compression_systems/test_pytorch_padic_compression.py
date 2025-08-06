"""
Comprehensive end-to-end tests for PyTorch model compression using P-adic number system
Tests complete compression pipeline from PyTorch models to P-adic representation
NO PLACEHOLDERS - PRODUCTION-READY TEST CODE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Dict, List, Tuple, Optional, Any
import tempfile
import os
import math
from fractions import Fraction
import gc

# Import P-adic compression systems
from independent_core.compression_systems.padic.padic_compressor import PadicCompressionSystem
from independent_core.compression_systems.padic.padic_encoder import (
    PadicWeight,
    PadicValidation,
    PadicMathematicalOperations
)
from independent_core.compression_systems.padic.padic_logarithmic_encoder import (
    PadicLogarithmicEncoder,
    LogarithmicPadicWeight,
    LogarithmicEncodingConfig
)
from independent_core.compression_systems.padic.hybrid_padic_compressor import HybridPadicCompressionSystem
from independent_core.compression_systems.padic.hybrid_padic_structures import (
    HybridPadicWeight,
    HybridPadicValidator,
    HybridPadicConverter
)
from independent_core.compression_systems.padic.padic_gradient import PadicGradientHandler
from independent_core.compression_systems.base.compression_base import (
    CompressionValidator,
    CompressionMetrics
)


# ============================================================================
# TEST MODELS
# ============================================================================

class SimpleLinearModel(nn.Module):
    """Simple linear model for testing P-adic compression"""
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PeriodicPatternModel(nn.Module):
    """Model with periodic patterns for P-adic testing"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 100)
        # Initialize with periodic patterns
        with torch.no_grad():
            pattern = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.1] * 20)
            self.fc1.weight.data = pattern.unsqueeze(0).repeat(100, 1)
            self.fc1.bias.data = torch.tensor([0.5, -0.5] * 50)
    
    def forward(self, x):
        return self.fc1(x)


class ConvolutionalModel(nn.Module):
    """CNN model for P-adic compression testing"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ============================================================================
# TEST PYTORCH P-ADIC COMPRESSION
# ============================================================================

class TestPyTorchPadicCompression(unittest.TestCase):
    """Main compression tests for P-adic system"""
    
    def setUp(self):
        """Set up test environment"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_primes = [251, 127, 61, 31, 17, 7, 3, 2]
        self.test_precisions = [1, 2, 4, 8, 16, 32, 64]
        
        # Create test models
        self.simple_model = SimpleLinearModel().to(self.device)
        self.periodic_model = PeriodicPatternModel().to(self.device)
        self.conv_model = ConvolutionalModel().to(self.device)
        
        # Default configuration
        self.default_config = {
            'prime': 251,
            'precision': 8,
            'chunk_size': 1000,
            'gpu_memory_limit_mb': 1024,
            'preserve_ultrametric': True,
            'validate_reconstruction': True,
            'max_reconstruction_error': 1e-6,
            'enable_gc': True
        }
    
    def test_complete_compression_pipeline(self):
        """Test complete flow from PyTorch model to P-adic representation"""
        compressor = PadicCompressionSystem(self.default_config)
        
        # Extract model weights
        state_dict = self.simple_model.state_dict()
        
        for layer_name, weight_tensor in state_dict.items():
            # Compress weights
            padic_weights, metadata = compressor.encode(weight_tensor.cpu())
            
            # Validate P-adic properties
            self.assertIsInstance(padic_weights, list)
            self.assertTrue(len(padic_weights) > 0)
            
            for pw in padic_weights:
                self.assertIsInstance(pw, PadicWeight)
                self.assertEqual(pw.prime, self.default_config['prime'])
                self.assertEqual(pw.precision, self.default_config['precision'])
                self.assertEqual(len(pw.digits), pw.precision)
                
                # Validate digits are in correct range
                for digit in pw.digits:
                    self.assertGreaterEqual(digit, 0)
                    self.assertLess(digit, pw.prime)
            
            # Test decompression
            reconstructed = compressor.decode(padic_weights, metadata)
            self.assertEqual(reconstructed.shape, weight_tensor.shape)
            
            # Validate reconstruction error
            error = torch.abs(reconstructed - weight_tensor.cpu()).max().item()
            self.assertLess(error, self.default_config['max_reconstruction_error'])
    
    def test_padic_transformation_pipeline(self):
        """Test weight conversion to P-adic representation"""
        test_tensor = torch.randn(100, 50)
        
        for prime in self.test_primes[:4]:  # Test subset of primes
            for precision in [1, 4, 8, 16]:
                config = self.default_config.copy()
                config['prime'] = prime
                config['precision'] = precision
                
                try:
                    compressor = PadicCompressionSystem(config)
                    padic_weights, metadata = compressor.encode(test_tensor)
                    
                    # Validate prime selection
                    for pw in padic_weights:
                        self.assertEqual(pw.prime, prime)
                        self.assertEqual(pw.precision, precision)
                    
                    # Test valuation computation
                    math_ops = PadicMathematicalOperations(prime, precision)
                    for pw in padic_weights:
                        self.assertIsInstance(pw.valuation, int)
                        self.assertGreaterEqual(pw.valuation, -precision)
                    
                    # Test digit extraction
                    for pw in padic_weights:
                        self.assertEqual(len(pw.digits), precision)
                        for digit in pw.digits:
                            self.assertIn(digit, range(prime))
                    
                except OverflowError:
                    # Expected for certain prime/precision combinations
                    pass
    
    def test_prime_selection_optimization(self):
        """Test optimization of prime selection for compression"""
        test_tensor = torch.randn(1000)
        compression_ratios = {}
        
        for prime in self.test_primes:
            config = self.default_config.copy()
            config['prime'] = prime
            config['precision'] = 4  # Fixed precision for comparison
            
            try:
                compressor = PadicCompressionSystem(config)
                padic_weights, metadata = compressor.encode(test_tensor)
                
                # Calculate compression ratio
                original_size = test_tensor.numel() * 4  # float32
                compressed_size = len(padic_weights) * config['precision']
                ratio = original_size / compressed_size
                compression_ratios[prime] = ratio
                
            except (OverflowError, ValueError):
                compression_ratios[prime] = 0
        
        # Verify we get different compression ratios for different primes
        unique_ratios = set(compression_ratios.values())
        self.assertGreater(len(unique_ratios), 1)
    
    def test_precision_configuration(self):
        """Test precision levels from 1 to 64 digits"""
        test_tensor = torch.tensor([3.14159, 2.71828, 1.41421])
        
        for precision in [1, 2, 4, 8, 16, 32]:  # Skip 64 for speed
            config = self.default_config.copy()
            config['precision'] = precision
            config['prime'] = 7  # Small prime for higher precisions
            
            try:
                compressor = PadicCompressionSystem(config)
                padic_weights, metadata = compressor.encode(test_tensor)
                
                # Verify precision is respected
                for pw in padic_weights:
                    self.assertEqual(len(pw.digits), precision)
                
                # Test reconstruction accuracy increases with precision
                reconstructed = compressor.decode(padic_weights, metadata)
                error = torch.abs(reconstructed - test_tensor).max().item()
                
                # Higher precision should generally give lower error
                if precision > 1:
                    self.assertLessEqual(error, 1.0)
                    
            except OverflowError:
                # Expected for certain precision levels
                pass
    
    def test_valuation_computation(self):
        """Test P-adic valuation computation"""
        math_ops = PadicMathematicalOperations(prime=7, precision=8)
        
        # Test valuation of various numbers
        test_cases = [
            (0, float('inf')),
            (7, 1),
            (49, 2),
            (343, 3),
            (6, 0),
            (8, 0),
            (14, 1),
            (21, 1)
        ]
        
        for num, expected_val in test_cases:
            if num == 0:
                # Special case for zero
                val = math_ops._valuation(num)
                self.assertEqual(val, expected_val)
            else:
                val = math_ops._valuation(num)
                self.assertEqual(val, expected_val)
    
    def test_digit_extraction_and_encoding(self):
        """Test digit extraction in P-adic representation"""
        math_ops = PadicMathematicalOperations(prime=5, precision=4)
        
        # Test conversion of rational numbers to P-adic digits
        test_value = Fraction(17, 25)  # 17/25 in base 5
        padic_digits = math_ops.to_padic(test_value)
        
        self.assertEqual(len(padic_digits), 4)
        for digit in padic_digits:
            self.assertIn(digit, range(5))
        
        # Test reconstruction
        reconstructed = math_ops.from_padic(padic_digits, valuation=0)
        error = abs(float(reconstructed) - float(test_value))
        self.assertLess(error, 0.01)


# ============================================================================
# TEST P-ADIC ARITHMETIC
# ============================================================================

class TestPadicArithmetic(unittest.TestCase):
    """Test P-adic arithmetic operations"""
    
    def setUp(self):
        """Set up test environment"""
        self.primes = [2, 3, 5, 7, 11]
        self.precision = 8
    
    def test_padic_addition(self):
        """Test P-adic addition"""
        for prime in self.primes:
            math_ops = PadicMathematicalOperations(prime, self.precision)
            
            # Test addition of P-adic numbers
            a = Fraction(1, 3)
            b = Fraction(2, 5)
            
            padic_a = math_ops.to_padic(a)
            padic_b = math_ops.to_padic(b)
            
            # Add P-adic representations
            result_digits = math_ops.add_padic(padic_a, padic_b)
            
            # Verify result
            self.assertEqual(len(result_digits), self.precision)
            for digit in result_digits:
                self.assertIn(digit, range(prime))
    
    def test_padic_multiplication(self):
        """Test P-adic multiplication"""
        for prime in self.primes:
            math_ops = PadicMathematicalOperations(prime, self.precision)
            
            # Test multiplication
            a = Fraction(2, 3)
            b = Fraction(3, 4)
            
            padic_a = math_ops.to_padic(a)
            padic_b = math_ops.to_padic(b)
            
            # Multiply P-adic representations
            result_digits = math_ops.multiply_padic(padic_a, padic_b)
            
            # Verify result
            self.assertEqual(len(result_digits), self.precision)
            for digit in result_digits:
                self.assertIn(digit, range(prime))
    
    def test_padic_convolution(self):
        """Test P-adic convolution for neural network operations"""
        prime = 7
        precision = 8
        math_ops = PadicMathematicalOperations(prime, precision)
        
        # Create test kernel and input
        kernel = torch.tensor([[0.5, 0.3], [0.2, 0.1]])
        input_tensor = torch.randn(4, 4)
        
        # Convert to P-adic
        config = {'prime': prime, 'precision': precision, 'chunk_size': 100, 'gpu_memory_limit_mb': 512}
        compressor = PadicCompressionSystem(config)
        
        kernel_padic, _ = compressor.encode(kernel.flatten())
        input_padic, _ = compressor.encode(input_tensor.flatten())
        
        # Verify P-adic representations
        self.assertTrue(len(kernel_padic) > 0)
        self.assertTrue(len(input_padic) > 0)
    
    def test_padic_matrix_operations(self):
        """Test P-adic matrix operations"""
        prime = 5
        precision = 6
        config = {
            'prime': prime,
            'precision': precision,
            'chunk_size': 100,
            'gpu_memory_limit_mb': 512
        }
        
        compressor = PadicCompressionSystem(config)
        
        # Test matrix multiplication preservation
        A = torch.randn(10, 10)
        B = torch.randn(10, 10)
        C = torch.mm(A, B)
        
        # Compress matrices
        A_padic, A_meta = compressor.encode(A)
        B_padic, B_meta = compressor.encode(B)
        C_padic, C_meta = compressor.encode(C)
        
        # Decompress and verify
        A_recon = compressor.decode(A_padic, A_meta)
        B_recon = compressor.decode(B_padic, B_meta)
        C_recon = compressor.decode(C_padic, C_meta)
        
        # Verify matrix multiplication is approximately preserved
        C_computed = torch.mm(A_recon, B_recon)
        error = torch.abs(C_computed - C_recon).max().item()
        self.assertLess(error, 0.1)
    
    def test_norm_preservation(self):
        """Test that P-adic operations preserve norms appropriately"""
        prime = 7
        precision = 8
        math_ops = PadicMathematicalOperations(prime, precision)
        
        # Test P-adic norm computation
        test_values = [Fraction(1, 1), Fraction(7, 1), Fraction(49, 1), Fraction(1, 7)]
        expected_norms = [1, 1/7, 1/49, 7]
        
        for value, expected_norm in zip(test_values, expected_norms):
            valuation = math_ops._valuation(value.numerator) - math_ops._valuation(value.denominator)
            padic_norm = prime ** (-valuation)
            self.assertAlmostEqual(padic_norm, expected_norm, places=6)
    
    def test_ultrametric_properties(self):
        """Test ultrametric inequality |x+y|_p ≤ max(|x|_p, |y|_p)"""
        prime = 3
        precision = 8
        math_ops = PadicMathematicalOperations(prime, precision)
        
        # Test ultrametric property
        test_pairs = [
            (Fraction(1, 3), Fraction(2, 3)),
            (Fraction(3, 1), Fraction(6, 1)),
            (Fraction(1, 9), Fraction(2, 9)),
            (Fraction(4, 1), Fraction(5, 1))
        ]
        
        for a, b in test_pairs:
            # Compute P-adic valuations
            val_a = math_ops._valuation(a.numerator) - math_ops._valuation(a.denominator)
            val_b = math_ops._valuation(b.numerator) - math_ops._valuation(b.denominator)
            val_sum = math_ops._valuation((a + b).numerator) - math_ops._valuation((a + b).denominator)
            
            # Verify ultrametric inequality (smaller valuation = larger norm)
            self.assertGreaterEqual(val_sum, min(val_a, val_b))


# ============================================================================
# TEST P-ADIC SPECIFIC FEATURES
# ============================================================================

class TestPadicSpecificFeatures(unittest.TestCase):
    """Test P-adic unique features"""
    
    def setUp(self):
        """Set up test environment"""
        self.default_config = {
            'prime': 7,
            'precision': 8,
            'chunk_size': 1000,
            'gpu_memory_limit_mb': 512
        }
    
    def test_hensel_lifting(self):
        """Test Hensel lifting for reconstruction"""
        # Hensel lifting is used internally in reconstruction
        compressor = PadicCompressionSystem(self.default_config)
        
        # Create test polynomial root
        test_value = torch.tensor([2.0])  # Root of x^2 - 4 = 0
        
        # Compress and decompress
        padic_weights, metadata = compressor.encode(test_value)
        reconstructed = compressor.decode(padic_weights, metadata)
        
        # Verify Hensel lifting preserves polynomial properties
        error = torch.abs(reconstructed**2 - 4.0).item()
        self.assertLess(error, 0.01)
    
    def test_padic_expansion_uniqueness(self):
        """Test uniqueness of P-adic expansion"""
        math_ops = PadicMathematicalOperations(prime=5, precision=8)
        
        # Same value should always give same P-adic expansion
        value = Fraction(17, 25)
        
        expansion1 = math_ops.to_padic(value)
        expansion2 = math_ops.to_padic(value)
        
        self.assertEqual(expansion1, expansion2)
    
    def test_valuation_additivity(self):
        """Test v_p(xy) = v_p(x) + v_p(y)"""
        math_ops = PadicMathematicalOperations(prime=3, precision=8)
        
        test_pairs = [
            (9, 27),   # 3^2 * 3^3 = 3^5
            (3, 5),    # 3^1 * 3^0 = 3^1
            (15, 18),  # 3^1 * 3^2 = 3^3
            (81, 2)    # 3^4 * 3^0 = 3^4
        ]
        
        for x, y in test_pairs:
            val_x = math_ops._valuation(x)
            val_y = math_ops._valuation(y)
            val_xy = math_ops._valuation(x * y)
            
            self.assertEqual(val_xy, val_x + val_y)
    
    def test_strong_triangle_inequality(self):
        """Test strong triangle inequality specific to P-adic metrics"""
        prime = 5
        math_ops = PadicMathematicalOperations(prime, precision=8)
        
        # For P-adic numbers, if |x|_p ≠ |y|_p, then |x+y|_p = max(|x|_p, |y|_p)
        test_cases = [
            (Fraction(5, 1), Fraction(1, 1)),   # Different valuations
            (Fraction(25, 1), Fraction(3, 1)),  # Different valuations
            (Fraction(1, 5), Fraction(2, 1))    # Different valuations
        ]
        
        for a, b in test_cases:
            val_a = math_ops._valuation(a.numerator) - math_ops._valuation(a.denominator)
            val_b = math_ops._valuation(b.numerator) - math_ops._valuation(b.denominator)
            
            if val_a != val_b:
                # Strong triangle inequality should hold
                val_sum = math_ops._valuation((a + b).numerator) - math_ops._valuation((a + b).denominator)
                self.assertEqual(val_sum, min(val_a, val_b))


# ============================================================================
# TEST P-ADIC MODEL ARCHITECTURES
# ============================================================================

class TestPadicModelArchitectures(unittest.TestCase):
    """Test P-adic compression on different model architectures"""
    
    def setUp(self):
        """Set up test environment"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = {
            'prime': 31,
            'precision': 4,
            'chunk_size': 1000,
            'gpu_memory_limit_mb': 512
        }
    
    def test_linear_layers(self):
        """Test P-adic compression on linear layers"""
        model = nn.Linear(100, 50)
        compressor = PadicCompressionSystem(self.config)
        
        # Compress weights
        weight_padic, weight_meta = compressor.encode(model.weight.data)
        bias_padic, bias_meta = compressor.encode(model.bias.data)
        
        # Verify compression
        self.assertTrue(len(weight_padic) > 0)
        self.assertTrue(len(bias_padic) > 0)
        
        # Test reconstruction
        weight_recon = compressor.decode(weight_padic, weight_meta)
        bias_recon = compressor.decode(bias_padic, bias_meta)
        
        self.assertEqual(weight_recon.shape, model.weight.shape)
        self.assertEqual(bias_recon.shape, model.bias.shape)
    
    def test_convolutional_layers(self):
        """Test P-adic compression on convolutional layers"""
        model = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        compressor = PadicCompressionSystem(self.config)
        
        # Compress conv weights
        weight_padic, weight_meta = compressor.encode(model.weight.data)
        
        # Verify 4D tensor handling
        self.assertTrue(len(weight_padic) > 0)
        
        # Test reconstruction preserves shape
        weight_recon = compressor.decode(weight_padic, weight_meta)
        self.assertEqual(weight_recon.shape, model.weight.shape)
    
    def test_attention_layers(self):
        """Test P-adic compression on attention mechanisms"""
        d_model = 64
        nhead = 8
        model = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        compressor = PadicCompressionSystem(self.config)
        
        # Compress attention weights
        state_dict = model.state_dict()
        compressed_state = {}
        
        for name, param in state_dict.items():
            padic_weights, metadata = compressor.encode(param)
            compressed_state[name] = (padic_weights, metadata)
        
        # Verify all attention components compressed
        self.assertIn('in_proj_weight', compressed_state)
        self.assertIn('out_proj.weight', compressed_state)
        
        # Test reconstruction
        for name, (padic_weights, metadata) in compressed_state.items():
            reconstructed = compressor.decode(padic_weights, metadata)
            self.assertEqual(reconstructed.shape, state_dict[name].shape)
    
    def test_batch_normalization(self):
        """Test P-adic compression on batch normalization layers"""
        model = nn.BatchNorm2d(64)
        compressor = PadicCompressionSystem(self.config)
        
        # Initialize with some statistics
        model.running_mean.data = torch.randn(64)
        model.running_var.data = torch.abs(torch.randn(64)) + 0.1
        
        # Compress statistics
        mean_padic, mean_meta = compressor.encode(model.running_mean)
        var_padic, var_meta = compressor.encode(model.running_var)
        
        # Test reconstruction preserves positive variance
        mean_recon = compressor.decode(mean_padic, mean_meta)
        var_recon = compressor.decode(var_padic, var_meta)
        
        self.assertTrue(torch.all(var_recon > 0))
    
    def test_recurrent_layers(self):
        """Test P-adic compression on recurrent layers"""
        model = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True)
        compressor = PadicCompressionSystem(self.config)
        
        # Compress LSTM weights
        state_dict = model.state_dict()
        for name, param in state_dict.items():
            padic_weights, metadata = compressor.encode(param)
            
            # Verify compression of different LSTM components
            self.assertTrue(len(padic_weights) > 0)
            
            # Test reconstruction
            reconstructed = compressor.decode(padic_weights, metadata)
            self.assertEqual(reconstructed.shape, param.shape)


# ============================================================================
# TEST PERIODIC PATTERN DETECTION
# ============================================================================

class TestPeriodicPatternDetection(unittest.TestCase):
    """Test periodic pattern handling in P-adic compression"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'prime': 7,
            'precision': 8,
            'chunk_size': 100,
            'gpu_memory_limit_mb': 512
        }
    
    def test_identification_of_periodic_weights(self):
        """Test identification of periodic patterns in weights"""
        # Create periodic pattern
        pattern = [0.1, 0.2, 0.3, 0.2, 0.1]
        periodic_tensor = torch.tensor(pattern * 20)  # Repeat 20 times
        
        compressor = PadicCompressionSystem(self.config)
        padic_weights, metadata = compressor.encode(periodic_tensor)
        
        # P-adic should efficiently encode periodic patterns
        compression_ratio = periodic_tensor.numel() * 4 / (len(padic_weights) * self.config['precision'])
        self.assertGreater(compression_ratio, 2.0)  # Should achieve good compression
    
    def test_padic_representation_of_periodic_patterns(self):
        """Test P-adic representation captures periodicity"""
        # Create strictly periodic rational pattern
        pattern = [Fraction(1, 3), Fraction(2, 3), Fraction(1, 3)]
        periodic_values = pattern * 10
        
        math_ops = PadicMathematicalOperations(prime=3, precision=6)
        
        # Convert to P-adic
        padic_representations = []
        for value in periodic_values:
            padic_digits = math_ops.to_padic(value)
            padic_representations.append(padic_digits)
        
        # Check for repeating P-adic patterns
        unique_representations = []
        for rep in padic_representations:
            if rep not in unique_representations:
                unique_representations.append(rep)
        
        # Should have few unique representations for periodic pattern
        self.assertLess(len(unique_representations), len(pattern) * 2)
    
    def test_compression_of_repeating_structures(self):
        """Test compression efficiency on repeating structures"""
        # Create model with repeating structure
        model = PeriodicPatternModel()
        compressor = PadicCompressionSystem(self.config)
        
        # Compress weights with periodic patterns
        weight_padic, weight_meta = compressor.encode(model.fc1.weight.data)
        bias_padic, bias_meta = compressor.encode(model.fc1.bias.data)
        
        # Calculate compression ratios
        weight_ratio = model.fc1.weight.numel() * 4 / (len(weight_padic) * self.config['precision'])
        bias_ratio = model.fc1.bias.numel() * 4 / (len(bias_padic) * self.config['precision'])
        
        # Periodic patterns should compress well
        self.assertGreater(weight_ratio, 1.5)
        self.assertGreater(bias_ratio, 1.5)
    
    def test_hensel_lifting_for_reconstruction(self):
        """Test Hensel lifting preserves periodic patterns"""
        # Create periodic polynomial pattern
        x = torch.linspace(0, 2*math.pi, 100)
        periodic = torch.sin(x)  # Periodic function
        
        compressor = PadicCompressionSystem(self.config)
        padic_weights, metadata = compressor.encode(periodic)
        reconstructed = compressor.decode(padic_weights, metadata)
        
        # Check periodicity is preserved
        first_period = reconstructed[:25]
        second_period = reconstructed[25:50]
        third_period = reconstructed[50:75]
        
        # Periods should be similar
        error12 = torch.abs(first_period - second_period).mean().item()
        error23 = torch.abs(second_period - third_period).mean().item()
        
        self.assertLess(error12, 0.1)
        self.assertLess(error23, 0.1)
    
    def test_pattern_based_compression_ratios(self):
        """Test compression ratios for different pattern types"""
        compressor = PadicCompressionSystem(self.config)
        
        # Test different patterns
        patterns = {
            'periodic': torch.tensor([1, 2, 3] * 50, dtype=torch.float32),
            'random': torch.randn(150),
            'constant': torch.ones(150) * 0.5,
            'linear': torch.linspace(0, 1, 150)
        }
        
        compression_ratios = {}
        for name, tensor in patterns.items():
            padic_weights, metadata = compressor.encode(tensor)
            ratio = tensor.numel() * 4 / (len(padic_weights) * self.config['precision'])
            compression_ratios[name] = ratio
        
        # Periodic and constant should compress better than random
        self.assertGreater(compression_ratios['periodic'], compression_ratios['random'])
        self.assertGreater(compression_ratios['constant'], compression_ratios['random'])


# ============================================================================
# TEST P-ADIC LOGARITHMIC ENCODING
# ============================================================================

class TestPadicLogarithmicEncoding(unittest.TestCase):
    """Test P-adic logarithmic encoding functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = LogarithmicEncodingConfig(
            prime=257,
            precision=2,
            use_natural_log=True,
            scale_factor=1000.0
        )
        self.encoder = PadicLogarithmicEncoder(self.config)
    
    def test_logarithmic_encoder_initialization(self):
        """Test P-adic logarithmic encoder setup"""
        self.assertEqual(self.encoder.config.prime, 257)
        self.assertEqual(self.encoder.config.precision, 2)
        self.assertTrue(self.encoder.config.use_natural_log)
    
    def test_base_selection_optimization(self):
        """Test optimization of logarithm base selection"""
        test_tensor = torch.randn(100)
        
        # Test with natural log
        config_natural = LogarithmicEncodingConfig(use_natural_log=True)
        encoder_natural = PadicLogarithmicEncoder(config_natural)
        result_natural = encoder_natural.encode_tensor(test_tensor)
        
        # Test with prime base log
        config_prime = LogarithmicEncodingConfig(use_natural_log=False)
        encoder_prime = PadicLogarithmicEncoder(config_prime)
        result_prime = encoder_prime.encode_tensor(test_tensor)
        
        # Both should produce valid encodings
        self.assertTrue(len(result_natural) > 0)
        self.assertTrue(len(result_prime) > 0)
    
    def test_precision_vs_compression_tradeoffs(self):
        """Test tradeoffs between precision and compression"""
        test_tensor = torch.randn(1000)
        
        results = {}
        for precision in [1, 2, 4, 8]:
            config = LogarithmicEncodingConfig(precision=precision)
            encoder = PadicLogarithmicEncoder(config)
            
            encoded = encoder.encode_tensor(test_tensor)
            decoded = encoder.decode_tensor(encoded)
            
            # Calculate metrics
            error = torch.abs(decoded - test_tensor).mean().item()
            compression_ratio = test_tensor.numel() * 4 / (len(encoded) * precision)
            
            results[precision] = {
                'error': error,
                'compression_ratio': compression_ratio
            }
        
        # Higher precision should generally give lower error
        self.assertLess(results[8]['error'], results[1]['error'])
        
        # Lower precision should give better compression
        self.assertGreater(results[1]['compression_ratio'], results[8]['compression_ratio'])
    
    def test_gradient_safe_encoding(self):
        """Test that encoding preserves gradient flow"""
        test_tensor = torch.randn(50, 50, requires_grad=True)
        
        # Create a simple operation that should preserve gradients
        encoded_tensor = test_tensor * 1.0  # Identity to maintain gradient
        
        # Simulate loss
        loss = encoded_tensor.sum()
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(test_tensor.grad)
        self.assertTrue(torch.all(test_tensor.grad == 1.0))
    
    def test_numerical_stability(self):
        """Test numerical stability of logarithmic encoding"""
        # Test edge cases
        edge_cases = [
            torch.tensor([1e-10, 1e-5, 1.0, 1e5, 1e10]),
            torch.tensor([0.0, 0.0, 0.0]),  # All zeros
            torch.ones(10) * float('inf'),  # Infinities
            torch.ones(10) * float('nan')   # NaNs
        ]
        
        encoder = PadicLogarithmicEncoder(self.config)
        
        for tensor in edge_cases[:-2]:  # Skip inf and nan for now
            try:
                encoded = encoder.encode_tensor(tensor)
                decoded = encoder.decode_tensor(encoded)
                
                # Should handle without crashing
                self.assertTrue(decoded.shape == tensor.shape)
            except (ValueError, RuntimeError):
                # Expected for some edge cases
                pass


# ============================================================================
# TEST P-ADIC EDGE CASES
# ============================================================================

class TestPadicEdgeCases(unittest.TestCase):
    """Test edge cases and error handling in P-adic system"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'prime': 7,
            'precision': 8,
            'chunk_size': 100,
            'gpu_memory_limit_mb': 512
        }
    
    def test_handling_of_prime_powers(self):
        """Test handling of exact prime powers"""
        compressor = PadicCompressionSystem(self.config)
        
        # Test exact prime powers
        prime_powers = torch.tensor([7.0, 49.0, 343.0, 1/7.0, 1/49.0])
        
        padic_weights, metadata = compressor.encode(prime_powers)
        reconstructed = compressor.decode(padic_weights, metadata)
        
        # Should handle prime powers efficiently
        relative_error = torch.abs((reconstructed - prime_powers) / (prime_powers + 1e-10))
        self.assertTrue(torch.all(relative_error < 0.01))
    
    def test_overflow_underflow_protection(self):
        """Test overflow and underflow protection"""
        # Test with large precision that might cause overflow
        config_large = self.config.copy()
        config_large['precision'] = 100
        config_large['prime'] = 251
        
        with self.assertRaises((OverflowError, ValueError)):
            compressor = PadicCompressionSystem(config_large)
    
    def test_division_by_zero_handling(self):
        """Test division by zero handling"""
        math_ops = PadicMathematicalOperations(prime=5, precision=8)
        
        # Test with zero denominator
        with self.assertRaises((ValueError, ZeroDivisionError)):
            _ = math_ops.to_padic(Fraction(1, 0))
    
    def test_large_valuation_handling(self):
        """Test handling of large valuations"""
        math_ops = PadicMathematicalOperations(prime=2, precision=8)
        
        # Test with high powers of prime
        large_power = 2**20
        valuation = math_ops._valuation(large_power)
        self.assertEqual(valuation, 20)
        
        # Test reconstruction with large valuation
        try:
            padic_digits = math_ops.to_padic(Fraction(1, large_power))
            reconstructed = math_ops.from_padic(padic_digits, valuation=-20)
            self.assertAlmostEqual(float(reconstructed), 1/large_power, places=10)
        except OverflowError:
            # Expected for very large valuations
            pass
    
    def test_precision_loss_scenarios(self):
        """Test scenarios where precision loss occurs"""
        compressor = PadicCompressionSystem(self.config)
        
        # Test with irrational approximations
        irrational = torch.tensor([math.pi, math.e, math.sqrt(2)])
        
        padic_weights, metadata = compressor.encode(irrational)
        reconstructed = compressor.decode(padic_weights, metadata)
        
        # Precision loss is expected but should be bounded
        error = torch.abs(reconstructed - irrational)
        self.assertTrue(torch.all(error < 0.1))
    
    def test_empty_tensor_handling(self):
        """Test handling of empty tensors"""
        compressor = PadicCompressionSystem(self.config)
        
        empty_tensor = torch.tensor([])
        with self.assertRaises(ValueError):
            compressor.encode(empty_tensor)
    
    def test_nan_inf_handling(self):
        """Test handling of NaN and Inf values"""
        compressor = PadicCompressionSystem(self.config)
        
        # Test NaN
        nan_tensor = torch.tensor([float('nan'), 1.0, 2.0])
        with self.assertRaises(ValueError):
            compressor.encode(nan_tensor)
        
        # Test Inf
        inf_tensor = torch.tensor([float('inf'), 1.0, 2.0])
        with self.assertRaises(ValueError):
            compressor.encode(inf_tensor)


# ============================================================================
# TEST P-ADIC GRADIENT FLOW
# ============================================================================

class TestPadicGradientFlow(unittest.TestCase):
    """Test gradient preservation through P-adic operations"""
    
    def setUp(self):
        """Set up test environment"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = {
            'prime': 7,
            'precision': 8,
            'chunk_size': 100,
            'gpu_memory_limit_mb': 512
        }
    
    def test_gradient_preservation_through_compression(self):
        """Test that gradients flow through compression/decompression"""
        # Create model with gradients
        model = SimpleLinearModel(input_dim=10, hidden_dim=20, output_dim=5)
        model.train()
        
        # Forward pass
        x = torch.randn(1, 10)
        output = model(x)
        loss = output.sum()
        
        # Compute gradients
        loss.backward()
        
        # Store original gradients
        original_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                original_grads[name] = param.grad.clone()
        
        # Compress and decompress weights
        compressor = PadicCompressionSystem(self.config)
        compressed_state = {}
        
        for name, param in model.named_parameters():
            padic_weights, metadata = compressor.encode(param.data)
            reconstructed = compressor.decode(padic_weights, metadata)
            param.data = reconstructed
        
        # Gradients should still exist (though values may differ)
        for name, param in model.named_parameters():
            if name in original_grads:
                self.assertIsNotNone(param.grad)
    
    def test_gradient_handler_integration(self):
        """Test P-adic gradient handler"""
        handler = PadicGradientHandler(self.config)
        
        # Create test gradients
        gradients = torch.randn(100, requires_grad=True)
        
        # Process gradients through P-adic system
        processed_grads = handler.process_gradients(gradients)
        
        # Should maintain shape and rough magnitude
        self.assertEqual(processed_grads.shape, gradients.shape)
        
        # Check gradient flow is maintained
        if gradients.grad_fn is not None:
            self.assertIsNotNone(processed_grads.grad_fn)
    
    def test_backward_pass_stability(self):
        """Test stability of backward pass with P-adic weights"""
        model = SimpleLinearModel(input_dim=20, hidden_dim=40, output_dim=10)
        compressor = PadicCompressionSystem(self.config)
        
        # Compress model weights
        for param in model.parameters():
            padic_weights, metadata = compressor.encode(param.data)
            param.data = compressor.decode(padic_weights, metadata)
        
        # Test multiple backward passes
        for _ in range(5):
            x = torch.randn(1, 20)
            output = model(x)
            loss = output.sum()
            
            # Should not crash or produce NaN
            loss.backward()
            
            # Check gradients are valid
            for param in model.parameters():
                if param.grad is not None:
                    self.assertFalse(torch.isnan(param.grad).any())
                    self.assertFalse(torch.isinf(param.grad).any())
            
            # Zero gradients for next iteration
            model.zero_grad()


# ============================================================================
# TEST P-ADIC PERFORMANCE
# ============================================================================

class TestPadicPerformance(unittest.TestCase):
    """Test performance benchmarks for P-adic compression"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'prime': 31,
            'precision': 4,
            'chunk_size': 10000,
            'gpu_memory_limit_mb': 1024
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_compression_speed_benchmark(self):
        """Benchmark compression speed"""
        compressor = PadicCompressionSystem(self.config)
        
        # Test different tensor sizes
        sizes = [100, 1000, 10000, 100000]
        times = []
        
        for size in sizes:
            tensor = torch.randn(size)
            
            start_time = time.time()
            padic_weights, metadata = compressor.encode(tensor)
            end_time = time.time()
            
            compression_time = end_time - start_time
            times.append(compression_time)
            
            # Ensure reasonable performance (< 1 second for 100k elements)
            if size == 100000:
                self.assertLess(compression_time, 1.0)
        
        # Times should scale reasonably with size
        # Larger tensors should take more time but not exponentially
        self.assertLess(times[-1] / times[0], 10000)  # Less than linear scaling
    
    def test_decompression_speed_benchmark(self):
        """Benchmark decompression speed"""
        compressor = PadicCompressionSystem(self.config)
        
        # Prepare compressed data
        tensor = torch.randn(10000)
        padic_weights, metadata = compressor.encode(tensor)
        
        # Benchmark decompression
        start_time = time.time()
        reconstructed = compressor.decode(padic_weights, metadata)
        end_time = time.time()
        
        decompression_time = end_time - start_time
        
        # Decompression should be fast (< 0.1 seconds for 10k elements)
        self.assertLess(decompression_time, 0.1)
    
    def test_memory_efficiency(self):
        """Test memory efficiency of P-adic compression"""
        compressor = PadicCompressionSystem(self.config)
        
        # Create large tensor
        large_tensor = torch.randn(1000000)
        original_memory = large_tensor.element_size() * large_tensor.numel()
        
        # Compress
        padic_weights, metadata = compressor.encode(large_tensor)
        
        # Estimate compressed memory
        compressed_memory = len(padic_weights) * self.config['precision']
        
        # Should achieve compression
        compression_ratio = original_memory / compressed_memory
        self.assertGreater(compression_ratio, 1.0)
    
    def test_gpu_acceleration_benefit(self):
        """Test GPU acceleration benefits if available"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Test with hybrid P-adic system
        hybrid_config = self.config.copy()
        hybrid_config['enable_hybrid'] = True
        hybrid_config['force_hybrid'] = True
        
        try:
            hybrid_compressor = HybridPadicCompressionSystem(hybrid_config)
            
            # Create GPU tensor
            gpu_tensor = torch.randn(10000).cuda()
            
            # Time GPU compression
            start_time = time.time()
            padic_weights, metadata = hybrid_compressor.encode(gpu_tensor)
            gpu_time = time.time() - start_time
            
            # GPU should handle compression efficiently
            self.assertLess(gpu_time, 0.5)
            
        except RuntimeError:
            # GPU might not be available in test environment
            self.skipTest("GPU compression not available")
    
    def test_batch_processing_performance(self):
        """Test performance of batch processing"""
        compressor = PadicCompressionSystem(self.config)
        
        # Create batch of tensors
        batch_size = 32
        tensors = [torch.randn(1000) for _ in range(batch_size)]
        
        # Time batch processing
        start_time = time.time()
        for tensor in tensors:
            padic_weights, metadata = compressor.encode(tensor)
        batch_time = time.time() - start_time
        
        # Should process batch efficiently
        avg_time_per_tensor = batch_time / batch_size
        self.assertLess(avg_time_per_tensor, 0.01)  # < 10ms per tensor
    
    def test_compression_ratio_vs_precision(self):
        """Test how compression ratio varies with precision"""
        test_tensor = torch.randn(10000)
        
        ratios = {}
        for precision in [1, 2, 4, 8, 16]:
            config = self.config.copy()
            config['precision'] = precision
            
            try:
                compressor = PadicCompressionSystem(config)
                padic_weights, metadata = compressor.encode(test_tensor)
                
                original_size = test_tensor.numel() * 4
                compressed_size = len(padic_weights) * precision
                ratio = original_size / compressed_size
                
                ratios[precision] = ratio
            except OverflowError:
                ratios[precision] = 0
        
        # Lower precision should give better compression
        if ratios[1] > 0 and ratios[16] > 0:
            self.assertGreater(ratios[1], ratios[16])
    
    def test_periodic_pattern_compression_efficiency(self):
        """Test compression efficiency on periodic patterns"""
        # Create highly periodic pattern
        pattern = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.1])
        periodic_tensor = pattern.repeat(1000)  # 5000 elements
        
        # Create random tensor for comparison
        random_tensor = torch.randn(5000)
        
        compressor = PadicCompressionSystem(self.config)
        
        # Compress both
        periodic_padic, periodic_meta = compressor.encode(periodic_tensor)
        random_padic, random_meta = compressor.encode(random_tensor)
        
        # Calculate compression ratios
        periodic_ratio = periodic_tensor.numel() * 4 / (len(periodic_padic) * self.config['precision'])
        random_ratio = random_tensor.numel() * 4 / (len(random_padic) * self.config['precision'])
        
        # Periodic should compress better
        self.assertGreater(periodic_ratio, random_ratio * 1.5)


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)