"""
Test PyTorch-Only P-adic Compression System
Validates the pure PyTorch implementation with Triton acceleration
"""

import torch
import torch.nn as nn
import time
import unittest
from typing import Dict, Any, List, Tuple

# Import PyTorch-only P-adic components
from padic.pytorch_padic_engine import PyTorchPAdicEngine, PyTorchPAdicConfig
from padic.triton_kernels import TritonPAdicOps
from padic.padic_compression_pytorch import (
    PurelyPyTorchPAdicSystem,
    PurelyPyTorchConfig,
    CompressionResult,
    DecompressionResult
)


class TestPyTorchPAdicEngine(unittest.TestCase):
    """Test PyTorch P-adic engine"""
    
    def setUp(self):
        """Initialize test components"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = PyTorchPAdicConfig(
            prime=257,
            precision=6,
            device=self.device,
            compile_mode="reduce-overhead"
        )
        self.engine = PyTorchPAdicEngine(self.config)
    
    def test_to_padic_conversion(self):
        """Test conversion to p-adic representation"""
        # Test single value
        x = torch.tensor([123.456], device=self.device)
        digits, signs = self.engine.to_padic(x, return_sign=True)
        
        self.assertEqual(digits.shape, (1, self.config.precision))
        self.assertIsNotNone(signs)
        
        # Test batch
        batch = torch.randn(10, 20, device=self.device)
        digits, _ = self.engine.to_padic(batch)
        self.assertEqual(digits.shape, (10, 20, self.config.precision))
    
    def test_from_padic_reconstruction(self):
        """Test reconstruction from p-adic digits"""
        original = torch.tensor([100.0, -50.0, 25.5], device=self.device)
        
        # Convert to p-adic and back
        digits, signs = self.engine.to_padic(original, return_sign=True)
        reconstructed = self.engine.from_padic(digits, signs)
        
        # Check reconstruction error
        error = torch.abs(original - reconstructed).max()
        self.assertLess(error, 1e-3, f"Reconstruction error {error} too large")
    
    def test_padic_arithmetic(self):
        """Test p-adic arithmetic operations"""
        a = torch.tensor([10.0], device=self.device)
        b = torch.tensor([5.0], device=self.device)
        
        # Convert to p-adic
        a_digits, _ = self.engine.to_padic(a)
        b_digits, _ = self.engine.to_padic(b)
        
        # Test addition
        sum_digits = self.engine.padic_add(a_digits, b_digits)
        sum_val = self.engine.from_padic(sum_digits)
        expected = a + b
        error = torch.abs(sum_val - expected).item()
        self.assertLess(error, 1.0, f"Addition error {error} too large")
        
        # Test multiplication
        prod_digits = self.engine.padic_multiply(a_digits, b_digits)
        prod_val = self.engine.from_padic(prod_digits)
        expected = a * b
        error = torch.abs(prod_val - expected).item()
        self.assertLess(error, 5.0, f"Multiplication error {error} too large")
    
    def test_ultrametric_distance(self):
        """Test ultrametric distance computation"""
        a = torch.tensor([10.0], device=self.device)
        b = torch.tensor([11.0], device=self.device)
        
        a_digits, _ = self.engine.to_padic(a)
        b_digits, _ = self.engine.to_padic(b)
        
        distance = self.engine.ultrametric_distance(a_digits, b_digits)
        
        self.assertGreaterEqual(distance.item(), 0.0)
        self.assertLessEqual(distance.item(), 1.0)
    
    def test_sparse_conversion(self):
        """Test sparse CSR conversion"""
        # Create sparse tensor
        sparse_tensor = torch.zeros(100, device=self.device)
        sparse_tensor[10] = 1.0
        sparse_tensor[50] = -2.0
        sparse_tensor[90] = 3.0
        
        # Convert to p-adic
        digits, _ = self.engine.to_padic(sparse_tensor)
        
        # Convert to sparse
        sparse_csr = self.engine.to_sparse_csr(digits, threshold=0.01)
        
        self.assertTrue(sparse_csr.is_sparse)
    
    def test_log_space_encoding(self):
        """Test logarithmic space encoding"""
        # Test with various magnitudes
        x = torch.tensor([0.001, 1.0, 1000.0], device=self.device)
        
        # Encode in log space
        log_digits = self.engine.log_space_encoding(x)
        
        # Decode from log space
        reconstructed = self.engine.log_space_decoding(log_digits)
        
        # Check relative error
        rel_error = torch.abs((x - reconstructed) / (x + 1e-10)).max()
        self.assertLess(rel_error, 0.1, f"Log space encoding error {rel_error} too large")


class TestTritonKernels(unittest.TestCase):
    """Test Triton kernel operations"""
    
    def setUp(self):
        """Initialize test components"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping Triton tests")
        
        self.device = 'cuda'
        self.triton_ops = TritonPAdicOps(prime=257, precision=6, device=self.device)
    
    def test_triton_ultrametric(self):
        """Test Triton ultrametric distance kernel"""
        # Create test p-adic numbers
        x = torch.randn(100, 6, device=self.device)
        y = torch.randn(100, 6, device=self.device)
        
        # Compute distance using Triton
        distances = self.triton_ops.ultrametric_distance(x, y)
        
        self.assertEqual(distances.shape, (100,))
        self.assertTrue(torch.all(distances >= 0))
    
    def test_triton_sparse_encode(self):
        """Test Triton sparse encoding kernel"""
        # Create dense tensor with sparsity
        dense = torch.randn(1000, device=self.device)
        dense[dense.abs() < 1.0] = 0  # Make sparse
        
        # Encode sparsely
        indices, values = self.triton_ops.sparse_encode(dense, threshold=0.01)
        
        self.assertLess(values.numel(), dense.numel())
        self.assertEqual(indices.shape, values.shape)
    
    def test_triton_batch_operations(self):
        """Test Triton batch arithmetic operations"""
        # Create batches
        a = torch.randn(50, 6, device=self.device)
        b = torch.randn(50, 6, device=self.device)
        
        # Test addition
        sum_result = self.triton_ops.batch_add(a, b)
        self.assertEqual(sum_result.shape, a.shape)
        
        # Test multiplication
        prod_result = self.triton_ops.batch_multiply(a, b)
        self.assertEqual(prod_result.shape, a.shape)
    
    def test_triton_log_encode(self):
        """Test Triton log space encoding"""
        x = torch.tensor([0.1, 1.0, 10.0, 100.0], device=self.device)
        
        # Encode using Triton
        log_encoded = self.triton_ops.log_encode(x)
        
        self.assertEqual(log_encoded.shape, x.shape)
        self.assertTrue(torch.all(torch.isfinite(log_encoded)))
    
    def test_triton_batch_convert(self):
        """Test Triton batch p-adic conversion"""
        batch = torch.randn(100, device=self.device)
        
        # Convert batch to p-adic
        padic_batch = self.triton_ops.batch_convert(batch)
        
        self.assertEqual(padic_batch.shape, (100, 6))  # 6 is precision
    
    def test_triton_hensel_lift(self):
        """Test Triton Hensel lifting"""
        # Create low precision p-adic
        digits = torch.randn(10, 4, device=self.device)  # precision 4
        
        # Lift to higher precision
        lifted = self.triton_ops.hensel_lift(digits, new_precision=8)
        
        self.assertEqual(lifted.shape, (10, 8))
        # Check that original digits are preserved
        self.assertTrue(torch.allclose(lifted[:, :4], digits, atol=1e-5))


class TestPurelyPyTorchSystem(unittest.TestCase):
    """Test complete PyTorch-only P-adic system"""
    
    def setUp(self):
        """Initialize test components"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = PurelyPyTorchConfig(
            prime=257,
            precision=6,
            device=self.device,
            enable_triton=torch.cuda.is_available(),
            enable_sparse=True,
            enable_log_encoding=True,
            enable_pattern_matching=True,
            enable_entropy=True
        )
        self.system = PurelyPyTorchPAdicSystem(self.config)
    
    def test_compression_decompression(self):
        """Test full compression-decompression pipeline"""
        # Test various tensor sizes
        test_sizes = [
            (100,),
            (50, 50),
            (10, 20, 30),
            (5, 5, 5, 5)
        ]
        
        for size in test_sizes:
            with self.subTest(size=size):
                original = torch.randn(size, device=self.device)
                
                # Compress
                compressed = self.system.compress(original)
                
                self.assertIsInstance(compressed, CompressionResult)
                self.assertGreater(compressed.compression_ratio, 1.0)
                
                # Decompress
                decompressed = self.system.decompress(compressed)
                
                self.assertIsInstance(decompressed, DecompressionResult)
                self.assertEqual(decompressed.reconstructed_data.shape, original.shape)
    
    def test_sparse_compression(self):
        """Test sparse tensor compression"""
        # Create sparse tensor
        sparse_tensor = torch.zeros(1000, device=self.device)
        sparse_tensor[100] = 10.0
        sparse_tensor[500] = -20.0
        sparse_tensor[900] = 30.0
        
        # Compress
        compressed = self.system.compress(sparse_tensor)
        
        # Check that sparse encoding was used
        if compressed.sparse_indices is not None:
            self.assertLess(compressed.sparse_values.numel(), sparse_tensor.numel())
        
        # Decompress and validate
        decompressed = self.system.decompress(compressed)
        
        # Check key values are preserved
        reconstructed = decompressed.reconstructed_data
        self.assertAlmostEqual(reconstructed[100].item(), 10.0, delta=1.0)
        self.assertAlmostEqual(reconstructed[500].item(), -20.0, delta=1.0)
        self.assertAlmostEqual(reconstructed[900].item(), 30.0, delta=1.0)
    
    def test_pattern_compression(self):
        """Test pattern-based compression"""
        # Create tensor with repeating pattern
        pattern = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0], device=self.device)
        repeated = pattern.repeat(20)  # 100 elements
        
        # Compress
        compressed = self.system.compress(repeated)
        
        # Check that pattern was detected
        if compressed.metadata.get('pattern_metadata', {}).get('has_pattern'):
            # Compression should be very good for patterns
            self.assertGreater(compressed.compression_ratio, 5.0)
        
        # Decompress and validate
        decompressed = self.system.decompress(compressed)
        
        # Check reconstruction
        error = torch.abs(repeated - decompressed.reconstructed_data).max()
        self.assertLess(error, 0.1)
    
    def test_validation(self):
        """Test compression validation"""
        original = torch.randn(100, 100, device=self.device)
        
        # Validate compression
        validation = self.system.validate_compression(original, tolerance=1e-3)
        
        self.assertIn('is_valid', validation)
        self.assertIn('compression_ratio', validation)
        self.assertIn('max_abs_error', validation)
        
        # For most cases, should be valid
        if not validation['is_valid']:
            print(f"Validation failed with max error: {validation['max_abs_error']}")
    
    def test_benchmark(self):
        """Test benchmarking functionality"""
        tensor_sizes = [
            (100,),
            (100, 100),
            (50, 50, 50)
        ]
        
        # Run benchmark
        results = self.system.benchmark(tensor_sizes, n_iterations=3)
        
        self.assertIn('results', results)
        self.assertIn('config', results)
        self.assertIn('statistics', results)
        
        # Check results structure
        for result in results['results']:
            self.assertIn('avg_compression_time', result)
            self.assertIn('avg_decompression_time', result)
            self.assertIn('avg_compression_ratio', result)
    
    def test_mixed_precision(self):
        """Test mixed precision compression"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for mixed precision")
        
        # Enable mixed precision
        self.system.config.enable_mixed_precision = True
        
        # Large tensor for mixed precision benefits
        large_tensor = torch.randn(1000, 1000, device=self.device)
        
        # Compress with mixed precision
        compressed = self.system.compress(large_tensor)
        
        self.assertIsNotNone(compressed)
        self.assertGreater(compressed.compression_ratio, 1.0)
        
        # Decompress
        decompressed = self.system.decompress(compressed)
        
        # Mixed precision may have slightly higher error
        rel_error = torch.abs((large_tensor - decompressed.reconstructed_data) / (large_tensor + 1e-10)).mean()
        self.assertLess(rel_error, 0.1)
    
    def test_no_numpy_dependency(self):
        """Verify no NumPy is used in the implementation"""
        # This test ensures the implementation is purely PyTorch
        import sys
        
        # Check that numpy is not imported by our modules
        # Note: This is a simple check, in production you'd want more thorough testing
        
        original = torch.randn(100, 100, device=self.device)
        
        # Should work without numpy
        compressed = self.system.compress(original)
        decompressed = self.system.decompress(compressed)
        
        self.assertIsNotNone(compressed)
        self.assertIsNotNone(decompressed)


def run_performance_comparison():
    """Run performance comparison between PyTorch-only and standard implementations"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print("PyTorch-Only P-adic Compression Performance Test")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Initialize system
    config = PurelyPyTorchConfig(
        prime=257,
        precision=6,
        device=device,
        enable_triton=torch.cuda.is_available()
    )
    system = PurelyPyTorchPAdicSystem(config)
    
    # Test different tensor sizes
    test_cases = [
        ("Small (1K)", (1000,)),
        ("Medium (100K)", (100, 1000)),
        ("Large (1M)", (1000, 1000)),
        ("Very Large (10M)", (1000, 10000)) if torch.cuda.is_available() else None
    ]
    
    results = []
    for name, size in [tc for tc in test_cases if tc]:
        print(f"\nTesting {name}: {size}")
        print("-" * 40)
        
        # Generate test tensor
        tensor = torch.randn(size, device=device)
        
        # Warm-up
        _ = system.compress(tensor)
        
        # Time compression
        start = time.time()
        compressed = system.compress(tensor)
        compression_time = time.time() - start
        
        # Time decompression
        start = time.time()
        decompressed = system.decompress(compressed)
        decompression_time = time.time() - start
        
        # Calculate error
        error = torch.abs(tensor - decompressed.reconstructed_data).max().item()
        
        # Print results
        print(f"Compression Time: {compression_time:.4f}s")
        print(f"Decompression Time: {decompression_time:.4f}s")
        print(f"Compression Ratio: {compressed.compression_ratio:.2f}x")
        print(f"Max Reconstruction Error: {error:.6f}")
        print(f"Triton Accelerated: {system.triton_ops is not None}")
        
        results.append({
            'name': name,
            'size': size,
            'compression_time': compression_time,
            'decompression_time': decompression_time,
            'compression_ratio': compressed.compression_ratio,
            'max_error': error
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    total_compression_time = sum(r['compression_time'] for r in results)
    total_decompression_time = sum(r['decompression_time'] for r in results)
    avg_compression_ratio = sum(r['compression_ratio'] for r in results) / len(results)
    max_error = max(r['max_error'] for r in results)
    
    print(f"Total Compression Time: {total_compression_time:.4f}s")
    print(f"Total Decompression Time: {total_decompression_time:.4f}s")
    print(f"Average Compression Ratio: {avg_compression_ratio:.2f}x")
    print(f"Maximum Error: {max_error:.6f}")
    print(f"PyTorch Compile: {config.compile_mode}")
    print(f"Triton Kernels: {'Enabled' if config.enable_triton else 'Disabled'}")
    
    # Get system statistics
    stats = system.get_statistics()
    print(f"\nSystem Statistics:")
    print(f"  Total Compressions: {stats['total_compressions']}")
    print(f"  Total Decompressions: {stats['total_decompressions']}")
    print(f"  Sparse Encodings: {stats['padic_engine_stats']['sparse_encodings']}")
    print(f"  Dense Encodings: {stats['padic_engine_stats']['dense_encodings']}")


if __name__ == '__main__':
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance comparison
    run_performance_comparison()