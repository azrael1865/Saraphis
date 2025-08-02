"""
Compatibility tests for hybrid p-adic compressor
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import pytest
import time
from typing import Dict, Any

from .hybrid_padic_compressor import HybridPadicCompressionSystem, HybridPadicIntegrationManager
from .padic_compressor import PadicCompressionSystem
from .hybrid_padic_structures import HybridPadicWeight


class TestHybridPadicCompressor:
    """Test suite for hybrid p-adic compressor compatibility"""
    
    def setup_method(self):
        """Setup test environment"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for hybrid p-adic tests")
        
        self.config = {
            'prime': 5,
            'precision': 4,
            'chunk_size': 1000,
            'gpu_memory_limit_mb': 512,
            'enable_hybrid': True,
            'hybrid_threshold': 100,
            'force_hybrid': False,
            'enable_dynamic_switching': True,
            'validate_reconstruction': True,
            'max_reconstruction_error': 1e-6
        }
        
        self.hybrid_system = HybridPadicCompressionSystem(self.config)
        
        # Create pure system config
        pure_config = self.config.copy()
        pure_config['enable_hybrid'] = False
        self.pure_system = PadicCompressionSystem(pure_config)
    
    def test_hybrid_compression_roundtrip(self):
        """Test that hybrid compression preserves data"""
        # Create test tensor (large enough to trigger hybrid)
        test_tensor = torch.randn(200, 200, device='cuda', dtype=torch.float32)
        
        # Compress with hybrid system
        compressed = self.hybrid_system.compress(test_tensor)
        
        # Verify it used hybrid compression
        assert compressed['compression_type'] == 'hybrid'
        assert 'encoded_data' in compressed
        assert 'metadata' in compressed
        
        # Verify encoded data contains hybrid weights
        encoded_data = compressed['encoded_data']
        assert isinstance(encoded_data, list)
        assert len(encoded_data) > 0
        assert isinstance(encoded_data[0], HybridPadicWeight)
        
        # Decompress
        decompressed = self.hybrid_system.decompress(compressed)
        
        # Verify data is preserved within tolerance
        assert decompressed.shape == test_tensor.shape
        assert decompressed.dtype == test_tensor.dtype
        assert torch.allclose(decompressed, test_tensor, atol=1e-6)
    
    def test_pure_padic_compression_fallback(self):
        """Test that small tensors use pure p-adic compression"""
        # Create small test tensor (below hybrid threshold)
        test_tensor = torch.randn(10, 10, device='cpu', dtype=torch.float32)
        
        # Compress with hybrid system
        compressed = self.hybrid_system.compress(test_tensor)
        
        # Verify it used pure p-adic compression
        assert compressed['compression_type'] == 'pure_padic'
        
        # Decompress
        decompressed = self.hybrid_system.decompress(compressed)
        
        # Verify data is preserved
        assert decompressed.shape == test_tensor.shape
        assert torch.allclose(decompressed, test_tensor, atol=1e-6)
    
    def test_forced_hybrid_compression(self):
        """Test forced hybrid compression for small tensors"""
        # Configure system to force hybrid
        config = self.config.copy()
        config['force_hybrid'] = True
        system = HybridPadicCompressionSystem(config)
        
        # Create small test tensor
        test_tensor = torch.randn(5, 5, device='cuda', dtype=torch.float32)
        
        # Compress with forced hybrid
        compressed = system.compress(test_tensor)
        
        # Verify it used hybrid compression despite small size
        assert compressed['compression_type'] == 'hybrid'
        
        # Verify round-trip
        decompressed = system.decompress(compressed)
        assert torch.allclose(decompressed, test_tensor, atol=1e-6)
    
    def test_hybrid_vs_pure_compatibility(self):
        """Test that hybrid and pure systems are compatible"""
        # Create test tensor
        test_tensor = torch.randn(50, 50, device='cuda', dtype=torch.float32)
        
        # Compress with hybrid system
        hybrid_compressed = self.hybrid_system.compress(test_tensor)
        
        # Compress with pure system (move to CPU first)
        pure_compressed = self.pure_system.compress(test_tensor.cpu())
        
        # Both should produce valid results
        hybrid_decompressed = self.hybrid_system.decompress(hybrid_compressed)
        pure_decompressed = self.pure_system.decompress(pure_compressed)
        
        # Both should preserve data
        assert torch.allclose(hybrid_decompressed, test_tensor, atol=1e-6)
        assert torch.allclose(pure_decompressed, test_tensor.cpu(), atol=1e-6)
    
    def test_integration_manager(self):
        """Test integration manager functionality"""
        manager = HybridPadicIntegrationManager()
        manager.initialize_systems(self.config)
        
        # Test large tensor (should use hybrid)
        large_tensor = torch.randn(150, 150, device='cuda', dtype=torch.float32)
        compressed_large = manager.compress_with_optimal_system(large_tensor)
        decompressed_large = manager.decompress_with_optimal_system(compressed_large)
        
        # Verify data preservation
        assert torch.allclose(decompressed_large, large_tensor, atol=1e-6)
        
        # Test small tensor (should use pure)
        small_tensor = torch.randn(20, 20, device='cpu', dtype=torch.float32)
        compressed_small = manager.compress_with_optimal_system(small_tensor)
        decompressed_small = manager.decompress_with_optimal_system(compressed_small)
        
        # Verify data preservation
        assert torch.allclose(decompressed_small, small_tensor, atol=1e-6)
        
        # Check stats
        stats = manager.get_integration_stats()
        assert stats['total_operations'] == 2
        assert stats['hybrid_operations'] >= 1
        assert stats['pure_operations'] >= 1
    
    def test_performance_tracking(self):
        """Test performance metrics tracking"""
        # Reset stats
        self.hybrid_system.reset_performance_stats()
        
        # Perform several compressions
        for i in range(3):
            if i % 2 == 0:
                # Large tensor for hybrid
                tensor = torch.randn(100, 100, device='cuda', dtype=torch.float32)
            else:
                # Small tensor for pure
                tensor = torch.randn(10, 10, device='cpu', dtype=torch.float32)
            
            compressed = self.hybrid_system.compress(tensor)
            decompressed = self.hybrid_system.decompress(compressed)
        
        # Check performance stats
        stats = self.hybrid_system.get_performance_stats()
        assert stats['total_compressions'] == 3
        assert stats['total_decompressions'] == 3
        assert stats['hybrid_compressions'] >= 1
        assert stats['pure_padic_compressions'] >= 1
        assert stats['average_compression_time'] > 0
        assert stats['average_decompression_time'] > 0
    
    def test_memory_usage_tracking(self):
        """Test GPU memory usage tracking"""
        # Get initial memory usage
        initial_memory = self.hybrid_system.get_memory_usage()
        
        # Create several hybrid weights
        test_tensors = []
        for i in range(3):
            tensor = torch.randn(100, 100, device='cuda', dtype=torch.float32)
            test_tensors.append(tensor)
        
        # Compress all tensors
        compressed_data = []
        for tensor in test_tensors:
            compressed = self.hybrid_system.compress(tensor)
            compressed_data.append(compressed)
        
        # Check memory usage
        final_memory = self.hybrid_system.get_memory_usage()
        assert 'hybrid_memory_usage' in final_memory
        assert 'current_gpu_usage' in final_memory
        
        # Clean up
        del compressed_data
        del test_tensors
        self.hybrid_system.hybrid_manager.cleanup_gpu_memory()
    
    def test_error_handling_invalid_input(self):
        """Test error handling for invalid inputs"""
        # Test with None input
        with pytest.raises(ValueError, match="cannot be None"):
            self.hybrid_system.compress(None)
        
        with pytest.raises(ValueError, match="cannot be None"):
            self.hybrid_system.decompress(None)
        
        # Test with invalid tensor type
        with pytest.raises(TypeError, match="must be torch.Tensor"):
            self.hybrid_system.compress([1, 2, 3])
        
        # Test with invalid compressed data type
        with pytest.raises(TypeError, match="must be dict"):
            self.hybrid_system.decompress([1, 2, 3])
        
        # Test with missing keys in compressed data
        with pytest.raises(KeyError, match="Missing required key"):
            self.hybrid_system.decompress({'invalid': 'data'})
    
    def test_configuration_validation(self):
        """Test configuration parameter validation"""
        # Test invalid prime
        with pytest.raises(ValueError, match="Prime"):
            invalid_config = self.config.copy()
            invalid_config['prime'] = 1
            HybridPadicCompressionSystem(invalid_config)
        
        # Test invalid precision
        with pytest.raises(ValueError, match="Precision"):
            invalid_config = self.config.copy()
            invalid_config['precision'] = 0
            HybridPadicCompressionSystem(invalid_config)
        
        # Test invalid hybrid threshold
        with pytest.raises(ValueError, match="hybrid_threshold"):
            invalid_config = self.config.copy()
            invalid_config['hybrid_threshold'] = -1
            HybridPadicCompressionSystem(invalid_config)
        
        # Test invalid chunk size
        with pytest.raises(ValueError, match="Chunk size"):
            invalid_config = self.config.copy()
            invalid_config['chunk_size'] = 0
            HybridPadicCompressionSystem(invalid_config)
    
    def test_integration_manager_error_handling(self):
        """Test integration manager error handling"""
        manager = HybridPadicIntegrationManager()
        
        # Test operation without initialization
        with pytest.raises(RuntimeError, match="not initialized"):
            tensor = torch.randn(10, 10)
            manager.compress_with_optimal_system(tensor)
        
        # Test invalid config
        with pytest.raises(TypeError, match="must be dict"):
            manager.initialize_systems(None)
        
        # Test invalid input types
        manager.initialize_systems(self.config)
        
        with pytest.raises(ValueError, match="cannot be None"):
            manager.compress_with_optimal_system(None)
        
        with pytest.raises(TypeError, match="must be torch.Tensor"):
            manager.compress_with_optimal_system([1, 2, 3])
    
    def test_reconstruction_validation(self):
        """Test reconstruction error validation"""
        # Create system with very strict reconstruction validation
        config = self.config.copy()
        config['validate_reconstruction'] = True
        config['max_reconstruction_error'] = 1e-10  # Very strict
        
        system = HybridPadicCompressionSystem(config)
        
        # Test with tensor that might exceed error tolerance
        test_tensor = torch.randn(50, 50, device='cuda', dtype=torch.float32)
        
        # This might fail with very strict tolerance, but should complete
        try:
            compressed = system.compress(test_tensor)
            decompressed = system.decompress(compressed)
            # If it succeeds, verify the data is close
            assert torch.allclose(decompressed, test_tensor, atol=1e-6)
        except ValueError as e:
            # If validation fails due to strict tolerance, that's expected
            assert "reconstruction error" in str(e).lower()


class TestHybridPadicPerformance:
    """Performance-focused tests for hybrid p-adic compressor"""
    
    def setup_method(self):
        """Setup performance test environment"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for hybrid p-adic tests")
        
        self.config = {
            'prime': 5,
            'precision': 4,
            'chunk_size': 1000,
            'gpu_memory_limit_mb': 1024,
            'enable_hybrid': True,
            'hybrid_threshold': 500,
            'force_hybrid': False,
            'validate_reconstruction': False  # Disable for performance
        }
    
    def test_large_tensor_performance(self):
        """Test performance with large tensors"""
        system = HybridPadicCompressionSystem(self.config)
        
        # Large tensor for performance test
        test_tensor = torch.randn(1000, 1000, device='cuda', dtype=torch.float32)
        
        # Time hybrid compression
        start_time = time.time()
        compressed = system.compress(test_tensor)
        compression_time = time.time() - start_time
        
        # Time decompression
        start_time = time.time()
        decompressed = system.decompress(compressed)
        decompression_time = time.time() - start_time
        
        # Verify performance is reasonable
        assert compression_time < 30.0  # Should complete within 30 seconds
        assert decompression_time < 30.0
        
        # Verify compression was hybrid
        assert compressed['compression_type'] == 'hybrid'
        
        # Verify data preservation
        assert torch.allclose(decompressed, test_tensor, atol=1e-5)
        
        # Check performance stats
        stats = system.get_performance_stats()
        assert stats['hybrid_compressions'] == 1
        assert stats['average_hybrid_time'] > 0
    
    def test_performance_comparison(self):
        """Test performance comparison between hybrid and pure systems"""
        hybrid_system = HybridPadicCompressionSystem(self.config)
        
        pure_config = self.config.copy()
        pure_config['enable_hybrid'] = False
        pure_system = PadicCompressionSystem(pure_config)
        
        # Medium-sized tensor
        test_tensor = torch.randn(200, 200, device='cuda', dtype=torch.float32)
        
        # Time hybrid compression
        start_time = time.time()
        hybrid_compressed = hybrid_system.compress(test_tensor)
        hybrid_time = time.time() - start_time
        
        # Time pure compression (move to CPU first)
        test_tensor_cpu = test_tensor.cpu()
        start_time = time.time()
        pure_compressed = pure_system.compress(test_tensor_cpu)
        pure_time = time.time() - start_time
        
        # Both should complete successfully
        assert hybrid_compressed['compression_type'] == 'hybrid'
        assert pure_compressed['algorithm'] == 'PadicCompressionSystem'
        
        # Performance comparison (hybrid might be faster for large tensors)
        print(f"Hybrid compression time: {hybrid_time:.4f}s")
        print(f"Pure compression time: {pure_time:.4f}s")
        
        # Both should preserve data
        hybrid_decompressed = hybrid_system.decompress(hybrid_compressed)
        pure_decompressed = pure_system.decompress(pure_compressed)
        
        assert torch.allclose(hybrid_decompressed, test_tensor, atol=1e-5)
        assert torch.allclose(pure_decompressed, test_tensor_cpu, atol=1e-5)


def test_hybrid_compression_basic():
    """Standalone test for basic hybrid compression - can be run without pytest"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    config = {
        'prime': 5,
        'precision': 4,
        'chunk_size': 1000,
        'gpu_memory_limit_mb': 512,
        'enable_hybrid': True,
        'hybrid_threshold': 100,
        'force_hybrid': True  # Force hybrid for testing
    }
    
    system = HybridPadicCompressionSystem(config)
    
    # Test tensor
    test_tensor = torch.randn(50, 50, device='cuda', dtype=torch.float32)
    
    # Compress
    compressed = system.compress(test_tensor)
    
    # Verify hybrid compression
    assert compressed['compression_type'] == 'hybrid'
    
    # Decompress
    decompressed = system.decompress(compressed)
    
    # Verify data preservation
    assert torch.allclose(decompressed, test_tensor, atol=1e-6)
    
    print("✅ Basic hybrid compression test passed")


def test_integration_manager_basic():
    """Standalone test for integration manager"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    config = {
        'prime': 5,
        'precision': 4,
        'chunk_size': 1000,
        'gpu_memory_limit_mb': 512,
        'enable_hybrid': True,
        'hybrid_threshold': 100
    }
    
    manager = HybridPadicIntegrationManager()
    manager.initialize_systems(config)
    
    # Test tensor
    test_tensor = torch.randn(150, 150, device='cuda', dtype=torch.float32)
    
    # Compress with optimal system
    compressed = manager.compress_with_optimal_system(test_tensor)
    
    # Decompress
    decompressed = manager.decompress_with_optimal_system(compressed)
    
    # Verify data preservation
    assert torch.allclose(decompressed, test_tensor, atol=1e-6)
    
    # Check stats
    stats = manager.get_integration_stats()
    assert stats['total_operations'] == 1
    
    print("✅ Integration manager test passed")


if __name__ == "__main__":
    """Run basic tests when executed as script"""
    print("Running hybrid p-adic compressor tests...")
    
    try:
        test_hybrid_compression_basic()
        test_integration_manager_basic()
        print("✅ All basic tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise