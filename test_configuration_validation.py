#!/usr/bin/env python3
"""
Test script for comprehensive configuration validation in p-adic compression system
"""

import torch
import sys
from independent_core.compression_systems.padic.padic_compressor import CompressionConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_valid_configurations():
    """Test valid configuration scenarios"""
    print("\n" + "="*80)
    print("TESTING VALID CONFIGURATIONS")
    print("="*80)
    
    # Test 1: Default configuration
    print("\n1. Default configuration:")
    try:
        config = CompressionConfig()
        print("✓ Default configuration validated successfully")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 2: CPU-only configuration
    print("\n2. CPU-only configuration:")
    try:
        config = CompressionConfig(
            compression_device='cpu',
            decompression_device='cpu'
        )
        print("✓ CPU-only configuration validated (with warnings)")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 3: GPU configuration (if available)
    print("\n3. GPU configuration:")
    try:
        config = CompressionConfig(
            compression_device='cuda',
            decompression_device='cuda',
            gpu_memory_limit_mb=2048
        )
        print("✓ GPU configuration validated")
    except Exception as e:
        print(f"✗ Failed (expected if no GPU): {e}")
    
    # Test 4: Mixed device configuration
    print("\n4. Mixed device configuration:")
    try:
        config = CompressionConfig(
            compression_device='cpu',
            decompression_device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("✓ Mixed device configuration validated")
    except Exception as e:
        print(f"✗ Failed: {e}")

def test_invalid_configurations():
    """Test invalid configuration scenarios that should raise errors"""
    print("\n" + "="*80)
    print("TESTING INVALID CONFIGURATIONS")
    print("="*80)
    
    test_cases = [
        # Test 1: Invalid device string
        {
            'name': 'Invalid device string',
            'config': {'compression_device': 'tpu'},
            'expected_error': 'Invalid compression_device'
        },
        # Test 2: Invalid precision bounds
        {
            'name': 'Invalid precision bounds',
            'config': {'min_precision': 5, 'base_precision': 3, 'max_precision': 4},
            'expected_error': 'Invalid precision bounds'
        },
        # Test 3: Invalid sparsity threshold
        {
            'name': 'Invalid sparsity threshold',
            'config': {'sparsity_threshold': 1.5},
            'expected_error': 'Sparsity threshold must be in (0, 1)'
        },
        # Test 4: Invalid pattern lengths
        {
            'name': 'Invalid pattern lengths',
            'config': {'min_pattern_length': 10, 'max_pattern_length': 5},
            'expected_error': 'min_pattern_length (10) must be less than max_pattern_length (5)'
        },
        # Test 5: Invalid entropy thresholds
        {
            'name': 'Invalid entropy thresholds',
            'config': {'huffman_threshold': 10.0, 'arithmetic_threshold': 5.0},
            'expected_error': 'Huffman threshold (10.0) must be less than arithmetic threshold (5.0)'
        },
        # Test 6: Excessive GPU memory limit
        {
            'name': 'Excessive GPU memory limit',
            'config': {'compression_device': 'cuda', 'gpu_memory_limit_mb': 1000000, 'enable_device_fallback': False},
            'expected_error': 'GPU memory limit'
        },
        # Test 7: Invalid pattern hash prime
        {
            'name': 'Invalid pattern hash prime (not prime)',
            'config': {'pattern_hash_prime': 4},
            'expected_error': 'pattern_hash_prime must be a prime number'
        },
        # Test 8: Negative max reconstruction error
        {
            'name': 'Negative max reconstruction error',
            'config': {'validate_reconstruction': True, 'max_reconstruction_error': -0.001},
            'expected_error': 'max_reconstruction_error must be positive'
        }
    ]
    
    for test in test_cases:
        print(f"\n{test['name']}:")
        try:
            config = CompressionConfig(**test['config'])
            print(f"✗ Expected error but validation passed!")
        except Exception as e:
            if test['expected_error'] in str(e):
                print(f"✓ Correctly raised error: {e}")
            else:
                print(f"✗ Unexpected error: {e}")

def test_warning_scenarios():
    """Test scenarios that should produce warnings but not errors"""
    print("\n" + "="*80)
    print("TESTING WARNING SCENARIOS")
    print("="*80)
    
    # Test 1: Low GPU memory limit
    print("\n1. Low GPU memory limit warning:")
    try:
        config = CompressionConfig(
            compression_device='cuda' if torch.cuda.is_available() else 'cpu',
            gpu_memory_limit_mb=256
        )
        print("✓ Configuration with low GPU memory limit created (should show warning)")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 2: Pattern length exceeds chunk size
    print("\n2. Pattern length exceeds chunk size:")
    try:
        config = CompressionConfig(
            max_pattern_length=20000,
            chunk_size=10000
        )
        print("✓ Configuration with oversized patterns created (should show warning)")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 3: Conflicting safety settings
    print("\n3. Conflicting safety settings:")
    try:
        config = CompressionConfig(
            enable_device_fallback=False,
            raise_on_error=False
        )
        print("✓ Configuration with conflicting safety settings created (should show warning)")
    except Exception as e:
        print(f"✗ Failed: {e}")

def test_device_normalization():
    """Test device string normalization"""
    print("\n" + "="*80)
    print("TESTING DEVICE NORMALIZATION")
    print("="*80)
    
    print("\n1. Testing 'gpu' -> 'cuda' normalization:")
    try:
        config = CompressionConfig(
            compression_device='gpu',
            decompression_device='gpu'
        )
        assert config.compression_device == 'cuda', f"Expected 'cuda', got '{config.compression_device}'"
        assert config.decompression_device == 'cuda', f"Expected 'cuda', got '{config.decompression_device}'"
        print("✓ Device normalization working correctly")
    except Exception as e:
        print(f"✗ Failed: {e}")

def test_triton_dependency():
    """Test Triton dependency validation"""
    print("\n" + "="*80)
    print("TESTING TRITON DEPENDENCY VALIDATION")
    print("="*80)
    
    # Check if Triton is available
    try:
        import triton
        triton_available = True
    except ImportError:
        triton_available = False
    
    print(f"\nTriton available: {triton_available}")
    
    if torch.cuda.is_available():
        print("\n1. Testing optimized sparse with GPU:")
        try:
            config = CompressionConfig(
                compression_device='cuda',
                use_optimized_sparse=True,
                raise_on_error=not triton_available  # Only raise if Triton not available
            )
            if triton_available:
                print("✓ Optimized sparse operations validated with Triton")
            else:
                print("✓ Configuration created with warning about missing Triton")
        except ImportError as e:
            if not triton_available:
                print(f"✓ Correctly raised error about missing Triton: {e}")
            else:
                print(f"✗ Unexpected error: {e}")
    else:
        print("Skipping GPU tests (no CUDA available)")

def main():
    """Run all configuration validation tests"""
    print("\n" + "="*80)
    print("P-ADIC COMPRESSION CONFIGURATION VALIDATION TEST SUITE")
    print("="*80)
    
    print(f"\nSystem info:")
    print(f"- PyTorch version: {torch.__version__}")
    print(f"- CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"- GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"- GPU memory: {props.total_memory / (1024**3):.1f} GB")
        print(f"- Compute capability: {props.major}.{props.minor}")
    
    # Run test suites
    test_valid_configurations()
    test_invalid_configurations()
    test_warning_scenarios()
    test_device_normalization()
    test_triton_dependency()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()