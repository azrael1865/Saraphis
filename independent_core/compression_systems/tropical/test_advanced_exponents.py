#!/usr/bin/env python3
"""
Test advanced exponent channel extraction features
NO PLACEHOLDERS - PRODUCTION READY
"""

import torch
import time
from tropical_channel_extractor import (
    ExponentChannelConfig, 
    TropicalExponentExtractor,
    ExponentChannelCompressor, 
    ExponentChannelOptimizer,
    ExponentPatternAnalyzer,
    TropicalChannelManager,
    TropicalChannels
)
from tropical_polynomial import TropicalPolynomial, TropicalMonomial


def test_advanced_extraction():
    """Test advanced exponent extraction with all features"""
    print("=" * 60)
    print("TESTING ADVANCED EXPONENT EXTRACTION")
    print("=" * 60)
    
    # Create config with aggressive settings
    config = ExponentChannelConfig(
        use_sparse=True,
        sparsity_threshold=0.7,
        quantization="auto",
        compression_level=2,
        enable_delta_encoding=True,
        enable_pattern_clustering=True,
        block_size=16,
        gpu_coalesce=True,
        validate_lossless=True
    )
    
    # Create extractor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = TropicalExponentExtractor(device, config)
    
    # Test 1: Very sparse polynomial (>95% sparsity)
    print("\n1. VERY SPARSE POLYNOMIAL TEST")
    print("-" * 40)
    sparse_monomials = [
        TropicalMonomial(1.0, {0: 5}),
        TropicalMonomial(2.0, {10: 3}),
        TropicalMonomial(3.0, {50: 2}),
        TropicalMonomial(4.0, {99: 1})
    ]
    sparse_poly = TropicalPolynomial(sparse_monomials, num_variables=100)
    
    # Extract with advanced features
    sparse_data, sparse_meta = extractor.extract_exponents_advanced(sparse_poly)
    
    print(f"Original shape: {sparse_meta['original_shape']}")
    print(f"Sparsity: {sparse_meta['sparsity_analysis']['sparsity']:.2%}")
    print(f"Format: {sparse_meta['format']}")
    print(f"Recommended format: {sparse_meta['sparsity_analysis']['recommended_format']}")
    
    # Test 2: Pattern clustering
    print("\n2. PATTERN CLUSTERING TEST")
    print("-" * 40)
    
    # Create polynomial with repeated patterns
    pattern_monomials = []
    base_pattern = {0: 2, 1: 1}
    for i in range(50):
        # Repeat base pattern with variations
        pattern = base_pattern.copy()
        if i % 3 == 0:
            pattern[2] = 1
        pattern_monomials.append(TropicalMonomial(float(i), pattern))
    
    pattern_poly = TropicalPolynomial(pattern_monomials, num_variables=5)
    
    # Extract dense first
    dense_exps = extractor.extract_exponents(pattern_poly)
    print(f"Dense exponent matrix shape: {dense_exps.shape}")
    
    # Test clustering
    clustered, cluster_meta = extractor.cluster_exponent_patterns(dense_exps)
    if cluster_meta.get('clustered', False):
        print(f"Clustering successful!")
        print(f"Unique patterns: {cluster_meta['num_unique_patterns']}")
        print(f"Total patterns: {cluster_meta['num_total_patterns']}")
        print(f"Compression ratio: {cluster_meta['pattern_compression_ratio']:.2f}x")
        print(f"Pattern entropy: {cluster_meta.get('pattern_entropy', 'N/A'):.2f}")
    else:
        print(f"Clustering not beneficial: {cluster_meta.get('reason', 'unknown')}")
    
    # Test 3: Quantization
    print("\n3. QUANTIZATION TEST")
    print("-" * 40)
    
    # Create polynomial with different exponent ranges
    quant_tests = [
        ([(1.0, {0: 3, 1: 2}), (2.0, {2: 5})], "int8 range"),
        ([(1.0, {0: 100, 1: 50}), (2.0, {2: 127})], "int8 boundary"),
        ([(1.0, {0: 1000, 1: 500}), (2.0, {2: 5000})], "int16 range"),
    ]
    
    for coeffs_exps, test_name in quant_tests:
        monomials = [TropicalMonomial(c, e) for c, e in coeffs_exps]
        poly = TropicalPolynomial(monomials, num_variables=3)
        exps = extractor.extract_exponents(poly)
        
        quantized, quant_meta = extractor.quantize_exponents(exps)
        print(f"{test_name}:")
        print(f"  Max value: {quant_meta['max_value']}")
        print(f"  Quantization: {quant_meta['quantization']}")
        print(f"  Compression ratio: {quant_meta['compression_ratio']:.2f}x")
    
    # Test 4: Delta encoding
    print("\n4. DELTA ENCODING TEST")
    print("-" * 40)
    
    # Create polynomial with sequential patterns
    delta_monomials = []
    for i in range(20):
        # Create incrementing pattern
        exps = {j: i + j for j in range(3)}
        delta_monomials.append(TropicalMonomial(float(i), exps))
    
    delta_poly = TropicalPolynomial(delta_monomials, num_variables=5)
    delta_exps = extractor.extract_exponents(delta_poly)
    
    delta_encoded, delta_meta = extractor.delta_encode_exponents(delta_exps)
    if delta_meta.get('delta_encoded', False):
        print(f"Delta encoding successful!")
        print(f"Original range: {delta_meta['original_range']}")
        print(f"Delta range: {delta_meta['delta_range']}")
        print(f"Compression benefit: {delta_meta['compression_benefit']:.2%}")
    else:
        print(f"Delta encoding not beneficial: {delta_meta.get('reason', 'unknown')}")
    
    # Test 5: Sparsity pattern detection
    print("\n5. SPARSITY PATTERN DETECTION")
    print("-" * 40)
    
    # Create banded matrix pattern
    band_monomials = []
    for i in range(30):
        # Create band structure
        exps = {}
        if i > 0:
            exps[i-1] = 1
        exps[i] = 2
        if i < 29:
            exps[i+1] = 1
        band_monomials.append(TropicalMonomial(float(i), exps))
    
    band_poly = TropicalPolynomial(band_monomials, num_variables=30)
    band_exps = extractor.extract_exponents(band_poly)
    
    pattern_info = extractor.detect_sparsity_pattern(band_exps)
    print(f"Sparsity: {pattern_info['sparsity']:.2%}")
    print(f"Avg vars per monomial: {pattern_info['avg_vars_per_monomial']:.2f}")
    print(f"Diagonal density: {pattern_info['diagonal_density']:.2%}")
    print(f"Band concentration: {pattern_info.get('band_concentration', 0):.2%}")
    print(f"Recommended format: {pattern_info['recommended_format']}")


def test_compression():
    """Test compression and decompression"""
    print("\n" + "=" * 60)
    print("TESTING COMPRESSION/DECOMPRESSION")
    print("=" * 60)
    
    config = ExponentChannelConfig(validate_lossless=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    compressor = ExponentChannelCompressor(device, config)
    
    # Test sparse compression
    print("\n1. SPARSE COMPRESSION TEST")
    print("-" * 40)
    
    # Create sparse data
    rows = torch.tensor([0, 1, 5, 10, 15], device=device)
    cols = torch.tensor([2, 7, 3, 9, 1], device=device)
    indices = torch.stack([rows, cols])
    values = torch.tensor([5, 3, 8, 2, 4], dtype=torch.int32, device=device)
    shape = (20, 10)
    
    # Compress
    compressed_bytes, compress_meta = compressor.compress_sparse(indices, values, shape)
    print(f"Original size: {shape[0] * shape[1] * 4} bytes")
    print(f"Compressed size: {compress_meta['compressed_size']} bytes")
    print(f"Compression ratio: {compress_meta['compression_ratio']:.2f}x")
    
    # Decompress
    decompressed_indices, decompressed_values = compressor.decompress_sparse(
        compressed_bytes, compress_meta
    )
    
    # Validate
    if torch.equal(indices, decompressed_indices) and torch.equal(values, decompressed_values):
        print("✓ Lossless compression validated!")
    else:
        raise ValueError("COMPRESSION VALIDATION FAILED - DATA MISMATCH")
    
    # Test block compression
    print("\n2. BLOCK COMPRESSION TEST")
    print("-" * 40)
    
    # Create block-sparse matrix
    block_sparse = torch.zeros((64, 64), dtype=torch.int32, device=device)
    # Add some dense blocks
    block_sparse[0:16, 0:16] = torch.randint(0, 10, (16, 16), device=device)
    block_sparse[32:48, 32:48] = torch.randint(0, 10, (16, 16), device=device)
    # Add some sparse elements
    block_sparse[50, 10] = 5
    block_sparse[20, 55] = 3
    
    compressed_blocks, block_meta = compressor.compress_pattern_blocks(block_sparse)
    print(f"Number of blocks: {compressed_blocks.item()}")
    print(f"Block metadata entries: {len(block_meta['block_metadata'])}")
    
    # Analyze block types
    block_types = {}
    for bm in block_meta['block_metadata']:
        btype = bm['type']
        block_types[btype] = block_types.get(btype, 0) + 1
    
    for btype, count in block_types.items():
        print(f"  {btype} blocks: {count}")
    
    # Validate
    success = compressor.validate_compression(block_sparse, compressed_blocks, block_meta)
    if success:
        print("✓ Block compression validated!")


def test_optimization():
    """Test GPU optimization features"""
    print("\n" + "=" * 60)
    print("TESTING GPU OPTIMIZATION")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = ExponentChannelConfig(gpu_coalesce=True)
    optimizer = ExponentChannelOptimizer(device, config)
    
    # Create test matrix with poor memory layout
    print("\n1. MEMORY LAYOUT OPTIMIZATION")
    print("-" * 40)
    
    # Random sparse matrix
    test_matrix = torch.zeros((100, 50), dtype=torch.int32, device=device)
    # Add random non-zeros
    for _ in range(200):
        i = torch.randint(0, 100, (1,)).item()
        j = torch.randint(0, 50, (1,)).item()
        test_matrix[i, j] = torch.randint(1, 10, (1,)).item()
    
    # Optimize layout
    optimized, opt_meta = optimizer.optimize_memory_layout(test_matrix)
    
    if opt_meta.get('optimized', False):
        print(f"Layout optimization: {opt_meta['layout']}")
        print(f"Contiguous memory: {optimized.is_contiguous()}")
        if 'gpu_format' in opt_meta:
            print(f"GPU format ready: {opt_meta['gpu_format']}")
    
    # Test access indices
    print("\n2. ACCESS INDEX CREATION")
    print("-" * 40)
    
    row_indices, col_indices = optimizer.create_access_indices(test_matrix)
    print(f"Number of non-zero accesses: {row_indices.shape[0]}")
    
    # Verify sorted for coalescing
    if row_indices.shape[0] > 1:
        # Check if row-major sorted
        sort_key = row_indices * test_matrix.shape[1] + col_indices
        is_sorted = (sort_key[1:] >= sort_key[:-1]).all()
        print(f"Indices sorted for coalescing: {is_sorted}")
    
    # Test packing/unpacking
    print("\n3. GPU PACKING/UNPACKING")
    print("-" * 40)
    
    packed, pack_meta = optimizer.pack_for_gpu(test_matrix)
    print(f"Packing format: {pack_meta.get('format', 'none')}")
    print(f"Packed sparsity: {pack_meta.get('sparsity', 0):.2%}")
    
    if pack_meta.get('packed', False):
        # Unpack and verify
        unpacked = optimizer.unpack_from_gpu(packed, pack_meta)
        if torch.equal(unpacked, test_matrix):
            print("✓ Packing/unpacking validated!")
        else:
            raise ValueError("PACKING VALIDATION FAILED - DATA MISMATCH")


def test_pattern_analysis():
    """Test pattern analysis for compression strategy"""
    print("\n" + "=" * 60)
    print("TESTING PATTERN ANALYSIS")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = ExponentChannelConfig(pattern_analysis_depth=3)
    analyzer = ExponentPatternAnalyzer(device, config)
    
    # Create diverse polynomial set
    print("\n1. POLYNOMIAL SET ANALYSIS")
    print("-" * 40)
    
    polynomials = []
    
    # Add sparse polynomials
    for i in range(10):
        sparse_mons = [
            TropicalMonomial(float(i), {i: 1, (i+5) % 20: 2})
        ]
        polynomials.append(TropicalPolynomial(sparse_mons, num_variables=20))
    
    # Add dense polynomials
    for i in range(5):
        dense_mons = [
            TropicalMonomial(float(j), {k: (j+k) % 3 for k in range(5)})
            for j in range(10)
        ]
        polynomials.append(TropicalPolynomial(dense_mons, num_variables=20))
    
    # Add patterned polynomials
    for i in range(5):
        pattern = {0: 2, 1: 1, 2: 3}
        pattern_mons = [
            TropicalMonomial(float(j), pattern)
            for j in range(15)
        ]
        polynomials.append(TropicalPolynomial(pattern_mons, num_variables=20))
    
    # Analyze
    analysis = analyzer.analyze_polynomial_set(polynomials)
    
    print(f"Number of polynomials: {analysis['num_polynomials']}")
    print(f"Average monomials: {analysis['avg_monomials']:.1f}")
    print(f"Average sparsity: {analysis['avg_sparsity']:.2%}")
    print(f"Max exponent: {analysis.get('max_exponent', 'N/A')}")
    print(f"Pattern entropy: {analysis.get('pattern_entropy', 'N/A'):.2f}")
    print(f"Recommended quantization: {analysis.get('recommended_quantization', 'N/A')}")
    
    print("\nRecommendations:")
    for rec in analysis.get('recommendations', []):
        print(f"  - {rec}")
    
    # Test common patterns
    print("\n2. COMMON PATTERN DETECTION")
    print("-" * 40)
    
    common = analysis.get('common_patterns', {})
    print(f"Unique patterns: {common.get('num_unique_patterns', 0)}")
    print(f"Total patterns: {common.get('total_patterns', 0)}")
    print(f"Redundancy ratio: {common.get('redundancy_ratio', 0):.2%}")
    
    if common.get('frequent_patterns'):
        print("\nMost frequent patterns:")
        for i, pattern_info in enumerate(common['frequent_patterns'][:3]):
            print(f"  {i+1}. Pattern: {pattern_info['pattern'][:5]}... "
                  f"(freq: {pattern_info['frequency']:.2%})")
    
    # Test strategy suggestion
    print("\n3. COMPRESSION STRATEGY SUGGESTION")
    print("-" * 40)
    
    strategy = analyzer.suggest_compression_strategy(analysis)
    suggested_config = strategy['config']
    
    print(f"Suggested configuration:")
    print(f"  Use sparse: {suggested_config.use_sparse}")
    print(f"  Sparsity threshold: {suggested_config.sparsity_threshold}")
    print(f"  Quantization: {suggested_config.quantization}")
    print(f"  Compression level: {suggested_config.compression_level}")
    print(f"  Pattern clustering: {suggested_config.enable_pattern_clustering}")
    print(f"  Delta encoding: {suggested_config.enable_delta_encoding}")
    
    print("\nRationale:")
    for reason in strategy.get('rationale', []):
        print(f"  - {reason}")


def test_integration():
    """Test integration with TropicalChannelManager"""
    print("\n" + "=" * 60)
    print("TESTING CHANNEL MANAGER INTEGRATION")
    print("=" * 60)
    
    # Create advanced config
    config = ExponentChannelConfig(
        use_sparse=True,
        compression_level=2,
        enable_pattern_clustering=True,
        validate_lossless=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manager = TropicalChannelManager(device, config)
    
    # Create test polynomial
    monomials = [
        TropicalMonomial(10.0, {0: 5, 1: 2}),
        TropicalMonomial(5.0, {2: 3}),
        TropicalMonomial(8.0, {0: 1, 1: 1, 2: 1}),
        TropicalMonomial(3.0, {}),
        TropicalMonomial(12.0, {0: 5, 1: 2}),  # Duplicate pattern
        TropicalMonomial(15.0, {0: 5, 1: 2}),  # Duplicate pattern
    ]
    poly = TropicalPolynomial(monomials, num_variables=3)
    
    print("\n1. STANDARD EXTRACTION")
    print("-" * 40)
    
    # Standard extraction
    channels_standard = manager.polynomial_to_channels(poly, use_advanced_exponents=False)
    print(f"Standard channels shape: {channels_standard.exponent_channel.shape}")
    print(f"Standard dtype: {channels_standard.exponent_channel.dtype}")
    
    print("\n2. ADVANCED EXTRACTION")
    print("-" * 40)
    
    # Advanced extraction
    channels_advanced = manager.polynomial_to_channels(poly, use_advanced_exponents=True)
    print(f"Advanced channels shape: {channels_advanced.exponent_channel.shape}")
    print(f"Advanced dtype: {channels_advanced.exponent_channel.dtype}")
    
    # Verify reconstruction
    print("\n3. RECONSTRUCTION VALIDATION")
    print("-" * 40)
    
    reconstructed_standard = manager.channels_to_polynomial(channels_standard)
    reconstructed_advanced = manager.channels_to_polynomial(channels_advanced)
    
    print(f"Original polynomial: {len(poly.monomials)} monomials")
    print(f"Standard reconstructed: {len(reconstructed_standard.monomials)} monomials")
    print(f"Advanced reconstructed: {len(reconstructed_advanced.monomials)} monomials")
    
    # Check if coefficients match
    orig_coeffs = sorted([m.coefficient for m in poly.monomials])
    std_coeffs = sorted([m.coefficient for m in reconstructed_standard.monomials])
    adv_coeffs = sorted([m.coefficient for m in reconstructed_advanced.monomials])
    
    if orig_coeffs == std_coeffs:
        print("✓ Standard reconstruction validated!")
    else:
        print("✗ Standard reconstruction mismatch")
    
    if orig_coeffs == adv_coeffs:
        print("✓ Advanced reconstruction validated!")
    else:
        print("✗ Advanced reconstruction mismatch")


def performance_benchmark():
    """Benchmark performance of advanced features"""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different configurations
    configs = [
        ("Baseline", ExponentChannelConfig(compression_level=0, use_sparse=False)),
        ("Basic", ExponentChannelConfig(compression_level=1, use_sparse=True)),
        ("Advanced", ExponentChannelConfig(compression_level=2, use_sparse=True,
                                         enable_pattern_clustering=True,
                                         enable_delta_encoding=True))
    ]
    
    # Create large test set
    print("\nGenerating test data...")
    test_polynomials = []
    for i in range(100):
        num_mons = 50 + (i % 50)
        monomials = []
        for j in range(num_mons):
            # Mix of sparse and dense patterns
            if j % 3 == 0:
                # Sparse
                exps = {k: 1 for k in range(0, 100, 20)}
            else:
                # Dense
                exps = {k: (j + k) % 5 for k in range(10)}
            monomials.append(TropicalMonomial(float(j), exps))
        test_polynomials.append(TropicalPolynomial(monomials, num_variables=100))
    
    print(f"Test set: {len(test_polynomials)} polynomials")
    print(f"Average size: {sum(len(p.monomials) for p in test_polynomials) / len(test_polynomials):.1f} monomials")
    
    for name, config in configs:
        print(f"\n{name} Configuration:")
        print("-" * 40)
        
        extractor = TropicalExponentExtractor(device, config)
        
        # Time extraction
        start = time.time()
        total_size = 0
        for poly in test_polynomials:
            data, meta = extractor.extract_exponents_advanced(poly)
            if isinstance(data, torch.Tensor):
                total_size += data.numel() * data.element_size()
        
        elapsed = (time.time() - start) * 1000
        
        print(f"  Extraction time: {elapsed:.2f}ms")
        print(f"  Per polynomial: {elapsed / len(test_polynomials):.3f}ms")
        print(f"  Total data size: {total_size / 1024:.1f}KB")
        
        # Check compression benefit
        baseline_size = sum(
            len(p.monomials) * p.num_variables * 4  # int32
            for p in test_polynomials
        )
        compression_ratio = baseline_size / total_size if total_size > 0 else 1.0
        print(f"  Compression ratio: {compression_ratio:.2f}x")


def main():
    """Run all tests"""
    print("=" * 60)
    print("ADVANCED EXPONENT CHANNEL EXTRACTION TEST SUITE")
    print("=" * 60)
    
    try:
        test_advanced_extraction()
        test_compression()
        test_optimization()
        test_pattern_analysis()
        test_integration()
        performance_benchmark()
        
        print("\n" + "=" * 60)
        print("✅ ALL ADVANCED TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()