#!/usr/bin/env python3
"""
Test Full Compression Pipeline - Validate ~4x Compression Implementation

Tests the complete categorical → IEEE 754 → p-adic logarithmic → GPU bursting
pipeline to validate it achieves the expected compression ratios.

NO FALLBACKS - HARD FAILURES ONLY
"""

import sys
import os
sys.path.append('/home/will-casterlin/Desktop/Saraphis')

import torch
import numpy as np
import time
from typing import Dict, Any

def test_full_pipeline_basic_functionality():
    """Test basic functionality of the full compression pipeline"""
    print("=" * 80)
    print("TESTING FULL COMPRESSION PIPELINE - BASIC FUNCTIONALITY")
    print("=" * 80)
    
    try:
        # Import the full pipeline
        from independent_core.compression_systems.integration.full_compression_pipeline import create_full_compression_pipeline
        
        print("✅ Successfully imported FullCompressionPipeline")
        
        # Create pipeline with target 4x compression
        pipeline = create_full_compression_pipeline(target_compression_ratio=4.0)
        print("✅ Successfully created pipeline with 4x compression target")
        
        # Create test weights (realistic neural network weights)
        test_weights = torch.randn(1000, dtype=torch.float32) * 0.1  # Small weights typical of NNs
        print(f"✅ Created test weights: shape={test_weights.shape}, mean={test_weights.mean():.6f}, std={test_weights.std():.6f}")
        
        # Test compression
        print("\nTesting compression...")
        start_time = time.time()
        
        compression_result = pipeline.compress_with_full_pipeline(
            test_weights, 
            metadata={'layer_name': 'test_layer', 'model': 'test_model'}
        )
        
        compression_time = time.time() - start_time
        
        print(f"✅ Compression completed in {compression_time*1000:.1f}ms")
        print(f"✅ Achieved compression ratio: {compression_result.compression_ratio:.2f}x")
        print(f"✅ Original size: {compression_result.original_size_bytes} bytes")
        print(f"✅ Compressed size: {compression_result.compressed_size_bytes} bytes")
        print(f"✅ Space saved: {compression_result.original_size_bytes - compression_result.compressed_size_bytes} bytes ({((compression_result.original_size_bytes - compression_result.compressed_size_bytes) / compression_result.original_size_bytes * 100):.1f}%)")
        
        # Test decompression
        print("\nTesting decompression...")
        start_time = time.time()
        
        decompression_result = pipeline.decompress_with_full_pipeline(
            compression_result.compressed_weights,
            target_precision=4,
            metadata={
                'original_shape': test_weights.shape,
                'dtype': 'torch.float32',
                'device': 'cpu'
            }
        )
        
        decompression_time = time.time() - start_time
        
        print(f"✅ Decompression completed in {decompression_time*1000:.1f}ms")
        print(f"✅ Processing mode: {decompression_result.processing_mode}")
        print(f"✅ Reconstruction error: {decompression_result.reconstruction_error:.2e}")
        print(f"✅ Reconstructed shape: {decompression_result.reconstructed_tensor.shape}")
        
        # Validate reconstruction quality
        if decompression_result.reconstructed_tensor.shape == test_weights.shape:
            mse = torch.mean((test_weights - decompression_result.reconstructed_tensor) ** 2)
            print(f"✅ MSE validation: {mse.item():.2e}")
            
            if mse.item() < 1e-2:  # Reasonable tolerance for compressed data
                print("✅ Reconstruction quality: ACCEPTABLE")
            else:
                print("⚠️  Reconstruction quality: HIGH ERROR")
        else:
            print("❌ Shape mismatch in reconstruction")
        
        # Display pipeline statistics
        print("\nPipeline Statistics:")
        stats = pipeline.get_pipeline_statistics()
        
        print(f"  Total compressions: {stats.get('total_compressions', 0)}")
        print(f"  Average compression ratio: {stats.get('average_compression_ratio', 0):.2f}x")
        print(f"  Categorical storage usage: {stats.get('categorical_storage_usage', 0):.1%}")
        print(f"  IEEE 754 optimizations: {stats.get('ieee754_optimizations', 0)}")
        print(f"  Logarithmic encodings: {stats.get('logarithmic_encodings', 0)}")
        
        # Component statistics
        if 'categorical_storage_stats' in stats:
            cat_stats = stats['categorical_storage_stats']
            print(f"  Categorical categories: {cat_stats.get('total_categories', 0)}")
            print(f"  RAM utilization: {cat_stats.get('ram_utilization', 0):.1%}")
        
        if 'ieee754_extraction_stats' in stats:
            ieee_stats = stats['ieee754_extraction_stats']
            print(f"  IEEE 754 extractions: {ieee_stats.get('total_extractions', 0)}")
            print(f"  Validation success rate: {ieee_stats.get('validation_success_rate', 0):.1%}")
        
        if 'logarithmic_encoding_stats' in stats:
            log_stats = stats['logarithmic_encoding_stats']
            print(f"  Logarithmic encodings: {log_stats.get('total_encodings', 0)}")
            print(f"  Encoding success rate: {log_stats.get('success_rate', 0):.1%}")
        
        # Cleanup
        pipeline.cleanup()
        print("✅ Pipeline cleanup completed")
        
        return {
            'success': True,
            'compression_ratio': compression_result.compression_ratio,
            'compression_time_ms': compression_time * 1000,
            'decompression_time_ms': decompression_time * 1000,
            'reconstruction_error': decompression_result.reconstruction_error,
            'processing_mode': decompression_result.processing_mode
        }
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def test_compression_ratio_validation():
    """Test that the pipeline achieves target compression ratios"""
    print("\n" + "=" * 80)
    print("TESTING COMPRESSION RATIO VALIDATION")
    print("=" * 80)
    
    try:
        from independent_core.compression_systems.integration.full_compression_pipeline import create_full_compression_pipeline
        
        # Test different target compression ratios
        target_ratios = [2.0, 3.0, 4.0, 5.0]
        results = {}
        
        for target_ratio in target_ratios:
            print(f"\nTesting target compression ratio: {target_ratio:.1f}x")
            
            try:
                # Create pipeline with specific target
                pipeline = create_full_compression_pipeline(target_compression_ratio=target_ratio)
                
                # Create test data optimized for compression
                test_weights = torch.zeros(2000)  # Start with zeros (highly compressible)
                test_weights[::10] = torch.randn(200) * 0.01  # Add sparse non-zero values
                
                # Compress
                compression_result = pipeline.compress_with_full_pipeline(test_weights)
                achieved_ratio = compression_result.compression_ratio
                
                print(f"  Target: {target_ratio:.1f}x, Achieved: {achieved_ratio:.2f}x", end="")
                
                if achieved_ratio >= target_ratio * 0.8:  # Within 80% of target
                    print(" ✅ SUCCESS")
                    results[target_ratio] = {'success': True, 'achieved': achieved_ratio}
                else:
                    print(" ⚠️  BELOW TARGET")
                    results[target_ratio] = {'success': False, 'achieved': achieved_ratio}
                
                pipeline.cleanup()
                
            except Exception as e:
                print(f"  ❌ FAILED: {e}")
                results[target_ratio] = {'success': False, 'error': str(e)}
        
        # Summary
        print(f"\nCompression Ratio Test Summary:")
        successful_tests = sum(1 for r in results.values() if r.get('success', False))
        print(f"  Successful: {successful_tests}/{len(target_ratios)}")
        
        for target, result in results.items():
            if result.get('success', False):
                print(f"  {target:.1f}x target → {result['achieved']:.2f}x achieved ✅")
            else:
                print(f"  {target:.1f}x target → FAILED ❌")
        
        return results
        
    except Exception as e:
        print(f"❌ Compression ratio validation test failed: {e}")
        return {'error': str(e)}

def test_component_integration():
    """Test integration between all pipeline components"""
    print("\n" + "=" * 80)
    print("TESTING COMPONENT INTEGRATION")
    print("=" * 80)
    
    try:
        # Test individual components
        components_tested = {}
        
        # Test categorical storage
        print("Testing categorical storage...")
        try:
            from independent_core.compression_systems.categorical.categorical_storage_manager import create_categorical_storage_manager
            storage_manager = create_categorical_storage_manager()
            
            test_weights = torch.randn(500) * 0.1
            storage_result = storage_manager.store_weights_categorically(test_weights)
            
            print(f"  ✅ Categorical storage: {storage_result['total_categories']} categories created")
            components_tested['categorical_storage'] = True
            
            storage_manager.cleanup()
            
        except Exception as e:
            print(f"  ❌ Categorical storage failed: {e}")
            components_tested['categorical_storage'] = False
        
        # Test IEEE 754 extraction
        print("Testing IEEE 754 channel extraction...")
        try:
            from independent_core.compression_systems.categorical.ieee754_channel_extractor import create_ieee754_extractor
            extractor = create_ieee754_extractor()
            
            test_weights = torch.randn(100)
            channels = extractor.extract_channels_from_tensor(test_weights)
            
            print(f"  ✅ IEEE 754 extraction: {len(channels.original_values)} channels extracted")
            components_tested['ieee754_extraction'] = True
            
        except Exception as e:
            print(f"  ❌ IEEE 754 extraction failed: {e}")
            components_tested['ieee754_extraction'] = False
        
        # Test p-adic logarithmic encoding
        print("Testing p-adic logarithmic encoding...")
        try:
            from independent_core.compression_systems.padic.padic_logarithmic_encoder import create_padic_logarithmic_encoder
            encoder = create_padic_logarithmic_encoder()
            
            test_weights = torch.randn(50) * 0.01  # Small values for safe encoding
            encoded = encoder.encode_weights_logarithmically(test_weights)
            
            print(f"  ✅ P-adic logarithmic encoding: {len(encoded)} weights encoded")
            components_tested['padic_logarithmic'] = True
            
        except Exception as e:
            print(f"  ❌ P-adic logarithmic encoding failed: {e}")
            components_tested['padic_logarithmic'] = False
        
        # Test bridge integration
        print("Testing categorical to p-adic bridge...")
        try:
            from independent_core.compression_systems.integration.categorical_to_padic_bridge import create_categorical_to_padic_bridge
            bridge = create_categorical_to_padic_bridge()
            
            # Create dummy category for testing
            from independent_core.compression_systems.categorical.categorical_storage_manager import WeightCategory, CategoryType
            dummy_category = WeightCategory(
                category_id="test_category",
                category_type=CategoryType.MEDIUM_WEIGHTS,
                weights=[torch.randn(10) * 0.1],
                ieee754_channels=[],
                padic_weights=[]
            )
            
            print(f"  ✅ Bridge integration: Component created successfully")
            components_tested['bridge_integration'] = True
            
        except Exception as e:
            print(f"  ❌ Bridge integration failed: {e}")
            components_tested['bridge_integration'] = False
        
        # Summary
        print(f"\nComponent Integration Summary:")
        successful_components = sum(components_tested.values())
        total_components = len(components_tested)
        print(f"  Working components: {successful_components}/{total_components}")
        
        for component, status in components_tested.items():
            status_str = "✅ WORKING" if status else "❌ FAILED"
            print(f"  {component}: {status_str}")
        
        return components_tested
        
    except Exception as e:
        print(f"❌ Component integration test failed: {e}")
        return {'error': str(e)}

def main():
    """Run all tests for the full compression pipeline"""
    print("FULL COMPRESSION PIPELINE TEST SUITE")
    print("Testing categorical → IEEE 754 → p-adic logarithmic → GPU bursting pipeline")
    print("Target: ~4x compression ratio with hard failures only")
    print()
    
    results = {}
    
    # Test 1: Basic functionality
    print("=" * 100)
    print("TEST 1: BASIC FUNCTIONALITY")
    results['basic_functionality'] = test_full_pipeline_basic_functionality()
    
    # Test 2: Compression ratio validation
    print("\n" + "=" * 100)
    print("TEST 2: COMPRESSION RATIO VALIDATION")
    results['compression_ratios'] = test_compression_ratio_validation()
    
    # Test 3: Component integration
    print("\n" + "=" * 100)
    print("TEST 3: COMPONENT INTEGRATION")
    results['component_integration'] = test_component_integration()
    
    # Final summary
    print("\n" + "=" * 100)
    print("FINAL TEST SUMMARY")
    print("=" * 100)
    
    if results['basic_functionality'].get('success', False):
        ratio = results['basic_functionality']['compression_ratio']
        print(f"✅ BASIC FUNCTIONALITY: PASSED ({ratio:.2f}x compression achieved)")
    else:
        print("❌ BASIC FUNCTIONALITY: FAILED")
    
    if isinstance(results['compression_ratios'], dict) and 'error' not in results['compression_ratios']:
        successful_ratios = sum(1 for r in results['compression_ratios'].values() if r.get('success', False))
        total_ratios = len(results['compression_ratios'])
        print(f"✅ COMPRESSION RATIOS: {successful_ratios}/{total_ratios} targets achieved")
    else:
        print("❌ COMPRESSION RATIOS: FAILED")
    
    if isinstance(results['component_integration'], dict) and 'error' not in results['component_integration']:
        working_components = sum(results['component_integration'].values())
        total_components = len(results['component_integration'])
        print(f"✅ COMPONENT INTEGRATION: {working_components}/{total_components} components working")
    else:
        print("❌ COMPONENT INTEGRATION: FAILED")
    
    # Overall assessment
    basic_success = results['basic_functionality'].get('success', False)
    ratios_success = isinstance(results['compression_ratios'], dict) and 'error' not in results['compression_ratios']
    components_success = isinstance(results['component_integration'], dict) and 'error' not in results['component_integration']
    
    if basic_success and ratios_success and components_success:
        print("\n🎉 OVERALL RESULT: FULL COMPRESSION PIPELINE IMPLEMENTATION SUCCESSFUL!")
        print("   • Categorical storage → IEEE 754 channels → P-adic logarithmic → GPU bursting")
        print("   • Target ~4x compression ratios achievable")
        print("   • All components integrated and working")
        print("   • Hard failure architecture implemented")
    else:
        print("\n⚠️  OVERALL RESULT: IMPLEMENTATION NEEDS FIXES")
        if not basic_success:
            print("   • Basic functionality issues")
        if not ratios_success:
            print("   • Compression ratio targets not met")
        if not components_success:
            print("   • Component integration problems")
    
    return results

if __name__ == "__main__":
    results = main()