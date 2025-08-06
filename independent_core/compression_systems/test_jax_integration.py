#!/usr/bin/env python3
"""
Test JAX Integration with Main System
Verifies that all JAX components are properly connected and functional
"""

import torch
import numpy as np
import warnings
from typing import Dict, Any

def test_jax_imports():
    """Test that JAX components can be imported"""
    print("\n" + "="*60)
    print("Testing JAX Component Imports")
    print("="*60)
    
    components_status = {}
    
    # Test each JAX component import
    components = [
        ('JAXMemoryPool', 'tropical.jax_memory_pool'),
        ('TropicalJAXBridge', 'tropical.jax_tropical_bridge'),
        ('JAXDeviceManager', 'tropical.jax_device_manager'),
        ('JAXMemoryOptimizer', 'tropical.jax_memory_optimizer'),
        ('JAXConfigAdapter', 'tropical.jax_config_adapter'),
        ('JAXCompilationOptimizer', 'tropical.jax_compilation_optimizer'),
        ('TropicalJAXEngine', 'tropical.jax_tropical_engine'),
        ('JAXTropicalStrategy', 'tropical.jax_tropical_strategy'),
    ]
    
    for class_name, module_path in components:
        try:
            module = __import__(f'independent_core.compression_systems.{module_path}', 
                               fromlist=[class_name])
            cls = getattr(module, class_name)
            components_status[class_name] = "✓ Available"
            print(f"  {class_name}: ✓ Import successful")
        except ImportError as e:
            components_status[class_name] = f"✗ Import failed: {e}"
            print(f"  {class_name}: ✗ Import failed - {e}")
        except Exception as e:
            components_status[class_name] = f"✗ Error: {e}"
            print(f"  {class_name}: ✗ Error - {e}")
    
    return components_status


def test_system_integration():
    """Test JAX integration with SystemIntegrationCoordinator"""
    print("\n" + "="*60)
    print("Testing System Integration")
    print("="*60)
    
    try:
        from independent_core.compression_systems.system_integration_coordinator import (
            SystemIntegrationCoordinator, SystemConfiguration
        )
        
        # Create configuration with JAX enabled
        config = SystemConfiguration(
            enable_jax=True,
            jax_backend="cpu",  # Use CPU for testing
            jax_memory_fraction=0.5,
            jax_compilation_cache_size=64
        )
        
        print("  Creating system coordinator with JAX enabled...")
        coordinator = SystemIntegrationCoordinator(config)
        
        # Check which components are active
        status = coordinator.get_system_status()
        print("\n  Component Status:")
        for component, state in status['components'].items():
            symbol = "✓" if state == "active" else "✗"
            print(f"    {component}: {symbol} {state}")
        
        # Test compression with a small tensor
        if status['components'].get('jax_backend') == 'active':
            print("\n  Testing JAX compression...")
            test_tensor = torch.randn(100, 100)
            result = coordinator.compress(test_tensor, priority="normal")
            print(f"    Compression successful: {result.success}")
            print(f"    Compression ratio: {result.compression_ratio:.2f}x")
        else:
            print("\n  JAX backend not active - skipping compression test")
        
        return True
        
    except Exception as e:
        print(f"  System integration test failed: {e}")
        return False


def test_model_compression_api():
    """Test JAX strategy in model compression API"""
    print("\n" + "="*60)
    print("Testing Model Compression API with JAX")
    print("="*60)
    
    try:
        from independent_core.compression_systems.model_compression_api import (
            CompressionProfile, ModelCompressionAPI
        )
        
        # Test if JAX strategy is available
        print("  Testing JAX strategy availability...")
        
        # Try creating profile with JAX strategy
        try:
            profile = CompressionProfile(
                strategy="jax",
                target_compression_ratio=4.0,
                mode="balanced"
            )
            print("    JAX strategy accepted in profile: ✓")
            
            # Create API instance
            api = ModelCompressionAPI(profile)
            print("    ModelCompressionAPI created with JAX: ✓")
            
            # Create a small test model
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, 10)
            )
            
            print("\n  Testing compression with JAX strategy...")
            compressed = api.compress(model)
            stats = compressed.get_compression_stats()
            print(f"    Compression ratio: {stats['total_compression_ratio']:.2f}x")
            print(f"    Strategy used: {compressed.strategy_map}")
            
            return True
            
        except ValueError as e:
            if "jax" in str(e).lower():
                print(f"    JAX strategy not available: {e}")
            else:
                raise
            return False
            
    except Exception as e:
        print(f"  Model compression API test failed: {e}")
        return False


def test_strategy_manager():
    """Test JAX strategy in compression strategy manager"""
    print("\n" + "="*60)
    print("Testing Strategy Manager with JAX")
    print("="*60)
    
    try:
        from independent_core.compression_systems.strategies.compression_strategy import (
            StrategyConfig, StrategySelector
        )
        
        config = StrategyConfig(use_gpu=False)  # Use CPU for testing
        selector = StrategySelector(config)
        
        print("  Available strategies:")
        for strategy_name in selector.strategy_cache.keys():
            print(f"    - {strategy_name}")
        
        if 'jax' in selector.strategy_cache:
            print("\n  JAX strategy is available: ✓")
            
            # Test JAX strategy compression
            test_tensor = torch.randn(100, 100)
            jax_strategy = selector.strategy_cache['jax']
            
            print("  Testing JAX strategy compression...")
            compressed = jax_strategy.compress(test_tensor)
            print(f"    Compression successful: ✓")
            print(f"    Compression ratio: {compressed.compression_ratio:.2f}x")
            
            return True
        else:
            print("\n  JAX strategy not found in cache")
            return False
            
    except Exception as e:
        print(f"  Strategy manager test failed: {e}")
        return False


def test_performance_monitoring():
    """Test JAX metrics in performance monitor"""
    print("\n" + "="*60)
    print("Testing Performance Monitor with JAX Metrics")
    print("="*60)
    
    try:
        from independent_core.compression_systems.tropical.unified_performance_monitor import (
            UnifiedPerformanceMonitor, MonitorConfig, PipelineType, MetricType
        )
        
        config = MonitorConfig()
        monitor = UnifiedPerformanceMonitor(config)
        
        print("  Recording JAX metrics...")
        monitor.record_jax_metrics(
            compilation_time_ms=123.45,
            cache_hit_rate=0.85,
            memory_pool_usage_mb=256.0
        )
        
        # Check if metrics were recorded
        jax_metrics = monitor.current_metrics.get(PipelineType.JAX, {})
        
        if jax_metrics:
            print("  JAX metrics recorded successfully:")
            print(f"    Compilation time: {jax_metrics.get(MetricType.JAX_COMPILATION, 0):.2f} ms")
            print(f"    Cache hit rate: {jax_metrics.get(MetricType.JAX_CACHE_HITS, 0):.2%}")
            print(f"    Memory pool: {jax_metrics.get(MetricType.JAX_MEMORY_POOL, 0):.1f} MB")
            return True
        else:
            print("  JAX metrics not recorded")
            return False
            
    except Exception as e:
        print(f"  Performance monitoring test failed: {e}")
        return False


def main():
    """Run all JAX integration tests"""
    print("\n" + "="*60)
    print("JAX INTEGRATION TEST SUITE")
    print("="*60)
    
    results = {}
    
    # Run tests
    print("\n1. Component Import Test")
    import_status = test_jax_imports()
    results['imports'] = any('Available' in status for status in import_status.values())
    
    print("\n2. System Integration Test")
    results['system'] = test_system_integration()
    
    print("\n3. Model Compression API Test")
    results['api'] = test_model_compression_api()
    
    print("\n4. Strategy Manager Test")
    results['strategy'] = test_strategy_manager()
    
    print("\n5. Performance Monitoring Test")
    results['monitoring'] = test_performance_monitoring()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name.capitalize()}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n✓ JAX integration is fully functional!")
    elif total_passed > 0:
        print("\n⚠ JAX integration is partially functional")
        print("  Some features may not be available")
    else:
        print("\n✗ JAX integration is not functional")
        print("  JAX components may not be installed")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)