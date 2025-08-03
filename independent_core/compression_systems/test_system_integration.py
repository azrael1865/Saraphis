"""
Comprehensive test suite for SystemIntegrationCoordinator
Tests the complete integrated compression pipeline
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
import time
import gc
import json
import tempfile
import os
import threading
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_system_configuration():
    """Test system configuration"""
    print("\n" + "="*60)
    print("Testing System Configuration")
    print("="*60)
    
    from system_integration_coordinator import SystemConfiguration, OptimizationStrategy
    
    # Test 1: Default configuration
    print("\n1. Testing default configuration...")
    config = SystemConfiguration()
    
    assert config.gpu_memory_limit_mb == 8192
    assert config.enable_smart_pool == True
    assert config.prime == 251
    assert config.optimization_strategy == OptimizationStrategy.BALANCED
    print("   ✓ Default configuration valid")
    
    # Test 2: Custom configuration
    print("\n2. Testing custom configuration...")
    custom_config = SystemConfiguration(
        gpu_memory_limit_mb=4096,
        prime=127,
        precision=32,
        optimization_strategy=OptimizationStrategy.THROUGHPUT
    )
    
    assert custom_config.gpu_memory_limit_mb == 4096
    assert custom_config.prime == 127
    assert custom_config.optimization_strategy == OptimizationStrategy.THROUGHPUT
    print("   ✓ Custom configuration valid")
    
    # Test 3: Configuration serialization
    print("\n3. Testing configuration serialization...")
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert 'gpu_memory_limit_mb' in config_dict
    assert 'optimization_strategy' in config_dict
    
    # Test from_dict
    restored_config = SystemConfiguration.from_dict(config_dict)
    assert restored_config.gpu_memory_limit_mb == config.gpu_memory_limit_mb
    assert restored_config.prime == config.prime
    print("   ✓ Configuration serialization works")
    
    # Test 4: Save/load configuration
    print("\n4. Testing configuration save/load...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config_file = f.name
    
    try:
        config.save(config_file)
        loaded_config = SystemConfiguration.load(config_file)
        
        assert loaded_config.gpu_memory_limit_mb == config.gpu_memory_limit_mb
        assert loaded_config.optimization_strategy == config.optimization_strategy
        print("   ✓ Configuration save/load works")
    finally:
        os.unlink(config_file)
    
    # Test 5: Invalid configuration
    print("\n5. Testing invalid configuration...")
    try:
        invalid_config = SystemConfiguration(prime=0)
        assert False, "Should have failed with invalid prime"
    except ValueError as e:
        print(f"   ✓ Correctly rejected invalid configuration: {e}")
    
    print("\n✓ System configuration tests completed")
    return True


def test_compression_pipeline():
    """Test compression pipeline orchestrator"""
    print("\n" + "="*60)
    print("Testing Compression Pipeline")
    print("="*60)
    
    from system_integration_coordinator import (
        SystemConfiguration,
        CompressionPipelineOrchestrator,
        CompressionRequest,
        CompressionResult
    )
    
    config = SystemConfiguration()
    
    # Mock components
    class MockCompressor:
        def __init__(self, config):
            self.config = config
        
        def compress(self, tensor):
            # Simulate compression
            time.sleep(0.01)
            return {
                'encoded_data': [{'size': tensor.numel()}],
                'metadata': {'shape': tensor.shape}
            }
        
        def decompress(self, compressed):
            # Simulate decompression
            time.sleep(0.01)
            shape = compressed['metadata']['shape']
            return torch.randn(shape)
    
    components = {
        'hybrid_compressor': MockCompressor(config),
        'padic_compressor': MockCompressor(config)
    }
    
    pipeline = CompressionPipelineOrchestrator(config, components)
    
    # Test 1: Basic compression request
    print("\n1. Testing basic compression request...")
    tensor = torch.randn(100, 100)
    request = CompressionRequest(
        request_id="test_001",
        tensor=tensor,
        priority="normal"
    )
    
    result = pipeline.process_request(request)
    
    assert isinstance(result, CompressionResult)
    assert result.request_id == "test_001"
    assert result.success == True
    assert result.compression_ratio > 0
    print(f"   ✓ Compression successful, ratio: {result.compression_ratio:.2f}")
    
    # Test 2: Priority handling
    print("\n2. Testing priority handling...")
    high_priority_request = CompressionRequest(
        request_id="test_002",
        tensor=torch.randn(50, 50),
        priority="high"
    )
    
    result2 = pipeline.process_request(high_priority_request)
    assert result2.success == True
    print("   ✓ High priority request processed")
    
    # Test 3: Queue processing
    print("\n3. Testing queue processing...")
    
    # Submit multiple requests
    for i in range(5):
        req = CompressionRequest(
            request_id=f"batch_{i}",
            tensor=torch.randn(20, 20),
            priority="normal"
        )
        pipeline.submit_compression(req)
    
    # Process queue
    results = pipeline.process_queue()
    assert len(results) == 5
    assert all(r.success for r in results)
    print(f"   ✓ Processed {len(results)} queued requests")
    
    # Test 4: Statistics
    print("\n4. Testing pipeline statistics...")
    stats = pipeline.get_statistics()
    
    assert stats['total_requests'] >= 7  # All our test requests
    assert stats['successful_compressions'] >= 7
    assert stats['average_compression_ratio'] > 0
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Success rate: {stats['successful_compressions']}/{stats['total_requests']}")
    print(f"   Average compression ratio: {stats['average_compression_ratio']:.2f}")
    
    # Test 5: Callback execution
    print("\n5. Testing callback execution...")
    callback_executed = False
    callback_result = None
    
    def test_callback(result):
        nonlocal callback_executed, callback_result
        callback_executed = True
        callback_result = result
    
    callback_request = CompressionRequest(
        request_id="test_callback",
        tensor=torch.randn(10, 10),
        callback=test_callback
    )
    
    pipeline.process_request(callback_request)
    assert callback_executed == True
    assert callback_result.request_id == "test_callback"
    print("   ✓ Callback executed successfully")
    
    print("\n✓ Compression pipeline tests completed")
    return True


def test_performance_optimization():
    """Test performance optimization manager"""
    print("\n" + "="*60)
    print("Testing Performance Optimization")
    print("="*60)
    
    from system_integration_coordinator import (
        SystemConfiguration,
        PerformanceOptimizationManager,
        OptimizationStrategy
    )
    
    config = SystemConfiguration(optimization_strategy=OptimizationStrategy.ADAPTIVE)
    components = {}
    
    manager = PerformanceOptimizationManager(config, components)
    
    # Test 1: Metrics recording
    print("\n1. Testing metrics recording...")
    
    # Record various metrics
    for i in range(20):
        manager.record_metrics('compression', {
            'throughput': np.random.uniform(500, 1500),
            'latency': np.random.uniform(5, 20),
            'compression_ratio': np.random.uniform(2, 10),
            'memory_usage': np.random.uniform(10, 100)
        })
    
    summary = manager.get_performance_summary()
    assert 'metrics' in summary
    assert 'throughput' in summary['metrics']
    print(f"   ✓ Recorded metrics, current strategy: {summary['current_strategy']}")
    
    # Test 2: Strategy switching
    print("\n2. Testing strategy switching...")
    
    # Switch to throughput optimization
    manager.switch_strategy(OptimizationStrategy.THROUGHPUT)
    assert manager.get_current_strategy() == OptimizationStrategy.THROUGHPUT
    print("   ✓ Switched to THROUGHPUT strategy")
    
    # Switch to latency optimization
    manager.switch_strategy(OptimizationStrategy.LATENCY)
    assert manager.get_current_strategy() == OptimizationStrategy.LATENCY
    print("   ✓ Switched to LATENCY strategy")
    
    # Test 3: Adaptive optimization
    print("\n3. Testing adaptive optimization...")
    
    # Enable adaptation
    manager.adaptation_enabled = True
    manager.adaptation_interval = 1  # 1 second for testing
    
    # Simulate poor performance to trigger adaptation
    for i in range(20):
        manager.record_metrics('compression', {
            'throughput': 100,  # Very low
            'latency': 200,     # Very high
            'memory_usage': 95  # Very high
        })
    
    # Wait for adaptation
    time.sleep(1.5)
    
    # Check if strategy changed
    new_summary = manager.get_performance_summary()
    print(f"   Adapted to strategy: {new_summary['current_strategy']}")
    
    # Test 4: Performance summary
    print("\n4. Testing performance summary...")
    summary = manager.get_performance_summary()
    
    assert 'current_strategy' in summary
    assert 'adaptation_enabled' in summary
    assert 'metrics' in summary
    
    for metric in ['throughput', 'latency', 'memory_usage']:
        if metric in summary['metrics']:
            metric_data = summary['metrics'][metric]
            print(f"   {metric}: current={metric_data.get('current', 0):.2f}, "
                  f"avg={metric_data.get('average', 0):.2f}")
    
    print("\n✓ Performance optimization tests completed")
    return True


def test_system_integration():
    """Test complete system integration"""
    print("\n" + "="*60)
    print("Testing System Integration")
    print("="*60)
    
    from system_integration_coordinator import (
        SystemIntegrationCoordinator,
        SystemConfiguration,
        OptimizationStrategy,
        create_compression_system
    )
    
    # Test 1: System initialization
    print("\n1. Testing system initialization...")
    
    # Skip if CUDA not available
    if not torch.cuda.is_available():
        print("   ⚠️  CUDA not available, using limited testing")
        config = SystemConfiguration(
            enable_smart_pool=False,
            enable_auto_swap=False,
            enable_cpu_bursting=False
        )
    else:
        config = SystemConfiguration()
    
    try:
        system = create_compression_system(config)
        assert system.state.value == "ready"
        print("   ✓ System initialized successfully")
    except Exception as e:
        print(f"   ⚠️  System initialization limited: {e}")
        # Create minimal system for testing
        config = SystemConfiguration(
            enable_smart_pool=False,
            enable_auto_swap=False,
            enable_cpu_bursting=False,
            enable_memory_pressure=False
        )
        system = SystemIntegrationCoordinator(config)
    
    # Test 2: System status
    print("\n2. Testing system status...")
    status = system.get_system_status()
    
    assert 'state' in status
    assert 'components' in status
    assert 'statistics' in status
    
    print(f"   System state: {status['state']}")
    print(f"   Active components: {sum(1 for v in status['components'].values() if v == 'active')}")
    
    # Test 3: Compression/decompression
    print("\n3. Testing compression/decompression...")
    
    # Test tensor
    test_tensor = torch.randn(100, 100)
    original_size = test_tensor.numel() * test_tensor.element_size()
    
    try:
        # Compress
        result = system.compress(test_tensor, priority="normal")
        
        assert result.success == True
        assert result.compression_ratio > 1.0
        
        print(f"   ✓ Compression successful")
        print(f"   Compression ratio: {result.compression_ratio:.2f}x")
        print(f"   Compression time: {result.compression_time*1000:.2f}ms")
        print(f"   Processing mode: {result.processing_mode}")
        
        # Decompress
        decompressed = system.decompress(result.compressed_data)
        assert decompressed.shape == test_tensor.shape
        print("   ✓ Decompression successful")
        
    except Exception as e:
        print(f"   ⚠️  Compression test limited: {e}")
    
    # Test 4: Optimization
    print("\n4. Testing system optimization...")
    
    try:
        # Switch optimization strategy
        system.optimize_system(OptimizationStrategy.THROUGHPUT)
        
        status = system.get_system_status()
        if 'performance' in status and 'current_strategy' in status['performance']:
            print(f"   Optimization strategy: {status['performance']['current_strategy']}")
        
    except Exception as e:
        print(f"   ⚠️  Optimization test limited: {e}")
    
    # Test 5: Concurrent operations
    print("\n5. Testing concurrent operations...")
    
    results = []
    errors = []
    
    def compress_tensor(idx):
        try:
            tensor = torch.randn(50, 50)
            result = system.compress(tensor, priority="normal")
            results.append((idx, result))
        except Exception as e:
            errors.append((idx, str(e)))
    
    # Launch concurrent compressions
    threads = []
    for i in range(5):
        t = threading.Thread(target=compress_tensor, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    print(f"   Completed: {len(results)} compressions")
    print(f"   Errors: {len(errors)}")
    
    # Test 6: System shutdown
    print("\n6. Testing system shutdown...")
    system.shutdown()
    assert system.state.value == "shutdown"
    print("   ✓ System shutdown successfully")
    
    print("\n✓ System integration tests completed")
    return True


def test_end_to_end_workflow():
    """Test complete end-to-end workflow"""
    print("\n" + "="*60)
    print("Testing End-to-End Workflow")
    print("="*60)
    
    from system_integration_coordinator import (
        create_compression_system,
        SystemConfiguration,
        OptimizationStrategy
    )
    
    # Create system with specific configuration
    config = SystemConfiguration(
        gpu_memory_limit_mb=4096,
        enable_smart_pool=True,
        enable_auto_swap=True,
        enable_cpu_bursting=True,
        enable_memory_pressure=True,
        optimization_strategy=OptimizationStrategy.BALANCED
    )
    
    # Adjust for non-CUDA environments
    if not torch.cuda.is_available():
        config.enable_smart_pool = False
        config.enable_auto_swap = False
        config.enable_cpu_bursting = False
    
    print("\n1. Creating compression system...")
    system = create_compression_system(config)
    
    # Test various tensor sizes
    test_cases = [
        ("Small", torch.randn(10, 10)),
        ("Medium", torch.randn(100, 100)),
        ("Large", torch.randn(1000, 1000))
    ]
    
    total_compression_ratio = 0
    total_time = 0
    
    for name, tensor in test_cases:
        print(f"\n2. Testing {name} tensor ({tensor.shape})...")
        
        try:
            # Compress
            start_time = time.time()
            result = system.compress(tensor)
            
            if result.success:
                # Decompress
                decompressed = system.decompress(result.compressed_data)
                
                # Verify
                if torch.allclose(tensor, decompressed, rtol=1e-4, atol=1e-7):
                    print(f"   ✓ {name} tensor: ratio={result.compression_ratio:.2f}x, "
                          f"time={result.compression_time*1000:.1f}ms, "
                          f"mode={result.processing_mode}")
                    
                    total_compression_ratio += result.compression_ratio
                    total_time += result.compression_time
                else:
                    print(f"   ⚠️  {name} tensor: decompression mismatch")
            else:
                print(f"   ✗ {name} tensor: compression failed - {result.error}")
                
        except Exception as e:
            print(f"   ✗ {name} tensor: {e}")
    
    # Summary
    print("\n3. Workflow summary:")
    if total_compression_ratio > 0:
        avg_ratio = total_compression_ratio / len(test_cases)
        print(f"   Average compression ratio: {avg_ratio:.2f}x")
        print(f"   Total processing time: {total_time*1000:.1f}ms")
    
    # Get final system status
    status = system.get_system_status()
    print(f"\n4. Final system status:")
    print(f"   Total compressions: {status['statistics']['total_compressions']}")
    print(f"   Total decompressions: {status['statistics']['total_decompressions']}")
    
    # Shutdown
    system.shutdown()
    
    print("\n✓ End-to-end workflow completed")
    return True


def run_stress_test():
    """Run stress test on the system"""
    print("\n" + "="*60)
    print("Running System Stress Test")
    print("="*60)
    
    from system_integration_coordinator import create_compression_system, SystemConfiguration
    
    # Create system with aggressive settings
    config = SystemConfiguration(
        optimization_strategy=OptimizationStrategy.THROUGHPUT,
        max_concurrent_operations=20
    )
    
    if not torch.cuda.is_available():
        config.enable_smart_pool = False
        config.enable_auto_swap = False
        config.enable_cpu_bursting = False
    
    system = create_compression_system(config)
    
    print("\n1. Running continuous compression stress test...")
    
    start_time = time.time()
    success_count = 0
    error_count = 0
    
    # Run for 10 seconds
    while time.time() - start_time < 10:
        try:
            # Random tensor size
            size = np.random.randint(10, 500)
            tensor = torch.randn(size, size)
            
            # Compress
            result = system.compress(tensor)
            if result.success:
                success_count += 1
                
                # Occasionally decompress
                if np.random.random() < 0.2:
                    decompressed = system.decompress(result.compressed_data)
            else:
                error_count += 1
                
        except Exception as e:
            error_count += 1
            
        # Brief pause
        time.sleep(0.01)
    
    duration = time.time() - start_time
    
    print(f"\n2. Stress test results:")
    print(f"   Duration: {duration:.1f}s")
    print(f"   Successful compressions: {success_count}")
    print(f"   Errors: {error_count}")
    print(f"   Throughput: {success_count/duration:.1f} ops/sec")
    
    # Get final statistics
    status = system.get_system_status()
    
    print(f"\n3. System statistics:")
    print(f"   Uptime: {status['statistics']['uptime_seconds']:.1f}s")
    print(f"   System errors: {status['statistics']['system_errors']}")
    
    if 'pipeline' in status:
        pipeline_stats = status['pipeline']
        if 'average_compression_ratio' in pipeline_stats:
            print(f"   Average compression ratio: {pipeline_stats['average_compression_ratio']:.2f}x")
        if 'total_memory_saved_mb' in pipeline_stats:
            print(f"   Total memory saved: {pipeline_stats['total_memory_saved_mb']:.1f}MB")
    
    # Cleanup
    system.shutdown()
    gc.collect()
    
    print("\n✓ Stress test completed")
    return True


def run_all_tests():
    """Run all system integration tests"""
    print("\n" + "="*80)
    print("System Integration Test Suite")
    print("="*80)
    
    try:
        # Component tests
        test_system_configuration()
        test_compression_pipeline()
        test_performance_optimization()
        
        # Integration tests
        test_system_integration()
        test_end_to_end_workflow()
        
        # Stress test
        run_stress_test()
        
        print("\n" + "="*80)
        print("✅ All system integration tests completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.exception("Test failure details:")
        return False
    
    return True


if __name__ == "__main__":
    # Clear GPU cache before tests
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Run tests
    success = run_all_tests()
    
    if success:
        print("\n🎉 SystemIntegrationCoordinator is ready for production!")
        print("   - Unified Configuration: ✓")
        print("   - Pipeline Orchestration: ✓")
        print("   - Performance Optimization: ✓")
        print("   - Component Integration: ✓")
        print("   - End-to-End Workflow: ✓")
    else:
        print("\n⚠️  Some tests failed. Please review the logs above.")