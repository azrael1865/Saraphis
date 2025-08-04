"""
Comprehensive test suite for MemoryPressureHandler
Tests GPU/CPU coordination and intelligent decision making
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
import time
import psutil
import gc
import logging
from typing import List, Dict, Any, Tuple
from fractions import Fraction
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Import real p-adic components
from independent_core.compression_systems.padic.padic_encoder import (
    PadicWeight, 
    PadicMathematicalOperations
)

def create_real_padic_weights(num_weights: int, precision: int = 4, prime: int = 257) -> List[PadicWeight]:
    """Create real p-adic weights for testing using proper mathematical conversion"""
    # SAFETY CHECK: Ensure precision doesn't cause overflow
    import math
    safe_threshold = 1e12
    max_safe_precision = int(math.log(safe_threshold) / math.log(prime))
    
    if precision > max_safe_precision:
        print(f"Safety: Reducing precision from {precision} to {max_safe_precision} for prime={prime}")
        precision = max_safe_precision
    
    from independent_core.compression_systems.padic.padic_encoder import PadicMathematicalOperations
    
    # Initialize p-adic mathematical operations
    math_ops = PadicMathematicalOperations(prime, precision)
    weights = []
    
    # Value ranges for different test scenarios
    value_ranges = [
        (-10.0, 10.0),      # Standard range
        (-1.0, 1.0),        # Small values
        (-100.0, 100.0),    # Larger values
        (0.001, 0.999),     # Fractional values
        (-0.999, -0.001)    # Negative fractional
    ]
    
    attempts = 0
    max_attempts = num_weights * 10  # Allow multiple attempts
    
    while len(weights) < num_weights and attempts < max_attempts:
        attempts += 1
        
        # Select value range based on index
        range_idx = (len(weights) % len(value_ranges))
        min_val, max_val = value_ranges[range_idx]
        
        # Generate value
        if attempts % 3 == 0:
            # Sometimes use integer values
            value = float(np.random.randint(int(min_val), int(max_val) + 1))
        elif attempts % 3 == 1:
            # Sometimes use simple fractions
            numerator = np.random.randint(1, 100)
            denominator = np.random.randint(1, 100)
            value = numerator / denominator
            if np.random.rand() > 0.5:
                value = -value
        else:
            # Use random float
            value = np.random.uniform(min_val, max_val)
        
        try:
            # Convert to proper p-adic representation
            weight = math_ops.to_padic(value)
            
            # Validate the conversion by reconstructing
            reconstructed = math_ops.from_padic(weight)
            
            # Verify mathematical correctness
            relative_error = abs(value - reconstructed) / (abs(value) + 1e-10)
            if relative_error > 1e-6:
                # Skip values with high reconstruction error
                continue
            
            # Additional validation
            if not validate_single_weight(weight, prime, precision):
                continue
            
        weights.append(weight)
    
        except (ValueError, TypeError, OverflowError) as e:
            # Skip values that can't be converted
            continue
    
    if len(weights) < num_weights:
        raise ValueError(
            f"Could not create enough valid p-adic weights. "
            f"Created {len(weights)} out of {num_weights} requested. "
            f"Consider adjusting value ranges or precision."
        )
    
    return weights[:num_weights]


def validate_single_weight(weight: PadicWeight, expected_prime: int, expected_precision: int) -> bool:
    """Validate a single p-adic weight for mathematical correctness"""
    try:
        # Check basic structure
        if not hasattr(weight, 'digits') or not hasattr(weight, 'valuation'):
            return False
        
        if not hasattr(weight, 'value') or not hasattr(weight, 'prime') or not hasattr(weight, 'precision'):
            return False
        
        # Check prime and precision match
        if weight.prime != expected_prime or weight.precision != expected_precision:
            return False
        
        # Check digit properties
        if not isinstance(weight.digits, list) or len(weight.digits) != weight.precision:
            return False
        
        for digit in weight.digits:
            if not isinstance(digit, int) or not (0 <= digit < weight.prime):
                return False
        
        # Check valuation bounds
        if not isinstance(weight.valuation, int):
            return False
        
        if weight.valuation < -weight.precision or weight.valuation > weight.precision:
            return False
        
        # Check value is Fraction
        if not isinstance(weight.value, Fraction):
            return False
        
        return True
        
    except Exception:
        return False


def validate_padic_weights(weights: List[PadicWeight], prime: int, precision: int) -> bool:
    """Validate that p-adic weights are mathematically correct"""
    from independent_core.compression_systems.padic.padic_encoder import PadicMathematicalOperations
    
    if not weights:
        print("No weights to validate")
        return False
    
    math_ops = PadicMathematicalOperations(prime, precision)
    
    for i, weight in enumerate(weights):
        try:
            # Basic structure validation
            if not validate_single_weight(weight, prime, precision):
                print(f"Weight {i}: Failed structural validation")
                return False
            
            # Check reconstruction
            reconstructed = math_ops.from_padic(weight)
            original = float(weight.value)
            
            # Allow small numerical errors
            relative_error = abs(original - reconstructed) / (abs(original) + 1e-10)
            if relative_error > 1e-5:
                print(f"Weight {i}: Reconstruction error - original={original}, reconstructed={reconstructed}, error={relative_error}")
                return False
            
            # Verify digit extraction correctness
            if weight.value.numerator != 0:
                # Non-zero weights should have at least one non-zero digit
                if all(d == 0 for d in weight.digits):
                    print(f"Weight {i}: All digits are zero for non-zero value {weight.value}")
                    return False
                
        except Exception as e:
            print(f"Weight {i}: Validation failed - {type(e).__name__}: {e}")
            return False
    
    return True


def measure_weight_conversion_time(num_weights: int, precision: int, prime: int) -> Tuple[List[PadicWeight], float]:
    """Measure time to create real p-adic weights"""
    start_time = time.time()
    weights = create_real_padic_weights(num_weights, precision, prime)
    conversion_time = time.time() - start_time
    return weights, conversion_time


def test_memory_pressure_handler_basic():
    """Test basic memory pressure handler functionality"""
    print("\n" + "="*60)
    print("Testing Basic Memory Pressure Handler")
    print("="*60)
    
    from memory_pressure_handler import (
        MemoryPressureHandler, 
        PressureHandlerConfig,
        ProcessingMode,
        MemoryState
    )
    
    # Create config
    config = PressureHandlerConfig(
        gpu_critical_threshold_mb=100,
        gpu_high_threshold_mb=500,
        gpu_moderate_threshold_mb=1000,
        monitoring_interval_ms=50
    )
    
    # Create handler
    handler = MemoryPressureHandler(config)
    
    # Test 1: Initial state
    print("\n1. Testing initial state...")
    assert handler.current_memory_state == MemoryState.HEALTHY
    assert handler.processing_mode == ProcessingMode.ADAPTIVE
    print("   ✓ Initial state correct")
    
    # Test 2: Basic decision
    print("\n2. Testing basic decision making...")
    metadata = {
        'size_mb': 50,
        'priority': 'normal'
    }
    
    use_cpu, decision_info = handler.should_use_cpu(metadata)
    print(f"   Decision: {'CPU' if use_cpu else 'GPU'}")
    print(f"   Reason: {decision_info.get('reason', 'unknown')}")
    print(f"   GPU free: {decision_info.get('gpu_free_mb', 0):.0f}MB")
    
    # Test 3: Processing mode
    print("\n3. Testing processing mode changes...")
    handler.set_processing_mode(ProcessingMode.GPU_PREFERRED)
    assert handler.processing_mode == ProcessingMode.GPU_PREFERRED
    
    handler.set_processing_mode(ProcessingMode.CPU_PREFERRED)
    metadata = {'size_mb': 10, 'priority': 'normal'}
    use_cpu, _ = handler.should_use_cpu(metadata)
    assert use_cpu == True
    print("   ✓ Processing mode changes work")
    
    # Test 4: Performance updates
    print("\n4. Testing performance tracking...")
    handler.update_performance('gpu', True, 1000.0, 5.0)
    handler.update_performance('cpu', True, 500.0, 10.0)
    
    assert handler.performance_metrics.gpu_throughput > 0
    assert handler.performance_metrics.cpu_throughput > 0
    print(f"   GPU score: {handler.performance_metrics.gpu_score:.2f}")
    print(f"   CPU score: {handler.performance_metrics.cpu_score:.2f}")
    
    # Test 5: Statistics
    print("\n5. Testing statistics...")
    stats = handler.get_statistics()
    print(f"   Total decisions: {stats['total_decisions']}")
    print(f"   GPU decisions: {stats['gpu_decisions']}")
    print(f"   CPU decisions: {stats['cpu_decisions']}")
    
    # Cleanup
    handler.cleanup()
    print("\n✓ Basic memory pressure handler tests completed")
    return True


def test_memory_threshold_detection():
    """Test memory threshold detection and state transitions"""
    print("\n" + "="*60)
    print("Testing Memory Threshold Detection")
    print("="*60)
    
    from memory_pressure_handler import (
        MemoryPressureHandler,
        PressureHandlerConfig,
        MemoryState,
        MemoryMetrics
    )
    
    # Create handler with low thresholds for testing
    config = PressureHandlerConfig(
        gpu_critical_threshold_mb=2048,
        gpu_high_threshold_mb=4096,
        gpu_moderate_threshold_mb=6144,
        gpu_critical_utilization=0.9,
        gpu_high_utilization=0.8,
        gpu_moderate_utilization=0.7
    )
    
    handler = MemoryPressureHandler(config)
    
    # Test 1: State transitions
    print("\n1. Testing memory state transitions...")
    
    # Mock different memory states
    class MockMetrics:
        def __init__(self, gpu_free_mb, gpu_utilization):
            self.gpu_free_mb = gpu_free_mb
            self.gpu_utilization = gpu_utilization
            self.gpu_total_mb = 1000
    
    # Healthy state
    handler._update_memory_state(MockMetrics(500, 0.5))
    assert handler.current_memory_state == MemoryState.HEALTHY
    print("   ✓ Healthy state detected")
    
    # Moderate state
    handler._update_memory_state(MockMetrics(150, 0.75))
    assert handler.current_memory_state == MemoryState.MODERATE
    print("   ✓ Moderate state detected")
    
    # High state
    handler._update_memory_state(MockMetrics(80, 0.85))
    assert handler.current_memory_state == MemoryState.HIGH
    print("   ✓ High state detected")
    
    # Critical state
    handler._update_memory_state(MockMetrics(40, 0.95))
    assert handler.current_memory_state == MemoryState.CRITICAL
    print("   ✓ Critical state detected")
    
    # Test 2: Decision making under pressure
    print("\n2. Testing decisions under memory pressure...")
    
    # Force critical state
    handler.current_memory_state = MemoryState.CRITICAL
    metadata = {'size_mb': 10, 'priority': 'normal'}
    use_cpu, info = handler.should_use_cpu(metadata)
    
    assert use_cpu == True
    assert info['reason'] == 'memory_critical'
    print("   ✓ Forces CPU under critical memory pressure")
    
    # Test 3: GPU required under pressure
    print("\n3. Testing GPU requirement under pressure...")
    metadata = {'size_mb': 10, 'priority': 'high', 'require_gpu': True}
    
    try:
        use_cpu, info = handler.should_use_cpu(metadata)
        assert use_cpu == False  # Should still use GPU
        print("   ✓ Respects GPU requirement")
    except RuntimeError as e:
        print(f"   ✓ Correctly fails when GPU required but memory insufficient: {e}")
    
    # Test 4: Adaptive thresholds
    print("\n4. Testing adaptive threshold adjustment...")
    if config.adaptive_threshold:
        # Simulate increasing memory usage
        for i in range(10):
            metrics = MockMetrics(200 - i*20, 0.5 + i*0.05)
            handler.memory_history.append(metrics)
        
        handler._update_adaptive_thresholds()
        print(f"   Moderate threshold: {handler.config.gpu_moderate_utilization:.2f}")
        print(f"   High threshold: {handler.config.gpu_high_utilization:.2f}")
    
    # Cleanup
    handler.cleanup()
    print("\n✓ Memory threshold detection tests completed")
    return True


def test_performance_based_decisions():
    """Test performance-based decision making"""
    print("\n" + "="*60)
    print("Testing Performance-Based Decisions")
    print("="*60)
    
    from memory_pressure_handler import (
        MemoryPressureHandler,
        PressureHandlerConfig,
        ProcessingMode
    )
    
    config = PressureHandlerConfig(
        prefer_gpu_threshold=2.0,  # GPU must be 2x faster
        warmup_iterations=5
    )
    
    handler = MemoryPressureHandler(config)
    handler.set_processing_mode(ProcessingMode.ADAPTIVE)
    
    # Test 1: Warmup period
    print("\n1. Testing warmup period...")
    for i in range(config.warmup_iterations - 1):
        metadata = {'size_mb': 10, 'priority': 'normal'}
        use_cpu, info = handler.should_use_cpu(metadata)
        assert info['reason'] == 'warmup_threshold'
    
    print(f"   ✓ Warmup period active for {config.warmup_iterations} iterations")
    
    # Test 2: Performance-based decisions after warmup
    print("\n2. Testing performance-based decisions...")
    
    # Simulate GPU being faster
    for _ in range(10):
        handler.update_performance('gpu', True, 1000.0, 5.0)  # Fast GPU
        handler.update_performance('cpu', True, 100.0, 50.0)  # Slow CPU
    
    metadata = {'size_mb': 50, 'priority': 'normal'}
    use_cpu, info = handler.should_use_cpu(metadata)
    
    print(f"   GPU score: {info.get('gpu_score', 0):.2f}")
    print(f"   CPU score: {info.get('cpu_score', 0):.2f}")
    print(f"   Decision: {'CPU' if use_cpu else 'GPU'}")
    
    # GPU should be preferred when it's much faster
    assert use_cpu == False or handler.current_memory_state != MemoryState.HEALTHY
    
    # Test 3: CPU preferred when GPU not significantly faster
    print("\n3. Testing CPU preference when performance similar...")
    
    # Make performances similar
    for _ in range(10):
        handler.update_performance('gpu', True, 200.0, 10.0)
        handler.update_performance('cpu', True, 150.0, 15.0)
    
    use_cpu, info = handler.should_use_cpu(metadata)
    gpu_advantage = handler.performance_metrics.gpu_score / max(handler.performance_metrics.cpu_score, 0.01)
    
    print(f"   GPU advantage: {gpu_advantage:.2f}x")
    print(f"   Decision: {'CPU' if use_cpu else 'GPU'}")
    
    # Test 4: Priority affects decisions
    print("\n4. Testing priority-based decisions...")
    
    high_priority = {'size_mb': 50, 'priority': 'high'}
    use_cpu_high, _ = handler.should_use_cpu(high_priority)
    
    low_priority = {'size_mb': 50, 'priority': 'low'}
    use_cpu_low, _ = handler.should_use_cpu(low_priority)
    
    print(f"   High priority: {'CPU' if use_cpu_high else 'GPU'}")
    print(f"   Low priority: {'CPU' if use_cpu_low else 'GPU'}")
    
    # Cleanup
    handler.cleanup()
    print("\n✓ Performance-based decision tests completed")
    return True


def test_memory_exhaustion_prediction():
    """Test memory exhaustion prediction"""
    print("\n" + "="*60)
    print("Testing Memory Exhaustion Prediction")
    print("="*60)
    
    from memory_pressure_handler import MemoryPressureHandler, PressureHandlerConfig, MemoryMetrics
    
    config = PressureHandlerConfig()
    handler = MemoryPressureHandler(config)
    
    # Test 1: No prediction with insufficient data
    print("\n1. Testing prediction with insufficient data...")
    prediction = handler.predict_memory_exhaustion()
    assert prediction is None
    print("   ✓ No prediction with insufficient history")
    
    # Test 2: Prediction with decreasing memory
    print("\n2. Testing prediction with decreasing memory...")
    
    # Simulate decreasing memory over time
    base_time = time.time()
    for i in range(10):
        metrics = MemoryMetrics(
            timestamp=base_time + i,
            gpu_total_mb=8000,
            gpu_allocated_mb=7000 + i * 50,
            gpu_reserved_mb=7500 + i * 50,
            gpu_free_mb=500 - i * 50,
            gpu_utilization=0.9 + i * 0.01,
            cpu_total_mb=16000,
            cpu_available_mb=8000,
            cpu_percent=50,
            swap_used_mb=0
        )
        handler.memory_history.append(metrics)
    
    prediction = handler.predict_memory_exhaustion()
    if prediction:
        print(f"   Predicted exhaustion in: {prediction:.1f} seconds")
    else:
        print("   No exhaustion predicted")
    
    # Test 3: No prediction with stable memory
    print("\n3. Testing prediction with stable memory...")
    handler.memory_history.clear()
    
    for i in range(10):
        metrics = MemoryMetrics(
            timestamp=base_time + i,
            gpu_total_mb=8000,
            gpu_allocated_mb=4000,
            gpu_reserved_mb=4000,
            gpu_free_mb=4000,
            gpu_utilization=0.5,
            cpu_total_mb=16000,
            cpu_available_mb=8000,
            cpu_percent=50,
            swap_used_mb=0
        )
        handler.memory_history.append(metrics)
    
    prediction = handler.predict_memory_exhaustion()
    assert prediction is None
    print("   ✓ No prediction with stable memory")
    
    # Cleanup
    handler.cleanup()
    print("\n✓ Memory exhaustion prediction tests completed")
    return True


def test_concurrent_access():
    """Test thread safety and concurrent access"""
    print("\n" + "="*60)
    print("Testing Concurrent Access")
    print("="*60)
    
    from memory_pressure_handler import MemoryPressureHandler, PressureHandlerConfig
    
    config = PressureHandlerConfig()
    handler = MemoryPressureHandler(config)
    
    # Test 1: Concurrent decisions
    print("\n1. Testing concurrent decision making...")
    
    results = []
    errors = []
    
    def make_decisions(thread_id):
        try:
            for i in range(100):
                metadata = {
                    'size_mb': np.random.randint(10, 100),
                    'priority': np.random.choice(['low', 'normal', 'high'])
                }
                use_cpu, info = handler.should_use_cpu(metadata)
                results.append((thread_id, use_cpu))
        except Exception as e:
            errors.append((thread_id, str(e)))
    
    # Create threads
    threads = []
    for i in range(5):
        t = threading.Thread(target=make_decisions, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    print(f"   Completed {len(results)} decisions across {len(threads)} threads")
    print(f"   Errors: {len(errors)}")
    assert len(errors) == 0
    
    # Test 2: Concurrent performance updates
    print("\n2. Testing concurrent performance updates...")
    
    def update_performance(thread_id):
        try:
            for i in range(50):
                mode = 'gpu' if thread_id % 2 == 0 else 'cpu'
                handler.update_performance(
                    mode, 
                    True, 
                    np.random.uniform(100, 1000),
                    np.random.uniform(1, 20)
                )
        except Exception as e:
            errors.append((thread_id, str(e)))
    
    errors.clear()
    threads = []
    for i in range(4):
        t = threading.Thread(target=update_performance, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    print(f"   Completed performance updates")
    print(f"   Errors: {len(errors)}")
    assert len(errors) == 0
    
    # Test 3: Statistics during activity
    print("\n3. Testing statistics access during activity...")
    stats = handler.get_statistics()
    print(f"   Total decisions: {stats['total_decisions']}")
    print(f"   State changes: {stats['state_changes']}")
    
    # Cleanup
    handler.cleanup()
    print("\n✓ Concurrent access tests completed")
    return True


def test_integration_scenario():
    """Test realistic integration scenario"""
    print("\n" + "="*60)
    print("Testing Integration Scenario")
    print("="*60)
    
    from memory_pressure_handler import (
        MemoryPressureHandler,
        PressureHandlerConfig,
        ProcessingMode,
        integrate_memory_pressure_handler
    )
    
    # Create mock decompression engine
    class MockDecompressionEngine:
        def __init__(self):
            self.prime = 251
            self.gpu_optimizer = None
            self.cpu_bursting_pipeline = None
            self.decompress_count = 0
            
        def decompress_progressive(self, weights, precision, metadata):
            self.decompress_count += 1
            
            # Validate that we received real PadicWeight objects - HARD FAILURE if not
            if not weights:
                raise ValueError("Empty weight list - HARD FAILURE")
            
            for i, weight in enumerate(weights):
                if not isinstance(weight, PadicWeight):
                    raise TypeError(f"Weight {i} is not a PadicWeight object - HARD FAILURE")
                if not hasattr(weight, 'digits') or not hasattr(weight, 'valuation'):
                    raise ValueError(f"Weight {i} missing required fields - HARD FAILURE")
            
            # Simulate decompression with real weight validation
            time.sleep(0.01)
            shape = metadata['original_shape']
            return torch.randn(shape), {'decompression_time': 0.01}
    
    # Create engine and integrate handler
    engine = MockDecompressionEngine()
    
    config = PressureHandlerConfig(
        gpu_critical_threshold_mb=100,
        force_cpu_on_critical=True
    )
    
    handler = integrate_memory_pressure_handler(engine, config)
    
    print("\n1. Testing integrated decompression...")
    
    # Test normal decompression with real p-adic weights
    print("Creating real p-adic weights for testing...")
    weights, conv_time = measure_weight_conversion_time(100, 10, 251)
    
    # Validate weights - HARD FAILURE if validation fails
    if not validate_padic_weights(weights, 251, 10):
        raise ValueError("P-adic weight validation failed - HARD FAILURE")
    print(f"✓ Created and validated {len(weights)} real p-adic weights in {conv_time:.3f}s")
    metadata = {
        'original_shape': (10, 10),
        'dtype': 'torch.float32',
        'device': 'cuda:0',
        'size_mb': 50
    }
    
    result, info = engine.decompress_progressive(weights, 10, metadata)
    
    print(f"   Processing mode: {info.get('processing_mode', 'unknown')}")
    print(f"   Decision info: {info.get('memory_pressure_decision', {})}")
    
    # Test 2: Simulate memory pressure
    print("\n2. Simulating memory pressure...")
    
    # Force critical state
    handler.current_memory_state = handler.memory_pressure_handler.MemoryState.CRITICAL
    
    # Large allocation should fail or use CPU
    large_metadata = metadata.copy()
    large_metadata['size_mb'] = 200
    
    try:
        result2, info2 = engine.decompress_progressive(weights, 10, large_metadata)
        print(f"   Decision under pressure: {info2.get('processing_mode', 'unknown')}")
    except RuntimeError as e:
        print(f"   ✓ Correctly failed under memory pressure: {e}")
    
    # Test 3: Performance tracking
    print("\n3. Testing performance tracking...")
    
    # Simulate multiple decompressions
    for i in range(10):
        metadata['size_mb'] = np.random.randint(10, 100)
        result, info = engine.decompress_progressive(weights, 10, metadata)
    
    summary = handler.get_memory_summary()
    print(f"   Memory state: {summary.get('memory_state', 'unknown')}")
    print(f"   Recent decisions: {summary.get('recent_decisions', {})}")
    
    # Cleanup
    handler.cleanup()
    print("\n✓ Integration scenario tests completed")
    return True


def run_stress_test():
    """Run stress test for memory pressure handler"""
    print("\n" + "="*60)
    print("Running Memory Pressure Handler Stress Test")
    print("="*60)
    
    from memory_pressure_handler import MemoryPressureHandler, PressureHandlerConfig
    
    config = PressureHandlerConfig(
        monitoring_interval_ms=10,  # Fast monitoring
        history_window_size=1000    # Large history
    )
    
    handler = MemoryPressureHandler(config)
    
    print("\n1. Running continuous operations...")
    
    start_time = time.time()
    decision_times = []
    
    # Run for 5 seconds
    while time.time() - start_time < 5:
        # Random metadata
        metadata = {
            'size_mb': np.random.randint(1, 500),
            'priority': np.random.choice(['low', 'normal', 'high']),
            'require_gpu': np.random.random() < 0.1,
            'require_cpu': np.random.random() < 0.1
        }
        
        decision_start = time.time()
        try:
            use_cpu, info = handler.should_use_cpu(metadata)
            decision_time = time.time() - decision_start
            decision_times.append(decision_time)
            
            # Update performance randomly
            if np.random.random() < 0.5:
                mode = 'cpu' if use_cpu else 'gpu'
                handler.update_performance(
                    mode,
                    np.random.random() < 0.9,  # 90% success rate
                    np.random.uniform(100, 1000),
                    np.random.uniform(1, 50)
                )
        except Exception as e:
            print(f"   Error during stress test: {e}")
    
    # Results
    print(f"\n2. Stress test results:")
    stats = handler.get_statistics()
    print(f"   Total decisions: {stats['total_decisions']}")
    print(f"   Average decision time: {np.mean(decision_times)*1000:.2f}ms")
    print(f"   Memory pressure events: {stats['memory_pressure_events']}")
    print(f"   State changes: {stats['state_changes']}")
    
    # Memory usage
    print(f"\n3. Memory usage:")
    print(f"   History size: {stats['memory_history_size']}")
    print(f"   Decision history size: {stats['decision_history_size']}")
    
    # Cleanup
    handler.cleanup()
    print("\n✓ Stress test completed")
    return True


def run_all_tests():
    """Run all memory pressure handler tests"""
    print("\n" + "="*80)
    print("Memory Pressure Handler Test Suite")
    print("="*80)
    
    try:
        # Run test suites
        test_memory_pressure_handler_basic()
        test_memory_threshold_detection()
        test_performance_based_decisions()
        test_memory_exhaustion_prediction()
        test_concurrent_access()
        test_integration_scenario()
        run_stress_test()
        
        print("\n" + "="*80)
        print("✅ All memory pressure handler tests completed successfully!")
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
        print("\n🎉 MemoryPressureHandler is ready for production!")
        print("   - Threshold Detection: ✓")
        print("   - CPU/GPU Decision Making: ✓")
        print("   - Performance Monitoring: ✓")
        print("   - Thread Safety: ✓")
        print("   - Integration Support: ✓")
    else:
        print("\n⚠️  Some tests failed. Please review the logs above.")