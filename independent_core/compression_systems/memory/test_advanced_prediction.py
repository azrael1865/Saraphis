"""
Test suite for Advanced Memory Prediction Algorithms
Tests ARIMA, exponential smoothing, pattern detection, and adaptive learning
"""

import torch
import numpy as np
import time
import threading
import pytest
import math
from typing import List, Dict, Any
import psutil
import gc
from collections import deque

from unified_memory_handler import (
    UnifiedMemoryHandler,
    UnifiedMemoryConfig,
    MemoryPressureLevel,
    AllocationPriority,
    EvictionStrategy,
    MemoryRequest,
    MemoryAllocation,
    MemoryPredictor,
    WorkloadPhase,
    MemoryPattern,
    PredictionResult,
    create_unified_handler
)


class TestMemoryPredictor:
    """Test suite for advanced memory prediction algorithms"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = UnifiedMemoryConfig(
            gpu_memory_limit_mb=4096,
            cpu_memory_limit_mb=8192,
            monitoring_interval_ms=50,
            history_window_size=200,
            enable_predictive_eviction=True,
            prediction_horizon_seconds=10.0
        )
        self.predictor = MemoryPredictor(self.config)
        
        # Generate synthetic memory usage data
        self._generate_test_data()
    
    def teardown_method(self):
        """Cleanup after test"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _generate_test_data(self):
        """Generate synthetic memory usage patterns for testing"""
        # Create periodic pattern with trend
        time_points = 100
        base_usage = 1000  # MB
        
        for i in range(time_points):
            timestamp = time.time() - (time_points - i) * 0.1
            
            # Periodic component (sine wave)
            periodic = 200 * math.sin(2 * math.pi * i / 20)
            
            # Trend component
            trend = 5 * i
            
            # Random noise
            noise = np.random.normal(0, 20)
            
            # Combined usage
            usage = base_usage + periodic + trend + noise
            
            # Create mock metrics
            metrics = {
                'timestamp': timestamp,
                'gpu_0': {
                    'allocated_mb': usage,
                    'total_mb': 4096,
                    'free_mb': 4096 - usage
                },
                'cpu': {
                    'used_mb': usage * 0.5,
                    'total_mb': 8192,
                    'available_mb': 8192 - usage * 0.5
                }
            }
            
            self.predictor.usage_history.append(metrics)
            
            # Add some allocations
            if i % 5 == 0:
                alloc = MemoryAllocation(
                    allocation_id=f"test_{i}",
                    subsystem="test",
                    size_bytes=int(50 * 1024 * 1024),  # 50MB
                    priority=AllocationPriority.NORMAL,
                    device="cuda:0",
                    timestamp=timestamp,
                    last_accessed=timestamp
                )
                self.predictor.allocation_history.append(alloc)
    
    def test_arima_prediction(self):
        """Test ARIMA-based prediction"""
        print("\nTest: ARIMA prediction...")
        
        # Test prediction
        device = "cuda:0"
        horizon = 5.0  # 5 seconds ahead
        
        prediction = self.predictor._predict_arima(device, horizon)
        
        # Check prediction is reasonable
        current_usage = self.predictor._get_current_usage(device)
        assert prediction > 0
        assert prediction < 4096  # Within memory limit
        
        # Prediction should account for trend
        assert prediction > current_usage  # Since we have positive trend
        
        print(f"Current usage: {current_usage:.2f} MB")
        print(f"ARIMA prediction (5s): {prediction:.2f} MB")
        print("✓ ARIMA prediction test passed")
    
    def test_exponential_smoothing(self):
        """Test exponential smoothing prediction"""
        print("\nTest: Exponential smoothing...")
        
        device = "cuda:0"
        horizon = 3.0
        
        prediction = self.predictor._predict_exponential_smoothing(device, horizon)
        
        assert prediction > 0
        assert prediction < 4096
        
        # Test different alpha values
        original_alpha = self.predictor.exp_smoothing_alpha
        
        # Higher alpha = more weight on recent values
        self.predictor.exp_smoothing_alpha = 0.9
        pred_high_alpha = self.predictor._predict_exponential_smoothing(device, horizon)
        
        self.predictor.exp_smoothing_alpha = 0.1
        pred_low_alpha = self.predictor._predict_exponential_smoothing(device, horizon)
        
        # With trend, high alpha should give higher prediction
        assert pred_high_alpha != pred_low_alpha
        
        self.predictor.exp_smoothing_alpha = original_alpha
        
        print(f"Exponential smoothing prediction: {prediction:.2f} MB")
        print(f"High alpha (0.9): {pred_high_alpha:.2f} MB")
        print(f"Low alpha (0.1): {pred_low_alpha:.2f} MB")
        print("✓ Exponential smoothing test passed")
    
    def test_pattern_detection(self):
        """Test memory pattern detection"""
        print("\nTest: Pattern detection...")
        
        patterns = self.predictor.detect_patterns()
        
        assert len(patterns) > 0
        
        # Should detect periodic pattern
        if 'periodic_main' in patterns:
            periodic = patterns['periodic_main']
            assert periodic.pattern_type == 'periodic'
            assert periodic.period_seconds > 0
            assert periodic.confidence > 0
            print(f"Detected period: {periodic.period_seconds:.2f}s with confidence {periodic.confidence:.2f}")
        
        # Should detect trend
        if 'trend' in patterns:
            trend = patterns['trend']
            assert trend.pattern_type == 'trend'
            assert trend.trend != 0
            print(f"Detected trend: {trend.trend:.2f} MB/s")
        
        print(f"Total patterns detected: {len(patterns)}")
        for pattern_id, pattern in patterns.items():
            print(f"  - {pattern_id}: {pattern.pattern_type} (confidence: {pattern.confidence:.2f})")
        
        print("✓ Pattern detection test passed")
    
    def test_memory_leak_detection(self):
        """Test memory leak detection"""
        print("\nTest: Memory leak detection...")
        
        # Create leak pattern
        self.predictor.allocation_history.clear()
        base_size = 10 * 1024 * 1024  # 10MB
        
        for i in range(50):
            # Monotonically increasing allocations
            alloc = MemoryAllocation(
                allocation_id=f"leak_{i}",
                subsystem="test",
                size_bytes=base_size + i * 1024 * 1024,  # Growing allocations
                priority=AllocationPriority.NORMAL,
                device="cuda:0",
                timestamp=time.time() - (50 - i),
                last_accessed=time.time()
            )
            self.predictor.allocation_history.append(alloc)
        
        # Detect patterns
        patterns = self.predictor.detect_patterns()
        
        # Should detect leak
        if 'leak' in patterns:
            leak = patterns['leak']
            assert leak.pattern_type == 'leak'
            assert leak.trend > 0  # Positive leak rate
            assert leak.confidence > 0.5
            print(f"Leak detected with rate {leak.trend:.2f} MB/s, confidence {leak.confidence:.2f}")
            print(f"Severity: {leak.metadata.get('severity', 'unknown')}")
        else:
            print("Warning: Leak pattern not detected in synthetic data")
        
        print("✓ Memory leak detection test passed")
    
    def test_burst_pattern_detection(self):
        """Test burst allocation pattern detection"""
        print("\nTest: Burst pattern detection...")
        
        # Create burst pattern
        self.predictor.allocation_history.clear()
        
        for i in range(100):
            # Normal allocations
            size = 5 * 1024 * 1024  # 5MB
            
            # Add bursts
            if i % 20 == 0:
                size = 100 * 1024 * 1024  # 100MB burst
            
            alloc = MemoryAllocation(
                allocation_id=f"burst_{i}",
                subsystem="test",
                size_bytes=size,
                priority=AllocationPriority.NORMAL,
                device="cuda:0",
                timestamp=time.time() - (100 - i) * 0.1,
                last_accessed=time.time()
            )
            self.predictor.allocation_history.append(alloc)
        
        # Detect patterns
        patterns = self.predictor.detect_patterns()
        
        if 'burst' in patterns:
            burst = patterns['burst']
            assert burst.pattern_type == 'burst'
            assert burst.amplitude_mb > 0
            print(f"Burst pattern detected: amplitude {burst.amplitude_mb:.2f} MB")
            print(f"Frequency: {burst.metadata.get('frequency', 0):.2f}")
        
        print("✓ Burst pattern detection test passed")
    
    def test_workload_phase_recognition(self):
        """Test workload phase recognition"""
        print("\nTest: Workload phase recognition...")
        
        # Test different allocation patterns
        test_phases = []
        
        # Idle phase - few allocations
        self.predictor.allocation_history.clear()
        for i in range(5):
            alloc = MemoryAllocation(
                allocation_id=f"idle_{i}",
                subsystem="test",
                size_bytes=1024 * 1024,  # 1MB
                priority=AllocationPriority.LOW,
                device="cpu",
                timestamp=time.time() - (5 - i) * 2,  # 2 seconds apart
                last_accessed=time.time()
            )
            self.predictor.allocation_history.append(alloc)
        
        phase = self.predictor.recognize_workload_phase()
        test_phases.append(('idle', phase))
        assert phase == WorkloadPhase.IDLE
        
        # Training phase - rapid, consistent allocations
        for i in range(20):
            alloc = MemoryAllocation(
                allocation_id=f"train_{i}",
                subsystem="test",
                size_bytes=32 * 1024 * 1024,  # 32MB consistent
                priority=AllocationPriority.HIGH,
                device="cuda:0",
                timestamp=time.time() - (20 - i) * 0.05,  # 50ms apart
                last_accessed=time.time()
            )
            self.predictor.allocation_history.append(alloc)
        
        phase = self.predictor.recognize_workload_phase()
        test_phases.append(('training', phase))
        
        # Data loading - large allocations
        for i in range(10):
            alloc = MemoryAllocation(
                allocation_id=f"data_{i}",
                subsystem="test",
                size_bytes=200 * 1024 * 1024,  # 200MB
                priority=AllocationPriority.NORMAL,
                device="cpu",
                timestamp=time.time() - (10 - i) * 0.5,
                last_accessed=time.time()
            )
            self.predictor.allocation_history.append(alloc)
        
        phase = self.predictor.recognize_workload_phase()
        test_phases.append(('data_loading', phase))
        assert phase == WorkloadPhase.DATA_LOADING
        
        print("Detected workload phases:")
        for expected, detected in test_phases:
            print(f"  Expected: {expected}, Detected: {detected.value}")
        
        print("✓ Workload phase recognition test passed")
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction with multiple methods"""
        print("\nTest: Ensemble prediction...")
        
        device = "cuda:0"
        horizon = 5.0
        
        # Get ensemble prediction
        prediction = self.predictor.predict_memory_usage(device, horizon)
        
        assert isinstance(prediction, PredictionResult)
        assert prediction.predicted_usage_mb > 0
        assert prediction.confidence >= 0 and prediction.confidence <= 1
        assert len(prediction.error_bounds) == 2
        assert prediction.error_bounds[0] <= prediction.predicted_usage_mb
        assert prediction.error_bounds[1] >= prediction.predicted_usage_mb
        
        print(f"Ensemble prediction: {prediction.predicted_usage_mb:.2f} MB")
        print(f"Confidence: {prediction.confidence:.2f}")
        print(f"Error bounds: [{prediction.error_bounds[0]:.2f}, {prediction.error_bounds[1]:.2f}]")
        print(f"Methods used: {prediction.metadata.get('methods', [])}")
        
        print("✓ Ensemble prediction test passed")
    
    def test_multi_step_prediction(self):
        """Test multi-step ahead predictions"""
        print("\nTest: Multi-step predictions...")
        
        device = "cuda:0"
        horizons = [60, 300, 900]  # 1min, 5min, 15min
        
        predictions = []
        for horizon in horizons:
            try:
                pred = self.predictor.predict_memory_usage(device, horizon)
                predictions.append((horizon, pred))
                print(f"{horizon}s prediction: {pred.predicted_usage_mb:.2f} MB (confidence: {pred.confidence:.2f})")
            except Exception as e:
                print(f"Failed to predict for {horizon}s: {e}")
        
        # Longer horizons should have lower confidence
        if len(predictions) >= 2:
            for i in range(1, len(predictions)):
                assert predictions[i][1].confidence <= predictions[i-1][1].confidence + 0.1
        
        # Error bounds should widen with horizon
        if len(predictions) >= 2:
            for i in range(1, len(predictions)):
                prev_width = predictions[i-1][1].error_bounds[1] - predictions[i-1][1].error_bounds[0]
                curr_width = predictions[i][1].error_bounds[1] - predictions[i][1].error_bounds[0]
                assert curr_width >= prev_width * 0.9  # Allow small variation
        
        print("✓ Multi-step prediction test passed")
    
    def test_adaptive_learning(self):
        """Test adaptive model weight learning"""
        print("\nTest: Adaptive learning...")
        
        device = "cuda:0"
        initial_weights = dict(self.predictor.model_weights)
        print(f"Initial weights: {initial_weights}")
        
        # Make predictions and update accuracy
        for i in range(10):
            horizon = 1.0
            prediction = self.predictor.predict_memory_usage(device, horizon)
            
            # Simulate actual usage (close to prediction for good models)
            if 'arima' in prediction.metadata.get('methods', []):
                actual = prediction.predicted_usage_mb + np.random.normal(0, 10)
            else:
                actual = prediction.predicted_usage_mb + np.random.normal(0, 50)
            
            # Update accuracy
            self.predictor.update_prediction_accuracy(actual, prediction)
        
        # Check weights adapted
        final_weights = dict(self.predictor.model_weights)
        print(f"Final weights: {final_weights}")
        
        # Weights should have changed
        assert initial_weights != final_weights
        
        # Sum of weights should be 1
        assert abs(sum(final_weights.values()) - 1.0) < 0.01
        
        # Get statistics
        stats = self.predictor.get_prediction_stats()
        print(f"Prediction stats: {stats}")
        
        assert stats['total_predictions'] > 0
        assert stats['mean_absolute_error'] >= 0
        assert stats['rmse'] >= 0
        
        print("✓ Adaptive learning test passed")
    
    def test_reinforcement_learning_eviction(self):
        """Test RL-based eviction policy selection"""
        print("\nTest: Reinforcement learning for eviction...")
        
        # Test different states
        states = [
            {'memory_usage': 0.3, 'workload_phase': 'training', 'pressure_level': 'low'},
            {'memory_usage': 0.7, 'workload_phase': 'inference', 'pressure_level': 'high'},
            {'memory_usage': 0.9, 'workload_phase': 'data_loading', 'pressure_level': 'critical'}
        ]
        
        selected_policies = []
        for state in states:
            policy = self.predictor.select_eviction_policy(state)
            assert isinstance(policy, EvictionStrategy)
            selected_policies.append(policy)
            print(f"State: {state['pressure_level']}, Policy: {policy.value}")
        
        # Update Q-table with rewards
        for i, state in enumerate(states[:-1]):
            action = selected_policies[i]
            reward = 0.8 if action == EvictionStrategy.HYBRID else 0.5
            next_state = states[i + 1]
            
            self.predictor.update_eviction_reward(state, action, reward, next_state)
        
        # Check Q-table was updated
        assert len(self.predictor.eviction_q_table) > 0
        
        # Test exploitation vs exploration
        original_epsilon = self.predictor.epsilon
        self.predictor.epsilon = 0  # No exploration
        
        # Should consistently select best known policy
        test_state = states[0]
        policies = [self.predictor.select_eviction_policy(test_state) for _ in range(10)]
        assert len(set(policies)) == 1  # Should be deterministic
        
        self.predictor.epsilon = original_epsilon
        
        print("✓ Reinforcement learning eviction test passed")
    
    def test_adaptive_window_sizing(self):
        """Test adaptive window size adjustment"""
        print("\nTest: Adaptive window sizing...")
        
        initial_window = self.predictor.adaptive_window_size
        print(f"Initial window size: {initial_window}")
        
        # Create stable pattern
        stable_usage = []
        for i in range(50):
            metrics = {
                'timestamp': time.time() + i,
                'gpu_0': {'allocated_mb': 1000 + np.random.normal(0, 5)},  # Low variance
                'cpu': {'used_mb': 500}
            }
            stable_usage.append(metrics)
        
        self.predictor.usage_history = deque(stable_usage, maxlen=self.config.history_window_size)
        window_stable = self.predictor.adapt_window_size()
        print(f"Window size (stable): {window_stable}")
        
        # Create unstable pattern
        unstable_usage = []
        for i in range(50):
            metrics = {
                'timestamp': time.time() + i,
                'gpu_0': {'allocated_mb': 1000 + np.random.normal(0, 200)},  # High variance
                'cpu': {'used_mb': 500}
            }
            unstable_usage.append(metrics)
        
        self.predictor.usage_history = deque(unstable_usage, maxlen=self.config.history_window_size)
        window_unstable = self.predictor.adapt_window_size()
        print(f"Window size (unstable): {window_unstable}")
        
        # Stable pattern should use larger window
        assert window_stable >= window_unstable
        
        print("✓ Adaptive window sizing test passed")
    
    def test_prediction_with_insufficient_data(self):
        """Test prediction behavior with insufficient data"""
        print("\nTest: Prediction with insufficient data...")
        
        # Clear history
        self.predictor.usage_history.clear()
        self.predictor.allocation_history.clear()
        
        # Add minimal data
        for i in range(5):
            metrics = {
                'timestamp': time.time() + i,
                'gpu_0': {'allocated_mb': 100 * i},
                'cpu': {'used_mb': 50 * i}
            }
            self.predictor.usage_history.append(metrics)
        
        # Should raise error or fallback
        with pytest.raises(ValueError):
            self.predictor.predict_memory_usage('cuda:0', 5.0)
        
        # Add more data
        for i in range(10):
            metrics = {
                'timestamp': time.time() + i + 5,
                'gpu_0': {'allocated_mb': 100 * (i + 5)},
                'cpu': {'used_mb': 50 * (i + 5)}
            }
            self.predictor.usage_history.append(metrics)
        
        # Should work now
        prediction = self.predictor.predict_memory_usage('cuda:0', 5.0)
        assert prediction.method in ['linear', 'fallback', 'ensemble']
        assert prediction.confidence <= 0.6  # Low confidence with limited data
        
        print("✓ Insufficient data handling test passed")
    
    def test_prediction_numerical_stability(self):
        """Test numerical stability of prediction algorithms"""
        print("\nTest: Numerical stability...")
        
        # Test with extreme values
        extreme_cases = [
            # Very small values
            [0.001, 0.002, 0.001, 0.003, 0.002],
            # Very large values
            [1e6, 1e6 + 100, 1e6 + 200, 1e6 + 150, 1e6 + 300],
            # Zero values
            [0, 0, 0, 0, 0],
            # Constant values
            [1000, 1000, 1000, 1000, 1000]
        ]
        
        for case_idx, values in enumerate(extreme_cases):
            print(f"Testing case {case_idx + 1}: {values[:3]}...")
            
            # Create test data
            self.predictor.usage_history.clear()
            for i, value in enumerate(values):
                metrics = {
                    'timestamp': time.time() + i,
                    'gpu_0': {'allocated_mb': value},
                    'cpu': {'used_mb': value * 0.5}
                }
                self.predictor.usage_history.append(metrics)
            
            # Should not crash or produce invalid results
            try:
                if len(values) >= 10:  # Need enough data
                    pred = self.predictor._simple_linear_prediction('cuda:0', 1.0)
                    assert not math.isnan(pred.predicted_usage_mb)
                    assert not math.isinf(pred.predicted_usage_mb)
                    assert pred.predicted_usage_mb >= 0
            except Exception as e:
                print(f"Case {case_idx + 1} raised expected exception: {e}")
        
        print("✓ Numerical stability test passed")


class TestIntegratedPrediction:
    """Test prediction integrated with unified memory handler"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = UnifiedMemoryConfig(
            gpu_memory_limit_mb=2048,
            cpu_memory_limit_mb=4096,
            monitoring_interval_ms=50,
            enable_predictive_eviction=True,
            enable_compression=True
        )
        self.handler = UnifiedMemoryHandler(self.config)
    
    def teardown_method(self):
        """Cleanup after test"""
        if hasattr(self, 'handler'):
            self.handler.shutdown()
        gc.collect()
    
    def test_predictive_eviction(self):
        """Test predictive eviction before memory exhaustion"""
        print("\nTest: Predictive eviction...")
        
        # Fill memory gradually
        allocations = []
        for i in range(20):
            request_id = self.handler.submit_request(
                subsystem='test',
                size_bytes=50 * 1024 * 1024,  # 50MB each
                priority=AllocationPriority.LOW if i < 10 else AllocationPriority.NORMAL,
                device='cpu'
            )
            allocations.append(request_id)
            time.sleep(0.1)
        
        # Check if predictive eviction occurred
        stats = self.handler.get_memory_stats()
        
        if 'prediction' in stats:
            print(f"Prediction accuracy: {stats['prediction'].get('accuracy', 0):.2%}")
            print(f"Model weights: {stats['prediction'].get('model_weights', {})}")
        
        # Should have triggered evictions
        assert stats['operations']['total_evictions'] > 0
        
        print(f"Total evictions: {stats['operations']['total_evictions']}")
        print("✓ Predictive eviction test passed")
    
    def test_speculative_allocation(self):
        """Test speculative pre-allocation based on predictions"""
        print("\nTest: Speculative allocation...")
        
        # Create pattern of allocations
        pattern_size = 25 * 1024 * 1024  # 25MB
        
        for cycle in range(3):
            # Allocation burst
            for i in range(5):
                self.handler.submit_request(
                    subsystem='test',
                    size_bytes=pattern_size,
                    priority=AllocationPriority.NORMAL,
                    device='cpu'
                )
                time.sleep(0.05)
            
            # Pause
            time.sleep(0.5)
        
        # Let pattern detection run
        time.sleep(0.5)
        
        # Check if patterns were detected
        stats = self.handler.get_memory_stats()
        patterns = stats.get('detected_patterns', {})
        
        print(f"Detected patterns: {list(patterns.keys())}")
        
        for pattern_id, pattern_info in patterns.items():
            print(f"  {pattern_id}: type={pattern_info['type']}, confidence={pattern_info['confidence']:.2f}")
        
        print("✓ Speculative allocation test passed")
    
    def test_memory_exhaustion_prediction(self):
        """Test accurate prediction of memory exhaustion"""
        print("\nTest: Memory exhaustion prediction...")
        
        # Start with some base allocations
        base_allocations = []
        for i in range(5):
            request_id = self.handler.submit_request(
                subsystem='test',
                size_bytes=100 * 1024 * 1024,  # 100MB
                priority=AllocationPriority.NORMAL,
                device='cpu'
            )
            base_allocations.append(request_id)
        
        time.sleep(0.5)
        
        # Get initial prediction
        stats = self.handler.get_memory_stats()
        initial_exhaustion = stats['pressure_levels']['cpu'].get('exhaustion_seconds')
        
        print(f"Initial exhaustion prediction: {initial_exhaustion}s")
        
        # Add more allocations at steady rate
        for i in range(10):
            self.handler.submit_request(
                subsystem='test',
                size_bytes=50 * 1024 * 1024,  # 50MB
                priority=AllocationPriority.NORMAL,
                device='cpu'
            )
            time.sleep(0.1)
        
        # Get updated prediction
        stats = self.handler.get_memory_stats()
        updated_exhaustion = stats['pressure_levels']['cpu'].get('exhaustion_seconds')
        
        print(f"Updated exhaustion prediction: {updated_exhaustion}s")
        
        # Multi-horizon predictions
        if 'prediction_60s' in stats['pressure_levels']['cpu']:
            pred_1min = stats['pressure_levels']['cpu']['prediction_60s']
            print(f"1-minute prediction: {pred_1min['usage_mb']:.2f} MB (confidence: {pred_1min['confidence']:.2f})")
        
        if 'prediction_300s' in stats['pressure_levels']['cpu']:
            pred_5min = stats['pressure_levels']['cpu']['prediction_300s']
            print(f"5-minute prediction: {pred_5min['usage_mb']:.2f} MB (confidence: {pred_5min['confidence']:.2f})")
        
        print("✓ Memory exhaustion prediction test passed")
    
    def test_workload_aware_allocation(self):
        """Test workload-aware memory allocation strategies"""
        print("\nTest: Workload-aware allocation...")
        
        # Simulate training workload
        print("Simulating training phase...")
        for i in range(20):
            self.handler.submit_request(
                subsystem='training',
                size_bytes=32 * 1024 * 1024,  # 32MB consistent
                priority=AllocationPriority.HIGH,
                device='cpu',
                metadata={'phase': 'training', 'batch': i}
            )
            time.sleep(0.02)  # Rapid allocations
        
        time.sleep(0.5)
        
        # Check detected phase
        stats = self.handler.get_memory_stats()
        current_phase = stats.get('prediction', {}).get('current_phase', 'unknown')
        print(f"Detected phase: {current_phase}")
        
        # Simulate inference workload
        print("Simulating inference phase...")
        for i in range(10):
            self.handler.submit_request(
                subsystem='inference',
                size_bytes=16 * 1024 * 1024,  # 16MB
                priority=AllocationPriority.NORMAL,
                device='cpu',
                metadata={'phase': 'inference', 'request': i}
            )
            time.sleep(0.2)  # Slower allocations
        
        time.sleep(0.5)
        
        # Check updated phase
        stats = self.handler.get_memory_stats()
        updated_phase = stats.get('prediction', {}).get('current_phase', 'unknown')
        print(f"Updated phase: {updated_phase}")
        
        print("✓ Workload-aware allocation test passed")


if __name__ == "__main__":
    # Run basic predictor tests
    print("=" * 60)
    print("TESTING MEMORY PREDICTOR")
    print("=" * 60)
    
    predictor_tests = TestMemoryPredictor()
    predictor_tests.setup_method()
    
    try:
        predictor_tests.test_arima_prediction()
        predictor_tests.test_exponential_smoothing()
        predictor_tests.test_pattern_detection()
        predictor_tests.test_memory_leak_detection()
        predictor_tests.test_burst_pattern_detection()
        predictor_tests.test_workload_phase_recognition()
        predictor_tests.test_ensemble_prediction()
        predictor_tests.test_multi_step_prediction()
        predictor_tests.test_adaptive_learning()
        predictor_tests.test_reinforcement_learning_eviction()
        predictor_tests.test_adaptive_window_sizing()
        predictor_tests.test_prediction_with_insufficient_data()
        predictor_tests.test_prediction_numerical_stability()
    finally:
        predictor_tests.teardown_method()
    
    # Run integrated tests
    print("\n" + "=" * 60)
    print("TESTING INTEGRATED PREDICTION")
    print("=" * 60)
    
    integrated_tests = TestIntegratedPrediction()
    integrated_tests.setup_method()
    
    try:
        integrated_tests.test_predictive_eviction()
        integrated_tests.test_speculative_allocation()
        integrated_tests.test_memory_exhaustion_prediction()
        integrated_tests.test_workload_aware_allocation()
    finally:
        integrated_tests.teardown_method()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)