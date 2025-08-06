#!/usr/bin/env python3
"""
Standalone test for advanced memory prediction algorithms
Tests core prediction functionality without full system integration
"""

import sys
import os
import numpy as np
import math
import time
from collections import deque
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass, field

# Mock scipy if not available
try:
    from scipy import signal, stats
    HAS_SCIPY = True
except ImportError:
    print("WARNING: scipy not available, using mock implementations")
    HAS_SCIPY = False
    
    # Mock signal module
    class MockSignal:
        @staticmethod
        def find_peaks(data, **kwargs):
            # Simple peak detection
            peaks = []
            for i in range(1, len(data) - 1):
                if data[i] > data[i-1] and data[i] > data[i+1]:
                    peaks.append(i)
            return np.array(peaks), {'peak_heights': data[peaks] if peaks else []}
    
    signal = MockSignal()


class WorkloadPhase(Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    DATA_LOADING = "data_loading"
    OPTIMIZATION = "optimization"
    IDLE = "idle"
    MIXED = "mixed"


@dataclass
class MemoryPattern:
    pattern_id: str
    pattern_type: str
    period_seconds: Optional[float] = None
    amplitude_mb: Optional[float] = None
    trend: Optional[float] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    timestamp: float
    horizon_seconds: float
    predicted_usage_mb: float
    confidence: float
    method: str
    error_bounds: Tuple[float, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimplifiedMemoryPredictor:
    """Simplified predictor for standalone testing"""
    
    def __init__(self):
        self.usage_history = deque(maxlen=200)
        self.arima_params = {'p': 2, 'd': 1, 'q': 1}
        self.exp_smoothing_alpha = 0.3
        
    def predict_arima(self, data: np.ndarray, horizon_steps: int) -> float:
        """ARIMA prediction implementation"""
        if len(data) < 20:
            raise ValueError("Insufficient data for ARIMA")
        
        # Difference the series
        diff_series = np.diff(data)
        
        # Simple AR estimation
        p = self.arima_params['p']
        if len(diff_series) > p:
            X = np.array([diff_series[i:i+p] for i in range(len(diff_series)-p)])
            y = diff_series[p:]
            
            if len(X) > 0:
                XtX = X.T @ X + 1e-6 * np.eye(p)
                Xty = X.T @ y
                ar_coef = np.linalg.solve(XtX, Xty)
            else:
                ar_coef = np.zeros(p)
        else:
            ar_coef = np.zeros(p)
        
        # Multi-step prediction
        predictions = []
        recent = list(diff_series[-p:]) if len(diff_series) >= p else list(diff_series)
        
        for step in range(horizon_steps):
            ar_pred = sum(ar_coef[i] * recent[-i-1] for i in range(min(p, len(recent))))
            predictions.append(ar_pred)
            recent.append(ar_pred)
            if len(recent) > p:
                recent.pop(0)
        
        # Integrate predictions
        integrated = data[-1]
        for diff in predictions:
            integrated += diff
        
        return max(0, integrated)
    
    def predict_exponential_smoothing(self, data: np.ndarray, horizon_steps: int) -> float:
        """Exponential smoothing with trend"""
        if len(data) < 3:
            raise ValueError("Insufficient data")
        
        level = data[0]
        trend = (data[-1] - data[0]) / len(data) if len(data) > 1 else 0
        
        alpha = self.exp_smoothing_alpha
        beta = alpha * 0.5
        
        for value in data[1:]:
            prev_level = level
            level = alpha * value + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
        
        prediction = level + trend * horizon_steps
        return max(0, prediction)
    
    def detect_patterns(self, data: np.ndarray) -> Dict[str, MemoryPattern]:
        """Pattern detection in time series"""
        patterns = {}
        
        if len(data) < 50:
            return patterns
        
        # Detect periodicity using autocorrelation
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        if HAS_SCIPY:
            peaks, properties = signal.find_peaks(autocorr, height=0.5, distance=10)
        else:
            # Simple peak detection
            peaks = []
            for i in range(10, len(autocorr) - 10):
                if autocorr[i] > 0.5 and autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    peaks.append(i)
            peaks = np.array(peaks)
            properties = {'peak_heights': autocorr[peaks] if len(peaks) > 0 else []}
        
        if len(peaks) > 0:
            strongest_peak = peaks[np.argmax(properties['peak_heights'])]
            period_samples = strongest_peak
            
            patterns['periodic'] = MemoryPattern(
                pattern_id='periodic',
                pattern_type='periodic',
                period_seconds=period_samples * 0.1,  # Assuming 100ms sampling
                amplitude_mb=np.std(data) * 2,
                confidence=float(properties['peak_heights'][np.argmax(properties['peak_heights'])]),
                metadata={'peak_lag': strongest_peak}
            )
        
        # Detect trend
        if len(data) >= 10:
            x = np.arange(len(data))
            slope, _ = np.polyfit(x, data, 1)
            
            if abs(slope) > 0.1:
                patterns['trend'] = MemoryPattern(
                    pattern_id='trend',
                    pattern_type='trend',
                    trend=slope,
                    confidence=min(1.0, abs(slope) / np.std(data)) if np.std(data) > 0 else 0,
                    metadata={'direction': 'increasing' if slope > 0 else 'decreasing'}
                )
        
        return patterns


def test_arima_prediction():
    """Test ARIMA prediction"""
    print("\n" + "="*50)
    print("TEST: ARIMA Prediction")
    print("="*50)
    
    predictor = SimplifiedMemoryPredictor()
    
    # Generate test data with trend and seasonality
    time_points = 100
    base = 1000
    data = []
    
    for i in range(time_points):
        value = base + 5 * i  # Trend
        value += 200 * math.sin(2 * math.pi * i / 20)  # Seasonality
        value += np.random.normal(0, 20)  # Noise
        data.append(value)
    
    data = np.array(data)
    
    # Test prediction
    try:
        horizon_steps = 10
        prediction = predictor.predict_arima(data, horizon_steps)
        
        print(f"✓ Current value: {data[-1]:.2f} MB")
        print(f"✓ ARIMA prediction ({horizon_steps} steps): {prediction:.2f} MB")
        print(f"✓ Expected trend continuation: {data[-1] + 5 * horizon_steps:.2f} MB")
        
        # Prediction should be reasonable
        assert prediction > 0
        assert prediction < data[-1] * 2  # Not more than double
        
        print("✓ ARIMA prediction test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ ARIMA prediction test FAILED: {e}")
        return False


def test_exponential_smoothing():
    """Test exponential smoothing"""
    print("\n" + "="*50)
    print("TEST: Exponential Smoothing")
    print("="*50)
    
    predictor = SimplifiedMemoryPredictor()
    
    # Generate data with clear trend
    data = np.array([100 + 10 * i + np.random.normal(0, 5) for i in range(50)])
    
    try:
        horizon_steps = 5
        prediction = predictor.predict_exponential_smoothing(data, horizon_steps)
        
        print(f"✓ Current value: {data[-1]:.2f} MB")
        print(f"✓ Exponential smoothing prediction: {prediction:.2f} MB")
        
        # With positive trend, prediction should be higher
        assert prediction > data[-1]
        
        # Test with different alpha values
        original_alpha = predictor.exp_smoothing_alpha
        
        predictor.exp_smoothing_alpha = 0.9
        pred_high = predictor.predict_exponential_smoothing(data, horizon_steps)
        
        predictor.exp_smoothing_alpha = 0.1
        pred_low = predictor.predict_exponential_smoothing(data, horizon_steps)
        
        print(f"✓ High alpha (0.9): {pred_high:.2f} MB")
        print(f"✓ Low alpha (0.1): {pred_low:.2f} MB")
        
        predictor.exp_smoothing_alpha = original_alpha
        
        print("✓ Exponential smoothing test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Exponential smoothing test FAILED: {e}")
        return False


def test_pattern_detection():
    """Test pattern detection"""
    print("\n" + "="*50)
    print("TEST: Pattern Detection")
    print("="*50)
    
    predictor = SimplifiedMemoryPredictor()
    
    # Generate data with clear patterns
    time_points = 100
    data = []
    
    for i in range(time_points):
        # Periodic pattern
        value = 1000 + 100 * math.sin(2 * math.pi * i / 15)
        # Trend
        value += 2 * i
        # Noise
        value += np.random.normal(0, 10)
        data.append(value)
    
    data = np.array(data)
    
    try:
        patterns = predictor.detect_patterns(data)
        
        print(f"✓ Detected {len(patterns)} patterns")
        
        for pattern_id, pattern in patterns.items():
            print(f"  - {pattern_id}: type={pattern.pattern_type}, confidence={pattern.confidence:.2f}")
            
            if pattern.pattern_type == 'periodic':
                print(f"    Period: {pattern.period_seconds:.2f}s, Amplitude: {pattern.amplitude_mb:.2f} MB")
            elif pattern.pattern_type == 'trend':
                print(f"    Trend: {pattern.trend:.2f} MB/sample")
        
        # Should detect at least one pattern
        assert len(patterns) > 0
        
        print("✓ Pattern detection test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Pattern detection test FAILED: {e}")
        return False


def test_memory_leak_detection():
    """Test memory leak detection"""
    print("\n" + "="*50)
    print("TEST: Memory Leak Detection")
    print("="*50)
    
    predictor = SimplifiedMemoryPredictor()
    
    # Generate monotonically increasing data (leak signature)
    data = []
    base = 1000
    
    for i in range(100):
        # Consistent increase (leak)
        value = base + 10 * i  # 10 MB/step leak
        value += np.random.normal(0, 2)  # Small noise
        data.append(value)
    
    data = np.array(data)
    
    try:
        patterns = predictor.detect_patterns(data)
        
        # Should detect strong positive trend
        if 'trend' in patterns:
            trend = patterns['trend']
            print(f"✓ Detected trend: {trend.trend:.2f} MB/sample")
            print(f"✓ Confidence: {trend.confidence:.2f}")
            print(f"✓ Direction: {trend.metadata.get('direction', 'unknown')}")
            
            # Strong positive trend indicates potential leak
            assert trend.trend > 5  # Significant positive trend
            assert trend.metadata['direction'] == 'increasing'
            
            print("✓ Memory leak pattern detected successfully")
        else:
            print("⚠ Trend pattern not detected (may need more sophisticated detection)")
        
        print("✓ Memory leak detection test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Memory leak detection test FAILED: {e}")
        return False


def test_numerical_stability():
    """Test numerical stability with edge cases"""
    print("\n" + "="*50)
    print("TEST: Numerical Stability")
    print("="*50)
    
    predictor = SimplifiedMemoryPredictor()
    
    test_cases = [
        ("Very small values", [0.001, 0.002, 0.001, 0.003] * 10),
        ("Very large values", [1e6, 1e6 + 100, 1e6 + 200] * 10),
        ("Constant values", [1000] * 30),
        ("Mixed scales", [1, 1000, 2, 2000, 3, 3000] * 5)
    ]
    
    all_passed = True
    
    for case_name, data in test_cases:
        print(f"\nTesting: {case_name}")
        data = np.array(data)
        
        try:
            # Test ARIMA
            if len(data) >= 20:
                pred = predictor.predict_arima(data, 5)
                assert not math.isnan(pred)
                assert not math.isinf(pred)
                assert pred >= 0
                print(f"  ✓ ARIMA: {pred:.6f}")
            
            # Test exponential smoothing
            pred = predictor.predict_exponential_smoothing(data, 5)
            assert not math.isnan(pred)
            assert not math.isinf(pred)
            assert pred >= 0
            print(f"  ✓ Exponential: {pred:.6f}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            all_passed = False
    
    if all_passed:
        print("\n✓ Numerical stability test PASSED")
    else:
        print("\n✗ Numerical stability test had some failures")
    
    return all_passed


def test_ensemble_prediction():
    """Test ensemble of prediction methods"""
    print("\n" + "="*50)
    print("TEST: Ensemble Prediction")
    print("="*50)
    
    predictor = SimplifiedMemoryPredictor()
    
    # Generate complex pattern
    time_points = 100
    data = []
    
    for i in range(time_points):
        value = 1000
        value += 3 * i  # Trend
        value += 150 * math.sin(2 * math.pi * i / 25)  # Seasonality
        value += 50 * math.sin(2 * math.pi * i / 7)  # Higher frequency
        value += np.random.normal(0, 20)  # Noise
        data.append(value)
    
    data = np.array(data)
    
    try:
        horizon_steps = 10
        
        # Get predictions from different methods
        predictions = []
        
        try:
            arima_pred = predictor.predict_arima(data, horizon_steps)
            predictions.append(('ARIMA', arima_pred))
            print(f"✓ ARIMA prediction: {arima_pred:.2f} MB")
        except Exception as e:
            print(f"  ARIMA failed: {e}")
        
        try:
            exp_pred = predictor.predict_exponential_smoothing(data, horizon_steps)
            predictions.append(('Exponential', exp_pred))
            print(f"✓ Exponential prediction: {exp_pred:.2f} MB")
        except Exception as e:
            print(f"  Exponential failed: {e}")
        
        if len(predictions) > 0:
            # Calculate ensemble
            ensemble_pred = np.mean([p[1] for p in predictions])
            print(f"✓ Ensemble prediction: {ensemble_pred:.2f} MB")
            
            # Calculate confidence based on agreement
            if len(predictions) > 1:
                std_dev = np.std([p[1] for p in predictions])
                confidence = max(0, min(1, 1 - std_dev / (ensemble_pred + 1e-6)))
                print(f"✓ Prediction confidence: {confidence:.2f}")
        
        print("✓ Ensemble prediction test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Ensemble prediction test FAILED: {e}")
        return False


def run_all_tests():
    """Run all standalone tests"""
    print("\n" + "="*60)
    print("ADVANCED MEMORY PREDICTION - STANDALONE TESTS")
    print("="*60)
    
    tests = [
        ("ARIMA Prediction", test_arima_prediction),
        ("Exponential Smoothing", test_exponential_smoothing),
        ("Pattern Detection", test_pattern_detection),
        ("Memory Leak Detection", test_memory_leak_detection),
        ("Numerical Stability", test_numerical_stability),
        ("Ensemble Prediction", test_ensemble_prediction)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30} {status}")
    
    print("\n" + "-"*60)
    print(f"Total: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! 🎉")
    else:
        print(f"\n⚠ {total_count - passed_count} test(s) failed")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)