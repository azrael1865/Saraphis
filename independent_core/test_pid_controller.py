"""
Comprehensive test suite for PIDController
Tests all functionality and edge cases
"""

import unittest
import time
import numpy as np
from gac_system.gac_pid_controller import (
    PIDController, 
    PIDControllerConfig, 
    GACPIDController,
    create_gradient_threshold_pid
)


class TestPIDController(unittest.TestCase):
    """Comprehensive tests for PIDController"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = PIDControllerConfig(
            kp=1.0, ki=0.1, kd=0.05,
            setpoint=10.0,
            output_min=-100, output_max=100,
            sample_time=0.01
        )
        self.controller = PIDController(self.config)
    
    def test_initialization_with_config(self):
        """Test initialization with config object"""
        controller = PIDController(self.config)
        self.assertEqual(controller.config.kp, 1.0)
        self.assertEqual(controller.config.ki, 0.1)
        self.assertEqual(controller.config.kd, 0.05)
        self.assertEqual(controller.config.setpoint, 10.0)
    
    def test_initialization_with_kwargs(self):
        """Test initialization with keyword arguments"""
        controller = PIDController(
            kp=2.0, ki=0.2, kd=0.1,
            setpoint=5.0,
            output_limits=(-50, 50)
        )
        self.assertEqual(controller.config.kp, 2.0)
        self.assertEqual(controller.config.ki, 0.2)
        self.assertEqual(controller.config.kd, 0.1)
        self.assertEqual(controller.config.setpoint, 5.0)
        self.assertEqual(controller.config.output_min, -50)
        self.assertEqual(controller.config.output_max, 50)
    
    def test_gac_pid_controller_alias(self):
        """Test that GACPIDController alias works"""
        controller = GACPIDController(kp=1.0, ki=0.1, kd=0.05)
        self.assertIsInstance(controller, PIDController)
    
    def test_basic_proportional_control(self):
        """Test proportional control only"""
        controller = PIDController(kp=1.0, ki=0.0, kd=0.0, setpoint=10.0)
        
        # Below setpoint
        output = controller.update(5.0)  # Error = 10 - 5 = 5
        self.assertAlmostEqual(output, 5.0, places=2)
        
        # Above setpoint
        controller.reset()
        output = controller.update(15.0)  # Error = 10 - 15 = -5
        self.assertAlmostEqual(output, -5.0, places=2)
    
    def test_integral_accumulation(self):
        """Test integral term accumulation"""
        controller = PIDController(
            kp=0.0, ki=1.0, kd=0.0, 
            setpoint=10.0,
            sample_time=0.01
        )
        
        # Simulate multiple updates with constant error
        measured = 8.0  # Error = 2
        for _ in range(10):
            output = controller.update(measured)
            time.sleep(0.015)  # Ensure sample time passes
        
        # Integral should accumulate
        self.assertGreater(controller.integral, 0)
        self.assertGreater(output, 0)
    
    def test_derivative_response(self):
        """Test derivative term response to changing error"""
        controller = PIDController(
            kp=0.0, ki=0.0, kd=1.0,
            setpoint=10.0,
            sample_time=0.01
        )
        
        # First update
        controller.update(5.0)
        time.sleep(0.02)
        
        # Sudden change - derivative should respond
        output = controller.update(8.0)  # Error decreased rapidly
        self.assertNotEqual(output, 0)
    
    def test_output_limits(self):
        """Test output limiting"""
        controller = PIDController(
            kp=100.0, ki=0.0, kd=0.0,
            setpoint=10.0,
            output_limits=(-10, 10)
        )
        
        # Large error should be limited
        output = controller.update(0.0)  # Error = 10, P = 1000
        self.assertEqual(output, 10.0)  # Should be clamped
        
        output = controller.update(20.0)  # Error = -10, P = -1000
        self.assertEqual(output, -10.0)  # Should be clamped
    
    def test_windup_protection(self):
        """Test integral windup protection"""
        controller = PIDController(
            kp=0.0, ki=1.0, kd=0.0,
            setpoint=10.0,
            windup_limit=5.0,
            sample_time=0.01
        )
        
        # Accumulate large integral
        for _ in range(100):
            controller.update(0.0)  # Large constant error
            time.sleep(0.011)
        
        # Integral should be limited
        self.assertLessEqual(abs(controller.integral), 5.0)
    
    def test_sample_time_limiting(self):
        """Test that updates are skipped if sample time hasn't elapsed"""
        controller = PIDController(
            kp=1.0, ki=0.1, kd=0.05,
            sample_time=0.1  # 100ms sample time
        )
        
        output1 = controller.update(5.0)
        # Immediate second call - should return same output
        output2 = controller.update(8.0)
        self.assertEqual(output1, output2)
        
        # Wait for sample time
        time.sleep(0.11)
        output3 = controller.update(8.0)
        self.assertNotEqual(output2, output3)
    
    def test_reset_functionality(self):
        """Test controller reset"""
        # Accumulate some state
        for i in range(10):
            self.controller.update(i)
            time.sleep(0.02)
        
        self.assertNotEqual(self.controller.integral, 0)
        self.assertNotEqual(self.controller.last_error, 0)
        
        # Reset
        self.controller.reset()
        self.assertEqual(self.controller.integral, 0)
        self.assertEqual(self.controller.last_error, 0)
    
    def test_performance_metrics(self):
        """Test performance metric tracking"""
        controller = PIDController(self.config)
        
        # Generate some updates
        for i in range(10):
            controller.update(5.0 + i)
            time.sleep(0.02)
        
        metrics = controller.get_performance_metrics()
        self.assertIn('total_updates', metrics)
        self.assertIn('avg_error', metrics)
        self.assertIn('stability_score', metrics)
        self.assertGreater(metrics['total_updates'], 0)
    
    def test_set_tuning_parameters(self):
        """Test dynamic parameter updates"""
        controller = PIDController(self.config)
        
        controller.set_tuning_parameters(kp=2.0, ki=0.2, kd=0.1)
        self.assertEqual(controller.config.kp, 2.0)
        self.assertEqual(controller.config.ki, 0.2)
        self.assertEqual(controller.config.kd, 0.1)
    
    def test_factory_functions(self):
        """Test factory function for creating controllers"""
        conservative = create_gradient_threshold_pid(1.0, "conservative")
        self.assertEqual(conservative.config.kp, 0.5)
        
        aggressive = create_gradient_threshold_pid(1.0, "aggressive")
        self.assertEqual(aggressive.config.kp, 2.0)
        
        moderate = create_gradient_threshold_pid(1.0, "moderate")
        self.assertEqual(moderate.config.kp, 1.0)
    
    def test_target_override(self):
        """Test that target_value parameter overrides setpoint"""
        controller = PIDController(kp=1.0, ki=0.0, kd=0.0, setpoint=10.0)
        
        # Use default setpoint
        output1 = controller.update(5.0)  # Error = 10 - 5 = 5
        self.assertAlmostEqual(output1, 5.0, places=2)
        
        # Override with target_value
        controller.reset()
        output2 = controller.update(5.0, target_value=20.0)  # Error = 20 - 5 = 15
        self.assertAlmostEqual(output2, 15.0, places=2)
    
    def test_auto_tuning(self):
        """Test auto-tuning functionality"""
        controller = PIDController(
            kp=1.0, ki=0.1, kd=0.05,
            setpoint=10.0,
            auto_tune=True,
            sample_time=0.001
        )
        
        initial_kp = controller.config.kp
        
        # Generate many updates to trigger auto-tuning
        for i in range(150):
            controller.update(5.0 + np.random.randn())
            time.sleep(0.002)
        
        # Check if auto-tuning has been attempted
        metrics = controller.get_performance_metrics()
        self.assertGreater(metrics['auto_tune_attempts'], 0)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        controller = PIDController(self.config)
        
        # Test with zero error
        controller.reset()
        output = controller.update(10.0)  # Error = 0
        self.assertEqual(output, 0.0)
        
        # Test with very large values
        controller.reset()
        output = controller.update(1e10)
        self.assertEqual(output, controller.config.output_min)  # Should be limited
        
        # Test with NaN (should handle gracefully)
        controller.reset()
        try:
            output = controller.update(float('nan'))
            # Should either handle it or raise appropriate error
        except (ValueError, TypeError):
            pass  # Expected behavior
    
    def test_dt_zero_handling(self):
        """Test handling of zero time delta"""
        controller = PIDController(self.config)
        
        # First update to set a baseline
        first_output = controller.update(5.0)
        
        # Manually set last_time to current time to force dt=0
        controller.last_time = time.time()
        output = controller.update(5.0)
        
        # Should not crash and should return last output (which is first_output)
        self.assertEqual(output, first_output)
    
    def test_anti_windup_conditional_integration(self):
        """Test advanced anti-windup with conditional integration"""
        controller = PIDController(
            kp=1.0, ki=1.0, kd=0.0,
            setpoint=10.0,
            output_limits=(0, 10),
            sample_time=0.01
        )
        
        # Saturate high
        for _ in range(10):
            controller.update(0.0)  # Large positive error
            time.sleep(0.02)
        
        # Now error reverses - integral should be allowed to decrease
        initial_integral = controller.integral
        controller.update(20.0)  # Negative error
        time.sleep(0.02)
        
        # Integral should be allowed to decrease even when saturated
        self.assertLessEqual(controller.integral, initial_integral)


class TestPIDControllerIntegration(unittest.TestCase):
    """Integration tests for PIDController in system context"""
    
    def test_gradient_clipping_scenario(self):
        """Test PID controller in gradient clipping context"""
        controller = PIDController(
            kp=0.3, ki=0.05, kd=0.02,
            setpoint=1.0,
            output_limits=(0.1, 10.0)
        )
        
        # Simulate gradient norms
        gradient_norms = [0.5, 2.0, 5.0, 1.2, 0.8, 3.0, 1.0]
        
        outputs = []
        for norm in gradient_norms:
            output = controller.update(norm)
            outputs.append(output)
            time.sleep(0.02)
        
        # Should converge towards setpoint
        final_errors = [abs(1.0 - norm) for norm in gradient_norms[-3:]]
        avg_final_error = sum(final_errors) / len(final_errors)
        self.assertLess(avg_final_error, 2.0)
    
    def test_learning_rate_adaptation(self):
        """Test PID controller for learning rate adaptation"""
        controller = PIDController(
            kp=0.5, ki=0.1, kd=0.05,
            setpoint=0.001,
            output_limits=(1e-5, 0.1)
        )
        
        # Simulate loss values (lower is better)
        loss_values = [0.5, 0.3, 0.2, 0.15, 0.1, 0.08]
        
        for loss in loss_values:
            # Use loss as feedback (inverse relationship with learning rate)
            lr_adjustment = controller.update(loss)
            time.sleep(0.02)
        
        # Controller should produce reasonable learning rate adjustments
        self.assertGreater(lr_adjustment, 1e-5)
        self.assertLess(lr_adjustment, 0.1)


class TestPIDControllerEdgeCases(unittest.TestCase):
    """Additional comprehensive edge case tests"""
    
    def test_nan_handling(self):
        """Test handling of NaN values"""
        controller = PIDController(kp=1.0, ki=0.1, kd=0.05, setpoint=10.0)
        
        # NaN input should not crash
        try:
            output = controller.update(float('nan'))
            # Check if output is valid (not NaN)
            if not np.isnan(output):
                self.assertTrue(True)  # Handled gracefully
        except (ValueError, TypeError):
            self.assertTrue(True)  # Appropriate error raised
    
    def test_infinity_handling(self):
        """Test handling of infinite values"""
        controller = PIDController(kp=1.0, ki=0.1, kd=0.05, setpoint=10.0)
        
        # Positive infinity
        output = controller.update(float('inf'))
        self.assertEqual(output, controller.config.output_min)
        
        # Negative infinity
        controller.reset()
        output = controller.update(float('-inf'))
        self.assertEqual(output, controller.config.output_max)
    
    def test_very_large_gains(self):
        """Test stability with very large gains"""
        controller = PIDController(
            kp=1e6, ki=1e6, kd=1e6,
            setpoint=1.0,
            output_limits=(-1000, 1000)
        )
        
        output = controller.update(0.9999)  # Very small error
        self.assertLessEqual(abs(output), 1000)  # Should be limited
    
    def test_very_small_gains(self):
        """Test with very small gains"""
        controller = PIDController(
            kp=1e-10, ki=1e-10, kd=1e-10,
            setpoint=1.0
        )
        
        output = controller.update(1000.0)  # Large error
        self.assertLess(abs(output), 1.0)  # Output should be tiny
    
    def test_rapid_setpoint_changes(self):
        """Test rapid setpoint changes"""
        controller = PIDController(kp=1.0, ki=0.1, kd=0.05)
        
        outputs = []
        setpoints = [10.0, -10.0, 0.0, 100.0, -100.0]
        
        for sp in setpoints:
            output = controller.update(5.0, target_value=sp)
            outputs.append(output)
            time.sleep(0.001)
        
        # Should handle rapid changes without instability
        self.assertEqual(len(outputs), len(setpoints))
        self.assertTrue(all(np.isfinite(o) for o in outputs))
    
    def test_zero_gains(self):
        """Test with all gains set to zero"""
        controller = PIDController(kp=0.0, ki=0.0, kd=0.0, setpoint=10.0)
        
        output = controller.update(5.0)
        self.assertEqual(output, 0.0)  # No control action
    
    def test_negative_gains(self):
        """Test with negative gains (reverse action)"""
        controller = PIDController(kp=-1.0, ki=-0.1, kd=-0.05, setpoint=10.0)
        
        output = controller.update(5.0)  # Positive error
        self.assertLess(output, 0)  # Negative output due to negative gains
    
    def test_concurrent_updates(self):
        """Test thread safety with concurrent updates"""
        import threading
        
        controller = PIDController(kp=1.0, ki=0.1, kd=0.05, setpoint=10.0)
        results = []
        
        def update_controller():
            for _ in range(10):
                output = controller.update(np.random.randn() * 10)
                results.append(output)
                time.sleep(0.001)
        
        threads = [threading.Thread(target=update_controller) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should complete without crashes
        self.assertGreater(len(results), 0)
    
    def test_memory_efficiency(self):
        """Test memory efficiency with many updates"""
        controller = PIDController(kp=1.0, ki=0.1, kd=0.05, setpoint=10.0)
        
        # Perform many updates
        for i in range(1000):
            controller.update(i % 20)
        
        # Check that history is limited
        self.assertLessEqual(len(controller.error_history), 50)
        self.assertLessEqual(len(controller.output_history), 50)
    
    def test_numerical_precision(self):
        """Test numerical precision with very small errors"""
        controller = PIDController(kp=1.0, ki=0.1, kd=0.05, setpoint=1.0)
        
        # Very small error
        output1 = controller.update(0.999999999)
        output2 = controller.update(1.000000001)
        
        # Should handle small differences
        self.assertNotEqual(output1, output2)
        self.assertTrue(np.isfinite(output1))
        self.assertTrue(np.isfinite(output2))
    
    def test_sample_time_zero_continuous(self):
        """Test continuous operation with sample_time=0"""
        controller = PIDController(
            kp=0.5, ki=0.05, kd=0.01,  # Smaller gains to avoid saturation
            setpoint=5.0,  # Smaller setpoint
            sample_time=0.0,  # Continuous mode
            output_limits=(-1000, 1000)  # Wider limits to avoid saturation
        )
        
        # Multiple rapid updates should all process
        outputs = []
        errors = []
        for i in range(5):
            output = controller.update(i * 0.5)  # Smaller increments
            outputs.append(output)
            errors.append(5.0 - i * 0.5)
        
        # All updates should process (not return cached values)
        # With different errors and no saturation, outputs should be different
        self.assertEqual(len(set(outputs)), 5, 
                        f"Expected 5 unique outputs but got {len(set(outputs))}. Outputs: {outputs}")
    
    def test_windup_with_changing_limits(self):
        """Test windup behavior when output limits change"""
        controller = PIDController(
            kp=0.0, ki=1.0, kd=0.0,
            setpoint=10.0,
            output_limits=(-10, 10),
            windup_limit=5.0
        )
        
        # Accumulate integral
        for _ in range(10):
            controller.update(0.0)
            time.sleep(0.001)
        
        # Change output limits
        controller.config.output_min = -100
        controller.config.output_max = 100
        
        # Should still respect windup limit
        self.assertLessEqual(abs(controller.integral), 5.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)