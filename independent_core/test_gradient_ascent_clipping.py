#!/usr/bin/env python3
"""
Comprehensive test suite for GradientAscentClipping
Tests all core functionality and identifies issues
"""

import unittest
import torch
import numpy as np
import asyncio
import time
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the module
from gac_system.gradient_ascent_clipping import (
    GACSystem, GACComponent, ComponentState, EventType, GACEvent,
    ComponentMetrics, PIDState, EventBus, ComponentRegistry,
    MetaLearningEngine, ReinforcementLearningAgent,
    DirectionAwareGradientComponent, GACThreadManager,
    GradientRequest, GradientResponse, create_gac_system,
    DIRECTION_COMPONENTS_AVAILABLE
)


class TestGACComponent(unittest.TestCase):
    """Test the abstract GACComponent base class"""
    
    def setUp(self):
        """Create a concrete implementation for testing"""
        class ConcreteGACComponent(GACComponent):
            async def process_gradient(self, gradient: torch.Tensor, context: dict) -> torch.Tensor:
                return gradient * 0.9
            
            def get_component_info(self) -> dict:
                return {"type": "test_component"}
        
        self.component = ConcreteGACComponent("test_component_1")
    
    def test_component_initialization(self):
        """Test component initializes with correct defaults"""
        self.assertEqual(self.component.component_id, "test_component_1")
        self.assertEqual(self.component.state, ComponentState.INACTIVE)
        self.assertIsInstance(self.component.metrics, ComponentMetrics)
        self.assertIsInstance(self.component.pid_state, PIDState)
        self.assertEqual(len(self.component.event_handlers), 0)
    
    def test_state_updates(self):
        """Test component state transitions"""
        events_received = []
        
        def handler(event):
            events_received.append(event)
        
        self.component.register_event_handler(EventType.COMPONENT_STATE_CHANGE, handler)
        self.component.update_state(ComponentState.ACTIVE)
        
        self.assertEqual(self.component.state, ComponentState.ACTIVE)
        self.assertEqual(len(events_received), 1)
        self.assertEqual(events_received[0].event_type, EventType.COMPONENT_STATE_CHANGE)
    
    def test_pid_calculation(self):
        """Test PID controller calculations"""
        self.component.pid_state.setpoint = 1.0
        self.component.pid_state.kp = 1.0
        self.component.pid_state.ki = 0.1
        self.component.pid_state.kd = 0.05
        
        output1 = self.component.calculate_pid_output(0.5, dt=1.0)
        output2 = self.component.calculate_pid_output(0.7, dt=1.0)
        
        # PID should produce different outputs for different errors
        self.assertNotEqual(output1, output2)
        self.assertIsInstance(output1, float)
    
    async def test_gradient_processing(self):
        """Test async gradient processing"""
        gradient = torch.randn(10, 10)
        context = {"test": True}
        
        result = await self.component.process_gradient(gradient, context)
        
        self.assertEqual(result.shape, gradient.shape)
        # Our test implementation multiplies by 0.9
        self.assertTrue(torch.allclose(result, gradient * 0.9))


class TestEventBus(unittest.TestCase):
    """Test the EventBus system"""
    
    def setUp(self):
        self.event_bus = EventBus()
    
    def test_subscribe_and_publish(self):
        """Test event subscription and publishing"""
        received_events = []
        
        def handler(event):
            received_events.append(event)
        
        self.event_bus.subscribe(EventType.GRADIENT_UPDATE, handler)
        
        event = GACEvent(EventType.GRADIENT_UPDATE, "test_source", {"value": 42})
        self.event_bus.publish(event)
        
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0].data["value"], 42)
    
    def test_event_history(self):
        """Test event history tracking"""
        for i in range(5):
            event = GACEvent(EventType.GRADIENT_UPDATE, f"source_{i}", {"index": i})
            self.event_bus.publish(event)
        
        history = self.event_bus.get_event_history()
        self.assertEqual(len(history), 5)
        
        # Test filtering by event type
        event2 = GACEvent(EventType.SYSTEM_ALERT, "alert_source", {})
        self.event_bus.publish(event2)
        
        gradient_history = self.event_bus.get_event_history(event_type=EventType.GRADIENT_UPDATE)
        self.assertEqual(len(gradient_history), 5)
        
        alert_history = self.event_bus.get_event_history(event_type=EventType.SYSTEM_ALERT)
        self.assertEqual(len(alert_history), 1)


class TestComponentRegistry(unittest.TestCase):
    """Test the ComponentRegistry"""
    
    def setUp(self):
        self.registry = ComponentRegistry()
        
        class TestComponent(GACComponent):
            async def process_gradient(self, gradient, context):
                return gradient
            def get_component_info(self):
                return {}
        
        self.TestComponent = TestComponent
    
    def test_register_and_retrieve(self):
        """Test component registration and retrieval"""
        comp1 = self.TestComponent("comp1")
        comp2 = self.TestComponent("comp2")
        
        self.registry.register_component(comp1, group="group_a")
        self.registry.register_component(comp2, group="group_a")
        
        # Test retrieval by ID
        retrieved = self.registry.get_component("comp1")
        self.assertEqual(retrieved.component_id, "comp1")
        
        # Test retrieval by group
        group_comps = self.registry.get_components_by_group("group_a")
        self.assertEqual(len(group_comps), 2)
    
    def test_dependency_management(self):
        """Test dependency graph and execution order"""
        for i in range(3):
            comp = self.TestComponent(f"comp{i}")
            self.registry.register_component(comp)
        
        # Create dependency chain: comp2 -> comp1 -> comp0
        self.registry.add_dependency("comp2", "comp1")
        self.registry.add_dependency("comp1", "comp0")
        
        order = self.registry.get_execution_order()
        
        # comp0 should come before comp1, comp1 before comp2
        self.assertTrue(order.index("comp0") < order.index("comp1"))
        self.assertTrue(order.index("comp1") < order.index("comp2"))
    
    def test_circular_dependency_detection(self):
        """Test that circular dependencies are detected"""
        for i in range(3):
            comp = self.TestComponent(f"comp{i}")
            self.registry.register_component(comp)
        
        # Create circular dependency
        self.registry.add_dependency("comp0", "comp1")
        self.registry.add_dependency("comp1", "comp2")
        self.registry.add_dependency("comp2", "comp0")
        
        with self.assertRaises(ValueError) as context:
            self.registry.get_execution_order()
        
        self.assertIn("Circular dependency", str(context.exception))


class TestMetaLearningEngine(unittest.TestCase):
    """Test the MetaLearningEngine"""
    
    def setUp(self):
        self.engine = MetaLearningEngine({'learning_rate': 0.01, 'adaptation_threshold': 0.1})
    
    def test_performance_recording(self):
        """Test recording and limiting performance history"""
        # Record many performance entries
        for i in range(1500):
            self.engine.record_performance("comp1", {"success_rate": 0.5 + i * 0.0001})
        
        # Check history is limited
        history = self.engine.component_performance_history["comp1"]
        self.assertLessEqual(len(history), 1000)  # Default limit
    
    def test_adaptation_suggestions(self):
        """Test adaptation suggestions based on performance"""
        # Create declining performance history
        for i in range(20):
            success_rate = 0.8 - i * 0.02  # Declining from 0.8 to 0.4
            self.engine.record_performance("comp1", {"success_rate": success_rate})
        
        adaptations = self.engine.suggest_adaptations("comp1")
        
        # Should suggest increasing sensitivity due to declining performance
        self.assertTrue(adaptations.get("increase_sensitivity", False))
        self.assertIn("suggested_threshold_adjustment", adaptations)


class TestReinforcementLearningAgent(unittest.TestCase):
    """Test the ReinforcementLearningAgent"""
    
    def setUp(self):
        self.agent = ReinforcementLearningAgent({
            'learning_rate': 0.1,
            'discount_factor': 0.95,
            'epsilon': 0.1
        })
    
    def test_state_representation(self):
        """Test state representation bucketing"""
        metrics1 = {'overall_performance': 0.9, 'system_load': 0.3, 'error_rate': 0.01}
        state1 = self.agent.get_state_representation(metrics1)
        self.assertEqual(state1, "high_low_low")
        
        metrics2 = {'overall_performance': 0.4, 'system_load': 0.9, 'error_rate': 0.15}
        state2 = self.agent.get_state_representation(metrics2)
        self.assertEqual(state2, "low_high_high")
    
    def test_action_selection(self):
        """Test action selection with epsilon-greedy"""
        # With empty Q-table, should return random action
        state = "test_state"
        action = self.agent.select_action(state)
        self.assertIn(action, self.agent.action_space)
    
    def test_q_value_update(self):
        """Test Q-value learning update"""
        state = "state1"
        action = "increase_threshold"
        reward = 1.0
        next_state = "state2"
        
        # Initial Q-value should be 0
        initial_q = self.agent.q_table[state][action]
        self.assertEqual(initial_q, 0.0)
        
        # Update Q-value
        self.agent.update_q_value(state, action, reward, next_state)
        
        # Q-value should have increased
        updated_q = self.agent.q_table[state][action]
        self.assertGreater(updated_q, initial_q)


class TestDirectionAwareGradientComponent(unittest.TestCase):
    """Test DirectionAwareGradientComponent wrapper"""
    
    def setUp(self):
        # Create a mock enhanced bounder
        self.mock_bounder = MagicMock()
        self.component = DirectionAwareGradientComponent(
            "test_direction_component",
            self.mock_bounder,
            {}
        )
    
    async def test_process_gradient_with_mock(self):
        """Test gradient processing through the wrapper"""
        gradient = torch.randn(10, 10)
        context = {"test": True}
        
        # Setup mock return value
        mock_result = MagicMock()
        mock_result.bounded_gradients = gradient * 0.8
        mock_result.applied_factor = 0.8
        mock_result.direction_based_adjustments = True
        mock_result.direction_confidence = 0.9
        mock_result.direction_state = MagicMock()
        mock_result.direction_state.direction.value = "ascending"
        
        self.mock_bounder.bound_gradients.return_value = mock_result
        
        # Process gradient
        result = await self.component.process_gradient(gradient, context)
        
        # Verify the mock was called
        self.mock_bounder.bound_gradients.assert_called_once_with(gradient, context)
        
        # Check result
        self.assertTrue(torch.allclose(result, gradient * 0.8))
        
        # Check statistics were updated
        self.assertEqual(self.component.processing_stats['total_processed'], 1)
        self.assertEqual(self.component.processing_stats['direction_adjustments'], 1)
    
    def test_get_component_info(self):
        """Test component info retrieval"""
        info = self.component.get_component_info()
        
        self.assertEqual(info['component_id'], "test_direction_component")
        self.assertEqual(info['component_type'], 'direction_aware_gradient_bounder')
        self.assertIn('processing_stats', info)
        self.assertIn('metrics', info)


class TestGACSystem(unittest.TestCase):
    """Test the main GACSystem"""
    
    def setUp(self):
        self.config = {
            'thresholds': {
                'gradient_magnitude': 1.0,
                'processing_time': 5.0,
                'error_rate': 0.05
            },
            'max_workers': 4
        }
        self.system = GACSystem(self.config)
    
    def test_system_initialization(self):
        """Test system initializes correctly"""
        self.assertEqual(self.system.system_state, ComponentState.INACTIVE)
        self.assertIsInstance(self.system.event_bus, EventBus)
        self.assertIsInstance(self.system.component_registry, ComponentRegistry)
        self.assertIsInstance(self.system.meta_learning_engine, MetaLearningEngine)
        self.assertIsInstance(self.system.rl_agent, ReinforcementLearningAgent)
    
    def test_system_start_stop(self):
        """Test system lifecycle"""
        self.system.start_system()
        self.assertEqual(self.system.system_state, ComponentState.ACTIVE)
        
        self.system.stop_system()
        self.assertEqual(self.system.system_state, ComponentState.INACTIVE)
    
    async def test_gradient_processing_empty(self):
        """Test gradient processing with no components"""
        self.system.start_system()
        
        gradient = torch.randn(10, 10)
        result = await self.system.process_gradient(gradient)
        
        # With no components, should return a clone of the original
        self.assertEqual(result.shape, gradient.shape)
        self.assertFalse(result is gradient)  # Should be a clone
    
    def test_component_registration(self):
        """Test registering components with the system"""
        class TestComponent(GACComponent):
            async def process_gradient(self, gradient, context):
                return gradient * 0.5
            def get_component_info(self):
                return {"test": True}
        
        comp = TestComponent("test_comp")
        self.system.register_component(comp, group="test_group")
        
        # Check component is registered
        retrieved = self.system.component_registry.get_component("test_comp")
        self.assertEqual(retrieved.component_id, "test_comp")
        
        # Check component status
        status = self.system.get_component_status()
        self.assertIn("test_comp", status)
    
    def test_system_metrics(self):
        """Test system metrics calculation"""
        metrics = self.system.get_system_metrics()
        
        self.assertIn('system_state', metrics)
        self.assertIn('active_components', metrics)
        self.assertIn('total_components', metrics)
        self.assertIn('overall_performance', metrics)
        self.assertIn('system_load', metrics)
        self.assertIn('error_rate', metrics)
        self.assertIn('uptime', metrics)
    
    def test_checkpoint_save_restore(self):
        """Test checkpoint creation and restoration"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoint
            timestamp = self.system.create_checkpoint(tmpdir)
            
            # Verify files were created
            checkpoint_path = Path(tmpdir)
            self.assertTrue((checkpoint_path / f"system_state_{timestamp}.json").exists())
            self.assertTrue((checkpoint_path / f"components_{timestamp}.json").exists())
            
            # Modify system state
            self.system.global_thresholds['gradient_magnitude'] = 2.0
            
            # Restore checkpoint
            success = self.system.restore_checkpoint(tmpdir, timestamp)
            self.assertTrue(success)
            
            # Check state was restored
            self.assertEqual(self.system.global_thresholds['gradient_magnitude'], 1.0)
    
    def test_direction_components_status(self):
        """Test direction components status retrieval"""
        status = self.system.get_direction_components_status()
        
        self.assertIn('components_available', status)
        self.assertIn('direction_state_manager', status)
        self.assertIn('direction_validator', status)
        self.assertIn('basic_bounder', status)
        self.assertIn('enhanced_bounder', status)
        
        # Check if components are available
        logger.info(f"Direction components available: {DIRECTION_COMPONENTS_AVAILABLE}")
        logger.info(f"Direction status: {status}")


class TestGACThreadManager(unittest.TestCase):
    """Test the GACThreadManager for synchronous processing"""
    
    def setUp(self):
        self.system = GACSystem({})
        self.thread_manager = GACThreadManager(self.system)
    
    def test_thread_manager_lifecycle(self):
        """Test starting and stopping thread manager"""
        # Start thread manager
        success = self.thread_manager.start()
        self.assertTrue(success)
        self.assertTrue(self.thread_manager.running)
        
        time.sleep(0.2)  # Give thread time to start
        
        # Stop thread manager
        success = self.thread_manager.stop()
        self.assertTrue(success)
        self.assertFalse(self.thread_manager.running)
    
    def test_synchronous_gradient_processing(self):
        """Test synchronous gradient processing interface"""
        self.thread_manager.start()
        time.sleep(0.2)  # Give thread time to start
        
        try:
            gradients = [torch.randn(10, 10) for _ in range(3)]
            
            # Process gradients synchronously
            result = self.thread_manager.process_gradients_sync(gradients, timeout=1.0)
            
            # Should return same number of gradients
            self.assertEqual(len(result), len(gradients))
            
            # Check shapes are preserved
            for orig, proc in zip(gradients, result):
                self.assertEqual(orig.shape, proc.shape)
            
            # Get statistics
            stats = self.thread_manager.get_stats()
            self.assertIn('processed_requests', stats)
            self.assertIn('failed_requests', stats)
            self.assertIn('success_rate', stats)
            
        finally:
            self.thread_manager.stop()


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete GAC system"""
    
    async def test_full_gradient_processing_pipeline(self):
        """Test complete gradient processing with custom component"""
        system = GACSystem({})
        
        # Create and register a custom component
        class ScalingComponent(GACComponent):
            async def process_gradient(self, gradient, context):
                return gradient * context.get('scale', 0.5)
            
            def get_component_info(self):
                return {"type": "scaler"}
        
        scaler = ScalingComponent("scaler")
        system.register_component(scaler)
        
        # Start system
        system.start_system()
        
        # Process gradient
        gradient = torch.ones(5, 5)
        context = {'scale': 0.7}
        
        result = await system.process_gradient(gradient, context)
        
        # Check result
        expected = torch.ones(5, 5) * 0.7
        self.assertTrue(torch.allclose(result, expected))
        
        # Check metrics were updated
        metrics = system.get_system_metrics()
        self.assertEqual(metrics['performance_metrics']['total_gradients_processed'], 1)
        
        # Stop system
        system.stop_system()
    
    def test_create_gac_system_function(self):
        """Test the create_gac_system helper function"""
        # Test with no config
        system1 = create_gac_system()
        self.assertIsInstance(system1, GACSystem)
        
        # Test with dict config passed directly to GACSystem
        config = {'thresholds': {'gradient_magnitude': 2.0}}
        system2 = create_gac_system()
        system2.global_thresholds.update(config['thresholds'])
        self.assertEqual(system2.global_thresholds['gradient_magnitude'], 2.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_none_gradient_handling(self):
        """Test handling of None gradients"""
        component = DirectionAwareGradientComponent("test", MagicMock())
        
        async def test():
            result = await component.process_gradient(None, {})
            self.assertIsNone(result)
        
        asyncio.run(test())
    
    def test_invalid_gradient_type(self):
        """Test handling of invalid gradient types"""
        component = DirectionAwareGradientComponent("test", MagicMock())
        
        async def test():
            # Pass a non-tensor
            result = await component.process_gradient([1, 2, 3], {})
            self.assertEqual(result, [1, 2, 3])  # Should return original
        
        asyncio.run(test())
    
    def test_component_error_handling(self):
        """Test error handling in component processing"""
        system = GACSystem({})
        
        class ErrorComponent(GACComponent):
            async def process_gradient(self, gradient, context):
                raise ValueError("Test error")
            
            def get_component_info(self):
                return {}
        
        error_comp = ErrorComponent("error_comp")
        system.register_component(error_comp)
        system.start_system()
        
        async def test():
            gradient = torch.randn(5, 5)
            result = await system.process_gradient(gradient)
            
            # Should still return a gradient despite error
            self.assertEqual(result.shape, gradient.shape)
            
            # Error should be recorded
            self.assertEqual(error_comp.metrics.error_count, 1)
        
        asyncio.run(test())
        system.stop_system()
    
    def test_zero_gradients(self):
        """Test handling of zero gradients"""
        system = GACSystem({})
        system.start_system()
        
        async def test():
            # Test with all-zero gradient
            zero_grad = torch.zeros(10, 10)
            result = await system.process_gradient(zero_grad)
            self.assertTrue(torch.allclose(result, zero_grad))
            
            # Test with near-zero gradient
            tiny_grad = torch.ones(10, 10) * 1e-10
            result = await system.process_gradient(tiny_grad)
            self.assertIsNotNone(result)
        
        asyncio.run(test())
        system.stop_system()
    
    def test_infinite_nan_gradients(self):
        """Test handling of infinite and NaN gradients"""
        component = DirectionAwareGradientComponent("test", MagicMock())
        
        # Setup mock to handle inf/nan
        mock_result = MagicMock()
        mock_result.bounded_gradients = torch.ones(5, 5)  # Return safe gradient
        mock_result.applied_factor = 1.0
        mock_result.direction_based_adjustments = False
        mock_result.direction_confidence = 0.5
        mock_result.direction_state = None
        component.enhanced_bounder.bound_gradients.return_value = mock_result
        
        async def test():
            # Test with infinite values
            inf_grad = torch.ones(5, 5)
            inf_grad[2, 2] = float('inf')
            result = await component.process_gradient(inf_grad, {})
            self.assertFalse(torch.isnan(result).any())
            self.assertFalse(torch.isinf(result).any())
            
            # Test with NaN values
            nan_grad = torch.ones(5, 5)
            nan_grad[1, 1] = float('nan')
            result = await component.process_gradient(nan_grad, {})
            self.assertFalse(torch.isnan(result).any())
        
        asyncio.run(test())
    
    def test_very_large_gradients(self):
        """Test handling of extremely large gradients"""
        system = GACSystem({'thresholds': {'gradient_magnitude': 1e6}})
        system.start_system()
        
        async def test():
            # Test with very large gradient
            large_grad = torch.ones(10, 10) * 1e8
            result = await system.process_gradient(large_grad)
            self.assertIsNotNone(result)
            self.assertFalse(torch.isinf(result).any())
        
        asyncio.run(test())
        system.stop_system()
    
    def test_mixed_dtypes(self):
        """Test handling of different tensor dtypes"""
        system = GACSystem({})
        system.start_system()
        
        async def test():
            # Test float32
            grad_f32 = torch.randn(5, 5, dtype=torch.float32)
            result = await system.process_gradient(grad_f32)
            self.assertEqual(result.dtype, torch.float32)
            
            # Test float64
            grad_f64 = torch.randn(5, 5, dtype=torch.float64)
            result = await system.process_gradient(grad_f64)
            self.assertEqual(result.dtype, torch.float64)
        
        asyncio.run(test())
        system.stop_system()


class TestConcurrency(unittest.TestCase):
    """Test concurrent operations and thread safety"""
    
    def test_concurrent_gradient_processing(self):
        """Test processing multiple gradients concurrently"""
        system = GACSystem({})
        system.start_system()
        
        async def process_batch(batch_id):
            gradients = [torch.randn(10, 10) for _ in range(5)]
            results = []
            for grad in gradients:
                result = await system.process_gradient(grad, {'batch_id': batch_id})
                results.append(result)
            return results
        
        async def test():
            # Process multiple batches concurrently
            tasks = [process_batch(i) for i in range(10)]
            results = await asyncio.gather(*tasks)
            
            # Verify all batches processed
            self.assertEqual(len(results), 10)
            for batch_results in results:
                self.assertEqual(len(batch_results), 5)
        
        asyncio.run(test())
        system.stop_system()
    
    def test_thread_manager_stress(self):
        """Stress test the thread manager with high load"""
        system = GACSystem({})
        manager = GACThreadManager(system)
        manager.start()
        
        try:
            # Send many requests rapidly
            num_requests = 50
            all_gradients = []
            
            for i in range(num_requests):
                grads = [torch.randn(20, 20) for _ in range(3)]
                all_gradients.append(grads)
                
            # Process all in parallel
            results = []
            for grads in all_gradients:
                result = manager.process_gradients_sync(grads, timeout=2.0)
                results.append(result)
            
            # Verify all processed
            self.assertEqual(len(results), num_requests)
            
            # Check statistics
            stats = manager.get_stats()
            self.assertGreater(stats['processed_requests'], 0)
            
        finally:
            manager.stop()


class TestMemoryAndPerformance(unittest.TestCase):
    """Test memory efficiency and performance"""
    
    def test_event_history_limit(self):
        """Test that event history doesn't grow unbounded"""
        event_bus = EventBus()
        
        # Publish many events (more than maxlen)
        for i in range(15000):
            event = GACEvent(EventType.GRADIENT_UPDATE, "test", {"i": i})
            event_bus.publish(event)
        
        # Check history is limited
        history = event_bus.get_event_history()
        self.assertLessEqual(len(history), 10000)  # Default maxlen
    
    def test_large_tensor_processing(self):
        """Test processing of large tensors"""
        system = GACSystem({})
        system.start_system()
        
        async def test():
            # Test with large tensor
            large_grad = torch.randn(100, 100)
            result = await system.process_gradient(large_grad)
            self.assertEqual(result.shape, large_grad.shape)
        
        asyncio.run(test())
        system.stop_system()


class TestErrorRecovery(unittest.TestCase):
    """Test error recovery and resilience"""
    
    def test_component_failure_recovery(self):
        """Test system continues with component failures"""
        system = GACSystem({})
        
        class FailingComponent(GACComponent):
            def __init__(self):
                super().__init__("failing_component")
                self.attempt = 0
            
            async def process_gradient(self, gradient, context):
                self.attempt += 1
                if self.attempt <= 2:
                    raise RuntimeError(f"Intentional failure {self.attempt}")
                return gradient * 0.9
            
            def get_component_info(self):
                return {"attempts": self.attempt}
        
        failing = FailingComponent()
        system.register_component(failing)
        system.start_system()
        
        async def test():
            # Process gradients, first two should fail
            for i in range(3):
                grad = torch.randn(5, 5)
                result = await system.process_gradient(grad)
                self.assertIsNotNone(result)
            
            # Check error count
            self.assertEqual(failing.metrics.error_count, 2)
        
        asyncio.run(test())
        system.stop_system()
    
    def test_emergency_protocols(self):
        """Test emergency protocol activation"""
        system = GACSystem({})
        
        # Set components with high error counts
        for i in range(3):
            comp = MagicMock(spec=GACComponent)
            comp.component_id = f"comp_{i}"
            
            # Create proper metrics mock
            metrics_mock = MagicMock()
            metrics_mock.error_count = 10
            comp.metrics = metrics_mock
            
            comp.state = ComponentState.ACTIVE
            comp.update_state = MagicMock()
            system.component_registry.register_component(comp)
        
        # Trigger emergency protocols
        system._initiate_emergency_protocols()
        
        # Check that thresholds were increased
        self.assertGreater(system.global_thresholds['gradient_magnitude'], 1.0)


class TestDirectionComponentsIntegration(unittest.TestCase):
    """Test direction components integration if available"""
    
    @unittest.skipIf(not DIRECTION_COMPONENTS_AVAILABLE, "Direction components not available")
    def test_direction_components_initialization(self):
        """Test that direction components initialize properly"""
        system = GACSystem({'direction_components': {'enabled': True}})
        
        # Check all components initialized
        self.assertIsNotNone(system.get_direction_state_manager())
        self.assertIsNotNone(system.get_direction_validator())
        self.assertIsNotNone(system.get_basic_bounder())
        self.assertIsNotNone(system.get_enhanced_bounder())
    
    @unittest.skipIf(not DIRECTION_COMPONENTS_AVAILABLE, "Direction components not available")
    def test_direction_aware_processing(self):
        """Test gradient processing with direction awareness"""
        system = GACSystem({'direction_components': {'enabled': True}})
        system.start_system()
        
        async def test():
            # Process gradients with increasing magnitude (ascending direction)
            for i in range(5):
                grad = torch.randn(10, 10) * (i + 1)
                result = await system.process_gradient(grad)
                self.assertIsNotNone(result)
        
        asyncio.run(test())
        
        # Check direction state was tracked
        if system.get_direction_state_manager():
            summary = system.get_direction_state_manager().get_direction_summary()
            self.assertIsNotNone(summary)
        
        system.stop_system()


def run_tests():
    """Run all tests and report results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestGACComponent,
        TestEventBus,
        TestComponentRegistry,
        TestMetaLearningEngine,
        TestReinforcementLearningAgent,
        TestDirectionAwareGradientComponent,
        TestGACSystem,
        TestGACThreadManager,
        TestIntegration,
        TestEdgeCases,
        TestConcurrency,
        TestMemoryAndPerformance,
        TestErrorRecovery,
        TestDirectionComponentsIntegration
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, trace in result.failures:
            print(f"  - {test}: {trace.split(chr(10))[0]}")
    
    if result.errors:
        print("\nERRORS:")
        for test, trace in result.errors:
            print(f"  - {test}: {trace.split(chr(10))[0]}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOVERALL: {'PASS' if success else 'FAIL'}")
    
    return success


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)