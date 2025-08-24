#!/usr/bin/env python3
"""
Test AutoScalingEngine Hard Failures - NO FALLBACKS
Verifies that all failures are hard failures with proper exceptions
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from production_scaling.auto_scaling_engine import AutoScalingEngine, create_scaling_engine


class BrokenMonitor:
    """Monitor that intentionally breaks to test hard failures"""
    
    def __init__(self, break_all_system=False, break_all_agent=False, 
                 break_get_system=False, break_get_agent=False):
        self.break_all_system = break_all_system
        self.break_all_agent = break_all_agent
        self.break_get_system = break_get_system
        self.break_get_agent = break_get_agent
    
    def get_all_system_status(self):
        if self.break_all_system:
            raise RuntimeError("Monitor get_all_system_status HARD FAILURE")
        return {'test_system': {
            'performance': {'response_time_ms': 100, 'error_rate': 0.01},
            'resources': {'cpu_percent': 85, 'memory_percent': 60},  # High CPU
            'health': 0.90
        }}
    
    def get_all_agent_status(self):
        if self.break_all_agent:
            raise RuntimeError("Monitor get_all_agent_status HARD FAILURE")
        return {'test_agent': {
            'active_tasks': 15,  # High task load
            'completed_tasks': 100,
            'task_success_rate': 0.95,
            'response_time': 100,
            'health': 0.90
        }}
    
    def get_system_status(self, name):
        if self.break_get_system:
            raise RuntimeError("Monitor get_system_status HARD FAILURE")
        return {
            'performance': {'response_time_ms': 100, 'error_rate': 0.01},
            'resources': {'cpu_percent': 70, 'memory_percent': 60},
            'health': 0.95
        }
    
    def get_agent_status(self, name):
        if self.break_get_agent:
            raise RuntimeError("Monitor get_agent_status HARD FAILURE")
        return {
            'active_tasks': 5,
            'completed_tasks': 100,
            'task_success_rate': 0.95,
            'response_time': 100,
            'health': 0.90
        }


class IncompleteMonitor:
    """Monitor missing required methods"""
    
    def get_all_system_status(self):
        return {}
    
    def get_all_agent_status(self):
        return {}
    
    # Missing get_system_status and get_agent_status methods


class TestHardFailures(unittest.TestCase):
    """Test that all failures are hard failures with no fallbacks"""
    
    def test_analyze_scaling_requirements_monitor_failure(self):
        """Test hard failure when monitor fails during analysis"""
        monitor = BrokenMonitor(break_all_system=True)
        engine = AutoScalingEngine(monitor, None, None)
        
        with self.assertRaises(RuntimeError) as context:
            engine.analyze_scaling_requirements()
        
        self.assertIn("get_all_system_status HARD FAILURE", str(context.exception))
    
    def test_analyze_scaling_requirements_agent_monitor_failure(self):
        """Test hard failure when agent monitor fails during analysis"""
        monitor = BrokenMonitor(break_all_agent=True)
        engine = AutoScalingEngine(monitor, None, None)
        
        with self.assertRaises(RuntimeError) as context:
            engine.analyze_scaling_requirements()
        
        self.assertIn("get_all_agent_status HARD FAILURE", str(context.exception))
    
    def test_validate_scaling_impact_missing_get_system_status(self):
        """Test hard failure when monitor missing get_system_status"""
        monitor = IncompleteMonitor()
        engine = AutoScalingEngine(monitor, None, None)
        
        operations = [{
            'success': True,
            'component': 'test_system',
            'type': 'systems',
            'action': 'scale_up'
        }]
        
        with self.assertRaises(AttributeError) as context:
            engine.validate_scaling_impact(operations)
        
        self.assertIn("Monitor must implement get_system_status() method - NO FALLBACKS", str(context.exception))
    
    def test_validate_scaling_impact_missing_get_agent_status(self):
        """Test hard failure when monitor missing get_agent_status"""
        monitor = IncompleteMonitor()
        engine = AutoScalingEngine(monitor, None, None)
        
        operations = [{
            'success': True,
            'component': 'test_agent',
            'type': 'agents',
            'action': 'scale_up'
        }]
        
        with self.assertRaises(AttributeError) as context:
            engine.validate_scaling_impact(operations)
        
        self.assertIn("Monitor must implement get_agent_status() method - NO FALLBACKS", str(context.exception))
    
    def test_validate_scaling_impact_failed_operation(self):
        """Test hard failure when trying to validate failed operation"""
        monitor = BrokenMonitor()
        engine = AutoScalingEngine(monitor, None, None)
        
        operations = [{
            'success': False,  # Failed operation
            'component': 'test_system',
            'type': 'systems',
            'action': 'scale_up',
            'error': 'Some scaling error'
        }]
        
        with self.assertRaises(RuntimeError) as context:
            engine.validate_scaling_impact(operations)
        
        self.assertIn("Cannot validate failed operation", str(context.exception))
    
    def test_predict_scaling_needs_insufficient_data(self):
        """Test hard failure when insufficient workload history"""
        monitor = BrokenMonitor()
        engine = AutoScalingEngine(monitor, None, None)
        
        # Add insufficient history (less than 60 entries)
        engine.workload_history['test_component'] = [
            {'load': 50, 'timestamp': 1000} for _ in range(30)  # Only 30 entries
        ]
        
        with self.assertRaises(ValueError) as context:
            engine.predict_scaling_needs()
        
        self.assertIn("Insufficient workload history for test_component: 30 < 60 required", str(context.exception))
    
    def test_update_workload_history_monitor_failure(self):
        """Test hard failure when monitor fails during workload update"""
        monitor = BrokenMonitor(break_all_system=True)
        engine = AutoScalingEngine(monitor, None, None)
        
        with self.assertRaises(RuntimeError) as context:
            engine._update_workload_history()
        
        self.assertIn("get_all_system_status HARD FAILURE", str(context.exception))
    
    def test_record_scaling_operation_failed_operation(self):
        """Test hard failure when recording failed operation"""
        monitor = BrokenMonitor()
        engine = AutoScalingEngine(monitor, None, None)
        
        requirements = {'scaling_needed': True}
        results = {
            'success': False,
            'operations': [{
                'success': False,  # Failed operation
                'type': 'systems',
                'action': 'scale_up',
                'error': 'Scaling failed'
            }]
        }
        
        with self.assertRaises(RuntimeError) as context:
            engine._record_scaling_operation(requirements, results, 5.0)
        
        self.assertIn("Cannot record metrics for failed operation", str(context.exception))
    
    def test_monitor_scaling_performance_no_operations(self):
        """Test hard failure when monitoring performance with no operations"""
        monitor = BrokenMonitor()
        engine = AutoScalingEngine(monitor, None, None)
        
        # No operations recorded
        with self.assertRaises(ValueError) as context:
            engine.monitor_scaling_performance()
        
        self.assertIn("No scaling operations recorded for systems", str(context.exception))
    
    def test_scale_up_component_hard_failure(self):
        """Test that scale up component failures are hard failures"""
        monitor = BrokenMonitor()
        engine = AutoScalingEngine(monitor, None, None)
        
        # Force an exception by providing invalid data
        item = {
            'name': None,  # Invalid name will cause exception
            'current_instances': 2,
            'reason': 'Test failure'
        }
        
        with self.assertRaises(Exception):
            engine._scale_up_component('systems', item)
    
    def test_scale_down_component_hard_failure(self):
        """Test that scale down component failures are hard failures"""
        monitor = BrokenMonitor()
        engine = AutoScalingEngine(monitor, None, None)
        
        # Force an exception by providing invalid data
        item = {
            'name': None,  # Invalid name will cause exception
            'current_instances': 2,
            'reason': 'Test failure'
        }
        
        with self.assertRaises(Exception):
            engine._scale_down_component('systems', item)
    
    def test_execute_system_scaling_hard_failure(self):
        """Test that system scaling failures are hard failures"""
        monitor = BrokenMonitor()
        engine = AutoScalingEngine(monitor, None, None)
        
        # Create invalid scaling request to force failure
        systems_to_scale = [{
            'name': None,  # Invalid name
            'target_instances': 3,
            'reason': 'Test failure'
        }]
        
        with self.assertRaises(Exception):
            engine.execute_system_scaling(systems_to_scale)
    
    def test_execute_agent_scaling_hard_failure(self):
        """Test that agent scaling failures are hard failures"""
        monitor = BrokenMonitor()
        engine = AutoScalingEngine(monitor, None, None)
        
        # Create invalid scaling request to force failure
        agents_to_scale = [{
            'name': None,  # Invalid name
            'target_instances': 3,
            'reason': 'Test failure'
        }]
        
        with self.assertRaises(Exception):
            engine.execute_agent_scaling(agents_to_scale)
    
    def test_execute_scaling_hard_failure(self):
        """Test that scaling execution failures are hard failures"""
        monitor = BrokenMonitor()
        engine = AutoScalingEngine(monitor, None, None)
        
        # Create invalid requirements to force failure
        requirements = {
            'scale_up': {
                'systems': [{
                    'name': None,  # Invalid name will cause exception
                    'current_instances': 1,
                    'reason': 'Test failure'
                }],
                'agents': []
            },
            'scale_down': {'systems': [], 'agents': []}
        }
        
        with self.assertRaises(Exception):
            engine._execute_scaling(requirements)
    
    def test_create_scaling_engine_failure_still_returns_error_dict(self):
        """Test that factory function still returns error dict for compatibility"""
        # This is the one exception where we return error dict instead of raising
        # because it's a factory function that consumers expect to return result dict
        
        # Pass invalid arguments to cause creation failure
        result = create_scaling_engine(None, None, None)
        
        self.assertFalse(result['success'])
        self.assertIn('error', result)


def run_tests():
    """Run all hard failure tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestHardFailures)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print("HARD FAILURE TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)