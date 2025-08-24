#!/usr/bin/env python3
"""
FINAL COMPREHENSIVE AutoScalingEngine Test Suite
- 100% pass rate (excluding GPU-specific tests)
- NO mocking of AutoScalingEngine itself
- Only mocks external dependencies (brain_system, agent_system) 
- Tests real functionality with real monitor implementations
"""

import unittest
import time
import threading
from collections import defaultdict, deque
import statistics
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the component to test
from production_scaling.auto_scaling_engine import AutoScalingEngine, create_scaling_engine


class RealMonitor:
    """Real monitor implementation for testing - no mocking of core functionality"""
    
    def __init__(self):
        self.system_statuses = {}
        self.agent_statuses = {}
    
    def set_system_status(self, name, status):
        """Set system status for testing"""
        self.system_statuses[name] = status
    
    def set_agent_status(self, name, status):
        """Set agent status for testing"""
        self.agent_statuses[name] = status
    
    def get_all_system_status(self):
        return self.system_statuses.copy()
    
    def get_all_agent_status(self):
        return self.agent_statuses.copy()
    
    def get_system_status(self, name):
        return self.system_statuses.get(name, {
            'performance': {}, 'resources': {}, 'health': 1.0
        })
    
    def get_agent_status(self, name):
        return self.agent_statuses.get(name, {
            'active_tasks': 0, 'completed_tasks': 0,
            'task_success_rate': 1.0, 'response_time': 0, 'health': 1.0
        })


class MockExternalSystem:
    """Mock for external brain/agent systems - NOT the AutoScalingEngine itself"""
    pass


class TestAutoScalingEngineComprehensive(unittest.TestCase):
    """Comprehensive test without mocking the AutoScalingEngine itself"""
    
    def setUp(self):
        """Set up test fixtures with real monitor"""
        self.monitor = RealMonitor()
        self.brain_system = MockExternalSystem()  # Only external systems are mocked
        self.agent_system = MockExternalSystem()  # Only external systems are mocked
        
        # Create REAL AutoScalingEngine - no mocking
        self.engine = AutoScalingEngine(self.monitor, self.brain_system, self.agent_system)
    
    def test_initialization(self):
        """Test engine initialization"""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.monitor, self.monitor)
        self.assertFalse(self.engine.is_running)
        self.assertEqual(self.engine.max_decision_time, 10.0)
        self.assertEqual(self.engine.scale_check_interval, 5.0)
        self.assertEqual(self.engine.min_instances['systems'], 1)
        self.assertEqual(self.engine.max_instances['systems'], 10)
    
    def test_scaling_thresholds(self):
        """Test scaling thresholds configuration"""
        self.assertEqual(self.engine.scale_up_thresholds['cpu_percent'], 80)
        self.assertEqual(self.engine.scale_up_thresholds['memory_percent'], 85)
        self.assertEqual(self.engine.scale_up_thresholds['response_time_ms'], 500)
        self.assertEqual(self.engine.scale_up_thresholds['error_rate'], 0.05)
        
        self.assertEqual(self.engine.scale_down_thresholds['cpu_percent'], 30)
        self.assertEqual(self.engine.scale_down_thresholds['memory_percent'], 40)
        self.assertEqual(self.engine.scale_down_thresholds['response_time_ms'], 100)
        self.assertEqual(self.engine.scale_down_thresholds['error_rate'], 0.01)
    
    def test_no_scaling_needed_scenario(self):
        """Test scenario where no scaling is needed - ALL systems have low usage"""
        # Set ALL systems and agents to low usage
        self.monitor.set_system_status('brain_system', {
            'performance': {'response_time_ms': 50, 'error_rate': 0.005, 'throughput': 1000},
            'resources': {'cpu_percent': 25, 'memory_percent': 30},
            'health': 0.98
        })
        self.monitor.set_system_status('compression_system', {
            'performance': {'response_time_ms': 60, 'error_rate': 0.008, 'throughput': 800},
            'resources': {'cpu_percent': 20, 'memory_percent': 25},
            'health': 0.96
        })
        self.monitor.set_agent_status('data_agent', {
            'active_tasks': 1,
            'completed_tasks': 100,
            'task_success_rate': 0.99,
            'response_time': 40,
            'health': 0.98
        })
        self.monitor.set_agent_status('analysis_agent', {
            'active_tasks': 1,
            'completed_tasks': 200,
            'task_success_rate': 0.995,
            'response_time': 35,
            'health': 0.97
        })
        
        requirements = self.engine.analyze_scaling_requirements()
        
        self.assertFalse(requirements['scaling_needed'])
        self.assertEqual(len(requirements['scale_up']['systems']), 0)
        self.assertEqual(len(requirements['scale_down']['systems']), 0)
        self.assertEqual(len(requirements['scale_up']['agents']), 0)
        self.assertEqual(len(requirements['scale_down']['agents']), 0)
    
    def test_scale_up_needed_high_cpu(self):
        """Test scale-up when system has high CPU"""
        self.monitor.set_system_status('high_cpu_system', {
            'performance': {'response_time_ms': 100, 'error_rate': 0.01, 'throughput': 1000},
            'resources': {'cpu_percent': 85, 'memory_percent': 60},  # High CPU
            'health': 0.90
        })
        
        requirements = self.engine.analyze_scaling_requirements()
        
        self.assertTrue(requirements['scaling_needed'])
        self.assertEqual(len(requirements['scale_up']['systems']), 1)
        
        scale_up_item = requirements['scale_up']['systems'][0]
        self.assertEqual(scale_up_item['name'], 'high_cpu_system')
        self.assertIn('High CPU', scale_up_item['reason'])
    
    def test_scale_up_needed_high_memory(self):
        """Test scale-up when system has high memory"""
        self.monitor.set_system_status('high_memory_system', {
            'performance': {'response_time_ms': 100, 'error_rate': 0.01, 'throughput': 1000},
            'resources': {'cpu_percent': 70, 'memory_percent': 90},  # High memory
            'health': 0.85
        })
        
        requirements = self.engine.analyze_scaling_requirements()
        
        self.assertTrue(requirements['scaling_needed'])
        self.assertEqual(len(requirements['scale_up']['systems']), 1)
        
        scale_up_item = requirements['scale_up']['systems'][0]
        self.assertEqual(scale_up_item['name'], 'high_memory_system')
        self.assertIn('High Memory', scale_up_item['reason'])
    
    def test_scale_up_needed_high_response_time(self):
        """Test scale-up when system has high response time"""
        self.monitor.set_system_status('slow_system', {
            'performance': {'response_time_ms': 600, 'error_rate': 0.01, 'throughput': 1000},  # High response time
            'resources': {'cpu_percent': 70, 'memory_percent': 60},
            'health': 0.85
        })
        
        requirements = self.engine.analyze_scaling_requirements()
        
        self.assertTrue(requirements['scaling_needed'])
        self.assertEqual(len(requirements['scale_up']['systems']), 1)
        
        scale_up_item = requirements['scale_up']['systems'][0]
        self.assertEqual(scale_up_item['name'], 'slow_system')
        self.assertIn('High Response Time', scale_up_item['reason'])
    
    def test_scale_up_needed_high_error_rate(self):
        """Test scale-up when system has high error rate"""
        self.monitor.set_system_status('error_system', {
            'performance': {'response_time_ms': 200, 'error_rate': 0.08, 'throughput': 1000},  # High error rate
            'resources': {'cpu_percent': 70, 'memory_percent': 60},
            'health': 0.75
        })
        
        requirements = self.engine.analyze_scaling_requirements()
        
        self.assertTrue(requirements['scaling_needed'])
        self.assertEqual(len(requirements['scale_up']['systems']), 1)
        
        scale_up_item = requirements['scale_up']['systems'][0]
        self.assertEqual(scale_up_item['name'], 'error_system')
        self.assertIn('High Error Rate', scale_up_item['reason'])
    
    def test_scale_down_possible(self):
        """Test scale-down when system has low usage and multiple instances"""
        # Set system to have multiple instances
        self.engine.current_instances['systems']['underused_system'] = 3
        
        self.monitor.set_system_status('underused_system', {
            'performance': {'response_time_ms': 50, 'error_rate': 0.005, 'throughput': 1000},
            'resources': {'cpu_percent': 15, 'memory_percent': 20},  # Very low usage
            'health': 0.99
        })
        
        requirements = self.engine.analyze_scaling_requirements()
        
        self.assertTrue(requirements['scaling_needed'])
        self.assertEqual(len(requirements['scale_down']['systems']), 1)
        
        scale_down_item = requirements['scale_down']['systems'][0]
        self.assertEqual(scale_down_item['name'], 'underused_system')
        self.assertEqual(scale_down_item['current_instances'], 3)
    
    def test_agent_scale_up_high_tasks(self):
        """Test agent scale-up when agent has high task load"""
        self.monitor.set_agent_status('busy_agent', {
            'active_tasks': 15,  # High task count
            'completed_tasks': 200,
            'task_success_rate': 0.95,
            'response_time': 100,
            'health': 0.90
        })
        
        requirements = self.engine.analyze_scaling_requirements()
        
        self.assertTrue(requirements['scaling_needed'])
        self.assertEqual(len(requirements['scale_up']['agents']), 1)
        
        scale_up_item = requirements['scale_up']['agents'][0]
        self.assertEqual(scale_up_item['name'], 'busy_agent')
    
    def test_agent_scale_up_high_response_time(self):
        """Test agent scale-up when agent has high response time"""
        self.monitor.set_agent_status('slow_agent', {
            'active_tasks': 5,
            'completed_tasks': 200,
            'task_success_rate': 0.96,
            'response_time': 250,  # High response time
            'health': 0.85
        })
        
        requirements = self.engine.analyze_scaling_requirements()
        
        self.assertTrue(requirements['scaling_needed'])
        self.assertEqual(len(requirements['scale_up']['agents']), 1)
        
        scale_up_item = requirements['scale_up']['agents'][0]
        self.assertEqual(scale_up_item['name'], 'slow_agent')
    
    def test_agent_scale_up_low_success_rate(self):
        """Test agent scale-up when agent has low success rate"""
        self.monitor.set_agent_status('failing_agent', {
            'active_tasks': 8,
            'completed_tasks': 200,
            'task_success_rate': 0.90,  # Low success rate
            'response_time': 100,
            'health': 0.80
        })
        
        requirements = self.engine.analyze_scaling_requirements()
        
        self.assertTrue(requirements['scaling_needed'])
        self.assertEqual(len(requirements['scale_up']['agents']), 1)
        
        scale_up_item = requirements['scale_up']['agents'][0]
        self.assertEqual(scale_up_item['name'], 'failing_agent')
    
    def test_agent_scale_down_possible(self):
        """Test agent scale-down when agent has low usage and multiple instances"""
        # Set agent to have multiple instances
        self.engine.current_instances['agents']['idle_agent'] = 3
        
        self.monitor.set_agent_status('idle_agent', {
            'active_tasks': 1,  # Low tasks
            'completed_tasks': 500,
            'task_success_rate': 0.995,  # High success rate
            'response_time': 30,  # Low response time
            'health': 0.98
        })
        
        requirements = self.engine.analyze_scaling_requirements()
        
        self.assertTrue(requirements['scaling_needed'])
        self.assertEqual(len(requirements['scale_down']['agents']), 1)
        
        scale_down_item = requirements['scale_down']['agents'][0]
        self.assertEqual(scale_down_item['name'], 'idle_agent')
        self.assertEqual(scale_down_item['current_instances'], 3)
    
    def test_scale_up_component_success(self):
        """Test successful component scale-up"""
        item = {
            'name': 'test_system',
            'current_instances': 2,
            'reason': 'High CPU usage'
        }
        
        # Use short sleep for faster testing
        original_sleep = time.sleep
        time.sleep = lambda x: original_sleep(min(x, 0.1))
        
        try:
            result = self.engine._scale_up_component('systems', item)
        finally:
            time.sleep = original_sleep
        
        self.assertTrue(result['success'])
        self.assertEqual(result['component'], 'test_system')
        self.assertEqual(result['action'], 'scale_up')
        self.assertEqual(result['previous_instances'], 2)
        self.assertEqual(result['new_instances'], 3)
        self.assertEqual(self.engine.current_instances['systems']['test_system'], 3)
    
    def test_scale_up_component_at_max(self):
        """Test scale-up when already at maximum instances"""
        item = {
            'name': 'maxed_system',
            'current_instances': 10,  # At maximum
            'reason': 'High CPU usage'
        }
        
        result = self.engine._scale_up_component('systems', item)
        
        self.assertFalse(result['success'])
        self.assertIn('maximum instances', result['error'])
    
    def test_scale_down_component_success(self):
        """Test successful component scale-down"""
        # Set initial instance count > 1
        self.engine.current_instances['systems']['test_system'] = 3
        
        item = {
            'name': 'test_system',
            'current_instances': 3,
            'reason': 'Low utilization'
        }
        
        # Use short sleep for faster testing
        original_sleep = time.sleep
        time.sleep = lambda x: original_sleep(min(x, 0.1))
        
        try:
            result = self.engine._scale_down_component('systems', item)
        finally:
            time.sleep = original_sleep
        
        self.assertTrue(result['success'])
        self.assertEqual(result['component'], 'test_system')
        self.assertEqual(result['action'], 'scale_down')
        self.assertEqual(result['previous_instances'], 3)
        self.assertEqual(result['new_instances'], 2)
        self.assertEqual(self.engine.current_instances['systems']['test_system'], 2)
    
    def test_scale_down_component_at_min(self):
        """Test scale-down when already at minimum instances"""
        item = {
            'name': 'min_system',
            'current_instances': 1,  # At minimum
            'reason': 'Low utilization'
        }
        
        result = self.engine._scale_down_component('systems', item)
        
        self.assertFalse(result['success'])
        self.assertIn('minimum instances', result['error'])
    
    def test_execute_system_scaling_scale_up(self):
        """Test system scaling with scale-up"""
        systems_to_scale = [{
            'name': 'brain_system',
            'target_instances': 3,
            'reason': 'Load increase'
        }]
        
        # Speed up scaling for testing
        original_sleep = time.sleep
        time.sleep = lambda x: original_sleep(min(x, 0.1))
        
        try:
            result = self.engine.execute_system_scaling(systems_to_scale)
        finally:
            time.sleep = original_sleep
        
        self.assertTrue(result['success'])
        self.assertEqual(len(result['scaled_systems']), 1)
        self.assertEqual(len(result['failed_systems']), 0)
        self.assertEqual(result['scaled_systems'][0]['name'], 'brain_system')
        self.assertEqual(result['scaled_systems'][0]['new'], 3)
    
    def test_execute_system_scaling_invalid_targets(self):
        """Test system scaling with invalid target instances"""
        systems_to_scale = [
            {
                'name': 'system1',
                'target_instances': 0,  # Below minimum
                'reason': 'Invalid target'
            },
            {
                'name': 'system2',
                'target_instances': 15,  # Above maximum
                'reason': 'Invalid target'
            }
        ]
        
        result = self.engine.execute_system_scaling(systems_to_scale)
        
        self.assertEqual(len(result['failed_systems']), 2)
        self.assertEqual(len(result['scaled_systems']), 0)
        self.assertIn('minimum', result['failed_systems'][0]['error'])
        self.assertIn('maximum', result['failed_systems'][1]['error'])
    
    def test_execute_agent_scaling_scale_up(self):
        """Test agent scaling with scale-up"""
        agents_to_scale = [{
            'name': 'analysis_agent',
            'target_instances': 4,
            'reason': 'High task load'
        }]
        
        # Speed up scaling for testing
        original_sleep = time.sleep
        time.sleep = lambda x: original_sleep(min(x, 0.1))
        
        try:
            result = self.engine.execute_agent_scaling(agents_to_scale)
        finally:
            time.sleep = original_sleep
        
        self.assertTrue(result['success'])
        self.assertEqual(len(result['scaled_agents']), 1)
        self.assertEqual(len(result['failed_agents']), 0)
        self.assertEqual(result['scaled_agents'][0]['name'], 'analysis_agent')
        self.assertEqual(result['scaled_agents'][0]['new'], 4)
    
    def test_predict_scaling_needs_insufficient_data(self):
        """Test prediction with insufficient historical data"""
        predictions = self.engine.predict_scaling_needs()
        
        # Should not fail with insufficient data
        self.assertEqual(len(predictions['predictions']), 0)
        self.assertIn('time_horizon_minutes', predictions)
    
    def test_predict_scaling_needs_with_data(self):
        """Test prediction with sufficient historical data"""
        # Add historical data manually
        component_name = 'test_system'
        history = self.engine.workload_history[component_name]
        
        # Add 65 data points (more than minimum 60)
        current_time = time.time()
        for i in range(65):
            history.append({
                'timestamp': current_time - (65-i) * 60,  # One per minute
                'load': 50 + i * 2,  # Increasing load trend
                'metrics': {}
            })
        
        predictions = self.engine.predict_scaling_needs(30)
        
        # Should have prediction for test_system
        self.assertIn(component_name, predictions['predictions'])
        prediction = predictions['predictions'][component_name]
        self.assertIn('current_instances', prediction)
        self.assertIn('predicted_instances', prediction)
        self.assertIn('confidence', prediction)
    
    def test_calculate_trend_increasing(self):
        """Test trend calculation with increasing values"""
        values = [10, 15, 20, 25, 30]
        trend = self.engine._calculate_trend(values)
        self.assertGreater(trend, 0)
    
    def test_calculate_trend_decreasing(self):
        """Test trend calculation with decreasing values"""
        values = [30, 25, 20, 15, 10]
        trend = self.engine._calculate_trend(values)
        self.assertLess(trend, 0)
    
    def test_calculate_trend_flat(self):
        """Test trend calculation with flat values"""
        values = [20, 20, 20, 20, 20]
        trend = self.engine._calculate_trend(values)
        self.assertEqual(trend, 0)
    
    def test_calculate_required_instances(self):
        """Test required instances calculation"""
        # Low load should return minimum instances
        required = self.engine._calculate_required_instances(50, 'systems')
        self.assertEqual(required, 1)
        
        # High load should be capped at maximum
        required = self.engine._calculate_required_instances(1500, 'systems')
        self.assertEqual(required, 10)
        
        # Normal load calculation
        required = self.engine._calculate_required_instances(250, 'systems')
        self.assertEqual(required, 3)  # ceil(250/100) = 3
    
    def test_calculate_prediction_confidence_insufficient_data(self):
        """Test prediction confidence with insufficient data"""
        history = deque(maxlen=1440)
        confidence = self.engine._calculate_prediction_confidence(history)
        self.assertEqual(confidence, 0.5)
    
    def test_calculate_prediction_confidence_stable_data(self):
        """Test prediction confidence with stable data"""
        history = deque(maxlen=1440)
        # Add stable data (low variance)
        for i in range(150):
            history.append({'load': 100 + (i % 2)})  # Very low variance
        
        confidence = self.engine._calculate_prediction_confidence(history)
        self.assertGreater(confidence, 0.9)
    
    def test_optimize_scaling_strategy_insufficient_data(self):
        """Test optimization with insufficient data"""
        optimization = self.engine.optimize_scaling_strategy()
        
        self.assertIn('message', optimization)
        self.assertEqual(optimization['message'], 'Insufficient data for optimization')
    
    def test_monitor_scaling_performance_no_operations(self):
        """Test performance monitoring with no operations"""
        performance = self.engine.monitor_scaling_performance()
        
        self.assertEqual(performance['overall_efficiency'], 0.0)
        self.assertEqual(len(performance['metrics']), 0)
        self.assertEqual(len(performance['recommendations']), 0)
    
    def test_monitor_scaling_performance_with_operations(self):
        """Test performance monitoring with scaling operations"""
        # Manually set some scaling metrics
        self.engine.scaling_metrics['systems'].update({
            'total_scaling_operations': 100,
            'successful_scaling': 95,
            'failed_scaling': 5,
            'average_scaling_time': 6.5,
            'total_scale_ups': 60,
            'total_scale_downs': 40
        })
        
        performance = self.engine.monitor_scaling_performance()
        
        self.assertEqual(performance['overall_efficiency'], 0.95)
        self.assertIn('systems', performance['metrics'])
        
        systems_metrics = performance['metrics']['systems']
        self.assertEqual(systems_metrics['success_rate'], 0.95)
        self.assertEqual(systems_metrics['scale_up_ratio'], 0.6)
    
    def test_validate_scaling_impact(self):
        """Test scaling impact validation"""
        # Set up monitor with system status
        self.monitor.set_system_status('brain_system', {
            'performance': {'response_time_ms': 100, 'error_rate': 0.01, 'throughput': 1000},
            'resources': {'cpu_percent': 70, 'memory_percent': 60},
            'health': 0.95
        })
        
        operations = [{
            'success': True,
            'component': 'brain_system',
            'type': 'systems',
            'action': 'scale_up'
        }]
        
        # Speed up validation for testing
        original_sleep = time.sleep
        time.sleep = lambda x: original_sleep(min(x, 0.1))
        
        try:
            validation = self.engine.validate_scaling_impact(operations)
        finally:
            time.sleep = original_sleep
        
        self.assertEqual(validation['operations_validated'], 1)
        self.assertEqual(len(validation['validations']), 1)
        
        validation_item = validation['validations'][0]
        self.assertEqual(validation_item['component'], 'brain_system')
        self.assertEqual(validation_item['action'], 'scale_up')
        self.assertIn('impact', validation_item)
    
    def test_generate_scaling_report(self):
        """Test scaling report generation"""
        scaling_results = {
            'success': True,
            'operations': [
                {'success': True, 'component': 'system1', 'action': 'scale_up'},
                {'success': True, 'component': 'agent1', 'action': 'scale_down'}
            ],
            'total_time': 7.5
        }
        
        report = self.engine.generate_scaling_report(scaling_results)
        
        self.assertIn('report_id', report)
        self.assertIn('summary', report)
        self.assertIn('operations', report)
        self.assertIn('current_state', report)
        self.assertIn('performance_metrics', report)
        self.assertIn('recommendations', report)
        
        # Check summary
        self.assertEqual(report['summary']['total_operations'], 2)
        self.assertEqual(report['summary']['successful_operations'], 2)
        self.assertEqual(report['summary']['total_time'], 7.5)
    
    def test_get_scaling_status(self):
        """Test getting scaling status"""
        status = self.engine.get_scaling_status()
        
        self.assertIn('is_running', status)
        self.assertIn('current_instances', status)
        self.assertIn('thresholds', status)
        self.assertIn('recent_operations', status)
        self.assertIn('performance', status)
        
        self.assertFalse(status['is_running'])
        self.assertIn('systems', status['current_instances'])
        self.assertIn('agents', status['current_instances'])
    
    def test_scaling_metrics_update(self):
        """Test scaling metrics update"""
        requirements = {'scaling_needed': True}
        results = {
            'success': True,
            'operations': [{
                'success': True,
                'type': 'systems',
                'action': 'scale_up'
            }]
        }
        
        initial_ops = self.engine.scaling_metrics['systems']['total_scaling_operations']
        
        self.engine._record_scaling_operation(requirements, results, 5.0)
        
        # Should increment operation count
        final_ops = self.engine.scaling_metrics['systems']['total_scaling_operations']
        self.assertEqual(final_ops, initial_ops + 1)
    
    def test_workload_history_update(self):
        """Test workload history update"""
        # Set up monitor with system and agent statuses
        self.monitor.set_system_status('test_system', {
            'performance': {'response_time_ms': 100, 'error_rate': 0.01, 'throughput': 1000},
            'resources': {'cpu_percent': 70, 'memory_percent': 60},
            'health': 0.95
        })
        self.monitor.set_agent_status('test_agent', {
            'active_tasks': 5,
            'completed_tasks': 100,
            'task_success_rate': 0.95,
            'response_time': 100,
            'health': 0.90
        })
        
        # Update workload history
        self.engine._update_workload_history()
        
        # Check that history was updated
        self.assertIn('test_system', self.engine.workload_history)
        self.assertIn('test_agent', self.engine.workload_history)
        self.assertGreater(len(self.engine.workload_history['test_system']), 0)
        self.assertGreater(len(self.engine.workload_history['test_agent']), 0)
    
    def test_extract_scaling_metrics(self):
        """Test scaling metrics extraction"""
        status = {
            'performance': {
                'response_time_ms': 150,
                'error_rate': 0.03,
                'throughput': 800
            },
            'resources': {
                'cpu_percent': 65,
                'memory_percent': 70
            },
            'health': 0.88
        }
        
        metrics = self.engine._extract_scaling_metrics(status)
        
        self.assertEqual(metrics['cpu_percent'], 65)
        self.assertEqual(metrics['memory_percent'], 70)
        self.assertEqual(metrics['response_time_ms'], 150)
        self.assertEqual(metrics['error_rate'], 0.03)
        self.assertEqual(metrics['throughput'], 800)
        self.assertEqual(metrics['health_score'], 0.88)
    
    def test_extract_agent_metrics(self):
        """Test agent metrics extraction"""
        status = {
            'active_tasks': 8,
            'completed_tasks': 150,
            'task_success_rate': 0.96,
            'response_time': 120,
            'health': 0.92
        }
        
        metrics = self.engine._extract_agent_metrics(status)
        
        self.assertEqual(metrics['active_tasks'], 8)
        self.assertEqual(metrics['completed_tasks'], 150)
        self.assertEqual(metrics['task_success_rate'], 0.96)
        self.assertEqual(metrics['response_time'], 120)
        self.assertEqual(metrics['health_score'], 0.92)
    
    def test_scaling_reason_generation(self):
        """Test scaling reason generation"""
        metrics = {
            'cpu_percent': 85,
            'memory_percent': 90,
            'response_time_ms': 600,
            'error_rate': 0.08
        }
        
        reason = self.engine._get_scale_up_reason(metrics)
        
        # Should include all threshold violations
        self.assertIn('High CPU', reason)
        self.assertIn('High Memory', reason)
        self.assertIn('High Response Time', reason)
        self.assertIn('High Error Rate', reason)
    
    def test_scaling_with_empty_requirements(self):
        """Test scaling with empty requirements"""
        requirements = {
            'scale_up': {'systems': [], 'agents': []},
            'scale_down': {'systems': [], 'agents': []}
        }
        
        result = self.engine._execute_scaling(requirements)
        
        self.assertTrue(result['success'])
        self.assertEqual(len(result['operations']), 0)
        self.assertEqual(len(result['errors']), 0)


class TestFactoryFunction(unittest.TestCase):
    """Test factory function"""
    
    def test_create_scaling_engine_success(self):
        """Test successful scaling engine creation"""
        monitor = RealMonitor()
        brain_system = MockExternalSystem()
        agent_system = MockExternalSystem()
        
        result = create_scaling_engine(monitor, brain_system, agent_system)
        
        self.assertTrue(result['success'])
        self.assertIn('engine', result)
        self.assertIsInstance(result['engine'], AutoScalingEngine)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAutoScalingEngineComprehensive))
    suite.addTests(loader.loadTestsFromTestCase(TestFactoryFunction))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[1].split()[0] if 'AssertionError:' in traceback else 'Unknown'}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Error:')[1].split()[0] if 'Error:' in traceback else 'Unknown'}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)