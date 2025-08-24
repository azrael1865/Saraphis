#!/usr/bin/env python3
"""
Comprehensive test suite for AutoScalingEngine
Tests all methods, edge cases, and integration scenarios
"""

import unittest
import tempfile
import shutil
import time
import threading
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the component to test
from production_scaling.auto_scaling_engine import AutoScalingEngine, create_scaling_engine


class MockMonitor:
    """Mock monitor for testing"""
    
    def __init__(self):
        self.system_statuses = {
            'brain_system': {
                'performance': {
                    'response_time_ms': 100,
                    'error_rate': 0.01,
                    'throughput': 1000
                },
                'resources': {
                    'cpu_percent': 45,
                    'memory_percent': 60
                },
                'health': 0.95
            },
            'compression_system': {
                'performance': {
                    'response_time_ms': 200,
                    'error_rate': 0.02,
                    'throughput': 500
                },
                'resources': {
                    'cpu_percent': 75,
                    'memory_percent': 80
                },
                'health': 0.90
            }
        }
        
        self.agent_statuses = {
            'data_agent': {
                'active_tasks': 5,
                'completed_tasks': 100,
                'task_success_rate': 0.98,
                'response_time': 50,
                'health': 0.95
            },
            'analysis_agent': {
                'active_tasks': 15,
                'completed_tasks': 200,
                'task_success_rate': 0.92,
                'response_time': 250,
                'health': 0.85
            }
        }
    
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
    
    def update_system_status(self, name, status):
        """Update system status for testing scenarios"""
        self.system_statuses[name] = status
    
    def update_agent_status(self, name, status):
        """Update agent status for testing scenarios"""
        self.agent_statuses[name] = status


class TestAutoScalingEngineBasic(unittest.TestCase):
    """Test basic functionality of AutoScalingEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = MockMonitor()
        self.brain_system = Mock()
        self.agent_system = Mock()
        self.engine = AutoScalingEngine(self.monitor, self.brain_system, self.agent_system)
    
    def test_initialization(self):
        """Test engine initialization"""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.monitor, self.monitor)
        self.assertEqual(self.engine.brain_system, self.brain_system)
        self.assertEqual(self.engine.agent_system, self.agent_system)
        self.assertFalse(self.engine.is_running)
        self.assertIsNone(self.engine.scaling_thread)
        
        # Check configuration
        self.assertEqual(self.engine.max_decision_time, 10.0)
        self.assertEqual(self.engine.scale_check_interval, 5.0)
        self.assertEqual(self.engine.min_instances['systems'], 1)
        self.assertEqual(self.engine.max_instances['systems'], 10)
        
        # Check thresholds
        self.assertEqual(self.engine.scale_up_thresholds['cpu_percent'], 80)
        self.assertEqual(self.engine.scale_down_thresholds['cpu_percent'], 30)
    
    def test_current_instances_initialization(self):
        """Test current instances initialization"""
        self.assertIsInstance(self.engine.current_instances['systems'], defaultdict)
        self.assertIsInstance(self.engine.current_instances['agents'], defaultdict)
        self.assertEqual(self.engine.current_instances['systems']['test'], 1)  # Default value
        self.assertEqual(self.engine.current_instances['agents']['test'], 1)  # Default value
    
    def test_scaling_history_initialization(self):
        """Test scaling history initialization"""
        self.assertIsInstance(self.engine.scaling_history, deque)
        self.assertEqual(self.engine.scaling_history.maxlen, 10000)
        self.assertEqual(len(self.engine.scaling_history), 0)
    
    def test_workload_history_initialization(self):
        """Test workload history initialization"""
        self.assertIsInstance(self.engine.workload_history, defaultdict)
        # Test that default creates deque with correct maxlen
        test_history = self.engine.workload_history['test_component']
        self.assertIsInstance(test_history, deque)
        self.assertEqual(test_history.maxlen, 1440)


class TestScalingAnalysis(unittest.TestCase):
    """Test scaling analysis functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = MockMonitor()
        self.brain_system = Mock()
        self.agent_system = Mock()
        self.engine = AutoScalingEngine(self.monitor, self.brain_system, self.agent_system)
    
    def test_analyze_scaling_requirements_no_scaling_needed(self):
        """Test analysis when no scaling is needed"""
        # Set low resource usage
        self.monitor.update_system_status('brain_system', {
            'performance': {'response_time_ms': 50, 'error_rate': 0.005, 'throughput': 1000},
            'resources': {'cpu_percent': 25, 'memory_percent': 30},
            'health': 0.98
        })
        
        requirements = self.engine.analyze_scaling_requirements()
        
        self.assertFalse(requirements['scaling_needed'])
        self.assertEqual(len(requirements['scale_up']['systems']), 0)
        self.assertEqual(len(requirements['scale_down']['systems']), 0)
        self.assertIn('current_load', requirements)
    
    def test_analyze_scaling_requirements_scale_up_needed(self):
        """Test analysis when scale-up is needed"""
        # Set high resource usage
        self.monitor.update_system_status('brain_system', {
            'performance': {'response_time_ms': 600, 'error_rate': 0.08, 'throughput': 1000},
            'resources': {'cpu_percent': 85, 'memory_percent': 90},
            'health': 0.80
        })
        
        requirements = self.engine.analyze_scaling_requirements()
        
        self.assertTrue(requirements['scaling_needed'])
        self.assertGreater(len(requirements['scale_up']['systems']), 0)
        
        # Check scale-up item structure
        scale_up_item = requirements['scale_up']['systems'][0]
        self.assertEqual(scale_up_item['name'], 'brain_system')
        self.assertEqual(scale_up_item['current_instances'], 1)
        self.assertIn('metrics', scale_up_item)
        self.assertIn('reason', scale_up_item)
    
    def test_analyze_scaling_requirements_scale_down_possible(self):
        """Test analysis when scale-down is possible"""
        # Set instance count > 1 and low usage
        self.engine.current_instances['systems']['brain_system'] = 3
        
        self.monitor.update_system_status('brain_system', {
            'performance': {'response_time_ms': 50, 'error_rate': 0.005, 'throughput': 1000},
            'resources': {'cpu_percent': 15, 'memory_percent': 20},
            'health': 0.99
        })
        
        requirements = self.engine.analyze_scaling_requirements()
        
        self.assertTrue(requirements['scaling_needed'])
        self.assertGreater(len(requirements['scale_down']['systems']), 0)
        
        # Check scale-down item structure
        scale_down_item = requirements['scale_down']['systems'][0]
        self.assertEqual(scale_down_item['name'], 'brain_system')
        self.assertEqual(scale_down_item['current_instances'], 3)
    
    def test_agent_scaling_analysis(self):
        """Test agent scaling analysis"""
        # High task load should trigger scale-up
        self.monitor.update_agent_status('analysis_agent', {
            'active_tasks': 25,
            'completed_tasks': 200,
            'task_success_rate': 0.85,
            'response_time': 300,
            'health': 0.75
        })
        
        requirements = self.engine.analyze_scaling_requirements()
        
        self.assertTrue(requirements['scaling_needed'])
        self.assertGreater(len(requirements['scale_up']['agents']), 0)
        
        agent_item = requirements['scale_up']['agents'][0]
        self.assertEqual(agent_item['name'], 'analysis_agent')
    
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


class TestScalingDecisions(unittest.TestCase):
    """Test scaling decision logic"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = MockMonitor()
        self.brain_system = Mock()
        self.agent_system = Mock()
        self.engine = AutoScalingEngine(self.monitor, self.brain_system, self.agent_system)
    
    def test_needs_scale_up_cpu(self):
        """Test scale-up decision based on CPU"""
        metrics = {
            'cpu_percent': 85,
            'memory_percent': 60,
            'response_time_ms': 200,
            'error_rate': 0.02
        }
        
        self.assertTrue(self.engine._needs_scale_up(metrics))
    
    def test_needs_scale_up_memory(self):
        """Test scale-up decision based on memory"""
        metrics = {
            'cpu_percent': 70,
            'memory_percent': 90,
            'response_time_ms': 200,
            'error_rate': 0.02
        }
        
        self.assertTrue(self.engine._needs_scale_up(metrics))
    
    def test_needs_scale_up_response_time(self):
        """Test scale-up decision based on response time"""
        metrics = {
            'cpu_percent': 70,
            'memory_percent': 60,
            'response_time_ms': 600,
            'error_rate': 0.02
        }
        
        self.assertTrue(self.engine._needs_scale_up(metrics))
    
    def test_needs_scale_up_error_rate(self):
        """Test scale-up decision based on error rate"""
        metrics = {
            'cpu_percent': 70,
            'memory_percent': 60,
            'response_time_ms': 200,
            'error_rate': 0.08
        }
        
        self.assertTrue(self.engine._needs_scale_up(metrics))
    
    def test_needs_scale_down_all_low(self):
        """Test scale-down when all metrics are low"""
        metrics = {
            'cpu_percent': 25,
            'memory_percent': 35,
            'response_time_ms': 80,
            'error_rate': 0.005
        }
        
        self.assertTrue(self.engine._needs_scale_down(metrics))
    
    def test_needs_scale_down_one_high(self):
        """Test scale-down rejection when one metric is high"""
        metrics = {
            'cpu_percent': 25,
            'memory_percent': 35,
            'response_time_ms': 200,  # High response time
            'error_rate': 0.005
        }
        
        self.assertFalse(self.engine._needs_scale_down(metrics))
    
    def test_agent_scale_up_decisions(self):
        """Test agent scale-up decisions"""
        # High active tasks
        metrics1 = {'active_tasks': 15, 'response_time': 100, 'task_success_rate': 0.98}
        self.assertTrue(self.engine._needs_agent_scale_up(metrics1))
        
        # High response time
        metrics2 = {'active_tasks': 5, 'response_time': 250, 'task_success_rate': 0.98}
        self.assertTrue(self.engine._needs_agent_scale_up(metrics2))
        
        # Low success rate
        metrics3 = {'active_tasks': 5, 'response_time': 100, 'task_success_rate': 0.90}
        self.assertTrue(self.engine._needs_agent_scale_up(metrics3))
    
    def test_agent_scale_down_decisions(self):
        """Test agent scale-down decisions"""
        # All metrics good for scale-down
        metrics1 = {'active_tasks': 1, 'response_time': 30, 'task_success_rate': 0.995}
        self.assertTrue(self.engine._needs_agent_scale_down(metrics1))
        
        # Too many active tasks
        metrics2 = {'active_tasks': 5, 'response_time': 30, 'task_success_rate': 0.995}
        self.assertFalse(self.engine._needs_agent_scale_down(metrics2))


class TestScalingExecution(unittest.TestCase):
    """Test scaling execution"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = MockMonitor()
        self.brain_system = Mock()
        self.agent_system = Mock()
        self.engine = AutoScalingEngine(self.monitor, self.brain_system, self.agent_system)
    
    def test_scale_up_component_success(self):
        """Test successful component scale-up"""
        item = {
            'name': 'test_system',
            'current_instances': 2,
            'reason': 'High CPU usage'
        }
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = self.engine._scale_up_component('systems', item)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['component'], 'test_system')
        self.assertEqual(result['action'], 'scale_up')
        self.assertEqual(result['previous_instances'], 2)
        self.assertEqual(result['new_instances'], 3)
        self.assertEqual(self.engine.current_instances['systems']['test_system'], 3)
    
    def test_scale_up_component_at_max(self):
        """Test scale-up when already at maximum instances"""
        item = {
            'name': 'test_system',
            'current_instances': 10,  # At maximum
            'reason': 'High CPU usage'
        }
        
        result = self.engine._scale_up_component('systems', item)
        
        self.assertFalse(result['success'])
        self.assertIn('maximum instances', result['error'])
    
    def test_scale_down_component_success(self):
        """Test successful component scale-down"""
        # Set initial instance count
        self.engine.current_instances['systems']['test_system'] = 3
        
        item = {
            'name': 'test_system',
            'current_instances': 3,
            'reason': 'Low utilization'
        }
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = self.engine._scale_down_component('systems', item)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['component'], 'test_system')
        self.assertEqual(result['action'], 'scale_down')
        self.assertEqual(result['previous_instances'], 3)
        self.assertEqual(result['new_instances'], 2)
        self.assertEqual(self.engine.current_instances['systems']['test_system'], 2)
    
    def test_scale_down_component_at_min(self):
        """Test scale-down when already at minimum instances"""
        item = {
            'name': 'test_system',
            'current_instances': 1,  # At minimum
            'reason': 'Low utilization'
        }
        
        result = self.engine._scale_down_component('systems', item)
        
        self.assertFalse(result['success'])
        self.assertIn('minimum instances', result['error'])
    
    def test_execute_scaling_success(self):
        """Test successful scaling execution"""
        requirements = {
            'scale_up': {
                'systems': [{
                    'name': 'system1',
                    'current_instances': 1,
                    'reason': 'High load'
                }],
                'agents': []
            },
            'scale_down': {
                'systems': [],
                'agents': []
            }
        }
        
        with patch('time.sleep'):
            result = self.engine._execute_scaling(requirements)
        
        self.assertTrue(result['success'])
        self.assertEqual(len(result['operations']), 1)
        self.assertEqual(len(result['errors']), 0)
    
    def test_execute_scaling_with_failures(self):
        """Test scaling execution with failures"""
        # Set system already at max instances
        self.engine.current_instances['systems']['system1'] = 10
        
        requirements = {
            'scale_up': {
                'systems': [{
                    'name': 'system1',
                    'current_instances': 10,  # Already at max
                    'reason': 'High load'
                }],
                'agents': []
            },
            'scale_down': {
                'systems': [],
                'agents': []
            }
        }
        
        result = self.engine._execute_scaling(requirements)
        
        self.assertFalse(result['success'])
        self.assertEqual(len(result['operations']), 1)
        self.assertEqual(len(result['errors']), 1)
        self.assertFalse(result['operations'][0]['success'])


class TestSystemScaling(unittest.TestCase):
    """Test system-level scaling operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = MockMonitor()
        self.brain_system = Mock()
        self.agent_system = Mock()
        self.engine = AutoScalingEngine(self.monitor, self.brain_system, self.agent_system)
    
    def test_execute_system_scaling_scale_up(self):
        """Test system scaling with scale-up"""
        systems_to_scale = [{
            'name': 'brain_system',
            'target_instances': 3,
            'reason': 'Load increase'
        }]
        
        with patch('time.sleep'):
            result = self.engine.execute_system_scaling(systems_to_scale)
        
        self.assertTrue(result['success'])
        self.assertEqual(len(result['scaled_systems']), 1)
        self.assertEqual(len(result['failed_systems']), 0)
        self.assertEqual(result['scaled_systems'][0]['name'], 'brain_system')
        self.assertEqual(result['scaled_systems'][0]['new'], 3)
    
    def test_execute_system_scaling_scale_down(self):
        """Test system scaling with scale-down"""
        # Set initial instances
        self.engine.current_instances['systems']['brain_system'] = 5
        
        systems_to_scale = [{
            'name': 'brain_system',
            'target_instances': 2,
            'reason': 'Load decrease'
        }]
        
        with patch('time.sleep'):
            result = self.engine.execute_system_scaling(systems_to_scale)
        
        self.assertTrue(result['success'])
        self.assertEqual(len(result['scaled_systems']), 1)
        self.assertEqual(result['scaled_systems'][0]['new'], 2)
    
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


class TestAgentScaling(unittest.TestCase):
    """Test agent-level scaling operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = MockMonitor()
        self.brain_system = Mock()
        self.agent_system = Mock()
        self.engine = AutoScalingEngine(self.monitor, self.brain_system, self.agent_system)
    
    def test_execute_agent_scaling_scale_up(self):
        """Test agent scaling with scale-up"""
        agents_to_scale = [{
            'name': 'analysis_agent',
            'target_instances': 4,
            'reason': 'High task load'
        }]
        
        with patch('time.sleep'):
            result = self.engine.execute_agent_scaling(agents_to_scale)
        
        self.assertTrue(result['success'])
        self.assertEqual(len(result['scaled_agents']), 1)
        self.assertEqual(len(result['failed_agents']), 0)
        self.assertEqual(result['scaled_agents'][0]['name'], 'analysis_agent')
        self.assertEqual(result['scaled_agents'][0]['new'], 4)
    
    def test_execute_agent_scaling_invalid_targets(self):
        """Test agent scaling with invalid targets"""
        agents_to_scale = [{
            'name': 'agent1',
            'target_instances': 25,  # Above maximum
            'reason': 'Invalid target'
        }]
        
        result = self.engine.execute_agent_scaling(agents_to_scale)
        
        self.assertEqual(len(result['failed_agents']), 1)
        self.assertEqual(len(result['scaled_agents']), 0)
        self.assertIn('maximum', result['failed_agents'][0]['error'])


class TestPredictiveScaling(unittest.TestCase):
    """Test predictive scaling functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = MockMonitor()
        self.brain_system = Mock()
        self.agent_system = Mock()
        self.engine = AutoScalingEngine(self.monitor, self.brain_system, self.agent_system)
    
    def test_predict_scaling_needs_insufficient_data(self):
        """Test prediction with insufficient historical data"""
        predictions = self.engine.predict_scaling_needs()
        
        self.assertEqual(len(predictions['predictions']), 0)
        # Should not fail with insufficient data
    
    def test_predict_scaling_needs_with_data(self):
        """Test prediction with sufficient historical data"""
        # Add historical data
        component_name = 'test_system'
        history = self.engine.workload_history[component_name]
        
        # Add 65 data points (more than minimum 60)
        for i in range(65):
            history.append({
                'timestamp': time.time() - (65-i) * 60,  # One per minute
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
    
    def test_calculate_trend(self):
        """Test trend calculation"""
        # Increasing trend
        values = [10, 15, 20, 25, 30]
        trend = self.engine._calculate_trend(values)
        self.assertGreater(trend, 0)
        
        # Decreasing trend
        values = [30, 25, 20, 15, 10]
        trend = self.engine._calculate_trend(values)
        self.assertLess(trend, 0)
        
        # No trend (flat)
        values = [20, 20, 20, 20, 20]
        trend = self.engine._calculate_trend(values)
        self.assertEqual(trend, 0)
    
    def test_calculate_required_instances(self):
        """Test required instances calculation"""
        # Low load
        required = self.engine._calculate_required_instances(50, 'systems')
        self.assertEqual(required, 1)  # Should be minimum
        
        # High load
        required = self.engine._calculate_required_instances(1500, 'systems')
        self.assertEqual(required, 10)  # Should be capped at maximum
        
        # Normal load
        required = self.engine._calculate_required_instances(250, 'systems')
        self.assertEqual(required, 3)  # 250/100 = 2.5 -> ceil(2.5) = 3
    
    def test_calculate_prediction_confidence(self):
        """Test prediction confidence calculation"""
        # Insufficient data
        history = deque(maxlen=1440)
        confidence = self.engine._calculate_prediction_confidence(history)
        self.assertEqual(confidence, 0.5)
        
        # Stable data (low variance)
        for i in range(150):
            history.append({'load': 100 + (i % 2)})  # Very low variance
        confidence = self.engine._calculate_prediction_confidence(history)
        self.assertGreater(confidence, 0.9)


class TestScalingOptimization(unittest.TestCase):
    """Test scaling optimization functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = MockMonitor()
        self.brain_system = Mock()
        self.agent_system = Mock()
        self.engine = AutoScalingEngine(self.monitor, self.brain_system, self.agent_system)
    
    def test_optimize_scaling_strategy_insufficient_data(self):
        """Test optimization with insufficient data"""
        optimization = self.engine.optimize_scaling_strategy()
        
        self.assertIn('message', optimization)
        self.assertEqual(optimization['message'], 'Insufficient data for optimization')
    
    def test_optimize_scaling_strategy_slow_scale_up(self):
        """Test optimization when scale-up is too slow"""
        # Add scaling history with slow response times
        for i in range(150):
            self.engine.scaling_history.append({
                'action': 'scale_up',
                'response_time': 10.0,  # Slow response time
                'success': True
            })
        
        initial_threshold = self.engine.scale_up_thresholds['cpu_percent']
        optimization = self.engine.optimize_scaling_strategy()
        
        self.assertGreater(len(optimization['optimizations']), 0)
        self.assertLess(
            self.engine.scale_up_thresholds['cpu_percent'], 
            initial_threshold
        )
    
    def test_optimize_scaling_strategy_high_failures(self):
        """Test optimization when scale-down failures are high"""
        # Add scaling history with many failures
        for i in range(150):
            self.engine.scaling_history.append({
                'action': 'scale_down',
                'response_time': 5.0,
                'success': i < 120  # 20% failure rate
            })
        
        initial_threshold = self.engine.scale_down_thresholds['cpu_percent']
        optimization = self.engine.optimize_scaling_strategy()
        
        self.assertGreater(len(optimization['optimizations']), 0)
        self.assertGreater(
            self.engine.scale_down_thresholds['cpu_percent'], 
            initial_threshold
        )


class TestScalingValidation(unittest.TestCase):
    """Test scaling validation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = MockMonitor()
        self.brain_system = Mock()
        self.agent_system = Mock()
        self.engine = AutoScalingEngine(self.monitor, self.brain_system, self.agent_system)
    
    def test_validate_scaling_impact(self):
        """Test scaling impact validation"""
        operations = [{
            'success': True,
            'component': 'brain_system',
            'type': 'systems',
            'action': 'scale_up'
        }]
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            validation = self.engine.validate_scaling_impact(operations)
        
        self.assertEqual(validation['operations_validated'], 1)
        self.assertEqual(len(validation['validations']), 1)
        
        validation_item = validation['validations'][0]
        self.assertEqual(validation_item['component'], 'brain_system')
        self.assertEqual(validation_item['action'], 'scale_up')
        self.assertIn('impact', validation_item)
    
    def test_calculate_scaling_impact_scale_up(self):
        """Test scaling impact calculation for scale-up"""
        operation = {'action': 'scale_up'}
        metrics = {'health_score': 0.95}
        
        impact = self.engine._calculate_scaling_impact(operation, metrics)
        
        self.assertIn('expected_improvement', impact)
        self.assertIn('actual_improvement', impact)
        self.assertIn('efficiency', impact)
        self.assertTrue(impact['performance_maintained'])
    
    def test_calculate_scaling_impact_scale_down(self):
        """Test scaling impact calculation for scale-down"""
        operation = {'action': 'scale_down'}
        metrics = {'health_score': 0.88}
        
        impact = self.engine._calculate_scaling_impact(operation, metrics)
        
        self.assertIn('expected_improvement', impact)
        self.assertIn('actual_improvement', impact)
        self.assertIn('efficiency', impact)
        self.assertTrue(impact['performance_maintained'])


class TestScalingPerformanceMonitoring(unittest.TestCase):
    """Test scaling performance monitoring"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = MockMonitor()
        self.brain_system = Mock()
        self.agent_system = Mock()
        self.engine = AutoScalingEngine(self.monitor, self.brain_system, self.agent_system)
    
    def test_monitor_scaling_performance_no_operations(self):
        """Test performance monitoring with no operations"""
        performance = self.engine.monitor_scaling_performance()
        
        self.assertEqual(performance['overall_efficiency'], 0.0)
        self.assertEqual(len(performance['metrics']), 0)
        self.assertEqual(len(performance['recommendations']), 0)
    
    def test_monitor_scaling_performance_with_operations(self):
        """Test performance monitoring with scaling operations"""
        # Add some scaling metrics
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
    
    def test_performance_recommendations(self):
        """Test performance monitoring recommendations"""
        # Set poor performance metrics
        self.engine.scaling_metrics['systems'].update({
            'total_scaling_operations': 100,
            'successful_scaling': 85,  # Low success rate
            'failed_scaling': 15,
            'average_scaling_time': 12.0,  # High scaling time
            'total_scale_ups': 85,  # High scale-up ratio
            'total_scale_downs': 15
        })
        
        performance = self.engine.monitor_scaling_performance()
        
        # Should have recommendations for all issues
        self.assertGreater(len(performance['recommendations']), 2)
        
        recommendations_text = ' '.join(performance['recommendations'])
        self.assertIn('Low success rate', recommendations_text)
        self.assertIn('High average scaling time', recommendations_text)
        self.assertIn('Frequent scale-ups', recommendations_text)


class TestAutoScalingLifecycle(unittest.TestCase):
    """Test auto-scaling lifecycle operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = MockMonitor()
        self.brain_system = Mock()
        self.agent_system = Mock()
        self.engine = AutoScalingEngine(self.monitor, self.brain_system, self.agent_system)
    
    def test_start_auto_scaling(self):
        """Test starting auto-scaling"""
        result = self.engine.start_auto_scaling()
        
        self.assertTrue(result['success'])
        self.assertTrue(self.engine.is_running)
        self.assertIsNotNone(self.engine.scaling_thread)
        
        # Clean up
        self.engine.stop_auto_scaling()
    
    def test_start_auto_scaling_already_running(self):
        """Test starting auto-scaling when already running"""
        self.engine.start_auto_scaling()
        
        # Try to start again
        result = self.engine.start_auto_scaling()
        
        self.assertFalse(result['success'])
        self.assertIn('already running', result['error'])
        
        # Clean up
        self.engine.stop_auto_scaling()
    
    def test_stop_auto_scaling(self):
        """Test stopping auto-scaling"""
        self.engine.start_auto_scaling()
        
        # Give it a moment to start
        time.sleep(0.1)
        
        result = self.engine.stop_auto_scaling()
        
        self.assertTrue(result['success'])
        self.assertFalse(self.engine.is_running)
        self.assertIn('final_metrics', result)
        self.assertIn('total_operations', result)
    
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


class TestScalingReporting(unittest.TestCase):
    """Test scaling reporting functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = MockMonitor()
        self.brain_system = Mock()
        self.agent_system = Mock()
        self.engine = AutoScalingEngine(self.monitor, self.brain_system, self.agent_system)
    
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
        
        # Should not have recommendations for successful operation
        self.assertEqual(len([r for r in report['recommendations'] 
                            if 'failed' in r.lower()]), 0)
    
    def test_generate_scaling_report_with_issues(self):
        """Test report generation with scaling issues"""
        # Set instances near max
        self.engine.current_instances['systems']['critical_system'] = 9
        
        scaling_results = {
            'success': False,
            'operations': [
                {'success': False, 'component': 'system1', 'action': 'scale_up'}
            ],
            'total_time': 12.0  # Exceeds limit
        }
        
        report = self.engine.generate_scaling_report(scaling_results)
        
        # Should have recommendations for both failure and slow time
        recommendations = report['recommendations']
        self.assertGreater(len(recommendations), 1)
        
        # Check for specific recommendations
        has_failure_rec = any('failed' in r.lower() for r in recommendations)
        has_time_rec = any('exceeds' in r for r in recommendations)
        has_capacity_rec = any('approaching' in r for r in recommendations)
        
        self.assertTrue(has_failure_rec)
        self.assertTrue(has_time_rec)
        self.assertTrue(has_capacity_rec)


class TestFactoryFunction(unittest.TestCase):
    """Test factory function"""
    
    def test_create_scaling_engine_success(self):
        """Test successful scaling engine creation"""
        monitor = MockMonitor()
        brain_system = Mock()
        agent_system = Mock()
        
        result = create_scaling_engine(monitor, brain_system, agent_system)
        
        self.assertTrue(result['success'])
        self.assertIn('engine', result)
        self.assertIsInstance(result['engine'], AutoScalingEngine)
    
    def test_create_scaling_engine_failure(self):
        """Test scaling engine creation failure"""
        # Pass invalid arguments
        with patch('production_scaling.auto_scaling_engine.AutoScalingEngine') as mock_engine:
            mock_engine.side_effect = Exception("Creation failed")
            
            result = create_scaling_engine(None, None, None)
            
            self.assertFalse(result['success'])
            self.assertIn('error', result)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = MockMonitor()
        self.brain_system = Mock()
        self.agent_system = Mock()
        self.engine = AutoScalingEngine(self.monitor, self.brain_system, self.agent_system)
    
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
    
    def test_workload_history_update_with_errors(self):
        """Test workload history update with monitor errors"""
        # Mock monitor to raise exception
        self.monitor.get_all_system_status = Mock(side_effect=Exception("Monitor error"))
        
        # Should not crash on error
        self.engine._update_workload_history()
        
        # History should remain empty
        self.assertEqual(len(self.engine.workload_history), 0)
    
    def test_scale_reason_generation(self):
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


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAutoScalingEngineBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestScalingAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestScalingDecisions))
    suite.addTests(loader.loadTestsFromTestCase(TestScalingExecution))
    suite.addTests(loader.loadTestsFromTestCase(TestSystemScaling))
    suite.addTests(loader.loadTestsFromTestCase(TestAgentScaling))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictiveScaling))
    suite.addTests(loader.loadTestsFromTestCase(TestScalingOptimization))
    suite.addTests(loader.loadTestsFromTestCase(TestScalingValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestScalingPerformanceMonitoring))
    suite.addTests(loader.loadTestsFromTestCase(TestAutoScalingLifecycle))
    suite.addTests(loader.loadTestsFromTestCase(TestScalingReporting))
    suite.addTests(loader.loadTestsFromTestCase(TestFactoryFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
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
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)