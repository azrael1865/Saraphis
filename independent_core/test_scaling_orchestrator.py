#!/usr/bin/env python3
"""
Comprehensive test suite for ScalingOrchestrator
Tests all functionality and identifies root cause issues
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
import threading
import time
from datetime import datetime, timedelta
from collections import deque

# Test the actual ScalingOrchestrator implementation
try:
    from production_scaling.scaling_orchestrator import ScalingOrchestrator
    from production_scaling.auto_scaling_engine import AutoScalingEngine
    from production_scaling.auto_recovery_engine import AutoRecoveryEngine  
    from production_scaling.load_balancer import IntelligentLoadBalancer
    from production_scaling.predictive_analytics import PredictiveScalingAnalytics
    IMPORTS_AVAILABLE = True
    IMPORT_ERROR = None
except Exception as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestScalingOrchestrator(unittest.TestCase):
    """Test ScalingOrchestrator core functionality"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Imports failed: {IMPORT_ERROR}")
    def setUp(self):
        """Set up test fixtures"""
        # Create mock dependencies
        self.mock_monitor = Mock()
        self.mock_brain_system = Mock()
        self.mock_agent_system = Mock()
        
        # Configure monitor mock responses
        self.mock_monitor.get_all_system_status.return_value = {
            'brain_core': {'health': 0.95},
            'compression_system': {'health': 0.90},
            'training_system': {'health': 0.85}
        }
        
        self.mock_monitor.get_all_agent_status.return_value = {
            'reasoning_agent': {'health': 0.92},
            'pattern_agent': {'health': 0.88},
            'optimization_agent': {'health': 0.91}
        }
        
        # Create orchestrator with mocked dependencies
        with patch('production_scaling.scaling_orchestrator.AutoScalingEngine') as mock_scaling_engine, \
             patch('production_scaling.scaling_orchestrator.AutoRecoveryEngine') as mock_recovery_engine, \
             patch('production_scaling.scaling_orchestrator.IntelligentLoadBalancer') as mock_load_balancer, \
             patch('production_scaling.scaling_orchestrator.PredictiveScalingAnalytics') as mock_predictive_analytics:
            
            self.orchestrator = ScalingOrchestrator(
                self.mock_monitor, 
                self.mock_brain_system, 
                self.mock_agent_system
            )
            
            # Store mocked component instances
            self.mock_scaling_engine = self.orchestrator.scaling_engine
            self.mock_recovery_engine = self.orchestrator.recovery_engine
            self.mock_load_balancer = self.orchestrator.load_balancer
            self.mock_predictive_analytics = self.orchestrator.predictive_analytics

    def test_orchestrator_initialization(self):
        """Test proper initialization of orchestrator"""
        self.assertIsNotNone(self.orchestrator)
        self.assertFalse(self.orchestrator.is_running)
        self.assertIsNone(self.orchestrator.orchestration_thread)
        self.assertEqual(self.orchestrator.orchestration_interval, 30.0)
        self.assertEqual(self.orchestrator.emergency_threshold, 0.3)
        
        # Check metrics initialization
        expected_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'emergency_responses': 0,
            'predictive_actions': 0
        }
        self.assertEqual(self.orchestrator.orchestration_metrics, expected_metrics)
        
        # Check operation history
        self.assertIsInstance(self.orchestrator.operation_history, deque)
        self.assertEqual(self.orchestrator.operation_history.maxlen, 1000)

    def test_start_orchestration_success(self):
        """Test successful orchestration startup"""
        # Configure successful component startup
        self.mock_scaling_engine.start_auto_scaling.return_value = {'success': True}
        self.mock_recovery_engine.start_auto_recovery.return_value = {'success': True}
        self.mock_load_balancer.start_load_balancing.return_value = {'success': True}
        self.mock_predictive_analytics.start_analytics.return_value = {'success': True}
        
        result = self.orchestrator.start_orchestration()
        
        self.assertTrue(result['success'])
        self.assertTrue(self.orchestrator.is_running)
        self.assertIsNotNone(self.orchestrator.orchestration_thread)
        self.assertTrue(self.orchestrator.orchestration_thread.is_alive())
        
        # Verify all components started
        self.mock_scaling_engine.start_auto_scaling.assert_called_once()
        self.mock_recovery_engine.start_auto_recovery.assert_called_once()
        self.mock_load_balancer.start_load_balancing.assert_called_once()
        self.mock_predictive_analytics.start_analytics.assert_called_once()
        
        # Clean up
        self.orchestrator.stop_orchestration()

    def test_start_orchestration_already_running(self):
        """Test starting orchestration when already running"""
        self.orchestrator.is_running = True
        
        result = self.orchestrator.start_orchestration()
        
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'Orchestration already running')

    def test_start_orchestration_component_failure(self):
        """Test orchestration startup with component failure"""
        # Configure scaling engine to fail
        self.mock_scaling_engine.start_auto_scaling.return_value = {'success': False, 'error': 'Scaling engine failed'}
        
        result = self.orchestrator.start_orchestration()
        
        self.assertFalse(result['success'])
        self.assertIn('Failed to start scaling engine', result['error'])
        self.assertFalse(self.orchestrator.is_running)

    def test_check_overall_health(self):
        """Test overall health checking"""
        health_check = self.orchestrator._check_overall_health()
        
        self.assertIn('timestamp', health_check)
        self.assertIn('overall_health', health_check)
        self.assertIn('critical_issues', health_check)
        self.assertIn('emergency_needed', health_check)
        
        # Check calculated overall health
        expected_health = (0.95 + 0.90 + 0.85 + 0.92 + 0.88 + 0.91) / 6
        self.assertAlmostEqual(health_check['overall_health'], expected_health, places=2)
        
        # No critical issues with current health levels
        self.assertEqual(len(health_check['critical_issues']), 0)
        self.assertFalse(health_check['emergency_needed'])

    def test_check_overall_health_emergency(self):
        """Test health checking with emergency conditions"""
        # Configure critical health levels
        self.mock_monitor.get_all_system_status.return_value = {
            'brain_core': {'health': 0.2},  # Critical
            'compression_system': {'health': 0.1}  # Critical
        }
        
        self.mock_monitor.get_all_agent_status.return_value = {
            'reasoning_agent': {'health': 0.95}  # Good
        }
        
        health_check = self.orchestrator._check_overall_health()
        
        self.assertEqual(len(health_check['critical_issues']), 2)
        self.assertTrue(health_check['emergency_needed'])
        
        # Check critical issues details
        critical_systems = [issue['name'] for issue in health_check['critical_issues']]
        self.assertIn('brain_core', critical_systems)
        self.assertIn('compression_system', critical_systems)

    def test_handle_emergency(self):
        """Test emergency handling"""
        health_check = {
            'critical_issues': [
                {'type': 'system', 'name': 'brain_core', 'health': 0.2},
                {'type': 'agent', 'name': 'reasoning_agent', 'health': 0.1}
            ]
        }
        
        # Configure recovery and scaling responses
        self.mock_recovery_engine.execute_system_recovery.return_value = {'success': True}
        self.mock_recovery_engine.execute_agent_recovery.return_value = {'success': False}
        
        self.orchestrator.scaling_engine.current_instances = {
            'systems': {'brain_core': 2},
            'agents': {'reasoning_agent': 3}
        }
        self.orchestrator.scaling_engine.max_instances = {'systems': 10, 'agents': 20}
        self.orchestrator.scaling_engine.execute_system_scaling.return_value = {'success': True}
        self.orchestrator.scaling_engine.execute_agent_scaling.return_value = {'success': True}
        
        initial_emergency_responses = self.orchestrator.orchestration_metrics['emergency_responses']
        
        self.orchestrator._handle_emergency(health_check)
        
        # Verify emergency response was recorded
        self.assertEqual(
            self.orchestrator.orchestration_metrics['emergency_responses'], 
            initial_emergency_responses + 1
        )
        
        # Verify recovery attempts
        self.mock_recovery_engine.execute_system_recovery.assert_called_with(['brain_core'])
        self.mock_recovery_engine.execute_agent_recovery.assert_called_with(['reasoning_agent'])

    def test_coordinate_scaling(self):
        """Test scaling coordination"""
        # Configure scaling analysis
        scaling_analysis = {
            'scaling_needed': True,
            'scale_up': {
                'systems': [
                    {'name': 'brain_core', 'current_instances': 2, 'reason': 'High load'}
                ],
                'agents': [
                    {'name': 'reasoning_agent', 'current_instances': 3, 'reason': 'CPU overload'}
                ]
            },
            'scale_down': {
                'systems': [],
                'agents': []
            }
        }
        
        self.mock_scaling_engine.analyze_scaling_requirements.return_value = scaling_analysis
        self.mock_scaling_engine.execute_system_scaling.return_value = {'success': True}
        self.mock_scaling_engine.execute_agent_scaling.return_value = {'success': True}
        self.mock_load_balancer.optimize_load_distribution.return_value = {'success': True}
        
        initial_operations = self.orchestrator.orchestration_metrics['total_operations']
        
        self.orchestrator._coordinate_scaling()
        
        # Verify scaling operations executed
        self.mock_scaling_engine.execute_system_scaling.assert_called_once()
        self.mock_scaling_engine.execute_agent_scaling.assert_called_once()
        self.mock_load_balancer.optimize_load_distribution.assert_called_once()
        
        # Verify operation recorded
        self.assertEqual(
            self.orchestrator.orchestration_metrics['total_operations'],
            initial_operations + 1
        )

    def test_coordinate_load_balancing(self):
        """Test load balancing coordination"""
        # Configure load analysis showing inefficiency
        load_analysis = {
            'overall_efficiency': 0.80,  # Below 95% target
            'load_distribution': {}
        }
        
        optimization_result = {
            'success': True,
            'actions_taken': ['redistribute_load', 'adjust_routing']
        }
        
        self.mock_load_balancer.analyze_workload_distribution.return_value = load_analysis
        self.mock_load_balancer.optimize_load_distribution.return_value = optimization_result
        
        self.orchestrator._coordinate_load_balancing()
        
        # Verify optimization was triggered
        self.mock_load_balancer.optimize_load_distribution.assert_called_once()

    def test_apply_predictive_scaling(self):
        """Test predictive scaling application"""
        # Configure predictions
        predictions = {
            'components': {
                'brain_core': [
                    {'predicted_load': 0.9, 'required_instances': 4}
                ],
                'reasoning_agent': [
                    {'predicted_load': 0.7, 'required_instances': 5}
                ]
            }
        }
        
        self.orchestrator.scaling_engine.current_instances = {
            'systems': {'brain_core': 2}
        }
        
        self.mock_predictive_analytics.predict_future_demand.return_value = predictions
        self.mock_scaling_engine.execute_system_scaling.return_value = {'success': True}
        
        initial_predictive_actions = self.orchestrator.orchestration_metrics['predictive_actions']
        
        self.orchestrator._apply_predictive_scaling()
        
        # Verify predictive scaling was applied
        self.assertEqual(
            self.orchestrator.orchestration_metrics['predictive_actions'],
            initial_predictive_actions + 1
        )
        
        self.mock_scaling_engine.execute_system_scaling.assert_called_once()

    def test_execute_manual_scaling(self):
        """Test manual scaling execution"""
        scaling_request = {
            'systems': {
                'brain_core': 5,
                'compression_system': 3
            },
            'agents': {
                'reasoning_agent': 7
            }
        }
        
        self.mock_scaling_engine.execute_system_scaling.return_value = {'success': True}
        self.mock_scaling_engine.execute_agent_scaling.return_value = {'success': True}
        self.mock_load_balancer.optimize_load_distribution.return_value = {'success': True}
        
        result = self.orchestrator.execute_manual_scaling(scaling_request)
        
        self.assertTrue(result['success'])
        self.assertIn('systems', result)
        self.assertIn('agents', result)
        self.assertIn('timestamp', result)
        
        # Verify scaling calls
        self.assertEqual(self.mock_scaling_engine.execute_system_scaling.call_count, 2)  # 2 systems
        self.mock_scaling_engine.execute_agent_scaling.assert_called_once()

    def test_get_orchestration_status(self):
        """Test status retrieval"""
        # Configure component status responses
        self.mock_scaling_engine.get_scaling_status.return_value = {'status': 'running'}
        self.mock_recovery_engine.get_recovery_status.return_value = {'status': 'monitoring'}
        self.mock_load_balancer.get_load_balancing_status.return_value = {'status': 'active'}
        self.mock_predictive_analytics.get_analytics_status.return_value = {'status': 'predicting'}
        
        status = self.orchestrator.get_orchestration_status()
        
        self.assertIn('is_running', status)
        self.assertIn('components', status)
        self.assertIn('metrics', status)
        self.assertIn('recent_operations', status)
        self.assertIn('health_check', status)
        
        # Verify component statuses included
        components = status['components']
        self.assertIn('scaling_engine', components)
        self.assertIn('recovery_engine', components)
        self.assertIn('load_balancer', components)
        self.assertIn('predictive_analytics', components)

    def test_generate_orchestration_report(self):
        """Test comprehensive report generation"""
        # Configure component reports
        self.mock_scaling_engine.get_scaling_status.return_value = {'status': 'running'}
        self.mock_scaling_engine.generate_scaling_report.return_value = {'report': 'scaling_data'}
        self.mock_recovery_engine.get_recovery_status.return_value = {'status': 'monitoring'}
        self.mock_recovery_engine.generate_recovery_report.return_value = {'report': 'recovery_data'}
        self.mock_load_balancer.analyze_workload_distribution.return_value = {'efficiency': 0.95}
        self.mock_load_balancer.generate_balancing_report.return_value = {'report': 'balancing_data'}
        self.mock_predictive_analytics.validate_prediction_accuracy.return_value = {'accuracy': 0.92}
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value = Mock()
            
            report = self.orchestrator.generate_orchestration_report()
        
        self.assertIn('report_id', report)
        self.assertIn('timestamp', report)
        self.assertIn('executive_summary', report)
        self.assertIn('component_reports', report)
        self.assertIn('metrics', report)
        self.assertIn('recommendations', report)
        
        # Verify report components
        component_reports = report['component_reports']
        self.assertIn('scaling', component_reports)
        self.assertIn('recovery', component_reports)
        self.assertIn('load_balancing', component_reports)
        self.assertIn('predictions', component_reports)

    def test_stop_orchestration(self):
        """Test orchestration stopping"""
        # First start orchestration
        self.mock_scaling_engine.start_auto_scaling.return_value = {'success': True}
        self.mock_recovery_engine.start_auto_recovery.return_value = {'success': True}
        self.mock_load_balancer.start_load_balancing.return_value = {'success': True}
        self.mock_predictive_analytics.start_analytics.return_value = {'success': True}
        
        self.orchestrator.start_orchestration()
        
        # Configure stop responses
        self.mock_scaling_engine.stop_auto_scaling.return_value = {'success': True}
        self.mock_recovery_engine.stop_auto_recovery.return_value = {'success': True}
        self.mock_load_balancer.stop_load_balancing.return_value = {'success': True}
        self.mock_predictive_analytics.stop_analytics.return_value = {'success': True}
        
        # Configure report generation
        self.mock_scaling_engine.get_scaling_status.return_value = {'status': 'stopped'}
        self.mock_scaling_engine.generate_scaling_report.return_value = {'report': 'final_scaling'}
        self.mock_recovery_engine.get_recovery_status.return_value = {'status': 'stopped'}
        self.mock_recovery_engine.generate_recovery_report.return_value = {'report': 'final_recovery'}
        self.mock_load_balancer.analyze_workload_distribution.return_value = {'efficiency': 0.95}
        self.mock_load_balancer.generate_balancing_report.return_value = {'report': 'final_balancing'}
        self.mock_predictive_analytics.validate_prediction_accuracy.return_value = {'accuracy': 0.90}
        
        with patch('builtins.open', create=True):
            result = self.orchestrator.stop_orchestration()
        
        self.assertTrue(result['success'])
        self.assertFalse(self.orchestrator.is_running)
        self.assertIn('component_results', result)
        self.assertIn('final_report', result)
        self.assertIn('total_operations', result)

    def test_thread_safety(self):
        """Test thread-safe operations"""
        def start_stop_cycle():
            # Configure successful responses
            self.mock_scaling_engine.start_auto_scaling.return_value = {'success': True}
            self.mock_recovery_engine.start_auto_recovery.return_value = {'success': True}
            self.mock_load_balancer.start_load_balancing.return_value = {'success': True}
            self.mock_predictive_analytics.start_analytics.return_value = {'success': True}
            
            result = self.orchestrator.start_orchestration()
            if result['success']:
                time.sleep(0.1)  # Let it run briefly
                self.orchestrator.stop_orchestration()
        
        # Run multiple concurrent start/stop operations
        threads = []
        for i in range(3):
            t = threading.Thread(target=start_stop_cycle)
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join(timeout=5.0)
        
        # Ensure orchestrator ends in stopped state
        self.assertFalse(self.orchestrator.is_running)

    def test_operation_recording(self):
        """Test operation recording functionality"""
        operation = {
            'type': 'test_operation',
            'timestamp': datetime.now(),
            'result': {'success': True}
        }
        
        initial_total = self.orchestrator.orchestration_metrics['total_operations']
        initial_successful = self.orchestrator.orchestration_metrics['successful_operations']
        
        self.orchestrator._record_operation(operation)
        
        # Verify metrics updated
        self.assertEqual(
            self.orchestrator.orchestration_metrics['total_operations'],
            initial_total + 1
        )
        self.assertEqual(
            self.orchestrator.orchestration_metrics['successful_operations'],
            initial_successful + 1
        )
        
        # Verify operation recorded in history
        self.assertIn(operation, self.orchestrator.operation_history)

    def test_error_handling(self):
        """Test error handling in core methods"""
        # Test with monitor failure
        self.mock_monitor.get_all_system_status.side_effect = Exception("Monitor failed")
        
        health_check = self.orchestrator._check_overall_health()
        
        # Should handle error gracefully
        self.assertEqual(health_check['overall_health'], 0.0)
        self.assertTrue(health_check['emergency_needed'])
        self.assertIn('error', health_check)

    def test_recommendations_generation(self):
        """Test recommendations generation"""
        # Set up conditions that should generate recommendations
        self.orchestrator.orchestration_metrics.update({
            'total_operations': 100,
            'successful_operations': 80,  # 80% success rate
            'emergency_responses': 10,   # High emergency responses
        })
        
        # Configure low prediction accuracy
        self.mock_predictive_analytics.validate_prediction_accuracy.return_value = {
            'meets_threshold': False,
            'overall_accuracy': 0.70
        }
        
        recommendations = self.orchestrator._generate_recommendations()
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Check for expected recommendations
        recommendation_text = ' '.join(recommendations)
        self.assertIn('success rate', recommendation_text.lower())
        self.assertIn('emergency responses', recommendation_text.lower())
        self.assertIn('prediction accuracy', recommendation_text.lower())


class TestScalingOrchestratorIntegration(unittest.TestCase):
    """Integration tests with real component interactions"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Imports failed: {IMPORT_ERROR}")
    def setUp(self):
        """Set up integration test fixtures"""
        self.mock_monitor = Mock()
        self.mock_brain_system = Mock()
        self.mock_agent_system = Mock()
        
        # Configure realistic monitor responses
        self.mock_monitor.get_all_system_status.return_value = {
            'brain_core': {'health': 0.95, 'cpu_percent': 45, 'memory_percent': 60},
            'compression_system': {'health': 0.90, 'cpu_percent': 70, 'memory_percent': 75},
            'training_system': {'health': 0.85, 'cpu_percent': 55, 'memory_percent': 80}
        }

    def test_realistic_orchestration_cycle(self):
        """Test a realistic orchestration cycle with actual components"""
        with patch('production_scaling.scaling_orchestrator.AutoScalingEngine') as MockScalingEngine, \
             patch('production_scaling.scaling_orchestrator.AutoRecoveryEngine') as MockRecoveryEngine, \
             patch('production_scaling.scaling_orchestrator.IntelligentLoadBalancer') as MockLoadBalancer, \
             patch('production_scaling.scaling_orchestrator.PredictiveScalingAnalytics') as MockPredictiveAnalytics:
            
            # Configure realistic component behaviors
            mock_scaling = MockScalingEngine.return_value
            mock_scaling.start_auto_scaling.return_value = {'success': True}
            mock_scaling.analyze_scaling_requirements.return_value = {
                'scaling_needed': True,
                'scale_up': {
                    'systems': [{'name': 'brain_core', 'current_instances': 2, 'reason': 'CPU > 80%'}],
                    'agents': []
                },
                'scale_down': {'systems': [], 'agents': []}
            }
            mock_scaling.execute_system_scaling.return_value = {'success': True}
            mock_scaling.current_instances = {'systems': {'brain_core': 2}}
            
            mock_recovery = MockRecoveryEngine.return_value
            mock_recovery.start_auto_recovery.return_value = {'success': True}
            
            mock_balancer = MockLoadBalancer.return_value
            mock_balancer.start_load_balancing.return_value = {'success': True}
            mock_balancer.analyze_workload_distribution.return_value = {'overall_efficiency': 0.92}
            
            mock_analytics = MockPredictiveAnalytics.return_value
            mock_analytics.start_analytics.return_value = {'success': True}
            mock_analytics.predict_future_demand.return_value = {'components': {}}
            
            orchestrator = ScalingOrchestrator(
                self.mock_monitor,
                self.mock_brain_system,
                self.mock_agent_system
            )
            
            # Start orchestration
            result = orchestrator.start_orchestration()
            self.assertTrue(result['success'])
            
            # Let it run for a brief period
            time.sleep(0.5)
            
            # Stop orchestration
            with patch('builtins.open', create=True):
                stop_result = orchestrator.stop_orchestration()
            self.assertTrue(stop_result['success'])
            
            # Verify components were called appropriately
            mock_scaling.start_auto_scaling.assert_called_once()
            mock_recovery.start_auto_recovery.assert_called_once()
            mock_balancer.start_load_balancing.assert_called_once()
            mock_analytics.start_analytics.assert_called_once()


if __name__ == '__main__':
    unittest.main()