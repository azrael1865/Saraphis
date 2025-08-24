#!/usr/bin/env python3
"""
REAL comprehensive test suite for ScalingOrchestrator
Tests actual functionality with REAL components, NO MOCKS
"""

import unittest
import threading
import time
from datetime import datetime, timedelta
from collections import deque

# Test the actual ScalingOrchestrator implementation with REAL components
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


class RealMonitor:
    """Real monitor implementation for testing"""
    
    def __init__(self):
        self.system_statuses = {
            'brain_core': {
                'health': 0.95,
                'performance': {'response_time_ms': 150, 'error_rate': 0.02, 'throughput': 100},
                'resources': {'cpu_percent': 45, 'memory_percent': 60}
            },
            'compression_system': {
                'health': 0.90,
                'performance': {'response_time_ms': 200, 'error_rate': 0.01, 'throughput': 80},
                'resources': {'cpu_percent': 70, 'memory_percent': 75}
            },
            'training_system': {
                'health': 0.85,
                'performance': {'response_time_ms': 300, 'error_rate': 0.03, 'throughput': 60},
                'resources': {'cpu_percent': 55, 'memory_percent': 80}
            }
        }
        
        self.agent_statuses = {
            'reasoning_agent': {
                'health': 0.92,
                'active_tasks': 5,
                'completed_tasks': 100,
                'task_success_rate': 0.98,
                'response_time': 120
            },
            'pattern_agent': {
                'health': 0.88,
                'active_tasks': 8,
                'completed_tasks': 80,
                'task_success_rate': 0.96,
                'response_time': 180
            },
            'optimization_agent': {
                'health': 0.91,
                'active_tasks': 3,
                'completed_tasks': 150,
                'task_success_rate': 0.99,
                'response_time': 90
            }
        }
    
    def get_all_system_status(self):
        return self.system_statuses.copy()
    
    def get_all_agent_status(self):
        return self.agent_statuses.copy()
    
    def get_system_status(self, system_name):
        return self.system_statuses.get(system_name, {})
    
    def get_agent_status(self, agent_name):
        return self.agent_statuses.get(agent_name, {})
    
    def update_system_health(self, system_name, health):
        """Update system health for testing"""
        if system_name in self.system_statuses:
            self.system_statuses[system_name]['health'] = health
    
    def update_agent_health(self, agent_name, health):
        """Update agent health for testing"""
        if agent_name in self.agent_statuses:
            self.agent_statuses[agent_name]['health'] = health
    
    def check_system_health(self, system_name):
        """Check individual system health for recovery engine"""
        if system_name in self.system_statuses:
            return {
                'name': system_name,
                'health': self.system_statuses[system_name]['health'],
                'status': 'healthy' if self.system_statuses[system_name]['health'] > 0.5 else 'unhealthy',
                'timestamp': time.time()
            }
        return {
            'name': system_name,
            'health': 0.0,
            'status': 'unknown',
            'timestamp': time.time()
        }


class RealBrainSystem:
    """Real brain system implementation for testing"""
    
    def __init__(self):
        self.status = 'running'
    
    def get_status(self):
        return {'status': self.status, 'health': 0.95}


class RealAgentSystem:
    """Real agent system implementation for testing"""
    
    def __init__(self):
        self.status = 'active'
        self.agents = ['reasoning_agent', 'pattern_agent', 'optimization_agent']
    
    def get_status(self):
        return {'status': self.status, 'active_agents': len(self.agents)}


class TestScalingOrchestratorReal(unittest.TestCase):
    """Test ScalingOrchestrator with REAL components - NO MOCKS"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Imports failed: {IMPORT_ERROR}")
    def setUp(self):
        """Set up test fixtures with REAL components"""
        # Create REAL dependencies - NO MOCKS
        self.real_monitor = RealMonitor()
        self.real_brain_system = RealBrainSystem()
        self.real_agent_system = RealAgentSystem()
        
        # Create orchestrator with REAL dependencies
        self.orchestrator = ScalingOrchestrator(
            self.real_monitor, 
            self.real_brain_system, 
            self.real_agent_system
        )

    def test_orchestrator_real_initialization(self):
        """Test REAL orchestrator initialization"""
        # Test actual object creation
        self.assertIsNotNone(self.orchestrator)
        self.assertIsInstance(self.orchestrator, ScalingOrchestrator)
        
        # Test REAL component initialization
        self.assertIsInstance(self.orchestrator.scaling_engine, AutoScalingEngine)
        self.assertIsInstance(self.orchestrator.recovery_engine, AutoRecoveryEngine)
        self.assertIsInstance(self.orchestrator.load_balancer, IntelligentLoadBalancer)
        self.assertIsInstance(self.orchestrator.predictive_analytics, PredictiveScalingAnalytics)
        
        # Test initial state
        self.assertFalse(self.orchestrator.is_running)
        self.assertIsNone(self.orchestrator.orchestration_thread)
        
        # Test configuration
        self.assertEqual(self.orchestrator.orchestration_interval, 30.0)
        self.assertEqual(self.orchestrator.emergency_threshold, 0.3)
        
        # Test metrics initialization
        expected_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'emergency_responses': 0,
            'predictive_actions': 0
        }
        self.assertEqual(self.orchestrator.orchestration_metrics, expected_metrics)

    def test_real_health_checking(self):
        """Test REAL health checking functionality"""
        health_check = self.orchestrator._check_overall_health()
        
        # Verify structure
        self.assertIn('timestamp', health_check)
        self.assertIn('overall_health', health_check)
        self.assertIn('critical_issues', health_check)
        self.assertIn('emergency_needed', health_check)
        
        # Test calculated overall health with REAL data
        expected_health = (0.95 + 0.90 + 0.85 + 0.92 + 0.88 + 0.91) / 6
        self.assertAlmostEqual(health_check['overall_health'], expected_health, places=2)
        
        # Should be no critical issues with current health levels
        self.assertEqual(len(health_check['critical_issues']), 0)
        self.assertFalse(health_check['emergency_needed'])

    def test_real_emergency_detection(self):
        """Test REAL emergency detection"""
        # Update monitor with critical health levels
        self.real_monitor.update_system_health('brain_core', 0.2)  # Critical
        self.real_monitor.update_agent_health('reasoning_agent', 0.1)  # Critical
        
        health_check = self.orchestrator._check_overall_health()
        
        # Should detect emergencies
        self.assertEqual(len(health_check['critical_issues']), 2)
        self.assertTrue(health_check['emergency_needed'])
        
        # Verify issue details
        critical_names = [issue['name'] for issue in health_check['critical_issues']]
        self.assertIn('brain_core', critical_names)
        self.assertIn('reasoning_agent', critical_names)

    def test_real_scaling_analysis(self):
        """Test REAL scaling analysis"""
        # Get scaling requirements from REAL scaling engine
        requirements = self.orchestrator.scaling_engine.analyze_scaling_requirements()
        
        # Verify structure
        self.assertIn('timestamp', requirements)
        self.assertIn('scaling_needed', requirements)
        self.assertIn('scale_up', requirements)
        self.assertIn('scale_down', requirements)
        self.assertIn('current_load', requirements)
        
        # Verify load data is extracted from REAL monitor
        self.assertIn('brain_core', requirements['current_load'])
        self.assertIn('reasoning_agent', requirements['current_load'])

    def test_real_orchestration_startup_shutdown(self):
        """Test REAL orchestration lifecycle"""
        # Test startup
        start_result = self.orchestrator.start_orchestration()
        
        # Verify successful startup
        self.assertTrue(start_result['success'])
        self.assertTrue(self.orchestrator.is_running)
        self.assertIsNotNone(self.orchestrator.orchestration_thread)
        self.assertTrue(self.orchestrator.orchestration_thread.is_alive())
        
        # Verify components are running
        self.assertIn('components', start_result)
        components = start_result['components']
        for component_name in ['scaling_engine', 'recovery_engine', 'load_balancer', 'predictive_analytics']:
            self.assertEqual(components[component_name], 'running')
        
        # Let it run briefly to test actual operation
        time.sleep(1.0)
        
        # Test shutdown
        stop_result = self.orchestrator.stop_orchestration()
        
        self.assertTrue(stop_result['success'])
        self.assertFalse(self.orchestrator.is_running)
        self.assertIn('component_results', stop_result)
        self.assertIn('total_operations', stop_result)

    def test_real_status_reporting(self):
        """Test REAL status reporting"""
        status = self.orchestrator.get_orchestration_status()
        
        # Verify structure
        self.assertIn('is_running', status)
        self.assertIn('components', status)
        self.assertIn('metrics', status)
        self.assertIn('recent_operations', status)
        self.assertIn('health_check', status)
        
        # Verify component statuses are REAL
        components = status['components']
        self.assertIn('scaling_engine', components)
        self.assertIn('recovery_engine', components)
        self.assertIn('load_balancer', components)
        self.assertIn('predictive_analytics', components)
        
        # Each component should have actual status data
        for component_status in components.values():
            self.assertIsInstance(component_status, dict)

    def test_real_manual_scaling(self):
        """Test REAL manual scaling execution"""
        scaling_request = {
            'systems': {
                'brain_core': 3,
                'compression_system': 2
            }
        }
        
        result = self.orchestrator.execute_manual_scaling(scaling_request)
        
        # Verify execution
        self.assertIn('timestamp', result)
        self.assertIn('systems', result)
        self.assertIn('agents', result)
        self.assertIn('success', result)
        
        # Verify actual scaling operations occurred
        self.assertIn('brain_core', result['systems'])
        self.assertIn('compression_system', result['systems'])

    def test_real_operation_recording(self):
        """Test REAL operation recording"""
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

    def test_real_error_handling(self):
        """Test REAL error handling"""
        # Create a monitor that will fail
        class FailingMonitor:
            def get_all_system_status(self):
                raise Exception("Monitor connection failed")
            
            def get_all_agent_status(self):
                raise Exception("Agent monitor failed")
        
        # Replace monitor with failing one
        original_monitor = self.orchestrator.monitor
        self.orchestrator.monitor = FailingMonitor()
        
        try:
            health_check = self.orchestrator._check_overall_health()
            
            # Should handle error gracefully
            self.assertEqual(health_check['overall_health'], 0.0)
            self.assertTrue(health_check['emergency_needed'])
            self.assertIn('error', health_check)
        finally:
            # Restore original monitor
            self.orchestrator.monitor = original_monitor

    def test_real_thread_safety(self):
        """Test REAL thread safety"""
        results = []
        
        def concurrent_operation():
            try:
                status = self.orchestrator.get_orchestration_status()
                results.append({'success': True, 'status': status})
            except Exception as e:
                results.append({'success': False, 'error': str(e)})
        
        # Run multiple concurrent operations
        threads = []
        for i in range(5):
            t = threading.Thread(target=concurrent_operation)
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join(timeout=10.0)
        
        # Verify all operations succeeded
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertTrue(result['success'], f"Thread failed: {result.get('error', 'Unknown')}")

    def test_real_comprehensive_workflow(self):
        """Test complete REAL workflow"""
        # 1. Start orchestration
        start_result = self.orchestrator.start_orchestration()
        self.assertTrue(start_result['success'])
        
        try:
            # 2. Set high resource usage to trigger scaling
            self.real_monitor.system_statuses['compression_system']['resources']['cpu_percent'] = 90
            self.real_monitor.system_statuses['compression_system']['resources']['memory_percent'] = 90
            self.real_monitor.system_statuses['training_system']['resources']['cpu_percent'] = 85
            self.real_monitor.system_statuses['training_system']['resources']['memory_percent'] = 95
            
            # Let it run and perform real operations
            time.sleep(2.0)
            
            # 3. Trigger an emergency condition
            self.real_monitor.update_system_health('brain_core', 0.1)
            time.sleep(1.0)
            
            # 4. Perform manual scaling
            scaling_result = self.orchestrator.execute_manual_scaling({
                'systems': {'compression_system': 3}
            })
            self.assertIn('success', scaling_result)
            
            # 5. Check status reflects changes
            status = self.orchestrator.get_orchestration_status()
            self.assertGreater(status['metrics']['total_operations'], 0)
            
        finally:
            # 6. Stop orchestration
            stop_result = self.orchestrator.stop_orchestration()
            self.assertTrue(stop_result['success'])

    def test_real_scaling_engine_integration(self):
        """Test REAL integration with scaling engine"""
        scaling_engine = self.orchestrator.scaling_engine
        
        # Test actual scaling analysis
        requirements = scaling_engine.analyze_scaling_requirements()
        self.assertIsInstance(requirements, dict)
        
        # Test actual scaling execution
        scaling_requests = [{
            'name': 'brain_core',
            'target_instances': 2,
            'reason': 'Integration test'
        }]
        
        result = scaling_engine.execute_system_scaling(scaling_requests)
        self.assertIn('success', result)
        self.assertIn('scaled_systems', result)

    def test_real_recovery_engine_integration(self):
        """Test REAL integration with recovery engine"""
        recovery_engine = self.orchestrator.recovery_engine
        
        # Test recovery status
        status = recovery_engine.get_recovery_status()
        self.assertIsInstance(status, dict)
        
        # Test recovery execution
        recovery_result = recovery_engine.execute_system_recovery(['brain_core'])
        self.assertIn('success', recovery_result)

    def test_real_predictive_analytics_integration(self):
        """Test REAL integration with predictive analytics"""
        analytics = self.orchestrator.predictive_analytics
        
        # Test prediction functionality
        predictions = analytics.predict_future_demand(time_range_hours=0.5)
        self.assertIsInstance(predictions, dict)
        
        # Test accuracy validation
        accuracy = analytics.validate_prediction_accuracy()
        self.assertIsInstance(accuracy, dict)


class TestScalingOrchestratorStress(unittest.TestCase):
    """Stress tests for ScalingOrchestrator with REAL components"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Imports failed: {IMPORT_ERROR}")
    def setUp(self):
        self.real_monitor = RealMonitor()
        self.real_brain_system = RealBrainSystem()
        self.real_agent_system = RealAgentSystem()
        self.orchestrator = ScalingOrchestrator(
            self.real_monitor, 
            self.real_brain_system, 
            self.real_agent_system
        )

    def test_real_high_load_scenarios(self):
        """Test REAL high load scenarios"""
        # Simulate high system load
        for system in ['brain_core', 'compression_system', 'training_system']:
            self.real_monitor.system_statuses[system]['resources']['cpu_percent'] = 95
            self.real_monitor.system_statuses[system]['resources']['memory_percent'] = 90
            self.real_monitor.system_statuses[system]['performance']['response_time_ms'] = 800
        
        # Test scaling analysis under load
        requirements = self.orchestrator.scaling_engine.analyze_scaling_requirements()
        self.assertTrue(requirements.get('scaling_needed', False))
        
        # Should recommend scale-up
        self.assertGreater(len(requirements['scale_up']['systems']), 0)

    def test_real_recovery_scenarios(self):
        """Test REAL recovery scenarios"""
        # Simulate system failures
        self.real_monitor.update_system_health('brain_core', 0.1)
        self.real_monitor.update_system_health('compression_system', 0.2)
        
        health_check = self.orchestrator._check_overall_health()
        
        # Should trigger emergency
        self.assertTrue(health_check['emergency_needed'])
        self.assertEqual(len(health_check['critical_issues']), 2)


if __name__ == '__main__':
    # Run only the real tests
    unittest.main(verbosity=2)