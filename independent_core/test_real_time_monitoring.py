"""
Test script for Saraphis Real-Time Production Monitoring & Optimization System
Demonstrates monitoring with <100ms latency and <30s alert response time
NO FALLBACKS - HARD FAILURES ONLY
"""

import sys
import json
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from brain import Brain
    from production_monitoring import (
        RealTimeProductionMonitor,
        ProductionOptimizationEngine,
        ProductionAlertSystem,
        ProductionAnalyticsDashboard,
        AutomatedResponseSystem,
        create_production_monitor
    )
except ImportError as e:
    logging.error(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimulatedProductionEnvironment:
    """Simulates production environment for testing"""
    
    def __init__(self):
        self.systems = {}
        self.agents = {}
        self.is_running = False
        self._lock = threading.Lock()
        
    def initialize_systems(self):
        """Initialize simulated systems"""
        system_names = [
            'brain_orchestration', 'uncertainty_system', 'proof_system',
            'gac_system', 'compression_systems', 'domain_management',
            'training_management', 'production_monitoring', 'production_security',
            'financial_fraud_domain', 'error_recovery'
        ]
        
        for name in system_names:
            self.systems[name] = {
                'health': 1.0,
                'performance': 1.0,
                'cpu_usage': random.uniform(20, 40),
                'memory_usage': random.uniform(30, 50),
                'response_time': random.uniform(10, 50),
                'error_rate': 0.0,
                'throughput': random.uniform(800, 1200)
            }
    
    def initialize_agents(self):
        """Initialize simulated agents"""
        agent_names = [
            'brain_orchestration_agent', 'proof_system_agent',
            'uncertainty_agent', 'training_agent', 'domain_agent',
            'compression_agent', 'production_agent', 'web_interface_agent'
        ]
        
        for name in agent_names:
            self.agents[name] = {
                'health': 1.0,
                'task_success_rate': 1.0,
                'response_time': random.uniform(50, 150),
                'active_tasks': random.randint(0, 5),
                'completed_tasks': 0,
                'coordination_score': 1.0
            }
    
    def start_simulation(self):
        """Start production simulation"""
        self.is_running = True
        self.simulation_thread = threading.Thread(
            target=self._simulation_loop,
            daemon=True
        )
        self.simulation_thread.start()
    
    def _simulation_loop(self):
        """Simulate production behavior"""
        iteration = 0
        while self.is_running:
            with self._lock:
                iteration += 1
                
                # Simulate normal variations
                for system in self.systems.values():
                    system['cpu_usage'] = max(0, min(100, 
                        system['cpu_usage'] + random.uniform(-5, 5)))
                    system['memory_usage'] = max(0, min(100,
                        system['memory_usage'] + random.uniform(-3, 3)))
                    system['response_time'] = max(1,
                        system['response_time'] + random.uniform(-10, 10))
                    system['throughput'] = max(0,
                        system['throughput'] + random.uniform(-50, 50))
                
                # Simulate issues at specific iterations
                if iteration == 20:
                    # High CPU usage
                    self.systems['brain_orchestration']['cpu_usage'] = 95
                    logger.info("SIMULATION: Injected high CPU usage in brain_orchestration")
                
                elif iteration == 40:
                    # Performance degradation
                    self.systems['proof_system']['performance'] = 0.6
                    self.systems['proof_system']['response_time'] = 500
                    logger.info("SIMULATION: Injected performance degradation in proof_system")
                
                elif iteration == 60:
                    # High error rate
                    self.systems['gac_system']['error_rate'] = 0.15
                    self.systems['gac_system']['health'] = 0.7
                    logger.info("SIMULATION: Injected high error rate in gac_system")
                
                elif iteration == 80:
                    # Agent failure
                    self.agents['uncertainty_agent']['health'] = 0.3
                    self.agents['uncertainty_agent']['task_success_rate'] = 0.5
                    logger.info("SIMULATION: Injected agent failure in uncertainty_agent")
                
                elif iteration == 100:
                    # System failure
                    self.systems['domain_management']['health'] = 0.2
                    self.systems['domain_management']['error_rate'] = 0.5
                    logger.info("SIMULATION: Injected system failure in domain_management")
                
                elif iteration == 120:
                    # Resource exhaustion
                    self.systems['compression_systems']['memory_usage'] = 95
                    self.systems['compression_systems']['cpu_usage'] = 92
                    logger.info("SIMULATION: Injected resource exhaustion in compression_systems")
                
                # Update agent tasks
                for agent in self.agents.values():
                    if agent['active_tasks'] > 0 and random.random() < 0.3:
                        agent['completed_tasks'] += 1
                        agent['active_tasks'] -= 1
                    if random.random() < 0.2:
                        agent['active_tasks'] = min(10, agent['active_tasks'] + 1)
            
            time.sleep(0.5)  # Update every 500ms
    
    def get_system_status(self, system_name: str) -> Dict[str, Any]:
        """Get simulated system status"""
        with self._lock:
            if system_name not in self.systems:
                return {'error': 'System not found'}
            
            system = self.systems[system_name]
            return {
                'name': system_name,
                'health': system['health'],
                'performance': {
                    'score': system['performance'],
                    'response_time_ms': system['response_time'],
                    'throughput': system['throughput'],
                    'error_rate': system['error_rate']
                },
                'resources': {
                    'cpu_percent': system['cpu_usage'],
                    'memory_percent': system['memory_usage']
                }
            }
    
    def get_agent_status(self, agent_name: str) -> Dict[str, Any]:
        """Get simulated agent status"""
        with self._lock:
            if agent_name not in self.agents:
                return {'error': 'Agent not found'}
            
            agent = self.agents[agent_name]
            return {
                'name': agent_name,
                'health': agent['health'],
                'task_success_rate': agent['task_success_rate'],
                'response_time': agent['response_time'],
                'active_tasks': agent['active_tasks'],
                'completed_tasks': agent['completed_tasks'],
                'coordination_score': agent['coordination_score']
            }
    
    def stop_simulation(self):
        """Stop production simulation"""
        self.is_running = False


def test_monitoring_latency(monitor: RealTimeProductionMonitor) -> bool:
    """Test monitoring latency is <100ms"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING MONITORING LATENCY")
    logger.info("=" * 80)
    
    try:
        latencies = []
        
        # Measure latency 10 times
        for i in range(10):
            start_time = time.time()
            
            # Get all system statuses
            statuses = monitor.get_all_system_status()
            
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            
            logger.info(f"  Iteration {i+1}: {latency:.2f}ms")
            time.sleep(0.1)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        logger.info(f"\nAverage latency: {avg_latency:.2f}ms")
        logger.info(f"Max latency: {max_latency:.2f}ms")
        logger.info(f"Requirement: <100ms")
        
        success = max_latency < 100
        if success:
            logger.info("✅ Monitoring latency test PASSED")
        else:
            logger.error("❌ Monitoring latency test FAILED")
        
        return success
        
    except Exception as e:
        logger.error(f"Monitoring latency test failed: {e}")
        return False


def test_alert_response_time(alert_system: ProductionAlertSystem, 
                           response_system: AutomatedResponseSystem) -> bool:
    """Test alert response time is <30 seconds"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING ALERT RESPONSE TIME")
    logger.info("=" * 80)
    
    try:
        # Create test alert
        alert = alert_system.create_alert(
            alert_type='PERFORMANCE',
            severity='HIGH',
            source='test/system',
            message='Test alert for response time measurement',
            data={'performance_score': 0.5}
        )
        
        alert_id = alert['alert_id']
        logger.info(f"Created test alert: {alert_id}")
        
        # Wait for response
        start_time = time.time()
        max_wait = 35  # 35 seconds max wait
        
        while time.time() - start_time < max_wait:
            # Check if alert has been handled
            alert_status = alert_system.get_alert_status(alert_id)
            if alert_status.get('handled'):
                response_time = alert_status.get('response_time', 0)
                logger.info(f"Alert handled in {response_time:.2f} seconds")
                
                success = response_time < 30
                if success:
                    logger.info("✅ Alert response time test PASSED")
                else:
                    logger.error("❌ Alert response time test FAILED")
                
                return success
            
            time.sleep(0.5)
        
        logger.error("❌ Alert was not handled within 35 seconds")
        return False
        
    except Exception as e:
        logger.error(f"Alert response time test failed: {e}")
        return False


def test_optimization_application(optimization_engine: ProductionOptimizationEngine) -> bool:
    """Test optimization application"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING OPTIMIZATION APPLICATION")
    logger.info("=" * 80)
    
    try:
        # Apply performance optimization
        result = optimization_engine.apply_optimization(
            optimization_type='performance',
            target='test/system',
            parameters={
                'current_performance': 0.6,
                'target_performance': 0.9
            }
        )
        
        logger.info(f"Optimization result: {result}")
        
        if result.get('success'):
            logger.info(f"✅ Optimization applied successfully")
            logger.info(f"  Impact: {result.get('impact', 0):.2f}")
            logger.info(f"  Time: {result.get('optimization_time', 0):.2f}s")
            return True
        else:
            logger.error(f"❌ Optimization failed: {result.get('error')}")
            return False
            
    except Exception as e:
        logger.error(f"Optimization application test failed: {e}")
        return False


def test_analytics_dashboard(dashboard: ProductionAnalyticsDashboard) -> bool:
    """Test analytics dashboard"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING ANALYTICS DASHBOARD")
    logger.info("=" * 80)
    
    try:
        # Let dashboard collect some data
        time.sleep(5)
        
        # Get dashboard data
        dashboard_data = dashboard.get_dashboard_data()
        
        logger.info("Dashboard data retrieved:")
        logger.info(f"  Timestamp: {dashboard_data.get('timestamp')}")
        logger.info(f"  Uptime: {dashboard_data.get('uptime')}")
        
        # Get performance summary
        summary = dashboard.get_performance_summary(time_range_minutes=1)
        
        logger.info("\nPerformance Summary:")
        logger.info(f"  Average Health: {summary['overall']['average_health']:.2%}")
        logger.info(f"  Average Performance: {summary['overall']['average_performance']:.2%}")
        logger.info(f"  Peak Resource Usage: {summary['overall']['peak_resource_usage']:.1f}%")
        logger.info(f"  Total Alerts: {summary['overall']['total_alerts']}")
        logger.info(f"  Total Optimizations: {summary['overall']['total_optimizations']}")
        
        # Export report
        report_result = dashboard.export_analytics_report('test_analytics_report.json')
        
        if report_result.get('success'):
            logger.info(f"✅ Analytics report exported to {report_result['filepath']}")
            return True
        else:
            logger.error("❌ Failed to export analytics report")
            return False
            
    except Exception as e:
        logger.error(f"Analytics dashboard test failed: {e}")
        return False


def test_full_monitoring_cycle(monitor, optimization_engine, alert_system, 
                             response_system, dashboard) -> bool:
    """Test complete monitoring cycle"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING FULL MONITORING CYCLE")
    logger.info("=" * 80)
    
    try:
        logger.info("Running 2-minute monitoring cycle...")
        
        start_time = time.time()
        duration = 120  # 2 minutes
        
        issue_counts = {
            'alerts_generated': 0,
            'optimizations_applied': 0,
            'responses_executed': 0
        }
        
        while time.time() - start_time < duration:
            # Check for issues
            issues = monitor.detect_performance_issues()
            if issues:
                issue_counts['alerts_generated'] += len(issues)
                logger.info(f"Detected {len(issues)} issues")
            
            # Check optimization history
            opt_history = optimization_engine.get_optimization_history(limit=1)
            if opt_history:
                issue_counts['optimizations_applied'] = len(
                    optimization_engine.get_optimization_history(limit=100)
                )
            
            # Check response metrics
            response_metrics = response_system.get_response_metrics()
            issue_counts['responses_executed'] = response_metrics['total_responses']
            
            # Display current status every 20 seconds
            if int(time.time() - start_time) % 20 == 0:
                metrics = monitor.get_overall_metrics()
                logger.info(f"\nCurrent Status (t={int(time.time() - start_time)}s):")
                logger.info(f"  Overall Health: {metrics['health_score']:.2%}")
                logger.info(f"  Performance: {metrics['performance_score']:.2%}")
                logger.info(f"  Alerts: {issue_counts['alerts_generated']}")
                logger.info(f"  Optimizations: {issue_counts['optimizations_applied']}")
                logger.info(f"  Responses: {issue_counts['responses_executed']}")
            
            time.sleep(1)
        
        # Final summary
        logger.info("\n" + "-" * 40)
        logger.info("MONITORING CYCLE SUMMARY")
        logger.info("-" * 40)
        logger.info(f"Duration: {duration} seconds")
        logger.info(f"Alerts Generated: {issue_counts['alerts_generated']}")
        logger.info(f"Optimizations Applied: {issue_counts['optimizations_applied']}")
        logger.info(f"Responses Executed: {issue_counts['responses_executed']}")
        
        # Check response metrics
        response_metrics = response_system.get_response_metrics()
        logger.info(f"\nResponse System Metrics:")
        logger.info(f"  Success Rate: {response_metrics['success_rate']:.2%}")
        logger.info(f"  Average Response Time: {response_metrics['average_response_time']:.2f}s")
        logger.info(f"  Max Response Time: {response_metrics['max_response_time']:.2f}s")
        logger.info(f"  Meets SLA (<30s): {response_metrics['meets_sla']}")
        
        success = response_metrics['meets_sla'] and response_metrics['success_rate'] > 0.8
        
        if success:
            logger.info("\n✅ Full monitoring cycle test PASSED")
        else:
            logger.error("\n❌ Full monitoring cycle test FAILED")
        
        return success
        
    except Exception as e:
        logger.error(f"Full monitoring cycle test failed: {e}")
        return False


def run_monitoring_tests():
    """Run all monitoring system tests"""
    logger.info("=" * 80)
    logger.info("SARAPHIS REAL-TIME MONITORING SYSTEM TEST")
    logger.info("=" * 80)
    
    # Initialize test environment
    logger.info("\n1. Initializing test environment...")
    
    # Create simulated environment
    sim_env = SimulatedProductionEnvironment()
    sim_env.initialize_systems()
    sim_env.initialize_agents()
    
    # Create monitoring configuration
    config = {
        'monitoring_interval': 0.1,  # 100ms
        'alert_check_interval': 0.5,  # 500ms
        'optimization_interval': 10,  # 10 seconds
        'performance_threshold': 0.7,
        'resource_threshold': 80,
        'error_rate_threshold': 0.05
    }
    
    try:
        # Initialize Brain system
        brain_system = Brain()
        brain_system.initialize_brain()
        logger.info("Brain system initialized")
        
        # Create monitoring components
        logger.info("\n2. Creating monitoring components...")
        
        # Use factory function to create monitor with all components
        monitor_result = create_production_monitor(
            brain_system=brain_system,
            agent_system=sim_env,  # Use simulated environment
            config=config
        )
        
        monitor = monitor_result['monitor']
        optimization_engine = monitor_result['optimization_engine']
        alert_system = monitor_result['alert_system']
        dashboard = monitor_result['dashboard']
        response_system = monitor_result['response_system']
        
        logger.info("All monitoring components created")
        
        # Override monitor methods to use simulated environment
        monitor.get_system_status = sim_env.get_system_status
        monitor.get_agent_status = sim_env.get_agent_status
        
        # Start monitoring
        logger.info("\n3. Starting monitoring systems...")
        
        monitor.start_monitoring()
        optimization_engine.start_optimization_engine()
        alert_system.start_alert_system()
        response_system.start_automated_responses()
        dashboard.start_dashboard()
        
        logger.info("All monitoring systems started")
        
        # Start simulation
        logger.info("\n4. Starting production simulation...")
        sim_env.start_simulation()
        
        # Run tests
        logger.info("\n5. Running tests...")
        
        test_results = {
            'monitoring_latency': test_monitoring_latency(monitor),
            'alert_response_time': test_alert_response_time(alert_system, response_system),
            'optimization_application': test_optimization_application(optimization_engine),
            'analytics_dashboard': test_analytics_dashboard(dashboard),
            'full_monitoring_cycle': test_full_monitoring_cycle(
                monitor, optimization_engine, alert_system, response_system, dashboard
            )
        }
        
        # Display test summary
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        
        total_tests = len(test_results)
        passed_tests = sum(1 for passed in test_results.values() if passed)
        
        for test_name, passed in test_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\nTotal Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {(passed_tests / total_tests * 100):.1f}%")
        
        # Stop all systems
        logger.info("\n6. Stopping monitoring systems...")
        
        sim_env.stop_simulation()
        monitor.stop_monitoring()
        optimization_engine.stop_optimization_engine()
        alert_system.stop_alert_system()
        response_system.stop_automated_responses()
        dashboard.stop_dashboard()
        
        # Cleanup
        brain_system.shutdown()
        
        return passed_tests == total_tests
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = run_monitoring_tests()
    sys.exit(0 if success else 1)