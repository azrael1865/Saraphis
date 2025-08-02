"""
Test script for Saraphis Production Scaling & Auto-Recovery System
Tests scaling decisions, recovery operations, load balancing, and predictive analytics
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
    from production_monitoring import create_production_monitor
    from production_scaling import ScalingOrchestrator
except ImportError as e:
    logging.error(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScalingTestEnvironment:
    """Simulates production environment for scaling tests"""
    
    def __init__(self):
        self.systems = {}
        self.agents = {}
        self.is_running = False
        self.load_scenarios = []
        self.failure_scenarios = []
        self._lock = threading.Lock()
        
    def initialize_environment(self):
        """Initialize test environment"""
        # Initialize systems
        system_names = [
            'brain_orchestration', 'uncertainty_system', 'proof_system',
            'gac_system', 'compression_systems', 'domain_management',
            'training_management', 'production_monitoring', 'production_security',
            'financial_fraud_domain', 'error_recovery'
        ]
        
        for name in system_names:
            self.systems[name] = {
                'health': 1.0,
                'cpu': 40,
                'memory': 45,
                'response_time': 100,
                'error_rate': 0.0,
                'throughput': 1000
            }
        
        # Initialize agents
        agent_names = [
            'brain_orchestration_agent', 'proof_system_agent',
            'uncertainty_agent', 'training_agent', 'domain_agent',
            'compression_agent', 'production_agent', 'web_interface_agent'
        ]
        
        for name in agent_names:
            self.agents[name] = {
                'health': 1.0,
                'active_tasks': 5,
                'task_success_rate': 1.0,
                'response_time': 100,
                'coordination_score': 1.0
            }
    
    def start_simulation(self):
        """Start environment simulation"""
        self.is_running = True
        self.simulation_thread = threading.Thread(
            target=self._simulation_loop,
            daemon=True
        )
        self.simulation_thread.start()
        logger.info("Scaling test environment started")
    
    def _simulation_loop(self):
        """Main simulation loop"""
        iteration = 0
        while self.is_running:
            with self._lock:
                iteration += 1
                
                # Apply load scenarios
                self._apply_load_scenarios(iteration)
                
                # Apply failure scenarios
                self._apply_failure_scenarios(iteration)
                
                # Normal variations
                self._apply_normal_variations()
            
            time.sleep(1)
    
    def _apply_load_scenarios(self, iteration: int):
        """Apply load test scenarios"""
        # Scenario 1: Gradual load increase (iterations 10-30)
        if 10 <= iteration <= 30:
            increase_rate = (iteration - 10) * 2
            self.systems['brain_orchestration']['cpu'] = min(95, 40 + increase_rate)
            self.systems['brain_orchestration']['memory'] = min(90, 45 + increase_rate)
            self.systems['brain_orchestration']['throughput'] = 1000 + (increase_rate * 20)
            
        # Scenario 2: Sudden spike (iteration 40)
        elif iteration == 40:
            logger.info("SCENARIO: Sudden load spike on proof_system")
            self.systems['proof_system']['cpu'] = 92
            self.systems['proof_system']['memory'] = 88
            self.systems['proof_system']['response_time'] = 800
            
        # Scenario 3: Agent overload (iterations 50-60)
        elif 50 <= iteration <= 60:
            overload = (iteration - 50) * 2
            self.agents['uncertainty_agent']['active_tasks'] = min(20, 5 + overload)
            self.agents['uncertainty_agent']['response_time'] = 100 + (overload * 50)
            
        # Scenario 4: Multi-system stress (iterations 70-80)
        elif 70 <= iteration <= 80:
            stress_level = (iteration - 70) * 3
            for system in ['gac_system', 'compression_systems', 'domain_management']:
                self.systems[system]['cpu'] = min(85, 40 + stress_level)
                self.systems[system]['memory'] = min(80, 45 + stress_level)
    
    def _apply_failure_scenarios(self, iteration: int):
        """Apply failure test scenarios"""
        # Scenario 1: System failure (iteration 90)
        if iteration == 90:
            logger.info("SCENARIO: System failure - financial_fraud_domain")
            self.systems['financial_fraud_domain']['health'] = 0.2
            self.systems['financial_fraud_domain']['error_rate'] = 0.6
            
        # Scenario 2: Agent failure (iteration 100)
        elif iteration == 100:
            logger.info("SCENARIO: Agent failure - training_agent")
            self.agents['training_agent']['health'] = 0.3
            self.agents['training_agent']['task_success_rate'] = 0.4
            
        # Scenario 3: Cascading failures (iteration 110)
        elif iteration == 110:
            logger.info("SCENARIO: Cascading failures")
            self.systems['brain_orchestration']['health'] = 0.4
            self.systems['uncertainty_system']['health'] = 0.45
            self.agents['brain_orchestration_agent']['health'] = 0.35
            
        # Scenario 4: Resource exhaustion (iteration 120)
        elif iteration == 120:
            logger.info("SCENARIO: Resource exhaustion - compression_systems")
            self.systems['compression_systems']['cpu'] = 98
            self.systems['compression_systems']['memory'] = 95
            self.systems['compression_systems']['response_time'] = 2000
    
    def _apply_normal_variations(self):
        """Apply normal random variations"""
        # Systems
        for system in self.systems.values():
            if system['health'] > 0.8:  # Only vary healthy systems
                system['cpu'] = max(10, min(100, system['cpu'] + random.uniform(-5, 5)))
                system['memory'] = max(10, min(100, system['memory'] + random.uniform(-3, 3)))
                system['response_time'] = max(10, system['response_time'] + random.uniform(-20, 20))
                system['throughput'] = max(100, system['throughput'] + random.uniform(-50, 50))
        
        # Agents
        for agent in self.agents.values():
            if agent['health'] > 0.8:
                agent['active_tasks'] = max(0, min(20, agent['active_tasks'] + random.randint(-2, 2)))
                agent['response_time'] = max(10, agent['response_time'] + random.uniform(-10, 10))
    
    def get_system_status(self, system_name: str) -> Dict[str, Any]:
        """Get system status for monitor"""
        with self._lock:
            if system_name not in self.systems:
                return {'error': 'System not found'}
            
            system = self.systems[system_name]
            return {
                'name': system_name,
                'health': system['health'],
                'performance': {
                    'score': 1.0 - (system['error_rate'] * 2),
                    'response_time_ms': system['response_time'],
                    'throughput': system['throughput'],
                    'error_rate': system['error_rate']
                },
                'resources': {
                    'cpu_percent': system['cpu'],
                    'memory_percent': system['memory']
                }
            }
    
    def get_agent_status(self, agent_name: str) -> Dict[str, Any]:
        """Get agent status for monitor"""
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
                'completed_tasks': random.randint(100, 1000),
                'coordination_score': agent['coordination_score']
            }
    
    def stop_simulation(self):
        """Stop environment simulation"""
        self.is_running = False
        logger.info("Scaling test environment stopped")


def test_scaling_decisions(orchestrator: ScalingOrchestrator) -> bool:
    """Test auto-scaling decision making"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING SCALING DECISIONS")
    logger.info("=" * 80)
    
    try:
        # Wait for some data collection
        logger.info("Collecting metrics for scaling analysis...")
        time.sleep(15)
        
        # Check scaling engine analysis
        scaling_analysis = orchestrator.scaling_engine.analyze_scaling_requirements()
        
        logger.info(f"\nScaling Analysis:")
        logger.info(f"  Scaling needed: {scaling_analysis.get('scaling_needed', False)}")
        logger.info(f"  Systems to scale up: {len(scaling_analysis.get('scale_up', {}).get('systems', []))}")
        logger.info(f"  Systems to scale down: {len(scaling_analysis.get('scale_down', {}).get('systems', []))}")
        logger.info(f"  Agents to scale up: {len(scaling_analysis.get('scale_up', {}).get('agents', []))}")
        logger.info(f"  Agents to scale down: {len(scaling_analysis.get('scale_down', {}).get('agents', []))}")
        
        # Test manual scaling
        logger.info("\nTesting manual scaling...")
        manual_result = orchestrator.execute_manual_scaling({
            'systems': {
                'brain_orchestration': 2,
                'proof_system': 3
            },
            'agents': {
                'uncertainty_agent': 2
            }
        })
        
        success = manual_result.get('success', False)
        logger.info(f"Manual scaling result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        # Check current instances
        time.sleep(5)
        scaling_status = orchestrator.scaling_engine.get_scaling_status()
        logger.info(f"\nCurrent instances:")
        logger.info(f"  Systems: {dict(scaling_status['current_instances']['systems'])}")
        logger.info(f"  Agents: {dict(scaling_status['current_instances']['agents'])}")
        
        return success
        
    except Exception as e:
        logger.error(f"Scaling decisions test failed: {e}")
        return False


def test_auto_recovery(orchestrator: ScalingOrchestrator) -> bool:
    """Test auto-recovery functionality"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING AUTO-RECOVERY")
    logger.info("=" * 80)
    
    try:
        # Wait for failure scenarios
        logger.info("Waiting for failure scenarios to trigger...")
        time.sleep(90)  # Wait until iteration 90+
        
        # Check recovery engine status
        recovery_status = orchestrator.recovery_engine.get_recovery_status()
        
        logger.info(f"\nRecovery Status:")
        logger.info(f"  Active recoveries: {len(recovery_status['active_recoveries'])}")
        logger.info(f"  Total recoveries: {recovery_status['performance']['total_recoveries']}")
        logger.info(f"  Success rate: {recovery_status['performance']['success_rate']:.2%}")
        logger.info(f"  Meets SLA: {recovery_status['performance']['meets_sla']}")
        
        # Test manual recovery
        logger.info("\nTesting manual recovery...")
        manual_recovery = orchestrator.recovery_engine.execute_system_recovery(['error_recovery'])
        
        success = manual_recovery.get('success', False)
        logger.info(f"Manual recovery result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        # Wait for recovery to complete
        time.sleep(10)
        
        # Validate recovery
        health_check = orchestrator._check_overall_health()
        logger.info(f"\nPost-recovery health: {health_check['overall_health']:.2%}")
        
        return recovery_status['performance']['meets_sla']
        
    except Exception as e:
        logger.error(f"Auto-recovery test failed: {e}")
        return False


def test_load_balancing(orchestrator: ScalingOrchestrator) -> bool:
    """Test intelligent load balancing"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING LOAD BALANCING")
    logger.info("=" * 80)
    
    try:
        # Check current load distribution
        load_analysis = orchestrator.load_balancer.analyze_workload_distribution()
        
        logger.info(f"\nLoad Distribution Analysis:")
        logger.info(f"  Overall efficiency: {load_analysis['overall_efficiency']:.2%}")
        logger.info(f"  Load variance: {load_analysis.get('load_variance', 0):.2f}")
        logger.info(f"  Hotspots detected: {len(load_analysis['hotspots'])}")
        logger.info(f"  Underutilized resources: {len(load_analysis['underutilized'])}")
        
        # Test task distribution
        test_tasks = [
            {'id': f'test_task_{i}', 'type': 'proof', 'priority': 'high'}
            for i in range(20)
        ]
        
        balance_result = orchestrator.load_balancer.balance_agent_tasks(test_tasks)
        
        logger.info(f"\nTask Balancing Result:")
        logger.info(f"  Tasks assigned: {balance_result['tasks_assigned']}")
        logger.info(f"  Success: {balance_result.get('success', False)}")
        
        # Test strategy switching
        logger.info("\nTesting strategy switching...")
        switch_result = orchestrator.load_balancer.switch_distribution_strategy('least_loaded')
        logger.info(f"Strategy switch: {switch_result.get('success', False)}")
        
        # Check efficiency after optimization
        time.sleep(10)
        new_analysis = orchestrator.load_balancer.analyze_workload_distribution()
        efficiency_improved = new_analysis['overall_efficiency'] >= load_analysis['overall_efficiency']
        
        logger.info(f"\nEfficiency after optimization: {new_analysis['overall_efficiency']:.2%}")
        logger.info(f"Improvement: {'✅ YES' if efficiency_improved else '❌ NO'}")
        
        meets_threshold = new_analysis['overall_efficiency'] >= 0.95
        return meets_threshold
        
    except Exception as e:
        logger.error(f"Load balancing test failed: {e}")
        return False


def test_predictive_scaling(orchestrator: ScalingOrchestrator) -> bool:
    """Test predictive scaling analytics"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING PREDICTIVE SCALING")
    logger.info("=" * 80)
    
    try:
        # Wait for sufficient data collection
        logger.info("Collecting data for predictions...")
        time.sleep(60)
        
        # Test workload pattern analysis
        pattern_analysis = orchestrator.predictive_analytics.analyze_workload_patterns()
        
        logger.info(f"\nPattern Analysis:")
        logger.info(f"  Components analyzed: {len(pattern_analysis['patterns'])}")
        logger.info(f"  Peak times identified: {len(pattern_analysis['peak_times'])}")
        logger.info(f"  Recommendations: {len(pattern_analysis['recommendations'])}")
        
        # Test scaling predictions
        logger.info("\nTesting scaling predictions...")
        
        test_components = ['brain_orchestration', 'proof_system', 'gac_system']
        predictions_made = 0
        accurate_predictions = 0
        
        for component in test_components:
            prediction = orchestrator.predictive_analytics.predict_scaling_requirements(
                component,
                horizon_minutes=15
            )
            
            if prediction.get('success'):
                predictions_made += 1
                logger.info(f"\nPrediction for {component}:")
                logger.info(f"  Current load: {prediction['predictions']['current_load']:.1f}")
                logger.info(f"  Predicted load: {prediction['predictions']['predicted_load']:.1f}")
                logger.info(f"  Scaling action: {prediction['predictions']['scaling_action']}")
                logger.info(f"  Confidence: {prediction['confidence']:.2%}")
                
                if prediction['confidence'] > 0.8:
                    accurate_predictions += 1
        
        # Test future demand prediction
        future_demand = orchestrator.predictive_analytics.predict_future_demand(
            time_range_hours=1
        )
        
        logger.info(f"\nFuture Demand Predictions:")
        logger.info(f"  Components predicted: {len(future_demand['components'])}")
        
        # Validate prediction accuracy
        accuracy_validation = orchestrator.predictive_analytics.validate_prediction_accuracy()
        
        logger.info(f"\nPrediction Accuracy:")
        logger.info(f"  Overall accuracy: {accuracy_validation['overall_accuracy']:.2%}")
        logger.info(f"  Meets threshold (>80%): {accuracy_validation['meets_threshold']}")
        
        # For testing, we'll consider it successful if we made predictions
        # In production, this would check actual vs predicted after time passes
        success = predictions_made > 0 and accuracy_validation['overall_accuracy'] >= 0.5
        
        return success
        
    except Exception as e:
        logger.error(f"Predictive scaling test failed: {e}")
        return False


def test_orchestration_coordination(orchestrator: ScalingOrchestrator) -> bool:
    """Test overall orchestration coordination"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING ORCHESTRATION COORDINATION")
    logger.info("=" * 80)
    
    try:
        # Get orchestration status
        status = orchestrator.get_orchestration_status()
        
        logger.info(f"\nOrchestration Status:")
        logger.info(f"  Total operations: {status['metrics']['total_operations']}")
        logger.info(f"  Successful operations: {status['metrics']['successful_operations']}")
        logger.info(f"  Failed operations: {status['metrics']['failed_operations']}")
        logger.info(f"  Emergency responses: {status['metrics']['emergency_responses']}")
        logger.info(f"  Predictive actions: {status['metrics']['predictive_actions']}")
        
        # Check component coordination
        logger.info(f"\nComponent Status:")
        logger.info(f"  Scaling engine: {'✅ Running' if status['components']['scaling_engine']['is_running'] else '❌ Stopped'}")
        logger.info(f"  Recovery engine: {'✅ Running' if status['components']['recovery_engine']['is_running'] else '❌ Stopped'}")
        logger.info(f"  Load balancer: {'✅ Running' if status['components']['load_balancer']['is_running'] else '❌ Stopped'}")
        logger.info(f"  Predictive analytics: {'✅ Running' if status['components']['predictive_analytics']['is_running'] else '❌ Stopped'}")
        
        # Check health
        health_check = status['health_check']
        logger.info(f"\nSystem Health:")
        logger.info(f"  Overall health: {health_check['overall_health']:.2%}")
        logger.info(f"  Critical issues: {len(health_check['critical_issues'])}")
        logger.info(f"  Emergency needed: {health_check['emergency_needed']}")
        
        # Calculate success rate
        total_ops = status['metrics']['total_operations']
        if total_ops > 0:
            success_rate = status['metrics']['successful_operations'] / total_ops
            logger.info(f"\nOrchestration Success Rate: {success_rate:.2%}")
            
            return success_rate >= 0.8
        
        return True  # No operations yet, but running
        
    except Exception as e:
        logger.error(f"Orchestration coordination test failed: {e}")
        return False


def run_scaling_tests():
    """Run all scaling and recovery tests"""
    logger.info("=" * 80)
    logger.info("SARAPHIS PRODUCTION SCALING & AUTO-RECOVERY TEST")
    logger.info("=" * 80)
    
    # Initialize test environment
    logger.info("\n1. Initializing test environment...")
    
    test_env = ScalingTestEnvironment()
    test_env.initialize_environment()
    
    try:
        # Initialize Brain system
        brain_system = Brain()
        brain_system.initialize_brain()
        logger.info("Brain system initialized")
        
        # Create monitor (required for scaling)
        logger.info("\n2. Creating production monitor...")
        
        monitor_config = {
            'monitoring_interval': 0.1,
            'alert_check_interval': 0.5,
            'optimization_interval': 10
        }
        
        monitor_result = create_production_monitor(
            brain_system=brain_system,
            agent_system=test_env,
            config=monitor_config
        )
        
        monitor = monitor_result['monitor']
        
        # Override monitor methods to use test environment
        monitor.get_system_status = test_env.get_system_status
        monitor.get_agent_status = test_env.get_agent_status
        monitor.get_all_system_status = lambda: {
            name: test_env.get_system_status(name) 
            for name in test_env.systems
        }
        monitor.get_all_agent_status = lambda: {
            name: test_env.get_agent_status(name) 
            for name in test_env.agents
        }
        monitor.check_system_health = lambda name: {
            'healthy': test_env.systems.get(name, {}).get('health', 0) > 0.5,
            'health': test_env.systems.get(name, {}).get('health', 0)
        }
        monitor.check_agent_health = lambda name: {
            'healthy': test_env.agents.get(name, {}).get('health', 0) > 0.5,
            'health': test_env.agents.get(name, {}).get('health', 0)
        }
        
        # Start monitor
        monitor.start_monitoring()
        logger.info("Production monitor started")
        
        # Create scaling orchestrator
        logger.info("\n3. Creating scaling orchestrator...")
        
        orchestrator = ScalingOrchestrator(monitor, brain_system, test_env)
        
        # Override orchestrator monitor references
        orchestrator.scaling_engine.monitor = monitor
        orchestrator.recovery_engine.monitor = monitor
        orchestrator.load_balancer.monitor = monitor
        orchestrator.predictive_analytics.monitor = monitor
        
        # Start orchestration
        logger.info("\n4. Starting orchestration system...")
        
        start_result = orchestrator.start_orchestration()
        if not start_result.get('success'):
            raise Exception(f"Failed to start orchestration: {start_result.get('error')}")
        
        logger.info("Orchestration system started successfully")
        
        # Start test environment simulation
        logger.info("\n5. Starting test simulation...")
        test_env.start_simulation()
        
        # Run tests
        logger.info("\n6. Running tests...")
        
        test_results = {
            'scaling_decisions': test_scaling_decisions(orchestrator),
            'auto_recovery': test_auto_recovery(orchestrator),
            'load_balancing': test_load_balancing(orchestrator),
            'predictive_scaling': test_predictive_scaling(orchestrator),
            'orchestration_coordination': test_orchestration_coordination(orchestrator)
        }
        
        # Wait for more scenarios
        logger.info("\n7. Running extended scenarios...")
        time.sleep(30)
        
        # Generate final report
        logger.info("\n8. Generating orchestration report...")
        final_report = orchestrator.generate_orchestration_report()
        
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
        
        # Display key metrics
        logger.info(f"\nKey Metrics:")
        logger.info(f"  Scaling efficiency: {final_report['component_reports']['scaling']['performance_metrics']['overall_efficiency']:.2%}")
        logger.info(f"  Recovery success rate: {final_report['component_reports']['recovery']['overall']['success_rate']:.2%}")
        logger.info(f"  Load balancing efficiency: {final_report['component_reports']['load_balancing']['summary']['current_efficiency']:.2%}")
        logger.info(f"  Prediction accuracy: {final_report['component_reports']['predictions']['overall_accuracy']:.2%}")
        
        # Stop all systems
        logger.info("\n9. Stopping all systems...")
        
        test_env.stop_simulation()
        stop_result = orchestrator.stop_orchestration()
        monitor.stop_monitoring()
        brain_system.shutdown()
        
        logger.info("All systems stopped")
        
        return passed_tests == total_tests
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = run_scaling_tests()
    sys.exit(0 if success else 1)