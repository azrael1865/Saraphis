"""
Test script for Saraphis Production Launch System
Demonstrates complete production deployment of all 11 systems with 8 specialized agents
NO FALLBACKS - HARD FAILURES ONLY
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from brain import Brain
    from production_launch import (
        ProductionLaunchOrchestrator,
        create_production_launcher,
        LaunchConfiguration
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


def create_production_config() -> Dict[str, Any]:
    """Create production configuration for testing"""
    return {
        'environment': 'production',
        'host': '0.0.0.0',
        'port': 8000,
        'workers': 4,
        'debug_mode': False,
        'log_level': 'INFO',
        
        # Database configuration
        'database': {
            'host': 'localhost',
            'port': 5432,
            'name': 'saraphis_prod',
            'user': 'saraphis',
            'password': 'secure_password_here'
        },
        
        # Redis configuration
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 0
        },
        
        # Security configuration
        'security': {
            'ssl_enabled': True,
            'authentication_required': True,
            'encryption_enabled': True,
            'encryption_key': 'your-32-character-encryption-key-here',
            'audit_logging_enabled': True,
            'rate_limiting_enabled': True
        },
        
        # Monitoring configuration
        'monitoring': {
            'enabled': True,
            'prometheus_url': 'http://localhost:9090',
            'grafana_url': 'http://localhost:3000',
            'metrics_port': 9091
        },
        
        # Logging configuration
        'logging': {
            'level': 'INFO',
            'centralized_logging': True,
            'log_retention_days': 30
        },
        
        # Backup configuration
        'backup': {
            'enabled': True,
            'schedule': '0 2 * * *',  # 2 AM daily
            'retention_days': 30,
            'backup_path': '/var/backups/saraphis'
        },
        
        # Resource limits
        'resource_limits': {
            'max_memory_gb': 32,
            'max_cpu_percent': 80,
            'max_connections': 10000
        },
        
        # GPU configuration
        'gpu_required': False,  # Set to False for testing without GPU
        
        # Other settings
        'orchestrator_workers': 8,
        'orchestrator_timeout': 30,
        'orchestrator_memory_limit': 4096,
        'admin_password': 'SecureAdminPassword123!',
        'api_secret': 'your-api-secret-key',
        'jwt_secret': 'your-jwt-secret-key'
    }


def create_mock_agent_system() -> Dict[str, Any]:
    """Create a mock agent system for testing"""
    return {
        'name': 'MockAgentSystem',
        'initialized': True,
        'agents': {}
    }


def test_pre_launch_validation(launcher: ProductionLaunchOrchestrator) -> bool:
    """Test pre-launch validation"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING PRE-LAUNCH VALIDATION")
    logger.info("=" * 80)
    
    try:
        result = launcher.validate_pre_launch_requirements()
        
        logger.info(f"\nValidation Result: {result.get('valid', False)}")
        
        if result.get('valid'):
            logger.info("✅ All pre-launch requirements validated")
            
            # Show validation checks
            checks = result.get('checks', {})
            for check, status in checks.items():
                status_symbol = "✅" if status else "❌"
                logger.info(f"  {status_symbol} {check}: {status}")
            
            return True
        else:
            logger.error("❌ Pre-launch validation failed")
            logger.error(f"Error: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"Pre-launch validation test failed: {e}")
        return False


def test_system_deployment(launcher: ProductionLaunchOrchestrator) -> bool:
    """Test system deployment"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING SYSTEM DEPLOYMENT")
    logger.info("=" * 80)
    
    try:
        logger.info("Deploying all 11 Saraphis systems...")
        
        result = launcher.deploy_all_systems()
        
        if result.get('success'):
            logger.info(f"✅ Successfully deployed {result.get('systems_deployed', 0)}/11 systems")
            logger.info(f"Success rate: {result.get('success_rate', 0) * 100:.1f}%")
            
            # Show failed deployments if any
            failed = result.get('failed_deployments', [])
            if failed:
                logger.warning(f"Failed deployments: {failed}")
            
            return result.get('success_rate', 0) >= launcher.launch_config.success_rate_threshold
        else:
            logger.error("❌ System deployment failed")
            logger.error(f"Error: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"System deployment test failed: {e}")
        return False


def test_agent_launch(launcher: ProductionLaunchOrchestrator) -> bool:
    """Test agent launch"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING AGENT LAUNCH")
    logger.info("=" * 80)
    
    try:
        logger.info("Launching all 8 specialized agents...")
        
        result = launcher.launch_all_agents()
        
        if result.get('success'):
            logger.info(f"✅ Successfully launched {result.get('agents_launched', 0)}/8 agents")
            logger.info(f"Success rate: {result.get('success_rate', 0) * 100:.1f}%")
            logger.info(f"Coordination score: {result.get('coordination_score', 0) * 100:.1f}%")
            
            # Show failed launches if any
            failed = result.get('failed_launches', [])
            if failed:
                logger.warning(f"Failed launches: {failed}")
            
            return result.get('success_rate', 0) >= launcher.launch_config.success_rate_threshold
        else:
            logger.error("❌ Agent launch failed")
            logger.error(f"Error: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"Agent launch test failed: {e}")
        return False


def test_system_integration(launcher: ProductionLaunchOrchestrator) -> bool:
    """Test system integration"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING SYSTEM INTEGRATION")
    logger.info("=" * 80)
    
    try:
        logger.info("Coordinating system and agent integration...")
        
        result = launcher.coordinate_system_integration()
        
        if result.get('success'):
            logger.info(f"✅ System integration successful")
            logger.info(f"Integration score: {result.get('integration_score', 0) * 100:.1f}%")
            logger.info(f"Tasks completed: {result.get('successful_tasks', 0)}/{result.get('total_tasks', 0)}")
            
            # Show failed integrations if any
            failed = result.get('failed_integrations', [])
            if failed:
                logger.warning(f"Failed integrations: {failed}")
            
            return True
        else:
            logger.error("❌ System integration failed")
            logger.error(f"Error: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"System integration test failed: {e}")
        return False


def test_production_workflows(launcher: ProductionLaunchOrchestrator) -> bool:
    """Test production workflows"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING PRODUCTION WORKFLOWS")
    logger.info("=" * 80)
    
    try:
        logger.info("Testing all production workflows...")
        
        result = launcher.test_production_workflows()
        
        if result.get('success'):
            logger.info(f"✅ Production workflows validated")
            logger.info(f"Workflows passed: {result.get('passed_workflows', 0)}/{result.get('total_workflows', 0)}")
            logger.info(f"Success rate: {result.get('success_rate', 0) * 100:.1f}%")
            
            # Show failed workflows if any
            failed = result.get('failed_workflows', [])
            if failed:
                logger.warning(f"Failed workflows: {failed}")
            
            return True
        else:
            logger.error("❌ Production workflow tests failed")
            logger.error(f"Error: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"Production workflow test failed: {e}")
        return False


def display_launch_report(report: Dict[str, Any]):
    """Display launch report"""
    logger.info("\n" + "=" * 80)
    logger.info("LAUNCH REPORT")
    logger.info("=" * 80)
    
    logger.info(f"\nReport ID: {report.get('report_id', 'Unknown')}")
    logger.info(f"Timestamp: {report.get('timestamp', 'Unknown')}")
    logger.info(f"Overall Success: {report.get('overall_success', False)}")
    logger.info(f"Launch Duration: {report.get('launch_duration', 0):.2f} seconds")
    
    logger.info(f"\nDeployment Summary:")
    logger.info(f"  Systems Deployed: {report.get('systems_deployed', 0)}/11")
    logger.info(f"  Agents Launched: {report.get('agents_launched', 0)}/8")
    logger.info(f"  Success Rate: {report.get('success_rate', 0) * 100:.1f}%")
    logger.info(f"  Performance Score: {report.get('performance_score', 0) * 100:.1f}%")
    logger.info(f"  Integration Status: {report.get('integration_status', 'Unknown')}")
    
    # Show critical issues
    critical_issues = report.get('critical_issues', [])
    if critical_issues:
        logger.warning(f"\nCritical Issues ({len(critical_issues)}):")
        for issue in critical_issues[:5]:  # Show first 5
            logger.warning(f"  - {issue}")
    
    # Show warnings
    warnings = report.get('warnings', [])
    if warnings:
        logger.info(f"\nWarnings ({len(warnings)}):")
        for warning in warnings[:5]:  # Show first 5
            logger.info(f"  - {warning}")
    
    # Show recommendations
    recommendations = report.get('recommendations', [])
    if recommendations:
        logger.info(f"\nRecommendations:")
        for rec in recommendations[:5]:  # Show first 5
            logger.info(f"  - {rec}")


def run_complete_launch_test():
    """Run complete production launch test"""
    logger.info("=" * 80)
    logger.info("SARAPHIS PRODUCTION LAUNCH SYSTEM TEST")
    logger.info("=" * 80)
    
    success = False
    launcher = None
    
    try:
        # Initialize components
        logger.info("\n1. Initializing test components...")
        
        # Create Brain system
        brain_system = Brain()
        brain_system.initialize_brain()
        logger.info("Brain system initialized")
        
        # Create mock agent system
        agent_system = create_mock_agent_system()
        logger.info("Agent system initialized (mock)")
        
        # Create production configuration
        production_config = create_production_config()
        logger.info("Production configuration created")
        
        # Create launch configuration
        launch_config = LaunchConfiguration(
            environment="production",
            max_launch_time_seconds=300,
            parallel_deployments=False,  # Sequential for testing
            validate_before_launch=True,
            monitor_during_launch=True,
            rollback_on_failure=True
        )
        
        # Create launcher
        launcher = create_production_launcher(
            brain_system=brain_system,
            agent_system=agent_system,
            production_config=production_config,
            launch_config=launch_config
        )
        logger.info("Production launcher created")
        
        # Run complete launch
        logger.info("\n2. Starting production launch sequence...")
        
        launch_result = launcher.launch_production_system()
        
        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("LAUNCH RESULT")
        logger.info("=" * 80)
        
        if launch_result.get('success'):
            logger.info("✅ PRODUCTION LAUNCH SUCCESSFUL!")
            display_launch_report(launch_result.get('report', {}))
            success = True
        else:
            logger.error("❌ PRODUCTION LAUNCH FAILED!")
            logger.error(f"Error: {launch_result.get('error', 'Unknown error')}")
            
            # Show recovery actions
            recovery = launch_result.get('recovery', {})
            if recovery.get('recovery_actions'):
                logger.info("\nRecovery Actions Taken:")
                for action in recovery['recovery_actions']:
                    logger.info(f"  - {action.get('action')}: {action.get('result', {}).get('success', False)}")
            
            display_launch_report(launch_result.get('report', {}))
        
        # Save report
        report_file = f"launch_report_{launch_result.get('launch_id', 'unknown')}.json"
        with open(report_file, 'w') as f:
            json.dump(launch_result, f, indent=2, default=str)
        logger.info(f"\nFull report saved to: {report_file}")
        
    except Exception as e:
        logger.error(f"Launch test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    finally:
        # Cleanup
        logger.info("\n3. Cleaning up...")
        if launcher:
            try:
                # Get final monitoring report
                monitor_report = launcher.launch_monitor.get_performance_report()
                logger.info(f"\nFinal Monitoring Report:")
                logger.info(f"  Monitoring Duration: {monitor_report.get('monitoring_duration', 0):.2f}s")
                logger.info(f"  Alerts Generated: {monitor_report.get('alerts_generated', 0)}")
                
                # Stop monitoring
                launcher.launch_monitor.stop_launch_monitoring()
            except:
                pass
        
        if 'brain_system' in locals():
            try:
                brain_system.shutdown()
                logger.info("Brain system shutdown completed")
            except:
                pass
    
    return success


def run_individual_tests():
    """Run individual component tests"""
    logger.info("=" * 80)
    logger.info("RUNNING INDIVIDUAL COMPONENT TESTS")
    logger.info("=" * 80)
    
    try:
        # Initialize components
        brain_system = Brain()
        brain_system.initialize_brain()
        agent_system = create_mock_agent_system()
        production_config = create_production_config()
        
        launcher = create_production_launcher(
            brain_system=brain_system,
            agent_system=agent_system,
            production_config=production_config
        )
        
        # Run individual tests
        tests = [
            ("Pre-Launch Validation", test_pre_launch_validation),
            ("System Deployment", test_system_deployment),
            ("Agent Launch", test_agent_launch),
            ("System Integration", test_system_integration),
            ("Production Workflows", test_production_workflows)
        ]
        
        results = {}
        for test_name, test_func in tests:
            logger.info(f"\n{'=' * 40}")
            logger.info(f"Running: {test_name}")
            logger.info('=' * 40)
            
            try:
                passed = test_func(launcher)
                results[test_name] = passed
                
                if passed:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
                    
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
                results[test_name] = False
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        
        total_tests = len(results)
        passed_tests = sum(1 for passed in results.values() if passed)
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {(passed_tests / total_tests * 100):.1f}%")
        
        # Cleanup
        brain_system.shutdown()
        
        return passed_tests == total_tests
        
    except Exception as e:
        logger.error(f"Individual tests failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test Saraphis Production Launch System"
    )
    parser.add_argument(
        '--individual',
        action='store_true',
        help='Run individual component tests'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        default=True,
        help='Run complete launch test (default)'
    )
    
    args = parser.parse_args()
    
    if args.individual:
        success = run_individual_tests()
    else:
        success = run_complete_launch_test()
    
    sys.exit(0 if success else 1)