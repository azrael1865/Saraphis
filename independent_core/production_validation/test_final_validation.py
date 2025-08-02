"""
Test script for Production Final Validator
Demonstrates complete production deployment validation
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from brain import Brain
    from production_monitoring_system import ProductionMonitoringSystem, MonitoringConfiguration
    from production_deployment_config import DeploymentConfigManager, DeploymentEnvironment
    from production_security_validator import ProductionSecurityValidator
    from production_config_manager import ProductionConfigManager
    from production_validation.final_validator import (
        ProductionFinalValidator, create_production_final_validator
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


def setup_test_components() -> Dict[str, Any]:
    """Setup test components for validation."""
    logger.info("Setting up test components...")
    
    components = {}
    
    try:
        # Initialize Brain system
        logger.info("Initializing Brain system...")
        brain = Brain()
        brain.initialize_brain()
        components['brain_system'] = brain
        
        # Initialize monitoring system
        logger.info("Initializing monitoring system...")
        monitoring_config = MonitoringConfiguration()
        monitoring_system = ProductionMonitoringSystem(monitoring_config)
        monitoring_system.start_monitoring()
        components['monitoring_system'] = monitoring_system
        
        # Initialize deployment manager
        logger.info("Initializing deployment manager...")
        deployment_manager = DeploymentConfigManager()
        deployment_manager.create_deployment_config(
            DeploymentEnvironment.PRODUCTION
        )
        components['deployment_manager'] = deployment_manager
        
        # Initialize security validator
        logger.info("Initializing security validator...")
        security_validator = ProductionSecurityValidator()
        components['security_validator'] = security_validator
        
        # Add other system components (mocked for testing)
        components.update({
            'brain_orchestrator': brain.brain_orchestrator,
            'neural_orchestrator': None,  # Would be actual component
            'reasoning_orchestrator': None,
            'uncertainty_orchestrator': None,
            'gac_system': None,
            'proof_system': None,
            'compression_api': None,
            'training_manager': None,
            'error_recovery_system': None,
            'universal_ai_core': None
        })
        
        logger.info("All test components initialized successfully")
        return components
        
    except Exception as e:
        logger.error(f"Failed to setup test components: {e}")
        raise


def create_production_config() -> Dict[str, Any]:
    """Create production configuration for testing."""
    return {
        'environment': 'production',
        'host': '0.0.0.0',
        'port': 8000,
        'workers': 4,
        'debug_mode': False,
        'log_level': 'INFO',
        'max_request_size': 10 * 1024 * 1024,  # 10MB
        'request_timeout': 30,
        'database_url': 'postgresql://localhost/saraphis',
        'redis_url': 'redis://localhost:6379',
        'security': {
            'ssl_enabled': True,
            'authentication_required': True,
            'rate_limiting_enabled': True
        },
        'monitoring': {
            'enabled': True,
            'metrics_port': 9090,
            'health_check_interval': 30
        },
        'scaling': {
            'auto_scaling_enabled': True,
            'min_instances': 2,
            'max_instances': 10,
            'target_cpu_percent': 70
        }
    }


def run_validation_test():
    """Run complete production validation test."""
    logger.info("=" * 80)
    logger.info("SARAPHIS BRAIN - PRODUCTION DEPLOYMENT FINAL VALIDATION")
    logger.info("=" * 80)
    
    try:
        # Setup components
        logger.info("\n1. Setting up system components...")
        components = setup_test_components()
        
        # Create production config
        logger.info("\n2. Creating production configuration...")
        production_config = create_production_config()
        
        # Create validator
        logger.info("\n3. Initializing Production Final Validator...")
        validator = create_production_final_validator(
            brain_system=components['brain_system'],
            all_components=components,
            production_config=production_config
        )
        
        # Run complete validation
        logger.info("\n4. Running complete production validation...")
        logger.info("This will validate all 11 systems and generate deployment decision")
        
        start_time = time.time()
        
        try:
            report = validator.run_complete_validation()
            
            # Display results
            logger.info("\n" + "=" * 80)
            logger.info("VALIDATION COMPLETE")
            logger.info("=" * 80)
            
            decision = report['deployment_decision']
            logger.info(f"\nDEPLOYMENT DECISION: {decision['decision']}")
            logger.info(f"Overall Score: {decision['overall_score']:.2%}")
            
            # Display scores
            logger.info("\nVALIDATION SCORES:")
            logger.info(f"  - System Health: {decision['overall_score']:.2%}")
            logger.info(f"  - Readiness: {decision['readiness_score']:.2%}")
            logger.info(f"  - Performance: {decision['performance_score']:.2%}")
            logger.info(f"  - Security: {decision['security_score']:.2%}")
            logger.info(f"  - Reliability: {decision['reliability_score']:.2%}")
            
            # Display issues
            if decision['critical_issues']:
                logger.warning("\nCRITICAL ISSUES:")
                for issue in decision['critical_issues']:
                    logger.warning(f"  - {issue}")
            
            if decision['warnings']:
                logger.info("\nWARNINGS:")
                for warning in decision['warnings']:
                    logger.info(f"  - {warning}")
            
            # Display recommendations
            logger.info("\nRECOMMENDATIONS:")
            for rec in decision['recommendations']:
                logger.info(f"  - {rec}")
            
            # Display component health
            logger.info("\nCOMPONENT HEALTH:")
            component_scores = report.get('scores', {}).get('health', {})
            if isinstance(component_scores, dict):
                for comp, score in component_scores.items():
                    logger.info(f"  - {comp}: {score:.2%}")
            
            # Save full report
            report_file = f"validation_report_{report['report_id']}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"\nFull report saved to: {report_file}")
            
            # Display timing
            duration = time.time() - start_time
            logger.info(f"\nTotal validation time: {duration:.2f} seconds")
            
            # Return success/failure
            if decision['decision'] == "GO":
                logger.info("\n✅ SYSTEM IS READY FOR PRODUCTION DEPLOYMENT!")
                return True
            else:
                logger.error("\n❌ SYSTEM IS NOT READY FOR PRODUCTION DEPLOYMENT!")
                return False
                
        except Exception as e:
            logger.error(f"\nValidation failed with error: {e}")
            logger.error("System is NOT ready for production deployment")
            return False
            
    except Exception as e:
        logger.error(f"Test setup failed: {e}")
        return False
    
    finally:
        # Cleanup
        logger.info("\n5. Cleaning up test components...")
        if 'components' in locals():
            if components.get('monitoring_system'):
                try:
                    components['monitoring_system'].stop_monitoring()
                except:
                    pass
            if components.get('brain_system'):
                try:
                    components['brain_system'].shutdown()
                except:
                    pass


def test_individual_validations():
    """Test individual validation methods."""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING INDIVIDUAL VALIDATIONS")
    logger.info("=" * 80)
    
    try:
        # Setup
        components = setup_test_components()
        production_config = create_production_config()
        validator = create_production_final_validator(
            components['brain_system'],
            components,
            production_config
        )
        
        # Test each validation
        validations = [
            ("System Health", validator.validate_final_system_health),
            ("Production Readiness", validator.test_production_readiness),
            ("Performance", validator.validate_final_performance),
            ("Security", validator.test_final_security),
            ("Scalability", validator.validate_final_scalability),
            ("Reliability", validator.test_final_reliability),
            ("Configuration", validator.validate_production_configuration),
            ("Disaster Recovery", validator.test_disaster_recovery),
            ("Monitoring Systems", validator.validate_monitoring_systems)
        ]
        
        for name, validation_func in validations:
            logger.info(f"\n--- Testing {name} ---")
            try:
                result = validation_func()
                passed = result.get('passed', False)
                score = result.get(f'{result["validation"]}_score', 0.0)
                
                if passed:
                    logger.info(f"✅ {name}: PASSED (Score: {score:.2%})")
                else:
                    logger.warning(f"❌ {name}: FAILED (Score: {score:.2%})")
                
                # Show details for failed validations
                if not passed and 'details' in result:
                    logger.info("  Details:")
                    for detail in result.get('details', []):
                        if isinstance(detail, dict):
                            logger.info(f"    - {detail}")
                            
            except Exception as e:
                logger.error(f"❌ {name}: ERROR - {e}")
        
        # Cleanup
        if components.get('monitoring_system'):
            components['monitoring_system'].stop_monitoring()
        if components.get('brain_system'):
            components['brain_system'].shutdown()
            
    except Exception as e:
        logger.error(f"Individual validation test failed: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test Production Final Validator"
    )
    parser.add_argument(
        '--individual',
        action='store_true',
        help='Test individual validations'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        default=True,
        help='Run full validation test (default)'
    )
    
    args = parser.parse_args()
    
    if args.individual:
        test_individual_validations()
    else:
        success = run_validation_test()
        sys.exit(0 if success else 1)