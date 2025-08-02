"""
Test script for Production Deployment Final Validation in Brain
Demonstrates comprehensive production deployment validation using Brain class methods
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from brain import Brain
except ImportError as e:
    logging.error(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_production_deployment_validation():
    """Run complete production deployment validation."""
    logger.info("=" * 80)
    logger.info("SARAPHIS BRAIN - PRODUCTION DEPLOYMENT FINAL VALIDATION")
    logger.info("=" * 80)
    
    try:
        # Initialize Brain system
        logger.info("\n1. Initializing Brain system...")
        brain = Brain()
        brain.initialize_brain()
        logger.info("Brain system initialized successfully")
        
        # Run production deployment validation
        logger.info("\n2. Starting production deployment final validation...")
        logger.info("This will validate:")
        logger.info("  - Final system health across all 11 systems")
        logger.info("  - Production readiness requirements")
        logger.info("  - Performance under production load")
        logger.info("  - Security hardening and compliance")
        logger.info("  - Scalability for expected growth")
        logger.info("  - Reliability and fault tolerance")
        logger.info("  - Production configuration")
        logger.info("  - Disaster recovery capabilities")
        logger.info("  - Monitoring and observability systems")
        
        validation_report = brain._validate_final_production_deployment()
        
        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION COMPLETE")
        logger.info("=" * 80)
        
        # Executive Summary
        exec_summary = validation_report.get('executive_summary', {})
        logger.info("\nEXECUTIVE SUMMARY:")
        logger.info(f"  Overall Status: {exec_summary.get('overall_status', 'unknown').upper()}")
        logger.info(f"  Validation Score: {exec_summary.get('validation_score', 0) * 100:.1f}%")
        logger.info(f"  Critical Issues: {exec_summary.get('critical_issues', 0)}")
        logger.info(f"  Warnings: {exec_summary.get('warnings', 0)}")
        logger.info(f"  Duration: {exec_summary.get('total_duration_seconds', 0):.2f} seconds")
        
        # Deployment Decision
        deployment_decision = validation_report.get('deployment_decision', {})
        logger.info("\nDEPLOYMENT DECISION:")
        logger.info(f"  Decision: {deployment_decision.get('decision', 'NO-GO')}")
        logger.info(f"  Overall Score: {deployment_decision.get('overall_score', 0) * 100:.1f}%")
        logger.info(f"  Ready for Production: {deployment_decision.get('ready_for_production', False)}")
        
        # Component Scores
        scores = deployment_decision.get('scores', {})
        logger.info("\nCOMPONENT SCORES:")
        logger.info(f"  - System Health: {scores.get('health', 0) * 100:.1f}%")
        logger.info(f"  - Production Readiness: {scores.get('readiness', 0) * 100:.1f}%")
        logger.info(f"  - Performance: {scores.get('performance', 0) * 100:.1f}%")
        logger.info(f"  - Security: {scores.get('security', 0) * 100:.1f}%")
        logger.info(f"  - Scalability: {scores.get('scalability', 0) * 100:.1f}%")
        logger.info(f"  - Reliability: {scores.get('reliability', 0) * 100:.1f}%")
        logger.info(f"  - Configuration: {scores.get('configuration', 0) * 100:.1f}%")
        logger.info(f"  - Disaster Recovery: {scores.get('disaster_recovery', 0) * 100:.1f}%")
        logger.info(f"  - Monitoring: {scores.get('monitoring', 0) * 100:.1f}%")
        
        # Critical Issues
        if deployment_decision.get('critical_issues'):
            logger.warning("\nCRITICAL ISSUES:")
            for issue in deployment_decision['critical_issues']:
                logger.warning(f"  - {issue}")
        
        # Warnings
        if deployment_decision.get('warnings'):
            logger.info("\nWARNINGS:")
            for warning in deployment_decision['warnings']:
                logger.info(f"  - {warning}")
        
        # Recommendations
        if deployment_decision.get('recommendations'):
            logger.info("\nRECOMMENDATIONS:")
            for rec in deployment_decision['recommendations']:
                logger.info(f"  - {rec}")
        
        # Detailed Validation Results
        validation_results = validation_report.get('validation_results', {})
        if validation_results:
            logger.info("\nDETAILED VALIDATION RESULTS:")
            for val_name, val_result in validation_results.items():
                status = "✅ PASSED" if val_result.get('passed', False) else "❌ FAILED"
                score = val_result.get(f'{val_result.get("validation", "")}_score', 0)
                logger.info(f"  {val_name}: {status} (Score: {score * 100:.1f}%)")
                if not val_result.get('passed', False) and 'error' in val_result:
                    logger.error(f"    Error: {val_result['error']}")
        
        # Save full report
        report_path = Path(f"production_deployment_report_{validation_report.get('report_id', 'unknown')}.json")
        with open(report_path, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        logger.info(f"\nFull report saved to: {report_path}")
        
        # Final status
        if deployment_decision.get('ready_for_production', False):
            logger.info("\n✅ SYSTEM IS READY FOR PRODUCTION DEPLOYMENT!")
            logger.info("All validation checks passed. System meets production requirements.")
            return True
        else:
            logger.error("\n❌ SYSTEM IS NOT READY FOR PRODUCTION DEPLOYMENT!")
            logger.error("Please address all critical issues and ensure all scores meet minimum requirements.")
            logger.error("Required minimum scores:")
            logger.error("  - System Health: 99.9%")
            logger.error("  - Production Readiness: 95%")
            logger.error("  - Performance: 90%")
            logger.error("  - Security: 90%")
            logger.error("  - Reliability: 99%")
            return False
            
    except Exception as e:
        logger.error(f"\nValidation failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    finally:
        # Cleanup
        if 'brain' in locals():
            try:
                brain.shutdown()
                logger.info("\nBrain system shutdown completed")
            except:
                pass


def test_individual_production_validations():
    """Test individual production validation components."""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING INDIVIDUAL PRODUCTION VALIDATION COMPONENTS")
    logger.info("=" * 80)
    
    try:
        # Initialize Brain
        brain = Brain()
        brain.initialize_brain()
        
        # Test each validation component
        validations = [
            ("System Health", brain._test_final_system_health),
            ("Production Readiness", brain._test_production_readiness),
            ("Performance", brain._validate_final_performance),
            ("Security", brain._test_final_security),
            ("Scalability", brain._validate_final_scalability),
            ("Reliability", brain._test_final_reliability),
            ("Configuration", brain._validate_production_configuration),
            ("Disaster Recovery", brain._test_disaster_recovery),
            ("Monitoring Systems", brain._validate_monitoring_systems)
        ]
        
        results = {}
        for name, test_func in validations:
            logger.info(f"\n--- Testing {name} ---")
            try:
                result = test_func()
                passed = result.get('passed', False)
                score = result.get(f'{result.get("validation", "")}_score', 0)
                results[name] = passed
                
                if passed:
                    logger.info(f"✅ {name}: PASSED (Score: {score * 100:.1f}%)")
                else:
                    logger.warning(f"❌ {name}: FAILED (Score: {score * 100:.1f}%)")
                    if 'error' in result:
                        logger.error(f"  Error: {result['error']}")
                
                # Show key details
                if 'details' in result:
                    logger.info("  Details:")
                    details = result['details']
                    if isinstance(details, dict):
                        for key, value in details.items():
                            logger.info(f"    - {key}: {value}")
                    elif isinstance(details, list):
                        for detail in details[:5]:  # Show first 5 details
                            logger.info(f"    - {detail}")
                        
            except Exception as e:
                logger.error(f"❌ {name}: ERROR - {e}")
                results[name] = False
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        
        total_tests = len(results)
        passed_tests = sum(1 for passed in results.values() if passed)
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {(passed_tests / total_tests * 100):.1f}%")
        
        # Cleanup
        brain.shutdown()
        
        return passed_tests == total_tests
        
    except Exception as e:
        logger.error(f"Individual validation test failed: {e}")
        return False


def test_deployment_report_generation():
    """Test deployment report generation."""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING DEPLOYMENT REPORT GENERATION")
    logger.info("=" * 80)
    
    try:
        # Initialize Brain
        brain = Brain()
        brain.initialize_brain()
        
        # Create mock validation results
        mock_results = {
            'health': {'passed': True, 'health_score': 0.999},
            'readiness': {'passed': True, 'readiness_score': 0.96},
            'performance': {'passed': True, 'performance_score': 0.92},
            'security': {'passed': True, 'security_score': 0.91},
            'scalability': {'passed': True, 'scalability_score': 0.88},
            'reliability': {'passed': True, 'reliability_score': 0.992},
            'configuration': {'passed': True, 'configuration_score': 0.95},
            'disaster_recovery': {'passed': True, 'disaster_recovery_score': 0.90},
            'monitoring': {'passed': True, 'monitoring_score': 0.93}
        }
        
        # Generate report
        logger.info("Generating deployment report...")
        report = brain._generate_final_deployment_report(mock_results)
        
        # Display report summary
        logger.info("\nREPORT GENERATED:")
        logger.info(f"  Report ID: {report.get('report_id', 'unknown')}")
        logger.info(f"  Timestamp: {report.get('timestamp', 'unknown')}")
        
        exec_summary = report.get('executive_summary', {})
        logger.info(f"\n  Executive Summary:")
        logger.info(f"    - Overall Status: {exec_summary.get('overall_status', 'unknown')}")
        logger.info(f"    - Validation Score: {exec_summary.get('validation_score', 0) * 100:.1f}%")
        
        deployment_decision = report.get('deployment_decision', {})
        logger.info(f"\n  Deployment Decision:")
        logger.info(f"    - Decision: {deployment_decision.get('decision', 'NO-GO')}")
        logger.info(f"    - Ready for Production: {deployment_decision.get('ready_for_production', False)}")
        
        # Cleanup
        brain.shutdown()
        
        return True
        
    except Exception as e:
        logger.error(f"Report generation test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test Saraphis Brain Production Deployment Validation"
    )
    parser.add_argument(
        '--individual',
        action='store_true',
        help='Test individual validation components'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Test report generation only'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        default=True,
        help='Run full production deployment validation (default)'
    )
    
    args = parser.parse_args()
    
    if args.individual:
        success = test_individual_production_validations()
    elif args.report:
        success = test_deployment_report_generation()
    else:
        success = run_production_deployment_validation()
    
    sys.exit(0 if success else 1)