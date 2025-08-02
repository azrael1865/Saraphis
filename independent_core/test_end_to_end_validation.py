"""
Test script for End-to-End System Validation
Demonstrates comprehensive system validation from web interface to predictions
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


def run_end_to_end_validation():
    """Run complete end-to-end system validation."""
    logger.info("=" * 80)
    logger.info("SARAPHIS BRAIN - END-TO-END SYSTEM VALIDATION")
    logger.info("=" * 80)
    
    try:
        # Initialize Brain system
        logger.info("\n1. Initializing Brain system...")
        brain = Brain()
        brain.initialize_brain()
        logger.info("Brain system initialized successfully")
        
        # Run end-to-end validation
        logger.info("\n2. Starting comprehensive end-to-end validation...")
        logger.info("This will validate:")
        logger.info("  - Complete pipeline from web interface to predictions")
        logger.info("  - All 11 system integrations")
        logger.info("  - Multi-domain workflows")
        logger.info("  - Real-time performance")
        logger.info("  - Training pipeline")
        logger.info("  - Production workflows")
        logger.info("  - Data consistency")
        logger.info("  - Error handling")
        logger.info("  - System reliability")
        
        validation_report = brain._validate_end_to_end_system()
        
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
        
        # System Health Dashboard
        health_dashboard = validation_report.get('system_health_dashboard', {})
        logger.info("\nSYSTEM HEALTH:")
        logger.info(f"  Overall Health: {health_dashboard.get('overall_health', 'unknown').upper()}")
        if 'systems_health' in health_dashboard:
            for system, health in health_dashboard['systems_health'].items():
                logger.info(f"  - {system}: {health}")
        
        # Performance Analysis
        perf_analysis = validation_report.get('performance_analysis', {})
        logger.info("\nPERFORMANCE ANALYSIS:")
        logger.info(f"  Meets Requirements: {perf_analysis.get('meets_requirements', False)}")
        if 'bottlenecks' in perf_analysis:
            logger.info(f"  Bottlenecks: {', '.join(perf_analysis['bottlenecks']) if perf_analysis['bottlenecks'] else 'None'}")
        
        # Reliability Assessment
        reliability = validation_report.get('reliability_assessment', {})
        logger.info("\nRELIABILITY ASSESSMENT:")
        logger.info(f"  System Reliable: {reliability.get('reliable', False)}")
        logger.info(f"  Reliability Score: {reliability.get('reliability_score', 0) * 100:.1f}%")
        logger.info(f"  MTBF: {reliability.get('mtbf_hours', 0)} hours")
        logger.info(f"  MTTR: {reliability.get('mttr_minutes', 0)} minutes")
        
        # Deployment Readiness
        deployment = validation_report.get('deployment_readiness', {})
        logger.info("\nDEPLOYMENT READINESS:")
        logger.info(f"  Ready for Deployment: {deployment.get('ready_for_deployment', False)}")
        logger.info(f"  Readiness Score: {deployment.get('readiness_score', 0) * 100:.1f}%")
        
        if 'blocking_issues' in deployment and deployment['blocking_issues']:
            logger.warning("\n  BLOCKING ISSUES:")
            for issue in deployment['blocking_issues']:
                logger.warning(f"    - {issue}")
        
        # Recommendations
        recommendations = validation_report.get('recommendations', {})
        if recommendations:
            logger.info("\nRECOMMENDATIONS:")
            for priority, items in recommendations.items():
                if items:
                    logger.info(f"  {priority.upper()}:")
                    for item in items:
                        logger.info(f"    - {item}")
        
        # Detailed Validation Results
        detailed_results = validation_report.get('detailed_results', {})
        if 'validations' in detailed_results:
            logger.info("\nDETAILED VALIDATION RESULTS:")
            validations = detailed_results['validations']
            for val_name, val_result in validations.items():
                status = "✅ PASSED" if val_result.get('passed', False) else "❌ FAILED"
                logger.info(f"  {val_name}: {status}")
                if not val_result.get('passed', False) and 'error' in val_result:
                    logger.error(f"    Error: {val_result['error']}")
        
        # Save full report
        report_path = Path(f"e2e_validation_report_{validation_report.get('report_id', 'unknown')}.json")
        with open(report_path, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        logger.info(f"\nFull report saved to: {report_path}")
        
        # Final status
        if deployment.get('ready_for_deployment', False):
            logger.info("\n✅ SYSTEM IS READY FOR PRODUCTION DEPLOYMENT!")
            return True
        else:
            logger.error("\n❌ SYSTEM IS NOT READY FOR PRODUCTION DEPLOYMENT!")
            logger.error("Please address all blocking issues and critical recommendations.")
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


def test_individual_validations():
    """Test individual validation components."""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING INDIVIDUAL VALIDATION COMPONENTS")
    logger.info("=" * 80)
    
    try:
        # Initialize Brain
        brain = Brain()
        brain.initialize_brain()
        
        # Test each validation component
        validations = [
            ("Complete Pipeline", brain._test_complete_pipeline),
            ("Multi-Domain", brain._test_multidomain_end_to_end),
            ("Real-time Performance", brain._validate_realtime_performance),
            ("Training Pipeline", brain._test_training_pipeline),
            ("Production Workflows", brain._validate_production_workflows),
            ("System Integration", brain._test_system_integration),
            ("Data Consistency", brain._validate_data_consistency),
            ("Error Propagation", brain._test_error_propagation),
            ("System Reliability", brain._validate_system_reliability)
        ]
        
        results = {}
        for name, test_func in validations:
            logger.info(f"\n--- Testing {name} ---")
            try:
                result = test_func()
                passed = result.get('passed', False)
                results[name] = passed
                
                if passed:
                    logger.info(f"✅ {name}: PASSED")
                else:
                    logger.warning(f"❌ {name}: FAILED")
                    if 'error' in result:
                        logger.error(f"  Error: {result['error']}")
                
                # Show key metrics
                if 'metrics' in result:
                    logger.info("  Metrics:")
                    for metric, value in result['metrics'].items():
                        logger.info(f"    - {metric}: {value}")
                        
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test Saraphis Brain End-to-End System Validation"
    )
    parser.add_argument(
        '--individual',
        action='store_true',
        help='Test individual validation components'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        default=True,
        help='Run full end-to-end validation (default)'
    )
    
    args = parser.parse_args()
    
    if args.individual:
        success = test_individual_validations()
    else:
        success = run_end_to_end_validation()
    
    sys.exit(0 if success else 1)