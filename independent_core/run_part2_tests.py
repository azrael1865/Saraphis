#!/usr/bin/env python3
"""
Test Runner for Part 2: Advanced Testing
Executes advanced integration, performance, and error recovery tests
"""

import sys
import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
import multiprocessing

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Part 2 test modules
from test_integration_scenarios import AdvancedIntegrationScenarios
from test_performance_validation import PerformanceValidationTests
from test_error_recovery import ErrorRecoveryTests

# Import Part 1 test results if available
try:
    from run_part1_tests import Part1TestRunner
except ImportError:
    Part1TestRunner = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('part2_test_results.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Part2TestRunner:
    """Runner for Part 2 advanced tests"""
    
    def __init__(self):
        """Initialize test runner"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = {}
        self.start_time = None
        self.end_time = None
        self.part1_results = None
        
        # Load Part 1 results if available
        self._load_part1_results()
        
    def _load_part1_results(self):
        """Load Part 1 results to check prerequisites"""
        try:
            # Look for most recent Part 1 results
            import glob
            part1_files = glob.glob('part1_results_*.json')
            if part1_files:
                latest_file = max(part1_files, key=os.path.getctime)
                with open(latest_file, 'r') as f:
                    self.part1_results = json.load(f)
                self.logger.info(f"Loaded Part 1 results from {latest_file}")
        except Exception as e:
            self.logger.warning(f"Could not load Part 1 results: {str(e)}")
            
    def check_prerequisites(self) -> bool:
        """Check if Part 1 was successful"""
        if self.part1_results:
            summary = self.part1_results.get('summary', {})
            ready = summary.get('ready_for_part2', False)
            
            if not ready:
                self.logger.error("Part 1 tests did not pass successfully!")
                self.logger.error("Please run and pass Part 1 tests before proceeding.")
                return False
                
            self.logger.info("Part 1 prerequisites satisfied ✓")
            return True
        else:
            self.logger.warning("Part 1 results not found. Proceeding anyway...")
            return True
            
    def run_part2_tests(self) -> Dict[str, Any]:
        """Run all Part 2 tests"""
        self.logger.info("="*80)
        self.logger.info("STARTING PART 2: ADVANCED TESTING")
        self.logger.info("="*80)
        
        # Check prerequisites
        if not self.check_prerequisites():
            return {'error': 'Prerequisites not met'}
            
        self.start_time = time.time()
        
        try:
            # Phase 1: Advanced Integration Scenarios
            self.logger.info("\nPhase 1: Running advanced integration scenarios...")
            self.results['integration_scenarios'] = self._run_integration_scenarios()
            
            # Phase 2: Performance Validation
            self.logger.info("\nPhase 2: Running performance validation tests...")
            self.results['performance_validation'] = self._run_performance_validation()
            
            # Phase 3: Error Recovery Testing
            self.logger.info("\nPhase 3: Running error recovery tests...")
            self.results['error_recovery'] = self._run_error_recovery()
            
            # Generate comprehensive report
            self.end_time = time.time()
            self.results['summary'] = self._generate_summary()
            self.results['combined_report'] = self._generate_combined_report()
            
            # Save results
            self._save_results()
            
            # Display final report
            self._display_report()
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Part 2 test execution failed: {str(e)}")
            raise
            
    def _run_integration_scenarios(self) -> Dict[str, Any]:
        """Run advanced integration scenarios"""
        try:
            runner = AdvancedIntegrationScenarios()
            results = runner.run_all_scenarios()
            
            # Extract key metrics
            summary = results.get('summary', {})
            
            return {
                'status': 'completed',
                'total_scenarios': summary.get('total_scenarios', 0),
                'passed_scenarios': summary.get('passed_scenarios', 0),
                'failed_scenarios': summary.get('failed_scenarios', 0),
                'success_rate': summary.get('success_rate', 0),
                'detailed_results': results,
                'key_findings': self._extract_integration_findings(results)
            }
            
        except Exception as e:
            self.logger.error(f"Integration scenarios failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
            
    def _run_performance_validation(self) -> Dict[str, Any]:
        """Run performance validation tests"""
        try:
            validator = PerformanceValidationTests()
            results = validator.run_all_tests()
            
            # Extract performance report
            report = results.get('performance_report', {})
            
            return {
                'status': 'completed',
                'performance_grade': report.get('performance_grade', 'N/A'),
                'executive_summary': report.get('executive_summary', {}),
                'recommendations': report.get('recommendations', []),
                'detailed_results': results,
                'meets_targets': self._check_performance_targets(results)
            }
            
        except Exception as e:
            self.logger.error(f"Performance validation failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
            
    def _run_error_recovery(self) -> Dict[str, Any]:
        """Run error recovery tests"""
        try:
            tester = ErrorRecoveryTests()
            results = tester.run_all_tests()
            
            # Extract recovery report
            report = results.get('recovery_report', {})
            
            return {
                'status': 'completed',
                'recovery_grade': report.get('recovery_grade', 'N/A'),
                'executive_summary': report.get('executive_summary', {}),
                'recommendations': report.get('recommendations', []),
                'detailed_results': results,
                'system_resilience': report.get('executive_summary', {}).get('system_resilience', 'unknown')
            }
            
        except Exception as e:
            self.logger.error(f"Error recovery tests failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
            
    def _extract_integration_findings(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key findings from integration tests"""
        findings = {}
        
        # End-to-end pipeline results
        if 'end_to_end_pipeline' in results:
            pipeline = results['end_to_end_pipeline']
            findings['pipeline_performance'] = {
                'all_stages_completed': all(
                    stage.get('completed', False) 
                    for stage in pipeline.get('pipeline_stages', {}).values()
                ),
                'accuracy': pipeline.get('metrics', {}).get('end_to_end_accuracy', 0),
                'overhead': pipeline.get('metrics', {}).get('proof_overhead_percent', 100)
            }
            
        # Production simulation results
        if 'production_simulation' in results:
            prod_sim = results['production_simulation']
            findings['production_readiness'] = prod_sim.get('readiness_report', {})
            
        return findings
        
    def _check_performance_targets(self, results: Dict[str, Any]) -> Dict[str, bool]:
        """Check if performance targets are met"""
        targets_met = {}
        
        # Check overhead target
        if 'overhead_analysis' in results:
            overhead = results['overhead_analysis']['analysis'].get('average_overhead_percent', 100)
            targets_met['overhead_target'] = overhead <= 10.0
            
        # Check throughput target
        if 'throughput_scaling' in results:
            throughput = results['throughput_scaling']['analysis'].get('max_throughput_tps', 0)
            targets_met['throughput_target'] = throughput >= 1000
            
        # Check latency target
        if 'latency_profiling' in results:
            p99_latency = results['latency_profiling']['percentiles'].get('total', {}).get('p99', 1000)
            targets_met['latency_target'] = p99_latency <= 100
            
        return targets_met
        
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of Part 2 results"""
        total_time = self.end_time - self.start_time
        
        # Calculate overall metrics
        total_phases = 3
        completed_phases = sum(
            1 for phase in ['integration_scenarios', 'performance_validation', 'error_recovery']
            if self.results.get(phase, {}).get('status') == 'completed'
        )
        
        # Integration metrics
        integration = self.results.get('integration_scenarios', {})
        integration_success = integration.get('success_rate', 0)
        
        # Performance metrics
        performance = self.results.get('performance_validation', {})
        performance_grade = performance.get('performance_grade', 'F')
        meets_all_targets = all(performance.get('meets_targets', {}).values())
        
        # Recovery metrics
        recovery = self.results.get('error_recovery', {})
        recovery_grade = recovery.get('recovery_grade', 'F')
        system_resilience = recovery.get('system_resilience', 'unknown')
        
        return {
            'execution_time_seconds': total_time,
            'timestamp': datetime.now().isoformat(),
            'overall_metrics': {
                'phases_completed': completed_phases,
                'total_phases': total_phases,
                'completion_rate': completed_phases / total_phases
            },
            'phase_results': {
                'integration': {
                    'success_rate': integration_success,
                    'production_ready': integration.get('key_findings', {}).get('production_readiness', {}).get('production_ready', False)
                },
                'performance': {
                    'grade': performance_grade,
                    'meets_all_targets': meets_all_targets
                },
                'recovery': {
                    'grade': recovery_grade,
                    'resilience_level': system_resilience
                }
            },
            'overall_status': self._determine_overall_status()
        }
        
    def _determine_overall_status(self) -> str:
        """Determine overall test status"""
        # Check all phases
        integration_passed = self.results.get('integration_scenarios', {}).get('success_rate', 0) > 0.8
        performance_passed = self.results.get('performance_validation', {}).get('performance_grade', 'F') in ['A', 'B']
        recovery_passed = self.results.get('error_recovery', {}).get('recovery_grade', 'F') in ['A', 'B']
        
        if all([integration_passed, performance_passed, recovery_passed]):
            return 'PRODUCTION_READY'
        elif sum([integration_passed, performance_passed, recovery_passed]) >= 2:
            return 'MOSTLY_READY'
        else:
            return 'NEEDS_IMPROVEMENT'
            
    def _generate_combined_report(self) -> Dict[str, Any]:
        """Generate combined report of Part 1 and Part 2"""
        combined = {
            'test_suite_version': '1.0',
            'execution_date': datetime.now().isoformat(),
            'part1_summary': {},
            'part2_summary': self.results['summary'],
            'overall_assessment': {}
        }
        
        # Include Part 1 summary if available
        if self.part1_results:
            combined['part1_summary'] = self.part1_results.get('summary', {})
            
        # Overall assessment
        part1_success = combined['part1_summary'].get('overall_metrics', {}).get('overall_success_rate', 0)
        part2_status = self.results['summary']['overall_status']
        
        combined['overall_assessment'] = {
            'foundation_tests': 'PASSED' if part1_success > 0.9 else 'FAILED',
            'advanced_tests': part2_status,
            'production_readiness': part2_status == 'PRODUCTION_READY',
            'next_steps': self._generate_next_steps(part2_status)
        }
        
        return combined
        
    def _generate_next_steps(self, status: str) -> List[str]:
        """Generate recommended next steps"""
        steps = []
        
        if status == 'PRODUCTION_READY':
            steps.extend([
                "System is ready for production deployment",
                "Consider running load tests at 2x expected capacity",
                "Set up production monitoring and alerting",
                "Create deployment runbook and rollback procedures"
            ])
        elif status == 'MOSTLY_READY':
            steps.extend([
                "Address remaining issues before production deployment",
                "Focus on failed test areas",
                "Re-run tests after fixes",
                "Consider phased rollout approach"
            ])
        else:
            steps.extend([
                "Significant improvements needed before production",
                "Review all test failures and recommendations",
                "Implement suggested optimizations",
                "Consider architectural improvements"
            ])
            
        return steps
        
    def _save_results(self):
        """Save test results to file"""
        filename = f"part2_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            self.logger.info(f"\nResults saved to: {filename}")
            
            # Also save combined report
            combined_filename = f"combined_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(combined_filename, 'w') as f:
                json.dump(self.results['combined_report'], f, indent=2, default=str)
            self.logger.info(f"Combined report saved to: {combined_filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            
    def _display_report(self):
        """Display final test report"""
        summary = self.results.get('summary', {})
        combined = self.results.get('combined_report', {})
        
        self.logger.info("\n" + "="*80)
        self.logger.info("PART 2 TEST EXECUTION SUMMARY")
        self.logger.info("="*80)
        
        # Phase results
        phase_results = summary.get('phase_results', {})
        
        self.logger.info("\nPHASE RESULTS:")
        
        # Integration
        integration = phase_results.get('integration', {})
        self.logger.info(f"\n  INTEGRATION SCENARIOS:")
        self.logger.info(f"    Success Rate: {integration.get('success_rate', 0):.1%}")
        self.logger.info(f"    Production Ready: {'YES' if integration.get('production_ready', False) else 'NO'}")
        
        # Performance
        performance = phase_results.get('performance', {})
        self.logger.info(f"\n  PERFORMANCE VALIDATION:")
        self.logger.info(f"    Grade: {performance.get('grade', 'N/A')}")
        self.logger.info(f"    Meets All Targets: {'YES' if performance.get('meets_all_targets', False) else 'NO'}")
        
        # Recovery
        recovery = phase_results.get('recovery', {})
        self.logger.info(f"\n  ERROR RECOVERY:")
        self.logger.info(f"    Grade: {recovery.get('grade', 'N/A')}")
        self.logger.info(f"    System Resilience: {recovery.get('resilience_level', 'N/A')}")
        
        # Overall status
        self.logger.info(f"\nOVERALL STATUS: {summary.get('overall_status', 'UNKNOWN')}")
        
        # Execution time
        self.logger.info(f"\nEXECUTION TIME: {summary.get('execution_time_seconds', 0):.2f} seconds")
        
        # Combined assessment
        assessment = combined.get('overall_assessment', {})
        self.logger.info("\n" + "="*80)
        self.logger.info("COMPREHENSIVE TEST SUITE ASSESSMENT")
        self.logger.info("="*80)
        
        self.logger.info(f"\nFoundation Tests (Part 1): {assessment.get('foundation_tests', 'N/A')}")
        self.logger.info(f"Advanced Tests (Part 2): {assessment.get('advanced_tests', 'N/A')}")
        self.logger.info(f"\nPRODUCTION READINESS: {'✅ YES' if assessment.get('production_readiness', False) else '❌ NO'}")
        
        # Next steps
        self.logger.info("\nRECOMMENDED NEXT STEPS:")
        for i, step in enumerate(assessment.get('next_steps', []), 1):
            self.logger.info(f"  {i}. {step}")
            
        self.logger.info("="*80)
        
        # Display key recommendations
        self._display_recommendations()
        
    def _display_recommendations(self):
        """Display consolidated recommendations"""
        self.logger.info("\nKEY RECOMMENDATIONS:")
        
        # Performance recommendations
        perf_recs = self.results.get('performance_validation', {}).get('recommendations', [])
        if perf_recs:
            self.logger.info("\n  Performance:")
            for rec in perf_recs[:3]:  # Top 3
                self.logger.info(f"    • {rec}")
                
        # Recovery recommendations
        recovery_recs = self.results.get('error_recovery', {}).get('recommendations', [])
        if recovery_recs:
            self.logger.info("\n  Error Recovery:")
            for rec in recovery_recs[:3]:  # Top 3
                self.logger.info(f"    • {rec}")
                
        self.logger.info("\n" + "="*80)


def run_parallel_tests(test_runner: Part2TestRunner) -> Dict[str, Any]:
    """Run tests in parallel for faster execution"""
    import concurrent.futures
    
    results = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        # Submit test phases
        futures = {
            executor.submit(test_runner._run_integration_scenarios): 'integration_scenarios',
            executor.submit(test_runner._run_performance_validation): 'performance_validation',
            executor.submit(test_runner._run_error_recovery): 'error_recovery'
        }
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            test_name = futures[future]
            try:
                results[test_name] = future.result()
            except Exception as e:
                results[test_name] = {'status': 'failed', 'error': str(e)}
                
    return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run Part 2: Advanced Testing for Proof System'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run tests in parallel'
    )
    parser.add_argument(
        '--skip-prerequisites',
        action='store_true',
        help='Skip Part 1 prerequisite check'
    )
    parser.add_argument(
        '--phase',
        choices=['integration', 'performance', 'recovery'],
        help='Run only specific test phase'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Run Part 2 tests
    runner = Part2TestRunner()
    
    # Override prerequisite check if requested
    if args.skip_prerequisites:
        runner.check_prerequisites = lambda: True
        
    try:
        if args.phase:
            # Run specific phase
            logger.info(f"Running only {args.phase} phase...")
            
            if args.phase == 'integration':
                results = {'integration_scenarios': runner._run_integration_scenarios()}
            elif args.phase == 'performance':
                results = {'performance_validation': runner._run_performance_validation()}
            elif args.phase == 'recovery':
                results = {'error_recovery': runner._run_error_recovery()}
                
            # Display phase results
            logger.info(f"\n{args.phase.upper()} RESULTS:")
            logger.info(json.dumps(results, indent=2, default=str))
            
        else:
            # Run all tests
            if args.parallel:
                logger.info("Running tests in parallel mode...")
                results = run_parallel_tests(runner)
                runner.results = results
                runner.results['summary'] = runner._generate_summary()
                runner.results['combined_report'] = runner._generate_combined_report()
                runner._save_results()
                runner._display_report()
            else:
                results = runner.run_part2_tests()
                
        # Exit code based on success
        if results.get('summary', {}).get('overall_status') == 'PRODUCTION_READY':
            sys.exit(0)
        elif results.get('summary', {}).get('overall_status') == 'MOSTLY_READY':
            sys.exit(1)
        else:
            sys.exit(2)
            
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        sys.exit(3)


if __name__ == "__main__":
    main()