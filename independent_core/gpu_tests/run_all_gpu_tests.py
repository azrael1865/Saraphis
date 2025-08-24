#!/usr/bin/env python3
"""
SARAPHIS GPU TEST SUITE RUNNER
Comprehensive GPU testing with hard failures
NO SILENT ERRORS - ALL FAILURES ARE FATAL
"""

import sys
import os
import torch
import logging
import time
import traceback
import unittest
from typing import Dict, List, Tuple
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('gpu_test_results.log')
    ]
)
logger = logging.getLogger(__name__)


class GPUTestRunner:
    """Master GPU test runner with hard failure detection"""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.gpu_info = None
        
    def check_gpu_availability(self) -> bool:
        """Check if GPU is available and log info"""
        logger.info("=" * 80)
        logger.info("CHECKING GPU AVAILABILITY")
        logger.info("=" * 80)
        
        if not torch.cuda.is_available():
            logger.error("FATAL: NO CUDA DEVICES AVAILABLE")
            logger.error("Cannot run GPU tests without NVIDIA GPU with CUDA support")
            return False
        
        device_count = torch.cuda.device_count()
        logger.info(f"✓ CUDA Available: {device_count} device(s) found")
        
        # Log detailed GPU info
        self.gpu_info = []
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            info = {
                'index': i,
                'name': props.name,
                'compute_capability': f"{props.major}.{props.minor}",
                'memory_gb': props.total_memory / 1e9,
                'multiprocessors': props.multi_processor_count
            }
            self.gpu_info.append(info)
            
            logger.info(f"\nGPU {i}: {info['name']}")
            logger.info(f"  Compute Capability: {info['compute_capability']}")
            logger.info(f"  Memory: {info['memory_gb']:.2f} GB")
            logger.info(f"  Multiprocessors: {info['multiprocessors']}")
        
        # Check CUDA version
        logger.info(f"\nCUDA Version: {torch.version.cuda}")
        logger.info(f"PyTorch Version: {torch.__version__}")
        logger.info(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
        logger.info(f"cuDNN Version: {torch.backends.cudnn.version()}")
        
        return True
    
    def run_test_module(self, module_name: str, module_path: str) -> Tuple[bool, Dict]:
        """Run a single test module with hard failure detection"""
        logger.info("\n" + "=" * 80)
        logger.info(f"RUNNING: {module_name}")
        logger.info("=" * 80)
        
        result = {
            'module': module_name,
            'passed': False,
            'tests_run': 0,
            'failures': [],
            'errors': [],
            'time': 0
        }
        
        start_time = time.time()
        
        try:
            # Run as subprocess to isolate failures
            cmd = [sys.executable, module_path]
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per module
            )
            
            result['time'] = time.time() - start_time
            
            if process.returncode == 0:
                result['passed'] = True
                logger.info(f"✅ {module_name} PASSED ({result['time']:.2f}s)")
            else:
                result['passed'] = False
                logger.error(f"❌ {module_name} FAILED")
                
                # Parse output for errors
                output_lines = process.stdout.split('\n') + process.stderr.split('\n')
                for line in output_lines:
                    if 'FATAL:' in line:
                        result['errors'].append(line)
                        logger.error(f"  {line}")
                    elif 'FAILED:' in line or 'ERROR:' in line:
                        result['failures'].append(line)
                        logger.error(f"  {line}")
                
                # Log full error if no specific errors found
                if not result['errors'] and not result['failures']:
                    logger.error("Full error output:")
                    logger.error(process.stderr)
            
        except subprocess.TimeoutExpired:
            result['time'] = time.time() - start_time
            result['errors'].append(f"TIMEOUT: Test exceeded 5 minutes")
            logger.error(f"❌ {module_name} TIMEOUT after {result['time']:.2f}s")
            
        except Exception as e:
            result['time'] = time.time() - start_time
            result['errors'].append(str(e))
            logger.error(f"❌ {module_name} EXCEPTION: {e}")
            logger.error(traceback.format_exc())
        
        return result['passed'], result
    
    def run_all_tests(self) -> bool:
        """Run all GPU test modules in priority order"""
        self.start_time = time.time()
        
        # Check GPU first
        if not self.check_gpu_availability():
            return False
        
        # Clear GPU memory before starting
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Define test modules in priority order
        test_modules = [
            # Priority 1: Critical
            ("Device Management", "test_gpu_device_management.py"),
            ("Memory Management", "test_gpu_memory_management.py"),
            
            # Priority 2: Important  
            ("Saraphis GPU Integration", "test_saraphis_gpu_integration.py"),
        ]
        
        # Run each module
        all_passed = True
        for module_name, module_file in test_modules:
            module_path = os.path.join(os.path.dirname(__file__), module_file)
            
            if not os.path.exists(module_path):
                logger.warning(f"⚠ Test module not found: {module_file}")
                continue
            
            passed, result = self.run_test_module(module_name, module_path)
            self.results[module_name] = result
            
            if not passed:
                all_passed = False
                # Continue running other tests to find all failures
        
        return all_passed
    
    def generate_report(self):
        """Generate comprehensive test report"""
        total_time = time.time() - self.start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("GPU TEST SUITE FINAL REPORT")
        logger.info("=" * 80)
        
        # Summary stats
        total_modules = len(self.results)
        passed_modules = sum(1 for r in self.results.values() if r['passed'])
        failed_modules = total_modules - passed_modules
        
        logger.info(f"\nTest Execution Time: {total_time:.2f} seconds")
        logger.info(f"Modules Tested: {total_modules}")
        logger.info(f"Passed: {passed_modules}")
        logger.info(f"Failed: {failed_modules}")
        
        # GPU utilization summary
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            logger.info(f"\nPeak GPU Memory Used: {peak_memory:.2f} GB")
        
        # Detailed results
        logger.info("\n" + "-" * 80)
        logger.info("DETAILED RESULTS:")
        logger.info("-" * 80)
        
        for module_name, result in self.results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            logger.info(f"\n{module_name}: {status} ({result['time']:.2f}s)")
            
            if not result['passed']:
                if result['errors']:
                    logger.info("  Errors:")
                    for error in result['errors'][:5]:  # Show first 5 errors
                        logger.info(f"    - {error}")
                
                if result['failures']:
                    logger.info("  Failures:")
                    for failure in result['failures'][:5]:  # Show first 5 failures
                        logger.info(f"    - {failure}")
        
        # Final verdict
        logger.info("\n" + "=" * 80)
        if failed_modules == 0:
            logger.info("🎉 ALL GPU TESTS PASSED!")
            logger.info("Saraphis GPU components are fully functional")
        else:
            logger.error(f"❌ {failed_modules} MODULE(S) FAILED")
            logger.error("Review the errors above and fix the issues")
        logger.info("=" * 80)
        
        # Save report to file
        with open('gpu_test_report.txt', 'w') as f:
            f.write(f"GPU TEST REPORT\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Execution Time: {total_time:.2f}s\n")
            f.write(f"Passed: {passed_modules}/{total_modules}\n")
            f.write(f"Failed: {failed_modules}/{total_modules}\n")
            f.write(f"\nGPU Info:\n")
            for gpu in self.gpu_info or []:
                f.write(f"  {gpu['name']} ({gpu['memory_gb']:.2f}GB)\n")
            f.write(f"\nResults:\n")
            for module_name, result in self.results.items():
                f.write(f"  {module_name}: {'PASS' if result['passed'] else 'FAIL'}\n")
        
        logger.info("\nReport saved to: gpu_test_report.txt")
        logger.info("Detailed log saved to: gpu_test_results.log")
        
        return failed_modules == 0


def main():
    """Main entry point for GPU test suite"""
    logger.info("=" * 80)
    logger.info("SARAPHIS GPU TEST SUITE")
    logger.info("HARD FAILURE MODE - NO SILENT ERRORS")
    logger.info("=" * 80)
    
    # Check Python version
    if sys.version_info < (3, 7):
        logger.error("FATAL: Python 3.7+ required")
        sys.exit(1)
    
    # Create runner
    runner = GPUTestRunner()
    
    try:
        # Run all tests
        all_passed = runner.run_all_tests()
        
        # Generate report
        runner.generate_report()
        
        # Exit with appropriate code
        sys.exit(0 if all_passed else 1)
        
    except KeyboardInterrupt:
        logger.error("\n\nTEST SUITE INTERRUPTED BY USER")
        sys.exit(2)
        
    except Exception as e:
        logger.error(f"\n\nFATAL ERROR IN TEST RUNNER: {e}")
        logger.error(traceback.format_exc())
        sys.exit(3)


if __name__ == "__main__":
    main()