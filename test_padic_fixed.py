#!/usr/bin/env python3
"""
Test FIXED P-adic Compression System
Validates that compression happens BEFORE P-adic transformation
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
import time
from typing import Dict, Any, List, Tuple

# Configure paths
base_path = Path(__file__).parent
sys.path.insert(0, str(base_path / "independent_core"))

# Import the fixed system
from compression_systems.padic.padic_compressor import (
    PadicCompressionSystem,
    CompressionConfig,
    CompressionResult,
    DecompressionResult
)

class Colors:
    """Terminal colors for pretty output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_info(text: str, value: Any = None):
    """Print info message"""
    if value is not None:
        print(f"{Colors.CYAN}  {text}: {Colors.BOLD}{value}{Colors.END}")
    else:
        print(f"{Colors.CYAN}  {text}{Colors.END}")

def format_size(bytes: int) -> str:
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"

def format_ratio(ratio: float) -> str:
    """Format compression ratio with color"""
    if ratio > 1.5:
        color = Colors.GREEN
        symbol = "↓"  # Good compression
    elif ratio > 1.0:
        color = Colors.YELLOW
        symbol = "→"  # Slight compression
    else:
        color = Colors.RED
        symbol = "↑"  # Expansion
    return f"{color}{ratio:.2f}x {symbol}{Colors.END}"

class TestPadicCompressionFixed:
    def __init__(self):
        self.system = None
        self.config = None
        self.device = None
        self.test_results = []
        
    def setup(self):
        """Set up test environment"""
        print_header("SETTING UP FIXED P-ADIC COMPRESSION SYSTEM")
        
        self.config = CompressionConfig(
            prime=257,
            base_precision=2,    # Reduced for less expansion
            min_precision=1,     # Minimum viable precision
            max_precision=3,     # Maximum safe precision
            target_error=1e-6,
            importance_threshold=0.1,
            compression_priority=0.8,  # Prioritize compression over accuracy
            enable_gpu=torch.cuda.is_available(),
            validate_reconstruction=True,
            chunk_size=1000,
            max_tensor_size=1_000_000,
            enable_memory_monitoring=True,
            sparsity_threshold=1e-6,
            huffman_threshold=2.0,
            arithmetic_threshold=6.0,
            enable_hybrid_entropy=True,
            raise_on_error=True
        )
        
        self.system = PadicCompressionSystem(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print_success(f"System initialized with prime={self.config.prime}")
        print_success(f"Base precision={self.config.base_precision} (reduced from 4)")
        print_success(f"Pipeline: Pattern→Sparse→Entropy→P-adic (FIXED ORDER)")
        print_success(f"Using device: {self.device}")

    def test_basic_compression(self) -> bool:
        """Test basic compression/decompression with fixed pipeline"""
        print_header("TEST 1: BASIC COMPRESSION")
        
        # Create test tensor
        test_tensor = torch.randn(100, 100, device=self.device)
        importance = torch.abs(test_tensor) + 0.1
        
        original_size = test_tensor.numel() * 4  # float32 = 4 bytes
        
        print_info("Input tensor shape", test_tensor.shape)
        print_info("Input size", format_size(original_size))
        
        try:
            # Compress
            print(f"\n{Colors.MAGENTA}Compressing...{Colors.END}")
            result = self.system.compress(test_tensor, importance)
            
            # Display stage metrics
            print(f"\n{Colors.BOLD}Stage Metrics:{Colors.END}")
            for stage, metrics in result.stage_metrics.items():
                time_ms = metrics.get('time', 0) * 1000
                if 'compression_ratio' in metrics:
                    ratio = format_ratio(metrics['compression_ratio'])
                    print(f"  {stage}: {time_ms:.1f}ms, ratio: {ratio}")
                elif 'expansion_ratio' in metrics:
                    ratio = metrics['expansion_ratio']
                    print(f"  {stage}: {time_ms:.1f}ms, expansion: {Colors.YELLOW}{ratio:.2f}x{Colors.END}")
                else:
                    print(f"  {stage}: {time_ms:.1f}ms")
            
            compressed_size = len(result.compressed_data)
            
            print_success("Compression successful")
            print_info("Compressed size", format_size(compressed_size))
            print_info("Compression ratio", format_ratio(result.compression_ratio))
            print_info("Processing time", f"{result.processing_time:.3f}s")
            print_info("Validation", "PASSED" if result.validation_passed else "FAILED")
            
            # Check if we achieved compression
            if result.compression_ratio > 1.0:
                print_success("ACHIEVED ACTUAL COMPRESSION!")
            else:
                print_warning("Still expanding - pipeline may need tuning")
            
            # Decompress
            print(f"\n{Colors.MAGENTA}Decompressing...{Colors.END}")
            decompressed = self.system.decompress(result.compressed_data)
            
            # Verify reconstruction
            error = torch.nn.functional.mse_loss(test_tensor, decompressed.reconstructed_tensor)
            max_error = torch.max(torch.abs(test_tensor - decompressed.reconstructed_tensor)).item()
            
            print_success("Decompression successful")
            print_info("Reconstruction MSE", f"{error:.2e}")
            print_info("Max absolute error", f"{max_error:.2e}")
            
            # Store result
            self.test_results.append({
                'test': 'basic_compression',
                'passed': error < self.config.target_error * 10,
                'compression_ratio': result.compression_ratio,
                'mse': error.item(),
                'time': result.processing_time
            })
            
            return error < self.config.target_error * 10
            
        except Exception as e:
            print_error(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_sparse_tensor(self) -> bool:
        """Test compression of sparse tensor"""
        print_header("TEST 2: SPARSE TENSOR COMPRESSION")
        
        # Create sparse tensor (95% zeros)
        test_tensor = torch.randn(200, 200, device=self.device)
        mask = torch.rand(200, 200, device=self.device) > 0.95
        test_tensor = test_tensor * mask.float()
        
        sparsity = 1.0 - (torch.count_nonzero(test_tensor).item() / test_tensor.numel())
        original_size = test_tensor.numel() * 4
        
        print_info("Tensor shape", test_tensor.shape)
        print_info("Sparsity", f"{sparsity:.1%}")
        print_info("Original size", format_size(original_size))
        
        try:
            # Compress
            result = self.system.compress(test_tensor)
            
            print_success("Compression successful")
            print_info("Compression ratio", format_ratio(result.compression_ratio))
            
            # Sparse tensors should compress very well
            if result.compression_ratio > 5.0:
                print_success(f"Excellent compression for sparse data!")
            elif result.compression_ratio > 2.0:
                print_success(f"Good compression for sparse data")
            else:
                print_warning(f"Sparse compression could be better")
            
            # Decompress and verify
            decompressed = self.system.decompress(result.compressed_data)
            error = torch.nn.functional.mse_loss(test_tensor, decompressed.reconstructed_tensor)
            
            print_info("Reconstruction MSE", f"{error:.2e}")
            
            self.test_results.append({
                'test': 'sparse_tensor',
                'passed': error < self.config.target_error * 10,
                'compression_ratio': result.compression_ratio,
                'mse': error.item(),
                'sparsity': sparsity
            })
            
            return error < self.config.target_error * 10
            
        except Exception as e:
            print_error(f"Test failed: {e}")
            return False

    def test_uniform_tensor(self) -> bool:
        """Test compression of uniform/repetitive tensor"""
        print_header("TEST 3: UNIFORM TENSOR COMPRESSION")
        
        # Create tensor with repetitive patterns
        pattern = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0], device=self.device)
        test_tensor = pattern.repeat(100, 100).view(100, 100, 5)[:, :, :4].contiguous()
        test_tensor = test_tensor.view(100, 100)
        
        original_size = test_tensor.numel() * 4
        
        print_info("Tensor shape", test_tensor.shape)
        print_info("Original size", format_size(original_size))
        print_info("Pattern", "Repetitive (should compress well)")
        
        try:
            # Compress
            result = self.system.compress(test_tensor)
            
            print_success("Compression successful")
            print_info("Compression ratio", format_ratio(result.compression_ratio))
            
            # Check pattern detection
            pattern_metrics = result.stage_metrics.get('pattern_detection', {})
            patterns_found = pattern_metrics.get('patterns_found', 0)
            print_info("Patterns detected", patterns_found)
            
            if result.compression_ratio > 10.0:
                print_success(f"Excellent compression for repetitive data!")
            elif result.compression_ratio > 3.0:
                print_success(f"Good compression for repetitive data")
            else:
                print_warning(f"Pattern detection may need improvement")
            
            # Decompress and verify
            decompressed = self.system.decompress(result.compressed_data)
            error = torch.nn.functional.mse_loss(test_tensor, decompressed.reconstructed_tensor)
            
            print_info("Reconstruction MSE", f"{error:.2e}")
            
            self.test_results.append({
                'test': 'uniform_tensor',
                'passed': error < self.config.target_error * 10,
                'compression_ratio': result.compression_ratio,
                'mse': error.item(),
                'patterns': patterns_found
            })
            
            return error < self.config.target_error * 10
            
        except Exception as e:
            print_error(f"Test failed: {e}")
            return False

    def test_large_tensor(self) -> bool:
        """Test compression of larger tensor"""
        print_header("TEST 4: LARGE TENSOR COMPRESSION")
        
        # Create larger tensor
        test_tensor = torch.randn(500, 500, device=self.device)
        importance = torch.abs(test_tensor) * 0.5 + 0.5
        
        original_size = test_tensor.numel() * 4
        
        print_info("Tensor shape", test_tensor.shape)
        print_info("Original size", format_size(original_size))
        
        try:
            # Time the compression
            start_time = time.perf_counter()
            result = self.system.compress(test_tensor, importance)
            compress_time = time.perf_counter() - start_time
            
            print_success("Compression successful")
            print_info("Compression ratio", format_ratio(result.compression_ratio))
            print_info("Compression time", f"{compress_time:.3f}s")
            print_info("Throughput", f"{original_size / compress_time / 1024 / 1024:.1f} MB/s")
            
            # Time the decompression
            start_time = time.perf_counter()
            decompressed = self.system.decompress(result.compressed_data)
            decompress_time = time.perf_counter() - start_time
            
            error = torch.nn.functional.mse_loss(test_tensor, decompressed.reconstructed_tensor)
            
            print_success("Decompression successful")
            print_info("Decompression time", f"{decompress_time:.3f}s")
            print_info("Reconstruction MSE", f"{error:.2e}")
            
            self.test_results.append({
                'test': 'large_tensor',
                'passed': error < self.config.target_error * 10,
                'compression_ratio': result.compression_ratio,
                'mse': error.item(),
                'compress_time': compress_time,
                'decompress_time': decompress_time
            })
            
            return error < self.config.target_error * 10
            
        except Exception as e:
            print_error(f"Test failed: {e}")
            return False

    def test_pipeline_order(self) -> bool:
        """Verify the pipeline executes in correct order"""
        print_header("TEST 5: PIPELINE ORDER VERIFICATION")
        
        test_tensor = torch.randn(50, 50, device=self.device)
        
        print_info("Testing pipeline execution order...")
        
        try:
            # Compress and check stage order
            result = self.system.compress(test_tensor)
            
            expected_order = [
                'pattern_detection',
                'sparse_encoding', 
                'entropy_coding',
                'padic_transformation',
                'metadata_compression'
            ]
            
            actual_order = list(result.stage_metrics.keys())
            
            print_info("Expected order", " → ".join(expected_order))
            print_info("Actual order", " → ".join(actual_order))
            
            order_correct = actual_order == expected_order
            
            if order_correct:
                print_success("Pipeline order is CORRECT!")
            else:
                print_error("Pipeline order is WRONG!")
            
            # Check that P-adic happens AFTER compression
            padic_metrics = result.stage_metrics.get('padic_transformation', {})
            expansion = padic_metrics.get('expansion_ratio', 1.0)
            
            if expansion < 10:
                print_success(f"P-adic expansion is reasonable: {expansion:.2f}x")
                print_info("This means compression happened BEFORE P-adic transform")
            else:
                print_warning(f"P-adic expansion is high: {expansion:.2f}x")
                print_info("May need to tune compression parameters")
            
            self.test_results.append({
                'test': 'pipeline_order',
                'passed': order_correct,
                'correct_order': order_correct,
                'padic_expansion': expansion
            })
            
            return order_correct
            
        except Exception as e:
            print_error(f"Test failed: {e}")
            return False

    def run_all_tests(self) -> bool:
        """Run all tests"""
        print_header("FIXED P-ADIC COMPRESSION TEST SUITE")
        print(f"{Colors.BOLD}Testing the corrected pipeline order{Colors.END}")
        print(f"{Colors.CYAN}Compress → Transform for GPU (not Transform → Compress){Colors.END}\n")
        
        try:
            self.setup()
            print_success("SETUP SUCCESSFUL")
            
            # Run tests
            test_methods = [
                self.test_basic_compression,
                self.test_sparse_tensor,
                self.test_uniform_tensor,
                self.test_large_tensor,
                self.test_pipeline_order,
            ]
            
            passed_count = 0
            failed_count = 0
            
            for test_method in test_methods:
                try:
                    if test_method():
                        passed_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    failed_count += 1
                    print_error(f"{test_method.__name__} crashed: {e}")
            
            # Summary
            print_header("TEST SUMMARY")
            
            print(f"\n{Colors.BOLD}Results:{Colors.END}")
            print_success(f"Tests Passed: {passed_count}")
            if failed_count > 0:
                print_error(f"Tests Failed: {failed_count}")
            
            # Analyze compression ratios
            print(f"\n{Colors.BOLD}Compression Performance:{Colors.END}")
            for result in self.test_results:
                test_name = result['test']
                ratio = result.get('compression_ratio', 0)
                passed = result.get('passed', False)
                
                status = "✓" if passed else "✗"
                color = Colors.GREEN if passed else Colors.RED
                
                print(f"  {color}{status}{Colors.END} {test_name}: {format_ratio(ratio)}")
            
            # Overall verdict
            print(f"\n{Colors.BOLD}Overall Verdict:{Colors.END}")
            
            avg_ratio = np.mean([r['compression_ratio'] for r in self.test_results])
            
            if avg_ratio > 1.0:
                print_success(f"SYSTEM ACHIEVES COMPRESSION! Average: {avg_ratio:.2f}x")
                print_success("The pipeline fix worked!")
            else:
                print_warning(f"System still expanding. Average: {avg_ratio:.2f}x")
                print_info("May need further tuning of compression parameters")
            
            if failed_count == 0:
                print(f"\n{Colors.GREEN}{Colors.BOLD}✓✓✓ ALL TESTS PASSED ✓✓✓{Colors.END}")
                return True
            else:
                print(f"\n{Colors.RED}{Colors.BOLD}✗✗✗ {failed_count} TESTS FAILED ✗✗✗{Colors.END}")
                return False
                
        except Exception as e:
            print_error(f"SETUP FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    def cleanup(self):
        """Clean up resources"""
        if self.system:
            self.system.cleanup()
            print_success("Cleanup complete")


def main():
    """Main test runner"""
    tester = TestPadicCompressionFixed()
    
    try:
        success = tester.run_all_tests()
    finally:
        tester.cleanup()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
