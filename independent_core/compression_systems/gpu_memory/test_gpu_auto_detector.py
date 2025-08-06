"""
Test GPU Auto-Detection System
Comprehensive tests for GPU detection and auto-configuration
"""

import torch
import unittest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock

from gpu_auto_detector import (
    GPUAutoDetector, 
    GPUSpecs, 
    GPUArchitecture,
    AutoOptimizedConfig,
    ConfigUpdater,
    get_gpu_detector,
    get_config_updater,
    auto_configure_system
)


class TestGPUAutoDetector(unittest.TestCase):
    """Test GPU auto-detection functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = GPUAutoDetector()
    
    def test_detect_all_gpus(self):
        """Test GPU detection"""
        gpus = self.detector.detect_all_gpus()
        
        if torch.cuda.is_available():
            # Should detect at least one GPU
            self.assertGreater(len(gpus), 0)
            
            # Check first GPU specs
            gpu_0 = gpus.get(0)
            self.assertIsNotNone(gpu_0)
            self.assertIsInstance(gpu_0, GPUSpecs)
            
            # Validate specs
            self.assertGreater(gpu_0.total_memory_gb, 0)
            self.assertGreater(gpu_0.multi_processor_count, 0)
            self.assertGreater(gpu_0.cuda_cores, 0)
            self.assertIsInstance(gpu_0.architecture, GPUArchitecture)
        else:
            # No GPUs available
            self.assertEqual(len(gpus), 0)
    
    def test_get_primary_gpu(self):
        """Test primary GPU detection"""
        primary_gpu = self.detector.get_primary_gpu()
        
        if torch.cuda.is_available():
            self.assertIsNotNone(primary_gpu)
            self.assertEqual(primary_gpu.device_id, 0)
        else:
            self.assertIsNone(primary_gpu)
    
    def test_architecture_detection(self):
        """Test GPU architecture family detection"""
        # Test known compute capabilities
        test_cases = [
            ("3.5", GPUArchitecture.KEPLER),
            ("5.2", GPUArchitecture.MAXWELL),
            ("6.1", GPUArchitecture.PASCAL),
            ("7.0", GPUArchitecture.VOLTA),
            ("7.5", GPUArchitecture.TURING),
            ("8.6", GPUArchitecture.AMPERE),
            ("8.9", GPUArchitecture.ADA),
            ("9.0", GPUArchitecture.HOPPER),
        ]
        
        for compute_cap, expected_arch in test_cases:
            arch = self.detector.get_architecture_family(compute_cap)
            self.assertEqual(arch, expected_arch)
    
    def test_cuda_core_estimation(self):
        """Test CUDA core estimation"""
        test_cases = [
            ("RTX 4090", 100, 12800),  # 100 SMs * 128 cores
            ("RTX 3090", 82, 10496),   # 82 SMs * 128 cores
            ("A100", 108, 6912),       # 108 SMs * 64 cores
            ("V100", 80, 5120),        # 80 SMs * 64 cores
        ]
        
        for gpu_name, sm_count, expected_cores in test_cases:
            cores = self.detector.estimate_cuda_cores(gpu_name, sm_count)
            self.assertEqual(cores, expected_cores)
    
    def test_memory_bandwidth_estimation(self):
        """Test memory bandwidth estimation"""
        test_cases = [
            ("RTX 4090", 1008.0),
            ("RTX 3090", 936.2),
            ("A100", 2039.0),
            ("H100", 3350.0),
        ]
        
        for gpu_name, expected_bandwidth in test_cases:
            bandwidth = self.detector.estimate_memory_bandwidth(gpu_name)
            self.assertEqual(bandwidth, expected_bandwidth)
    
    def test_numa_node_detection(self):
        """Test NUMA node detection"""
        numa_nodes = GPUAutoDetector._detect_numa_nodes()
        
        # Should always return at least one node
        self.assertIsInstance(numa_nodes, list)
        self.assertGreater(len(numa_nodes), 0)
        self.assertIn(0, numa_nodes)
    
    def test_cache_functionality(self):
        """Test caching of detection results"""
        # First detection
        gpus1 = self.detector.detect_all_gpus()
        timestamp1 = self.detector.detection_timestamp
        
        # Second detection (should use cache)
        gpus2 = self.detector.detect_all_gpus()
        timestamp2 = self.detector.detection_timestamp
        
        # Timestamps should be the same (cached)
        self.assertEqual(timestamp1, timestamp2)
        
        # Results should be identical
        self.assertEqual(gpus1, gpus2)


class TestAutoOptimizedConfig(unittest.TestCase):
    """Test auto-optimized configuration generation"""
    
    def test_config_from_high_end_gpu(self):
        """Test configuration for high-end GPU (24GB+)"""
        gpu_specs = GPUSpecs(
            device_id=0,
            name="RTX 4090",
            total_memory_gb=24.0,
            total_memory_mb=24576.0,
            compute_capability="8.9",
            multi_processor_count=128,
            max_threads_per_block=1024,
            max_shared_memory_per_block=49152,
            warp_size=32,
            is_integrated=False,
            is_multi_gpu_board=False,
            cuda_cores=16384,
            memory_bandwidth_gb=1008.0,
            architecture=GPUArchitecture.ADA,
            pcie_generation=5,
            memory_clock_mhz=21000.0,
            gpu_clock_mhz=2520.0,
            l2_cache_size_mb=72.0,
            tensor_cores=512,
            rt_cores=128
        )
        
        config = AutoOptimizedConfig.from_gpu_specs(gpu_specs)
        
        # Check memory thresholds
        self.assertEqual(config.memory_pressure_threshold, 0.92)
        self.assertGreater(config.gpu_critical_threshold_mb, 20000)
        
        # Check batch sizes
        self.assertGreater(config.gpu_batch_size, 50000)
        self.assertGreater(config.cpu_batch_size, 10000)
        
        # Check features
        self.assertTrue(config.use_tensor_cores)
        self.assertTrue(config.use_cuda_graphs)
        self.assertTrue(config.use_flash_attention)
    
    def test_config_from_mid_range_gpu(self):
        """Test configuration for mid-range GPU (12GB)"""
        gpu_specs = GPUSpecs(
            device_id=0,
            name="RTX 4070",
            total_memory_gb=12.0,
            total_memory_mb=12288.0,
            compute_capability="8.9",
            multi_processor_count=46,
            max_threads_per_block=1024,
            max_shared_memory_per_block=49152,
            warp_size=32,
            is_integrated=False,
            is_multi_gpu_board=False,
            cuda_cores=5888,
            memory_bandwidth_gb=504.2,
            architecture=GPUArchitecture.ADA,
            pcie_generation=4,
            memory_clock_mhz=21000.0,
            gpu_clock_mhz=2475.0,
            l2_cache_size_mb=36.0,
            tensor_cores=184,
            rt_cores=46
        )
        
        config = AutoOptimizedConfig.from_gpu_specs(gpu_specs)
        
        # Check memory thresholds
        self.assertEqual(config.memory_pressure_threshold, 0.88)
        self.assertLess(config.gpu_critical_threshold_mb, 10000)
        self.assertGreater(config.gpu_critical_threshold_mb, 8000)
        
        # Check burst multiplier
        self.assertEqual(config.burst_multiplier, 4.0)
    
    def test_config_from_low_end_gpu(self):
        """Test configuration for low-end GPU (8GB)"""
        gpu_specs = GPUSpecs(
            device_id=0,
            name="RTX 4060",
            total_memory_gb=8.0,
            total_memory_mb=8192.0,
            compute_capability="8.9",
            multi_processor_count=30,
            max_threads_per_block=1024,
            max_shared_memory_per_block=49152,
            warp_size=32,
            is_integrated=False,
            is_multi_gpu_board=False,
            cuda_cores=3840,
            memory_bandwidth_gb=272.0,
            architecture=GPUArchitecture.ADA,
            pcie_generation=4,
            memory_clock_mhz=17000.0,
            gpu_clock_mhz=2460.0,
            l2_cache_size_mb=24.0,
            tensor_cores=120,
            rt_cores=30
        )
        
        config = AutoOptimizedConfig.from_gpu_specs(gpu_specs)
        
        # Check memory thresholds
        self.assertEqual(config.memory_pressure_threshold, 0.85)
        self.assertLess(config.gpu_critical_threshold_mb, 6000)
        
        # Check burst multiplier (should be higher for low memory)
        self.assertEqual(config.burst_multiplier, 5.0)
        
        # Check memory pools (should be limited)
        self.assertEqual(config.memory_pool_count, 1)
    
    def test_config_serialization(self):
        """Test configuration serialization"""
        gpu_specs = GPUSpecs(
            device_id=0,
            name="Test GPU",
            total_memory_gb=16.0,
            total_memory_mb=16384.0,
            compute_capability="8.6",
            multi_processor_count=60,
            max_threads_per_block=1024,
            max_shared_memory_per_block=49152,
            warp_size=32,
            is_integrated=False,
            is_multi_gpu_board=False,
            cuda_cores=7680,
            memory_bandwidth_gb=600.0,
            architecture=GPUArchitecture.AMPERE,
            pcie_generation=4,
            memory_clock_mhz=19000.0,
            gpu_clock_mhz=1800.0,
            l2_cache_size_mb=4.0,
            tensor_cores=240,
            rt_cores=60
        )
        
        config = AutoOptimizedConfig.from_gpu_specs(gpu_specs)
        
        # Convert to dict
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        
        # Check key fields are present
        self.assertIn('gpu_memory_threshold_mb', config_dict)
        self.assertIn('memory_pressure_threshold', config_dict)
        self.assertIn('cpu_batch_size', config_dict)
        self.assertIn('use_tensor_cores', config_dict)


class TestConfigUpdater(unittest.TestCase):
    """Test configuration updater functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.updater = ConfigUpdater()
    
    def test_initialization(self):
        """Test updater initialization"""
        self.assertIsNotNone(self.updater.detector)
        self.assertIsNotNone(self.updater.optimized_config)
    
    def test_cpu_only_config(self):
        """Test CPU-only configuration"""
        with patch('torch.cuda.is_available', return_value=False):
            updater = ConfigUpdater()
            config = updater.optimized_config
            
            # Should have CPU-only settings
            self.assertEqual(config.gpu_memory_threshold_mb, 0)
            self.assertEqual(config.gpu_batch_size, 0)
            self.assertFalse(config.use_tensor_cores)
            self.assertFalse(config.use_cuda_graphs)
            self.assertGreater(config.cpu_batch_size, 0)
    
    def test_update_cpu_bursting_config(self):
        """Test updating CPU bursting configuration"""
        # Create mock config
        mock_config = MagicMock()
        
        # Update it
        updated = self.updater.update_cpu_bursting_config(mock_config)
        
        # Check attributes were set
        self.assertEqual(updated.gpu_memory_threshold_mb, 
                        self.updater.optimized_config.gpu_memory_threshold_mb)
        self.assertEqual(updated.memory_pressure_threshold,
                        self.updater.optimized_config.memory_pressure_threshold)
    
    def test_export_import_config(self):
        """Test configuration export and import"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name
        
        try:
            # Export configuration
            self.updater.export_config(config_file)
            
            # Check file exists
            self.assertTrue(os.path.exists(config_file))
            
            # Load configuration
            loaded_updater = ConfigUpdater.load_config(config_file)
            
            # Compare configurations
            self.assertEqual(
                self.updater.optimized_config.gpu_memory_threshold_mb,
                loaded_updater.optimized_config.gpu_memory_threshold_mb
            )
            
        finally:
            # Clean up
            if os.path.exists(config_file):
                os.unlink(config_file)
    
    def test_gpu_info_string(self):
        """Test GPU info string generation"""
        info = self.updater.get_gpu_info_string()
        self.assertIsInstance(info, str)
        
        if torch.cuda.is_available():
            self.assertIn("GPU:", info)
            self.assertIn("Memory:", info)
            self.assertIn("Architecture:", info)
        else:
            self.assertIn("No GPU detected", info)


class TestGlobalFunctions(unittest.TestCase):
    """Test global utility functions"""
    
    def test_singleton_detector(self):
        """Test singleton detector instance"""
        detector1 = get_gpu_detector()
        detector2 = get_gpu_detector()
        
        # Should be the same instance
        self.assertIs(detector1, detector2)
    
    def test_singleton_updater(self):
        """Test singleton updater instance"""
        updater1 = get_config_updater()
        updater2 = get_config_updater()
        
        # Should be the same instance
        self.assertIs(updater1, updater2)
    
    def test_auto_configure_system(self):
        """Test automatic system configuration"""
        config = auto_configure_system()
        
        self.assertIsInstance(config, dict)
        self.assertIn('gpu_memory_threshold_mb', config)
        self.assertIn('memory_pressure_threshold', config)
        self.assertIn('cpu_batch_size', config)
        
        # Values should be positive
        self.assertGreater(config['cpu_batch_size'], 0)
        self.assertGreater(config['num_cpu_workers'], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the auto-detection system"""
    
    def test_end_to_end_configuration(self):
        """Test end-to-end configuration flow"""
        # Detect GPU
        detector = GPUAutoDetector()
        gpu_specs = detector.get_primary_gpu()
        
        if gpu_specs:
            # Generate optimized config
            config = AutoOptimizedConfig.from_gpu_specs(gpu_specs)
            
            # Create updater
            updater = ConfigUpdater()
            
            # Create mock compression configs
            cpu_bursting_config = MagicMock()
            memory_pressure_config = MagicMock()
            gpu_memory_config = MagicMock()
            
            # Update all configs
            updater.update_cpu_bursting_config(cpu_bursting_config)
            updater.update_memory_pressure_config(memory_pressure_config)
            updater.update_gpu_memory_pool_config(gpu_memory_config)
            
            # Verify all were updated
            self.assertTrue(hasattr(cpu_bursting_config, 'gpu_memory_threshold_mb'))
            self.assertTrue(hasattr(memory_pressure_config, 'gpu_critical_threshold_mb'))
    
    def test_performance_benchmarks(self):
        """Test detection performance"""
        import time
        
        detector = GPUAutoDetector()
        
        # Time GPU detection
        start = time.time()
        gpus = detector.detect_all_gpus()
        detection_time = (time.time() - start) * 1000  # ms
        
        print(f"GPU detection time: {detection_time:.2f} ms")
        
        # Should complete quickly
        self.assertLess(detection_time, 100)  # < 100ms
        
        # Time config generation
        if gpus:
            gpu_specs = list(gpus.values())[0]
            start = time.time()
            config = AutoOptimizedConfig.from_gpu_specs(gpu_specs)
            config_time = (time.time() - start) * 1000
            
            print(f"Config generation time: {config_time:.2f} ms")
            self.assertLess(config_time, 10)  # < 10ms


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)