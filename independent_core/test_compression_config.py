#!/usr/bin/env python3
"""
Comprehensive unit tests for CompressionConfig, ConfigurationManager, and ConfigurationValidator.
Tests all validation logic, edge cases, and error handling.
"""

import unittest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from compression_systems.services.compression_config import (
    CompressionConfig,
    ConfigurationManager,
    ConfigurationValidator
)


class TestCompressionConfig(unittest.TestCase):
    """Test CompressionConfig dataclass and validation"""
    
    def setUp(self):
        """Set up valid default configuration for tests"""
        self.valid_config_params = {
            'padic_base': 7,  # Prime number
            'padic_precision': 32,
            'padic_adaptive': True,
            'sheaf_locality_radius': 5,
            'sheaf_cohomology_dim': 3,
            'sheaf_validate_invariants': True,
            'tensor_decomp_method': 'tucker',
            'tensor_rank_ratio': 0.5,
            'tensor_adaptive_rank': True,
            'gpu_memory_threshold': 0.8,
            'gpu_cache_size_mb': 1024,
            'gpu_prefetch_enabled': True,
            'max_concurrent_requests': 100,
            'request_timeout_seconds': 30,
            'max_requests_per_module': 50,
            'max_compression_time_ms': 100.0,
            'min_compression_ratio': 2.0,
            'max_reconstruction_error': 0.001
        }
    
    def test_valid_configuration_creation(self):
        """Test creating a valid configuration"""
        config = CompressionConfig(**self.valid_config_params)
        self.assertIsInstance(config, CompressionConfig)
        self.assertEqual(config.padic_base, 7)
        self.assertEqual(config.tensor_decomp_method, 'tucker')
    
    def test_padic_base_validation_non_prime(self):
        """Test that non-prime p-adic base raises error"""
        invalid_params = self.valid_config_params.copy()
        invalid_params['padic_base'] = 4  # Not prime
        
        with self.assertRaises(ValueError) as context:
            CompressionConfig(**invalid_params)
        self.assertIn("P-adic base must be prime", str(context.exception))
    
    def test_padic_base_validation_too_small(self):
        """Test that p-adic base < 2 raises error"""
        invalid_params = self.valid_config_params.copy()
        invalid_params['padic_base'] = 1
        
        with self.assertRaises(ValueError) as context:
            CompressionConfig(**invalid_params)
        self.assertIn("P-adic base must be >= 2", str(context.exception))
    
    def test_padic_precision_validation_negative(self):
        """Test that negative precision raises error"""
        invalid_params = self.valid_config_params.copy()
        invalid_params['padic_precision'] = -1
        
        with self.assertRaises(ValueError) as context:
            CompressionConfig(**invalid_params)
        self.assertIn("P-adic precision must be > 0", str(context.exception))
    
    def test_padic_precision_validation_too_high(self):
        """Test that precision > 128 raises error"""
        invalid_params = self.valid_config_params.copy()
        invalid_params['padic_precision'] = 129
        
        with self.assertRaises(ValueError) as context:
            CompressionConfig(**invalid_params)
        self.assertIn("P-adic precision too high", str(context.exception))
    
    def test_sheaf_locality_radius_validation(self):
        """Test sheaf locality radius validation"""
        invalid_params = self.valid_config_params.copy()
        invalid_params['sheaf_locality_radius'] = 0
        
        with self.assertRaises(ValueError) as context:
            CompressionConfig(**invalid_params)
        self.assertIn("Locality radius must be > 0", str(context.exception))
    
    def test_sheaf_cohomology_dim_validation_negative(self):
        """Test sheaf cohomology dimension validation"""
        invalid_params = self.valid_config_params.copy()
        invalid_params['sheaf_cohomology_dim'] = 0
        
        with self.assertRaises(ValueError) as context:
            CompressionConfig(**invalid_params)
        self.assertIn("Cohomology dimension must be > 0", str(context.exception))
    
    def test_sheaf_cohomology_dim_validation_too_high(self):
        """Test sheaf cohomology dimension upper bound"""
        invalid_params = self.valid_config_params.copy()
        invalid_params['sheaf_cohomology_dim'] = 11
        
        with self.assertRaises(ValueError) as context:
            CompressionConfig(**invalid_params)
        self.assertIn("Cohomology dimension too high", str(context.exception))
    
    def test_tensor_method_validation(self):
        """Test tensor decomposition method validation"""
        invalid_params = self.valid_config_params.copy()
        invalid_params['tensor_decomp_method'] = 'invalid_method'
        
        with self.assertRaises(ValueError) as context:
            CompressionConfig(**invalid_params)
        self.assertIn("Invalid tensor method", str(context.exception))
    
    def test_tensor_rank_ratio_validation_bounds(self):
        """Test tensor rank ratio bounds validation"""
        # Test lower bound
        invalid_params = self.valid_config_params.copy()
        invalid_params['tensor_rank_ratio'] = 0.0
        
        with self.assertRaises(ValueError) as context:
            CompressionConfig(**invalid_params)
        self.assertIn("Rank ratio must be in (0,1)", str(context.exception))
        
        # Test upper bound
        invalid_params['tensor_rank_ratio'] = 1.0
        
        with self.assertRaises(ValueError) as context:
            CompressionConfig(**invalid_params)
        self.assertIn("Rank ratio must be in (0,1)", str(context.exception))
    
    def test_gpu_memory_threshold_validation(self):
        """Test GPU memory threshold validation"""
        # Test lower bound
        invalid_params = self.valid_config_params.copy()
        invalid_params['gpu_memory_threshold'] = 0.0
        
        with self.assertRaises(ValueError) as context:
            CompressionConfig(**invalid_params)
        self.assertIn("GPU memory threshold must be in (0,1)", str(context.exception))
        
        # Test upper bound
        invalid_params['gpu_memory_threshold'] = 1.0
        
        with self.assertRaises(ValueError) as context:
            CompressionConfig(**invalid_params)
        self.assertIn("GPU memory threshold must be in (0,1)", str(context.exception))
    
    def test_gpu_cache_size_validation(self):
        """Test GPU cache size validation"""
        invalid_params = self.valid_config_params.copy()
        invalid_params['gpu_cache_size_mb'] = 0
        
        with self.assertRaises(ValueError) as context:
            CompressionConfig(**invalid_params)
        self.assertIn("GPU cache size must be > 0", str(context.exception))
    
    def test_service_config_validation(self):
        """Test service configuration validation"""
        # Test max_concurrent_requests
        invalid_params = self.valid_config_params.copy()
        invalid_params['max_concurrent_requests'] = 0
        
        with self.assertRaises(ValueError) as context:
            CompressionConfig(**invalid_params)
        self.assertIn("Max concurrent requests must be > 0", str(context.exception))
        
        # Test request_timeout_seconds
        invalid_params = self.valid_config_params.copy()
        invalid_params['request_timeout_seconds'] = 0
        
        with self.assertRaises(ValueError) as context:
            CompressionConfig(**invalid_params)
        self.assertIn("Request timeout must be > 0", str(context.exception))
        
        # Test max_requests_per_module
        invalid_params = self.valid_config_params.copy()
        invalid_params['max_requests_per_module'] = 0
        
        with self.assertRaises(ValueError) as context:
            CompressionConfig(**invalid_params)
        self.assertIn("Max requests per module must be > 0", str(context.exception))
    
    def test_performance_config_validation(self):
        """Test performance configuration validation"""
        # Test max_compression_time_ms
        invalid_params = self.valid_config_params.copy()
        invalid_params['max_compression_time_ms'] = 0
        
        with self.assertRaises(ValueError) as context:
            CompressionConfig(**invalid_params)
        self.assertIn("Max compression time must be > 0", str(context.exception))
        
        # Test min_compression_ratio
        invalid_params = self.valid_config_params.copy()
        invalid_params['min_compression_ratio'] = 1.0
        
        with self.assertRaises(ValueError) as context:
            CompressionConfig(**invalid_params)
        self.assertIn("Min compression ratio must be > 1", str(context.exception))
        
        # Test max_reconstruction_error
        invalid_params = self.valid_config_params.copy()
        invalid_params['max_reconstruction_error'] = 0
        
        with self.assertRaises(ValueError) as context:
            CompressionConfig(**invalid_params)
        self.assertIn("Max reconstruction error must be > 0", str(context.exception))
    
    def test_is_prime_method(self):
        """Test the _is_prime static method"""
        # Test prime numbers
        self.assertTrue(CompressionConfig._is_prime(2))
        self.assertTrue(CompressionConfig._is_prime(3))
        self.assertTrue(CompressionConfig._is_prime(5))
        self.assertTrue(CompressionConfig._is_prime(7))
        self.assertTrue(CompressionConfig._is_prime(11))
        self.assertTrue(CompressionConfig._is_prime(13))
        
        # Test non-prime numbers
        self.assertFalse(CompressionConfig._is_prime(1))
        self.assertFalse(CompressionConfig._is_prime(4))
        self.assertFalse(CompressionConfig._is_prime(6))
        self.assertFalse(CompressionConfig._is_prime(8))
        self.assertFalse(CompressionConfig._is_prime(9))
        self.assertFalse(CompressionConfig._is_prime(10))
    
    def test_all_tensor_methods(self):
        """Test all valid tensor decomposition methods"""
        valid_methods = ['tucker', 'cp', 'tensor_train']
        
        for method in valid_methods:
            params = self.valid_config_params.copy()
            params['tensor_decomp_method'] = method
            config = CompressionConfig(**params)
            self.assertEqual(config.tensor_decomp_method, method)


class TestConfigurationManager(unittest.TestCase):
    """Test ConfigurationManager class"""
    
    def setUp(self):
        """Set up test configuration manager"""
        self.base_config_params = {
            'padic_base': 7,
            'padic_precision': 32,
            'padic_adaptive': True,
            'sheaf_locality_radius': 5,
            'sheaf_cohomology_dim': 3,
            'sheaf_validate_invariants': True,
            'tensor_decomp_method': 'tucker',
            'tensor_rank_ratio': 0.5,
            'tensor_adaptive_rank': True,
            'gpu_memory_threshold': 0.8,
            'gpu_cache_size_mb': 1024,
            'gpu_prefetch_enabled': True,
            'max_concurrent_requests': 100,
            'request_timeout_seconds': 30,
            'max_requests_per_module': 50,
            'max_compression_time_ms': 100.0,
            'min_compression_ratio': 2.0,
            'max_reconstruction_error': 0.001
        }
        self.base_config = CompressionConfig(**self.base_config_params)
        self.manager = ConfigurationManager(self.base_config)
    
    def test_manager_initialization(self):
        """Test ConfigurationManager initialization"""
        self.assertIsInstance(self.manager, ConfigurationManager)
        self.assertEqual(self.manager.base_config, self.base_config)
        self.assertEqual(len(self.manager._module_overrides), 0)
        self.assertEqual(len(self.manager._service_configs), 0)
    
    def test_manager_initialization_with_invalid_type(self):
        """Test ConfigurationManager with invalid base config type"""
        with self.assertRaises(TypeError) as context:
            ConfigurationManager("not a config")
        self.assertIn("Base config must be CompressionConfig instance", str(context.exception))
    
    def test_add_module_override(self):
        """Test adding module-specific overrides"""
        overrides = {'padic_precision': 64, 'gpu_cache_size_mb': 2048}
        self.manager.add_module_override('test_module', overrides)
        
        self.assertIn('test_module', self.manager._module_overrides)
        self.assertEqual(self.manager._module_overrides['test_module'], overrides)
    
    def test_add_module_override_empty_name(self):
        """Test adding override with empty module name"""
        with self.assertRaises(ValueError) as context:
            self.manager.add_module_override('', {'padic_precision': 64})
        self.assertIn("Module name cannot be empty", str(context.exception))
    
    def test_add_module_override_invalid_type(self):
        """Test adding override with invalid type"""
        with self.assertRaises(TypeError) as context:
            self.manager.add_module_override('test_module', "not a dict")
        self.assertIn("Overrides must be dict", str(context.exception))
    
    def test_add_module_override_invalid_key(self):
        """Test adding override with invalid configuration key"""
        with self.assertRaises(KeyError) as context:
            self.manager.add_module_override('test_module', {'invalid_key': 123})
        self.assertIn("Invalid override key: invalid_key", str(context.exception))
    
    def test_get_config_for_module(self):
        """Test getting configuration with module overrides"""
        # Add overrides
        overrides = {'padic_precision': 64, 'gpu_cache_size_mb': 2048}
        self.manager.add_module_override('test_module', overrides)
        
        # Get config with overrides
        module_config = self.manager.get_config_for_module('test_module')
        
        # Check overrides were applied
        self.assertEqual(module_config.padic_precision, 64)
        self.assertEqual(module_config.gpu_cache_size_mb, 2048)
        
        # Check other values unchanged
        self.assertEqual(module_config.padic_base, self.base_config.padic_base)
        self.assertEqual(module_config.tensor_decomp_method, self.base_config.tensor_decomp_method)
    
    def test_get_config_for_module_no_overrides(self):
        """Test getting configuration for module without overrides"""
        module_config = self.manager.get_config_for_module('unknown_module')
        
        # Should return config equal to base config
        self.assertEqual(module_config.padic_precision, self.base_config.padic_precision)
        self.assertEqual(module_config.gpu_cache_size_mb, self.base_config.gpu_cache_size_mb)
    
    def test_get_config_for_module_empty_name(self):
        """Test getting config with empty module name"""
        with self.assertRaises(ValueError) as context:
            self.manager.get_config_for_module('')
        self.assertIn("Module name cannot be empty", str(context.exception))
    
    def test_register_service_config(self):
        """Test registering service-specific configuration"""
        service_config = CompressionConfig(**self.base_config_params)
        self.manager.register_service_config('test_service', service_config)
        
        self.assertIn('test_service', self.manager._service_configs)
        self.assertEqual(self.manager._service_configs['test_service'], service_config)
    
    def test_register_service_config_empty_name(self):
        """Test registering service config with empty name"""
        service_config = CompressionConfig(**self.base_config_params)
        
        with self.assertRaises(ValueError) as context:
            self.manager.register_service_config('', service_config)
        self.assertIn("Service name cannot be empty", str(context.exception))
    
    def test_register_service_config_invalid_type(self):
        """Test registering service config with invalid type"""
        with self.assertRaises(TypeError) as context:
            self.manager.register_service_config('test_service', "not a config")
        self.assertIn("Config must be CompressionConfig instance", str(context.exception))
    
    def test_get_config_for_service(self):
        """Test getting service-specific configuration"""
        service_config = CompressionConfig(**self.base_config_params)
        self.manager.register_service_config('test_service', service_config)
        
        retrieved_config = self.manager.get_config_for_service('test_service')
        self.assertEqual(retrieved_config, service_config)
    
    def test_get_config_for_service_not_found(self):
        """Test getting config for non-existent service"""
        with self.assertRaises(KeyError) as context:
            self.manager.get_config_for_service('unknown_service')
        self.assertIn("No config found for service: unknown_service", str(context.exception))
    
    def test_validate_all_configs(self):
        """Test validating all registered configurations"""
        # Add valid module override
        self.manager.add_module_override('module1', {'padic_precision': 64})
        
        # Add valid service config
        service_config = CompressionConfig(**self.base_config_params)
        self.manager.register_service_config('service1', service_config)
        
        # Validate all
        results = self.manager.validate_all_configs()
        
        self.assertTrue(results['base_config'])
        self.assertTrue(results['module_module1'])
        self.assertTrue(results['service_service1'])
    
    def test_validate_all_configs_with_invalid_override(self):
        """Test validation with invalid module override"""
        # Add override that would make config invalid
        self.manager.add_module_override('bad_module', {'padic_base': 4})  # Non-prime
        
        with self.assertRaises(ValueError) as context:
            self.manager.validate_all_configs()
        self.assertIn("Module bad_module config validation failed", str(context.exception))
    
    def test_export_config(self):
        """Test exporting configuration to file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_config.json'
            
            # Add some overrides
            self.manager.add_module_override('test_module', {'padic_precision': 64})
            
            # Export config
            self.manager.export_config('test_module', filepath)
            
            # Verify file exists and contains correct data
            self.assertTrue(filepath.exists())
            
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            self.assertEqual(config_dict['padic_precision'], 64)
            self.assertEqual(config_dict['padic_base'], self.base_config.padic_base)
    
    def test_load_config_from_file(self):
        """Test loading configuration from JSON file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_config.json'
            
            # Write config to file
            with open(filepath, 'w') as f:
                json.dump(self.base_config_params, f)
            
            # Load config
            loaded_config = ConfigurationManager.load_config_from_file(filepath)
            
            self.assertIsInstance(loaded_config, CompressionConfig)
            self.assertEqual(loaded_config.padic_base, self.base_config_params['padic_base'])
            self.assertEqual(loaded_config.tensor_decomp_method, self.base_config_params['tensor_decomp_method'])
    
    def test_load_config_from_file_not_found(self):
        """Test loading config from non-existent file"""
        with self.assertRaises(FileNotFoundError) as context:
            ConfigurationManager.load_config_from_file(Path('/nonexistent/config.json'))
        self.assertIn("Config file not found", str(context.exception))
    
    def test_get_configuration_summary(self):
        """Test getting configuration summary"""
        # Add module override
        self.manager.add_module_override('module1', {'padic_precision': 64})
        
        # Add service config
        service_config = CompressionConfig(**self.base_config_params)
        self.manager.register_service_config('service1', service_config)
        
        # Get summary
        summary = self.manager.get_configuration_summary()
        
        self.assertIn('base_config', summary)
        self.assertIn('module_overrides', summary)
        self.assertIn('service_configs', summary)
        self.assertEqual(summary['total_modules'], 1)
        self.assertEqual(summary['total_services'], 1)
        self.assertEqual(summary['module_overrides']['module1'], {'padic_precision': 64})


class TestConfigurationValidator(unittest.TestCase):
    """Test ConfigurationValidator class"""
    
    def setUp(self):
        """Set up test configurations"""
        self.base_params = {
            'padic_base': 7,
            'padic_precision': 32,
            'padic_adaptive': True,
            'sheaf_locality_radius': 5,
            'sheaf_cohomology_dim': 3,
            'sheaf_validate_invariants': True,
            'tensor_decomp_method': 'tucker',
            'tensor_rank_ratio': 0.5,
            'tensor_adaptive_rank': True,
            'gpu_memory_threshold': 0.8,
            'gpu_cache_size_mb': 1024,
            'gpu_prefetch_enabled': True,
            'max_concurrent_requests': 100,
            'request_timeout_seconds': 30,
            'max_requests_per_module': 50,
            'max_compression_time_ms': 100.0,
            'min_compression_ratio': 2.0,
            'max_reconstruction_error': 0.001
        }
    
    def test_validate_system_compatibility_compatible(self):
        """Test validation of compatible configurations"""
        # Create multiple compatible configs
        config1 = CompressionConfig(**self.base_params)
        
        params2 = self.base_params.copy()
        params2['gpu_cache_size_mb'] = 512
        config2 = CompressionConfig(**params2)
        
        # Should not raise any errors
        ConfigurationValidator.validate_system_compatibility([config1, config2])
    
    def test_validate_system_compatibility_empty_list(self):
        """Test validation with empty configuration list"""
        with self.assertRaises(ValueError) as context:
            ConfigurationValidator.validate_system_compatibility([])
        self.assertIn("No configurations provided", str(context.exception))
    
    def test_validate_system_compatibility_incompatible_bases(self):
        """Test validation with incompatible P-adic bases"""
        config1 = CompressionConfig(**self.base_params)
        
        params2 = self.base_params.copy()
        params2['padic_base'] = 11  # Different prime
        config2 = CompressionConfig(**params2)
        
        with self.assertRaises(ValueError) as context:
            ConfigurationValidator.validate_system_compatibility([config1, config2])
        self.assertIn("Incompatible P-adic bases", str(context.exception))
    
    def test_validate_system_compatibility_gpu_memory_exceeded(self):
        """Test validation with GPU memory limit exceeded"""
        configs = []
        
        # Create configs that exceed GPU memory limit
        for i in range(10):
            params = self.base_params.copy()
            params['gpu_cache_size_mb'] = 1024  # 1GB each
            configs.append(CompressionConfig(**params))
        
        with self.assertRaises(ValueError) as context:
            ConfigurationValidator.validate_system_compatibility(configs)
        self.assertIn("Total GPU cache exceeds limit", str(context.exception))
    
    def test_validate_system_compatibility_concurrent_requests_exceeded(self):
        """Test validation with concurrent requests limit exceeded"""
        configs = []
        
        # Create configs that exceed concurrent request limit
        # Use small GPU cache to avoid hitting GPU memory limit first
        for i in range(110):
            params = self.base_params.copy()
            params['max_concurrent_requests'] = 100
            params['gpu_cache_size_mb'] = 10  # Small cache to avoid GPU limit
            configs.append(CompressionConfig(**params))
        
        with self.assertRaises(ValueError) as context:
            ConfigurationValidator.validate_system_compatibility(configs)
        self.assertIn("Total concurrent requests too high", str(context.exception))
    
    def test_validate_performance_feasibility_valid(self):
        """Test performance feasibility validation with valid sizes"""
        config = CompressionConfig(**self.base_params)
        
        # Small data sizes that should be feasible
        expected_sizes = [1000, 5000, 10000]
        
        # Should not raise any errors
        ConfigurationValidator.validate_performance_feasibility(config, expected_sizes)
    
    def test_validate_performance_feasibility_infeasible(self):
        """Test performance feasibility validation with infeasible sizes"""
        params = self.base_params.copy()
        params['max_compression_time_ms'] = 1.0  # Very strict limit
        config = CompressionConfig(**params)
        
        # Large data size that would be infeasible
        expected_sizes = [1000000]  # Would require ~1000ms at 1 microsecond per element
        
        with self.assertRaises(ValueError) as context:
            ConfigurationValidator.validate_performance_feasibility(config, expected_sizes)
        self.assertIn("Performance requirement infeasible", str(context.exception))
        self.assertIn("1000000", str(context.exception))


class TestEdgeCasesAndIntegration(unittest.TestCase):
    """Test edge cases and integration scenarios"""
    
    def test_boundary_prime_values(self):
        """Test boundary cases for prime validation"""
        base_params = {
            'padic_base': 2,  # Smallest prime
            'padic_precision': 1,  # Minimum precision
            'padic_adaptive': False,
            'sheaf_locality_radius': 1,
            'sheaf_cohomology_dim': 1,
            'sheaf_validate_invariants': False,
            'tensor_decomp_method': 'cp',
            'tensor_rank_ratio': 0.001,  # Near lower bound
            'tensor_adaptive_rank': False,
            'gpu_memory_threshold': 0.001,  # Near lower bound
            'gpu_cache_size_mb': 1,  # Minimum
            'gpu_prefetch_enabled': False,
            'max_concurrent_requests': 1,
            'request_timeout_seconds': 1,
            'max_requests_per_module': 1,
            'max_compression_time_ms': 0.001,
            'min_compression_ratio': 1.001,
            'max_reconstruction_error': 0.000001
        }
        
        # Should create successfully
        config = CompressionConfig(**base_params)
        self.assertEqual(config.padic_base, 2)
    
    def test_large_prime_values(self):
        """Test with larger prime numbers"""
        large_primes = [97, 101, 103, 107, 109, 113]
        
        for prime in large_primes:
            params = {
                'padic_base': prime,
                'padic_precision': 128,  # Maximum precision
                'padic_adaptive': True,
                'sheaf_locality_radius': 100,
                'sheaf_cohomology_dim': 10,  # Maximum
                'sheaf_validate_invariants': True,
                'tensor_decomp_method': 'tensor_train',
                'tensor_rank_ratio': 0.999,  # Near upper bound
                'tensor_adaptive_rank': True,
                'gpu_memory_threshold': 0.999,  # Near upper bound
                'gpu_cache_size_mb': 8192,
                'gpu_prefetch_enabled': True,
                'max_concurrent_requests': 10000,
                'request_timeout_seconds': 3600,
                'max_requests_per_module': 1000,
                'max_compression_time_ms': 10000.0,
                'min_compression_ratio': 100.0,
                'max_reconstruction_error': 1.0
            }
            
            config = CompressionConfig(**params)
            self.assertEqual(config.padic_base, prime)
    
    def test_configuration_immutability(self):
        """Test that configuration is properly handled after creation"""
        params = {
            'padic_base': 7,
            'padic_precision': 32,
            'padic_adaptive': True,
            'sheaf_locality_radius': 5,
            'sheaf_cohomology_dim': 3,
            'sheaf_validate_invariants': True,
            'tensor_decomp_method': 'tucker',
            'tensor_rank_ratio': 0.5,
            'tensor_adaptive_rank': True,
            'gpu_memory_threshold': 0.8,
            'gpu_cache_size_mb': 1024,
            'gpu_prefetch_enabled': True,
            'max_concurrent_requests': 100,
            'request_timeout_seconds': 30,
            'max_requests_per_module': 50,
            'max_compression_time_ms': 100.0,
            'min_compression_ratio': 2.0,
            'max_reconstruction_error': 0.001
        }
        
        config = CompressionConfig(**params)
        
        # Modify attribute directly
        config.padic_base = 4  # Non-prime
        
        # Re-validation should catch the invalid value
        with self.assertRaises(ValueError) as context:
            config.__post_init__()
        self.assertIn("P-adic base must be prime", str(context.exception))
    
    def test_manager_with_multiple_modules_and_services(self):
        """Test ConfigurationManager with multiple modules and services"""
        base_params = {
            'padic_base': 7,
            'padic_precision': 32,
            'padic_adaptive': True,
            'sheaf_locality_radius': 5,
            'sheaf_cohomology_dim': 3,
            'sheaf_validate_invariants': True,
            'tensor_decomp_method': 'tucker',
            'tensor_rank_ratio': 0.5,
            'tensor_adaptive_rank': True,
            'gpu_memory_threshold': 0.8,
            'gpu_cache_size_mb': 1024,
            'gpu_prefetch_enabled': True,
            'max_concurrent_requests': 100,
            'request_timeout_seconds': 30,
            'max_requests_per_module': 50,
            'max_compression_time_ms': 100.0,
            'min_compression_ratio': 2.0,
            'max_reconstruction_error': 0.001
        }
        
        base_config = CompressionConfig(**base_params)
        manager = ConfigurationManager(base_config)
        
        # Add multiple module overrides
        for i in range(5):
            overrides = {
                'padic_precision': 32 + i * 8,
                'gpu_cache_size_mb': 512 + i * 256
            }
            manager.add_module_override(f'module_{i}', overrides)
        
        # Add multiple service configs
        for i in range(3):
            params = base_params.copy()
            params['max_concurrent_requests'] = 50 + i * 25
            service_config = CompressionConfig(**params)
            manager.register_service_config(f'service_{i}', service_config)
        
        # Validate all configurations
        results = manager.validate_all_configs()
        
        # Check all validations passed
        self.assertTrue(all(results.values()))
        self.assertEqual(len([k for k in results if k.startswith('module_')]), 5)
        self.assertEqual(len([k for k in results if k.startswith('service_')]), 3)
    
    def test_json_serialization_deserialization(self):
        """Test JSON serialization and deserialization cycle"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_config.json'
            
            # Create original config
            original_params = {
                'padic_base': 13,
                'padic_precision': 48,
                'padic_adaptive': False,
                'sheaf_locality_radius': 7,
                'sheaf_cohomology_dim': 4,
                'sheaf_validate_invariants': False,
                'tensor_decomp_method': 'cp',
                'tensor_rank_ratio': 0.3,
                'tensor_adaptive_rank': False,
                'gpu_memory_threshold': 0.6,
                'gpu_cache_size_mb': 2048,
                'gpu_prefetch_enabled': False,
                'max_concurrent_requests': 200,
                'request_timeout_seconds': 60,
                'max_requests_per_module': 100,
                'max_compression_time_ms': 200.0,
                'min_compression_ratio': 3.0,
                'max_reconstruction_error': 0.002
            }
            
            original_config = CompressionConfig(**original_params)
            manager = ConfigurationManager(original_config)
            
            # Export to JSON
            manager.export_config('test', filepath)
            
            # Load from JSON
            loaded_config = ConfigurationManager.load_config_from_file(filepath)
            
            # Compare all fields
            for key, value in original_params.items():
                self.assertEqual(getattr(loaded_config, key), value)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)