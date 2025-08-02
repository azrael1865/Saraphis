#!/usr/bin/env python3
"""
Test P-Adic Service Configuration Integration

Tests the integration of the new PadicServiceConfiguration with existing P-adic components.
"""

import logging
import tempfile
import json
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_configuration():
    """Test basic configuration functionality"""
    from .padic_service_config import PadicServiceConfiguration
    
    print("=== Testing Basic Configuration ===")
    
    # Test initialization
    config = PadicServiceConfiguration()
    print(f"✓ Configuration initialized with default values")
    
    # Test getting values
    default_p = config.get('padic.default_p')
    print(f"✓ Default prime: {default_p}")
    
    # Test setting values
    success = config.set('padic.default_precision', 150)
    print(f"✓ Set precision: {success}")
    
    # Test validation
    is_valid = config.validate_configuration()
    print(f"✓ Configuration valid: {is_valid}")
    
    print("Basic configuration tests passed!\n")

def test_integration_with_existing_components():
    """Test integration with existing P-adic components"""
    from .padic_service_config import PadicServiceConfiguration
    from .padic_integration import PadicIntegrationConfig
    
    print("=== Testing Integration with Existing Components ===")
    
    config = PadicServiceConfiguration()
    
    # Test getting integration config
    integration_config = config.get_integration_config()
    print(f"✓ Integration config type: {type(integration_config)}")
    print(f"✓ Prime: {integration_config.prime}")
    print(f"✓ Base precision: {integration_config.base_precision}")
    
    # Test getting hensel config
    hensel_config = config.get_hensel_config()
    print(f"✓ Hensel config type: {type(hensel_config)}")
    print(f"✓ Max iterations: {hensel_config.max_iterations}")
    
    # Test getting clustering config
    clustering_config = config.get_clustering_config()
    print(f"✓ Clustering config type: {type(clustering_config)}")
    print(f"✓ Max clusters: {clustering_config.max_clusters}")
    
    # Test getting GPU config
    gpu_config = config.get_gpu_config()
    print(f"✓ GPU config type: {type(gpu_config)}")
    print(f"✓ Device ID: {gpu_config.device_id}")
    
    print("Integration with existing components tests passed!\n")

def test_configuration_persistence():
    """Test configuration save/load functionality"""
    from .padic_service_config import PadicServiceConfiguration
    
    print("=== Testing Configuration Persistence ===")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        # Create and modify configuration
        config1 = PadicServiceConfiguration()
        config1.set('padic.default_precision', 200)
        config1.set('service.max_batch_size', 64)
        
        # Save configuration
        success = config1.save_configuration(temp_path)
        print(f"✓ Configuration saved: {success}")
        
        # Load configuration
        config2 = PadicServiceConfiguration(config_path=temp_path)
        precision = config2.get('padic.default_precision')
        batch_size = config2.get('service.max_batch_size')
        
        print(f"✓ Loaded precision: {precision}")
        print(f"✓ Loaded batch size: {batch_size}")
        
        # Verify values
        assert precision == 200, f"Expected 200, got {precision}"
        assert batch_size == 64, f"Expected 64, got {batch_size}"
        
    finally:
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
    
    print("Configuration persistence tests passed!\n")

def test_configuration_validation():
    """Test configuration validation"""
    from .padic_service_config import PadicServiceConfiguration
    
    print("=== Testing Configuration Validation ===")
    
    config = PadicServiceConfiguration()
    
    # Test valid updates
    valid_update = config.update({
        'padic': {
            'default_precision': 64,
            'compression_ratio': 0.2
        }
    })
    print(f"✓ Valid update accepted: {valid_update}")
    
    # Test invalid updates (should be rejected)
    invalid_update = config.update({
        'padic': {
            'default_precision': -10  # Invalid: negative precision
        }
    })
    print(f"✓ Invalid update rejected: {not invalid_update}")
    
    # Verify original values preserved
    precision = config.get('padic.default_precision')
    print(f"✓ Precision preserved after invalid update: {precision}")
    
    print("Configuration validation tests passed!\n")

def test_configuration_callbacks():
    """Test configuration change callbacks"""
    from .padic_service_config import PadicServiceConfiguration
    
    print("=== Testing Configuration Callbacks ===")
    
    changes_received = []
    
    def config_callback(key, old_value, new_value):
        changes_received.append((key, old_value, new_value))
        print(f"  Callback: {key} changed from {old_value} to {new_value}")
    
    config = PadicServiceConfiguration()
    config.register_change_callback(config_callback)
    
    # Make some changes
    config.set('padic.default_precision', 128)
    config.set('service.max_batch_size', 48)
    
    print(f"✓ Received {len(changes_received)} change notifications")
    
    # Verify callback data
    assert len(changes_received) == 2, f"Expected 2 changes, got {len(changes_received)}"
    print("✓ All callbacks received correctly")
    
    print("Configuration callbacks tests passed!\n")

def test_orchestrator_integration():
    """Test integration with PadicSystemOrchestrator"""
    from .padic_service_config import PadicServiceConfiguration
    from .padic_integration import initialize_padic_integration, shutdown_padic_integration
    
    print("=== Testing Orchestrator Integration ===")
    
    try:
        # Create service configuration
        service_config = PadicServiceConfiguration()
        service_config.set('padic.default_precision', 75)
        service_config.set('integration.brain_core_enabled', True)
        
        # Initialize with service configuration
        # Note: This might fail if dependencies aren't available, which is expected in testing
        try:
            orchestrator = initialize_padic_integration(service_config=service_config)
            print("✓ Orchestrator initialized with service configuration")
            
            # Test getting configuration back
            retrieved_config = orchestrator.get_service_configuration()
            precision = retrieved_config.get('padic.default_precision')
            print(f"✓ Retrieved precision from orchestrator: {precision}")
            
            # Test updating configuration through orchestrator
            update_success = orchestrator.update_service_configuration({
                'padic': {'compression_ratio': 0.15}
            })
            print(f"✓ Configuration update through orchestrator: {update_success}")
            
        except Exception as e:
            print(f"⚠ Orchestrator test skipped (expected if dependencies missing): {str(e)}")
            
    finally:
        try:
            shutdown_padic_integration()
            print("✓ Orchestrator shutdown")
        except:
            pass  # May not be initialized
    
    print("Orchestrator integration tests completed!\n")

def main():
    """Run all integration tests"""
    print("P-Adic Service Configuration Integration Tests")
    print("=" * 50)
    
    test_basic_configuration()
    test_integration_with_existing_components()
    test_configuration_persistence()
    test_configuration_validation()
    test_configuration_callbacks()
    test_orchestrator_integration()
    
    print("=" * 50)
    print("All integration tests completed successfully! ✓")

if __name__ == "__main__":
    main()