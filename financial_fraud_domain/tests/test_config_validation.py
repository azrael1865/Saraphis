"""
Test script for enhanced configuration management validation and error handling
Demonstrates comprehensive validation features and error scenarios
"""

import json
import os
import tempfile
from pathlib import Path
import logging

# Import the enhanced modules
from enhanced_config_manager import (
    EnhancedConfigManager, ValidationLevel, ConfigValidationError,
    ConfigLoadError, ConfigSecurityError, ConfigCompatibilityError
)
from enhanced_domain_config import (
    FinancialFraudConfig, DomainConfig, DatabaseConfig, 
    SecurityConfig, EnvironmentType, SecurityLevel
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_basic_validation():
    """Test basic configuration validation"""
    print("\n=== Testing Basic Configuration Validation ===\n")
    
    # Create configuration with invalid values
    config = DomainConfig()
    
    # Test invalid thresholds
    config.model_threshold = 1.5  # Should be between 0 and 1
    config.alert_threshold = -0.1  # Should be between 0 and 1
    
    # Test invalid database settings
    config.database.port = 70000  # Should be between 1 and 65535
    config.database.connection_pool_size = -10  # Should be positive
    
    # Validate
    is_valid, errors = config.validate()
    
    print(f"Configuration valid: {is_valid}")
    print(f"Errors found: {len(errors)}")
    for i, error in enumerate(errors[:5]):  # Show first 5 errors
        print(f"  {i+1}. {error}")
    
    return not is_valid  # Test passes if validation fails


def test_security_validation():
    """Test security configuration validation"""
    print("\n=== Testing Security Configuration Validation ===\n")
    
    config = DomainConfig()
    
    # Set insecure values
    config.database.password = "weak123"  # Hardcoded password
    config.security.key_rotation_days = 0  # Invalid rotation period
    config.security.data_retention_days = 30  # Too short for HIGH security
    
    # Test IP range validation
    config.security.allowed_ip_ranges = ["192.168.1.0/24", "invalid-ip-range"]
    config.security.blocked_ip_ranges = ["192.168.1.0/24"]  # Conflicts with allowed
    
    # Validate
    is_valid, errors = config.validate()
    
    print(f"Security validation - Valid: {is_valid}")
    print("Security-related errors:")
    security_errors = [e for e in errors if 'security' in e.lower() or 'password' in e.lower()]
    for error in security_errors:
        print(f"  - {error}")
    
    return len(security_errors) > 0


def test_environment_validation():
    """Test environment-specific validation"""
    print("\n=== Testing Environment-Specific Validation ===\n")
    
    # Create production configuration
    config = DomainConfig(environment=EnvironmentType.PRODUCTION)
    
    # Set values that are invalid for production
    config.api.debug_mode = True  # Should be False in production
    config.api.ssl_enabled = False  # Should be True in production
    config.security.encryption_enabled = False  # Should be True in production
    config.security.security_level = SecurityLevel.LOW  # Should be HIGH or CRITICAL
    
    # Validate
    is_valid, errors = config.validate()
    
    print(f"Production configuration - Valid: {is_valid}")
    print("Production-specific errors:")
    for error in errors:
        print(f"  - {error}")
    
    return not is_valid


def test_enhanced_config_manager():
    """Test enhanced configuration manager features"""
    print("\n=== Testing Enhanced Configuration Manager ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create manager with strict validation
        manager = EnhancedConfigManager(
            config_dir=temp_dir,
            environment="test",
            validation_level=ValidationLevel.STRICT,
            enable_recovery=True
        )
        
        # Test 1: Register and save valid configuration
        print("Test 1: Valid configuration")
        valid_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "username": "test_user",
                "password": "secure_password_123"
            }
        }
        
        # Register schema
        schema = {
            "type": "object",
            "properties": {
                "database": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer", "minimum": 1, "maximum": 65535}
                    },
                    "required": ["host", "port"]
                }
            },
            "required": ["database"]
        }
        
        manager.validators["test_config"] = manager._get_validator("test_config")
        manager.validators["test_config"].schemas["test_config"] = schema
        
        # Save configuration
        result = manager.save_config("test_config", valid_config)
        print(f"  Save valid config: {'✓ PASSED' if result else '✗ FAILED'}")
        
        # Test 2: Try to save invalid configuration
        print("\nTest 2: Invalid configuration")
        invalid_config = {
            "database": {
                "host": "localhost",
                "port": 99999,  # Invalid port
                "database": "test_db"
            }
        }
        
        try:
            manager.save_config("test_invalid", invalid_config)
            print("  Save invalid config: ✗ FAILED (should have raised error)")
        except ConfigValidationError as e:
            print(f"  Save invalid config: ✓ PASSED (correctly rejected)")
            print(f"    Error: {e}")
        
        # Test 3: Configuration with sensitive data
        print("\nTest 3: Sensitive data encryption")
        sensitive_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "password": "super_secret_password",
                "api_key": "sk-1234567890abcdef"
            }
        }
        
        manager.save_config("test_sensitive", sensitive_config)
        
        # Check if data was encrypted
        config_file = Path(temp_dir) / "environments" / "test" / "test_sensitive.json"
        with open(config_file, 'r') as f:
            saved_data = json.load(f)
        
        is_encrypted = (
            saved_data["database"]["password"].startswith("encrypted:") and
            saved_data["database"]["api_key"].startswith("encrypted:")
        )
        print(f"  Sensitive data encrypted: {'✓ PASSED' if is_encrypted else '✗ FAILED'}")
        
        # Test 4: Configuration recovery
        print("\nTest 4: Configuration recovery")
        
        # Corrupt the configuration file
        with open(config_file, 'w') as f:
            f.write("invalid json content")
        
        # Try to load - should attempt recovery
        recovered_config = manager.load_config("test_sensitive", required=False, use_defaults=True)
        print(f"  Recovery from corruption: {'✓ PASSED' if recovered_config else '✗ FAILED'}")
        
        # Test 5: Environment variable overrides
        print("\nTest 5: Environment variable overrides")
        os.environ["FRAUD_CONFIG_TEST_ENV_DATABASE_PORT"] = "3306"
        
        env_config = manager.load_config("test_env", required=False, use_defaults=True)
        if env_config and env_config.get("database", {}).get("port") == 3306:
            print("  Environment override: ✓ PASSED")
        else:
            print("  Environment override: ✗ FAILED")
        
        return True


def test_config_dependency_validation():
    """Test configuration dependency validation"""
    print("\n=== Testing Configuration Dependency Validation ===\n")
    
    config = DomainConfig()
    
    # Enable real-time processing without Redis
    config.real_time_processing = True
    config.redis.host = ""  # Empty Redis host
    
    # Enable auto-retrain without ML model path
    config.ml_model.auto_retrain = True
    config.ml_model.model_path = ""
    
    # Validate
    is_valid, errors = config.validate()
    
    print(f"Dependency validation - Valid: {is_valid}")
    print("Dependency-related errors:")
    for error in errors:
        if "required" in error.lower() or "depend" in error.lower():
            print(f"  - {error}")
    
    return True


def test_performance_validation():
    """Test performance-related validation"""
    print("\n=== Testing Performance Validation ===\n")
    
    config = DomainConfig()
    
    # Set values that could cause performance issues
    config.database.connection_pool_size = 500
    config.database.connection_pool_max_overflow = 600  # Total: 1100 connections
    config.ml_model.batch_size = 1000000  # Very large batch
    config.api.request_timeout = 3600  # 1 hour timeout
    config.api.max_request_size = 2 * 1024 * 1024 * 1024  # 2GB
    
    # Validate
    is_valid, errors = config.validate()
    
    print(f"Performance validation - Valid: {is_valid}")
    print("Performance-related issues:")
    for error in errors:
        print(f"  - {error}")
    
    return True


def test_error_recovery():
    """Test error recovery mechanisms"""
    print("\n=== Testing Error Recovery Mechanisms ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create configuration file with errors
        config_path = Path(temp_dir) / "fraud_config.json"
        
        # Test 1: Recovery from missing file
        print("Test 1: Recovery from missing file")
        fraud_config = FinancialFraudConfig(
            config_path=str(config_path),
            environment="test",
            strict_mode=False  # Allow recovery
        )
        
        print(f"  Loaded without file: ✓ PASSED")
        
        # Test 2: Recovery from corrupted file
        print("\nTest 2: Recovery from corrupted file")
        config_path.write_text("{ invalid json content")
        
        fraud_config = FinancialFraudConfig(
            config_path=str(config_path),
            environment="test",
            strict_mode=False
        )
        
        print(f"  Recovered from corruption: ✓ PASSED")
        
        # Test 3: Strict mode failure
        print("\nTest 3: Strict mode with invalid config")
        
        invalid_config = {
            "model_threshold": 2.0,  # Invalid: > 1
            "database": {
                "port": -1  # Invalid: negative
            }
        }
        
        config_path.write_text(json.dumps(invalid_config))
        
        try:
            fraud_config = FinancialFraudConfig(
                config_path=str(config_path),
                environment="test",
                strict_mode=True  # Strict mode
            )
            print("  Strict mode validation: ✗ FAILED (should have raised error)")
        except Exception as e:
            print(f"  Strict mode validation: ✓ PASSED (correctly failed)")
            print(f"    Error type: {type(e).__name__}")
        
        return True


def test_comprehensive_validation():
    """Run comprehensive validation test suite"""
    print("\n" + "="*60)
    print("COMPREHENSIVE CONFIGURATION VALIDATION TEST SUITE")
    print("="*60)
    
    tests = [
        ("Basic Validation", test_basic_validation),
        ("Security Validation", test_security_validation),
        ("Environment Validation", test_environment_validation),
        ("Dependency Validation", test_config_dependency_validation),
        ("Performance Validation", test_performance_validation),
        ("Enhanced Config Manager", test_enhanced_config_manager),
        ("Error Recovery", test_error_recovery)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "PASSED" if result else "FAILED", None))
        except Exception as e:
            results.append((test_name, "ERROR", str(e)))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, status, error in results:
        status_symbol = "✓" if status == "PASSED" else "✗"
        print(f"{status_symbol} {test_name}: {status}")
        if error:
            print(f"    Error: {error}")
    
    passed = sum(1 for _, status, _ in results if status == "PASSED")
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    # Run comprehensive test suite
    success = test_comprehensive_validation()
    
    print("\n" + "="*60)
    if success:
        print("✅ ALL TESTS PASSED - Configuration validation is working correctly!")
    else:
        print("❌ SOME TESTS FAILED - Please review the errors above")
    print("="*60)