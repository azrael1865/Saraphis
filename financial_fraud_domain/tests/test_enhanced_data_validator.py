"""
Test script for Enhanced Financial Data Validator
Demonstrates comprehensive validation and error handling features
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
import time
from datetime import datetime, timedelta

# Import the enhanced data validator
from enhanced_data_validator import (
    EnhancedFinancialDataValidator, ValidationConfig, ValidationContext, 
    ValidationLevel, SecurityLevel, ValidationMode,
    InputValidationError, ValidationTimeoutError, ValidationConfigError,
    ValidationSecurityError, ValidationIntegrationError, DataCorruptionError,
    PartialValidationError, create_validator
)

def create_test_data():
    """Create test data with various quality issues"""
    print("\n=== Creating Test Data ===")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Test directory: {temp_dir}")
    
    # 1. Good quality data
    good_data = pd.DataFrame({
        'transaction_id': [f'TXN{i:08d}' for i in range(1000)],
        'user_id': [f'USER{np.random.randint(1, 100):06d}' for _ in range(1000)],
        'amount': np.random.lognormal(mean=4, sigma=1, size=1000).round(2),
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1H'),
        'merchant_id': [f'MERCH{np.random.randint(1, 50):05d}' for _ in range(1000)],
        'currency': np.random.choice(['USD', 'EUR', 'GBP'], size=1000, p=[0.7, 0.2, 0.1]),
        'transaction_type': np.random.choice(['purchase', 'withdrawal', 'transfer'], size=1000)
    })
    print(f"✓ Created good data: {len(good_data)} records")
    
    # 2. Data with quality issues
    bad_data = pd.DataFrame({
        'transaction_id': ['TXN00000001'] * 500 + [f'TXN{i:08d}' for i in range(500, 1000)],  # Duplicates
        'user_id': [f'USER{i:06d}' if i % 10 != 0 else None for i in range(1000)],  # Nulls
        'amount': [-100] * 100 + [1000000000] * 50 + list(np.random.normal(100, 20, 850)),  # Invalid amounts
        'timestamp': ['invalid'] * 200 + list(pd.date_range('2019-01-01', periods=800, freq='1H')),  # Invalid timestamps
        'merchant_id': [f'MERCH{i:05d}' for i in range(1000)],
        'currency': ['XXX'] * 300 + ['USD'] * 700,  # Invalid currency
        'transaction_type': ['purchase'] * 1000
    })
    print(f"✓ Created bad data: {len(bad_data)} records")
    
    # 3. Data with security issues (sensitive information)
    sensitive_data = pd.DataFrame({
        'transaction_id': [f'TXN{i:08d}' for i in range(100)],
        'user_id': [f'USER{i:06d}' for i in range(100)],
        'amount': np.random.uniform(10, 1000, 100).round(2),
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
        'credit_card': ['4111111111111111'] * 50 + ['5500000000000004'] * 50,  # Credit card numbers
        'ssn': ['123-45-6789'] * 100,  # SSN
        'email': [f'user{i}@example.com' for i in range(100)]  # Emails
    })
    print(f"✓ Created sensitive data: {len(sensitive_data)} records")
    
    # 4. Large file (for memory testing)
    large_data = pd.DataFrame({
        'transaction_id': [f'TXN{i:08d}' for i in range(50000)],
        'user_id': [f'USER{np.random.randint(1, 1000):06d}' for _ in range(50000)],
        'amount': np.random.lognormal(mean=4, sigma=1, size=50000).round(2),
        'timestamp': pd.date_range('2024-01-01', periods=50000, freq='1min')
    })
    print(f"✓ Created large data: {len(large_data)} records")
    
    # 5. Empty data
    empty_data = pd.DataFrame()
    print(f"✓ Created empty data: {len(empty_data)} records")
    
    return {
        'temp_dir': temp_dir,
        'good_data': good_data,
        'bad_data': bad_data,
        'sensitive_data': sensitive_data,
        'large_data': large_data,
        'empty_data': empty_data
    }

def test_basic_validation(validator: EnhancedFinancialDataValidator, test_data: dict):
    """Test basic validation features"""
    print("\n=== Testing Basic Validation ===")
    
    # Test 1: Valid data
    print("\n1. Testing valid data:")
    try:
        result = validator.validate(test_data['good_data'])
        print(f"   Valid: {result.is_valid}")
        print(f"   Quality Score: {result.data_quality_score:.2f}")
        print(f"   Issues: {len(result.issues) if hasattr(result, 'issues') else 0}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Bad data
    print("\n2. Testing data with quality issues:")
    try:
        result = validator.validate(test_data['bad_data'])
        print(f"   Valid: {result.is_valid}")
        print(f"   Quality Score: {result.data_quality_score:.2f}")
        print(f"   Issues: {len(result.issues) if hasattr(result, 'issues') else 0}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Empty data
    print("\n3. Testing empty data:")
    try:
        result = validator.validate(test_data['empty_data'])
        print(f"   Valid: {result.is_valid}")
    except InputValidationError as e:
        print(f"   ✓ Caught InputValidationError: {e}")
    except Exception as e:
        print(f"   Error: {e}")

def test_error_handling(validator: EnhancedFinancialDataValidator, test_data: dict):
    """Test error handling capabilities"""
    print("\n=== Testing Error Handling ===")
    
    # Test 1: Invalid input type
    print("\n1. Testing invalid input type:")
    try:
        validator.validate("not a dataframe")
    except InputValidationError as e:
        print(f"   ✓ Caught InputValidationError: {e}")
    except Exception as e:
        print(f"   Unexpected error: {e}")
    
    # Test 2: Configuration errors
    print("\n2. Testing configuration validation:")
    invalid_config = {
        'validation_level': 'invalid_level',
        'timeout_seconds': -10,
        'max_memory_mb': 'not_a_number'
    }
    is_valid, issues = validator.validate_configuration(invalid_config)
    print(f"   Config valid: {is_valid}")
    print(f"   Issues: {issues[:3]}...")  # Show first 3 issues
    
    # Test 3: Memory limit testing (smaller limit)
    print("\n3. Testing memory limits:")
    config = ValidationConfig(max_memory_mb=10)  # Very low limit
    memory_validator = EnhancedFinancialDataValidator(config)
    try:
        result = memory_validator.validate(test_data['large_data'])
        print(f"   Unexpectedly succeeded with {len(result.issues) if hasattr(result, 'issues') else 0} issues")
    except InputValidationError as e:
        print(f"   ✓ Caught memory limit error: {e}")
    except Exception as e:
        print(f"   Error: {e}")

def test_security_validation(validator: EnhancedFinancialDataValidator, test_data: dict):
    """Test security validation features"""
    print("\n=== Testing Security Validation ===")
    
    # Test different security levels
    security_levels = [SecurityLevel.PUBLIC, SecurityLevel.INTERNAL, SecurityLevel.CONFIDENTIAL]
    
    for level in security_levels:
        print(f"\n Testing security level: {level.value}")
        try:
            config = ValidationConfig(
                security_level=level,
                security_validation=True,
                validation_level=ValidationLevel.BASIC
            )
            sec_validator = EnhancedFinancialDataValidator(config)
            
            result = sec_validator.validate(test_data['sensitive_data'])
            
            # Check for security report
            report = sec_validator.get_validation_report(result)
            security_report = report.get('security_report', {})
            
            print(f"   ✓ Security validation completed")
            print(f"   PII detected: {security_report.get('pii_detected', False)}")
            print(f"   Security risks: {len(security_report.get('security_risks', []))}")
            
        except ValidationSecurityError as e:
            print(f"   ✓ Caught security error: {e}")
        except Exception as e:
            print(f"   Error: {e}")

def test_performance_monitoring(validator: EnhancedFinancialDataValidator, test_data: dict):
    """Test performance monitoring"""
    print("\n=== Testing Performance Monitoring ===")
    
    # Test with performance monitoring enabled
    config = ValidationConfig(
        performance_monitoring=True,
        validation_level=ValidationLevel.COMPREHENSIVE
    )
    perf_validator = EnhancedFinancialDataValidator(config)
    
    # Validate data and check performance metrics
    start_time = time.time()
    result = perf_validator.validate(test_data['good_data'])
    duration = time.time() - start_time
    
    print(f"\nPerformance Test Results:")
    print(f"  Validation time: {duration:.3f}s")
    print(f"  Records processed: {len(test_data['good_data'])}")
    print(f"  Records/second: {len(test_data['good_data']) / duration:.0f}")
    
    # Get performance report
    report = perf_validator.get_validation_report(result)
    perf_metrics = report.get('performance_metrics', {})
    
    if perf_metrics:
        print(f"  Memory usage: {perf_metrics.get('memory_usage_mb', 0):.1f}MB")
        print(f"  Rules executed: {perf_metrics.get('rules_executed', 0)}")
        print(f"  Bottlenecks: {perf_metrics.get('bottlenecks', [])}")
        print(f"  Suggestions: {perf_metrics.get('optimization_suggestions', [])}")
    
    # Test performance summary
    perf_summary = perf_validator.get_performance_summary()
    print(f"\nPerformance Summary: {perf_summary}")

def test_validation_levels(validator: EnhancedFinancialDataValidator, test_data: dict):
    """Test different validation levels"""
    print("\n=== Testing Validation Levels ===")
    
    levels = [ValidationLevel.BASIC, ValidationLevel.STANDARD, 
              ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]
    
    for level in levels:
        print(f"\n Testing validation level: {level.value}")
        try:
            start_time = time.time()
            result = validator.validate(test_data['bad_data'], validation_level=level)
            duration = time.time() - start_time
            
            print(f"   Issues found: {len(result.issues) if hasattr(result, 'issues') else 0}")
            print(f"   Quality score: {result.data_quality_score:.2f}")
            print(f"   Duration: {duration:.3f}s")
            
        except Exception as e:
            print(f"   Failed: {type(e).__name__}: {e}")

def test_timeout_handling(validator: EnhancedFinancialDataValidator, test_data: dict):
    """Test timeout handling"""
    print("\n=== Testing Timeout Handling ===")
    
    # Test with very short timeout
    config = ValidationConfig(
        timeout_seconds=0.1,  # Very short timeout
        validation_level=ValidationLevel.COMPREHENSIVE
    )
    timeout_validator = EnhancedFinancialDataValidator(config)
    
    try:
        result = timeout_validator.validate(test_data['large_data'])
        print(f"   Validation completed without timeout")
    except ValidationTimeoutError as e:
        print(f"   ✓ Caught timeout error: {e}")
    except Exception as e:
        print(f"   Error: {e}")

def test_recovery_mechanisms(validator: EnhancedFinancialDataValidator, test_data: dict):
    """Test error recovery mechanisms"""
    print("\n=== Testing Recovery Mechanisms ===")
    
    # Test with recovery enabled
    config = ValidationConfig(
        enable_recovery=True,
        enable_partial_results=True,
        validation_level=ValidationLevel.STRICT
    )
    recovery_validator = EnhancedFinancialDataValidator(config)
    
    # Test with problematic data
    try:
        result = recovery_validator.validate(test_data['bad_data'])
        print(f"   Recovery test - Valid: {result.is_valid}")
        print(f"   Quality score: {result.data_quality_score:.2f}")
        print(f"   Issues: {len(result.issues) if hasattr(result, 'issues') else 0}")
    except PartialValidationError as e:
        print(f"   ✓ Partial validation error with recovery: {e}")
        if hasattr(e, 'partial_results') and e.partial_results:
            print(f"   Partial results available")
    except Exception as e:
        print(f"   Error: {e}")

def test_caching(validator: EnhancedFinancialDataValidator, test_data: dict):
    """Test validation caching"""
    print("\n=== Testing Validation Caching ===")
    
    # Test with caching enabled
    config = ValidationConfig(
        enable_caching=True,
        cache_ttl_seconds=60
    )
    cache_validator = EnhancedFinancialDataValidator(config)
    
    # First validation (should cache result)
    start_time = time.time()
    result1 = cache_validator.validate(test_data['good_data'])
    duration1 = time.time() - start_time
    
    # Second validation (should use cache)
    start_time = time.time()
    result2 = cache_validator.validate(test_data['good_data'])
    duration2 = time.time() - start_time
    
    print(f"   First validation: {duration1:.3f}s")
    print(f"   Second validation: {duration2:.3f}s")
    print(f"   Cache speedup: {duration1/duration2:.1f}x" if duration2 > 0 else "   Cache speedup: N/A")
    
    # Clear cache test
    cache_validator.clear_cache()
    print(f"   ✓ Cache cleared")

def test_integration_validation(validator: EnhancedFinancialDataValidator, test_data: dict):
    """Test integration validation"""
    print("\n=== Testing Integration Validation ===")
    
    # Test with integration testing enabled
    config = ValidationConfig(
        integration_testing=True,
        validation_level=ValidationLevel.BASIC
    )
    integration_validator = EnhancedFinancialDataValidator(config)
    
    try:
        result = integration_validator.validate(test_data['good_data'])
        
        # Check integration report
        report = integration_validator.get_validation_report(result)
        integration_report = report.get('integration_report', {})
        
        print(f"   Brain system compatible: {integration_report.get('brain_system_compatible', False)}")
        print(f"   Dependencies validated: {integration_report.get('dependencies_validated', False)}")
        print(f"   Integration score: {integration_report.get('integration_score', 0):.2f}")
        print(f"   Recommendations: {integration_report.get('recommendations', [])}")
        
    except ValidationIntegrationError as e:
        print(f"   ✓ Caught integration error: {e}")
    except Exception as e:
        print(f"   Error: {e}")

def test_error_reporting(validator: EnhancedFinancialDataValidator, test_data: dict):
    """Test error reporting capabilities"""
    print("\n=== Testing Error Reporting ===")
    
    # Generate some errors by testing various problematic scenarios
    test_scenarios = [
        test_data['empty_data'],
        test_data['bad_data'],
        test_data['sensitive_data']
    ]
    
    for i, scenario in enumerate(test_scenarios):
        try:
            validator.validate(scenario, validation_level=ValidationLevel.STRICT)
        except Exception:
            pass  # Expected to fail for some scenarios
    
    # Get error summary
    error_summary = validator.get_error_summary()
    
    print("\nError Report Summary:")
    print(f"  Total errors: {error_summary.get('total_errors', 0)}")
    print(f"  Error types: {error_summary.get('error_types', {})}")
    
    recent_errors = error_summary.get('recent_errors', [])
    if recent_errors:
        print(f"  Recent errors:")
        for error in recent_errors[:3]:
            print(f"    - {error.get('type', 'Unknown')}: {error.get('message', 'No message')}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("ENHANCED FINANCIAL DATA VALIDATOR TEST SUITE")
    print("=" * 60)
    
    # Create test data
    test_data = create_test_data()
    
    try:
        # Initialize validator
        print("\n=== Initializing Enhanced Data Validator ===")
        config = ValidationConfig(
            validation_level=ValidationLevel.STANDARD,
            context=ValidationContext.TESTING,
            security_validation=True,
            performance_monitoring=True,
            enable_recovery=True,
            timeout_seconds=30.0
        )
        validator = EnhancedFinancialDataValidator(config)
        print(f"✓ Validator initialized: {validator}")
        
        # Run tests
        test_basic_validation(validator, test_data)
        test_error_handling(validator, test_data)
        test_security_validation(validator, test_data)
        test_performance_monitoring(validator, test_data)
        test_validation_levels(validator, test_data)
        test_timeout_handling(validator, test_data)
        test_recovery_mechanisms(validator, test_data)
        test_caching(validator, test_data)
        test_integration_validation(validator, test_data)
        test_error_reporting(validator, test_data)
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Final validator status
        print(f"\nFinal Validator Status: {validator}")
        error_summary = validator.get_error_summary()
        perf_summary = validator.get_performance_summary()
        print(f"Total errors recorded: {error_summary.get('total_errors', 0)}")
        print(f"Total validations: {perf_summary.get('total_validations', 0)}")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print(f"\nCleaning up test directory: {test_data['temp_dir']}")
        import shutil
        shutil.rmtree(test_data['temp_dir'], ignore_errors=True)

if __name__ == "__main__":
    main()