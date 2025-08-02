"""
Test script for Enhanced Financial Data Loader
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

# Import the enhanced data loader
from enhanced_data_loader import (
    EnhancedFinancialDataLoader, DataSource, DataSourceType, 
    ValidationLevel, SecurityLevel, LoadStatus,
    DataSourceValidationError, DataAccessError, DataFormatError,
    DataSizeError, DataIntegrityError, DataSecurityError,
    DataQualityError, DataLoadTimeoutError, DataMemoryError,
    create_enhanced_data_loader
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
    good_file = temp_dir / 'good_data.csv'
    good_data.to_csv(good_file, index=False)
    print(f"✓ Created good data: {good_file}")
    
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
    bad_file = temp_dir / 'bad_data.csv'
    bad_data.to_csv(bad_file, index=False)
    print(f"✓ Created bad data: {bad_file}")
    
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
    sensitive_file = temp_dir / 'sensitive_data.csv'
    sensitive_data.to_csv(sensitive_file, index=False)
    print(f"✓ Created sensitive data: {sensitive_file}")
    
    # 4. Large file (for memory testing)
    large_data = pd.DataFrame({
        'transaction_id': [f'TXN{i:08d}' for i in range(100000)],
        'user_id': [f'USER{np.random.randint(1, 1000):06d}' for _ in range(100000)],
        'amount': np.random.lognormal(mean=4, sigma=1, size=100000).round(2),
        'timestamp': pd.date_range('2024-01-01', periods=100000, freq='1min')
    })
    large_file = temp_dir / 'large_data.csv'
    large_data.to_csv(large_file, index=False)
    print(f"✓ Created large data: {large_file} ({large_file.stat().st_size / 1024 / 1024:.1f}MB)")
    
    # 5. Malformed CSV
    with open(temp_dir / 'malformed.csv', 'w') as f:
        f.write('transaction_id,amount,timestamp\n')
        f.write('TXN001,100.50,2024-01-01\n')
        f.write('TXN002,"broken,data",2024-01-02\n')
        f.write('TXN003,not_a_number,invalid_date\n')
        f.write('TXN004,200.00\n')  # Missing column
    print(f"✓ Created malformed CSV: {temp_dir / 'malformed.csv'}")
    
    # 6. JSON data
    json_data = {
        'transactions': [
            {'transaction_id': f'TXN{i:08d}', 'amount': float(i * 10), 'timestamp': '2024-01-01'}
            for i in range(100)
        ]
    }
    json_file = temp_dir / 'data.json'
    with open(json_file, 'w') as f:
        json.dump(json_data, f)
    print(f"✓ Created JSON data: {json_file}")
    
    # 7. Empty file
    empty_file = temp_dir / 'empty.csv'
    empty_file.touch()
    print(f"✓ Created empty file: {empty_file}")
    
    return temp_dir

def test_basic_validation(loader: EnhancedFinancialDataLoader, test_dir: Path):
    """Test basic validation features"""
    print("\n=== Testing Basic Validation ===")
    
    # Test 1: Valid data source
    print("\n1. Testing valid data source:")
    source = DataSource(
        name="good_data",
        source_type=DataSourceType.CSV,
        location=str(test_dir / 'good_data.csv'),
        validation_level=ValidationLevel.BASIC
    )
    
    is_valid, issues = loader.validate_data_source(source)
    print(f"   Valid: {is_valid}")
    print(f"   Issues: {issues}")
    
    # Test 2: Invalid source location
    print("\n2. Testing invalid source location:")
    invalid_source = DataSource(
        name="nonexistent",
        source_type=DataSourceType.CSV,
        location=str(test_dir / 'nonexistent.csv'),
        validation_level=ValidationLevel.BASIC
    )
    
    is_valid, issues = loader.validate_data_source(invalid_source)
    print(f"   Valid: {is_valid}")
    print(f"   Issues: {issues[:2]}...")  # Show first 2 issues
    
    # Test 3: Invalid configuration
    print("\n3. Testing invalid configuration:")
    bad_config_source = DataSource(
        name="bad_config",
        source_type=DataSourceType.CSV,
        location=str(test_dir / 'good_data.csv'),
        timeout=-1,  # Invalid
        retry_attempts=100,  # Too high
        max_memory_mb=-10  # Invalid
    )
    
    is_valid, issues = loader.validate_data_source(bad_config_source)
    print(f"   Valid: {is_valid}")
    print(f"   Issues: {issues}")

def test_error_handling(loader: EnhancedFinancialDataLoader, test_dir: Path):
    """Test error handling capabilities"""
    print("\n=== Testing Error Handling ===")
    
    # Test 1: DataAccessError - File not found
    print("\n1. Testing DataAccessError (file not found):")
    try:
        loader.load(str(test_dir / 'nonexistent.csv'))
    except DataAccessError as e:
        print(f"   ✓ Caught DataAccessError: {e}")
        print(f"   Details: {e.details}")
    
    # Test 2: DataFormatError - Malformed CSV
    print("\n2. Testing DataFormatError (malformed CSV):")
    try:
        loader.load(
            str(test_dir / 'malformed.csv'),
            validation_level=ValidationLevel.STRICT
        )
    except (DataFormatError, DataQualityError) as e:
        print(f"   ✓ Caught {type(e).__name__}: {e}")
    
    # Test 3: DataQualityError - Low quality data
    print("\n3. Testing DataQualityError (low quality data):")
    try:
        source = DataSource(
            name="bad_data",
            source_type=DataSourceType.CSV,
            location=str(test_dir / 'bad_data.csv'),
            quality_threshold=0.9,  # High threshold
            enable_degradation=False
        )
        loader.load(source, validation_level=ValidationLevel.STRICT)
    except DataQualityError as e:
        print(f"   ✓ Caught DataQualityError: {e}")
        print(f"   Quality score: {e.details.get('quality_score', 'N/A')}")
    
    # Test 4: Empty file
    print("\n4. Testing empty file:")
    try:
        loader.load(str(test_dir / 'empty.csv'))
    except (DataAccessError, DataFormatError) as e:
        print(f"   ✓ Caught {type(e).__name__}: {e}")

def test_security_validation(loader: EnhancedFinancialDataLoader, test_dir: Path):
    """Test security validation features"""
    print("\n=== Testing Security Validation ===")
    
    # Test different security levels
    security_levels = [SecurityLevel.NONE, SecurityLevel.BASIC, SecurityLevel.HIGH]
    
    for level in security_levels:
        print(f"\n Testing security level: {level.value}")
        try:
            source = DataSource(
                name="sensitive_data",
                source_type=DataSourceType.CSV,
                location=str(test_dir / 'sensitive_data.csv'),
                security_level=level,
                validation_level=ValidationLevel.BASIC
            )
            
            if level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                # Should fail without credentials
                try:
                    loader.load(source)
                except DataSecurityError as e:
                    print(f"   ✓ Security validation caught sensitive data")
            else:
                # Should succeed with warning
                data = loader.load(source)
                print(f"   ✓ Loaded {len(data)} records with security level {level.value}")
                
        except DataSecurityError as e:
            print(f"   ✓ Caught DataSecurityError: {e}")

def test_memory_management(loader: EnhancedFinancialDataLoader, test_dir: Path):
    """Test memory management features"""
    print("\n=== Testing Memory Management ===")
    
    # Test 1: Load with memory limit
    print("\n1. Testing memory limit enforcement:")
    source = DataSource(
        name="large_data",
        source_type=DataSourceType.CSV,
        location=str(test_dir / 'large_data.csv'),
        max_memory_mb=50,  # Low limit
        batch_size=10000
    )
    
    try:
        # Should succeed with chunking
        data = loader.load(source, chunk_size=5000)
        print(f"   ✓ Successfully loaded {len(data)} records with memory limit")
    except DataMemoryError as e:
        print(f"   Memory limit exceeded: {e}")

def test_validation_levels(loader: EnhancedFinancialDataLoader, test_dir: Path):
    """Test different validation levels"""
    print("\n=== Testing Validation Levels ===")
    
    levels = [ValidationLevel.NONE, ValidationLevel.BASIC, 
              ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]
    
    source = DataSource(
        name="bad_data",
        source_type=DataSourceType.CSV,
        location=str(test_dir / 'bad_data.csv'),
        enable_degradation=True
    )
    
    for level in levels:
        print(f"\n Testing validation level: {level.value}")
        try:
            start_time = time.time()
            data = loader.load(source, validation_level=level)
            load_time = time.time() - start_time
            
            print(f"   Loaded: {len(data)} records")
            print(f"   Time: {load_time:.2f}s")
            
            # Get metrics
            latest_metric = loader.load_metrics[-1]
            print(f"   Quality score: {latest_metric.data_quality_score:.2f}")
            print(f"   Warnings: {latest_metric.warning_count}")
            print(f"   Errors: {latest_metric.error_count}")
            
        except Exception as e:
            print(f"   Failed: {type(e).__name__}: {e}")

def test_graceful_degradation(loader: EnhancedFinancialDataLoader, test_dir: Path):
    """Test graceful degradation features"""
    print("\n=== Testing Graceful Degradation ===")
    
    # Create source with strict requirements
    source = DataSource(
        name="bad_data",
        source_type=DataSourceType.CSV,
        location=str(test_dir / 'bad_data.csv'),
        quality_threshold=0.95,  # Very high threshold
        enable_degradation=True,  # Enable degradation
        validation_level=ValidationLevel.COMPREHENSIVE
    )
    
    print("\n1. Testing with degradation enabled:")
    try:
        data = loader.load(source)
        latest_metric = loader.load_metrics[-1]
        print(f"   ✓ Loaded in {latest_metric.load_status.value} mode")
        print(f"   Records: {len(data)}")
        print(f"   Quality: {latest_metric.data_quality_score:.2f}")
    except Exception as e:
        print(f"   Failed: {e}")
    
    # Test without degradation
    source.enable_degradation = False
    print("\n2. Testing without degradation:")
    try:
        data = loader.load(source)
        print(f"   Loaded: {len(data)} records")
    except DataQualityError as e:
        print(f"   ✓ Correctly failed with DataQualityError")
        print(f"   Quality score: {e.details.get('quality_score', 'N/A')}")

def test_error_reporting(loader: EnhancedFinancialDataLoader, test_dir: Path):
    """Test error reporting capabilities"""
    print("\n=== Testing Error Reporting ===")
    
    # Generate some errors
    test_sources = [
        str(test_dir / 'nonexistent.csv'),
        str(test_dir / 'malformed.csv'),
        str(test_dir / 'bad_data.csv'),
        str(test_dir / 'empty.csv')
    ]
    
    for source in test_sources:
        try:
            loader.load(source, validation_level=ValidationLevel.STRICT)
        except Exception:
            pass  # Expected to fail
    
    # Get error report
    report = loader.get_error_report()
    
    print("\nError Report Summary:")
    print(f"  Total loads: {report['total_loads']}")
    print(f"  Successful: {report['successful_loads']}")
    print(f"  Failed: {report['failed_loads']}")
    print(f"  Degraded: {report['degraded_loads']}")
    print(f"  Timeouts: {report['timeout_loads']}")
    print(f"  Total errors: {report['total_errors']}")
    print(f"  Total warnings: {report['total_warnings']}")
    print(f"  Average quality: {report['average_quality_score']:.2f}")
    
    if report['failure_details']:
        print("\nFailure Details:")
        for failure in report['failure_details'][:3]:
            print(f"  - {failure['source']}: {failure['errors'][0] if failure['errors'] else 'Unknown error'}")

def test_performance_monitoring(loader: EnhancedFinancialDataLoader, test_dir: Path):
    """Test performance monitoring"""
    print("\n=== Testing Performance Monitoring ===")
    
    # Load data and check performance metrics
    source = DataSource(
        name="good_data",
        source_type=DataSourceType.CSV,
        location=str(test_dir / 'good_data.csv')
    )
    
    data = loader.load(source)
    
    # Get latest metrics
    latest_metric = loader.load_metrics[-1]
    
    print("\nPerformance Metrics:")
    print(f"  Load time: {latest_metric.load_time_seconds:.2f}s")
    print(f"  Validation time: {latest_metric.validation_time_seconds:.2f}s")
    print(f"  Preprocessing time: {latest_metric.preprocessing_time_seconds:.2f}s")
    print(f"  Memory usage: {latest_metric.memory_usage_mb:.2f}MB")
    
    if latest_metric.performance_metrics:
        print("\nSystem Metrics:")
        for key, value in latest_metric.performance_metrics.items():
            print(f"  {key}: {value:.2f}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("ENHANCED FINANCIAL DATA LOADER TEST SUITE")
    print("=" * 60)
    
    # Create test data
    test_dir = create_test_data()
    
    try:
        # Initialize loader
        print("\n=== Initializing Enhanced Data Loader ===")
        loader = create_enhanced_data_loader(
            max_memory_mb=1024,
            enable_degradation=True
        )
        print(f"✓ Loader initialized (session: {loader._session_id})")
        
        # Run tests
        test_basic_validation(loader, test_dir)
        test_error_handling(loader, test_dir)
        test_security_validation(loader, test_dir)
        test_memory_management(loader, test_dir)
        test_validation_levels(loader, test_dir)
        test_graceful_degradation(loader, test_dir)
        test_error_reporting(loader, test_dir)
        test_performance_monitoring(loader, test_dir)
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print(f"\nCleaning up test directory: {test_dir}")
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)

if __name__ == "__main__":
    main()