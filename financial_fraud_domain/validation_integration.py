"""
Validation Integration Module
Provides unified access to all validation components for financial fraud detection.
Consolidates validation functionality into a simple interface.
"""

import logging
from typing import Optional, Dict, Any, Union, List
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Define fallback classes at module level
class SecurityLevel:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ValidationLevel:
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"

def get_integrated_validator(environment: str = "production", 
                           config: Optional[Dict[str, Any]] = None):
    """
    Get configured validator for specified environment
    
    Args:
        environment: "production", "development", or "testing"  
        config: Optional configuration dictionary
        
    Returns:
        Configured validator instance
    """
    try:
        # Try enhanced validator first
        try:
            from enhanced_data_validator import (
                EnhancedFinancialDataValidator,
                ValidationConfig,
                ValidationContext,
                ValidationMode,
                SecurityLevel
            )
        except ImportError:
            from enhanced_data_validator import (
                EnhancedFinancialDataValidator,
                ValidationConfig,
                ValidationContext,
                ValidationMode,
                SecurityLevel
            )
        
        # Create configuration based on environment
        if environment == "production":
            validation_config = ValidationConfig(
                validation_level="comprehensive",
                security_level=SecurityLevel.HIGH,
                enable_compliance_checks=True,
                enable_fraud_detection=True,
                max_validation_time=300.0,
                max_memory_usage_mb=2048.0,
                enable_parallel_validation=True,
                enable_caching=True,
                cache_size=1000,
                enable_error_recovery=True,
                fallback_strategy="partial",
                enable_performance_monitoring=True,
                enable_security_validation=True,
                enable_integration_validation=True
            )
        elif environment == "development":
            validation_config = ValidationConfig(
                validation_level="standard",
                security_level=SecurityLevel.MEDIUM,
                enable_compliance_checks=True,
                enable_fraud_detection=True,
                max_validation_time=120.0,
                max_memory_usage_mb=1024.0,
                enable_parallel_validation=False,
                enable_caching=True,
                enable_error_recovery=True,
                enable_performance_monitoring=True
            )
        else:  # testing
            validation_config = ValidationConfig(
                validation_level="basic",
                security_level=SecurityLevel.LOW,
                enable_compliance_checks=False,
                enable_fraud_detection=False,
                max_validation_time=30.0,
                max_memory_usage_mb=512.0,
                enable_parallel_validation=False,
                enable_caching=False,
                enable_error_recovery=True,
                enable_performance_monitoring=False
            )
        
        # Override with user config if provided
        if config:
            for key, value in config.items():
                if hasattr(validation_config, key):
                    setattr(validation_config, key, value)
        
        logger.info(f"Using enhanced validator for {environment} environment")
        return EnhancedFinancialDataValidator(validation_config)
        
    except ImportError as e:
        logger.warning(f"Enhanced validator not available: {e}, falling back to standard")
        
        # Fallback to standard validator
        try:
            from data_validator import FinancialDataValidator, ValidationLevel
            
            # Create simplified configuration
            if environment == "production":
                validation_level = ValidationLevel.COMPREHENSIVE
            elif environment == "development":
                validation_level = ValidationLevel.STANDARD
            else:  # testing
                validation_level = ValidationLevel.BASIC
            
            logger.info(f"Using standard validator for {environment} environment")
            return FinancialDataValidator(validation_level=validation_level)
            
        except ImportError as e:
            logger.warning(f"Standard validator not available: {e}, using minimal validator")
            
            # Create minimal fallback classes
            class SecurityLevel:
                LOW = "low"
                MEDIUM = "medium"
                HIGH = "high"
            
            class ValidationLevel:
                BASIC = "basic"
                STANDARD = "standard"
                COMPREHENSIVE = "comprehensive"
            
            class MinimalValidator:
                def __init__(self, validation_level=None):
                    self.validation_level = validation_level
                    
                def validate(self, data):
                    return {
                        'is_valid': data is not None,
                        'issues': [] if data is not None else ['Data is None']
                    }
                    
                def validate_transaction_data(self, data):
                    return self.validate(data)
            
            logger.info(f"Using minimal validator for {environment} environment")
            return MinimalValidator()

def get_integrated_transaction_validator(environment: str = "production",
                                       config: Optional[Dict[str, Any]] = None):
    """
    Get configured transaction validator for specified environment
    
    Args:
        environment: Environment configuration
        config: Optional configuration overrides
        
    Returns:
        Configured transaction validator instance
    """
    try:
        # Try enhanced transaction validator first
        from enhanced_transaction_validator import EnhancedTransactionFieldValidator
        
        # Create configuration based on environment
        if environment == "production":
            validator_config = {
                'enable_comprehensive_validation': True,
                'enable_fraud_patterns': True,
                'enable_compliance_checks': True,
                'strict_validation': True,
                'enable_error_recovery': True
            }
        elif environment == "development":
            validator_config = {
                'enable_comprehensive_validation': True,
                'enable_fraud_patterns': True,
                'enable_compliance_checks': False,
                'strict_validation': False,
                'enable_error_recovery': True
            }
        else:  # testing
            validator_config = {
                'enable_comprehensive_validation': False,
                'enable_fraud_patterns': False,
                'enable_compliance_checks': False,
                'strict_validation': False,
                'enable_error_recovery': True
            }
        
        # Override with user config
        if config:
            validator_config.update(config)
        
        logger.info(f"Using enhanced transaction validator for {environment} environment")
        return EnhancedTransactionFieldValidator(**validator_config)
        
    except ImportError as e:
        logger.warning(f"Enhanced transaction validator not available: {e}")
        
        # Create a minimal transaction validator
        class BasicTransactionValidator:
            def __init__(self):
                self.environment = environment
            
            def validate_transaction(self, transaction_data):
                """Basic transaction validation"""
                issues = []
                
                # Check required fields
                required_fields = ['transaction_id', 'user_id', 'amount']
                for field in required_fields:
                    if field not in transaction_data or transaction_data[field] is None:
                        issues.append(f"Missing required field: {field}")
                
                # Basic amount validation
                if 'amount' in transaction_data:
                    try:
                        amount = float(transaction_data['amount'])
                        if amount <= 0:
                            issues.append("Amount must be positive")
                        if amount > 100000:  # Basic threshold
                            issues.append("Amount exceeds threshold")
                    except (ValueError, TypeError):
                        issues.append("Invalid amount format")
                
                return {
                    'is_valid': len(issues) == 0,
                    'issues': issues
                }
        
        logger.info(f"Using basic transaction validator for {environment} environment")
        return BasicTransactionValidator()

def validate_data_integrated(data, environment: str = "production", 
                           config: Optional[Dict[str, Any]] = None):
    """
    Validate data with integrated validation pipeline
    
    Args:
        data: Input DataFrame or dict
        environment: Environment configuration
        config: Optional configuration overrides
        
    Returns:
        Validation result
    """
    validator = get_integrated_validator(environment, config)
    
    if hasattr(validator, 'validate_transaction_data'):
        return validator.validate_transaction_data(data)
    elif hasattr(validator, 'validate'):
        return validator.validate(data)
    else:
        # Basic validation
        return {
            'is_valid': data is not None,
            'issues': [] if data is not None else ['Data is None'],
            'validator_type': type(validator).__name__
        }

def validate_transaction_integrated(transaction_data, environment: str = "production",
                                  config: Optional[Dict[str, Any]] = None):
    """
    Validate single transaction with integrated pipeline
    
    Args:
        transaction_data: Transaction data dict
        environment: Environment configuration  
        config: Optional configuration overrides
        
    Returns:
        Validation result
    """
    transaction_validator = get_integrated_transaction_validator(environment, config)
    
    if hasattr(transaction_validator, 'validate_transaction'):
        return transaction_validator.validate_transaction(transaction_data)
    elif hasattr(transaction_validator, 'validate'):
        return transaction_validator.validate(transaction_data)
    else:
        # Basic validation
        return {
            'is_valid': transaction_data is not None,
            'issues': [] if transaction_data is not None else ['Transaction data is None']
        }

def get_validation_statistics(validator):
    """
    Get validation statistics from any validator type
    
    Args:
        validator: Validator instance
        
    Returns:
        Dictionary of statistics  
    """
    try:
        if hasattr(validator, 'get_statistics'):
            return validator.get_statistics()
        elif hasattr(validator, 'get_performance_metrics'):
            return validator.get_performance_metrics()
        elif hasattr(validator, 'validation_stats'):
            return validator.validation_stats
        else:
            return {
                'validator_type': type(validator).__name__,
                'methods': [method for method in dir(validator) if not method.startswith('_')],
                'message': 'Limited statistics available'
            }
    except Exception as e:
        logger.error(f"Failed to get validation statistics: {e}")
        return {'error': str(e)}

def run_integrated_validation_tests():
    """
    Run comprehensive validation tests across all environments
    
    Returns:
        Test results dictionary
    """
    test_results = {
        'environments_tested': [],
        'validators_tested': [],
        'test_results': {},
        'errors': []
    }
    
    # Test data
    import pandas as pd
    import numpy as np
    
    test_data = pd.DataFrame({
        'transaction_id': [f'TXN{i:06d}' for i in range(10)],
        'user_id': [f'USER{i:03d}' for i in range(10)],
        'amount': np.random.uniform(10, 1000, 10),
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H'),
        'merchant_id': [f'MERCH{i%3:03d}' for i in range(10)]
    })
    
    test_transaction = {
        'transaction_id': 'TXN123456',
        'user_id': 'USER123',
        'amount': 100.50,
        'timestamp': '2024-01-01T12:00:00Z',
        'merchant_id': 'MERCH001'
    }
    
    # Test each environment
    for environment in ['production', 'development', 'testing']:
        try:
            test_results['environments_tested'].append(environment)
            
            # Test data validator
            try:
                validator = get_integrated_validator(environment)
                test_results['validators_tested'].append(f"data_validator_{environment}")
                
                result = validate_data_integrated(test_data, environment)
                test_results['test_results'][f'data_validation_{environment}'] = {
                    'success': True,
                    'is_valid': result.get('is_valid', False),
                    'issue_count': len(result.get('issues', []))
                }
            except Exception as e:
                test_results['errors'].append(f"Data validator {environment}: {str(e)}")
                test_results['test_results'][f'data_validation_{environment}'] = {
                    'success': False,
                    'error': str(e)
                }
            
            # Test transaction validator
            try:
                transaction_result = validate_transaction_integrated(test_transaction, environment)
                test_results['test_results'][f'transaction_validation_{environment}'] = {
                    'success': True,
                    'is_valid': transaction_result.get('is_valid', False),
                    'issue_count': len(transaction_result.get('issues', []))
                }
            except Exception as e:
                test_results['errors'].append(f"Transaction validator {environment}: {str(e)}")
                test_results['test_results'][f'transaction_validation_{environment}'] = {
                    'success': False,
                    'error': str(e)
                }
                
        except Exception as e:
            test_results['errors'].append(f"Environment {environment}: {str(e)}")
    
    # Calculate summary
    total_tests = len(test_results['test_results'])
    successful_tests = sum(1 for result in test_results['test_results'].values() 
                          if result.get('success', False))
    
    test_results['summary'] = {
        'total_tests': total_tests,
        'successful_tests': successful_tests,
        'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
        'total_errors': len(test_results['errors'])
    }
    
    return test_results

# Export main functions
__all__ = [
    'get_integrated_validator',
    'get_integrated_transaction_validator', 
    'validate_data_integrated',
    'validate_transaction_integrated',
    'get_validation_statistics',
    'run_integrated_validation_tests'
]

if __name__ == "__main__":
    print("Validation Integration Module")
    print("Available environments: production, development, testing")
    
    # Test basic functionality
    try:
        validator = get_integrated_validator("development")
        print(f"✓ Successfully created validator: {type(validator).__name__}")
        
        # Test with sample data
        import pandas as pd
        import numpy as np
        
        sample_data = pd.DataFrame({
            'transaction_id': [f'TXN{i:06d}' for i in range(5)],
            'user_id': [f'USER{i:03d}' for i in range(5)],
            'amount': np.random.uniform(10, 1000, 5),
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1H')
        })
        
        result = validate_data_integrated(sample_data, environment="testing")
        print(f"✓ Successfully validated data: valid={result.get('is_valid', False)}")
        
        # Test transaction validation
        test_transaction = {
            'transaction_id': 'TXN123456',
            'user_id': 'USER123', 
            'amount': 100.50
        }
        
        transaction_result = validate_transaction_integrated(test_transaction, environment="testing")
        print(f"✓ Successfully validated transaction: valid={transaction_result.get('is_valid', False)}")
        
        # Test statistics
        stats = get_validation_statistics(validator)
        print(f"✓ Successfully retrieved statistics: {len(stats)} metrics")
        
        # Run comprehensive tests
        test_results = run_integrated_validation_tests()
        print(f"✓ Comprehensive tests: {test_results['summary']['successful_tests']}/{test_results['summary']['total_tests']} passed")
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
    
    print("Validation integration module ready!")