"""
Preprocessing Integration Module
Provides unified access to preprocessing components for financial fraud detection.
Consolidates all preprocessing functionality into a simple interface.
"""

import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

def get_integrated_preprocessor(environment: str = "production", 
                              config: Optional[Dict[str, Any]] = None):
    """
    Get configured preprocessor for specified environment
    
    Args:
        environment: "production", "development", or "testing"
        config: Optional configuration dictionary
        
    Returns:
        Configured preprocessor instance
    """
    try:
        # Try enhanced preprocessor first
        try:
            from enhanced_data_preprocessor import (
                EnhancedFinancialDataPreprocessor, 
                PreprocessingConfig,
                ValidationConfig,
                SecurityConfig
            )
        except ImportError:
            from enhanced_data_preprocessor import (
                EnhancedFinancialDataPreprocessor, 
                PreprocessingConfig,
                ValidationConfig,
                SecurityConfig
            )
        
        # Create configuration based on environment
        if environment == "production":
            preprocessing_config = PreprocessingConfig(
                handle_missing=True,
                missing_strategy="fill",
                remove_duplicates=True,
                outlier_detection=True,
                outlier_method="iqr",
                create_time_features=True,
                create_aggregation_features=True,
                create_ratio_features=True,
                create_lag_features=True,
                feature_selection=True,
                selection_method="mutual_info",
                max_features=50,
                categorical_encoding="onehot",
                max_categories=20,
                validation_config=ValidationConfig(
                    validate_inputs=True,
                    validate_outputs=True,
                    check_data_consistency=True,
                    check_feature_ranges=True
                ),
                security_config=SecurityConfig(
                    max_memory_mb=4096.0,
                    max_execution_time_seconds=600.0,
                    protect_sensitive_features=True
                ),
                enable_error_recovery=True,
                fallback_strategy="partial"
            )
        elif environment == "development":
            preprocessing_config = PreprocessingConfig(
                handle_missing=True,
                missing_strategy="fill",
                create_time_features=True,
                create_aggregation_features=True,
                feature_selection=True,
                max_features=30,
                validation_config=ValidationConfig(
                    validate_inputs=True,
                    validate_outputs=True
                ),
                security_config=SecurityConfig(
                    max_memory_mb=2048.0,
                    max_execution_time_seconds=300.0
                ),
                enable_error_recovery=True
            )
        else:  # testing
            preprocessing_config = PreprocessingConfig(
                handle_missing=True,
                missing_strategy="drop",
                create_time_features=False,
                create_aggregation_features=False,
                feature_selection=False,
                validation_config=ValidationConfig(
                    validate_inputs=False,
                    validate_outputs=False
                ),
                enable_error_recovery=True
            )
        
        # Override with user config if provided
        if config:
            for key, value in config.items():
                if hasattr(preprocessing_config, key):
                    setattr(preprocessing_config, key, value)
        
        logger.info(f"Using enhanced preprocessor for {environment} environment")
        return EnhancedFinancialDataPreprocessor(preprocessing_config)
        
    except ImportError as e:
        logger.warning(f"Enhanced preprocessor not available: {e}, falling back to standard")
        
        # Fallback to standard preprocessor
        try:
            from data_preprocessor import FinancialDataPreprocessor, PreprocessingConfig
            
            # Create simplified configuration
            if environment == "production":
                preprocessing_config = PreprocessingConfig(
                    handle_missing=True,
                    missing_strategy="fill",
                    remove_duplicates=True,
                    outlier_detection=True,
                    create_time_features=True,
                    create_aggregation_features=True,
                    create_ratio_features=True,
                    feature_selection=True,
                    max_features=50
                )
            elif environment == "development":
                preprocessing_config = PreprocessingConfig(
                    handle_missing=True,
                    create_time_features=True,
                    create_aggregation_features=True,
                    feature_selection=True,
                    max_features=30
                )
            else:  # testing
                preprocessing_config = PreprocessingConfig(
                    handle_missing=True,
                    missing_strategy="drop",
                    create_time_features=False,
                    feature_selection=False
                )
            
            # Override with user config
            if config:
                for key, value in config.items():
                    if hasattr(preprocessing_config, key):
                        setattr(preprocessing_config, key, value)
            
            logger.info(f"Using standard preprocessor for {environment} environment")
            return FinancialDataPreprocessor(preprocessing_config)
            
        except ImportError as e:
            logger.error(f"No preprocessor available: {e}")
            raise ImportError("No preprocessing components available")

def process_data_integrated(data, target=None, environment: str = "production", 
                          config: Optional[Dict[str, Any]] = None):
    """
    Process data with integrated preprocessing pipeline
    
    Args:
        data: Input DataFrame
        target: Optional target variable
        environment: Environment configuration
        config: Optional configuration overrides
        
    Returns:
        Preprocessed DataFrame
    """
    preprocessor = get_integrated_preprocessor(environment, config)
    return preprocessor.fit_transform(data, target)

def create_fraud_features_integrated(data, environment: str = "production"):
    """
    Create fraud-specific features using integrated preprocessor
    
    Args:
        data: Input DataFrame
        environment: Environment configuration
        
    Returns:
        DataFrame with fraud features
    """
    try:
        preprocessor = get_integrated_preprocessor(environment)
        
        # Check if enhanced preprocessor has fraud feature creation
        if hasattr(preprocessor, 'create_fraud_features'):
            return preprocessor.create_fraud_features(data)
        
        # Fallback to basic fraud features
        fraud_data = data.copy()
        
        # Basic fraud indicators
        if 'amount' in data.columns:
            fraud_data['high_amount_flag'] = (fraud_data['amount'] > fraud_data['amount'].quantile(0.95)).astype(int)
            fraud_data['round_amount_flag'] = (fraud_data['amount'] % 100 == 0).astype(int)
        
        if 'timestamp' in data.columns:
            import pandas as pd
            timestamp = pd.to_datetime(fraud_data['timestamp'], errors='coerce')
            fraud_data['night_transaction'] = ((timestamp.dt.hour >= 23) | (timestamp.dt.hour <= 5)).astype(int)
            fraud_data['weekend_transaction'] = timestamp.dt.dayofweek.isin([5, 6]).astype(int)
        
        return fraud_data
        
    except Exception as e:
        logger.error(f"Fraud feature creation failed: {e}")
        return data

def get_preprocessing_statistics(preprocessor):
    """
    Get preprocessing statistics from any preprocessor type
    
    Args:
        preprocessor: Fitted preprocessor instance
        
    Returns:
        Dictionary of statistics
    """
    try:
        if hasattr(preprocessor, 'get_statistics'):
            return preprocessor.get_statistics()
        elif hasattr(preprocessor, 'preprocessing_stats'):
            return preprocessor.preprocessing_stats
        else:
            return {
                'fitted': getattr(preprocessor, 'fitted', False),
                'feature_count': len(getattr(preprocessor, 'feature_names', [])),
                'message': 'Limited statistics available'
            }
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        return {'error': str(e)}

# Export main functions
__all__ = [
    'get_integrated_preprocessor',
    'process_data_integrated', 
    'create_fraud_features_integrated',
    'get_preprocessing_statistics'
]

if __name__ == "__main__":
    print("Preprocessing Integration Module")
    print("Available environments: production, development, testing")
    
    # Test basic functionality
    try:
        preprocessor = get_integrated_preprocessor("development")
        print(f"✓ Successfully created preprocessor: {type(preprocessor).__name__}")
        
        # Test with sample data
        import pandas as pd
        import numpy as np
        
        sample_data = pd.DataFrame({
            'amount': np.random.lognormal(4, 1, 100),
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'user_id': [f'USER{i%10:03d}' for i in range(100)],
            'merchant_id': [f'MERCH{i%5:03d}' for i in range(100)]
        })
        
        processed_data = process_data_integrated(sample_data, environment="testing")
        print(f"✓ Successfully processed data: {sample_data.shape} -> {processed_data.shape}")
        
        # Test fraud features
        fraud_data = create_fraud_features_integrated(sample_data, environment="testing")
        print(f"✓ Successfully created fraud features: {fraud_data.shape}")
        
        # Test statistics
        stats = get_preprocessing_statistics(preprocessor)
        print(f"✓ Successfully retrieved statistics: {len(stats)} metrics")
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
    
    print("Preprocessing integration module ready!")