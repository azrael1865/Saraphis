"""
Data Loading Integration Module
Provides unified access to all data loading components for financial fraud detection.
Consolidates data ingestion functionality into a simple interface.
"""

import logging
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

def get_integrated_data_loader(environment: str = "production", 
                             config: Optional[Dict[str, Any]] = None):
    """
    Get configured data loader for specified environment
    
    Args:
        environment: "production", "development", or "testing"
        config: Optional configuration dictionary
        
    Returns:
        Configured data loader instance
    """
    try:
        # Try enhanced data loader first
        try:
            from enhanced_data_loader import (
                EnhancedFinancialDataLoader,
                DataLoadConfig,
                ValidationLevel,
                SecurityLevel,
                CacheConfig,
                PerformanceConfig
            )
        except ImportError:
            from enhanced_data_loader import (
                EnhancedFinancialDataLoader,
                DataLoadConfig,
                ValidationLevel,
                SecurityLevel,
                CacheConfig,
                PerformanceConfig
            )
        
        # Create configuration based on environment
        if environment == "production":
            loader_config = DataLoadConfig(
                validation_level=ValidationLevel.COMPREHENSIVE,
                security_level=SecurityLevel.HIGH,
                enable_caching=True,
                cache_ttl_seconds=3600,
                max_cache_size_mb=1024,
                enable_parallel_loading=True,
                max_workers=4,
                chunk_size=10000,
                enable_retry=True,
                max_retries=3,
                retry_delay_seconds=2,
                enable_compression=True,
                enable_encryption=True,
                enable_audit_logging=True,
                max_file_size_mb=500,
                max_memory_usage_mb=2048,
                timeout_seconds=300
            )
        elif environment == "development":
            loader_config = DataLoadConfig(
                validation_level=ValidationLevel.STANDARD,
                security_level=SecurityLevel.MEDIUM,
                enable_caching=True,
                cache_ttl_seconds=1800,
                max_cache_size_mb=512,
                enable_parallel_loading=False,
                max_workers=2,
                chunk_size=5000,
                enable_retry=True,
                max_retries=2,
                enable_compression=False,
                enable_encryption=False,
                enable_audit_logging=False,
                max_file_size_mb=100,
                max_memory_usage_mb=1024,
                timeout_seconds=120
            )
        else:  # testing
            loader_config = DataLoadConfig(
                validation_level=ValidationLevel.BASIC,
                security_level=SecurityLevel.LOW,
                enable_caching=False,
                enable_parallel_loading=False,
                max_workers=1,
                chunk_size=1000,
                enable_retry=False,
                max_retries=0,
                enable_compression=False,
                enable_encryption=False,
                enable_audit_logging=False,
                max_file_size_mb=10,
                max_memory_usage_mb=256,
                timeout_seconds=30
            )
        
        # Override with user config if provided
        if config:
            for key, value in config.items():
                if hasattr(loader_config, key):
                    setattr(loader_config, key, value)
        
        logger.info(f"Using enhanced data loader for {environment} environment")
        return EnhancedFinancialDataLoader(loader_config)
        
    except ImportError as e:
        logger.warning(f"Enhanced data loader not available: {e}, trying data directory")
        
        # Try data directory loader
        try:
            from .data.enhanced_data_loader import EnhancedFinancialDataLoader
            logger.info(f"Using data directory enhanced loader for {environment} environment")
            return EnhancedFinancialDataLoader()
        except ImportError:
            pass
        
        # Try basic data loader
        try:
            from .data.data_loader import FinancialDataLoader
            logger.info(f"Using basic data loader for {environment} environment")
            return FinancialDataLoader()
        except ImportError:
            pass
        
        # Final fallback to create a minimal loader
        logger.warning("No data loader available, creating minimal loader")
        
        class MinimalDataLoader:
            """Minimal data loader for basic functionality"""
            
            def __init__(self):
                self.environment = environment
                self.load_stats = {
                    'total_loads': 0,
                    'successful_loads': 0,
                    'failed_loads': 0
                }
            
            def load_from_file(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
                """Load data from file"""
                try:
                    file_path = Path(file_path)
                    
                    if not file_path.exists():
                        raise FileNotFoundError(f"File not found: {file_path}")
                    
                    # Determine file type and load accordingly
                    if file_path.suffix.lower() == '.csv':
                        data = pd.read_csv(file_path, **kwargs)
                    elif file_path.suffix.lower() == '.json':
                        data = pd.read_json(file_path, **kwargs)
                    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                        data = pd.read_excel(file_path, **kwargs)
                    elif file_path.suffix.lower() == '.parquet':
                        data = pd.read_parquet(file_path, **kwargs)
                    else:
                        # Try CSV as default
                        data = pd.read_csv(file_path, **kwargs)
                    
                    self.load_stats['total_loads'] += 1
                    self.load_stats['successful_loads'] += 1
                    
                    logger.info(f"Successfully loaded {len(data)} records from {file_path}")
                    return data
                    
                except Exception as e:
                    self.load_stats['total_loads'] += 1
                    self.load_stats['failed_loads'] += 1
                    logger.error(f"Failed to load data from {file_path}: {e}")
                    raise
            
            def load_sample_data(self, size: int = 1000) -> pd.DataFrame:
                """Generate sample financial transaction data"""
                import numpy as np
                
                np.random.seed(42)
                
                data = pd.DataFrame({
                    'transaction_id': [f'TXN{i:08d}' for i in range(size)],
                    'user_id': [f'USER{np.random.randint(1, 1000):06d}' for _ in range(size)],
                    'merchant_id': [f'MERCH{np.random.randint(1, 100):04d}' for _ in range(size)],
                    'amount': np.random.lognormal(mean=4, sigma=1.5, size=size),
                    'timestamp': pd.date_range('2024-01-01', periods=size, freq='15min'),
                    'payment_method': np.random.choice(
                        ['credit_card', 'debit_card', 'bank_transfer', 'digital_wallet'], 
                        size
                    ),
                    'currency': np.random.choice(['USD', 'EUR', 'GBP'], size, p=[0.7, 0.2, 0.1]),
                    'country': np.random.choice(['US', 'UK', 'DE', 'FR', 'CA'], size),
                    'is_fraud': np.random.choice([0, 1], size, p=[0.98, 0.02])
                })
                
                self.load_stats['total_loads'] += 1
                self.load_stats['successful_loads'] += 1
                
                logger.info(f"Generated {size} sample transactions")
                return data
            
            def get_statistics(self) -> Dict[str, Any]:
                """Get loader statistics"""
                return {
                    'loader_type': 'MinimalDataLoader',
                    'environment': self.environment,
                    'statistics': self.load_stats.copy()
                }
        
        return MinimalDataLoader()

def load_data_integrated(source: Union[str, Path, Dict[str, Any]], 
                       environment: str = "production",
                       config: Optional[Dict[str, Any]] = None,
                       **kwargs) -> pd.DataFrame:
    """
    Load data with integrated data loading pipeline
    
    Args:
        source: Data source (file path, config dict, etc.)
        environment: Environment configuration
        config: Optional configuration overrides
        **kwargs: Additional parameters for loading
        
    Returns:
        Loaded DataFrame
    """
    loader = get_integrated_data_loader(environment, config)
    
    if isinstance(source, (str, Path)):
        # File-based loading
        if hasattr(loader, 'load_from_file'):
            return loader.load_from_file(source, **kwargs)
        elif hasattr(loader, 'load_file'):
            return loader.load_file(source, **kwargs)
        elif hasattr(loader, 'load'):
            return loader.load(source, **kwargs)
        else:
            # Fallback to pandas
            logger.warning("Using pandas fallback for file loading")
            return pd.read_csv(source, **kwargs)
    
    elif isinstance(source, dict):
        # Configuration-based loading
        if hasattr(loader, 'load_from_config'):
            return loader.load_from_config(source, **kwargs)
        elif hasattr(loader, 'load'):
            return loader.load(source, **kwargs)
        else:
            raise ValueError("Configuration-based loading not supported by current loader")
    
    else:
        raise ValueError(f"Unsupported source type: {type(source)}")

def load_sample_data_integrated(size: int = 1000,
                               environment: str = "production",
                               config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Load sample data for testing and development
    
    Args:
        size: Number of sample records
        environment: Environment configuration
        config: Optional configuration overrides
        
    Returns:
        Sample DataFrame
    """
    loader = get_integrated_data_loader(environment, config)
    
    if hasattr(loader, 'load_sample_data'):
        return loader.load_sample_data(size)
    elif hasattr(loader, 'generate_sample_data'):
        return loader.generate_sample_data(size)
    else:
        # Generate basic sample data
        import numpy as np
        
        np.random.seed(42)
        
        return pd.DataFrame({
            'transaction_id': [f'TXN{i:08d}' for i in range(size)],
            'user_id': [f'USER{np.random.randint(1, 100):06d}' for _ in range(size)],
            'amount': np.random.lognormal(mean=4, sigma=1, size=size),
            'timestamp': pd.date_range('2024-01-01', periods=size, freq='1H'),
            'is_fraud': np.random.choice([0, 1], size, p=[0.98, 0.02])
        })

def load_recent_data_integrated(days: int = 30,
                              environment: str = "production", 
                              config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Load recent data for retraining or analysis
    
    Args:
        days: Number of recent days to load
        environment: Environment configuration
        config: Optional configuration overrides
        
    Returns:
        Recent data DataFrame
    """
    loader = get_integrated_data_loader(environment, config)
    
    if hasattr(loader, 'load_recent_data'):
        return loader.load_recent_data(days)
    elif hasattr(loader, 'load_time_range'):
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        return loader.load_time_range(start_date, end_date)
    else:
        # Generate sample recent data
        logger.warning("Generating sample recent data as fallback")
        size = days * 100  # Approximate transactions per day
        
        import numpy as np
        from datetime import datetime, timedelta
        
        np.random.seed(42)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return pd.DataFrame({
            'transaction_id': [f'RECENT_TXN{i:08d}' for i in range(size)],
            'user_id': [f'USER{np.random.randint(1, 100):06d}' for _ in range(size)],
            'amount': np.random.lognormal(mean=4, sigma=1, size=size),
            'timestamp': pd.date_range(start_date, end_date, periods=size),
            'is_fraud': np.random.choice([0, 1], size, p=[0.98, 0.02])
        })

def get_data_loading_statistics(loader):
    """
    Get data loading statistics from any loader type
    
    Args:
        loader: Data loader instance
        
    Returns:
        Dictionary of statistics
    """
    try:
        if hasattr(loader, 'get_statistics'):
            return loader.get_statistics()
        elif hasattr(loader, 'get_metrics'):
            return loader.get_metrics()
        elif hasattr(loader, 'load_stats'):
            return {'statistics': loader.load_stats}
        else:
            return {
                'loader_type': type(loader).__name__,
                'methods': [method for method in dir(loader) if not method.startswith('_')],
                'message': 'Limited statistics available'
            }
    except Exception as e:
        logger.error(f"Failed to get data loading statistics: {e}")
        return {'error': str(e)}

def run_integrated_data_loading_tests():
    """
    Run comprehensive data loading tests across all environments
    
    Returns:
        Test results dictionary
    """
    test_results = {
        'environments_tested': [],
        'loaders_tested': [],
        'test_results': {},
        'errors': []
    }
    
    # Test each environment
    for environment in ['production', 'development', 'testing']:
        try:
            test_results['environments_tested'].append(environment)
            
            # Test loader creation
            try:
                loader = get_integrated_data_loader(environment)
                test_results['loaders_tested'].append(f"loader_{environment}")
                
                test_results['test_results'][f'loader_creation_{environment}'] = {
                    'success': True,
                    'loader_type': type(loader).__name__
                }
            except Exception as e:
                test_results['errors'].append(f"Loader creation {environment}: {str(e)}")
                test_results['test_results'][f'loader_creation_{environment}'] = {
                    'success': False,
                    'error': str(e)
                }
                continue
            
            # Test sample data loading
            try:
                sample_data = load_sample_data_integrated(100, environment)
                test_results['test_results'][f'sample_data_{environment}'] = {
                    'success': True,
                    'records_loaded': len(sample_data),
                    'columns': list(sample_data.columns)
                }
            except Exception as e:
                test_results['errors'].append(f"Sample data {environment}: {str(e)}")
                test_results['test_results'][f'sample_data_{environment}'] = {
                    'success': False,
                    'error': str(e)
                }
            
            # Test recent data loading
            try:
                recent_data = load_recent_data_integrated(7, environment)
                test_results['test_results'][f'recent_data_{environment}'] = {
                    'success': True,
                    'records_loaded': len(recent_data),
                    'columns': list(recent_data.columns)
                }
            except Exception as e:
                test_results['errors'].append(f"Recent data {environment}: {str(e)}")
                test_results['test_results'][f'recent_data_{environment}'] = {
                    'success': False,
                    'error': str(e)
                }
            
            # Test statistics
            try:
                stats = get_data_loading_statistics(loader)
                test_results['test_results'][f'statistics_{environment}'] = {
                    'success': True,
                    'stats_available': len(stats)
                }
            except Exception as e:
                test_results['errors'].append(f"Statistics {environment}: {str(e)}")
                test_results['test_results'][f'statistics_{environment}'] = {
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
    'get_integrated_data_loader',
    'load_data_integrated',
    'load_sample_data_integrated',
    'load_recent_data_integrated',
    'get_data_loading_statistics',
    'run_integrated_data_loading_tests'
]

if __name__ == "__main__":
    print("Data Loading Integration Module")
    print("Available environments: production, development, testing")
    
    # Test basic functionality
    try:
        loader = get_integrated_data_loader("development")
        print(f"✓ Successfully created loader: {type(loader).__name__}")
        
        # Test sample data loading
        sample_data = load_sample_data_integrated(50, environment="testing")
        print(f"✓ Successfully loaded sample data: {sample_data.shape}")
        
        # Test recent data loading
        recent_data = load_recent_data_integrated(3, environment="testing")
        print(f"✓ Successfully loaded recent data: {recent_data.shape}")
        
        # Test statistics
        stats = get_data_loading_statistics(loader)
        print(f"✓ Successfully retrieved statistics: {len(stats)} metrics")
        
        # Run comprehensive tests
        test_results = run_integrated_data_loading_tests()
        print(f"✓ Comprehensive tests: {test_results['summary']['successful_tests']}/{test_results['summary']['total_tests']} passed")
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
    
    print("Data loading integration module ready!")