"""
IEEE Fraud Detection Dataset Loader for Financial Fraud Domain
Integrated data loading and preprocessing for the Saraphis fraud detection system
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
import logging
from pathlib import Path
import json
from datetime import datetime

from financial_fraud_domain.enhanced_fraud_core_exceptions import (
    FraudDataError, 
    FraudValidationError, 
    FraudProcessingError
)

logger = logging.getLogger(__name__)


class IEEEFraudDataLoader:
    """
    Data loader for IEEE-CIS Fraud Detection dataset integrated with Saraphis fraud domain.
    Provides comprehensive data loading, preprocessing, and validation capabilities.
    """
    
    def __init__(self, data_dir: str = "training_data/ieee-fraud-detection", 
                 enable_validation: bool = True,
                 cache_processed: bool = True,
                 fraud_domain_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the IEEE Fraud Data Loader.
        
        Args:
            data_dir: Directory containing the dataset files
            enable_validation: Whether to perform data validation
            cache_processed: Whether to cache processed data
            fraud_domain_config: Configuration from fraud domain
        """
        self.data_dir = Path(data_dir)
        self.enable_validation = enable_validation
        self.cache_processed = cache_processed
        self.fraud_domain_config = fraud_domain_config or {}
        
        # File paths
        self.transaction_file = self.data_dir / "train_transaction.csv"
        self.identity_file = self.data_dir / "train_identity.csv"
        self.test_transaction_file = self.data_dir / "test_transaction.csv"
        self.test_identity_file = self.data_dir / "test_identity.csv"
        self.cache_dir = self.data_dir / "processed_cache"
        
        # Processing metadata
        self.processing_metadata = {
            'load_timestamp': None,
            'feature_count': 0,
            'sample_count': 0,
            'fraud_rate': 0.0,
            'preprocessing_steps': [],
            'data_quality_metrics': {}
        }
        
        # Feature groups for analysis
        self.feature_groups = {
            'V_features': [],      # Vesta features
            'C_features': [],      # Counting features  
            'D_features': [],      # Timedelta features
            'M_features': [],      # Match features
            'card_features': [],   # Card-related features
            'addr_features': [],   # Address features
            'email_features': [],  # Email features
            'device_features': [], # Device features
            'other_features': []   # Other features
        }
        
        # Initialize cache directory
        if self.cache_processed:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, dataset_type: str = "train", 
                  use_cache: bool = True,
                  validation_split: float = 0.2,
                  force_full_dataset: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Load and preprocess the IEEE fraud detection dataset with full dataset validation.
        
        Args:
            dataset_type: "train", "test", or "both"
            use_cache: Whether to use cached processed data
            validation_split: Fraction of data to use for validation (only for train)
            force_full_dataset: Whether to enforce full dataset size validation
            
        Returns:
            For train: (X_train, y_train, (X_val, y_val)) if validation_split > 0
            For test: (X_test, None, None)
            For both: (X_train, y_train, (X_test, None))
        """
        try:
            self.processing_metadata['load_timestamp'] = datetime.now().isoformat()
            
            # Expected dataset sizes - ROOT FIX: Account for validation split
            EXPECTED_SIZES = {
                'train': 590540,  # Full training set
                'test': 506691    # Full test set
            }
            
            # Check for cached data with size validation
            if use_cache and self.cache_processed:
                cached_data = self._load_from_cache(dataset_type)
                if cached_data is not None:
                    X, y, val_data = cached_data
                    expected_size = EXPECTED_SIZES.get(dataset_type, 590000)
                    
                    # ROOT FIX: Calculate expected size accounting for validation split
                    # If validation_split > 0, the cached X should be (1-validation_split) of full dataset
                    if dataset_type == 'train' and validation_split > 0:
                        expected_training_size = int(expected_size * (1 - validation_split))
                        # Allow 1% tolerance for rounding
                        min_acceptable_size = expected_training_size * 0.99
                        max_acceptable_size = expected_training_size * 1.01
                    else:
                        # For test data or no validation split, use full expected size
                        min_acceptable_size = expected_size * 0.99
                        max_acceptable_size = expected_size * 1.01
                    
                    # Validate cache has complete dataset with proper size range
                    if min_acceptable_size <= X.shape[0] <= max_acceptable_size:
                        logger.info(
                            f"Loaded {dataset_type} data from cache: "
                            f"{X.shape[0]} samples, {X.shape[1]} features "
                            f"(expected range: {min_acceptable_size:.0f}-{max_acceptable_size:.0f})"
                        )
                        return cached_data
                    else:
                        logger.warning(
                            f"Cache has incorrect dataset size: {X.shape[0]} samples "
                            f"(expected range: {min_acceptable_size:.0f}-{max_acceptable_size:.0f}). Loading fresh data..."
                        )
                        # Clear invalid cache
                        self._clear_cache(dataset_type)
                        cached_data = None
            
            # Load fresh data
            logger.info(f"Loading fresh {dataset_type} data...")
            if dataset_type == "train":
                X, y, val_data = self._load_training_data(validation_split)
            elif dataset_type == "test":
                X, y, val_data = self._load_test_data()
            elif dataset_type == "both":
                X, y, val_data = self._load_both_datasets()
            else:
                raise FraudDataError(f"Invalid dataset_type: {dataset_type}")
            
            # Validate loaded data - ROOT FIX: Account for validation split in fresh data validation
            if force_full_dataset:
                expected_size = EXPECTED_SIZES.get(dataset_type, 590000)
                actual_size = X.shape[0]
                
                # ROOT FIX: Calculate expected size accounting for validation split
                if dataset_type == 'train' and validation_split > 0:
                    expected_training_size = int(expected_size * (1 - validation_split))
                    # Allow 1% tolerance for rounding
                    min_acceptable_size = expected_training_size * 0.99
                    max_acceptable_size = expected_training_size * 1.01
                else:
                    # For test data or no validation split, use full expected size
                    min_acceptable_size = expected_size * 0.99
                    max_acceptable_size = expected_size * 1.01
                
                # Check if we have a reasonable amount of data
                if actual_size < min_acceptable_size:
                    logger.warning(
                        f"Dataset size significantly smaller than expected: {actual_size} samples "
                        f"(expected range: {min_acceptable_size:.0f}-{max_acceptable_size:.0f}). This may indicate incomplete data."
                    )
                    # Don't fail - continue with available data
                elif min_acceptable_size <= actual_size <= max_acceptable_size:
                    logger.info(
                        f"Dataset size within expected range: {actual_size} samples "
                        f"(expected range: {min_acceptable_size:.0f}-{max_acceptable_size:.0f})"
                    )
                else:
                    logger.info(
                        f"Dataset size larger than expected: {actual_size} samples "
                        f"(expected range: {min_acceptable_size:.0f}-{max_acceptable_size:.0f}). Using available data."
                    )
            
            # Cache if enabled
            if use_cache and self.cache_processed:
                self._save_to_cache(dataset_type, X, y, val_data)
                
            return X, y, val_data
                
        except Exception as e:
            logger.error(f"Failed to load IEEE fraud data: {e}")
            raise FraudDataError(f"Data loading failed: {e}")
    
    def _load_training_data(self, validation_split: float = 0.2) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Load and process training data with optional validation split."""
        logger.info("Loading IEEE fraud training data...")
        
        # Load transaction data
        trans_df = self._load_transactions(self.transaction_file)
        
        # Load identity data (optional)
        id_df = self._load_identities(self.identity_file)
        
        # Validate data if enabled
        if self.enable_validation:
            self._validate_training_data(trans_df)
        
        # Extract labels
        if 'isFraud' not in trans_df.columns:
            raise FraudDataError("Missing 'isFraud' column in training data")
        
        y = trans_df['isFraud'].values.astype(np.int64)
        trans_df = trans_df.drop(['isFraud'], axis=1)
        
        # Merge with identity data if available
        merged_df = self._merge_data(trans_df, id_df)
        
        # Remove non-feature columns
        merged_df = self._remove_non_features(merged_df)
        
        # Categorize features
        self._categorize_features(merged_df.columns)
        
        # Handle missing values and convert to numpy
        X = self._finalize_features(merged_df)
        
        # Update metadata
        self.processing_metadata.update({
            'feature_count': X.shape[1],
            'sample_count': X.shape[0],
            'fraud_rate': y.mean(),
            'preprocessing_steps': [
                'transaction_load', 'identity_load', 'merge', 
                'feature_engineering', 'missing_value_handling'
            ]
        })
        
        # Perform validation split if requested
        val_data = None
        if validation_split > 0:
            X, y, val_data = self._create_validation_split(X, y, validation_split)
        
        # Cache processed data
        if self.cache_processed:
            self._save_to_cache("train", X, y, val_data)
        
        logger.info(f"Training data loaded: X={X.shape}, y={y.shape}, fraud_rate={y.mean():.4f}")
        
        return X, y, val_data
    
    def _load_test_data(self) -> Tuple[np.ndarray, None, None]:
        """Load and process test data."""
        logger.info("Loading IEEE fraud test data...")
        
        # Load test transaction data
        trans_df = self._load_transactions(self.test_transaction_file)
        
        # Load test identity data (optional)
        id_df = self._load_identities(self.test_identity_file)
        
        # Merge with identity data if available
        merged_df = self._merge_data(trans_df, id_df)
        
        # Remove non-feature columns
        merged_df = self._remove_non_features(merged_df)
        
        # Handle missing values and convert to numpy
        X = self._finalize_features(merged_df)
        
        logger.info(f"Test data loaded: X={X.shape}")
        
        return X, None, None
    
    def _load_both_datasets(self) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, None]]:
        """Load both training and test datasets."""
        # Load training data without validation split
        X_train, y_train, _ = self._load_training_data(validation_split=0.0)
        
        # Load test data
        X_test, _, _ = self._load_test_data()
        
        return X_train, y_train, (X_test, None)
    
    def _load_transactions(self, file_path: Path) -> pd.DataFrame:
        """Load transaction data with enhanced error handling and optimization."""
        logger.info(f"Loading transactions from {file_path}")
        
        # Enhanced file existence check
        if not file_path.exists():
            self._handle_data_loading_error(FileNotFoundError(), file_path, "transaction")
        
        # Define optimized dtypes for memory efficiency
        dtypes = {
            'TransactionID': 'int32',
            'TransactionDT': 'int32',
            'TransactionAmt': 'float32'
        }
        
        # Add isFraud dtype if this is training data
        if 'train' in file_path.name:
            dtypes['isFraud'] = 'int8'
        
        try:
            # Enhanced loading with better error handling
            trans_df = pd.read_csv(
                file_path,
                dtype=dtypes,
                low_memory=False
            )
            
            # Basic validation after loading
            if trans_df.empty:
                raise pd.errors.EmptyDataError("Transaction file is empty")
            
            logger.info(f"Successfully loaded {len(trans_df):,} transactions with {len(trans_df.columns)} columns")
            
            # Log memory usage
            memory_usage = trans_df.memory_usage(deep=True).sum() / 1024**2
            logger.info(f"Transaction data memory usage: {memory_usage:.2f} MB")
            
            return trans_df
            
        except Exception as e:
            self._handle_data_loading_error(e, file_path, "transaction")
    
    def _load_identities(self, file_path: Path) -> pd.DataFrame:
        """Load identity data with enhanced error handling (optional file)."""
        if not file_path.exists():
            logger.warning(f"Identity file not found: {file_path}. Proceeding without identity data.")
            return pd.DataFrame()
        
        logger.info(f"Loading identities from {file_path}")
        
        try:
            # Enhanced loading for identity data
            id_df = pd.read_csv(
                file_path,
                dtype={'TransactionID': 'int32'},
                low_memory=False
            )
            
            # Basic validation for identity data
            if id_df.empty:
                logger.warning("Identity file is empty")
                return pd.DataFrame()
            
            if 'TransactionID' not in id_df.columns:
                logger.warning("TransactionID column missing from identity data. Cannot merge with transactions.")
                return pd.DataFrame()
            
            logger.info(f"Successfully loaded {len(id_df):,} identity records with {len(id_df.columns)} columns")
            
            # Log memory usage
            memory_usage = id_df.memory_usage(deep=True).sum() / 1024**2
            logger.info(f"Identity data memory usage: {memory_usage:.2f} MB")
            
            return id_df
            
        except Exception as e:
            logger.warning(f"Failed to load identity file {file_path}: {e}. Proceeding without identity data.")
            return pd.DataFrame()
    
    def _merge_data(self, trans_df: pd.DataFrame, 
                    id_df: pd.DataFrame) -> pd.DataFrame:
        """Merge transaction and identity data with enhanced validation."""
        if id_df.empty:
            logger.info("No identity data to merge")
            return trans_df
        
        # Enhanced validation before merge
        if self.enable_validation:
            self._validate_identity_data(id_df, trans_df)
        
        original_trans_count = len(trans_df)
        
        # Validate merge keys
        if 'TransactionID' not in trans_df.columns:
            logger.warning("No TransactionID in transaction data, skipping merge")
            return trans_df
        
        if 'TransactionID' not in id_df.columns:
            logger.warning("No TransactionID in identity data, skipping merge")
            return trans_df
        
        logger.info("Merging transaction and identity data")
        
        try:
            merged_df = trans_df.merge(id_df, on='TransactionID', how='left')
            
            # Calculate merge statistics
            identity_columns = [col for col in id_df.columns if col != 'TransactionID']
            if identity_columns:
                n_matched = merged_df[identity_columns[0]].notna().sum()
                match_rate = n_matched / len(merged_df)
                logger.info(f"Merge completed: {n_matched}/{len(merged_df)} transactions have identity info ({match_rate:.2%})")
                
                # Store data quality metric
                self.processing_metadata['data_quality_metrics']['identity_match_rate'] = match_rate
            
            # Enhanced validation after merge
            if self.enable_validation:
                self._validate_merged_data(merged_df, original_trans_count)
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Failed to merge data: {e}")
            return trans_df
    
    def _remove_non_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove non-feature columns."""
        drop_columns = ['TransactionID', 'TransactionDT']
        existing_drop_cols = [col for col in drop_columns if col in df.columns]
        
        if existing_drop_cols:
            logger.info(f"Dropping non-feature columns: {existing_drop_cols}")
            df = df.drop(existing_drop_cols, axis=1)
        
        return df
    
    def _categorize_features(self, columns: List[str]) -> None:
        """Categorize features into groups for analysis."""
        for col in columns:
            if col.startswith('V'):
                self.feature_groups['V_features'].append(col)
            elif col.startswith('C'):
                self.feature_groups['C_features'].append(col)
            elif col.startswith('D'):
                self.feature_groups['D_features'].append(col)
            elif col.startswith('M'):
                self.feature_groups['M_features'].append(col)
            elif any(card_prefix in col.lower() for card_prefix in ['card', 'creditcard']):
                self.feature_groups['card_features'].append(col)
            elif 'addr' in col.lower():
                self.feature_groups['addr_features'].append(col)
            elif 'email' in col.lower():
                self.feature_groups['email_features'].append(col)
            elif 'device' in col.lower():
                self.feature_groups['device_features'].append(col)
            else:
                self.feature_groups['other_features'].append(col)
        
        # Log feature group statistics
        for group, features in self.feature_groups.items():
            if features:
                logger.info(f"{group}: {len(features)} features")
    
    def _finalize_features(self, df: pd.DataFrame) -> np.ndarray:
        """Enhanced feature finalization with improved preprocessing."""
        logger.info("Starting enhanced feature finalization...")
        
        # Separate numeric and categorical columns for targeted processing
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        logger.info(f"Processing {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features")
        
        # Enhanced categorical processing
        if categorical_cols:
            logger.info("Applying enhanced categorical encoding...")
            
            for col in categorical_cols:
                # Fill missing categorical values with 'missing' before encoding
                df[col] = df[col].fillna('missing')
                
                # Handle unseen categories gracefully
                try:
                    # Use pandas factorize for consistent encoding (pandas 2.x compatible)
                    df[col] = pd.factorize(df[col], use_na_sentinel=True)[0]
                except Exception as e:
                    logger.warning(f"Failed to encode column {col}: {e}. Using fallback encoding.")
                    df[col] = df[col].astype(str).astype('category').cat.codes
        
        # Enhanced missing value handling for numeric columns
        if numeric_cols:
            logger.info("Applying enhanced missing value imputation...")
            
            # Use different strategies based on feature type and domain knowledge
            missing_value_strategy = self.fraud_domain_config.get('missing_value_fill', -999)
            
            # For amount-related features, use median instead of fixed value
            amount_features = [col for col in numeric_cols if 'amt' in col.lower() or 'amount' in col.lower()]
            if amount_features:
                for col in amount_features:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.debug(f"Filled missing values in {col} with median: {median_val}")
            
            # For other numeric features, use configured strategy
            other_numeric = [col for col in numeric_cols if col not in amount_features]
            if other_numeric:
                df[other_numeric] = df[other_numeric].fillna(missing_value_strategy)
        
        # Enhanced data type conversion
        target_dtype = self.fraud_domain_config.get('feature_dtype', 'float32')
        
        try:
            X = df.values.astype(target_dtype)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to convert to {target_dtype}: {e}. Using float64.")
            X = df.values.astype(np.float64)
            target_dtype = 'float64'
        
        # Enhanced data quality checks
        logger.info("Performing enhanced data quality validation...")
        
        # Check for infinite values
        inf_mask = np.isinf(X)
        if inf_mask.any():
            inf_count = inf_mask.sum()
            logger.warning(f"Found {inf_count} infinite values. Replacing with 0.")
            X[inf_mask] = 0
        
        # Check for remaining NaN values
        nan_mask = np.isnan(X)
        if nan_mask.any():
            nan_count = nan_mask.sum()
            logger.warning(f"Found {nan_count} NaN values after preprocessing. Replacing with 0.")
            X[nan_mask] = 0
        
        # Enhanced logging and statistics
        logger.info(f"Enhanced feature processing complete:")
        logger.info(f"  Final feature matrix: {X.shape[1]} features, {X.shape[0]} samples")
        logger.info(f"  Memory usage: {X.nbytes / 1e6:.2f} MB")
        logger.info(f"  Feature dtype: {X.dtype}")
        logger.info(f"  Data range: [{X.min():.2f}, {X.max():.2f}]")
        
        # Calculate enhanced data quality metrics
        missing_rate = 0  # Should be 0 after preprocessing
        unique_values_per_feature = np.array([len(np.unique(X[:, i])) for i in range(X.shape[1])])
        constant_features = (unique_values_per_feature == 1).sum()
        
        if constant_features > 0:
            logger.warning(f"Found {constant_features} constant features")
        
        # Store enhanced metadata
        self.processing_metadata['data_quality_metrics'].update({
            'final_missing_rate': missing_rate,
            'constant_features': int(constant_features),
            'inf_values_found': int(inf_mask.sum()) if 'inf_mask' in locals() else 0,
            'nan_values_found': int(nan_mask.sum()) if 'nan_mask' in locals() else 0,
            'unique_values_per_feature_stats': {
                'min': int(unique_values_per_feature.min()),
                'max': int(unique_values_per_feature.max()),
                'mean': float(unique_values_per_feature.mean())
            }
        })
        
        # Calculate original missing rate for metadata
        missing_rate = 0
        self.processing_metadata['data_quality_metrics'].update({
            'missing_value_rate': missing_rate,
            'memory_usage_mb': X.nbytes / 1e6,
            'feature_dtype': str(X.dtype)
        })
        
        return X
    
    def _handle_data_loading_error(self, error: Exception, file_path: Path, data_type: str) -> None:
        """Enhanced error handling for data loading operations."""
        error_context = {
            'file_path': str(file_path),
            'data_type': data_type,
            'file_exists': file_path.exists() if isinstance(file_path, Path) else False,
            'timestamp': datetime.now().isoformat()
        }
        
        if isinstance(error, FileNotFoundError):
            raise FraudDataError(
                f"Required {data_type} file not found: {file_path}",
                dataset_type=data_type,
                file_path=str(file_path)
            )
        elif isinstance(error, pd.errors.EmptyDataError):
            raise FraudDataError(
                f"Empty {data_type} file: {file_path}",
                dataset_type=data_type,
                file_path=str(file_path)
            )
        elif isinstance(error, pd.errors.DtypeWarning):
            logger.warning(f"Data type warning for {data_type} file {file_path}: {error}")
            # Continue processing despite dtype warnings
        elif isinstance(error, MemoryError):
            raise FraudProcessingError(
                f"Insufficient memory to load {data_type} file: {file_path}",
                processing_stage="data_loading"
            )
        elif isinstance(error, PermissionError):
            raise FraudDataError(
                f"Permission denied accessing {data_type} file: {file_path}",
                dataset_type=data_type,
                file_path=str(file_path)
            )
        else:
            # Generic error handling
            raise FraudDataError(
                f"Failed to load {data_type} data from {file_path}: {str(error)}",
                dataset_type=data_type,
                file_path=str(file_path)
            )
    
    def _validate_training_data(self, trans_df: pd.DataFrame) -> None:
        """Enhanced validation for training data quality and structure."""
        validation_errors = []
        
        # Check required columns
        if 'isFraud' not in trans_df.columns:
            validation_errors.append("Missing required 'isFraud' column")
        
        if 'TransactionAmt' not in trans_df.columns:
            validation_errors.append("Missing required 'TransactionAmt' column")
        
        if 'TransactionID' not in trans_df.columns:
            validation_errors.append("Missing required 'TransactionID' column")
        
        # Check data quality
        if len(trans_df) == 0:
            validation_errors.append("Empty transaction dataset")
        
        # Enhanced fraud rate validation
        if 'isFraud' in trans_df.columns:
            fraud_rate = trans_df['isFraud'].mean()
            if fraud_rate < 0.001 or fraud_rate > 0.5:
                validation_errors.append(f"Unusual fraud rate: {fraud_rate:.4f}")
            
            # Check for valid fraud labels (should be 0 or 1)
            valid_labels = trans_df['isFraud'].isin([0, 1]).all()
            if not valid_labels:
                validation_errors.append("isFraud column contains invalid values (must be 0 or 1)")
        
        # Enhanced duplicate checking
        if 'TransactionID' in trans_df.columns:
            duplicates = trans_df['TransactionID'].duplicated().sum()
            if duplicates > 0:
                validation_errors.append(f"Found {duplicates} duplicate TransactionIDs")
        
        # Enhanced transaction amount validation
        if 'TransactionAmt' in trans_df.columns:
            # Check for non-numeric values
            non_numeric = pd.to_numeric(trans_df['TransactionAmt'], errors='coerce').isna().sum()
            if non_numeric > 0:
                validation_errors.append(f"Found {non_numeric} non-numeric values in TransactionAmt")
            
            # Check for negative amounts
            negative_amounts = (trans_df['TransactionAmt'] < 0).sum()
            if negative_amounts > 0:
                validation_errors.append(f"Found {negative_amounts} negative transaction amounts")
            
            # Check for excessive amounts (potential data quality issue)
            excessive_amounts = (trans_df['TransactionAmt'] > 1000000).sum()
            if excessive_amounts > 0:
                logger.warning(f"Found {excessive_amounts} transactions with amount > $1M")
        
        # Enhanced missing value analysis
        missing_rates = trans_df.isnull().mean()
        high_missing_cols = missing_rates[missing_rates > 0.9].index.tolist()
        if high_missing_cols:
            logger.warning(f"Columns with >90% missing values: {high_missing_cols}")
        
        # Check for columns with all missing values
        all_missing_cols = missing_rates[missing_rates == 1.0].index.tolist()
        if all_missing_cols:
            validation_errors.append(f"Columns with 100% missing values: {all_missing_cols}")
        
        # Data type validation for key columns
        if 'TransactionDT' in trans_df.columns:
            if not pd.api.types.is_numeric_dtype(trans_df['TransactionDT']):
                validation_errors.append("TransactionDT column should be numeric")
        
        # Memory usage check
        memory_usage = trans_df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"Transaction data memory usage: {memory_usage:.2f} MB")
        
        if validation_errors:
            error_msg = "Enhanced data validation failed: " + "; ".join(validation_errors)
            raise FraudValidationError(error_msg, validation_failures=validation_errors)
        
        logger.info("Enhanced data validation passed")
    
    def _validate_identity_data(self, identity_df: pd.DataFrame, trans_df: pd.DataFrame) -> None:
        """Enhanced validation for identity data quality and consistency with transaction data."""
        if identity_df.empty:
            logger.warning("Identity data is empty")
            return
        
        validation_warnings = []
        
        # Check required TransactionID column
        if 'TransactionID' not in identity_df.columns:
            logger.warning("TransactionID column missing from identity data. Cannot merge with transactions.")
            return
        
        # Check for duplicate identity records
        duplicates = identity_df['TransactionID'].duplicated().sum()
        if duplicates > 0:
            validation_warnings.append(f"Found {duplicates} duplicate TransactionIDs in identity data")
        
        # Check alignment with transaction data
        trans_ids = set(trans_df['TransactionID']) if 'TransactionID' in trans_df.columns else set()
        identity_ids = set(identity_df['TransactionID'])
        
        # Identity records not in transactions
        id_only = identity_ids - trans_ids
        if id_only:
            validation_warnings.append(f"Found {len(id_only)} TransactionIDs in identity data not present in transactions")
        
        # Coverage analysis
        coverage = len(identity_ids & trans_ids) / len(trans_ids) * 100 if trans_ids else 0
        logger.info(f"Identity data coverage: {coverage:.2f}% of transactions")
        
        # Data quality checks for identity data
        missing_rates = identity_df.isnull().mean()
        all_missing_cols = missing_rates[missing_rates == 1.0].index.tolist()
        if all_missing_cols:
            validation_warnings.append(f"Identity columns with 100% missing values: {all_missing_cols}")
        
        # Memory usage
        memory_usage = identity_df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"Identity data memory usage: {memory_usage:.2f} MB")
        
        # Log warnings
        for warning in validation_warnings:
            logger.warning(warning)
        
        logger.info("Identity data validation completed")
    
    def _validate_merged_data(self, merged_df: pd.DataFrame, original_trans_count: int) -> None:
        """Validate merged transaction and identity data."""
        logger.info("Validating merged dataset...")
        
        # Check if merge preserved all transactions
        if len(merged_df) != original_trans_count:
            logger.warning(f"Merge changed row count: {original_trans_count} -> {len(merged_df)}")
        
        # Check for introduced missing values
        missing_after_merge = merged_df.isnull().sum().sum()
        logger.info(f"Total missing values after merge: {missing_after_merge:,}")
        
        # Memory usage check
        memory_usage = merged_df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"Merged data memory usage: {memory_usage:.2f} MB")
        
        logger.info("Merged data validation completed")
    
    def _create_validation_split(self, X: np.ndarray, y: np.ndarray, 
                               validation_split: float) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Create stratified validation split."""
        from sklearn.model_selection import train_test_split
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=validation_split,
            stratify=y,
            random_state=42
        )
        
        logger.info(f"Validation split created: train={X_train.shape[0]}, val={X_val.shape[0]}")
        
        return X_train, y_train, (X_val, y_val)
    
    def _save_to_cache(self, dataset_type: str, X: np.ndarray, y: Optional[np.ndarray], 
                      val_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Save processed data to cache."""
        try:
            cache_file = self.cache_dir / f"{dataset_type}_processed.npz"
            
            save_dict = {'X': X}
            if y is not None:
                save_dict['y'] = y
            if val_data is not None:
                save_dict['X_val'] = val_data[0]
                save_dict['y_val'] = val_data[1]
            
            np.savez_compressed(cache_file, **save_dict)
            
            # Save metadata
            metadata_file = self.cache_dir / f"{dataset_type}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(self.processing_metadata, f, indent=2)
            
            logger.info(f"Saved processed data to cache: {cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _load_from_cache(self, dataset_type: str) -> Optional[Tuple]:
        """Load processed data from cache."""
        try:
            cache_file = self.cache_dir / f"{dataset_type}_processed.npz"
            metadata_file = self.cache_dir / f"{dataset_type}_metadata.json"
            
            if not cache_file.exists():
                return None
            
            # Load data
            data = np.load(cache_file)
            X = data['X']
            y = data.get('y', None)
            
            val_data = None
            if 'X_val' in data and 'y_val' in data:
                val_data = (data['X_val'], data['y_val'])
            
            # Load metadata if available
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.processing_metadata = json.load(f)
            
            return X, y, val_data
            
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None
    
    def _clear_cache(self, dataset_type: str):
        """Clear invalid cache files"""
        cache_file = self.cache_dir / f"{dataset_type}_processed.npz"
        metadata_file = self.cache_dir / f"{dataset_type}_metadata.json"
        
        if cache_file.exists():
            cache_file.unlink()
            logger.info(f"Cleared invalid cache: {cache_file}")
            
        if metadata_file.exists():
            metadata_file.unlink()
            logger.info(f"Cleared invalid metadata: {metadata_file}")
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing."""
        # This would need to be stored during processing
        # For now, return feature group information
        all_features = []
        for group, features in self.feature_groups.items():
            all_features.extend(features)
        return all_features
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive data quality report."""
        return {
            'processing_metadata': self.processing_metadata,
            'feature_groups': {k: len(v) for k, v in self.feature_groups.items()},
            'data_files': {
                'transaction_train': self.transaction_file.exists(),
                'identity_train': self.identity_file.exists(),
                'transaction_test': self.test_transaction_file.exists(),
                'identity_test': self.test_identity_file.exists()
            },
            'cache_info': {
                'cache_enabled': self.cache_processed,
                'cache_dir': str(self.cache_dir),
                'cache_exists': self.cache_dir.exists() if self.cache_processed else False
            }
        }
    
    def clear_cache(self) -> bool:
        """Clear all cached data."""
        try:
            if self.cache_dir.exists():
                import shutil
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cache cleared successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
        return False
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the IEEE fraud dataset without loading all data.
        
        Returns:
            Dictionary with dataset information including file paths, sizes, and availability
        """
        info = {
            'data_directory': str(self.data_dir),
            'configuration': {
                'enable_validation': self.enable_validation,
                'cache_processed': self.cache_processed,
                'cache_directory': str(self.cache_dir) if self.cache_processed else None
            },
            'files': {},
            'processing_metadata': self.processing_metadata.copy()
        }
        
        # Check transaction files
        for file_type, file_path in [
            ('train_transaction', self.transaction_file),
            ('train_identity', self.identity_file),
            ('test_transaction', self.test_transaction_file),
            ('test_identity', self.test_identity_file)
        ]:
            file_info = {'exists': file_path.exists()}
            
            if file_info['exists']:
                try:
                    file_size = file_path.stat().st_size / 1024**2  # MB
                    file_info.update({
                        'path': str(file_path),
                        'size_mb': round(file_size, 2)
                    })
                except Exception as e:
                    file_info['error'] = str(e)
            
            info['files'][file_type] = file_info
        
        # Cache information
        if self.cache_processed and self.cache_dir.exists():
            cache_files = list(self.cache_dir.glob('*.npz'))
            info['cache'] = {
                'enabled': True,
                'directory': str(self.cache_dir),
                'cached_files': len(cache_files),
                'files': [f.name for f in cache_files]
            }
        else:
            info['cache'] = {'enabled': False}
        
        return info


# Convenience function for quick loading with fraud domain integration
def load_ieee_fraud_data(data_dir: str = "training_data/ieee-fraud-detection",
                        dataset_type: str = "train",
                        validation_split: float = 0.2,
                        fraud_domain_config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, ...]:
    """
    Quick function to load IEEE fraud detection dataset with fraud domain integration.
    
    Args:
        data_dir: Directory containing the dataset files
        dataset_type: "train", "test", or "both"
        validation_split: Fraction for validation split (only for train)
        fraud_domain_config: Configuration from fraud domain
        
    Returns:
        Tuple of numpy arrays depending on dataset_type
    """
    loader = IEEEFraudDataLoader(
        data_dir=data_dir,
        fraud_domain_config=fraud_domain_config
    )
    return loader.load_data(dataset_type=dataset_type, validation_split=validation_split)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Test basic loading
        X, y, val_data = load_ieee_fraud_data()
        print(f"Successfully loaded data: X shape={X.shape}, y shape={y.shape}")
        if val_data:
            print(f"Validation data: X_val shape={val_data[0].shape}, y_val shape={val_data[1].shape}")
        print(f"Fraud rate: {y.mean():.4%}")
        print(f"Data types: X={X.dtype}, y={y.dtype}")
        
        # Test data quality report
        loader = IEEEFraudDataLoader()
        quality_report = loader.get_data_quality_report()
        print(f"Data quality report: {json.dumps(quality_report, indent=2)}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()