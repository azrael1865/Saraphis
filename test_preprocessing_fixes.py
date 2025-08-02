#!/usr/bin/env python3
"""
Test script to validate Brain data preprocessing fixes for IEEE fraud detection dataset.
"""

import pandas as pd
import numpy as np
import sys
import os
import logging
from typing import Dict, Any

# Add the independent_core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'independent_core'))

try:
    from brain import Brain, BrainSystemConfig
except ImportError as e:
    print(f"Failed to import Brain modules: {e}")
    print("Make sure you're running this from the Saraphis directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_ieee_data():
    """Create sample IEEE fraud detection data for testing."""
    n_samples = 1000
    
    # Create sample transaction data
    trans_data = {
        'TransactionID': list(range(1, n_samples + 1)),
        'TransactionDT': np.random.randint(86400, 15811131, n_samples),
        'TransactionAmt': np.random.exponential(50, n_samples),
        'ProductCD': np.random.choice(['W', 'H', 'C', 'S', 'R'], n_samples),
        'card1': np.random.randint(1000, 20000, n_samples),
        'card2': np.random.choice([111.0, 222.0, 333.0, np.nan], n_samples),
        'card3': np.random.choice([150.0, 185.0, np.nan], n_samples),
        'card4': np.random.choice(['discover', 'mastercard', 'visa', 'american express', np.nan], n_samples),
        'card5': np.random.choice([226.0, 166.0, 117.0, np.nan], n_samples),
        'card6': np.random.choice(['debit', 'credit', 'charge card', np.nan], n_samples),
        'addr1': np.random.choice([204.0, 325.0, np.nan], n_samples),
        'addr2': np.random.choice([87.0, np.nan], n_samples),
        'P_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', np.nan], n_samples),
        'R_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', np.nan], n_samples),
        'M1': np.random.choice(['T', 'F', np.nan], n_samples),
        'M2': np.random.choice(['T', 'F', np.nan], n_samples),
        'M3': np.random.choice(['T', 'F', np.nan], n_samples),
        'isFraud': np.random.choice([0, 1], n_samples, p=[0.965, 0.035])  # 3.5% fraud rate
    }
    
    # Add V columns (Vesta engineered features)
    for i in range(1, 21):  # V1 to V20
        trans_data[f'V{i}'] = np.random.randn(n_samples)
        # Add some NaN values
        nan_mask = np.random.random(n_samples) < 0.1
        trans_data[f'V{i}'][nan_mask] = np.nan
    
    # Add C columns (counting features)
    for i in range(1, 6):  # C1 to C5
        trans_data[f'C{i}'] = np.random.randint(0, 10, n_samples)
    
    # Add D columns (timedelta features)
    for i in range(1, 6):  # D1 to D5
        trans_data[f'D{i}'] = np.random.randint(0, 500, n_samples).astype(float)
        # Add some NaN values
        nan_mask = np.random.random(n_samples) < 0.15
        trans_data[f'D{i}'][nan_mask] = np.nan
    
    trans_df = pd.DataFrame(trans_data)
    
    # Create sample identity data
    id_data = {
        'TransactionID': list(range(1, n_samples // 2 + 1)),  # Only half have identity info
        'id_01': np.random.randint(-10, 100, n_samples // 2).astype(float),
        'id_02': np.random.randint(10000, 700000, n_samples // 2).astype(float),
        'id_03': np.random.choice([0.0, 1.0, np.nan], n_samples // 2),
        'id_04': np.random.choice([0.0, 1.0, np.nan], n_samples // 2),
        'id_05': np.random.randint(0, 50, n_samples // 2).astype(float),
        'id_06': np.random.randint(0, 50, n_samples // 2).astype(float),
        'DeviceType': np.random.choice(['desktop', 'mobile', 'tablet', np.nan], n_samples // 2),
        'DeviceInfo': np.random.choice(['Windows', 'MacOS', 'iOS', 'Android', np.nan], n_samples // 2),
    }
    
    id_df = pd.DataFrame(id_data)
    
    return trans_df, id_df


def test_data_preprocessing():
    """Test the data preprocessing pipeline."""
    logger.info("=" * 80)
    logger.info("Testing Brain Data Preprocessing Pipeline")
    logger.info("=" * 80)
    
    # Create sample data
    logger.info("\\n1. Creating sample IEEE fraud detection data...")
    trans_df, id_df = create_sample_ieee_data()
    logger.info(f"   Transaction data shape: {trans_df.shape}")
    logger.info(f"   Identity data shape: {id_df.shape}")
    logger.info(f"   Fraud rate: {trans_df['isFraud'].mean():.2%}")
    
    # Initialize Brain
    logger.info("\\n2. Initializing Brain system...")
    config = BrainSystemConfig(
        enable_monitoring=True,
        enable_parallel_predictions=False,
        max_prediction_threads=1
    )
    
    brain = Brain(config)
    logger.info("   Brain initialized successfully")
    
    # Add fraud detection domain
    logger.info("\\n3. Adding fraud detection domain...")
    domain_result = brain.add_domain(
        'fraud_detection',
        {
            'type': 'specialized',
            'description': 'Fraud detection domain',
            'hidden_layers': [128, 64, 32],
            'learning_rate': 0.001
        }
    )
    logger.info(f"   Domain added: {domain_result['success']}")
    
    # Test data preparation
    logger.info("\\n4. Testing data preparation...")
    try:
        # Prepare data in the expected format
        train_data = {
            'transactions': trans_df,
            'identities': id_df
        }
        
        # Test the preprocessing directly
        X, y = brain._prepare_fraud_detection_data(train_data)
        logger.info(f"   ✓ Data preparation successful!")
        logger.info(f"   Features shape: {X.shape}")
        logger.info(f"   Labels shape: {y.shape}")
        logger.info(f"   Feature dtype: {X.dtype}")
        logger.info(f"   Label dtype: {y.dtype}")
        logger.info(f"   Feature range: [{X.min():.2f}, {X.max():.2f}]")
        logger.info(f"   NaN count in features: {np.isnan(X).sum()}")
        logger.info(f"   Inf count in features: {np.isinf(X).sum()}")
        
    except Exception as e:
        logger.error(f"   ✗ Data preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test training with preprocessed data
    logger.info("\\n5. Testing training with preprocessed data...")
    try:
        from training_manager import TrainingConfig
        config = TrainingConfig(epochs=2)  # Just 2 epochs for testing
        result = brain.train_domain(
            'fraud_detection',
            train_data,
            training_config=config
        )
        
        if result['success']:
            logger.info(f"   ✓ Training successful!")
            logger.info(f"   Session ID: {result.get('session_id')}")
            logger.info(f"   Epochs completed: {result.get('epochs_completed')}")
            logger.info(f"   Training time: {result.get('training_time', 0):.2f}s")
        else:
            logger.error(f"   ✗ Training failed: {result.get('error')}")
            return False
            
    except Exception as e:
        logger.error(f"   ✗ Training failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test prediction
    logger.info("\\n6. Testing prediction with trained model...")
    try:
        # Create a sample transaction for prediction
        sample_transaction = trans_df.iloc[0:1].drop(['isFraud'], axis=1).to_dict('records')[0]
        
        prediction = brain.predict(
            {'data': sample_transaction},
            domain='fraud_detection'
        )
        
        if prediction.success:
            logger.info(f"   ✓ Prediction successful!")
            logger.info(f"   Prediction: {prediction.prediction}")
            logger.info(f"   Confidence: {prediction.confidence:.2%}")
            logger.info(f"   Domain used: {prediction.domain}")
        else:
            logger.error(f"   ✗ Prediction failed: {prediction.error}")
            return False
            
    except Exception as e:
        logger.error(f"   ✗ Prediction failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with missing values
    logger.info("\\n7. Testing with high missing value ratio...")
    try:
        # Create data with many missing values
        trans_missing = trans_df.copy()
        for col in trans_missing.columns:
            if col not in ['TransactionID', 'isFraud']:
                mask = np.random.random(len(trans_missing)) < 0.3  # 30% missing
                trans_missing.loc[mask, col] = np.nan
        
        train_data_missing = {
            'transactions': trans_missing,
            'identities': id_df
        }
        
        X_missing, y_missing = brain._prepare_fraud_detection_data(train_data_missing)
        logger.info(f"   ✓ Missing value handling successful!")
        logger.info(f"   Features shape: {X_missing.shape}")
        logger.info(f"   NaN count after processing: {np.isnan(X_missing).sum()}")
        
    except Exception as e:
        logger.error(f"   ✗ Missing value handling failed: {e}")
        return False
    
    # Test with different data types
    logger.info("\\n8. Testing with mixed data types...")
    try:
        # Add some problematic data types
        trans_mixed = trans_df.copy()
        trans_mixed['text_col'] = ['text_' + str(i) for i in range(len(trans_mixed))]
        trans_mixed['bool_col'] = np.random.choice([True, False], len(trans_mixed))
        trans_mixed['inf_col'] = np.random.randn(len(trans_mixed))
        trans_mixed.loc[0:10, 'inf_col'] = np.inf
        trans_mixed.loc[11:20, 'inf_col'] = -np.inf
        
        train_data_mixed = {
            'transactions': trans_mixed,
            'identities': id_df
        }
        
        X_mixed, y_mixed = brain._prepare_fraud_detection_data(train_data_mixed)
        logger.info(f"   ✓ Mixed data type handling successful!")
        logger.info(f"   Features shape: {X_mixed.shape}")
        logger.info(f"   Inf values after processing: {np.isinf(X_mixed).sum()}")
        
    except Exception as e:
        logger.error(f"   ✗ Mixed data type handling failed: {e}")
        return False
    
    logger.info("\\n" + "=" * 80)
    logger.info("All preprocessing tests passed! ✓")
    logger.info("=" * 80)
    
    # Cleanup
    brain.shutdown()
    
    return True


if __name__ == "__main__":
    success = test_data_preprocessing()
    sys.exit(0 if success else 1)