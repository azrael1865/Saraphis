#!/usr/bin/env python3
"""
Simple Training Test - Bypasses complex import issues and uses basic training
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add the independent_core to path
sys.path.append('independent_core')

def load_ieee_data():
    """Load IEEE fraud detection data with fallback handling"""
    try:
        # Try to load the processed data
        data_path = Path('training_data/ieee-fraud-detection/processed_cache/train_processed.npz')
        if data_path.exists():
            print(f"Loading processed data from {data_path}")
            data = np.load(data_path)
            X = data['X']
            y = data['y']
            print(f"✅ Loaded {X.shape[0]:,} samples with {X.shape[1]} features")
            print(f"✅ Fraud rate: {np.mean(y)*100:.2f}%")
            return X, y
        else:
            raise FileNotFoundError(f"Processed data not found at {data_path}")
    except Exception as e:
        print(f"❌ Failed to load IEEE data: {e}")
        print("Creating synthetic data for testing...")
        
        # Create synthetic data for testing
        np.random.seed(42)
        X = np.random.random((10000, 100)).astype(np.float32)
        y = np.random.randint(0, 2, 10000).astype(np.int64)
        print(f"✅ Created synthetic data: {X.shape[0]:,} samples with {X.shape[1]} features")
        return X, y

def run_simple_training():
    """Run simple training using the Brain system"""
    try:
        print("🧠 Initializing Brain system...")
        from brain import Brain
        
        # Initialize Brain
        brain = Brain()
        print("✅ Brain system initialized")
        
        # Load data
        print("📊 Loading training data...")
        X, y = load_ieee_data()
        
        # Prepare training data
        training_data = {'X': X, 'y': y}
        
        # Run training with simple configuration
        print("🚀 Starting training...")
        result = brain.train_domain(
            'fraud_detection', 
            training_data,
            epochs=5,  # Small number for testing
            batch_size=128,
            learning_rate=0.001
        )
        
        # Display results
        print("\n" + "="*60)
        print("TRAINING RESULTS")
        print("="*60)
        print(f"Success: {result.get('success', False)}")
        print(f"Training time: {result.get('training_time', 'N/A')}")
        print(f"Best performance: {result.get('best_performance', 'N/A')}")
        print(f"Epochs completed: {result.get('details', {}).get('epochs_completed', 'N/A')}")
        
        if result.get('success'):
            print("✅ Training completed successfully!")
        else:
            print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Training execution failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    print("="*60)
    print("SIMPLE TRAINING TEST")
    print("="*60)
    
    result = run_simple_training()
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60) 