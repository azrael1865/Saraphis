import numpy as np
import sys
sys.path.append('independent_core')
from brain import Brain

print("RUNNING FULL TRAINING WITH DEFAULT CONFIGURATION")
print("100 epochs with early stopping enabled")
brain = Brain()

# Load the full dataset
data = np.load('training_data/ieee-fraud-detection/processed_cache/train_processed.npz')
X = data['X']
y = data['y']

print(f"Dataset: {X.shape[0]:,} samples with {X.shape[1]} features")
print(f"Fraud rate: {np.mean(y)*100:.2f}%")
print("Starting training with default config (100 epochs, early stopping enabled)...")

# Run training with default configuration (100 epochs, early stopping)
result = brain.train_domain('fraud_detection', {'X': X, 'y': y})

print("Training completed!")
print(f"Success: {result['success']}")
print(f"Training time: {result.get('training_time', 0):.2f}s")
print(f"Best performance: {result.get('best_performance', 'N/A')}")
# CRITICAL FIX: Check for epochs_completed at top level first, then in details
epochs_completed = result.get('epochs_completed', result.get('details', {}).get('epochs_completed', 'N/A'))
print(f"Epochs completed: {epochs_completed}")
print(f"Early stopping triggered: {'Yes' if isinstance(epochs_completed, int) and epochs_completed < 100 else 'No'}") 