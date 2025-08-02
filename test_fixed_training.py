import numpy as np
import sys
sys.path.append('independent_core')
from brain import Brain
import brain_training_integration

print("FULL DATASET TRAINING WITH ALL FIXES APPLIED")
brain = Brain()

# Load the full dataset
data = np.load('training_data/ieee-fraud-detection/processed_cache/train_processed.npz')
X = data['X']
y = data['y']

print(f"Dataset: {X.shape[0]:,} samples with {X.shape[1]} features")
print(f"Fraud rate: {np.mean(y)*100:.2f}%")
print(f"Memory usage: {X.nbytes / 1024 / 1024:.1f} MB")
print("Starting REAL training with GPU acceleration...")

# Run training with fixes applied - SHOW EVERY BATCH
result = brain.train_domain('fraud_detection', {'X': X, 'y': y}, {
    'epochs': 2, 
    'batch_size': 256,
    'log_frequency': 1  # Log EVERY batch to see real-time progress
})

print("Training completed!")
print(f"Success: {result['success']}")
print(f"Training time: {result.get('training_time', 0):.2f}s")
best_perf = result.get("best_performance")
print(f"Best performance: {best_perf if best_perf is not None else 'N/A'}")
print(f"Epochs completed: {result.get('details', {}).get('epochs_completed', 'N/A')}") 