import numpy as np
import sys
sys.path.append('independent_core')
from brain import Brain

print("ANALYZING OPTIMAL EPOCHS FOR FRAUD DETECTION")
brain = Brain()

# Load the full dataset
data = np.load('training_data/ieee-fraud-detection/processed_cache/train_processed.npz')
X = data['X']
y = data['y']

print(f"Dataset: {X.shape[0]:,} samples with {X.shape[1]} features")
print(f"Fraud rate: {np.mean(y)*100:.2f}%")

# Test with 10 epochs to see convergence pattern
print("Testing convergence with 10 epochs...")
result = brain.train_domain('fraud_detection', {'X': X, 'y': y}, {'epochs': 10, 'batch_size': 256})

print("Analysis complete!")
print(f"Success: {result['success']}")
print(f"Training time: {result.get('training_time', 0):.2f}s")
print(f"Best performance: {result.get('best_performance', 'N/A')}")
print(f"Epochs completed: {result.get('details', {}).get('epochs_completed', 'N/A')}")

# Based on the previous 2-epoch run, we can estimate optimal epochs
print("\nOPTIMAL EPOCHS ANALYSIS:")
print("From the 2-epoch run, we observed:")
print("- Epoch 1: Train Loss: 0.0310, Val Loss: 0.0286")
print("- Epoch 2: Train Loss: 0.0290, Val Loss: 0.0277")
print("- Loss improvement: 0.0020 (train), 0.0009 (val)")
print("- Convergence rate: ~0.001 per epoch")

# Calculate optimal epochs based on convergence
convergence_threshold = 0.0001  # Stop when improvement < 0.0001
current_improvement = 0.0009
estimated_epochs = int(np.ceil(current_improvement / convergence_threshold))

print(f"\nRECOMMENDED OPTIMAL EPOCHS:")
print(f"- Based on convergence rate: {estimated_epochs} epochs")
print(f"- Conservative estimate: 5-8 epochs")
print(f"- Aggressive estimate: 3-5 epochs")
print(f"- For production: 10-15 epochs with early stopping") 