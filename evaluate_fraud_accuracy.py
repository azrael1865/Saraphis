import numpy as np
import sys
sys.path.append('independent_core')
from brain import Brain
import torch

print("EVALUATING FRAUD DETECTION ACCURACY")
brain = Brain()

# Load the full dataset
data = np.load('training_data/ieee-fraud-detection/processed_cache/train_processed.npz')
X = data['X']
y = data['y']

print(f"Dataset: {X.shape[0]:,} samples with {X.shape[1]} features")
print(f"Fraud rate: {np.mean(y)*100:.2f}%")

# Get the trained model from the brain
model = brain.training_manager._get_or_create_model('fraud_detection', {'num_features': X.shape[1]}, brain.training_manager._sessions[list(brain.training_manager._sessions.keys())[0]].config)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

print(f"Evaluating on device: {device}")

# Split data for evaluation (use same split as training)
X_train, X_val, y_train, y_val = brain.training_manager._prepare_training_data({'X': X, 'y': y}, brain.training_manager._sessions[list(brain.training_manager._sessions.keys())[0]].config)

print(f"Validation set: {X_val.shape[0]:,} samples")

# Evaluate on validation set
correct = 0
total = 0
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

with torch.no_grad():
    for i in range(0, len(X_val), 256):
        batch_X = X_val[i:i+256]
        batch_y = y_val[i:i+256]
        
        # Create tensors on device
        batch_X_tensor = torch.FloatTensor(batch_X).to(device)
        batch_y_tensor = torch.FloatTensor(batch_y).to(device)
        
        # Fix shape mismatch
        if len(batch_y_tensor.shape) == 1:
            batch_y_tensor = batch_y_tensor.unsqueeze(1)
        
        # Get predictions
        outputs = model(batch_X_tensor)
        predictions = (outputs > 0.5).float()  # Threshold at 0.5
        
        # Calculate metrics
        correct += ((predictions == batch_y_tensor).sum().item())
        total += batch_y_tensor.size(0)
        
        # Calculate confusion matrix components
        true_positives += ((predictions == 1) & (batch_y_tensor == 1)).sum().item()
        false_positives += ((predictions == 1) & (batch_y_tensor == 0)).sum().item()
        true_negatives += ((predictions == 0) & (batch_y_tensor == 0)).sum().item()
        false_negatives += ((predictions == 0) & (batch_y_tensor == 1)).sum().item()

# Calculate accuracy metrics
accuracy = correct / total
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n" + "="*50)
print("FRAUD DETECTION ACCURACY RESULTS")
print("="*50)
print(f"Overall Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1-Score: {f1_score*100:.2f}%")
print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"True Negatives: {true_negatives}")
print(f"False Negatives: {false_negatives}")
print(f"Total Samples: {total:,}")
print("="*50) 