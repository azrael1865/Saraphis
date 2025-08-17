"""
Integration example for tropical semiring operations.
Shows how to use tropical mathematics for neural network compression.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import time
from .tropical_core import (
    TROPICAL_ZERO,
    TropicalNumber,
    TropicalMathematicalOperations,
    TropicalGradientTracker,
    is_tropical_zero,
    to_tropical_safe,
    from_tropical_safe,
    tropical_distance
)
from ..base.compression_base import CompressionBase


class TropicalLinearLayer(nn.Module):
    """
    A neural network layer using tropical arithmetic.
    Performs tropical matrix multiplication: max-plus operations.
    """
    
    def __init__(self, in_features: int, out_features: int, device: torch.device = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device or torch.device('cpu')
        
        # Initialize tropical operations
        self.ops = TropicalMathematicalOperations(device=self.device)
        self.grad_tracker = TropicalGradientTracker(device=self.device)
        
        # Initialize weights with tropical-friendly values
        # Using uniform distribution in log space for better tropical properties
        self.weight = nn.Parameter(
            torch.log(torch.rand(out_features, in_features, device=self.device) + 1e-6)
        )
        
        # Tropical bias (optional)
        self.bias = nn.Parameter(
            torch.zeros(out_features, device=self.device)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using tropical operations.
        x: Input tensor of shape (batch_size, in_features)
        """
        batch_size = x.shape[0]
        
        # Ensure input is on correct device
        if x.device != self.device:
            x = x.to(self.device)
        
        # Initialize output with tropical zeros
        output = torch.full((batch_size, self.out_features), TROPICAL_ZERO, device=self.device)
        
        # Tropical matrix multiplication
        # out[b,j] = max_i(x[b,i] + weight[j,i]) + bias[j]
        for b in range(batch_size):
            for j in range(self.out_features):
                # Compute tropical dot product
                products = x[b, :] + self.weight[j, :]
                
                # Handle tropical zeros
                non_zero_mask = products > TROPICAL_ZERO
                if non_zero_mask.any():
                    # Tropical sum (max)
                    max_val = products[non_zero_mask].max()
                    # Add bias using tropical multiplication (standard addition)
                    output[b, j] = max_val + self.bias[j]
        
        return output


class TropicalCompressionExample:
    """
    Example of using tropical operations for neural network weight compression.
    Demonstrates piecewise linear approximation using tropical polynomials.
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')
        self.ops = TropicalMathematicalOperations(device=self.device)
    
    def compress_weights_tropical(self, weights: torch.Tensor, num_pieces: int = 4) -> Dict[str, Any]:
        """
        Compress neural network weights using tropical piecewise approximation.
        
        Args:
            weights: Tensor of weights to compress
            num_pieces: Number of linear pieces in approximation
            
        Returns:
            Dictionary containing compressed representation
        """
        if weights.device != self.device:
            weights = weights.to(self.device)
        
        # Flatten weights for processing
        original_shape = weights.shape
        flat_weights = weights.flatten()
        
        # Convert to tropical representation
        tropical_weights = to_tropical_safe(flat_weights)
        
        # Find breakpoints for piecewise approximation
        sorted_weights, indices = torch.sort(tropical_weights)
        n = len(sorted_weights)
        
        # Compute breakpoints
        breakpoints = []
        slopes = []
        
        for i in range(num_pieces):
            start_idx = i * n // num_pieces
            end_idx = (i + 1) * n // num_pieces
            
            if end_idx > start_idx:
                # Extract piece
                piece = sorted_weights[start_idx:end_idx]
                
                # Compute tropical linear approximation
                # Using least squares in log domain
                x_vals = torch.arange(len(piece), dtype=torch.float32, device=self.device)
                
                # Avoid tropical zeros in fitting
                valid_mask = piece > TROPICAL_ZERO
                if valid_mask.any():
                    valid_x = x_vals[valid_mask]
                    valid_y = piece[valid_mask]
                    
                    # Simple linear fit
                    if len(valid_x) > 1:
                        x_mean = valid_x.mean()
                        y_mean = valid_y.mean()
                        
                        slope = ((valid_x - x_mean) * (valid_y - y_mean)).sum() / ((valid_x - x_mean) ** 2).sum()
                        intercept = y_mean - slope * x_mean
                        
                        breakpoints.append({
                            'start': start_idx,
                            'end': end_idx,
                            'slope': slope.item(),
                            'intercept': intercept.item()
                        })
                        slopes.append(slope.item())
        
        # Store compression info
        compressed = {
            'original_shape': original_shape,
            'num_pieces': num_pieces,
            'breakpoints': breakpoints,
            'slopes': torch.tensor(slopes, device=self.device),
            'permutation': indices,  # For reconstruction
            'compression_ratio': weights.numel() / (len(breakpoints) * 4)  # Approximate
        }
        
        return compressed
    
    def decompress_weights_tropical(self, compressed: Dict[str, Any]) -> torch.Tensor:
        """
        Decompress weights from tropical piecewise representation.
        
        Args:
            compressed: Dictionary containing compressed representation
            
        Returns:
            Reconstructed weight tensor
        """
        original_shape = compressed['original_shape']
        breakpoints = compressed['breakpoints']
        permutation = compressed['permutation']
        
        # Reconstruct flattened weights
        n = permutation.numel()
        reconstructed = torch.zeros(n, device=self.device)
        
        for bp in breakpoints:
            start = bp['start']
            end = bp['end']
            slope = bp['slope']
            intercept = bp['intercept']
            
            # Generate linear piece
            x_vals = torch.arange(end - start, dtype=torch.float32, device=self.device)
            piece = slope * x_vals + intercept
            
            # Place in correct positions
            reconstructed[start:end] = piece
        
        # Inverse permutation
        inverse_perm = torch.argsort(permutation)
        reconstructed = reconstructed[inverse_perm]
        
        # Reshape to original
        return reconstructed.reshape(original_shape)


def example_tropical_neural_compression():
    """
    Complete example of using tropical operations for neural network compression.
    """
    print("Tropical Neural Network Compression Example")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a simple neural network layer
    in_features, out_features = 64, 32
    tropical_layer = TropicalLinearLayer(in_features, out_features, device=device)
    
    # Generate sample input
    batch_size = 16
    x = torch.randn(batch_size, in_features, device=device)
    
    # Forward pass with tropical operations
    print("\n1. Tropical Layer Forward Pass:")
    output = tropical_layer(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Compress weights using tropical approximation
    print("\n2. Weight Compression:")
    compressor = TropicalCompressionExample(device=device)
    
    # Get original weights
    original_weights = tropical_layer.weight.data
    print(f"   Original weight shape: {original_weights.shape}")
    print(f"   Original weight size: {original_weights.numel() * 4} bytes")
    
    # Compress
    compressed = compressor.compress_weights_tropical(original_weights, num_pieces=8)
    print(f"   Number of pieces: {compressed['num_pieces']}")
    print(f"   Compression ratio: {compressed['compression_ratio']:.2f}x")
    
    # Decompress
    reconstructed_weights = compressor.decompress_weights_tropical(compressed)
    
    # Compute reconstruction error
    mse = torch.nn.functional.mse_loss(original_weights, reconstructed_weights)
    print(f"   Reconstruction MSE: {mse:.6f}")
    
    # Tropical distance metric
    print("\n3. Tropical Distance Analysis:")
    # Compute tropical distance between original and reconstructed
    flat_original = original_weights.flatten()
    flat_reconstructed = reconstructed_weights.flatten()
    
    # Sample some distances
    sample_size = min(100, flat_original.numel())
    sample_indices = torch.randperm(flat_original.numel())[:sample_size]
    
    distances = []
    for idx in sample_indices:
        dist = tropical_distance(flat_original[idx].item(), flat_reconstructed[idx].item())
        if not torch.isinf(torch.tensor(dist)):
            distances.append(dist)
    
    if distances:
        avg_distance = sum(distances) / len(distances)
        print(f"   Average tropical distance: {avg_distance:.6f}")
    
    # Demonstrate gradient tracking
    print("\n4. Gradient Tracking Example:")
    x_grad = torch.randn(5, device=device, requires_grad=True)
    y_grad = torch.randn(5, device=device, requires_grad=True)
    
    # Tropical operations with gradient
    ops = TropicalMathematicalOperations(device=device)
    z = ops.tropical_add(x_grad, y_grad)
    
    print(f"   x: {x_grad.data}")
    print(f"   y: {y_grad.data}")
    print(f"   tropical_add(x, y): {z}")
    
    # Show which elements "win" in the max operation
    winners = (x_grad > y_grad).float()
    print(f"   Winners (1 if x > y): {winners}")
    
    print("\nExample completed successfully!")


def example_tropical_polynomial_approximation():
    """
    Example of using tropical polynomials for function approximation.
    """
    print("\nTropical Polynomial Approximation Example")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ops = TropicalMathematicalOperations(device=device)
    
    # Create a nonlinear function to approximate
    x = torch.linspace(-5, 5, 100, device=device)
    y_true = torch.abs(x) + 0.5 * torch.sin(x)  # Piecewise smooth function
    
    # Tropical polynomial approximation
    # P(x) = max(a0, a1 + x, a2 + 2x, ...)
    degree = 5
    
    # Find coefficients using sampling
    coeffs = []
    for i in range(degree + 1):
        # Sample points where this term might dominate
        sample_x = -5 + 10 * i / degree
        sample_y = torch.abs(torch.tensor(sample_x)) + 0.5 * torch.sin(torch.tensor(sample_x))
        coeff = sample_y - i * sample_x
        coeffs.append(coeff.item())
    
    print(f"   Tropical polynomial coefficients: {coeffs}")
    
    # Evaluate tropical polynomial
    y_approx = torch.full_like(x, TROPICAL_ZERO)
    for i, coeff in enumerate(coeffs):
        term = coeff + i * x
        y_approx = ops.tropical_add(y_approx, term)
    
    # Compute approximation error
    mse = torch.nn.functional.mse_loss(y_true, y_approx)
    print(f"   Approximation MSE: {mse:.6f}")
    
    # Find active regions (which term dominates where)
    active_terms = []
    for i in range(len(x)):
        terms = [coeff + j * x[i].item() for j, coeff in enumerate(coeffs)]
        active_terms.append(terms.index(max(terms)))
    
    print(f"   Number of distinct active regions: {len(set(active_terms))}")
    
    print("\nTropical polynomial approximation completed!")


class TropicalCompressionSystem(CompressionBase):
    """
    High-level Tropical compression system that integrates with existing components.
    Uses the TropicalCompressionExample for actual compression work.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'compression_pieces': 4,
            'device': 'cpu',
            'enable_validation': True
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        
        # Initialize device
        if isinstance(self.config['device'], str):
            self.device = torch.device(self.config['device'])
        else:
            self.device = self.config['device']
        
        # Use existing tropical compression example
        self.compressor = TropicalCompressionExample(device=self.device)
        
    def _validate_config(self) -> None:
        """Validate tropical-specific configuration."""
        if 'compression_pieces' in self.config and self.config['compression_pieces'] <= 0:
            raise ValueError("compression_pieces must be positive")
    
    def encode(self, data: torch.Tensor) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Encode tensor using tropical compression."""
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(data)}")
        
        start_time = time.time()
        
        # Move data to correct device
        if data.device != self.device:
            data = data.to(self.device)
        
        # Use existing tropical compression
        compressed_data = self.compressor.compress_weights_tropical(
            data, 
            num_pieces=self.config.get('compression_pieces', 4)
        )
        
        encoding_time = time.time() - start_time
        
        # Extract encoded data and metadata
        encoded_data = {
            'breakpoints': compressed_data['breakpoints'],
            'indices': compressed_data.get('indices', compressed_data.get('permutation', []))
        }
        
        metadata = {
            'original_shape': data.shape,
            'original_dtype': str(data.dtype),
            'device': str(self.device),
            'compression_pieces': self.config.get('compression_pieces', 4),
            'encoding_time': encoding_time,
            'compression_algorithm': 'tropical_piecewise_linear'
        }
        
        return encoded_data, metadata
    
    def decode(self, encoded_data: Dict[str, Any], metadata: Dict[str, Any]) -> torch.Tensor:
        """Decode tropical compressed data back to tensor."""
        if not isinstance(encoded_data, dict):
            raise TypeError(f"Expected dict for encoded_data, got {type(encoded_data)}")
        
        if not isinstance(metadata, dict):
            raise TypeError(f"Expected dict for metadata, got {type(metadata)}")
        
        # Reconstruct using the existing decompression method
        reconstructed_data = self.compressor.decompress_weights_tropical({
            'breakpoints': encoded_data['breakpoints'],
            'permutation': encoded_data['indices'],  # Map indices back to permutation
            'original_shape': metadata['original_shape']
        })
        
        # Ensure correct device
        if reconstructed_data.device != self.device:
            reconstructed_data = reconstructed_data.to(self.device)
        
        return reconstructed_data


if __name__ == "__main__":
    # Run examples
    example_tropical_neural_compression()
    example_tropical_polynomial_approximation()