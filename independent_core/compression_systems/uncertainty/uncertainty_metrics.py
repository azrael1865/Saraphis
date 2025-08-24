"""
Uncertainty Metrics for Compression Systems
Quantifies and tracks uncertainty in compressed representations
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings

@dataclass
class UncertaintyStats:
    """Statistics for uncertainty quantification"""
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    entropy: float = 0.0

@dataclass
class UncertaintyQuantification:
    """
    Quantifies uncertainty in neural network compression.
    
    Tracks multiple sources of uncertainty:
    - Aleatoric: Inherent data uncertainty
    - Epistemic: Model uncertainty
    - Compression: Uncertainty from lossy compression
    """
    
    # Configuration
    num_samples: int = 100
    confidence_level: float = 0.95
    use_monte_carlo: bool = True
    track_history: bool = True
    
    # State
    history: List[UncertaintyStats] = field(default_factory=list)
    _current_stats: Optional[UncertaintyStats] = None
    
    def estimate_aleatoric_uncertainty(self, 
                                      data: torch.Tensor,
                                      model: Optional[torch.nn.Module] = None) -> UncertaintyStats:
        """
        Estimate aleatoric (data) uncertainty.
        
        Args:
            data: Input data tensor
            model: Optional model for prediction-based uncertainty
            
        Returns:
            Uncertainty statistics
        """
        # Compute basic statistics
        mean = data.mean().item()
        std = data.std().item()
        min_val = data.min().item()
        max_val = data.max().item()
        
        # Compute entropy as measure of uncertainty
        if data.numel() > 0:
            # Normalize data to [0, 1] for entropy calculation
            data_norm = (data - min_val) / (max_val - min_val + 1e-8)
            # Use histogram-based entropy estimation
            hist, _ = np.histogram(data_norm.cpu().numpy(), bins=50)
            hist = hist / hist.sum()  # Normalize
            hist = hist[hist > 0]  # Remove zeros
            entropy = -np.sum(hist * np.log(hist + 1e-8))
        else:
            entropy = 0.0
        
        # Compute confidence interval
        z_score = 1.96  # 95% confidence
        margin = z_score * std / np.sqrt(data.numel())
        confidence_interval = (mean - margin, mean + margin)
        
        stats = UncertaintyStats(
            mean=mean,
            std=std,
            min=min_val,
            max=max_val,
            confidence_interval=confidence_interval,
            entropy=entropy
        )
        
        if self.track_history:
            self.history.append(stats)
        
        self._current_stats = stats
        return stats
    
    def estimate_epistemic_uncertainty(self,
                                      model: torch.nn.Module,
                                      data: torch.Tensor,
                                      num_forward_passes: Optional[int] = None) -> UncertaintyStats:
        """
        Estimate epistemic (model) uncertainty using Monte Carlo dropout.
        
        Args:
            model: Neural network model with dropout
            data: Input data
            num_forward_passes: Number of stochastic forward passes
            
        Returns:
            Uncertainty statistics
        """
        if num_forward_passes is None:
            num_forward_passes = self.num_samples
        
        # Enable dropout during inference
        model.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_forward_passes):
                pred = model(data)
                predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions)
        
        # Compute statistics across predictions
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Aggregate statistics
        mean = mean_pred.mean().item()
        std = std_pred.mean().item()
        min_val = predictions.min().item()
        max_val = predictions.max().item()
        
        # Compute prediction entropy
        entropy = self._compute_prediction_entropy(predictions)
        
        # Confidence interval
        z_score = 1.96
        margin = z_score * std / np.sqrt(num_forward_passes)
        confidence_interval = (mean - margin, mean + margin)
        
        stats = UncertaintyStats(
            mean=mean,
            std=std,
            min=min_val,
            max=max_val,
            confidence_interval=confidence_interval,
            entropy=entropy
        )
        
        if self.track_history:
            self.history.append(stats)
        
        model.eval()  # Reset to evaluation mode
        return stats
    
    def estimate_compression_uncertainty(self,
                                        original: torch.Tensor,
                                        compressed: torch.Tensor,
                                        reconstruction: Optional[torch.Tensor] = None) -> UncertaintyStats:
        """
        Estimate uncertainty introduced by compression.
        
        Args:
            original: Original tensor
            compressed: Compressed representation
            reconstruction: Optional reconstructed tensor
            
        Returns:
            Uncertainty statistics
        """
        # Compute compression ratio
        compression_ratio = compressed.numel() / original.numel()
        
        if reconstruction is not None:
            # Compute reconstruction error
            error = (original - reconstruction).abs()
            mean_error = error.mean().item()
            std_error = error.std().item()
            min_error = error.min().item()
            max_error = error.max().item()
        else:
            # Estimate based on compression ratio
            mean_error = 1.0 - compression_ratio
            std_error = mean_error * 0.1  # Heuristic
            min_error = 0.0
            max_error = mean_error * 2.0
        
        # Information loss as entropy
        entropy = -compression_ratio * np.log(compression_ratio + 1e-8)
        
        # Confidence interval for error
        z_score = 1.96
        margin = z_score * std_error / np.sqrt(original.numel())
        confidence_interval = (mean_error - margin, mean_error + margin)
        
        stats = UncertaintyStats(
            mean=mean_error,
            std=std_error,
            min=min_error,
            max=max_error,
            confidence_interval=confidence_interval,
            entropy=entropy
        )
        
        if self.track_history:
            self.history.append(stats)
        
        return stats
    
    def _compute_prediction_entropy(self, predictions: torch.Tensor) -> float:
        """
        Compute entropy of predictions.
        
        Args:
            predictions: Tensor of predictions [num_samples, ...]
            
        Returns:
            Entropy value
        """
        # Flatten predictions
        preds_flat = predictions.view(predictions.size(0), -1)
        
        # Compute mean and variance
        mean = preds_flat.mean(dim=0)
        var = preds_flat.var(dim=0)
        
        # Approximate entropy using Gaussian assumption
        # H = 0.5 * log(2 * pi * e * var)
        entropy = 0.5 * torch.log(2 * np.pi * np.e * (var + 1e-8)).mean().item()
        
        return entropy
    
    def combine_uncertainties(self, 
                            aleatoric: Optional[UncertaintyStats] = None,
                            epistemic: Optional[UncertaintyStats] = None,
                            compression: Optional[UncertaintyStats] = None) -> UncertaintyStats:
        """
        Combine multiple uncertainty sources.
        
        Args:
            aleatoric: Aleatoric uncertainty stats
            epistemic: Epistemic uncertainty stats  
            compression: Compression uncertainty stats
            
        Returns:
            Combined uncertainty statistics
        """
        uncertainties = [u for u in [aleatoric, epistemic, compression] if u is not None]
        
        if not uncertainties:
            return UncertaintyStats()
        
        # Combine uncertainties (assuming independence)
        total_var = sum(u.std ** 2 for u in uncertainties)
        total_std = np.sqrt(total_var)
        
        # Weighted mean
        weights = [1.0 / (u.std + 1e-8) for u in uncertainties]
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        mean = sum(w * u.mean for w, u in zip(weights, uncertainties))
        
        # Min/max across all
        min_val = min(u.min for u in uncertainties)
        max_val = max(u.max for u in uncertainties)
        
        # Combined entropy
        entropy = sum(u.entropy for u in uncertainties)
        
        # Combined confidence interval
        z_score = 1.96
        margin = z_score * total_std
        confidence_interval = (mean - margin, mean + margin)
        
        return UncertaintyStats(
            mean=mean,
            std=total_std,
            min=min_val,
            max=max_val,
            confidence_interval=confidence_interval,
            entropy=entropy
        )
    
    def get_current_stats(self) -> Optional[UncertaintyStats]:
        """Get the most recent uncertainty statistics"""
        return self._current_stats
    
    def get_history_summary(self) -> Dict[str, float]:
        """
        Get summary statistics of uncertainty history.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.history:
            return {}
        
        means = [s.mean for s in self.history]
        stds = [s.std for s in self.history]
        entropies = [s.entropy for s in self.history]
        
        return {
            'mean_uncertainty': np.mean(means),
            'std_uncertainty': np.mean(stds),
            'max_uncertainty': np.max(stds),
            'min_uncertainty': np.min(stds),
            'mean_entropy': np.mean(entropies),
            'trend': np.polyfit(range(len(means)), means, 1)[0] if len(means) > 1 else 0.0
        }
    
    def reset(self):
        """Reset uncertainty tracking"""
        self.history.clear()
        self._current_stats = None

__all__ = ['UncertaintyQuantification', 'UncertaintyStats']