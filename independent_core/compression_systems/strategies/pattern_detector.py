"""
Pattern Detection System for Weight Distribution Analysis.
Analyzes statistical properties of weight tensors to inform compression strategy selection.
NO PLACEHOLDERS - COMPLETE PRODUCTION IMPLEMENTATION
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Check scipy availability and raise error if not available
try:
    import scipy.stats
    from scipy.signal import find_peaks
    from scipy.stats import gaussian_kde
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    raise ImportError("scipy is required for pattern detection. Install with: pip install scipy")


@dataclass
class DistributionAnalysis:
    """Results from weight distribution analysis"""
    mean: float
    std: float
    skewness: float
    kurtosis: float
    num_modes: int
    sparsity: float
    quantization_levels: int
    distribution_type: str  # "gaussian", "bimodal", "multimodal", "uniform", "sparse", "heavy_tailed", "unknown"
    mode_locations: List[float]
    valley_points: List[float]  # For multimodal distributions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'mean': self.mean,
            'std': self.std,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'num_modes': self.num_modes,
            'sparsity': self.sparsity,
            'quantization_levels': self.quantization_levels,
            'distribution_type': self.distribution_type,
            'mode_locations': self.mode_locations,
            'valley_points': self.valley_points
        }


class WeightDistributionAnalyzer:
    """Analyzes weight tensor distributions for optimal compression strategy selection"""
    
    def __init__(self, max_sample_size: int = 100_000):
        """
        Initialize the weight distribution analyzer.
        
        Args:
            max_sample_size: Maximum number of elements to sample for large tensors
        """
        self.max_sample_size = max_sample_size
        
    def analyze_distribution(self, weights: torch.Tensor) -> DistributionAnalysis:
        """
        Perform comprehensive distribution analysis on weight tensor.
        
        Uses scipy.stats for advanced statistical analysis including:
        - Skewness and kurtosis calculation
        - Mode detection using kernel density estimation
        - Distribution type classification
        
        Args:
            weights: Input weight tensor to analyze
            
        Returns:
            DistributionAnalysis object containing all statistical metrics
        """
        # Flatten and convert to numpy
        weights_np = weights.flatten().cpu().numpy()
        
        # Sample if tensor is too large to avoid memory issues
        if len(weights_np) > self.max_sample_size:
            logger.info(f"Sampling {self.max_sample_size} elements from tensor with {len(weights_np)} elements")
            indices = np.random.choice(len(weights_np), self.max_sample_size, replace=False)
            weights_np = weights_np[indices]
        
        # Basic statistics
        mean = float(np.mean(weights_np))
        std = float(np.std(weights_np))
        
        # Higher moments using scipy
        skewness = float(scipy.stats.skew(weights_np))
        kurtosis = float(scipy.stats.kurtosis(weights_np))
        
        # Sparsity calculation
        sparsity = float((weights_np == 0).sum() / len(weights_np))
        
        # Quantization levels
        unique_values = np.unique(weights_np)
        quantization_levels = len(unique_values)
        
        # Mode detection
        num_modes, mode_locations, valley_points = self.detect_modes(weights)
        
        # Classify distribution type
        distribution_type = self.classify_distribution(
            skewness, kurtosis, num_modes, sparsity
        )
        
        return DistributionAnalysis(
            mean=mean,
            std=std,
            skewness=skewness,
            kurtosis=kurtosis,
            num_modes=num_modes,
            sparsity=sparsity,
            quantization_levels=quantization_levels,
            distribution_type=distribution_type,
            mode_locations=mode_locations,
            valley_points=valley_points
        )
    
    def detect_modes(self, weights: torch.Tensor) -> Tuple[int, List[float], List[float]]:
        """
        Detect modes in the weight distribution using kernel density estimation.
        
        Uses scipy.stats.gaussian_kde for smooth density estimation and
        scipy.signal.find_peaks for robust peak detection.
        
        Args:
            weights: Input weight tensor
            
        Returns:
            Tuple of (number of modes, mode locations, valley points between modes)
        """
        weights_np = weights.flatten().cpu().numpy()
        
        # Sample if necessary
        if len(weights_np) > self.max_sample_size:
            indices = np.random.choice(len(weights_np), self.max_sample_size, replace=False)
            weights_np = weights_np[indices]
        
        # Remove zeros for KDE (zeros can dominate the distribution)
        non_zero_weights = weights_np[weights_np != 0]
        
        # Need sufficient data for KDE
        if len(non_zero_weights) < 10:
            # Not enough non-zero data, return single mode at mean
            return 1, [float(np.mean(weights_np))], []
        
        try:
            # Kernel Density Estimation
            kde = gaussian_kde(non_zero_weights)
            
            # Create evaluation points
            x_min, x_max = non_zero_weights.min(), non_zero_weights.max()
            x_range = np.linspace(x_min, x_max, 1000)
            
            # Evaluate density
            density = kde(x_range)
            
            # Find peaks with adaptive parameters
            # Require peaks to be at least 10% of maximum density
            min_height = np.max(density) * 0.1
            # Require minimum distance between peaks (5% of range)
            min_distance = len(x_range) // 20
            
            peaks, properties = find_peaks(
                density,
                height=min_height,
                distance=min_distance,
                prominence=np.max(density) * 0.05  # Add prominence requirement
            )
            
            # Get mode locations
            mode_locations = x_range[peaks].tolist()
            
            # Find valleys between peaks
            valley_points = []
            if len(peaks) > 1:
                for i in range(len(peaks) - 1):
                    # Find minimum between consecutive peaks
                    valley_region = density[peaks[i]:peaks[i+1]]
                    valley_idx = np.argmin(valley_region) + peaks[i]
                    valley_points.append(float(x_range[valley_idx]))
            
            # If no modes found, report single mode at maximum density
            if len(mode_locations) == 0:
                max_density_idx = np.argmax(density)
                mode_locations = [float(x_range[max_density_idx])]
            
            return len(mode_locations), mode_locations, valley_points
            
        except Exception as e:
            logger.warning(f"Mode detection failed: {e}, returning single mode")
            # Fallback to simple mode detection
            return 1, [float(np.mean(non_zero_weights))], []
    
    def classify_distribution(self, skewness: float, kurtosis: float, 
                             num_modes: int, sparsity: float) -> str:
        """
        Classify the distribution type based on statistical properties.
        
        Distribution types:
        - sparse: >70% zeros
        - bimodal: exactly 2 modes
        - multimodal: >2 modes
        - gaussian: low skewness, normal kurtosis
        - heavy_tailed: high kurtosis
        - uniform: low kurtosis
        - unknown: doesn't fit other categories
        
        Args:
            skewness: Skewness of the distribution
            kurtosis: Excess kurtosis of the distribution
            num_modes: Number of detected modes
            sparsity: Proportion of zero values
            
        Returns:
            String classification of distribution type
        """
        # Check sparsity first (most definitive)
        if sparsity > 0.7:
            return "sparse"
        
        # Check modality
        if num_modes == 2:
            return "bimodal"
        elif num_modes > 2:
            return "multimodal"
        
        # Check for Gaussian-like distribution
        # Normal distribution has skewness ≈ 0 and excess kurtosis ≈ 0
        if abs(skewness) < 0.5 and -0.5 < kurtosis < 0.5:
            return "gaussian"
        
        # Check for heavy-tailed distribution (high kurtosis)
        if kurtosis > 3:
            return "heavy_tailed"
        
        # Check for uniform-like distribution (low kurtosis)
        if kurtosis < -1.2:
            return "uniform"
        
        # Default classification
        return "unknown"
    
    def analyze_quantization(self, weights: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze quantization properties of the weight tensor.
        
        Args:
            weights: Input weight tensor
            
        Returns:
            Dictionary containing quantization analysis results
        """
        weights_np = weights.flatten().cpu().numpy()
        
        # Get unique values
        unique_values = np.unique(weights_np)
        num_levels = len(unique_values)
        
        # Check if values appear to be quantized
        if num_levels < 256:
            # Might be quantized, check for uniform spacing
            if num_levels > 1:
                sorted_values = np.sort(unique_values)
                differences = np.diff(sorted_values)
                
                # Check if differences are approximately uniform
                std_diff = np.std(differences)
                mean_diff = np.mean(differences)
                
                is_uniform_quantization = (std_diff / mean_diff < 0.1) if mean_diff > 0 else False
            else:
                is_uniform_quantization = False
                
            return {
                'is_quantized': True,
                'num_levels': num_levels,
                'is_uniform': is_uniform_quantization,
                'bit_depth': int(np.ceil(np.log2(num_levels))) if num_levels > 0 else 0
            }
        else:
            return {
                'is_quantized': False,
                'num_levels': num_levels,
                'is_uniform': False,
                'bit_depth': 32  # Assume float32
            }
    
    def analyze_clustering(self, weights: torch.Tensor, n_clusters: int = 5) -> Dict[str, Any]:
        """
        Analyze natural clustering in the weight distribution.
        
        Args:
            weights: Input weight tensor
            n_clusters: Number of clusters to attempt finding
            
        Returns:
            Dictionary containing clustering analysis results
        """
        weights_np = weights.flatten().cpu().numpy()
        
        # Sample if necessary
        if len(weights_np) > self.max_sample_size:
            indices = np.random.choice(len(weights_np), self.max_sample_size, replace=False)
            weights_np = weights_np[indices]
        
        # Remove zeros for clustering
        non_zero_weights = weights_np[weights_np != 0]
        
        if len(non_zero_weights) < n_clusters:
            return {
                'natural_clusters': 1,
                'cluster_centers': [float(np.mean(weights_np))],
                'cluster_sizes': [len(weights_np)]
            }
        
        try:
            from sklearn.cluster import KMeans
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(non_zero_weights.reshape(-1, 1))
            
            # Get cluster statistics
            cluster_centers = kmeans.cluster_centers_.flatten().tolist()
            cluster_sizes = [int(np.sum(labels == i)) for i in range(n_clusters)]
            
            # Calculate silhouette score to assess clustering quality
            from sklearn.metrics import silhouette_score
            if len(np.unique(labels)) > 1:
                score = silhouette_score(non_zero_weights.reshape(-1, 1), labels)
            else:
                score = 0.0
            
            return {
                'natural_clusters': n_clusters,
                'cluster_centers': cluster_centers,
                'cluster_sizes': cluster_sizes,
                'silhouette_score': float(score)
            }
            
        except ImportError:
            logger.warning("scikit-learn not available for clustering analysis")
            return {
                'natural_clusters': 1,
                'cluster_centers': [float(np.mean(weights_np))],
                'cluster_sizes': [len(weights_np)]
            }
    
    def compute_compression_hints(self, analysis: DistributionAnalysis) -> Dict[str, float]:
        """
        Compute compression strategy hints based on distribution analysis.
        
        Args:
            analysis: DistributionAnalysis results
            
        Returns:
            Dictionary of strategy scores and hints
        """
        hints = {
            'padic_score': 0.0,
            'tropical_score': 0.0,
            'hybrid_score': 0.0,
            'quantization_benefit': 0.0,
            'clustering_benefit': 0.0
        }
        
        # P-adic hints
        if analysis.distribution_type == 'gaussian':
            hints['padic_score'] += 0.3
        elif analysis.distribution_type == 'heavy_tailed':
            hints['padic_score'] += 0.25
        
        if analysis.quantization_levels < 32:
            hints['padic_score'] += 0.2
            hints['quantization_benefit'] = 0.8
        
        # Tropical hints
        if analysis.distribution_type == 'sparse':
            hints['tropical_score'] += 0.5
        elif analysis.sparsity > 0.3:
            hints['tropical_score'] += 0.2
        
        if analysis.distribution_type == 'uniform':
            hints['tropical_score'] -= 0.1
        
        # Hybrid hints
        if analysis.distribution_type in ['bimodal', 'multimodal']:
            hints['hybrid_score'] += 0.4
            hints['clustering_benefit'] = 0.6
        
        if analysis.num_modes > 3:
            hints['hybrid_score'] += 0.2
        
        # Normalize scores
        for key in ['padic_score', 'tropical_score', 'hybrid_score']:
            hints[key] = min(1.0, max(0.0, hints[key]))
        
        return hints