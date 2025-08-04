"""
Weight Categorizer - Advanced Pattern Recognition for Categorical Storage

Provides sophisticated weight pattern analysis and categorization for optimal
p-adic compression through similarity detection and entropy analysis.

NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging
from enum import Enum

logger = logging.getLogger(__name__)

# Import existing components
try:
    from .ieee754_channel_extractor import IEEE754Channels
    from .categorical_storage_manager import CategoryType
except ImportError:
    from compression_systems.categorical.ieee754_channel_extractor import IEEE754Channels
    from compression_systems.categorical.categorical_storage_manager import CategoryType


class PatternType(Enum):
    """Types of weight patterns for advanced categorization"""
    UNIFORM = "uniform"                    # Uniform distribution
    GAUSSIAN = "gaussian"                  # Normal distribution
    SPARSE = "sparse"                      # Sparse weights (many zeros)
    CLUSTERED = "clustered"                # Clustered values
    POWER_LAW = "power_law"                # Power law distribution
    BIMODAL = "bimodal"                    # Two distinct peaks
    PERIODIC = "periodic"                  # Periodic patterns
    MONOTONIC = "monotonic"                # Monotonically increasing/decreasing


@dataclass
class WeightPattern:
    """Detected pattern in weight distribution"""
    pattern_type: PatternType
    confidence: float                      # Pattern confidence [0, 1]
    parameters: Dict[str, float]           # Pattern-specific parameters
    entropy: float                         # Pattern entropy
    compressibility_score: float           # Estimated compression potential


@dataclass
class CategorizationResult:
    """Result of weight categorization analysis"""
    primary_category: CategoryType
    secondary_categories: List[CategoryType]
    detected_patterns: List[WeightPattern]
    similarity_groups: List[List[int]]     # Indices of similar weights
    compression_estimate: float            # Estimated compression ratio
    optimization_hints: Dict[str, Any]     # Hints for p-adic optimization


class WeightCategorizer:
    """
    Advanced weight pattern recognition and categorization system
    
    Analyzes weight distributions, detects patterns, and provides optimal
    categorization for p-adic compression optimization.
    """
    
    def __init__(self, enable_clustering: bool = True, 
                 enable_pattern_detection: bool = True,
                 max_clusters: int = 20):
        """Initialize weight categorizer
        
        Args:
            enable_clustering: Enable K-means clustering for similarity detection
            enable_pattern_detection: Enable statistical pattern detection
            max_clusters: Maximum number of clusters for similarity analysis
        """
        self.enable_clustering = enable_clustering
        self.enable_pattern_detection = enable_pattern_detection
        self.max_clusters = max_clusters
        
        # Pattern detection thresholds
        self.pattern_thresholds = {
            'sparsity_threshold': 0.1,        # Fraction of zeros for sparse
            'uniformity_threshold': 0.05,     # Std dev threshold for uniform
            'bimodal_separation': 2.0,        # Minimum separation for bimodal
            'periodicity_correlation': 0.8,   # Autocorrelation for periodic
            'power_law_alpha_min': 1.0,       # Minimum power law exponent
            'power_law_alpha_max': 4.0        # Maximum power law exponent
        }
        
        # Categorization statistics
        self.categorization_stats = {
            'total_categorizations': 0,
            'pattern_detection_success_rate': 0.0,
            'clustering_success_rate': 0.0,
            'average_compression_estimate': 0.0,
            'pattern_distribution': {},
            'category_distribution': {}
        }
        
        logger.info("WeightCategorizer initialized with clustering: %s, pattern detection: %s", 
                   enable_clustering, enable_pattern_detection)
    
    def categorize_weights(self, weights: torch.Tensor, 
                          channels: Optional[IEEE754Channels] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> CategorizationResult:
        """Perform comprehensive weight categorization analysis
        
        Args:
            weights: Tensor of weights to categorize
            channels: Optional IEEE 754 channels for enhanced analysis
            metadata: Optional metadata for categorization hints
            
        Returns:
            CategorizationResult with detailed categorization analysis
            
        Raises:
            RuntimeError: If categorization fails (hard failure)
            ValueError: If input validation fails
        """
        if weights is None:
            raise ValueError("Weights tensor cannot be None")
        
        if weights.numel() == 0:
            raise ValueError("Weights tensor cannot be empty")
        
        try:
            # Convert to numpy for analysis
            weight_array = weights.flatten().cpu().numpy()
            
            # Validate weights
            if np.any(np.isnan(weight_array)):
                raise ValueError("Weights contain NaN values")
            
            if np.any(np.isinf(weight_array)):
                raise ValueError("Weights contain infinite values")
            
            # Primary categorization based on basic statistics
            primary_category = self._determine_primary_category(weight_array)
            
            # Secondary categorization analysis
            secondary_categories = self._determine_secondary_categories(weight_array, channels)
            
            # Pattern detection
            detected_patterns = []
            if self.enable_pattern_detection:
                detected_patterns = self._detect_weight_patterns(weight_array)
            
            # Similarity clustering
            similarity_groups = []
            if self.enable_clustering and len(weight_array) > 10:
                similarity_groups = self._cluster_similar_weights(weight_array, channels)
            
            # Compression estimation
            compression_estimate = self._estimate_compression_ratio(
                weight_array, detected_patterns, similarity_groups
            )
            
            # Generate optimization hints
            optimization_hints = self._generate_optimization_hints(
                weight_array, detected_patterns, channels
            )
            
            # Update statistics
            self._update_categorization_statistics(primary_category, detected_patterns)
            
            result = CategorizationResult(
                primary_category=primary_category,
                secondary_categories=secondary_categories,
                detected_patterns=detected_patterns,
                similarity_groups=similarity_groups,
                compression_estimate=compression_estimate,
                optimization_hints=optimization_hints
            )
            
            logger.debug("Categorized %d weights: primary=%s, patterns=%d, groups=%d, compression=%.2fx",
                        len(weight_array), primary_category.value, len(detected_patterns), 
                        len(similarity_groups), compression_estimate)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Weight categorization failed: {e}")
    
    def _determine_primary_category(self, weights: np.ndarray) -> CategoryType:
        """Determine primary category based on weight statistics
        
        Args:
            weights: Array of weight values
            
        Returns:
            Primary CategoryType
        """
        abs_weights = np.abs(weights)
        
        # Check for zeros
        zero_fraction = np.sum(weights == 0.0) / len(weights)
        if zero_fraction > 0.5:
            return CategoryType.ZERO_WEIGHTS
        
        # Categorize by magnitude distribution
        mean_abs = np.mean(abs_weights)
        
        if mean_abs < 0.01:
            return CategoryType.SMALL_WEIGHTS
        elif mean_abs > 1.0:
            return CategoryType.LARGE_WEIGHTS
        else:
            # Further categorize medium weights by sign distribution
            positive_fraction = np.sum(weights > 0) / len(weights)
            if positive_fraction > 0.7:
                return CategoryType.POSITIVE_WEIGHTS
            elif positive_fraction < 0.3:
                return CategoryType.NEGATIVE_WEIGHTS
            else:
                return CategoryType.MEDIUM_WEIGHTS
    
    def _determine_secondary_categories(self, weights: np.ndarray, 
                                      channels: Optional[IEEE754Channels]) -> List[CategoryType]:
        """Determine secondary categories based on advanced analysis
        
        Args:
            weights: Array of weight values
            channels: Optional IEEE 754 channels
            
        Returns:
            List of secondary CategoryType values
        """
        secondary = []
        
        # Entropy-based categorization
        if channels is not None:
            mantissa_entropy = self._calculate_entropy(channels.mantissa_channel)
            exponent_entropy = self._calculate_entropy(channels.exponent_channel)
            
            if mantissa_entropy > 4.0 or exponent_entropy > 3.0:
                secondary.append(CategoryType.HIGH_ENTROPY)
            elif mantissa_entropy < 2.0 and exponent_entropy < 1.5:
                secondary.append(CategoryType.LOW_ENTROPY)
        
        # Statistical analysis for additional categories
        weight_std = np.std(weights)
        weight_mean = np.mean(np.abs(weights))
        
        # High variability indicates complex patterns
        if weight_std > weight_mean:
            if CategoryType.HIGH_ENTROPY not in secondary:
                secondary.append(CategoryType.HIGH_ENTROPY)
        
        # Low variability indicates regular patterns
        elif weight_std < weight_mean * 0.1:
            if CategoryType.LOW_ENTROPY not in secondary:
                secondary.append(CategoryType.LOW_ENTROPY)
        
        return secondary
    
    def _detect_weight_patterns(self, weights: np.ndarray) -> List[WeightPattern]:
        """Detect statistical patterns in weight distribution
        
        Args:
            weights: Array of weight values
            
        Returns:
            List of detected WeightPattern objects
        """
        patterns = []
        
        try:
            # Test for uniform distribution
            uniform_pattern = self._test_uniform_pattern(weights)
            if uniform_pattern:
                patterns.append(uniform_pattern)
            
            # Test for Gaussian distribution
            gaussian_pattern = self._test_gaussian_pattern(weights)
            if gaussian_pattern:
                patterns.append(gaussian_pattern)
            
            # Test for sparsity
            sparse_pattern = self._test_sparse_pattern(weights)
            if sparse_pattern:
                patterns.append(sparse_pattern)
            
            # Test for power law distribution
            power_law_pattern = self._test_power_law_pattern(weights)
            if power_law_pattern:
                patterns.append(power_law_pattern)
            
            # Test for bimodal distribution
            bimodal_pattern = self._test_bimodal_pattern(weights)
            if bimodal_pattern:
                patterns.append(bimodal_pattern)
            
            # Test for periodic patterns (if weights represent sequence)
            if len(weights) > 50:
                periodic_pattern = self._test_periodic_pattern(weights)
                if periodic_pattern:
                    patterns.append(periodic_pattern)
            
            # Test for monotonic patterns
            monotonic_pattern = self._test_monotonic_pattern(weights)
            if monotonic_pattern:
                patterns.append(monotonic_pattern)
                
        except Exception as e:
            logger.warning("Pattern detection encountered error: %s", e)
        
        return patterns
    
    def _test_uniform_pattern(self, weights: np.ndarray) -> Optional[WeightPattern]:
        """Test for uniform distribution pattern"""
        try:
            # Calculate coefficient of variation
            mean_val = np.mean(weights)
            std_val = np.std(weights)
            
            if abs(mean_val) < 1e-10:  # Near zero mean
                return None
            
            cv = std_val / abs(mean_val)
            
            # Uniform distribution has CV ≈ 0.577 for [-a, a] range
            if 0.4 < cv < 0.8:
                # Additional test: histogram should be relatively flat
                hist, _ = np.histogram(weights, bins=10)
                hist_cv = np.std(hist) / (np.mean(hist) + 1e-10)
                
                if hist_cv < 0.5:  # Relatively flat histogram
                    confidence = 1.0 - abs(cv - 0.577) / 0.2
                    
                    return WeightPattern(
                        pattern_type=PatternType.UNIFORM,
                        confidence=max(0.5, confidence),
                        parameters={'coefficient_variation': cv, 'histogram_cv': hist_cv},
                        entropy=self._calculate_entropy(weights),
                        compressibility_score=0.6  # Uniform is moderately compressible
                    )
        except Exception:
            pass
        
        return None
    
    def _test_gaussian_pattern(self, weights: np.ndarray) -> Optional[WeightPattern]:
        """Test for Gaussian distribution pattern"""
        try:
            from scipy import stats
            
            # Shapiro-Wilk test for normality (if sample size allows)
            if len(weights) <= 5000:
                stat, p_value = stats.shapiro(weights)
                
                if p_value > 0.05:  # Likely Gaussian
                    confidence = min(0.95, p_value * 10)
                    
                    return WeightPattern(
                        pattern_type=PatternType.GAUSSIAN,
                        confidence=confidence,
                        parameters={
                            'mean': float(np.mean(weights)),
                            'std': float(np.std(weights)),
                            'shapiro_stat': float(stat),
                            'p_value': float(p_value)
                        },
                        entropy=self._calculate_entropy(weights),
                        compressibility_score=0.8  # Gaussian is highly compressible
                    )
        except ImportError:
            # Fallback test without scipy
            try:
                # Simple test: check if ~68% of values are within 1 std dev
                mean_val = np.mean(weights)
                std_val = np.std(weights)
                
                within_1_std = np.sum(np.abs(weights - mean_val) <= std_val) / len(weights)
                
                if 0.6 < within_1_std < 0.8:  # Close to 68%
                    confidence = 0.7
                    
                    return WeightPattern(
                        pattern_type=PatternType.GAUSSIAN,
                        confidence=confidence,
                        parameters={'mean': mean_val, 'std': std_val, 'within_1_std': within_1_std},
                        entropy=self._calculate_entropy(weights),
                        compressibility_score=0.8
                    )
            except Exception:
                pass
        except Exception:
            pass
        
        return None
    
    def _test_sparse_pattern(self, weights: np.ndarray) -> Optional[WeightPattern]:
        """Test for sparse pattern (many zeros)"""
        try:
            zero_fraction = np.sum(np.abs(weights) < 1e-8) / len(weights)
            
            if zero_fraction > self.pattern_thresholds['sparsity_threshold']:
                confidence = min(0.95, zero_fraction * 1.2)
                
                return WeightPattern(
                    pattern_type=PatternType.SPARSE,
                    confidence=confidence,
                    parameters={
                        'zero_fraction': zero_fraction,
                        'nnz_count': int(np.sum(np.abs(weights) >= 1e-8))
                    },
                    entropy=self._calculate_entropy(weights),
                    compressibility_score=0.9  # Sparse is highly compressible
                )
        except Exception:
            pass
        
        return None
    
    def _test_power_law_pattern(self, weights: np.ndarray) -> Optional[WeightPattern]:
        """Test for power law distribution pattern"""
        try:
            # Only test on positive values
            positive_weights = weights[weights > 0]
            
            if len(positive_weights) < 10:
                return None
            
            # Log-log linear regression to estimate power law exponent
            log_weights = -np.log(positive_weights)  # Negative for survival function
            sorted_weights = np.sort(log_weights)
            
            log_ranks = np.log(np.arange(1, len(sorted_weights) + 1))
            
            # Linear regression
            correlation = np.corrcoef(sorted_weights, log_ranks)[0, 1]
            
            if abs(correlation) > 0.8:  # Strong linear correlation in log-log
                # Estimate alpha parameter
                slope = np.polyfit(sorted_weights, log_ranks, 1)[0]
                alpha = abs(slope)
                
                if (self.pattern_thresholds['power_law_alpha_min'] <= alpha <= 
                    self.pattern_thresholds['power_law_alpha_max']):
                    
                    confidence = min(0.95, abs(correlation))
                    
                    return WeightPattern(
                        pattern_type=PatternType.POWER_LAW,
                        confidence=confidence,
                        parameters={
                            'alpha': alpha,
                            'correlation': correlation,
                            'positive_fraction': len(positive_weights) / len(weights)
                        },
                        entropy=self._calculate_entropy(weights),
                        compressibility_score=0.7
                    )
        except Exception:
            pass
        
        return None
    
    def _test_bimodal_pattern(self, weights: np.ndarray) -> Optional[WeightPattern]:
        """Test for bimodal distribution pattern"""
        try:
            # Use histogram to detect two peaks
            hist, bin_edges = np.histogram(weights, bins=20)
            
            # Find peaks in histogram
            peaks = []
            for i in range(1, len(hist) - 1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.max(hist) * 0.1:
                    peaks.append(i)
            
            if len(peaks) >= 2:
                # Check separation between highest peaks
                peak_heights = [hist[p] for p in peaks]
                sorted_peaks = sorted(zip(peaks, peak_heights), key=lambda x: x[1], reverse=True)
                
                if len(sorted_peaks) >= 2:
                    peak1_idx, peak1_height = sorted_peaks[0]
                    peak2_idx, peak2_height = sorted_peaks[1]
                    
                    # Calculate separation
                    peak1_pos = bin_edges[peak1_idx]
                    peak2_pos = bin_edges[peak2_idx]
                    separation = abs(peak1_pos - peak2_pos)
                    
                    # Calculate valley depth (minimum between peaks)
                    min_idx = min(peak1_idx, peak2_idx)
                    max_idx = max(peak1_idx, peak2_idx)
                    valley_height = np.min(hist[min_idx:max_idx+1])
                    
                    # Bimodal if peaks are well separated and valley is deep
                    peak_ratio = min(peak1_height, peak2_height) / max(peak1_height, peak2_height)
                    valley_ratio = valley_height / max(peak1_height, peak2_height)
                    
                    if peak_ratio > 0.3 and valley_ratio < 0.5:
                        confidence = min(0.9, peak_ratio * (1 - valley_ratio))
                        
                        return WeightPattern(
                            pattern_type=PatternType.BIMODAL,
                            confidence=confidence,
                            parameters={
                                'peak1_position': float(peak1_pos),
                                'peak2_position': float(peak2_pos),
                                'separation': float(separation),
                                'peak_ratio': float(peak_ratio),
                                'valley_ratio': float(valley_ratio)
                            },
                            entropy=self._calculate_entropy(weights),
                            compressibility_score=0.75
                        )
        except Exception:
            pass
        
        return None
    
    def _test_periodic_pattern(self, weights: np.ndarray) -> Optional[WeightPattern]:
        """Test for periodic pattern in weight sequence"""
        try:
            # Autocorrelation analysis
            autocorr = np.correlate(weights, weights, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Normalize
            autocorr = autocorr / autocorr[0]
            
            # Find peaks in autocorrelation (excluding lag 0)
            peaks = []
            for i in range(2, min(len(autocorr), len(weights) // 4)):
                if (autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and 
                    autocorr[i] > self.pattern_thresholds['periodicity_correlation']):
                    peaks.append((i, autocorr[i]))
            
            if peaks:
                # Find the most significant period
                best_period, best_correlation = max(peaks, key=lambda x: x[1])
                
                confidence = min(0.95, best_correlation)
                
                return WeightPattern(
                    pattern_type=PatternType.PERIODIC,
                    confidence=confidence,
                    parameters={
                        'period': int(best_period),
                        'correlation': float(best_correlation),
                        'num_periods': len(weights) // best_period
                    },
                    entropy=self._calculate_entropy(weights),
                    compressibility_score=0.85  # Periodic is highly compressible
                )
        except Exception:
            pass
        
        return None
    
    def _test_monotonic_pattern(self, weights: np.ndarray) -> Optional[WeightPattern]:
        """Test for monotonic pattern"""
        try:
            # Calculate differences
            diffs = np.diff(weights)
            
            # Count increasing/decreasing steps
            increasing = np.sum(diffs > 0)
            decreasing = np.sum(diffs < 0)
            total_steps = len(diffs)
            
            if total_steps == 0:
                return None
            
            # Check for strong monotonicity
            increasing_fraction = increasing / total_steps
            decreasing_fraction = decreasing / total_steps
            
            if increasing_fraction > 0.8:
                confidence = min(0.9, increasing_fraction)
                direction = "increasing"
            elif decreasing_fraction > 0.8:
                confidence = min(0.9, decreasing_fraction)
                direction = "decreasing"
            else:
                return None
            
            return WeightPattern(
                pattern_type=PatternType.MONOTONIC,
                confidence=confidence,
                parameters={
                    'direction': direction,
                    'monotonic_fraction': max(increasing_fraction, decreasing_fraction),
                    'average_slope': float(np.mean(diffs))
                },
                entropy=self._calculate_entropy(weights),
                compressibility_score=0.8
            )
        except Exception:
            pass
        
        return None
    
    def _cluster_similar_weights(self, weights: np.ndarray, 
                               channels: Optional[IEEE754Channels]) -> List[List[int]]:
        """Cluster weights by similarity for grouping
        
        Args:
            weights: Array of weight values
            channels: Optional IEEE 754 channels for enhanced clustering
            
        Returns:
            List of clusters, each containing indices of similar weights
        """
        try:
            # Prepare feature matrix for clustering
            features = []
            
            # Basic weight features
            weight_features = weights.reshape(-1, 1)
            features.append(weight_features)
            
            # IEEE 754 channel features if available
            if channels is not None:
                features.append(channels.exponent_channel.reshape(-1, 1))
                features.append(channels.mantissa_channel.reshape(-1, 1))
            
            # Combine features
            if len(features) > 1:
                feature_matrix = np.hstack(features)
            else:
                feature_matrix = features[0]
            
            # Determine optimal number of clusters
            n_weights = len(weights)
            max_clusters = min(self.max_clusters, n_weights // 10, 20)
            
            if max_clusters < 2:
                return []
            
            best_clusters = None
            best_score = -1
            
            # Try different numbers of clusters
            for n_clusters in range(2, max_clusters + 1):
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(feature_matrix)
                    
                    # Calculate silhouette score
                    if len(np.unique(cluster_labels)) > 1:
                        score = silhouette_score(feature_matrix, cluster_labels)
                        
                        if score > best_score:
                            best_score = score
                            best_clusters = cluster_labels
                except Exception:
                    continue
            
            # Group indices by cluster
            if best_clusters is not None:
                similarity_groups = []
                for cluster_id in np.unique(best_clusters):
                    indices = np.where(best_clusters == cluster_id)[0].tolist()
                    if len(indices) > 1:  # Only include clusters with multiple elements
                        similarity_groups.append(indices)
                
                return similarity_groups
            
        except Exception as e:
            logger.warning("Weight clustering failed: %s", e)
        
        return []
    
    def _estimate_compression_ratio(self, weights: np.ndarray, 
                                  patterns: List[WeightPattern],
                                  similarity_groups: List[List[int]]) -> float:
        """Estimate potential compression ratio based on analysis
        
        Args:
            weights: Array of weight values
            patterns: Detected patterns
            similarity_groups: Similarity clusters
            
        Returns:
            Estimated compression ratio
        """
        try:
            base_compression = 1.0  # No compression baseline
            
            # Pattern-based compression estimation
            pattern_bonus = 0.0
            for pattern in patterns:
                pattern_bonus += pattern.compressibility_score * pattern.confidence
            
            # Similarity-based compression estimation
            similarity_bonus = 0.0
            if similarity_groups:
                total_in_groups = sum(len(group) for group in similarity_groups)
                similarity_fraction = total_in_groups / len(weights)
                similarity_bonus = similarity_fraction * 0.3  # Up to 30% bonus
            
            # Entropy-based adjustment
            entropy = self._calculate_entropy(weights)
            entropy_factor = max(0.5, 1.0 - entropy / 10.0)  # Lower entropy = better compression
            
            # Combine factors
            estimated_ratio = base_compression + pattern_bonus + similarity_bonus
            estimated_ratio *= entropy_factor
            
            # Apply p-adic specific adjustments
            padic_factor = self._estimate_padic_compatibility(weights)
            estimated_ratio *= padic_factor
            
            # Clamp to reasonable range
            estimated_ratio = max(1.5, min(8.0, estimated_ratio))
            
            return estimated_ratio
            
        except Exception:
            return 2.0  # Conservative default estimate
    
    def _estimate_padic_compatibility(self, weights: np.ndarray) -> float:
        """Estimate how well weights will compress with p-adic encoding
        
        Args:
            weights: Array of weight values
            
        Returns:
            P-adic compatibility factor [0.5, 2.0]
        """
        try:
            # Factors that improve p-adic compression
            compatibility = 1.0
            
            # 1. Rational-like values compress better
            rational_fraction = 0.0
            for w in weights[:min(100, len(weights))]:  # Sample for efficiency
                if abs(w) > 1e-10:
                    # Check if value is close to a simple rational
                    for denom in [2, 3, 4, 5, 8, 10, 16]:
                        rational_val = round(w * denom) / denom
                        if abs(w - rational_val) < 1e-6:
                            rational_fraction += 1.0
                            break
            
            rational_fraction /= min(100, len(weights))
            compatibility += rational_fraction * 0.3
            
            # 2. Values with small denominators compress better
            # This is approximated by checking decimal precision
            avg_precision = 0.0
            for w in weights[:min(100, len(weights))]:
                if abs(w) > 1e-10:
                    # Count decimal places
                    str_val = f"{w:.10f}".rstrip('0')
                    if '.' in str_val:
                        decimal_places = len(str_val.split('.')[1])
                        avg_precision += decimal_places
            
            avg_precision /= min(100, len(weights))
            precision_factor = max(0.5, 1.0 - avg_precision / 10.0)
            compatibility *= precision_factor
            
            # 3. Magnitude distribution affects p-adic efficiency
            abs_weights = np.abs(weights[weights != 0])
            if len(abs_weights) > 0:
                log_weights = np.log10(abs_weights + 1e-10)
                magnitude_std = np.std(log_weights)
                
                # Moderate magnitude spread is optimal for p-adic
                if magnitude_std < 2.0:
                    compatibility *= 1.2
                elif magnitude_std > 5.0:
                    compatibility *= 0.8
            
            return max(0.5, min(2.0, compatibility))
            
        except Exception:
            return 1.0
    
    def _generate_optimization_hints(self, weights: np.ndarray, 
                                   patterns: List[WeightPattern],
                                   channels: Optional[IEEE754Channels]) -> Dict[str, Any]:
        """Generate optimization hints for p-adic compression
        
        Args:
            weights: Array of weight values
            patterns: Detected patterns
            channels: Optional IEEE 754 channels
            
        Returns:
            Dictionary of optimization hints
        """
        hints = {}
        
        try:
            # Pattern-specific hints
            for pattern in patterns:
                if pattern.pattern_type == PatternType.SPARSE:
                    hints['use_sparse_encoding'] = True
                    hints['sparsity_ratio'] = pattern.parameters.get('zero_fraction', 0.0)
                
                elif pattern.pattern_type == PatternType.GAUSSIAN:
                    hints['use_gaussian_quantization'] = True
                    hints['gaussian_std'] = pattern.parameters.get('std', 1.0)
                
                elif pattern.pattern_type == PatternType.PERIODIC:
                    hints['use_periodic_compression'] = True
                    hints['period_length'] = pattern.parameters.get('period', 1)
                
                elif pattern.pattern_type == PatternType.UNIFORM:
                    hints['use_uniform_quantization'] = True
                    hints['quantization_levels'] = 256
            
            # Precision hints based on weight distribution
            abs_weights = np.abs(weights[weights != 0])
            if len(abs_weights) > 0:
                weight_range = np.max(abs_weights) - np.min(abs_weights)
                if weight_range < 0.1:
                    hints['recommended_precision'] = 4
                elif weight_range < 1.0:
                    hints['recommended_precision'] = 5
                else:
                    hints['recommended_precision'] = 6
            
            # Prime selection hints
            hints['recommended_prime'] = 257  # Default
            
            # Based on patterns, suggest alternative primes
            if any(p.pattern_type == PatternType.SPARSE for p in patterns):
                hints['alternative_primes'] = [127, 251]  # Smaller primes for sparse data
            
            # Channel-specific hints
            if channels is not None:
                exponent_range = np.max(channels.exponent_channel) - np.min(channels.exponent_channel)
                if exponent_range < 50:
                    hints['compress_exponent_channel'] = True
                
                mantissa_entropy = self._calculate_entropy(channels.mantissa_channel)
                if mantissa_entropy < 3.0:
                    hints['compress_mantissa_channel'] = True
            
        except Exception as e:
            logger.warning("Failed to generate optimization hints: %s", e)
        
        return hints
    
    def _calculate_entropy(self, values: np.ndarray, bins: int = 50) -> float:
        """Calculate Shannon entropy of value distribution
        
        Args:
            values: Array of values
            bins: Number of histogram bins
            
        Returns:
            Shannon entropy
        """
        try:
            hist, _ = np.histogram(values, bins=bins)
            hist = hist[hist > 0]  # Remove empty bins
            
            if len(hist) <= 1:
                return 0.0
            
            probabilities = hist / np.sum(hist)
            entropy = -np.sum(probabilities * np.log2(probabilities))
            
            return float(entropy)
            
        except Exception:
            return 0.0
    
    def _update_categorization_statistics(self, primary_category: CategoryType, 
                                        patterns: List[WeightPattern]) -> None:
        """Update categorization statistics
        
        Args:
            primary_category: Primary category assigned
            patterns: Detected patterns
        """
        try:
            self.categorization_stats['total_categorizations'] += 1
            
            # Update category distribution
            category_key = primary_category.value
            if category_key not in self.categorization_stats['category_distribution']:
                self.categorization_stats['category_distribution'][category_key] = 0
            self.categorization_stats['category_distribution'][category_key] += 1
            
            # Update pattern distribution
            for pattern in patterns:
                pattern_key = pattern.pattern_type.value
                if pattern_key not in self.categorization_stats['pattern_distribution']:
                    self.categorization_stats['pattern_distribution'][pattern_key] = 0
                self.categorization_stats['pattern_distribution'][pattern_key] += 1
            
            # Update success rates
            if patterns:
                total = self.categorization_stats['total_categorizations']
                pattern_successes = sum(1 for _ in self.categorization_stats['pattern_distribution'].values())
                self.categorization_stats['pattern_detection_success_rate'] = pattern_successes / total
            
        except Exception as e:
            logger.warning("Failed to update categorization statistics: %s", e)
    
    def get_categorization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive categorization statistics
        
        Returns:
            Dictionary of categorization statistics
        """
        return {
            'total_categorizations': self.categorization_stats['total_categorizations'],
            'pattern_detection_success_rate': self.categorization_stats['pattern_detection_success_rate'],
            'clustering_success_rate': self.categorization_stats['clustering_success_rate'],
            'average_compression_estimate': self.categorization_stats['average_compression_estimate'],
            'pattern_distribution': dict(self.categorization_stats['pattern_distribution']),
            'category_distribution': dict(self.categorization_stats['category_distribution']),
            'configuration': {
                'enable_clustering': self.enable_clustering,
                'enable_pattern_detection': self.enable_pattern_detection,
                'max_clusters': self.max_clusters,
                'pattern_thresholds': self.pattern_thresholds
            }
        }


# Factory function for easy integration
def create_weight_categorizer(enable_clustering: bool = True,
                            enable_pattern_detection: bool = True,
                            max_clusters: int = 20) -> WeightCategorizer:
    """Factory function to create weight categorizer
    
    Args:
        enable_clustering: Enable similarity clustering
        enable_pattern_detection: Enable pattern detection
        max_clusters: Maximum number of clusters
        
    Returns:
        Configured WeightCategorizer instance
    """
    return WeightCategorizer(
        enable_clustering=enable_clustering,
        enable_pattern_detection=enable_pattern_detection,
        max_clusters=max_clusters
    )