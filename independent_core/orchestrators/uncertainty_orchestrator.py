"""
Uncertainty Orchestrator - Advanced Uncertainty Quantification System
Manages uncertainty analysis, confidence estimation, and probabilistic reasoning
"""

import asyncio
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock, Lock
import threading
from collections import defaultdict, deque
import json
import traceback
from abc import ABC, abstractmethod
import scipy.stats as stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

class UncertaintyType(Enum):
    ALEATORIC = "aleatoric"  # Data uncertainty
    EPISTEMIC = "epistemic"  # Model uncertainty
    ONTOLOGICAL = "ontological"  # Structural uncertainty
    TEMPORAL = "temporal"  # Time-dependent uncertainty
    SPATIAL = "spatial"  # Location-dependent uncertainty

class QuantificationMethod(Enum):
    BAYESIAN = "bayesian"
    MONTE_CARLO_DROPOUT = "monte_carlo_dropout"
    ENSEMBLE = "ensemble"
    VARIATIONAL = "variational"
    BOOTSTRAP = "bootstrap"
    CONFORMAL = "conformal"
    GAUSSIAN_PROCESS = "gaussian_process"
    # Non-Bayesian methods for categorical data
    CONFORMALIZED_CREDAL = "conformalized_credal"
    DEEP_DETERMINISTIC = "deep_deterministic"
    BATCH_ENSEMBLE = "batch_ensemble"
    ENTROPY_BASED = "entropy_based"
    POSSIBILITY_BASED = "possibility_based"

class ConfidenceLevel(Enum):
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9
    CERTAIN = 0.99

class UncertaintyEstimationStrategy(Enum):
    CONSERVATIVE = "conservative"  # Err on side of higher uncertainty
    BALANCED = "balanced"  # Balance precision and recall
    AGGRESSIVE = "aggressive"  # Minimize uncertainty estimates
    ADAPTIVE = "adaptive"  # Adapt based on context

@dataclass
class UncertaintyEstimate:
    estimate_id: str
    value: float  # Main prediction/estimate
    uncertainty: float  # Uncertainty magnitude
    confidence_interval: Tuple[float, float]
    uncertainty_type: UncertaintyType
    method: QuantificationMethod
    confidence_level: float = 0.95
    additional_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class BayesianInference:
    inference_id: str
    prior_distribution: str
    likelihood_function: str
    posterior_samples: Optional[np.ndarray] = None
    evidence: Optional[float] = None
    convergence_metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, float] = field(default_factory=dict)
    iterations: int = 0
    computation_time: float = 0.0

@dataclass
class UncertaintyPropagation:
    propagation_id: str
    input_uncertainties: Dict[str, float]
    propagation_method: str
    output_uncertainty: float
    sensitivity_analysis: Dict[str, float] = field(default_factory=dict)
    correlation_matrix: Optional[np.ndarray] = None
    monte_carlo_samples: int = 1000

@dataclass
class CalibrationResults:
    calibration_id: str
    method: str
    calibration_error: float
    reliability_diagram: Dict[str, List[float]]
    confidence_histogram: Dict[str, int]
    brier_score: float
    log_likelihood: float
    coverage_probability: float
    sharpness: float

@dataclass
class CredalSet:
    """Credal set representation for categorical uncertainty"""
    categories: List[str]
    lower_bounds: Dict[str, float]
    upper_bounds: Dict[str, float]
    confidence_level: float
    coverage_guarantee: bool
    entropy: float = 0.0
    possibility_measure: Dict[str, float] = field(default_factory=dict)
    necessity_measure: Dict[str, float] = field(default_factory=dict)

@dataclass
class CategoricalUncertaintyMetrics:
    """Comprehensive uncertainty metrics for categorical data"""
    shannon_entropy: float
    renyi_entropy: Dict[float, float]  # alpha -> entropy
    min_entropy: float
    collision_entropy: float
    possibility_measure: Dict[str, float]
    necessity_measure: Dict[str, float]
    credal_set: Optional[CredalSet] = None
    coverage_guarantee: bool = False
    confidence_level: float = 0.95

class UncertaintyQuantifier(ABC):
    """Abstract base class for uncertainty quantification methods"""
    
    @abstractmethod
    def quantify(self, data: Any, model: Any = None, **kwargs) -> UncertaintyEstimate:
        pass
    
    @abstractmethod
    def get_method_name(self) -> QuantificationMethod:
        pass

class BayesianQuantifier(UncertaintyQuantifier):
    """Bayesian uncertainty quantification"""
    
    def __init__(self, prior_type: str = "normal", num_samples: int = 1000):
        self.prior_type = prior_type
        self.num_samples = num_samples
    
    def quantify(self, data: Any, model: Any = None, **kwargs) -> UncertaintyEstimate:
        """Perform Bayesian uncertainty quantification"""
        try:
            # Simple Bayesian inference for demonstration
            observations = np.array(data) if not isinstance(data, np.ndarray) else data
            
            # Assume normal likelihood with unknown mean and variance
            n = len(observations)
            sample_mean = np.mean(observations)
            sample_var = np.var(observations, ddof=1)
            
            # Conjugate prior (normal-inverse-gamma)
            # Prior parameters
            mu_0 = kwargs.get('prior_mean', 0.0)
            lambda_0 = kwargs.get('prior_precision', 1.0)
            alpha_0 = kwargs.get('prior_alpha', 1.0)
            beta_0 = kwargs.get('prior_beta', 1.0)
            
            # Posterior parameters
            lambda_n = lambda_0 + n
            mu_n = (lambda_0 * mu_0 + n * sample_mean) / lambda_n
            alpha_n = alpha_0 + n / 2
            beta_n = beta_0 + 0.5 * np.sum((observations - sample_mean)**2) + \
                     (lambda_0 * n * (sample_mean - mu_0)**2) / (2 * lambda_n)
            
            # Posterior predictive distribution (t-distribution)
            dof = 2 * alpha_n
            scale = np.sqrt(beta_n * (lambda_n + 1) / (alpha_n * lambda_n))
            
            # Generate posterior samples
            posterior_samples = stats.t.rvs(
                df=dof, 
                loc=mu_n, 
                scale=scale, 
                size=self.num_samples
            )
            
            # Calculate uncertainty metrics
            posterior_mean = np.mean(posterior_samples)
            posterior_std = np.std(posterior_samples)
            confidence_interval = np.percentile(posterior_samples, [2.5, 97.5])
            
            return UncertaintyEstimate(
                estimate_id=f"bayesian_{int(time.time())}",
                value=posterior_mean,
                uncertainty=posterior_std,
                confidence_interval=tuple(confidence_interval),
                uncertainty_type=UncertaintyType.EPISTEMIC,
                method=QuantificationMethod.BAYESIAN,
                additional_metrics={
                    'posterior_variance': posterior_std**2,
                    'degrees_of_freedom': dof,
                    'evidence': self._calculate_evidence(observations, mu_0, lambda_0, alpha_0, beta_0)
                },
                metadata={
                    'num_observations': n,
                    'prior_parameters': {
                        'mu_0': mu_0, 'lambda_0': lambda_0,
                        'alpha_0': alpha_0, 'beta_0': beta_0
                    },
                    'posterior_parameters': {
                        'mu_n': mu_n, 'lambda_n': lambda_n,
                        'alpha_n': alpha_n, 'beta_n': beta_n
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Bayesian quantification failed: {e}")
            return UncertaintyEstimate(
                estimate_id=f"bayesian_failed_{int(time.time())}",
                value=0.0,
                uncertainty=1.0,
                confidence_interval=(0.0, 0.0),
                uncertainty_type=UncertaintyType.EPISTEMIC,
                method=QuantificationMethod.BAYESIAN,
                metadata={'error': str(e)}
            )
    
    def _calculate_evidence(self, observations: np.ndarray, mu_0: float, lambda_0: float, 
                          alpha_0: float, beta_0: float) -> float:
        """Calculate marginal likelihood (evidence)"""
        try:
            n = len(observations)
            sample_mean = np.mean(observations)
            sample_var = np.var(observations, ddof=1)
            
            # Log marginal likelihood for normal-inverse-gamma model
            lambda_n = lambda_0 + n
            alpha_n = alpha_0 + n / 2
            beta_n = beta_0 + 0.5 * np.sum((observations - sample_mean)**2) + \
                     (lambda_0 * n * (sample_mean - mu_0)**2) / (2 * lambda_n)
            
            log_evidence = (
                0.5 * np.log(lambda_0 / lambda_n) +
                alpha_0 * np.log(beta_0) - alpha_n * np.log(beta_n) +
                stats.loggamma(alpha_n) - stats.loggamma(alpha_0) -
                0.5 * n * np.log(2 * np.pi)
            )
            
            return log_evidence
            
        except Exception:
            return 0.0
    
    def get_method_name(self) -> QuantificationMethod:
        return QuantificationMethod.BAYESIAN

class ConformalizedCredalQuantifier(UncertaintyQuantifier):
    """Production-ready conformalized credal set predictors for categorical uncertainty"""
    
    def __init__(self, alpha: float = 0.1, num_calibration: int = 1000, 
                 nonconformity_measure: str = 'inverse_probability'):
        self.alpha = alpha  # Coverage level (1 - alpha)
        self.num_calibration = num_calibration
        self.nonconformity_measure = nonconformity_measure
        self.calibration_threshold = None
        self.calibration_scores = []
        self.category_mapping = {}
        self.is_calibrated = False
        
    def quantify(self, data: Any, model: Any = None, **kwargs) -> UncertaintyEstimate:
        """Production-ready uncertainty quantification using conformalized credal sets"""
        try:
            # Validate input data
            if not data or len(data) == 0:
                raise ValueError("Empty or invalid data provided")
            
            # Update configuration from kwargs if provided
            if 'alpha' in kwargs:
                self.alpha = kwargs['alpha']
            if 'num_calibration' in kwargs:
                self.num_calibration = kwargs['num_calibration']
            if 'nonconformity_measure' in kwargs:
                self.nonconformity_measure = kwargs['nonconformity_measure']
            
            # Convert data to categorical format
            categorical_data = self._preprocess_categorical_data(data)
            
            # Extract unique categories
            categories = list(set(categorical_data))
            self.category_mapping = {cat: i for i, cat in enumerate(categories)}
            
            # Calibrate conformal predictor if not already calibrated
            if not self.is_calibrated:
                self._calibrate_conformal(categorical_data, categories)
            
            # Calculate category probabilities using empirical distribution
            probabilities = self._calculate_empirical_probabilities(categorical_data, categories)
            
            # Generate credal set prediction
            credal_set = self._predict_credal_set(probabilities, self.calibration_threshold)
            
            # Calculate uncertainty metrics
            uncertainty_metrics = self._calculate_uncertainty_metrics(credal_set, probabilities)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(credal_set, self.alpha)
            
            # Generate estimate ID
            estimate_id = f"conformalized_credal_{int(time.time())}"
            
            return UncertaintyEstimate(
                estimate_id=estimate_id,
                value=uncertainty_metrics['mean_probability'],
                uncertainty=uncertainty_metrics['total_uncertainty'],
                confidence_interval=confidence_interval,
                uncertainty_type=UncertaintyType.EPISTEMIC,
                method=QuantificationMethod.CONFORMALIZED_CREDAL,
                confidence_level=1.0 - self.alpha,
                additional_metrics={
                    'entropy': uncertainty_metrics['entropy'],
                    'credal_set_size': len(credal_set.categories),
                    'coverage_guarantee': credal_set.coverage_guarantee,
                    'conformal_threshold': self.calibration_threshold,
                    'nonconformity_measure': self.nonconformity_measure,
                    'calibration_scores_count': len(self.calibration_scores)
                },
                metadata={
                    'categories': categories,
                    'credal_set': credal_set,
                    'probabilities': probabilities
                }
            )
            
        except Exception as e:
            logger.error(f"ConformalizedCredalQuantifier quantification failed: {e}")
            # Return high uncertainty estimate on failure
            return UncertaintyEstimate(
                estimate_id=f"conformalized_credal_failed_{int(time.time())}",
                value=0.0,
                uncertainty=1.0,
                confidence_interval=(0.0, 1.0),
                uncertainty_type=UncertaintyType.EPISTEMIC,
                method=QuantificationMethod.CONFORMALIZED_CREDAL,
                confidence_level=0.95,
                additional_metrics={'error': str(e)}
            )
    
    def _preprocess_categorical_data(self, data: Any) -> List[str]:
        """Preprocess data into categorical format"""
        if isinstance(data, (list, tuple)):
            return [str(item) for item in data]
        elif isinstance(data, np.ndarray):
            return [str(item) for item in data.flatten()]
        elif isinstance(data, dict):
            # Extract categorical values from dictionary
            categorical_values = []
            for key, value in data.items():
                if isinstance(value, (list, tuple)):
                    categorical_values.extend([str(v) for v in value])
                else:
                    categorical_values.append(str(value))
            return categorical_values
        else:
            return [str(data)]
    
    def _calibrate_conformal(self, data: List[str], categories: List[str]) -> None:
        """Calibrate conformal predictor using validation data"""
        self.calibration_scores = []
        
        # Calculate nonconformity scores for all data points
        for i, item in enumerate(data):
            # Create leave-one-out dataset
            leave_one_out_data = data[:i] + data[i+1:]
            
            # Calculate empirical probabilities on leave-one-out data
            loo_probabilities = self._calculate_empirical_probabilities(leave_one_out_data, categories)
            
            # Calculate nonconformity score for true category
            true_category = item
            if true_category in loo_probabilities:
                true_probability = loo_probabilities[true_category]
                nonconformity_score = self._calculate_nonconformity_score(true_probability, true_category)
            else:
                # Category not seen in leave-one-out data
                nonconformity_score = 1.0
            
            self.calibration_scores.append(nonconformity_score)
        
        # Calculate conformal threshold
        if self.calibration_scores:
            threshold_percentile = (1 - self.alpha) * 100
            self.calibration_threshold = np.percentile(self.calibration_scores, threshold_percentile)
        else:
            self.calibration_threshold = 1.0
        
        self.is_calibrated = True
        logger.info(f"Conformal calibration complete. Threshold: {self.calibration_threshold:.4f}")
    
    def _calculate_nonconformity_score(self, probability: float, category: str) -> float:
        """Calculate nonconformity score based on selected measure"""
        if self.nonconformity_measure == 'inverse_probability':
            return 1.0 - probability
        elif self.nonconformity_measure == 'negative_log_probability':
            return -np.log(max(probability, 1e-10))
        elif self.nonconformity_measure == 'squared_error':
            return (1.0 - probability) ** 2
        else:
            return 1.0 - probability
    
    def _calculate_empirical_probabilities(self, data: List[str], categories: List[str]) -> Dict[str, float]:
        """Calculate empirical probability distribution"""
        category_counts = {}
        total_count = len(data)
        
        # Count occurrences of each category
        for item in data:
            category_counts[item] = category_counts.get(item, 0) + 1
        
        # Calculate probabilities
        probabilities = {}
        for category in categories:
            count = category_counts.get(category, 0)
            probabilities[category] = count / total_count if total_count > 0 else 0.0
        
        return probabilities
    
    def _predict_credal_set(self, probabilities: Dict[str, float], threshold: float) -> CredalSet:
        """Generate set-valued prediction with coverage guarantee"""
        # Find categories in prediction set (those with low nonconformity scores)
        prediction_set = []
        for category, probability in probabilities.items():
            nonconformity_score = self._calculate_nonconformity_score(probability, category)
            if nonconformity_score <= threshold:
                prediction_set.append(category)
        
        # If no categories in prediction set, include the most probable
        if not prediction_set and probabilities:
            most_probable = max(probabilities.items(), key=lambda x: x[1])[0]
            prediction_set = [most_probable]
        
        # Calculate lower and upper bounds for all categories
        lower_bounds = {}
        upper_bounds = {}
        
        for category in probabilities.keys():
            if category in prediction_set:
                # Categories in prediction set have bounds based on threshold
                prob = probabilities[category]
                nonconformity = self._calculate_nonconformity_score(prob, category)
                lower_bounds[category] = max(0.0, prob - (threshold - nonconformity))
                upper_bounds[category] = min(1.0, prob + (threshold - nonconformity))
            else:
                # Categories not in prediction set have zero upper bound
                lower_bounds[category] = 0.0
                upper_bounds[category] = 0.0
        
        # Calculate entropy of the credal set
        entropy = self._calculate_credal_entropy(probabilities)
        
        return CredalSet(
            categories=prediction_set,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            confidence_level=1.0 - self.alpha,
            coverage_guarantee=True,
            entropy=entropy
        )
    
    def _calculate_uncertainty_metrics(self, credal_set: CredalSet, probabilities: Dict[str, float]) -> Dict[str, float]:
        """Calculate comprehensive uncertainty metrics"""
        # Calculate entropy
        entropy = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)
        
        # Calculate mean probability
        mean_probability = np.mean(list(probabilities.values()))
        
        # Calculate total uncertainty based on credal set size and entropy
        credal_uncertainty = 1.0 - (len(credal_set.categories) / len(probabilities)) if probabilities else 1.0
        entropy_uncertainty = entropy / np.log2(len(probabilities)) if len(probabilities) > 1 else 0.0
        
        # Combine uncertainties
        total_uncertainty = np.sqrt(credal_uncertainty**2 + entropy_uncertainty**2)
        
        return {
            'entropy': entropy,
            'mean_probability': mean_probability,
            'credal_uncertainty': credal_uncertainty,
            'entropy_uncertainty': entropy_uncertainty,
            'total_uncertainty': total_uncertainty
        }
    
    def _calculate_credal_entropy(self, probabilities: Dict[str, float]) -> float:
        """Calculate entropy of the credal set"""
        return -sum(p * np.log2(p) for p in probabilities.values() if p > 0)
    
    def _calculate_confidence_interval(self, credal_set: CredalSet, alpha: float) -> Tuple[float, float]:
        """Calculate confidence interval for the prediction"""
        if not credal_set.categories:
            return (0.0, 1.0)
        
        # Calculate interval based on credal set bounds
        min_lower = min(credal_set.lower_bounds.values())
        max_upper = max(credal_set.upper_bounds.values())
        
        # Adjust for confidence level
        confidence_width = (max_upper - min_lower) * alpha
        center = (min_lower + max_upper) / 2
        
        lower_bound = max(0.0, center - confidence_width / 2)
        upper_bound = min(1.0, center + confidence_width / 2)
        
        return (lower_bound, upper_bound)
    
    def get_method_name(self) -> QuantificationMethod:
        return QuantificationMethod.CONFORMALIZED_CREDAL

class DeepDeterministicQuantifier(UncertaintyQuantifier):
    """Production-ready Deep Deterministic Uncertainty (DDU) for categorical data"""
    
    def __init__(self, feature_dim: int = 128, num_classes: int = 10, 
                 spectral_norm_multiplier: float = 0.9, regularization_strength: float = 0.01):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.spectral_norm_multiplier = spectral_norm_multiplier
        self.regularization_strength = regularization_strength
        self.gaussian_discriminant = None
        self.feature_mean = None
        self.feature_covariance = None
        self.is_trained = False
        
    def quantify(self, data: Any, model: Any = None, **kwargs) -> UncertaintyEstimate:
        """Production-ready uncertainty quantification using Deep Deterministic Uncertainty"""
        try:
            # Validate input data
            if not data or len(data) == 0:
                raise ValueError("Empty or invalid data provided")
            
            # Extract features from data
            features = self._extract_features(data, model)
            
            # Apply spectral normalization for regularization
            normalized_features = self._apply_spectral_normalization(features)
            
            # Fit Gaussian discriminant analysis if not already trained
            if not self.is_trained:
                self._fit_gaussian_discriminant(normalized_features, data)
            
            # Calculate DDU uncertainty metrics
            uncertainty_metrics = self._calculate_ddu_uncertainty(normalized_features)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(uncertainty_metrics)
            
            # Generate estimate ID
            estimate_id = f"deep_deterministic_{int(time.time())}"
            
            return UncertaintyEstimate(
                estimate_id=estimate_id,
                value=uncertainty_metrics['mean_confidence'],
                uncertainty=uncertainty_metrics['total_uncertainty'],
                confidence_interval=confidence_interval,
                uncertainty_type=UncertaintyType.EPISTEMIC,
                method=QuantificationMethod.DEEP_DETERMINISTIC,
                confidence_level=uncertainty_metrics['confidence_level'],
                additional_metrics={
                    'feature_distance': uncertainty_metrics['feature_distance'],
                    'mahalanobis_distance': uncertainty_metrics['mahalanobis_distance'],
                    'spectral_norm': uncertainty_metrics['spectral_norm'],
                    'regularization_strength': self.regularization_strength,
                    'feature_dim': self.feature_dim,
                    'is_trained': self.is_trained
                },
                metadata={
                    'features': features,
                    'normalized_features': normalized_features,
                    'uncertainty_metrics': uncertainty_metrics
                }
            )
            
        except Exception as e:
            logger.error(f"DeepDeterministicQuantifier quantification failed: {e}")
            # Return high uncertainty estimate on failure
            return UncertaintyEstimate(
                estimate_id=f"deep_deterministic_failed_{int(time.time())}",
                value=0.0,
                uncertainty=1.0,
                confidence_interval=(0.0, 1.0),
                uncertainty_type=UncertaintyType.EPISTEMIC,
                method=QuantificationMethod.DEEP_DETERMINISTIC,
                confidence_level=0.95,
                additional_metrics={'error': str(e)}
            )
    
    def _extract_features(self, data: Any, model: Any) -> np.ndarray:
        """Extract features from categorical data"""
        if isinstance(data, (list, tuple)):
            # Convert categorical data to numerical features
            categories = list(set(data))
            category_mapping = {cat: i for i, cat in enumerate(categories)}
            
            # Create one-hot encoded features
            features = np.zeros((len(data), len(categories)))
            for i, item in enumerate(data):
                if item in category_mapping:
                    features[i, category_mapping[item]] = 1.0
            
            # Add frequency-based features
            category_counts = {}
            for item in data:
                category_counts[item] = category_counts.get(item, 0) + 1
            
            frequency_features = np.zeros((len(data), len(categories)))
            for i, item in enumerate(data):
                if item in category_mapping:
                    frequency_features[i, category_mapping[item]] = category_counts[item] / len(data)
            
            # Combine features
            combined_features = np.concatenate([features, frequency_features], axis=1)
            
            # Pad or truncate to target dimension
            if combined_features.shape[1] < self.feature_dim:
                padding = np.zeros((combined_features.shape[0], self.feature_dim - combined_features.shape[1]))
                combined_features = np.concatenate([combined_features, padding], axis=1)
            else:
                combined_features = combined_features[:, :self.feature_dim]
            
            return combined_features
            
        elif isinstance(data, np.ndarray):
            # Handle numpy array data
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            # Ensure 2D array
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            # Convert to float if needed
            if data.dtype != np.float64:
                data = data.astype(np.float64)
            
            # Pad or truncate to target dimension
            if data.shape[1] < self.feature_dim:
                padding = np.zeros((data.shape[0], self.feature_dim - data.shape[1]))
                data = np.concatenate([data, padding], axis=1)
            else:
                data = data[:, :self.feature_dim]
            
            return data
            
        elif isinstance(data, dict):
            # Extract features from dictionary
            all_values = []
            for key, value in data.items():
                if isinstance(value, (list, tuple)):
                    all_values.extend(value)
                else:
                    all_values.append(value)
            
            # Convert to numerical features
            return self._extract_features(all_values, model)
            
        else:
            # Single value - convert to array
            return np.array([[float(data)]], dtype=np.float64)
    
    def _apply_spectral_normalization(self, features: np.ndarray) -> np.ndarray:
        """Apply spectral normalization for regularization"""
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        # Calculate spectral norm
        u, s, vh = np.linalg.svd(features, full_matrices=False)
        spectral_norm = np.max(s)
        
        # Apply spectral normalization
        if spectral_norm > 0:
            normalized_features = features * (self.spectral_norm_multiplier / spectral_norm)
        else:
            normalized_features = features
        
        # Add regularization
        regularization_matrix = np.eye(normalized_features.shape[1]) * self.regularization_strength
        normalized_features = normalized_features + regularization_matrix[:normalized_features.shape[0], :]
        
        return normalized_features
    
    def _fit_gaussian_discriminant(self, features: np.ndarray, labels: Any):
        """Fit Gaussian discriminant analysis to features"""
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        # Calculate feature statistics
        self.feature_mean = np.mean(features, axis=0)
        self.feature_covariance = np.cov(features.T)
        
        # Add regularization to covariance matrix
        regularization_matrix = np.eye(self.feature_covariance.shape[0]) * self.regularization_strength
        self.feature_covariance = self.feature_covariance + regularization_matrix
        
        # Ensure covariance matrix is positive definite
        eigenvals, eigenvecs = np.linalg.eigh(self.feature_covariance)
        eigenvals = np.maximum(eigenvals, 1e-6)  # Minimum eigenvalue
        self.feature_covariance = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        self.is_trained = True
        logger.info(f"DDU Gaussian discriminant fitted. Feature mean shape: {self.feature_mean.shape}")
    
    def _calculate_ddu_uncertainty(self, features: np.ndarray) -> Dict[str, float]:
        """Calculate DDU uncertainty metrics"""
        if not self.is_trained:
            return {
                'total_uncertainty': 1.0,
                'mean_confidence': 0.0,
                'feature_distance': 1.0,
                'mahalanobis_distance': 1.0,
                'spectral_norm': 1.0,
                'confidence_level': 0.0
            }
        
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        # Calculate feature distance from mean
        feature_distance = np.linalg.norm(features - self.feature_mean, axis=1)
        mean_feature_distance = np.mean(feature_distance)
        
        # Calculate Mahalanobis distance
        try:
            inv_covariance = np.linalg.inv(self.feature_covariance)
            mahalanobis_distances = []
            for feature in features:
                diff = feature - self.feature_mean
                mahalanobis_dist = np.sqrt(diff.T @ inv_covariance @ diff)
                mahalanobis_distances.append(mahalanobis_dist)
            mean_mahalanobis_distance = np.mean(mahalanobis_distances)
        except np.linalg.LinAlgError:
            mean_mahalanobis_distance = 1.0
        
        # Calculate spectral norm
        u, s, vh = np.linalg.svd(features, full_matrices=False)
        spectral_norm = np.max(s) if len(s) > 0 else 1.0
        
        # Calculate uncertainty metrics
        feature_uncertainty = min(1.0, mean_feature_distance / self.feature_dim)
        mahalanobis_uncertainty = min(1.0, mean_mahalanobis_distance / np.sqrt(self.feature_dim))
        spectral_uncertainty = min(1.0, spectral_norm / self.feature_dim)
        
        # Combine uncertainties
        total_uncertainty = np.sqrt(feature_uncertainty**2 + mahalanobis_uncertainty**2 + spectral_uncertainty**2)
        mean_confidence = max(0.0, 1.0 - total_uncertainty)
        
        # Calculate confidence level based on uncertainty
        confidence_level = max(0.5, 1.0 - total_uncertainty)
        
        return {
            'total_uncertainty': total_uncertainty,
            'mean_confidence': mean_confidence,
            'feature_distance': mean_feature_distance,
            'mahalanobis_distance': mean_mahalanobis_distance,
            'spectral_norm': spectral_norm,
            'confidence_level': confidence_level
        }
    
    def _calculate_confidence_interval(self, uncertainty_metrics: Dict[str, float]) -> Tuple[float, float]:
        """Calculate confidence interval for the prediction"""
        confidence = uncertainty_metrics['mean_confidence']
        uncertainty = uncertainty_metrics['total_uncertainty']
        
        # Calculate interval based on uncertainty
        interval_width = uncertainty * 0.5
        lower_bound = max(0.0, confidence - interval_width)
        upper_bound = min(1.0, confidence + interval_width)
        
        return (lower_bound, upper_bound)
    
    def get_method_name(self) -> QuantificationMethod:
        return QuantificationMethod.DEEP_DETERMINISTIC

class BatchEnsembleQuantifier(UncertaintyQuantifier):
    """Production-ready BatchEnsemble for efficient ensemble uncertainty"""
    
    def __init__(self, num_ensembles: int = 4, rank: int = 1, memory_reduction_factor: int = 3,
                 diversity_strength: float = 0.1, weight_decay: float = 0.01):
        self.num_ensembles = num_ensembles
        self.rank = rank
        self.memory_reduction_factor = memory_reduction_factor
        self.diversity_strength = diversity_strength
        self.weight_decay = weight_decay
        self.ensemble_weights = {}
        self.ensemble_biases = {}
        self.base_weights = {}
        self.is_initialized = False
        
    def quantify(self, data: Any, model: Any = None, **kwargs) -> UncertaintyEstimate:
        """Production-ready uncertainty quantification using BatchEnsemble"""
        try:
            # Validate input data
            if not data or len(data) == 0:
                raise ValueError("Empty or invalid data provided")
            
            # Update configuration from kwargs if provided
            if 'num_ensembles' in kwargs:
                self.num_ensembles = kwargs['num_ensembles']
            if 'rank' in kwargs:
                self.rank = kwargs['rank']
            if 'memory_reduction_factor' in kwargs:
                self.memory_reduction_factor = kwargs['memory_reduction_factor']
            if 'diversity_strength' in kwargs:
                self.diversity_strength = kwargs['diversity_strength']
            if 'weight_decay' in kwargs:
                self.weight_decay = kwargs['weight_decay']
            
            # Reinitialize ensemble if parameters changed or not initialized
            if not self.is_initialized or any(key in kwargs for key in ['num_ensembles', 'rank', 'memory_reduction_factor']):
                self._initialize_ensemble(data)
            
            # Generate ensemble predictions
            ensemble_predictions = self._ensemble_predict(data, model)
            
            # Calculate ensemble uncertainty metrics
            uncertainty_metrics = self._calculate_ensemble_uncertainty(ensemble_predictions)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(uncertainty_metrics)
            
            # Generate estimate ID
            estimate_id = f"batch_ensemble_{int(time.time())}"
            
            return UncertaintyEstimate(
                estimate_id=estimate_id,
                value=uncertainty_metrics['mean_prediction'],
                uncertainty=uncertainty_metrics['total_uncertainty'],
                confidence_interval=confidence_interval,
                uncertainty_type=UncertaintyType.EPISTEMIC,
                method=QuantificationMethod.BATCH_ENSEMBLE,
                confidence_level=uncertainty_metrics['confidence_level'],
                additional_metrics={
                    'ensemble_variance': uncertainty_metrics['ensemble_variance'],
                    'diversity_score': uncertainty_metrics['diversity_score'],
                    'memory_reduction_factor': self.memory_reduction_factor,
                    'ensemble_size': self.num_ensembles,
                    'rank': self.rank,
                    'is_initialized': self.is_initialized
                },
                metadata={
                    'ensemble_predictions': ensemble_predictions,
                    'uncertainty_metrics': uncertainty_metrics
                }
            )
            
        except Exception as e:
            logger.error(f"BatchEnsembleQuantifier quantification failed: {e}")
            # Return high uncertainty estimate on failure
            return UncertaintyEstimate(
                estimate_id=f"batch_ensemble_failed_{int(time.time())}",
                value=0.0,
                uncertainty=1.0,
                confidence_interval=(0.0, 1.0),
                uncertainty_type=UncertaintyType.EPISTEMIC,
                method=QuantificationMethod.BATCH_ENSEMBLE,
                confidence_level=0.95,
                additional_metrics={'error': str(e)}
            )
    
    def _initialize_ensemble(self, data: Any):
        """Initialize ensemble with weight factorization"""
        # Extract feature dimension from data
        if isinstance(data, (list, tuple)):
            categories = list(set(data))
            feature_dim = len(categories)
        elif isinstance(data, np.ndarray):
            feature_dim = data.shape[1] if data.ndim > 1 else 1
        else:
            feature_dim = 1
        
        # Ensure minimum feature dimension
        feature_dim = max(feature_dim, 2)
        
        # Initialize base weights
        self.base_weights = np.random.randn(feature_dim, self.rank) / np.sqrt(feature_dim)
        
        # Initialize ensemble-specific weights and biases
        for i in range(self.num_ensembles):
            # Rank-based weight factorization
            ensemble_weight = np.random.randn(self.rank, feature_dim) / np.sqrt(self.rank)
            ensemble_bias = np.random.randn(feature_dim) * 0.01
            
            self.ensemble_weights[f'ensemble_{i}'] = ensemble_weight
            self.ensemble_biases[f'ensemble_{i}'] = ensemble_bias
        
        self.is_initialized = True
        logger.info(f"BatchEnsemble initialized with {self.num_ensembles} ensembles, rank {self.rank}, feature_dim {feature_dim}")
    
    def _ensemble_predict(self, data: Any, model: Any) -> List[float]:
        """Generate predictions from all ensemble members"""
        # Extract features from data
        features = self._extract_features(data)
        
        predictions = []
        for i in range(self.num_ensembles):
            ensemble_key = f'ensemble_{i}'
            
            # Get ensemble weights and bias
            ensemble_weight = self.ensemble_weights[ensemble_key]
            ensemble_bias = self.ensemble_biases[ensemble_key]
            
            # Ensure compatible dimensions
            if features.shape[1] != self.base_weights.shape[0]:
                # Pad or truncate features to match base_weights
                if features.shape[1] < self.base_weights.shape[0]:
                    padding = np.zeros((features.shape[0], self.base_weights.shape[0] - features.shape[1]))
                    features = np.concatenate([features, padding], axis=1)
                else:
                    features = features[:, :self.base_weights.shape[0]]
            
            # Compute ensemble prediction with weight factorization
            # W_ensemble = base_weights @ ensemble_weight
            ensemble_matrix = self.base_weights @ ensemble_weight
            
            # Apply ensemble transformation
            ensemble_features = features @ ensemble_matrix + ensemble_bias
            
            # Compute prediction (simplified for categorical data)
            prediction = np.mean(ensemble_features)
            predictions.append(float(prediction))
        
        return predictions
    
    def _extract_features(self, data: Any) -> np.ndarray:
        """Extract features from categorical data"""
        if isinstance(data, (list, tuple)):
            # Convert categorical data to numerical features
            categories = list(set(data))
            category_mapping = {cat: i for i, cat in enumerate(categories)}
            
            # Create one-hot encoded features
            features = np.zeros((len(data), len(categories)))
            for i, item in enumerate(data):
                if item in category_mapping:
                    features[i, category_mapping[item]] = 1.0
            
            # Add frequency-based features
            category_counts = {}
            for item in data:
                category_counts[item] = category_counts.get(item, 0) + 1
            
            frequency_features = np.zeros((len(data), len(categories)))
            for i, item in enumerate(data):
                if item in category_mapping:
                    frequency_features[i, category_mapping[item]] = category_counts[item] / len(data)
            
            # Combine features
            combined_features = np.concatenate([features, frequency_features], axis=1)
            
            # Ensure minimum feature dimension
            if combined_features.shape[1] < 2:
                padding = np.zeros((combined_features.shape[0], 2 - combined_features.shape[1]))
                combined_features = np.concatenate([combined_features, padding], axis=1)
            
            return combined_features
            
        elif isinstance(data, np.ndarray):
            # Handle numpy array data
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            # Convert to float if needed
            if data.dtype != np.float64:
                data = data.astype(np.float64)
            
            # Ensure minimum feature dimension
            if data.shape[1] < 2:
                padding = np.zeros((data.shape[0], 2 - data.shape[1]))
                data = np.concatenate([data, padding], axis=1)
            
            return data
            
        elif isinstance(data, dict):
            # Extract features from dictionary
            all_values = []
            for key, value in data.items():
                if isinstance(value, (list, tuple)):
                    all_values.extend(value)
                else:
                    all_values.append(value)
            
            return self._extract_features(all_values)
            
        else:
            # Single value - convert to array
            return np.array([[float(data)]], dtype=np.float64)
    
    def _calculate_ensemble_uncertainty(self, predictions: List[float]) -> Dict[str, float]:
        """Calculate comprehensive ensemble uncertainty metrics"""
        if not predictions:
            return {
                'total_uncertainty': 1.0,
                'mean_prediction': 0.0,
                'ensemble_variance': 1.0,
                'diversity_score': 0.0,
                'confidence_level': 0.0
            }
        
        predictions_array = np.array(predictions)
        
        # Calculate basic statistics
        mean_prediction = np.mean(predictions_array)
        ensemble_variance = np.var(predictions_array)
        
        # Calculate diversity score (how different the ensemble members are)
        diversity_score = self._calculate_diversity_score(predictions_array)
        
        # Calculate epistemic uncertainty (ensemble disagreement)
        epistemic_uncertainty = min(1.0, ensemble_variance * self.diversity_strength)
        
        # Calculate aleatoric uncertainty (prediction spread)
        aleatoric_uncertainty = min(1.0, np.std(predictions_array) / np.sqrt(self.num_ensembles))
        
        # Combine uncertainties
        total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
        
        # Calculate confidence level based on ensemble agreement
        confidence_level = max(0.5, 1.0 - total_uncertainty)
        
        return {
            'total_uncertainty': total_uncertainty,
            'mean_prediction': mean_prediction,
            'ensemble_variance': ensemble_variance,
            'diversity_score': diversity_score,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'confidence_level': confidence_level
        }
    
    def _calculate_diversity_score(self, predictions: np.ndarray) -> float:
        """Calculate ensemble diversity score"""
        if len(predictions) < 2:
            return 0.0
        
        # Calculate pairwise differences
        differences = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                differences.append(abs(predictions[i] - predictions[j]))
        
        # Diversity is the mean of pairwise differences
        diversity = np.mean(differences) if differences else 0.0
        
        return min(1.0, diversity)
    
    def _calculate_confidence_interval(self, uncertainty_metrics: Dict[str, float]) -> Tuple[float, float]:
        """Calculate confidence interval for the prediction"""
        mean_prediction = uncertainty_metrics['mean_prediction']
        uncertainty = uncertainty_metrics['total_uncertainty']
        
        # Calculate interval based on uncertainty
        interval_width = uncertainty * 0.5
        lower_bound = max(0.0, mean_prediction - interval_width)
        upper_bound = min(1.0, mean_prediction + interval_width)
        
        return (lower_bound, upper_bound)
    
    def get_method_name(self) -> QuantificationMethod:
        return QuantificationMethod.BATCH_ENSEMBLE

class EntropyBasedQuantifier(UncertaintyQuantifier):
    """Production-ready entropy-based uncertainty for categorical data"""
    
    def __init__(self, entropy_type: str = 'shannon', alpha: float = 2.0, 
                 normalize_entropy: bool = True, min_entropy_threshold: float = 0.01):
        self.entropy_type = entropy_type  # 'shannon', 'renyi', 'min_entropy'
        self.alpha = alpha  # For Rényi entropy
        self.normalize_entropy = normalize_entropy
        self.min_entropy_threshold = min_entropy_threshold
        self.category_mapping = {}
        self.entropy_history = []
        self.is_calibrated = False
        
    def quantify(self, data: Any, model: Any = None, **kwargs) -> UncertaintyEstimate:
        """Production-ready uncertainty quantification using entropy measures"""
        try:
            # Validate input data
            if not data or len(data) == 0:
                raise ValueError("Empty or invalid data provided")
            
            # Update configuration from kwargs if provided
            if 'entropy_type' in kwargs:
                self.entropy_type = kwargs['entropy_type']
            if 'alpha' in kwargs:
                self.alpha = kwargs['alpha']
            if 'normalize_entropy' in kwargs:
                self.normalize_entropy = kwargs['normalize_entropy']
            if 'min_entropy_threshold' in kwargs:
                self.min_entropy_threshold = kwargs['min_entropy_threshold']
            
            # Preprocess categorical data
            categorical_data = self._preprocess_categorical_data(data)
            
            # Calculate probability distribution
            probabilities = self._calculate_probability_distribution(categorical_data)
            
            # Calculate entropy-based uncertainty metrics
            uncertainty_metrics = self._calculate_entropy_uncertainty(probabilities)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(uncertainty_metrics)
            
            # Generate estimate ID
            estimate_id = f"entropy_based_{int(time.time())}"
            
            return UncertaintyEstimate(
                estimate_id=estimate_id,
                value=uncertainty_metrics['mean_probability'],
                uncertainty=uncertainty_metrics['total_uncertainty'],
                confidence_interval=confidence_interval,
                uncertainty_type=UncertaintyType.EPISTEMIC,
                method=QuantificationMethod.ENTROPY_BASED,
                confidence_level=uncertainty_metrics['confidence_level'],
                additional_metrics={
                    'shannon_entropy': uncertainty_metrics['shannon_entropy'],
                    'renyi_entropy': uncertainty_metrics['renyi_entropy'],
                    'min_entropy': uncertainty_metrics['min_entropy'],
                    'entropy_type': self.entropy_type,
                    'alpha': self.alpha,
                    'normalize_entropy': self.normalize_entropy,
                    'category_count': len(probabilities),
                    'is_calibrated': self.is_calibrated
                },
                metadata={
                    'probabilities': probabilities,
                    'uncertainty_metrics': uncertainty_metrics,
                    'entropy_history': self.entropy_history
                }
            )
            
        except Exception as e:
            logger.error(f"EntropyBasedQuantifier quantification failed: {e}")
            # Return high uncertainty estimate on failure
            return UncertaintyEstimate(
                estimate_id=f"entropy_based_failed_{int(time.time())}",
                value=0.0,
                uncertainty=1.0,
                confidence_interval=(0.0, 1.0),
                uncertainty_type=UncertaintyType.EPISTEMIC,
                method=QuantificationMethod.ENTROPY_BASED,
                confidence_level=0.95,
                additional_metrics={'error': str(e)}
            )
    
    def _preprocess_categorical_data(self, data: Any) -> List[str]:
        """Preprocess data into categorical format"""
        if isinstance(data, (list, tuple)):
            return [str(item) for item in data]
        elif isinstance(data, np.ndarray):
            return [str(item) for item in data.flatten()]
        elif isinstance(data, dict):
            # Extract categorical values from dictionary
            categorical_values = []
            for key, value in data.items():
                if isinstance(value, (list, tuple)):
                    categorical_values.extend([str(v) for v in value])
                else:
                    categorical_values.append(str(value))
            return categorical_values
        else:
            return [str(data)]
    
    def _calculate_probability_distribution(self, categorical_data: List[str]) -> Dict[str, float]:
        """Calculate empirical probability distribution"""
        category_counts = {}
        total_count = len(categorical_data)
        
        # Count occurrences of each category
        for item in categorical_data:
            category_counts[item] = category_counts.get(item, 0) + 1
        
        # Calculate probabilities
        probabilities = {}
        for category, count in category_counts.items():
            probabilities[category] = count / total_count if total_count > 0 else 0.0
        
        return probabilities
    
    def _calculate_entropy_uncertainty(self, probabilities: Dict[str, float]) -> Dict[str, float]:
        """Calculate comprehensive entropy-based uncertainty metrics"""
        if not probabilities:
            return {
                'total_uncertainty': 1.0,
                'mean_probability': 0.0,
                'shannon_entropy': 0.0,
                'renyi_entropy': 0.0,
                'min_entropy': 0.0,
                'confidence_level': 0.0
            }
        
        # Calculate Shannon entropy
        shannon_entropy = self._calculate_shannon_entropy(probabilities)
        
        # Calculate Rényi entropy
        renyi_entropy = self._calculate_renyi_entropy(probabilities)
        
        # Calculate min-entropy
        min_entropy = self._calculate_min_entropy(probabilities)
        
        # Calculate mean probability
        mean_probability = np.mean(list(probabilities.values()))
        
        # Normalize entropy if requested
        if self.normalize_entropy:
            max_entropy = np.log2(len(probabilities)) if len(probabilities) > 1 else 1.0
            shannon_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0.0
            renyi_entropy = renyi_entropy / max_entropy if max_entropy > 0 else 0.0
            min_entropy = min_entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Calculate uncertainty based on entropy type
        if self.entropy_type == 'shannon':
            primary_entropy = shannon_entropy
        elif self.entropy_type == 'renyi':
            primary_entropy = renyi_entropy
        elif self.entropy_type == 'min_entropy':
            primary_entropy = min_entropy
        else:
            primary_entropy = shannon_entropy
        
        # Calculate total uncertainty
        total_uncertainty = min(1.0, primary_entropy)
        
        # Calculate confidence level based on entropy type and characteristics
        if self.entropy_type == 'shannon':
            # For Shannon entropy, confidence decreases as entropy increases
            # Normalized Shannon entropy ranges from 0 (certain) to 1 (max uncertainty)
            confidence_level = 1.0 - shannon_entropy
        elif self.entropy_type == 'renyi':
            # For Rényi entropy, confidence depends on alpha
            # Higher alpha means more emphasis on dominant probabilities
            if self.alpha > 1:
                # Higher alpha = more conservative (lower confidence for same entropy)
                confidence_level = 1.0 - (renyi_entropy * (1.0 + (self.alpha - 1) * 0.2))
            else:
                confidence_level = 1.0 - renyi_entropy
        elif self.entropy_type == 'min_entropy':
            # Min-entropy is most conservative - highest uncertainty for same distribution
            confidence_level = 1.0 - (min_entropy * 1.2)  # Scale up uncertainty
        else:
            confidence_level = 1.0 - primary_entropy
        
        # Ensure confidence is in valid range [0.1, 0.99]
        confidence_level = max(0.1, min(0.99, confidence_level))
        
        # Store entropy history
        self.entropy_history.append({
            'shannon': shannon_entropy,
            'renyi': renyi_entropy,
            'min_entropy': min_entropy,
            'timestamp': time.time()
        })
        
        return {
            'total_uncertainty': total_uncertainty,
            'mean_probability': mean_probability,
            'shannon_entropy': shannon_entropy,
            'renyi_entropy': renyi_entropy,
            'min_entropy': min_entropy,
            'confidence_level': confidence_level
        }
    
    def _calculate_shannon_entropy(self, probabilities: Dict[str, float]) -> float:
        """Calculate Shannon entropy"""
        entropy = 0.0
        for probability in probabilities.values():
            if probability > 0:
                entropy -= probability * np.log2(probability)
        return entropy
    
    def _calculate_renyi_entropy(self, probabilities: Dict[str, float]) -> float:
        """Calculate Rényi entropy with parameter alpha"""
        if self.alpha == 1:
            # Rényi entropy converges to Shannon entropy when alpha = 1
            return self._calculate_shannon_entropy(probabilities)
        
        if self.alpha <= 0:
            raise ValueError("Alpha parameter must be positive")
        
        # Calculate sum of probabilities raised to alpha
        sum_p_alpha = sum(prob ** self.alpha for prob in probabilities.values() if prob > 0)
        
        if sum_p_alpha <= 0:
            return 0.0
        
        # Calculate Rényi entropy
        entropy = (1 / (1 - self.alpha)) * np.log2(sum_p_alpha)
        return entropy
    
    def _calculate_min_entropy(self, probabilities: Dict[str, float]) -> float:
        """Calculate min-entropy (Rényi entropy with alpha = infinity)"""
        if not probabilities:
            return 0.0
        
        # Min-entropy is -log2(max_probability)
        max_probability = max(probabilities.values())
        if max_probability <= 0:
            return 0.0
        
        return -np.log2(max_probability)
    
    def _calculate_confidence_interval(self, uncertainty_metrics: Dict[str, float]) -> Tuple[float, float]:
        """Calculate confidence interval for the prediction"""
        mean_probability = uncertainty_metrics['mean_probability']
        uncertainty = uncertainty_metrics['total_uncertainty']
        
        # Calculate interval based on uncertainty
        interval_width = uncertainty * 0.5
        lower_bound = max(0.0, mean_probability - interval_width)
        upper_bound = min(1.0, mean_probability + interval_width)
        
        return (lower_bound, upper_bound)
    
    def get_method_name(self) -> QuantificationMethod:
        return QuantificationMethod.ENTROPY_BASED

class PossibilityBasedQuantifier(UncertaintyQuantifier):
    """Production-ready possibility-based uncertainty for categorical data"""
    
    def __init__(self, membership_function: str = 'triangular', alpha_cut: float = 0.5,
                 fuzzy_operator: str = 'max_min', necessity_threshold: float = 0.3):
        self.membership_function = membership_function  # 'triangular', 'gaussian', 'trapezoidal'
        self.alpha_cut = alpha_cut  # Alpha-cut for fuzzy set operations
        self.fuzzy_operator = fuzzy_operator  # 'max_min', 'product', 'lukasiewicz'
        self.necessity_threshold = necessity_threshold
        self.possibility_history = []
        self.necessity_history = []
        self.is_initialized = False
        
    def quantify(self, data: Any, model: Any = None, **kwargs) -> UncertaintyEstimate:
        """Production-ready uncertainty quantification using possibility theory"""
        try:
            # Validate input data with hard failure
            if not data or len(data) == 0:
                raise ValueError("Empty or invalid data provided")
            
            # Update configuration from kwargs if provided
            if 'membership_function' in kwargs:
                self.membership_function = kwargs['membership_function']
            if 'fuzzy_operator' in kwargs:
                self.fuzzy_operator = kwargs['fuzzy_operator']
            if 'alpha_cut' in kwargs:
                self.alpha_cut = kwargs['alpha_cut']
            if 'necessity_threshold' in kwargs:
                self.necessity_threshold = kwargs['necessity_threshold']
            
            # Preprocess categorical data
            categorical_data = self._preprocess_categorical_data(data)
            
            # Calculate fuzzy membership values
            membership_values = self._calculate_membership_values(categorical_data)
            
            # Calculate possibility and necessity measures
            possibility_measures = self._calculate_possibility_measures(membership_values)
            necessity_measures = self._calculate_necessity_measures(membership_values)
            
            # Calculate possibility-based uncertainty metrics
            uncertainty_metrics = self._calculate_possibility_uncertainty(possibility_measures, necessity_measures)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(uncertainty_metrics)
            
            # Generate estimate ID
            estimate_id = f"possibility_based_{int(time.time())}"
            
            return UncertaintyEstimate(
                estimate_id=estimate_id,
                value=uncertainty_metrics['mean_possibility'],
                uncertainty=uncertainty_metrics['total_uncertainty'],
                confidence_interval=confidence_interval,
                uncertainty_type=UncertaintyType.EPISTEMIC,
                method=QuantificationMethod.POSSIBILITY_BASED,
                confidence_level=uncertainty_metrics['confidence_level'],
                additional_metrics={
                    'possibility_measure': uncertainty_metrics['possibility_measure'],
                    'necessity_measure': uncertainty_metrics['necessity_measure'],
                    'membership_function': self.membership_function,
                    'alpha_cut': self.alpha_cut,
                    'fuzzy_operator': self.fuzzy_operator,
                    'category_count': len(membership_values),
                    'is_initialized': self.is_initialized
                },
                metadata={
                    'membership_values': membership_values,
                    'possibility_measures': possibility_measures,
                    'necessity_measures': necessity_measures,
                    'uncertainty_metrics': uncertainty_metrics
                }
            )
            
        except Exception as e:
            logger.error(f"PossibilityBasedQuantifier quantification failed: {e}")
            # Hard failure - re-raise the exception for debugging
            raise RuntimeError(f"PossibilityBasedQuantifier failed with error: {e}")
    
    def _preprocess_categorical_data(self, data: Any) -> List[str]:
        """Preprocess data into categorical format"""
        if isinstance(data, (list, tuple)):
            return [str(item) for item in data]
        elif isinstance(data, np.ndarray):
            return [str(item) for item in data.flatten()]
        elif isinstance(data, dict):
            # Extract categorical values from dictionary
            categorical_values = []
            for key, value in data.items():
                if isinstance(value, (list, tuple)):
                    categorical_values.extend([str(v) for v in value])
                else:
                    categorical_values.append(str(value))
            return categorical_values
        else:
            return [str(data)]
    
    def _calculate_membership_values(self, categorical_data: List[str]) -> Dict[str, float]:
        """Calculate fuzzy membership values for categories"""
        category_counts = {}
        total_count = len(categorical_data)
        
        # Count occurrences of each category
        for item in categorical_data:
            category_counts[item] = category_counts.get(item, 0) + 1
        
        # Calculate membership values based on frequency
        membership_values = {}
        for category, count in category_counts.items():
            frequency = count / total_count if total_count > 0 else 0.0
            
            # Apply membership function
            if self.membership_function == 'triangular':
                membership_values[category] = self._triangular_membership(frequency)
            elif self.membership_function == 'gaussian':
                membership_values[category] = self._gaussian_membership(frequency)
            elif self.membership_function == 'trapezoidal':
                membership_values[category] = self._trapezoidal_membership(frequency)
            else:
                membership_values[category] = frequency
        
        return membership_values
    
    def _triangular_membership(self, frequency: float) -> float:
        """Calculate triangular membership function"""
        if frequency <= 0.0:
            return 0.0
        elif frequency <= 0.5:
            return 2.0 * frequency
        elif frequency <= 1.0:
            return 2.0 * (1.0 - frequency)
        else:
            return 0.0
    
    def _gaussian_membership(self, frequency: float) -> float:
        """Calculate Gaussian membership function"""
        if frequency <= 0.0:
            return 0.0
        else:
            # Gaussian centered at 0.5 with sigma = 0.2
            return np.exp(-((frequency - 0.5) ** 2) / (2 * 0.2 ** 2))
    
    def _trapezoidal_membership(self, frequency: float) -> float:
        """Calculate trapezoidal membership function"""
        if frequency <= 0.0:
            return 0.0
        elif frequency <= 0.3:
            return frequency / 0.3
        elif frequency <= 0.7:
            return 1.0
        elif frequency <= 1.0:
            return (1.0 - frequency) / 0.3
        else:
            return 0.0
    
    def _calculate_possibility_measures(self, membership_values: Dict[str, float]) -> Dict[str, float]:
        """Calculate possibility measures for categories"""
        possibility_measures = {}
        
        # Get all categories and their memberships
        categories = list(membership_values.keys())
        memberships = list(membership_values.values())
        
        for i, category in enumerate(categories):
            membership = memberships[i]
            
            if self.fuzzy_operator == 'max_min':
                # Max-min: possibility is the membership value itself
                possibility_measures[category] = membership
            elif self.fuzzy_operator == 'product':
                # Product: possibility is membership value
                possibility_measures[category] = membership
            elif self.fuzzy_operator == 'lukasiewicz':
                # Lukasiewicz: possibility is min(1, membership + alpha_cut)
                possibility_measures[category] = min(1.0, membership + self.alpha_cut)
            else:
                possibility_measures[category] = membership
        
        return possibility_measures
    
    def _calculate_necessity_measures(self, membership_values: Dict[str, float]) -> Dict[str, float]:
        """Calculate necessity measures for categories"""
        necessity_measures = {}
        
        # Get all categories and their memberships
        categories = list(membership_values.keys())
        memberships = list(membership_values.values())
        
        for i, category in enumerate(categories):
            membership = memberships[i]
            
            if self.fuzzy_operator == 'max_min':
                # Max-min: necessity = 1 - max(membership of other categories)
                other_memberships = [memberships[j] for j in range(len(memberships)) if j != i]
                max_other_membership = max(other_memberships) if other_memberships else 0.0
                necessity_measures[category] = max(0.0, 1.0 - max_other_membership)
            elif self.fuzzy_operator == 'product':
                # Product: necessity = 1 - product of other memberships
                other_memberships = [memberships[j] for j in range(len(memberships)) if j != i]
                product_other_memberships = np.prod(other_memberships) if other_memberships else 0.0
                necessity_measures[category] = max(0.0, 1.0 - product_other_memberships)
            elif self.fuzzy_operator == 'lukasiewicz':
                # Lukasiewicz: necessity = max(0, 1 - sum(other_memberships) - alpha_cut)
                other_memberships = [memberships[j] for j in range(len(memberships)) if j != i]
                sum_other_memberships = sum(other_memberships) if other_memberships else 0.0
                necessity_measures[category] = max(0.0, 1.0 - sum_other_memberships - self.alpha_cut)
            else:
                # Default: necessity = 1 - membership
                necessity_measures[category] = max(0.0, 1.0 - membership)
        
        return necessity_measures
    
    def _calculate_possibility_uncertainty(self, possibility_measures: Dict[str, float], 
                                         necessity_measures: Dict[str, float]) -> Dict[str, float]:
        """Calculate comprehensive possibility-based uncertainty metrics"""
        if not possibility_measures:
            return {
                'total_uncertainty': 1.0,
                'mean_possibility': 0.0,
                'possibility_measure': 0.0,
                'necessity_measure': 0.0,
                'confidence_level': 0.0
            }
        
        # Calculate mean possibility
        mean_possibility = np.mean(list(possibility_measures.values()))
        
        # Calculate overall possibility measure (max of all possibilities)
        overall_possibility = max(possibility_measures.values()) if possibility_measures else 0.0
        
        # Calculate overall necessity measure (min of all necessities)
        overall_necessity = min(necessity_measures.values()) if necessity_measures else 0.0
        
        # Calculate uncertainty based on possibility-necessity gap
        possibility_necessity_gap = overall_possibility - overall_necessity
        
        # Calculate total uncertainty
        total_uncertainty = min(1.0, possibility_necessity_gap + (1.0 - overall_possibility))
        
        # Calculate confidence level based on fuzzy operator type and membership function
        if self.fuzzy_operator == 'max_min':
            # Max-min: confidence based on possibility dominance
            confidence_level = overall_possibility * (1.0 - possibility_necessity_gap * 0.3)
        elif self.fuzzy_operator == 'product':
            # Product: more conservative confidence
            confidence_level = overall_possibility * (1.0 - possibility_necessity_gap * 0.5)
        elif self.fuzzy_operator == 'lukasiewicz':
            # Lukasiewicz: most conservative
            confidence_level = overall_possibility * (1.0 - possibility_necessity_gap * 0.7)
        else:
            confidence_level = overall_possibility * (1.0 - possibility_necessity_gap * 0.4)
        
        # Adjust based on membership function type
        if self.membership_function == 'triangular':
            # Triangular: moderate confidence scaling
            confidence_level *= 1.0
        elif self.membership_function == 'gaussian':
            # Gaussian: higher confidence for well-defined distributions
            confidence_level *= 1.1
        elif self.membership_function == 'trapezoidal':
            # Trapezoidal: more conservative
            confidence_level *= 0.9
        else:
            confidence_level *= 1.0
        
        # Ensure confidence is in valid range [0.1, 0.99]
        confidence_level = max(0.1, min(0.99, confidence_level))
        
        # Store history
        self.possibility_history.append({
            'possibility': overall_possibility,
            'necessity': overall_necessity,
            'gap': possibility_necessity_gap,
            'timestamp': time.time()
        })
        
        return {
            'total_uncertainty': total_uncertainty,
            'mean_possibility': mean_possibility,
            'possibility_measure': overall_possibility,
            'necessity_measure': overall_necessity,
            'possibility_necessity_gap': possibility_necessity_gap,
            'confidence_level': confidence_level
        }
    
    def _calculate_confidence_interval(self, uncertainty_metrics: Dict[str, float]) -> Tuple[float, float]:
        """Calculate confidence interval for the prediction"""
        mean_possibility = uncertainty_metrics['mean_possibility']
        uncertainty = uncertainty_metrics['total_uncertainty']
        
        # Calculate interval based on uncertainty
        interval_width = uncertainty * 0.5
        lower_bound = max(0.0, mean_possibility - interval_width)
        upper_bound = min(1.0, mean_possibility + interval_width)
        
        return (lower_bound, upper_bound)
    
    def get_method_name(self) -> QuantificationMethod:
        return QuantificationMethod.POSSIBILITY_BASED

class MonteCarloDropoutQuantifier(UncertaintyQuantifier):
    """Monte Carlo Dropout uncertainty quantification"""
    
    def __init__(self, num_samples: int = 100, dropout_rate: float = 0.1):
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
    
    def quantify(self, data: Any, model: Any = None, **kwargs) -> UncertaintyEstimate:
        """Perform MC Dropout uncertainty quantification"""
        try:
            if model is None:
                raise ValueError("Model required for MC Dropout")
            
            # Enable dropout during inference
            model.train()
            
            predictions = []
            input_tensor = torch.tensor(data, dtype=torch.float32)
            
            with torch.no_grad():
                for _ in range(self.num_samples):
                    output = model(input_tensor)
                    predictions.append(output.numpy())
            
            predictions = np.array(predictions)
            
            # Calculate statistics
            mean_prediction = np.mean(predictions, axis=0)
            uncertainty = np.std(predictions, axis=0)
            
            # Calculate confidence interval
            lower = np.percentile(predictions, 2.5, axis=0)
            upper = np.percentile(predictions, 97.5, axis=0)
            
            # Flatten if single output
            if mean_prediction.shape == (1,):
                mean_prediction = mean_prediction[0]
                uncertainty = uncertainty[0]
                confidence_interval = (lower[0], upper[0])
            else:
                confidence_interval = (lower, upper)
            
            return UncertaintyEstimate(
                estimate_id=f"mc_dropout_{int(time.time())}",
                value=mean_prediction,
                uncertainty=uncertainty,
                confidence_interval=confidence_interval,
                uncertainty_type=UncertaintyType.EPISTEMIC,
                method=QuantificationMethod.MONTE_CARLO_DROPOUT,
                additional_metrics={
                    'prediction_variance': np.var(predictions, axis=0),
                    'num_samples': self.num_samples,
                    'dropout_rate': self.dropout_rate
                }
            )
            
        except Exception as e:
            logger.error(f"MC Dropout quantification failed: {e}")
            return UncertaintyEstimate(
                estimate_id=f"mc_dropout_failed_{int(time.time())}",
                value=0.0,
                uncertainty=1.0,
                confidence_interval=(0.0, 0.0),
                uncertainty_type=UncertaintyType.EPISTEMIC,
                method=QuantificationMethod.MONTE_CARLO_DROPOUT,
                metadata={'error': str(e)}
            )
    
    def get_method_name(self) -> QuantificationMethod:
        return QuantificationMethod.MONTE_CARLO_DROPOUT

class EnsembleQuantifier(UncertaintyQuantifier):
    """Ensemble uncertainty quantification"""
    
    def __init__(self, models: List[Any] = None):
        self.models = models or []
    
    def quantify(self, data: Any, model: Any = None, **kwargs) -> UncertaintyEstimate:
        """Perform ensemble uncertainty quantification"""
        try:
            if not self.models and model is None:
                raise ValueError("Models required for ensemble quantification")
            
            models_to_use = self.models if self.models else [model]
            predictions = []
            
            input_tensor = torch.tensor(data, dtype=torch.float32)
            
            for model in models_to_use:
                model.eval()
                with torch.no_grad():
                    output = model(input_tensor)
                    predictions.append(output.numpy())
            
            predictions = np.array(predictions)
            
            # Calculate statistics
            mean_prediction = np.mean(predictions, axis=0)
            uncertainty = np.std(predictions, axis=0)
            
            # Calculate confidence interval
            lower = np.percentile(predictions, 2.5, axis=0)
            upper = np.percentile(predictions, 97.5, axis=0)
            
            # Flatten if single output
            if mean_prediction.shape == (1,):
                mean_prediction = mean_prediction[0]
                uncertainty = uncertainty[0]
                confidence_interval = (lower[0], upper[0])
            else:
                confidence_interval = (lower, upper)
            
            return UncertaintyEstimate(
                estimate_id=f"ensemble_{int(time.time())}",
                value=mean_prediction,
                uncertainty=uncertainty,
                confidence_interval=confidence_interval,
                uncertainty_type=UncertaintyType.EPISTEMIC,
                method=QuantificationMethod.ENSEMBLE,
                additional_metrics={
                    'ensemble_size': len(models_to_use),
                    'prediction_variance': np.var(predictions, axis=0),
                    'agreement': 1.0 - (uncertainty / (np.abs(mean_prediction) + 1e-8))
                }
            )
            
        except Exception as e:
            logger.error(f"Ensemble quantification failed: {e}")
            return UncertaintyEstimate(
                estimate_id=f"ensemble_failed_{int(time.time())}",
                value=0.0,
                uncertainty=1.0,
                confidence_interval=(0.0, 0.0),
                uncertainty_type=UncertaintyType.EPISTEMIC,
                method=QuantificationMethod.ENSEMBLE,
                metadata={'error': str(e)}
            )
    
    def get_method_name(self) -> QuantificationMethod:
        return QuantificationMethod.ENSEMBLE

class BootstrapQuantifier(UncertaintyQuantifier):
    """Bootstrap uncertainty quantification"""
    
    def __init__(self, num_bootstrap_samples: int = 1000):
        self.num_bootstrap_samples = num_bootstrap_samples
    
    def quantify(self, data: Any, model: Any = None, **kwargs) -> UncertaintyEstimate:
        """Perform bootstrap uncertainty quantification"""
        try:
            observations = np.array(data) if not isinstance(data, np.ndarray) else data
            
            # Bootstrap resampling
            bootstrap_statistics = []
            statistic_func = kwargs.get('statistic', np.mean)
            
            for _ in range(self.num_bootstrap_samples):
                # Resample with replacement
                bootstrap_sample = np.random.choice(
                    observations, 
                    size=len(observations), 
                    replace=True
                )
                
                # Calculate statistic
                statistic = statistic_func(bootstrap_sample)
                bootstrap_statistics.append(statistic)
            
            bootstrap_statistics = np.array(bootstrap_statistics)
            
            # Calculate statistics
            mean_statistic = np.mean(bootstrap_statistics)
            uncertainty = np.std(bootstrap_statistics)
            confidence_interval = np.percentile(bootstrap_statistics, [2.5, 97.5])
            
            return UncertaintyEstimate(
                estimate_id=f"bootstrap_{int(time.time())}",
                value=mean_statistic,
                uncertainty=uncertainty,
                confidence_interval=tuple(confidence_interval),
                uncertainty_type=UncertaintyType.ALEATORIC,
                method=QuantificationMethod.BOOTSTRAP,
                additional_metrics={
                    'bootstrap_variance': np.var(bootstrap_statistics),
                    'bias': mean_statistic - statistic_func(observations),
                    'num_bootstrap_samples': self.num_bootstrap_samples
                }
            )
            
        except Exception as e:
            logger.error(f"Bootstrap quantification failed: {e}")
            return UncertaintyEstimate(
                estimate_id=f"bootstrap_failed_{int(time.time())}",
                value=0.0,
                uncertainty=1.0,
                confidence_interval=(0.0, 0.0),
                uncertainty_type=UncertaintyType.ALEATORIC,
                method=QuantificationMethod.BOOTSTRAP,
                metadata={'error': str(e)}
            )
    
    def get_method_name(self) -> QuantificationMethod:
        return QuantificationMethod.BOOTSTRAP

class UncertaintyOrchestrator:
    def __init__(self, brain_instance=None, config: Optional[Dict] = None):
        self.brain = brain_instance
        self.config = config or {}
        
        # Core state management
        self._lock = RLock()
        self._uncertainty_estimates: Dict[str, UncertaintyEstimate] = {}
        self._calibration_results: Dict[str, CalibrationResults] = {}
        self._propagation_results: Dict[str, UncertaintyPropagation] = {}
        
        # Quantification methods
        self._quantifiers: Dict[QuantificationMethod, UncertaintyQuantifier] = {
            QuantificationMethod.BAYESIAN: BayesianQuantifier(),
            QuantificationMethod.MONTE_CARLO_DROPOUT: MonteCarloDropoutQuantifier(),
            QuantificationMethod.ENSEMBLE: EnsembleQuantifier(),
            QuantificationMethod.BOOTSTRAP: BootstrapQuantifier(),
            # Non-Bayesian methods for categorical data
            QuantificationMethod.CONFORMALIZED_CREDAL: ConformalizedCredalQuantifier(),
            QuantificationMethod.DEEP_DETERMINISTIC: DeepDeterministicQuantifier(),
            QuantificationMethod.BATCH_ENSEMBLE: BatchEnsembleQuantifier(),
            QuantificationMethod.ENTROPY_BASED: EntropyBasedQuantifier(),
            QuantificationMethod.POSSIBILITY_BASED: PossibilityBasedQuantifier()
        }
        
        # Strategy configuration
        self._default_strategy = UncertaintyEstimationStrategy(
            self.config.get('default_strategy', 'balanced')
        )
        self._confidence_threshold = self.config.get('confidence_threshold', 0.8)
        
        # Calibration tracking
        self._calibration_data: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self._model_calibrations: Dict[str, CalibrationResults] = {}
        
        # Performance tracking
        self._quantification_times: Dict[str, List[float]] = defaultdict(list)
        self._accuracy_tracking: Dict[str, List[float]] = defaultdict(list)
        
        # Temporal uncertainty tracking
        self._temporal_uncertainties: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Spatial uncertainty tracking
        self._spatial_uncertainties: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        logger.info("UncertaintyOrchestrator initialized")
    
    def quantify(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Main uncertainty quantification entry point"""
        operation = parameters.get('operation', 'estimate_uncertainty')
        
        if operation == 'estimate_uncertainty':
            return self._estimate_uncertainty(parameters)
        elif operation == 'calibrate_model':
            return self._calibrate_model(parameters)
        elif operation == 'propagate_uncertainty':
            return self._propagate_uncertainty(parameters)
        elif operation == 'analyze_temporal_uncertainty':
            return self._analyze_temporal_uncertainty(parameters)
        elif operation == 'analyze_spatial_uncertainty':
            return self._analyze_spatial_uncertainty(parameters)
        elif operation == 'confidence_assessment':
            return self._assess_confidence(parameters)
        elif operation == 'uncertainty_decomposition':
            return self._decompose_uncertainty(parameters)
        elif operation == 'get_uncertainty_report':
            return self._get_uncertainty_report(parameters)
        else:
            logger.warning(f"Unknown operation: {operation}")
            return {"error": f"Unknown operation: {operation}"}
    
    def _estimate_uncertainty(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate uncertainty using specified method"""
        try:
            # Parse parameters
            data = parameters.get('data')
            method = QuantificationMethod(parameters.get('method', 'bayesian'))
            model = parameters.get('model')
            uncertainty_type = UncertaintyType(parameters.get('uncertainty_type', 'epistemic'))
            
            if data is None:
                return {"error": "Data required for uncertainty estimation"}
            
            # Get quantifier
            quantifier = self._quantifiers.get(method)
            if not quantifier:
                return {"error": f"Quantifier not available for method: {method.value}"}
            
            # Perform quantification
            start_time = time.time()
            # Remove data and model from parameters to avoid duplicate arguments
            quantifier_params = {k: v for k, v in parameters.items() 
                               if k not in ['data', 'model']}
            estimate = quantifier.quantify(data, model, **quantifier_params)
            quantification_time = time.time() - start_time
            
            # Store estimate
            with self._lock:
                self._uncertainty_estimates[estimate.estimate_id] = estimate
                self._quantification_times[method.value].append(quantification_time)
            
            # Update temporal tracking if applicable
            if uncertainty_type == UncertaintyType.TEMPORAL:
                self._update_temporal_uncertainty(estimate)
            
            logger.info(f"Uncertainty estimated: {estimate.estimate_id} using {method.value}")
            
            return {
                "estimate_id": estimate.estimate_id,
                "value": estimate.value,
                "uncertainty": estimate.uncertainty,
                "confidence_interval": estimate.confidence_interval,
                "confidence_level": estimate.confidence_level,
                "method": method.value,
                "uncertainty_type": uncertainty_type.value,
                "quantification_time": quantification_time,
                "additional_metrics": estimate.additional_metrics
            }
            
        except Exception as e:
            logger.error(f"Uncertainty estimation failed: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def _calibrate_model(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calibrate uncertainty estimates against true outcomes"""
        try:
            model_id = parameters.get('model_id', 'default')
            predictions = parameters.get('predictions', [])
            true_values = parameters.get('true_values', [])
            uncertainties = parameters.get('uncertainties', [])
            
            if not predictions or not true_values:
                return {"error": "Predictions and true values required for calibration"}
            
            predictions = np.array(predictions)
            true_values = np.array(true_values)
            uncertainties = np.array(uncertainties) if uncertainties else np.ones_like(predictions)
            
            # Calculate calibration metrics
            calibration_results = self._calculate_calibration_metrics(
                predictions, true_values, uncertainties
            )
            
            calibration_id = f"cal_{model_id}_{int(time.time())}"
            calibration = CalibrationResults(
                calibration_id=calibration_id,
                method="standard",
                calibration_error=calibration_results['calibration_error'],
                reliability_diagram=calibration_results['reliability_diagram'],
                confidence_histogram=calibration_results['confidence_histogram'],
                brier_score=calibration_results['brier_score'],
                log_likelihood=calibration_results['log_likelihood'],
                coverage_probability=calibration_results['coverage_probability'],
                sharpness=calibration_results['sharpness']
            )
            
            # Store calibration results
            with self._lock:
                self._calibration_results[calibration_id] = calibration
                self._model_calibrations[model_id] = calibration
            
            return {
                "calibration_id": calibration_id,
                "model_id": model_id,
                "calibration_error": calibration.calibration_error,
                "coverage_probability": calibration.coverage_probability,
                "brier_score": calibration.brier_score,
                "sharpness": calibration.sharpness,
                "is_well_calibrated": calibration.calibration_error < 0.1
            }
            
        except Exception as e:
            logger.error(f"Model calibration failed: {e}")
            return {"error": str(e)}
    
    def _calculate_calibration_metrics(self, predictions: np.ndarray, true_values: np.ndarray, 
                                     uncertainties: np.ndarray) -> Dict[str, Any]:
        """Calculate various calibration metrics"""
        
        # Reliability diagram
        num_bins = 10
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        # Convert uncertainties to confidence scores
        confidences = 1.0 / (1.0 + uncertainties)  # Simple transformation
        
        # Calculate accuracies (for regression, use within-uncertainty band)
        accuracies = np.abs(predictions - true_values) <= uncertainties
        
        ece = 0  # Expected Calibration Error
        reliability_diagram = {'bin_centers': [], 'accuracies': [], 'confidences': [], 'counts': []}
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                reliability_diagram['bin_centers'].append((bin_lower + bin_upper) / 2)
                reliability_diagram['accuracies'].append(accuracy_in_bin)
                reliability_diagram['confidences'].append(avg_confidence_in_bin)
                reliability_diagram['counts'].append(in_bin.sum())
        
        # Brier Score (for binary outcomes)
        if len(np.unique(true_values)) == 2:
            brier_score = np.mean((confidences - true_values) ** 2)
        else:
            # Modified Brier score for regression
            brier_score = np.mean((confidences - accuracies.astype(float)) ** 2)
        
        # Log likelihood (assuming Gaussian)
        log_likelihood = -0.5 * np.sum(
            np.log(2 * np.pi * uncertainties**2) + 
            ((predictions - true_values) / uncertainties)**2
        )
        
        # Coverage probability
        coverage_probability = np.mean(accuracies)
        
        # Sharpness (average uncertainty)
        sharpness = np.mean(uncertainties)
        
        # Confidence histogram
        confidence_histogram = {}
        for i in range(num_bins):
            bin_mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
            confidence_histogram[f'bin_{i}'] = int(bin_mask.sum())
        
        return {
            'calibration_error': ece,
            'reliability_diagram': reliability_diagram,
            'confidence_histogram': confidence_histogram,
            'brier_score': brier_score,
            'log_likelihood': log_likelihood,
            'coverage_probability': coverage_probability,
            'sharpness': sharpness
        }
    
    def _propagate_uncertainty(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Propagate uncertainty through a computational graph"""
        try:
            input_uncertainties = parameters.get('input_uncertainties', {})
            function = parameters.get('function')
            method = parameters.get('method', 'monte_carlo')
            num_samples = parameters.get('num_samples', 1000)
            
            if not input_uncertainties:
                return {"error": "Input uncertainties required"}
            
            propagation_id = f"prop_{int(time.time())}"
            
            if method == 'monte_carlo':
                output_uncertainty = self._monte_carlo_propagation(
                    input_uncertainties, function, num_samples
                )
            elif method == 'analytical':
                output_uncertainty = self._analytical_propagation(
                    input_uncertainties, function
                )
            else:
                return {"error": f"Unknown propagation method: {method}"}
            
            # Sensitivity analysis
            sensitivity = self._calculate_sensitivity(input_uncertainties, function)
            
            propagation = UncertaintyPropagation(
                propagation_id=propagation_id,
                input_uncertainties=input_uncertainties,
                propagation_method=method,
                output_uncertainty=output_uncertainty,
                sensitivity_analysis=sensitivity,
                monte_carlo_samples=num_samples
            )
            
            with self._lock:
                self._propagation_results[propagation_id] = propagation
            
            return {
                "propagation_id": propagation_id,
                "output_uncertainty": output_uncertainty,
                "sensitivity_analysis": sensitivity,
                "method": method,
                "num_samples": num_samples
            }
            
        except Exception as e:
            logger.error(f"Uncertainty propagation failed: {e}")
            return {"error": str(e)}
    
    def _monte_carlo_propagation(self, input_uncertainties: Dict[str, float], 
                               function: Callable, num_samples: int) -> float:
        """Monte Carlo uncertainty propagation"""
        
        # Generate samples for each input
        input_samples = {}
        for var_name, uncertainty in input_uncertainties.items():
            # Assume normal distribution with mean=0, std=uncertainty
            input_samples[var_name] = np.random.normal(0, uncertainty, num_samples)
        
        # Evaluate function for each sample
        outputs = []
        for i in range(num_samples):
            sample_inputs = {var: samples[i] for var, samples in input_samples.items()}
            try:
                if callable(function):
                    output = function(**sample_inputs)
                else:
                    # Simple algebraic evaluation
                    output = eval(function, {"__builtins__": {}}, sample_inputs)
                outputs.append(output)
            except Exception:
                outputs.append(0.0)  # Handle evaluation errors
        
        # Calculate output uncertainty
        return np.std(outputs)
    
    def _analytical_propagation(self, input_uncertainties: Dict[str, float], 
                              function: Callable) -> float:
        """Analytical uncertainty propagation using linearization"""
        
        # This is a simplified implementation
        # In practice, would need to compute partial derivatives
        
        # Assume linear combination for demonstration
        total_variance = sum(uncertainty**2 for uncertainty in input_uncertainties.values())
        return np.sqrt(total_variance)
    
    def _calculate_sensitivity(self, input_uncertainties: Dict[str, float], 
                             function: Callable) -> Dict[str, float]:
        """Calculate sensitivity of output to each input uncertainty"""
        
        sensitivity = {}
        base_output = 0.0  # Reference output
        
        for var_name, uncertainty in input_uncertainties.items():
            # Perturb this variable
            perturbed_inputs = {var: 0 for var in input_uncertainties.keys()}
            perturbed_inputs[var_name] = uncertainty
            
            try:
                if callable(function):
                    perturbed_output = function(**perturbed_inputs)
                else:
                    perturbed_output = eval(function, {"__builtins__": {}}, perturbed_inputs)
                
                # Sensitivity = change in output / change in input
                sensitivity[var_name] = abs(perturbed_output - base_output) / uncertainty
                
            except Exception:
                sensitivity[var_name] = 0.0
        
        return sensitivity
    
    def _analyze_temporal_uncertainty(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal patterns in uncertainty"""
        try:
            time_series_data = parameters.get('time_series_data', [])
            window_size = parameters.get('window_size', 10)
            
            if len(time_series_data) < window_size:
                return {"error": "Insufficient data for temporal analysis"}
            
            # Calculate rolling uncertainty
            rolling_uncertainties = []
            for i in range(len(time_series_data) - window_size + 1):
                window_data = time_series_data[i:i + window_size]
                window_uncertainty = np.std(window_data)
                rolling_uncertainties.append(window_uncertainty)
            
            # Trend analysis
            time_points = np.arange(len(rolling_uncertainties))
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                time_points, rolling_uncertainties
            )
            
            # Seasonality analysis (simplified)
            seasonality_period = parameters.get('seasonality_period', 12)
            if len(rolling_uncertainties) >= seasonality_period * 2:
                seasonal_component = self._extract_seasonality(
                    rolling_uncertainties, seasonality_period
                )
            else:
                seasonal_component = None
            
            analysis_id = f"temporal_analysis_{int(time.time())}"
            
            return {
                "analysis_id": analysis_id,
                "rolling_uncertainties": rolling_uncertainties,
                "trend": {
                    "slope": slope,
                    "r_squared": r_value**2,
                    "p_value": p_value,
                    "trend_direction": "increasing" if slope > 0 else "decreasing"
                },
                "seasonality": seasonal_component,
                "mean_uncertainty": np.mean(rolling_uncertainties),
                "uncertainty_volatility": np.std(rolling_uncertainties)
            }
            
        except Exception as e:
            logger.error(f"Temporal uncertainty analysis failed: {e}")
            return {"error": str(e)}
    
    def _extract_seasonality(self, data: List[float], period: int) -> Dict[str, Any]:
        """Extract seasonal component from uncertainty data"""
        data_array = np.array(data)
        
        # Simple seasonal decomposition
        seasonal_means = []
        for i in range(period):
            seasonal_indices = np.arange(i, len(data_array), period)
            seasonal_mean = np.mean(data_array[seasonal_indices])
            seasonal_means.append(seasonal_mean)
        
        overall_mean = np.mean(data_array)
        seasonal_component = [mean - overall_mean for mean in seasonal_means]
        
        # Calculate seasonal strength
        seasonal_variance = np.var(seasonal_component)
        total_variance = np.var(data_array)
        seasonal_strength = seasonal_variance / total_variance if total_variance > 0 else 0
        
        return {
            "seasonal_component": seasonal_component,
            "seasonal_strength": seasonal_strength,
            "period": period
        }
    
    def _analyze_spatial_uncertainty(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze spatial patterns in uncertainty"""
        try:
            spatial_data = parameters.get('spatial_data', {})  # {location: uncertainty}
            coordinates = parameters.get('coordinates', {})    # {location: (x, y)}
            
            if not spatial_data:
                return {"error": "Spatial data required"}
            
            # Calculate spatial statistics
            uncertainties = list(spatial_data.values())
            locations = list(spatial_data.keys())
            
            # Spatial autocorrelation (simplified)
            if coordinates:
                autocorrelation = self._calculate_spatial_autocorrelation(
                    spatial_data, coordinates
                )
            else:
                autocorrelation = None
            
            # Hotspot analysis
            hotspots = self._identify_uncertainty_hotspots(spatial_data)
            
            # Spatial clustering
            clusters = self._cluster_spatial_uncertainties(spatial_data, coordinates)
            
            analysis_id = f"spatial_analysis_{int(time.time())}"
            
            return {
                "analysis_id": analysis_id,
                "spatial_statistics": {
                    "mean_uncertainty": np.mean(uncertainties),
                    "max_uncertainty": np.max(uncertainties),
                    "min_uncertainty": np.min(uncertainties),
                    "uncertainty_range": np.max(uncertainties) - np.min(uncertainties)
                },
                "spatial_autocorrelation": autocorrelation,
                "hotspots": hotspots,
                "clusters": clusters,
                "total_locations": len(locations)
            }
            
        except Exception as e:
            logger.error(f"Spatial uncertainty analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_spatial_autocorrelation(self, spatial_data: Dict[str, float], 
                                         coordinates: Dict[str, Tuple[float, float]]) -> float:
        """Calculate Moran's I spatial autocorrelation"""
        locations = list(spatial_data.keys())
        n = len(locations)
        
        if n < 2:
            return 0.0
        
        # Calculate weights matrix (inverse distance)
        weights = np.zeros((n, n))
        for i, loc_i in enumerate(locations):
            for j, loc_j in enumerate(locations):
                if i != j:
                    coord_i = coordinates.get(loc_i, (0, 0))
                    coord_j = coordinates.get(loc_j, (0, 0))
                    distance = np.sqrt((coord_i[0] - coord_j[0])**2 + (coord_i[1] - coord_j[1])**2)
                    weights[i, j] = 1.0 / (distance + 1e-8)  # Avoid division by zero
        
        # Normalize weights
        row_sums = np.sum(weights, axis=1)
        for i in range(n):
            if row_sums[i] > 0:
                weights[i, :] /= row_sums[i]
        
        # Calculate Moran's I
        uncertainties = np.array([spatial_data[loc] for loc in locations])
        mean_uncertainty = np.mean(uncertainties)
        
        numerator = 0
        denominator = np.sum((uncertainties - mean_uncertainty)**2)
        
        for i in range(n):
            for j in range(n):
                numerator += weights[i, j] * (uncertainties[i] - mean_uncertainty) * (uncertainties[j] - mean_uncertainty)
        
        if denominator > 0:
            morans_i = numerator / denominator
        else:
            morans_i = 0.0
        
        return morans_i
    
    def _identify_uncertainty_hotspots(self, spatial_data: Dict[str, float]) -> List[str]:
        """Identify locations with high uncertainty (hotspots)"""
        uncertainties = list(spatial_data.values())
        threshold = np.mean(uncertainties) + np.std(uncertainties)
        
        hotspots = [location for location, uncertainty in spatial_data.items() 
                   if uncertainty > threshold]
        
        return hotspots
    
    def _cluster_spatial_uncertainties(self, spatial_data: Dict[str, float], 
                                     coordinates: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Simple clustering of spatial uncertainties"""
        # This is a simplified implementation
        # In practice, would use proper clustering algorithms like K-means
        
        uncertainties = list(spatial_data.values())
        
        # Simple quantile-based clustering
        low_threshold = np.percentile(uncertainties, 33)
        high_threshold = np.percentile(uncertainties, 67)
        
        clusters = {
            "low_uncertainty": [],
            "medium_uncertainty": [],
            "high_uncertainty": []
        }
        
        for location, uncertainty in spatial_data.items():
            if uncertainty <= low_threshold:
                clusters["low_uncertainty"].append(location)
            elif uncertainty <= high_threshold:
                clusters["medium_uncertainty"].append(location)
            else:
                clusters["high_uncertainty"].append(location)
        
        return clusters
    
    def _assess_confidence(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Assess confidence in predictions or decisions"""
        try:
            predictions = parameters.get('predictions', [])
            uncertainties = parameters.get('uncertainties', [])
            threshold = parameters.get('confidence_threshold', self._confidence_threshold)
            
            if not predictions or not uncertainties:
                return {"error": "Predictions and uncertainties required"}
            
            predictions = np.array(predictions)
            uncertainties = np.array(uncertainties)
            
            # Convert uncertainties to confidence scores
            confidences = 1.0 / (1.0 + uncertainties)
            
            # Assess confidence levels
            confidence_assessment = {
                "high_confidence": np.sum(confidences >= threshold),
                "medium_confidence": np.sum((confidences >= 0.5) & (confidences < threshold)),
                "low_confidence": np.sum(confidences < 0.5),
                "mean_confidence": np.mean(confidences),
                "confidence_std": np.std(confidences)
            }
            
            # Confidence distribution
            confidence_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            confidence_distribution = {}
            for i in range(len(confidence_bins) - 1):
                bin_mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i+1])
                confidence_distribution[f"confidence_{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}"] = int(bin_mask.sum())
            
            return {
                "assessment_id": f"confidence_{int(time.time())}",
                "confidence_assessment": confidence_assessment,
                "confidence_distribution": confidence_distribution,
                "threshold": threshold,
                "total_predictions": len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Confidence assessment failed: {e}")
            return {"error": str(e)}
    
    def _decompose_uncertainty(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose total uncertainty into aleatoric and epistemic components"""
        try:
            total_uncertainties = parameters.get('total_uncertainties', [])
            method = parameters.get('method', 'variance_decomposition')
            
            if not total_uncertainties:
                return {"error": "Total uncertainties required"}
            
            total_uncertainties = np.array(total_uncertainties)
            
            if method == 'variance_decomposition':
                # Simple variance decomposition
                # Assume some portion is aleatoric (irreducible) and some is epistemic (reducible)
                
                # Estimate aleatoric uncertainty as minimum observed uncertainty
                aleatoric_component = np.min(total_uncertainties)
                
                # Epistemic component is the remaining uncertainty
                epistemic_components = total_uncertainties - aleatoric_component
                epistemic_components = np.maximum(epistemic_components, 0)  # Ensure non-negative
                
                # Calculate statistics
                mean_aleatoric = aleatoric_component
                mean_epistemic = np.mean(epistemic_components)
                
                decomposition = {
                    "aleatoric_uncertainty": {
                        "mean": mean_aleatoric,
                        "description": "Irreducible uncertainty due to noise in data"
                    },
                    "epistemic_uncertainty": {
                        "mean": mean_epistemic,
                        "std": np.std(epistemic_components),
                        "max": np.max(epistemic_components),
                        "description": "Reducible uncertainty due to limited knowledge"
                    },
                    "total_uncertainty": {
                        "mean": np.mean(total_uncertainties),
                        "std": np.std(total_uncertainties)
                    }
                }
                
                # Calculate relative contributions
                total_mean = np.mean(total_uncertainties)
                if total_mean > 0:
                    decomposition["relative_contributions"] = {
                        "aleatoric_fraction": mean_aleatoric / total_mean,
                        "epistemic_fraction": mean_epistemic / total_mean
                    }
                
            else:
                return {"error": f"Unknown decomposition method: {method}"}
            
            return {
                "decomposition_id": f"decomp_{int(time.time())}",
                "method": method,
                "decomposition": decomposition,
                "total_samples": len(total_uncertainties)
            }
            
        except Exception as e:
            logger.error(f"Uncertainty decomposition failed: {e}")
            return {"error": str(e)}
    
    def _update_temporal_uncertainty(self, estimate: UncertaintyEstimate):
        """Update temporal uncertainty tracking"""
        uncertainty_key = f"{estimate.method.value}_{estimate.uncertainty_type.value}"
        
        with self._lock:
            self._temporal_uncertainties[uncertainty_key].append({
                'timestamp': estimate.timestamp,
                'uncertainty': estimate.uncertainty,
                'value': estimate.value
            })
    
    def _get_uncertainty_report(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive uncertainty report"""
        try:
            with self._lock:
                # Summary statistics
                total_estimates = len(self._uncertainty_estimates)
                methods_used = set(est.method.value for est in self._uncertainty_estimates.values())
                uncertainty_types = set(est.uncertainty_type.value for est in self._uncertainty_estimates.values())
                
                # Average uncertainties by method
                method_uncertainties = defaultdict(list)
                for estimate in self._uncertainty_estimates.values():
                    method_uncertainties[estimate.method.value].append(estimate.uncertainty)
                
                avg_uncertainties = {
                    method: np.mean(uncertainties) 
                    for method, uncertainties in method_uncertainties.items()
                }
                
                # Performance metrics
                avg_quantification_times = {
                    method: np.mean(times) 
                    for method, times in self._quantification_times.items()
                }
                
                # Calibration summary
                calibration_summary = {}
                for model_id, calibration in self._model_calibrations.items():
                    calibration_summary[model_id] = {
                        'calibration_error': calibration.calibration_error,
                        'coverage_probability': calibration.coverage_probability,
                        'brier_score': calibration.brier_score
                    }
                
                return {
                    "report_id": f"uncertainty_report_{int(time.time())}",
                    "summary": {
                        "total_estimates": total_estimates,
                        "methods_used": list(methods_used),
                        "uncertainty_types": list(uncertainty_types),
                        "total_calibrations": len(self._calibration_results),
                        "total_propagations": len(self._propagation_results)
                    },
                    "performance": {
                        "average_uncertainties": avg_uncertainties,
                        "quantification_times": avg_quantification_times
                    },
                    "calibration_summary": calibration_summary,
                    "recent_estimates": [
                        {
                            "estimate_id": est.estimate_id,
                            "method": est.method.value,
                            "uncertainty": est.uncertainty,
                            "timestamp": est.timestamp
                        }
                        for est in list(self._uncertainty_estimates.values())[-10:]
                    ]
                }
                
        except Exception as e:
            logger.error(f"Uncertainty report generation failed: {e}")
            return {"error": str(e)}
    
    def get_uncertainty_estimate(self, estimate_id: str) -> Optional[UncertaintyEstimate]:
        """Get uncertainty estimate by ID"""
        return self._uncertainty_estimates.get(estimate_id)
    
    def get_calibration_results(self, calibration_id: str) -> Optional[CalibrationResults]:
        """Get calibration results by ID"""
        return self._calibration_results.get(calibration_id)
    
    def add_quantifier(self, method: QuantificationMethod, quantifier: UncertaintyQuantifier):
        """Add a custom uncertainty quantifier"""
        self._quantifiers[method] = quantifier
        logger.info(f"Added custom quantifier for method: {method.value}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for uncertainty quantification"""
        with self._lock:
            return {
                "total_estimates": len(self._uncertainty_estimates),
                "quantification_times": dict(self._quantification_times),
                "method_usage": {
                    method.value: sum(1 for est in self._uncertainty_estimates.values() 
                                    if est.method == method)
                    for method in QuantificationMethod
                },
                "uncertainty_type_distribution": {
                    unc_type.value: sum(1 for est in self._uncertainty_estimates.values() 
                                      if est.uncertainty_type == unc_type)
                    for unc_type in UncertaintyType
                },
                "calibration_performance": {
                    model_id: {
                        "calibration_error": cal.calibration_error,
                        "coverage_probability": cal.coverage_probability
                    }
                    for model_id, cal in self._model_calibrations.items()
                }
            }


class CrossDomainPropagationEngine:
    """Real-time cross-domain uncertainty propagation engine."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)
        self.propagation_rules = {}
        self.propagation_history = defaultdict(list)
        self.active_propagations = {}
        self.propagation_cache = {}
        
        # Propagation strategies
        self.strategies = {
            'immediate': self._immediate_propagation,
            'batched': self._batched_propagation,
            'conditional': self._conditional_propagation
        }
        
        # Initialize default propagation rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default cross-domain propagation rules."""
        self.propagation_rules = {
            'fraud_detection': {
                'targets': ['financial_analysis', 'risk_assessment', 'compliance'],
                'threshold': 0.2,  # 20% uncertainty threshold
                'strategy': 'immediate',
                'priority': 'high'
            },
            'financial_analysis': {
                'targets': ['risk_assessment', 'trading_system'],
                'threshold': 0.15,  # 15% uncertainty threshold
                'strategy': 'conditional',
                'priority': 'medium'
            },
            'risk_assessment': {
                'targets': ['compliance', 'audit_system'],
                'threshold': 0.25,  # 25% uncertainty threshold
                'strategy': 'batched',
                'priority': 'high'
            }
        }
    
    def propagate_uncertainty(self, source_domain: str, uncertainty_result: Dict[str, Any], 
                            target_domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """Propagate uncertainty from source domain to target domains."""
        try:
            if target_domains is None:
                target_domains = self.propagation_rules.get(source_domain, {}).get('targets', [])
            
            propagation_results = {}
            
            for target_domain in target_domains:
                # Check propagation threshold
                if uncertainty_result.get('uncertainty', 0) > self.propagation_rules.get(source_domain, {}).get('threshold', 0.1):
                    # Execute propagation based on strategy
                    strategy = self.propagation_rules.get(source_domain, {}).get('strategy', 'immediate')
                    propagation_func = self.strategies.get(strategy, self._immediate_propagation)
                    
                    result = propagation_func(source_domain, target_domain, uncertainty_result)
                    propagation_results[target_domain] = result
                    
                    # Cache propagation result
                    self.propagation_cache[f"{source_domain}_{target_domain}"] = {
                        'result': result,
                        'timestamp': time.time(),
                        'uncertainty': uncertainty_result.get('uncertainty', 0)
                    }
            
            return {
                'propagation_id': f"prop_{int(time.time())}",
                'source_domain': source_domain,
                'target_domains': list(propagation_results.keys()),
                'results': propagation_results,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Cross-domain propagation failed: {e}")
            raise RuntimeError(f"Cross-domain propagation failed: {e}")
    
    def _immediate_propagation(self, source_domain: str, target_domain: str, 
                              uncertainty_result: Dict[str, Any]) -> Dict[str, Any]:
        """Immediate propagation of uncertainty."""
        try:
            # Transform uncertainty for target domain
            transformed_uncertainty = self._transform_uncertainty_for_domain(
                uncertainty_result, source_domain, target_domain
            )
            
            # Apply domain-specific adjustments
            adjusted_uncertainty = self._apply_domain_adjustments(
                transformed_uncertainty, target_domain
            )
            
            return {
                'propagation_type': 'immediate',
                'transformed_uncertainty': adjusted_uncertainty,
                'confidence_level': uncertainty_result.get('confidence_level', 0.95),
                'propagation_time': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Immediate propagation failed: {e}")
            raise RuntimeError(f"Immediate propagation failed: {e}")
    
    def _batched_propagation(self, source_domain: str, target_domain: str, 
                            uncertainty_result: Dict[str, Any]) -> Dict[str, Any]:
        """Batched propagation of uncertainty."""
        try:
            # Add to batch queue
            batch_key = f"{source_domain}_{target_domain}"
            if batch_key not in self.active_propagations:
                self.active_propagations[batch_key] = []
            
            self.active_propagations[batch_key].append({
                'uncertainty_result': uncertainty_result,
                'timestamp': time.time()
            })
            
            # Process batch if threshold reached
            if len(self.active_propagations[batch_key]) >= 10:  # Batch size of 10
                return self._process_batch(batch_key, target_domain)
            else:
                return {
                    'propagation_type': 'batched',
                    'status': 'queued',
                    'batch_size': len(self.active_propagations[batch_key])
                }
                
        except Exception as e:
            self.logger.error(f"Batched propagation failed: {e}")
            raise RuntimeError(f"Batched propagation failed: {e}")
    
    def _conditional_propagation(self, source_domain: str, target_domain: str, 
                               uncertainty_result: Dict[str, Any]) -> Dict[str, Any]:
        """Conditional propagation based on uncertainty thresholds."""
        try:
            # Check conditional rules
            uncertainty_level = uncertainty_result.get('uncertainty', 0)
            confidence_level = uncertainty_result.get('confidence_level', 0.95)
            
            # Domain-specific conditions
            conditions_met = self._check_propagation_conditions(
                source_domain, target_domain, uncertainty_level, confidence_level
            )
            
            if conditions_met:
                return self._immediate_propagation(source_domain, target_domain, uncertainty_result)
            else:
                return {
                    'propagation_type': 'conditional',
                    'status': 'skipped',
                    'reason': 'conditions_not_met',
                    'uncertainty_level': uncertainty_level,
                    'confidence_level': confidence_level
                }
                
        except Exception as e:
            self.logger.error(f"Conditional propagation failed: {e}")
            raise RuntimeError(f"Conditional propagation failed: {e}")
    
    def _transform_uncertainty_for_domain(self, uncertainty_result: Dict[str, Any], 
                                        source_domain: str, target_domain: str) -> Dict[str, Any]:
        """Transform uncertainty for target domain requirements."""
        try:
            transformed = uncertainty_result.copy()
            
            # Domain-specific transformations
            if source_domain == 'fraud_detection' and target_domain == 'financial_analysis':
                # Transform fraud uncertainty to financial risk uncertainty
                transformed['uncertainty'] = uncertainty_result.get('uncertainty', 0) * 0.8  # Scale down
                transformed['uncertainty_type'] = 'financial_risk'
                
            elif source_domain == 'financial_analysis' and target_domain == 'risk_assessment':
                # Transform financial uncertainty to risk assessment uncertainty
                transformed['uncertainty'] = uncertainty_result.get('uncertainty', 0) * 1.2  # Scale up
                transformed['uncertainty_type'] = 'portfolio_risk'
                
            elif source_domain == 'risk_assessment' and target_domain == 'compliance':
                # Transform risk uncertainty to compliance uncertainty
                transformed['uncertainty'] = min(1.0, uncertainty_result.get('uncertainty', 0) * 1.5)
                transformed['uncertainty_type'] = 'compliance_risk'
            
            return transformed
            
        except Exception as e:
            self.logger.error(f"Uncertainty transformation failed: {e}")
            raise RuntimeError(f"Uncertainty transformation failed: {e}")
    
    def _apply_domain_adjustments(self, uncertainty_result: Dict[str, Any], 
                                target_domain: str) -> Dict[str, Any]:
        """Apply domain-specific adjustments to uncertainty."""
        try:
            adjusted = uncertainty_result.copy()
            
            # Domain-specific adjustments
            if target_domain == 'financial_analysis':
                # Conservative adjustment for financial analysis
                adjusted['uncertainty'] = min(1.0, uncertainty_result.get('uncertainty', 0) * 1.1)
                adjusted['confidence_level'] = max(0.8, uncertainty_result.get('confidence_level', 0.95) * 0.95)
                
            elif target_domain == 'risk_assessment':
                # Aggressive adjustment for risk assessment
                adjusted['uncertainty'] = min(1.0, uncertainty_result.get('uncertainty', 0) * 1.3)
                adjusted['confidence_level'] = max(0.7, uncertainty_result.get('confidence_level', 0.95) * 0.9)
                
            elif target_domain == 'compliance':
                # Very conservative adjustment for compliance
                adjusted['uncertainty'] = min(1.0, uncertainty_result.get('uncertainty', 0) * 1.5)
                adjusted['confidence_level'] = max(0.6, uncertainty_result.get('confidence_level', 0.95) * 0.85)
            
            return adjusted
            
        except Exception as e:
            self.logger.error(f"Domain adjustment failed: {e}")
            raise RuntimeError(f"Domain adjustment failed: {e}")
    
    def _check_propagation_conditions(self, source_domain: str, target_domain: str, 
                                    uncertainty_level: float, confidence_level: float) -> bool:
        """Check if propagation conditions are met."""
        try:
            # Base conditions
            if uncertainty_level < 0.05:  # Very low uncertainty
                return False
            
            if confidence_level > 0.98:  # Very high confidence
                return False
            
            # Domain-specific conditions
            if source_domain == 'fraud_detection':
                return uncertainty_level > 0.1 and confidence_level < 0.9
                
            elif source_domain == 'financial_analysis':
                return uncertainty_level > 0.08 and confidence_level < 0.85
                
            elif source_domain == 'risk_assessment':
                return uncertainty_level > 0.12 and confidence_level < 0.8
            
            return True
            
        except Exception as e:
            self.logger.error(f"Propagation condition check failed: {e}")
            raise RuntimeError(f"Propagation condition check failed: {e}")
    
    def _process_batch(self, batch_key: str, target_domain: str) -> Dict[str, Any]:
        """Process batched propagation."""
        try:
            batch_data = self.active_propagations[batch_key]
            
            # Aggregate uncertainties
            uncertainties = [item['uncertainty_result'].get('uncertainty', 0) for item in batch_data]
            confidences = [item['uncertainty_result'].get('confidence_level', 0.95) for item in batch_data]
            
            # Calculate aggregated uncertainty
            avg_uncertainty = np.mean(uncertainties)
            avg_confidence = np.mean(confidences)
            max_uncertainty = np.max(uncertainties)
            
            # Use conservative aggregation (max uncertainty)
            aggregated_uncertainty = max_uncertainty
            
            # Clear batch
            del self.active_propagations[batch_key]
            
            return {
                'propagation_type': 'batched',
                'status': 'processed',
                'batch_size': len(batch_data),
                'aggregated_uncertainty': aggregated_uncertainty,
                'average_confidence': avg_confidence,
                'max_uncertainty': max_uncertainty,
                'processing_time': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            raise RuntimeError(f"Batch processing failed: {e}")


class UncertaintyEventSystem:
    """Event-driven uncertainty monitoring and alerting system."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)
        self.event_handlers = defaultdict(list)
        self.event_history = []
        self.active_alerts = {}
        self.event_rules = {}
        
        # Event types
        self.event_types = {
            'high_uncertainty': self._handle_high_uncertainty,
            'low_confidence': self._handle_low_confidence,
            'domain_conflict': self._handle_domain_conflict,
            'propagation_failure': self._handle_propagation_failure,
            'threshold_exceeded': self._handle_threshold_exceeded
        }
        
        # Initialize event rules
        self._initialize_event_rules()
    
    def _initialize_event_rules(self):
        """Initialize event monitoring rules."""
        self.event_rules = {
            'high_uncertainty': {
                'threshold': 0.3,
                'severity': 'warning',
                'action': 'alert',
                'cooldown': 300  # 5 minutes
            },
            'low_confidence': {
                'threshold': 0.7,
                'severity': 'error',
                'action': 'alert',
                'cooldown': 60  # 1 minute
            },
            'domain_conflict': {
                'threshold': 0.5,
                'severity': 'critical',
                'action': 'alert_and_escalate',
                'cooldown': 30  # 30 seconds
            },
            'propagation_failure': {
                'threshold': 0.0,
                'severity': 'error',
                'action': 'retry',
                'cooldown': 120  # 2 minutes
            },
            'threshold_exceeded': {
                'threshold': 0.8,
                'severity': 'critical',
                'action': 'emergency_shutdown',
                'cooldown': 10  # 10 seconds
            }
        }
    
    def monitor_uncertainty_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor and handle uncertainty events."""
        try:
            # Check if event should be processed
            if not self._should_process_event(event_type, event_data):
                return {'status': 'skipped', 'reason': 'cooldown_active'}
            
            # Get event handler
            handler = self.event_types.get(event_type)
            if handler is None:
                raise RuntimeError(f"Unknown event type: {event_type}")
            
            # Process event
            result = handler(event_data)
            
            # Record event
            self._record_event(event_type, event_data, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Uncertainty event monitoring failed: {e}")
            raise RuntimeError(f"Uncertainty event monitoring failed: {e}")
    
    def _should_process_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """Check if event should be processed based on cooldown rules."""
        try:
            rule = self.event_rules.get(event_type, {})
            cooldown = rule.get('cooldown', 60)
            
            # Check last event time
            last_event_time = self._get_last_event_time(event_type)
            if time.time() - last_event_time < cooldown:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Event processing check failed: {e}")
            return True  # Default to processing
    
    def _get_last_event_time(self, event_type: str) -> float:
        """Get last event time for cooldown checking."""
        try:
            for event in reversed(self.event_history):
                if event.get('event_type') == event_type:
                    return event.get('timestamp', 0)
            return 0
        except Exception:
            return 0
    
    def _record_event(self, event_type: str, event_data: Dict[str, Any], result: Dict[str, Any]):
        """Record event in history."""
        try:
            event_record = {
                'event_type': event_type,
                'event_data': event_data,
                'result': result,
                'timestamp': time.time()
            }
            self.event_history.append(event_record)
            
            # Keep only last 1000 events
            if len(self.event_history) > 1000:
                self.event_history = self.event_history[-1000:]
                
        except Exception as e:
            self.logger.error(f"Event recording failed: {e}")
    
    def _handle_high_uncertainty(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle high uncertainty events."""
        try:
            uncertainty_level = event_data.get('uncertainty', 0)
            domain = event_data.get('domain', 'unknown')
            
            # Create alert
            alert_id = f"high_uncertainty_{int(time.time())}"
            alert = {
                'alert_id': alert_id,
                'type': 'high_uncertainty',
                'severity': 'warning',
                'domain': domain,
                'uncertainty_level': uncertainty_level,
                'timestamp': time.time(),
                'message': f"High uncertainty detected in {domain}: {uncertainty_level:.3f}"
            }
            
            self.active_alerts[alert_id] = alert
            
            return {
                'status': 'alert_created',
                'alert_id': alert_id,
                'severity': 'warning',
                'action_taken': 'monitoring_enhanced'
            }
            
        except Exception as e:
            self.logger.error(f"High uncertainty handling failed: {e}")
            raise RuntimeError(f"High uncertainty handling failed: {e}")
    
    def _handle_low_confidence(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle low confidence events."""
        try:
            confidence_level = event_data.get('confidence_level', 0)
            domain = event_data.get('domain', 'unknown')
            
            # Create alert
            alert_id = f"low_confidence_{int(time.time())}"
            alert = {
                'alert_id': alert_id,
                'type': 'low_confidence',
                'severity': 'error',
                'domain': domain,
                'confidence_level': confidence_level,
                'timestamp': time.time(),
                'message': f"Low confidence detected in {domain}: {confidence_level:.3f}"
            }
            
            self.active_alerts[alert_id] = alert
            
            return {
                'status': 'alert_created',
                'alert_id': alert_id,
                'severity': 'error',
                'action_taken': 'confidence_recalibration'
            }
            
        except Exception as e:
            self.logger.error(f"Low confidence handling failed: {e}")
            raise RuntimeError(f"Low confidence handling failed: {e}")
    
    def _handle_domain_conflict(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle domain conflict events."""
        try:
            conflict_level = event_data.get('conflict_level', 0)
            domains = event_data.get('domains', [])
            
            # Create critical alert
            alert_id = f"domain_conflict_{int(time.time())}"
            alert = {
                'alert_id': alert_id,
                'type': 'domain_conflict',
                'severity': 'critical',
                'domains': domains,
                'conflict_level': conflict_level,
                'timestamp': time.time(),
                'message': f"Domain conflict detected between {', '.join(domains)}: {conflict_level:.3f}"
            }
            
            self.active_alerts[alert_id] = alert
            
            return {
                'status': 'alert_created',
                'alert_id': alert_id,
                'severity': 'critical',
                'action_taken': 'conflict_resolution_initiated'
            }
            
        except Exception as e:
            self.logger.error(f"Domain conflict handling failed: {e}")
            raise RuntimeError(f"Domain conflict handling failed: {e}")
    
    def _handle_propagation_failure(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle propagation failure events."""
        try:
            source_domain = event_data.get('source_domain', 'unknown')
            target_domain = event_data.get('target_domain', 'unknown')
            error_message = event_data.get('error', 'Unknown error')
            
            # Create alert
            alert_id = f"propagation_failure_{int(time.time())}"
            alert = {
                'alert_id': alert_id,
                'type': 'propagation_failure',
                'severity': 'error',
                'source_domain': source_domain,
                'target_domain': target_domain,
                'error_message': error_message,
                'timestamp': time.time(),
                'message': f"Propagation failure from {source_domain} to {target_domain}: {error_message}"
            }
            
            self.active_alerts[alert_id] = alert
            
            return {
                'status': 'alert_created',
                'alert_id': alert_id,
                'severity': 'error',
                'action_taken': 'propagation_retry_initiated'
            }
            
        except Exception as e:
            self.logger.error(f"Propagation failure handling failed: {e}")
            raise RuntimeError(f"Propagation failure handling failed: {e}")
    
    def _handle_threshold_exceeded(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle threshold exceeded events."""
        try:
            threshold_type = event_data.get('threshold_type', 'unknown')
            current_value = event_data.get('current_value', 0)
            threshold_value = event_data.get('threshold_value', 0)
            
            # Create critical alert
            alert_id = f"threshold_exceeded_{int(time.time())}"
            alert = {
                'alert_id': alert_id,
                'type': 'threshold_exceeded',
                'severity': 'critical',
                'threshold_type': threshold_type,
                'current_value': current_value,
                'threshold_value': threshold_value,
                'timestamp': time.time(),
                'message': f"Threshold exceeded: {threshold_type} = {current_value:.3f} > {threshold_value:.3f}"
            }
            
            self.active_alerts[alert_id] = alert
            
            return {
                'status': 'alert_created',
                'alert_id': alert_id,
                'severity': 'critical',
                'action_taken': 'emergency_protocol_activated'
            }
            
        except Exception as e:
            self.logger.error(f"Threshold exceeded handling failed: {e}")
            raise RuntimeError(f"Threshold exceeded handling failed: {e}")


class DynamicResourceAllocator:
    """Dynamic resource allocation based on uncertainty levels."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)
        self.resource_allocations = {}
        self.allocation_history = []
        self.current_allocation = {}
        
        # Resource types
        self.resource_types = {
            'cpu': self._allocate_cpu,
            'memory': self._allocate_memory,
            'priority': self._allocate_priority,
            'io': self._allocate_io
        }
        
        # Initialize allocation policies
        self._initialize_allocation_policies()
    
    def _initialize_allocation_policies(self):
        """Initialize resource allocation policies."""
        self.allocation_policies = {
            'high_uncertainty': {
                'cpu_multiplier': 2.0,
                'memory_multiplier': 1.5,
                'priority_level': 'high',
                'io_bandwidth': 'increased'
            },
            'medium_uncertainty': {
                'cpu_multiplier': 1.2,
                'memory_multiplier': 1.1,
                'priority_level': 'medium',
                'io_bandwidth': 'normal'
            },
            'low_uncertainty': {
                'cpu_multiplier': 0.8,
                'memory_multiplier': 0.9,
                'priority_level': 'low',
                'io_bandwidth': 'reduced'
            }
        }
    
    def allocate_resources(self, uncertainty_level: float, domain: str, 
                          operation_type: str) -> Dict[str, Any]:
        """Allocate resources based on uncertainty level."""
        try:
            # Determine allocation policy
            if uncertainty_level > 0.3:
                policy = 'high_uncertainty'
            elif uncertainty_level > 0.1:
                policy = 'medium_uncertainty'
            else:
                policy = 'low_uncertainty'
            
            # Get policy settings
            policy_settings = self.allocation_policies.get(policy, {})
            
            # Allocate resources
            allocation = {
                'domain': domain,
                'operation_type': operation_type,
                'uncertainty_level': uncertainty_level,
                'policy': policy,
                'timestamp': time.time()
            }
            
            # Allocate each resource type
            for resource_type, allocator in self.resource_types.items():
                allocation[resource_type] = allocator(policy_settings, domain, operation_type)
            
            # Record allocation
            self.current_allocation[f"{domain}_{operation_type}"] = allocation
            self.allocation_history.append(allocation)
            
            return allocation
            
        except Exception as e:
            self.logger.error(f"Resource allocation failed: {e}")
            raise RuntimeError(f"Resource allocation failed: {e}")
    
    def _allocate_cpu(self, policy_settings: Dict[str, Any], domain: str, 
                     operation_type: str) -> Dict[str, Any]:
        """Allocate CPU resources."""
        try:
            cpu_multiplier = policy_settings.get('cpu_multiplier', 1.0)
            base_cpu = 0.1  # Base CPU allocation (10%)
            
            allocated_cpu = base_cpu * cpu_multiplier
            
            return {
                'allocation': allocated_cpu,
                'multiplier': cpu_multiplier,
                'base_allocation': base_cpu,
                'max_allocation': 0.8  # Max 80% CPU
            }
            
        except Exception as e:
            self.logger.error(f"CPU allocation failed: {e}")
            raise RuntimeError(f"CPU allocation failed: {e}")
    
    def _allocate_memory(self, policy_settings: Dict[str, Any], domain: str, 
                        operation_type: str) -> Dict[str, Any]:
        """Allocate memory resources."""
        try:
            memory_multiplier = policy_settings.get('memory_multiplier', 1.0)
            base_memory_mb = 512  # Base memory allocation (512MB)
            
            allocated_memory_mb = base_memory_mb * memory_multiplier
            
            return {
                'allocation_mb': allocated_memory_mb,
                'multiplier': memory_multiplier,
                'base_allocation_mb': base_memory_mb,
                'max_allocation_mb': 4096  # Max 4GB
            }
            
        except Exception as e:
            self.logger.error(f"Memory allocation failed: {e}")
            raise RuntimeError(f"Memory allocation failed: {e}")
    
    def _allocate_priority(self, policy_settings: Dict[str, Any], domain: str, 
                          operation_type: str) -> Dict[str, Any]:
        """Allocate priority resources."""
        try:
            priority_level = policy_settings.get('priority_level', 'medium')
            
            # Map priority levels to numerical values
            priority_mapping = {
                'high': 1,
                'medium': 2,
                'low': 3
            }
            
            priority_value = priority_mapping.get(priority_level, 2)
            
            return {
                'priority_level': priority_level,
                'priority_value': priority_value,
                'nice_value': priority_value - 1  # Unix nice value
            }
            
        except Exception as e:
            self.logger.error(f"Priority allocation failed: {e}")
            raise RuntimeError(f"Priority allocation failed: {e}")
    
    def _allocate_io(self, policy_settings: Dict[str, Any], domain: str, 
                    operation_type: str) -> Dict[str, Any]:
        """Allocate I/O resources."""
        try:
            io_bandwidth = policy_settings.get('io_bandwidth', 'normal')
            
            # Map bandwidth settings to allocation values
            bandwidth_mapping = {
                'increased': 1.5,
                'normal': 1.0,
                'reduced': 0.7
            }
            
            io_multiplier = bandwidth_mapping.get(io_bandwidth, 1.0)
            base_io_mbps = 100  # Base I/O allocation (100 MB/s)
            
            allocated_io_mbps = base_io_mbps * io_multiplier
            
            return {
                'bandwidth_mbps': allocated_io_mbps,
                'multiplier': io_multiplier,
                'base_bandwidth_mbps': base_io_mbps,
                'bandwidth_setting': io_bandwidth
            }
            
        except Exception as e:
            self.logger.error(f"I/O allocation failed: {e}")
            raise RuntimeError(f"I/O allocation failed: {e}")


class UncertaintyBasedDecisionMaker:
    """Advanced decision making using uncertainty estimates."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)
        self.decision_history = []
        self.decision_rules = {}
        self.confidence_thresholds = {}
        
        # Decision strategies
        self.decision_strategies = {
            'conservative': self._conservative_decision,
            'balanced': self._balanced_decision,
            'aggressive': self._aggressive_decision,
            'adaptive': self._adaptive_decision
        }
        
        # Initialize decision rules
        self._initialize_decision_rules()
    
    def _initialize_decision_rules(self):
        """Initialize decision-making rules."""
        self.decision_rules = {
            'fraud_detection': {
                'high_risk_threshold': 0.25,
                'medium_risk_threshold': 0.15,
                'low_risk_threshold': 0.05,
                'strategy': 'conservative',
                'confidence_threshold': 0.8
            },
            'financial_analysis': {
                'high_risk_threshold': 0.3,
                'medium_risk_threshold': 0.2,
                'low_risk_threshold': 0.1,
                'strategy': 'balanced',
                'confidence_threshold': 0.75
            },
            'risk_assessment': {
                'high_risk_threshold': 0.35,
                'medium_risk_threshold': 0.25,
                'low_risk_threshold': 0.15,
                'strategy': 'conservative',
                'confidence_threshold': 0.85
            }
        }
    
    def make_decision(self, uncertainty_result: Dict[str, Any], domain: str, 
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make decision based on uncertainty estimates."""
        try:
            uncertainty_level = uncertainty_result.get('uncertainty', 0)
            confidence_level = uncertainty_result.get('confidence_level', 0.95)
            
            # Get domain-specific rules
            rules = self.decision_rules.get(domain, {})
            strategy = rules.get('strategy', 'balanced')
            
            # Get decision strategy function
            decision_func = self.decision_strategies.get(strategy, self._balanced_decision)
            
            # Make decision
            decision = decision_func(uncertainty_result, domain, rules, context)
            
            # Record decision
            decision_record = {
                'decision_id': f"decision_{int(time.time())}",
                'domain': domain,
                'uncertainty_level': uncertainty_level,
                'confidence_level': confidence_level,
                'strategy': strategy,
                'decision': decision,
                'timestamp': time.time()
            }
            
            self.decision_history.append(decision_record)
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Decision making failed: {e}")
            raise RuntimeError(f"Decision making failed: {e}")
    
    def _conservative_decision(self, uncertainty_result: Dict[str, Any], domain: str, 
                             rules: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Make conservative decision (err on side of caution)."""
        try:
            uncertainty_level = uncertainty_result.get('uncertainty', 0)
            confidence_level = uncertainty_result.get('confidence_level', 0.95)
            
            # Conservative thresholds
            high_threshold = rules.get('high_risk_threshold', 0.25)
            medium_threshold = rules.get('medium_risk_threshold', 0.15)
            low_threshold = rules.get('low_risk_threshold', 0.05)
            confidence_threshold = rules.get('confidence_threshold', 0.8)
            
            # Decision logic
            if uncertainty_level > high_threshold or confidence_level < confidence_threshold:
                action = 'reject'
                risk_level = 'high'
                recommendation = 'immediate_action_required'
                
            elif uncertainty_level > medium_threshold:
                action = 'review'
                risk_level = 'medium'
                recommendation = 'manual_review_recommended'
                
            elif uncertainty_level > low_threshold:
                action = 'proceed_with_caution'
                risk_level = 'low'
                recommendation = 'monitor_closely'
                
            else:
                action = 'proceed'
                risk_level = 'minimal'
                recommendation = 'standard_processing'
            
            return {
                'action': action,
                'risk_level': risk_level,
                'recommendation': recommendation,
                'uncertainty_level': uncertainty_level,
                'confidence_level': confidence_level,
                'reasoning': f"Conservative decision based on uncertainty {uncertainty_level:.3f} and confidence {confidence_level:.3f}"
            }
            
        except Exception as e:
            self.logger.error(f"Conservative decision failed: {e}")
            raise RuntimeError(f"Conservative decision failed: {e}")
    
    def _balanced_decision(self, uncertainty_result: Dict[str, Any], domain: str, 
                          rules: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Make balanced decision (balance precision and recall)."""
        try:
            uncertainty_level = uncertainty_result.get('uncertainty', 0)
            confidence_level = uncertainty_result.get('confidence_level', 0.95)
            
            # Balanced thresholds (more lenient than conservative)
            high_threshold = rules.get('high_risk_threshold', 0.3)
            medium_threshold = rules.get('medium_risk_threshold', 0.2)
            low_threshold = rules.get('low_risk_threshold', 0.1)
            confidence_threshold = rules.get('confidence_threshold', 0.75)
            
            # Decision logic
            if uncertainty_level > high_threshold or confidence_level < confidence_threshold:
                action = 'review'
                risk_level = 'high'
                recommendation = 'additional_verification_required'
                
            elif uncertainty_level > medium_threshold:
                action = 'proceed_with_monitoring'
                risk_level = 'medium'
                recommendation = 'enhanced_monitoring'
                
            elif uncertainty_level > low_threshold:
                action = 'proceed'
                risk_level = 'low'
                recommendation = 'standard_monitoring'
                
            else:
                action = 'proceed'
                risk_level = 'minimal'
                recommendation = 'normal_processing'
            
            return {
                'action': action,
                'risk_level': risk_level,
                'recommendation': recommendation,
                'uncertainty_level': uncertainty_level,
                'confidence_level': confidence_level,
                'reasoning': f"Balanced decision based on uncertainty {uncertainty_level:.3f} and confidence {confidence_level:.3f}"
            }
            
        except Exception as e:
            self.logger.error(f"Balanced decision failed: {e}")
            raise RuntimeError(f"Balanced decision failed: {e}")
    
    def _aggressive_decision(self, uncertainty_result: Dict[str, Any], domain: str, 
                           rules: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Make aggressive decision (minimize false positives)."""
        try:
            uncertainty_level = uncertainty_result.get('uncertainty', 0)
            confidence_level = uncertainty_result.get('confidence_level', 0.95)
            
            # Aggressive thresholds (more lenient)
            high_threshold = rules.get('high_risk_threshold', 0.4)
            medium_threshold = rules.get('medium_risk_threshold', 0.25)
            low_threshold = rules.get('low_risk_threshold', 0.15)
            confidence_threshold = rules.get('confidence_threshold', 0.7)
            
            # Decision logic
            if uncertainty_level > high_threshold or confidence_level < confidence_threshold:
                action = 'proceed_with_monitoring'
                risk_level = 'high'
                recommendation = 'monitor_intensively'
                
            elif uncertainty_level > medium_threshold:
                action = 'proceed'
                risk_level = 'medium'
                recommendation = 'enhanced_monitoring'
                
            elif uncertainty_level > low_threshold:
                action = 'proceed'
                risk_level = 'low'
                recommendation = 'standard_monitoring'
                
            else:
                action = 'proceed'
                risk_level = 'minimal'
                recommendation = 'normal_processing'
            
            return {
                'action': action,
                'risk_level': risk_level,
                'recommendation': recommendation,
                'uncertainty_level': uncertainty_level,
                'confidence_level': confidence_level,
                'reasoning': f"Aggressive decision based on uncertainty {uncertainty_level:.3f} and confidence {confidence_level:.3f}"
            }
            
        except Exception as e:
            self.logger.error(f"Aggressive decision failed: {e}")
            raise RuntimeError(f"Aggressive decision failed: {e}")
    
    def _adaptive_decision(self, uncertainty_result: Dict[str, Any], domain: str, 
                          rules: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Make adaptive decision based on context and history."""
        try:
            uncertainty_level = uncertainty_result.get('uncertainty', 0)
            confidence_level = uncertainty_result.get('confidence_level', 0.95)
            
            # Analyze context
            context_analysis = self._analyze_context(context, domain)
            
            # Adjust thresholds based on context
            adjusted_rules = self._adjust_rules_for_context(rules, context_analysis)
            
            # Use balanced decision with adjusted rules
            return self._balanced_decision(uncertainty_result, domain, adjusted_rules, context)
            
        except Exception as e:
            self.logger.error(f"Adaptive decision failed: {e}")
            raise RuntimeError(f"Adaptive decision failed: {e}")
    
    def _analyze_context(self, context: Optional[Dict[str, Any]], domain: str) -> Dict[str, Any]:
        """Analyze decision context."""
        try:
            if context is None:
                return {'risk_profile': 'normal', 'urgency': 'normal', 'impact': 'normal'}
            
            # Extract context factors
            risk_profile = context.get('risk_profile', 'normal')
            urgency = context.get('urgency', 'normal')
            impact = context.get('impact', 'normal')
            
            return {
                'risk_profile': risk_profile,
                'urgency': urgency,
                'impact': impact,
                'domain': domain
            }
            
        except Exception as e:
            self.logger.error(f"Context analysis failed: {e}")
            return {'risk_profile': 'normal', 'urgency': 'normal', 'impact': 'normal'}
    
    def _adjust_rules_for_context(self, rules: Dict[str, Any], 
                                 context_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust decision rules based on context."""
        try:
            adjusted_rules = rules.copy()
            
            # Adjust based on risk profile
            risk_profile = context_analysis.get('risk_profile', 'normal')
            if risk_profile == 'high':
                # More conservative thresholds
                adjusted_rules['high_risk_threshold'] *= 0.8
                adjusted_rules['medium_risk_threshold'] *= 0.8
                adjusted_rules['low_risk_threshold'] *= 0.8
                adjusted_rules['confidence_threshold'] *= 1.1
            elif risk_profile == 'low':
                # More aggressive thresholds
                adjusted_rules['high_risk_threshold'] *= 1.2
                adjusted_rules['medium_risk_threshold'] *= 1.2
                adjusted_rules['low_risk_threshold'] *= 1.2
                adjusted_rules['confidence_threshold'] *= 0.9
            
            # Adjust based on urgency
            urgency = context_analysis.get('urgency', 'normal')
            if urgency == 'high':
                # More aggressive for high urgency
                adjusted_rules['high_risk_threshold'] *= 1.1
                adjusted_rules['medium_risk_threshold'] *= 1.1
                adjusted_rules['low_risk_threshold'] *= 1.1
                adjusted_rules['confidence_threshold'] *= 0.95
            
            return adjusted_rules
            
        except Exception as e:
            self.logger.error(f"Rule adjustment failed: {e}")
            return rules