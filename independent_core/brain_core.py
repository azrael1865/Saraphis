"""
Brain Core Foundation Class for Universal AI Core.
Manages shared reasoning capabilities across multiple domains while preventing catastrophic forgetting.
Now with enhanced uncertainty quantification capabilities.
"""

import logging
import threading
import json
import time
import uuid
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import warnings
from enum import Enum


class UncertaintyType(Enum):
    """Types of uncertainty in predictions."""
    EPISTEMIC = "epistemic"  # Model uncertainty
    ALEATORIC = "aleatoric"  # Data uncertainty
    TOTAL = "total"         # Combined uncertainty


@dataclass
class UncertaintyMetrics:
    """Comprehensive uncertainty metrics for a prediction."""
    mean: float
    variance: float
    std: float
    confidence_interval: Tuple[float, float]
    prediction_interval: Tuple[float, float]
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    model_confidence: float
    credible_regions: Dict[str, Tuple[float, float]]
    entropy: float = 0.0
    mutual_information: float = 0.0
    reliability_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'mean': self.mean,
            'variance': self.variance,
            'std': self.std,
            'confidence_interval': list(self.confidence_interval),
            'prediction_interval': list(self.prediction_interval),
            'epistemic_uncertainty': self.epistemic_uncertainty,
            'aleatoric_uncertainty': self.aleatoric_uncertainty,
            'model_confidence': self.model_confidence,
            'credible_regions': {k: list(v) for k, v in self.credible_regions.items()},
            'entropy': self.entropy,
            'mutual_information': self.mutual_information,
            'reliability_score': self.reliability_score
        }


@dataclass
class BrainConfig:
    """Configuration for Brain Core system."""
    
    # Shared knowledge settings
    shared_memory_size: int = 10000
    knowledge_persistence: bool = True
    knowledge_path: Optional[Path] = None
    
    # Reasoning settings
    reasoning_depth: int = 5
    context_window_size: int = 2048
    
    # Performance settings
    enable_caching: bool = True
    cache_size: int = 1000
    thread_safe: bool = True
    
    # Uncertainty settings
    enable_uncertainty: bool = True
    uncertainty_history_size: int = 1000
    confidence_threshold: float = 0.7
    reliability_window: int = 100
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.shared_memory_size <= 0:
            raise ValueError("shared_memory_size must be positive")
        
        if self.reasoning_depth <= 0:
            raise ValueError("reasoning_depth must be positive")
        
        if self.context_window_size <= 0:
            raise ValueError("context_window_size must be positive")
        
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        if self.knowledge_persistence and self.knowledge_path:
            self.knowledge_path = Path(self.knowledge_path)
            self.knowledge_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class PredictionResult:
    """Result from a brain prediction operation."""
    prediction_id: str
    success: bool
    predicted_value: Any
    confidence: float
    domain: Optional[str] = None
    reasoning_steps: List[str] = field(default_factory=list)
    uncertainty_metrics: Optional[UncertaintyMetrics] = None
    computation_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction result to dictionary."""
        result = {
            'prediction_id': self.prediction_id,
            'success': self.success,
            'predicted_value': self.predicted_value,
            'confidence': self.confidence,
            'domain': self.domain,
            'reasoning_steps': self.reasoning_steps,
            'computation_time': self.computation_time,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'error_message': self.error_message
        }
        
        if self.uncertainty_metrics:
            result['uncertainty_metrics'] = self.uncertainty_metrics.to_dict()
        
        return result


class BrainCore:
    """
    Core Brain class that manages shared reasoning capabilities.
    Provides foundation for domain-specific lobes while maintaining shared knowledge.
    Enhanced with uncertainty quantification capabilities.
    """
    
    def __init__(self, config: Optional[Union[BrainConfig, Dict[str, Any]]] = None):
        """
        Initialize Brain Core with configuration.
        
        Args:
            config: BrainConfig object or dict with configuration parameters
        """
        # Handle configuration
        if config is None:
            self.config = BrainConfig()
        elif isinstance(config, dict):
            self.config = BrainConfig(**config)
        elif isinstance(config, BrainConfig):
            self.config = config
        else:
            raise TypeError(f"Invalid config type: {type(config)}")
        
        # Setup logging
        self._setup_logging()
        
        # Initialize shared state
        self._shared_state: Dict[str, Any] = {
            'knowledge_base': defaultdict(dict),
            'reasoning_patterns': {},
            'cross_domain_insights': [],
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '1.0.0',
                'total_domains': 0,
                'last_updated': datetime.now().isoformat()
            }
        }
        
        # Knowledge management indices for efficient retrieval
        self._knowledge_index: Dict[str, Set[str]] = defaultdict(set)  # key -> domains containing that key
        self._text_search_index: Dict[str, Set[str]] = defaultdict(set)  # search terms -> knowledge keys
        self._knowledge_relationships: Dict[str, Set[str]] = defaultdict(set)  # key -> related keys
        
        # Uncertainty tracking
        self._uncertainty_history = deque(maxlen=self.config.uncertainty_history_size)
        self._prediction_accuracy_history = deque(maxlen=self.config.reliability_window)
        self._domain_confidence_scores: Dict[str, float] = defaultdict(lambda: 0.5)
        self._uncertainty_calibration_data = {
            'predictions': [],
            'true_values': [],
            'confidence_intervals': []
        }
        
        # Thread safety
        self._state_lock = threading.RLock() if self.config.thread_safe else None
        
        # Caching
        self._cache: Dict[str, Any] = {} if self.config.enable_caching else None
        self._cache_lock = threading.Lock() if self.config.enable_caching and self.config.thread_safe else None
        
        # Auto-backup state variables
        self._auto_backup_enabled = False
        self._auto_backup_interval = 3600  # 1 hour default
        self._auto_backup_path: Optional[Path] = None
        self._auto_backup_thread: Optional[threading.Thread] = None
        self._last_backup_time: Optional[datetime] = None
        
        # Load persisted knowledge if available
        if self.config.knowledge_persistence and self.config.knowledge_path and self.config.knowledge_path.exists():
            self._load_persisted_knowledge()
        
        self.logger.info(f"BrainCore initialized with config: shared_memory_size={self.config.shared_memory_size}, "
                        f"reasoning_depth={self.config.reasoning_depth}, uncertainty={self.config.enable_uncertainty}")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration for Brain Core."""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Don't add handlers if they already exist
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(self.config.log_format)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Set log level
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        
        self.logger.debug("Logging initialized for BrainCore")
    
    def predict(self, input_data: Any, domain: Optional[str] = None) -> PredictionResult:
        """
        Make a prediction using the brain's reasoning capabilities.
        
        Args:
            input_data: Input data for prediction (can be dict, list, numpy array, or primitive)
            domain: Optional domain hint for specialized processing
            
        Returns:
            PredictionResult with prediction details and metadata
            
        Raises:
            ValueError: If input_data is invalid
            RuntimeError: If prediction fails
        """
        start_time = time.time()
        prediction_id = str(uuid.uuid4())
        reasoning_steps = []
        
        try:
            # Input validation and sanitization
            validated_input, input_type = self._validate_and_sanitize_input(input_data)
            reasoning_steps.append(f"Input validated: type={input_type}, size={self._get_input_size(validated_input)}")
            
            # Check cache first if enabled
            cache_key = None
            if self.config.enable_caching:
                cache_key = self._generate_cache_key(validated_input, domain)
                cached_result = self._get_cached_prediction(cache_key)
                if cached_result:
                    self.logger.debug(f"Cache hit for prediction {prediction_id}")
                    cached_result.prediction_id = prediction_id  # Update ID for this request
                    cached_result.computation_time = time.time() - start_time
                    return cached_result
            
            # Prepare context from shared state
            context = self._prepare_prediction_context(validated_input, domain)
            reasoning_steps.append(f"Context prepared: domains={len(context['available_domains'])}, patterns={len(context['relevant_patterns'])}")
            
            # Domain routing preparation
            selected_domain = self._select_domain(validated_input, domain, context)
            reasoning_steps.append(f"Domain selected: {selected_domain or 'general'}")
            
            # Core prediction logic
            if selected_domain and selected_domain in context['available_domains']:
                # Domain-specific prediction (placeholder for future lobe integration)
                prediction_value, base_confidence, base_uncertainty = self._domain_specific_prediction(
                    validated_input, selected_domain, context
                )
                reasoning_steps.append(f"Domain-specific prediction completed: confidence={base_confidence:.3f}")
            else:
                # General prediction using shared knowledge
                prediction_value, base_confidence, base_uncertainty = self._general_prediction(
                    validated_input, context
                )
                reasoning_steps.append(f"General prediction completed: confidence={base_confidence:.3f}")
            
            # Apply reasoning patterns
            enhanced_prediction = self._apply_reasoning_patterns(
                prediction_value, context, reasoning_steps
            )
            if enhanced_prediction != prediction_value:
                reasoning_steps.append("Reasoning patterns applied to enhance prediction")
                prediction_value = enhanced_prediction
            
            # Calculate comprehensive uncertainty metrics
            uncertainty_metrics = None
            if self.config.enable_uncertainty:
                uncertainty_metrics = self.calculate_uncertainty({
                    'predicted_value': prediction_value,
                    'base_confidence': base_confidence,
                    'base_uncertainty': base_uncertainty,
                    'context': context,
                    'domain': selected_domain,
                    'input_type': input_type,
                    'input_data': validated_input
                })
                reasoning_steps.append(f"Uncertainty quantified: epistemic={uncertainty_metrics.epistemic_uncertainty:.3f}, "
                                     f"aleatoric={uncertainty_metrics.aleatoric_uncertainty:.3f}")
            
            # Calculate final confidence based on uncertainty
            if uncertainty_metrics:
                final_confidence = uncertainty_metrics.model_confidence
            else:
                final_confidence = self._calculate_final_confidence(base_confidence, base_uncertainty, context)
            
            # Create successful result
            result = PredictionResult(
                prediction_id=prediction_id,
                success=True,
                predicted_value=prediction_value,
                confidence=final_confidence,
                domain=selected_domain,
                reasoning_steps=reasoning_steps,
                uncertainty_metrics=uncertainty_metrics,
                computation_time=time.time() - start_time,
                metadata={
                    'input_type': input_type,
                    'context_size': len(str(context)),
                    'cache_hit': False,
                    'reasoning_depth': min(len(reasoning_steps), self.config.reasoning_depth),
                    'reliability_score': self.assess_reliability(
                        {'predicted_value': prediction_value, 'confidence': final_confidence, 'domain': selected_domain}
                    ) if self.config.enable_uncertainty else None
                }
            )
            
            # Track uncertainty history
            if self.config.enable_uncertainty:
                self._track_uncertainty(result)
            
            # Cache the result if enabled
            if self.config.enable_caching and cache_key:
                self._cache_prediction(cache_key, result)
            
            # Update cross-domain insights
            self._update_cross_domain_insights(result, context)
            
            return result
            
        except ValueError as e:
            # Input validation errors
            return PredictionResult(
                prediction_id=prediction_id,
                success=False,
                predicted_value=None,
                confidence=0.0,
                domain=domain,
                reasoning_steps=reasoning_steps,
                error_message=f"Invalid input: {str(e)}",
                computation_time=time.time() - start_time
            )
        except Exception as e:
            # Other errors
            self.logger.error(f"Prediction failed: {e}")
            return PredictionResult(
                prediction_id=prediction_id,
                success=False,
                predicted_value=None,
                confidence=0.0,
                domain=domain,
                reasoning_steps=reasoning_steps,
                error_message=f"Prediction failed: {str(e)}",
                computation_time=time.time() - start_time
            )
    
    def calculate_uncertainty(self, prediction_data: Dict[str, Any]) -> UncertaintyMetrics:
        """
        Calculate comprehensive uncertainty metrics for a prediction.
        
        Args:
            prediction_data: Dictionary containing prediction information
                - predicted_value: The prediction output
                - base_confidence: Initial confidence score
                - base_uncertainty: Base uncertainty estimates
                - context: Prediction context
                - domain: Domain used for prediction
                - input_type: Type of input data
                - input_data: The actual input data
        
        Returns:
            UncertaintyMetrics with comprehensive uncertainty quantification
        """
        try:
            predicted_value = prediction_data.get('predicted_value')
            base_confidence = prediction_data.get('base_confidence', 0.5)
            base_uncertainty = prediction_data.get('base_uncertainty', {})
            context = prediction_data.get('context', {})
            domain = prediction_data.get('domain', 'general')
            input_type = prediction_data.get('input_type', 'unknown')
            
            # Extract base uncertainties
            epistemic_base = base_uncertainty.get('epistemic', 0.2)
            aleatoric_base = base_uncertainty.get('aleatoric', 0.1)
            
            # Calculate domain-specific adjustments
            domain_confidence = self._domain_confidence_scores.get(domain, 0.5)
            domain_adjustment = (1 - domain_confidence) * 0.1
            
            # Adjust uncertainties based on context
            context_factor = self._calculate_context_uncertainty_factor(context)
            
            # Calculate epistemic uncertainty (model uncertainty)
            epistemic_uncertainty = epistemic_base * context_factor + domain_adjustment
            
            # Calculate aleatoric uncertainty (data uncertainty)
            aleatoric_uncertainty = aleatoric_base * self._calculate_data_uncertainty_factor(
                prediction_data.get('input_data'), input_type
            )
            
            # Calculate total uncertainty
            total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
            
            # Calculate variance and standard deviation
            variance = total_uncertainty**2
            std = total_uncertainty
            
            # Calculate mean (for numerical predictions)
            if isinstance(predicted_value, (int, float)):
                mean = float(predicted_value)
            elif isinstance(predicted_value, np.ndarray) and predicted_value.size == 1:
                mean = float(predicted_value)
            else:
                # For non-numerical predictions, use confidence as proxy
                mean = base_confidence
            
            # Calculate confidence and prediction intervals
            confidence_level = 0.95
            z_score = 1.96  # For 95% confidence
            
            # Confidence interval (epistemic uncertainty only)
            ci_lower = mean - z_score * epistemic_uncertainty
            ci_upper = mean + z_score * epistemic_uncertainty
            
            # Prediction interval (total uncertainty)
            pi_lower = mean - z_score * total_uncertainty
            pi_upper = mean + z_score * total_uncertainty
            
            # Calculate credible regions for different confidence levels
            credible_regions = {}
            for level in [0.5, 0.68, 0.80, 0.90, 0.95, 0.99]:
                z = self._get_z_score(level)
                credible_regions[f"{level:.0%}"] = (
                    mean - z * total_uncertainty,
                    mean + z * total_uncertainty
                )
            
            # Calculate entropy (information content)
            entropy = -0.5 * np.log(2 * np.pi * np.e * variance) if variance > 0 else 0.0
            
            # Calculate mutual information (simplified)
            mutual_information = entropy * base_confidence
            
            # Calculate model confidence incorporating uncertainties
            model_confidence = self._calculate_model_confidence(
                base_confidence, epistemic_uncertainty, aleatoric_uncertainty, context
            )
            
            # Calculate reliability score
            reliability_score = self._calculate_reliability_score(
                model_confidence, epistemic_uncertainty, domain
            )
            
            return UncertaintyMetrics(
                mean=mean,
                variance=variance,
                std=std,
                confidence_interval=(ci_lower, ci_upper),
                prediction_interval=(pi_lower, pi_upper),
                epistemic_uncertainty=epistemic_uncertainty,
                aleatoric_uncertainty=aleatoric_uncertainty,
                model_confidence=model_confidence,
                credible_regions=credible_regions,
                entropy=entropy,
                mutual_information=mutual_information,
                reliability_score=reliability_score
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating uncertainty: {e}")
            # Return default uncertainty metrics on error
            return UncertaintyMetrics(
                mean=0.0,
                variance=1.0,
                std=1.0,
                confidence_interval=(0.0, 1.0),
                prediction_interval=(0.0, 1.0),
                epistemic_uncertainty=0.5,
                aleatoric_uncertainty=0.5,
                model_confidence=0.1,
                credible_regions={'95%': (0.0, 1.0)},
                entropy=0.0,
                mutual_information=0.0,
                reliability_score=0.0
            )
    
    def get_confidence_score(self, prediction_data: Dict[str, Any]) -> float:
        """
        Get confidence score for a prediction.
        
        Args:
            prediction_data: Dictionary containing prediction information
        
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # If we have uncertainty metrics, use model confidence
            if 'uncertainty_metrics' in prediction_data and isinstance(prediction_data['uncertainty_metrics'], UncertaintyMetrics):
                return prediction_data['uncertainty_metrics'].model_confidence
            
            # Otherwise calculate from available data
            base_confidence = prediction_data.get('confidence', 0.5)
            domain = prediction_data.get('domain', 'general')
            
            # Adjust for domain confidence
            domain_confidence = self._domain_confidence_scores.get(domain, 0.5)
            
            # Adjust for prediction history
            if self._prediction_accuracy_history:
                historical_accuracy = np.mean([acc for acc in self._prediction_accuracy_history])
                confidence_adjustment = (historical_accuracy - 0.5) * 0.2
            else:
                confidence_adjustment = 0.0
            
            # Calculate final confidence
            confidence = base_confidence * 0.7 + domain_confidence * 0.2 + confidence_adjustment * 0.1
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    def assess_reliability(self, prediction_data: Dict[str, Any], historical_data: Optional[List[Dict[str, Any]]] = None) -> float:
        """
        Assess the reliability of a prediction based on historical performance and current metrics.
        
        Args:
            prediction_data: Current prediction data
            historical_data: Optional historical predictions for comparison
        
        Returns:
            Reliability score between 0 and 1
        """
        try:
            reliability_factors = []
            
            # Factor 1: Current confidence
            current_confidence = self.get_confidence_score(prediction_data)
            reliability_factors.append(current_confidence * 0.3)
            
            # Factor 2: Domain reliability
            domain = prediction_data.get('domain', 'general')
            if domain in self._domain_confidence_scores:
                domain_reliability = self._domain_confidence_scores[domain]
                reliability_factors.append(domain_reliability * 0.2)
            else:
                reliability_factors.append(0.1)  # Low reliability for unknown domains
            
            # Factor 3: Historical accuracy (if available)
            if self._prediction_accuracy_history:
                recent_accuracy = np.mean(list(self._prediction_accuracy_history)[-20:])
                reliability_factors.append(recent_accuracy * 0.25)
            else:
                reliability_factors.append(0.5 * 0.25)  # Neutral if no history
            
            # Factor 4: Uncertainty consistency
            if historical_data:
                uncertainty_consistency = self._calculate_uncertainty_consistency(prediction_data, historical_data)
                reliability_factors.append(uncertainty_consistency * 0.15)
            else:
                # Check against recent uncertainty history
                if self._uncertainty_history:
                    recent_uncertainties = [h['epistemic'] + h['aleatoric'] for h in list(self._uncertainty_history)[-10:]]
                    if recent_uncertainties:
                        avg_uncertainty = np.mean(recent_uncertainties)
                        current_uncertainty = prediction_data.get('epistemic_uncertainty', 0.5) + prediction_data.get('aleatoric_uncertainty', 0.5)
                        consistency = 1.0 - min(1.0, abs(current_uncertainty - avg_uncertainty) / avg_uncertainty)
                        reliability_factors.append(consistency * 0.15)
                    else:
                        reliability_factors.append(0.5 * 0.15)
                else:
                    reliability_factors.append(0.5 * 0.15)
            
            # Factor 5: Prediction stability
            stability_score = self._calculate_prediction_stability(prediction_data)
            reliability_factors.append(stability_score * 0.1)
            
            # Calculate overall reliability
            reliability = sum(reliability_factors)
            
            # Apply threshold
            if current_confidence < self.config.confidence_threshold:
                reliability *= 0.8  # Penalize low confidence predictions
            
            return max(0.0, min(1.0, reliability))
            
        except Exception as e:
            self.logger.error(f"Error assessing reliability: {e}")
            return 0.5
    
    def _uncertainty_metrics(self, data: Any) -> Dict[str, float]:
        """
        Internal method to calculate basic uncertainty metrics from data.
        
        Args:
            data: Input data for uncertainty calculation
        
        Returns:
            Dictionary of uncertainty metrics
        """
        try:
            metrics = {
                'data_uncertainty': 0.1,
                'model_uncertainty': 0.2,
                'total_uncertainty': 0.3
            }
            
            # Calculate data-specific uncertainties
            if isinstance(data, np.ndarray):
                if data.size > 1:
                    # Calculate variance-based uncertainty
                    data_std = np.std(data)
                    data_range = np.ptp(data)
                    if data_range > 0:
                        metrics['data_uncertainty'] = min(1.0, data_std / data_range)
                    
                    # Estimate model uncertainty from data characteristics
                    metrics['model_uncertainty'] = 0.1 + 0.1 * min(1.0, data.ndim / 10)
                    
            elif isinstance(data, list) and len(data) > 1:
                # For lists, check homogeneity
                if all(isinstance(x, (int, float)) for x in data):
                    values = np.array(data)
                    metrics['data_uncertainty'] = min(1.0, np.std(values) / (np.mean(np.abs(values)) + 1e-8))
                else:
                    # Higher uncertainty for heterogeneous data
                    metrics['data_uncertainty'] = 0.3
                    
            elif isinstance(data, dict):
                # Dictionary complexity affects uncertainty
                depth = self._get_dict_depth(data)
                metrics['model_uncertainty'] = min(0.5, 0.1 * depth)
                metrics['data_uncertainty'] = 0.1 + 0.05 * len(data)
                
            # Calculate total uncertainty
            metrics['total_uncertainty'] = np.sqrt(
                metrics['data_uncertainty']**2 + metrics['model_uncertainty']**2
            )
            
            return metrics
            
        except Exception as e:
            self.logger.debug(f"Error in _uncertainty_metrics: {e}")
            return {
                'data_uncertainty': 0.5,
                'model_uncertainty': 0.5,
                'total_uncertainty': 0.7
            }
    
    def _calculate_context_uncertainty_factor(self, context: Dict[str, Any]) -> float:
        """Calculate uncertainty factor based on context."""
        factor = 1.0
        
        # More available domains and patterns reduce uncertainty
        if 'available_domains' in context:
            num_domains = len(context['available_domains'])
            factor *= (1.0 - min(0.3, num_domains * 0.05))
        
        if 'relevant_patterns' in context:
            num_patterns = len(context['relevant_patterns'])
            factor *= (1.0 - min(0.2, num_patterns * 0.02))
        
        if 'recent_insights' in context:
            num_insights = len(context['recent_insights'])
            factor *= (1.0 - min(0.1, num_insights * 0.01))
        
        return max(0.1, factor)
    
    def _calculate_data_uncertainty_factor(self, input_data: Any, input_type: str) -> float:
        """Calculate uncertainty factor based on input data characteristics."""
        base_factor = 1.0
        
        if input_type == 'numpy_array' and isinstance(input_data, np.ndarray):
            # Check for data quality issues
            if np.any(np.isnan(input_data)):
                base_factor *= 1.5
            if input_data.size < 10:
                base_factor *= 1.2
            
        elif input_type == 'dictionary' and isinstance(input_data, dict):
            # Sparse dictionaries have higher uncertainty
            if len(input_data) < 5:
                base_factor *= 1.3
            
        elif input_type == 'string':
            # Short strings have higher uncertainty
            if len(str(input_data)) < 20:
                base_factor *= 1.2
        
        return min(2.0, base_factor)
    
    def _get_z_score(self, confidence_level: float) -> float:
        """Get z-score for confidence level."""
        # Simplified z-score lookup
        z_scores = {
            0.50: 0.674,
            0.68: 1.0,
            0.80: 1.282,
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
        
        # Find closest confidence level
        closest_level = min(z_scores.keys(), key=lambda x: abs(x - confidence_level))
        return z_scores[closest_level]
    
    def _calculate_model_confidence(self, base_confidence: float, epistemic: float, 
                                  aleatoric: float, context: Dict[str, Any]) -> float:
        """Calculate model confidence incorporating uncertainties."""
        # Start with base confidence
        confidence = base_confidence
        
        # Reduce confidence based on uncertainties
        uncertainty_penalty = (epistemic + aleatoric) / 2
        confidence *= (1 - uncertainty_penalty * 0.5)
        
        # Boost confidence if we have good context
        if context.get('domain_knowledge'):
            confidence += 0.05
        if len(context.get('relevant_patterns', [])) > 3:
            confidence += 0.03
        
        # Apply bounds
        return max(0.1, min(0.99, confidence))
    
    def _calculate_reliability_score(self, model_confidence: float, epistemic_uncertainty: float, domain: str) -> float:
        """Calculate reliability score for uncertainty metrics."""
        # Base reliability on model confidence
        reliability = model_confidence
        
        # Penalize high epistemic uncertainty
        if epistemic_uncertainty > 0.3:
            reliability *= 0.8
        
        # Boost for well-known domains
        if domain in self._domain_confidence_scores and self._domain_confidence_scores[domain] > 0.7:
            reliability *= 1.1
        
        return max(0.0, min(1.0, reliability))
    
    def _calculate_uncertainty_consistency(self, current_prediction: Dict[str, Any], 
                                         historical_data: List[Dict[str, Any]]) -> float:
        """Calculate consistency of uncertainty estimates."""
        if not historical_data:
            return 0.5
        
        try:
            # Extract uncertainty values
            current_uncertainty = current_prediction.get('total_uncertainty', 0.5)
            historical_uncertainties = [h.get('total_uncertainty', 0.5) for h in historical_data[-10:]]
            
            if not historical_uncertainties:
                return 0.5
            
            # Calculate consistency as inverse of variance
            variance = np.var(historical_uncertainties + [current_uncertainty])
            consistency = 1.0 / (1.0 + variance * 10)
            
            return consistency
            
        except Exception:
            return 0.5
    
    def _calculate_prediction_stability(self, prediction_data: Dict[str, Any]) -> float:
        """Calculate prediction stability score."""
        # Check if prediction value is stable type
        predicted_value = prediction_data.get('predicted_value')
        
        if predicted_value is None:
            return 0.0
        
        # Numerical predictions are generally more stable
        if isinstance(predicted_value, (int, float, np.number)):
            return 0.9
        elif isinstance(predicted_value, np.ndarray):
            return 0.8
        elif isinstance(predicted_value, dict) and len(predicted_value) > 0:
            return 0.7
        elif isinstance(predicted_value, str) and len(predicted_value) > 10:
            return 0.6
        else:
            return 0.5
    
    def _track_uncertainty(self, result: PredictionResult) -> None:
        """Track uncertainty metrics for historical analysis."""
        if result.uncertainty_metrics:
            self._uncertainty_history.append({
                'timestamp': result.timestamp,
                'domain': result.domain,
                'epistemic': result.uncertainty_metrics.epistemic_uncertainty,
                'aleatoric': result.uncertainty_metrics.aleatoric_uncertainty,
                'total': result.uncertainty_metrics.std,
                'confidence': result.uncertainty_metrics.model_confidence,
                'reliability': result.uncertainty_metrics.reliability_score
            })
            
            # Update domain confidence scores based on reliability
            if result.domain:
                current_score = self._domain_confidence_scores[result.domain]
                new_score = 0.9 * current_score + 0.1 * result.uncertainty_metrics.reliability_score
                self._domain_confidence_scores[result.domain] = new_score
    
    def get_uncertainty_statistics(self) -> Dict[str, Any]:
        """Get statistics about uncertainty tracking."""
        if not self._uncertainty_history:
            return {
                'total_predictions': 0,
                'average_epistemic': 0.0,
                'average_aleatoric': 0.0,
                'average_confidence': 0.0,
                'domain_confidences': dict(self._domain_confidence_scores)
            }
        
        history_list = list(self._uncertainty_history)
        
        return {
            'total_predictions': len(history_list),
            'average_epistemic': np.mean([h['epistemic'] for h in history_list]),
            'average_aleatoric': np.mean([h['aleatoric'] for h in history_list]),
            'average_confidence': np.mean([h['confidence'] for h in history_list]),
            'average_reliability': np.mean([h['reliability'] for h in history_list]),
            'confidence_trend': self._calculate_confidence_trend(),
            'domain_confidences': dict(self._domain_confidence_scores),
            'recent_predictions': history_list[-5:] if len(history_list) >= 5 else history_list
        }
    
    def _calculate_confidence_trend(self) -> str:
        """Calculate trend in confidence scores."""
        if len(self._uncertainty_history) < 10:
            return 'insufficient_data'
        
        recent = list(self._uncertainty_history)[-10:]
        older = list(self._uncertainty_history)[-20:-10] if len(self._uncertainty_history) >= 20 else list(self._uncertainty_history)[:10]
        
        recent_avg = np.mean([h['confidence'] for h in recent])
        older_avg = np.mean([h['confidence'] for h in older])
        
        if recent_avg > older_avg * 1.05:
            return 'improving'
        elif recent_avg < older_avg * 0.95:
            return 'declining'
        else:
            return 'stable'
    
    def _validate_and_sanitize_input(self, input_data: Any) -> Tuple[Any, str]:
        """
        Validate and sanitize input data.
        
        Args:
            input_data: Raw input data
            
        Returns:
            Tuple of (validated_input, input_type)
            
        Raises:
            ValueError: If input is invalid
        """
        if input_data is None:
            raise ValueError("Input data cannot be None")
        
        # Handle numpy arrays
        if isinstance(input_data, np.ndarray):
            if input_data.size == 0:
                raise ValueError("Empty numpy array provided")
            return input_data.copy(), 'numpy_array'
        
        # Handle dictionaries
        if isinstance(input_data, dict):
            if not input_data:
                raise ValueError("Empty dictionary provided")
            # Deep copy to prevent external modifications
            return json.loads(json.dumps(input_data, default=str)), 'dictionary'
        
        # Handle lists
        if isinstance(input_data, list):
            if not input_data:
                raise ValueError("Empty list provided")
            return input_data.copy(), 'list'
        
        # Handle strings
        if isinstance(input_data, str):
            if not input_data.strip():
                raise ValueError("Empty or whitespace-only string provided")
            return input_data.strip(), 'string'
        
        # Handle numeric types
        if isinstance(input_data, (int, float)):
            if not np.isfinite(input_data):
                raise ValueError("Non-finite numeric value provided")
            return input_data, 'numeric'
        
        # Handle other types by converting to string
        try:
            return str(input_data), 'converted_string'
        except Exception as e:
            raise ValueError(f"Cannot process input type {type(input_data)}: {e}")
    
    def _get_input_size(self, validated_input: Any) -> int:
        """Get the size of validated input for logging."""
        if isinstance(validated_input, np.ndarray):
            return validated_input.size
        elif isinstance(validated_input, (dict, list)):
            return len(validated_input)
        elif isinstance(validated_input, str):
            return len(validated_input)
        else:
            return 1
    
    def _generate_cache_key(self, validated_input: Any, domain: Optional[str]) -> str:
        """Generate cache key for prediction."""
        try:
            # Create a string representation for hashing
            if isinstance(validated_input, np.ndarray):
                input_str = f"array_{validated_input.shape}_{validated_input.dtype}_{hash(validated_input.data.tobytes())}"
            else:
                input_str = json.dumps(validated_input, sort_keys=True, default=str)
            
            domain_str = domain or 'general'
            return f"prediction_{domain_str}_{hash(input_str)}"
        except Exception:
            # Fallback to simple hash
            return f"prediction_{domain or 'general'}_{id(validated_input)}"
    
    def _get_cached_prediction(self, cache_key: str) -> Optional[PredictionResult]:
        """Get cached prediction if available."""
        if self._cache is None:
            return None
        
        if self._cache_lock:
            with self._cache_lock:
                return self._cache.get(cache_key)
        else:
            return self._cache.get(cache_key)
    
    def _cache_prediction(self, cache_key: str, result: PredictionResult) -> None:
        """Cache prediction result."""
        if self._cache is None:
            return
        
        # Manage cache size
        if self._cache_lock:
            with self._cache_lock:
                if len(self._cache) >= self.config.cache_size:
                    # Remove oldest entries (simple FIFO)
                    oldest_keys = list(self._cache.keys())[:len(self._cache) - self.config.cache_size + 1]
                    for key in oldest_keys:
                        del self._cache[key]
                
                self._cache[cache_key] = result
        else:
            if len(self._cache) >= self.config.cache_size:
                oldest_keys = list(self._cache.keys())[:len(self._cache) - self.config.cache_size + 1]
                for key in oldest_keys:
                    del self._cache[key]
            
            self._cache[cache_key] = result
    
    def _prepare_prediction_context(self, validated_input: Any, domain: Optional[str]) -> Dict[str, Any]:
        """Prepare context from shared state for prediction."""
        if self._state_lock:
            with self._state_lock:
                knowledge_base = dict(self._shared_state['knowledge_base'])
                reasoning_patterns = self._shared_state['reasoning_patterns'].copy()
                insights = self._shared_state['cross_domain_insights'][-10:]  # Last 10 insights
        else:
            knowledge_base = dict(self._shared_state['knowledge_base'])
            reasoning_patterns = self._shared_state['reasoning_patterns'].copy()
            insights = self._shared_state['cross_domain_insights'][-10:]
        
        context = {
            'available_domains': list(knowledge_base.keys()),
            'relevant_patterns': reasoning_patterns,
            'recent_insights': insights,
            'domain_knowledge': knowledge_base.get(domain, {}) if domain else {},
            'input_characteristics': self._analyze_input_characteristics(validated_input)
        }
        
        return context
    
    def _analyze_input_characteristics(self, validated_input: Any) -> Dict[str, Any]:
        """Analyze characteristics of the input for better prediction."""
        characteristics = {
            'type': type(validated_input).__name__,
            'complexity': 'simple'
        }
        
        if isinstance(validated_input, np.ndarray):
            characteristics.update({
                'shape': validated_input.shape,
                'dtype': str(validated_input.dtype),
                'dimensions': validated_input.ndim,
                'complexity': 'complex' if validated_input.ndim > 2 else 'moderate'
            })
        elif isinstance(validated_input, dict):
            characteristics.update({
                'keys': list(validated_input.keys()),
                'depth': self._get_dict_depth(validated_input),
                'complexity': 'complex' if len(validated_input) > 10 else 'moderate'
            })
        elif isinstance(validated_input, list):
            characteristics.update({
                'length': len(validated_input),
                'homogeneous': len(set(type(x) for x in validated_input)) == 1,
                'complexity': 'complex' if len(validated_input) > 100 else 'moderate'
            })
        
        return characteristics
    
    def _get_dict_depth(self, d: dict, current_depth: int = 0) -> int:
        """Get the maximum depth of a nested dictionary."""
        if not isinstance(d, dict) or not d:
            return current_depth
        
        return max(self._get_dict_depth(v, current_depth + 1) 
                  for v in d.values() 
                  if isinstance(v, dict)) or current_depth + 1
    
    def _select_domain(self, validated_input: Any, requested_domain: Optional[str], context: Dict[str, Any]) -> Optional[str]:
        """Select the most appropriate domain for prediction."""
        # If domain explicitly requested and available, use it
        if requested_domain and requested_domain in context['available_domains']:
            return requested_domain
        
        # For now, return None to use general prediction
        # This will be enhanced when domain lobes are implemented
        return None
    
    def _domain_specific_prediction(self, validated_input: Any, domain: str, context: Dict[str, Any]) -> Tuple[Any, float, Dict[str, float]]:
        """
        Placeholder for domain-specific prediction.
        Will be implemented when domain lobes are added.
        """
        # For now, fall back to general prediction
        return self._general_prediction(validated_input, context)
    
    def _general_prediction(self, validated_input: Any, context: Dict[str, Any]) -> Tuple[Any, float, Dict[str, float]]:
        """
        Make a general prediction using shared knowledge.
        """
        # Extract features based on input type
        if isinstance(validated_input, np.ndarray):
            # Numerical prediction
            features = {
                'mean': np.mean(validated_input),
                'std': np.std(validated_input),
                'min': np.min(validated_input),
                'max': np.max(validated_input)
            }
            
            # Simple prediction based on statistics
            prediction_value = features['mean'] + 0.1 * features['std']
            base_confidence = 0.7
            
        elif isinstance(validated_input, dict):
            # Dictionary-based prediction
            num_keys = len(validated_input)
            depth = context['input_characteristics']['depth']
            
            # Generate prediction based on structure
            prediction_value = {
                'analyzed_keys': num_keys,
                'structure_depth': depth,
                'prediction_type': 'structural_analysis'
            }
            base_confidence = min(0.8, 0.5 + 0.1 * num_keys)
            
        elif isinstance(validated_input, list):
            # List-based prediction
            if context['input_characteristics']['homogeneous'] and validated_input:
                if all(isinstance(x, (int, float)) for x in validated_input):
                    prediction_value = np.median(validated_input)
                else:
                    prediction_value = {'list_length': len(validated_input), 'type': 'heterogeneous'}
            else:
                prediction_value = {'list_length': len(validated_input), 'type': 'mixed'}
            base_confidence = 0.6
            
        elif isinstance(validated_input, str):
            # String-based prediction
            prediction_value = {
                'length': len(validated_input),
                'words': len(validated_input.split()),
                'prediction_type': 'text_analysis'
            }
            base_confidence = 0.65
            
        else:
            # Numeric or other simple types
            prediction_value = validated_input * 1.1 if isinstance(validated_input, (int, float)) else str(validated_input)
            base_confidence = 0.5
        
        # Calculate uncertainty metrics using internal method
        uncertainty_metrics = self._uncertainty_metrics(validated_input)
        
        # Adjust confidence based on available knowledge
        if context['domain_knowledge']:
            base_confidence += 0.1
            uncertainty_metrics['epistemic'] = uncertainty_metrics.get('model_uncertainty', 0.2) * 0.8
        else:
            uncertainty_metrics['epistemic'] = uncertainty_metrics.get('model_uncertainty', 0.2)
        
        if len(context['relevant_patterns']) > 0:
            base_confidence += 0.05
            uncertainty_metrics['total'] = uncertainty_metrics.get('total_uncertainty', 0.3) * 0.9
        
        uncertainty_metrics['aleatoric'] = uncertainty_metrics.get('data_uncertainty', 0.1)
        
        return prediction_value, min(base_confidence, 0.95), uncertainty_metrics
    
    def _apply_reasoning_patterns(self, prediction_value: Any, context: Dict[str, Any], reasoning_steps: List[str]) -> Any:
        """Apply reasoning patterns to enhance prediction."""
        # Check if any patterns apply
        applied_patterns = []
        enhanced_value = prediction_value
        
        for pattern_name, pattern_data in context['relevant_patterns'].items():
            if self._pattern_applies(prediction_value, pattern_data):
                enhanced_value = self._apply_pattern(enhanced_value, pattern_data)
                applied_patterns.append(pattern_name)
        
        if applied_patterns:
            reasoning_steps.append(f"Applied patterns: {', '.join(applied_patterns)}")
        
        return enhanced_value
    
    def _pattern_applies(self, value: Any, pattern_data: Dict[str, Any]) -> bool:
        """Check if a reasoning pattern applies to current prediction."""
        # Simple heuristic for now
        if 'value_type' in pattern_data:
            return type(value).__name__ == pattern_data['value_type']
        return False
    
    def _apply_pattern(self, value: Any, pattern_data: Dict[str, Any]) -> Any:
        """Apply a specific reasoning pattern."""
        # Simple transformation for now
        if 'transformation' in pattern_data:
            if pattern_data['transformation'] == 'scale' and isinstance(value, (int, float)):
                return value * pattern_data.get('factor', 1.0)
        return value
    
    def _calculate_final_confidence(self, base_confidence: float, uncertainty: Dict[str, float], context: Dict[str, Any]) -> float:
        """Calculate final confidence score considering all factors."""
        # Start with base confidence
        final_confidence = base_confidence
        
        # Adjust for uncertainty
        total_uncertainty = uncertainty.get('total', 0.3)
        final_confidence *= (1 - total_uncertainty)
        
        # Boost for recent successful insights
        if context['recent_insights']:
            insight_boost = min(0.1, len(context['recent_insights']) * 0.01)
            final_confidence = min(0.99, final_confidence + insight_boost)
        
        # Ensure bounds
        return max(0.01, min(0.99, final_confidence))
    
    def _update_cross_domain_insights(self, result: PredictionResult, context: Dict[str, Any]) -> None:
        """Update cross-domain insights based on prediction result."""
        if not result.success:
            return
        
        insight = {
            'timestamp': result.timestamp.isoformat(),
            'domain': result.domain,
            'confidence': result.confidence,
            'input_type': result.metadata.get('input_type'),
            'reasoning_depth': len(result.reasoning_steps),
            'uncertainty': result.uncertainty_metrics.to_dict() if result.uncertainty_metrics else None
        }
        
        if self._state_lock:
            with self._state_lock:
                self._shared_state['cross_domain_insights'].append(insight)
                # Limit insights
                max_insights = self.config.shared_memory_size // 10
                if len(self._shared_state['cross_domain_insights']) > max_insights:
                    self._shared_state['cross_domain_insights'] = self._shared_state['cross_domain_insights'][-max_insights:]
        else:
            self._shared_state['cross_domain_insights'].append(insight)
            max_insights = self.config.shared_memory_size // 10
            if len(self._shared_state['cross_domain_insights']) > max_insights:
                self._shared_state['cross_domain_insights'] = self._shared_state['cross_domain_insights'][-max_insights:]
    
    def get_shared_state(self) -> Dict[str, Any]:
        """
        Get current shared knowledge state.
        
        Returns:
            Dict containing shared state across all domains
        """
        if self._state_lock:
            with self._state_lock:
                # Return deep copy to prevent external modifications
                return self._deep_copy_state(self._shared_state)
        else:
            return self._deep_copy_state(self._shared_state)
    
    def set_shared_state(self, state: Dict[str, Any]) -> None:
        """
        Update shared knowledge state.
        
        Args:
            state: New state dictionary to merge with existing state
            
        Raises:
            ValueError: If state is invalid
            TypeError: If state is not a dictionary
        """
        # Validate input
        if not isinstance(state, dict):
            raise TypeError(f"State must be a dictionary, got {type(state)}")
        
        if not state:
            raise ValueError("Cannot set empty state")
        
        # Validate state structure
        self._validate_state_structure(state)
        
        # Update state with thread safety
        if self._state_lock:
            with self._state_lock:
                self._merge_state(state)
        else:
            self._merge_state(state)
        
        # Clear cache on state update
        if self._cache is not None:
            if self._cache_lock:
                with self._cache_lock:
                    self._cache.clear()
            else:
                self._cache.clear()
        
        # Persist if configured
        if self.config.knowledge_persistence and self.config.knowledge_path:
            self._persist_knowledge()
        
        self.logger.debug("Shared state updated successfully")
    
    def _validate_state_structure(self, state: Dict[str, Any]) -> None:
        """
        Validate state structure before merging.
        
        Args:
            state: State dictionary to validate
            
        Raises:
            ValueError: If state structure is invalid
        """
        # Check for required top-level keys
        valid_keys = {'knowledge_base', 'reasoning_patterns', 'cross_domain_insights', 'metadata'}
        
        for key in state:
            if key not in valid_keys:
                self.logger.warning(f"Unknown state key '{key}' will be ignored")
        
        # Validate knowledge_base structure
        if 'knowledge_base' in state:
            if not isinstance(state['knowledge_base'], dict):
                raise ValueError("knowledge_base must be a dictionary")
            
            # Ensure it can be used as defaultdict
            for domain, knowledge in state['knowledge_base'].items():
                if not isinstance(knowledge, dict):
                    raise ValueError(f"knowledge_base['{domain}'] must be a dictionary")
        
        # Validate reasoning_patterns
        if 'reasoning_patterns' in state:
            if not isinstance(state['reasoning_patterns'], dict):
                raise ValueError("reasoning_patterns must be a dictionary")
        
        # Validate cross_domain_insights
        if 'cross_domain_insights' in state:
            if not isinstance(state['cross_domain_insights'], list):
                raise ValueError("cross_domain_insights must be a list")
        
        # Validate metadata
        if 'metadata' in state:
            if not isinstance(state['metadata'], dict):
                raise ValueError("metadata must be a dictionary")
    
    def _merge_state(self, new_state: Dict[str, Any]) -> None:
        """
        Merge new state with existing state.
        
        Args:
            new_state: State to merge
        """
        # Update timestamp
        self._shared_state['metadata']['last_updated'] = datetime.now().isoformat()
        
        # Merge knowledge_base
        if 'knowledge_base' in new_state:
            for domain, knowledge in new_state['knowledge_base'].items():
                self._shared_state['knowledge_base'][domain].update(knowledge)
                
                # Update domain count
                self._shared_state['metadata']['total_domains'] = len(self._shared_state['knowledge_base'])
        
        # Merge reasoning_patterns
        if 'reasoning_patterns' in new_state:
            self._shared_state['reasoning_patterns'].update(new_state['reasoning_patterns'])
        
        # Extend cross_domain_insights
        if 'cross_domain_insights' in new_state:
            # Limit insights to prevent unbounded growth
            max_insights = self.config.shared_memory_size // 10
            self._shared_state['cross_domain_insights'].extend(new_state['cross_domain_insights'])
            
            # Keep only the most recent insights
            if len(self._shared_state['cross_domain_insights']) > max_insights:
                self._shared_state['cross_domain_insights'] = self._shared_state['cross_domain_insights'][-max_insights:]
        
        # Update metadata (selective merge)
        if 'metadata' in new_state:
            # Preserve critical metadata
            preserved_keys = {'created_at', 'version'}
            for key, value in new_state['metadata'].items():
                if key not in preserved_keys:
                    self._shared_state['metadata'][key] = value
    
    def _deep_copy_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a deep copy of the state.
        
        Args:
            state: State to copy
            
        Returns:
            Deep copy of the state
        """
        # Use JSON serialization for deep copy to handle complex structures
        try:
            return json.loads(json.dumps(state, default=str))
        except (TypeError, ValueError) as e:
            self.logger.warning(f"Failed to deep copy state using JSON: {e}")
            
            # Fallback to manual copy
            return {
                'knowledge_base': dict(state.get('knowledge_base', {})),
                'reasoning_patterns': dict(state.get('reasoning_patterns', {})),
                'cross_domain_insights': list(state.get('cross_domain_insights', [])),
                'metadata': dict(state.get('metadata', {}))
            }
    
    def _persist_knowledge(self) -> None:
        """Persist current knowledge state to disk."""
        if not self.config.knowledge_path:
            return
        
        try:
            # Create backup of existing file
            if self.config.knowledge_path.exists():
                backup_path = self.config.knowledge_path.with_suffix('.backup')
                self.config.knowledge_path.rename(backup_path)
            
            # Write current state
            with open(self.config.knowledge_path, 'w') as f:
                json.dump(self._shared_state, f, indent=2, default=str)
            
            # Remove backup on success
            backup_path = self.config.knowledge_path.with_suffix('.backup')
            if backup_path.exists():
                backup_path.unlink()
            
            self.logger.debug(f"Knowledge persisted to {self.config.knowledge_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to persist knowledge: {e}")
            
            # Restore backup if it exists
            backup_path = self.config.knowledge_path.with_suffix('.backup')
            if backup_path.exists():
                backup_path.rename(self.config.knowledge_path)
    
    def _load_persisted_knowledge(self) -> None:
        """Load persisted knowledge from disk."""
        if not self.config.knowledge_path or not self.config.knowledge_path.exists():
            return
        
        try:
            with open(self.config.knowledge_path, 'r') as f:
                loaded_state = json.load(f)
            
            # Validate loaded state
            self._validate_state_structure(loaded_state)
            
            # Convert knowledge_base back to defaultdict
            if 'knowledge_base' in loaded_state:
                loaded_state['knowledge_base'] = defaultdict(dict, loaded_state['knowledge_base'])
            
            # Merge with current state (in case of partial load)
            self._merge_state(loaded_state)
            
            self.logger.info(f"Loaded persisted knowledge from {self.config.knowledge_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load persisted knowledge: {e}")
            
            # Continue with empty state on failure
            self.logger.warning("Starting with fresh knowledge state")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the Brain Core state.
        
        Returns:
            Dictionary with state statistics
        """
        if self._state_lock:
            with self._state_lock:
                stats = self._calculate_statistics()
        else:
            stats = self._calculate_statistics()
        
        # Add uncertainty statistics
        stats['uncertainty_stats'] = self.get_uncertainty_statistics()
        
        return stats
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate statistics about current state."""
        knowledge_base = self._shared_state.get('knowledge_base', {})
        
        stats = {
            'total_domains': len(knowledge_base),
            'total_knowledge_items': sum(len(domain_knowledge) for domain_knowledge in knowledge_base.values()),
            'reasoning_patterns_count': len(self._shared_state.get('reasoning_patterns', {})),
            'cross_domain_insights_count': len(self._shared_state.get('cross_domain_insights', [])),
            'metadata': self._shared_state.get('metadata', {}),
            'cache_size': len(self._cache) if self._cache is not None else 0,
            'memory_estimate_mb': self._estimate_memory_usage() / (1024 * 1024)
        }
        
        # Domain-specific stats
        if knowledge_base:
            stats['domains'] = {
                domain: len(knowledge) 
                for domain, knowledge in knowledge_base.items()
            }
        
        return stats
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of shared state in bytes."""
        try:
            # Serialize to estimate size
            serialized = json.dumps(self._shared_state, default=str)
            return len(serialized.encode('utf-8'))
        except:
            # Fallback estimate
            return 0.0
    
    def clear_cache(self) -> None:
        """Clear the internal cache if caching is enabled."""
        if self._cache is not None:
            if self._cache_lock:
                with self._cache_lock:
                    self._cache.clear()
            else:
                self._cache.clear()
            
            self.logger.debug("Cache cleared")
    
    def add_shared_knowledge(self, key: str, value: Any, domain: str = "general", 
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add knowledge to the shared knowledge base with indexing.
        
        Args:
            key: Unique identifier for the knowledge
            value: The knowledge value (can be any type)
            domain: Domain to store the knowledge in
            metadata: Optional metadata about the knowledge
            
        Returns:
            True if knowledge was added successfully, False otherwise
            
        Raises:
            ValueError: If key or value is invalid
            TypeError: If parameters have wrong types
        """
        # Validate inputs
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Key must be a non-empty string")
        
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError("Domain must be a non-empty string")
        
        if value is None:
            raise ValueError("Value cannot be None")
        
        # Clean key and domain
        key = key.strip()
        domain = domain.strip()
        
        try:
            # Create knowledge entry
            knowledge_entry = {
                'value': value,
                'domain': domain,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {},
                'access_count': 0,
                'last_accessed': None
            }
            
            # Validate knowledge entry
            if not self._validate_knowledge_entry(key, knowledge_entry):
                return False
            
            # Thread-safe knowledge addition
            if self._state_lock:
                with self._state_lock:
                    success = self._add_knowledge_with_indexing(key, knowledge_entry, domain)
            else:
                success = self._add_knowledge_with_indexing(key, knowledge_entry, domain)
            
            if success:
                self.logger.debug(f"Added shared knowledge: {key} in domain {domain}")
                
                # Clear cache since knowledge base changed
                if self._cache is not None:
                    if self._cache_lock:
                        with self._cache_lock:
                            self._cache.clear()
                    else:
                        self._cache.clear()
                
                # Persist if configured
                if self.config.knowledge_persistence and self.config.knowledge_path:
                    self._persist_knowledge()
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to add shared knowledge {key}: {e}")
            return False
    
    def get_shared_knowledge(self, key: str, domain: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve shared knowledge by key.
        
        Args:
            key: Knowledge identifier
            domain: Optional domain to search in. If None, searches all domains
            
        Returns:
            Knowledge entry with metadata, or None if not found
        """
        if not isinstance(key, str) or not key.strip():
            return None
        
        key = key.strip()
        
        try:
            if self._state_lock:
                with self._state_lock:
                    return self._get_knowledge_from_store(key, domain)
            else:
                return self._get_knowledge_from_store(key, domain)
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve shared knowledge {key}: {e}")
            return None
    
    def search_shared_knowledge(self, query: str, domain: Optional[str] = None, 
                              max_results: int = 10, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search shared knowledge using text similarity and fuzzy matching.
        
        Args:
            query: Search query string
            domain: Optional domain to search in
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of matching knowledge entries with similarity scores
        """
        if not isinstance(query, str) or not query.strip():
            return []
        
        if max_results <= 0:
            max_results = 10
        
        if not (0.0 <= similarity_threshold <= 1.0):
            similarity_threshold = 0.3
        
        query = query.strip().lower()
        
        try:
            if self._state_lock:
                with self._state_lock:
                    return self._search_knowledge_store(query, domain, max_results, similarity_threshold)
            else:
                return self._search_knowledge_store(query, domain, max_results, similarity_threshold)
                
        except Exception as e:
            self.logger.error(f"Failed to search shared knowledge with query '{query}': {e}")
            return []
    
    def _validate_knowledge_entry(self, key: str, entry: Dict[str, Any]) -> bool:
        """
        Validate a knowledge entry before adding to the store.
        
        Args:
            key: Knowledge key
            entry: Knowledge entry to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = ['value', 'domain', 'timestamp', 'metadata']
            for field in required_fields:
                if field not in entry:
                    self.logger.warning(f"Knowledge entry {key} missing required field: {field}")
                    return False
            
            # Validate field types
            if not isinstance(entry['domain'], str):
                self.logger.warning(f"Knowledge entry {key} has invalid domain type")
                return False
            
            if not isinstance(entry['metadata'], dict):
                self.logger.warning(f"Knowledge entry {key} has invalid metadata type")
                return False
            
            # Check size constraints
            entry_size = len(json.dumps(entry, default=str))
            max_entry_size = 1024 * 1024  # 1MB per entry
            if entry_size > max_entry_size:
                self.logger.warning(f"Knowledge entry {key} exceeds size limit: {entry_size} bytes")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating knowledge entry {key}: {e}")
            return False
    
    def _add_knowledge_with_indexing(self, key: str, entry: Dict[str, Any], domain: str) -> bool:
        """
        Add knowledge with proper indexing for efficient retrieval.
        
        Args:
            key: Knowledge key
            entry: Knowledge entry
            domain: Domain name
            
        Returns:
            True if added successfully
        """
        try:
            # Add to main knowledge base
            self._shared_state['knowledge_base'][domain][key] = entry
            
            # Update indices
            self._knowledge_index[key].add(domain)
            
            # Index searchable text from value and metadata
            searchable_text = self._extract_searchable_text(entry)
            search_terms = self._tokenize_for_search(searchable_text)
            
            for term in search_terms:
                self._text_search_index[term].add(key)
            
            # Update relationships based on similar keys or content
            related_keys = self._find_related_keys(key, entry)
            for related_key in related_keys:
                self._knowledge_relationships[key].add(related_key)
                self._knowledge_relationships[related_key].add(key)
            
            # Update metadata
            self._shared_state['metadata']['last_updated'] = datetime.now().isoformat()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding knowledge with indexing for {key}: {e}")
            return False
    
    def _get_knowledge_from_store(self, key: str, domain: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Retrieve knowledge from the store with access tracking.
        
        Args:
            key: Knowledge key
            domain: Optional domain filter
            
        Returns:
            Knowledge entry or None
        """
        try:
            # Search in specific domain if provided
            if domain:
                knowledge_base = self._shared_state['knowledge_base'].get(domain, {})
                if key in knowledge_base:
                    entry = knowledge_base[key].copy()
                    # Update access tracking
                    knowledge_base[key]['access_count'] += 1
                    knowledge_base[key]['last_accessed'] = datetime.now().isoformat()
                    return entry
                return None
            
            # Search across all domains
            for domain_name, domain_knowledge in self._shared_state['knowledge_base'].items():
                if key in domain_knowledge:
                    entry = domain_knowledge[key].copy()
                    # Update access tracking
                    domain_knowledge[key]['access_count'] += 1
                    domain_knowledge[key]['last_accessed'] = datetime.now().isoformat()
                    return entry
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving knowledge {key}: {e}")
            return None
    
    def _search_knowledge_store(self, query: str, domain: Optional[str], max_results: int, 
                              similarity_threshold: float) -> List[Dict[str, Any]]:
        """
        Search knowledge store with similarity scoring.
        
        Args:
            query: Search query
            domain: Optional domain filter
            max_results: Maximum results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of matching knowledge entries with scores
        """
        try:
            results = []
            query_terms = self._tokenize_for_search(query)
            
            # Get candidate keys from text index
            candidate_keys = set()
            for term in query_terms:
                if term in self._text_search_index:
                    candidate_keys.update(self._text_search_index[term])
            
            # Also check for partial matches in keys
            all_keys = set()
            for domain_name, domain_knowledge in self._shared_state['knowledge_base'].items():
                if domain is None or domain_name == domain:
                    all_keys.update(domain_knowledge.keys())
            
            # Add keys that partially match the query
            for key in all_keys:
                if any(term in key.lower() for term in query_terms):
                    candidate_keys.add(key)
            
            # Score and rank candidates
            for key in candidate_keys:
                entry = self.get_shared_knowledge(key, domain)
                if entry:
                    similarity_score = self._calculate_similarity_score(query, key, entry)
                    
                    if similarity_score >= similarity_threshold:
                        result_entry = entry.copy()
                        result_entry['similarity_score'] = similarity_score
                        result_entry['key'] = key
                        results.append(result_entry)
            
            # Sort by similarity score (descending) and limit results
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return results[:max_results]
            
        except Exception as e:
            self.logger.error(f"Error searching knowledge store: {e}")
            return []
    
    def _extract_searchable_text(self, entry: Dict[str, Any]) -> str:
        """
        Extract searchable text from a knowledge entry.
        
        Args:
            entry: Knowledge entry
            
        Returns:
            Searchable text string
        """
        text_parts = []
        
        # Extract from value
        value = entry.get('value', '')
        if isinstance(value, str):
            text_parts.append(value)
        elif isinstance(value, dict):
            text_parts.extend(str(v) for v in value.values() if isinstance(v, (str, int, float)))
        elif isinstance(value, list):
            text_parts.extend(str(item) for item in value if isinstance(item, (str, int, float)))
        else:
            text_parts.append(str(value))
        
        # Extract from metadata
        metadata = entry.get('metadata', {})
        if isinstance(metadata, dict):
            for key, val in metadata.items():
                if isinstance(val, (str, int, float)):
                    text_parts.append(f"{key}:{val}")
        
        return ' '.join(text_parts)
    
    def _tokenize_for_search(self, text: str) -> Set[str]:
        """
        Tokenize text for search indexing.
        
        Args:
            text: Text to tokenize
            
        Returns:
            Set of search terms
        """
        if not text:
            return set()
        
        # Simple tokenization - split on whitespace and punctuation, convert to lowercase
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out very short tokens and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        filtered_tokens = {token for token in tokens if len(token) >= 2 and token not in stop_words}
        
        return filtered_tokens
    
    def _find_related_keys(self, key: str, entry: Dict[str, Any]) -> Set[str]:
        """
        Find keys related to the given key and entry.
        
        Args:
            key: Knowledge key
            entry: Knowledge entry
            
        Returns:
            Set of related keys
        """
        related_keys = set()
        
        try:
            # Find keys with similar names
            key_lower = key.lower()
            for domain_knowledge in self._shared_state['knowledge_base'].values():
                for existing_key in domain_knowledge.keys():
                    if existing_key != key:
                        # Simple similarity check
                        if self._calculate_key_similarity(key_lower, existing_key.lower()) > 0.7:
                            related_keys.add(existing_key)
            
            # Limit to prevent explosion of relationships
            return set(list(related_keys)[:5])
            
        except Exception as e:
            self.logger.debug(f"Error finding related keys for {key}: {e}")
            return set()
    
    def _calculate_similarity_score(self, query: str, key: str, entry: Dict[str, Any]) -> float:
        """
        Calculate similarity score between query and knowledge entry.
        
        Args:
            query: Search query
            key: Knowledge key
            entry: Knowledge entry
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            scores = []
            
            # Key similarity
            key_score = self._calculate_key_similarity(query.lower(), key.lower())
            scores.append(key_score * 0.4)  # Weight key matching highly
            
            # Content similarity
            searchable_text = self._extract_searchable_text(entry).lower()
            content_score = self._calculate_text_similarity(query.lower(), searchable_text)
            scores.append(content_score * 0.6)  # Weight content matching
            
            # Boost score based on access patterns
            access_count = entry.get('access_count', 0)
            popularity_boost = min(0.1, access_count * 0.01)  # Small boost for popular items
            
            final_score = sum(scores) + popularity_boost
            return min(1.0, final_score)
            
        except Exception as e:
            self.logger.debug(f"Error calculating similarity score: {e}")
            return 0.0
    
    def _calculate_key_similarity(self, query: str, key: str) -> float:
        """Calculate similarity between query and key using fuzzy matching."""
        try:
            # Exact match
            if query == key:
                return 1.0
            
            # Substring match
            if query in key or key in query:
                return 0.8
            
            # Token overlap
            query_tokens = set(query.split())
            key_tokens = set(key.split())
            
            if query_tokens and key_tokens:
                overlap = len(query_tokens & key_tokens)
                union = len(query_tokens | key_tokens)
                return overlap / union if union > 0 else 0.0
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_text_similarity(self, query: str, text: str) -> float:
        """Calculate similarity between query and text content."""
        try:
            if not query or not text:
                return 0.0
            
            query_tokens = set(self._tokenize_for_search(query))
            text_tokens = set(self._tokenize_for_search(text))
            
            if not query_tokens or not text_tokens:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(query_tokens & text_tokens)
            union = len(query_tokens | text_tokens)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def save_state(self, filepath: Union[str, Path]) -> bool:
        """
        Save complete brain state to disk including all knowledge, uncertainty models, and configuration.
        
        Args:
            filepath: Path where to save the state file
            
        Returns:
            True if save was successful, False otherwise
        """
        filepath = Path(filepath)
        
        try:
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup if file exists
            backup_path = None
            if filepath.exists():
                backup_path = filepath.with_suffix(filepath.suffix + '.backup')
                filepath.rename(backup_path)
                self.logger.debug(f"Created backup at {backup_path}")
            
            # Serialize complete state
            serialized_state = self._serialize_state()
            
            # Write to file with atomic operation
            temp_path = filepath.with_suffix(filepath.suffix + '.tmp')
            with open(temp_path, 'w') as f:
                json.dump(serialized_state, f, indent=2, default=str)
            
            # Atomic rename
            temp_path.rename(filepath)
            
            # Remove backup on success
            if backup_path and backup_path.exists():
                backup_path.unlink()
            
            self.logger.info(f"Brain state saved successfully to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save brain state: {e}")
            
            # Restore backup if available
            if backup_path and backup_path.exists():
                try:
                    backup_path.rename(filepath)
                    self.logger.info("Restored backup after save failure")
                except Exception as restore_error:
                    self.logger.error(f"Failed to restore backup: {restore_error}")
            
            return False
    
    def load_state(self, filepath: Union[str, Path]) -> bool:
        """
        Load complete brain state from disk including all knowledge, uncertainty models, and configuration.
        
        Args:
            filepath: Path to the state file to load
            
        Returns:
            True if load was successful, False otherwise
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            self.logger.error(f"State file not found: {filepath}")
            return False
        
        try:
            # Load state from file
            with open(filepath, 'r') as f:
                serialized_state = json.load(f)
            
            # Validate state structure
            if not self._validate_serialized_state(serialized_state):
                self.logger.error("Invalid state file structure")
                return False
            
            # Create backup of current state before loading
            current_state_backup = self._serialize_state()
            
            try:
                # Deserialize and apply state
                self._deserialize_state(serialized_state)
                
                # Verify state was loaded correctly
                if self._verify_loaded_state(serialized_state):
                    self.logger.info(f"Brain state loaded successfully from {filepath}")
                    
                    # Clear cache after state load
                    self.clear_cache()
                    
                    return True
                else:
                    raise RuntimeError("State verification failed after loading")
                    
            except Exception as e:
                # Restore previous state on failure
                self.logger.error(f"Failed to deserialize state: {e}")
                self._deserialize_state(current_state_backup)
                return False
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in state file: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to load brain state: {e}")
            return False
    
    def update_config(self, config_dict: Dict[str, Any]) -> bool:
        """
        Update brain configuration dynamically.
        
        Args:
            config_dict: Dictionary with configuration updates
            
        Returns:
            True if update was successful, False otherwise
        """
        if not isinstance(config_dict, dict):
            self.logger.error(f"config_dict must be a dictionary, got {type(config_dict)}")
            return False
        
        try:
            # Create backup of current config
            current_config = self.get_config()
            
            # Validate new configuration values
            validated_updates = self._validate_config_updates(config_dict)
            
            # Apply updates
            for key, value in validated_updates.items():
                if hasattr(self.config, key):
                    old_value = getattr(self.config, key)
                    setattr(self.config, key, value)
                    self.logger.debug(f"Updated config.{key}: {old_value} -> {value}")
                    
                    # Handle specific config changes
                    self._handle_config_change(key, old_value, value)
                else:
                    self.logger.warning(f"Unknown configuration key: {key}")
            
            # Re-validate entire config
            try:
                self.config.__post_init__()
            except Exception as e:
                # Restore previous config on validation failure
                self.logger.error(f"Config validation failed: {e}")
                for key, value in current_config.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                return False
            
            # Update logging if log level changed
            if 'log_level' in validated_updates:
                self._setup_logging()
            
            self.logger.info(f"Configuration updated with {len(validated_updates)} changes")
            
            # Persist state if configured
            if self.config.knowledge_persistence and self.config.knowledge_path:
                # Save full state, not just knowledge
                state_path = self.config.knowledge_path.with_suffix('.state')
                self.save_state(state_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current brain configuration as a dictionary.
        
        Returns:
            Dictionary representation of current configuration
        """
        try:
            config_dict = {}
            
            # Get all configuration attributes
            for field_name in self.config.__dataclass_fields__:
                value = getattr(self.config, field_name)
                
                # Convert Path objects to strings
                if isinstance(value, Path):
                    config_dict[field_name] = str(value)
                else:
                    config_dict[field_name] = value
            
            return config_dict
            
        except Exception as e:
            self.logger.error(f"Failed to get configuration: {e}")
            return {}
    
    def _serialize_state(self) -> Dict[str, Any]:
        """
        Serialize complete brain state for persistence.
        
        Returns:
            Dictionary containing serialized brain state
        """
        try:
            # Prepare shared state for serialization
            shared_state_copy = self._deep_copy_state(self._shared_state)
            
            # Convert defaultdict to regular dict for JSON serialization
            if 'knowledge_base' in shared_state_copy:
                shared_state_copy['knowledge_base'] = dict(shared_state_copy['knowledge_base'])
            
            # Serialize state components
            state = {
                'version': '1.1.0',  # State format version
                'timestamp': datetime.now().isoformat(),
                'config': self.get_config(),
                'shared_state': shared_state_copy,
                'knowledge_index': {k: list(v) for k, v in self._knowledge_index.items()},
                'text_search_index': {k: list(v) for k, v in self._text_search_index.items()},
                'knowledge_relationships': {k: list(v) for k, v in self._knowledge_relationships.items()},
                'uncertainty_history': list(self._uncertainty_history),
                'prediction_accuracy_history': list(self._prediction_accuracy_history),
                'domain_confidence_scores': dict(self._domain_confidence_scores),
                'uncertainty_calibration_data': self._uncertainty_calibration_data.copy(),
                'statistics': {
                    'total_predictions': len(self._uncertainty_history),
                    'cache_size': len(self._cache) if self._cache is not None else 0,
                    'memory_estimate_mb': self._estimate_memory_usage() / (1024 * 1024)
                }
            }
            
            # Add cache state if caching is enabled
            if self.config.enable_caching and self._cache is not None:
                # Serialize only cache metadata, not actual cached results
                cache_keys = list(self._cache.keys()) if self._cache else []
                state['cache_metadata'] = {
                    'enabled': True,
                    'size': len(cache_keys),
                    'max_size': self.config.cache_size,
                    'keys_sample': cache_keys[:10]  # Sample for debugging
                }
            else:
                state['cache_metadata'] = {'enabled': False}
            
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to serialize state: {e}")
            raise
    
    def _deserialize_state(self, data: Dict[str, Any]) -> None:
        """
        Deserialize and apply brain state from persistence.
        
        Args:
            data: Dictionary containing serialized brain state
            
        Raises:
            ValueError: If state data is invalid
            RuntimeError: If state application fails
        """
        try:
            # Check version compatibility
            state_version = data.get('version', '1.0.0')
            if not self._is_compatible_version(state_version):
                raise ValueError(f"Incompatible state version: {state_version}")
            
            # Update configuration first
            if 'config' in data:
                config_dict = data['config']
                # Don't update knowledge_path from saved state to avoid conflicts
                if 'knowledge_path' in config_dict:
                    del config_dict['knowledge_path']
                self.update_config(config_dict)
            
            # Restore shared state
            if 'shared_state' in data:
                if self._state_lock:
                    with self._state_lock:
                        self._restore_shared_state(data['shared_state'])
                else:
                    self._restore_shared_state(data['shared_state'])
            
            # Restore indices
            if 'knowledge_index' in data:
                self._knowledge_index = defaultdict(set, {k: set(v) for k, v in data['knowledge_index'].items()})
            
            if 'text_search_index' in data:
                self._text_search_index = defaultdict(set, {k: set(v) for k, v in data['text_search_index'].items()})
            
            if 'knowledge_relationships' in data:
                self._knowledge_relationships = defaultdict(set, {k: set(v) for k, v in data['knowledge_relationships'].items()})
            
            # Restore uncertainty tracking
            if 'uncertainty_history' in data:
                self._uncertainty_history = deque(data['uncertainty_history'], maxlen=self.config.uncertainty_history_size)
            
            if 'prediction_accuracy_history' in data:
                self._prediction_accuracy_history = deque(data['prediction_accuracy_history'], maxlen=self.config.reliability_window)
            
            if 'domain_confidence_scores' in data:
                self._domain_confidence_scores = defaultdict(lambda: 0.5, data['domain_confidence_scores'])
            
            if 'uncertainty_calibration_data' in data:
                self._uncertainty_calibration_data = data['uncertainty_calibration_data'].copy()
            
            # Log restoration summary
            stats = data.get('statistics', {})
            self.logger.info(f"State restored: {stats.get('total_predictions', 0)} predictions, "
                           f"{len(self._shared_state['knowledge_base'])} domains")
            
        except Exception as e:
            self.logger.error(f"Failed to deserialize state: {e}")
            raise
    
    def _validate_serialized_state(self, state: Dict[str, Any]) -> bool:
        """
        Validate structure of serialized state.
        
        Args:
            state: Serialized state dictionary
            
        Returns:
            True if state structure is valid
        """
        try:
            # Check required top-level keys
            required_keys = {'version', 'timestamp', 'shared_state'}
            if not all(key in state for key in required_keys):
                self.logger.error("Missing required keys in state")
                return False
            
            # Validate shared state structure
            if 'shared_state' in state:
                shared_state = state['shared_state']
                if not isinstance(shared_state, dict):
                    self.logger.error("Invalid shared_state type")
                    return False
                
                # Check shared state components
                if 'knowledge_base' not in shared_state:
                    self.logger.error("Missing knowledge_base in shared_state")
                    return False
            
            # Validate version format
            version = state.get('version', '')
            if not re.match(r'^\d+\.\d+\.\d+$', version):
                self.logger.error(f"Invalid version format: {version}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating state structure: {e}")
            return False
    
    def _validate_config_updates(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration updates before applying.
        
        Args:
            config_dict: Configuration updates
            
        Returns:
            Validated configuration updates
            
        Raises:
            ValueError: If configuration values are invalid
        """
        validated = {}
        
        for key, value in config_dict.items():
            # Skip unknown keys
            if not hasattr(self.config, key):
                continue
            
            # Type validation
            expected_type = type(getattr(self.config, key))
            
            # Handle Path type
            if expected_type == Path and isinstance(value, str):
                validated[key] = Path(value)
            elif expected_type == bool and isinstance(value, str):
                # Handle string booleans
                validated[key] = value.lower() in ('true', '1', 'yes', 'on')
            elif expected_type in (int, float) and isinstance(value, str):
                # Handle numeric strings
                try:
                    validated[key] = expected_type(value)
                except ValueError:
                    self.logger.warning(f"Invalid {expected_type.__name__} value for {key}: {value}")
                    continue
            else:
                # Direct assignment for matching types
                if isinstance(value, expected_type):
                    validated[key] = value
                else:
                    self.logger.warning(f"Type mismatch for {key}: expected {expected_type}, got {type(value)}")
                    continue
            
            # Value range validation
            if key == 'shared_memory_size' and validated[key] <= 0:
                raise ValueError(f"{key} must be positive")
            elif key == 'reasoning_depth' and validated[key] <= 0:
                raise ValueError(f"{key} must be positive")
            elif key == 'confidence_threshold' and not (0.0 <= validated[key] <= 1.0):
                raise ValueError(f"{key} must be between 0 and 1")
            elif key == 'log_level' and validated[key] not in ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'):
                raise ValueError(f"Invalid log level: {validated[key]}")
        
        return validated
    
    def _handle_config_change(self, key: str, old_value: Any, new_value: Any) -> None:
        """
        Handle specific configuration changes that require additional actions.
        
        Args:
            key: Configuration key that changed
            old_value: Previous value
            new_value: New value
        """
        # Handle uncertainty history size change
        if key == 'uncertainty_history_size':
            # Resize uncertainty history deque
            history = list(self._uncertainty_history)
            self._uncertainty_history = deque(history, maxlen=new_value)
            self.logger.debug(f"Resized uncertainty history to {new_value}")
        
        elif key == 'reliability_window':
            # Resize prediction accuracy history
            history = list(self._prediction_accuracy_history)
            self._prediction_accuracy_history = deque(history, maxlen=new_value)
            self.logger.debug(f"Resized prediction accuracy history to {new_value}")
        
        elif key == 'cache_size':
            # Clear cache if it exceeds new size
            if self._cache is not None and len(self._cache) > new_value:
                self.clear_cache()
                self.logger.debug(f"Cleared cache due to size reduction")
        
        elif key == 'enable_caching':
            if new_value and self._cache is None:
                # Enable caching
                self._cache = {}
                if self.config.thread_safe:
                    self._cache_lock = threading.Lock()
                self.logger.debug("Enabled caching")
            elif not new_value and self._cache is not None:
                # Disable caching
                self._cache = None
                self._cache_lock = None
                self.logger.debug("Disabled caching")
        
        elif key == 'thread_safe':
            if new_value and not old_value:
                # Enable thread safety
                self._state_lock = threading.RLock()
                if self._cache is not None:
                    self._cache_lock = threading.Lock()
                self.logger.debug("Enabled thread safety")
            elif not new_value and old_value:
                # Disable thread safety (use with caution)
                self._state_lock = None
                self._cache_lock = None
                self.logger.warning("Disabled thread safety - use with caution in multi-threaded environments")
    
    def _restore_shared_state(self, shared_state: Dict[str, Any]) -> None:
        """
        Restore shared state from deserialized data.
        
        Args:
            shared_state: Shared state dictionary
        """
        # Convert knowledge_base back to defaultdict
        if 'knowledge_base' in shared_state:
            shared_state['knowledge_base'] = defaultdict(dict, shared_state['knowledge_base'])
        
        # Restore with proper structure
        self._shared_state = {
            'knowledge_base': shared_state.get('knowledge_base', defaultdict(dict)),
            'reasoning_patterns': shared_state.get('reasoning_patterns', {}),
            'cross_domain_insights': shared_state.get('cross_domain_insights', []),
            'metadata': shared_state.get('metadata', {
                'created_at': datetime.now().isoformat(),
                'version': '1.0.0',
                'total_domains': 0,
                'last_updated': datetime.now().isoformat()
            })
        }
        
        # Update metadata
        self._shared_state['metadata']['last_updated'] = datetime.now().isoformat()
        self._shared_state['metadata']['total_domains'] = len(self._shared_state['knowledge_base'])
    
    def _is_compatible_version(self, version: str) -> bool:
        """
        Check if state version is compatible with current implementation.
        
        Args:
            version: Version string (e.g., "1.0.0")
            
        Returns:
            True if compatible
        """
        try:
            major, minor, patch = map(int, version.split('.'))
            # Compatible with 1.x.x versions
            return major == 1
        except Exception:
            return False
    
    def _verify_loaded_state(self, original_state: Dict[str, Any]) -> bool:
        """
        Verify that state was loaded correctly.
        
        Args:
            original_state: Original state that was loaded
            
        Returns:
            True if verification passes
        """
        try:
            # Verify knowledge base
            if 'shared_state' in original_state:
                original_kb = original_state['shared_state'].get('knowledge_base', {})
                current_kb = self._shared_state.get('knowledge_base', {})
                
                if len(original_kb) != len(current_kb):
                    self.logger.error("Knowledge base size mismatch after loading")
                    return False
            
            # Verify uncertainty history
            if 'uncertainty_history' in original_state:
                if len(original_state['uncertainty_history']) != len(self._uncertainty_history):
                    self.logger.warning("Uncertainty history size adjusted due to configuration")
            
            return True
            
        except Exception as e:
            self.logger.error(f"State verification failed: {e}")
            return False
    
    def create_checkpoint(self, checkpoint_name: Optional[str] = None) -> Optional[Path]:
        """
        Create a checkpoint of current brain state.
        
        Args:
            checkpoint_name: Optional name for checkpoint. If None, timestamp is used.
            
        Returns:
            Path to checkpoint file if successful, None otherwise
        """
        try:
            # Determine checkpoint directory
            if self.config.knowledge_path:
                checkpoint_dir = self.config.knowledge_path.parent / "checkpoints"
            else:
                checkpoint_dir = Path("brain_checkpoints")
            
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate checkpoint filename
            if checkpoint_name:
                filename = f"checkpoint_{checkpoint_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            else:
                filename = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            checkpoint_path = checkpoint_dir / filename
            
            # Save state as checkpoint
            if self.save_state(checkpoint_path):
                self.logger.info(f"Checkpoint created: {checkpoint_path}")
                
                # Manage checkpoint retention (keep last 10 by default)
                self._cleanup_old_checkpoints(checkpoint_dir, keep_count=10)
                
                return checkpoint_path
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
            return None
    
    def restore_checkpoint(self, checkpoint_path: Union[str, Path]) -> bool:
        """
        Restore brain state from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if restoration was successful
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            self.logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            # Create backup of current state before restoration
            backup_state = self._serialize_state()
            
            # Load checkpoint
            if self.load_state(checkpoint_path):
                self.logger.info(f"Restored from checkpoint: {checkpoint_path}")
                return True
            else:
                # Restore backup if checkpoint load failed
                self._deserialize_state(backup_state)
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to restore checkpoint: {e}")
            return False
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List available checkpoints.
        
        Returns:
            List of checkpoint information dictionaries
        """
        try:
            # Determine checkpoint directory
            if self.config.knowledge_path:
                checkpoint_dir = self.config.knowledge_path.parent / "checkpoints"
            else:
                checkpoint_dir = Path("brain_checkpoints")
            
            if not checkpoint_dir.exists():
                return []
            
            checkpoints = []
            for checkpoint_file in checkpoint_dir.glob("checkpoint_*.json"):
                try:
                    # Extract checkpoint info
                    stat = checkpoint_file.stat()
                    checkpoints.append({
                        'name': checkpoint_file.name,
                        'path': str(checkpoint_file),
                        'size_mb': stat.st_size / (1024 * 1024),
                        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except Exception as e:
                    self.logger.debug(f"Error reading checkpoint {checkpoint_file}: {e}")
            
            # Sort by modification time (newest first)
            checkpoints.sort(key=lambda x: x['modified'], reverse=True)
            
            return checkpoints
            
        except Exception as e:
            self.logger.error(f"Failed to list checkpoints: {e}")
            return []
    
    def export_state(self, format: str = 'json', include_cache: bool = False) -> Union[str, bytes, None]:
        """
        Export brain state in specified format.
        
        Args:
            format: Export format ('json', 'pickle', 'yaml')
            include_cache: Whether to include cache in export
            
        Returns:
            Exported state data or None on failure
        """
        try:
            # Get serialized state
            state = self._serialize_state()
            
            # Optionally exclude cache
            if not include_cache and 'cache_metadata' in state:
                state['cache_metadata'] = {'enabled': False}
            
            if format == 'json':
                return json.dumps(state, indent=2, default=str)
                
            elif format == 'pickle':
                import pickle
                return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
                
            elif format == 'yaml':
                try:
                    import yaml
                    return yaml.dump(state, default_flow_style=False, allow_unicode=True)
                except ImportError:
                    self.logger.error("PyYAML not installed. Install with: pip install pyyaml")
                    return None
                    
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to export state in {format} format: {e}")
            return None
    
    def import_state(self, data: Union[str, bytes, Dict[str, Any]], format: str = 'json') -> bool:
        """
        Import brain state from specified format.
        
        Args:
            data: State data to import
            format: Import format ('json', 'pickle', 'yaml', 'dict')
            
        Returns:
            True if import was successful
        """
        try:
            # Parse data based on format
            if format == 'json':
                if isinstance(data, str):
                    state = json.loads(data)
                else:
                    raise ValueError("JSON data must be a string")
                    
            elif format == 'pickle':
                import pickle
                if isinstance(data, bytes):
                    state = pickle.loads(data)
                else:
                    raise ValueError("Pickle data must be bytes")
                    
            elif format == 'yaml':
                try:
                    import yaml
                    if isinstance(data, str):
                        state = yaml.safe_load(data)
                    else:
                        raise ValueError("YAML data must be a string")
                except ImportError:
                    self.logger.error("PyYAML not installed. Install with: pip install pyyaml")
                    return False
                    
            elif format == 'dict':
                if isinstance(data, dict):
                    state = data
                else:
                    raise ValueError("Dict format requires dictionary input")
                    
            else:
                self.logger.error(f"Unsupported import format: {format}")
                return False
            
            # Validate and apply state
            if not self._validate_serialized_state(state):
                return False
            
            # Create backup before import
            backup_state = self._serialize_state()
            
            try:
                self._deserialize_state(state)
                self.logger.info(f"State imported successfully from {format} format")
                return True
            except Exception as e:
                # Restore backup on failure
                self._deserialize_state(backup_state)
                raise
                
        except Exception as e:
            self.logger.error(f"Failed to import state from {format} format: {e}")
            return False
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current brain state without full serialization.
        
        Returns:
            Dictionary containing state summary
        """
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'uncertainty_enabled': self.config.enable_uncertainty,
                    'caching_enabled': self.config.enable_caching,
                    'persistence_enabled': self.config.knowledge_persistence,
                    'shared_memory_size': self.config.shared_memory_size
                },
                'knowledge': {
                    'total_domains': len(self._shared_state['knowledge_base']),
                    'total_items': sum(len(d) for d in self._shared_state['knowledge_base'].values()),
                    'domains': list(self._shared_state['knowledge_base'].keys())
                },
                'uncertainty': {
                    'history_size': len(self._uncertainty_history),
                    'tracked_domains': len(self._domain_confidence_scores),
                    'average_confidence': np.mean(list(self._domain_confidence_scores.values())) if self._domain_confidence_scores else 0.0
                },
                'performance': {
                    'cache_size': len(self._cache) if self._cache is not None else 0,
                    'memory_usage_mb': self._estimate_memory_usage() / (1024 * 1024),
                    'total_predictions': len(self._uncertainty_history)
                },
                'indices': {
                    'knowledge_index_size': len(self._knowledge_index),
                    'search_index_size': len(self._text_search_index),
                    'relationships_count': sum(len(v) for v in self._knowledge_relationships.values())
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate state summary: {e}")
            return {'error': str(e)}
    
    def _cleanup_old_checkpoints(self, checkpoint_dir: Path, keep_count: int = 10) -> None:
        """
        Clean up old checkpoints, keeping only the most recent ones.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            keep_count: Number of checkpoints to keep
        """
        try:
            # Get all checkpoint files
            checkpoints = list(checkpoint_dir.glob("checkpoint_*.json"))
            
            if len(checkpoints) <= keep_count:
                return
            
            # Sort by modification time
            checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old checkpoints
            for old_checkpoint in checkpoints[keep_count:]:
                try:
                    old_checkpoint.unlink()
                    self.logger.debug(f"Removed old checkpoint: {old_checkpoint.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove checkpoint {old_checkpoint.name}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup checkpoints: {e}")
    
    def reset_state(self, preserve_config: bool = True) -> None:
        """
        Reset brain to initial state.
        
        Args:
            preserve_config: Whether to preserve current configuration
        """
        try:
            # Save current config if requested
            current_config = self.get_config() if preserve_config else None
            
            # Create fresh state
            self._shared_state = {
                'knowledge_base': defaultdict(dict),
                'reasoning_patterns': {},
                'cross_domain_insights': [],
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'version': '1.0.0',
                    'total_domains': 0,
                    'last_updated': datetime.now().isoformat()
                }
            }
            
            # Reset indices
            self._knowledge_index = defaultdict(set)
            self._text_search_index = defaultdict(set)
            self._knowledge_relationships = defaultdict(set)
            
            # Reset uncertainty tracking
            self._uncertainty_history = deque(maxlen=self.config.uncertainty_history_size)
            self._prediction_accuracy_history = deque(maxlen=self.config.reliability_window)
            self._domain_confidence_scores = defaultdict(lambda: 0.5)
            self._uncertainty_calibration_data = {
                'predictions': [],
                'true_values': [],
                'confidence_intervals': []
            }
            
            # Clear cache
            if self._cache is not None:
                self.clear_cache()
            
            # Restore config if requested
            if preserve_config and current_config:
                # Remove knowledge_path to avoid overwriting
                if 'knowledge_path' in current_config:
                    del current_config['knowledge_path']
                self.update_config(current_config)
            
            self.logger.info("Brain state reset to initial state")
            
        except Exception as e:
            self.logger.error(f"Failed to reset state: {e}")
            raise
    
    def enable_auto_backup(self, interval_seconds: int = 3600, backup_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Enable automatic state backup at specified intervals.
        
        Args:
            interval_seconds: Backup interval in seconds (default: 1 hour)
            backup_path: Optional custom backup path. If None, uses default location.
            
        Returns:
            True if auto-backup was enabled successfully
        """
        if interval_seconds <= 0:
            self.logger.error("Backup interval must be positive")
            return False
        
        try:
            self._auto_backup_interval = interval_seconds
            
            # Set backup path
            if backup_path:
                self._auto_backup_path = Path(backup_path)
            else:
                if self.config.knowledge_path:
                    self._auto_backup_path = self.config.knowledge_path.with_suffix('.auto_backup')
                else:
                    self._auto_backup_path = Path("brain_auto_backup.json")
            
            # Start backup thread
            self._auto_backup_enabled = True
            self._auto_backup_thread = threading.Thread(
                target=self._auto_backup_loop,
                daemon=True,
                name="BrainCore-AutoBackup"
            )
            self._auto_backup_thread.start()
            
            self.logger.info(f"Auto-backup enabled with interval: {interval_seconds}s, path: {self._auto_backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to enable auto-backup: {e}")
            self._auto_backup_enabled = False
            return False
    
    def disable_auto_backup(self) -> None:
        """Disable automatic state backup."""
        self._auto_backup_enabled = False
        
        if self._auto_backup_thread and self._auto_backup_thread.is_alive():
            # Wait for thread to finish (with timeout)
            self._auto_backup_thread.join(timeout=5.0)
            
        self.logger.info("Auto-backup disabled")
    
    def _auto_backup_loop(self) -> None:
        """Background thread loop for automatic backups."""
        while self._auto_backup_enabled:
            try:
                # Wait for interval or until disabled
                for _ in range(int(self._auto_backup_interval)):
                    if not self._auto_backup_enabled:
                        break
                    time.sleep(1)
                
                if self._auto_backup_enabled:
                    # Perform backup
                    success = self.save_state(self._auto_backup_path)
                    
                    if success:
                        self._last_backup_time = datetime.now()
                        self.logger.debug(f"Auto-backup completed at {self._last_backup_time.isoformat()}")
                    else:
                        self.logger.warning("Auto-backup failed")
                        
            except Exception as e:
                self.logger.error(f"Error in auto-backup loop: {e}")
                # Continue loop despite errors
    
    def get_backup_status(self) -> Dict[str, Any]:
        """
        Get current backup status information.
        
        Returns:
            Dictionary containing backup status
        """
        return {
            'auto_backup_enabled': self._auto_backup_enabled,
            'backup_interval_seconds': self._auto_backup_interval,
            'last_backup_time': self._last_backup_time.isoformat() if self._last_backup_time else None,
            'backup_path': str(self._auto_backup_path) if hasattr(self, '_auto_backup_path') else None,
            'time_since_last_backup': (datetime.now() - self._last_backup_time).total_seconds() if self._last_backup_time else None,
            'checkpoints_available': len(self.list_checkpoints())
        }
    
    def __enter__(self):
        """Context manager entry - enables auto-backup."""
        # Enable auto-backup when entering context
        if self.config.knowledge_persistence:
            self.enable_auto_backup(interval_seconds=300)  # 5 minute auto-backup in context
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - saves state and disables auto-backup."""
        # Save final state
        if self.config.knowledge_persistence and self.config.knowledge_path:
            try:
                self.save_state(self.config.knowledge_path.with_suffix('.final'))
            except Exception as e:
                self.logger.error(f"Failed to save final state: {e}")
        
        # Disable auto-backup
        self.disable_auto_backup()
        
        # Don't suppress exceptions
        return False
    
    def __repr__(self) -> str:
        """String representation of BrainCore."""
        stats = self.get_statistics()
        return (f"BrainCore(domains={stats['total_domains']}, "
                f"knowledge_items={stats['total_knowledge_items']}, "
                f"memory_mb={stats['memory_estimate_mb']:.2f}, "
                f"uncertainty_enabled={self.config.enable_uncertainty})")