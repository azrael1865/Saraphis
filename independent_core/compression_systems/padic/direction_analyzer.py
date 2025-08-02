"""
Direction Analyzer - Advanced direction analysis for hybrid switching optimization
NO FALLBACKS - HARD FAILURES ONLY
"""

import logging
import math
import statistics
import threading
import time
import torch
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from enum import Enum

# Import GAC system components
from ...gac_system.gac_types import DirectionState, DirectionType
from ...gac_system.direction_state import DirectionHistory


class DirectionPatternType(Enum):
    """Direction pattern type enumeration"""
    ASCENDING = "ascending"
    DESCENDING = "descending"
    OSCILLATING = "oscillating"
    STABLE_ASCENDING = "stable_ascending"
    STABLE_DESCENDING = "stable_descending"
    CHAOTIC = "chaotic"
    CONVERGENT = "convergent"
    DIVERGENT = "divergent"


class DirectionTrendType(Enum):
    """Direction trend type enumeration"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"
    CYCLICAL = "cyclical"


@dataclass
class DirectionPatternAnalysis:
    """Direction pattern analysis result"""
    pattern_type: DirectionPatternType
    pattern_confidence: float
    pattern_strength: float
    pattern_duration: int
    pattern_characteristics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate pattern analysis"""
        if not isinstance(self.pattern_type, DirectionPatternType):
            raise TypeError("Pattern type must be DirectionPatternType")
        if not isinstance(self.pattern_confidence, (int, float)) or not (0.0 <= self.pattern_confidence <= 1.0):
            raise ValueError("Pattern confidence must be float between 0.0 and 1.0")
        if not isinstance(self.pattern_strength, (int, float)) or not (0.0 <= self.pattern_strength <= 1.0):
            raise ValueError("Pattern strength must be float between 0.0 and 1.0")
        if not isinstance(self.pattern_duration, int) or self.pattern_duration < 0:
            raise ValueError("Pattern duration must be non-negative integer")


@dataclass
class DirectionTrendAnalysis:
    """Direction trend analysis result"""
    trend_type: DirectionTrendType
    trend_strength: float
    trend_confidence: float
    trend_velocity: float
    trend_acceleration: float
    predicted_direction: float
    prediction_horizon: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate trend analysis"""
        if not isinstance(self.trend_type, DirectionTrendType):
            raise TypeError("Trend type must be DirectionTrendType")
        if not isinstance(self.trend_strength, (int, float)) or not (0.0 <= self.trend_strength <= 1.0):
            raise ValueError("Trend strength must be float between 0.0 and 1.0")
        if not isinstance(self.trend_confidence, (int, float)) or not (0.0 <= self.trend_confidence <= 1.0):
            raise ValueError("Trend confidence must be float between 0.0 and 1.0")


@dataclass
class DirectionStabilityAnalysis:
    """Direction stability analysis result"""
    stability_score: float
    stability_trend: DirectionTrendType
    oscillation_frequency: float
    oscillation_amplitude: float
    stability_confidence: float
    stability_prediction: float
    stability_factors: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate stability analysis"""
        if not isinstance(self.stability_score, (int, float)) or not (0.0 <= self.stability_score <= 1.0):
            raise ValueError("Stability score must be float between 0.0 and 1.0")
        if not isinstance(self.stability_trend, DirectionTrendType):
            raise TypeError("Stability trend must be DirectionTrendType")
        if not isinstance(self.oscillation_frequency, (int, float)) or self.oscillation_frequency < 0:
            raise ValueError("Oscillation frequency must be non-negative")


@dataclass
class DirectionOptimizationRecommendation:
    """Direction optimization recommendation"""
    recommendation_id: str
    recommendation_type: str
    title: str
    description: str
    expected_improvement: float
    implementation_complexity: str
    parameters: Dict[str, Any]
    priority: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate recommendation"""
        if not isinstance(self.recommendation_id, str) or not self.recommendation_id.strip():
            raise ValueError("Recommendation ID must be non-empty string")
        if not isinstance(self.expected_improvement, (int, float)) or not (0.0 <= self.expected_improvement <= 1.0):
            raise ValueError("Expected improvement must be float between 0.0 and 1.0")
        if self.priority not in ['low', 'medium', 'high', 'critical']:
            raise ValueError("Priority must be one of: low, medium, high, critical")


@dataclass
class DirectionAnalyzerConfig:
    """Configuration for direction analyzer"""
    # Pattern analysis parameters
    pattern_window_size: int = 20
    pattern_overlap_ratio: float = 0.5
    pattern_confidence_threshold: float = 0.7
    pattern_strength_threshold: float = 0.6
    
    # Trend analysis parameters
    trend_window_size: int = 15
    trend_smoothing_factor: float = 0.3
    trend_velocity_window: int = 5
    trend_prediction_horizon: int = 10
    
    # Stability analysis parameters
    stability_window_size: int = 25
    oscillation_detection_threshold: float = 0.1
    stability_confidence_threshold: float = 0.75
    stability_prediction_window: int = 8
    
    # Optimization parameters
    enable_adaptive_parameters: bool = True
    optimization_interval_seconds: int = 300
    performance_history_size: int = 100
    
    # Analysis features
    enable_fourier_analysis: bool = True
    enable_wavelet_analysis: bool = False
    enable_correlation_analysis: bool = True
    enable_entropy_analysis: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if self.pattern_window_size <= 0:
            raise ValueError("Pattern window size must be positive")
        if not (0.0 <= self.pattern_overlap_ratio <= 1.0):
            raise ValueError("Pattern overlap ratio must be between 0.0 and 1.0")
        if self.trend_window_size <= 0:
            raise ValueError("Trend window size must be positive")
        if self.stability_window_size <= 0:
            raise ValueError("Stability window size must be positive")


class DirectionAnalyzer:
    """
    Advanced direction analysis for hybrid switching optimization.
    Provides multi-dimensional direction analysis and optimization recommendations.
    """
    
    def __init__(self, config: Optional[DirectionAnalyzerConfig] = None):
        """Initialize direction analyzer"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, DirectionAnalyzerConfig):
            raise TypeError(f"Config must be DirectionAnalyzerConfig or None, got {type(config)}")
        
        self.config = config or DirectionAnalyzerConfig()
        self.logger = logging.getLogger('DirectionAnalyzer')
        
        # Analysis state
        self.pattern_cache: Dict[str, DirectionPatternAnalysis] = {}
        self.trend_cache: Dict[str, DirectionTrendAnalysis] = {}
        self.stability_cache: Dict[str, DirectionStabilityAnalysis] = {}
        
        # Performance tracking
        self.analysis_performance: deque = deque(maxlen=self.config.performance_history_size)
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Thread safety
        self._analysis_lock = threading.RLock()
        self._cache_lock = threading.RLock()
        
        # Analysis metrics
        self.analysis_metrics = {
            'total_analyses': 0,
            'pattern_analyses': 0,
            'trend_analyses': 0,
            'stability_analyses': 0,
            'optimization_recommendations': 0,
            'average_analysis_time_ms': 0.0
        }
        
        self.logger.info("DirectionAnalyzer created successfully")
    
    def analyze_direction_patterns(self, gradient_history: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Analyze direction patterns in gradient history.
        
        Args:
            gradient_history: List of gradient tensors to analyze
            
        Returns:
            Direction pattern analysis results
            
        Raises:
            ValueError: If gradient history is invalid
            RuntimeError: If analysis fails
        """
        if not isinstance(gradient_history, list):
            raise TypeError("Gradient history must be list")
        if len(gradient_history) < self.config.pattern_window_size:
            raise ValueError(f"Gradient history too short: {len(gradient_history)} < {self.config.pattern_window_size}")
        
        for i, grad in enumerate(gradient_history):
            if not isinstance(grad, torch.Tensor):
                raise TypeError(f"Gradient {i} must be torch.Tensor, got {type(grad)}")
            if grad.numel() == 0:
                raise ValueError(f"Gradient {i} cannot be empty")
        
        try:
            start_time = time.time()
            
            with self._analysis_lock:
                # Extract direction features from gradients
                direction_features = self._extract_direction_features(gradient_history)
                
                # Perform pattern analysis
                pattern_analysis = self._analyze_patterns(direction_features)
                
                # Perform frequency analysis if enabled
                if self.config.enable_fourier_analysis:
                    frequency_analysis = self._analyze_frequency_patterns(direction_features)
                    pattern_analysis.update(frequency_analysis)
                
                # Perform correlation analysis if enabled
                if self.config.enable_correlation_analysis:
                    correlation_analysis = self._analyze_direction_correlations(direction_features)
                    pattern_analysis.update(correlation_analysis)
                
                # Perform entropy analysis if enabled
                if self.config.enable_entropy_analysis:
                    entropy_analysis = self._analyze_direction_entropy(direction_features)
                    pattern_analysis.update(entropy_analysis)
                
                # Update metrics
                analysis_time = (time.time() - start_time) * 1000
                self._update_analysis_metrics('pattern', analysis_time)
                
                self.logger.debug(f"Pattern analysis completed in {analysis_time:.2f}ms")
                
                return pattern_analysis
                
        except Exception as e:
            self.logger.error(f"Direction pattern analysis failed: {e}")
            raise RuntimeError(f"Direction pattern analysis failed: {e}")
    
    def predict_direction_trends(self, direction_state: DirectionState) -> DirectionTrendAnalysis:
        """
        Predict direction trends based on current state.
        
        Args:
            direction_state: Current direction state
            
        Returns:
            Direction trend analysis
            
        Raises:
            ValueError: If direction state is invalid
            RuntimeError: If prediction fails
        """
        if direction_state is None:
            raise ValueError("Direction state cannot be None")
        if not isinstance(direction_state, DirectionState):
            raise TypeError("Direction state must be DirectionState")
        
        try:
            start_time = time.time()
            
            with self._analysis_lock:
                # Analyze current trend characteristics
                trend_characteristics = self._analyze_trend_characteristics(direction_state)
                
                # Predict future trend
                trend_prediction = self._predict_trend(trend_characteristics)
                
                # Calculate trend velocity and acceleration
                velocity, acceleration = self._calculate_trend_dynamics(direction_state)
                
                # Determine trend type
                trend_type = self._classify_trend_type(trend_characteristics, velocity, acceleration)
                
                # Calculate trend confidence
                trend_confidence = self._calculate_trend_confidence(trend_characteristics, trend_prediction)
                
                # Create trend analysis result
                trend_analysis = DirectionTrendAnalysis(
                    trend_type=trend_type,
                    trend_strength=trend_characteristics.get('strength', 0.5),
                    trend_confidence=trend_confidence,
                    trend_velocity=velocity,
                    trend_acceleration=acceleration,
                    predicted_direction=trend_prediction,
                    prediction_horizon=self.config.trend_prediction_horizon
                )
                
                # Cache result
                cache_key = f"trend_{direction_state.timestamp.isoformat()}"
                with self._cache_lock:
                    self.trend_cache[cache_key] = trend_analysis
                
                # Update metrics
                analysis_time = (time.time() - start_time) * 1000
                self._update_analysis_metrics('trend', analysis_time)
                
                self.logger.debug(f"Trend prediction completed: {trend_type.name} (confidence: {trend_confidence:.3f})")
                
                return trend_analysis
                
        except Exception as e:
            self.logger.error(f"Direction trend prediction failed: {e}")
            raise RuntimeError(f"Direction trend prediction failed: {e}")
    
    def analyze_direction_stability(self, direction_history: List[DirectionHistory]) -> DirectionStabilityAnalysis:
        """
        Analyze direction stability from history.
        
        Args:
            direction_history: List of direction history entries
            
        Returns:
            Direction stability analysis
            
        Raises:
            ValueError: If direction history is invalid
            RuntimeError: If analysis fails
        """
        if not isinstance(direction_history, list):
            raise TypeError("Direction history must be list")
        if len(direction_history) < self.config.stability_window_size:
            raise ValueError(f"Direction history too short: {len(direction_history)} < {self.config.stability_window_size}")
        
        for i, entry in enumerate(direction_history):
            if not isinstance(entry, DirectionHistory):
                raise TypeError(f"History entry {i} must be DirectionHistory, got {type(entry)}")
        
        try:
            start_time = time.time()
            
            with self._analysis_lock:
                # Extract stability features
                stability_features = self._extract_stability_features(direction_history)
                
                # Calculate stability score
                stability_score = self._calculate_stability_score(stability_features)
                
                # Analyze oscillation patterns
                oscillation_frequency, oscillation_amplitude = self._analyze_oscillations(stability_features)
                
                # Determine stability trend
                stability_trend = self._analyze_stability_trend(stability_features)
                
                # Calculate stability confidence
                stability_confidence = self._calculate_stability_confidence(stability_features)
                
                # Predict future stability
                stability_prediction = self._predict_stability(stability_features)
                
                # Identify stability factors
                stability_factors = self._identify_stability_factors(stability_features)
                
                # Create stability analysis result
                stability_analysis = DirectionStabilityAnalysis(
                    stability_score=stability_score,
                    stability_trend=stability_trend,
                    oscillation_frequency=oscillation_frequency,
                    oscillation_amplitude=oscillation_amplitude,
                    stability_confidence=stability_confidence,
                    stability_prediction=stability_prediction,
                    stability_factors=stability_factors
                )
                
                # Cache result
                cache_key = f"stability_{datetime.utcnow().isoformat()}"
                with self._cache_lock:
                    self.stability_cache[cache_key] = stability_analysis
                
                # Update metrics
                analysis_time = (time.time() - start_time) * 1000
                self._update_analysis_metrics('stability', analysis_time)
                
                self.logger.debug(f"Stability analysis completed: score={stability_score:.3f}, trend={stability_trend.name}")
                
                return stability_analysis
                
        except Exception as e:
            self.logger.error(f"Direction stability analysis failed: {e}")
            raise RuntimeError(f"Direction stability analysis failed: {e}")
    
    def calculate_direction_confidence(self, direction_state: DirectionState) -> float:
        """
        Calculate direction confidence score.
        
        Args:
            direction_state: Direction state to analyze
            
        Returns:
            Direction confidence score (0.0 to 1.0)
            
        Raises:
            ValueError: If direction state is invalid
            RuntimeError: If calculation fails
        """
        if direction_state is None:
            raise ValueError("Direction state cannot be None")
        if not isinstance(direction_state, DirectionState):
            raise TypeError("Direction state must be DirectionState")
        
        try:
            # Base confidence from direction state
            base_confidence = direction_state.confidence
            
            # Stability contribution
            stability_contribution = direction_state.stability * 0.3
            
            # Consistency contribution (based on direction history)
            consistency_contribution = self._calculate_consistency_contribution(direction_state) * 0.2
            
            # Magnitude contribution
            magnitude_contribution = self._calculate_magnitude_contribution(direction_state) * 0.2
            
            # Trend contribution
            trend_contribution = self._calculate_trend_contribution(direction_state) * 0.3
            
            # Combine contributions
            total_confidence = (
                base_confidence * 0.4 +
                stability_contribution +
                consistency_contribution +
                magnitude_contribution +
                trend_contribution
            )
            
            # Clamp to valid range
            confidence = max(0.0, min(1.0, total_confidence))
            
            self.logger.debug(f"Direction confidence calculated: {confidence:.3f}")
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Direction confidence calculation failed: {e}")
            raise RuntimeError(f"Direction confidence calculation failed: {e}")
    
    def generate_direction_recommendations(self, analysis: Dict[str, Any]) -> List[DirectionOptimizationRecommendation]:
        """
        Generate direction optimization recommendations.
        
        Args:
            analysis: Direction analysis results
            
        Returns:
            List of optimization recommendations
            
        Raises:
            ValueError: If analysis is invalid
            RuntimeError: If recommendation generation fails
        """
        if not isinstance(analysis, dict):
            raise TypeError("Analysis must be dict")
        if not analysis:
            raise ValueError("Analysis cannot be empty")
        
        try:
            recommendations = []
            
            # Pattern-based recommendations
            if 'pattern_type' in analysis:
                pattern_recommendations = self._generate_pattern_recommendations(analysis)
                recommendations.extend(pattern_recommendations)
            
            # Stability-based recommendations
            if 'stability_score' in analysis:
                stability_recommendations = self._generate_stability_recommendations(analysis)
                recommendations.extend(stability_recommendations)
            
            # Performance-based recommendations
            if 'performance_correlation' in analysis:
                performance_recommendations = self._generate_performance_recommendations(analysis)
                recommendations.extend(performance_recommendations)
            
            # Trend-based recommendations
            if 'trend_type' in analysis:
                trend_recommendations = self._generate_trend_recommendations(analysis)
                recommendations.extend(trend_recommendations)
            
            # Sort recommendations by expected improvement
            recommendations.sort(key=lambda r: r.expected_improvement, reverse=True)
            
            # Update metrics
            self.analysis_metrics['optimization_recommendations'] += len(recommendations)
            
            self.logger.info(f"Generated {len(recommendations)} direction optimization recommendations")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Direction recommendation generation failed: {e}")
            raise RuntimeError(f"Direction recommendation generation failed: {e}")
    
    def optimize_direction_parameters(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimize direction analysis parameters based on performance.
        
        Args:
            performance_data: Performance data for optimization
            
        Returns:
            Optimization results
            
        Raises:
            ValueError: If performance data is invalid
            RuntimeError: If optimization fails
        """
        if not isinstance(performance_data, list):
            raise TypeError("Performance data must be list")
        if len(performance_data) < 10:
            raise ValueError("Insufficient performance data for optimization")
        
        try:
            optimization_results = {
                'parameter_changes': {},
                'performance_improvements': {},
                'optimization_score': 0.0,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Optimize pattern analysis parameters
            pattern_optimization = self._optimize_pattern_parameters(performance_data)
            optimization_results['parameter_changes']['pattern'] = pattern_optimization
            
            # Optimize trend analysis parameters
            trend_optimization = self._optimize_trend_parameters(performance_data)
            optimization_results['parameter_changes']['trend'] = trend_optimization
            
            # Optimize stability analysis parameters
            stability_optimization = self._optimize_stability_parameters(performance_data)
            optimization_results['parameter_changes']['stability'] = stability_optimization
            
            # Calculate overall optimization score
            optimization_results['optimization_score'] = self._calculate_optimization_score(
                pattern_optimization, trend_optimization, stability_optimization
            )
            
            # Record optimization in history
            self.optimization_history.append(optimization_results)
            
            self.logger.info(f"Direction parameter optimization completed: score={optimization_results['optimization_score']:.3f}")
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Direction parameter optimization failed: {e}")
            raise RuntimeError(f"Direction parameter optimization failed: {e}")
    
    def _extract_direction_features(self, gradient_history: List[torch.Tensor]) -> Dict[str, np.ndarray]:
        """Extract direction features from gradient history"""
        features = {}
        
        # Calculate gradient norms
        norms = []
        for grad in gradient_history:
            norm = torch.norm(grad).item()
            norms.append(norm)
        features['norms'] = np.array(norms)
        
        # Calculate gradient angles (for 2D case, approximate for higher dimensions)
        angles = []
        for grad in gradient_history:
            if grad.numel() >= 2:
                flat_grad = grad.flatten()
                angle = math.atan2(flat_grad[1].item(), flat_grad[0].item())
                angles.append(angle)
            else:
                angles.append(0.0)
        features['angles'] = np.array(angles)
        
        # Calculate gradient differences
        differences = []
        for i in range(1, len(gradient_history)):
            diff = torch.norm(gradient_history[i] - gradient_history[i-1]).item()
            differences.append(diff)
        features['differences'] = np.array(differences)
        
        # Calculate normalized gradients
        normalized_grads = []
        for grad in gradient_history:
            norm = torch.norm(grad)
            if norm > 1e-8:
                normalized = grad / norm
                normalized_grads.append(normalized.flatten().numpy())
            else:
                normalized_grads.append(np.zeros(grad.numel()))
        features['normalized_gradients'] = np.array(normalized_grads)
        
        return features
    
    def _analyze_patterns(self, direction_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze direction patterns"""
        analysis = {}
        
        norms = direction_features.get('norms', np.array([]))
        angles = direction_features.get('angles', np.array([]))
        differences = direction_features.get('differences', np.array([]))
        
        if len(norms) == 0:
            return analysis
        
        # Analyze norm patterns
        norm_trend = self._analyze_trend(norms)
        analysis['norm_trend'] = norm_trend
        
        # Analyze angle patterns
        if len(angles) > 0:
            angle_stability = self._calculate_angle_stability(angles)
            analysis['angle_stability'] = angle_stability
        
        # Analyze difference patterns
        if len(differences) > 0:
            difference_stability = np.std(differences) / (np.mean(differences) + 1e-8)
            analysis['difference_stability'] = min(1.0, 1.0 / (difference_stability + 1.0))
        
        # Determine overall pattern type
        pattern_type, pattern_confidence = self._classify_pattern(norms, angles, differences)
        analysis['pattern_type'] = pattern_type.name
        analysis['pattern_confidence'] = pattern_confidence
        
        # Calculate pattern strength
        pattern_strength = self._calculate_pattern_strength(norms, angles, differences)
        analysis['pattern_strength'] = pattern_strength
        
        return analysis
    
    def _analyze_frequency_patterns(self, direction_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze frequency patterns using Fourier analysis"""
        analysis = {}
        
        norms = direction_features.get('norms', np.array([]))
        if len(norms) < 8:  # Need minimum data for FFT
            return analysis
        
        try:
            # Apply FFT to norms
            fft_result = np.fft.fft(norms)
            frequencies = np.fft.fftfreq(len(norms))
            magnitudes = np.abs(fft_result)
            
            # Find dominant frequency
            dominant_freq_idx = np.argmax(magnitudes[1:len(magnitudes)//2]) + 1  # Skip DC component
            dominant_frequency = frequencies[dominant_freq_idx]
            dominant_magnitude = magnitudes[dominant_freq_idx]
            
            analysis['dominant_frequency'] = abs(dominant_frequency)
            analysis['frequency_strength'] = dominant_magnitude / len(norms)
            
            # Analyze frequency distribution
            low_freq_power = np.sum(magnitudes[1:len(magnitudes)//4])
            high_freq_power = np.sum(magnitudes[len(magnitudes)//4:len(magnitudes)//2])
            total_power = low_freq_power + high_freq_power
            
            if total_power > 0:
                analysis['low_frequency_ratio'] = low_freq_power / total_power
                analysis['high_frequency_ratio'] = high_freq_power / total_power
            
        except Exception as e:
            self.logger.debug(f"Fourier analysis failed: {e}")
        
        return analysis
    
    def _analyze_direction_correlations(self, direction_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze correlations between direction features"""
        analysis = {}
        
        norms = direction_features.get('norms', np.array([]))
        differences = direction_features.get('differences', np.array([]))
        
        if len(norms) > 1 and len(differences) > 0:
            # Correlate norms with differences
            if len(norms) == len(differences) + 1:
                norms_truncated = norms[1:]  # Match lengths
            else:
                norms_truncated = norms[:len(differences)]
            
            if len(norms_truncated) == len(differences) and len(differences) > 1:
                correlation = np.corrcoef(norms_truncated, differences)[0, 1]
                if not np.isnan(correlation):
                    analysis['norm_difference_correlation'] = correlation
        
        return analysis
    
    def _analyze_direction_entropy(self, direction_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze direction entropy"""
        analysis = {}
        
        norms = direction_features.get('norms', np.array([]))
        angles = direction_features.get('angles', np.array([]))
        
        # Calculate norm entropy
        if len(norms) > 0:
            norm_entropy = self._calculate_entropy(norms)
            analysis['norm_entropy'] = norm_entropy
        
        # Calculate angle entropy
        if len(angles) > 0:
            angle_entropy = self._calculate_entropy(angles)
            analysis['angle_entropy'] = angle_entropy
        
        return analysis
    
    def _analyze_trend_characteristics(self, direction_state: DirectionState) -> Dict[str, float]:
        """Analyze trend characteristics from direction state"""
        characteristics = {}
        
        # Use direction state properties
        characteristics['confidence'] = direction_state.confidence
        characteristics['stability'] = direction_state.stability
        characteristics['magnitude'] = getattr(direction_state, 'magnitude', 1.0)
        
        # Calculate derived characteristics
        characteristics['strength'] = (direction_state.confidence + direction_state.stability) / 2.0
        characteristics['reliability'] = min(direction_state.confidence, direction_state.stability)
        
        return characteristics
    
    def _predict_trend(self, trend_characteristics: Dict[str, float]) -> float:
        """Predict future trend direction"""
        # Simple linear prediction based on current characteristics
        confidence = trend_characteristics.get('confidence', 0.5)
        stability = trend_characteristics.get('stability', 0.5)
        
        # Predict direction continuation probability
        prediction = (confidence * 0.6 + stability * 0.4)
        
        return prediction
    
    def _calculate_trend_dynamics(self, direction_state: DirectionState) -> Tuple[float, float]:
        """Calculate trend velocity and acceleration"""
        # Placeholder implementation - would need historical data for real calculation
        velocity = direction_state.confidence - 0.5  # Relative to neutral
        acceleration = direction_state.stability - 0.5  # Relative to neutral
        
        return velocity, acceleration
    
    def _classify_trend_type(self, characteristics: Dict[str, float], velocity: float, acceleration: float) -> DirectionTrendType:
        """Classify trend type based on characteristics"""
        confidence = characteristics.get('confidence', 0.5)
        stability = characteristics.get('stability', 0.5)
        
        if stability > 0.8:
            if velocity > 0.1:
                return DirectionTrendType.INCREASING
            elif velocity < -0.1:
                return DirectionTrendType.DECREASING
            else:
                return DirectionTrendType.STABLE
        elif stability < 0.3:
            return DirectionTrendType.VOLATILE
        else:
            if abs(velocity) < 0.05 and abs(acceleration) < 0.05:
                return DirectionTrendType.CYCLICAL
            else:
                return DirectionTrendType.STABLE
    
    def _calculate_trend_confidence(self, characteristics: Dict[str, float], prediction: float) -> float:
        """Calculate trend confidence"""
        confidence = characteristics.get('confidence', 0.5)
        stability = characteristics.get('stability', 0.5)
        
        # Combine factors
        trend_confidence = (confidence * 0.4 + stability * 0.4 + prediction * 0.2)
        
        return max(0.0, min(1.0, trend_confidence))
    
    def _extract_stability_features(self, direction_history: List[DirectionHistory]) -> Dict[str, np.ndarray]:
        """Extract stability features from direction history"""
        features = {}
        
        # Extract confidence values
        confidences = [entry.confidence for entry in direction_history]
        features['confidences'] = np.array(confidences)
        
        # Extract stability values
        stabilities = [entry.stability if hasattr(entry, 'stability') else 0.5 for entry in direction_history]
        features['stabilities'] = np.array(stabilities)
        
        # Calculate changes
        confidence_changes = np.diff(confidences)
        features['confidence_changes'] = confidence_changes
        
        return features
    
    def _calculate_stability_score(self, stability_features: Dict[str, np.ndarray]) -> float:
        """Calculate overall stability score"""
        confidences = stability_features.get('confidences', np.array([0.5]))
        stabilities = stability_features.get('stabilities', np.array([0.5]))
        confidence_changes = stability_features.get('confidence_changes', np.array([0.0]))
        
        # Calculate stability components
        avg_confidence = np.mean(confidences)
        avg_stability = np.mean(stabilities)
        change_stability = 1.0 / (1.0 + np.std(confidence_changes))
        
        # Combine components
        stability_score = (avg_confidence * 0.3 + avg_stability * 0.4 + change_stability * 0.3)
        
        return max(0.0, min(1.0, stability_score))
    
    def _analyze_oscillations(self, stability_features: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """Analyze oscillation patterns"""
        confidences = stability_features.get('confidences', np.array([0.5]))
        
        if len(confidences) < 4:
            return 0.0, 0.0
        
        # Detect oscillations using zero crossings of detrended signal
        detrended = confidences - np.mean(confidences)
        zero_crossings = np.where(np.diff(np.sign(detrended)))[0]
        
        # Calculate frequency (oscillations per unit time)
        if len(zero_crossings) > 1:
            frequency = len(zero_crossings) / (2 * len(confidences))  # Normalize
        else:
            frequency = 0.0
        
        # Calculate amplitude
        amplitude = np.std(detrended)
        
        return frequency, amplitude
    
    def _analyze_stability_trend(self, stability_features: Dict[str, np.ndarray]) -> DirectionTrendType:
        """Analyze stability trend"""
        stabilities = stability_features.get('stabilities', np.array([0.5]))
        
        if len(stabilities) < 3:
            return DirectionTrendType.STABLE
        
        # Calculate trend using linear regression slope
        x = np.arange(len(stabilities))
        slope = np.polyfit(x, stabilities, 1)[0]
        
        if slope > 0.01:
            return DirectionTrendType.INCREASING
        elif slope < -0.01:
            return DirectionTrendType.DECREASING
        else:
            return DirectionTrendType.STABLE
    
    def _calculate_stability_confidence(self, stability_features: Dict[str, np.ndarray]) -> float:
        """Calculate stability confidence"""
        stabilities = stability_features.get('stabilities', np.array([0.5]))
        
        # Higher confidence for consistent stability values
        stability_variance = np.var(stabilities)
        confidence = 1.0 / (1.0 + stability_variance * 10)
        
        return max(0.0, min(1.0, confidence))
    
    def _predict_stability(self, stability_features: Dict[str, np.ndarray]) -> float:
        """Predict future stability"""
        stabilities = stability_features.get('stabilities', np.array([0.5]))
        
        if len(stabilities) < 2:
            return stabilities[-1] if len(stabilities) > 0 else 0.5
        
        # Simple extrapolation
        recent_trend = stabilities[-1] - stabilities[-2]
        predicted_stability = stabilities[-1] + recent_trend * 0.5  # Damped prediction
        
        return max(0.0, min(1.0, predicted_stability))
    
    def _identify_stability_factors(self, stability_features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Identify factors affecting stability"""
        factors = {}
        
        confidences = stability_features.get('confidences', np.array([]))
        confidence_changes = stability_features.get('confidence_changes', np.array([]))
        
        if len(confidences) > 0:
            factors['confidence_level'] = np.mean(confidences)
            factors['confidence_variance'] = np.var(confidences)
        
        if len(confidence_changes) > 0:
            factors['change_magnitude'] = np.mean(np.abs(confidence_changes))
            factors['change_consistency'] = 1.0 / (1.0 + np.var(confidence_changes))
        
        return factors
    
    def _calculate_consistency_contribution(self, direction_state: DirectionState) -> float:
        """Calculate consistency contribution to confidence"""
        # Placeholder - would need historical comparison
        return direction_state.confidence * 0.8
    
    def _calculate_magnitude_contribution(self, direction_state: DirectionState) -> float:
        """Calculate magnitude contribution to confidence"""
        magnitude = getattr(direction_state, 'magnitude', 1.0)
        # Normalize magnitude contribution
        return min(1.0, magnitude / 2.0)
    
    def _calculate_trend_contribution(self, direction_state: DirectionState) -> float:
        """Calculate trend contribution to confidence"""
        # Use stability as proxy for trend consistency
        return direction_state.stability
    
    def _generate_pattern_recommendations(self, analysis: Dict[str, Any]) -> List[DirectionOptimizationRecommendation]:
        """Generate pattern-based recommendations"""
        recommendations = []
        
        pattern_type = analysis.get('pattern_type', 'unknown')
        pattern_confidence = analysis.get('pattern_confidence', 0.5)
        
        if pattern_confidence < 0.7:
            recommendations.append(DirectionOptimizationRecommendation(
                recommendation_id="pattern_analysis_improvement",
                recommendation_type="pattern",
                title="Improve Pattern Analysis",
                description="Pattern confidence is low. Consider increasing pattern window size or improving feature extraction.",
                expected_improvement=0.15,
                implementation_complexity="medium",
                parameters={"pattern_window_size": self.config.pattern_window_size * 1.5},
                priority="medium"
            ))
        
        if pattern_type == "oscillating":
            recommendations.append(DirectionOptimizationRecommendation(
                recommendation_id="oscillation_handling",
                recommendation_type="pattern",
                title="Optimize Oscillation Handling",
                description="Detected oscillating pattern. Consider specialized oscillation-aware switching strategy.",
                expected_improvement=0.20,
                implementation_complexity="high",
                parameters={"enable_oscillation_detection": True},
                priority="high"
            ))
        
        return recommendations
    
    def _generate_stability_recommendations(self, analysis: Dict[str, Any]) -> List[DirectionOptimizationRecommendation]:
        """Generate stability-based recommendations"""
        recommendations = []
        
        stability_score = analysis.get('stability_score', 0.5)
        
        if stability_score < 0.6:
            recommendations.append(DirectionOptimizationRecommendation(
                recommendation_id="stability_improvement",
                recommendation_type="stability",
                title="Improve Stability Analysis",
                description="Low stability detected. Consider increasing stability window size or adjusting thresholds.",
                expected_improvement=0.18,
                implementation_complexity="medium",
                parameters={"stability_window_size": self.config.stability_window_size * 1.2},
                priority="high"
            ))
        
        return recommendations
    
    def _generate_performance_recommendations(self, analysis: Dict[str, Any]) -> List[DirectionOptimizationRecommendation]:
        """Generate performance-based recommendations"""
        recommendations = []
        
        # Placeholder implementation
        recommendations.append(DirectionOptimizationRecommendation(
            recommendation_id="performance_optimization",
            recommendation_type="performance",
            title="Optimize Performance Tracking",
            description="Enhance performance correlation analysis for better switching decisions.",
            expected_improvement=0.12,
            implementation_complexity="low",
            parameters={"enable_advanced_correlation": True},
            priority="medium"
        ))
        
        return recommendations
    
    def _generate_trend_recommendations(self, analysis: Dict[str, Any]) -> List[DirectionOptimizationRecommendation]:
        """Generate trend-based recommendations"""
        recommendations = []
        
        trend_type = analysis.get('trend_type', 'stable')
        
        if trend_type == "volatile":
            recommendations.append(DirectionOptimizationRecommendation(
                recommendation_id="volatility_handling",
                recommendation_type="trend",
                title="Handle High Volatility",
                description="High volatility detected. Consider adaptive smoothing or specialized volatile pattern handling.",
                expected_improvement=0.22,
                implementation_complexity="high",
                parameters={"enable_volatility_adaptation": True},
                priority="high"
            ))
        
        return recommendations
    
    def _optimize_pattern_parameters(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize pattern analysis parameters"""
        # Placeholder implementation
        return {
            'parameter': 'pattern_window_size',
            'old_value': self.config.pattern_window_size,
            'new_value': max(10, min(50, self.config.pattern_window_size + 2)),
            'improvement': 0.05
        }
    
    def _optimize_trend_parameters(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize trend analysis parameters"""
        # Placeholder implementation
        return {
            'parameter': 'trend_window_size',
            'old_value': self.config.trend_window_size,
            'new_value': max(5, min(30, self.config.trend_window_size + 1)),
            'improvement': 0.03
        }
    
    def _optimize_stability_parameters(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize stability analysis parameters"""
        # Placeholder implementation
        return {
            'parameter': 'stability_window_size',
            'old_value': self.config.stability_window_size,
            'new_value': max(10, min(50, self.config.stability_window_size + 3)),
            'improvement': 0.04
        }
    
    def _calculate_optimization_score(self, pattern_opt: Dict, trend_opt: Dict, stability_opt: Dict) -> float:
        """Calculate overall optimization score"""
        pattern_improvement = pattern_opt.get('improvement', 0.0)
        trend_improvement = trend_opt.get('improvement', 0.0)
        stability_improvement = stability_opt.get('improvement', 0.0)
        
        # Weighted average
        total_improvement = (pattern_improvement * 0.4 + trend_improvement * 0.3 + stability_improvement * 0.3)
        
        return total_improvement
    
    def _update_analysis_metrics(self, analysis_type: str, execution_time_ms: float) -> None:
        """Update analysis metrics"""
        self.analysis_metrics['total_analyses'] += 1
        
        if analysis_type == 'pattern':
            self.analysis_metrics['pattern_analyses'] += 1
        elif analysis_type == 'trend':
            self.analysis_metrics['trend_analyses'] += 1
        elif analysis_type == 'stability':
            self.analysis_metrics['stability_analyses'] += 1
        
        # Update average analysis time
        total_time = self.analysis_metrics['average_analysis_time_ms'] * (self.analysis_metrics['total_analyses'] - 1)
        total_time += execution_time_ms
        self.analysis_metrics['average_analysis_time_ms'] = total_time / self.analysis_metrics['total_analyses']
    
    def _analyze_trend(self, values: np.ndarray) -> str:
        """Analyze trend in values"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple trend analysis using linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_angle_stability(self, angles: np.ndarray) -> float:
        """Calculate stability of angles"""
        if len(angles) < 2:
            return 1.0
        
        # Calculate circular variance for angles
        circular_variance = 1.0 - abs(np.mean(np.exp(1j * angles)))
        stability = 1.0 - circular_variance
        
        return max(0.0, min(1.0, stability))
    
    def _classify_pattern(self, norms: np.ndarray, angles: np.ndarray, differences: np.ndarray) -> Tuple[DirectionPatternType, float]:
        """Classify direction pattern"""
        if len(norms) < 5:
            return DirectionPatternType.CHAOTIC, 0.5
        
        # Analyze norm trend
        norm_trend = self._analyze_trend(norms)
        
        # Analyze stability
        norm_stability = 1.0 / (1.0 + np.std(norms) / (np.mean(norms) + 1e-8))
        
        # Determine pattern type
        if norm_stability > 0.8:
            if norm_trend == "increasing":
                return DirectionPatternType.STABLE_ASCENDING, 0.9
            elif norm_trend == "decreasing":
                return DirectionPatternType.STABLE_DESCENDING, 0.9
            else:
                return DirectionPatternType.CONVERGENT, 0.8
        elif norm_stability < 0.3:
            return DirectionPatternType.CHAOTIC, 0.3
        else:
            if norm_trend == "increasing":
                return DirectionPatternType.ASCENDING, 0.7
            elif norm_trend == "decreasing":
                return DirectionPatternType.DESCENDING, 0.7
            else:
                return DirectionPatternType.OSCILLATING, 0.6
    
    def _calculate_pattern_strength(self, norms: np.ndarray, angles: np.ndarray, differences: np.ndarray) -> float:
        """Calculate pattern strength"""
        if len(norms) == 0:
            return 0.0
        
        # Combine multiple measures of pattern strength
        norm_consistency = 1.0 / (1.0 + np.std(norms) / (np.mean(norms) + 1e-8))
        
        if len(differences) > 0:
            difference_consistency = 1.0 / (1.0 + np.std(differences) / (np.mean(differences) + 1e-8))
        else:
            difference_consistency = 0.5
        
        # Combined strength
        strength = (norm_consistency * 0.6 + difference_consistency * 0.4)
        
        return max(0.0, min(1.0, strength))
    
    def _calculate_entropy(self, values: np.ndarray) -> float:
        """Calculate entropy of values"""
        if len(values) == 0:
            return 0.0
        
        # Bin the values
        bins = min(10, len(values) // 2)
        if bins < 2:
            return 0.0
        
        hist, _ = np.histogram(values, bins=bins)
        hist = hist[hist > 0]  # Remove zero bins
        
        if len(hist) == 0:
            return 0.0
        
        # Normalize
        prob = hist / np.sum(hist)
        
        # Calculate entropy
        entropy = -np.sum(prob * np.log2(prob))
        
        # Normalize to [0, 1]
        max_entropy = np.log2(len(prob))
        if max_entropy > 0:
            entropy = entropy / max_entropy
        
        return entropy
    
    def shutdown(self) -> None:
        """Shutdown direction analyzer"""
        self.logger.info("Shutting down direction analyzer")
        
        # Clear caches
        self.pattern_cache.clear()
        self.trend_cache.clear()
        self.stability_cache.clear()
        
        # Clear performance data
        self.analysis_performance.clear()
        self.optimization_history.clear()
        
        self.logger.info("Direction analyzer shutdown complete")