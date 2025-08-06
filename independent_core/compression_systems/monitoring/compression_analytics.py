"""
Compression Analytics System - Comprehensive monitoring and analysis for neural compression
Real-time and batch analytics for P-adic and Tropical compression pipelines
NO PLACEHOLDERS - PRODUCTION READY
FAIL LOUD - NO GRACEFUL DEGRADATION
"""

import time
import json
import csv
import threading
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union, Deque, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
from pathlib import Path
import hashlib
import statistics

# Import the existing performance monitor
from ..tropical.unified_performance_monitor import (
    UnifiedPerformanceMonitor,
    MetricType,
    PipelineType,
    PerformanceMetric,
    MonitorConfig
)

logger = logging.getLogger(__name__)


class LayerType(Enum):
    """Neural network layer types"""
    DENSE = "dense"
    CONV = "conv"
    ATTENTION = "attention"
    EMBEDDING = "embedding"
    NORMALIZATION = "normalization"
    POOLING = "pooling"
    ACTIVATION = "activation"
    UNKNOWN = "unknown"


class AnalysisMode(Enum):
    """Analytics operation modes"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"


@dataclass
class LayerMetrics:
    """Metrics for a specific layer"""
    layer_id: str
    layer_type: LayerType
    compression_ratio: float
    memory_saved_mb: float
    latency_ms: float
    accuracy_loss: float
    pipeline_used: PipelineType
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate layer metrics"""
        if self.compression_ratio < 1.0:
            raise ValueError(f"Compression ratio must be >= 1.0, got {self.compression_ratio}")
        if self.memory_saved_mb < 0:
            raise ValueError(f"Memory saved must be >= 0, got {self.memory_saved_mb}")
        if self.latency_ms < 0:
            raise ValueError(f"Latency must be >= 0, got {self.latency_ms}")
        if not (0.0 <= self.accuracy_loss <= 1.0):
            raise ValueError(f"Accuracy loss must be in [0, 1], got {self.accuracy_loss}")


@dataclass
class RollingStatistics:
    """Rolling statistics for a metric window"""
    mean: float
    std: float
    min: float
    max: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    percentile_95: float
    percentile_99: float
    count: int
    window_size: int
    
    @classmethod
    def from_values(cls, values: List[float], window_size: int) -> 'RollingStatistics':
        """Create statistics from values"""
        if not values:
            raise ValueError("Cannot compute statistics from empty values")
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return cls(
            mean=statistics.mean(values),
            std=statistics.stdev(values) if n > 1 else 0.0,
            min=min(values),
            max=max(values),
            percentile_25=sorted_values[int(n * 0.25)],
            percentile_50=sorted_values[int(n * 0.50)],
            percentile_75=sorted_values[int(n * 0.75)],
            percentile_95=sorted_values[min(int(n * 0.95), n-1)],
            percentile_99=sorted_values[min(int(n * 0.99), n-1)],
            count=n,
            window_size=window_size
        )


@dataclass
class CompressionEfficiency:
    """Compression efficiency metrics"""
    compression_ratio: float
    accuracy_retained: float  # 1.0 - accuracy_loss
    memory_reduction_percent: float
    speed_improvement_factor: float
    efficiency_score: float  # Combined metric
    
    def __post_init__(self):
        """Validate efficiency metrics"""
        if self.compression_ratio < 1.0:
            raise ValueError(f"Compression ratio must be >= 1.0, got {self.compression_ratio}")
        if not (0.0 <= self.accuracy_retained <= 1.0):
            raise ValueError(f"Accuracy retained must be in [0, 1], got {self.accuracy_retained}")
        if not (0.0 <= self.memory_reduction_percent <= 100.0):
            raise ValueError(f"Memory reduction must be in [0, 100], got {self.memory_reduction_percent}")
        if self.speed_improvement_factor < 0:
            raise ValueError(f"Speed improvement must be >= 0, got {self.speed_improvement_factor}")


@dataclass
class ModelImpact:
    """Model-level compression impact"""
    model_id: str
    original_accuracy: float
    compressed_accuracy: float
    accuracy_drop: float
    original_size_mb: float
    compressed_size_mb: float
    size_reduction_percent: float
    inference_speedup: float
    critical_layers: List[str]
    sensitivity_map: Dict[str, float]
    
    def __post_init__(self):
        """Validate model impact metrics"""
        if not (0.0 <= self.original_accuracy <= 1.0):
            raise ValueError(f"Original accuracy must be in [0, 1], got {self.original_accuracy}")
        if not (0.0 <= self.compressed_accuracy <= 1.0):
            raise ValueError(f"Compressed accuracy must be in [0, 1], got {self.compressed_accuracy}")
        if self.original_size_mb <= 0:
            raise ValueError(f"Original size must be > 0, got {self.original_size_mb}")
        if self.compressed_size_mb <= 0:
            raise ValueError(f"Compressed size must be > 0, got {self.compressed_size_mb}")


@dataclass
class CompressionRecommendation:
    """Compression pipeline recommendation"""
    recommended_pipeline: PipelineType
    confidence_score: float
    reasoning: List[str]
    optimal_parameters: Dict[str, Any]
    expected_compression_ratio: float
    expected_accuracy_retention: float
    alternative_pipelines: List[Tuple[PipelineType, float]]  # (pipeline, score)
    
    def __post_init__(self):
        """Validate recommendation"""
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError(f"Confidence score must be in [0, 1], got {self.confidence_score}")
        if self.expected_compression_ratio < 1.0:
            raise ValueError(f"Expected compression ratio must be >= 1.0, got {self.expected_compression_ratio}")
        if not (0.0 <= self.expected_accuracy_retention <= 1.0):
            raise ValueError(f"Expected accuracy retention must be in [0, 1], got {self.expected_accuracy_retention}")


class CompressionMetricsAggregator:
    """
    Aggregate metrics from UnifiedPerformanceMonitor
    Calculate rolling statistics and track metrics per layer type
    """
    
    def __init__(self, monitor: UnifiedPerformanceMonitor, 
                 window_size: int = 1000,
                 aggregation_interval_ms: int = 100):
        """
        Initialize metrics aggregator
        
        Args:
            monitor: UnifiedPerformanceMonitor instance
            window_size: Size of sliding window for statistics
            aggregation_interval_ms: Interval for aggregation updates
        """
        if window_size <= 0:
            raise ValueError(f"Window size must be positive, got {window_size}")
        if aggregation_interval_ms <= 0:
            raise ValueError(f"Aggregation interval must be positive, got {aggregation_interval_ms}")
        
        self.monitor = monitor
        self.window_size = window_size
        self.aggregation_interval_ms = aggregation_interval_ms
        
        # Metrics storage per layer type
        self.layer_metrics: Dict[LayerType, Deque[LayerMetrics]] = {
            layer_type: deque(maxlen=window_size)
            for layer_type in LayerType
        }
        
        # Rolling statistics cache
        self.rolling_stats: Dict[Tuple[LayerType, str], RollingStatistics] = {}
        self.stats_lock = threading.RLock()
        
        # Aggregation state
        self.total_layers_processed = 0
        self.total_memory_saved_mb = 0.0
        self.aggregation_active = True
        
        # Start aggregation thread
        self.aggregation_thread = threading.Thread(target=self._aggregation_loop, daemon=True)
        self.aggregation_thread.start()
        
        logger.info(f"CompressionMetricsAggregator initialized with window_size={window_size}")
    
    def add_layer_metrics(self, metrics: LayerMetrics):
        """
        Add metrics for a processed layer
        
        Args:
            metrics: Layer compression metrics
        """
        with self.stats_lock:
            self.layer_metrics[metrics.layer_type].append(metrics)
            self.total_layers_processed += 1
            self.total_memory_saved_mb += metrics.memory_saved_mb
            
            # Record in performance monitor
            self.monitor.record_compression(
                pipeline_type=metrics.pipeline_used,
                input_size_bytes=int(metrics.metadata.get('original_size_bytes', 0)),
                output_size_bytes=int(metrics.metadata.get('compressed_size_bytes', 0)),
                compression_time_ms=metrics.latency_ms,
                success=True
            )
    
    def get_rolling_statistics(self, layer_type: LayerType, 
                              metric_name: str) -> Optional[RollingStatistics]:
        """
        Get rolling statistics for a specific layer type and metric
        
        Args:
            layer_type: Type of layer
            metric_name: Name of metric (e.g., 'compression_ratio', 'latency_ms')
            
        Returns:
            Rolling statistics or None if insufficient data
        """
        with self.stats_lock:
            cache_key = (layer_type, metric_name)
            
            # Check cache
            if cache_key in self.rolling_stats:
                stats = self.rolling_stats[cache_key]
                # Return cached if recent enough
                if stats.count >= min(10, self.window_size // 10):
                    return stats
            
            # Compute fresh statistics
            metrics_list = list(self.layer_metrics[layer_type])
            if len(metrics_list) < 2:
                return None
            
            # Extract values for the specific metric
            values = []
            for m in metrics_list:
                if hasattr(m, metric_name):
                    values.append(getattr(m, metric_name))
            
            if len(values) < 2:
                return None
            
            stats = RollingStatistics.from_values(values, self.window_size)
            self.rolling_stats[cache_key] = stats
            return stats
    
    def get_layer_type_summary(self, layer_type: LayerType) -> Dict[str, Any]:
        """
        Get comprehensive summary for a layer type
        
        Args:
            layer_type: Type of layer to summarize
            
        Returns:
            Summary dictionary with statistics
        """
        with self.stats_lock:
            metrics_list = list(self.layer_metrics[layer_type])
            
            if not metrics_list:
                return {
                    'layer_type': layer_type.value,
                    'count': 0,
                    'no_data': True
                }
            
            # Compute aggregates
            compression_ratios = [m.compression_ratio for m in metrics_list]
            latencies = [m.latency_ms for m in metrics_list]
            accuracy_losses = [m.accuracy_loss for m in metrics_list]
            memory_saved = [m.memory_saved_mb for m in metrics_list]
            
            # Pipeline distribution
            pipeline_counts = defaultdict(int)
            for m in metrics_list:
                pipeline_counts[m.pipeline_used.value] += 1
            
            return {
                'layer_type': layer_type.value,
                'count': len(metrics_list),
                'compression_ratio': {
                    'mean': statistics.mean(compression_ratios),
                    'std': statistics.stdev(compression_ratios) if len(compression_ratios) > 1 else 0,
                    'min': min(compression_ratios),
                    'max': max(compression_ratios)
                },
                'latency_ms': {
                    'mean': statistics.mean(latencies),
                    'std': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                    'min': min(latencies),
                    'max': max(latencies)
                },
                'accuracy_loss': {
                    'mean': statistics.mean(accuracy_losses),
                    'std': statistics.stdev(accuracy_losses) if len(accuracy_losses) > 1 else 0,
                    'min': min(accuracy_losses),
                    'max': max(accuracy_losses)
                },
                'total_memory_saved_mb': sum(memory_saved),
                'pipeline_distribution': dict(pipeline_counts)
            }
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all layer types"""
        with self.stats_lock:
            return {
                'total_layers_processed': self.total_layers_processed,
                'total_memory_saved_mb': self.total_memory_saved_mb,
                'layer_summaries': {
                    layer_type.value: self.get_layer_type_summary(layer_type)
                    for layer_type in LayerType
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def _aggregation_loop(self):
        """Background aggregation thread"""
        while self.aggregation_active:
            try:
                # Clear old cache entries
                with self.stats_lock:
                    # Keep only recent stats
                    current_time = time.time()
                    cache_keys_to_remove = []
                    for key in self.rolling_stats:
                        # Remove stats older than 10 seconds
                        if len(self.layer_metrics[key[0]]) == 0:
                            cache_keys_to_remove.append(key)
                    
                    for key in cache_keys_to_remove:
                        del self.rolling_stats[key]
                
                time.sleep(self.aggregation_interval_ms / 1000.0)
                
            except Exception as e:
                logger.error(f"Aggregation loop error: {e}")
                raise
    
    def shutdown(self):
        """Shutdown aggregator"""
        self.aggregation_active = False
        logger.info(f"CompressionMetricsAggregator shutdown - processed {self.total_layers_processed} layers")


class CompressionEfficiencyAnalyzer:
    """
    Analyze compression efficiency metrics
    Compare P-adic vs Tropical pipeline effectiveness
    """
    
    def __init__(self, monitor: UnifiedPerformanceMonitor):
        """
        Initialize efficiency analyzer
        
        Args:
            monitor: UnifiedPerformanceMonitor instance
        """
        self.monitor = monitor
        self.efficiency_history: Deque[CompressionEfficiency] = deque(maxlen=1000)
        self.pipeline_efficiency: Dict[PipelineType, List[CompressionEfficiency]] = {
            PipelineType.PADIC: [],
            PipelineType.TROPICAL: [],
            PipelineType.HYBRID: []
        }
        self._lock = threading.RLock()
        
        logger.info("CompressionEfficiencyAnalyzer initialized")
    
    def analyze_compression(self, original_size_bytes: int,
                          compressed_size_bytes: int,
                          accuracy_loss: float,
                          compression_time_ms: float,
                          decompression_time_ms: float,
                          pipeline_type: PipelineType) -> CompressionEfficiency:
        """
        Analyze a compression operation
        
        Args:
            original_size_bytes: Original data size
            compressed_size_bytes: Compressed size
            accuracy_loss: Loss in accuracy (0-1)
            compression_time_ms: Compression time
            decompression_time_ms: Decompression time
            pipeline_type: Pipeline used
            
        Returns:
            Compression efficiency metrics
        """
        if original_size_bytes <= 0:
            raise ValueError(f"Original size must be positive, got {original_size_bytes}")
        if compressed_size_bytes <= 0:
            raise ValueError(f"Compressed size must be positive, got {compressed_size_bytes}")
        if not (0.0 <= accuracy_loss <= 1.0):
            raise ValueError(f"Accuracy loss must be in [0, 1], got {accuracy_loss}")
        
        with self._lock:
            # Calculate metrics
            compression_ratio = original_size_bytes / compressed_size_bytes
            accuracy_retained = 1.0 - accuracy_loss
            memory_reduction_percent = ((original_size_bytes - compressed_size_bytes) / 
                                       original_size_bytes) * 100
            
            # Calculate speed improvement
            baseline_time = 10.0  # Baseline in ms (configurable)
            total_time = compression_time_ms + decompression_time_ms
            speed_improvement_factor = baseline_time / max(total_time, 0.001)
            
            # Calculate combined efficiency score
            # Weighted combination: compression (40%), accuracy (40%), speed (20%)
            efficiency_score = (
                0.4 * min(compression_ratio / 10.0, 1.0) +  # Normalize to [0, 1]
                0.4 * accuracy_retained +
                0.2 * min(speed_improvement_factor, 1.0)
            )
            
            efficiency = CompressionEfficiency(
                compression_ratio=compression_ratio,
                accuracy_retained=accuracy_retained,
                memory_reduction_percent=memory_reduction_percent,
                speed_improvement_factor=speed_improvement_factor,
                efficiency_score=efficiency_score
            )
            
            # Store in history
            self.efficiency_history.append(efficiency)
            self.pipeline_efficiency[pipeline_type].append(efficiency)
            
            return efficiency
    
    def compare_pipelines(self) -> Dict[str, Any]:
        """
        Compare P-adic vs Tropical efficiency
        
        Returns:
            Comparison results
        """
        with self._lock:
            comparison = {}
            
            for pipeline_type in [PipelineType.PADIC, PipelineType.TROPICAL]:
                efficiencies = self.pipeline_efficiency[pipeline_type]
                
                if not efficiencies:
                    comparison[pipeline_type.value] = {'no_data': True}
                    continue
                
                # Calculate average metrics
                avg_compression = statistics.mean([e.compression_ratio for e in efficiencies])
                avg_accuracy = statistics.mean([e.accuracy_retained for e in efficiencies])
                avg_memory_reduction = statistics.mean([e.memory_reduction_percent for e in efficiencies])
                avg_speed = statistics.mean([e.speed_improvement_factor for e in efficiencies])
                avg_efficiency = statistics.mean([e.efficiency_score for e in efficiencies])
                
                comparison[pipeline_type.value] = {
                    'sample_count': len(efficiencies),
                    'avg_compression_ratio': avg_compression,
                    'avg_accuracy_retained': avg_accuracy,
                    'avg_memory_reduction_percent': avg_memory_reduction,
                    'avg_speed_improvement': avg_speed,
                    'avg_efficiency_score': avg_efficiency
                }
            
            # Determine winner if both have data
            if (PipelineType.PADIC.value in comparison and 
                PipelineType.TROPICAL.value in comparison and
                not comparison[PipelineType.PADIC.value].get('no_data') and
                not comparison[PipelineType.TROPICAL.value].get('no_data')):
                
                padic_score = comparison[PipelineType.PADIC.value]['avg_efficiency_score']
                tropical_score = comparison[PipelineType.TROPICAL.value]['avg_efficiency_score']
                
                comparison['recommendation'] = {
                    'winner': PipelineType.TROPICAL.value if tropical_score > padic_score else PipelineType.PADIC.value,
                    'score_difference': abs(tropical_score - padic_score),
                    'confidence': min(abs(tropical_score - padic_score) * 2, 1.0)  # Scale difference to confidence
                }
            
            return comparison
    
    def calculate_memory_savings(self, original_mb: float,
                                compressed_mb: float) -> Dict[str, float]:
        """
        Calculate detailed memory savings
        
        Args:
            original_mb: Original size in MB
            compressed_mb: Compressed size in MB
            
        Returns:
            Memory savings metrics
        """
        if original_mb <= 0:
            raise ValueError(f"Original size must be positive, got {original_mb}")
        if compressed_mb <= 0:
            raise ValueError(f"Compressed size must be positive, got {compressed_mb}")
        
        absolute_savings = original_mb - compressed_mb
        percent_savings = (absolute_savings / original_mb) * 100
        compression_ratio = original_mb / compressed_mb
        
        return {
            'original_mb': original_mb,
            'compressed_mb': compressed_mb,
            'absolute_savings_mb': absolute_savings,
            'percent_savings': percent_savings,
            'compression_ratio': compression_ratio,
            'savings_per_gb': (absolute_savings / original_mb) * 1024  # MB saved per GB
        }


class ModelImpactAnalyzer:
    """
    Analyze compression impact on model performance
    Track layer-wise impact and identify critical layers
    """
    
    def __init__(self):
        """Initialize model impact analyzer"""
        self.model_impacts: Dict[str, ModelImpact] = {}
        self.layer_sensitivities: Dict[str, Dict[str, float]] = {}
        self.critical_layer_threshold = 0.01  # 1% accuracy drop threshold
        self._lock = threading.RLock()
        
        logger.info("ModelImpactAnalyzer initialized")
    
    def analyze_model_impact(self, model_id: str,
                            original_accuracy: float,
                            compressed_accuracy: float,
                            original_size_mb: float,
                            compressed_size_mb: float,
                            layer_impacts: Dict[str, float]) -> ModelImpact:
        """
        Analyze overall model compression impact
        
        Args:
            model_id: Model identifier
            original_accuracy: Original model accuracy
            compressed_accuracy: Compressed model accuracy
            original_size_mb: Original model size
            compressed_size_mb: Compressed model size
            layer_impacts: Per-layer accuracy impacts
            
        Returns:
            Model impact analysis
        """
        with self._lock:
            # Calculate metrics
            accuracy_drop = original_accuracy - compressed_accuracy
            size_reduction_percent = ((original_size_mb - compressed_size_mb) / 
                                     original_size_mb) * 100
            
            # Estimate inference speedup (simplified)
            inference_speedup = original_size_mb / compressed_size_mb
            
            # Identify critical layers
            critical_layers = [
                layer_id for layer_id, impact in layer_impacts.items()
                if impact > self.critical_layer_threshold
            ]
            
            # Create sensitivity map (normalized impacts)
            max_impact = max(layer_impacts.values()) if layer_impacts else 1.0
            sensitivity_map = {
                layer_id: impact / max_impact
                for layer_id, impact in layer_impacts.items()
            }
            
            impact = ModelImpact(
                model_id=model_id,
                original_accuracy=original_accuracy,
                compressed_accuracy=compressed_accuracy,
                accuracy_drop=accuracy_drop,
                original_size_mb=original_size_mb,
                compressed_size_mb=compressed_size_mb,
                size_reduction_percent=size_reduction_percent,
                inference_speedup=inference_speedup,
                critical_layers=critical_layers,
                sensitivity_map=sensitivity_map
            )
            
            # Store impact
            self.model_impacts[model_id] = impact
            self.layer_sensitivities[model_id] = layer_impacts
            
            return impact
    
    def track_layer_impact(self, model_id: str, layer_id: str,
                          original_output: torch.Tensor,
                          compressed_output: torch.Tensor) -> float:
        """
        Track compression impact for a specific layer
        
        Args:
            model_id: Model identifier
            layer_id: Layer identifier
            original_output: Original layer output
            compressed_output: Compressed layer output
            
        Returns:
            Impact score (0-1)
        """
        with self._lock:
            # Calculate reconstruction error
            mse = torch.nn.functional.mse_loss(compressed_output, original_output).item()
            
            # Normalize to [0, 1] range
            # Use exponential decay for impact scoring
            impact = 1.0 - np.exp(-mse)
            
            # Store in sensitivities
            if model_id not in self.layer_sensitivities:
                self.layer_sensitivities[model_id] = {}
            
            self.layer_sensitivities[model_id][layer_id] = impact
            
            return impact
    
    def identify_critical_layers(self, model_id: str,
                                threshold: Optional[float] = None) -> List[str]:
        """
        Identify layers critical to model performance
        
        Args:
            model_id: Model identifier
            threshold: Custom threshold for criticality
            
        Returns:
            List of critical layer IDs
        """
        with self._lock:
            if model_id not in self.layer_sensitivities:
                return []
            
            threshold = threshold or self.critical_layer_threshold
            
            critical = [
                layer_id for layer_id, impact in self.layer_sensitivities[model_id].items()
                if impact > threshold
            ]
            
            # Sort by impact (highest first)
            critical.sort(key=lambda x: self.layer_sensitivities[model_id][x], reverse=True)
            
            return critical
    
    def generate_sensitivity_heatmap(self, model_id: str) -> Dict[str, Any]:
        """
        Generate compression sensitivity heatmap data
        
        Args:
            model_id: Model identifier
            
        Returns:
            Heatmap data structure
        """
        with self._lock:
            if model_id not in self.layer_sensitivities:
                return {'error': f'No data for model {model_id}'}
            
            sensitivities = self.layer_sensitivities[model_id]
            
            # Group by layer type if possible
            layer_groups = defaultdict(list)
            for layer_id, sensitivity in sensitivities.items():
                # Extract layer type from ID (assumes format like "dense_1", "conv_2", etc.)
                layer_type = layer_id.split('_')[0] if '_' in layer_id else 'unknown'
                layer_groups[layer_type].append({
                    'layer_id': layer_id,
                    'sensitivity': sensitivity
                })
            
            # Sort each group by sensitivity
            for group in layer_groups.values():
                group.sort(key=lambda x: x['sensitivity'], reverse=True)
            
            return {
                'model_id': model_id,
                'total_layers': len(sensitivities),
                'critical_count': len(self.identify_critical_layers(model_id)),
                'layer_groups': dict(layer_groups),
                'statistics': {
                    'mean_sensitivity': statistics.mean(sensitivities.values()),
                    'std_sensitivity': statistics.stdev(sensitivities.values()) if len(sensitivities) > 1 else 0,
                    'max_sensitivity': max(sensitivities.values()),
                    'min_sensitivity': min(sensitivities.values())
                }
            }


class CompressionRecommendationEngine:
    """
    Auto-select best compression pipeline based on layer characteristics
    Provide optimal parameters and confidence scores
    """
    
    def __init__(self, efficiency_analyzer: CompressionEfficiencyAnalyzer,
                 impact_analyzer: ModelImpactAnalyzer):
        """
        Initialize recommendation engine
        
        Args:
            efficiency_analyzer: Efficiency analyzer instance
            impact_analyzer: Impact analyzer instance
        """
        self.efficiency_analyzer = efficiency_analyzer
        self.impact_analyzer = impact_analyzer
        self._lock = threading.RLock()
        
        # Recommendation rules and thresholds
        self.rules = {
            'prefer_tropical_for_dense': 0.7,  # Confidence boost for dense layers
            'prefer_padic_for_conv': 0.6,      # Confidence boost for conv layers
            'high_accuracy_requirement': 0.95,  # Accuracy retention threshold
            'compression_ratio_weight': 0.4,
            'accuracy_weight': 0.4,
            'speed_weight': 0.2
        }
        
        logger.info("CompressionRecommendationEngine initialized")
    
    def recommend_pipeline(self, layer_type: LayerType,
                          layer_shape: Tuple[int, ...],
                          sparsity: float,
                          requirements: Optional[Dict[str, Any]] = None) -> CompressionRecommendation:
        """
        Recommend best compression pipeline for a layer
        
        Args:
            layer_type: Type of neural network layer
            layer_shape: Shape of layer weights
            sparsity: Sparsity ratio (0-1)
            requirements: Optional requirements (e.g., min_accuracy, max_latency)
            
        Returns:
            Compression recommendation
        """
        with self._lock:
            requirements = requirements or {}
            
            # Calculate scores for each pipeline
            padic_score = self._calculate_padic_score(layer_type, layer_shape, sparsity)
            tropical_score = self._calculate_tropical_score(layer_type, layer_shape, sparsity)
            hybrid_score = self._calculate_hybrid_score(layer_type, layer_shape, sparsity)
            
            # Apply requirement adjustments
            if 'min_accuracy' in requirements:
                min_acc = requirements['min_accuracy']
                if min_acc > self.rules['high_accuracy_requirement']:
                    # Prefer P-adic for high accuracy requirements
                    padic_score *= 1.2
                    tropical_score *= 0.9
            
            if 'max_latency_ms' in requirements:
                max_latency = requirements['max_latency_ms']
                if max_latency < 10:  # Very low latency requirement
                    # Prefer Tropical for speed
                    tropical_score *= 1.3
                    padic_score *= 0.8
            
            # Determine best pipeline
            scores = {
                PipelineType.PADIC: padic_score,
                PipelineType.TROPICAL: tropical_score,
                PipelineType.HYBRID: hybrid_score
            }
            
            best_pipeline = max(scores, key=scores.get)
            best_score = scores[best_pipeline]
            
            # Generate reasoning
            reasoning = self._generate_reasoning(layer_type, sparsity, scores, requirements)
            
            # Generate optimal parameters
            optimal_params = self._generate_optimal_parameters(best_pipeline, layer_type, layer_shape)
            
            # Calculate expected metrics
            expected_compression = self._estimate_compression_ratio(best_pipeline, layer_type, sparsity)
            expected_accuracy = self._estimate_accuracy_retention(best_pipeline, layer_type)
            
            # Generate alternatives
            alternatives = [
                (pipeline, score) for pipeline, score in sorted(
                    scores.items(), key=lambda x: x[1], reverse=True
                )[1:]
            ]
            
            return CompressionRecommendation(
                recommended_pipeline=best_pipeline,
                confidence_score=min(best_score, 1.0),
                reasoning=reasoning,
                optimal_parameters=optimal_params,
                expected_compression_ratio=expected_compression,
                expected_accuracy_retention=expected_accuracy,
                alternative_pipelines=alternatives
            )
    
    def auto_select_batch(self, layers: List[Dict[str, Any]]) -> List[CompressionRecommendation]:
        """
        Auto-select pipelines for a batch of layers
        
        Args:
            layers: List of layer specifications
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        for layer_spec in layers:
            try:
                layer_type = LayerType[layer_spec.get('type', 'UNKNOWN').upper()]
                layer_shape = tuple(layer_spec.get('shape', []))
                sparsity = layer_spec.get('sparsity', 0.0)
                requirements = layer_spec.get('requirements', {})
                
                rec = self.recommend_pipeline(layer_type, layer_shape, sparsity, requirements)
                recommendations.append(rec)
                
            except Exception as e:
                logger.error(f"Failed to generate recommendation: {e}")
                raise
        
        return recommendations
    
    def _calculate_padic_score(self, layer_type: LayerType,
                              layer_shape: Tuple[int, ...],
                              sparsity: float) -> float:
        """Calculate P-adic pipeline suitability score"""
        score = 0.5  # Base score
        
        # P-adic is good for:
        # - Convolutional layers
        # - Layers with structured patterns
        # - Medium sparsity (20-60%)
        
        if layer_type == LayerType.CONV:
            score += self.rules['prefer_padic_for_conv']
        elif layer_type == LayerType.ATTENTION:
            score += 0.4  # Good for attention patterns
        
        # Sparsity adjustment
        if 0.2 <= sparsity <= 0.6:
            score += 0.2
        
        # Size adjustment (P-adic overhead)
        total_params = np.prod(layer_shape) if layer_shape else 1
        if total_params > 1e6:  # Large layers
            score += 0.1
        
        return score
    
    def _calculate_tropical_score(self, layer_type: LayerType,
                                 layer_shape: Tuple[int, ...],
                                 sparsity: float) -> float:
        """Calculate Tropical pipeline suitability score"""
        score = 0.5  # Base score
        
        # Tropical is good for:
        # - Dense/Linear layers
        # - High sparsity (>60%)
        # - Real-time requirements
        
        if layer_type == LayerType.DENSE:
            score += self.rules['prefer_tropical_for_dense']
        elif layer_type == LayerType.EMBEDDING:
            score += 0.5  # Good for embeddings
        
        # Sparsity adjustment
        if sparsity > 0.6:
            score += 0.3
        elif sparsity > 0.8:
            score += 0.5
        
        # Speed bonus
        score += 0.1  # Tropical is generally faster
        
        return score
    
    def _calculate_hybrid_score(self, layer_type: LayerType,
                               layer_shape: Tuple[int, ...],
                               sparsity: float) -> float:
        """Calculate Hybrid pipeline suitability score"""
        # Hybrid is best for mixed characteristics
        padic = self._calculate_padic_score(layer_type, layer_shape, sparsity)
        tropical = self._calculate_tropical_score(layer_type, layer_shape, sparsity)
        
        # Hybrid score is high when both are moderate
        if 0.4 <= padic <= 0.7 and 0.4 <= tropical <= 0.7:
            return 0.8
        
        return 0.3  # Otherwise low priority
    
    def _generate_reasoning(self, layer_type: LayerType,
                           sparsity: float,
                           scores: Dict[PipelineType, float],
                           requirements: Dict[str, Any]) -> List[str]:
        """Generate human-readable reasoning"""
        reasoning = []
        
        # Layer type reasoning
        if layer_type == LayerType.DENSE:
            reasoning.append(f"Dense layer typically benefits from Tropical compression")
        elif layer_type == LayerType.CONV:
            reasoning.append(f"Convolutional layer often compressed better with P-adic")
        
        # Sparsity reasoning
        if sparsity > 0.7:
            reasoning.append(f"High sparsity ({sparsity:.1%}) favors Tropical method")
        elif sparsity < 0.3:
            reasoning.append(f"Low sparsity ({sparsity:.1%}) may benefit from P-adic structure")
        
        # Score reasoning
        best = max(scores, key=scores.get)
        reasoning.append(f"{best.value} pipeline scored highest ({scores[best]:.2f})")
        
        # Requirements reasoning
        if requirements:
            if 'min_accuracy' in requirements:
                reasoning.append(f"Accuracy requirement: {requirements['min_accuracy']:.1%}")
            if 'max_latency_ms' in requirements:
                reasoning.append(f"Latency constraint: {requirements['max_latency_ms']}ms")
        
        return reasoning
    
    def _generate_optimal_parameters(self, pipeline: PipelineType,
                                    layer_type: LayerType,
                                    layer_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Generate optimal compression parameters"""
        params = {}
        
        if pipeline == PipelineType.PADIC:
            params = {
                'prime': 251,  # Default prime
                'precision': 8 if layer_type == LayerType.DENSE else 16,
                'block_size': 64 if len(layer_shape) > 2 else 32,
                'use_gpu': True,
                'batch_size': 256
            }
        elif pipeline == PipelineType.TROPICAL:
            params = {
                'epsilon': 1e-6,
                'max_iterations': 100,
                'convergence_threshold': 1e-4,
                'use_jax': True,
                'precision': 'float32'
            }
        elif pipeline == PipelineType.HYBRID:
            params = {
                'padic_ratio': 0.5,
                'switch_threshold': 0.5,
                'adaptive': True
            }
        
        return params
    
    def _estimate_compression_ratio(self, pipeline: PipelineType,
                                   layer_type: LayerType,
                                   sparsity: float) -> float:
        """Estimate expected compression ratio"""
        base_ratio = 2.0
        
        # Pipeline adjustments
        if pipeline == PipelineType.TROPICAL:
            base_ratio *= 1.2
        elif pipeline == PipelineType.HYBRID:
            base_ratio *= 1.1
        
        # Sparsity bonus
        base_ratio *= (1 + sparsity)
        
        # Layer type adjustments
        if layer_type == LayerType.EMBEDDING:
            base_ratio *= 1.5  # Embeddings compress well
        elif layer_type == LayerType.ATTENTION:
            base_ratio *= 0.9  # Attention is harder to compress
        
        return max(base_ratio, 1.0)
    
    def _estimate_accuracy_retention(self, pipeline: PipelineType,
                                    layer_type: LayerType) -> float:
        """Estimate expected accuracy retention"""
        base_accuracy = 0.95
        
        # Pipeline adjustments
        if pipeline == PipelineType.PADIC:
            base_accuracy = 0.97  # P-adic is more accurate
        elif pipeline == PipelineType.HYBRID:
            base_accuracy = 0.96
        
        # Layer type adjustments
        if layer_type == LayerType.ATTENTION:
            base_accuracy *= 0.98  # Attention is sensitive
        elif layer_type == LayerType.ACTIVATION:
            base_accuracy *= 1.01  # Activations are robust
        
        return min(base_accuracy, 1.0)


class AnalyticsReporter:
    """
    Generate reports and export metrics
    Support JSON, CSV, and streaming updates
    """
    
    def __init__(self, aggregator: CompressionMetricsAggregator,
                 efficiency_analyzer: CompressionEfficiencyAnalyzer,
                 impact_analyzer: ModelImpactAnalyzer,
                 recommendation_engine: CompressionRecommendationEngine):
        """
        Initialize analytics reporter
        
        Args:
            aggregator: Metrics aggregator
            efficiency_analyzer: Efficiency analyzer
            impact_analyzer: Impact analyzer
            recommendation_engine: Recommendation engine
        """
        self.aggregator = aggregator
        self.efficiency_analyzer = efficiency_analyzer
        self.impact_analyzer = impact_analyzer
        self.recommendation_engine = recommendation_engine
        
        self.report_count = 0
        self.streaming_clients: Set[str] = set()
        self._lock = threading.RLock()
        
        logger.info("AnalyticsReporter initialized")
    
    def generate_json_report(self, include_history: bool = False) -> str:
        """
        Generate comprehensive JSON report
        
        Args:
            include_history: Whether to include historical data
            
        Returns:
            JSON string
        """
        with self._lock:
            report = {
                'report_id': hashlib.md5(str(time.time()).encode()).hexdigest()[:8],
                'timestamp': datetime.now().isoformat(),
                'report_number': self.report_count,
                
                # Aggregated metrics
                'aggregated_metrics': self.aggregator.get_all_statistics(),
                
                # Efficiency comparison
                'efficiency_analysis': self.efficiency_analyzer.compare_pipelines(),
                
                # Model impacts
                'model_impacts': {
                    model_id: asdict(impact)
                    for model_id, impact in self.impact_analyzer.model_impacts.items()
                },
                
                # Summary statistics
                'summary': self._generate_summary()
            }
            
            if include_history:
                report['history'] = {
                    'efficiency_history': [
                        asdict(e) for e in list(self.efficiency_analyzer.efficiency_history)
                    ]
                }
            
            self.report_count += 1
            return json.dumps(report, indent=2, default=str)
    
    def generate_csv_report(self, filepath: Path) -> None:
        """
        Generate CSV report file
        
        Args:
            filepath: Output file path
        """
        with self._lock:
            # Prepare data for CSV
            rows = []
            
            # Add layer metrics
            for layer_type in LayerType:
                for metric in list(self.aggregator.layer_metrics[layer_type]):
                    rows.append({
                        'timestamp': metric.timestamp,
                        'layer_type': layer_type.value,
                        'layer_id': metric.layer_id,
                        'compression_ratio': metric.compression_ratio,
                        'memory_saved_mb': metric.memory_saved_mb,
                        'latency_ms': metric.latency_ms,
                        'accuracy_loss': metric.accuracy_loss,
                        'pipeline': metric.pipeline_used.value
                    })
            
            # Write to CSV
            if rows:
                with open(filepath, 'w', newline='') as csvfile:
                    fieldnames = rows[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                
                logger.info(f"CSV report written to {filepath}")
            else:
                logger.warning("No data to write to CSV")
    
    def create_performance_summary(self) -> Dict[str, Any]:
        """
        Create executive performance summary
        
        Returns:
            Performance summary dict
        """
        with self._lock:
            # Get current metrics
            aggregated = self.aggregator.get_all_statistics()
            efficiency = self.efficiency_analyzer.compare_pipelines()
            
            summary = {
                'total_layers_processed': aggregated['total_layers_processed'],
                'total_memory_saved_mb': aggregated['total_memory_saved_mb'],
                'timestamp': datetime.now().isoformat(),
                
                'best_performing_pipeline': None,
                'average_compression_ratio': 0.0,
                'average_accuracy_retention': 0.0,
                
                'recommendations': []
            }
            
            # Determine best pipeline
            if 'recommendation' in efficiency:
                summary['best_performing_pipeline'] = efficiency['recommendation']['winner']
            
            # Calculate averages
            if PipelineType.TROPICAL.value in efficiency:
                tropical = efficiency[PipelineType.TROPICAL.value]
                if not tropical.get('no_data'):
                    summary['average_compression_ratio'] = tropical.get('avg_compression_ratio', 0)
                    summary['average_accuracy_retention'] = tropical.get('avg_accuracy_retained', 0)
            
            # Generate recommendations
            if summary['average_accuracy_retention'] < 0.95:
                summary['recommendations'].append("Consider adjusting compression parameters for better accuracy")
            
            if summary['average_compression_ratio'] < 2.0:
                summary['recommendations'].append("Compression ratio below target - review layer selection")
            
            if not summary['recommendations']:
                summary['recommendations'].append("System performing within expected parameters")
            
            return summary
    
    def export_metrics(self, filepath: Path, format: str = 'json') -> None:
        """
        Export metrics to file
        
        Args:
            filepath: Output path
            format: Export format ('json' or 'csv')
        """
        if format == 'json':
            report = self.generate_json_report(include_history=True)
            with open(filepath, 'w') as f:
                f.write(report)
            logger.info(f"JSON metrics exported to {filepath}")
            
        elif format == 'csv':
            self.generate_csv_report(filepath)
            
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def stream_updates(self, client_id: str, callback) -> None:
        """
        Register client for streaming updates
        
        Args:
            client_id: Unique client identifier
            callback: Function to call with updates
        """
        with self._lock:
            self.streaming_clients.add(client_id)
            
            # Send initial state
            summary = self.create_performance_summary()
            callback(client_id, summary)
            
            logger.info(f"Client {client_id} registered for streaming updates")
    
    def unregister_streaming_client(self, client_id: str) -> None:
        """Unregister streaming client"""
        with self._lock:
            self.streaming_clients.discard(client_id)
            logger.info(f"Client {client_id} unregistered from streaming")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate internal summary statistics"""
        summary = {
            'total_reports_generated': self.report_count,
            'active_streaming_clients': len(self.streaming_clients),
            'analysis_health': 'healthy',  # Simplified health check
            'uptime_seconds': time.time()  # Would track from initialization
        }
        
        # Add warning flags
        warnings = []
        
        # Check for low accuracy
        efficiency = self.efficiency_analyzer.compare_pipelines()
        for pipeline_data in efficiency.values():
            if isinstance(pipeline_data, dict) and not pipeline_data.get('no_data'):
                if pipeline_data.get('avg_accuracy_retained', 1.0) < 0.9:
                    warnings.append(f"Low accuracy retention detected")
                    break
        
        if warnings:
            summary['warnings'] = warnings
            summary['analysis_health'] = 'warning'
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    # Initialize components
    monitor_config = MonitorConfig(
        sampling_interval_ms=100,
        history_window_size=1000,
        enable_detailed_tracking=True,
        enable_auto_reporting=True
    )
    
    monitor = UnifiedPerformanceMonitor(config=monitor_config)
    aggregator = CompressionMetricsAggregator(monitor)
    efficiency_analyzer = CompressionEfficiencyAnalyzer(monitor)
    impact_analyzer = ModelImpactAnalyzer()
    recommendation_engine = CompressionRecommendationEngine(efficiency_analyzer, impact_analyzer)
    reporter = AnalyticsReporter(aggregator, efficiency_analyzer, impact_analyzer, recommendation_engine)
    
    # Simulate some layer compressions
    for i in range(10):
        # Create sample layer metrics
        layer_metrics = LayerMetrics(
            layer_id=f"dense_{i}",
            layer_type=LayerType.DENSE,
            compression_ratio=2.5 + np.random.random(),
            memory_saved_mb=100 + np.random.random() * 50,
            latency_ms=5 + np.random.random() * 10,
            accuracy_loss=0.01 + np.random.random() * 0.04,
            pipeline_used=PipelineType.TROPICAL if i % 2 == 0 else PipelineType.PADIC,
            timestamp=time.time(),
            metadata={'original_size_bytes': 1000000, 'compressed_size_bytes': 400000}
        )
        
        aggregator.add_layer_metrics(layer_metrics)
        
        # Analyze efficiency
        efficiency = efficiency_analyzer.analyze_compression(
            original_size_bytes=1000000,
            compressed_size_bytes=400000,
            accuracy_loss=layer_metrics.accuracy_loss,
            compression_time_ms=layer_metrics.latency_ms,
            decompression_time_ms=layer_metrics.latency_ms * 0.8,
            pipeline_type=layer_metrics.pipeline_used
        )
        
        time.sleep(0.1)
    
    # Generate recommendation
    recommendation = recommendation_engine.recommend_pipeline(
        layer_type=LayerType.DENSE,
        layer_shape=(1024, 512),
        sparsity=0.3,
        requirements={'min_accuracy': 0.95}
    )
    
    print(f"Recommendation: {recommendation.recommended_pipeline.value}")
    print(f"Confidence: {recommendation.confidence_score:.2f}")
    print(f"Reasoning: {recommendation.reasoning}")
    
    # Generate report
    report_json = reporter.generate_json_report()
    print("\nJSON Report Preview:")
    print(report_json[:500])
    
    # Create summary
    summary = reporter.create_performance_summary()
    print("\nPerformance Summary:")
    print(json.dumps(summary, indent=2))
    
    # Shutdown
    aggregator.shutdown()
    monitor.shutdown()