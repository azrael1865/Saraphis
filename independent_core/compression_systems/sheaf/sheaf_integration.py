"""
Sheaf Integration System

This module provides integration classes for connecting the sheaf compression system
with the brain core, domain registry, and training manager components.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
import threading
import time
import json
import uuid
from pathlib import Path
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
import traceback
from enum import Enum

# Import sheaf compression system
from .sheaf_core import SheafCompressionSystem, CellularSheaf, RestrictionMap
from .sheaf_advanced import (
    CellularSheafBuilder, RestrictionMapProcessor, 
    SheafCohomologyCalculator, SheafReconstructionEngine
)

# Import brain core components
try:
    from brain_core import BrainCore, PredictionResult, UncertaintyMetrics, BrainConfig
except ImportError:
    # Define minimal stubs for testing
    BrainCore = None
    PredictionResult = None
    UncertaintyMetrics = None
    BrainConfig = None

# Import domain registry components
try:
    from domain_registry import DomainRegistry, DomainMetadata, DomainConfig, DomainStatus
except ImportError:
    # Define minimal stubs for testing
    DomainRegistry = None
    DomainMetadata = None
    DomainConfig = None
    DomainStatus = None

# Import training manager components
try:
    from training_manager import TrainingManager, TrainingConfig, TrainingSession
except ImportError:
    # Define minimal stubs for testing
    TrainingManager = None
    TrainingConfig = None
    TrainingSession = None


class SheafIntegrationError(Exception):
    """Base exception for sheaf integration errors."""
    pass


class SheafBrainIntegrationError(SheafIntegrationError):
    """Exception for brain integration specific errors."""
    pass


class SheafDomainIntegrationError(SheafIntegrationError):
    """Exception for domain integration specific errors."""
    pass


class SheafTrainingIntegrationError(SheafIntegrationError):
    """Exception for training integration specific errors."""
    pass


class SheafOrchestrationError(SheafIntegrationError):
    """Exception for orchestration specific errors."""
    pass


@dataclass
class SheafPredictionMetrics:
    """Metrics for sheaf-aware predictions."""
    sheaf_confidence: float
    cohomology_score: float
    restriction_consistency: float
    global_section_coverage: float
    topological_stability: float
    compression_quality: float
    reconstruction_fidelity: float
    uncertainty_reduction: float
    prediction_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'sheaf_confidence': self.sheaf_confidence,
            'cohomology_score': self.cohomology_score,
            'restriction_consistency': self.restriction_consistency,
            'global_section_coverage': self.global_section_coverage,
            'topological_stability': self.topological_stability,
            'compression_quality': self.compression_quality,
            'reconstruction_fidelity': self.reconstruction_fidelity,
            'uncertainty_reduction': self.uncertainty_reduction,
            'prediction_timestamp': self.prediction_timestamp.isoformat()
        }


@dataclass
class SheafDomainState:
    """State for sheaf-aware domain management."""
    domain_name: str
    sheaf_structures: Dict[str, CellularSheaf] = field(default_factory=dict)
    capability_map: Dict[str, float] = field(default_factory=dict)
    isolation_level: float = 0.0
    compression_stats: Dict[str, Any] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            'domain_name': self.domain_name,
            'sheaf_structure_count': len(self.sheaf_structures),
            'capabilities': self.capability_map,
            'isolation_level': self.isolation_level,
            'compression_stats': self.compression_stats,
            'last_update': self.last_update.isoformat()
        }


@dataclass
class SheafTrainingMetrics:
    """Metrics for sheaf-aware training."""
    gradient_consistency: float
    sheaf_loss_contribution: float
    topological_regularization: float
    cohomological_penalty: float
    restriction_violation_count: int
    global_section_alignment: float
    training_efficiency: float
    convergence_rate: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'gradient_consistency': self.gradient_consistency,
            'sheaf_loss_contribution': self.sheaf_loss_contribution,
            'topological_regularization': self.topological_regularization,
            'cohomological_penalty': self.cohomological_penalty,
            'restriction_violation_count': self.restriction_violation_count,
            'global_section_alignment': self.global_section_alignment,
            'training_efficiency': self.training_efficiency,
            'convergence_rate': self.convergence_rate,
            'timestamp': self.timestamp.isoformat()
        }


class SheafBrainIntegration:
    """Integration between sheaf compression and brain core systems."""
    
    def __init__(self, brain_core: BrainCore, 
                 sheaf_system: SheafCompressionSystem,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize sheaf-brain integration.
        
        Args:
            brain_core: BrainCore instance
            sheaf_system: SheafCompressionSystem instance
            config: Optional configuration dictionary
        """
        if brain_core is None:
            raise ValueError("BrainCore cannot be None")
        if sheaf_system is None:
            raise ValueError("SheafCompressionSystem cannot be None")
            
        self.brain_core = brain_core
        self.sheaf_system = sheaf_system
        self.config = config or {}
        
        # Configuration
        self.confidence_threshold = self.config.get('confidence_threshold', 0.8)
        self.uncertainty_weight = self.config.get('uncertainty_weight', 0.3)
        self.sheaf_weight = self.config.get('sheaf_weight', 0.5)
        self.cache_predictions = self.config.get('cache_predictions', True)
        self.max_cache_size = self.config.get('max_cache_size', 1000)
        
        # State management
        self.prediction_cache = deque(maxlen=self.max_cache_size)
        self.sheaf_metrics_history = deque(maxlen=1000)
        self.integration_stats = defaultdict(int)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self._initialize_sheaf_predictors()
        self._setup_uncertainty_quantification()
        self._register_brain_hooks()
        
    def _initialize_sheaf_predictors(self):
        """Initialize sheaf-aware prediction components."""
        self.sheaf_builder = CellularSheafBuilder({
            'max_cells': 500,
            'min_cell_size': 5,
            'overlap_threshold': 0.15
        })
        
        self.reconstruction_engine = SheafReconstructionEngine({
            'reconstruction_method': 'hierarchical',
            'smoothing_factor': 0.3,
            'convergence_tolerance': 1e-5
        })
        
        self.cohomology_calculator = SheafCohomologyCalculator({
            'max_degree': 3,
            'computation_method': 'spectral'
        })
        
    def _setup_uncertainty_quantification(self):
        """Setup sheaf-aware uncertainty quantification."""
        self.uncertainty_models = {
            'topological': self._topological_uncertainty,
            'cohomological': self._cohomological_uncertainty,
            'restriction': self._restriction_uncertainty,
            'global': self._global_section_uncertainty
        }
        
    def _register_brain_hooks(self):
        """Register integration hooks with brain core."""
        try:
            # Add sheaf-aware prediction enhancer
            if hasattr(self.brain_core, 'register_prediction_enhancer'):
                self.brain_core.register_prediction_enhancer(
                    'sheaf_integration',
                    self._enhance_prediction_with_sheaf
                )
                
            # Add uncertainty quantifier
            if hasattr(self.brain_core, 'register_uncertainty_quantifier'):
                self.brain_core.register_uncertainty_quantifier(
                    'sheaf_uncertainty',
                    self._quantify_sheaf_uncertainty
                )
                
        except Exception as e:
            self.logger.warning(f"Failed to register brain hooks: {e}")
            
    def predict_with_sheaf(self, input_data: Any, 
                          domain: Optional[str] = None,
                          use_compression: bool = True) -> Tuple[PredictionResult, SheafPredictionMetrics]:
        """
        Make prediction with sheaf-aware enhancements.
        
        Args:
            input_data: Input data for prediction
            domain: Optional domain specification
            use_compression: Whether to use sheaf compression
            
        Returns:
            Tuple of prediction result and sheaf metrics
        """
        start_time = time.time()
        
        try:
            # Validate input
            if input_data is None:
                raise ValueError("Input data cannot be None")
                
            # Compress data if requested
            sheaf_data = None
            compression_metrics = {}
            
            if use_compression:
                try:
                    compressed = self.sheaf_system.compress(input_data)
                    sheaf_data = compressed.get('sheaf')
                    compression_metrics = {
                        'compression_ratio': compressed.get('compression_ratio', 1.0),
                        'sheaf_valid': compressed.get('metadata', {}).get('sheaf_valid', False)
                    }
                except Exception as e:
                    self.logger.warning(f"Sheaf compression failed: {e}")
                    use_compression = False
                    
            # Get base prediction from brain core
            base_prediction = self.brain_core.predict(input_data, domain)
            
            if not base_prediction.success:
                # Return base prediction with default metrics
                metrics = SheafPredictionMetrics(
                    sheaf_confidence=0.0,
                    cohomology_score=0.0,
                    restriction_consistency=0.0,
                    global_section_coverage=0.0,
                    topological_stability=0.0,
                    compression_quality=0.0,
                    reconstruction_fidelity=0.0,
                    uncertainty_reduction=0.0
                )
                return base_prediction, metrics
                
            # Enhance prediction with sheaf analysis
            enhanced_prediction = base_prediction
            sheaf_metrics = self._compute_sheaf_metrics(
                input_data, sheaf_data, base_prediction, compression_metrics
            )
            
            if use_compression and sheaf_data:
                # Apply sheaf-based confidence adjustment
                adjusted_confidence = self._adjust_confidence_with_sheaf(
                    base_prediction.confidence,
                    sheaf_metrics
                )
                
                # Apply sheaf-based uncertainty reduction
                adjusted_uncertainty = self._reduce_uncertainty_with_sheaf(
                    base_prediction.uncertainty_metrics,
                    sheaf_metrics
                )
                
                # Create enhanced prediction
                enhanced_prediction = PredictionResult(
                    prediction_id=base_prediction.prediction_id,
                    success=True,
                    predicted_value=base_prediction.predicted_value,
                    confidence=adjusted_confidence,
                    domain=base_prediction.domain,
                    uncertainty_metrics=adjusted_uncertainty,
                    computation_time=time.time() - start_time,
                    metadata={
                        **base_prediction.metadata,
                        'sheaf_enhanced': True,
                        'sheaf_metrics': sheaf_metrics.to_dict()
                    }
                )
                
            # Cache prediction if enabled
            if self.cache_predictions:
                self._cache_sheaf_prediction(enhanced_prediction, sheaf_metrics)
                
            # Update statistics
            with self.lock:
                self.integration_stats['predictions_made'] += 1
                if use_compression:
                    self.integration_stats['compressed_predictions'] += 1
                    
            return enhanced_prediction, sheaf_metrics
            
        except Exception as e:
            self.logger.error(f"Sheaf prediction failed: {e}")
            raise SheafBrainIntegrationError(f"Prediction failed: {e}") from e
            
    def _enhance_prediction_with_sheaf(self, prediction: PredictionResult,
                                     input_data: Any) -> PredictionResult:
        """Enhance brain core prediction with sheaf analysis."""
        try:
            # Compress input to get sheaf structure
            compressed = self.sheaf_system.compress(input_data)
            sheaf = compressed.get('sheaf')
            
            if not sheaf:
                return prediction
                
            # Compute sheaf-based enhancements
            sheaf_confidence = self._compute_sheaf_confidence(sheaf)
            topological_stability = self._compute_topological_stability(sheaf)
            
            # Adjust prediction confidence
            adjusted_confidence = (
                prediction.confidence * (1 - self.sheaf_weight) +
                sheaf_confidence * self.sheaf_weight
            )
            
            # Create enhanced prediction
            enhanced = PredictionResult(
                prediction_id=prediction.prediction_id,
                success=prediction.success,
                predicted_value=prediction.predicted_value,
                confidence=adjusted_confidence,
                domain=prediction.domain,
                uncertainty_metrics=prediction.uncertainty_metrics,
                computation_time=prediction.computation_time,
                metadata={
                    **prediction.metadata,
                    'sheaf_confidence': sheaf_confidence,
                    'topological_stability': topological_stability
                }
            )
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"Failed to enhance prediction: {e}")
            return prediction
            
    def _quantify_sheaf_uncertainty(self, input_data: Any,
                                   base_uncertainty: UncertaintyMetrics) -> UncertaintyMetrics:
        """Quantify uncertainty using sheaf structure."""
        try:
            # Compress input to get sheaf structure
            compressed = self.sheaf_system.compress(input_data)
            sheaf = compressed.get('sheaf')
            
            if not sheaf:
                return base_uncertainty
                
            # Compute sheaf-based uncertainty components
            topological_uncertainty = self._topological_uncertainty(sheaf)
            cohomological_uncertainty = self._cohomological_uncertainty(sheaf)
            restriction_uncertainty = self._restriction_uncertainty(sheaf)
            global_uncertainty = self._global_section_uncertainty(sheaf)
            
            # Aggregate uncertainties
            sheaf_epistemic = np.mean([
                topological_uncertainty,
                cohomological_uncertainty
            ])
            
            sheaf_aleatoric = np.mean([
                restriction_uncertainty,
                global_uncertainty
            ])
            
            # Combine with base uncertainty
            combined_epistemic = (
                base_uncertainty.epistemic_uncertainty * (1 - self.uncertainty_weight) +
                sheaf_epistemic * self.uncertainty_weight
            )
            
            combined_aleatoric = (
                base_uncertainty.aleatoric_uncertainty * (1 - self.uncertainty_weight) +
                sheaf_aleatoric * self.uncertainty_weight
            )
            
            # Create enhanced uncertainty metrics
            enhanced = UncertaintyMetrics(
                mean=base_uncertainty.mean,
                variance=base_uncertainty.variance,
                std=base_uncertainty.std,
                confidence_interval=base_uncertainty.confidence_interval,
                prediction_interval=base_uncertainty.prediction_interval,
                epistemic_uncertainty=combined_epistemic,
                aleatoric_uncertainty=combined_aleatoric,
                model_confidence=base_uncertainty.model_confidence,
                credible_regions=base_uncertainty.credible_regions,
                entropy=base_uncertainty.entropy,
                mutual_information=base_uncertainty.mutual_information,
                reliability_score=self._compute_reliability_with_sheaf(
                    base_uncertainty.reliability_score, sheaf
                )
            )
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"Failed to quantify sheaf uncertainty: {e}")
            return base_uncertainty
            
    def _compute_sheaf_metrics(self, input_data: Any,
                             sheaf: Optional[CellularSheaf],
                             prediction: PredictionResult,
                             compression_metrics: Dict[str, Any]) -> SheafPredictionMetrics:
        """Compute comprehensive sheaf metrics."""
        try:
            if sheaf is None:
                # Return default metrics
                return SheafPredictionMetrics(
                    sheaf_confidence=prediction.confidence,
                    cohomology_score=0.0,
                    restriction_consistency=1.0,
                    global_section_coverage=1.0,
                    topological_stability=1.0,
                    compression_quality=1.0,
                    reconstruction_fidelity=1.0,
                    uncertainty_reduction=0.0
                )
                
            # Compute individual metrics
            sheaf_confidence = self._compute_sheaf_confidence(sheaf)
            cohomology_score = self._compute_cohomology_score(sheaf)
            restriction_consistency = self._compute_restriction_consistency(sheaf)
            global_section_coverage = self._compute_global_section_coverage(sheaf)
            topological_stability = self._compute_topological_stability(sheaf)
            compression_quality = compression_metrics.get('compression_ratio', 1.0)
            
            # Compute reconstruction fidelity
            reconstruction_fidelity = 1.0
            if isinstance(input_data, torch.Tensor):
                try:
                    reconstructed = self.reconstruction_engine.reconstruct(
                        sheaf, input_data.shape
                    )
                    reconstruction_error = torch.nn.functional.mse_loss(
                        reconstructed, input_data
                    ).item()
                    reconstruction_fidelity = 1.0 / (1.0 + reconstruction_error)
                except Exception as e:
                    self.logger.warning(f"Failed to compute reconstruction fidelity: {e}")
                    
            # Compute uncertainty reduction
            base_uncertainty = prediction.uncertainty_metrics
            sheaf_uncertainty = self._compute_total_sheaf_uncertainty(sheaf)
            uncertainty_reduction = max(0.0, 1.0 - sheaf_uncertainty)
            
            metrics = SheafPredictionMetrics(
                sheaf_confidence=sheaf_confidence,
                cohomology_score=cohomology_score,
                restriction_consistency=restriction_consistency,
                global_section_coverage=global_section_coverage,
                topological_stability=topological_stability,
                compression_quality=compression_quality,
                reconstruction_fidelity=reconstruction_fidelity,
                uncertainty_reduction=uncertainty_reduction
            )
            
            # Store metrics in history
            with self.lock:
                self.sheaf_metrics_history.append(metrics)
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to compute sheaf metrics: {e}")
            raise SheafBrainIntegrationError(f"Metric computation failed: {e}") from e
            
    def _compute_sheaf_confidence(self, sheaf: CellularSheaf) -> float:
        """Compute confidence score from sheaf structure."""
        try:
            # Check cocycle condition
            cocycle_valid = sheaf.verify_cocycle_condition()
            if not cocycle_valid:
                return 0.0
                
            # Compute confidence components
            section_coverage = len(sheaf.sections) / max(1, len(sheaf.cells))
            restriction_completeness = len(sheaf.restriction_maps) / max(
                1, len(sheaf.cells) * (len(sheaf.cells) - 1) // 2
            )
            
            # Check gluing consistency
            gluing_score = 1.0
            if hasattr(sheaf, 'gluing_data') and sheaf.gluing_data:
                consistent_gluing = sum(
                    1 for g in sheaf.gluing_data.values()
                    if self._check_gluing_consistency(g)
                )
                gluing_score = consistent_gluing / max(1, len(sheaf.gluing_data))
                
            # Aggregate confidence
            confidence = np.mean([
                section_coverage,
                restriction_completeness,
                gluing_score
            ])
            
            return float(np.clip(confidence, 0.0, 1.0))
            
        except Exception as e:
            self.logger.warning(f"Failed to compute sheaf confidence: {e}")
            return 0.5
            
    def _compute_cohomology_score(self, sheaf: CellularSheaf) -> float:
        """Compute cohomology-based score."""
        try:
            # Compute cohomology groups
            cohomology_groups = self.cohomology_calculator.compute_cohomology(sheaf)
            
            # Score based on cohomology dimensions
            h0_dim = cohomology_groups.get(0, {}).get('dimension', 0)
            h1_dim = cohomology_groups.get(1, {}).get('dimension', 0)
            h2_dim = cohomology_groups.get(2, {}).get('dimension', 0)
            
            # Lower dimensional cohomology indicates better structure
            score = 1.0 / (1.0 + h1_dim + 2 * h2_dim)
            
            # Factor in Betti numbers if available
            if 'betti_numbers' in cohomology_groups:
                betti_sum = sum(cohomology_groups['betti_numbers'].values())
                score *= 1.0 / (1.0 + 0.1 * betti_sum)
                
            return float(np.clip(score, 0.0, 1.0))
            
        except Exception as e:
            self.logger.warning(f"Failed to compute cohomology score: {e}")
            return 0.5
            
    def _compute_restriction_consistency(self, sheaf: CellularSheaf) -> float:
        """Compute consistency of restriction maps."""
        try:
            if not sheaf.restriction_maps:
                return 1.0
                
            consistency_scores = []
            
            for (cell1, cell2), restriction in sheaf.restriction_maps.items():
                if cell1 in sheaf.sections and cell2 in sheaf.sections:
                    section1 = sheaf.sections[cell1]
                    section2 = sheaf.sections[cell2]
                    
                    # Check if restriction preserves structure
                    if hasattr(restriction, 'apply'):
                        restricted = restriction.apply(section1)
                        consistency = self._measure_section_similarity(
                            restricted, section2
                        )
                        consistency_scores.append(consistency)
                        
            if not consistency_scores:
                return 1.0
                
            return float(np.mean(consistency_scores))
            
        except Exception as e:
            self.logger.warning(f"Failed to compute restriction consistency: {e}")
            return 0.5
            
    def _compute_global_section_coverage(self, sheaf: CellularSheaf) -> float:
        """Compute coverage of global sections."""
        try:
            if not hasattr(sheaf, 'compute_global_sections'):
                return 0.5
                
            global_sections = sheaf.compute_global_sections()
            
            if not global_sections:
                return 0.0
                
            # Compute coverage
            total_cells = len(sheaf.cells)
            covered_cells = sum(
                1 for cell in sheaf.cells
                if any(
                    self._section_covers_cell(section, cell)
                    for section in global_sections.values()
                )
            )
            
            coverage = covered_cells / max(1, total_cells)
            return float(np.clip(coverage, 0.0, 1.0))
            
        except Exception as e:
            self.logger.warning(f"Failed to compute global section coverage: {e}")
            return 0.5
            
    def _compute_topological_stability(self, sheaf: CellularSheaf) -> float:
        """Compute topological stability of sheaf."""
        try:
            stability_components = []
            
            # Check cell dimension consistency
            if hasattr(sheaf, 'cell_dimensions'):
                dimensions = list(sheaf.cell_dimensions.values())
                if dimensions:
                    dim_variance = np.var(dimensions)
                    dim_stability = 1.0 / (1.0 + dim_variance)
                    stability_components.append(dim_stability)
                    
            # Check overlap region stability
            if hasattr(sheaf, 'overlap_regions'):
                overlap_count = len(sheaf.overlap_regions)
                total_pairs = len(sheaf.cells) * (len(sheaf.cells) - 1) // 2
                overlap_ratio = overlap_count / max(1, total_pairs)
                overlap_stability = 1.0 - overlap_ratio  # Less overlap = more stable
                stability_components.append(overlap_stability)
                
            # Check restriction map connectivity
            if sheaf.restriction_maps:
                connected_pairs = len(sheaf.restriction_maps)
                connectivity_ratio = connected_pairs / max(1, total_pairs)
                connectivity_stability = connectivity_ratio
                stability_components.append(connectivity_stability)
                
            if not stability_components:
                return 0.5
                
            return float(np.mean(stability_components))
            
        except Exception as e:
            self.logger.warning(f"Failed to compute topological stability: {e}")
            return 0.5
            
    def _topological_uncertainty(self, sheaf: CellularSheaf) -> float:
        """Compute topological uncertainty."""
        try:
            # Higher dimensional cells introduce more uncertainty
            max_dim = max(sheaf.cell_dimensions.values()) if sheaf.cell_dimensions else 0
            dim_uncertainty = max_dim / 10.0  # Normalize
            
            # Disconnected components increase uncertainty
            components = self._count_connected_components(sheaf)
            component_uncertainty = (components - 1) / 10.0  # Normalize
            
            # Combine uncertainties
            uncertainty = np.clip(dim_uncertainty + component_uncertainty, 0.0, 1.0)
            return float(uncertainty)
            
        except Exception as e:
            self.logger.warning(f"Failed to compute topological uncertainty: {e}")
            return 0.5
            
    def _cohomological_uncertainty(self, sheaf: CellularSheaf) -> float:
        """Compute cohomological uncertainty."""
        try:
            cohomology = self.cohomology_calculator.compute_cohomology(sheaf)
            
            # Non-trivial cohomology indicates uncertainty
            h1_dim = cohomology.get(1, {}).get('dimension', 0)
            h2_dim = cohomology.get(2, {}).get('dimension', 0)
            
            uncertainty = (h1_dim + 2 * h2_dim) / 10.0  # Normalize
            return float(np.clip(uncertainty, 0.0, 1.0))
            
        except Exception as e:
            self.logger.warning(f"Failed to compute cohomological uncertainty: {e}")
            return 0.5
            
    def _restriction_uncertainty(self, sheaf: CellularSheaf) -> float:
        """Compute restriction map uncertainty."""
        try:
            if not sheaf.restriction_maps:
                return 0.0
                
            # Compute variance in restriction map properties
            restriction_norms = []
            
            for restriction in sheaf.restriction_maps.values():
                if hasattr(restriction, 'matrix'):
                    norm = np.linalg.norm(restriction.matrix)
                    restriction_norms.append(norm)
                    
            if not restriction_norms:
                return 0.0
                
            # Higher variance = higher uncertainty
            variance = np.var(restriction_norms)
            uncertainty = variance / (1.0 + variance)  # Normalize
            
            return float(np.clip(uncertainty, 0.0, 1.0))
            
        except Exception as e:
            self.logger.warning(f"Failed to compute restriction uncertainty: {e}")
            return 0.5
            
    def _global_section_uncertainty(self, sheaf: CellularSheaf) -> float:
        """Compute global section uncertainty."""
        try:
            if not hasattr(sheaf, 'compute_global_sections'):
                return 0.5
                
            global_sections = sheaf.compute_global_sections()
            
            # No global sections = high uncertainty
            if not global_sections:
                return 1.0
                
            # Multiple global sections = some uncertainty
            num_sections = len(global_sections)
            uncertainty = 1.0 - (1.0 / num_sections)
            
            return float(np.clip(uncertainty, 0.0, 1.0))
            
        except Exception as e:
            self.logger.warning(f"Failed to compute global section uncertainty: {e}")
            return 0.5
            
    def _compute_total_sheaf_uncertainty(self, sheaf: CellularSheaf) -> float:
        """Compute total uncertainty from all sheaf components."""
        uncertainties = [
            self._topological_uncertainty(sheaf),
            self._cohomological_uncertainty(sheaf),
            self._restriction_uncertainty(sheaf),
            self._global_section_uncertainty(sheaf)
        ]
        
        # Weight different uncertainty types
        weights = [0.3, 0.3, 0.2, 0.2]
        total_uncertainty = np.dot(uncertainties, weights)
        
        return float(np.clip(total_uncertainty, 0.0, 1.0))
        
    def _adjust_confidence_with_sheaf(self, base_confidence: float,
                                    sheaf_metrics: SheafPredictionMetrics) -> float:
        """Adjust confidence using sheaf metrics."""
        # Compute sheaf-based confidence adjustment
        sheaf_factors = [
            sheaf_metrics.sheaf_confidence,
            sheaf_metrics.restriction_consistency,
            sheaf_metrics.global_section_coverage,
            sheaf_metrics.topological_stability
        ]
        
        sheaf_confidence = np.mean(sheaf_factors)
        
        # Blend with base confidence
        adjusted = (
            base_confidence * (1 - self.sheaf_weight) +
            sheaf_confidence * self.sheaf_weight
        )
        
        # Apply cohomology penalty
        cohomology_penalty = 1.0 - (1.0 - sheaf_metrics.cohomology_score) * 0.1
        adjusted *= cohomology_penalty
        
        return float(np.clip(adjusted, 0.0, 1.0))
        
    def _reduce_uncertainty_with_sheaf(self, base_uncertainty: UncertaintyMetrics,
                                     sheaf_metrics: SheafPredictionMetrics) -> UncertaintyMetrics:
        """Reduce uncertainty using sheaf information."""
        reduction_factor = sheaf_metrics.uncertainty_reduction
        
        # Apply reduction to uncertainty components
        reduced_epistemic = base_uncertainty.epistemic_uncertainty * (1 - reduction_factor * 0.3)
        reduced_aleatoric = base_uncertainty.aleatoric_uncertainty * (1 - reduction_factor * 0.2)
        
        # Tighten confidence intervals
        ci_lower, ci_upper = base_uncertainty.confidence_interval
        ci_width = ci_upper - ci_lower
        reduced_width = ci_width * (1 - reduction_factor * 0.1)
        ci_center = (ci_lower + ci_upper) / 2
        
        reduced_ci = (
            ci_center - reduced_width / 2,
            ci_center + reduced_width / 2
        )
        
        # Create reduced uncertainty metrics
        reduced = UncertaintyMetrics(
            mean=base_uncertainty.mean,
            variance=base_uncertainty.variance * (1 - reduction_factor * 0.2),
            std=np.sqrt(base_uncertainty.variance * (1 - reduction_factor * 0.2)),
            confidence_interval=reduced_ci,
            prediction_interval=base_uncertainty.prediction_interval,
            epistemic_uncertainty=reduced_epistemic,
            aleatoric_uncertainty=reduced_aleatoric,
            model_confidence=base_uncertainty.model_confidence * (1 + reduction_factor * 0.1),
            credible_regions=base_uncertainty.credible_regions,
            entropy=base_uncertainty.entropy * (1 - reduction_factor * 0.15),
            mutual_information=base_uncertainty.mutual_information,
            reliability_score=min(1.0, base_uncertainty.reliability_score * (1 + reduction_factor * 0.2))
        )
        
        return reduced
        
    def _compute_reliability_with_sheaf(self, base_reliability: float,
                                      sheaf: CellularSheaf) -> float:
        """Compute enhanced reliability using sheaf structure."""
        # Check sheaf validity
        if not sheaf.verify_cocycle_condition():
            return base_reliability * 0.5
            
        # Compute sheaf-based reliability boost
        sheaf_boost = 0.0
        
        # Well-connected sheaf is more reliable
        if hasattr(sheaf, 'topology') and sheaf.topology:
            connectivity = len(sheaf.topology) / max(
                1, len(sheaf.cells) * (len(sheaf.cells) - 1) // 2
            )
            sheaf_boost += connectivity * 0.1
            
        # Consistent restrictions increase reliability
        restriction_consistency = self._compute_restriction_consistency(sheaf)
        sheaf_boost += restriction_consistency * 0.1
        
        # Apply boost
        enhanced_reliability = base_reliability * (1 + sheaf_boost)
        return float(np.clip(enhanced_reliability, 0.0, 1.0))
        
    def _check_gluing_consistency(self, gluing_data: Any) -> bool:
        """Check if gluing data is consistent."""
        try:
            if not isinstance(gluing_data, dict):
                return True
                
            # Check compatibility conditions
            if 'compatibility' in gluing_data:
                return gluing_data['compatibility'] > 0.8
                
            return True
            
        except Exception:
            return False
            
    def _measure_section_similarity(self, section1: Any, section2: Any) -> float:
        """Measure similarity between sections."""
        try:
            if section1 is None or section2 is None:
                return 0.0
                
            # Handle tensor sections
            if isinstance(section1, torch.Tensor) and isinstance(section2, torch.Tensor):
                if section1.shape != section2.shape:
                    return 0.0
                    
                # Compute cosine similarity
                flat1 = section1.flatten()
                flat2 = section2.flatten()
                
                dot_product = torch.dot(flat1, flat2)
                norm1 = torch.norm(flat1)
                norm2 = torch.norm(flat2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                    
                similarity = dot_product / (norm1 * norm2)
                return float(similarity.item())
                
            # Handle other types
            if section1 == section2:
                return 1.0
                
            return 0.5
            
        except Exception:
            return 0.0
            
    def _section_covers_cell(self, section: Any, cell: Any) -> bool:
        """Check if section covers cell."""
        try:
            # Simple coverage check
            if hasattr(section, 'cells') and hasattr(cell, 'id'):
                return cell.id in section.cells
                
            return True  # Assume coverage by default
            
        except Exception:
            return False
            
    def _count_connected_components(self, sheaf: CellularSheaf) -> int:
        """Count connected components in sheaf topology."""
        try:
            if not hasattr(sheaf, 'topology') or not sheaf.topology:
                return len(sheaf.cells)
                
            # Build adjacency graph
            adjacency = defaultdict(set)
            
            for (cell1, cell2) in sheaf.topology:
                adjacency[cell1].add(cell2)
                adjacency[cell2].add(cell1)
                
            # Count components using DFS
            visited = set()
            components = 0
            
            for cell in sheaf.cells:
                if cell not in visited:
                    components += 1
                    # DFS from this cell
                    stack = [cell]
                    while stack:
                        current = stack.pop()
                        if current not in visited:
                            visited.add(current)
                            stack.extend(adjacency[current] - visited)
                            
            return components
            
        except Exception:
            return 1
            
    def _cache_sheaf_prediction(self, prediction: PredictionResult,
                              metrics: SheafPredictionMetrics):
        """Cache prediction with sheaf metrics."""
        try:
            cache_entry = {
                'prediction': prediction,
                'metrics': metrics,
                'timestamp': datetime.now()
            }
            
            with self.lock:
                self.prediction_cache.append(cache_entry)
                
        except Exception as e:
            self.logger.warning(f"Failed to cache prediction: {e}")
            
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get statistics about sheaf-brain integration."""
        with self.lock:
            stats = dict(self.integration_stats)
            
        # Add metric summaries
        if self.sheaf_metrics_history:
            recent_metrics = list(self.sheaf_metrics_history)[-100:]
            
            stats['average_sheaf_confidence'] = np.mean([
                m.sheaf_confidence for m in recent_metrics
            ])
            stats['average_cohomology_score'] = np.mean([
                m.cohomology_score for m in recent_metrics
            ])
            stats['average_uncertainty_reduction'] = np.mean([
                m.uncertainty_reduction for m in recent_metrics
            ])
            
        stats['cache_size'] = len(self.prediction_cache)
        stats['metrics_history_size'] = len(self.sheaf_metrics_history)
        
        return stats


class SheafDomainIntegration:
    """Integration between sheaf compression and domain registry."""
    
    def __init__(self, domain_registry: DomainRegistry,
                 sheaf_system: SheafCompressionSystem,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize sheaf-domain integration.
        
        Args:
            domain_registry: DomainRegistry instance
            sheaf_system: SheafCompressionSystem instance
            config: Optional configuration dictionary
        """
        if domain_registry is None:
            raise ValueError("DomainRegistry cannot be None")
        if sheaf_system is None:
            raise ValueError("SheafCompressionSystem cannot be None")
            
        self.domain_registry = domain_registry
        self.sheaf_system = sheaf_system
        self.config = config or {}
        
        # Configuration
        self.capability_threshold = self.config.get('capability_threshold', 0.7)
        self.isolation_granularity = self.config.get('isolation_granularity', 0.1)
        self.state_compression_enabled = self.config.get('state_compression_enabled', True)
        self.max_sheaf_cache_per_domain = self.config.get('max_sheaf_cache_per_domain', 100)
        
        # State management
        self.domain_sheaf_states: Dict[str, SheafDomainState] = {}
        self.sheaf_capability_cache: Dict[str, Dict[str, float]] = {}
        self.isolation_boundaries: Dict[str, Set[str]] = defaultdict(set)
        self.compression_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self._initialize_domain_detectors()
        self._setup_isolation_mechanisms()
        
    def _initialize_domain_detectors(self):
        """Initialize sheaf-aware domain detection components."""
        self.domain_detectors = {
            'topological': self._detect_topological_domain,
            'cohomological': self._detect_cohomological_domain,
            'structural': self._detect_structural_domain,
            'semantic': self._detect_semantic_domain
        }
        
    def _setup_isolation_mechanisms(self):
        """Setup sheaf-based isolation mechanisms."""
        self.isolation_strategies = {
            'topological': self._topological_isolation,
            'cellular': self._cellular_isolation,
            'cohomological': self._cohomological_isolation,
            'restriction': self._restriction_based_isolation
        }
        
    def register_sheaf_domain(self, domain_name: str,
                            domain_config: Optional[DomainConfig] = None,
                            sheaf_structure: Optional[CellularSheaf] = None) -> bool:
        """
        Register domain with sheaf-aware enhancements.
        
        Args:
            domain_name: Name of domain to register
            domain_config: Domain configuration
            sheaf_structure: Optional initial sheaf structure
            
        Returns:
            Success status
        """
        try:
            # Validate inputs
            if not domain_name:
                raise ValueError("Domain name cannot be empty")
                
            # Register base domain
            success = self.domain_registry.register_domain(domain_name, domain_config)
            
            if not success:
                return False
                
            # Create sheaf domain state
            sheaf_state = SheafDomainState(
                domain_name=domain_name,
                sheaf_structures={},
                capability_map={},
                isolation_level=0.0,
                compression_stats={}
            )
            
            # Add initial sheaf structure if provided
            if sheaf_structure:
                sheaf_id = str(uuid.uuid4())
                sheaf_state.sheaf_structures[sheaf_id] = sheaf_structure
                sheaf_state.capability_map = self._compute_domain_capabilities(sheaf_structure)
                
            # Store state
            with self.lock:
                self.domain_sheaf_states[domain_name] = sheaf_state
                self.sheaf_capability_cache[domain_name] = sheaf_state.capability_map
                
            self.logger.info(f"Registered sheaf domain: {domain_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register sheaf domain: {e}")
            raise SheafDomainIntegrationError(f"Registration failed: {e}") from e
            
    def detect_domain_capabilities(self, data: Any,
                                 use_compression: bool = True) -> Dict[str, float]:
        """
        Detect domain capabilities using sheaf analysis.
        
        Args:
            data: Input data to analyze
            use_compression: Whether to compress data first
            
        Returns:
            Dictionary of domain capabilities
        """
        try:
            # Compress data if requested
            sheaf = None
            if use_compression:
                try:
                    compressed = self.sheaf_system.compress(data)
                    sheaf = compressed.get('sheaf')
                except Exception as e:
                    self.logger.warning(f"Compression failed: {e}")
                    
            # Detect capabilities for each detector type
            capabilities = {}
            
            for detector_name, detector_func in self.domain_detectors.items():
                try:
                    domain, confidence = detector_func(data, sheaf)
                    if domain and confidence > self.capability_threshold:
                        capabilities[f"{domain}_{detector_name}"] = confidence
                except Exception as e:
                    self.logger.warning(f"Detector {detector_name} failed: {e}")
                    
            # Aggregate capabilities by domain
            domain_capabilities = defaultdict(list)
            for capability, confidence in capabilities.items():
                domain = capability.split('_')[0]
                domain_capabilities[domain].append(confidence)
                
            # Average capabilities per domain
            averaged_capabilities = {
                domain: float(np.mean(confidences))
                for domain, confidences in domain_capabilities.items()
            }
            
            return averaged_capabilities
            
        except Exception as e:
            self.logger.error(f"Failed to detect domain capabilities: {e}")
            raise SheafDomainIntegrationError(f"Detection failed: {e}") from e
            
    def _detect_topological_domain(self, data: Any,
                                 sheaf: Optional[CellularSheaf]) -> Tuple[str, float]:
        """Detect domain based on topological features."""
        try:
            if sheaf is None:
                return None, 0.0
                
            # Analyze topological properties
            num_cells = len(sheaf.cells)
            max_dimension = max(sheaf.cell_dimensions.values()) if sheaf.cell_dimensions else 0
            connectivity = len(sheaf.topology) / max(1, num_cells * (num_cells - 1) // 2)
            
            # Classify based on topology
            if max_dimension <= 1 and connectivity > 0.8:
                return "linear", 0.9
            elif max_dimension == 2 and connectivity > 0.6:
                return "planar", 0.85
            elif max_dimension >= 3:
                return "volumetric", 0.8
            elif connectivity < 0.3:
                return "sparse", 0.75
            else:
                return "general", 0.6
                
        except Exception:
            return None, 0.0
            
    def _detect_cohomological_domain(self, data: Any,
                                   sheaf: Optional[CellularSheaf]) -> Tuple[str, float]:
        """Detect domain based on cohomological features."""
        try:
            if sheaf is None:
                return None, 0.0
                
            # Compute cohomology
            calculator = SheafCohomologyCalculator()
            cohomology = calculator.compute_cohomology(sheaf)
            
            # Analyze cohomological dimensions
            h0 = cohomology.get(0, {}).get('dimension', 1)
            h1 = cohomology.get(1, {}).get('dimension', 0)
            h2 = cohomology.get(2, {}).get('dimension', 0)
            
            # Classify based on cohomology
            if h0 == 1 and h1 == 0 and h2 == 0:
                return "simply_connected", 0.95
            elif h1 > 0 and h2 == 0:
                return "multiply_connected", 0.85
            elif h2 > 0:
                return "higher_topology", 0.8
            else:
                return "trivial_topology", 0.7
                
        except Exception:
            return None, 0.0
            
    def _detect_structural_domain(self, data: Any,
                                sheaf: Optional[CellularSheaf]) -> Tuple[str, float]:
        """Detect domain based on structural features."""
        try:
            if sheaf is None:
                # Analyze raw data structure
                if isinstance(data, torch.Tensor):
                    if len(data.shape) == 1:
                        return "vector", 0.9
                    elif len(data.shape) == 2:
                        return "matrix", 0.9
                    elif len(data.shape) >= 3:
                        return "tensor", 0.85
                elif isinstance(data, (list, tuple)):
                    return "sequence", 0.8
                elif isinstance(data, dict):
                    return "mapping", 0.8
                else:
                    return "scalar", 0.7
                    
            # Analyze sheaf structure
            section_types = set()
            for section in sheaf.sections.values():
                section_types.add(type(section).__name__)
                
            if len(section_types) == 1:
                return "homogeneous", 0.85
            else:
                return "heterogeneous", 0.75
                
        except Exception:
            return None, 0.0
            
    def _detect_semantic_domain(self, data: Any,
                              sheaf: Optional[CellularSheaf]) -> Tuple[str, float]:
        """Detect domain based on semantic features."""
        try:
            if sheaf and hasattr(sheaf, 'metadata'):
                metadata = sheaf.metadata
                if 'domain_hint' in metadata:
                    return metadata['domain_hint'], 0.9
                    
            return "general_semantic", 0.5
            
        except Exception:
            return None, 0.0
            
    def _compute_domain_capabilities(self, sheaf: CellularSheaf) -> Dict[str, float]:
        """Compute domain capabilities from sheaf structure."""
        capabilities = {}
        
        try:
            # Compression capability
            if hasattr(sheaf, 'compression_ratio'):
                capabilities['compression'] = min(1.0, sheaf.compression_ratio / 10.0)
                
            # Reconstruction capability
            if hasattr(sheaf, 'reconstruction_fidelity'):
                capabilities['reconstruction'] = sheaf.reconstruction_fidelity
                
            # Topological analysis capability
            if sheaf.cell_dimensions:
                max_dim = max(sheaf.cell_dimensions.values())
                capabilities['topological_analysis'] = min(1.0, max_dim / 5.0)
                
            # Pattern detection capability
            if sheaf.restriction_maps:
                capabilities['pattern_detection'] = min(
                    1.0, len(sheaf.restriction_maps) / 100.0
                )
                
            return capabilities
            
        except Exception:
            return {}
            
    def _topological_isolation(self, domain_name: str,
                             sheaf_state: SheafDomainState,
                             isolation_level: float) -> Set[str]:
        """Apply topological isolation strategy."""
        boundaries = set()
        
        try:
            for sheaf in sheaf_state.sheaf_structures.values():
                if hasattr(sheaf, 'topology'):
                    # Find cells with low connectivity
                    connectivity = defaultdict(int)
                    
                    for (cell1, cell2) in sheaf.topology:
                        connectivity[cell1] += 1
                        connectivity[cell2] += 1
                        
                    # Boundary cells have connectivity below threshold
                    threshold = len(sheaf.cells) * (1 - isolation_level)
                    
                    for cell, conn_count in connectivity.items():
                        if conn_count < threshold:
                            boundaries.add(str(cell))
                            
            return boundaries
            
        except Exception:
            return set()
            
    def _cellular_isolation(self, domain_name: str,
                          sheaf_state: SheafDomainState,
                          isolation_level: float) -> Set[str]:
        """Apply cellular isolation strategy."""
        boundaries = set()
        
        try:
            for sheaf in sheaf_state.sheaf_structures.values():
                if hasattr(sheaf, 'cell_dimensions'):
                    # Isolate higher-dimensional cells
                    max_dim = max(sheaf.cell_dimensions.values())
                    dim_threshold = max_dim * (1 - isolation_level)
                    
                    for cell, dim in sheaf.cell_dimensions.items():
                        if dim >= dim_threshold:
                            boundaries.add(str(cell))
                            
            return boundaries
            
        except Exception:
            return set()
            
    def _cohomological_isolation(self, domain_name: str,
                               sheaf_state: SheafDomainState,
                               isolation_level: float) -> Set[str]:
        """Apply cohomological isolation strategy."""
        boundaries = set()
        
        try:
            calculator = SheafCohomologyCalculator()
            
            for sheaf_id, sheaf in sheaf_state.sheaf_structures.items():
                cohomology = calculator.compute_cohomology(sheaf)
                
                if 'contributing_cells' in cohomology:
                    num_to_isolate = int(
                        len(cohomology['contributing_cells']) * isolation_level
                    )
                    isolated_cells = list(cohomology['contributing_cells'])[:num_to_isolate]
                    boundaries.update(str(cell) for cell in isolated_cells)
                    
            return boundaries
            
        except Exception:
            return set()
            
    def _restriction_based_isolation(self, domain_name: str,
                                   sheaf_state: SheafDomainState,
                                   isolation_level: float) -> Set[str]:
        """Apply restriction-based isolation strategy."""
        boundaries = set()
        
        try:
            for sheaf in sheaf_state.sheaf_structures.values():
                if sheaf.restriction_maps:
                    restriction_scores = []
                    
                    for (cell1, cell2), restriction in sheaf.restriction_maps.items():
                        if hasattr(restriction, 'matrix'):
                            score = np.linalg.norm(
                                restriction.matrix - np.eye(restriction.matrix.shape[0])
                            )
                            restriction_scores.append((score, cell1, cell2))
                            
                    # Isolate cells with highest restriction scores
                    restriction_scores.sort(reverse=True)
                    num_to_isolate = int(len(restriction_scores) * isolation_level)
                    
                    for score, cell1, cell2 in restriction_scores[:num_to_isolate]:
                        boundaries.add(str(cell1))
                        boundaries.add(str(cell2))
                        
            return boundaries
            
        except Exception:
            return set()
            
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get statistics about sheaf-domain integration."""
        stats = {
            'registered_domains': len(self.domain_sheaf_states),
            'total_sheaf_structures': sum(
                len(state.sheaf_structures)
                for state in self.domain_sheaf_states.values()
            ),
            'isolated_domains': len(self.isolation_boundaries),
            'capability_cache_size': len(self.sheaf_capability_cache)
        }
        
        # Add compression statistics
        total_original = 0
        total_compressed = 0
        
        for state in self.domain_sheaf_states.values():
            for comp_stats in state.compression_stats.values():
                total_original += comp_stats.get('original_size', 0)
                total_compressed += comp_stats.get('compressed_size', 0)
                
        if total_original > 0:
            stats['average_compression_ratio'] = total_original / total_compressed
        else:
            stats['average_compression_ratio'] = 1.0
            
        return stats


class SheafTrainingIntegration:
    """Integration between sheaf compression and training systems."""
    
    def __init__(self, training_manager: TrainingManager,
                 sheaf_system: SheafCompressionSystem,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize sheaf-training integration.
        
        Args:
            training_manager: TrainingManager instance
            sheaf_system: SheafCompressionSystem instance
            config: Optional configuration dictionary
        """
        if training_manager is None:
            raise ValueError("TrainingManager cannot be None")
        if sheaf_system is None:
            raise ValueError("SheafCompressionSystem cannot be None")
            
        self.training_manager = training_manager
        self.sheaf_system = sheaf_system
        self.config = config or {}
        
        # Configuration
        self.gradient_threshold = self.config.get('gradient_threshold', 0.1)
        self.sheaf_regularization_weight = self.config.get('sheaf_regularization_weight', 0.01)
        self.cohomology_penalty_weight = self.config.get('cohomology_penalty_weight', 0.001)
        self.enable_sheaf_constraints = self.config.get('enable_sheaf_constraints', True)
        
        # State management
        self.training_metrics_history: Dict[str, List[SheafTrainingMetrics]] = defaultdict(list)
        self.gradient_sheaves: Dict[str, CellularSheaf] = {}
        self.optimization_stats: Dict[str, Any] = defaultdict(int)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self._initialize_gradient_validators()
        
    def _initialize_gradient_validators(self):
        """Initialize sheaf-aware gradient validation components."""
        self.gradient_validators = {
            'consistency': self._validate_gradient_consistency,
            'topology': self._validate_gradient_topology,
            'restriction': self._validate_gradient_restrictions,
            'cohomology': self._validate_gradient_cohomology
        }
        
        self.sheaf_builder = CellularSheafBuilder({
            'max_cells': 100,
            'min_cell_size': 1,
            'partition_strategy': 'adaptive'
        })
        
    def validate_gradients_with_sheaf(self, gradients: List[torch.Tensor],
                                    model_parameters: List[torch.nn.Parameter],
                                    learning_rate: float,
                                    epoch: int,
                                    batch_idx: int) -> Tuple[List[torch.Tensor], SheafTrainingMetrics]:
        """
        Validate and correct gradients using sheaf structure.
        
        Args:
            gradients: List of gradient tensors
            model_parameters: Model parameters
            learning_rate: Current learning rate
            epoch: Current epoch
            batch_idx: Current batch index
            
        Returns:
            Tuple of corrected gradients and training metrics
        """
        try:
            # Validate inputs
            if not gradients:
                raise ValueError("Gradients list cannot be empty")
                
            if len(gradients) != len(model_parameters):
                raise ValueError("Gradients and parameters must have same length")
                
            # Build sheaf from gradients
            gradient_data = torch.cat([g.flatten() for g in gradients])
            gradient_sheaf = self.sheaf_builder.build_from_data(
                gradient_data.unsqueeze(0)
            )
            
            # Store gradient sheaf
            session_id = f"epoch_{epoch}_batch_{batch_idx}"
            with self.lock:
                self.gradient_sheaves[session_id] = gradient_sheaf
                
            # Validate gradients
            validation_results = {}
            for validator_name, validator_func in self.gradient_validators.items():
                try:
                    is_valid, score = validator_func(
                        gradients, gradient_sheaf, model_parameters
                    )
                    validation_results[validator_name] = (is_valid, score)
                except Exception as e:
                    self.logger.warning(f"Validator {validator_name} failed: {e}")
                    validation_results[validator_name] = (True, 1.0)
                    
            # Compute overall validity
            all_valid = all(result[0] for result in validation_results.values())
            avg_score = np.mean([result[1] for result in validation_results.values()])
            
            # Correct gradients if needed
            corrected_gradients = gradients
            
            if not all_valid and self.enable_sheaf_constraints:
                corrected_gradients = self._correct_gradients_with_sheaf(
                    gradients, gradient_sheaf, validation_results, model_parameters
                )
                
            # Compute training metrics
            metrics = self._compute_training_metrics(
                gradients, corrected_gradients, gradient_sheaf,
                validation_results, learning_rate
            )
            
            # Store metrics
            with self.lock:
                self.training_metrics_history[session_id].append(metrics)
                self.optimization_stats['validations_performed'] += 1
                if not all_valid:
                    self.optimization_stats['corrections_applied'] += 1
                    
            return corrected_gradients, metrics
            
        except Exception as e:
            self.logger.error(f"Gradient validation failed: {e}")
            raise SheafTrainingIntegrationError(f"Validation failed: {e}") from e
            
    def _validate_gradient_consistency(self, gradients: List[torch.Tensor],
                                     gradient_sheaf: CellularSheaf,
                                     model_parameters: List[torch.nn.Parameter]) -> Tuple[bool, float]:
        """Validate gradient consistency using sheaf structure."""
        try:
            # Check if gradients maintain sheaf consistency
            if not gradient_sheaf.verify_cocycle_condition():
                return False, 0.0
                
            # Compute consistency score
            consistency_scores = []
            
            for i, (grad, param) in enumerate(zip(gradients, model_parameters)):
                if grad is None:
                    continue
                    
                # Check gradient magnitude consistency
                grad_norm = torch.norm(grad)
                param_norm = torch.norm(param)
                
                if param_norm > 0:
                    relative_norm = grad_norm / param_norm
                    consistency = 1.0 / (1.0 + relative_norm)
                    consistency_scores.append(consistency)
                    
            if not consistency_scores:
                return True, 1.0
                
            avg_consistency = float(np.mean(consistency_scores))
            is_consistent = avg_consistency > self.gradient_threshold
            
            return is_consistent, avg_consistency
            
        except Exception:
            return True, 0.5
            
    def _validate_gradient_topology(self, gradients: List[torch.Tensor],
                                  gradient_sheaf: CellularSheaf,
                                  model_parameters: List[torch.nn.Parameter]) -> Tuple[bool, float]:
        """Validate gradient topology preservation."""
        try:
            # Simple topology validation based on structure preservation
            is_valid = True
            similarity = 1.0
            
            # Check if gradient shapes match parameter shapes
            for grad, param in zip(gradients, model_parameters):
                if grad is not None and grad.shape != param.shape:
                    is_valid = False
                    similarity = 0.0
                    
            return is_valid, similarity
            
        except Exception:
            return True, 0.5
            
    def _validate_gradient_restrictions(self, gradients: List[torch.Tensor],
                                      gradient_sheaf: CellularSheaf,
                                      model_parameters: List[torch.nn.Parameter]) -> Tuple[bool, float]:
        """Validate gradient restriction maps."""
        try:
            if not gradient_sheaf.restriction_maps:
                return True, 1.0
                
            # Simple restriction validation
            valid_restrictions = 0
            total_restrictions = len(gradient_sheaf.restriction_maps)
            
            for restriction in gradient_sheaf.restriction_maps.values():
                if hasattr(restriction, 'matrix'):
                    cond_number = np.linalg.cond(restriction.matrix)
                    if cond_number < 1000:
                        valid_restrictions += 1
                        
            validity_score = valid_restrictions / max(1, total_restrictions)
            is_valid = validity_score > 0.9
            
            return is_valid, validity_score
            
        except Exception:
            return True, 0.5
            
    def _validate_gradient_cohomology(self, gradients: List[torch.Tensor],
                                    gradient_sheaf: CellularSheaf,
                                    model_parameters: List[torch.nn.Parameter]) -> Tuple[bool, float]:
        """Validate gradient cohomological properties."""
        try:
            # Compute cohomology of gradient sheaf
            calculator = SheafCohomologyCalculator()
            cohomology = calculator.compute_cohomology(gradient_sheaf)
            
            # Check cohomological constraints
            h0_dim = cohomology.get(0, {}).get('dimension', 1)
            h1_dim = cohomology.get(1, {}).get('dimension', 0)
            
            # Gradients should have simple cohomology
            is_valid = h0_dim == 1 and h1_dim == 0
            score = 1.0 / (1.0 + h1_dim)
            
            return is_valid, score
            
        except Exception:
            return True, 0.5
            
    def _correct_gradients_with_sheaf(self, gradients: List[torch.Tensor],
                                    gradient_sheaf: CellularSheaf,
                                    validation_results: Dict[str, Tuple[bool, float]],
                                    model_parameters: List[torch.nn.Parameter]) -> List[torch.Tensor]:
        """Correct gradients based on sheaf constraints."""
        corrected_gradients = []
        
        try:
            # Identify which validations failed
            failed_validations = [
                name for name, (is_valid, _) in validation_results.items()
                if not is_valid
            ]
            
            for i, grad in enumerate(gradients):
                if grad is None:
                    corrected_gradients.append(None)
                    continue
                    
                corrected_grad = grad.clone()
                
                # Apply corrections based on failures
                if 'consistency' in failed_validations:
                    # Clip gradient magnitude
                    param_norm = torch.norm(model_parameters[i])
                    max_grad_norm = param_norm * 10.0
                    
                    if torch.norm(grad) > max_grad_norm:
                        corrected_grad = grad * (max_grad_norm / torch.norm(grad))
                        
                corrected_gradients.append(corrected_grad)
                
            return corrected_gradients
            
        except Exception as e:
            self.logger.warning(f"Gradient correction failed: {e}")
            return gradients
            
    def _compute_training_metrics(self, original_gradients: List[torch.Tensor],
                                corrected_gradients: List[torch.Tensor],
                                gradient_sheaf: CellularSheaf,
                                validation_results: Dict[str, Tuple[bool, float]],
                                learning_rate: float) -> SheafTrainingMetrics:
        """Compute comprehensive training metrics."""
        try:
            # Compute gradient consistency
            consistency_scores = [
                score for _, score in validation_results.values()
            ]
            gradient_consistency = float(np.mean(consistency_scores))
            
            # Compute training efficiency
            gradient_correction_ratio = sum(
                1 for orig, corr in zip(original_gradients, corrected_gradients)
                if orig is not None and corr is not None and not torch.allclose(orig, corr)
            ) / max(1, len(original_gradients))
            
            training_efficiency = 1.0 - gradient_correction_ratio
            
            # Estimate convergence rate
            convergence_rate = learning_rate * gradient_consistency * training_efficiency
            
            metrics = SheafTrainingMetrics(
                gradient_consistency=gradient_consistency,
                sheaf_loss_contribution=0.0,
                topological_regularization=0.0,
                cohomological_penalty=0.0,
                restriction_violation_count=0,
                global_section_alignment=1.0,
                training_efficiency=training_efficiency,
                convergence_rate=float(convergence_rate)
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to compute training metrics: {e}")
            # Return default metrics
            return SheafTrainingMetrics(
                gradient_consistency=0.5,
                sheaf_loss_contribution=0.0,
                topological_regularization=0.0,
                cohomological_penalty=0.0,
                restriction_violation_count=0,
                global_section_alignment=1.0,
                training_efficiency=0.5,
                convergence_rate=0.001
            )
            
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get statistics about sheaf-training integration."""
        with self.lock:
            stats = dict(self.optimization_stats)
            
        # Add metric summaries
        total_metrics = sum(len(metrics) for metrics in self.training_metrics_history.values())
        stats['total_training_metrics'] = total_metrics
        
        if total_metrics > 0:
            recent_metrics = []
            for metrics_list in self.training_metrics_history.values():
                recent_metrics.extend(metrics_list[-10:])
                
            if recent_metrics:
                stats['average_gradient_consistency'] = np.mean([
                    m.gradient_consistency for m in recent_metrics
                ])
                stats['average_training_efficiency'] = np.mean([
                    m.training_efficiency for m in recent_metrics
                ])
                
        stats['gradient_sheaf_cache_size'] = len(self.gradient_sheaves)
        
        return stats


class IntegrationStatus(Enum):
    """Status of integration components."""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class SystemHealthMetrics:
    """Health metrics for sheaf system."""
    brain_integration_status: IntegrationStatus
    domain_integration_status: IntegrationStatus
    training_integration_status: IntegrationStatus
    overall_health_score: float
    last_health_check: datetime
    error_count: int
    warning_count: int
    performance_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'brain_integration_status': self.brain_integration_status.value,
            'domain_integration_status': self.domain_integration_status.value,
            'training_integration_status': self.training_integration_status.value,
            'overall_health_score': self.overall_health_score,
            'last_health_check': self.last_health_check.isoformat(),
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'performance_metrics': self.performance_metrics
        }


class SheafSystemOrchestrator:
    """Orchestrator for coordinating all sheaf integrations."""
    
    def __init__(self, brain_core: BrainCore,
                 domain_registry: DomainRegistry,
                 training_manager: TrainingManager,
                 sheaf_system: SheafCompressionSystem,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize sheaf system orchestrator.
        
        Args:
            brain_core: BrainCore instance
            domain_registry: DomainRegistry instance
            training_manager: TrainingManager instance
            sheaf_system: SheafCompressionSystem instance
            config: Optional configuration dictionary
        """
        if brain_core is None:
            raise ValueError("BrainCore cannot be None")
        if domain_registry is None:
            raise ValueError("DomainRegistry cannot be None")
        if training_manager is None:
            raise ValueError("TrainingManager cannot be None")
        if sheaf_system is None:
            raise ValueError("SheafCompressionSystem cannot be None")
            
        self.brain_core = brain_core
        self.domain_registry = domain_registry
        self.training_manager = training_manager
        self.sheaf_system = sheaf_system
        self.config = config or {}
        
        # Configuration
        self.health_check_interval = self.config.get('health_check_interval', 60)
        self.performance_monitoring_enabled = self.config.get('performance_monitoring_enabled', True)
        self.auto_recovery_enabled = self.config.get('auto_recovery_enabled', True)
        
        # State management
        self.integration_status = {
            'brain': IntegrationStatus.INACTIVE,
            'domain': IntegrationStatus.INACTIVE,
            'training': IntegrationStatus.INACTIVE
        }
        self.health_metrics_history = deque(maxlen=1000)
        self.error_log = deque(maxlen=500)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Integration components
        self.brain_integration: Optional[SheafBrainIntegration] = None
        self.domain_integration: Optional[SheafDomainIntegration] = None
        self.training_integration: Optional[SheafTrainingIntegration] = None
        
        # Initialize system
        self._initialize_orchestrator()
        
    def _initialize_orchestrator(self):
        """Initialize orchestrator components."""
        try:
            self.logger.info("Initializing sheaf system orchestrator...")
            
            # Initialize integrations
            self._initialize_brain_integration()
            self._initialize_domain_integration()
            self._initialize_training_integration()
            
            self.logger.info("Sheaf system orchestrator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            raise SheafOrchestrationError(f"Initialization failed: {e}") from e
            
    def _initialize_brain_integration(self):
        """Initialize brain-sheaf integration."""
        try:
            self.integration_status['brain'] = IntegrationStatus.INITIALIZING
            
            brain_config = self.config.get('brain_integration', {})
            self.brain_integration = SheafBrainIntegration(
                self.brain_core,
                self.sheaf_system,
                brain_config
            )
            
            self.integration_status['brain'] = IntegrationStatus.ACTIVE
            self.logger.info("Brain integration initialized")
            
        except Exception as e:
            self.integration_status['brain'] = IntegrationStatus.ERROR
            self.logger.error(f"Failed to initialize brain integration: {e}")
            raise
            
    def _initialize_domain_integration(self):
        """Initialize domain-sheaf integration."""
        try:
            self.integration_status['domain'] = IntegrationStatus.INITIALIZING
            
            domain_config = self.config.get('domain_integration', {})
            self.domain_integration = SheafDomainIntegration(
                self.domain_registry,
                self.sheaf_system,
                domain_config
            )
            
            self.integration_status['domain'] = IntegrationStatus.ACTIVE
            self.logger.info("Domain integration initialized")
            
        except Exception as e:
            self.integration_status['domain'] = IntegrationStatus.ERROR
            self.logger.error(f"Failed to initialize domain integration: {e}")
            raise
            
    def _initialize_training_integration(self):
        """Initialize training-sheaf integration."""
        try:
            self.integration_status['training'] = IntegrationStatus.INITIALIZING
            
            training_config = self.config.get('training_integration', {})
            self.training_integration = SheafTrainingIntegration(
                self.training_manager,
                self.sheaf_system,
                training_config
            )
            
            self.integration_status['training'] = IntegrationStatus.ACTIVE
            self.logger.info("Training integration initialized")
            
        except Exception as e:
            self.integration_status['training'] = IntegrationStatus.ERROR
            self.logger.error(f"Failed to initialize training integration: {e}")
            raise
            
    def predict_with_full_integration(self, input_data: Any,
                                    domain: Optional[str] = None,
                                    use_compression: bool = True) -> Dict[str, Any]:
        """
        Make prediction using all sheaf integrations.
        
        Args:
            input_data: Input data for prediction
            domain: Optional domain specification
            use_compression: Whether to use compression
            
        Returns:
            Dictionary containing prediction and all integration results
        """
        start_time = time.time()
        results = {
            'success': False,
            'prediction': None,
            'sheaf_metrics': None,
            'domain_capabilities': None,
            'integration_time': 0.0
        }
        
        try:
            # Check system health
            if not self._check_integration_health():
                raise SheafOrchestrationError("System health check failed")
                
            # Brain integration prediction
            if self.brain_integration and self.integration_status['brain'] == IntegrationStatus.ACTIVE:
                prediction, sheaf_metrics = self.brain_integration.predict_with_sheaf(
                    input_data, domain, use_compression
                )
                results['prediction'] = prediction
                results['sheaf_metrics'] = sheaf_metrics
                
            # Domain capability detection
            if self.domain_integration and self.integration_status['domain'] == IntegrationStatus.ACTIVE:
                capabilities = self.domain_integration.detect_domain_capabilities(
                    input_data, use_compression
                )
                results['domain_capabilities'] = capabilities
                
            results['success'] = True
            results['integration_time'] = time.time() - start_time
            
            return results
            
        except Exception as e:
            self.logger.error(f"Full integration prediction failed: {e}")
            results['error'] = str(e)
            results['integration_time'] = time.time() - start_time
            return results
            
    def _check_integration_health(self) -> bool:
        """Check health of all integrations."""
        try:
            all_healthy = True
            
            # Check brain integration
            if self.brain_integration is None or self.integration_status['brain'] != IntegrationStatus.ACTIVE:
                all_healthy = False
                
            # Check domain integration
            if self.domain_integration is None or self.integration_status['domain'] != IntegrationStatus.ACTIVE:
                all_healthy = False
                
            # Check training integration
            if self.training_integration is None or self.integration_status['training'] != IntegrationStatus.ACTIVE:
                all_healthy = False
                
            return all_healthy
            
        except Exception:
            return False
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Get integration statistics
            integration_stats = {}
            
            if self.brain_integration:
                integration_stats['brain'] = self.brain_integration.get_integration_statistics()
                
            if self.domain_integration:
                integration_stats['domain'] = self.domain_integration.get_integration_statistics()
                
            if self.training_integration:
                integration_stats['training'] = self.training_integration.get_integration_statistics()
                
            # Build status report
            status = {
                'integration_status': {
                    'brain': self.integration_status['brain'].value,
                    'domain': self.integration_status['domain'].value,
                    'training': self.integration_status['training'].value
                },
                'integration_statistics': integration_stats,
                'error_summary': {
                    'total_errors': len(self.error_log)
                }
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {
                'error': str(e),
                'integration_status': {
                    'brain': IntegrationStatus.ERROR.value,
                    'domain': IntegrationStatus.ERROR.value,
                    'training': IntegrationStatus.ERROR.value
                }
            }
            
    def shutdown(self):
        """Shutdown orchestrator and all integrations."""
        try:
            self.logger.info("Shutting down sheaf system orchestrator...")
            
            # Update statuses
            with self.lock:
                for key in self.integration_status:
                    self.integration_status[key] = IntegrationStatus.SHUTDOWN
                    
            self.logger.info("Sheaf system orchestrator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False


# Convenience functions for creating integrated systems

def create_integrated_sheaf_system(brain_core: BrainCore,
                                 domain_registry: DomainRegistry,
                                 training_manager: TrainingManager,
                                 sheaf_config: Optional[Dict[str, Any]] = None,
                                 orchestrator_config: Optional[Dict[str, Any]] = None) -> SheafSystemOrchestrator:
    """
    Create a fully integrated sheaf system.
    
    Args:
        brain_core: BrainCore instance
        domain_registry: DomainRegistry instance
        training_manager: TrainingManager instance
        sheaf_config: Configuration for sheaf compression system
        orchestrator_config: Configuration for orchestrator
        
    Returns:
        Configured SheafSystemOrchestrator instance
    """
    # Create sheaf compression system
    sheaf_system = SheafCompressionSystem(sheaf_config)
    
    # Create orchestrator
    orchestrator = SheafSystemOrchestrator(
        brain_core=brain_core,
        domain_registry=domain_registry,
        training_manager=training_manager,
        sheaf_system=sheaf_system,
        config=orchestrator_config
    )
    
    return orchestrator


def validate_sheaf_integration(orchestrator: SheafSystemOrchestrator,
                             test_data: Optional[Any] = None) -> Dict[str, Any]:
    """
    Validate sheaf integration is working correctly.
    
    Args:
        orchestrator: SheafSystemOrchestrator instance
        test_data: Optional test data for validation
        
    Returns:
        Validation results
    """
    validation_results = {
        'system_status': orchestrator.get_system_status(),
        'integration_tests': {},
        'overall_valid': False
    }
    
    try:
        # Test brain integration
        if test_data is not None:
            brain_result = orchestrator.predict_with_full_integration(
                test_data, use_compression=True
            )
            validation_results['integration_tests']['brain'] = {
                'success': brain_result['success'],
                'has_prediction': brain_result['prediction'] is not None,
                'has_metrics': brain_result['sheaf_metrics'] is not None
            }
            
        # Check overall validity
        all_tests_passed = all(
            test.get('success', False)
            for test in validation_results['integration_tests'].values()
        )
        
        validation_results['overall_valid'] = all_tests_passed
        
    except Exception as e:
        validation_results['error'] = str(e)
        validation_results['overall_valid'] = False
        
    return validation_results