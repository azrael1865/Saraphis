#!/usr/bin/env python3
"""
Brain Golf Connector - Integration between golf domain and Saraphis Brain system.
Implements DomainInterface for proper Brain system integration.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import threading
import time

from ..brain import BrainCore
from ..domain_registry import DomainInterface, DomainData, DomainRequest, DomainResponse
from .domain_config import GolfDomainConfig
from .enhanced_golf_core_main import GolfPredictionResult, GolfAnalysisContext


@dataclass
class GolfDomainData(DomainData):
    """Golf-specific domain data."""
    tournament_id: str
    player_data: List[Dict[str, Any]]
    course_conditions: Dict[str, Any]
    weather_data: Dict[str, Any]
    historical_performance: Dict[str, List[float]]
    salary_cap: float
    lineup_constraints: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Brain processing."""
        return {
            'tournament_id': self.tournament_id,
            'player_count': len(self.player_data),
            'salary_cap': self.salary_cap,
            'lineup_size': sum(self.lineup_constraints.values()),
            'weather_conditions': self.weather_data,
            'course_difficulty': self.course_conditions.get('difficulty_rating', 'medium')
        }


@dataclass
class GolfPredictionRequest(DomainRequest):
    """Golf prediction request."""
    tournament_id: str
    analysis_context: GolfAnalysisContext
    prediction_type: str = 'lineup_optimization'  # 'lineup_optimization', 'player_projections', 'risk_analysis'
    request_options: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary."""
        return {
            'tournament_id': self.tournament_id,
            'prediction_type': self.prediction_type,
            'salary_cap': self.analysis_context.salary_cap,
            'lineup_size': self.analysis_context.lineup_size,
            'options': self.request_options
        }


class BrainGolfConnector(DomainInterface):
    """
    Connector that integrates golf domain with Saraphis Brain system.
    Implements DomainInterface for proper orchestration and coordination.
    """
    
    def __init__(self, brain_core: BrainCore, config: GolfDomainConfig, gpu_optimizer=None):
        """Initialize brain golf connector."""
        self.brain_core = brain_core
        self.config = config
        self.gpu_optimizer = gpu_optimizer
        self.logger = logging.getLogger('BrainGolfConnector')
        
        # Domain interface properties
        self.domain_name = config.brain_integration.domain_name
        self.domain_type = "specialized"
        self.capabilities = {
            'lineup_optimization',
            'player_projections', 
            'risk_analysis',
            'tournament_prediction',
            'ensemble_modeling'
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Components
        self.enhanced_golf_core = None
        
        # State tracking
        self.is_initialized = False
        self.request_count = 0
        self.last_prediction_time = None
        
        # Performance metrics
        self.performance_metrics = {
            'total_requests': 0,
            'successful_predictions': 0,
            'average_response_time': 0.0,
            'error_count': 0,
            'cache_hit_rate': 0.0
        }
        
        # Brain integration features
        self.orchestrator_enabled = config.brain_integration.use_orchestrators
        self.proof_strategies_enabled = config.brain_integration.use_proof_strategies
        self.uncertainty_enabled = config.brain_integration.enable_uncertainty_quantification
        
        self.logger.info(f"Brain Golf Connector initialized for domain: {self.domain_name}")
    
    async def initialize(self) -> bool:
        """Initialize the connector and register with brain system."""
        if self.is_initialized:
            return True
        
        try:
            self.logger.info("Initializing Brain Golf Connector...")
            
            # Initialize enhanced golf core
            from .enhanced_golf_core_main import EnhancedGolfCore
            self.enhanced_golf_core = EnhancedGolfCore(
                config=self.config,
                brain_core=self.brain_core,
                gpu_optimizer=self.gpu_optimizer
            )
            await self.enhanced_golf_core.initialize()
            
            # Register with brain's domain registry
            if self.brain_core and hasattr(self.brain_core, 'domain_registry'):
                domain_config = self.config.get_saraphis_domain_config()
                registration_success = await self.brain_core.domain_registry.register_domain(
                    domain_name=self.domain_name,
                    domain_interface=self,
                    domain_config=domain_config
                )
                
                if not registration_success:
                    raise RuntimeError("Failed to register with brain domain registry")
            
            self.is_initialized = True
            self.logger.info("Brain Golf Connector initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Brain Golf Connector: {e}")
            return False
    
    async def process_request(self, request: DomainRequest) -> DomainResponse:
        """Process domain request through brain system."""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        self.request_count += 1
        self.performance_metrics['total_requests'] += 1
        
        try:
            # Validate request
            if not isinstance(request, GolfPredictionRequest):
                raise ValueError("Invalid request type for golf domain")
            
            # Use brain orchestrators if enabled
            if self.orchestrator_enabled and self.brain_core:
                response = await self._process_with_orchestrators(request)
            else:
                response = await self._process_direct(request)
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time, success=True)
            
            self.last_prediction_time = datetime.now()
            self.logger.info(f"Processed golf prediction request in {processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to process golf request: {e}")
            self.performance_metrics['error_count'] += 1
            
            # Return error response
            return DomainResponse(
                success=False,
                data={},
                metadata={
                    'error': str(e),
                    'request_id': getattr(request, 'request_id', 'unknown'),
                    'processing_time': time.time() - start_time
                }
            )
    
    async def _process_with_orchestrators(self, request: GolfPredictionRequest) -> DomainResponse:
        """Process request using brain orchestrators."""
        try:
            # Create orchestration task
            task_data = {
                'domain': self.domain_name,
                'request_type': request.prediction_type,
                'tournament_id': request.tournament_id,
                'context': request.analysis_context.__dict__
            }
            
            # Use brain orchestrator if available
            if hasattr(self.brain_core, 'orchestrator'):
                orchestration_result = await self.brain_core.orchestrator.execute_task(
                    task_id=f"golf_{request.tournament_id}_{int(time.time())}",
                    task_type="prediction",
                    task_data=task_data,
                    priority="normal",
                    timeout=self.config.brain_integration.orchestrator_timeout
                )
                
                if orchestration_result and orchestration_result.success:
                    # Process the orchestrated request
                    golf_result = await self.enhanced_golf_core.predict_lineup(request.analysis_context)
                    
                    # Add orchestration metadata
                    return DomainResponse(
                        success=True,
                        data=self._convert_golf_result_to_domain_data(golf_result),
                        metadata={
                            'orchestrated': True,
                            'orchestration_id': orchestration_result.task_id,
                            'prediction_type': request.prediction_type,
                            'tournament_id': request.tournament_id
                        }
                    )
            
            # Fallback to direct processing
            return await self._process_direct(request)
            
        except Exception as e:
            self.logger.warning(f"Orchestrated processing failed, falling back to direct: {e}")
            return await self._process_direct(request)
    
    async def _process_direct(self, request: GolfPredictionRequest) -> DomainResponse:
        """Process request directly without orchestrators."""
        try:
            # Process based on request type
            if request.prediction_type == 'lineup_optimization':
                result = await self.enhanced_golf_core.predict_lineup(request.analysis_context)
                return DomainResponse(
                    success=True,
                    data=self._convert_golf_result_to_domain_data(result),
                    metadata={
                        'prediction_type': 'lineup_optimization',
                        'tournament_id': request.tournament_id,
                        'models_used': result.model_versions
                    }
                )
            
            elif request.prediction_type == 'player_projections':
                # Extract just player projections
                result = await self.enhanced_golf_core.predict_lineup(request.analysis_context)
                return DomainResponse(
                    success=True,
                    data={
                        'player_projections': result.player_projections,
                        'confidence_scores': result.confidence_scores
                    },
                    metadata={
                        'prediction_type': 'player_projections',
                        'tournament_id': request.tournament_id
                    }
                )
            
            elif request.prediction_type == 'risk_analysis':
                result = await self.enhanced_golf_core.predict_lineup(request.analysis_context)
                return DomainResponse(
                    success=True,
                    data={
                        'risk_metrics': result.risk_metrics,
                        'lineup_recommendations': result.lineup_recommendations[:3]  # Top 3 lineups
                    },
                    metadata={
                        'prediction_type': 'risk_analysis',
                        'tournament_id': request.tournament_id
                    }
                )
            
            else:
                raise ValueError(f"Unknown prediction type: {request.prediction_type}")
                
        except Exception as e:
            self.logger.error(f"Direct processing failed: {e}")
            raise
    
    def _convert_golf_result_to_domain_data(self, golf_result: GolfPredictionResult) -> Dict[str, Any]:
        """Convert golf prediction result to domain data format."""
        return {
            'lineup_recommendations': golf_result.lineup_recommendations,
            'player_projections': golf_result.player_projections,
            'confidence_scores': golf_result.confidence_scores,
            'risk_metrics': golf_result.risk_metrics,
            'optimization_details': golf_result.optimization_details,
            'prediction_timestamp': golf_result.prediction_timestamp.isoformat(),
            'model_versions': golf_result.model_versions
        }
    
    async def validate_data(self, data: DomainData) -> bool:
        """Validate golf domain data."""
        if not isinstance(data, GolfDomainData):
            return False
        
        # Check required fields
        required_fields = ['tournament_id', 'player_data', 'salary_cap']
        for field in required_fields:
            if not hasattr(data, field) or getattr(data, field) is None:
                self.logger.warning(f"Missing required field: {field}")
                return False
        
        # Validate data ranges
        if data.salary_cap <= 0:
            self.logger.warning("Invalid salary cap")
            return False
        
        if not data.player_data:
            self.logger.warning("No player data provided")
            return False
        
        return True
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get domain capabilities."""
        return {
            'domain_name': self.domain_name,
            'domain_type': self.domain_type,
            'capabilities': list(self.capabilities),
            'supported_request_types': [
                'lineup_optimization',
                'player_projections', 
                'risk_analysis',
                'tournament_prediction'
            ],
            'brain_integration': {
                'orchestrators_enabled': self.orchestrator_enabled,
                'proof_strategies_enabled': self.proof_strategies_enabled,
                'uncertainty_enabled': self.uncertainty_enabled
            },
            'performance_metrics': self.performance_metrics.copy()
        }
    
    async def get_domain_status(self) -> Dict[str, Any]:
        """Get current domain status."""
        status = {
            'initialized': self.is_initialized,
            'request_count': self.request_count,
            'last_prediction': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            'performance_metrics': self.performance_metrics.copy()
        }
        
        # Add enhanced golf core status
        if self.enhanced_golf_core:
            try:
                core_metrics = await self.enhanced_golf_core.get_performance_metrics()
                status['enhanced_golf_core'] = core_metrics
            except Exception as e:
                self.logger.warning(f"Failed to get enhanced golf core status: {e}")
        
        return status
    
    def _update_performance_metrics(self, processing_time: float, success: bool):
        """Update performance metrics."""
        if success:
            self.performance_metrics['successful_predictions'] += 1
        
        # Update average response time
        total_requests = self.performance_metrics['total_requests']
        current_avg = self.performance_metrics['average_response_time']
        new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
        self.performance_metrics['average_response_time'] = new_avg
    
    async def process_with_proof_strategies(self, request: GolfPredictionRequest) -> DomainResponse:
        """Process request with proof strategy validation."""
        if not self.proof_strategies_enabled:
            return await self.process_request(request)
        
        try:
            # Get base prediction
            base_response = await self.process_request(request)
            
            # Apply proof strategies if available
            if (self.brain_core and 
                hasattr(self.brain_core, 'proof_system') and 
                hasattr(self.brain_core.proof_system, 'proof_strategies')):
                
                # Use adaptive strategy for golf predictions
                proof_strategy = self.brain_core.proof_system.proof_strategies.get('adaptive')
                if proof_strategy:
                    proof_context = {
                        'domain': self.domain_name,
                        'prediction_data': base_response.data,
                        'request_context': request.to_dict(),
                        'confidence_threshold': self.config.brain_integration.proof_confidence_threshold
                    }
                    
                    proof_result = await proof_strategy.verify_prediction(proof_context)
                    
                    # Add proof validation to response metadata
                    base_response.metadata['proof_validation'] = {
                        'strategy_used': 'adaptive',
                        'confidence_score': proof_result.confidence,
                        'validation_passed': proof_result.success,
                        'proof_details': proof_result.details
                    }
            
            return base_response
            
        except Exception as e:
            self.logger.warning(f"Proof strategy processing failed: {e}")
            return await self.process_request(request)
    
    async def process_with_uncertainty_quantification(self, request: GolfPredictionRequest) -> DomainResponse:
        """Process request with uncertainty quantification."""
        if not self.uncertainty_enabled:
            return await self.process_request(request)
        
        try:
            # Get base prediction
            base_response = await self.process_request(request)
            
            # Add uncertainty quantification if orchestrator available
            if (self.brain_core and 
                hasattr(self.brain_core, 'uncertainty_orchestrator')):
                
                uncertainty_context = {
                    'domain': self.domain_name,
                    'prediction_data': base_response.data,
                    'model_ensemble': True,
                    'confidence_scores': base_response.data.get('confidence_scores', {})
                }
                
                uncertainty_result = await self.brain_core.uncertainty_orchestrator.quantify_uncertainty(
                    uncertainty_context
                )
                
                # Add uncertainty metrics to response
                base_response.metadata['uncertainty_quantification'] = {
                    'overall_uncertainty': uncertainty_result.get('overall_uncertainty', 0.0),
                    'prediction_intervals': uncertainty_result.get('prediction_intervals', {}),
                    'uncertainty_sources': uncertainty_result.get('sources', []),
                    'reliability_score': uncertainty_result.get('reliability_score', 0.0)
                }
            
            return base_response
            
        except Exception as e:
            self.logger.warning(f"Uncertainty quantification failed: {e}")
            return await self.process_request(request)
    
    async def batch_process_requests(self, requests: List[GolfPredictionRequest]) -> List[DomainResponse]:
        """Process multiple requests in batch."""
        if not requests:
            return []
        
        self.logger.info(f"Processing batch of {len(requests)} golf requests")
        
        # Process requests concurrently
        tasks = [self.process_request(request) for request in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                self.logger.error(f"Batch request {i} failed: {response}")
                results.append(DomainResponse(
                    success=False,
                    data={},
                    metadata={'error': str(response), 'batch_index': i}
                ))
            else:
                results.append(response)
        
        return results
    
    async def shutdown(self):
        """Shutdown brain golf connector."""
        self.logger.info("Shutting down Brain Golf Connector...")
        
        try:
            # Shutdown enhanced golf core
            if self.enhanced_golf_core:
                await self.enhanced_golf_core.shutdown()
            
            # Unregister from brain system
            if (self.brain_core and 
                hasattr(self.brain_core, 'domain_registry') and 
                self.is_initialized):
                await self.brain_core.domain_registry.unregister_domain(self.domain_name)
            
            self.is_initialized = False
            self.logger.info("Brain Golf Connector shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during Brain Golf Connector shutdown: {e}")
    
    def __str__(self) -> str:
        """String representation."""
        return f"BrainGolfConnector(domain={self.domain_name}, initialized={self.is_initialized})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"BrainGolfConnector(domain={self.domain_name}, "
                f"requests={self.request_count}, "
                f"orchestrators={self.orchestrator_enabled}, "
                f"initialized={self.is_initialized})")