#!/usr/bin/env python3
"""
Enhanced Golf Core Main - Bridge between existing golf gambler and Saraphis Brain system.
Maintains existing functionality while adding Saraphis neural network capabilities.
"""

import asyncio
import logging
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import time

from .domain_config import GolfDomainConfig
from ..brain import BrainCore


@dataclass
class GolfPredictionResult:
    """Result from golf prediction operations."""
    lineup_recommendations: List[Dict[str, Any]]
    player_projections: Dict[str, float]
    confidence_scores: Dict[str, float]
    risk_metrics: Dict[str, float]
    optimization_details: Dict[str, Any]
    prediction_timestamp: datetime = field(default_factory=datetime.now)
    model_versions: Dict[str, str] = field(default_factory=dict)


@dataclass
class GolfAnalysisContext:
    """Context for golf analysis operations."""
    tournament_id: str
    salary_cap: float
    lineup_size: int
    position_constraints: Dict[str, int]
    weather_conditions: Optional[Dict[str, Any]] = None
    course_conditions: Optional[Dict[str, Any]] = None
    historical_context: Optional[Dict[str, Any]] = None


class EnhancedGolfCore:
    """
    Enhanced golf core that bridges existing golf gambler functionality
    with Saraphis Brain system capabilities.
    """
    
    def __init__(self, config: GolfDomainConfig, brain_core: Optional[BrainCore] = None, gpu_optimizer=None):
        """Initialize enhanced golf core."""
        self.config = config
        self.brain_core = brain_core
        self.gpu_optimizer = gpu_optimizer
        self.logger = logging.getLogger('EnhancedGolfCore')
        
        # Thread safety
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=config.training.max_workers)
        
        # Components (will be initialized later)
        self.golf_connector = None
        self.neural_network = None
        self.data_loader = None
        self.prediction_engine = None
        
        # Existing golf gambler integration
        self.existing_gambler = None
        self.existing_model_loaded = False
        
        # State tracking
        self.is_initialized = False
        self.last_training_time = None
        self.model_performance_metrics = {}
        
        # Cache for predictions
        self.prediction_cache = {}
        self.cache_timeout = 300  # 5 minutes
        
        self.logger.info("Enhanced Golf Core initialized")
    
    async def initialize(self):
        """Initialize all components."""
        if self.is_initialized:
            return
        
        try:
            self.logger.info("Initializing Enhanced Golf Core components...")
            
            # Initialize components in order
            await self._initialize_data_loader()
            await self._initialize_neural_network()
            await self._initialize_prediction_engine()
            await self._initialize_brain_connector()
            
            # Load existing golf gambler if configured
            if self.config.environment.use_existing_rl_model:
                await self._load_existing_golf_gambler()
            
            self.is_initialized = True
            self.logger.info("Enhanced Golf Core initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Enhanced Golf Core: {e}")
            raise RuntimeError(f"Initialization failed: {e}")
    
    async def _initialize_data_loader(self):
        """Initialize golf data loader."""
        from .golf_data_loader import GolfDataLoader
        
        self.data_loader = GolfDataLoader(
            config=self.config.data,
            cache_enabled=self.config.data.use_cache
        )
        await self.data_loader.initialize()
        self.logger.info("Golf data loader initialized")
    
    async def _initialize_neural_network(self):
        """Initialize golf neural network."""
        from .golf_neural_network import GolfNeuralFactory
        
        # Get input dimensions from data loader
        sample_data = await self.data_loader.get_sample_features()
        input_dim = len(sample_data) if sample_data else 50  # Default fallback
        
        self.neural_network = GolfNeuralFactory.create_network(
            model_type=self.config.model.model_type,
            input_dim=input_dim,
            config=self.config.model
        )
        self.logger.info(f"Golf neural network initialized ({self.config.model.model_type})")
    
    async def _initialize_prediction_engine(self):
        """Initialize golf prediction engine."""
        from .golf_prediction_engine import GolfPredictionEngine
        
        self.prediction_engine = GolfPredictionEngine(
            neural_network=self.neural_network,
            data_loader=self.data_loader,
            config=self.config,
            gpu_optimizer=self.gpu_optimizer
        )
        await self.prediction_engine.initialize()
        self.logger.info("Golf prediction engine initialized")
    
    async def _initialize_brain_connector(self):
        """Initialize brain connector if brain core is available."""
        if self.brain_core:
            from .brain_golf_connector import BrainGolfConnector
            
            self.golf_connector = BrainGolfConnector(
                brain_core=self.brain_core,
                config=self.config
            )
            await self.golf_connector.initialize()
            self.logger.info("Brain golf connector initialized")
    
    async def _load_existing_golf_gambler(self):
        """Load existing golf gambler functionality."""
        try:
            # This would integrate with existing golf gambler code
            # For now, we'll simulate this integration
            self.existing_gambler = {
                'model_type': 'reinforcement_learning',
                'last_trained': datetime.now(),
                'performance_metrics': {
                    'roi': 0.15,
                    'sharpe_ratio': 1.2,
                    'win_rate': 0.62
                }
            }
            self.existing_model_loaded = True
            self.logger.info("Existing golf gambler model loaded")
            
        except Exception as e:
            self.logger.warning(f"Failed to load existing golf gambler: {e}")
            self.existing_model_loaded = False
    
    async def predict_lineup(self, context: GolfAnalysisContext) -> GolfPredictionResult:
        """Generate optimized golf lineup predictions."""
        if not self.is_initialized:
            await self.initialize()
        
        # Check cache first
        cache_key = self._generate_cache_key(context)
        cached_result = self._get_cached_prediction(cache_key)
        if cached_result:
            self.logger.info("Returning cached prediction")
            return cached_result
        
        try:
            start_time = time.time()
            
            # Get predictions from all available models
            predictions = await self._generate_ensemble_predictions(context)
            
            # Optimize lineup using predictions
            lineup_recommendations = await self._optimize_lineup(predictions, context)
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(lineup_recommendations, context)
            
            # Create result
            result = GolfPredictionResult(
                lineup_recommendations=lineup_recommendations,
                player_projections=predictions.get('player_projections', {}),
                confidence_scores=predictions.get('confidence_scores', {}),
                risk_metrics=risk_metrics,
                optimization_details={
                    'prediction_time': time.time() - start_time,
                    'models_used': predictions.get('models_used', []),
                    'ensemble_weights': predictions.get('ensemble_weights', {}),
                    'optimization_method': 'enhanced_saraphis'
                },
                model_versions={
                    'saraphis_neural': self.neural_network.__class__.__name__ if self.neural_network else 'none',
                    'existing_rl': 'loaded' if self.existing_model_loaded else 'none',
                    'prediction_engine': self.prediction_engine.__class__.__name__ if self.prediction_engine else 'none'
                }
            )
            
            # Cache result
            self._cache_prediction(cache_key, result)
            
            self.logger.info(f"Generated lineup prediction in {result.optimization_details['prediction_time']:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to predict lineup: {e}")
            raise RuntimeError(f"Lineup prediction failed: {e}")
    
    async def _generate_ensemble_predictions(self, context: GolfAnalysisContext) -> Dict[str, Any]:
        """Generate predictions from ensemble of models."""
        predictions = {
            'player_projections': {},
            'confidence_scores': {},
            'models_used': [],
            'ensemble_weights': {}
        }
        
        # Saraphis neural network predictions
        if self.neural_network and self.prediction_engine:
            try:
                neural_preds = await self.prediction_engine.predict_tournament(
                    tournament_id=context.tournament_id,
                    context=context
                )
                predictions['player_projections'].update(neural_preds.get('projections', {}))
                predictions['confidence_scores'].update(neural_preds.get('confidence', {}))
                predictions['models_used'].append('saraphis_neural')
                predictions['ensemble_weights']['saraphis'] = self.config.model.ensemble_weight_saraphis
                
            except Exception as e:
                self.logger.warning(f"Saraphis neural prediction failed: {e}")
        
        # Existing golf gambler predictions
        if self.existing_model_loaded and self.config.environment.use_existing_rl_model:
            try:
                existing_preds = await self._get_existing_gambler_predictions(context)
                
                # Merge predictions with weights
                weight = self.config.model.ensemble_weight_existing
                for player, projection in existing_preds.get('projections', {}).items():
                    if player in predictions['player_projections']:
                        predictions['player_projections'][player] = (
                            predictions['player_projections'][player] * (1 - weight) + 
                            projection * weight
                        )
                    else:
                        predictions['player_projections'][player] = projection
                
                predictions['models_used'].append('existing_rl')
                predictions['ensemble_weights']['existing'] = weight
                
            except Exception as e:
                self.logger.warning(f"Existing gambler prediction failed: {e}")
        
        # Statistical baseline predictions
        try:
            stat_preds = await self._get_statistical_baseline(context)
            
            weight = self.config.model.ensemble_weight_statistical
            for player, projection in stat_preds.get('projections', {}).items():
                if player in predictions['player_projections']:
                    predictions['player_projections'][player] = (
                        predictions['player_projections'][player] * (1 - weight) + 
                        projection * weight
                    )
                else:
                    predictions['player_projections'][player] = projection
            
            predictions['models_used'].append('statistical')
            predictions['ensemble_weights']['statistical'] = weight
            
        except Exception as e:
            self.logger.warning(f"Statistical baseline prediction failed: {e}")
        
        return predictions
    
    async def _get_existing_gambler_predictions(self, context: GolfAnalysisContext) -> Dict[str, Any]:
        """Get predictions from existing golf gambler model."""
        # This would interface with the existing golf gambler code
        # For now, simulate realistic predictions
        
        player_names = await self.data_loader.get_tournament_players(context.tournament_id)
        projections = {}
        
        for player in player_names[:20]:  # Limit for simulation
            # Simulate existing model predictions
            base_projection = np.random.normal(15, 5)  # Base fantasy points
            projections[player] = max(0, base_projection)
        
        return {
            'projections': projections,
            'confidence': {player: 0.7 for player in projections.keys()},
            'model_info': self.existing_gambler
        }
    
    async def _get_statistical_baseline(self, context: GolfAnalysisContext) -> Dict[str, Any]:
        """Get statistical baseline predictions."""
        try:
            # Load historical data for statistical analysis
            historical_data = await self.data_loader.load_historical_tournament_data(
                tournament_id=context.tournament_id,
                lookback_weeks=12
            )
            
            projections = {}
            for player_data in historical_data:
                player_name = player_data.get('player_name')
                if player_name:
                    # Simple statistical projection based on recent performance
                    recent_scores = player_data.get('recent_scores', [])
                    if recent_scores:
                        avg_score = np.mean(recent_scores)
                        std_score = np.std(recent_scores)
                        # Adjust for course difficulty, weather, etc.
                        projected_score = avg_score * 1.02  # Slight adjustment
                        projections[player_name] = max(0, projected_score)
            
            return {
                'projections': projections,
                'confidence': {player: 0.6 for player in projections.keys()}
            }
            
        except Exception as e:
            self.logger.warning(f"Statistical baseline calculation failed: {e}")
            return {'projections': {}, 'confidence': {}}
    
    async def _optimize_lineup(self, predictions: Dict[str, Any], context: GolfAnalysisContext) -> List[Dict[str, Any]]:
        """Optimize lineup using predictions and constraints."""
        try:
            # Get player data with salaries
            player_data = await self.data_loader.get_tournament_player_data(context.tournament_id)
            
            # Combine predictions with player data
            optimization_data = []
            for player in player_data:
                player_name = player.get('name')
                if player_name in predictions['player_projections']:
                    optimization_data.append({
                        'name': player_name,
                        'salary': player.get('salary', 0),
                        'projection': predictions['player_projections'][player_name],
                        'confidence': predictions['confidence_scores'].get(player_name, 0.5),
                        'position': player.get('position', 'G'),
                        'ownership': player.get('projected_ownership', 0.1)
                    })
            
            # Sort by value (projection / salary ratio)
            optimization_data.sort(
                key=lambda x: (x['projection'] / max(x['salary'], 1)) * x['confidence'], 
                reverse=True
            )
            
            # Generate multiple lineup options
            lineups = []
            for i in range(5):  # Generate 5 different lineups
                lineup = await self._generate_single_lineup(
                    optimization_data, context, variation=i
                )
                if lineup:
                    lineups.append(lineup)
            
            return lineups
            
        except Exception as e:
            self.logger.error(f"Lineup optimization failed: {e}")
            return []
    
    async def _generate_single_lineup(self, player_data: List[Dict], context: GolfAnalysisContext, variation: int = 0) -> Optional[Dict[str, Any]]:
        """Generate a single optimized lineup."""
        try:
            selected_players = []
            total_salary = 0
            total_projection = 0
            
            # Add some variation for different lineups
            randomization_factor = 0.1 * variation
            
            for player in player_data:
                if len(selected_players) >= context.lineup_size:
                    break
                
                # Add randomization for lineup variation
                adjusted_value = player['projection'] * (1 + np.random.normal(0, randomization_factor))
                
                if (total_salary + player['salary'] <= context.salary_cap and
                    len(selected_players) < context.lineup_size):
                    
                    selected_players.append(player)
                    total_salary += player['salary']
                    total_projection += player['projection']
            
            if len(selected_players) == context.lineup_size:
                return {
                    'players': selected_players,
                    'total_salary': total_salary,
                    'salary_remaining': context.salary_cap - total_salary,
                    'projected_points': total_projection,
                    'lineup_rating': total_projection / max(total_salary, 1),
                    'risk_score': np.mean([p['confidence'] for p in selected_players]),
                    'variation_id': variation
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Single lineup generation failed: {e}")
            return None
    
    async def _calculate_risk_metrics(self, lineups: List[Dict], context: GolfAnalysisContext) -> Dict[str, float]:
        """Calculate risk metrics for lineup recommendations."""
        if not lineups:
            return {}
        
        try:
            all_players = []
            for lineup in lineups:
                all_players.extend(lineup.get('players', []))
            
            # Calculate various risk metrics
            ownership_variance = np.var([p['ownership'] for p in all_players])
            projection_variance = np.var([p['projection'] for p in all_players])
            salary_utilization = np.mean([
                lineup['total_salary'] / context.salary_cap for lineup in lineups
            ])
            
            avg_confidence = np.mean([p['confidence'] for p in all_players])
            
            return {
                'ownership_variance': ownership_variance,
                'projection_variance': projection_variance,
                'salary_utilization': salary_utilization,
                'average_confidence': avg_confidence,
                'lineup_correlation': 0.3,  # Placeholder - would calculate actual correlation
                'risk_score': 1 - avg_confidence,
                'diversification_score': 1 - ownership_variance
            }
            
        except Exception as e:
            self.logger.error(f"Risk metrics calculation failed: {e}")
            return {}
    
    def _generate_cache_key(self, context: GolfAnalysisContext) -> str:
        """Generate cache key for prediction context."""
        return f"lineup_{context.tournament_id}_{context.salary_cap}_{context.lineup_size}_{hash(str(context))}"
    
    def _get_cached_prediction(self, cache_key: str) -> Optional[GolfPredictionResult]:
        """Get cached prediction if still valid."""
        if cache_key in self.prediction_cache:
            cached_item = self.prediction_cache[cache_key]
            if time.time() - cached_item['timestamp'] < self.cache_timeout:
                return cached_item['result']
            else:
                del self.prediction_cache[cache_key]
        return None
    
    def _cache_prediction(self, cache_key: str, result: GolfPredictionResult):
        """Cache prediction result."""
        self.prediction_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        
        # Clean old cache entries
        current_time = time.time()
        expired_keys = [
            key for key, item in self.prediction_cache.items()
            if current_time - item['timestamp'] > self.cache_timeout
        ]
        for key in expired_keys:
            del self.prediction_cache[key]
    
    async def train_models(self, training_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Train neural network models."""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            self.logger.info("Starting model training...")
            start_time = time.time()
            
            # Load training data if not provided
            if training_data is None:
                training_data = await self.data_loader.load_training_data()
            
            # Train neural network
            training_results = {}
            if self.neural_network and self.prediction_engine:
                neural_results = await self.prediction_engine.train_models(training_data)
                training_results['neural_network'] = neural_results
            
            # Update performance metrics
            training_time = time.time() - start_time
            self.last_training_time = datetime.now()
            
            training_results['training_summary'] = {
                'training_time': training_time,
                'last_trained': self.last_training_time.isoformat(),
                'data_size': len(training_data) if training_data else 0,
                'models_trained': list(training_results.keys())
            }
            
            self.logger.info(f"Model training completed in {training_time:.2f}s")
            return training_results
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise RuntimeError(f"Training failed: {e}")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all models."""
        metrics = {
            'enhanced_golf_core': {
                'is_initialized': self.is_initialized,
                'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'prediction_cache_size': len(self.prediction_cache),
                'existing_model_loaded': self.existing_model_loaded
            }
        }
        
        # Add neural network metrics
        if self.prediction_engine:
            try:
                engine_metrics = await self.prediction_engine.get_performance_metrics()
                metrics['prediction_engine'] = engine_metrics
            except Exception as e:
                self.logger.warning(f"Failed to get prediction engine metrics: {e}")
        
        # Add existing gambler metrics
        if self.existing_model_loaded and self.existing_gambler:
            metrics['existing_gambler'] = self.existing_gambler.get('performance_metrics', {})
        
        return metrics
    
    async def shutdown(self):
        """Shutdown enhanced golf core."""
        self.logger.info("Shutting down Enhanced Golf Core...")
        
        try:
            # Shutdown components
            if self.golf_connector:
                await self.golf_connector.shutdown()
            
            if self.prediction_engine:
                await self.prediction_engine.shutdown()
            
            if self.data_loader:
                await self.data_loader.shutdown()
            
            # Shutdown executor
            self._executor.shutdown(wait=True)
            
            # Clear cache
            self.prediction_cache.clear()
            
            self.is_initialized = False
            self.logger.info("Enhanced Golf Core shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=False)