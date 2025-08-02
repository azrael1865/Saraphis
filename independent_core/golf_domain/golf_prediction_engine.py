#!/usr/bin/env python3
"""
Golf Prediction Engine - Unified prediction interface for golf lineup optimization.
Coordinates neural networks, data processing, and ensemble predictions.
"""

import asyncio
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .domain_config import GolfDomainConfig
from .golf_neural_network import GolfNeuralNetwork, GolfNeuralFactory, GolfNeuralTrainer, NetworkPrediction
from .golf_data_loader import GolfDataLoader
from .enhanced_golf_core_main import GolfAnalysisContext


@dataclass
class TournamentPrediction:
    """Prediction results for a tournament."""
    tournament_id: str
    player_projections: Dict[str, float]
    confidence_scores: Dict[str, float]
    lineup_recommendations: List[Dict[str, Any]]
    risk_assessment: Dict[str, float]
    model_metadata: Dict[str, Any]
    prediction_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LineupPrediction:
    """Prediction for a specific lineup."""
    lineup_id: str
    players: List[str]
    projected_score: float
    confidence: float
    risk_score: float
    lineup_metadata: Dict[str, Any]


@dataclass
class PredictionRequest:
    """Request for prediction services."""
    request_id: str
    tournament_id: str
    prediction_type: str  # 'tournament', 'lineup', 'player'
    context: GolfAnalysisContext
    options: Dict[str, Any] = field(default_factory=dict)


class GolfPredictionEngine:
    """
    Main prediction engine that coordinates all golf prediction activities.
    Manages neural networks, data processing, and ensemble predictions.
    """
    
    def __init__(self, neural_network: GolfNeuralNetwork, data_loader: GolfDataLoader, config: GolfDomainConfig, gpu_optimizer=None):
        """Initialize golf prediction engine."""
        self.neural_network = neural_network
        self.data_loader = data_loader
        self.config = config
        self.logger = logging.getLogger('GolfPredictionEngine')
        
        # Thread safety
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=config.training.max_workers)
        
        # Device management
        self.device = self._get_device()
        
        # GPU optimization
        self.gpu_optimizer = gpu_optimizer
        
        # Training components
        self.trainer = None
        self.is_trained = False
        
        # Performance tracking
        self.prediction_cache = {}
        self.cache_timeout = 300  # 5 minutes
        self.performance_metrics = {
            'total_predictions': 0,
            'cache_hits': 0,
            'average_prediction_time': 0.0,
            'model_accuracy': 0.0,
            'last_training_time': None
        }
        
        # State tracking
        self.is_initialized = False
        
        self.logger.info("Golf Prediction Engine initialized")
    
    def _get_device(self) -> torch.device:
        """Get appropriate device for training/inference."""
        device_config = self.config.training.device.lower()
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = torch.device('cpu')
                self.logger.info("Using CPU device")
        elif device_config == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                self.logger.info("Using CUDA device (forced)")
            else:
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
            self.logger.info("Using CPU device (forced)")
        
        return device
    
    async def initialize(self):
        """Initialize the prediction engine."""
        if self.is_initialized:
            return
        
        try:
            self.logger.info("Initializing Golf Prediction Engine...")
            
            # Move network to device
            self.neural_network = self.neural_network.to(self.device)
            
            # Initialize trainer
            self.trainer = GolfNeuralTrainer(self.neural_network, self.config.model)
            
            # Check if pre-trained model exists and load it
            await self._load_pretrained_model()
            
            self.is_initialized = True
            self.logger.info("Golf Prediction Engine initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Golf Prediction Engine: {e}")
            raise RuntimeError(f"Prediction engine initialization failed: {e}")
    
    async def _load_pretrained_model(self):
        """Load pre-trained model if available."""
        try:
            # This would load from a saved checkpoint
            # For now, we'll skip this as we're creating new models
            self.logger.info("No pre-trained model found, will train from scratch")
            
        except Exception as e:
            self.logger.warning(f"Failed to load pre-trained model: {e}")
    
    async def predict_tournament(self, tournament_id: str, context: GolfAnalysisContext) -> Dict[str, Any]:
        """Generate predictions for an entire tournament."""
        if not self.is_initialized:
            await self.initialize()
        
        # Check cache
        cache_key = f"tournament_{tournament_id}_{hash(str(context))}"
        cached_result = self._get_cached_prediction(cache_key)
        if cached_result:
            self.performance_metrics['cache_hits'] += 1
            return cached_result
        
        try:
            start_time = time.time()
            self.performance_metrics['total_predictions'] += 1
            
            self.logger.info(f"Generating tournament predictions for {tournament_id}")
            
            # Load tournament data
            tournament_players = await self.data_loader.get_tournament_player_data(tournament_id)
            
            # Generate features for all players
            player_features = await self._generate_tournament_features(tournament_id, tournament_players, context)
            
            # Get neural network predictions
            neural_predictions = await self._predict_with_neural_network(player_features)
            
            # Process predictions into tournament format
            tournament_result = await self._process_tournament_predictions(
                tournament_id, tournament_players, neural_predictions, context
            )
            
            # Cache result
            self._cache_prediction(cache_key, tournament_result)
            
            # Update performance metrics
            prediction_time = time.time() - start_time
            self._update_performance_metrics(prediction_time)
            
            self.logger.info(f"Tournament prediction completed in {prediction_time:.2f}s")
            return tournament_result
            
        except Exception as e:
            self.logger.error(f"Tournament prediction failed: {e}")
            raise RuntimeError(f"Tournament prediction failed: {e}")
    
    async def _generate_tournament_features(self, tournament_id: str, players: List[Dict], context: GolfAnalysisContext) -> torch.Tensor:
        """Generate feature matrix for tournament players."""
        try:
            feature_list = []
            
            # Get tournament data
            tournament = self.data_loader.tournament_database.get(tournament_id)
            if not tournament:
                raise ValueError(f"Tournament {tournament_id} not found")
            
            for player_data in players:
                player_id = player_data.get('id')
                player = self.data_loader.player_database.get(player_id)
                
                if player:
                    # Generate features for this player-tournament combination
                    features = await self.data_loader._generate_player_tournament_features(player, tournament)
                    feature_list.append(features)
                else:
                    # Generate default features if player not found
                    self.logger.warning(f"Player {player_id} not found, using default features")
                    sample_features = await self.data_loader.get_sample_features()
                    feature_list.append(sample_features)
            
            if not feature_list:
                raise ValueError("No features generated for tournament")
            
            # Convert to tensor
            features_array = np.array(feature_list)
            return torch.FloatTensor(features_array).to(self.device)
            
        except Exception as e:
            self.logger.error(f"Feature generation failed: {e}")
            raise
    
    async def _predict_with_neural_network(self, features: torch.Tensor) -> NetworkPrediction:
        """Get predictions from neural network."""
        try:
            self.neural_network.eval()
            
            # Optimize GPU memory if optimizer available
            if self.gpu_optimizer and features.device.type == 'cuda':
                with self.gpu_optimizer.optimize_memory_context():
                    with torch.no_grad():
                        if hasattr(self.neural_network, 'predict_with_uncertainty'):
                            # Use uncertainty estimation if available
                            prediction = self.neural_network.predict_with_uncertainty(features)
                        else:
                            # Standard forward pass
                            prediction = self.neural_network(features)
            else:
                with torch.no_grad():
                    if hasattr(self.neural_network, 'predict_with_uncertainty'):
                        # Use uncertainty estimation if available
                        prediction = self.neural_network.predict_with_uncertainty(features)
                    else:
                        # Standard forward pass
                        prediction = self.neural_network(features)
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Neural network prediction failed: {e}")
            raise
    
    async def _process_tournament_predictions(self, tournament_id: str, players: List[Dict], 
                                           predictions: NetworkPrediction, context: GolfAnalysisContext) -> Dict[str, Any]:
        """Process neural network predictions into tournament format."""
        try:
            projections = {}
            confidence_scores = {}
            
            # Convert tensor predictions to dictionary
            pred_values = predictions.predictions.cpu().numpy()
            conf_values = predictions.confidence.cpu().numpy()
            
            for i, player_data in enumerate(players):
                player_name = player_data.get('name', f"Player_{i}")
                
                if i < len(pred_values):
                    projections[player_name] = float(pred_values[i])
                    confidence_scores[player_name] = float(conf_values[i])
                else:
                    # Fallback for mismatched indices
                    projections[player_name] = 0.0
                    confidence_scores[player_name] = 0.0
            
            # Generate lineup recommendations
            lineup_recommendations = await self._generate_lineup_recommendations(
                players, projections, confidence_scores, context
            )
            
            # Calculate risk assessment
            risk_assessment = self._calculate_risk_assessment(projections, confidence_scores)
            
            return {
                'projections': projections,
                'confidence': confidence_scores,
                'lineup_recommendations': lineup_recommendations,
                'risk_assessment': risk_assessment,
                'model_metadata': {
                    'model_type': self.neural_network.__class__.__name__,
                    'device_used': str(self.device),
                    'prediction_features': len(predictions.predictions),
                    'uncertainty_available': predictions.uncertainty_estimates is not None
                }
            }
            
        except Exception as e:
            self.logger.error(f"Tournament prediction processing failed: {e}")
            raise
    
    async def _generate_lineup_recommendations(self, players: List[Dict], projections: Dict[str, float], 
                                            confidence_scores: Dict[str, float], context: GolfAnalysisContext) -> List[Dict[str, Any]]:
        """Generate lineup recommendations based on predictions."""
        try:
            # Combine player data with projections
            lineup_data = []
            for player in players:
                player_name = player.get('name')
                if player_name in projections:
                    lineup_data.append({
                        'name': player_name,
                        'salary': player.get('salary', 0),
                        'projection': projections[player_name],
                        'confidence': confidence_scores.get(player_name, 0.5),
                        'value': projections[player_name] / max(player.get('salary', 1), 1)
                    })
            
            # Sort by value (projection per dollar)
            lineup_data.sort(key=lambda x: x['value'] * x['confidence'], reverse=True)
            
            # Generate multiple lineup configurations
            lineups = []
            for i in range(3):  # Generate 3 different lineups
                lineup = await self._build_single_lineup(lineup_data, context, variation=i)
                if lineup:
                    lineups.append(lineup)
            
            return lineups
            
        except Exception as e:
            self.logger.error(f"Lineup recommendation generation failed: {e}")
            return []
    
    async def _build_single_lineup(self, player_data: List[Dict], context: GolfAnalysisContext, variation: int = 0) -> Optional[Dict[str, Any]]:
        """Build a single optimized lineup."""
        try:
            selected_players = []
            total_salary = 0
            total_projection = 0
            
            # Add variation for different lineups
            randomization = 0.1 * variation
            
            for player in player_data:
                if len(selected_players) >= context.lineup_size:
                    break
                
                # Apply slight randomization for lineup diversity
                adjusted_value = player['value'] * (1 + np.random.normal(0, randomization))
                
                if (total_salary + player['salary'] <= context.salary_cap and
                    len(selected_players) < context.lineup_size):
                    
                    selected_players.append(player)
                    total_salary += player['salary']
                    total_projection += player['projection']
            
            if len(selected_players) == context.lineup_size:
                return {
                    'lineup_id': f"lineup_{variation}",
                    'players': selected_players,
                    'total_salary': total_salary,
                    'salary_remaining': context.salary_cap - total_salary,
                    'projected_points': total_projection,
                    'average_confidence': np.mean([p['confidence'] for p in selected_players]),
                    'value_rating': total_projection / max(total_salary, 1),
                    'variation': variation
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Single lineup building failed: {e}")
            return None
    
    def _calculate_risk_assessment(self, projections: Dict[str, float], confidence_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate risk assessment metrics."""
        try:
            if not projections or not confidence_scores:
                return {}
            
            projection_values = list(projections.values())
            confidence_values = list(confidence_scores.values())
            
            return {
                'projection_variance': float(np.var(projection_values)),
                'confidence_mean': float(np.mean(confidence_values)),
                'confidence_std': float(np.std(confidence_values)),
                'risk_score': 1.0 - float(np.mean(confidence_values)),
                'uncertainty_level': float(np.std(confidence_values)),
                'prediction_range': float(np.max(projection_values) - np.min(projection_values))
            }
            
        except Exception as e:
            self.logger.error(f"Risk assessment calculation failed: {e}")
            return {}
    
    async def train_models(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train the neural network models."""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            self.logger.info("Starting neural network training...")
            start_time = time.time()
            
            # Extract training data
            features = training_data.get('features')
            targets = training_data.get('targets')
            
            if features is None or targets is None or len(features) == 0:
                raise ValueError("No training data provided")
            
            # Create data loaders
            train_size = int(0.8 * len(features))
            
            train_features = features[:train_size]
            train_targets = targets[:train_size]
            val_features = features[train_size:]
            val_targets = targets[train_size:]
            
            train_loader = self.data_loader.create_data_loader(
                train_features, train_targets, 
                training_data.get('player_ids', [])[:train_size],
                batch_size=self.config.training.batch_size,
                shuffle=True
            )
            
            val_loader = self.data_loader.create_data_loader(
                val_features, val_targets,
                training_data.get('player_ids', [])[train_size:],
                batch_size=self.config.training.batch_size,
                shuffle=False
            )
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            training_history = []
            
            for epoch in range(self.config.training.epochs):
                # Optimize GPU memory before training if available
                if self.gpu_optimizer and self.device.type == 'cuda':
                    await self.gpu_optimizer.optimize_memory_allocation()
                
                # Train epoch
                train_metrics = self.trainer.train_epoch(train_loader, self.device)
                
                # Validate
                val_metrics = self.trainer.validate(val_loader, self.device)
                
                # Record metrics
                epoch_metrics = {
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['val_loss'],
                    'learning_rate': train_metrics['lr']
                }
                training_history.append(epoch_metrics)
                
                # Early stopping check
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    patience_counter = 0
                    
                    # Save best model
                    if epoch % self.config.training.checkpoint_frequency == 0:
                        checkpoint_path = f"./checkpoints/golf_model_epoch_{epoch}.pt"
                        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                        self.trainer.save_checkpoint(checkpoint_path, epoch, val_metrics['val_loss'])
                else:
                    patience_counter += 1
                
                # Log progress
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                                   f"Val Loss: {val_metrics['val_loss']:.4f}")
                
                # Early stopping
                if patience_counter >= self.config.training.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            training_time = time.time() - start_time
            self.is_trained = True
            self.performance_metrics['last_training_time'] = datetime.now().isoformat()
            self.performance_metrics['model_accuracy'] = 1.0 - best_val_loss  # Simple accuracy proxy
            
            training_results = {
                'training_time': training_time,
                'epochs_completed': len(training_history),
                'best_val_loss': best_val_loss,
                'final_train_loss': training_history[-1]['train_loss'] if training_history else 0.0,
                'training_history': training_history,
                'model_parameters': sum(p.numel() for p in self.neural_network.parameters()),
                'device_used': str(self.device)
            }
            
            self.logger.info(f"Neural network training completed in {training_time:.2f}s")
            return training_results
            
        except Exception as e:
            self.logger.error(f"Neural network training failed: {e}")
            raise RuntimeError(f"Training failed: {e}")
    
    async def predict_player_performance(self, player_id: str, tournament_id: str, context: GolfAnalysisContext) -> Dict[str, Any]:
        """Predict individual player performance."""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Get player and tournament data
            player = self.data_loader.player_database.get(player_id)
            tournament = self.data_loader.tournament_database.get(tournament_id)
            
            if not player or not tournament:
                raise ValueError(f"Player {player_id} or tournament {tournament_id} not found")
            
            # Generate features
            features = await self.data_loader._generate_player_tournament_features(player, tournament)
            feature_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Get prediction
            prediction = await self._predict_with_neural_network(feature_tensor)
            
            return {
                'player_id': player_id,
                'player_name': player.name,
                'projected_points': float(prediction.predictions[0]),
                'confidence': float(prediction.confidence[0]),
                'uncertainty': float(prediction.uncertainty_estimates[0]) if prediction.uncertainty_estimates is not None else None,
                'feature_importance': prediction.feature_importance.cpu().numpy().tolist() if prediction.feature_importance is not None else None
            }
            
        except Exception as e:
            self.logger.error(f"Player performance prediction failed: {e}")
            raise RuntimeError(f"Player prediction failed: {e}")
    
    def _get_cached_prediction(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction if available and not expired."""
        if cache_key in self.prediction_cache:
            cached_item = self.prediction_cache[cache_key]
            if time.time() - cached_item['timestamp'] < self.cache_timeout:
                return cached_item['result']
            else:
                del self.prediction_cache[cache_key]
        return None
    
    def _cache_prediction(self, cache_key: str, result: Dict[str, Any]):
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
    
    def _update_performance_metrics(self, prediction_time: float):
        """Update performance metrics."""
        total_predictions = self.performance_metrics['total_predictions']
        current_avg = self.performance_metrics['average_prediction_time']
        
        # Update running average
        new_avg = ((current_avg * (total_predictions - 1)) + prediction_time) / total_predictions
        self.performance_metrics['average_prediction_time'] = new_avg
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = self.performance_metrics.copy()
        
        # Add current state info
        metrics.update({
            'is_initialized': self.is_initialized,
            'is_trained': self.is_trained,
            'device': str(self.device),
            'model_type': self.neural_network.__class__.__name__,
            'cache_size': len(self.prediction_cache),
            'cache_hit_rate': (metrics['cache_hits'] / max(metrics['total_predictions'], 1)) * 100
        })
        
        return metrics
    
    async def clear_cache(self):
        """Clear prediction cache."""
        with self._lock:
            self.prediction_cache.clear()
            self.logger.info("Prediction cache cleared")
    
    async def shutdown(self):
        """Shutdown prediction engine."""
        self.logger.info("Shutting down Golf Prediction Engine...")
        
        try:
            # Clear cache
            await self.clear_cache()
            
            # Shutdown executor
            self._executor.shutdown(wait=True)
            
            # Clear CUDA cache if using GPU
            if self.device.type == 'cuda':
                if self.gpu_optimizer:
                    await self.gpu_optimizer.cleanup_streams()
                torch.cuda.empty_cache()
            
            self.is_initialized = False
            self.logger.info("Golf Prediction Engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during Golf Prediction Engine shutdown: {e}")
    
    def __str__(self) -> str:
        """String representation."""
        return f"GolfPredictionEngine(model={self.neural_network.__class__.__name__}, trained={self.is_trained})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"GolfPredictionEngine(model={self.neural_network.__class__.__name__}, "
                f"device={self.device}, trained={self.is_trained}, "
                f"predictions={self.performance_metrics['total_predictions']})")