"""
Golf Gambling Domain for Saraphis Brain System

Provides comprehensive golf lineup optimization and prediction capabilities
combining existing golf gambler functionality with Saraphis neural networks.
"""

from .enhanced_golf_core_main import EnhancedGolfCore, GolfPredictionResult
from .domain_config import GolfDomainConfig
from .brain_golf_connector import BrainGolfConnector, GolfDomainData, GolfPredictionRequest
from .golf_neural_network import GolfNeuralNetwork, GolfEnsembleNetwork, GolfNeuralFactory
from .golf_data_loader import GolfDataLoader, GolfDataset, GolfDataConfig
from .golf_prediction_engine import GolfPredictionEngine, TournamentPrediction, LineupPrediction, PredictionRequest

__all__ = [
    'EnhancedGolfCore',
    'GolfPredictionResult', 
    'GolfDomainConfig',
    'BrainGolfConnector',
    'GolfDomainData',
    'GolfPredictionRequest',
    'GolfNeuralNetwork',
    'GolfEnsembleNetwork', 
    'GolfNeuralFactory',
    'GolfDataLoader',
    'GolfDataset',
    'GolfDataConfig',
    'GolfPredictionEngine',
    'TournamentPrediction',
    'LineupPrediction',
    'PredictionRequest'
]

__version__ = '1.0.0'