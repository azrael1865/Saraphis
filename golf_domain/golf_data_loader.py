#!/usr/bin/env python3
"""
Golf Data Loader - Data loading and preprocessing for golf prediction models.
Handles player data, historical results, course information, and weather data.
"""

import asyncio
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import time
from pathlib import Path
import hashlib

from .domain_config import DataConfig


@dataclass
class GolfDataConfig:
    """Configuration for golf data loading."""
    normalize_features: bool = True
    handle_missing_values: str = 'mean'  # 'mean', 'median', 'drop'
    outlier_removal: bool = True
    feature_engineering: bool = True
    use_cache: bool = True
    cache_directory: str = './cache/golf/'
    min_tournaments_per_player: int = 5
    lookback_weeks: int = 52
    
    def __post_init__(self):
        """Ensure cache directory exists."""
        if self.use_cache:
            os.makedirs(self.cache_directory, exist_ok=True)


@dataclass
class PlayerData:
    """Individual player data structure."""
    player_id: str
    name: str
    recent_scores: List[float]
    average_score: float
    driving_distance: float
    driving_accuracy: float
    greens_in_regulation: float
    putting_average: float
    historical_performance: Dict[str, List[float]]
    injury_status: str = 'healthy'
    form_rating: float = 0.0
    world_ranking: int = 999


@dataclass
class TournamentData:
    """Tournament-specific data structure."""
    tournament_id: str
    name: str
    course_name: str
    start_date: datetime
    prize_pool: float
    field_size: int
    cut_line: Optional[int] = None
    weather_forecast: Dict[str, Any] = field(default_factory=dict)
    course_conditions: Dict[str, Any] = field(default_factory=dict)


class GolfDataset(Dataset):
    """PyTorch dataset for golf data."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, player_ids: List[str]):
        """Initialize golf dataset."""
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.player_ids = player_ids
        
        assert len(self.features) == len(self.targets) == len(self.player_ids)
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        return self.features[idx], self.targets[idx]
    
    def get_player_data(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get item with player ID."""
        return self.features[idx], self.targets[idx], self.player_ids[idx]


class GolfDataLoader:
    """Main data loader for golf prediction system."""
    
    def __init__(self, config: DataConfig, cache_enabled: bool = True):
        """Initialize golf data loader."""
        self.config = config
        self.cache_enabled = cache_enabled
        self.logger = logging.getLogger('GolfDataLoader')
        
        # Data paths
        self.data_paths = {
            'player_data': config.player_data_path,
            'historical_results': config.historical_results_path,
            'course_data': config.course_data_path,
            'weather_data': config.weather_data_path
        }
        
        # Cache management
        self.cache_dir = Path(config.cache_directory)
        self.cache_expiry = timedelta(hours=config.cache_expiry_hours)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Data storage
        self.player_database = {}
        self.tournament_database = {}
        self.course_database = {}
        self.weather_database = {}
        
        # Feature engineering
        self.feature_columns = []
        self.target_columns = ['fantasy_points']
        self.scaler = None
        
        # State tracking
        self.is_initialized = False
        self.last_data_load = None
        
        self.logger.info("Golf Data Loader initialized")
    
    async def initialize(self):
        """Initialize data loader and load initial data."""
        if self.is_initialized:
            return
        
        try:
            self.logger.info("Initializing Golf Data Loader...")
            
            # Create cache directory
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Load base data
            await self._load_base_data()
            
            # Initialize feature engineering
            await self._initialize_feature_engineering()
            
            self.is_initialized = True
            self.last_data_load = datetime.now()
            self.logger.info("Golf Data Loader initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Golf Data Loader: {e}")
            raise RuntimeError(f"Data loader initialization failed: {e}")
    
    async def _load_base_data(self):
        """Load base data files."""
        try:
            # Load player data
            if os.path.exists(self.data_paths['player_data']):
                player_df = pd.read_csv(self.data_paths['player_data'])
                self.player_database = self._process_player_data(player_df)
                self.logger.info(f"Loaded {len(self.player_database)} players")
            else:
                self.logger.warning(f"Player data file not found: {self.data_paths['player_data']}")
                self.player_database = self._generate_sample_player_data()
            
            # Load historical results
            if os.path.exists(self.data_paths['historical_results']):
                results_df = pd.read_csv(self.data_paths['historical_results'])
                self.tournament_database = self._process_tournament_data(results_df)
                self.logger.info(f"Loaded {len(self.tournament_database)} tournaments")
            else:
                self.logger.warning(f"Historical results file not found: {self.data_paths['historical_results']}")
                self.tournament_database = self._generate_sample_tournament_data()
            
            # Load course data
            if os.path.exists(self.data_paths['course_data']):
                course_df = pd.read_csv(self.data_paths['course_data'])
                self.course_database = self._process_course_data(course_df)
                self.logger.info(f"Loaded {len(self.course_database)} courses")
            else:
                self.logger.warning(f"Course data file not found: {self.data_paths['course_data']}")
                self.course_database = self._generate_sample_course_data()
            
            # Load weather data
            if os.path.exists(self.data_paths['weather_data']):
                weather_df = pd.read_csv(self.data_paths['weather_data'])
                self.weather_database = self._process_weather_data(weather_df)
                self.logger.info(f"Loaded {len(self.weather_database)} weather records")
            else:
                self.logger.warning(f"Weather data file not found: {self.data_paths['weather_data']}")
                self.weather_database = self._generate_sample_weather_data()
            
        except Exception as e:
            self.logger.error(f"Failed to load base data: {e}")
            raise
    
    def _process_player_data(self, df: pd.DataFrame) -> Dict[str, PlayerData]:
        """Process player data DataFrame into PlayerData objects."""
        player_db = {}
        
        for _, row in df.iterrows():
            player_id = str(row.get('player_id', row.get('id', '')))
            
            player_data = PlayerData(
                player_id=player_id,
                name=row.get('name', f'Player_{player_id}'),
                recent_scores=self._parse_scores(row.get('recent_scores', '[]')),
                average_score=float(row.get('average_score', 72.0)),
                driving_distance=float(row.get('driving_distance', 280.0)),
                driving_accuracy=float(row.get('driving_accuracy', 0.65)),
                greens_in_regulation=float(row.get('greens_in_regulation', 0.68)),
                putting_average=float(row.get('putting_average', 1.8)),
                historical_performance=self._parse_historical_performance(row.get('historical_performance', '{}')),
                injury_status=row.get('injury_status', 'healthy'),
                form_rating=float(row.get('form_rating', 0.0)),
                world_ranking=int(row.get('world_ranking', 999))
            )
            
            player_db[player_id] = player_data
        
        return player_db
    
    def _parse_scores(self, scores_str: str) -> List[float]:
        """Parse scores from string representation."""
        try:
            if isinstance(scores_str, str):
                return json.loads(scores_str)
            elif isinstance(scores_str, list):
                return [float(s) for s in scores_str]
            else:
                return []
        except:
            return []
    
    def _parse_historical_performance(self, perf_str: str) -> Dict[str, List[float]]:
        """Parse historical performance from string representation."""
        try:
            if isinstance(perf_str, str):
                return json.loads(perf_str)
            elif isinstance(perf_str, dict):
                return perf_str
            else:
                return {}
        except:
            return {}
    
    def _process_tournament_data(self, df: pd.DataFrame) -> Dict[str, TournamentData]:
        """Process tournament data DataFrame."""
        tournament_db = {}
        
        for _, row in df.iterrows():
            tournament_id = str(row.get('tournament_id', ''))
            
            tournament_data = TournamentData(
                tournament_id=tournament_id,
                name=row.get('tournament_name', f'Tournament_{tournament_id}'),
                course_name=row.get('course_name', 'Unknown Course'),
                start_date=pd.to_datetime(row.get('start_date', datetime.now())),
                prize_pool=float(row.get('prize_pool', 1000000)),
                field_size=int(row.get('field_size', 144)),
                cut_line=int(row.get('cut_line', 70)) if pd.notna(row.get('cut_line')) else None,
                weather_forecast=self._parse_json_field(row.get('weather_forecast', '{}')),
                course_conditions=self._parse_json_field(row.get('course_conditions', '{}'))
            )
            
            tournament_db[tournament_id] = tournament_data
        
        return tournament_db
    
    def _process_course_data(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Process course data DataFrame."""
        course_db = {}
        
        for _, row in df.iterrows():
            course_name = row.get('course_name', '')
            
            course_db[course_name] = {
                'yardage': int(row.get('yardage', 7200)),
                'par': int(row.get('par', 72)),
                'difficulty_rating': float(row.get('difficulty_rating', 3.5)),
                'green_speed': float(row.get('green_speed', 11.0)),
                'rough_height': float(row.get('rough_height', 3.0)),
                'wind_factor': float(row.get('wind_factor', 1.0)),
                'elevation_changes': float(row.get('elevation_changes', 200.0)),
                'course_type': row.get('course_type', 'parkland')
            }
        
        return course_db
    
    def _process_weather_data(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Process weather data DataFrame."""
        weather_db = {}
        
        for _, row in df.iterrows():
            date_key = str(row.get('date', ''))
            location = row.get('location', '')
            key = f"{date_key}_{location}"
            
            weather_db[key] = {
                'temperature': float(row.get('temperature', 70.0)),
                'humidity': float(row.get('humidity', 0.6)),
                'wind_speed': float(row.get('wind_speed', 10.0)),
                'wind_direction': row.get('wind_direction', 'N'),
                'precipitation_chance': float(row.get('precipitation_chance', 0.0)),
                'visibility': float(row.get('visibility', 10.0)),
                'pressure': float(row.get('pressure', 30.0))
            }
        
        return weather_db
    
    def _parse_json_field(self, field_value: str) -> Dict[str, Any]:
        """Parse JSON field safely."""
        try:
            if isinstance(field_value, str):
                return json.loads(field_value)
            elif isinstance(field_value, dict):
                return field_value
            else:
                return {}
        except:
            return {}
    
    def _generate_sample_player_data(self) -> Dict[str, PlayerData]:
        """Generate sample player data for testing."""
        self.logger.info("Generating sample player data")
        player_db = {}
        
        for i in range(150):  # Generate 150 sample players
            player_id = f"player_{i:03d}"
            
            # Generate realistic golf statistics
            avg_score = np.random.normal(72, 3)
            recent_scores = [avg_score + np.random.normal(0, 2) for _ in range(10)]
            
            player_data = PlayerData(
                player_id=player_id,
                name=f"Player {i:03d}",
                recent_scores=recent_scores,
                average_score=avg_score,
                driving_distance=np.random.normal(290, 20),
                driving_accuracy=np.random.beta(6, 4),  # Biased toward higher accuracy
                greens_in_regulation=np.random.beta(7, 3),
                putting_average=np.random.normal(1.8, 0.2),
                historical_performance={
                    'last_6_tournaments': [avg_score + np.random.normal(0, 3) for _ in range(6)]
                },
                injury_status=np.random.choice(['healthy', 'minor_injury'], p=[0.9, 0.1]),
                form_rating=np.random.normal(0, 1),
                world_ranking=i + 1
            )
            
            player_db[player_id] = player_data
        
        return player_db
    
    def _generate_sample_tournament_data(self) -> Dict[str, TournamentData]:
        """Generate sample tournament data for testing."""
        self.logger.info("Generating sample tournament data")
        tournament_db = {}
        
        tournament_names = [
            'Masters Tournament', 'PGA Championship', 'U.S. Open',
            'The Open Championship', 'Players Championship', 'WGC Match Play',
            'Arnold Palmer Invitational', 'Memorial Tournament'
        ]
        
        for i, name in enumerate(tournament_names):
            tournament_id = f"tournament_{i:03d}"
            
            tournament_data = TournamentData(
                tournament_id=tournament_id,
                name=name,
                course_name=f"{name} Course",
                start_date=datetime.now() + timedelta(days=i*7),
                prize_pool=np.random.uniform(5000000, 15000000),
                field_size=np.random.choice([144, 156, 132]),
                cut_line=70,
                weather_forecast={
                    'temperature': np.random.uniform(60, 85),
                    'wind_speed': np.random.uniform(5, 20),
                    'precipitation_chance': np.random.uniform(0, 0.3)
                },
                course_conditions={
                    'difficulty_rating': np.random.uniform(3.0, 5.0),
                    'green_speed': np.random.uniform(10.0, 13.0)
                }
            )
            
            tournament_db[tournament_id] = tournament_data
        
        return tournament_db
    
    def _generate_sample_course_data(self) -> Dict[str, Dict[str, Any]]:
        """Generate sample course data for testing."""
        self.logger.info("Generating sample course data")
        course_db = {}
        
        course_names = [
            'Augusta National', 'Pebble Beach', 'St. Andrews', 'Pinehurst No. 2',
            'Bethpage Black', 'TPC Sawgrass', 'Bay Hill', 'Muirfield Village'
        ]
        
        for name in course_names:
            course_db[name] = {
                'yardage': np.random.randint(7000, 7500),
                'par': 72,
                'difficulty_rating': np.random.uniform(3.0, 5.0),
                'green_speed': np.random.uniform(10.0, 13.0),
                'rough_height': np.random.uniform(2.0, 4.0),
                'wind_factor': np.random.uniform(0.8, 1.5),
                'elevation_changes': np.random.uniform(50, 400),
                'course_type': np.random.choice(['parkland', 'links', 'desert', 'mountain'])
            }
        
        return course_db
    
    def _generate_sample_weather_data(self) -> Dict[str, Dict[str, Any]]:
        """Generate sample weather data for testing."""
        self.logger.info("Generating sample weather data")
        weather_db = {}
        
        for i in range(30):  # 30 days of weather data
            date = datetime.now() + timedelta(days=i)
            date_key = date.strftime('%Y-%m-%d')
            
            weather_db[f"{date_key}_default"] = {
                'temperature': np.random.uniform(60, 85),
                'humidity': np.random.uniform(0.3, 0.8),
                'wind_speed': np.random.uniform(5, 25),
                'wind_direction': np.random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']),
                'precipitation_chance': np.random.uniform(0, 0.4),
                'visibility': np.random.uniform(8, 15),
                'pressure': np.random.uniform(29.5, 30.5)
            }
        
        return weather_db
    
    async def _initialize_feature_engineering(self):
        """Initialize feature engineering components."""
        # Define feature columns
        self.feature_columns = [
            # Player statistics
            'average_score', 'driving_distance', 'driving_accuracy',
            'greens_in_regulation', 'putting_average', 'form_rating',
            'world_ranking_normalized',
            
            # Recent performance
            'recent_avg_score', 'recent_score_std', 'recent_trend',
            
            # Course factors
            'course_difficulty', 'course_yardage_normalized', 'green_speed',
            'wind_factor', 'course_fit_score',
            
            # Weather factors
            'temperature', 'wind_speed', 'humidity', 'precipitation_chance',
            
            # Tournament factors
            'field_strength', 'prize_pool_normalized', 'cut_pressure'
        ]
        
        self.logger.info(f"Initialized {len(self.feature_columns)} feature columns")
    
    async def load_training_data(self, lookback_weeks: int = 12) -> Dict[str, Any]:
        """Load training data for model training."""
        if not self.is_initialized:
            await self.initialize()
        
        cache_key = f"training_data_{lookback_weeks}weeks"
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self.logger.info(f"Loading training data for {lookback_weeks} weeks")
            
            # Combine all data sources
            training_samples = []
            
            for tournament_id, tournament in self.tournament_database.items():
                # Check if tournament is within lookback period
                cutoff_date = datetime.now() - timedelta(weeks=lookback_weeks)
                if tournament.start_date < cutoff_date:
                    continue
                
                # Generate training samples for this tournament
                tournament_samples = await self._generate_tournament_training_samples(tournament_id)
                training_samples.extend(tournament_samples)
            
            if not training_samples:
                self.logger.warning("No training samples generated")
                return {'features': np.array([]), 'targets': np.array([]), 'player_ids': []}
            
            # Convert to arrays
            features = np.array([sample['features'] for sample in training_samples])
            targets = np.array([sample['target'] for sample in training_samples])
            player_ids = [sample['player_id'] for sample in training_samples]
            
            training_data = {
                'features': features,
                'targets': targets,
                'player_ids': player_ids,
                'feature_names': self.feature_columns,
                'sample_count': len(training_samples)
            }
            
            # Cache the data
            await self._cache_data(cache_key, training_data)
            
            self.logger.info(f"Loaded {len(training_samples)} training samples")
            return training_data
            
        except Exception as e:
            self.logger.error(f"Failed to load training data: {e}")
            raise
    
    async def _generate_tournament_training_samples(self, tournament_id: str) -> List[Dict[str, Any]]:
        """Generate training samples for a specific tournament."""
        tournament = self.tournament_database.get(tournament_id)
        if not tournament:
            return []
        
        samples = []
        
        # Select subset of players for this tournament
        available_players = list(self.player_database.keys())[:50]  # Limit for simulation
        
        for player_id in available_players:
            player = self.player_database[player_id]
            
            # Generate features for this player-tournament combination
            features = await self._generate_player_tournament_features(player, tournament)
            
            # Generate target (simulated fantasy points)
            target = self._simulate_fantasy_points(player, tournament)
            
            samples.append({
                'features': features,
                'target': target,
                'player_id': player_id,
                'tournament_id': tournament_id
            })
        
        return samples
    
    async def _generate_player_tournament_features(self, player: PlayerData, tournament: TournamentData) -> np.ndarray:
        """Generate feature vector for player-tournament combination."""
        features = []
        
        # Player statistics
        features.extend([
            player.average_score,
            player.driving_distance,
            player.driving_accuracy,
            player.greens_in_regulation,
            player.putting_average,
            player.form_rating,
            1.0 / max(player.world_ranking, 1)  # Normalized ranking
        ])
        
        # Recent performance
        if player.recent_scores:
            features.extend([
                np.mean(player.recent_scores),
                np.std(player.recent_scores),
                self._calculate_trend(player.recent_scores)
            ])
        else:
            features.extend([player.average_score, 2.0, 0.0])
        
        # Course factors
        course_data = self.course_database.get(tournament.course_name, {})
        features.extend([
            course_data.get('difficulty_rating', 3.5),
            course_data.get('yardage', 7200) / 7500,  # Normalized
            course_data.get('green_speed', 11.0),
            course_data.get('wind_factor', 1.0),
            self._calculate_course_fit(player, course_data)
        ])
        
        # Weather factors
        weather_key = f"{tournament.start_date.strftime('%Y-%m-%d')}_default"
        weather_data = self.weather_database.get(weather_key, {})
        features.extend([
            weather_data.get('temperature', 70.0),
            weather_data.get('wind_speed', 10.0),
            weather_data.get('humidity', 0.6),
            weather_data.get('precipitation_chance', 0.0)
        ])
        
        # Tournament factors
        features.extend([
            self._calculate_field_strength(tournament),
            tournament.prize_pool / 15000000,  # Normalized
            1.0 if tournament.cut_line else 0.0
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_trend(self, scores: List[float]) -> float:
        """Calculate performance trend from recent scores."""
        if len(scores) < 3:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(scores))
        y = np.array(scores)
        slope = np.polyfit(x, y, 1)[0]
        return -slope  # Negative because lower scores are better
    
    def _calculate_course_fit(self, player: PlayerData, course_data: Dict[str, Any]) -> float:
        """Calculate how well player fits the course."""
        # Simple fit calculation based on player strengths vs course requirements
        course_type = course_data.get('course_type', 'parkland')
        
        fit_score = 0.5  # Base fit
        
        # Adjust based on course characteristics
        if course_type == 'links':
            # Links courses favor accuracy over distance
            fit_score += (player.driving_accuracy - 0.65) * 0.5
        elif course_type == 'desert':
            # Desert courses favor distance
            fit_score += (player.driving_distance - 280) / 40 * 0.3
        
        # Green speed factor
        green_speed = course_data.get('green_speed', 11.0)
        if green_speed > 12:
            # Fast greens favor good putters
            fit_score += (1.9 - player.putting_average) * 0.3
        
        return max(0.0, min(1.0, fit_score))
    
    def _calculate_field_strength(self, tournament: TournamentData) -> float:
        """Calculate field strength for tournament."""
        # Simple field strength calculation
        base_strength = 0.5
        
        # Adjust based on prize pool (proxy for field quality)
        if tournament.prize_pool > 10000000:
            base_strength += 0.3
        elif tournament.prize_pool > 7500000:
            base_strength += 0.2
        
        return min(1.0, base_strength)
    
    def _simulate_fantasy_points(self, player: PlayerData, tournament: TournamentData) -> float:
        """Simulate fantasy points for training data."""
        # Base points from average score
        base_points = 80 - player.average_score  # Simple conversion
        
        # Add randomness and factors
        course_factor = self.course_database.get(tournament.course_name, {}).get('difficulty_rating', 3.5)
        difficulty_adjustment = (5.0 - course_factor) * 2  # Easier courses = more points
        
        # Form factor
        form_factor = player.form_rating * 3
        
        # Random variance
        variance = np.random.normal(0, 5)
        
        fantasy_points = base_points + difficulty_adjustment + form_factor + variance
        return max(0, fantasy_points)
    
    async def get_tournament_players(self, tournament_id: str) -> List[str]:
        """Get list of players for tournament."""
        # Return subset of available players
        return list(self.player_database.keys())[:100]  # Limit for simulation
    
    async def get_tournament_player_data(self, tournament_id: str) -> List[Dict[str, Any]]:
        """Get player data for tournament with salaries."""
        player_names = await self.get_tournament_players(tournament_id)
        
        player_data = []
        for player_id in player_names:
            player = self.player_database.get(player_id)
            if player:
                # Simulate salary based on world ranking and form
                base_salary = 12000 - (player.world_ranking * 20)
                form_adjustment = player.form_rating * 500
                salary = max(5000, min(15000, base_salary + form_adjustment))
                
                player_data.append({
                    'id': player_id,
                    'name': player.name,
                    'salary': int(salary),
                    'position': 'G',  # Golf position
                    'projected_ownership': np.random.uniform(0.05, 0.4),
                    'world_ranking': player.world_ranking,
                    'average_score': player.average_score
                })
        
        return player_data
    
    async def load_historical_tournament_data(self, tournament_id: str, lookback_weeks: int = 12) -> List[Dict[str, Any]]:
        """Load historical data for tournament analysis."""
        cutoff_date = datetime.now() - timedelta(weeks=lookback_weeks)
        
        historical_data = []
        for player_id, player in self.player_database.items():
            if player.recent_scores:
                historical_data.append({
                    'player_name': player.name,
                    'player_id': player_id,
                    'recent_scores': player.recent_scores,
                    'average_score': player.average_score,
                    'form_rating': player.form_rating
                })
        
        return historical_data
    
    async def get_sample_features(self) -> Optional[np.ndarray]:
        """Get sample feature vector for network initialization."""
        if not self.feature_columns:
            await self._initialize_feature_engineering()
        
        # Return dummy features with correct dimensionality
        return np.zeros(len(self.feature_columns), dtype=np.float32)
    
    def create_data_loader(self, features: np.ndarray, targets: np.ndarray, player_ids: List[str], 
                          batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """Create PyTorch DataLoader."""
        dataset = GolfDataset(features, targets, player_ids)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    
    async def _get_cached_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache if available and not expired."""
        if not self.cache_enabled:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_item = json.load(f)
                
                cache_time = datetime.fromisoformat(cached_item['timestamp'])
                if datetime.now() - cache_time < self.cache_expiry:
                    # Convert numpy arrays back
                    data = cached_item['data']
                    if 'features' in data and isinstance(data['features'], list):
                        data['features'] = np.array(data['features'])
                    if 'targets' in data and isinstance(data['targets'], list):
                        data['targets'] = np.array(data['targets'])
                    
                    self.logger.info(f"Loaded data from cache: {cache_key}")
                    return data
                
            except Exception as e:
                self.logger.warning(f"Failed to load cached data: {e}")
        
        return None
    
    async def _cache_data(self, cache_key: str, data: Dict[str, Any]):
        """Cache data to disk."""
        if not self.cache_enabled:
            return
        
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            # Prepare data for JSON serialization
            serializable_data = data.copy()
            if 'features' in serializable_data and isinstance(serializable_data['features'], np.ndarray):
                serializable_data['features'] = serializable_data['features'].tolist()
            if 'targets' in serializable_data and isinstance(serializable_data['targets'], np.ndarray):
                serializable_data['targets'] = serializable_data['targets'].tolist()
            
            cached_item = {
                'timestamp': datetime.now().isoformat(),
                'data': serializable_data
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cached_item, f)
            
            self.logger.info(f"Cached data: {cache_key}")
            
        except Exception as e:
            self.logger.warning(f"Failed to cache data: {e}")
    
    async def shutdown(self):
        """Shutdown data loader."""
        self.logger.info("Shutting down Golf Data Loader...")
        
        try:
            # Clear in-memory data
            self.player_database.clear()
            self.tournament_database.clear()
            self.course_database.clear()
            self.weather_database.clear()
            
            self.is_initialized = False
            self.logger.info("Golf Data Loader shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during Golf Data Loader shutdown: {e}")