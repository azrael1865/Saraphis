"""
Golf Domain Configuration
Configuration for golf-specific domain operations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

@dataclass
class GolfDomainConfig:
    """Configuration for golf domain operations"""
    
    # Domain parameters
    domain_name: str = "golf"
    version: str = "1.0.0"
    
    # Golf-specific settings
    max_strokes: int = 10
    par_values: Dict[int, int] = field(default_factory=lambda: {
        3: 3,  # Par 3
        4: 4,  # Par 4  
        5: 5   # Par 5
    })
    
    # Scoring configuration
    scoring_method: str = "stroke_play"  # stroke_play, match_play, stableford
    handicap_enabled: bool = True
    max_handicap: int = 36
    
    # Course configuration
    num_holes: int = 18
    tee_positions: List[str] = field(default_factory=lambda: ["back", "middle", "front"])
    
    # Analytics settings
    track_statistics: bool = True
    statistics_window: int = 20  # Number of rounds to analyze
    
    # Performance thresholds
    good_score_threshold: float = 0.9  # 90% of par
    excellent_score_threshold: float = 0.8  # 80% of par
    
    # Integration settings
    enable_ml_predictions: bool = False
    enable_weather_integration: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.num_holes not in [9, 18]:
            raise ValueError(f"num_holes must be 9 or 18, got {self.num_holes}")
        
        if self.max_handicap < 0 or self.max_handicap > 54:
            raise ValueError(f"max_handicap must be between 0 and 54, got {self.max_handicap}")
        
        if self.scoring_method not in ["stroke_play", "match_play", "stableford"]:
            raise ValueError(f"Invalid scoring_method: {self.scoring_method}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "domain_name": self.domain_name,
            "version": self.version,
            "max_strokes": self.max_strokes,
            "par_values": self.par_values,
            "scoring_method": self.scoring_method,
            "handicap_enabled": self.handicap_enabled,
            "max_handicap": self.max_handicap,
            "num_holes": self.num_holes,
            "tee_positions": self.tee_positions,
            "track_statistics": self.track_statistics,
            "statistics_window": self.statistics_window,
            "good_score_threshold": self.good_score_threshold,
            "excellent_score_threshold": self.excellent_score_threshold,
            "enable_ml_predictions": self.enable_ml_predictions,
            "enable_weather_integration": self.enable_weather_integration
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GolfDomainConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    def save(self, path: Path) -> None:
        """Save configuration to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'GolfDomainConfig':
        """Load configuration from JSON file"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)

__all__ = ['GolfDomainConfig']