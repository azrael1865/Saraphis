"""
Compression strategies module for intelligent strategy selection.
"""

from .compression_strategy import (
    # Configuration
    StrategyConfig,
    CompressedData,
    
    # Abstract base
    CompressionStrategy,
    
    # Concrete strategies
    TropicalStrategy,
    PadicStrategy,
    HybridStrategy,
    
    # Strategy selection
    StrategySelector,
    AdaptiveStrategyManager,
)

__all__ = [
    'StrategyConfig',
    'CompressedData',
    'CompressionStrategy',
    'TropicalStrategy',
    'PadicStrategy',
    'HybridStrategy',
    'StrategySelector',
    'AdaptiveStrategyManager',
]