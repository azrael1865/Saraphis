#!/usr/bin/env python3
"""
Feature Extractor Plugin Base Classes
====================================

This module provides abstract base classes for feature extraction plugins in the Universal AI Core system.
Adapted from existing feature extraction patterns in the Saraphis codebase, made domain-agnostic.

Base Classes:
- FeatureExtractorPlugin: Abstract base for all feature extractors
- PluginMetadata: Metadata and versioning for plugins
- FeatureExtractionResult: Standardized result container
"""

import logging
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from enum import Enum
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Types of features that can be extracted"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical" 
    BINARY = "binary"
    TEXT = "text"
    VECTOR = "vector"
    GRAPH = "graph"
    IMAGE = "image"
    SEQUENCE = "sequence"


class ExtractionStatus(Enum):
    """Status of feature extraction operation"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class PluginMetadata:
    """Metadata for feature extractor plugins"""
    name: str
    version: str
    author: str
    description: str
    supported_input_types: List[str]
    supported_feature_types: List[FeatureType]
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    plugin_id: str = ""
    
    def __post_init__(self):
        """Generate plugin ID if not provided"""
        if not self.plugin_id:
            content = f"{self.name}:{self.version}:{self.author}"
            self.plugin_id = hashlib.md5(content.encode()).hexdigest()


@dataclass
class FeatureExtractionResult:
    """Result container for feature extraction operations"""
    features: Dict[str, Any]
    feature_names: List[str]
    feature_types: Dict[str, FeatureType]
    extraction_time: float
    status: ExtractionStatus
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    input_hash: str = ""
    
    def __post_init__(self):
        """Validate result after initialization"""
        if self.status == ExtractionStatus.SUCCESS and not self.features:
            raise ValueError("Successful extraction must have features")
        if self.status == ExtractionStatus.FAILED and not self.error_message:
            self.error_message = "Feature extraction failed"


class FeatureExtractorPlugin(ABC):
    """
    Abstract base class for feature extraction plugins.
    
    This class defines the interface that all feature extractor plugins must implement.
    Adapted from existing feature extraction patterns in the Saraphis codebase.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature extractor plugin.
        
        Args:
            config: Plugin-specific configuration dictionary
        """
        self.config = config or {}
        self._metadata = self._create_metadata()
        self._is_initialized = False
        self._feature_cache = {}
        self._extraction_count = 0
        self._last_extraction_time = None
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"Initialized feature extractor: {self._metadata.name}")
    
    @abstractmethod
    def _create_metadata(self) -> PluginMetadata:
        """
        Create metadata for this plugin.
        
        Returns:
            PluginMetadata instance with plugin information
        """
        pass
    
    @abstractmethod
    def extract_features(self, input_data: Any) -> FeatureExtractionResult:
        """
        Extract features from input data.
        
        Args:
            input_data: Input data to extract features from
            
        Returns:
            FeatureExtractionResult containing extracted features and metadata
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get the names of features that this extractor produces.
        
        Returns:
            List of feature names
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate that input data is compatible with this extractor.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        pass
    
    def initialize(self) -> bool:
        """
        Initialize the plugin. Called before first use.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self._perform_initialization()
            self._is_initialized = True
            logger.info(f"Plugin {self._metadata.name} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize plugin {self._metadata.name}: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown the plugin and clean up resources"""
        try:
            self._perform_shutdown()
            self._is_initialized = False
            logger.info(f"Plugin {self._metadata.name} shutdown successfully")
        except Exception as e:
            logger.error(f"Error shutting down plugin {self._metadata.name}: {e}")
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        return self._metadata
    
    def is_initialized(self) -> bool:
        """Check if plugin is initialized"""
        return self._is_initialized
    
    def batch_extract_features(self, input_batch: List[Any]) -> List[FeatureExtractionResult]:
        """
        Extract features from a batch of input data.
        
        Args:
            input_batch: List of input data items
            
        Returns:
            List of FeatureExtractionResult objects
        """
        if not self._is_initialized:
            raise RuntimeError(f"Plugin {self._metadata.name} not initialized")
        
        results = []
        for input_data in input_batch:
            try:
                result = self.extract_features(input_data)
                results.append(result)
            except Exception as e:
                error_result = FeatureExtractionResult(
                    features={},
                    feature_names=[],
                    feature_types={},
                    extraction_time=0.0,
                    status=ExtractionStatus.ERROR,
                    error_message=str(e)
                )
                results.append(error_result)
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores if available.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        return {}
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """
        Get statistics about feature extractions performed.
        
        Returns:
            Dictionary with extraction statistics
        """
        return {
            "extraction_count": self._extraction_count,
            "last_extraction_time": self._last_extraction_time,
            "cache_size": len(self._feature_cache),
            "is_initialized": self._is_initialized,
            "plugin_name": self._metadata.name,
            "plugin_version": self._metadata.version
        }
    
    def clear_cache(self) -> None:
        """Clear the feature cache"""
        self._feature_cache.clear()
        logger.info(f"Cleared feature cache for {self._metadata.name}")
    
    def _validate_config(self) -> None:
        """Validate plugin configuration. Override in subclasses."""
        pass
    
    def _perform_initialization(self) -> None:
        """Perform plugin-specific initialization. Override in subclasses."""
        pass
    
    def _perform_shutdown(self) -> None:
        """Perform plugin-specific shutdown. Override in subclasses."""
        pass
    
    def _generate_input_hash(self, input_data: Any) -> str:
        """
        Generate hash for input data for caching purposes.
        
        Args:
            input_data: Input data to hash
            
        Returns:
            String hash of the input data
        """
        try:
            if isinstance(input_data, (str, int, float, bool)):
                content = str(input_data)
            elif isinstance(input_data, (list, tuple)):
                content = str(sorted(input_data) if all(isinstance(x, (str, int, float)) for x in input_data) else input_data)
            elif isinstance(input_data, dict):
                content = str(sorted(input_data.items()))
            elif hasattr(input_data, '__dict__'):
                content = str(sorted(input_data.__dict__.items()))
            else:
                content = str(input_data)
            
            return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            # Fallback to object id if hashing fails
            return str(id(input_data))
    
    def _update_extraction_stats(self, extraction_time: float) -> None:
        """Update extraction statistics"""
        self._extraction_count += 1
        self._last_extraction_time = datetime.utcnow()
    
    def _check_cache(self, input_hash: str) -> Optional[FeatureExtractionResult]:
        """Check if features are cached for this input"""
        return self._feature_cache.get(input_hash)
    
    def _store_in_cache(self, input_hash: str, result: FeatureExtractionResult) -> None:
        """Store extraction result in cache"""
        if len(self._feature_cache) < self.config.get('max_cache_size', 1000):
            self._feature_cache[input_hash] = result


class BaseNumericalExtractor(FeatureExtractorPlugin):
    """Base class for numerical feature extractors"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.scaler = None
        self.feature_statistics = {}
    
    def compute_statistics(self, features: np.ndarray) -> Dict[str, float]:
        """Compute basic statistics for numerical features"""
        return {
            'mean': float(np.mean(features)),
            'std': float(np.std(features)),
            'min': float(np.min(features)),
            'max': float(np.max(features)),
            'median': float(np.median(features))
        }
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using configured method"""
        normalization_method = self.config.get('normalization', 'none')
        
        if normalization_method == 'standard':
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            return (features - mean) / (std + 1e-8)
        elif normalization_method == 'minmax':
            min_val = np.min(features, axis=0)
            max_val = np.max(features, axis=0)
            return (features - min_val) / (max_val - min_val + 1e-8)
        else:
            return features


class BaseCategoricalExtractor(FeatureExtractorPlugin):
    """Base class for categorical feature extractors"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.category_mappings = {}
        self.vocabulary = set()
    
    def encode_categories(self, categories: List[str]) -> np.ndarray:
        """Encode categorical data"""
        encoding_method = self.config.get('encoding', 'onehot')
        
        if encoding_method == 'onehot':
            return self._onehot_encode(categories)
        elif encoding_method == 'label':
            return self._label_encode(categories)
        else:
            raise ValueError(f"Unknown encoding method: {encoding_method}")
    
    def _onehot_encode(self, categories: List[str]) -> np.ndarray:
        """One-hot encode categories"""
        unique_categories = sorted(set(categories))
        encoding = np.zeros((len(categories), len(unique_categories)))
        
        for i, cat in enumerate(categories):
            if cat in unique_categories:
                j = unique_categories.index(cat)
                encoding[i, j] = 1
        
        return encoding
    
    def _label_encode(self, categories: List[str]) -> np.ndarray:
        """Label encode categories"""
        unique_categories = sorted(set(categories))
        category_to_label = {cat: i for i, cat in enumerate(unique_categories)}
        
        return np.array([category_to_label.get(cat, -1) for cat in categories])


# Example implementation for testing
class ExampleFeatureExtractor(FeatureExtractorPlugin):
    """Example feature extractor for testing purposes"""
    
    def _create_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="ExampleFeatureExtractor",
            version="1.0.0",
            author="Universal AI Core",
            description="Example feature extractor for testing",
            supported_input_types=["dict", "str"],
            supported_feature_types=[FeatureType.NUMERICAL, FeatureType.CATEGORICAL]
        )
    
    def extract_features(self, input_data: Any) -> FeatureExtractionResult:
        """Extract example features"""
        start_time = time.time()
        
        try:
            if not self.validate_input(input_data):
                raise ValueError("Invalid input data")
            
            # Generate input hash
            input_hash = self._generate_input_hash(input_data)
            
            # Check cache
            cached_result = self._check_cache(input_hash)
            if cached_result:
                return cached_result
            
            # Extract features
            if isinstance(input_data, dict):
                features = {
                    'dict_size': len(input_data),
                    'has_numeric_values': any(isinstance(v, (int, float)) for v in input_data.values()),
                    'key_count': len(input_data.keys())
                }
                feature_types = {
                    'dict_size': FeatureType.NUMERICAL,
                    'has_numeric_values': FeatureType.BINARY,
                    'key_count': FeatureType.NUMERICAL
                }
            elif isinstance(input_data, str):
                features = {
                    'string_length': len(input_data),
                    'word_count': len(input_data.split()),
                    'has_numbers': any(c.isdigit() for c in input_data)
                }
                feature_types = {
                    'string_length': FeatureType.NUMERICAL,
                    'word_count': FeatureType.NUMERICAL,
                    'has_numbers': FeatureType.BINARY
                }
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
            
            extraction_time = time.time() - start_time
            
            result = FeatureExtractionResult(
                features=features,
                feature_names=list(features.keys()),
                feature_types=feature_types,
                extraction_time=extraction_time,
                status=ExtractionStatus.SUCCESS,
                input_hash=input_hash
            )
            
            # Update stats and cache
            self._update_extraction_stats(extraction_time)
            self._store_in_cache(input_hash, result)
            
            return result
            
        except Exception as e:
            extraction_time = time.time() - start_time
            return FeatureExtractionResult(
                features={},
                feature_names=[],
                feature_types={},
                extraction_time=extraction_time,
                status=ExtractionStatus.ERROR,
                error_message=str(e),
                input_hash=""
            )
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for this extractor"""
        return ['dict_size', 'has_numeric_values', 'key_count', 'string_length', 'word_count', 'has_numbers']
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data"""
        return isinstance(input_data, (dict, str))


# Plugin factory function
def create_feature_extractor(extractor_type: str, config: Optional[Dict[str, Any]] = None) -> FeatureExtractorPlugin:
    """
    Factory function to create feature extractor plugins.
    
    Args:
        extractor_type: Type of extractor to create
        config: Configuration for the extractor
        
    Returns:
        FeatureExtractorPlugin instance
    """
    extractors = {
        'example': ExampleFeatureExtractor,
    }
    
    if extractor_type not in extractors:
        raise ValueError(f"Unknown extractor type: {extractor_type}")
    
    return extractors[extractor_type](config)


if __name__ == "__main__":
    # Test the feature extractor base classes
    print("🔧 Feature Extractor Plugin Base Classes Test")
    print("=" * 50)
    
    # Test example extractor
    extractor = create_feature_extractor('example')
    
    # Initialize
    success = extractor.initialize()
    print(f"✅ Extractor initialized: {success}")
    
    # Test metadata
    metadata = extractor.get_metadata()
    print(f"📋 Plugin: {metadata.name} v{metadata.version}")
    
    # Test feature extraction
    test_data = {"key1": "value1", "key2": 42, "key3": True}
    result = extractor.extract_features(test_data)
    
    print(f"🔍 Extraction status: {result.status.value}")
    print(f"📊 Features extracted: {len(result.features)}")
    print(f"⏱️ Extraction time: {result.extraction_time:.4f}s")
    
    # Test batch extraction
    batch_data = [{"a": 1}, {"b": 2}, "test string"]
    batch_results = extractor.batch_extract_features(batch_data)
    print(f"📦 Batch processed: {len(batch_results)} items")
    
    # Test statistics
    stats = extractor.get_extraction_stats()
    print(f"📈 Extraction count: {stats['extraction_count']}")
    
    # Shutdown
    extractor.shutdown()
    print("\n✅ Feature extractor plugin test completed!")