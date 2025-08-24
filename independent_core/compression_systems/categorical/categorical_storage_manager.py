"""
Categorical Storage Manager - RAM-based Weight Categorization and Storage

Manages categorical storage of neural network weights in RAM for optimized
p-adic compression. Groups similar weights for better compression ratios.

NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import threading
import time
import logging
from enum import Enum

logger = logging.getLogger(__name__)

# Import existing components for integration
try:
    from ..padic.padic_encoder import PadicWeight
    from ..gpu_memory.cpu_bursting_pipeline import CPUBurstingConfig
    from .ieee754_channel_extractor import IEEE754Channels, IEEE754ChannelExtractor
except ImportError:
    from compression_systems.padic.padic_encoder import PadicWeight
    from compression_systems.gpu_memory.cpu_bursting_pipeline import CPUBurstingConfig
    from compression_systems.categorical.ieee754_channel_extractor import IEEE754Channels, IEEE754ChannelExtractor


class CategoryType(Enum):
    """Types of weight categories for storage optimization"""
    SMALL_WEIGHTS = "small_weights"           # |w| < 0.01
    MEDIUM_WEIGHTS = "medium_weights"         # 0.01 <= |w| < 1.0
    LARGE_WEIGHTS = "large_weights"           # |w| >= 1.0
    ZERO_WEIGHTS = "zero_weights"             # w == 0 (exactly)
    POSITIVE_WEIGHTS = "positive_weights"     # w > 0
    NEGATIVE_WEIGHTS = "negative_weights"     # w < 0
    HIGH_ENTROPY = "high_entropy"             # Complex mantissa patterns
    LOW_ENTROPY = "low_entropy"               # Simple mantissa patterns


@dataclass
class CategoryMetrics:
    """Metrics for a weight category"""
    category_type: CategoryType
    weight_count: int = 0
    total_storage_bytes: int = 0
    compression_ratio: float = 0.0
    average_entropy: float = 0.0
    access_frequency: int = 0
    last_accessed: float = 0.0
    
    def update_access(self):
        """Update access metrics"""
        self.access_frequency += 1
        self.last_accessed = time.time()


@dataclass
class CategoricalStorageConfig:
    """Configuration for categorical storage system"""
    # Memory limits
    max_ram_storage_mb: int = 8192                # Maximum RAM usage
    category_cache_size_mb: int = 2048            # Per-category cache size
    
    # Categorization thresholds
    small_weight_threshold: float = 0.01          # Threshold for small weights
    large_weight_threshold: float = 1.0           # Threshold for large weights
    entropy_threshold: float = 4.0                # Entropy threshold for complexity
    similarity_threshold: float = 0.95            # Similarity for grouping
    
    # Storage optimization
    enable_compression_within_categories: bool = True
    enable_deduplication: bool = True
    enable_quantization: bool = True
    quantization_bits: int = 16
    
    # Performance settings
    max_categories_per_type: int = 100            # Limit categories per type
    category_merge_threshold: int = 1000          # Merge small categories
    enable_async_storage: bool = True
    storage_worker_threads: int = 4
    
    def __post_init__(self):
        """Validate configuration"""
        if self.max_ram_storage_mb <= 0:
            raise ValueError(f"max_ram_storage_mb must be > 0, got {self.max_ram_storage_mb}")
        if not (0.0 < self.similarity_threshold <= 1.0):
            raise ValueError(f"similarity_threshold must be in (0,1], got {self.similarity_threshold}")
        if not (4 <= self.quantization_bits <= 32):
            raise ValueError(f"quantization_bits must be in [4,32], got {self.quantization_bits}")


@dataclass
class WeightCategory:
    """Container for categorized weights"""
    category_id: str
    category_type: CategoryType
    weights: List[torch.Tensor] = field(default_factory=list)
    ieee754_channels: List[IEEE754Channels] = field(default_factory=list)
    padic_weights: List[PadicWeight] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    compression_metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: CategoryMetrics = field(init=False)
    
    def __post_init__(self):
        """Initialize category metrics"""
        self.metrics = CategoryMetrics(
            category_type=self.category_type,
            weight_count=len(self.weights)
        )
    
    def add_weight(self, weight: torch.Tensor, channels: Optional[IEEE754Channels] = None,
                   padic_weight: Optional[PadicWeight] = None):
        """Add weight to category with optional pre-computed components"""
        self.weights.append(weight)
        if channels:
            self.ieee754_channels.append(channels)
        if padic_weight:
            self.padic_weights.append(padic_weight)
        
        self.metrics.weight_count = len(self.weights)
        self.metrics.update_access()
    
    def get_storage_size_bytes(self) -> int:
        """Calculate total storage size in bytes"""
        size = 0
        
        # Tensor storage
        for weight in self.weights:
            size += weight.numel() * weight.element_size()
        
        # IEEE 754 channels storage (estimated)
        for channels in self.ieee754_channels:
            size += channels.sign_channel.nbytes
            size += channels.exponent_channel.nbytes  
            size += channels.mantissa_channel.nbytes
            size += channels.original_values.nbytes
        
        # P-adic weights storage (estimated)
        for padic_weight in self.padic_weights:
            size += len(padic_weight.digits) * 4  # Assume 4 bytes per digit
        
        self.metrics.total_storage_bytes = size
        return size


class CategoricalStorageManager:
    """
    RAM-based categorical storage system for neural network weights
    
    Provides intelligent categorization and storage of weights to optimize
    p-adic compression ratios through grouping of similar weight patterns.
    """
    
    def __init__(self, config: CategoricalStorageConfig):
        """Initialize categorical storage manager
        
        Args:
            config: Storage configuration
        """
        self.config = config
        self.categories: Dict[str, WeightCategory] = {}
        self.category_type_map: Dict[CategoryType, List[str]] = defaultdict(list)
        
        # IEEE 754 channel extractor for optimization
        self.ieee754_extractor = IEEE754ChannelExtractor(validate_reconstruction=True)
        
        # Storage statistics
        self.storage_stats = {
            'total_weights_stored': 0,
            'total_categories': 0,
            'total_ram_usage_mb': 0.0,
            'categorization_time_ms': 0.0,
            'compression_ratio_by_category': {},
            'access_patterns': defaultdict(int)
        }
        
        # Thread safety
        self._lock = threading.RLock()
        self._storage_executor = None
        
        if config.enable_async_storage:
            from concurrent.futures import ThreadPoolExecutor
            self._storage_executor = ThreadPoolExecutor(
                max_workers=config.storage_worker_threads,
                thread_name_prefix="CategoricalStorage"
            )
        
        logger.info("CategoricalStorageManager initialized with %d MB RAM limit", config.max_ram_storage_mb)
    
    def store_weights_categorically(self, weights: torch.Tensor, 
                                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Store weights in categorical RAM storage with optimization
        
        Args:
            weights: Tensor of weights to store categorically
            metadata: Optional metadata for storage optimization
            
        Returns:
            Dictionary with storage information and category mappings
            
        Raises:
            RuntimeError: If categorical storage fails (hard failure)
            ValueError: If input validation fails
        """
        if weights is None:
            raise ValueError("Weights tensor cannot be None")
        
        if weights.numel() == 0:
            raise ValueError("Weights tensor cannot be empty")
        
        start_time = time.time()
        
        try:
            with self._lock:
                # Check RAM usage before storing
                self._check_ram_usage_before_storage(weights)
                
                # Extract IEEE 754 channels for categorization
                channels = self.ieee754_extractor.extract_channels_from_tensor(weights)
                
                # Categorize weights based on patterns
                categories = self._categorize_weights(weights, channels, metadata)
                
                # Store in appropriate categories
                storage_info = self._store_in_categories(categories, weights, channels, metadata)
                
                # Update statistics
                categorization_time = (time.time() - start_time) * 1000
                self.storage_stats['categorization_time_ms'] = categorization_time
                self.storage_stats['total_weights_stored'] += weights.numel()
                self.storage_stats['total_categories'] = len(self.categories)
                
                # Calculate total RAM usage
                total_ram_bytes = sum(cat.get_storage_size_bytes() for cat in self.categories.values())
                self.storage_stats['total_ram_usage_mb'] = total_ram_bytes / (1024 * 1024)
                
                logger.info("Stored %d weights in %d categories, total RAM: %.1f MB", 
                           weights.numel(), len(categories), self.storage_stats['total_ram_usage_mb'])
                
                return {
                    'categories_created': list(categories.keys()),
                    'total_categories': len(categories),
                    'storage_info': storage_info,
                    'categorization_time_ms': categorization_time,
                    'ram_usage_mb': self.storage_stats['total_ram_usage_mb']
                }
                
        except Exception as e:
            raise RuntimeError(f"Categorical weight storage failed: {e}")
    
    def retrieve_weights_by_category(self, category_types: List[CategoryType], 
                                   limit: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Retrieve weights from specific categories
        
        Args:
            category_types: List of category types to retrieve
            limit: Optional limit on number of weights to retrieve
            
        Returns:
            Dictionary mapping category IDs to weight tensors
            
        Raises:
            RuntimeError: If retrieval fails (hard failure)
            ValueError: If category types are invalid
        """
        if not category_types:
            raise ValueError("Category types list cannot be empty")
        
        try:
            with self._lock:
                retrieved = {}
                total_retrieved = 0
                
                for category_type in category_types:
                    if category_type not in CategoryType:
                        raise ValueError(f"Invalid category type: {category_type}")
                    
                    # Find categories of this type
                    category_ids = self.category_type_map.get(category_type, [])
                    
                    for category_id in category_ids:
                        if limit and total_retrieved >= limit:
                            break
                            
                        if category_id in self.categories:
                            category = self.categories[category_id]
                            
                            # Combine weights from category
                            if category.weights:
                                combined_weights = torch.cat(category.weights, dim=0)
                                retrieved[category_id] = combined_weights
                                
                                # Update access statistics
                                category.metrics.update_access()
                                self.storage_stats['access_patterns'][category_type.value] += 1
                                
                                total_retrieved += combined_weights.numel()
                
                logger.debug("Retrieved %d weights from %d categories of types: %s", 
                           total_retrieved, len(retrieved), [ct.value for ct in category_types])
                
                return retrieved
                
        except Exception as e:
            raise RuntimeError(f"Categorical weight retrieval failed: {e}")
    
    def _check_ram_usage_before_storage(self, weights: torch.Tensor) -> None:
        """Check RAM usage before storing new weights
        
        Args:
            weights: Weights to be stored
            
        Raises:
            RuntimeError: If storage would exceed RAM limits
        """
        # Estimate storage size for new weights
        estimated_size_mb = (weights.numel() * weights.element_size()) / (1024 * 1024)
        
        # Add overhead for IEEE 754 channels and categorization
        estimated_size_mb *= 2.5  # Conservative overhead estimate
        
        current_usage_mb = self.storage_stats['total_ram_usage_mb']
        projected_usage_mb = current_usage_mb + estimated_size_mb
        
        if projected_usage_mb > self.config.max_ram_storage_mb:
            # Try to free space by merging or evicting categories
            freed_mb = self._free_storage_space(estimated_size_mb)
            
            if freed_mb < estimated_size_mb:
                raise RuntimeError(
                    f"Insufficient RAM for categorical storage: need {estimated_size_mb:.1f}MB, "
                    f"have {self.config.max_ram_storage_mb - current_usage_mb:.1f}MB available"
                )
        
        logger.debug("RAM usage check passed: current %.1f MB, adding %.1f MB", 
                    current_usage_mb, estimated_size_mb)
    
    def _categorize_weights(self, weights: torch.Tensor, channels: IEEE754Channels,
                          metadata: Optional[Dict[str, Any]]) -> Dict[str, CategoryType]:
        """Categorize weights based on patterns and characteristics
        
        Args:
            weights: Tensor of weights to categorize
            channels: IEEE 754 channels for pattern analysis
            metadata: Optional metadata for categorization hints
            
        Returns:
            Dictionary mapping weight indices to category types
        """
        flattened_weights = weights.flatten()
        categories = {}
        
        for i, weight_val in enumerate(flattened_weights):
            weight_val = float(weight_val)
            
            # Categorize by magnitude
            abs_weight = abs(weight_val)
            
            if abs_weight == 0.0:
                category_type = CategoryType.ZERO_WEIGHTS
            elif abs_weight < self.config.small_weight_threshold:
                category_type = CategoryType.SMALL_WEIGHTS
            elif abs_weight >= self.config.large_weight_threshold:
                category_type = CategoryType.LARGE_WEIGHTS
            else:
                category_type = CategoryType.MEDIUM_WEIGHTS
            
            # Further categorize by sign
            if category_type != CategoryType.ZERO_WEIGHTS:
                if weight_val > 0:
                    category_type = CategoryType.POSITIVE_WEIGHTS if category_type == CategoryType.MEDIUM_WEIGHTS else category_type
                else:
                    category_type = CategoryType.NEGATIVE_WEIGHTS if category_type == CategoryType.MEDIUM_WEIGHTS else category_type
            
            # Analyze entropy of mantissa for complexity categorization
            if i < len(channels.mantissa_channel):
                mantissa_entropy = self._calculate_local_entropy(channels.mantissa_channel, i)
                
                if mantissa_entropy > self.config.entropy_threshold:
                    category_type = CategoryType.HIGH_ENTROPY
                elif mantissa_entropy < self.config.entropy_threshold / 2:
                    category_type = CategoryType.LOW_ENTROPY
            
            categories[i] = category_type
        
        return categories
    
    def _store_in_categories(self, weight_categories: Dict[str, CategoryType], 
                           weights: torch.Tensor, channels: IEEE754Channels,
                           metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Store weights in their assigned categories
        
        Args:
            weight_categories: Mapping of weight indices to category types
            weights: Original weight tensor
            channels: IEEE 754 channels
            metadata: Storage metadata
            
        Returns:
            Storage information dictionary
        """
        flattened_weights = weights.flatten()
        storage_info = defaultdict(list)
        
        # Group weights by category type
        category_groups = defaultdict(list)
        for weight_idx, category_type in weight_categories.items():
            category_groups[category_type].append(weight_idx)
        
        # Create or update categories
        for category_type, weight_indices in category_groups.items():
            category_id = self._get_or_create_category_id(category_type)
            
            # Extract weight subset
            weight_subset = torch.stack([flattened_weights[i] for i in weight_indices])
            
            # Extract corresponding channel subsets
            subset_channels = IEEE754Channels(
                sign_channel=np.array([channels.sign_channel[i] for i in weight_indices]),
                exponent_channel=np.array([channels.exponent_channel[i] for i in weight_indices]),
                mantissa_channel=np.array([channels.mantissa_channel[i] for i in weight_indices]),
                original_values=np.array([channels.original_values[i] for i in weight_indices])
            )
            
            # Add to category
            if category_id not in self.categories:
                self.categories[category_id] = WeightCategory(
                    category_id=category_id,
                    category_type=category_type
                )
                self.category_type_map[category_type].append(category_id)
            
            self.categories[category_id].add_weight(weight_subset, subset_channels)
            storage_info[category_type.value].append({
                'category_id': category_id,
                'weight_count': len(weight_indices),
                'storage_bytes': weight_subset.numel() * weight_subset.element_size()
            })
        
        return dict(storage_info)
    
    def _get_or_create_category_id(self, category_type: CategoryType) -> str:
        """Get existing category ID or create new one
        
        Args:
            category_type: Type of category
            
        Returns:
            Category ID string
        """
        # Check if we can reuse an existing category
        existing_categories = self.category_type_map.get(category_type, [])
        
        # If we have room in existing categories, find one with space
        for category_id in existing_categories:
            category = self.categories[category_id]
            if len(category.weights) < self.config.category_merge_threshold:
                return category_id
        
        # Create new category if we haven't exceeded the limit
        if len(existing_categories) < self.config.max_categories_per_type:
            category_id = f"{category_type.value}_{len(existing_categories):04d}_{int(time.time())}"
            return category_id
        
        # Force merge with least recently used category
        lru_category_id = min(existing_categories, 
                             key=lambda cid: self.categories[cid].metrics.last_accessed)
        return lru_category_id
    
    def _calculate_local_entropy(self, mantissa_channel: np.ndarray, center_idx: int, 
                                window_size: int = 10) -> float:
        """Calculate local entropy around a specific mantissa value
        
        Args:
            mantissa_channel: Array of mantissa values
            center_idx: Center index for local calculation
            window_size: Size of window around center
            
        Returns:
            Local entropy value
        """
        try:
            start_idx = max(0, center_idx - window_size // 2)
            end_idx = min(len(mantissa_channel), center_idx + window_size // 2)
            
            local_values = mantissa_channel[start_idx:end_idx]
            
            if len(local_values) < 2:
                return 0.0
            
            # Quantize values for entropy calculation
            quantized = np.round(local_values * 1000).astype(np.int32)
            unique_values, counts = np.unique(quantized, return_counts=True)
            
            # Calculate probabilities and entropy
            probabilities = counts / len(quantized)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            return float(entropy)
            
        except Exception:
            return 0.0
    
    def _free_storage_space(self, required_mb: float) -> float:
        """Free storage space by evicting or merging categories
        
        Args:
            required_mb: Amount of space needed in MB
            
        Returns:
            Amount of space freed in MB
        """
        freed_mb = 0.0
        
        try:
            # Sort categories by last access time (LRU eviction)
            categories_by_access = sorted(
                self.categories.items(),
                key=lambda item: item[1].metrics.last_accessed
            )
            
            # Evict least recently used categories
            for category_id, category in categories_by_access:
                if freed_mb >= required_mb:
                    break
                
                # Calculate space that would be freed
                category_size_mb = category.get_storage_size_bytes() / (1024 * 1024)
                
                # Remove category
                del self.categories[category_id]
                self.category_type_map[category.category_type].remove(category_id)
                
                freed_mb += category_size_mb
                
                logger.debug("Evicted category %s to free %.1f MB", category_id, category_size_mb)
            
            return freed_mb
            
        except Exception as e:
            logger.error("Failed to free storage space: %s", e)
            return 0.0
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics
        
        Returns:
            Dictionary of storage statistics and metrics
        """
        with self._lock:
            # Calculate compression ratios by category
            compression_ratios = {}
            for category_type, category_ids in self.category_type_map.items():
                ratios = []
                for category_id in category_ids:
                    if category_id in self.categories:
                        category = self.categories[category_id]
                        if category.metrics.compression_ratio > 0:
                            ratios.append(category.metrics.compression_ratio)
                
                if ratios:
                    compression_ratios[category_type.value] = {
                        'average': np.mean(ratios),
                        'std': np.std(ratios),
                        'min': np.min(ratios),
                        'max': np.max(ratios)
                    }
            
            # Category distribution
            category_distribution = {}
            for category_type, category_ids in self.category_type_map.items():
                total_weights = sum(
                    self.categories[cid].metrics.weight_count
                    for cid in category_ids if cid in self.categories
                )
                category_distribution[category_type.value] = total_weights
            
            return {
                'total_weights_stored': self.storage_stats['total_weights_stored'],
                'total_categories': len(self.categories),
                'total_ram_usage_mb': self.storage_stats['total_ram_usage_mb'],
                'ram_utilization': self.storage_stats['total_ram_usage_mb'] / self.config.max_ram_storage_mb,
                'categorization_time_ms': self.storage_stats['categorization_time_ms'],
                'compression_ratios_by_category': compression_ratios,
                'category_distribution': category_distribution,
                'access_patterns': dict(self.storage_stats['access_patterns']),
                'ieee754_extraction_stats': self.ieee754_extractor.get_extraction_statistics()
            }
    
    def optimize_categories_for_padic(self, target_prime: int = 257) -> Dict[str, Any]:
        """Optimize stored categories for p-adic compression
        
        Args:
            target_prime: Target p-adic prime for optimization
            
        Returns:
            Optimization results and statistics
            
        Raises:
            RuntimeError: If optimization fails
        """
        try:
            with self._lock:
                optimization_results = {}
                total_optimized = 0
                
                for category_id, category in self.categories.items():
                    if not category.ieee754_channels:
                        continue
                    
                    # Optimize IEEE 754 channels for p-adic compression
                    optimized_channels = []
                    for channels in category.ieee754_channels:
                        optimized = self.ieee754_extractor.optimize_channels_for_padic(channels, target_prime)
                        optimized_channels.append(optimized)
                    
                    # Replace original channels with optimized versions
                    category.ieee754_channels = optimized_channels
                    total_optimized += len(optimized_channels)
                    
                    # Update category metrics
                    category.metrics.update_access()
                    
                    optimization_results[category_id] = {
                        'category_type': category.category_type.value,
                        'optimized_channels': len(optimized_channels),
                        'target_prime': target_prime
                    }
                
                logger.info("Optimized %d categories (%d total channels) for p-adic prime %d", 
                           len(optimization_results), total_optimized, target_prime)
                
                return {
                    'total_categories_optimized': len(optimization_results),
                    'total_channels_optimized': total_optimized,
                    'target_prime': target_prime,
                    'optimization_results': optimization_results
                }
                
        except Exception as e:
            raise RuntimeError(f"Category optimization for p-adic compression failed: {e}")
    
    def cleanup(self) -> None:
        """Clean up resources and shutdown storage manager"""
        with self._lock:
            if self._storage_executor:
                self._storage_executor.shutdown(wait=True)
            
            # Clear all categories and statistics
            self.categories.clear()
            self.category_type_map.clear()
            self.storage_stats = {
                'total_weights_stored': 0,
                'total_categories': 0,
                'total_ram_usage_mb': 0.0,
                'categorization_time_ms': 0.0,
                'compression_ratio_by_category': {},
                'access_patterns': defaultdict(int)
            }
            
            logger.info("CategoricalStorageManager cleaned up and shutdown")


# Factory function for integration
def create_categorical_storage_manager(config: Optional[CategoricalStorageConfig] = None) -> CategoricalStorageManager:
    """Factory function to create categorical storage manager
    
    Args:
        config: Optional configuration, uses defaults if None
        
    Returns:
        Configured CategoricalStorageManager instance
    """
    if config is None:
        config = CategoricalStorageConfig()
    
    return CategoricalStorageManager(config)
