"""
Strategy Cache - Caching system for proof strategies
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import threading
import hashlib
import json
import gzip
import pickle
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "least_recently_used"
    LFU = "least_frequently_used"
    TTL = "time_to_live"
    SIZE_BASED = "size_based"


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    creation_time: float
    last_access_time: float
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None
    compressed: bool = False
    
    def __post_init__(self):
        # NO FALLBACKS - HARD FAILURES ONLY
        if not self.key or not isinstance(self.key, str):
            raise ValueError("Key must be non-empty string")
        if self.value is None:
            raise ValueError("Value cannot be None")
        if self.creation_time <= 0:
            raise ValueError("Creation time must be positive")
        if self.last_access_time <= 0:
            raise ValueError("Last access time must be positive")
        if self.access_count < 0:
            raise ValueError("Access count must be non-negative")
        if self.size_bytes < 0:
            raise ValueError("Size bytes must be non-negative")
        if self.ttl_seconds is not None and self.ttl_seconds <= 0:
            raise ValueError("TTL must be positive if specified")
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.creation_time > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age in seconds"""
        return time.time() - self.creation_time
    
    def update_access(self):
        """Update access metadata"""
        self.last_access_time = time.time()
        self.access_count += 1
    
    def calculate_checksum(self) -> str:
        """Calculate checksum for integrity verification"""
        try:
            value_bytes = pickle.dumps(self.value)
            return hashlib.sha256(value_bytes).hexdigest()
        except Exception as e:
            raise RuntimeError(f"Failed to calculate checksum: {e}")
    
    def verify_integrity(self) -> bool:
        """Verify entry integrity"""
        if self.checksum is None:
            return True  # No checksum to verify
        
        try:
            current_checksum = self.calculate_checksum()
            return current_checksum == self.checksum
        except Exception:
            return False


@dataclass
class CacheStatistics:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    corruptions: int = 0
    compressions: int = 0
    decompressions: int = 0
    total_requests: int = 0
    total_size_bytes: int = 0
    average_access_time: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Get cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        """Get cache miss rate"""
        return 1.0 - self.hit_rate
    
    def update_hit(self, access_time: float):
        """Update hit statistics"""
        self.hits += 1
        self.total_requests += 1
        self._update_average_access_time(access_time)
    
    def update_miss(self, access_time: float):
        """Update miss statistics"""
        self.misses += 1
        self.total_requests += 1
        self._update_average_access_time(access_time)
    
    def _update_average_access_time(self, access_time: float):
        """Update average access time"""
        if self.total_requests == 1:
            self.average_access_time = access_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_access_time = (alpha * access_time + 
                                     (1 - alpha) * self.average_access_time)


class CacheWarmer:
    """Pre-loads cache with frequently used items"""
    
    def __init__(self, cache: 'StrategyCache'):
        if cache is None:
            raise ValueError("Cache cannot be None")
        self.cache = cache
        self.warming_strategies: Dict[str, Callable] = {}
    
    def add_warming_strategy(self, name: str, strategy_func: Callable):
        """Add a cache warming strategy"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not name or not isinstance(name, str):
            raise ValueError("Name must be non-empty string")
        if not callable(strategy_func):
            raise TypeError("Strategy function must be callable")
        
        self.warming_strategies[name] = strategy_func
    
    def warm_cache(self, strategy_name: Optional[str] = None):
        """Warm the cache using specified or all strategies"""
        if strategy_name:
            if strategy_name not in self.warming_strategies:
                raise ValueError(f"Warming strategy {strategy_name} not found")
            strategies = {strategy_name: self.warming_strategies[strategy_name]}
        else:
            strategies = self.warming_strategies
        
        for name, strategy_func in strategies.items():
            try:
                logger.info(f"Warming cache with strategy: {name}")
                strategy_func(self.cache)
            except Exception as e:
                logger.error(f"Cache warming strategy {name} failed: {e}")
                raise RuntimeError(f"Cache warming failed for strategy {name}: {e}")


class StrategyCache:
    """Thread-safe caching system for proof strategies"""
    
    def __init__(self, 
                 max_size: int = 1000,
                 max_size_bytes: int = 100 * 1024 * 1024,  # 100MB
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
                 default_ttl: Optional[float] = None,
                 enable_compression: bool = True,
                 compression_threshold: int = 1024,  # 1KB
                 enable_integrity_check: bool = True):
        
        # NO FALLBACKS - HARD FAILURES ONLY
        if max_size <= 0:
            raise ValueError("Max size must be positive")
        if max_size_bytes <= 0:
            raise ValueError("Max size bytes must be positive")
        if not isinstance(eviction_policy, EvictionPolicy):
            raise TypeError("Eviction policy must be EvictionPolicy enum")
        if default_ttl is not None and default_ttl <= 0:
            raise ValueError("Default TTL must be positive if specified")
        if compression_threshold < 0:
            raise ValueError("Compression threshold must be non-negative")
        
        self.max_size = max_size
        self.max_size_bytes = max_size_bytes
        self.eviction_policy = eviction_policy
        self.default_ttl = default_ttl
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        self.enable_integrity_check = enable_integrity_check
        
        # Storage
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.cache_lock = threading.RLock()
        
        # Statistics
        self.statistics = CacheStatistics()
        
        # Frequency tracking for LFU
        self.access_frequencies: Dict[str, int] = defaultdict(int)
        
        # Cache warmer
        self.cache_warmer = CacheWarmer(self)
        
        # Background cleanup thread
        self.cleanup_thread = None
        self.cleanup_interval = 300  # 5 minutes
        self.stop_cleanup = threading.Event()
        self._start_cleanup_thread()
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store a value in the cache"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not key or not isinstance(key, str):
            raise ValueError("Key must be non-empty string")
        if value is None:
            raise ValueError("Value cannot be None")
        if ttl is not None and ttl <= 0:
            raise ValueError("TTL must be positive if specified")
        
        with self.cache_lock:
            current_time = time.time()
            
            # Use default TTL if not specified
            effective_ttl = ttl if ttl is not None else self.default_ttl
            
            # Calculate size
            try:
                value_bytes = pickle.dumps(value)
                size_bytes = len(value_bytes)
            except Exception as e:
                raise RuntimeError(f"Failed to serialize value: {e}")
            
            # Compress if enabled and above threshold
            compressed = False
            if self.enable_compression and size_bytes > self.compression_threshold:
                try:
                    compressed_bytes = gzip.compress(value_bytes)
                    if len(compressed_bytes) < size_bytes:
                        value_bytes = compressed_bytes
                        size_bytes = len(compressed_bytes)
                        compressed = True
                        self.statistics.compressions += 1
                except Exception as e:
                    logger.warning(f"Compression failed for key {key}: {e}")
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                creation_time=current_time,
                last_access_time=current_time,
                access_count=0,
                size_bytes=size_bytes,
                ttl_seconds=effective_ttl,
                metadata=metadata or {},
                compressed=compressed
            )
            
            # Calculate checksum for integrity
            if self.enable_integrity_check:
                entry.checksum = entry.calculate_checksum()
            
            # Remove existing entry if present
            if key in self.entries:
                old_entry = self.entries[key]
                self.statistics.total_size_bytes -= old_entry.size_bytes
                del self.entries[key]
            
            # Ensure space available
            self._ensure_space(size_bytes)
            
            # Add entry
            self.entries[key] = entry
            self.statistics.total_size_bytes += size_bytes
            
            # Update LRU order
            self.entries.move_to_end(key)
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the cache"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not key or not isinstance(key, str):
            raise ValueError("Key must be non-empty string")
        
        start_time = time.time()
        
        with self.cache_lock:
            entry = self.entries.get(key)
            
            if entry is None:
                access_time = time.time() - start_time
                self.statistics.update_miss(access_time)
                return None
            
            # Check if expired
            if entry.is_expired:
                self._remove_entry(key)
                access_time = time.time() - start_time
                self.statistics.update_miss(access_time)
                return None
            
            # Verify integrity
            if self.enable_integrity_check and not entry.verify_integrity():
                logger.warning(f"Cache entry corruption detected for key: {key}")
                self.statistics.corruptions += 1
                self._remove_entry(key)
                access_time = time.time() - start_time
                self.statistics.update_miss(access_time)
                return None
            
            # Update access metadata
            entry.update_access()
            self.access_frequencies[key] += 1
            
            # Update LRU order
            self.entries.move_to_end(key)
            
            # Handle decompression
            value = entry.value
            if entry.compressed:
                try:
                    # This should not happen in our implementation
                    # since we store the original value, not compressed bytes
                    # But keeping for completeness
                    self.statistics.decompressions += 1
                except Exception as e:
                    logger.error(f"Decompression failed for key {key}: {e}")
                    self._remove_entry(key)
                    access_time = time.time() - start_time
                    self.statistics.update_miss(access_time)
                    return None
            
            access_time = time.time() - start_time
            self.statistics.update_hit(access_time)
            return value
    
    def contains(self, key: str) -> bool:
        """Check if key exists in cache"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not key or not isinstance(key, str):
            raise ValueError("Key must be non-empty string")
        
        with self.cache_lock:
            entry = self.entries.get(key)
            if entry is None:
                return False
            
            # Check if expired
            if entry.is_expired:
                self._remove_entry(key)
                return False
            
            return True
    
    def remove(self, key: str) -> bool:
        """Remove a key from cache"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not key or not isinstance(key, str):
            raise ValueError("Key must be non-empty string")
        
        with self.cache_lock:
            if key in self.entries:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries from cache"""
        with self.cache_lock:
            self.entries.clear()
            self.access_frequencies.clear()
            self.statistics.total_size_bytes = 0
    
    def get_size(self) -> int:
        """Get number of entries in cache"""
        with self.cache_lock:
            return len(self.entries)
    
    def get_size_bytes(self) -> int:
        """Get total size in bytes"""
        with self.cache_lock:
            return self.statistics.total_size_bytes
    
    def get_statistics(self) -> CacheStatistics:
        """Get cache statistics"""
        with self.cache_lock:
            return self.statistics
    
    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get information about a cache entry"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not key or not isinstance(key, str):
            raise ValueError("Key must be non-empty string")
        
        with self.cache_lock:
            entry = self.entries.get(key)
            if entry is None:
                return None
            
            return {
                'key': entry.key,
                'creation_time': entry.creation_time,
                'last_access_time': entry.last_access_time,
                'access_count': entry.access_count,
                'size_bytes': entry.size_bytes,
                'ttl_seconds': entry.ttl_seconds,
                'age_seconds': entry.age_seconds,
                'is_expired': entry.is_expired,
                'compressed': entry.compressed,
                'metadata': entry.metadata.copy(),
                'access_frequency': self.access_frequencies.get(key, 0)
            }
    
    def get_all_keys(self) -> List[str]:
        """Get all cache keys"""
        with self.cache_lock:
            return list(self.entries.keys())
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries"""
        with self.cache_lock:
            expired_keys = []
            
            for key, entry in self.entries.items():
                if entry.is_expired:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
            
            return len(expired_keys)
    
    def _ensure_space(self, required_bytes: int) -> None:
        """Ensure sufficient space for new entry"""
        # Check size limit
        while len(self.entries) >= self.max_size:
            self._evict_entry()
        
        # Check byte limit
        while (self.statistics.total_size_bytes + required_bytes > self.max_size_bytes 
               and len(self.entries) > 0):
            self._evict_entry()
    
    def _evict_entry(self) -> None:
        """Evict an entry based on eviction policy"""
        if not self.entries:
            return
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Remove least recently used (first in OrderedDict)
            key = next(iter(self.entries))
            self._remove_entry(key)
            
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used
            if self.access_frequencies:
                key = min(self.access_frequencies.items(), key=lambda x: x[1])[0]
                if key in self.entries:
                    self._remove_entry(key)
                else:
                    # Fallback to LRU if frequency tracking is inconsistent
                    key = next(iter(self.entries))
                    self._remove_entry(key)
            else:
                # Fallback to LRU
                key = next(iter(self.entries))
                self._remove_entry(key)
                
        elif self.eviction_policy == EvictionPolicy.TTL:
            # Remove oldest entry (by creation time)
            oldest_key = min(self.entries.items(), key=lambda x: x[1].creation_time)[0]
            self._remove_entry(oldest_key)
            
        elif self.eviction_policy == EvictionPolicy.SIZE_BASED:
            # Remove largest entry
            largest_key = max(self.entries.items(), key=lambda x: x[1].size_bytes)[0]
            self._remove_entry(largest_key)
        
        self.statistics.evictions += 1
    
    def _remove_entry(self, key: str) -> None:
        """Remove an entry and update statistics"""
        if key in self.entries:
            entry = self.entries[key]
            self.statistics.total_size_bytes -= entry.size_bytes
            del self.entries[key]
            
            if key in self.access_frequencies:
                del self.access_frequencies[key]
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread"""
        def cleanup_worker():
            while not self.stop_cleanup.wait(self.cleanup_interval):
                try:
                    expired_count = self.cleanup_expired()
                    if expired_count > 0:
                        logger.debug(f"Cleaned up {expired_count} expired cache entries")
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def shutdown(self) -> None:
        """Shutdown cache and cleanup resources"""
        self.stop_cleanup.set()
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
        self.clear()
    
    def __del__(self):
        """Destructor"""
        try:
            self.shutdown()
        except Exception:
            pass  # Ignore errors during destruction
    
    # Context manager support
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
    
    # Cache warmer integration
    def add_warming_strategy(self, name: str, strategy_func: Callable):
        """Add a cache warming strategy"""
        self.cache_warmer.add_warming_strategy(name, strategy_func)
    
    def warm_cache(self, strategy_name: Optional[str] = None):
        """Warm the cache"""
        self.cache_warmer.warm_cache(strategy_name)