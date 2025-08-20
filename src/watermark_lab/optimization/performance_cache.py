"""High-performance caching system for watermark operations."""

import time
import hashlib
import pickle
import threading
import logging
from typing import Any, Dict, Optional, Callable, TypeVar, Generic, NamedTuple
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum
import weakref

T = TypeVar('T')

class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"
    LFU = "lfu" 
    TTL = "ttl"
    ADAPTIVE = "adaptive"

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_access = time.time()

class PerformanceCache(Generic[T]):
    """High-performance cache with multiple eviction strategies."""
    
    def __init__(self, 
                 max_size: int = 1000,
                 max_memory_mb: float = 100.0,
                 default_ttl: Optional[float] = None,
                 strategy: CacheStrategy = CacheStrategy.LRU,
                 cleanup_interval: float = 60.0):
        """
        Initialize performance cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default time-to-live in seconds
            strategy: Cache eviction strategy
            cleanup_interval: Cleanup interval in seconds
        """
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.default_ttl = default_ttl
        self.strategy = strategy
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._current_memory = 0
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        # Cleanup thread
        self._cleanup_interval = cleanup_interval
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        
        self.logger = logging.getLogger(f"cache.{id(self)}")
        self._start_cleanup()
    
    def _start_cleanup(self) -> None:
        """Start background cleanup thread."""
        if self._cleanup_interval > 0:
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop, 
                daemon=True
            )
            self._cleanup_thread.start()
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self._stop_cleanup.wait(self._cleanup_interval):
            try:
                self._cleanup_expired()
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Create a stable hash from arguments
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            # Fallback estimation
            return 1024  # Default 1KB
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry and update memory tracking."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._current_memory -= entry.size_bytes
            self._evictions += 1
    
    def _evict_entries(self) -> None:
        """Evict entries based on strategy."""
        if self.strategy == CacheStrategy.LRU:
            self._evict_lru()
        elif self.strategy == CacheStrategy.LFU:
            self._evict_lfu()
        elif self.strategy == CacheStrategy.TTL:
            self._cleanup_expired()
        elif self.strategy == CacheStrategy.ADAPTIVE:
            self._evict_adaptive()
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        while (len(self._cache) >= self.max_size or 
               self._current_memory > self.max_memory_bytes):
            if not self._cache:
                break
            # OrderedDict maintains insertion order, move_to_end on access maintains LRU
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key)
    
    def _evict_lfu(self) -> None:
        """Evict least frequently used entries."""
        while (len(self._cache) >= self.max_size or 
               self._current_memory > self.max_memory_bytes):
            if not self._cache:
                break
            
            # Find entry with lowest access count
            lfu_key = min(self._cache.keys(), 
                         key=lambda k: self._cache[k].access_count)
            self._remove_entry(lfu_key)
    
    def _evict_adaptive(self) -> None:
        """Adaptive eviction combining multiple factors."""
        while (len(self._cache) >= self.max_size or 
               self._current_memory > self.max_memory_bytes):
            if not self._cache:
                break
            
            current_time = time.time()
            
            # Score entries based on multiple factors
            def score_entry(key: str, entry: CacheEntry) -> float:
                age = current_time - entry.timestamp
                last_access_age = current_time - entry.last_access
                access_frequency = entry.access_count / max(age, 1)
                size_penalty = entry.size_bytes / 1024  # Size in KB
                
                # Lower score = higher priority for eviction
                return access_frequency - (last_access_age / 3600) - (size_penalty / 100)
            
            # Evict entry with lowest score
            worst_key = min(self._cache.keys(),
                          key=lambda k: score_entry(k, self._cache[k]))
            self._remove_entry(worst_key)
    
    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                if entry.is_expired():
                    self._remove_entry(key)
                    self._misses += 1
                    return default
                
                # Update access metadata
                entry.touch()
                
                # Move to end for LRU
                if self.strategy == CacheStrategy.LRU:
                    self._cache.move_to_end(key)
                
                self._hits += 1
                return entry.value
            
            self._misses += 1
            return default
    
    def put(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Put value in cache."""
        with self._lock:
            # Estimate size
            size_bytes = self._estimate_size(value)
            
            # Check if single item exceeds memory limit
            if size_bytes > self.max_memory_bytes:
                self.logger.warning(f"Item too large for cache: {size_bytes} bytes")
                return
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Evict entries if necessary
            self._evict_entries()
            
            # Create new entry
            entry = CacheEntry(
                value=value,
                size_bytes=size_bytes,
                ttl=ttl or self.default_ttl
            )
            
            self._cache[key] = entry
            self._current_memory += size_bytes
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "memory_mb": self._current_memory / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "strategy": self.strategy.value
            }
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, '_stop_cleanup'):
            self._stop_cleanup.set()

class CacheManager:
    """Global cache manager for different cache types."""
    
    def __init__(self):
        self._caches: Dict[str, PerformanceCache] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger("cache.manager")
    
    def get_cache(self, name: str, **kwargs) -> PerformanceCache:
        """Get or create a named cache."""
        with self._lock:
            if name not in self._caches:
                self._caches[name] = PerformanceCache(**kwargs)
                self.logger.info(f"Created cache: {name}")
            return self._caches[name]
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        with self._lock:
            return {name: cache.get_stats() 
                   for name, cache in self._caches.items()}
    
    def clear_all(self) -> None:
        """Clear all caches."""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()

# Global cache manager
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

def cached(cache_name: str = "default", 
           ttl: Optional[float] = None,
           key_func: Optional[Callable] = None,
           cache_kwargs: Optional[Dict] = None) -> Callable:
    """
    Decorator for caching function results.
    
    Args:
        cache_name: Name of cache to use
        ttl: Time-to-live for cached entries
        key_func: Custom function to generate cache keys
        cache_kwargs: Arguments for cache creation
    """
    def decorator(func: Callable) -> Callable:
        cache = get_cache_manager().get_cache(
            cache_name, 
            **(cache_kwargs or {})
        )
        
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache._generate_key(*args, **kwargs)
            
            # Try cache first
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl=ttl)
            
            return result
        
        # Add cache management methods to function
        wrapper.cache = cache
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_stats = lambda: cache.get_stats()
        
        return wrapper
    
    return decorator

# Specialized caches for common operations
def get_model_cache() -> PerformanceCache:
    """Get cache for model objects."""
    return get_cache_manager().get_cache(
        "models",
        max_size=10,  # Models are large
        max_memory_mb=2000,  # 2GB for models
        strategy=CacheStrategy.LRU
    )

def get_tokenizer_cache() -> PerformanceCache:
    """Get cache for tokenizers."""
    return get_cache_manager().get_cache(
        "tokenizers", 
        max_size=50,
        max_memory_mb=500,  # 500MB for tokenizers
        strategy=CacheStrategy.LRU
    )

def get_detection_cache() -> PerformanceCache:
    """Get cache for detection results."""
    return get_cache_manager().get_cache(
        "detection",
        max_size=10000,
        max_memory_mb=100,
        default_ttl=3600,  # 1 hour TTL
        strategy=CacheStrategy.ADAPTIVE
    )

def get_generation_cache() -> PerformanceCache:
    """Get cache for generation results."""
    return get_cache_manager().get_cache(
        "generation",
        max_size=5000,
        max_memory_mb=200,
        default_ttl=1800,  # 30 minutes TTL
        strategy=CacheStrategy.LRU
    )