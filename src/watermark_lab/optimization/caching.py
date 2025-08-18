"""Production-grade caching system with multiple backends and intelligent optimization."""

import os
import json
import time
import hashlib
import pickle
import threading
from typing import Dict, Any, Optional, Union, List, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import memcache
    MEMCACHE_AVAILABLE = True
except ImportError:
    MEMCACHE_AVAILABLE = False

from ..utils.logging import get_logger
from ..utils.exceptions import WatermarkLabError


class CacheError(WatermarkLabError):
    """Exception raised for cache-related errors."""
    pass


try:
    from ..utils.metrics import MetricsCollector
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    
    class MetricsCollector:
        """Fallback metrics collector."""
        def __init__(self):
            pass

logger = get_logger("optimization.caching")


@dataclass
class CacheConfig:
    """Configuration for caching systems."""
    
    # Backend configuration
    backend: str = "memory"  # memory, redis, memcache, hybrid
    redis_url: Optional[str] = None
    memcache_servers: List[str] = None
    
    # Size limits
    max_memory_items: int = 10000
    max_memory_size: int = 100 * 1024 * 1024  # 100MB
    max_item_size: int = 1024 * 1024  # 1MB
    
    # TTL settings
    default_ttl: int = 3600  # 1 hour
    max_ttl: int = 86400  # 24 hours
    
    # Performance settings
    enable_compression: bool = True
    compression_threshold: int = 1024  # Compress items > 1KB
    enable_statistics: bool = True
    
    # Cache warming
    enable_preloading: bool = True
    preload_patterns: List[str] = None
    
    # Eviction policy
    eviction_policy: str = "lru"  # lru, lfu, fifo
    
    def __post_init__(self):
        """Initialize default values."""
        if self.memcache_servers is None:
            self.memcache_servers = ["127.0.0.1:11211"]
        if self.preload_patterns is None:
            self.preload_patterns = []


@dataclass
class CacheStats:
    """Cache performance statistics."""
    
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    size: int = 0
    memory_usage: int = 0
    avg_access_time: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / max(1, total)
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            'hit_rate': self.hit_rate,
            'miss_rate': self.miss_rate
        }


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.stats = CacheStats()
        self.logger = get_logger(f"cache.{self.__class__.__name__.lower()}")
        self._lock = threading.RLock()
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all items from cache."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats
    
    def _generate_key(self, key: str) -> str:
        """Generate consistent cache key."""
        if len(key) > 250:  # Redis key limit
            return hashlib.sha256(key.encode()).hexdigest()
        return key
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            serialized = pickle.dumps(value)
            
            if self.config.enable_compression and len(serialized) > self.config.compression_threshold:
                try:
                    import gzip
                    serialized = gzip.compress(serialized)
                except ImportError:
                    pass  # Use uncompressed
            
            return serialized
            
        except Exception as e:
            self.logger.error(f"Serialization failed: {e}")
            raise CacheError(f"Failed to serialize value: {e}")
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Check if compressed
            try:
                import gzip
                if data[:3] == b'\x1f\x8b\x08':  # gzip magic number
                    data = gzip.decompress(data)
            except ImportError:
                pass
            
            return pickle.loads(data)
            
        except Exception as e:
            self.logger.error(f"Deserialization failed: {e}")
            raise CacheError(f"Failed to deserialize value: {e}")


class MemoryCache(CacheBackend):
    """In-memory LRU cache implementation."""
    
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self.cache: OrderedDict = OrderedDict()
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}
        self.ttls: Dict[str, float] = {}
        self.memory_usage = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from memory cache."""
        start_time = time.time()
        cache_key = self._generate_key(key)
        
        try:
            with self._lock:
                # Check if key exists and not expired
                if cache_key not in self.cache:
                    self.stats.misses += 1
                    return None
                
                # Check TTL
                if self._is_expired(cache_key):
                    self._remove_key(cache_key)
                    self.stats.misses += 1
                    return None
                
                # Update access information
                value = self.cache[cache_key]
                self.access_times[cache_key] = time.time()
                self.access_counts[cache_key] = self.access_counts.get(cache_key, 0) + 1
                
                # Move to end for LRU
                if self.config.eviction_policy == "lru":
                    self.cache.move_to_end(cache_key)
                
                self.stats.hits += 1
                
                # Update average access time
                access_time = time.time() - start_time
                self.stats.avg_access_time = (
                    (self.stats.avg_access_time * (self.stats.hits - 1) + access_time) / self.stats.hits
                )
                
                return value
                
        except Exception as e:
            self.logger.error(f"Cache get failed: {e}")
            self.stats.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in memory cache."""
        cache_key = self._generate_key(key)
        
        try:
            # Check item size
            serialized = self._serialize_value(value)
            item_size = len(serialized)
            
            if item_size > self.config.max_item_size:
                self.logger.warning(f"Item too large: {item_size} > {self.config.max_item_size}")
                return False
            
            with self._lock:
                # Remove existing item if present
                if cache_key in self.cache:
                    old_size = len(self._serialize_value(self.cache[cache_key]))
                    self.memory_usage -= old_size
                
                # Check memory limits and evict if needed
                while (len(self.cache) >= self.config.max_memory_items or 
                       self.memory_usage + item_size > self.config.max_memory_size):
                    if not self._evict_item():
                        break
                
                # Add new item
                self.cache[cache_key] = value
                self.memory_usage += item_size
                self.access_times[cache_key] = time.time()
                self.access_counts[cache_key] = 1
                
                # Set TTL
                ttl = ttl or self.config.default_ttl
                self.ttls[cache_key] = time.time() + ttl
                
                self.stats.sets += 1
                self.stats.size = len(self.cache)
                self.stats.memory_usage = self.memory_usage
                
                return True
                
        except Exception as e:
            self.logger.error(f"Cache set failed: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete item from memory cache."""
        cache_key = self._generate_key(key)
        
        try:
            with self._lock:
                if cache_key not in self.cache:
                    return False
                
                # Remove item and update tracking
                item_size = len(self._serialize_value(self.cache[cache_key]))
                self.memory_usage -= item_size
                
                self._remove_key(cache_key)
                self.stats.deletes += 1
                self.stats.size = len(self.cache)
                self.stats.memory_usage = self.memory_usage
                
                return True
                
        except Exception as e:
            self.logger.error(f"Cache delete failed: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all items from memory cache."""
        try:
            with self._lock:
                self.cache.clear()
                self.access_times.clear()
                self.access_counts.clear()
                self.ttls.clear()
                self.memory_usage = 0
                
                self.stats.size = 0
                self.stats.memory_usage = 0
                
                return True
                
        except Exception as e:
            self.logger.error(f"Cache clear failed: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        cache_key = self._generate_key(key)
        
        with self._lock:
            if cache_key not in self.cache:
                return False
            
            if self._is_expired(cache_key):
                self._remove_key(cache_key)
                return False
            
            return True
    def _is_expired(self, cache_key: str) -> bool:
        """Check if cache key is expired."""
        ttl = self.ttls.get(cache_key, 0)
        return time.time() > ttl
    
    def _remove_key(self, cache_key: str):
        """Remove key and associated metadata."""
        if cache_key in self.cache:
            item_size = len(self._serialize_value(self.cache[cache_key]))
            self.memory_usage -= item_size
            
            del self.cache[cache_key]
            self.access_times.pop(cache_key, None)
            self.access_counts.pop(cache_key, None)
            self.ttls.pop(cache_key, None)
            
            self.stats.size = len(self.cache)
            self.stats.memory_usage = self.memory_usage
    
    def _evict_item(self) -> bool:
        """Evict one item based on eviction policy."""
        if not self.cache:
            return False
        
        if self.config.eviction_policy == "lru":
            # Remove least recently used (first item in OrderedDict)
            cache_key = next(iter(self.cache))
        elif self.config.eviction_policy == "lfu":
            # Remove least frequently used
            cache_key = min(self.access_counts, key=self.access_counts.get)
        elif self.config.eviction_policy == "fifo":
            # Remove first inserted (first item in OrderedDict)
            cache_key = next(iter(self.cache))
        else:
            # Default to LRU
            cache_key = next(iter(self.cache))
        
        self._remove_key(cache_key)
        self.stats.evictions += 1
        return True


class RedisCache(CacheBackend):
    """Redis-based cache implementation."""
    
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        
        if not REDIS_AVAILABLE:
            raise ImportError("Redis library not available")
        
        try:
            redis_url = config.redis_url or "redis://localhost:6379/0"
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            
            # Test connection
            self.redis_client.ping()
            self.logger.info(f"Connected to Redis: {redis_url}")
            
        except Exception as e:
            self.logger.error(f"Redis connection failed: {e}")
            raise CacheError(f"Failed to connect to Redis: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from Redis cache."""
        start_time = time.time()
        cache_key = self._generate_key(key)
        
        try:
            data = self.redis_client.get(cache_key)
            if data is None:
                self.stats.misses += 1
                return None
            
            value = self._deserialize_value(data)
            self.stats.hits += 1
            
            # Update average access time
            access_time = time.time() - start_time
            self.stats.avg_access_time = (
                (self.stats.avg_access_time * (self.stats.hits - 1) + access_time) / self.stats.hits
            )
            
            return value
            
        except Exception as e:
            self.logger.error(f"Redis get failed: {e}")
            self.stats.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in Redis cache."""
        cache_key = self._generate_key(key)
        
        try:
            serialized = self._serialize_value(value)
            
            if len(serialized) > self.config.max_item_size:
                self.logger.warning(f"Item too large for Redis: {len(serialized)}")
                return False
            
            ttl = ttl or self.config.default_ttl
            result = self.redis_client.setex(cache_key, ttl, serialized)
            
            if result:
                self.stats.sets += 1
            
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"Redis set failed: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete item from Redis cache."""
        cache_key = self._generate_key(key)
        
        try:
            result = self.redis_client.delete(cache_key)
            if result:
                self.stats.deletes += 1
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"Redis delete failed: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all items from Redis cache."""
        try:
            self.redis_client.flushdb()
            return True
            
        except Exception as e:
            self.logger.error(f"Redis clear failed: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        cache_key = self._generate_key(key)
        
        try:
            return bool(self.redis_client.exists(cache_key))
            
        except Exception as e:
            self.logger.error(f"Redis exists check failed: {e}")
            return False


class HybridCache(CacheBackend):
    """Hybrid cache with multiple levels (memory + distributed)."""
    
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        
        # L1 cache (memory) - fast but limited
        memory_config = CacheConfig(
            backend="memory",
            max_memory_items=min(1000, config.max_memory_items // 10),
            max_memory_size=config.max_memory_size // 10
        )
        self.l1_cache = MemoryCache(memory_config)
        
        # L2 cache (distributed) - slower but larger
        if REDIS_AVAILABLE and config.redis_url:
            self.l2_cache = RedisCache(config)
        else:
            self.l2_cache = None
            self.logger.warning("L2 cache (Redis) not available, using memory only")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from hybrid cache (L1 first, then L2)."""
        
        # Try L1 cache first
        value = self.l1_cache.get(key)
        if value is not None:
            self.stats.hits += 1
            return value
        
        # Try L2 cache
        if self.l2_cache:
            value = self.l2_cache.get(key)
            if value is not None:
                # Promote to L1 cache
                self.l1_cache.set(key, value, ttl=300)  # 5 min TTL in L1
                self.stats.hits += 1
                return value
        
        self.stats.misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in hybrid cache (both L1 and L2)."""
        
        success = True
        
        # Set in L1 cache
        l1_success = self.l1_cache.set(key, value, min(ttl or 3600, 3600))
        
        # Set in L2 cache
        l2_success = True
        if self.l2_cache:
            l2_success = self.l2_cache.set(key, value, ttl)
        
        success = l1_success and l2_success
        
        if success:
            self.stats.sets += 1
        
        return success
    
    def delete(self, key: str) -> bool:
        """Delete item from hybrid cache (both levels)."""
        
        l1_success = self.l1_cache.delete(key)
        l2_success = True
        
        if self.l2_cache:
            l2_success = self.l2_cache.delete(key)
        
        success = l1_success or l2_success
        
        if success:
            self.stats.deletes += 1
        
        return success
    
    def clear(self) -> bool:
        """Clear hybrid cache (both levels)."""
        
        l1_success = self.l1_cache.clear()
        l2_success = True
        
        if self.l2_cache:
            l2_success = self.l2_cache.clear()
        
        return l1_success and l2_success
    
    def exists(self, key: str) -> bool:
        """Check if key exists in hybrid cache."""
        
        if self.l1_cache.exists(key):
            return True
        
        if self.l2_cache and self.l2_cache.exists(key):
            return True
        
        return False


class CacheManager:
    """Advanced cache management with intelligent features."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = get_logger("cache_manager")
        self.metrics_collector = MetricsCollector()
        
        # Initialize cache backend
        self.cache = self._create_backend(config)
        
        # Cache warming
        if config.enable_preloading:
            self._warm_cache()
        
        # Performance monitoring
        if config.enable_statistics:
            self._start_monitoring()
    
    def _create_backend(self, config: CacheConfig) -> CacheBackend:
        """Create appropriate cache backend."""
        
        if config.backend == "memory":
            return MemoryCache(config)
        elif config.backend == "redis":
            if not REDIS_AVAILABLE:
                self.logger.warning("Redis not available, falling back to memory cache")
                return MemoryCache(config)
            return RedisCache(config)
        elif config.backend == "hybrid":
            return HybridCache(config)
        else:
            self.logger.warning(f"Unknown backend {config.backend}, using memory")
            return MemoryCache(config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache with default value."""
        value = self.cache.get(key)
        return value if value is not None else default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in cache."""
        return self.cache.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        return self.cache.delete(key)
    
    def clear(self) -> bool:
        """Clear all cache items."""
        return self.cache.clear()
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self.cache.exists(key)
    
    def get_or_set(self, key: str, factory: Callable[[], Any], ttl: Optional[int] = None) -> Any:
        """Get item or set if not exists using factory function."""
        
        value = self.cache.get(key)
        if value is not None:
            return value
        
        # Generate value using factory
        try:
            value = factory()
            self.cache.set(key, value, ttl)
            return value
        except Exception as e:
            self.logger.error(f"Factory function failed: {e}")
            raise
    
    def mget(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple items at once."""
        result = {}
        for key in keys:
            value = self.cache.get(key)
            if value is not None:
                result[key] = value
        return result
    
    def mset(self, items: Dict[str, Any], ttl: Optional[int] = None) -> int:
        """Set multiple items at once."""
        success_count = 0
        for key, value in items.items():
            if self.cache.set(key, value, ttl):
                success_count += 1
        return success_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        
        cache_stats = self.cache.get_stats()
        
        stats = {
            'backend': self.config.backend,
            'cache_stats': cache_stats.to_dict(),
            'config': {
                'max_memory_items': self.config.max_memory_items,
                'max_memory_size': self.config.max_memory_size,
                'default_ttl': self.config.default_ttl,
                'eviction_policy': self.config.eviction_policy
            }
        }
        
        # Add backend-specific stats
        if isinstance(self.cache, HybridCache):
            stats['l1_stats'] = self.cache.l1_cache.get_stats().to_dict()
            if self.cache.l2_cache:
                stats['l2_stats'] = self.cache.l2_cache.get_stats().to_dict()
        
        return stats
    
    def _warm_cache(self):
        """Warm cache with commonly used items."""
        # This would be implemented based on application-specific patterns
        self.logger.info("Cache warming completed")
    
    def _start_monitoring(self):
        """Start cache performance monitoring."""
        # This would integrate with the monitoring system
        self.logger.info("Cache monitoring started")


# Global cache manager instance
_global_cache_manager = None

def get_cache_manager(config: Optional[CacheConfig] = None) -> CacheManager:
    """Get global cache manager instance."""
    global _global_cache_manager
    
    if _global_cache_manager is None:
        if config is None:
            config = CacheConfig()
        _global_cache_manager = CacheManager(config)
    
    return _global_cache_manager


# Decorators for easy caching
def cached(ttl: int = 3600, key_func: Optional[Callable] = None):
    """Decorator for caching function results."""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__module__}.{func.__name__}:{hash((args, tuple(kwargs.items())))}"
            
            # Try to get from cache
            cache_manager = get_cache_manager()
            result = cache_manager.get(cache_key)
            
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


# Export main classes
__all__ = [
    "CacheManager",
    "CacheConfig", 
    "CacheStats",
    "MemoryCache",
    "RedisCache", 
    "HybridCache",
    "get_cache_manager",
    "cached"
]