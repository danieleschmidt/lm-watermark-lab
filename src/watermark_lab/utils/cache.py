"""Caching utilities for performance optimization."""

import time
import hashlib
import pickle
import threading
from typing import Any, Dict, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

from .logging import get_logger
from .exceptions import WatermarkLabError


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    value: Any
    timestamp: float
    ttl: Optional[float] = None
    access_count: int = 0
    last_access: float = 0.0
    
    def __post_init__(self):
        if self.last_access == 0.0:
            self.last_access = self.timestamp
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def access(self) -> Any:
        """Access cache entry and update metadata."""
        self.access_count += 1
        self.last_access = time.time()
        return self.value


class MemoryCache:
    """Thread-safe in-memory cache with TTL and LRU eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()
        self.logger = get_logger("memory_cache")
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            
            if entry.is_expired():
                del self.cache[key]
                self.misses += 1
                return None
            
            self.hits += 1
            return entry.access()
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        with self.lock:
            if ttl is None:
                ttl = self.default_ttl
            
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl
            )
            
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = entry
            self.logger.debug(f"Cached key: {key[:50]}...")
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.logger.info("Cache cleared")
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.cache:
            return
        
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].last_access
        )
        
        del self.cache[lru_key]
        self.evictions += 1
        self.logger.debug(f"Evicted LRU key: {lru_key[:50]}...")
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count."""
        with self.lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "total_requests": total_requests
            }


class FileCache:
    """Persistent file-based cache."""
    
    def __init__(self, cache_dir: Union[str, Path], max_size: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("file_cache")
        
        # Index file to track cache entries
        self.index_file = self.cache_dir / "_cache_index.pkl"
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from file."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache index: {e}")
        return {}
    
    def _save_index(self) -> None:
        """Save cache index to file."""
        try:
            with open(self.index_file, 'wb') as f:
                pickle.dump(self.index, f)
        except Exception as e:
            self.logger.error(f"Failed to save cache index: {e}")
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from file cache."""
        if key not in self.index:
            return None
        
        entry_info = self.index[key]
        
        # Check TTL
        if entry_info.get('ttl') and time.time() - entry_info['timestamp'] > entry_info['ttl']:
            self.delete(key)
            return None
        
        file_path = self._get_file_path(key)
        
        if not file_path.exists():
            # Clean up stale index entry
            del self.index[key]
            self._save_index()
            return None
        
        try:
            with open(file_path, 'rb') as f:
                value = pickle.load(f)
            
            # Update access info
            self.index[key]['last_access'] = time.time()
            self.index[key]['access_count'] = self.index[key].get('access_count', 0) + 1
            self._save_index()
            
            return value
            
        except Exception as e:
            self.logger.error(f"Failed to load cached value for key {key}: {e}")
            self.delete(key)
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in file cache."""
        # Evict if at capacity
        if len(self.index) >= self.max_size and key not in self.index:
            self._evict_lru()
        
        file_path = self._get_file_path(key)
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
            
            self.index[key] = {
                'timestamp': time.time(),
                'ttl': ttl,
                'last_access': time.time(),
                'access_count': 0,
                'file_path': str(file_path)
            }
            
            self._save_index()
            self.logger.debug(f"Cached key to file: {key[:50]}...")
            
        except Exception as e:
            self.logger.error(f"Failed to cache value for key {key}: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete value from file cache."""
        if key not in self.index:
            return False
        
        file_path = self._get_file_path(key)
        
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            self.logger.warning(f"Failed to delete cache file: {e}")
        
        del self.index[key]
        self._save_index()
        return True
    
    def clear(self) -> None:
        """Clear all cache entries."""
        for key in list(self.index.keys()):
            self.delete(key)
        
        self.index.clear()
        self._save_index()
        self.logger.info("File cache cleared")
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.index:
            return
        
        lru_key = min(
            self.index.keys(),
            key=lambda k: self.index[k].get('last_access', 0)
        )
        
        self.delete(lru_key)
        self.logger.debug(f"Evicted LRU key from file cache: {lru_key[:50]}...")


class CacheManager:
    """Unified cache manager with multiple cache levels."""
    
    def __init__(
        self,
        memory_cache_size: int = 1000,
        file_cache_size: int = 10000,
        cache_dir: Optional[Union[str, Path]] = None,
        default_ttl: Optional[float] = 3600
    ):
        self.memory_cache = MemoryCache(memory_cache_size, default_ttl)
        
        if cache_dir:
            self.file_cache = FileCache(cache_dir, file_cache_size)
        else:
            self.file_cache = None
        
        self.logger = get_logger("cache_manager")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory first, then file)."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try file cache if available
        if self.file_cache:
            value = self.file_cache.get(key)
            if value is not None:
                # Promote to memory cache
                self.memory_cache.set(key, value)
                return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None, persist: bool = True) -> None:
        """Set value in cache."""
        # Always set in memory cache
        self.memory_cache.set(key, value, ttl)
        
        # Set in file cache if persistence is enabled
        if persist and self.file_cache:
            self.file_cache.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete value from all cache levels."""
        memory_deleted = self.memory_cache.delete(key)
        file_deleted = self.file_cache.delete(key) if self.file_cache else False
        
        return memory_deleted or file_deleted
    
    def clear(self) -> None:
        """Clear all cache levels."""
        self.memory_cache.clear()
        if self.file_cache:
            self.file_cache.clear()
    
    def cleanup(self) -> Dict[str, int]:
        """Cleanup expired entries from all cache levels."""
        memory_cleaned = self.memory_cache.cleanup_expired()
        file_cleaned = 0  # File cache cleanup would be more complex
        
        return {
            "memory_cleaned": memory_cleaned,
            "file_cleaned": file_cleaned
        }


# Global cache instance
_global_cache = None


def get_cache() -> CacheManager:
    """Get global cache instance."""
    global _global_cache
    
    if _global_cache is None:
        cache_dir = Path.home() / ".watermark_lab" / "cache"
        _global_cache = CacheManager(cache_dir=cache_dir)
    
    return _global_cache


def cached(
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None,
    persist: bool = True,
    cache_instance: Optional[CacheManager] = None
):
    """Decorator for caching function results."""
    
    def decorator(func: Callable) -> Callable:
        cache = cache_instance or get_cache()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = _generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl, persist=persist)
            
            return result
        
        # Add cache management methods
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_delete = lambda *args, **kwargs: cache.delete(
            key_func(*args, **kwargs) if key_func else _generate_cache_key(func.__name__, args, kwargs)
        )
        
        return wrapper
    
    return decorator


def _generate_cache_key(func_name: str, args: Tuple, kwargs: Dict[str, Any]) -> str:
    """Generate cache key from function name and arguments."""
    # Create a deterministic string representation
    key_parts = [func_name]
    
    # Add positional arguments
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        else:
            # For complex objects, use their hash
            key_parts.append(str(hash(str(arg))))
    
    # Add keyword arguments
    for key, value in sorted(kwargs.items()):
        if isinstance(value, (str, int, float, bool)):
            key_parts.append(f"{key}={value}")
        else:
            key_parts.append(f"{key}={hash(str(value))}")
    
    cache_key = "|".join(key_parts)
    
    # Hash if too long
    if len(cache_key) > 200:
        cache_key = hashlib.md5(cache_key.encode()).hexdigest()
    
    return cache_key


# Specific cache decorators for watermarking components
def cache_watermark_generation(ttl: float = 1800):  # 30 minutes
    """Cache watermark generation results."""
    return cached(ttl=ttl, persist=True)


def cache_detection_results(ttl: float = 3600):  # 1 hour
    """Cache detection results."""
    return cached(ttl=ttl, persist=True)


def cache_model_outputs(ttl: float = 7200):  # 2 hours
    """Cache model outputs."""
    return cached(ttl=ttl, persist=True)