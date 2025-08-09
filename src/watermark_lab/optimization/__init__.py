"""Advanced optimization and performance enhancement systems."""

from .caching import CacheManager, CacheConfig, MemoryCache, HybridCache
from .resource_manager import ResourceManager, ResourceLimits, ResourcePool

# Import optional modules with fallbacks
try:
    from .caching import RedisCache
    REDIS_CACHING_AVAILABLE = True
except ImportError:
    REDIS_CACHING_AVAILABLE = False

try:
    from .performance import PerformanceOptimizer, OptimizationConfig
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False

try:
    from .async_processing import AsyncProcessor, ProcessingConfig
    ASYNC_PROCESSING_AVAILABLE = True
except ImportError:
    ASYNC_PROCESSING_AVAILABLE = False

try:
    from .batch_processing import BatchProcessor, BatchConfig
    BATCH_PROCESSING_AVAILABLE = True
except ImportError:
    BATCH_PROCESSING_AVAILABLE = False

# Build __all__ dynamically based on available modules
__all__ = [
    "CacheManager",
    "CacheConfig", 
    "MemoryCache",
    "HybridCache",
    "ResourceManager",
    "ResourceLimits",
    "ResourcePool"
]

if REDIS_CACHING_AVAILABLE:
    __all__.append("RedisCache")

if PERFORMANCE_AVAILABLE:
    __all__.extend(["PerformanceOptimizer", "OptimizationConfig"])

if ASYNC_PROCESSING_AVAILABLE:
    __all__.extend(["AsyncProcessor", "ProcessingConfig"])

if BATCH_PROCESSING_AVAILABLE:
    __all__.extend(["BatchProcessor", "BatchConfig"])