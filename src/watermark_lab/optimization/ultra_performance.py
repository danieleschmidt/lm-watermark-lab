"""Ultra-high performance optimizations for massive scale deployment."""

import asyncio
import multiprocessing as mp
import threading
import time
import math
import hashlib
import weakref
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

from ..utils.logging import get_logger
from ..utils.exceptions import PerformanceError

logger = get_logger(__name__)


@dataclass
class UltraPerformanceConfig:
    """Configuration for ultra-high performance mode."""
    
    enable_multiprocessing: bool = True
    max_processes: int = mp.cpu_count()
    max_threads_per_process: int = 4
    enable_distributed_cache: bool = False
    redis_url: str = "redis://localhost:6379"
    enable_memory_mapping: bool = True
    enable_vectorization: bool = HAS_NUMPY
    batch_size_multiplier: float = 2.0
    precompute_cache_size: int = 10000
    enable_predictive_caching: bool = True


class UltraPerformanceManager:
    """Ultra-high performance manager for extreme scalability."""
    
    def __init__(self, config: UltraPerformanceConfig = None):
        self.config = config or UltraPerformanceConfig()
        self.process_pool = None
        self.thread_pool = None
        self.redis_client = None
        self.precomputed_cache = {}
        self.memory_pool = weakref.WeakValueDictionary()
        self.performance_metrics = {
            'operations_per_second': deque(maxlen=1000),
            'cache_hit_rate': deque(maxlen=1000),
            'memory_efficiency': deque(maxlen=1000),
            'processing_time': deque(maxlen=1000)
        }
        
        self._initialize_resources()
        
    def _initialize_resources(self):
        """Initialize high-performance resources."""
        if self.config.enable_multiprocessing:
            self.process_pool = ProcessPoolExecutor(
                max_workers=self.config.max_processes
            )
            
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.max_threads_per_process
        )
        
        if self.config.enable_distributed_cache and HAS_REDIS:
            try:
                self.redis_client = redis.from_url(self.config.redis_url)
                self.redis_client.ping()
                logger.info("Redis distributed cache enabled")
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}")
                self.redis_client = None
    
    def ultra_batch_process(self, 
                           func: Callable,
                           data_items: List[Any],
                           chunk_size: Optional[int] = None) -> List[Any]:
        """Ultra-high performance batch processing."""
        start_time = time.time()
        
        if not data_items:
            return []
        
        # Adaptive chunk sizing
        if chunk_size is None:
            chunk_size = max(1, int(len(data_items) / self.config.max_processes))
            chunk_size = int(chunk_size * self.config.batch_size_multiplier)
        
        # Split data into chunks
        chunks = [
            data_items[i:i + chunk_size]
            for i in range(0, len(data_items), chunk_size)
        ]
        
        results = []
        
        if self.config.enable_multiprocessing and self.process_pool:
            # Multiprocessing approach
            futures = []
            for chunk in chunks:
                future = self.process_pool.submit(self._process_chunk, func, chunk)
                futures.append(future)
            
            for future in futures:
                try:
                    chunk_results = future.result(timeout=300)  # 5 minute timeout
                    results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
        else:
            # Threading approach
            futures = []
            for chunk in chunks:
                future = self.thread_pool.submit(self._process_chunk, func, chunk)
                futures.append(future)
            
            for future in futures:
                try:
                    chunk_results = future.result(timeout=180)  # 3 minute timeout
                    results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Thread processing failed: {e}")
        
        processing_time = time.time() - start_time
        self.performance_metrics['processing_time'].append(processing_time)
        
        # Calculate operations per second
        ops_per_second = len(data_items) / processing_time if processing_time > 0 else 0
        self.performance_metrics['operations_per_second'].append(ops_per_second)
        
        return results
    
    def _process_chunk(self, func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a single chunk of data."""
        results = []
        for item in chunk:
            try:
                result = func(item)
                results.append(result)
            except Exception as e:
                logger.warning(f"Item processing failed: {e}")
                results.append(None)
        return results
    
    def ultra_cache_get(self, key: str, compute_func: Callable = None) -> Any:
        """Ultra-fast caching with multiple layers."""
        # Level 1: Memory cache
        if key in self.precomputed_cache:
            self.performance_metrics['cache_hit_rate'].append(1.0)
            return self.precomputed_cache[key]
        
        # Level 2: Redis distributed cache
        if self.redis_client:
            try:
                cached_value = self.redis_client.get(key)
                if cached_value:
                    value = pickle.loads(cached_value)
                    # Store in local cache for faster access
                    self.precomputed_cache[key] = value
                    self.performance_metrics['cache_hit_rate'].append(0.8)
                    return value
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")
        
        # Level 3: Compute if function provided
        if compute_func:
            value = compute_func()
            self.ultra_cache_set(key, value)
            self.performance_metrics['cache_hit_rate'].append(0.0)
            return value
        
        self.performance_metrics['cache_hit_rate'].append(0.0)
        return None
    
    def ultra_cache_set(self, key: str, value: Any, ttl: int = 3600):
        """Ultra-fast cache storage with TTL."""
        # Store in local memory cache
        if len(self.precomputed_cache) < self.config.precompute_cache_size:
            self.precomputed_cache[key] = value
        
        # Store in Redis if available
        if self.redis_client:
            try:
                serialized = pickle.dumps(value)
                self.redis_client.setex(key, ttl, serialized)
            except Exception as e:
                logger.warning(f"Redis cache write failed: {e}")
    
    def vectorized_batch_operation(self, 
                                 operation: str,
                                 data_arrays: List[Union[List, np.ndarray]]) -> Union[List, np.ndarray]:
        """Vectorized operations for massive performance gains."""
        if not HAS_NUMPY or not self.config.enable_vectorization:
            # Fallback to regular processing
            return self._fallback_batch_operation(operation, data_arrays)
        
        try:
            # Convert to numpy arrays for vectorization
            arrays = [np.array(data) if not isinstance(data, np.ndarray) else data 
                     for data in data_arrays]
            
            start_time = time.time()
            
            if operation == 'cosine_similarity':
                result = self._vectorized_cosine_similarity(arrays)
            elif operation == 'euclidean_distance':
                result = self._vectorized_euclidean_distance(arrays)
            elif operation == 'matrix_multiply':
                result = self._vectorized_matrix_multiply(arrays)
            elif operation == 'statistical_analysis':
                result = self._vectorized_statistics(arrays)
            else:
                raise ValueError(f"Unknown vectorized operation: {operation}")
            
            processing_time = time.time() - start_time
            self.performance_metrics['processing_time'].append(processing_time)
            
            return result
            
        except Exception as e:
            logger.warning(f"Vectorized operation failed: {e}")
            return self._fallback_batch_operation(operation, data_arrays)
    
    def _vectorized_cosine_similarity(self, arrays: List[np.ndarray]) -> np.ndarray:
        """Vectorized cosine similarity calculation."""
        if len(arrays) != 2:
            raise ValueError("Cosine similarity requires exactly 2 arrays")
        
        a, b = arrays
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _vectorized_euclidean_distance(self, arrays: List[np.ndarray]) -> np.ndarray:
        """Vectorized Euclidean distance calculation."""
        if len(arrays) != 2:
            raise ValueError("Euclidean distance requires exactly 2 arrays")
        
        a, b = arrays
        return np.linalg.norm(a - b)
    
    def _vectorized_matrix_multiply(self, arrays: List[np.ndarray]) -> np.ndarray:
        """Vectorized matrix multiplication."""
        result = arrays[0]
        for arr in arrays[1:]:
            result = np.dot(result, arr)
        return result
    
    def _vectorized_statistics(self, arrays: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Vectorized statistical analysis."""
        stats = {}
        for i, arr in enumerate(arrays):
            stats[f'array_{i}'] = {
                'mean': np.mean(arr),
                'std': np.std(arr),
                'min': np.min(arr),
                'max': np.max(arr),
                'median': np.median(arr)
            }
        return stats
    
    def _fallback_batch_operation(self, operation: str, data_arrays: List) -> List:
        """Fallback batch operations when vectorization unavailable."""
        # Simple fallback implementations
        results = []
        for data in data_arrays:
            if operation == 'mean':
                results.append(sum(data) / len(data) if data else 0)
            elif operation == 'sum':
                results.append(sum(data))
            else:
                results.append(data)  # Pass-through
        return results
    
    async def async_ultra_process(self, 
                                async_func: Callable,
                                data_items: List[Any],
                                max_concurrent: int = 100) -> List[Any]:
        """Ultra-high performance async processing."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(item):
            async with semaphore:
                return await async_func(item)
        
        tasks = [process_with_semaphore(item) for item in data_items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]
        error_count = len(results) - len(valid_results)
        
        if error_count > 0:
            logger.warning(f"Async processing had {error_count} errors")
        
        return valid_results
    
    def memory_efficient_generator(self, 
                                 data_source: Union[List, Callable],
                                 batch_size: int = 1000):
        """Memory-efficient data processing generator."""
        if isinstance(data_source, list):
            for i in range(0, len(data_source), batch_size):
                yield data_source[i:i + batch_size]
        elif callable(data_source):
            while True:
                batch = data_source(batch_size)
                if not batch:
                    break
                yield batch
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = {}
        
        for metric_name, values in self.performance_metrics.items():
            if values:
                metrics[metric_name] = {
                    'current': values[-1],
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
            else:
                metrics[metric_name] = {
                    'current': 0.0,
                    'average': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'count': 0
                }
        
        return metrics
    
    def optimize_for_workload(self, workload_type: str):
        """Dynamically optimize configuration for specific workload types."""
        if workload_type == 'compute_intensive':
            self.config.max_processes = mp.cpu_count()
            self.config.batch_size_multiplier = 1.5
        elif workload_type == 'io_intensive':
            self.config.max_threads_per_process = 8
            self.config.batch_size_multiplier = 3.0
        elif workload_type == 'memory_intensive':
            self.config.precompute_cache_size = 5000
            self.config.enable_memory_mapping = True
        elif workload_type == 'network_intensive':
            self.config.enable_distributed_cache = True
            self.config.max_threads_per_process = 6
        
        logger.info(f"Optimized configuration for {workload_type} workload")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception:
                pass


# Global ultra-performance manager
_ultra_manager = None

def get_ultra_performance_manager() -> UltraPerformanceManager:
    """Get global ultra-performance manager."""
    global _ultra_manager
    if _ultra_manager is None:
        _ultra_manager = UltraPerformanceManager()
    return _ultra_manager


def ultra_performance_decorator(workload_type: str = 'balanced'):
    """Decorator for ultra-high performance function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_ultra_performance_manager()
            manager.optimize_for_workload(workload_type)
            
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            manager.performance_metrics['processing_time'].append(execution_time)
            
            return result
        return wrapper
    return decorator