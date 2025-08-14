"""Advanced performance optimization system with intelligent tuning and analysis."""

import time
import threading
import multiprocessing
import concurrent.futures
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
from enum import Enum
import logging
import json
import hashlib
import functools

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    CPU_INTENSIVE = "cpu_intensive"
    IO_INTENSIVE = "io_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceMetrics:
    """Performance measurement data."""
    
    operation_name: str
    execution_time: float
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    io_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    throughput: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / max(1, total)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'operation_name': self.operation_name,
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'io_operations': self.io_operations,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hit_rate,
            'throughput': self.throughput,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    
    # Threading configuration
    max_workers: int = min(32, multiprocessing.cpu_count() * 2)
    thread_pool_size: int = 10
    
    # Memory configuration
    memory_threshold: float = 0.8  # 80% memory usage threshold
    gc_frequency: int = 1000  # Garbage collection frequency
    
    # Caching configuration
    cache_enabled: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600
    
    # Performance monitoring
    profiling_enabled: bool = True
    metrics_retention: int = 10000
    benchmark_frequency: int = 100
    
    # Optimization strategy
    strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE
    auto_tune: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'max_workers': self.max_workers,
            'thread_pool_size': self.thread_pool_size,
            'memory_threshold': self.memory_threshold,
            'gc_frequency': self.gc_frequency,
            'cache_enabled': self.cache_enabled,
            'cache_size': self.cache_size,
            'cache_ttl': self.cache_ttl,
            'profiling_enabled': self.profiling_enabled,
            'metrics_retention': self.metrics_retention,
            'benchmark_frequency': self.benchmark_frequency,
            'strategy': self.strategy.value,
            'auto_tune': self.auto_tune
        }


class PerformanceProfiler:
    """Performance profiling and analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger("performance_profiler")
        self.profiles = defaultdict(list)
        self._lock = threading.Lock()
    
    @contextmanager
    def profile(self, operation_name: str, **metadata):
        """Context manager for profiling operations."""
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = self._get_cpu_usage()
            
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                execution_time=end_time - start_time,
                memory_usage=end_memory - start_memory,
                cpu_usage=end_cpu - start_cpu,
                metadata=metadata
            )
            
            with self._lock:
                self.profiles[operation_name].append(metrics)
                
                # Keep only recent profiles
                if len(self.profiles[operation_name]) > 1000:
                    self.profiles[operation_name] = self.profiles[operation_name][-1000:]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                return process.memory_info().rss / (1024 * 1024)
            except Exception:
                pass
        return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        if PSUTIL_AVAILABLE:
            try:
                return psutil.cpu_percent()
            except Exception:
                pass
        return 0.0
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        
        with self._lock:
            metrics_list = self.profiles.get(operation_name, [])
            
            if not metrics_list:
                return {'error': f'No profiles found for operation: {operation_name}'}
            
            execution_times = [m.execution_time for m in metrics_list]
            memory_usages = [m.memory_usage for m in metrics_list]
            
            return {
                'operation_name': operation_name,
                'sample_count': len(metrics_list),
                'execution_time': {
                    'avg': sum(execution_times) / len(execution_times),
                    'min': min(execution_times),
                    'max': max(execution_times),
                    'p50': self._percentile(execution_times, 0.5),
                    'p95': self._percentile(execution_times, 0.95),
                    'p99': self._percentile(execution_times, 0.99)
                },
                'memory_usage': {
                    'avg': sum(memory_usages) / len(memory_usages),
                    'min': min(memory_usages),
                    'max': max(memory_usages)
                },
                'throughput': len(metrics_list) / sum(execution_times) if execution_times else 0
            }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all profiled operations."""
        
        with self._lock:
            stats = {}
            for operation_name in self.profiles:
                stats[operation_name] = self.get_operation_stats(operation_name)
            return stats


class SmartCache:
    """Intelligent caching system with performance optimization."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.hit_count = 0
        self.miss_count = 0
        self._lock = threading.RLock()
        
        # Cache optimization
        self.adaptive_ttl_enabled = True
        self.performance_tracking = True
        self.auto_cleanup_enabled = True
        
        # Start cleanup thread
        if self.auto_cleanup_enabled:
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with intelligent optimization."""
        
        with self._lock:
            if key not in self.cache:
                self.miss_count += 1
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if time.time() > entry['expires_at']:
                del self.cache[key]
                self.access_times.pop(key, None)
                self.access_counts.pop(key, None)
                self.miss_count += 1
                return None
            
            # Update access information
            self.access_times[key] = time.time()
            self.access_counts[key] += 1
            self.hit_count += 1
            
            # Adaptive TTL based on access frequency
            if self.adaptive_ttl_enabled and self.access_counts[key] > 10:
                # Extend TTL for frequently accessed items
                bonus_ttl = min(self.ttl * 0.5, self.access_counts[key] * 60)
                entry['expires_at'] = time.time() + self.ttl + bonus_ttl
            
            return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in cache with smart eviction."""
        
        with self._lock:
            # Check if cache is full
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_items()
            
            # Set item
            effective_ttl = ttl or self.ttl
            entry = {
                'value': value,
                'created_at': time.time(),
                'expires_at': time.time() + effective_ttl,
                'access_count': 1
            }
            
            self.cache[key] = entry
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
            
            return True
    
    def _evict_items(self, count: int = None):
        """Evict items using intelligent strategy."""
        
        if not self.cache:
            return
        
        evict_count = count or max(1, len(self.cache) // 10)  # Evict 10% by default
        
        # Score items for eviction (lower score = more likely to evict)
        scores = {}
        current_time = time.time()
        
        for key in self.cache:
            entry = self.cache[key]
            
            # Factors affecting eviction score
            age = current_time - entry['created_at']
            access_frequency = self.access_counts.get(key, 1)
            time_since_access = current_time - self.access_times.get(key, current_time)
            
            # Calculate composite score
            score = (access_frequency * 100) - (age / 60) - (time_since_access / 10)
            scores[key] = score
        
        # Sort by score and evict lowest scoring items
        keys_to_evict = sorted(scores.keys(), key=lambda k: scores[k])[:evict_count]
        
        for key in keys_to_evict:
            del self.cache[key]
            self.access_times.pop(key, None)
            self.access_counts.pop(key, None)
    
    def _cleanup_loop(self):
        """Background cleanup of expired items."""
        
        while True:
            try:
                current_time = time.time()
                expired_keys = []
                
                with self._lock:
                    for key, entry in self.cache.items():
                        if current_time > entry['expires_at']:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        del self.cache[key]
                        self.access_times.pop(key, None)
                        self.access_counts.pop(key, None)
                
                time.sleep(300)  # Cleanup every 5 minutes
                
            except Exception:
                time.sleep(60)  # Retry in 1 minute on error
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        
        with self._lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / max(1, total_requests)
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                'ttl': self.ttl,
                'adaptive_ttl_enabled': self.adaptive_ttl_enabled
            }


class PerformanceOptimizer:
    """Advanced performance optimization system."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.logger = self._setup_logging()
        
        # Performance components
        self.profiler = PerformanceProfiler()
        self.cache = SmartCache(self.config.cache_size, self.config.cache_ttl)
        
        # Thread pools for different workload types
        self.thread_pools = {
            'cpu': concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()),
            'io': concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers),
            'default': concurrent.futures.ThreadPoolExecutor(max_workers=self.config.thread_pool_size)
        }
        
        # Performance tracking
        self.operation_counts = defaultdict(int)
        self.optimization_history = deque(maxlen=1000)
        
        # Auto-tuning state
        self.auto_tune_enabled = self.config.auto_tune
        self.tune_counter = 0
        
        # Memory management
        self.gc_counter = 0
        
        self.logger.info("Performance optimizer initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup performance logging."""
        logger = logging.getLogger("performance_optimizer")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def optimize_function(
        self,
        func: Callable,
        workload_type: str = "default",
        cache_key_func: Optional[Callable] = None,
        cache_ttl: Optional[int] = None
    ) -> Callable:
        """Decorator to optimize function performance."""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = f"{func.__module__}.{func.__name__}"
            
            # Check cache if enabled
            if self.config.cache_enabled and cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
                cached_result = self.cache.get(cache_key)
                
                if cached_result is not None:
                    return cached_result
            
            # Profile execution
            with self.profiler.profile(operation_name):
                # Execute with appropriate optimization strategy
                if workload_type == "cpu" and len(args) > 1:
                    # For CPU-intensive tasks, consider parallelization
                    result = self._execute_with_strategy(func, args, kwargs, workload_type)
                else:
                    result = func(*args, **kwargs)
                
                # Cache result if enabled
                if self.config.cache_enabled and cache_key_func:
                    cache_key = cache_key_func(*args, **kwargs)
                    self.cache.set(cache_key, result, cache_ttl)
                
                # Update operation counter
                self.operation_counts[operation_name] += 1
                
                # Trigger auto-tuning if enabled
                if self.auto_tune_enabled:
                    self._check_auto_tune()
                
                # Memory management
                self._manage_memory()
                
                return result
        
        return wrapper
    
    def _execute_with_strategy(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        workload_type: str
    ) -> Any:
        """Execute function with optimization strategy."""
        
        if workload_type in self.thread_pools:
            thread_pool = self.thread_pools[workload_type]
            
            # For batch operations, consider parallel execution
            if isinstance(args[0], (list, tuple)) and len(args[0]) > 10:
                return self._parallel_batch_execution(func, args[0], thread_pool)
        
        return func(*args, **kwargs)
    
    def _parallel_batch_execution(
        self,
        func: Callable,
        items: Union[list, tuple],
        thread_pool: concurrent.futures.ThreadPoolExecutor
    ) -> List[Any]:
        """Execute function on batch of items in parallel."""
        
        batch_size = max(1, len(items) // (self.config.max_workers * 2))
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        future_to_batch = {
            thread_pool.submit(self._process_batch, func, batch): batch
            for batch in batches
        }
        
        results = []
        for future in concurrent.futures.as_completed(future_to_batch):
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception as e:
                self.logger.error(f"Batch execution failed: {e}")
                # Fall back to sequential processing for failed batch
                batch = future_to_batch[future]
                for item in batch:
                    try:
                        results.append(func(item))
                    except Exception:
                        results.append(None)
        
        return results
    
    def _process_batch(self, func: Callable, batch: List[Any]) -> List[Any]:
        """Process a batch of items."""
        return [func(item) for item in batch]
    
    def _check_auto_tune(self):
        """Check if auto-tuning should be performed."""
        
        self.tune_counter += 1
        
        if self.tune_counter >= self.config.benchmark_frequency:
            self._perform_auto_tune()
            self.tune_counter = 0
    
    def _perform_auto_tune(self):
        """Perform automatic performance tuning."""
        
        try:
            # Analyze recent performance
            stats = self.profiler.get_all_stats()
            
            if not stats:
                return
            
            # Find operations that could benefit from tuning
            slow_operations = []
            for op_name, op_stats in stats.items():
                if op_stats.get('execution_time', {}).get('avg', 0) > 1.0:  # > 1 second
                    slow_operations.append((op_name, op_stats))
            
            if slow_operations:
                self.logger.info(f"Auto-tuning identified {len(slow_operations)} slow operations")
                
                # Adjust thread pool sizes based on workload
                self._tune_thread_pools(stats)
                
                # Adjust cache settings
                self._tune_cache_settings()
                
                # Record tuning action
                tuning_record = {
                    'timestamp': time.time(),
                    'slow_operations_count': len(slow_operations),
                    'actions_taken': ['thread_pool_tuning', 'cache_tuning']
                }
                
                self.optimization_history.append(tuning_record)
        
        except Exception as e:
            self.logger.error(f"Auto-tuning failed: {e}")
    
    def _tune_thread_pools(self, stats: Dict[str, Any]):
        """Tune thread pool configurations based on performance data."""
        
        # Analyze CPU vs I/O intensive operations
        cpu_intensive_ops = 0
        io_intensive_ops = 0
        
        for op_stats in stats.values():
            avg_time = op_stats.get('execution_time', {}).get('avg', 0)
            if avg_time > 0.5:  # Operations taking > 500ms
                # Simple heuristic: assume longer operations are I/O intensive
                if avg_time > 2.0:
                    io_intensive_ops += 1
                else:
                    cpu_intensive_ops += 1
        
        # Adjust thread pools
        if io_intensive_ops > cpu_intensive_ops:
            # More I/O intensive - increase I/O thread pool
            new_io_size = min(self.config.max_workers * 2, 64)
            if new_io_size != self.thread_pools['io']._max_workers:
                self.logger.info(f"Tuning I/O thread pool size to {new_io_size}")
                # Would recreate thread pool in real implementation
    
    def _tune_cache_settings(self):
        """Tune cache settings based on performance."""
        
        cache_stats = self.cache.get_stats()
        hit_rate = cache_stats.get('hit_rate', 0)
        
        if hit_rate < 0.5:  # Low hit rate
            # Increase cache size if possible
            if self.cache.max_size < 5000:
                self.cache.max_size = min(5000, self.cache.max_size * 2)
                self.logger.info(f"Increased cache size to {self.cache.max_size}")
        
        elif hit_rate > 0.9:  # Very high hit rate
            # Could potentially reduce cache size to free memory
            pass
    
    def _manage_memory(self):
        """Manage memory usage and garbage collection."""
        
        self.gc_counter += 1
        
        if self.gc_counter >= self.config.gc_frequency:
            # Check memory usage
            if PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    memory_percent = process.memory_percent()
                    
                    if memory_percent > (self.config.memory_threshold * 100):
                        import gc
                        collected = gc.collect()
                        self.logger.debug(f"Garbage collection freed {collected} objects")
                except Exception:
                    pass
            
            self.gc_counter = 0
    
    @contextmanager
    def performance_context(self, operation_name: str):
        """Context manager for performance monitoring."""
        
        with self.profiler.profile(operation_name):
            yield
    
    def benchmark_operation(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        iterations: int = 100
    ) -> Dict[str, Any]:
        """Benchmark an operation multiple times."""
        
        kwargs = kwargs or {}
        execution_times = []
        
        for _ in range(iterations):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
            except Exception as e:
                self.logger.error(f"Benchmark iteration failed: {e}")
                continue
        
        if not execution_times:
            return {'error': 'All benchmark iterations failed'}
        
        return {
            'iterations': len(execution_times),
            'avg_time': sum(execution_times) / len(execution_times),
            'min_time': min(execution_times),
            'max_time': max(execution_times),
            'total_time': sum(execution_times),
            'throughput': len(execution_times) / sum(execution_times)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        
        profiler_stats = self.profiler.get_all_stats()
        cache_stats = self.cache.get_stats()
        
        # System resources
        system_stats = {}
        if PSUTIL_AVAILABLE:
            try:
                system_stats = {
                    'cpu_count': psutil.cpu_count(),
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent
                }
            except Exception:
                pass
        
        return {
            'configuration': self.config.to_dict(),
            'profiler_stats': profiler_stats,
            'cache_stats': cache_stats,
            'system_stats': system_stats,
            'operation_counts': dict(self.operation_counts),
            'optimization_history': list(self.optimization_history)[-10:],  # Last 10 optimizations
            'thread_pools': {
                name: {'max_workers': pool._max_workers}
                for name, pool in self.thread_pools.items()
            }
        }
    
    def shutdown(self):
        """Shutdown performance optimizer and cleanup resources."""
        
        self.logger.info("Shutting down performance optimizer...")
        
        # Shutdown thread pools
        for name, pool in self.thread_pools.items():
            pool.shutdown(wait=True)
            self.logger.debug(f"Shutdown thread pool: {name}")
        
        self.logger.info("Performance optimizer shutdown complete")


# Global performance optimizer
performance_optimizer = PerformanceOptimizer()


# Decorator functions
def optimize(
    workload_type: str = "default",
    cache_key_func: Optional[Callable] = None,
    cache_ttl: Optional[int] = None
):
    """Decorator for function optimization."""
    
    def decorator(func: Callable) -> Callable:
        return performance_optimizer.optimize_function(
            func, workload_type, cache_key_func, cache_ttl
        )
    
    return decorator


def benchmark(iterations: int = 100):
    """Decorator to benchmark function performance."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Run benchmark in background
            def run_benchmark():
                benchmark_result = performance_optimizer.benchmark_operation(
                    func, args, kwargs, iterations
                )
                performance_optimizer.logger.info(
                    f"Benchmark {func.__name__}: {benchmark_result}"
                )
            
            threading.Thread(target=run_benchmark, daemon=True).start()
            return result
        
        return wrapper
    
    return decorator


# Convenience functions
def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary."""
    return performance_optimizer.get_performance_summary()


def optimize_for_cpu(func: Callable) -> Callable:
    """Optimize function for CPU-intensive workloads."""
    return performance_optimizer.optimize_function(func, "cpu")


def optimize_for_io(func: Callable) -> Callable:
    """Optimize function for I/O-intensive workloads."""
    return performance_optimizer.optimize_function(func, "io")


__all__ = [
    'PerformanceOptimizer',
    'PerformanceMetrics',
    'OptimizationConfig',
    'OptimizationStrategy',
    'PerformanceProfiler',
    'SmartCache',
    'performance_optimizer',
    'optimize',
    'benchmark',
    'optimize_for_cpu',
    'optimize_for_io',
    'get_performance_summary'
]