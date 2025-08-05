"""Performance optimization utilities."""

import time
import functools
import threading
import numpy as np
from typing import Callable, Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque

from .logging import get_logger
from .cache import cached


@dataclass
class PerformanceProfile:
    """Performance profiling data."""
    
    function_name: str
    call_count: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    memory_usage: Optional[float] = None
    
    @property
    def time_per_call(self) -> float:
        """Average time per call."""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0


class PerformanceProfiler:
    """Thread-safe performance profiler."""
    
    def __init__(self):
        self.profiles: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "call_count": 0,
            "total_time": 0.0,
            "times": deque(maxlen=1000),  # Keep last 1000 calls
            "memory_samples": deque(maxlen=100)
        })
        self.lock = threading.Lock()
        self.logger = get_logger("performance_profiler")
    
    def profile(self, func: Callable) -> Callable:
        """Decorator to profile function performance."""
        func_name = f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                success = False
                raise
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                
                self._record_execution(func_name, execution_time, success)
            
            return result
        
        return wrapper
    
    def _record_execution(self, func_name: str, execution_time: float, success: bool):
        """Record execution metrics."""
        with self.lock:
            profile = self.profiles[func_name]
            profile["call_count"] += 1
            profile["total_time"] += execution_time
            profile["times"].append(execution_time)
            
            if not success:
                profile["error_count"] = profile.get("error_count", 0) + 1
    
    def get_profile(self, func_name: str) -> Optional[PerformanceProfile]:
        """Get performance profile for function."""
        with self.lock:
            if func_name not in self.profiles:
                return None
            
            profile_data = self.profiles[func_name]
            times = list(profile_data["times"])
            
            if not times:
                return None
            
            return PerformanceProfile(
                function_name=func_name,
                call_count=profile_data["call_count"],
                total_time=profile_data["total_time"],
                avg_time=profile_data["total_time"] / profile_data["call_count"],
                min_time=min(times),
                max_time=max(times)
            )
    
    def get_all_profiles(self) -> List[PerformanceProfile]:
        """Get all performance profiles."""
        profiles = []
        
        with self.lock:
            for func_name in self.profiles:
                profile = self.get_profile(func_name)
                if profile:
                    profiles.append(profile)
        
        return sorted(profiles, key=lambda p: p.total_time, reverse=True)
    
    def reset(self):
        """Reset all profiles."""
        with self.lock:
            self.profiles.clear()
    
    def report(self) -> str:
        """Generate performance report."""
        profiles = self.get_all_profiles()
        
        if not profiles:
            return "No performance data available"
        
        lines = ["Performance Report", "=" * 50]
        
        for profile in profiles:
            lines.append(f"Function: {profile.function_name}")
            lines.append(f"  Calls: {profile.call_count}")
            lines.append(f"  Total Time: {profile.total_time:.3f}s")
            lines.append(f"  Avg Time: {profile.avg_time:.3f}s")
            lines.append(f"  Min Time: {profile.min_time:.3f}s")
            lines.append(f"  Max Time: {profile.max_time:.3f}s")
            lines.append("")
        
        return "\n".join(lines)


class MemoryMonitor:
    """Monitor memory usage of functions."""
    
    def __init__(self):
        self.logger = get_logger("memory_monitor")
    
    def monitor(self, func: Callable) -> Callable:
        """Decorator to monitor memory usage."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                import psutil
                process = psutil.Process()
                start_memory = process.memory_info().rss
            except ImportError:
                start_memory = None
            
            try:
                result = func(*args, **kwargs)
            finally:
                if start_memory is not None:
                    end_memory = process.memory_info().rss
                    memory_delta = end_memory - start_memory
                    
                    if memory_delta > 1024 * 1024:  # > 1MB
                        self.logger.info(
                            f"Function {func.__name__} used {memory_delta / 1024 / 1024:.2f} MB"
                        )
            
            return result
        
        return wrapper


class BatchOptimizer:
    """Optimize batch processing operations."""
    
    def __init__(self):
        self.logger = get_logger("batch_optimizer")
        self.optimal_batch_sizes = {}
    
    def find_optimal_batch_size(
        self,
        func: Callable,
        test_data: List[Any],
        batch_sizes: Optional[List[int]] = None,
        max_test_items: int = 100
    ) -> int:
        """Find optimal batch size for function."""
        
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        
        # Limit test data
        test_items = test_data[:max_test_items]
        
        if not test_items:
            return batch_sizes[0]
        
        results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(test_items):
                continue
            
            try:
                start_time = time.time()
                
                # Process in batches
                for i in range(0, len(test_items), batch_size):
                    batch = test_items[i:i + batch_size]
                    func(batch)
                
                total_time = time.time() - start_time
                throughput = len(test_items) / total_time
                
                results[batch_size] = {
                    "time": total_time,
                    "throughput": throughput
                }
                
                self.logger.debug(
                    f"Batch size {batch_size}: {throughput:.2f} items/sec"
                )
                
            except Exception as e:
                self.logger.warning(f"Batch size {batch_size} failed: {e}")
                continue
        
        if not results:
            return batch_sizes[0]
        
        # Find batch size with highest throughput
        optimal_batch_size = max(results.keys(), key=lambda k: results[k]["throughput"])
        
        func_name = func.__name__
        self.optimal_batch_sizes[func_name] = optimal_batch_size
        
        self.logger.info(
            f"Optimal batch size for {func_name}: {optimal_batch_size} "
            f"({results[optimal_batch_size]['throughput']:.2f} items/sec)"
        )
        
        return optimal_batch_size
    
    def get_optimal_batch_size(self, func_name: str, default: int = 32) -> int:
        """Get previously determined optimal batch size."""
        return self.optimal_batch_sizes.get(func_name, default)


def memoize_with_lru(maxsize: int = 128):
    """LRU cache decorator with configurable size."""
    def decorator(func):
        return functools.lru_cache(maxsize=maxsize)(func)
    return decorator


def lazy_property(func):
    """Decorator for lazy property evaluation."""
    attr_name = f"_lazy_{func.__name__}"
    
    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    
    return wrapper


class VectorizedOperations:
    """Vectorized operations for numerical computations."""
    
    @staticmethod
    def cosine_similarity_batch(vectors1: np.ndarray, vectors2: np.ndarray) -> np.ndarray:
        """Compute cosine similarity for batches of vectors."""
        # Normalize vectors
        norm1 = np.linalg.norm(vectors1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(vectors2, axis=1, keepdims=True)
        
        normalized1 = vectors1 / (norm1 + 1e-8)
        normalized2 = vectors2 / (norm2 + 1e-8)
        
        # Compute dot product
        return np.sum(normalized1 * normalized2, axis=1)
    
    @staticmethod
    def jaccard_similarity_batch(sets1: List[set], sets2: List[set]) -> np.ndarray:
        """Compute Jaccard similarity for batches of sets."""
        similarities = []
        
        for s1, s2 in zip(sets1, sets2):
            intersection = len(s1 & s2)
            union = len(s1 | s2)
            similarity = intersection / union if union > 0 else 0.0
            similarities.append(similarity)
        
        return np.array(similarities)
    
    @staticmethod
    def edit_distance_batch(strings1: List[str], strings2: List[str]) -> np.ndarray:
        """Compute edit distance for batches of strings."""
        distances = []
        
        for s1, s2 in zip(strings1, strings2):
            distance = VectorizedOperations._edit_distance(s1, s2)
            distances.append(distance)
        
        return np.array(distances)
    
    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Compute edit distance between two strings."""
        if len(s1) < len(s2):
            return VectorizedOperations._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


class AlgorithmOptimizer:
    """Optimize watermarking algorithms for specific use cases."""
    
    def __init__(self):
        self.logger = get_logger("algorithm_optimizer")
        self.optimization_cache = {}
    
    @cached(ttl=3600)  # Cache for 1 hour
    def optimize_kirchenbauer_params(
        self,
        target_detection_rate: float = 0.95,
        target_quality_threshold: float = 0.8,
        test_texts: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Optimize Kirchenbauer watermark parameters."""
        if test_texts is None:
            # Use default test texts
            test_texts = [
                "This is a test text for parameter optimization.",
                "Machine learning models can generate realistic text.",
                "Watermarking helps identify AI-generated content."
            ]
        
        best_params = {"gamma": 0.25, "delta": 2.0}
        best_score = 0.0
        
        # Grid search over parameter space
        gamma_values = np.linspace(0.1, 0.5, 5)
        delta_values = np.linspace(1.0, 4.0, 4)
        
        for gamma in gamma_values:
            for delta in delta_values:
                try:
                    # Simulate performance with these parameters
                    detection_rate = self._simulate_detection_rate(gamma, delta, test_texts)
                    quality_score = self._simulate_quality_score(gamma, delta, test_texts)
                    
                    # Combined score (weighted)
                    score = (
                        0.7 * min(detection_rate / target_detection_rate, 1.0) +
                        0.3 * min(quality_score / target_quality_threshold, 1.0)
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_params = {"gamma": gamma, "delta": delta}
                
                except Exception as e:
                    self.logger.warning(f"Parameter optimization failed for gamma={gamma}, delta={delta}: {e}")
                    continue
        
        self.logger.info(f"Optimized Kirchenbauer parameters: {best_params} (score: {best_score:.3f})")
        return best_params
    
    def _simulate_detection_rate(self, gamma: float, delta: float, test_texts: List[str]) -> float:
        """Simulate detection rate for given parameters."""
        # Simplified simulation - in reality would use actual watermarking
        base_rate = 0.9
        gamma_effect = min(gamma * 2, 0.1)  # Gamma improves detection up to 0.1
        delta_effect = min((delta - 1) * 0.02, 0.08)  # Delta improves detection up to 0.08
        
        return min(1.0, base_rate + gamma_effect + delta_effect)
    
    def _simulate_quality_score(self, gamma: float, delta: float, test_texts: List[str]) -> float:
        """Simulate quality score for given parameters."""
        # Simplified simulation - in reality would measure actual quality
        base_quality = 0.95
        gamma_penalty = gamma * 0.2  # Higher gamma reduces quality
        delta_penalty = (delta - 1) * 0.05  # Higher delta reduces quality
        
        return max(0.0, base_quality - gamma_penalty - delta_penalty)


# Global instances
_global_profiler = PerformanceProfiler()
_global_memory_monitor = MemoryMonitor()
_global_batch_optimizer = BatchOptimizer()
_global_algorithm_optimizer = AlgorithmOptimizer()


def profile_performance(func: Callable) -> Callable:
    """Global performance profiling decorator."""
    return _global_profiler.profile(func)


def monitor_memory(func: Callable) -> Callable:
    """Global memory monitoring decorator."""
    return _global_memory_monitor.monitor(func)


def get_profiler() -> PerformanceProfiler:
    """Get global profiler instance."""
    return _global_profiler


def get_batch_optimizer() -> BatchOptimizer:
    """Get global batch optimizer instance."""
    return _global_batch_optimizer


def get_algorithm_optimizer() -> AlgorithmOptimizer:
    """Get global algorithm optimizer instance."""
    return _global_algorithm_optimizer


# Utility functions
def time_function(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """Time function execution and return result with duration."""
    start_time = time.time()
    result = func(*args, **kwargs)
    duration = time.time() - start_time
    return result, duration


def benchmark_function(func: Callable, num_runs: int = 10, *args, **kwargs) -> Dict[str, float]:
    """Benchmark function performance over multiple runs."""
    times = []
    
    for _ in range(num_runs):
        _, duration = time_function(func, *args, **kwargs)
        times.append(duration)
    
    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "median": np.median(times)
    }