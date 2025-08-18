"""Fallback implementations for optional dependencies."""

import math

# Fallback numpy functions
class MockNumpy:
    @staticmethod
    def random():
        class RandomState:
            def __init__(self, seed):
                import random
                random.seed(seed)
                self._random = random
            
            def permutation(self, n):
                items = list(range(n))
                self._random.shuffle(items)
                return items
        
        class Random:
            @staticmethod
            def RandomState(seed):
                return RandomState(seed)
        
        return Random()
    
    @staticmethod
    def mean(data):
        return sum(data) / len(data) if data else 0.0
    
    @staticmethod
    def std(data):
        if not data:
            return 0.0
        mean_val = sum(data) / len(data)
        variance = sum((x - mean_val) ** 2 for x in data) / len(data)
        return math.sqrt(variance)

# Try to import numpy, fall back if not available
try:
    import numpy as np
    numpy_available = True
except ImportError:
    np = MockNumpy()
    numpy_available = False

# Mock scipy.stats
class MockStats:
    class norm:
        @staticmethod
        def cdf(x):
            # Simple approximation of normal CDF
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

try:
    from scipy import stats
    scipy_available = True
except ImportError:
    stats = MockStats()
    scipy_available = False

# Mock psutil
class MockPsutil:
    @staticmethod
    def cpu_percent():
        return 0.0
    
    class Process:
        def memory_info(self):
            class MemInfo:
                def __init__(self):
                    self.rss = 1024 * 1024 * 50  # 50MB
            return MemInfo()

try:
    import psutil
    psutil_available = True
except ImportError:
    psutil = MockPsutil()
    psutil_available = False

__all__ = ['np', 'stats', 'psutil', 'numpy_available', 'scipy_available', 'psutil_available']