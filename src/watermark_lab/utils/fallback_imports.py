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
    def cpu_percent(interval=None):
        import random
        return random.uniform(10, 80)
    
    @staticmethod
    def virtual_memory():
        class MemoryInfo:
            def __init__(self):
                self.total = 8 * 1024 * 1024 * 1024  # 8GB
                self.available = 4 * 1024 * 1024 * 1024  # 4GB  
                self.percent = 50.0
                self.used = 4 * 1024 * 1024 * 1024
        return MemoryInfo()
    
    @staticmethod
    def disk_usage(path):
        class DiskUsage:
            def __init__(self):
                self.total = 100 * 1024 * 1024 * 1024  # 100GB
                self.used = 50 * 1024 * 1024 * 1024   # 50GB
                self.free = 50 * 1024 * 1024 * 1024    # 50GB
        return DiskUsage()
    
    @staticmethod  
    def net_connections():
        return [1, 2, 3]  # Mock connections
    
    @staticmethod
    def Process():
        class MockProcess:
            def open_files(self):
                return [1, 2, 3]  # Mock file handles
            def rlimit(self, resource):
                return (1024, 1024)
            def memory_info(self):
                class MemInfo:
                    def __init__(self):
                        self.rss = 1024 * 1024 * 50  # 50MB
                return MemInfo()
        return MockProcess()

try:
    import psutil
    psutil_available = True
except ImportError:
    psutil = MockPsutil()
    psutil_available = False

__all__ = ['np', 'stats', 'psutil', 'numpy_available', 'scipy_available', 'psutil_available']