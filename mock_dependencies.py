"""
Mock dependencies for autonomous testing without external packages.
"""

import sys
import math
import random

# Mock transformers
class MockTokenizer:
    def encode(self, text):
        return [hash(word) % 1000 for word in text.split()]
    
    def decode(self, token_ids):
        return " ".join([f"token_{tid}" for tid in token_ids])

class MockModel:
    def generate(self, input_ids, max_length=50, **kwargs):
        # Simple mock generation
        return [[random.randint(0, 999) for _ in range(max_length // 4)]]

class MockTransformers:
    AutoTokenizer = MockTokenizer
    AutoModel = MockModel

# Mock numpy
class MockNumpyRandom:
    class RandomState:
        def __init__(self, seed):
            random.seed(seed)
        def permutation(self, n):
            items = list(range(n))
            random.shuffle(items)
            return items
        def choice(self, items, p=None):
            if isinstance(items, int):
                return random.randint(0, items-1)
            return random.choice(items)

class MockNumpy:
    random = MockNumpyRandom
    
    @staticmethod
    def array(data):
        return data
    
    @staticmethod
    def mean(data):
        return sum(data) / len(data) if data else 0
    
    @staticmethod
    def std(data):
        if not data:
            return 0
        mean_val = sum(data) / len(data)
        variance = sum((x - mean_val) ** 2 for x in data) / len(data)
        return math.sqrt(variance)
    
    @staticmethod
    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))
    
    @staticmethod
    def linalg():
        class LinAlg:
            @staticmethod
            def norm(data):
                return math.sqrt(sum(x * x for x in data))
        return LinAlg()

# Mock torch
class MockTorch:
    device = lambda x: x
    cuda = type('cuda', (), {'is_available': lambda: False})
    
    @staticmethod
    def is_available():
        return False

# Mock scipy
class MockScipy:
    class stats:
        class norm:
            @staticmethod
            def cdf(x):
                return 0.5 * (1 + math.erf(x / math.sqrt(2)))

# Mock sklearn
class MockSklearn:
    class metrics:
        class pairwise:
            @staticmethod
            def cosine_similarity(a, b):
                return [[0.8]]

# Install mocks
sys.modules['numpy'] = MockNumpy()
sys.modules['torch'] = MockTorch()
sys.modules['transformers'] = MockTransformers()
sys.modules['scipy'] = MockScipy()
sys.modules['scipy.stats'] = MockScipy.stats()
sys.modules['sklearn'] = MockSklearn()
sys.modules['sklearn.metrics'] = MockSklearn.metrics()
sys.modules['sklearn.metrics.pairwise'] = MockSklearn.metrics.pairwise()