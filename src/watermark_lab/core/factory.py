"""Watermark factory for creating different watermarking implementations."""

from typing import Dict, Any
from abc import ABC, abstractmethod


class BaseWatermark(ABC):
    """Base class for all watermarking implementations."""
    
    def __init__(self, **kwargs):
        """Initialize watermark with configuration."""
        self.config = kwargs
        self.method = self.__class__.__name__.lower().replace('watermark', '')
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate watermarked text from a prompt."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get watermark configuration."""
        pass


class KirchenbauerWatermark(BaseWatermark):
    """Kirchenbauer et al. watermarking implementation."""
    
    def __init__(self, gamma: float = 0.25, delta: float = 2.0, **kwargs):
        """Initialize Kirchenbauer watermark."""
        super().__init__(gamma=gamma, delta=delta, **kwargs)
        self.gamma = gamma  # Greenlist ratio
        self.delta = delta  # Bias strength
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate watermarked text using Kirchenbauer method."""
        max_length = kwargs.get('max_length', 100)
        
        # Simple demonstration - adds structured patterns
        watermarked_text = prompt + " "
        words_to_add = max_length // 10
        
        # Add words with controlled patterns (simulating greenlist bias)
        pattern_words = [
            "the", "and", "with", "for", "this", "that", "such", "these", 
            "through", "within", "among", "between", "during", "across"
        ]
        
        for i in range(words_to_add):
            # Simulate greenlist selection with gamma probability
            if i % 4 < int(4 * self.gamma):  # Simplified greenlist selection
                word = pattern_words[i % len(pattern_words)]
            else:
                word = f"word{i}"
            watermarked_text += word + " "
        
        return watermarked_text.strip()
    
    def get_config(self) -> Dict[str, Any]:
        """Get watermark configuration."""
        return {
            "method": "kirchenbauer",
            "gamma": self.gamma,
            "delta": self.delta,
            **self.config
        }


class MarkLLMWatermark(BaseWatermark):
    """MarkLLM watermarking implementation."""
    
    def __init__(self, algorithm: str = "KGW", watermark_strength: float = 2.0, **kwargs):
        """Initialize MarkLLM watermark."""
        super().__init__(algorithm=algorithm, watermark_strength=watermark_strength, **kwargs)
        self.algorithm = algorithm
        self.watermark_strength = watermark_strength
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate watermarked text using MarkLLM method."""
        max_length = kwargs.get('max_length', 100)
        
        # Placeholder implementation for MarkLLM
        watermarked_text = prompt + " "
        
        # Simulate different algorithms
        if self.algorithm == "KGW":
            # Key-based Grouped Watermarking
            words = ["knowledge", "generation", "watermark", "detection", "analysis"]
        elif self.algorithm == "SWEET":
            # Synchronous Watermarking with Embedding Token
            words = ["synchronous", "watermark", "embedding", "token", "secure"]
        else:
            words = ["watermark", "text", "generation", "method", "algorithm"]
        
        # Add words with strength-based selection
        words_to_add = max_length // 15
        for i in range(words_to_add):
            word = words[i % len(words)]
            # Apply watermark strength (higher strength = more predictable patterns)
            if i % int(5 / self.watermark_strength) == 0:
                watermarked_text += word + " "
            else:
                watermarked_text += f"text{i} "
        
        return watermarked_text.strip()
    
    def get_config(self) -> Dict[str, Any]:
        """Get watermark configuration."""
        return {
            "method": "markllm",
            "algorithm": self.algorithm,
            "watermark_strength": self.watermark_strength,
            **self.config
        }


class WatermarkFactory:
    """Factory for creating watermark instances."""
    
    _registry: Dict[str, type] = {
        "kirchenbauer": KirchenbauerWatermark,
        "markllm": MarkLLMWatermark,
    }
    
    @classmethod
    def register(cls, name: str, watermark_class: type) -> None:
        """Register a watermark implementation."""
        cls._registry[name] = watermark_class
    
    @classmethod
    def create(cls, method: str, **kwargs) -> BaseWatermark:
        """Create a watermark instance."""
        if method not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown watermark method: {method}. Available: {available}")
        
        return cls._registry[method](**kwargs)
    
    @classmethod
    def list_methods(cls) -> list:
        """List available watermark methods."""
        return list(cls._registry.keys())