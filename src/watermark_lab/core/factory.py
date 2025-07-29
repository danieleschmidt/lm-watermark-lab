"""Watermark factory for creating different watermarking implementations."""

from typing import Dict, Any
from abc import ABC, abstractmethod


class BaseWatermark(ABC):
    """Base class for all watermarking implementations."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate watermarked text from a prompt."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get watermark configuration."""
        pass


class WatermarkFactory:
    """Factory for creating watermark instances."""
    
    _registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, watermark_class: type) -> None:
        """Register a watermark implementation."""
        cls._registry[name] = watermark_class
    
    @classmethod
    def create(cls, method: str, **kwargs) -> BaseWatermark:
        """Create a watermark instance."""
        if method not in cls._registry:
            raise ValueError(f"Unknown watermark method: {method}")
        
        return cls._registry[method](**kwargs)
    
    @classmethod
    def list_methods(cls) -> list:
        """List available watermark methods."""
        return list(cls._registry.keys())