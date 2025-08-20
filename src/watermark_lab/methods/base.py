"""Base watermarking interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import hashlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@dataclass
class WatermarkConfig:
    """Configuration for watermark detection."""
    method: str
    key: str
    params: Dict[str, Any]

@dataclass
class DetectionResult:
    """Result of watermark detection."""
    is_watermarked: bool
    confidence: float
    p_value: float
    test_statistic: Optional[float] = None
    token_scores: Optional[list] = None

class BaseWatermark(ABC):
    """Base class for all watermarking methods."""
    
    def __init__(self, model_name: str, key: str = "default", **kwargs):
        self.model_name = model_name
        self.key = key
        self.params = kwargs
        self._tokenizer = None
        self._model = None
        
    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer
    
    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
        return self._model
    
    def hash_key(self, key: str) -> int:
        """Create reproducible hash from key."""
        return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    
    @abstractmethod
    def generate(self, prompt: str, max_length: int = 100, **kwargs) -> str:
        """Generate watermarked text."""
        pass
    
    @abstractmethod
    def detect(self, text: str) -> DetectionResult:
        """Detect watermark in text."""
        pass
    
    def get_config(self) -> WatermarkConfig:
        """Get watermark configuration for detection."""
        return WatermarkConfig(
            method=self.__class__.__name__.lower().replace("watermark", ""),
            key=self.key,
            params=self.params
        )
    
    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text."""
        return self.tokenizer.encode(text, return_tensors="pt")