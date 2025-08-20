"""Robust base watermarking interface with error handling and validation."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
import hashlib
import logging
from enum import Enum

# Fallback imports for environments without ML dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    HAS_TRANSFORMERS = False

class WatermarkError(Exception):
    """Base exception for watermarking errors."""
    pass

class ValidationError(WatermarkError):
    """Validation error for inputs."""
    pass

class ModelError(WatermarkError):
    """Model loading or inference error."""
    pass

class DetectionError(WatermarkError):
    """Watermark detection error."""
    pass

class WatermarkStrength(Enum):
    """Watermark strength levels."""
    LIGHT = "light"
    MEDIUM = "medium"
    STRONG = "strong"

@dataclass
class WatermarkConfig:
    """Enhanced configuration for watermark detection."""
    method: str
    key: str
    params: Dict[str, Any] = field(default_factory=dict)
    strength: WatermarkStrength = WatermarkStrength.MEDIUM
    model_name: Optional[str] = None
    version: str = "1.0"

@dataclass
class DetectionResult:
    """Enhanced result of watermark detection."""
    is_watermarked: bool
    confidence: float
    p_value: float
    test_statistic: Optional[float] = None
    token_scores: Optional[List[float]] = None
    method: str = ""
    processing_time: float = 0.0
    model_used: Optional[str] = None
    warning_messages: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate detection result."""
        if not 0 <= self.confidence <= 1:
            raise ValidationError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if not 0 <= self.p_value <= 1:
            raise ValidationError(f"P-value must be between 0 and 1, got {self.p_value}")

class RobustBaseWatermark(ABC):
    """Enhanced base class with robust error handling and validation."""
    
    def __init__(self, model_name: str = "gpt2", key: str = "default", **kwargs):
        """Initialize with validation and error handling."""
        self.model_name = self._validate_model_name(model_name)
        self.key = self._validate_key(key)
        self.params = kwargs
        self._tokenizer = None
        self._model = None
        self.logger = logging.getLogger(f"watermark.{self.__class__.__name__}")
        
        # Configuration validation
        self._validate_params(kwargs)
        
    def _validate_model_name(self, model_name: str) -> str:
        """Validate model name."""
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValidationError("Model name must be a non-empty string")
        return model_name.strip()
    
    def _validate_key(self, key: str) -> str:
        """Validate watermark key."""
        if not isinstance(key, str) or len(key) < 3:
            raise ValidationError("Key must be a string with at least 3 characters")
        return key
    
    def _validate_params(self, params: Dict[str, Any]) -> None:
        """Validate initialization parameters."""
        # Override in subclasses for specific validation
        pass
    
    def _validate_text_input(self, text: str, min_length: int = 1, max_length: int = 100000) -> str:
        """Validate text input."""
        if not isinstance(text, str):
            raise ValidationError(f"Text must be a string, got {type(text)}")
        
        text = text.strip()
        if len(text) < min_length:
            raise ValidationError(f"Text too short: {len(text)} < {min_length}")
        if len(text) > max_length:
            raise ValidationError(f"Text too long: {len(text)} > {max_length}")
        
        return text
    
    def _validate_generation_params(self, max_length: int, temperature: float) -> None:
        """Validate text generation parameters."""
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValidationError(f"max_length must be positive integer, got {max_length}")
        if max_length > 10000:
            self.logger.warning(f"Large max_length {max_length} may cause memory issues")
        
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            raise ValidationError(f"temperature must be positive number, got {temperature}")
        if temperature > 2.0:
            self.logger.warning(f"High temperature {temperature} may reduce watermark quality")
    
    @property
    def tokenizer(self):
        """Lazy load tokenizer with error handling."""
        if not HAS_TRANSFORMERS:
            raise ModelError("transformers library not available")
            
        if self._tokenizer is None:
            try:
                self.logger.info(f"Loading tokenizer: {self.model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
            except Exception as e:
                raise ModelError(f"Failed to load tokenizer {self.model_name}: {e}")
        
        return self._tokenizer
    
    @property
    def model(self):
        """Lazy load model with error handling."""
        if not HAS_TRANSFORMERS or not HAS_TORCH:
            raise ModelError("torch and transformers libraries required")
            
        if self._model is None:
            try:
                self.logger.info(f"Loading model: {self.model_name}")
                
                # Determine device and dtype
                if torch.cuda.is_available():
                    device_map = "auto"
                    torch_dtype = torch.float16
                    self.logger.info("Using CUDA with float16")
                else:
                    device_map = None
                    torch_dtype = torch.float32
                    self.logger.info("Using CPU with float32")
                
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    low_cpu_mem_usage=True
                )
                
            except Exception as e:
                raise ModelError(f"Failed to load model {self.model_name}: {e}")
        
        return self._model
    
    def hash_key(self, key: str) -> int:
        """Create reproducible hash from key with validation."""
        if not isinstance(key, str):
            raise ValidationError("Key must be a string")
        
        try:
            return int(hashlib.sha256(key.encode('utf-8')).hexdigest()[:8], 16)
        except Exception as e:
            raise ValidationError(f"Failed to hash key: {e}")
    
    @abstractmethod
    def generate(self, prompt: str, max_length: int = 100, **kwargs) -> str:
        """Generate watermarked text with validation."""
        pass
    
    @abstractmethod
    def detect(self, text: str) -> DetectionResult:
        """Detect watermark in text with validation."""
        pass
    
    def get_config(self) -> WatermarkConfig:
        """Get watermark configuration."""
        return WatermarkConfig(
            method=self.__class__.__name__.lower().replace("watermark", ""),
            key=self.key,
            params=self.params.copy(),
            model_name=self.model_name
        )
    
    def tokenize(self, text: str):
        """Tokenize text with error handling."""
        try:
            text = self._validate_text_input(text)
            if HAS_TORCH:
                return self.tokenizer.encode(text, return_tensors="pt")
            else:
                return self.tokenizer.encode(text)
        except Exception as e:
            raise ValidationError(f"Tokenization failed: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on watermark instance."""
        status = {
            "model_name": self.model_name,
            "key_set": bool(self.key),
            "tokenizer_loaded": self._tokenizer is not None,
            "model_loaded": self._model is not None,
            "torch_available": HAS_TORCH,
            "transformers_available": HAS_TRANSFORMERS,
            "cuda_available": HAS_TORCH and torch.cuda.is_available(),
            "issues": []
        }
        
        # Check for potential issues
        if not HAS_TORCH:
            status["issues"].append("PyTorch not available - limited functionality")
        if not HAS_TRANSFORMERS:
            status["issues"].append("Transformers not available - no model support")
        
        return status