"""Advanced model loading and management utilities for production-grade watermarking."""

import os
import warnings
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    import torch
    import torch.nn.functional as F
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        GPT2LMHeadModel, GPT2Tokenizer,
        LlamaForCausalLM, LlamaTokenizer,
        BloomForCausalLM, BloomTokenizerFast
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    # Define comprehensive dummy torch module for fallback
    import numpy as np
    
    class TorchDevice:
        """Dummy torch.device replacement."""
        def __init__(self, device_type="cpu"):
            self.type = device_type
        
        def __str__(self):
            return self.type
    
    class TorchTensor:
        """Dummy torch.Tensor replacement using numpy arrays."""
        def __init__(self, data):
            if isinstance(data, (list, tuple)):
                self.data = np.array(data)
            else:
                self.data = np.array([data]) if np.isscalar(data) else data
            self.device = TorchDevice("cpu")
        
        def to(self, device):
            """No-op device transfer."""
            return self
        
        def __getitem__(self, key):
            return TorchTensor(self.data[key])
        
        def tolist(self):
            return self.data.tolist()
        
        @property
        def shape(self):
            return self.data.shape
    
    class TorchCuda:
        """Dummy torch.cuda replacement."""
        @staticmethod
        def is_available():
            return False
    
    class TorchBackends:
        """Dummy torch.backends replacement."""
        class mps:
            @staticmethod
            def is_available():
                return False
    
    class torch:
        """Comprehensive dummy torch module."""
        Tensor = TorchTensor
        device = TorchDevice
        cuda = TorchCuda
        backends = TorchBackends
        float16 = "float16"
        float32 = "float32"
        
        @staticmethod
        def tensor(data, device=None):
            """Create a dummy tensor."""
            return TorchTensor(data)
        
        @staticmethod
        def randn(*shape, device=None):
            """Generate random tensor using numpy."""
            if len(shape) == 1:
                return TorchTensor(np.random.randn(shape[0]))
            return TorchTensor(np.random.randn(*shape))
        
        @staticmethod
        def no_grad():
            """Dummy context manager."""
            class NoGradContext:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return NoGradContext()
        
        # Add other torch dtypes that might be used
        int32 = "int32"
        int64 = "int64"
        bool = "bool"
        
        @classmethod
        def __getattr__(cls, name):
            """Fallback for any missing torch attributes."""
            # Return string representation for dtype attributes
            return name
    
    # Add torch.nn.functional as dummy
    class F:
        pass
    
    torch.nn = type('nn', (), {'functional': F})()
        
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not available. Using fallback tokenization.")

from .logging import get_logger
from .exceptions import ModelLoadError, ValidationError
from .cache import get_cache_manager

logger = get_logger("model_loader")


@dataclass
class ModelConfig:
    """Configuration for model loading and inference."""
    
    model_name: str
    device: str = "auto"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    use_cache: bool = True
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "use_cache": self.use_cache,
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": self.trust_remote_code,
        }


class BaseModelWrapper(ABC):
    """Abstract base class for model wrappers."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = get_logger(f"model.{config.model_name}")
    
    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text into token IDs."""
        pass
    
    @abstractmethod
    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        pass
    
    @abstractmethod
    def generate_logits(self, input_ids: List[int], context_length: int = 50) -> torch.Tensor:
        """Generate logits for next token prediction."""
        pass
    
    @abstractmethod
    def generate_tokens(self, prompt: str, max_new_tokens: int = 50, **kwargs) -> List[str]:
        """Generate tokens from prompt."""
        pass


class FallbackModelWrapper(BaseModelWrapper):
    """Fallback model wrapper when transformers is not available."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Simple vocabulary for fallback
        self.vocab = [
            "the", "and", "to", "of", "a", "in", "is", "it", "you", "that",
            "he", "was", "for", "on", "are", "as", "with", "his", "they", "be",
            "at", "one", "have", "this", "from", "or", "had", "by", "not", "word",
            "but", "what", "some", "we", "can", "out", "other", "were", "all", "your",
            "when", "up", "use", "each", "which", "she", "do", "how", "their", "if",
            "will", "way", "about", "many", "then", "them", "would", "write", "like", "so",
            "algorithm", "watermark", "detection", "security", "model", "text", "generation",
            "method", "analysis", "system", "data", "content", "research", "evaluation"
        ] * 20  # Extend for larger vocab
        
        self.word_to_id = {word: i for i, word in enumerate(self.vocab)}
        self.id_to_word = {i: word for i, word in enumerate(self.vocab)}
        
        logger.warning(f"Using fallback tokenization for {config.model_name}")
    
    def tokenize(self, text: str) -> List[int]:
        """Simple whitespace tokenization with word-to-id mapping."""
        words = text.lower().split()
        return [self.word_to_id.get(word, 0) for word in words]
    
    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        words = [self.id_to_word.get(tid, "<unk>") for tid in token_ids]
        return " ".join(words)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def generate_logits(self, input_ids: List[int], context_length: int = 50) -> torch.Tensor:
        """Generate random logits as fallback."""
        vocab_size = self.get_vocab_size()
        return torch.randn(vocab_size)
    
    def generate_tokens(self, prompt: str, max_new_tokens: int = 50, **kwargs) -> List[str]:
        """Generate tokens using simple heuristics."""
        import random
        
        # Simple token generation
        common_tokens = ["the", "and", "to", "of", "a", "in", "with", "for"]
        content_tokens = ["watermark", "detection", "analysis", "method", "system"]
        
        tokens = []
        for i in range(min(max_new_tokens, 20)):  # Limit fallback generation
            if random.random() < 0.7:
                tokens.append(random.choice(common_tokens))
            else:
                tokens.append(random.choice(content_tokens))
        
        return tokens


class TransformersModelWrapper(BaseModelWrapper):
    """Wrapper for Hugging Face transformers models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ModelLoadError("Transformers library not available")
        
        self.device = self._get_device(config.device)
        self.model = None
        self.tokenizer = None
        self.cache_manager = get_cache_manager()
        
        self._load_model_and_tokenizer()
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer with caching."""
        cache_key = f"model_{self.config.model_name}_{self.device}"
        
        cached_model = self.cache_manager.get(cache_key) if self.config.use_cache else None
        if cached_model:
            self.model, self.tokenizer = cached_model
            logger.info(f"Loaded cached model: {self.config.model_name}")
            return
        
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Handle pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Determine torch dtype
            torch_dtype = torch.float16 if self.config.torch_dtype == "auto" else getattr(torch, self.config.torch_dtype)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=self.config.trust_remote_code,
                low_cpu_mem_usage=True
            )
            
            if self.device.type != "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()  # Set to evaluation mode
            
            # Cache the loaded model
            if self.config.use_cache:
                self.cache_manager.set(cache_key, (self.model, self.tokenizer), ttl=3600)
            
            logger.info(f"Successfully loaded {self.config.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.config.model_name}: {e}")
            raise ModelLoadError(f"Failed to load model: {e}")
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text using the model's tokenizer."""
        if not self.tokenizer:
            raise ModelLoadError("Tokenizer not loaded")
        
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return tokens
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise ValidationError(f"Tokenization failed: {e}")
    
    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        if not self.tokenizer:
            raise ModelLoadError("Tokenizer not loaded")
        
        try:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            return text
        except Exception as e:
            logger.error(f"Detokenization failed: {e}")
            return "<decoding_error>"
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if not self.tokenizer:
            return 50257  # GPT-2 default
        return len(self.tokenizer)
    
    def generate_logits(self, input_ids: List[int], context_length: int = 50) -> torch.Tensor:
        """Generate logits for next token prediction."""
        if not self.model:
            raise ModelLoadError("Model not loaded")
        
        try:
            # Limit context length
            if len(input_ids) > context_length:
                input_ids = input_ids[-context_length:]
            
            # Convert to tensor
            input_tensor = torch.tensor([input_ids], device=self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs.logits[0, -1, :]  # Last token logits
            
            return logits
            
        except Exception as e:
            logger.error(f"Logits generation failed: {e}")
            # Fallback to random logits
            return torch.randn(self.get_vocab_size(), device=self.device)
    
    def generate_tokens(self, prompt: str, max_new_tokens: int = 50, **kwargs) -> List[str]:
        """Generate tokens from prompt using the model."""
        if not self.model or not self.tokenizer:
            raise ModelLoadError("Model or tokenizer not loaded")
        
        try:
            # Tokenize input
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generation parameters
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "top_k": kwargs.get("top_k", self.config.top_k),
                "repetition_penalty": kwargs.get("repetition_penalty", self.config.repetition_penalty),
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "do_sample": True,
                "use_cache": True
            }
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(input_ids, **generation_kwargs)
            
            # Extract new tokens
            new_token_ids = output_ids[0, input_ids.shape[1]:].tolist()
            
            # Convert to words
            tokens = []
            for token_id in new_token_ids:
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=True).strip()
                if token_text:
                    tokens.extend(token_text.split())
            
            return tokens
            
        except Exception as e:
            logger.error(f"Token generation failed: {e}")
            # Fallback to simple generation
            fallback = FallbackModelWrapper(self.config)
            return fallback.generate_tokens(prompt, max_new_tokens, **kwargs)


class ModelManager:
    """Manages multiple models and provides unified interface."""
    
    def __init__(self):
        self.models: Dict[str, BaseModelWrapper] = {}
        self.logger = get_logger("model_manager")
    
    def load_model(self, model_name: str, config: Optional[ModelConfig] = None) -> BaseModelWrapper:
        """Load a model with optional configuration."""
        if model_name in self.models:
            return self.models[model_name]
        
        if config is None:
            config = ModelConfig(model_name=model_name)
        
        try:
            if TRANSFORMERS_AVAILABLE and not model_name.startswith("fallback"):
                model = TransformersModelWrapper(config)
            else:
                model = FallbackModelWrapper(config)
            
            self.models[model_name] = model
            self.logger.info(f"Loaded model: {model_name}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            # Fall back to fallback model
            fallback_config = ModelConfig(model_name=f"fallback_{model_name}")
            fallback_model = FallbackModelWrapper(fallback_config)
            self.models[model_name] = fallback_model
            return fallback_model
    
    def get_model(self, model_name: str) -> BaseModelWrapper:
        """Get a loaded model."""
        if model_name not in self.models:
            return self.load_model(model_name)
        return self.models[model_name]
    
    def list_models(self) -> List[str]:
        """List all loaded models."""
        return list(self.models.keys())
    
    def unload_model(self, model_name: str):
        """Unload a model to free memory."""
        if model_name in self.models:
            del self.models[model_name]
            self.logger.info(f"Unloaded model: {model_name}")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model."""
        if model_name not in self.models:
            return {"status": "not_loaded"}
        
        model = self.models[model_name]
        return {
            "status": "loaded",
            "type": model.__class__.__name__,
            "vocab_size": model.get_vocab_size(),
            "config": model.config.to_dict() if hasattr(model.config, 'to_dict') else str(model.config)
        }


# Global model manager instance
_global_model_manager = None

def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = ModelManager()
    return _global_model_manager


# Common model configurations
COMMON_MODEL_CONFIGS = {
    "gpt2": ModelConfig(
        model_name="gpt2",
        max_length=1024,
        temperature=0.7
    ),
    "gpt2-medium": ModelConfig(
        model_name="gpt2-medium",
        max_length=1024,
        temperature=0.7
    ),
    "gpt2-large": ModelConfig(
        model_name="gpt2-large",
        max_length=1024,
        temperature=0.6
    ),
    "distilgpt2": ModelConfig(
        model_name="distilgpt2",
        max_length=1024,
        temperature=0.8
    ),
    "microsoft/DialoGPT-medium": ModelConfig(
        model_name="microsoft/DialoGPT-medium",
        max_length=512,
        temperature=0.7
    ),
}

def get_common_config(model_name: str) -> Optional[ModelConfig]:
    """Get a common model configuration."""
    return COMMON_MODEL_CONFIGS.get(model_name)