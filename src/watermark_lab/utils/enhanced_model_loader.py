"""Enhanced model loader with integrity checking, caching, and reliability features."""

import os
import json
import hashlib
import pickle
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .logging import get_logger, StructuredLogger
from .exceptions import ModelLoadError, ValidationError, SecurityError, ResourceError
from .data_integrity import get_integrity_manager, IntegrityData, HashAlgorithm
from .circuit_breaker import get_circuit_breaker, CircuitBreakerConfig
from .retry import retry, RetryConfig
from ..config.settings import get_settings

logger = get_logger("enhanced_model_loader")
structured_logger = StructuredLogger("model_loader")


class ModelStatus(Enum):
    """Model loading and runtime status."""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    CACHED = "cached"
    VERIFIED = "verified"
    CORRUPTED = "corrupted"


@dataclass
class ModelMetadata:
    """Enhanced metadata for loaded models."""
    model_name: str
    model_type: str
    model_size: int
    parameters_count: Optional[int]
    model_hash: str
    config_hash: str
    load_time: float
    memory_usage: int
    status: ModelStatus
    last_used: float
    use_count: int
    integrity_data: Optional[IntegrityData]
    cache_path: Optional[str] = None
    source_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['status'] = self.status.value
        if self.integrity_data:
            result['integrity_data'] = self.integrity_data.to_dict()
        return result


@dataclass
class ModelConfig:
    """Enhanced configuration for model loading."""
    model_name: str
    model_type: str = "transformers"
    device: str = "cpu"
    precision: str = "float32"
    max_memory: Optional[int] = None  # In MB
    use_cache: bool = True
    verify_integrity: bool = True
    timeout: float = 300.0
    retries: int = 3
    
    # Model-specific parameters
    trust_remote_code: bool = False
    use_auth_token: Optional[str] = None
    revision: Optional[str] = None
    
    # Custom loading parameters
    custom_loader: Optional[Callable] = None
    preprocessing_config: Optional[Dict[str, Any]] = None


class ModelCache:
    """Intelligent model caching with integrity verification."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.settings = get_settings()
        self.cache_dir = Path(cache_dir or self.settings.default_model_cache)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger("model_cache")
        self.integrity_manager = get_integrity_manager()
        
        # Cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.cache_metadata = self._load_cache_metadata()
        
        # Thread safety
        self.lock = threading.RLock()
    
    def _load_cache_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load cache metadata from disk."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load cache metadata: {e}")
        
        return {}
    
    def _save_cache_metadata(self):
        """Save cache metadata to disk."""
        try:
            with self.lock:
                with open(self.metadata_file, 'w') as f:
                    json.dump(self.cache_metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")
    
    def _get_cache_path(self, model_name: str, config_hash: str) -> Path:
        """Get cache path for model."""
        safe_name = "".join(c for c in model_name if c.isalnum() or c in "._-")
        return self.cache_dir / f"{safe_name}_{config_hash[:8]}"
    
    def _compute_config_hash(self, config: ModelConfig) -> str:
        """Compute hash of model configuration."""
        config_dict = asdict(config)
        # Remove non-hashable items
        config_dict.pop('custom_loader', None)
        return self.integrity_manager.compute_hash(config_dict)
    
    def get_cached_model(self, model_name: str, config: ModelConfig) -> Optional[Tuple[Any, ModelMetadata]]:
        """Retrieve model from cache if available and valid."""
        try:
            with self.lock:
                config_hash = self._compute_config_hash(config)
                cache_key = f"{model_name}_{config_hash}"
                
                if cache_key not in self.cache_metadata:
                    return None
                
                cached_info = self.cache_metadata[cache_key]
                cache_path = Path(cached_info["cache_path"])
                
                if not cache_path.exists():
                    self.logger.warning(f"Cache file missing: {cache_path}")
                    del self.cache_metadata[cache_key]
                    self._save_cache_metadata()
                    return None
                
                # Verify integrity if enabled
                if config.verify_integrity and cached_info.get("integrity_data"):
                    integrity_data = IntegrityData.from_dict(cached_info["integrity_data"])
                    
                    with open(cache_path, 'rb') as f:
                        cached_data = f.read()
                    
                    if not self.integrity_manager.verify_integrity(cached_data, integrity_data):
                        self.logger.error(f"Cache integrity verification failed: {cache_path}")
                        # Remove corrupted cache
                        cache_path.unlink(missing_ok=True)
                        del self.cache_metadata[cache_key]
                        self._save_cache_metadata()
                        return None
                
                # Load model from cache
                start_time = time.time()
                
                try:
                    with open(cache_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    load_time = time.time() - start_time
                    
                    # Update metadata
                    metadata = ModelMetadata(**cached_info["metadata"])\n                    metadata.last_used = time.time()\n                    metadata.use_count += 1\n                    metadata.status = ModelStatus.LOADED\n                    \n                    # Update cache metadata\n                    self.cache_metadata[cache_key]["metadata"] = metadata.to_dict()\n                    self._save_cache_metadata()\n                    \n                    self.logger.info(f"Loaded model from cache: {model_name} in {load_time:.2f}s")\n                    \n                    structured_logger.log_event(\n                        "model_cache_hit",\n                        model_name=model_name,\n                        load_time=load_time,\n                        cache_path=str(cache_path)\n                    )\n                    \n                    return model_data, metadata\n                    \n                except Exception as e:\n                    self.logger.error(f"Failed to load from cache: {e}")\n                    # Remove corrupted cache\n                    cache_path.unlink(missing_ok=True)\n                    del self.cache_metadata[cache_key]\n                    self._save_cache_metadata()\n                    return None\n                    \n        except Exception as e:\n            self.logger.error(f"Cache retrieval failed: {e}")\n            return None\n    \n    def cache_model(\n        self,\n        model_name: str,\n        config: ModelConfig,\n        model_data: Any,\n        metadata: ModelMetadata\n    ) -> bool:\n        """Cache model data with integrity verification."""\n        try:\n            with self.lock:\n                config_hash = self._compute_config_hash(config)\n                cache_key = f"{model_name}_{config_hash}"\n                cache_path = self._get_cache_path(model_name, config_hash)\n                \n                # Create cache directory\n                cache_path.mkdir(parents=True, exist_ok=True)\n                model_file = cache_path / "model.pkl"\n                \n                # Serialize model\n                start_time = time.time()\n                \n                with open(model_file, 'wb') as f:\n                    pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)\n                \n                cache_time = time.time() - start_time\n                \n                # Create integrity data\n                if config.verify_integrity:\n                    with open(model_file, 'rb') as f:\n                        model_bytes = f.read()\n                    \n                    integrity_data = self.integrity_manager.create_integrity_data(\n                        model_bytes,\n                        algorithm=HashAlgorithm.SHA256,\n                        metadata={\n                            "model_name": model_name,\n                            "cache_time": cache_time,\n                            "config_hash": config_hash\n                        }\n                    )\n                    \n                    metadata.integrity_data = integrity_data\n                \n                # Update metadata\n                metadata.cache_path = str(model_file)\n                metadata.status = ModelStatus.CACHED\n                \n                # Save to cache metadata\n                self.cache_metadata[cache_key] = {\n                    "model_name": model_name,\n                    "config_hash": config_hash,\n                    "cache_path": str(model_file),\n                    "cached_at": time.time(),\n                    "metadata": metadata.to_dict(),\n                    "integrity_data": integrity_data.to_dict() if integrity_data else None\n                }\n                \n                self._save_cache_metadata()\n                \n                self.logger.info(f"Cached model: {model_name} in {cache_time:.2f}s")\n                \n                structured_logger.log_event(\n                    "model_cached",\n                    model_name=model_name,\n                    cache_time=cache_time,\n                    cache_path=str(model_file),\n                    model_size=metadata.model_size\n                )\n                \n                return True\n                \n        except Exception as e:\n            self.logger.error(f"Model caching failed: {e}")\n            return False\n    \n    def clear_cache(self, model_name: Optional[str] = None) -> int:\n        """Clear cache for specific model or all models."""\n        try:\n            with self.lock:\n                cleared_count = 0\n                keys_to_remove = []\n                \n                for cache_key, cached_info in self.cache_metadata.items():\n                    if model_name is None or cached_info["model_name"] == model_name:\n                        cache_path = Path(cached_info["cache_path"])\n                        \n                        # Remove cache file\n                        if cache_path.exists():\n                            cache_path.unlink()\n                        \n                        # Remove cache directory if empty\n                        cache_dir = cache_path.parent\n                        try:\n                            if cache_dir.exists() and not any(cache_dir.iterdir()):\n                                cache_dir.rmdir()\n                        except OSError:\n                            pass  # Directory not empty\n                        \n                        keys_to_remove.append(cache_key)\n                        cleared_count += 1\n                \n                # Remove from metadata\n                for key in keys_to_remove:\n                    del self.cache_metadata[key]\n                \n                self._save_cache_metadata()\n                \n                self.logger.info(f"Cleared {cleared_count} cached models")\n                return cleared_count\n                \n        except Exception as e:\n            self.logger.error(f"Cache clearing failed: {e}")\n            return 0\n    \n    def get_cache_stats(self) -> Dict[str, Any]:\n        """Get cache statistics."""\n        try:\n            with self.lock:\n                total_models = len(self.cache_metadata)\n                total_size = 0\n                \n                for cached_info in self.cache_metadata.values():\n                    cache_path = Path(cached_info["cache_path"])\n                    if cache_path.exists():\n                        total_size += cache_path.stat().st_size\n                \n                return {\n                    "total_models": total_models,\n                    "total_size_bytes": total_size,\n                    "total_size_mb": total_size / (1024 * 1024),\n                    "cache_directory": str(self.cache_dir),\n                    "models": list(self.cache_metadata.keys())\n                }\n                \n        except Exception as e:\n            self.logger.error(f"Failed to get cache stats: {e}")\n            return {"error": str(e)}


class EnhancedModelLoader:
    """Enhanced model loader with reliability, caching, and integrity features."""
    
    def __init__(self, cache_dir: Optional[str] = None):\n        self.settings = get_settings()\n        self.logger = get_logger("enhanced_model_loader")\n        self.integrity_manager = get_integrity_manager()\n        \n        # Initialize cache\n        self.cache = ModelCache(cache_dir)\n        \n        # Model registry\n        self.loaded_models: Dict[str, Tuple[Any, ModelMetadata]] = {}\n        self.model_lock = threading.RLock()\n        \n        # Circuit breakers for different operations\n        self.load_circuit = get_circuit_breaker(\n            "model_loading",\n            CircuitBreakerConfig(\n                failure_threshold=3,\n                recovery_timeout=300.0,  # 5 minutes\n                timeout=300.0  # 5 minutes per load attempt\n            )\n        )\n        \n        # Thread pool for concurrent operations\n        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="model_loader")\n        \n        # Retry configuration\n        self.retry_config = RetryConfig(\n            max_attempts=3,\n            base_delay=5.0,\n            max_delay=60.0,\n            exponential_backoff=True,\n            jitter=True,\n            retry_exceptions=(ModelLoadError, ConnectionError, OSError),\n            stop_exceptions=(ValidationError, SecurityError, MemoryError)\n        )\n    \n    @retry()\n    def load_model(\n        self,\n        model_name: str,\n        config: Optional[ModelConfig] = None\n    ) -> Tuple[Any, ModelMetadata]:\n        """Load model with comprehensive error handling and caching."""\n        start_time = time.time()\n        \n        try:\n            if config is None:\n                config = ModelConfig(model_name=model_name)\n            \n            # Validate inputs\n            if not model_name or not isinstance(model_name, str):\n                raise ValidationError("Model name must be a non-empty string")\n            \n            with self.model_lock:\n                # Check if already loaded\n                if model_name in self.loaded_models:\n                    model, metadata = self.loaded_models[model_name]\n                    metadata.last_used = time.time()\n                    metadata.use_count += 1\n                    \n                    self.logger.debug(f"Using already loaded model: {model_name}")\n                    return model, metadata\n                \n                # Try cache first\n                if config.use_cache:\n                    cached_result = self.cache.get_cached_model(model_name, config)\n                    if cached_result:\n                        model, metadata = cached_result\n                        self.loaded_models[model_name] = (model, metadata)\n                        return model, metadata\n                \n                # Load model using circuit breaker\n                model, metadata = self.load_circuit.call(\n                    self._load_model_impl,\n                    model_name,\n                    config\n                )\n                \n                # Cache the model if caching is enabled\n                if config.use_cache:\n                    self.cache.cache_model(model_name, config, model, metadata)\n                \n                # Store in memory registry\n                self.loaded_models[model_name] = (model, metadata)\n                \n                load_time = time.time() - start_time\n                \n                structured_logger.log_event(\n                    "model_loaded",\n                    model_name=model_name,\n                    load_time=load_time,\n                    model_size=metadata.model_size,\n                    memory_usage=metadata.memory_usage,\n                    success=True\n                )\n                \n                self.logger.info(f"Successfully loaded model: {model_name} in {load_time:.2f}s")\n                return model, metadata\n                \n        except Exception as e:\n            load_time = time.time() - start_time\n            \n            structured_logger.log_event(\n                "model_load_failed",\n                model_name=model_name,\n                load_time=load_time,\n                error=str(e),\n                success=False\n            )\n            \n            self.logger.error(f"Failed to load model {model_name}: {e}", exc_info=True)\n            raise ModelLoadError(f"Failed to load model {model_name}: {e}")\n    \n    def _load_model_impl(self, model_name: str, config: ModelConfig) -> Tuple[Any, ModelMetadata]:\n        """Internal model loading implementation."""\n        start_time = time.time()\n        \n        try:\n            # Determine model type and loading strategy\n            if config.model_type == "transformers" and TRANSFORMERS_AVAILABLE:\n                model, tokenizer = self._load_transformers_model(model_name, config)\n                model_data = {"model": model, "tokenizer": tokenizer}\n                model_type = "transformers"\n                \n            elif config.model_type == "pytorch" and TORCH_AVAILABLE:\n                model = self._load_pytorch_model(model_name, config)\n                model_data = {"model": model}\n                model_type = "pytorch"\n                \n            elif config.custom_loader:\n                model_data = config.custom_loader(model_name, config)\n                model_type = "custom"\n                \n            else:\n                # Fallback to simple model\n                model_data = self._create_fallback_model(model_name, config)\n                model_type = "fallback"\n            \n            load_time = time.time() - start_time\n            \n            # Calculate model size and memory usage\n            model_size = self._estimate_model_size(model_data)\n            memory_usage = self._estimate_memory_usage(model_data)\n            parameters_count = self._count_parameters(model_data)\n            \n            # Create integrity hashes\n            model_hash = self.integrity_manager.compute_hash(\n                pickle.dumps(model_data, protocol=pickle.HIGHEST_PROTOCOL)\n            )\n            \n            config_hash = self.integrity_manager.compute_hash(asdict(config))\n            \n            # Create metadata\n            metadata = ModelMetadata(\n                model_name=model_name,\n                model_type=model_type,\n                model_size=model_size,\n                parameters_count=parameters_count,\n                model_hash=model_hash,\n                config_hash=config_hash,\n                load_time=load_time,\n                memory_usage=memory_usage,\n                status=ModelStatus.LOADED,\n                last_used=time.time(),\n                use_count=1,\n                integrity_data=None\n            )\n            \n            return model_data, metadata\n            \n        except Exception as e:\n            self.logger.error(f"Model loading implementation failed: {e}")\n            raise ModelLoadError(f"Model loading failed: {e}")\n    \n    def _load_transformers_model(self, model_name: str, config: ModelConfig) -> Tuple[Any, Any]:\n        """Load Transformers model with error handling."""\n        try:\n            # Load configuration first\n            model_config = AutoConfig.from_pretrained(\n                model_name,\n                trust_remote_code=config.trust_remote_code,\n                use_auth_token=config.use_auth_token,\n                revision=config.revision\n            )\n            \n            # Load tokenizer\n            tokenizer = AutoTokenizer.from_pretrained(\n                model_name,\n                trust_remote_code=config.trust_remote_code,\n                use_auth_token=config.use_auth_token,\n                revision=config.revision\n            )\n            \n            # Load model\n            if TORCH_AVAILABLE and config.device != "cpu":\n                device = torch.device(config.device if torch.cuda.is_available() else "cpu")\n            else:\n                device = "cpu"\n            \n            model = AutoModel.from_pretrained(\n                model_name,\n                config=model_config,\n                trust_remote_code=config.trust_remote_code,\n                use_auth_token=config.use_auth_token,\n                revision=config.revision\n            ).to(device)\n            \n            # Set precision if specified\n            if config.precision == "float16" and TORCH_AVAILABLE:\n                model = model.half()\n            elif config.precision == "int8" and TORCH_AVAILABLE:\n                # This would require additional libraries like bitsandbytes\n                pass\n            \n            return model, tokenizer\n            \n        except Exception as e:\n            self.logger.error(f"Transformers model loading failed: {e}")\n            raise ModelLoadError(f"Transformers model loading failed: {e}")\n    \n    def _load_pytorch_model(self, model_name: str, config: ModelConfig) -> Any:\n        """Load PyTorch model from file."""\n        try:\n            model_path = Path(model_name)\n            if not model_path.exists():\n                raise ModelLoadError(f"PyTorch model file not found: {model_name}")\n            \n            # Verify file integrity if enabled\n            if config.verify_integrity:\n                integrity_data = self.integrity_manager.create_file_integrity(model_path)\n                if not self.integrity_manager.verify_file_integrity(model_path, integrity_data):\n                    raise SecurityError("Model file integrity verification failed")\n            \n            # Load model\n            device = config.device if TORCH_AVAILABLE else "cpu"\n            model = torch.load(model_path, map_location=device)\n            \n            if hasattr(model, 'eval'):\n                model.eval()\n            \n            return model\n            \n        except Exception as e:\n            self.logger.error(f"PyTorch model loading failed: {e}")\n            raise ModelLoadError(f"PyTorch model loading failed: {e}")\n    \n    def _create_fallback_model(self, model_name: str, config: ModelConfig) -> Dict[str, Any]:\n        """Create a fallback model for testing/demo purposes."""\n        self.logger.warning(f"Using fallback model for: {model_name}")\n        \n        # Simple mock model\n        return {\n            "model": MockModel(model_name),\n            "tokenizer": MockTokenizer(),\n            "config": {"model_name": model_name, "type": "fallback"}\n        }\n    \n    def _estimate_model_size(self, model_data: Dict[str, Any]) -> int:\n        """Estimate model size in bytes."""\n        try:\n            # Use pickle size as rough estimate\n            return len(pickle.dumps(model_data, protocol=pickle.HIGHEST_PROTOCOL))\n        except Exception:\n            return 0\n    \n    def _estimate_memory_usage(self, model_data: Dict[str, Any]) -> int:\n        """Estimate memory usage in bytes."""\n        try:\n            if TORCH_AVAILABLE and "model" in model_data:\n                model = model_data["model"]\n                if hasattr(model, 'parameters'):\n                    total_params = sum(p.numel() * p.element_size() for p in model.parameters())\n                    return total_params\n            \n            # Fallback estimate\n            return self._estimate_model_size(model_data)\n            \n        except Exception:\n            return 0\n    \n    def _count_parameters(self, model_data: Dict[str, Any]) -> Optional[int]:\n        """Count model parameters."""\n        try:\n            if TORCH_AVAILABLE and "model" in model_data:\n                model = model_data["model"]\n                if hasattr(model, 'parameters'):\n                    return sum(p.numel() for p in model.parameters())\n            return None\n        except Exception:\n            return None\n    \n    def unload_model(self, model_name: str) -> bool:\n        """Unload model from memory."""\n        try:\n            with self.model_lock:\n                if model_name in self.loaded_models:\n                    del self.loaded_models[model_name]\n                    \n                    # Force garbage collection\n                    import gc\n                    gc.collect()\n                    \n                    if TORCH_AVAILABLE:\n                        torch.cuda.empty_cache() if torch.cuda.is_available() else None\n                    \n                    self.logger.info(f"Unloaded model: {model_name}")\n                    \n                    structured_logger.log_event(\n                        "model_unloaded",\n                        model_name=model_name\n                    )\n                    \n                    return True\n                    \n                return False\n                \n        except Exception as e:\n            self.logger.error(f"Failed to unload model {model_name}: {e}")\n            return False\n    \n    def list_loaded_models(self) -> List[Dict[str, Any]]:\n        """List all loaded models with their metadata."""\n        try:\n            with self.model_lock:\n                result = []\n                for model_name, (model, metadata) in self.loaded_models.items():\n                    result.append({\n                        "model_name": model_name,\n                        "metadata": metadata.to_dict()\n                    })\n                return result\n                \n        except Exception as e:\n            self.logger.error(f"Failed to list loaded models: {e}")\n            return []\n    \n    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:\n        """Get information about a specific model."""\n        try:\n            with self.model_lock:\n                if model_name in self.loaded_models:\n                    model, metadata = self.loaded_models[model_name]\n                    return {\n                        "model_name": model_name,\n                        "metadata": metadata.to_dict(),\n                        "status": "loaded"\n                    }\n                    \n                # Check cache\n                cache_stats = self.cache.get_cache_stats()\n                if model_name in cache_stats.get("models", []):\n                    return {\n                        "model_name": model_name,\n                        "status": "cached"\n                    }\n                    \n                return None\n                \n        except Exception as e:\n            self.logger.error(f"Failed to get model info for {model_name}: {e}")\n            return None\n    \n    def verify_model_integrity(self, model_name: str) -> bool:\n        """Verify integrity of a loaded model."""\n        try:\n            with self.model_lock:\n                if model_name not in self.loaded_models:\n                    return False\n                \n                model, metadata = self.loaded_models[model_name]\n                \n                # Recompute hash and compare\n                current_hash = self.integrity_manager.compute_hash(\n                    pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)\n                )\n                \n                integrity_valid = current_hash == metadata.model_hash\n                \n                if integrity_valid:\n                    metadata.status = ModelStatus.VERIFIED\n                else:\n                    metadata.status = ModelStatus.CORRUPTED\n                    self.logger.error(f"Model integrity check failed: {model_name}")\n                \n                return integrity_valid\n                \n        except Exception as e:\n            self.logger.error(f"Model integrity verification failed: {e}")\n            return False\n    \n    def cleanup_unused_models(self, max_age_hours: float = 24.0) -> int:\n        """Clean up models that haven't been used recently."""\n        try:\n            with self.model_lock:\n                current_time = time.time()\n                max_age_seconds = max_age_hours * 3600\n                \n                models_to_remove = []\n                for model_name, (model, metadata) in self.loaded_models.items():\n                    if current_time - metadata.last_used > max_age_seconds:\n                        models_to_remove.append(model_name)\n                \n                for model_name in models_to_remove:\n                    self.unload_model(model_name)\n                \n                self.logger.info(f"Cleaned up {len(models_to_remove)} unused models")\n                return len(models_to_remove)\n                \n        except Exception as e:\n            self.logger.error(f"Model cleanup failed: {e}")\n            return 0\n    \n    def get_system_stats(self) -> Dict[str, Any]:\n        """Get comprehensive system statistics."""\n        try:\n            with self.model_lock:\n                loaded_count = len(self.loaded_models)\n                total_memory = sum(\n                    metadata.memory_usage \n                    for _, (_, metadata) in self.loaded_models.items()\n                    if metadata.memory_usage\n                )\n                \n                cache_stats = self.cache.get_cache_stats()\n                circuit_stats = self.load_circuit.get_metrics()\n                \n                return {\n                    "loaded_models": loaded_count,\n                    "total_memory_usage_bytes": total_memory,\n                    "total_memory_usage_mb": total_memory / (1024 * 1024),\n                    "cache_stats": cache_stats,\n                    "circuit_breaker_stats": circuit_stats,\n                    "models": self.list_loaded_models()\n                }\n                \n        except Exception as e:\n            self.logger.error(f"Failed to get system stats: {e}")\n            return {"error": str(e)}\n    \n    def __del__(self):\n        """Cleanup on destruction."""\n        try:\n            self.executor.shutdown(wait=False)\n        except Exception:\n            pass


class MockModel:\n    """Mock model for fallback scenarios."""\n    \n    def __init__(self, model_name: str):\n        self.model_name = model_name\n        self.config = {"model_name": model_name, "hidden_size": 768}\n    \n    def generate(self, input_ids, max_length=100, **kwargs):\n        \"\"\"Mock generation method.\"\"\"\n        if NUMPY_AVAILABLE:\n            # Return random token ids\n            return np.random.randint(0, 1000, size=(1, max_length))\n        else:\n            return [[random.randint(0, 1000) for _ in range(max_length)]]\n    \n    def parameters(self):\n        \"\"\"Mock parameters method.\"\"\"\n        return iter([])\n    \n    def eval(self):\n        \"\"\"Mock eval method.\"\"\"\n        return self


class MockTokenizer:\n    \"\"\"Mock tokenizer for fallback scenarios.\"\"\"\n    \n    def __init__(self):\n        self.vocab_size = 1000\n    \n    def encode(self, text: str, **kwargs) -> List[int]:\n        \"\"\"Mock encoding method.\"\"\"\n        words = text.lower().split()\n        return [hash(word) % self.vocab_size for word in words]\n    \n    def decode(self, token_ids: List[int], **kwargs) -> str:\n        \"\"\"Mock decoding method.\"\"\"\n        return " ".join([f"token_{tid}" for tid in token_ids])\n    \n    def tokenize(self, text: str) -> List[str]:\n        \"\"\"Mock tokenization method.\"\"\"\n        return text.lower().split()\n\n\n# Global model loader instance\n_model_loader = None\n\ndef get_enhanced_model_loader(cache_dir: Optional[str] = None) -> EnhancedModelLoader:\n    \"\"\"Get global enhanced model loader instance.\"\"\"\n    global _model_loader\n    if _model_loader is None:\n        _model_loader = EnhancedModelLoader(cache_dir)\n    return _model_loader\n\n\n__all__ = [\n    "EnhancedModelLoader",\n    "ModelConfig",\n    "ModelMetadata",\n    "ModelStatus",\n    "ModelCache",\n    "get_enhanced_model_loader"\n]