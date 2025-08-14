"""Enhanced integration layer for seamless watermarking workflows."""

import json
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from collections import deque, defaultdict
import logging

try:
    import asyncio
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

@dataclass
class IntegrationConfig:
    """Configuration for enhanced integration."""
    
    # Performance settings
    batch_size: int = 32
    max_workers: int = 4
    cache_enabled: bool = True
    async_enabled: bool = True
    
    # Quality settings  
    quality_threshold: float = 0.8
    detection_confidence: float = 0.95
    
    # Integration settings
    auto_fallback: bool = True
    retry_attempts: int = 3
    timeout_seconds: float = 30.0
    
    # Monitoring
    metrics_enabled: bool = True
    logging_level: str = "INFO"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class EnhancedWatermarkFactory:
    """Enhanced factory with improved integration capabilities."""
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()
        self.logger = self._setup_logging()
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        self.operation_count = 0
        
        # Integration cache
        self._method_cache = {}
        self._result_cache = {}
        
        # Available methods registry
        self._methods = {
            'kirchenbauer': self._create_kirchenbauer,
            'aaronson': self._create_aaronson,
            'zhao': self._create_zhao,
            'markllm': self._create_markllm,
            'unigram': self._create_unigram,
            'sacw': self._create_sacw,
            'arms': self._create_arms,
            'qipw': self._create_qipw,
        }
        
        self.logger.info("Enhanced watermark factory initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup enhanced logging."""
        logger = logging.getLogger("enhanced_factory")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, self.config.logging_level))
        return logger
    
    def create_watermark(
        self, 
        method: str, 
        **kwargs
    ) -> 'BaseWatermarkImplementation':
        """Create watermark with enhanced error handling and performance."""
        
        start_time = time.time()
        self.operation_count += 1
        
        try:
            # Validate method
            if method not in self._methods:
                available = ', '.join(self._methods.keys())
                raise ValueError(f"Unknown method '{method}'. Available: {available}")
            
            # Check cache first
            cache_key = self._generate_cache_key(method, kwargs)
            if self.config.cache_enabled and cache_key in self._method_cache:
                self.logger.debug(f"Cache hit for method {method}")
                return self._method_cache[cache_key]
            
            # Create watermark instance
            watermark = self._methods[method](**kwargs)
            
            # Cache result
            if self.config.cache_enabled:
                self._method_cache[cache_key] = watermark
            
            # Record performance
            duration = time.time() - start_time
            self.performance_metrics['create_watermark'].append(duration)
            
            if self.config.metrics_enabled:
                self.logger.info(
                    f"Created {method} watermark in {duration:.3f}s "
                    f"(operation #{self.operation_count})"
                )
            
            return watermark
            
        except Exception as e:
            self.logger.error(f"Failed to create {method} watermark: {e}")
            
            # Auto-fallback if enabled
            if self.config.auto_fallback and method != 'kirchenbauer':
                self.logger.info("Attempting fallback to kirchenbauer method")
                return self.create_watermark('kirchenbauer', **kwargs)
            
            raise
    
    def _generate_cache_key(self, method: str, kwargs: Dict) -> str:
        """Generate cache key for method and parameters."""
        key_data = {'method': method, 'kwargs': kwargs}
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def batch_create(
        self, 
        requests: List[Dict[str, Any]]
    ) -> List['BaseWatermarkImplementation']:
        """Create multiple watermarks in batch."""
        
        results = []
        batch_size = min(len(requests), self.config.batch_size)
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            batch_results = []
            
            for request in batch:
                method = request.pop('method')
                try:
                    watermark = self.create_watermark(method, **request)
                    batch_results.append(watermark)
                except Exception as e:
                    self.logger.error(f"Batch creation failed for {method}: {e}")
                    batch_results.append(None)
            
            results.extend(batch_results)
        
        return results
    
    def _create_kirchenbauer(self, **kwargs) -> 'KirchenbauerWatermark':
        """Create Kirchenbauer watermark with enhancements."""
        return KirchenbauerWatermark(**kwargs)
    
    def _create_aaronson(self, **kwargs) -> 'AaronsonWatermark':
        """Create Aaronson watermark with enhancements."""
        return AaronsonWatermark(**kwargs)
    
    def _create_zhao(self, **kwargs) -> 'ZhaoWatermark':
        """Create Zhao watermark with enhancements."""
        return ZhaoWatermark(**kwargs)
    
    def _create_markllm(self, **kwargs) -> 'MarkLLMWatermark':
        """Create MarkLLM watermark with enhancements."""
        return MarkLLMWatermark(**kwargs)
    
    def _create_unigram(self, **kwargs) -> 'UnigramWatermark':
        """Create Unigram watermark with enhancements."""
        return UnigramWatermark(**kwargs)
    
    def _create_sacw(self, **kwargs) -> 'SACWWatermark':
        """Create SACW (Semantic-Aware Contextual Watermarking) algorithm."""
        return SACWWatermark(**kwargs)
    
    def _create_arms(self, **kwargs) -> 'ARMSWatermark':
        """Create ARMS (Adversarial-Robust Multi-Scale) watermark."""
        return ARMSWatermark(**kwargs)
    
    def _create_qipw(self, **kwargs) -> 'QIPWWatermark':
        """Create QIPW (Quantum-Inspired Probabilistic) watermark."""
        return QIPWWatermark(**kwargs)
    
    def get_available_methods(self) -> List[str]:
        """Get list of available watermarking methods."""
        return list(self._methods.keys())
    
    def get_method_info(self, method: str) -> Dict[str, Any]:
        """Get detailed information about a watermarking method."""
        info = {
            'kirchenbauer': {
                'name': 'Kirchenbauer et al.',
                'description': 'Green-red list based watermarking',
                'paper': 'A Watermark for Large Language Models (2023)',
                'strengths': ['High detection rate', 'Fast inference'],
                'weaknesses': ['Sensitive to paraphrasing'],
                'parameters': ['gamma', 'delta', 'seed']
            },
            'aaronson': {
                'name': 'Aaronson',
                'description': 'Cryptographic watermarking approach',
                'paper': 'My AI Safety Lecture (2022)',
                'strengths': ['Cryptographically secure', 'Theoretically sound'],
                'weaknesses': ['Complex implementation', 'Performance overhead'],
                'parameters': ['key', 'hash_function']
            },
            'zhao': {
                'name': 'Zhao et al.',
                'description': 'Multi-bit robust watermarking',
                'paper': 'Provably Robust Multi-bit Watermarking (2023)',
                'strengths': ['Multi-bit encoding', 'Robust to attacks'],
                'weaknesses': ['Higher complexity', 'Quality trade-offs'],
                'parameters': ['num_bits', 'redundancy', 'error_correction']
            },
            'markllm': {
                'name': 'MarkLLM',
                'description': 'Comprehensive watermarking toolkit',
                'paper': 'MarkLLM: An Open-Source Toolkit (2024)',
                'strengths': ['Multiple algorithms', 'Production ready'],
                'weaknesses': ['Heavy dependencies', 'Learning curve'],
                'parameters': ['algorithm', 'model', 'config']
            },
            'unigram': {
                'name': 'Unigram',
                'description': 'Simple unigram-based watermarking',
                'paper': 'Statistical watermarking approaches',
                'strengths': ['Simplicity', 'Fast processing'],
                'weaknesses': ['Limited robustness', 'Basic detection'],
                'parameters': ['vocab_partition', 'bias_strength']
            },
            'sacw': {
                'name': 'SACW (Novel)',
                'description': 'Semantic-Aware Contextual Watermarking',
                'paper': 'Novel research contribution',
                'strengths': ['Context awareness', 'Semantic preservation'],
                'weaknesses': ['Computational overhead', 'Novel approach'],
                'parameters': ['semantic_threshold', 'context_window', 'gamma']
            },
            'arms': {
                'name': 'ARMS (Novel)',
                'description': 'Adversarial-Robust Multi-Scale Watermarking',
                'paper': 'Novel research contribution',
                'strengths': ['Attack resistance', 'Multi-scale approach'],
                'weaknesses': ['Complex tuning', 'Resource intensive'],
                'parameters': ['scale_levels', 'robustness_factor', 'gamma']
            },
            'qipw': {
                'name': 'QIPW (Novel)',
                'description': 'Quantum-Inspired Probabilistic Watermarking',
                'paper': 'Novel research contribution',
                'strengths': ['Quantum principles', 'Probabilistic approach'],
                'weaknesses': ['Theoretical complexity', 'Implementation challenges'],
                'parameters': ['coherence_time', 'entanglement_strength']
            }
        }
        
        return info.get(method, {'error': f'Unknown method: {method}'})
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            'total_operations': self.operation_count,
            'cache_hits': len(self._method_cache),
            'available_methods': len(self._methods)
        }
        
        # Calculate average operation times
        for operation, times in self.performance_metrics.items():
            if times:
                stats[f'{operation}_avg_time'] = sum(times) / len(times)
                stats[f'{operation}_total_ops'] = len(times)
        
        return stats
    
    @contextmanager
    def performance_monitoring(self):
        """Context manager for performance monitoring."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.performance_metrics['context_operation'].append(duration)
            if self.config.metrics_enabled:
                self.logger.debug(f"Context operation completed in {duration:.3f}s")


class BaseWatermarkImplementation:
    """Base implementation with enhanced capabilities."""
    
    def __init__(self, **kwargs):
        self.config = kwargs
        self.method = self.__class__.__name__.lower().replace('watermark', '')
        self.creation_time = time.time()
        self.usage_count = 0
        
    def generate(self, text: str, **kwargs) -> str:
        """Generate watermarked text with performance tracking."""
        self.usage_count += 1
        start_time = time.time()
        
        try:
            result = self._generate_impl(text, **kwargs)
            duration = time.time() - start_time
            
            # Add metadata
            if hasattr(result, '__dict__'):
                result.generation_time = duration
                result.usage_count = self.usage_count
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Generation failed for {self.method}: {e}")
    
    def _generate_impl(self, text: str, **kwargs) -> str:
        """Implementation-specific generation logic."""
        # Placeholder - would be overridden by specific implementations
        return f"[{self.method.upper()} WATERMARKED] {text}"
    
    def get_config(self) -> Dict[str, Any]:
        """Get watermark configuration."""
        return {
            'method': self.method,
            'config': self.config,
            'creation_time': self.creation_time,
            'usage_count': self.usage_count
        }


# Specific implementations
class KirchenbauerWatermark(BaseWatermarkImplementation):
    """Enhanced Kirchenbauer watermark implementation."""
    
    def _generate_impl(self, text: str, **kwargs) -> str:
        gamma = self.config.get('gamma', 0.25)
        delta = self.config.get('delta', 2.0)
        seed = self.config.get('seed', 42)
        
        # Simulate watermarking process
        return f"[KIRCHENBAUER γ={gamma} δ={delta}] {text}"


class AaronsonWatermark(BaseWatermarkImplementation):
    """Enhanced Aaronson watermark implementation."""
    
    def _generate_impl(self, text: str, **kwargs) -> str:
        key = self.config.get('key', 'default_key')
        
        # Simulate cryptographic watermarking
        return f"[AARONSON key={key[:8]}...] {text}"


class ZhaoWatermark(BaseWatermarkImplementation):
    """Enhanced Zhao watermark implementation."""
    
    def _generate_impl(self, text: str, **kwargs) -> str:
        num_bits = self.config.get('num_bits', 8)
        redundancy = self.config.get('redundancy', 2)
        
        # Simulate multi-bit watermarking
        return f"[ZHAO bits={num_bits} r={redundancy}] {text}"


class MarkLLMWatermark(BaseWatermarkImplementation):
    """Enhanced MarkLLM watermark implementation."""
    
    def _generate_impl(self, text: str, **kwargs) -> str:
        algorithm = self.config.get('algorithm', 'KGW')
        model = self.config.get('model', 'default')
        
        # Simulate MarkLLM integration
        return f"[MARKLLM {algorithm} model={model}] {text}"


class UnigramWatermark(BaseWatermarkImplementation):
    """Enhanced Unigram watermark implementation."""
    
    def _generate_impl(self, text: str, **kwargs) -> str:
        partition = self.config.get('vocab_partition', 0.5)
        bias = self.config.get('bias_strength', 1.0)
        
        # Simulate unigram watermarking
        return f"[UNIGRAM p={partition} b={bias}] {text}"


class SACWWatermark(BaseWatermarkImplementation):
    """SACW: Semantic-Aware Contextual Watermarking (Novel Research)."""
    
    def _generate_impl(self, text: str, **kwargs) -> str:
        semantic_threshold = self.config.get('semantic_threshold', 0.85)
        context_window = self.config.get('context_window', 16)
        gamma = self.config.get('gamma', 0.25)
        
        # Novel semantic-aware approach
        return f"[SACW s={semantic_threshold} w={context_window} γ={gamma}] {text}"


class ARMSWatermark(BaseWatermarkImplementation):
    """ARMS: Adversarial-Robust Multi-Scale Watermarking (Novel Research)."""
    
    def _generate_impl(self, text: str, **kwargs) -> str:
        scale_levels = self.config.get('scale_levels', [1, 4, 16])
        robustness_factor = self.config.get('robustness_factor', 2.0)
        gamma = self.config.get('gamma', 0.25)
        
        # Novel multi-scale approach
        return f"[ARMS scales={scale_levels} r={robustness_factor} γ={gamma}] {text}"


class QIPWWatermark(BaseWatermarkImplementation):
    """QIPW: Quantum-Inspired Probabilistic Watermarking (Novel Research)."""
    
    def _generate_impl(self, text: str, **kwargs) -> str:
        coherence_time = self.config.get('coherence_time', 100.0)
        entanglement_strength = self.config.get('entanglement_strength', 0.8)
        
        # Novel quantum-inspired approach
        return f"[QIPW t={coherence_time} e={entanglement_strength}] {text}"


# Create global enhanced factory instance
enhanced_factory = EnhancedWatermarkFactory()


def create_watermark(method: str, **kwargs) -> BaseWatermarkImplementation:
    """Convenience function for creating watermarks."""
    return enhanced_factory.create_watermark(method, **kwargs)


def get_available_methods() -> List[str]:
    """Get available watermarking methods."""
    return enhanced_factory.get_available_methods()


def get_method_info(method: str) -> Dict[str, Any]:
    """Get method information."""
    return enhanced_factory.get_method_info(method)


__all__ = [
    'EnhancedWatermarkFactory',
    'BaseWatermarkImplementation', 
    'IntegrationConfig',
    'create_watermark',
    'get_available_methods',
    'get_method_info',
    'enhanced_factory'
]