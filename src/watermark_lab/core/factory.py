"""Watermark factory for creating different watermarking implementations."""

import math
import random
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict


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
        """Generate watermarked text using Kirchenbauer method with actual statistical watermarking."""
        max_length = kwargs.get('max_length', 100)
        seed = kwargs.get('seed', self.config.get('seed', 42))
        vocab_size = kwargs.get('vocab_size', 1000)  # Simplified vocab
        
        # Initialize random generator with seed for reproducibility
        rng = np.random.RandomState(seed)
        
        # Simulate tokenization - in production would use actual tokenizer
        tokens = self._simple_tokenize(prompt)
        generated_tokens = []
        
        # Generate tokens with watermarked distribution
        for i in range(max_length // 5):  # Generate fewer tokens for demo
            # Create context-dependent seed
            context_seed = self._hash_context(tokens[-4:] if len(tokens) >= 4 else tokens)
            context_rng = np.random.RandomState((seed + context_seed) % (2**32))
            
            # Partition vocabulary into green and red lists
            green_list_size = int(vocab_size * self.gamma)
            green_list = set(context_rng.permutation(vocab_size)[:green_list_size])
            
            # Sample next token with bias toward green list
            # Simulate logits - in production would come from actual model
            logits = np.random.normal(0, 1, vocab_size)
            
            # Apply green list bias
            for token_id in green_list:
                logits[token_id] += self.delta
            
            # Sample from modified distribution
            probs = self._softmax(logits)
            next_token_id = np.random.choice(vocab_size, p=probs)
            
            # Convert back to word (simplified)
            next_word = self._id_to_word(next_token_id)
            generated_tokens.append(next_word)
            tokens.append(next_word)
        
        watermarked_text = prompt + " " + " ".join(generated_tokens)
        return watermarked_text.strip()
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization for demonstration."""
        return text.lower().split()
    
    def _hash_context(self, context: List[str]) -> int:
        """Hash context tokens to create reproducible seed."""
        context_str = "||".join(context)
        return int(hashlib.md5(context_str.encode()).hexdigest()[:8], 16)
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        return exp_logits / np.sum(exp_logits)
    
    def _id_to_word(self, token_id: int) -> str:
        """Convert token ID to word (simplified vocabulary)."""
        # Simple vocabulary for demonstration
        vocab = [
            "and", "the", "to", "of", "a", "in", "is", "it", "you", "that",
            "he", "was", "for", "on", "are", "as", "with", "his", "they", "be",
            "at", "one", "have", "this", "from", "or", "had", "by", "not", "word",
            "but", "what", "some", "we", "can", "out", "other", "were", "all", "your",
            "when", "up", "use", "each", "which", "she", "do", "how", "their", "if",
            "will", "way", "about", "many", "then", "them", "would", "write", "like", "so",
            "these", "her", "long", "make", "thing", "see", "him", "two", "has", "look",
            "more", "day", "could", "go", "come", "did", "my", "sound", "no", "most",
            "people", "over", "know", "water", "than", "call", "first", "who", "may", "down",
            "side", "been", "now", "find", "any", "new", "work", "part", "take", "get",
            "place", "made", "live", "where", "after", "back", "little", "only", "round", "man"
        ] * 10  # Extend vocabulary
        return vocab[token_id % len(vocab)]
    
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
        """Generate watermarked text using MarkLLM method with key-based watermarking."""
        max_length = kwargs.get('max_length', 100)
        secret_key = kwargs.get('key', self.config.get('key', 'default_key'))
        
        # Tokenize input
        tokens = self._simple_tokenize(prompt)
        generated_tokens = []
        
        # Key-based watermarking implementation
        if self.algorithm == "KGW":  # Key-based Grouped Watermarking
            generated_tokens = self._generate_kgw(tokens, max_length, secret_key)
        elif self.algorithm == "SWEET":  # Synchronous Watermarking with Embedding Token
            generated_tokens = self._generate_sweet(tokens, max_length, secret_key)
        else:
            generated_tokens = self._generate_basic(tokens, max_length)
        
        watermarked_text = prompt + " " + " ".join(generated_tokens)
        return watermarked_text.strip()
    
    def _generate_kgw(self, context: List[str], max_length: int, key: str) -> List[str]:
        """Key-based Grouped Watermarking implementation."""
        tokens = []
        vocab_groups = self._create_vocab_groups(key)
        
        for i in range(max_length // 8):
            # Hash context with key to determine group
            context_hash = self._hash_with_key(context[-3:] if len(context) >= 3 else context, key, i)
            group_id = context_hash % len(vocab_groups)
            
            # Select from specific group with watermark strength bias
            group = vocab_groups[group_id]
            if random.random() < (self.watermark_strength / 5.0):  # Apply strength
                # Choose predictable token from group
                token_idx = context_hash % len(group)
                token = group[token_idx]
            else:
                # Choose random token from any group
                all_tokens = [token for group in vocab_groups for token in group]
                token = random.choice(all_tokens)
            
            tokens.append(token)
            context.append(token)
        
        return tokens
    
    def _generate_sweet(self, context: List[str], max_length: int, key: str) -> List[str]:
        """Synchronous Watermarking with Embedding Token implementation."""
        tokens = []
        embedding_tokens = self._get_embedding_tokens(key)
        
        for i in range(max_length // 8):
            if i % 3 == 0:  # Insert embedding token synchronously
                # Select embedding token based on context and key
                context_hash = self._hash_with_key(context[-2:] if len(context) >= 2 else context, key, i)
                embedding_token = embedding_tokens[context_hash % len(embedding_tokens)]
                tokens.append(embedding_token)
            else:
                # Generate normal token with slight bias
                normal_tokens = ["the", "and", "to", "of", "a", "in", "is", "for", "on", "with"]
                if random.random() < (self.watermark_strength / 3.0):
                    # Biased selection
                    context_hash = self._hash_with_key(context[-1:] if context else [], key, i)
                    token = normal_tokens[context_hash % len(normal_tokens)]
                else:
                    token = random.choice(normal_tokens + ["text", "data", "model", "system"])
                tokens.append(token)
            
            context.append(tokens[-1])
        
        return tokens
    
    def _generate_basic(self, context: List[str], max_length: int) -> List[str]:
        """Basic watermarking without key dependency."""
        tokens = []
        basic_vocab = ["watermark", "text", "generation", "method", "algorithm", "detection", "security"]
        
        for i in range(max_length // 10):
            if random.random() < (self.watermark_strength / 4.0):
                token = basic_vocab[i % len(basic_vocab)]
            else:
                token = f"word{i}"
            tokens.append(token)
        
        return tokens
    
    def _create_vocab_groups(self, key: str) -> List[List[str]]:
        """Create vocabulary groups based on key."""
        # Simple vocabulary partitioning
        vocab = [
            "the", "and", "to", "of", "a", "in", "is", "it", "you", "that",
            "he", "was", "for", "on", "are", "as", "with", "his", "they", "be",
            "knowledge", "generation", "watermark", "detection", "analysis", "security",
            "algorithm", "method", "system", "model", "data", "text", "content"
        ]
        
        # Hash-based grouping
        groups = [[] for _ in range(4)]  # 4 groups
        for word in vocab:
            word_hash = int(hashlib.md5((key + word).encode()).hexdigest()[:4], 16)
            group_id = word_hash % 4
            groups[group_id].append(word)
        
        return groups
    
    def _get_embedding_tokens(self, key: str) -> List[str]:
        """Get embedding tokens for SWEET algorithm."""
        base_tokens = ["sync", "embed", "token", "mark", "sign", "code", "flag", "tag"]
        # Shuffle based on key
        key_hash = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        random.seed(key_hash)
        random.shuffle(base_tokens)
        random.seed()  # Reset to avoid affecting other randomness
        return base_tokens
    
    def _hash_with_key(self, context: List[str], key: str, position: int) -> int:
        """Hash context with key and position."""
        context_str = "|".join(context) + f"#{key}#{position}"
        return int(hashlib.md5(context_str.encode()).hexdigest()[:8], 16)
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization for demonstration."""
        return text.lower().split()
    
    def get_config(self) -> Dict[str, Any]:
        """Get watermark configuration."""
        return {
            "method": "markllm",
            "algorithm": self.algorithm,
            "watermark_strength": self.watermark_strength,
            **self.config
        }


class AaronsonWatermark(BaseWatermark):
    """Aaronson watermarking implementation using cryptographic pseudorandom functions."""
    
    def __init__(self, secret_key: str = "secret", **kwargs):
        """Initialize Aaronson watermark."""
        super().__init__(secret_key=secret_key, **kwargs)
        self.secret_key = secret_key
        self.threshold = kwargs.get('threshold', 0.5)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate watermarked text using Aaronson's cryptographic method."""
        max_length = kwargs.get('max_length', 100)
        
        tokens = self._simple_tokenize(prompt)
        generated_tokens = []
        
        for i in range(max_length // 6):
            # Create cryptographic hash of context + key + position
            context = tokens[-5:] if len(tokens) >= 5 else tokens
            hash_input = "|".join(context) + f"#{self.secret_key}#{i}"
            hash_value = hashlib.sha256(hash_input.encode()).hexdigest()
            
            # Convert hash to pseudorandom value
            pseudo_random = int(hash_value[:8], 16) / (2**32)
            
            # Select token based on pseudorandom threshold
            if pseudo_random > self.threshold:
                # High-probability token
                high_prob_tokens = ["the", "and", "to", "a", "in", "of", "is", "for"]
                token_idx = int(hash_value[8:16], 16) % len(high_prob_tokens)
                token = high_prob_tokens[token_idx]
            else:
                # Lower-probability token
                low_prob_tokens = ["algorithm", "cryptographic", "pseudorandom", "watermark", "detection"]
                token_idx = int(hash_value[16:24], 16) % len(low_prob_tokens)
                token = low_prob_tokens[token_idx]
            
            generated_tokens.append(token)
            tokens.append(token)
        
        watermarked_text = prompt + " " + " ".join(generated_tokens)
        return watermarked_text.strip()
    
    def get_config(self) -> Dict[str, Any]:
        """Get watermark configuration."""
        return {
            "method": "aaronson",
            "secret_key": "[REDACTED]",  # Don't expose actual key
            "threshold": self.threshold,
            **self.config
        }
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization for demonstration."""
        return text.lower().split()


class ZhaoWatermark(BaseWatermark):
    """Zhao et al. robust multi-bit watermarking implementation."""
    
    def __init__(self, message_bits: str = "101010", redundancy: int = 3, **kwargs):
        """Initialize Zhao watermark."""
        super().__init__(message_bits=message_bits, redundancy=redundancy, **kwargs)
        self.message_bits = message_bits
        self.redundancy = redundancy  # Repetitions for robustness
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate watermarked text with embedded multi-bit message."""
        max_length = kwargs.get('max_length', 100)
        
        tokens = self._simple_tokenize(prompt)
        generated_tokens = []
        
        # Repeat message for redundancy
        extended_message = (self.message_bits * self.redundancy)[:max_length//4]
        
        for i, bit in enumerate(extended_message):
            # Embed each bit in token selection
            if bit == '1':
                # Bit 1: choose from odd-indexed vocabulary
                vocab_subset = ["one", "three", "five", "seven", "nine", "eleven", "thirteen"]
            else:
                # Bit 0: choose from even-indexed vocabulary
                vocab_subset = ["two", "four", "six", "eight", "ten", "twelve", "fourteen"]
            
            # Add some randomness while preserving bit information
            if random.random() < 0.8:  # 80% adherence to bit encoding
                token = vocab_subset[i % len(vocab_subset)]
            else:
                # Add noise tokens occasionally
                noise_tokens = ["the", "and", "to", "of", "a", "in"]
                token = random.choice(noise_tokens)
            
            generated_tokens.append(token)
            tokens.append(token)
        
        watermarked_text = prompt + " " + " ".join(generated_tokens)
        return watermarked_text.strip()
    
    def get_config(self) -> Dict[str, Any]:
        """Get watermark configuration."""
        return {
            "method": "zhao",
            "message_bits": self.message_bits,
            "redundancy": self.redundancy,
            **self.config
        }
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization for demonstration."""
        return text.lower().split()


class WatermarkFactory:
    """Factory for creating watermark instances."""
    
    _registry: Dict[str, type] = {
        "kirchenbauer": KirchenbauerWatermark,
        "markllm": MarkLLMWatermark,
        "aaronson": AaronsonWatermark,
        "zhao": ZhaoWatermark,
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