"""Watermark detection functionality with statistical methods."""

import math
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from collections import defaultdict, Counter


@dataclass
class DetectionResult:
    """Result of watermark detection."""
    is_watermarked: bool
    confidence: float
    p_value: float
    test_statistic: float
    method: str
    token_scores: Optional[List[float]] = None
    green_list_hits: Optional[int] = None
    total_tokens: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "is_watermarked": self.is_watermarked,
            "confidence": self.confidence,
            "p_value": self.p_value,
            "test_statistic": self.test_statistic,
            "method": self.method,
            "details": {
                "green_list_hits": self.green_list_hits,
                "total_tokens": self.total_tokens,
                "token_scores": self.token_scores[:10] if self.token_scores else None  # Limit for API
            }
        }


class WatermarkDetector:
    """Detects watermarks in text."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize detector with watermark configuration."""
        self.config = config
        self.method = config.get("method", "unknown")
    
    def detect(self, text: str) -> DetectionResult:
        """Detect watermark in text using method-specific statistical analysis."""
        if not text or not text.strip():
            return self._empty_result()
        
        tokens = self._simple_tokenize(text)
        
        if len(tokens) < 10:
            return self._empty_result()
        
        # Route to appropriate detection method
        method = self.config.get("method", "unknown")
        
        if method == "kirchenbauer":
            return self._detect_kirchenbauer(tokens)
        elif method == "markllm":
            return self._detect_markllm(tokens)
        elif method == "aaronson":
            return self._detect_aaronson(tokens)
        elif method == "zhao":
            return self._detect_zhao(tokens)
        else:
            return self._detect_generic(tokens)
    
    def _detect_kirchenbauer(self, tokens: List[str]) -> DetectionResult:
        """Detect Kirchenbauer watermark using statistical tests."""
        gamma = self.config.get("gamma", 0.25)
        seed = self.config.get("seed", 42)
        vocab_size = self.config.get("vocab_size", 1000)
        
        green_hits = 0
        token_scores = []
        
        # Simulate detection by checking green list membership
        for i, token in enumerate(tokens[1:], 1):  # Skip first token
            # Recreate context-dependent seed
            context = tokens[max(0, i-4):i]
            context_seed = self._hash_context(context)
            context_rng = np.random.RandomState((seed + context_seed) % (2**32))
            
            # Recreate green list
            green_list_size = int(vocab_size * gamma)
            green_list = set(context_rng.permutation(vocab_size)[:green_list_size])
            
            # Check if token is in green list (simplified)
            token_id = self._word_to_id(token)
            if token_id in green_list:
                green_hits += 1
                token_scores.append(1.0)
            else:
                token_scores.append(0.0)
        
        # Statistical test for green list hits
        n = len(tokens) - 1
        expected_hits = n * gamma
        
        if n == 0:
            return self._empty_result()
        
        # Z-test for proportion
        observed_proportion = green_hits / n
        expected_proportion = gamma
        
        # Standard error for binomial proportion
        se = math.sqrt(expected_proportion * (1 - expected_proportion) / n)
        
        if se == 0:
            z_score = 0
        else:
            z_score = (observed_proportion - expected_proportion) / se
        
        # P-value (one-tailed test)
        p_value = 1 - stats.norm.cdf(z_score)
        
        # Confidence and decision
        confidence = max(0.0, min(0.99, 1 - p_value))
        is_watermarked = p_value < 0.05  # Alpha = 0.05
        
        return DetectionResult(
            is_watermarked=is_watermarked,
            confidence=confidence,
            p_value=p_value,
            test_statistic=z_score,
            method="kirchenbauer",
            token_scores=token_scores,
            green_list_hits=green_hits,
            total_tokens=n
        )
    
    def _detect_markllm(self, tokens: List[str]) -> DetectionResult:
        """Detect MarkLLM watermark using algorithm-specific patterns."""
        algorithm = self.config.get("algorithm", "KGW")
        key = self.config.get("key", "default_key")
        
        if algorithm == "KGW":
            return self._detect_kgw_pattern(tokens, key)
        elif algorithm == "SWEET":
            return self._detect_sweet_pattern(tokens, key)
        else:
            return self._detect_generic(tokens)
    
    def _detect_kgw_pattern(self, tokens: List[str], key: str) -> DetectionResult:
        """Detect Key-based Grouped Watermarking patterns."""
        vocab_groups = self._create_vocab_groups(key)
        expected_hits = 0
        actual_hits = 0
        token_scores = []
        
        for i, token in enumerate(tokens[3:], 3):  # Need context
            context = tokens[max(0, i-3):i]
            context_hash = self._hash_with_key(context, key, i)
            expected_group = context_hash % len(vocab_groups)
            
            # Check if token is in expected group
            if token in vocab_groups[expected_group]:
                actual_hits += 1
                token_scores.append(1.0)
            else:
                token_scores.append(0.0)
            expected_hits += 1
        
        if expected_hits == 0:
            return self._empty_result()
        
        # Chi-square goodness of fit test
        observed_rate = actual_hits / expected_hits
        expected_rate = 1.0 / len(vocab_groups)  # Random chance
        
        # Simple z-test for proportion
        se = math.sqrt(expected_rate * (1 - expected_rate) / expected_hits)
        if se > 0:
            z_score = (observed_rate - expected_rate) / se
            p_value = 1 - stats.norm.cdf(z_score)
        else:
            z_score = 0
            p_value = 0.5
        
        confidence = max(0.0, min(0.99, 1 - p_value))
        is_watermarked = p_value < 0.05
        
        return DetectionResult(
            is_watermarked=is_watermarked,
            confidence=confidence,
            p_value=p_value,
            test_statistic=z_score,
            method="markllm_kgw",
            token_scores=token_scores,
            green_list_hits=actual_hits,
            total_tokens=expected_hits
        )
    
    def _detect_sweet_pattern(self, tokens: List[str], key: str) -> DetectionResult:
        """Detect SWEET watermark patterns."""
        embedding_tokens = self._get_embedding_tokens(key)
        embedding_hits = 0
        expected_positions = 0
        token_scores = []
        
        for i, token in enumerate(tokens):
            if i % 3 == 0:  # Expected embedding positions
                expected_positions += 1
                if token in embedding_tokens:
                    embedding_hits += 1
                    token_scores.append(1.0)
                else:
                    token_scores.append(0.0)
        
        if expected_positions == 0:
            return self._empty_result()
        
        # Test for embedding token frequency
        observed_rate = embedding_hits / expected_positions
        expected_rate = len(embedding_tokens) / 1000  # Rough vocabulary estimate
        
        se = math.sqrt(expected_rate * (1 - expected_rate) / expected_positions)
        if se > 0:
            z_score = (observed_rate - expected_rate) / se
            p_value = 1 - stats.norm.cdf(z_score)
        else:
            z_score = 0
            p_value = 0.5
        
        confidence = max(0.0, min(0.99, 1 - p_value))
        is_watermarked = p_value < 0.05
        
        return DetectionResult(
            is_watermarked=is_watermarked,
            confidence=confidence,
            p_value=p_value,
            test_statistic=z_score,
            method="markllm_sweet",
            token_scores=token_scores,
            green_list_hits=embedding_hits,
            total_tokens=expected_positions
        )
    
    def _detect_aaronson(self, tokens: List[str]) -> DetectionResult:
        """Detect Aaronson cryptographic watermark."""
        secret_key = self.config.get("secret_key", "secret")
        threshold = self.config.get("threshold", 0.5)
        
        high_prob_tokens = {"the", "and", "to", "a", "in", "of", "is", "for"}
        expected_highs = 0
        actual_highs = 0
        token_scores = []
        
        for i, token in enumerate(tokens[5:], 5):  # Need context
            context = tokens[max(0, i-5):i]
            hash_input = "|".join(context) + f"#{secret_key}#{i}"
            hash_value = hashlib.sha256(hash_input.encode()).hexdigest()
            pseudo_random = int(hash_value[:8], 16) / (2**32)
            
            if pseudo_random > threshold:
                expected_highs += 1
                if token in high_prob_tokens:
                    actual_highs += 1
                    token_scores.append(1.0)
                else:
                    token_scores.append(0.0)
        
        if expected_highs == 0:
            return self._empty_result()
        
        # Test correlation between pseudorandom values and token types
        observed_rate = actual_highs / expected_highs
        expected_rate = len(high_prob_tokens) / 100  # Rough estimate
        
        se = math.sqrt(expected_rate * (1 - expected_rate) / expected_highs)
        if se > 0:
            z_score = (observed_rate - expected_rate) / se
            p_value = 1 - stats.norm.cdf(z_score)
        else:
            z_score = 0
            p_value = 0.5
        
        confidence = max(0.0, min(0.99, 1 - p_value))
        is_watermarked = p_value < 0.05
        
        return DetectionResult(
            is_watermarked=is_watermarked,
            confidence=confidence,
            p_value=p_value,
            test_statistic=z_score,
            method="aaronson",
            token_scores=token_scores,
            green_list_hits=actual_highs,
            total_tokens=expected_highs
        )
    
    def _detect_zhao(self, tokens: List[str]) -> DetectionResult:
        """Detect Zhao multi-bit watermark."""
        message_bits = self.config.get("message_bits", "101010")
        redundancy = self.config.get("redundancy", 3)
        
        # Detect embedded message bits
        detected_bits = []
        token_scores = []
        
        for token in tokens:
            if token in ["one", "three", "five", "seven", "nine", "eleven", "thirteen"]:
                detected_bits.append('1')
                token_scores.append(1.0)
            elif token in ["two", "four", "six", "eight", "ten", "twelve", "fourteen"]:
                detected_bits.append('0')
                token_scores.append(1.0)
            else:
                token_scores.append(0.0)
        
        if not detected_bits:
            return self._empty_result()
        
        # Compare with expected message pattern
        detected_message = ''.join(detected_bits)
        extended_message = (message_bits * redundancy)[:len(detected_bits)]
        
        # Hamming distance
        matches = sum(1 for a, b in zip(detected_message, extended_message) if a == b)
        match_rate = matches / len(detected_message) if detected_message else 0
        
        # Statistical test
        expected_rate = 0.5  # Random chance
        n = len(detected_message)
        
        if n > 0:
            se = math.sqrt(expected_rate * (1 - expected_rate) / n)
            if se > 0:
                z_score = (match_rate - expected_rate) / se
                p_value = 1 - stats.norm.cdf(z_score)
            else:
                z_score = 0
                p_value = 0.5
        else:
            z_score = 0
            p_value = 0.5
        
        confidence = max(0.0, min(0.99, 1 - p_value))
        is_watermarked = p_value < 0.05
        
        return DetectionResult(
            is_watermarked=is_watermarked,
            confidence=confidence,
            p_value=p_value,
            test_statistic=z_score,
            method="zhao",
            token_scores=token_scores,
            green_list_hits=matches,
            total_tokens=n
        )
    
    def _detect_generic(self, tokens: List[str]) -> DetectionResult:
        """Generic pattern-based detection for unknown watermarks."""
        pattern_score = self._analyze_patterns(tokens)
        
        test_statistic = pattern_score * 2.5
        p_value = max(0.001, 1.0 - (pattern_score / 100.0))
        confidence = min(0.99, pattern_score / 100.0)
        is_watermarked = pattern_score > 50.0
        
        return DetectionResult(
            is_watermarked=is_watermarked,
            confidence=confidence,
            p_value=p_value,
            test_statistic=test_statistic,
            method="generic"
        )
    
    def _analyze_patterns(self, tokens: List[str]) -> float:
        """Analyze token patterns for watermark detection."""
        if not tokens:
            return 0.0
        
        # Multiple pattern analysis techniques
        scores = []
        
        # 1. Token length distribution analysis
        avg_length = sum(len(token) for token in tokens) / len(tokens)
        length_variance = sum((len(token) - avg_length) ** 2 for token in tokens) / len(tokens)
        if avg_length > 5.0:
            scores.append(20.0)
        if length_variance > 4.0:
            scores.append(25.0)
        
        # 2. Vocabulary diversity analysis
        unique_tokens = len(set(tokens))
        diversity_ratio = unique_tokens / len(tokens)
        if diversity_ratio < 0.8:
            scores.append(30.0)
        
        # 3. Repetition pattern analysis
        token_counts = Counter(tokens)
        most_common_freq = token_counts.most_common(1)[0][1] if token_counts else 1
        repetition_ratio = most_common_freq / len(tokens)
        if repetition_ratio > 0.15:
            scores.append(25.0)
        
        # 4. N-gram pattern analysis
        bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
        bigram_counts = Counter(bigrams)
        if bigrams and max(bigram_counts.values()) / len(bigrams) > 0.1:
            scores.append(20.0)
        
        return min(100.0, sum(scores))
    
    def _empty_result(self) -> DetectionResult:
        """Return empty detection result."""
        return DetectionResult(
            is_watermarked=False,
            confidence=0.0,
            p_value=1.0,
            test_statistic=0.0,
            method=self.method
        )
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization for demonstration."""
        return text.lower().split()
    
    def _hash_context(self, context: List[str]) -> int:
        """Hash context tokens to create reproducible seed."""
        context_str = "||".join(context)
        return int(hashlib.md5(context_str.encode()).hexdigest()[:8], 16)
    
    def _word_to_id(self, word: str) -> int:
        """Convert word to token ID (simplified)."""
        return int(hashlib.md5(word.encode()).hexdigest()[:8], 16) % 1000
    
    def _create_vocab_groups(self, key: str) -> List[List[str]]:
        """Create vocabulary groups based on key."""
        vocab = [
            "the", "and", "to", "of", "a", "in", "is", "it", "you", "that",
            "he", "was", "for", "on", "are", "as", "with", "his", "they", "be",
            "knowledge", "generation", "watermark", "detection", "analysis", "security",
            "algorithm", "method", "system", "model", "data", "text", "content"
        ]
        
        groups = [[] for _ in range(4)]
        for word in vocab:
            word_hash = int(hashlib.md5((key + word).encode()).hexdigest()[:4], 16)
            group_id = word_hash % 4
            groups[group_id].append(word)
        
        return groups
    
    def _get_embedding_tokens(self, key: str) -> List[str]:
        """Get embedding tokens for SWEET algorithm."""
        base_tokens = ["sync", "embed", "token", "mark", "sign", "code", "flag", "tag"]
        key_hash = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        np.random.seed(key_hash)
        np.random.shuffle(base_tokens)
        np.random.seed()  # Reset
        return base_tokens
    
    def _hash_with_key(self, context: List[str], key: str, position: int) -> int:
        """Hash context with key and position."""
        context_str = "|".join(context) + f"#{key}#{position}"
        return int(hashlib.md5(context_str.encode()).hexdigest()[:8], 16)
    
    def detect_batch(self, texts: list) -> list:
        """Detect watermarks in multiple texts."""
        return [self.detect(text) for text in texts]