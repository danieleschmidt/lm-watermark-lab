"""Robust Kirchenbauer watermark implementation with comprehensive error handling."""

import time
import logging
from typing import List, Optional, Dict, Any
import numpy as np

# Fallback imports
try:
    import torch
    from scipy.stats import norm
    from transformers import GenerationConfig
    HAS_DEPENDENCIES = True
except ImportError:
    torch = None
    norm = None
    GenerationConfig = None
    HAS_DEPENDENCIES = False

from .robust_base import (
    RobustBaseWatermark, DetectionResult, ValidationError, 
    ModelError, DetectionError, WatermarkStrength
)

class RobustKirchenbauerWatermark(RobustBaseWatermark):
    """
    Robust implementation of Kirchenbauer et al. watermarking with error handling.
    
    Reference: "A Watermark for Large Language Models" (2023)
    https://arxiv.org/abs/2301.10226
    """
    
    def __init__(self, model_name: str = "gpt2", gamma: float = 0.25, delta: float = 2.0, 
                 key: str = "default", seed: int = 42, strength: WatermarkStrength = WatermarkStrength.MEDIUM, **kwargs):
        """
        Initialize robust Kirchenbauer watermark.
        
        Args:
            model_name: Name of the language model
            gamma: Fraction of vocabulary in green list (0 < gamma < 1)
            delta: Strength of watermark bias
            key: Secret key for watermark
            seed: Random seed for reproducibility
            strength: Watermark strength level
        """
        # Validate specific parameters before calling super
        self._validate_kirchenbauer_params(gamma, delta, seed)
        
        super().__init__(model_name, key, **kwargs)
        
        self.gamma = gamma
        self.delta = delta
        self.seed = seed
        self.strength = strength
        
        # Adjust parameters based on strength
        self._adjust_for_strength()
        
        # Initialize RNG with validation
        try:
            self.rng = np.random.RandomState(self.hash_key(key))
        except Exception as e:
            raise ValidationError(f"Failed to initialize random state: {e}")
    
    def _validate_kirchenbauer_params(self, gamma: float, delta: float, seed: int) -> None:
        """Validate Kirchenbauer-specific parameters."""
        if not isinstance(gamma, (int, float)) or not 0 < gamma < 1:
            raise ValidationError(f"gamma must be between 0 and 1, got {gamma}")
        
        if not isinstance(delta, (int, float)) or delta <= 0:
            raise ValidationError(f"delta must be positive, got {delta}")
        
        if not isinstance(seed, int):
            raise ValidationError(f"seed must be integer, got {type(seed)}")
    
    def _adjust_for_strength(self) -> None:
        """Adjust parameters based on watermark strength."""
        if self.strength == WatermarkStrength.LIGHT:
            self.delta *= 0.7
            self.gamma *= 0.8
        elif self.strength == WatermarkStrength.STRONG:
            self.delta *= 1.5
            self.gamma = min(0.4, self.gamma * 1.2)
        
        self.logger.info(f"Adjusted for {self.strength.value}: gamma={self.gamma:.3f}, delta={self.delta:.3f}")
    
    def _get_greenlist_ids(self, context_tokens) -> List[int]:
        """Get green list token IDs with error handling."""
        try:
            # Handle different input types
            if HAS_DEPENDENCIES and torch is not None and hasattr(context_tokens, 'shape'):
                # PyTorch tensor
                context_list = context_tokens.tolist() if context_tokens.dim() > 0 else []
            elif isinstance(context_tokens, (list, tuple)):
                context_list = list(context_tokens)
            else:
                context_list = []
            
            # Get vocabulary size safely
            vocab_size = getattr(self.tokenizer, 'vocab_size', 50257)  # Default to GPT2 size
            
            if len(context_list) == 0:
                seed = self.seed
            else:
                seed = context_list[-1] if isinstance(context_list[-1], int) else self.seed
            
            # Create reproducible green list
            if HAS_DEPENDENCIES and torch is not None:
                generator = torch.Generator()
                generator.manual_seed(seed)
                greenlist_size = int(vocab_size * self.gamma)
                greenlist_ids = torch.randperm(vocab_size, generator=generator)[:greenlist_size]
                return greenlist_ids.tolist()
            else:
                # Fallback using numpy
                local_rng = np.random.RandomState(seed)
                greenlist_size = int(vocab_size * self.gamma)
                greenlist_ids = local_rng.choice(vocab_size, greenlist_size, replace=False)
                return greenlist_ids.tolist()
                
        except Exception as e:
            self.logger.error(f"Error generating green list: {e}")
            raise DetectionError(f"Failed to generate green list: {e}")
    
    def _apply_watermark_bias(self, logits, context) -> Any:
        """Apply watermark bias with error handling."""
        if not HAS_DEPENDENCIES or torch is None:
            raise ModelError("PyTorch required for watermark bias application")
        
        try:
            greenlist_ids = self._get_greenlist_ids(context)
            
            # Create bias mask
            bias_mask = torch.zeros_like(logits)
            if len(greenlist_ids) > 0:
                bias_mask[greenlist_ids] = self.delta
            
            return logits + bias_mask
            
        except Exception as e:
            self.logger.error(f"Error applying watermark bias: {e}")
            raise ModelError(f"Failed to apply watermark bias: {e}")
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 1.0, 
                 top_p: float = 0.9, **kwargs) -> str:
        """Generate watermarked text with comprehensive error handling."""
        start_time = time.time()
        
        try:
            # Validate inputs
            prompt = self._validate_text_input(prompt, min_length=1, max_length=10000)
            self._validate_generation_params(max_length, temperature)
            
            if not HAS_DEPENDENCIES:
                raise ModelError("PyTorch and transformers required for text generation")
            
            # Validate additional parameters
            if not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
                raise ValidationError(f"top_p must be between 0 and 1, got {top_p}")
            
            self.logger.info(f"Generating text: prompt_len={len(prompt)}, max_length={max_length}")
            
            # Tokenize prompt
            try:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                if torch.cuda.is_available() and hasattr(input_ids, 'cuda'):
                    input_ids = input_ids.cuda()
            except Exception as e:
                raise ValidationError(f"Failed to tokenize prompt: {e}")
            
            # Safety checks
            if len(input_ids[0]) + max_length > 2048:  # Common context limit
                self.logger.warning("Generation may exceed context limit")
            
            # Generation with watermark
            output_ids = input_ids.clone()
            generation_time = time.time()
            
            for step in range(max_length):
                try:
                    # Get logits for next token
                    with torch.no_grad():
                        outputs = self.model(output_ids)
                        logits = outputs.logits[0, -1, :]
                        
                        # Apply watermark bias
                        watermarked_logits = self._apply_watermark_bias(logits, output_ids[0])
                        
                        # Apply temperature
                        if temperature != 1.0:
                            watermarked_logits = watermarked_logits / temperature
                        
                        # Convert to probabilities
                        probs = torch.softmax(watermarked_logits, dim=-1)
                        
                        # Apply top-p filtering if needed
                        if top_p < 1.0:
                            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                            mask = cumulative_probs > top_p
                            if mask.sum() < len(mask):  # Ensure at least one token
                                mask[0] = False
                            sorted_probs[mask] = 0
                            probs = torch.zeros_like(probs)
                            probs.scatter_(-1, sorted_indices, sorted_probs)
                        
                        # Sample next token
                        if probs.sum() == 0:
                            # Fallback if all probabilities are zero
                            probs = torch.softmax(logits, dim=-1)
                        
                        next_token = torch.multinomial(probs, 1)
                        output_ids = torch.cat([output_ids, next_token.unsqueeze(0)], dim=-1)
                        
                        # Stop conditions
                        if next_token.item() == self.tokenizer.eos_token_id:
                            break
                        
                        # Prevent infinite loops
                        if time.time() - generation_time > 60:  # 60 second timeout
                            self.logger.warning("Generation timeout, stopping early")
                            break
                            
                except Exception as e:
                    self.logger.error(f"Error at generation step {step}: {e}")
                    if step == 0:
                        raise ModelError(f"Generation failed immediately: {e}")
                    else:
                        self.logger.warning(f"Stopping generation early due to error: {e}")
                        break
            
            # Decode output
            try:
                output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                # Return only the generated part
                generated_text = output_text[len(prompt):]
                
                self.logger.info(f"Generation completed in {time.time() - start_time:.2f}s")
                return generated_text
                
            except Exception as e:
                raise ModelError(f"Failed to decode generated text: {e}")
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise
    
    def detect(self, text: str) -> DetectionResult:
        """Detect watermark with comprehensive error handling."""
        start_time = time.time()
        warnings = []
        
        try:
            # Validate input
            text = self._validate_text_input(text, min_length=5, max_length=100000)
            
            self.logger.info(f"Detecting watermark in text of length {len(text)}")
            
            # Tokenize text
            try:
                if HAS_DEPENDENCIES and torch is not None:
                    tokens = self.tokenizer.encode(text)
                else:
                    # Fallback tokenization
                    tokens = list(range(len(text.split())))  # Simple word-based fallback
                    warnings.append("Using fallback tokenization - results may be inaccurate")
            except Exception as e:
                raise DetectionError(f"Failed to tokenize text: {e}")
            
            if len(tokens) < 10:
                warnings.append("Text too short for reliable detection")
                return DetectionResult(
                    is_watermarked=False,
                    confidence=0.0,
                    p_value=1.0,
                    test_statistic=0.0,
                    method="kirchenbauer",
                    processing_time=time.time() - start_time,
                    model_used=self.model_name,
                    warning_messages=warnings
                )
            
            # Count green tokens
            green_count = 0
            token_scores = []
            
            for i in range(1, len(tokens)):
                try:
                    context = tokens[:i]
                    current_token = tokens[i]
                    
                    # Get green list for this context
                    greenlist_ids = self._get_greenlist_ids(context)
                    
                    # Check if current token is in green list
                    is_green = current_token in greenlist_ids
                    if is_green:
                        green_count += 1
                    
                    token_scores.append(1.0 if is_green else 0.0)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing token {i}: {e}")
                    token_scores.append(0.0)
            
            # Statistical test
            n_tokens = len(tokens) - 1
            if n_tokens == 0:
                warnings.append("No tokens to analyze")
                return DetectionResult(
                    is_watermarked=False,
                    confidence=0.0,
                    p_value=1.0,
                    test_statistic=0.0,
                    method="kirchenbauer",
                    processing_time=time.time() - start_time,
                    model_used=self.model_name,
                    warning_messages=warnings
                )
            
            expected_green = n_tokens * self.gamma
            variance = n_tokens * self.gamma * (1 - self.gamma)
            
            if variance == 0:
                test_statistic = 0.0
                p_value = 1.0
                warnings.append("Zero variance in statistical test")
            else:
                # Z-test for proportion
                test_statistic = (green_count - expected_green) / np.sqrt(variance)
                
                # One-tailed test (expecting more green tokens if watermarked)
                if norm is not None:
                    p_value = 1 - norm.cdf(test_statistic)
                else:
                    # Fallback normal approximation
                    from math import exp, sqrt, pi
                    p_value = 0.5 * (1 - test_statistic / sqrt(2 * pi))
                    p_value = max(0, min(1, p_value))
                    warnings.append("Using fallback statistical computation")
            
            # Determine result
            confidence = max(0, min(1, 1 - p_value))
            significance_level = 0.01
            is_watermarked = p_value < significance_level
            
            # Additional validation
            if green_count / n_tokens > 0.8:
                warnings.append("Unusually high green token ratio - check parameters")
            
            self.logger.info(f"Detection completed: green_ratio={green_count/n_tokens:.3f}, p_value={p_value:.6f}")
            
            return DetectionResult(
                is_watermarked=is_watermarked,
                confidence=confidence,
                p_value=p_value,
                test_statistic=test_statistic,
                token_scores=token_scores,
                method="kirchenbauer",
                processing_time=time.time() - start_time,
                model_used=self.model_name,
                warning_messages=warnings
            )
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            raise DetectionError(f"Watermark detection failed: {e}")
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration."""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "parameters": {
                "gamma": self.gamma,
                "delta": self.delta,
                "seed": self.seed,
                "strength": self.strength.value
            }
        }
        
        # Parameter validation
        if self.gamma <= 0.1:
            validation["warnings"].append("Very low gamma may reduce detectability")
        if self.gamma >= 0.7:
            validation["warnings"].append("Very high gamma may impact text quality")
        
        if self.delta < 0.5:
            validation["warnings"].append("Low delta may reduce watermark strength")
        if self.delta > 5.0:
            validation["warnings"].append("High delta may degrade text quality")
        
        # Environment validation
        if not HAS_DEPENDENCIES:
            validation["errors"].append("Missing required dependencies (torch, transformers, scipy)")
            validation["valid"] = False
        
        return validation