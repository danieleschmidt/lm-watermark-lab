"""Kirchenbauer et al. watermarking method implementation."""

import torch
import numpy as np
from typing import List, Optional
from scipy.stats import norm
from transformers import GenerationConfig

from watermark_lab.methods.base import BaseWatermark, DetectionResult

class KirchenbauerWatermark(BaseWatermark):
    """
    Implementation of the Kirchenbauer et al. watermarking method.
    
    Reference: "A Watermark for Large Language Models" (2023)
    https://arxiv.org/abs/2301.10226
    """
    
    def __init__(self, model_name: str, gamma: float = 0.25, delta: float = 2.0, 
                 key: str = "default", seed: int = 42, **kwargs):
        """
        Initialize Kirchenbauer watermark.
        
        Args:
            model_name: Name of the language model
            gamma: Fraction of vocabulary in green list (0 < gamma < 1)
            delta: Strength of watermark bias
            key: Secret key for watermark
            seed: Random seed for reproducibility
        """
        super().__init__(model_name, key, **kwargs)
        self.gamma = gamma
        self.delta = delta
        self.seed = seed
        self.rng = np.random.RandomState(self.hash_key(key))
        
    def _get_greenlist_ids(self, context_tokens: torch.Tensor) -> torch.Tensor:
        """Get green list token IDs based on context."""
        vocab_size = self.tokenizer.vocab_size
        if len(context_tokens) == 0:
            # Use seed for initial token
            seed = self.seed
        else:
            # Use last token as seed
            seed = context_tokens[-1].item()
            
        # Create reproducible green list
        generator = torch.Generator()
        generator.manual_seed(seed)
        greenlist_size = int(vocab_size * self.gamma)
        
        # Sample without replacement
        greenlist_ids = torch.randperm(vocab_size, generator=generator)[:greenlist_size]
        return greenlist_ids
    
    def _apply_watermark_bias(self, logits: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply watermark bias to logits."""
        greenlist_ids = self._get_greenlist_ids(context)
        
        # Create bias mask
        bias_mask = torch.zeros_like(logits)
        bias_mask[greenlist_ids] = self.delta
        
        return logits + bias_mask
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 1.0, 
                 top_p: float = 0.9, **kwargs) -> str:
        """Generate watermarked text."""
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            
        # Generation parameters
        generation_config = GenerationConfig(
            max_length=len(input_ids[0]) + max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        # Generate with watermark
        output_ids = input_ids.clone()
        
        for _ in range(max_length):
            # Get logits for next token
            with torch.no_grad():
                outputs = self.model(output_ids)
                logits = outputs.logits[0, -1, :]  # Last token logits
                
                # Apply watermark bias
                watermarked_logits = self._apply_watermark_bias(logits, output_ids[0])
                
                # Sample next token
                probs = torch.softmax(watermarked_logits / temperature, dim=-1)
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumulative_probs > top_p
                    mask[0] = False  # Keep at least one token
                    sorted_probs[mask] = 0
                    probs = torch.zeros_like(probs)
                    probs.scatter_(-1, sorted_indices, sorted_probs)
                
                # Sample
                next_token = torch.multinomial(probs, 1)
                output_ids = torch.cat([output_ids, next_token.unsqueeze(0)], dim=-1)
                
                # Stop if EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode output
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Return only the generated part
        return output_text[len(prompt):]
    
    def detect(self, text: str) -> DetectionResult:
        """Detect watermark in text using statistical test."""
        # Tokenize text
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) < 10:  # Need minimum tokens for reliable detection
            return DetectionResult(
                is_watermarked=False,
                confidence=0.0,
                p_value=1.0,
                test_statistic=0.0
            )
        
        # Count green tokens
        green_count = 0
        token_scores = []
        
        for i in range(1, len(tokens)):  # Skip first token (no context)
            context = torch.tensor(tokens[:i])
            current_token = tokens[i]
            
            # Get green list for this context
            greenlist_ids = self._get_greenlist_ids(context)
            
            # Check if current token is in green list
            is_green = current_token in greenlist_ids
            if is_green:
                green_count += 1
            
            token_scores.append(1.0 if is_green else 0.0)
        
        # Statistical test
        n_tokens = len(tokens) - 1
        expected_green = n_tokens * self.gamma
        variance = n_tokens * self.gamma * (1 - self.gamma)
        
        if variance == 0:
            test_statistic = 0.0
            p_value = 1.0
        else:
            # Z-test for proportion
            test_statistic = (green_count - expected_green) / np.sqrt(variance)
            # One-tailed test (we expect more green tokens if watermarked)
            p_value = 1 - norm.cdf(test_statistic)
        
        # Confidence based on p-value
        confidence = 1 - p_value
        is_watermarked = p_value < 0.01  # Significance level
        
        return DetectionResult(
            is_watermarked=is_watermarked,
            confidence=confidence,
            p_value=p_value,
            test_statistic=test_statistic,
            token_scores=token_scores
        )