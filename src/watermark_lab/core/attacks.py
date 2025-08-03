"""Attack simulation framework for testing watermark robustness."""

import random
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import Counter


@dataclass
class AttackResult:
    """Result of an attack on watermarked text."""
    original_text: str
    attacked_text: str
    attack_type: str
    success: bool
    quality_score: float
    similarity_score: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_text": self.original_text,
            "attacked_text": self.attacked_text,
            "attack_type": self.attack_type,
            "success": self.success,
            "quality_score": self.quality_score,
            "similarity_score": self.similarity_score,
            "metadata": self.metadata
        }


class BaseAttack(ABC):
    """Base class for all attack implementations."""
    
    def __init__(self, strength: str = "medium", **kwargs):
        """Initialize attack with configuration."""
        self.strength = strength
        self.config = kwargs
        self.name = self.__class__.__name__.lower().replace('attack', '')
    
    @abstractmethod
    def attack(self, text: str) -> str:
        """Apply attack to text."""
        pass
    
    def evaluate_attack(self, original: str, attacked: str) -> Tuple[float, float]:
        """Evaluate attack quality and similarity."""
        quality_score = self._calculate_quality_preservation(original, attacked)
        similarity_score = self._calculate_similarity(original, attacked)
        return quality_score, similarity_score
    
    def _calculate_quality_preservation(self, original: str, attacked: str) -> float:
        """Calculate how well the attack preserves text quality."""
        if not original or not attacked:
            return 0.0
        
        # Simple quality metrics
        orig_words = original.split()
        attacked_words = attacked.split()
        
        # Length preservation
        length_ratio = min(len(attacked_words), len(orig_words)) / max(len(attacked_words), len(orig_words), 1)
        
        # Vocabulary preservation
        orig_vocab = set(w.lower() for w in orig_words)
        attacked_vocab = set(w.lower() for w in attacked_words)
        vocab_overlap = len(orig_vocab & attacked_vocab) / len(orig_vocab | attacked_vocab) if orig_vocab | attacked_vocab else 0
        
        return (length_ratio + vocab_overlap) / 2
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union


class ParaphraseAttack(BaseAttack):
    """Paraphrasing attack using various strategies."""
    
    def __init__(self, method: str = "synonym", strength: str = "medium", **kwargs):
        """Initialize paraphrase attack."""
        super().__init__(strength, **kwargs)
        self.method = method
        self.synonym_dict = self._build_synonym_dict()
    
    def attack(self, text: str) -> str:
        """Apply paraphrasing attack."""
        if self.method == "synonym":
            return self._synonym_replacement(text)
        elif self.method == "reorder":
            return self._sentence_reordering(text)
        elif self.method == "expansion":
            return self._text_expansion(text)
        else:
            return self._combined_paraphrase(text)
    
    def _synonym_replacement(self, text: str) -> str:
        """Replace words with synonyms."""
        words = text.split()
        attacked_words = []
        
        # Replacement probability based on strength
        replacement_prob = {"light": 0.1, "medium": 0.3, "heavy": 0.5}.get(self.strength, 0.3)
        
        for word in words:
            if random.random() < replacement_prob and word.lower() in self.synonym_dict:
                synonyms = self.synonym_dict[word.lower()]
                new_word = random.choice(synonyms)
                # Preserve capitalization
                if word[0].isupper():
                    new_word = new_word.capitalize()
                attacked_words.append(new_word)
            else:
                attacked_words.append(word)
        
        return " ".join(attacked_words)
    
    def _sentence_reordering(self, text: str) -> str:
        """Reorder sentences while preserving meaning."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return text
        
        # Randomly swap adjacent sentences
        reorder_prob = {"light": 0.2, "medium": 0.4, "heavy": 0.6}.get(self.strength, 0.4)
        
        reordered = sentences.copy()
        for i in range(len(reordered) - 1):
            if random.random() < reorder_prob:
                reordered[i], reordered[i + 1] = reordered[i + 1], reordered[i]
        
        return ". ".join(reordered) + "."
    
    def _text_expansion(self, text: str) -> str:
        """Expand text with additional phrases."""
        words = text.split()
        expanded_words = []
        
        expansion_phrases = [
            "in other words", "that is to say", "furthermore", "additionally",
            "specifically", "in particular", "for example", "such as"
        ]
        
        expansion_prob = {"light": 0.05, "medium": 0.1, "heavy": 0.2}.get(self.strength, 0.1)
        
        for i, word in enumerate(words):
            expanded_words.append(word)
            if random.random() < expansion_prob and i < len(words) - 1:
                phrase = random.choice(expansion_phrases)
                expanded_words.append(phrase)
        
        return " ".join(expanded_words)
    
    def _combined_paraphrase(self, text: str) -> str:
        """Apply multiple paraphrasing techniques."""
        result = text
        result = self._synonym_replacement(result)
        result = self._text_expansion(result)
        result = self._sentence_reordering(result)
        return result
    
    def _build_synonym_dict(self) -> Dict[str, List[str]]:
        """Build dictionary of synonyms."""
        return {
            "algorithm": ["method", "procedure", "technique", "approach"],
            "method": ["approach", "technique", "procedure", "strategy"],
            "system": ["framework", "structure", "architecture", "platform"],
            "model": ["framework", "structure", "representation", "paradigm"],
            "data": ["information", "content", "material", "input"],
            "text": ["content", "material", "document", "passage"],
            "analysis": ["examination", "evaluation", "assessment", "study"],
            "result": ["outcome", "finding", "conclusion", "output"],
            "approach": ["method", "strategy", "technique", "way"],
            "technique": ["method", "approach", "strategy", "procedure"],
            "the": ["this", "that", "such"],
            "and": ["plus", "also", "as well as"],
            "with": ["using", "through", "via"],
            "for": ["regarding", "concerning", "about"],
            "watermark": ["signature", "marker", "identifier", "tag"],
            "detection": ["identification", "recognition", "discovery"],
            "generation": ["creation", "production", "synthesis"],
            "security": ["protection", "safety", "defense"]
        }


class TruncationAttack(BaseAttack):
    """Truncation attack removing parts of text."""
    
    def attack(self, text: str) -> str:
        """Apply truncation attack."""
        words = text.split()
        if len(words) <= 1:
            return text
        
        # Truncation ratio based on strength
        truncation_ratio = {"light": 0.1, "medium": 0.3, "heavy": 0.5}.get(self.strength, 0.3)
        
        truncation_type = random.choice(["beginning", "end", "middle", "random"])
        
        words_to_remove = int(len(words) * truncation_ratio)
        
        if truncation_type == "beginning":
            return " ".join(words[words_to_remove:])
        elif truncation_type == "end":
            return " ".join(words[:-words_to_remove])
        elif truncation_type == "middle":
            start = len(words) // 4
            end = start + words_to_remove
            return " ".join(words[:start] + words[end:])
        else:  # random
            indices_to_remove = set(random.sample(range(len(words)), words_to_remove))
            return " ".join(word for i, word in enumerate(words) if i not in indices_to_remove)


class InsertionAttack(BaseAttack):
    """Insertion attack adding noise words."""
    
    def attack(self, text: str) -> str:
        """Apply insertion attack."""
        words = text.split()
        insertion_prob = {"light": 0.05, "medium": 0.15, "heavy": 0.3}.get(self.strength, 0.15)
        
        noise_words = [
            "actually", "basically", "essentially", "fundamentally", "generally",
            "importantly", "particularly", "specifically", "typically", "usually",
            "clearly", "obviously", "certainly", "definitely", "probably"
        ]
        
        attacked_words = []
        for word in words:
            attacked_words.append(word)
            if random.random() < insertion_prob:
                noise_word = random.choice(noise_words)
                attacked_words.append(noise_word)
        
        return " ".join(attacked_words)


class SubstitutionAttack(BaseAttack):
    """Character-level substitution attack."""
    
    def attack(self, text: str) -> str:
        """Apply character substitution attack."""
        substitution_prob = {"light": 0.01, "medium": 0.03, "heavy": 0.08}.get(self.strength, 0.03)
        
        # Common character substitutions
        substitutions = {
            'a': ['@', 'α'], 'e': ['3', 'ε'], 'i': ['1', '!'], 'o': ['0', 'ο'],
            's': ['$', '5'], 'g': ['9'], 'l': ['1', '|'], 't': ['7']
        }
        
        attacked_chars = []
        for char in text:
            if char.lower() in substitutions and random.random() < substitution_prob:
                substitution = random.choice(substitutions[char.lower()])
                attacked_chars.append(substitution)
            else:
                attacked_chars.append(char)
        
        return "".join(attacked_chars)


class TranslationAttack(BaseAttack):
    """Back-translation attack simulation."""
    
    def attack(self, text: str) -> str:
        """Simulate back-translation attack."""
        # Simulate translation artifacts
        words = text.split()
        
        # Translation probability based on strength
        translation_prob = {"light": 0.1, "medium": 0.2, "heavy": 0.4}.get(self.strength, 0.2)
        
        # Common translation artifacts
        translation_changes = {
            "the": ["a", "this", "that"],
            "a": ["the", "one", "some"],
            "is": ["was", "becomes", "appears"],
            "are": ["were", "become", "appear"],
            "will": ["shall", "would", "going to"],
            "can": ["could", "able to", "may"],
            "and": ["as well as", "plus", "together with"],
            "but": ["however", "although", "yet"],
            "very": ["extremely", "quite", "really"],
            "good": ["well", "fine", "excellent"],
            "bad": ["poor", "terrible", "awful"]
        }
        
        attacked_words = []
        for word in words:
            if word.lower() in translation_changes and random.random() < translation_prob:
                replacement = random.choice(translation_changes[word.lower()])
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                attacked_words.append(replacement)
            else:
                attacked_words.append(word)
        
        return " ".join(attacked_words)


class CombinedAttack(BaseAttack):
    """Combined attack using multiple strategies."""
    
    def __init__(self, attacks: List[str] = None, strength: str = "medium", **kwargs):
        """Initialize combined attack."""
        super().__init__(strength, **kwargs)
        self.attacks = attacks or ["paraphrase", "truncation", "insertion"]
        self.attack_instances = self._initialize_attacks()
    
    def attack(self, text: str) -> str:
        """Apply combined attack."""
        result = text
        
        # Apply attacks in sequence
        for attack_instance in self.attack_instances:
            if random.random() < 0.7:  # 70% chance to apply each attack
                result = attack_instance.attack(result)
        
        return result
    
    def _initialize_attacks(self) -> List[BaseAttack]:
        """Initialize attack instances."""
        attack_map = {
            "paraphrase": ParaphraseAttack,
            "truncation": TruncationAttack,
            "insertion": InsertionAttack,
            "substitution": SubstitutionAttack,
            "translation": TranslationAttack
        }
        
        return [attack_map[name](strength=self.strength) for name in self.attacks if name in attack_map]


class AttackSimulator:
    """Main attack simulation orchestrator."""
    
    def __init__(self):
        """Initialize attack simulator."""
        self.attack_registry = {
            "paraphrase": ParaphraseAttack,
            "truncation": TruncationAttack,
            "insertion": InsertionAttack,
            "substitution": SubstitutionAttack,
            "translation": TranslationAttack,
            "combined": CombinedAttack
        }
    
    def register_attack(self, name: str, attack_class: type):
        """Register new attack type."""
        self.attack_registry[name] = attack_class
    
    def run_attack(self, text: str, attack_name: str, **kwargs) -> AttackResult:
        """Run specific attack on text."""
        if attack_name not in self.attack_registry:
            raise ValueError(f"Unknown attack: {attack_name}")
        
        attack = self.attack_registry[attack_name](**kwargs)
        attacked_text = attack.attack(text)
        
        # Evaluate attack
        quality_score, similarity_score = attack.evaluate_attack(text, attacked_text)
        
        # Determine success (simplified - would need actual detector)
        success = similarity_score < 0.8  # Threshold for "successful" attack
        
        return AttackResult(
            original_text=text,
            attacked_text=attacked_text,
            attack_type=attack_name,
            success=success,
            quality_score=quality_score,
            similarity_score=similarity_score,
            metadata={
                "strength": kwargs.get("strength", "medium"),
                "method": kwargs.get("method", "default")
            }
        )
    
    def run_attack_suite(self, texts: List[str], attacks: List[str], **kwargs) -> Dict[str, List[AttackResult]]:
        """Run multiple attacks on multiple texts."""
        results = {}
        
        for attack_name in attacks:
            results[attack_name] = []
            for text in texts:
                try:
                    result = self.run_attack(text, attack_name, **kwargs)
                    results[attack_name].append(result)
                except Exception as e:
                    # Log error and continue
                    print(f"Error running {attack_name} on text: {e}")
        
        return results
    
    def evaluate_robustness(self, original_texts: List[str], attack_results: Dict[str, List[AttackResult]]) -> Dict[str, float]:
        """Evaluate overall robustness against attacks."""
        robustness_scores = {}
        
        for attack_name, results in attack_results.items():
            if not results:
                continue
            
            # Average quality preservation
            avg_quality = sum(r.quality_score for r in results) / len(results)
            
            # Average similarity preservation
            avg_similarity = sum(r.similarity_score for r in results) / len(results)
            
            # Robustness score (higher is better)
            robustness_score = (avg_quality + avg_similarity) / 2
            robustness_scores[attack_name] = robustness_score
        
        return robustness_scores
    
    def list_attacks(self) -> List[str]:
        """List available attacks."""
        return list(self.attack_registry.keys())