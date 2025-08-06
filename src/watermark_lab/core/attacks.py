"""Attack simulation framework for testing watermark robustness."""

import random
import re
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import statistics

from ..utils.logging import get_logger
from ..utils.exceptions import AttackError


@dataclass
class AttackResult:
    """Result of an attack on watermarked text."""
    original_text: str
    attacked_text: str
    attack_type: str
    attack_strength: str
    quality_score: float
    similarity_score: float
    attack_success: bool
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_text": self.original_text,
            "attacked_text": self.attacked_text,
            "attack_type": self.attack_type,
            "attack_strength": self.attack_strength,
            "quality_score": self.quality_score,
            "similarity_score": self.similarity_score,
            "attack_success": self.attack_success,
            "metadata": self.metadata
        }


class BaseAttack(ABC):
    """Base class for watermark attacks."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"attack.{name}")
    
    @abstractmethod
    def attack(self, text: str, strength: str = "medium") -> AttackResult:
        """Execute attack on text."""
        pass
    
    def _calculate_quality_score(self, original: str, attacked: str) -> float:
        """Calculate quality preservation score."""
        orig_tokens = original.lower().split()
        att_tokens = attacked.lower().split()
        
        if not orig_tokens or not att_tokens:
            return 0.0
        
        # Simple quality score based on length preservation and vocabulary overlap
        length_ratio = len(att_tokens) / len(orig_tokens)
        length_score = max(0.0, 1.0 - abs(1.0 - length_ratio))
        
        orig_vocab = set(orig_tokens)
        att_vocab = set(att_tokens)
        vocab_overlap = len(orig_vocab & att_vocab) / max(len(orig_vocab | att_vocab), 1)
        
        return (length_score * 0.4 + vocab_overlap * 0.6)
    
    def _calculate_similarity_score(self, original: str, attacked: str) -> float:
        """Calculate semantic similarity score."""
        orig_tokens = set(original.lower().split())
        att_tokens = set(attacked.lower().split())
        
        if not orig_tokens and not att_tokens:
            return 1.0
        
        intersection = len(orig_tokens & att_tokens)
        union = len(orig_tokens | att_tokens)
        
        return intersection / max(union, 1)


class ParaphraseAttack(BaseAttack):
    """Paraphrasing attack to test watermark robustness."""
    
    def __init__(self):
        super().__init__("paraphrase")
        self.synonyms = self._build_synonym_dict()
        self.sentence_transforms = self._build_sentence_transforms()
    
    def attack(self, text: str, strength: str = "medium") -> AttackResult:
        """Execute paraphrasing attack."""
        try:
            attacked_text = self._paraphrase_text(text, strength)
            
            quality_score = self._calculate_quality_score(text, attacked_text)
            similarity_score = self._calculate_similarity_score(text, attacked_text)
            
            # Attack considered successful if text is significantly changed
            attack_success = similarity_score < 0.8
            
            return AttackResult(
                original_text=text,
                attacked_text=attacked_text,
                attack_type="paraphrase",
                attack_strength=strength,
                quality_score=quality_score,
                similarity_score=similarity_score,
                attack_success=attack_success,
                metadata={"paraphrase_method": "synonym_replacement"}
            )
            
        except Exception as e:
            self.logger.error(f"Paraphrase attack failed: {e}")
            raise AttackError(f"Paraphrase attack failed: {e}")
    
    def _paraphrase_text(self, text: str, strength: str) -> str:
        """Paraphrase text with given strength."""
        tokens = text.split()
        
        # Determine replacement probability based on strength
        replacement_probs = {"light": 0.2, "medium": 0.4, "heavy": 0.6}
        prob = replacement_probs.get(strength, 0.4)
        
        paraphrased_tokens = []
        
        for token in tokens:
            if random.random() < prob:
                # Try to replace with synonym
                clean_token = re.sub(r'[^\w]', '', token.lower())
                if clean_token in self.synonyms:
                    synonym = random.choice(self.synonyms[clean_token])
                    # Preserve original capitalization
                    if token[0].isupper():
                        synonym = synonym.capitalize()
                    paraphrased_tokens.append(synonym)
                else:
                    paraphrased_tokens.append(token)
            else:
                paraphrased_tokens.append(token)
        
        paraphrased_text = " ".join(paraphrased_tokens)
        
        # Apply sentence-level transformations for medium/heavy attacks
        if strength in ["medium", "heavy"]:
            paraphrased_text = self._apply_sentence_transforms(paraphrased_text, strength)
        
        return paraphrased_text
    
    def _apply_sentence_transforms(self, text: str, strength: str) -> str:
        """Apply sentence-level transformations."""
        sentences = re.split(r'([.!?])', text)
        transformed_sentences = []
        
        transform_prob = 0.3 if strength == "medium" else 0.5
        
        for i in range(0, len(sentences), 2):  # Process sentence and punctuation pairs
            sentence = sentences[i].strip()
            punct = sentences[i+1] if i+1 < len(sentences) else ""
            
            if sentence and random.random() < transform_prob:
                # Apply random transformation
                transformation = random.choice(self.sentence_transforms)
                sentence = transformation(sentence)
            
            transformed_sentences.append(sentence + punct)
        
        return "".join(transformed_sentences)
    
    def _build_synonym_dict(self) -> Dict[str, List[str]]:
        """Build synonym dictionary for replacements."""
        return {
            "algorithm": ["method", "approach", "technique", "procedure"],
            "method": ["approach", "technique", "algorithm", "way"],
            "system": ["framework", "platform", "structure", "model"],
            "model": ["system", "framework", "structure", "design"],
            "data": ["information", "content", "material", "dataset"],
            "text": ["content", "document", "passage", "material"],
            "analysis": ["examination", "study", "investigation", "assessment"],
            "research": ["study", "investigation", "analysis", "exploration"],
            "result": ["outcome", "finding", "conclusion", "output"],
            "approach": ["method", "technique", "strategy", "way"],
            "technique": ["method", "approach", "strategy", "procedure"],
            "process": ["procedure", "method", "operation", "workflow"],
            "information": ["data", "content", "details", "material"],
            "knowledge": ["understanding", "information", "expertise", "wisdom"],
            "learning": ["training", "education", "acquisition", "development"],
            "intelligence": ["reasoning", "cognition", "understanding", "smarts"],
            "artificial": ["synthetic", "simulated", "manufactured", "constructed"],
            "machine": ["computer", "device", "system", "apparatus"],
            "computer": ["machine", "system", "device", "processor"],
            "technology": ["tech", "innovation", "advancement", "system"],
            "science": ["research", "study", "field", "discipline"],
            "application": ["use", "implementation", "deployment", "usage"],
            "generation": ["creation", "production", "formation", "development"],
            "detection": ["identification", "discovery", "recognition", "finding"],
            "watermark": ["signature", "marker", "identifier", "stamp"],
            "security": ["protection", "safety", "defense", "safeguarding"]
        }
    
    def _build_sentence_transforms(self) -> List[callable]:
        """Build sentence transformation functions."""
        
        def passive_to_active(sentence: str) -> str:
            """Simple passive to active voice transformation."""
            # Very basic transformation - just for demonstration
            if "is" in sentence and "by" in sentence:
                parts = sentence.split(" is ")
                if len(parts) == 2:
                    return f"The system {parts[1].replace(' by', ' uses')}"
            return sentence
        
        def reorder_clauses(sentence: str) -> str:
            """Reorder sentence clauses."""
            parts = sentence.split(", ")
            if len(parts) > 1:
                random.shuffle(parts)
                return ", ".join(parts)
            return sentence
        
        def add_connectives(sentence: str) -> str:
            """Add connecting words."""
            connectives = ["Furthermore", "Moreover", "Additionally", "In addition"]
            if not sentence.startswith(tuple(connectives)):
                return f"{random.choice(connectives)}, " + sentence.lower()
            return sentence
        
        return [passive_to_active, reorder_clauses, add_connectives]
    

class TruncationAttack(BaseAttack):
    """Truncation attack by removing parts of text."""
    
    def __init__(self):
        super().__init__("truncation")
    
    def attack(self, text: str, strength: str = "medium") -> AttackResult:
        """Execute truncation attack."""
        try:
            attacked_text = self._truncate_text(text, strength)
            
            quality_score = self._calculate_quality_score(text, attacked_text)
            similarity_score = self._calculate_similarity_score(text, attacked_text)
            
            # Attack success based on amount removed
            original_length = len(text.split())
            attacked_length = len(attacked_text.split())
            removal_ratio = 1.0 - (attacked_length / max(original_length, 1))
            attack_success = removal_ratio > 0.2
            
            return AttackResult(
                original_text=text,
                attacked_text=attacked_text,
                attack_type="truncation",
                attack_strength=strength,
                quality_score=quality_score,
                similarity_score=similarity_score,
                attack_success=attack_success,
                metadata={"removal_ratio": removal_ratio}
            )
            
        except Exception as e:
            self.logger.error(f"Truncation attack failed: {e}")
            raise AttackError(f"Truncation attack failed: {e}")
    
    def _truncate_text(self, text: str, strength: str) -> str:
        """Truncate text with given strength."""
        tokens = text.split()
        
        if not tokens:
            return text
        
        # Determine truncation ratios
        truncation_ratios = {"light": 0.1, "medium": 0.25, "heavy": 0.4}
        ratio = truncation_ratios.get(strength, 0.25)
        
        tokens_to_remove = int(len(tokens) * ratio)
        
        if tokens_to_remove == 0:
            return text
        
        # Random truncation strategies
        strategy = random.choice(["beginning", "end", "middle", "random"])
        
        if strategy == "beginning":
            truncated_tokens = tokens[tokens_to_remove:]
        elif strategy == "end":
            truncated_tokens = tokens[:-tokens_to_remove]
        elif strategy == "middle":
            start_remove = len(tokens) // 4
            end_remove = start_remove + tokens_to_remove
            truncated_tokens = tokens[:start_remove] + tokens[end_remove:]
        else:  # random
            indices_to_remove = set(random.sample(range(len(tokens)), tokens_to_remove))
            truncated_tokens = [token for i, token in enumerate(tokens) if i not in indices_to_remove]
        
        return " ".join(truncated_tokens)


class InsertionAttack(BaseAttack):
    """Insertion attack by adding noise text."""
    
    def __init__(self):
        super().__init__("insertion")
        self.noise_phrases = self._build_noise_phrases()
    
    def attack(self, text: str, strength: str = "medium") -> AttackResult:
        """Execute insertion attack."""
        try:
            attacked_text = self._insert_noise(text, strength)
            
            quality_score = self._calculate_quality_score(text, attacked_text)
            similarity_score = self._calculate_similarity_score(text, attacked_text)
            
            # Attack success based on amount inserted
            original_length = len(text.split())
            attacked_length = len(attacked_text.split())
            insertion_ratio = (attacked_length / max(original_length, 1)) - 1.0
            attack_success = insertion_ratio > 0.15
            
            return AttackResult(
                original_text=text,
                attacked_text=attacked_text,
                attack_type="insertion",
                attack_strength=strength,
                quality_score=quality_score,
                similarity_score=similarity_score,
                attack_success=attack_success,
                metadata={"insertion_ratio": insertion_ratio}
            )
            
        except Exception as e:
            self.logger.error(f"Insertion attack failed: {e}")
            raise AttackError(f"Insertion attack failed: {e}")
    
    def _insert_noise(self, text: str, strength: str) -> str:
        """Insert noise into text."""
        tokens = text.split()
        
        if not tokens:
            return text
        
        # Determine insertion ratios
        insertion_ratios = {"light": 0.1, "medium": 0.2, "heavy": 0.35}
        ratio = insertion_ratios.get(strength, 0.2)
        
        tokens_to_insert = max(1, int(len(tokens) * ratio))
        
        # Insert noise at random positions
        for _ in range(tokens_to_insert):
            insert_position = random.randint(0, len(tokens))
            noise_phrase = random.choice(self.noise_phrases)
            
            # Insert as individual tokens
            noise_tokens = noise_phrase.split()
            for i, noise_token in enumerate(noise_tokens):
                tokens.insert(insert_position + i, noise_token)
        
        return " ".join(tokens)
    
    def _build_noise_phrases(self) -> List[str]:
        """Build noise phrases for insertion."""
        return [
            "furthermore",
            "in addition",
            "moreover",
            "also",
            "specifically",
            "particularly",
            "especially",
            "notably",
            "importantly",
            "significantly",
            "as mentioned",
            "for example",
            "for instance",
            "such as",
            "including",
            "namely",
            "that is",
            "in other words",
            "to clarify",
            "to elaborate",
            "overall",
            "in summary",
            "in conclusion",
            "ultimately",
            "finally"
        ]


class SubstitutionAttack(BaseAttack):
    """Token substitution attack."""
    
    def __init__(self):
        super().__init__("substitution")
    
    def attack(self, text: str, strength: str = "medium") -> AttackResult:
        """Execute substitution attack."""
        try:
            attacked_text = self._substitute_tokens(text, strength)
            
            quality_score = self._calculate_quality_score(text, attacked_text)
            similarity_score = self._calculate_similarity_score(text, attacked_text)
            
            attack_success = similarity_score < 0.85
            
            return AttackResult(
                original_text=text,
                attacked_text=attacked_text,
                attack_type="substitution",
                attack_strength=strength,
                quality_score=quality_score,
                similarity_score=similarity_score,
                attack_success=attack_success,
                metadata={"substitution_method": "random_replacement"}
            )
            
        except Exception as e:
            self.logger.error(f"Substitution attack failed: {e}")
            raise AttackError(f"Substitution attack failed: {e}")
    
    def _substitute_tokens(self, text: str, strength: str) -> str:
        """Substitute random tokens."""
        tokens = text.split()
        
        if not tokens:
            return text
        
        # Determine substitution probability
        substitution_probs = {"light": 0.05, "medium": 0.15, "heavy": 0.3}
        prob = substitution_probs.get(strength, 0.15)
        
        substitution_vocab = [
            "item", "element", "component", "part", "aspect", "feature",
            "factor", "entity", "object", "thing", "concept", "idea",
            "notion", "principle", "characteristic", "property"
        ]
        
        substituted_tokens = []
        for token in tokens:
            if random.random() < prob and len(token) > 3:  # Don't substitute short words
                substitute = random.choice(substitution_vocab)
                # Preserve capitalization
                if token[0].isupper():
                    substitute = substitute.capitalize()
                substituted_tokens.append(substitute)
            else:
                substituted_tokens.append(token)
        
        return " ".join(substituted_tokens)


class AttackSimulator:
    """Simulates various attacks on watermarked text."""
    
    def __init__(self):
        self.attacks = {
            "paraphrase": ParaphraseAttack(),
            "truncation": TruncationAttack(),
            "insertion": InsertionAttack(),
            "substitution": SubstitutionAttack()
        }
        self.logger = get_logger("attack.simulator")
    
    def register_attack(self, name: str, attack: BaseAttack) -> None:
        """Register a new attack."""
        self.attacks[name] = attack
        self.logger.info(f"Registered attack: {name}")
    
    def run_attack(self, text: str, attack_name: str, strength: str = "medium") -> AttackResult:
        """Run specific attack on text."""
        if attack_name not in self.attacks:
            available = ", ".join(self.attacks.keys())
            raise AttackError(f"Unknown attack: {attack_name}. Available: {available}")
        
        attack = self.attacks[attack_name]
        return attack.attack(text, strength)
    
    def run_attack_suite(self, text: str, attack_names: Optional[List[str]] = None, 
                        strength: str = "medium") -> Dict[str, AttackResult]:
        """Run multiple attacks on text."""
        if attack_names is None:
            attack_names = list(self.attacks.keys())
        
        results = {}
        
        for attack_name in attack_names:
            try:
                result = self.run_attack(text, attack_name, strength)
                results[attack_name] = result
                self.logger.info(f"Attack {attack_name} completed with success: {result.attack_success}")
            except Exception as e:
                self.logger.error(f"Attack {attack_name} failed: {e}")
                continue
        
        return results