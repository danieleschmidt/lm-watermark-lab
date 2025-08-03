"""Comprehensive evaluation framework for watermarking quality and performance."""

import math
import statistics
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import re


@dataclass
class QualityMetrics:
    """Quality assessment metrics for watermarked text."""
    perplexity_increase: float
    bleu_score: float
    semantic_similarity: float
    diversity_score: float
    fluency_score: float
    coherence_score: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "perplexity_increase": self.perplexity_increase,
            "bleu_score": self.bleu_score,
            "semantic_similarity": self.semantic_similarity,
            "diversity_score": self.diversity_score,
            "fluency_score": self.fluency_score,
            "coherence_score": self.coherence_score
        }


@dataclass
class DetectabilityMetrics:
    """Detectability performance metrics."""
    true_positive_rate: float
    false_positive_rate: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confidence_avg: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "true_positive_rate": self.true_positive_rate,
            "false_positive_rate": self.false_positive_rate,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
            "confidence_avg": self.confidence_avg
        }


@dataclass
class RobustnessMetrics:
    """Robustness metrics against attacks."""
    attack_success_rate: Dict[str, float]
    signal_degradation: Dict[str, float]
    recovery_rate: Dict[str, float]
    overall_robustness: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "attack_success_rate": self.attack_success_rate,
            "signal_degradation": self.signal_degradation,
            "recovery_rate": self.recovery_rate,
            "overall_robustness": self.overall_robustness
        }


class QualityEvaluator:
    """Evaluates text quality metrics."""
    
    def __init__(self):
        """Initialize quality evaluator."""
        self.reference_vocab = self._build_reference_vocab()
    
    def evaluate_quality(self, original_text: str, watermarked_text: str) -> QualityMetrics:
        """Evaluate quality of watermarked text against original."""
        perplexity_increase = self._calculate_perplexity_increase(original_text, watermarked_text)
        bleu_score = self._calculate_bleu_score(original_text, watermarked_text)
        semantic_similarity = self._calculate_semantic_similarity(original_text, watermarked_text)
        diversity_score = self._calculate_diversity_score(watermarked_text)
        fluency_score = self._calculate_fluency_score(watermarked_text)
        coherence_score = self._calculate_coherence_score(watermarked_text)
        
        return QualityMetrics(
            perplexity_increase=perplexity_increase,
            bleu_score=bleu_score,
            semantic_similarity=semantic_similarity,
            diversity_score=diversity_score,
            fluency_score=fluency_score,
            coherence_score=coherence_score
        )
    
    def _calculate_perplexity_increase(self, original: str, watermarked: str) -> float:
        """Calculate perplexity increase (simplified implementation)."""
        original_tokens = original.lower().split()
        watermarked_tokens = watermarked.lower().split()
        
        # Simple unigram probability model
        original_probs = self._calculate_token_probabilities(original_tokens)
        watermarked_probs = self._calculate_token_probabilities(watermarked_tokens)
        
        # Calculate cross-entropy
        original_entropy = -sum(prob * math.log2(prob) for prob in original_probs.values() if prob > 0)
        watermarked_entropy = -sum(prob * math.log2(prob) for prob in watermarked_probs.values() if prob > 0)
        
        # Perplexity = 2^entropy
        original_perplexity = 2 ** original_entropy
        watermarked_perplexity = 2 ** watermarked_entropy
        
        return (watermarked_perplexity - original_perplexity) / original_perplexity if original_perplexity > 0 else 0.0
    
    def _calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score (simplified implementation)."""
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        if not ref_tokens or not cand_tokens:
            return 0.0
        
        # 1-gram precision
        ref_1grams = Counter(ref_tokens)
        cand_1grams = Counter(cand_tokens)
        
        overlap = 0
        for token in cand_1grams:
            overlap += min(cand_1grams[token], ref_1grams.get(token, 0))
        
        precision_1 = overlap / len(cand_tokens) if cand_tokens else 0
        
        # 2-gram precision
        ref_2grams = Counter(zip(ref_tokens[:-1], ref_tokens[1:]))
        cand_2grams = Counter(zip(cand_tokens[:-1], cand_tokens[1:]))
        
        overlap_2 = 0
        for bigram in cand_2grams:
            overlap_2 += min(cand_2grams[bigram], ref_2grams.get(bigram, 0))
        
        precision_2 = overlap_2 / max(1, len(cand_tokens) - 1)
        
        # Brevity penalty
        bp = min(1.0, len(cand_tokens) / len(ref_tokens)) if ref_tokens else 0
        
        # BLEU-2 score
        if precision_1 > 0 and precision_2 > 0:
            bleu = bp * (precision_1 * precision_2) ** 0.5
        else:
            bleu = 0.0
        
        return bleu
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity (simplified implementation)."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        # Jaccard similarity
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_diversity_score(self, text: str) -> float:
        """Calculate lexical diversity (Type-Token Ratio)."""
        tokens = text.lower().split()
        if not tokens:
            return 0.0
        
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        
        return unique_tokens / total_tokens
    
    def _calculate_fluency_score(self, text: str) -> float:
        """Calculate fluency score based on linguistic patterns."""
        tokens = text.split()
        if len(tokens) < 2:
            return 0.0
        
        score = 0.0
        
        # Sentence structure score
        sentences = re.split(r'[.!?]+', text)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / len([s for s in sentences if s.strip()])
            # Optimal sentence length around 15-20 words
            length_score = max(0, 1.0 - abs(avg_sentence_length - 17.5) / 17.5)
            score += length_score * 0.3
        
        # Grammar pattern score (simplified)
        grammar_score = self._assess_grammar_patterns(tokens)
        score += grammar_score * 0.4
        
        # Vocabulary appropriateness
        vocab_score = self._assess_vocabulary_appropriateness(tokens)
        score += vocab_score * 0.3
        
        return min(1.0, score)
    
    def _calculate_coherence_score(self, text: str) -> float:
        """Calculate text coherence score."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 1.0
        
        coherence_scores = []
        
        for i in range(len(sentences) - 1):
            # Simple coherence based on shared vocabulary
            tokens1 = set(sentences[i].lower().split())
            tokens2 = set(sentences[i + 1].lower().split())
            
            if tokens1 and tokens2:
                overlap = len(tokens1 & tokens2)
                similarity = overlap / min(len(tokens1), len(tokens2))
                coherence_scores.append(similarity)
        
        return statistics.mean(coherence_scores) if coherence_scores else 0.5
    
    def _calculate_token_probabilities(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate token probabilities."""
        if not tokens:
            return {}
        
        counts = Counter(tokens)
        total = len(tokens)
        
        return {token: count / total for token, count in counts.items()}
    
    def _assess_grammar_patterns(self, tokens: List[str]) -> float:
        """Assess basic grammar patterns."""
        if not tokens:
            return 0.0
        
        # Simple heuristics for grammar assessment
        score = 0.0
        
        # Check for common function words
        function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        function_word_count = sum(1 for token in tokens if token.lower() in function_words)
        function_word_ratio = function_word_count / len(tokens)
        
        # Optimal function word ratio around 0.3-0.5
        if 0.2 <= function_word_ratio <= 0.6:
            score += 0.5
        
        # Check for repeated patterns (negative indicator)
        token_counts = Counter(tokens)
        max_repetition = max(token_counts.values()) / len(tokens)
        if max_repetition < 0.3:  # Not too repetitive
            score += 0.5
        
        return score
    
    def _assess_vocabulary_appropriateness(self, tokens: List[str]) -> float:
        """Assess vocabulary appropriateness."""
        if not tokens:
            return 0.0
        
        # Check against reference vocabulary
        appropriate_tokens = sum(1 for token in tokens if token.lower() in self.reference_vocab)
        return appropriate_tokens / len(tokens)
    
    def _build_reference_vocab(self) -> set:
        """Build reference vocabulary for quality assessment."""
        # Common English words for quality assessment
        return {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'algorithm', 'method', 'system', 'model', 'data', 'text', 'analysis',
            'research', 'study', 'result', 'approach', 'technique', 'process',
            'information', 'knowledge', 'learning', 'intelligence', 'artificial',
            'machine', 'computer', 'technology', 'science', 'application'
        }


class DetectabilityEvaluator:
    """Evaluates watermark detectability performance."""
    
    def evaluate_detectability(self, 
                             true_labels: List[bool], 
                             predicted_labels: List[bool],
                             confidence_scores: List[float]) -> DetectabilityMetrics:
        """Evaluate detectability metrics."""
        if len(true_labels) != len(predicted_labels):
            raise ValueError("True labels and predictions must have same length")
        
        # Confusion matrix components
        tp = sum(1 for true, pred in zip(true_labels, predicted_labels) if true and pred)
        fp = sum(1 for true, pred in zip(true_labels, predicted_labels) if not true and pred)
        tn = sum(1 for true, pred in zip(true_labels, predicted_labels) if not true and not pred)
        fn = sum(1 for true, pred in zip(true_labels, predicted_labels) if true and not pred)
        
        # Metrics
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # AUC-ROC (simplified)
        auc_roc = self._calculate_auc_roc(true_labels, confidence_scores)
        
        # Average confidence
        confidence_avg = statistics.mean(confidence_scores) if confidence_scores else 0.0
        
        return DetectabilityMetrics(
            true_positive_rate=tpr,
            false_positive_rate=fpr,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            confidence_avg=confidence_avg
        )
    
    def _calculate_auc_roc(self, true_labels: List[bool], scores: List[float]) -> float:
        """Calculate AUC-ROC (simplified implementation)."""
        if not true_labels or not scores:
            return 0.5
        
        # Sort by scores descending
        paired = list(zip(scores, true_labels))
        paired.sort(reverse=True)
        
        # Calculate AUC using trapezoidal rule
        thresholds = sorted(set(scores), reverse=True)
        tpr_values = []
        fpr_values = []
        
        for threshold in thresholds:
            predictions = [score >= threshold for score in scores]
            
            tp = sum(1 for true, pred in zip(true_labels, predictions) if true and pred)
            fp = sum(1 for true, pred in zip(true_labels, predictions) if not true and pred)
            tn = sum(1 for true, pred in zip(true_labels, predictions) if not true and not pred)
            fn = sum(1 for true, pred in zip(true_labels, predictions) if true and not pred)
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            tpr_values.append(tpr)
            fpr_values.append(fpr)
        
        # Trapezoidal integration
        auc = 0.0
        for i in range(1, len(fpr_values)):
            auc += (fpr_values[i] - fpr_values[i-1]) * (tpr_values[i] + tpr_values[i-1]) / 2
        
        return max(0.0, min(1.0, auc))


class RobustnessEvaluator:
    """Evaluates watermark robustness against attacks."""
    
    def evaluate_robustness(self, 
                          original_detections: List[bool],
                          attacked_detections: Dict[str, List[bool]],
                          attack_qualities: Dict[str, List[float]]) -> RobustnessMetrics:
        """Evaluate robustness against various attacks."""
        
        attack_success_rates = {}
        signal_degradations = {}
        recovery_rates = {}
        
        for attack_name, attacked_results in attacked_detections.items():
            if len(original_detections) != len(attacked_results):
                continue
            
            # Attack success rate (watermark removal)
            successful_attacks = sum(1 for orig, attacked in zip(original_detections, attacked_results) 
                                   if orig and not attacked)
            total_watermarked = sum(original_detections)
            success_rate = successful_attacks / total_watermarked if total_watermarked > 0 else 0.0
            attack_success_rates[attack_name] = success_rate
            
            # Signal degradation
            if attack_name in attack_qualities:
                qualities = attack_qualities[attack_name]
                avg_quality_loss = 1.0 - statistics.mean(qualities) if qualities else 1.0
                signal_degradations[attack_name] = avg_quality_loss
            
            # Recovery rate (watermark preservation)
            preserved_watermarks = sum(1 for orig, attacked in zip(original_detections, attacked_results)
                                     if orig and attacked)
            recovery_rate = preserved_watermarks / total_watermarked if total_watermarked > 0 else 0.0
            recovery_rates[attack_name] = recovery_rate
        
        # Overall robustness score
        if recovery_rates:
            overall_robustness = statistics.mean(recovery_rates.values())
        else:
            overall_robustness = 0.0
        
        return RobustnessMetrics(
            attack_success_rate=attack_success_rates,
            signal_degradation=signal_degradations,
            recovery_rate=recovery_rates,
            overall_robustness=overall_robustness
        )


class EvaluationSuite:
    """Comprehensive evaluation suite combining all metrics."""
    
    def __init__(self):
        """Initialize evaluation suite."""
        self.quality_evaluator = QualityEvaluator()
        self.detectability_evaluator = DetectabilityEvaluator()
        self.robustness_evaluator = RobustnessEvaluator()
    
    def comprehensive_evaluation(self,
                                original_texts: List[str],
                                watermarked_texts: List[str],
                                detection_results: List[bool],
                                confidence_scores: List[float],
                                true_labels: List[bool],
                                attack_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform comprehensive evaluation."""
        
        results = {}
        
        # Quality evaluation
        quality_metrics = []
        for orig, watermarked in zip(original_texts, watermarked_texts):
            metrics = self.quality_evaluator.evaluate_quality(orig, watermarked)
            quality_metrics.append(metrics)
        
        # Aggregate quality metrics
        if quality_metrics:
            results['quality'] = {
                'perplexity_increase': statistics.mean(m.perplexity_increase for m in quality_metrics),
                'bleu_score': statistics.mean(m.bleu_score for m in quality_metrics),
                'semantic_similarity': statistics.mean(m.semantic_similarity for m in quality_metrics),
                'diversity_score': statistics.mean(m.diversity_score for m in quality_metrics),
                'fluency_score': statistics.mean(m.fluency_score for m in quality_metrics),
                'coherence_score': statistics.mean(m.coherence_score for m in quality_metrics)
            }
        
        # Detectability evaluation
        detectability = self.detectability_evaluator.evaluate_detectability(
            true_labels, detection_results, confidence_scores
        )
        results['detectability'] = detectability.to_dict()
        
        # Robustness evaluation
        if attack_results:
            robustness = self.robustness_evaluator.evaluate_robustness(
                detection_results,
                attack_results.get('attacked_detections', {}),
                attack_results.get('attack_qualities', {})
            )
            results['robustness'] = robustness.to_dict()
        
        return results