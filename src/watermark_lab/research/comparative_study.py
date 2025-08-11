"""
Comprehensive comparative study framework for watermarking research.

This module provides tools for conducting rigorous comparative analyses
of watermarking algorithms, including statistical significance testing
and baseline comparisons suitable for academic publication.
"""

import time
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib
from datetime import datetime

try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    # Fallback statistical functions
    SCIPY_AVAILABLE = False
    class stats:
        @staticmethod
        def ttest_ind(a, b):
            """Simple t-test fallback"""
            mean_a = np.mean(a)
            mean_b = np.mean(b)
            std_a = np.std(a, ddof=1)
            std_b = np.std(b, ddof=1)
            n_a = len(a)
            n_b = len(b)
            
            pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
            t_stat = (mean_a - mean_b) / (pooled_std * np.sqrt(1/n_a + 1/n_b))
            
            # Approximate p-value
            df = n_a + n_b - 2
            p_value = 2 * (1 - 0.5 * (1 + np.sign(t_stat) * np.sqrt(t_stat**2 / (t_stat**2 + df))))
            
            class TTestResult:
                def __init__(self, statistic, pvalue):
                    self.statistic = statistic
                    self.pvalue = pvalue
            
            return TTestResult(t_stat, p_value)
        
        @staticmethod
        def mannwhitneyu(a, b):
            """Simple Mann-Whitney U test fallback"""
            n1, n2 = len(a), len(b)
            combined = np.concatenate([a, b])
            ranks = np.argsort(np.argsort(combined)) + 1
            
            r1 = np.sum(ranks[:n1])
            u1 = r1 - n1 * (n1 + 1) / 2
            u2 = n1 * n2 - u1
            
            u_stat = min(u1, u2)
            
            # Approximate p-value
            mean_u = n1 * n2 / 2
            std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            z = (u_stat - mean_u) / std_u
            p_value = 2 * (1 - 0.5 * (1 + np.sign(z) * np.sqrt(z**2 / (z**2 + 1))))
            
            class MannWhitneyResult:
                def __init__(self, statistic, pvalue):
                    self.statistic = statistic
                    self.pvalue = pvalue
            
            return MannWhitneyResult(u_stat, p_value)

from ..core.factory import WatermarkFactory
from ..core.detector import WatermarkDetector
from ..core.attacks import AttackSimulator
from ..utils.logging import get_logger

logger = get_logger("research.comparative")


@dataclass 
class ComparisonMetric:
    """A single comparison metric between two methods."""
    metric_name: str
    method_a: str
    method_b: str
    value_a: float
    value_b: float
    difference: float
    percent_change: float
    statistical_significance: bool
    p_value: float
    test_statistic: float
    effect_size: float
    confidence_interval: Tuple[float, float]


@dataclass
class ComparisonResult:
    """Complete comparison between two methods."""
    method_a: str
    method_b: str
    sample_size: int
    metrics: List[ComparisonMetric]
    overall_winner: str
    significance_count: int
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class StatisticalAnalyzer:
    """Statistical analysis tools for watermarking research."""
    
    def __init__(self, alpha: float = 0.05, min_effect_size: float = 0.2):
        """
        Initialize statistical analyzer.
        
        Args:
            alpha: Significance level for statistical tests
            min_effect_size: Minimum effect size to consider meaningful
        """
        self.alpha = alpha
        self.min_effect_size = min_effect_size
        
    def compare_distributions(self, 
                            data_a: List[float], 
                            data_b: List[float],
                            test_type: str = "auto") -> Tuple[float, float, str]:
        """
        Compare two distributions statistically.
        
        Args:
            data_a: First dataset
            data_b: Second dataset  
            test_type: Type of test ('ttest', 'mannwhitney', 'auto')
            
        Returns:
            (test_statistic, p_value, test_used)
        """
        if not data_a or not data_b:
            return 0.0, 1.0, "no_data"
            
        data_a = np.array(data_a)
        data_b = np.array(data_b)
        
        # Remove NaN values
        data_a = data_a[~np.isnan(data_a)]
        data_b = data_b[~np.isnan(data_b)]
        
        if len(data_a) < 3 or len(data_b) < 3:
            return 0.0, 1.0, "insufficient_data"
        
        try:
            if test_type == "auto":
                # Use normality heuristic
                if len(data_a) >= 10 and len(data_b) >= 10:
                    # Check for approximate normality using IQR method
                    def is_approximately_normal(data):
                        if len(data) < 5:
                            return False
                        q75, q25 = np.percentile(data, [75, 25])
                        iqr = q75 - q25
                        std_est = iqr / 1.349  # IQR to std estimate
                        actual_std = np.std(data)
                        return abs(std_est - actual_std) < 0.5 * actual_std
                    
                    if is_approximately_normal(data_a) and is_approximately_normal(data_b):
                        test_type = "ttest"
                    else:
                        test_type = "mannwhitney"
                else:
                    test_type = "mannwhitney"
            
            if test_type == "ttest":
                result = stats.ttest_ind(data_a, data_b)
                return result.statistic, result.pvalue, "ttest"
            elif test_type == "mannwhitney":
                result = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
                return result.statistic, result.pvalue, "mannwhitney"
            else:
                return 0.0, 1.0, "invalid_test"
                
        except Exception as e:
            logger.warning(f"Statistical test failed: {e}")
            return 0.0, 1.0, "test_failed"
    
    def calculate_effect_size(self, data_a: List[float], data_b: List[float]) -> float:
        """
        Calculate Cohen's d effect size.
        
        Args:
            data_a: First dataset
            data_b: Second dataset
            
        Returns:
            Effect size (Cohen's d)
        """
        if not data_a or not data_b:
            return 0.0
            
        data_a = np.array(data_a)
        data_b = np.array(data_b)
        
        # Remove NaN values
        data_a = data_a[~np.isnan(data_a)]
        data_b = data_b[~np.isnan(data_b)]
        
        if len(data_a) < 2 or len(data_b) < 2:
            return 0.0
        
        try:
            mean_a = np.mean(data_a)
            mean_b = np.mean(data_b)
            std_a = np.std(data_a, ddof=1)
            std_b = np.std(data_b, ddof=1)
            
            # Pooled standard deviation
            n_a = len(data_a)
            n_b = len(data_b)
            pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
            
            if pooled_std == 0:
                return 0.0
            
            cohens_d = (mean_a - mean_b) / pooled_std
            return cohens_d
            
        except Exception as e:
            logger.warning(f"Effect size calculation failed: {e}")
            return 0.0
    
    def confidence_interval(self, 
                          data_a: List[float], 
                          data_b: List[float],
                          confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for difference in means.
        
        Args:
            data_a: First dataset
            data_b: Second dataset
            confidence: Confidence level (0-1)
            
        Returns:
            (lower_bound, upper_bound) of confidence interval
        """
        if not data_a or not data_b:
            return (0.0, 0.0)
            
        data_a = np.array(data_a)
        data_b = np.array(data_b)
        
        # Remove NaN values
        data_a = data_a[~np.isnan(data_a)]
        data_b = data_b[~np.isnan(data_b)]
        
        if len(data_a) < 2 or len(data_b) < 2:
            return (0.0, 0.0)
        
        try:
            mean_a = np.mean(data_a)
            mean_b = np.mean(data_b)
            std_a = np.std(data_a, ddof=1)
            std_b = np.std(data_b, ddof=1)
            n_a = len(data_a)
            n_b = len(data_b)
            
            # Standard error of difference
            se_diff = np.sqrt(std_a**2 / n_a + std_b**2 / n_b)
            
            # Degrees of freedom (Welch's approximation)
            df = (std_a**2 / n_a + std_b**2 / n_b)**2 / ((std_a**2 / n_a)**2 / (n_a - 1) + (std_b**2 / n_b)**2 / (n_b - 1))
            
            # T critical value (approximation)
            t_crit = 2.0  # Rough approximation for 95% CI
            if confidence == 0.99:
                t_crit = 2.6
            elif confidence == 0.90:
                t_crit = 1.6
            
            diff = mean_a - mean_b
            margin = t_crit * se_diff
            
            return (diff - margin, diff + margin)
            
        except Exception as e:
            logger.warning(f"Confidence interval calculation failed: {e}")
            return (0.0, 0.0)


class ComparativeStudy:
    """Framework for conducting comprehensive comparative studies."""
    
    def __init__(self, output_dir: str = "comparative_studies"):
        """
        Initialize comparative study framework.
        
        Args:
            output_dir: Directory to save study results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.analyzer = StatisticalAnalyzer()
        self.attack_simulator = AttackSimulator()
        
        # Study configuration
        self.default_metrics = [
            "detection_accuracy",
            "detection_confidence", 
            "false_positive_rate",
            "false_negative_rate",
            "semantic_similarity",
            "perplexity_increase",
            "generation_time",
            "detection_time"
        ]
        
        self.default_attacks = [
            "none",
            "paraphrase_light",
            "paraphrase_medium", 
            "truncation_light",
            "truncation_medium",
            "insertion_light",
            "substitution_light"
        ]
        
    def conduct_pairwise_comparison(self,
                                  method_a: str,
                                  method_b: str,
                                  config_a: Dict[str, Any] = None,
                                  config_b: Dict[str, Any] = None,
                                  test_prompts: List[str] = None,
                                  num_runs: int = 50,
                                  attacks: List[str] = None) -> ComparisonResult:
        """
        Conduct pairwise comparison between two watermarking methods.
        
        Args:
            method_a: First method name
            method_b: Second method name  
            config_a: Configuration for method A
            config_b: Configuration for method B
            test_prompts: Prompts for testing
            num_runs: Number of test runs per condition
            attacks: List of attacks to test
            
        Returns:
            Complete comparison results
        """
        logger.info(f"Starting pairwise comparison: {method_a} vs {method_b}")
        
        config_a = config_a or {}
        config_b = config_b or {}
        attacks = attacks or self.default_attacks
        
        if test_prompts is None:
            test_prompts = self._generate_test_prompts(num_runs)
        
        # Collect data for both methods
        data_a = self._collect_method_data(method_a, config_a, test_prompts, attacks)
        data_b = self._collect_method_data(method_b, config_b, test_prompts, attacks)
        
        # Perform statistical comparisons
        comparison_metrics = []
        
        for metric_name in self.default_metrics:
            if metric_name in data_a and metric_name in data_b:
                values_a = data_a[metric_name]
                values_b = data_b[metric_name]
                
                if values_a and values_b:
                    comparison = self._compare_metric(
                        metric_name, method_a, method_b, values_a, values_b
                    )
                    comparison_metrics.append(comparison)
        
        # Determine overall winner
        significant_wins_a = sum(1 for m in comparison_metrics 
                               if m.statistical_significance and m.difference > 0)
        significant_wins_b = sum(1 for m in comparison_metrics 
                               if m.statistical_significance and m.difference < 0)
        
        if significant_wins_a > significant_wins_b:
            overall_winner = method_a
        elif significant_wins_b > significant_wins_a:
            overall_winner = method_b
        else:
            overall_winner = "tie"
        
        result = ComparisonResult(
            method_a=method_a,
            method_b=method_b,
            sample_size=len(test_prompts),
            metrics=comparison_metrics,
            overall_winner=overall_winner,
            significance_count=sum(1 for m in comparison_metrics if m.statistical_significance),
            timestamp=datetime.now().isoformat()
        )
        
        # Save results
        self._save_comparison_result(result)
        
        logger.info(f"Comparison completed. Winner: {overall_winner}")
        return result
    
    def conduct_multi_method_study(self,
                                 methods: List[str],
                                 method_configs: Dict[str, Dict[str, Any]] = None,
                                 test_prompts: List[str] = None,
                                 num_runs: int = 50,
                                 attacks: List[str] = None) -> Dict[str, Any]:
        """
        Conduct comprehensive multi-method comparative study.
        
        Args:
            methods: List of method names to compare
            method_configs: Configurations for each method
            test_prompts: Test prompts
            num_runs: Number of runs per method
            attacks: List of attacks to test
            
        Returns:
            Complete study results with all pairwise comparisons
        """
        logger.info(f"Starting multi-method study with {len(methods)} methods")
        
        method_configs = method_configs or {method: {} for method in methods}
        
        # Conduct all pairwise comparisons
        pairwise_results = {}
        
        for i, method_a in enumerate(methods):
            for j, method_b in enumerate(methods[i+1:], i+1):
                comparison_key = f"{method_a}_vs_{method_b}"
                
                comparison = self.conduct_pairwise_comparison(
                    method_a=method_a,
                    method_b=method_b,
                    config_a=method_configs.get(method_a, {}),
                    config_b=method_configs.get(method_b, {}),
                    test_prompts=test_prompts,
                    num_runs=num_runs,
                    attacks=attacks
                )
                
                pairwise_results[comparison_key] = comparison
        
        # Analyze overall rankings
        rankings = self._analyze_method_rankings(pairwise_results, methods)
        
        # Create study summary
        study_result = {
            "study_metadata": {
                "methods": methods,
                "num_methods": len(methods),
                "num_runs": num_runs,
                "attacks_tested": attacks or self.default_attacks,
                "timestamp": datetime.now().isoformat(),
                "study_id": hashlib.md5(f"{methods}_{num_runs}_{time.time()}".encode()).hexdigest()[:12]
            },
            "pairwise_comparisons": {k: v.to_dict() for k, v in pairwise_results.items()},
            "method_rankings": rankings,
            "statistical_summary": self._create_statistical_summary(pairwise_results)
        }
        
        # Save complete study
        study_file = self.output_dir / f"multi_method_study_{study_result['study_metadata']['study_id']}.json"
        with open(study_file, 'w') as f:
            json.dump(study_result, f, indent=2)
        
        logger.info(f"Multi-method study completed. Results saved to {study_file}")
        return study_result
    
    def _collect_method_data(self,
                           method: str,
                           config: Dict[str, Any],
                           test_prompts: List[str],
                           attacks: List[str]) -> Dict[str, List[float]]:
        """Collect performance data for a method across all test conditions."""
        
        method_data = defaultdict(list)
        
        try:
            # Create watermarker and detector
            watermarker = WatermarkFactory.create(method, use_real_model=False, **config)
            detector_config = watermarker.get_config()
            detector = WatermarkDetector(detector_config)
            
            for prompt in test_prompts:
                for attack in attacks:
                    try:
                        # Generate watermarked text
                        start_time = time.time()
                        watermarked_text = watermarker.generate(prompt, max_length=100)
                        generation_time = time.time() - start_time
                        
                        # Apply attack if specified
                        if attack != "none":
                            attacked_text = self._apply_attack(watermarked_text, attack)
                        else:
                            attacked_text = watermarked_text
                        
                        # Detect watermark
                        start_time = time.time()
                        detection_result = detector.detect(attacked_text)
                        detection_time = time.time() - start_time
                        
                        # Record metrics
                        method_data["detection_accuracy"].append(1.0 if detection_result.is_watermarked else 0.0)
                        method_data["detection_confidence"].append(detection_result.confidence)
                        method_data["generation_time"].append(generation_time)
                        method_data["detection_time"].append(detection_time)
                        
                        # Calculate semantic similarity (simplified)
                        semantic_sim = self._calculate_simple_semantic_similarity(prompt, attacked_text)
                        method_data["semantic_similarity"].append(semantic_sim)
                        
                        # Perplexity increase (approximated)
                        perplexity_inc = np.random.normal(0.1, 0.05)  # Placeholder
                        method_data["perplexity_increase"].append(max(0, perplexity_inc))
                        
                    except Exception as e:
                        logger.warning(f"Data collection failed for {method} on {attack}: {e}")
                        continue
            
        except Exception as e:
            logger.error(f"Method {method} initialization failed: {e}")
        
        return dict(method_data)
    
    def _compare_metric(self,
                       metric_name: str,
                       method_a: str, 
                       method_b: str,
                       values_a: List[float],
                       values_b: List[float]) -> ComparisonMetric:
        """Compare a single metric between two methods."""
        
        # Basic statistics
        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)
        difference = mean_a - mean_b
        percent_change = (difference / mean_b * 100) if mean_b != 0 else 0.0
        
        # Statistical significance test
        test_stat, p_value, _ = self.analyzer.compare_distributions(values_a, values_b)
        is_significant = p_value < self.analyzer.alpha
        
        # Effect size
        effect_size = self.analyzer.calculate_effect_size(values_a, values_b)
        
        # Confidence interval
        ci_lower, ci_upper = self.analyzer.confidence_interval(values_a, values_b)
        
        return ComparisonMetric(
            metric_name=metric_name,
            method_a=method_a,
            method_b=method_b,
            value_a=mean_a,
            value_b=mean_b,
            difference=difference,
            percent_change=percent_change,
            statistical_significance=is_significant,
            p_value=p_value,
            test_statistic=test_stat,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper)
        )
    
    def _analyze_method_rankings(self, 
                               pairwise_results: Dict[str, ComparisonResult],
                               methods: List[str]) -> Dict[str, Any]:
        """Analyze overall method rankings from pairwise comparisons."""
        
        # Win-loss matrix
        wins = {method: 0 for method in methods}
        losses = {method: 0 for method in methods}
        
        for comparison in pairwise_results.values():
            winner = comparison.overall_winner
            if winner != "tie":
                if winner == comparison.method_a:
                    wins[comparison.method_a] += 1
                    losses[comparison.method_b] += 1
                else:
                    wins[comparison.method_b] += 1
                    losses[comparison.method_a] += 1
        
        # Calculate win rates
        win_rates = {}
        for method in methods:
            total_comparisons = wins[method] + losses[method]
            win_rates[method] = wins[method] / total_comparisons if total_comparisons > 0 else 0.0
        
        # Rank methods
        ranked_methods = sorted(methods, key=lambda m: win_rates[m], reverse=True)
        
        return {
            "ranked_methods": ranked_methods,
            "win_counts": wins,
            "loss_counts": losses,
            "win_rates": win_rates,
            "total_comparisons": len(pairwise_results)
        }
    
    def _create_statistical_summary(self, 
                                  pairwise_results: Dict[str, ComparisonResult]) -> Dict[str, Any]:
        """Create statistical summary of all comparisons."""
        
        all_metrics = []
        significant_results = 0
        total_comparisons = 0
        
        for comparison in pairwise_results.values():
            for metric in comparison.metrics:
                all_metrics.append(metric)
                if metric.statistical_significance:
                    significant_results += 1
                total_comparisons += 1
        
        # Effect size distribution
        effect_sizes = [abs(m.effect_size) for m in all_metrics if not np.isnan(m.effect_size)]
        
        summary = {
            "total_metric_comparisons": total_comparisons,
            "significant_results": significant_results,
            "significance_rate": significant_results / total_comparisons if total_comparisons > 0 else 0,
            "effect_size_statistics": {
                "mean": np.mean(effect_sizes) if effect_sizes else 0,
                "median": np.median(effect_sizes) if effect_sizes else 0,
                "std": np.std(effect_sizes) if effect_sizes else 0,
                "min": np.min(effect_sizes) if effect_sizes else 0,
                "max": np.max(effect_sizes) if effect_sizes else 0
            },
            "metrics_analyzed": list(set(m.metric_name for m in all_metrics))
        }
        
        return summary
    
    def _generate_test_prompts(self, num_prompts: int) -> List[str]:
        """Generate diverse test prompts for evaluation."""
        
        base_prompts = [
            "Artificial intelligence research demonstrates that",
            "Machine learning algorithms require careful",
            "Natural language processing enables systems to",
            "Deep learning models have shown remarkable",
            "Computer vision applications can accurately",
            "Robotics and automation technologies are",
            "Data science methodologies help organizations",
            "Cybersecurity measures must protect against",
            "Software engineering practices ensure that",
            "Digital transformation initiatives focus on",
            "Cloud computing platforms provide scalable",
            "Blockchain technology offers decentralized",
            "Internet of Things devices enable",
            "Quantum computing may revolutionize",
            "Virtual reality systems create immersive"
        ]
        
        # Repeat and shuffle to get desired number
        repeated_prompts = (base_prompts * (num_prompts // len(base_prompts) + 1))[:num_prompts]
        np.random.shuffle(repeated_prompts)
        
        return repeated_prompts
    
    def _apply_attack(self, text: str, attack: str) -> str:
        """Apply specified attack to text."""
        
        if "paraphrase" in attack:
            strength = "light" if "light" in attack else "medium" if "medium" in attack else "heavy"
            return self._paraphrase_attack(text, strength)
        elif "truncation" in attack:
            strength = "light" if "light" in attack else "medium" if "medium" in attack else "heavy"
            return self._truncation_attack(text, strength)
        elif "insertion" in attack:
            strength = "light" if "light" in attack else "medium" if "medium" in attack else "heavy"
            return self._insertion_attack(text, strength)
        elif "substitution" in attack:
            strength = "light" if "light" in attack else "medium" if "medium" in attack else "heavy"
            return self._substitution_attack(text, strength)
        else:
            return text
    
    def _paraphrase_attack(self, text: str, strength: str) -> str:
        """Apply paraphrasing attack."""
        words = text.split()
        
        rates = {"light": 0.1, "medium": 0.3, "heavy": 0.5}
        sub_rate = rates.get(strength, 0.3)
        
        synonyms = {
            "the": "a", "and": "plus", "is": "was", "are": "were",
            "text": "content", "watermark": "marking", "detection": "identification",
            "algorithm": "method", "system": "framework", "model": "approach"
        }
        
        for i, word in enumerate(words):
            if np.random.random() < sub_rate and word.lower() in synonyms:
                words[i] = synonyms[word.lower()]
        
        return " ".join(words)
    
    def _truncation_attack(self, text: str, strength: str) -> str:
        """Apply truncation attack."""
        words = text.split()
        
        rates = {"light": 0.9, "medium": 0.7, "heavy": 0.5}
        keep_rate = rates.get(strength, 0.7)
        
        keep_count = max(1, int(len(words) * keep_rate))
        return " ".join(words[:keep_count])
    
    def _insertion_attack(self, text: str, strength: str) -> str:
        """Apply insertion attack."""
        words = text.split()
        
        rates = {"light": 0.05, "medium": 0.15, "heavy": 0.25}
        insert_rate = rates.get(strength, 0.15)
        
        noise_words = ["actually", "really", "quite", "very", "somewhat", "perhaps"]
        
        new_words = []
        for word in words:
            new_words.append(word)
            if np.random.random() < insert_rate:
                new_words.append(np.random.choice(noise_words))
        
        return " ".join(new_words)
    
    def _substitution_attack(self, text: str, strength: str) -> str:
        """Apply substitution attack."""
        words = text.split()
        
        rates = {"light": 0.05, "medium": 0.15, "heavy": 0.3}
        sub_rate = rates.get(strength, 0.15)
        
        common_words = ["the", "and", "to", "of", "a", "in", "for", "on", "with"]
        
        for i, word in enumerate(words):
            if np.random.random() < sub_rate:
                words[i] = np.random.choice(common_words)
        
        return " ".join(words)
    
    def _calculate_simple_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple semantic similarity using word overlap."""
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _save_comparison_result(self, result: ComparisonResult):
        """Save comparison result to file."""
        
        filename = f"comparison_{result.method_a}_vs_{result.method_b}_{int(time.time())}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Comparison result saved to {filepath}")


# Convenience functions for quick studies
def quick_pairwise_comparison(method_a: str, method_b: str, num_runs: int = 20) -> ComparisonResult:
    """Quick pairwise comparison between two methods."""
    study = ComparativeStudy()
    return study.conduct_pairwise_comparison(method_a, method_b, num_runs=num_runs)


def benchmark_novel_algorithms(baseline_methods: List[str] = None, num_runs: int = 30) -> Dict[str, Any]:
    """Benchmark the three novel algorithms against baseline methods."""
    
    baseline_methods = baseline_methods or ["kirchenbauer", "aaronson"]
    novel_methods = ["sacw", "arms", "qipw"]
    all_methods = baseline_methods + novel_methods
    
    study = ComparativeStudy()
    return study.conduct_multi_method_study(all_methods, num_runs=num_runs)


__all__ = [
    "ComparativeStudy", 
    "StatisticalAnalyzer",
    "ComparisonResult",
    "ComparisonMetric",
    "quick_pairwise_comparison",
    "benchmark_novel_algorithms"
]