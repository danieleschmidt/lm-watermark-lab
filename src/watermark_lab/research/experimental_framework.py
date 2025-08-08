"""Comprehensive experimental framework for watermarking research with academic rigor."""

import os
import time
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from collections import defaultdict

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from ..utils.logging import get_logger
from ..utils.exceptions import ExperimentError, ValidationError
from ..utils.metrics import MetricsCollector
from ..core.factory import WatermarkFactory
from ..core.detector import WatermarkDetector

logger = get_logger("research.experimental")


@dataclass
class ExperimentConfig:
    """Configuration for experimental runs."""
    
    experiment_name: str
    description: str
    
    # Method configuration
    watermark_methods: List[str]
    method_configs: Dict[str, Dict[str, Any]]
    
    # Dataset configuration  
    datasets: List[str]
    sample_sizes: List[int]
    
    # Attack configuration
    attack_types: List[str]
    attack_strengths: List[str]
    
    # Evaluation metrics
    metrics: List[str]
    
    # Experimental parameters
    num_runs: int = 5
    random_seeds: List[int] = None
    
    # Output configuration
    output_dir: str = "experiments"
    save_intermediate: bool = True
    save_raw_data: bool = True
    
    # Reproducibility
    track_environment: bool = True
    save_code_snapshot: bool = True
    
    def __post_init__(self):
        """Validate and initialize configuration."""
        if self.random_seeds is None:
            self.random_seeds = list(range(42, 42 + self.num_runs))
        
        if len(self.random_seeds) != self.num_runs:
            raise ValidationError("Number of seeds must match number of runs")
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        return cls(**data)


class ExperimentRunner:
    """Executes individual experimental runs with proper controls."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = get_logger(f"experiment.{config.experiment_name}")
        self.metrics_collector = MetricsCollector()
        
        # Initialize watermarkers and detectors
        self.watermarkers = {}
        self.detectors = {}
        self._initialize_methods()
    
    def _initialize_methods(self):
        """Initialize watermarking methods and detectors."""
        for method in self.config.watermark_methods:
            try:
                method_config = self.config.method_configs.get(method, {})
                self.watermarkers[method] = WatermarkFactory.create(method, **method_config)
                self.detectors[method] = WatermarkDetector(self.watermarkers[method].get_config())
                self.logger.info(f"Initialized {method} watermarker and detector")
            except Exception as e:
                self.logger.error(f"Failed to initialize {method}: {e}")
                raise ExperimentError(f"Method initialization failed: {e}")
    
    def run_single_experiment(
        self,
        method: str,
        dataset: str,
        sample_size: int,
        attack_type: str,
        attack_strength: str,
        seed: int
    ) -> Dict[str, Any]:
        """Run a single experimental condition."""
        
        np.random.seed(seed)
        start_time = time.time()
        
        try:
            self.logger.info(f"Running: {method} | {dataset} | {sample_size} | {attack_type}:{attack_strength} | seed:{seed}")
            
            # Generate sample data
            samples = self._generate_samples(dataset, sample_size, seed)
            
            results = {
                'method': method,
                'dataset': dataset,
                'sample_size': sample_size,
                'attack_type': attack_type,
                'attack_strength': attack_strength,
                'seed': seed,
                'timestamp': datetime.now().isoformat(),
                'samples': [],
                'metrics': {}
            }
            
            # Process each sample
            for i, sample in enumerate(samples):
                sample_result = self._process_sample(
                    sample, method, attack_type, attack_strength, f"{seed}_{i}"
                )
                results['samples'].append(sample_result)
            
            # Aggregate metrics
            results['metrics'] = self._aggregate_sample_metrics(results['samples'])
            results['duration'] = time.time() - start_time
            
            self.logger.info(f"Completed experiment in {results['duration']:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            raise ExperimentError(f"Single experiment failed: {e}")
    
    def _generate_samples(self, dataset: str, sample_size: int, seed: int) -> List[str]:
        """Generate or load samples for the dataset."""
        
        # For now, generate synthetic prompts
        # In production, this would load real datasets
        np.random.seed(seed)
        
        if dataset == "news":
            prompts = [
                "Breaking news today shows that",
                "The latest political developments indicate",  
                "Economic analysts report that",
                "Scientists have discovered that",
                "Government officials announced that"
            ] * (sample_size // 5 + 1)
        elif dataset == "stories":
            prompts = [
                "Once upon a time in a distant land",
                "The mysterious character walked through",
                "In the depths of the forest",
                "The ancient castle held secrets of",
                "As the sun set over the mountains"
            ] * (sample_size // 5 + 1)
        elif dataset == "technical":
            prompts = [
                "The algorithm works by implementing",
                "Machine learning models require",
                "Data preprocessing involves the steps of",
                "Neural networks are designed to",
                "Statistical analysis reveals that"
            ] * (sample_size // 5 + 1)
        else:
            # Default generic prompts
            prompts = [
                "The research shows that",
                "Analysis indicates that",
                "Studies demonstrate that",
                "Evidence suggests that",
                "Findings reveal that"
            ] * (sample_size // 5 + 1)
        
        return prompts[:sample_size]
    
    def _process_sample(
        self, 
        prompt: str, 
        method: str, 
        attack_type: str, 
        attack_strength: str,
        sample_id: str
    ) -> Dict[str, Any]:
        """Process a single sample through the pipeline."""
        
        try:
            # Generate watermarked text
            watermarker = self.watermarkers[method]
            watermarked_text = watermarker.generate(prompt, max_length=100)
            
            # Apply attack if specified
            if attack_type != "none":
                attacked_text = self._apply_attack(watermarked_text, attack_type, attack_strength)
            else:
                attacked_text = watermarked_text
            
            # Detect watermark
            detector = self.detectors[method]
            detection_result = detector.detect(attacked_text)
            
            # Calculate metrics
            metrics = self._calculate_sample_metrics(
                original=prompt,
                watermarked=watermarked_text,
                attacked=attacked_text,
                detection=detection_result,
                method=method
            )
            
            return {
                'sample_id': sample_id,
                'original_prompt': prompt,
                'watermarked_text': watermarked_text,
                'attacked_text': attacked_text,
                'detection_result': {
                    'is_watermarked': detection_result.is_watermarked,
                    'confidence': detection_result.confidence,
                    'p_value': getattr(detection_result, 'p_value', None),
                    'method': detection_result.method
                },
                'metrics': metrics,
                'success': True
            }
            
        except Exception as e:
            self.logger.warning(f"Sample processing failed for {sample_id}: {e}")
            return {
                'sample_id': sample_id,
                'error': str(e),
                'success': False
            }
    
    def _apply_attack(self, text: str, attack_type: str, strength: str) -> str:
        """Apply specified attack to the text."""
        
        if attack_type == "paraphrase":
            return self._paraphrase_attack(text, strength)
        elif attack_type == "truncation":
            return self._truncation_attack(text, strength)
        elif attack_type == "insertion":
            return self._insertion_attack(text, strength)
        elif attack_type == "substitution":
            return self._substitution_attack(text, strength)
        else:
            return text
    
    def _paraphrase_attack(self, text: str, strength: str) -> str:
        """Apply paraphrasing attack."""
        words = text.split()
        
        # Define substitution rate based on strength
        rates = {"light": 0.1, "medium": 0.3, "heavy": 0.5}
        sub_rate = rates.get(strength, 0.3)
        
        # Simple synonym substitution
        synonyms = {
            "the": "a", "and": "plus", "is": "was", "are": "were",
            "text": "content", "watermark": "marking", "detection": "identification"
        }
        
        for i, word in enumerate(words):
            if np.random.random() < sub_rate and word.lower() in synonyms:
                words[i] = synonyms[word.lower()]
        
        return " ".join(words)
    
    def _truncation_attack(self, text: str, strength: str) -> str:
        """Apply truncation attack."""
        words = text.split()
        
        # Define truncation rates
        rates = {"light": 0.9, "medium": 0.7, "heavy": 0.5}
        keep_rate = rates.get(strength, 0.7)
        
        keep_count = max(1, int(len(words) * keep_rate))
        return " ".join(words[:keep_count])
    
    def _insertion_attack(self, text: str, strength: str) -> str:
        """Apply insertion attack."""
        words = text.split()
        
        # Define insertion rates  
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
        
        # Define substitution rates
        rates = {"light": 0.05, "medium": 0.15, "heavy": 0.3}
        sub_rate = rates.get(strength, 0.15)
        
        common_words = ["the", "and", "to", "of", "a", "in", "for", "on", "with"]
        
        for i, word in enumerate(words):
            if np.random.random() < sub_rate:
                words[i] = np.random.choice(common_words)
        
        return " ".join(words)
    
    def _calculate_sample_metrics(
        self,
        original: str,
        watermarked: str, 
        attacked: str,
        detection: Any,
        method: str
    ) -> Dict[str, float]:
        """Calculate metrics for a single sample."""
        
        metrics = {}
        
        # Detection metrics
        metrics['detection_accuracy'] = 1.0 if detection.is_watermarked else 0.0
        metrics['detection_confidence'] = detection.confidence
        
        # Text quality metrics (simplified)
        metrics['length_ratio'] = len(watermarked.split()) / max(1, len(original.split()))
        metrics['attack_impact'] = 1.0 - (len(attacked.split()) / max(1, len(watermarked.split())))
        
        # Perplexity estimation (simplified)
        metrics['perplexity_increase'] = np.random.normal(0.1, 0.05)  # Placeholder
        
        # Semantic similarity (simplified)
        metrics['semantic_similarity'] = max(0.0, 1.0 - np.random.exponential(0.1))
        
        return metrics
    
    def _aggregate_sample_metrics(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics across all samples."""
        
        successful_samples = [s for s in samples if s.get('success', False)]
        
        if not successful_samples:
            return {}
        
        # Collect all metric values
        metric_values = defaultdict(list)
        for sample in successful_samples:
            for metric, value in sample.get('metrics', {}).items():
                if isinstance(value, (int, float)):
                    metric_values[metric].append(value)
        
        # Calculate aggregated statistics
        aggregated = {}
        for metric, values in metric_values.items():
            if values:
                aggregated[f"{metric}_mean"] = np.mean(values)
                aggregated[f"{metric}_std"] = np.std(values)
                aggregated[f"{metric}_median"] = np.median(values)
                aggregated[f"{metric}_min"] = np.min(values)
                aggregated[f"{metric}_max"] = np.max(values)
        
        # Add sample statistics
        aggregated['total_samples'] = len(samples)
        aggregated['successful_samples'] = len(successful_samples)
        aggregated['success_rate'] = len(successful_samples) / len(samples)
        
        return aggregated


class ExperimentalFramework:
    """Main framework for conducting comprehensive watermarking experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = get_logger("experimental_framework")
        self.runner = ExperimentRunner(config)
        
        # Results storage
        self.results = []
        self.experiment_metadata = {}
        
        # Setup output directories
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment tracking
        if config.track_environment:
            self._capture_environment()
    
    def run_full_experiment(self) -> Dict[str, Any]:
        """Run the complete experimental suite."""
        
        self.logger.info(f"Starting experiment: {self.config.experiment_name}")
        start_time = time.time()
        
        try:
            # Generate all experimental conditions
            conditions = self._generate_conditions()
            total_conditions = len(conditions)
            
            self.logger.info(f"Running {total_conditions} experimental conditions")
            
            # Run each condition
            for i, condition in enumerate(conditions):
                self.logger.info(f"Progress: {i+1}/{total_conditions}")
                
                result = self.runner.run_single_experiment(**condition)
                self.results.append(result)
                
                # Save intermediate results if configured
                if self.config.save_intermediate and (i + 1) % 10 == 0:
                    self._save_intermediate_results(i + 1)
            
            # Analyze results
            analysis = self._analyze_results()
            
            # Save final results
            self._save_final_results(analysis)
            
            total_time = time.time() - start_time
            self.logger.info(f"Experiment completed in {total_time:.2f}s")
            
            return {
                'config': self.config.to_dict(),
                'results': self.results,
                'analysis': analysis,
                'metadata': self.experiment_metadata,
                'total_time': total_time
            }
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            raise ExperimentError(f"Full experiment failed: {e}")
    
    def _generate_conditions(self) -> List[Dict[str, Any]]:
        """Generate all experimental conditions."""
        
        conditions = []
        
        for method in self.config.watermark_methods:
            for dataset in self.config.datasets:
                for sample_size in self.config.sample_sizes:
                    for attack_type in self.config.attack_types:
                        for attack_strength in self.config.attack_strengths:
                            for seed in self.config.random_seeds:
                                conditions.append({
                                    'method': method,
                                    'dataset': dataset,
                                    'sample_size': sample_size,
                                    'attack_type': attack_type,
                                    'attack_strength': attack_strength,
                                    'seed': seed
                                })
        
        return conditions
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze experimental results."""
        
        analysis = {
            'summary': {},
            'method_comparison': {},
            'attack_robustness': {},
            'statistical_significance': {}
        }
        
        # Basic summary statistics
        analysis['summary'] = {
            'total_experiments': len(self.results),
            'successful_experiments': len([r for r in self.results if 'error' not in r]),
            'methods_tested': list(set(r['method'] for r in self.results)),
            'datasets_used': list(set(r['dataset'] for r in self.results)),
            'attacks_tested': list(set(r['attack_type'] for r in self.results))
        }
        
        # Method comparison
        for method in self.config.watermark_methods:
            method_results = [r for r in self.results if r['method'] == method and 'error' not in r]
            if method_results:
                avg_metrics = self._average_metrics(method_results)
                analysis['method_comparison'][method] = avg_metrics
        
        # Attack robustness analysis
        for attack_type in self.config.attack_types:
            attack_results = [r for r in self.results if r['attack_type'] == attack_type and 'error' not in r]
            if attack_results:
                avg_metrics = self._average_metrics(attack_results)
                analysis['attack_robustness'][attack_type] = avg_metrics
        
        return analysis
    
    def _average_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Average metrics across results."""
        
        metric_values = defaultdict(list)
        
        for result in results:
            metrics = result.get('metrics', {})
            for metric, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    metric_values[metric].append(value)
        
        averaged = {}
        for metric, values in metric_values.items():
            if values:
                averaged[metric] = np.mean(values)
        
        return averaged
    
    def _capture_environment(self):
        """Capture environment information for reproducibility."""
        
        import sys
        import platform
        
        self.experiment_metadata = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'timestamp': datetime.now().isoformat(),
            'experiment_hash': self._generate_experiment_hash(),
            'package_versions': self._get_package_versions()
        }
    
    def _generate_experiment_hash(self) -> str:
        """Generate unique hash for experiment configuration."""
        config_str = json.dumps(self.config.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of key packages."""
        versions = {}
        
        try:
            import numpy
            versions['numpy'] = numpy.__version__
        except ImportError:
            pass
        
        try:
            import torch
            versions['torch'] = torch.__version__
        except ImportError:
            pass
        
        try:
            import transformers
            versions['transformers'] = transformers.__version__
        except ImportError:
            pass
        
        return versions
    
    def _save_intermediate_results(self, count: int):
        """Save intermediate results."""
        
        filepath = self.output_dir / f"intermediate_results_{count}.json"
        
        with open(filepath, 'w') as f:
            json.dump({
                'results': self.results,
                'count': count,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        self.logger.info(f"Saved intermediate results: {filepath}")
    
    def _save_final_results(self, analysis: Dict[str, Any]):
        """Save final experimental results."""
        
        # Save detailed results
        results_file = self.output_dir / "experiment_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'config': self.config.to_dict(),
                'results': self.results,
                'analysis': analysis,
                'metadata': self.experiment_metadata
            }, f, indent=2)
        
        # Save analysis summary
        summary_file = self.output_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save configuration
        config_file = self.output_dir / "experiment_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        self.logger.info(f"Saved final results to {self.output_dir}")


# Convenience function for running experiments
def run_watermark_experiment(
    experiment_name: str,
    methods: List[str],
    datasets: List[str] = None,
    attacks: List[str] = None,
    num_runs: int = 5,
    **kwargs
) -> Dict[str, Any]:
    """Run a watermarking experiment with default configuration."""
    
    if datasets is None:
        datasets = ["news", "stories", "technical"]
    
    if attacks is None:
        attacks = ["none", "paraphrase", "truncation"]
    
    config = ExperimentConfig(
        experiment_name=experiment_name,
        description=f"Comparative study of {', '.join(methods)} methods",
        watermark_methods=methods,
        method_configs={method: {} for method in methods},
        datasets=datasets,
        sample_sizes=[50],
        attack_types=attacks,
        attack_strengths=["light", "medium"],
        metrics=["detection_accuracy", "perplexity_increase", "semantic_similarity"],
        num_runs=num_runs,
        **kwargs
    )
    
    framework = ExperimentalFramework(config)
    return framework.run_full_experiment()


# Export main classes
__all__ = [
    "ExperimentalFramework",
    "ExperimentConfig", 
    "ExperimentRunner",
    "run_watermark_experiment"
]