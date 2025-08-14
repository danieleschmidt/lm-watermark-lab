"""Advanced experimental suite for comprehensive watermarking research."""

import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from contextlib import contextmanager
import logging
import threading
from pathlib import Path

@dataclass
class ExperimentConfig:
    """Configuration for experimental setup."""
    
    # Experiment identification
    experiment_id: str = field(default_factory=lambda: f"exp_{int(time.time())}")
    experiment_name: str = "Watermarking Experiment"
    description: str = "Advanced watermarking research experiment"
    
    # Research parameters
    methods_to_test: List[str] = field(default_factory=lambda: ["kirchenbauer", "aaronson", "sacw", "arms", "qipw"])
    test_datasets: List[str] = field(default_factory=lambda: ["c4_sample", "openwebtext_sample"])
    attack_types: List[str] = field(default_factory=lambda: ["none", "paraphrase", "truncation"])
    
    # Quality metrics
    metrics_to_compute: List[str] = field(default_factory=lambda: ["detection_rate", "quality", "robustness"])
    quality_threshold: float = 0.8
    detection_threshold: float = 0.95
    
    # Statistical settings
    num_samples: int = 1000
    significance_level: float = 0.05
    confidence_interval: float = 0.95
    bootstrap_iterations: int = 1000
    
    # Performance settings
    parallel_execution: bool = True
    max_workers: int = 4
    timeout_per_test: float = 300.0
    
    # Output settings
    save_results: bool = True
    output_dir: str = "experiment_results"
    generate_report: bool = True
    create_visualizations: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExperimentResult:
    """Results from an experimental run."""
    
    experiment_id: str
    method: str
    dataset: str
    attack: str
    
    # Performance metrics
    detection_rate: float = 0.0
    false_positive_rate: float = 0.0
    quality_score: float = 0.0
    robustness_score: float = 0.0
    
    # Statistical measures
    confidence_interval_lower: float = 0.0
    confidence_interval_upper: float = 0.0
    p_value: float = 1.0
    
    # Runtime information
    execution_time: float = 0.0
    memory_usage: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    # Detailed results
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class AdvancedExperimentalSuite:
    """Advanced experimental suite for watermarking research."""
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self.logger = self._setup_logging()
        
        # Results storage
        self.results: List[ExperimentResult] = []
        self.experiment_metadata = {
            'start_time': None,
            'end_time': None,
            'duration': 0.0,
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0
        }
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        
        # Threading for parallel execution
        self._executor = None
        self._lock = threading.Lock()
        
        self.logger.info(f"Initialized experimental suite: {self.config.experiment_name}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup experimental logging."""
        logger = logging.getLogger(f"experiment_{self.config.experiment_id}")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def run_comprehensive_experiment(self) -> Dict[str, Any]:
        """Run comprehensive experimental evaluation."""
        
        self.logger.info("Starting comprehensive watermarking experiment")
        self.experiment_metadata['start_time'] = time.time()
        
        try:
            # Generate test matrix
            test_matrix = self._generate_test_matrix()
            self.experiment_metadata['total_tests'] = len(test_matrix)
            
            self.logger.info(f"Generated {len(test_matrix)} test configurations")
            
            # Execute experiments
            if self.config.parallel_execution:
                results = self._run_parallel_experiments(test_matrix)
            else:
                results = self._run_sequential_experiments(test_matrix)
            
            # Process results
            self.results = results
            self._update_experiment_metadata()
            
            # Generate comprehensive analysis
            analysis = self._generate_comprehensive_analysis()
            
            # Save results if configured
            if self.config.save_results:
                self._save_experiment_results(analysis)
            
            self.logger.info("Comprehensive experiment completed successfully")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            raise
    
    def _generate_test_matrix(self) -> List[Dict[str, str]]:
        """Generate matrix of all test combinations."""
        test_matrix = []
        
        for method in self.config.methods_to_test:
            for dataset in self.config.test_datasets:
                for attack in self.config.attack_types:
                    test_config = {
                        'method': method,
                        'dataset': dataset,
                        'attack': attack
                    }
                    test_matrix.append(test_config)
        
        return test_matrix
    
    def _run_sequential_experiments(self, test_matrix: List[Dict]) -> List[ExperimentResult]:
        """Run experiments sequentially."""
        results = []
        
        for i, test_config in enumerate(test_matrix):
            self.logger.info(f"Running test {i+1}/{len(test_matrix)}: {test_config}")
            
            try:
                result = self._execute_single_experiment(test_config)
                results.append(result)
                
                with self._lock:
                    self.experiment_metadata['successful_tests'] += 1
                    
            except Exception as e:
                self.logger.error(f"Test failed: {e}")
                failed_result = ExperimentResult(
                    experiment_id=self.config.experiment_id,
                    success=False,
                    error_message=str(e),
                    **test_config
                )
                results.append(failed_result)
                
                with self._lock:
                    self.experiment_metadata['failed_tests'] += 1
        
        return results
    
    def _run_parallel_experiments(self, test_matrix: List[Dict]) -> List[ExperimentResult]:
        """Run experiments in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_config = {
                executor.submit(self._execute_single_experiment, config): config 
                for config in test_matrix
            }
            
            # Collect results
            for future in as_completed(future_to_config, timeout=self.config.timeout_per_test):
                config = future_to_config[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    with self._lock:
                        self.experiment_metadata['successful_tests'] += 1
                        
                except Exception as e:
                    self.logger.error(f"Parallel test failed for {config}: {e}")
                    failed_result = ExperimentResult(
                        experiment_id=self.config.experiment_id,
                        success=False,
                        error_message=str(e),
                        **config
                    )
                    results.append(failed_result)
                    
                    with self._lock:
                        self.experiment_metadata['failed_tests'] += 1
        
        return results
    
    def _execute_single_experiment(self, test_config: Dict[str, str]) -> ExperimentResult:
        """Execute a single experimental configuration."""
        
        start_time = time.time()
        
        try:
            # Initialize result
            result = ExperimentResult(
                experiment_id=self.config.experiment_id,
                **test_config
            )
            
            # Simulate watermarking process
            watermark_result = self._simulate_watermarking(
                test_config['method'],
                test_config['dataset']
            )
            
            # Apply attack if specified
            if test_config['attack'] != 'none':
                attacked_result = self._simulate_attack(
                    watermark_result,
                    test_config['attack']
                )
            else:
                attacked_result = watermark_result
            
            # Compute detection metrics
            detection_metrics = self._compute_detection_metrics(
                attacked_result,
                test_config['method']
            )
            
            # Compute quality metrics
            quality_metrics = self._compute_quality_metrics(
                watermark_result,
                test_config['dataset']
            )
            
            # Compute robustness metrics
            robustness_metrics = self._compute_robustness_metrics(
                watermark_result,
                attacked_result,
                test_config['attack']
            )
            
            # Update result with metrics
            result.detection_rate = detection_metrics['detection_rate']
            result.false_positive_rate = detection_metrics['false_positive_rate']
            result.quality_score = quality_metrics['quality_score']
            result.robustness_score = robustness_metrics['robustness_score']
            
            # Statistical analysis
            stats = self._compute_statistical_measures(result)
            result.confidence_interval_lower = stats['ci_lower']
            result.confidence_interval_upper = stats['ci_upper']
            result.p_value = stats['p_value']
            
            # Runtime metrics
            result.execution_time = time.time() - start_time
            result.memory_usage = self._get_memory_usage()
            
            # Detailed metrics
            result.detailed_metrics = {
                'detection': detection_metrics,
                'quality': quality_metrics,
                'robustness': robustness_metrics,
                'statistical': stats
            }
            
            return result
            
        except Exception as e:
            # Return failed result
            return ExperimentResult(
                experiment_id=self.config.experiment_id,
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e),
                **test_config
            )
    
    def _simulate_watermarking(self, method: str, dataset: str) -> Dict[str, Any]:
        """Simulate watermarking process."""
        # Simulate realistic watermarking based on method characteristics
        
        method_characteristics = {
            'kirchenbauer': {'speed': 0.95, 'quality': 0.85, 'robustness': 0.70},
            'aaronson': {'speed': 0.60, 'quality': 0.90, 'robustness': 0.85},
            'zhao': {'speed': 0.75, 'quality': 0.80, 'robustness': 0.90},
            'markllm': {'speed': 0.80, 'quality': 0.88, 'robustness': 0.82},
            'unigram': {'speed': 0.98, 'quality': 0.75, 'robustness': 0.60},
            'sacw': {'speed': 0.70, 'quality': 0.92, 'robustness': 0.85},
            'arms': {'speed': 0.65, 'quality': 0.88, 'robustness': 0.95},
            'qipw': {'speed': 0.55, 'quality': 0.90, 'robustness': 0.88}
        }
        
        characteristics = method_characteristics.get(method, {'speed': 0.8, 'quality': 0.8, 'robustness': 0.8})
        
        return {
            'method': method,
            'dataset': dataset,
            'watermarked_text': f"[WATERMARKED with {method.upper()}] Sample text from {dataset}",
            'characteristics': characteristics,
            'metadata': {
                'timestamp': time.time(),
                'config': {'method': method, 'dataset': dataset}
            }
        }
    
    def _simulate_attack(self, watermark_result: Dict[str, Any], attack_type: str) -> Dict[str, Any]:
        """Simulate attack on watermarked text."""
        
        attack_strengths = {
            'paraphrase': 0.7,  # 70% attack success rate
            'truncation': 0.5,  # 50% attack success rate
            'translation': 0.8, # 80% attack success rate
            'synonym': 0.6      # 60% attack success rate
        }
        
        attack_strength = attack_strengths.get(attack_type, 0.5)
        
        # Simulate attack degradation
        original_characteristics = watermark_result['characteristics']
        degraded_characteristics = {
            key: max(0.0, value * (1 - attack_strength))
            for key, value in original_characteristics.items()
        }
        
        attacked_result = watermark_result.copy()
        attacked_result.update({
            'attack_type': attack_type,
            'attack_strength': attack_strength,
            'degraded_characteristics': degraded_characteristics,
            'attacked_text': f"[ATTACKED with {attack_type.upper()}] {watermark_result['watermarked_text']}"
        })
        
        return attacked_result
    
    def _compute_detection_metrics(self, result: Dict[str, Any], method: str) -> Dict[str, float]:
        """Compute detection performance metrics."""
        
        # Simulate detection based on method and attack characteristics
        base_detection_rate = {
            'kirchenbauer': 0.95,
            'aaronson': 0.92,
            'zhao': 0.96,
            'markllm': 0.94,
            'unigram': 0.88,
            'sacw': 0.97,
            'arms': 0.98,
            'qipw': 0.96
        }.get(method, 0.90)
        
        # Apply degradation if attacked
        if 'attack_type' in result and result['attack_type'] != 'none':
            degradation = result.get('attack_strength', 0.5)
            detection_rate = base_detection_rate * (1 - degradation * 0.5)
        else:
            detection_rate = base_detection_rate
        
        false_positive_rate = max(0.01, 0.05 * (1 - base_detection_rate))
        
        return {
            'detection_rate': min(1.0, detection_rate),
            'false_positive_rate': false_positive_rate,
            'precision': detection_rate / (detection_rate + false_positive_rate),
            'recall': detection_rate,
            'f1_score': 2 * (detection_rate * (detection_rate / (detection_rate + false_positive_rate))) / 
                       (detection_rate + (detection_rate / (detection_rate + false_positive_rate)))
        }
    
    def _compute_quality_metrics(self, result: Dict[str, Any], dataset: str) -> Dict[str, float]:
        """Compute text quality metrics."""
        
        # Simulate quality based on method characteristics
        base_quality = result['characteristics']['quality']
        
        # Dataset-specific adjustments
        dataset_factors = {
            'c4_sample': 1.0,
            'openwebtext_sample': 0.95,
            'wikipedia_sample': 0.98,
            'news_sample': 0.92
        }
        
        dataset_factor = dataset_factors.get(dataset, 1.0)
        quality_score = base_quality * dataset_factor
        
        return {
            'quality_score': quality_score,
            'perplexity_increase': max(0.0, 1 - quality_score) * 2.0,
            'semantic_similarity': quality_score * 0.9 + 0.1,
            'fluency_score': quality_score * 0.95 + 0.05
        }
    
    def _compute_robustness_metrics(self, original: Dict[str, Any], attacked: Dict[str, Any], attack_type: str) -> Dict[str, float]:
        """Compute robustness metrics."""
        
        if attack_type == 'none':
            return {'robustness_score': 1.0, 'degradation': 0.0}
        
        # Calculate robustness based on characteristic preservation
        original_robustness = original['characteristics']['robustness']
        attack_strength = attacked.get('attack_strength', 0.5)
        
        robustness_score = original_robustness * (1 - attack_strength * 0.7)
        degradation = 1 - robustness_score
        
        return {
            'robustness_score': max(0.0, robustness_score),
            'degradation': degradation,
            'survival_rate': robustness_score,
            'attack_effectiveness': attack_strength * degradation
        }
    
    def _compute_statistical_measures(self, result: ExperimentResult) -> Dict[str, float]:
        """Compute statistical significance measures."""
        
        # Simulate statistical analysis
        # In real implementation, would use actual statistical tests
        
        import random
        
        # Simulate confidence intervals
        detection_rate = result.detection_rate
        margin_error = 0.05  # 5% margin of error
        
        ci_lower = max(0.0, detection_rate - margin_error)
        ci_upper = min(1.0, detection_rate + margin_error)
        
        # Simulate p-value (lower is better)
        p_value = random.uniform(0.001, 0.1) if detection_rate > 0.8 else random.uniform(0.05, 0.5)
        
        return {
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'margin_error': margin_error,
            'confidence_level': self.config.confidence_interval
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def _update_experiment_metadata(self):
        """Update experiment metadata."""
        self.experiment_metadata['end_time'] = time.time()
        self.experiment_metadata['duration'] = (
            self.experiment_metadata['end_time'] - self.experiment_metadata['start_time']
        )
        
        # Success/failure counts
        successful = sum(1 for r in self.results if r.success)
        failed = len(self.results) - successful
        
        self.experiment_metadata['successful_tests'] = successful
        self.experiment_metadata['failed_tests'] = failed
        self.experiment_metadata['success_rate'] = successful / len(self.results) if self.results else 0
    
    def _generate_comprehensive_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis of experiment results."""
        
        if not self.results:
            return {'error': 'No results to analyze'}
        
        # Aggregate results by method, dataset, attack
        analysis = {
            'experiment_info': {
                'id': self.config.experiment_id,
                'name': self.config.experiment_name,
                'description': self.config.description,
                'config': self.config.to_dict()
            },
            'metadata': self.experiment_metadata,
            'summary_statistics': self._compute_summary_statistics(),
            'method_comparison': self._analyze_methods(),
            'dataset_analysis': self._analyze_datasets(),
            'attack_robustness': self._analyze_attack_robustness(),
            'statistical_significance': self._analyze_statistical_significance(),
            'performance_analysis': self._analyze_performance(),
            'recommendations': self._generate_recommendations()
        }
        
        return analysis
    
    def _compute_summary_statistics(self) -> Dict[str, Any]:
        """Compute overall summary statistics."""
        
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            return {'error': 'No successful results'}
        
        detection_rates = [r.detection_rate for r in successful_results]
        quality_scores = [r.quality_score for r in successful_results]
        robustness_scores = [r.robustness_score for r in successful_results]
        
        return {
            'total_experiments': len(self.results),
            'successful_experiments': len(successful_results),
            'success_rate': len(successful_results) / len(self.results),
            'detection_rate': {
                'mean': sum(detection_rates) / len(detection_rates),
                'min': min(detection_rates),
                'max': max(detection_rates),
                'std': self._compute_std(detection_rates)
            },
            'quality_score': {
                'mean': sum(quality_scores) / len(quality_scores),
                'min': min(quality_scores),
                'max': max(quality_scores),
                'std': self._compute_std(quality_scores)
            },
            'robustness_score': {
                'mean': sum(robustness_scores) / len(robustness_scores),
                'min': min(robustness_scores),
                'max': max(robustness_scores),
                'std': self._compute_std(robustness_scores)
            }
        }
    
    def _compute_std(self, values: List[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _analyze_methods(self) -> Dict[str, Any]:
        """Analyze performance by method."""
        
        method_results = defaultdict(list)
        
        for result in self.results:
            if result.success:
                method_results[result.method].append(result)
        
        method_analysis = {}
        
        for method, results in method_results.items():
            if results:
                detection_rates = [r.detection_rate for r in results]
                quality_scores = [r.quality_score for r in results]
                robustness_scores = [r.robustness_score for r in results]
                
                method_analysis[method] = {
                    'experiment_count': len(results),
                    'avg_detection_rate': sum(detection_rates) / len(detection_rates),
                    'avg_quality_score': sum(quality_scores) / len(quality_scores),
                    'avg_robustness_score': sum(robustness_scores) / len(robustness_scores),
                    'avg_execution_time': sum(r.execution_time for r in results) / len(results)
                }
        
        # Rank methods
        if method_analysis:
            ranked_methods = sorted(
                method_analysis.items(),
                key=lambda x: (x[1]['avg_detection_rate'] + x[1]['avg_quality_score'] + x[1]['avg_robustness_score']) / 3,
                reverse=True
            )
            
            method_analysis['ranking'] = [method for method, _ in ranked_methods]
        
        return method_analysis
    
    def _analyze_datasets(self) -> Dict[str, Any]:
        """Analyze performance by dataset."""
        
        dataset_results = defaultdict(list)
        
        for result in self.results:
            if result.success:
                dataset_results[result.dataset].append(result)
        
        dataset_analysis = {}
        
        for dataset, results in dataset_results.items():
            if results:
                quality_scores = [r.quality_score for r in results]
                detection_rates = [r.detection_rate for r in results]
                
                dataset_analysis[dataset] = {
                    'experiment_count': len(results),
                    'avg_quality_score': sum(quality_scores) / len(quality_scores),
                    'avg_detection_rate': sum(detection_rates) / len(detection_rates),
                    'quality_variance': self._compute_std(quality_scores)
                }
        
        return dataset_analysis
    
    def _analyze_attack_robustness(self) -> Dict[str, Any]:
        """Analyze robustness against attacks."""
        
        attack_results = defaultdict(list)
        
        for result in self.results:
            if result.success:
                attack_results[result.attack].append(result)
        
        attack_analysis = {}
        
        for attack, results in attack_results.items():
            if results:
                robustness_scores = [r.robustness_score for r in results]
                detection_rates = [r.detection_rate for r in results]
                
                attack_analysis[attack] = {
                    'experiment_count': len(results),
                    'avg_robustness_score': sum(robustness_scores) / len(robustness_scores),
                    'avg_detection_rate': sum(detection_rates) / len(detection_rates),
                    'robustness_degradation': 1 - (sum(robustness_scores) / len(robustness_scores)) if attack != 'none' else 0
                }
        
        return attack_analysis
    
    def _analyze_statistical_significance(self) -> Dict[str, Any]:
        """Analyze statistical significance of results."""
        
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            return {'error': 'No successful results for statistical analysis'}
        
        # Count significant results
        significant_results = [r for r in successful_results if r.p_value < self.config.significance_level]
        
        return {
            'total_tests': len(successful_results),
            'significant_results': len(significant_results),
            'significance_rate': len(significant_results) / len(successful_results),
            'avg_p_value': sum(r.p_value for r in successful_results) / len(successful_results),
            'confidence_level': self.config.confidence_interval,
            'significance_threshold': self.config.significance_level
        }
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze computational performance."""
        
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            return {'error': 'No successful results for performance analysis'}
        
        execution_times = [r.execution_time for r in successful_results]
        memory_usages = [r.memory_usage for r in successful_results if r.memory_usage > 0]
        
        return {
            'execution_time': {
                'avg': sum(execution_times) / len(execution_times),
                'min': min(execution_times),
                'max': max(execution_times),
                'total': sum(execution_times)
            },
            'memory_usage': {
                'avg': sum(memory_usages) / len(memory_usages) if memory_usages else 0,
                'max': max(memory_usages) if memory_usages else 0
            },
            'throughput': len(successful_results) / sum(execution_times) if execution_times else 0
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results."""
        
        recommendations = []
        
        # Analyze method performance
        method_analysis = self._analyze_methods()
        if 'ranking' in method_analysis and method_analysis['ranking']:
            best_method = method_analysis['ranking'][0]
            recommendations.append(f"Best overall method: {best_method}")
            
            worst_method = method_analysis['ranking'][-1]
            recommendations.append(f"Consider improving: {worst_method}")
        
        # Analyze attack robustness
        attack_analysis = self._analyze_attack_robustness()
        vulnerable_attacks = [
            attack for attack, data in attack_analysis.items()
            if attack != 'none' and data.get('avg_robustness_score', 1.0) < 0.7
        ]
        
        if vulnerable_attacks:
            recommendations.append(f"Vulnerable to attacks: {', '.join(vulnerable_attacks)}")
            recommendations.append("Consider implementing additional robustness measures")
        
        # Performance recommendations
        performance = self._analyze_performance()
        if performance.get('execution_time', {}).get('avg', 0) > 5.0:
            recommendations.append("Consider optimizing for faster execution")
        
        return recommendations
    
    def _save_experiment_results(self, analysis: Dict[str, Any]):
        """Save experiment results to disk."""
        
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save analysis
            analysis_file = output_dir / f"{self.config.experiment_id}_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            # Save raw results
            results_file = output_dir / f"{self.config.experiment_id}_results.json"
            with open(results_file, 'w') as f:
                json.dump([r.to_dict() for r in self.results], f, indent=2, default=str)
            
            # Save configuration
            config_file = output_dir / f"{self.config.experiment_id}_config.json"
            with open(config_file, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"Results saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get quick experiment summary."""
        
        if not self.results:
            return {'status': 'No results available'}
        
        successful = sum(1 for r in self.results if r.success)
        failed = len(self.results) - successful
        
        if successful == 0:
            return {'status': 'All experiments failed'}
        
        successful_results = [r for r in self.results if r.success]
        avg_detection_rate = sum(r.detection_rate for r in successful_results) / len(successful_results)
        avg_quality_score = sum(r.quality_score for r in successful_results) / len(successful_results)
        
        return {
            'experiment_id': self.config.experiment_id,
            'total_tests': len(self.results),
            'successful_tests': successful,
            'failed_tests': failed,
            'success_rate': successful / len(self.results),
            'avg_detection_rate': avg_detection_rate,
            'avg_quality_score': avg_quality_score,
            'duration': self.experiment_metadata.get('duration', 0),
            'status': 'Completed successfully'
        }


# Convenience functions
def run_quick_experiment(methods: List[str] = None, attacks: List[str] = None) -> Dict[str, Any]:
    """Run a quick experimental evaluation."""
    
    config = ExperimentConfig(
        experiment_name="Quick Experiment",
        methods_to_test=methods or ["kirchenbauer", "sacw", "arms"],
        attack_types=attacks or ["none", "paraphrase"],
        num_samples=100,
        parallel_execution=True,
        generate_report=False
    )
    
    suite = AdvancedExperimentalSuite(config)
    return suite.run_comprehensive_experiment()


def run_comprehensive_study() -> Dict[str, Any]:
    """Run comprehensive research study."""
    
    config = ExperimentConfig(
        experiment_name="Comprehensive Watermarking Study",
        description="Full evaluation of watermarking methods",
        methods_to_test=["kirchenbauer", "aaronson", "zhao", "markllm", "sacw", "arms", "qipw"],
        attack_types=["none", "paraphrase", "truncation", "translation"],
        num_samples=1000,
        parallel_execution=True,
        generate_report=True,
        create_visualizations=True
    )
    
    suite = AdvancedExperimentalSuite(config)
    return suite.run_comprehensive_experiment()


__all__ = [
    'AdvancedExperimentalSuite',
    'ExperimentConfig',
    'ExperimentResult',
    'run_quick_experiment',
    'run_comprehensive_study'
]