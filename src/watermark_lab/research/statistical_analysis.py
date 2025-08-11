"""
Statistical analysis tools for watermarking research.

This module provides comprehensive statistical analysis capabilities
specifically designed for watermarking research, including significance
testing, power analysis, and effect size calculations.
"""

import numpy as np
import time
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import json
from pathlib import Path

try:
    import scipy.stats as stats
    from scipy.stats import chi2_contingency, fisher_exact
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Fallback implementations
    class stats:
        @staticmethod
        def ttest_1samp(a, popmean):
            """One-sample t-test fallback"""
            mean_a = np.mean(a)
            std_a = np.std(a, ddof=1)
            n = len(a)
            t_stat = (mean_a - popmean) / (std_a / np.sqrt(n))
            
            # Approximate p-value using normal distribution
            z = abs(t_stat)
            p_value = 2 * (1 - 0.5 * (1 + np.sign(z) * np.sqrt(z**2 / (z**2 + n - 1))))
            
            class TTestResult:
                def __init__(self, statistic, pvalue):
                    self.statistic = statistic
                    self.pvalue = pvalue
            
            return TTestResult(t_stat, p_value)
        
        @staticmethod
        def chi2_contingency(observed):
            """Chi-square test fallback"""
            observed = np.array(observed)
            
            # Calculate expected frequencies
            row_sums = observed.sum(axis=1)
            col_sums = observed.sum(axis=0)
            total = observed.sum()
            
            expected = np.outer(row_sums, col_sums) / total
            
            # Chi-square statistic
            chi2_stat = ((observed - expected) ** 2 / expected).sum()
            
            # Degrees of freedom
            df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
            
            # Approximate p-value
            p_value = 1 - (chi2_stat / (chi2_stat + df))
            
            return chi2_stat, p_value, df, expected
        
        @staticmethod
        def fisher_exact(table):
            """Fisher exact test fallback"""
            a, b, c, d = table[0][0], table[0][1], table[1][0], table[1][1]
            n = a + b + c + d
            
            # Simplified Fisher exact test
            # Calculate odds ratio
            if b == 0 or c == 0:
                odds_ratio = float('inf') if a * d > 0 else 0
            else:
                odds_ratio = (a * d) / (b * c)
            
            # Approximate p-value using hypergeometric distribution
            p_value = 0.5  # Placeholder
            
            return odds_ratio, p_value
        
        @staticmethod
        def normaltest(a):
            """Normality test fallback"""
            # Simple normality test using skewness and kurtosis
            mean_a = np.mean(a)
            std_a = np.std(a)
            
            if std_a == 0:
                return 0, 1.0
            
            # Calculate skewness
            skew = np.mean(((a - mean_a) / std_a) ** 3)
            # Calculate kurtosis
            kurt = np.mean(((a - mean_a) / std_a) ** 4) - 3
            
            # Combined test statistic
            stat = (skew ** 2) / 6 + (kurt ** 2) / 24
            p_value = max(0.01, 1 - stat)  # Approximate p-value
            
            return stat, p_value

from ..utils.logging import get_logger

logger = get_logger("research.statistical")


@dataclass
class StatisticalTest:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[int]
    critical_value: Optional[float]
    effect_size: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    interpretation: str
    significant: bool
    power: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a method."""
    method_name: str
    sample_size: int
    detection_rate: float
    false_positive_rate: float
    false_negative_rate: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float]
    confidence_scores: List[float]
    processing_times: List[float]
    semantic_similarities: List[float]
    

class StatisticalAnalyzer:
    """Advanced statistical analyzer for watermarking research."""
    
    def __init__(self, alpha: float = 0.05, power_threshold: float = 0.8):
        """
        Initialize statistical analyzer.
        
        Args:
            alpha: Significance level for hypothesis tests
            power_threshold: Minimum statistical power threshold
        """
        self.alpha = alpha
        self.power_threshold = power_threshold
        
        # Critical values for common tests
        self.critical_values = {
            0.05: 1.96,  # 95% confidence
            0.01: 2.58,  # 99% confidence
            0.001: 3.29  # 99.9% confidence
        }
    
    def detection_rate_test(self, 
                          watermarked_detections: List[bool],
                          unwatermarked_detections: List[bool],
                          expected_detection_rate: float = 0.95) -> StatisticalTest:
        """
        Test if detection rate meets the expected threshold.
        
        Args:
            watermarked_detections: Detection results for watermarked texts
            unwatermarked_detections: Detection results for unwatermarked texts
            expected_detection_rate: Expected detection rate threshold
            
        Returns:
            Statistical test results
        """
        try:
            watermarked_rate = np.mean(watermarked_detections)
            unwatermarked_rate = np.mean(unwatermarked_detections)
            
            # One-sample t-test for watermarked detection rate
            watermarked_array = np.array(watermarked_detections, dtype=float)
            t_stat, p_value = stats.ttest_1samp(watermarked_array, expected_detection_rate)
            
            # Effect size (Cohen's d)
            effect_size = (watermarked_rate - expected_detection_rate) / np.std(watermarked_array)
            
            # Confidence interval
            se = np.std(watermarked_array) / np.sqrt(len(watermarked_array))
            margin = self.critical_values[self.alpha] * se
            ci = (watermarked_rate - margin, watermarked_rate + margin)
            
            # Interpretation
            if p_value < self.alpha and watermarked_rate >= expected_detection_rate:
                interpretation = f"Detection rate ({watermarked_rate:.3f}) significantly meets threshold"
            elif p_value < self.alpha:
                interpretation = f"Detection rate ({watermarked_rate:.3f}) significantly below threshold"
            else:
                interpretation = f"Detection rate ({watermarked_rate:.3f}) not significantly different from threshold"
            
            return StatisticalTest(
                test_name="detection_rate_test",
                statistic=t_stat,
                p_value=p_value,
                degrees_of_freedom=len(watermarked_detections) - 1,
                critical_value=self.critical_values[self.alpha],
                effect_size=effect_size,
                confidence_interval=ci,
                interpretation=interpretation,
                significant=p_value < self.alpha
            )
            
        except Exception as e:
            logger.error(f"Detection rate test failed: {e}")
            return self._create_failed_test("detection_rate_test")
    
    def false_positive_test(self,
                          unwatermarked_detections: List[bool],
                          max_false_positive_rate: float = 0.05) -> StatisticalTest:
        """
        Test if false positive rate is within acceptable bounds.
        
        Args:
            unwatermarked_detections: Detection results for unwatermarked texts
            max_false_positive_rate: Maximum acceptable false positive rate
            
        Returns:
            Statistical test results
        """
        try:
            fp_rate = np.mean(unwatermarked_detections)
            n = len(unwatermarked_detections)
            
            # Binomial test for false positive rate
            # Using normal approximation for large samples
            expected_fp = max_false_positive_rate
            se = np.sqrt(expected_fp * (1 - expected_fp) / n)
            
            if se > 0:
                z_stat = (fp_rate - expected_fp) / se
                p_value = 2 * (1 - 0.5 * (1 + np.sign(z_stat) * np.sqrt(z_stat**2 / (z_stat**2 + 1))))
            else:
                z_stat = 0
                p_value = 1.0
            
            # Effect size
            effect_size = (fp_rate - expected_fp) / np.sqrt(expected_fp * (1 - expected_fp))
            
            # Confidence interval for proportion
            margin = self.critical_values[self.alpha] * np.sqrt(fp_rate * (1 - fp_rate) / n)
            ci = (max(0, fp_rate - margin), min(1, fp_rate + margin))
            
            # Interpretation
            if fp_rate <= max_false_positive_rate:
                interpretation = f"False positive rate ({fp_rate:.3f}) within acceptable bounds"
            else:
                interpretation = f"False positive rate ({fp_rate:.3f}) exceeds threshold ({max_false_positive_rate})"
            
            return StatisticalTest(
                test_name="false_positive_test",
                statistic=z_stat,
                p_value=p_value,
                degrees_of_freedom=n - 1,
                critical_value=self.critical_values[self.alpha],
                effect_size=effect_size,
                confidence_interval=ci,
                interpretation=interpretation,
                significant=p_value < self.alpha
            )
            
        except Exception as e:
            logger.error(f"False positive test failed: {e}")
            return self._create_failed_test("false_positive_test")
    
    def semantic_preservation_test(self,
                                 semantic_similarities: List[float],
                                 min_similarity: float = 0.8) -> StatisticalTest:
        """
        Test if semantic similarity meets preservation requirements.
        
        Args:
            semantic_similarities: List of semantic similarity scores
            min_similarity: Minimum required semantic similarity
            
        Returns:
            Statistical test results
        """
        try:
            similarities = np.array(semantic_similarities)
            mean_similarity = np.mean(similarities)
            
            # One-sample t-test
            t_stat, p_value = stats.ttest_1samp(similarities, min_similarity)
            
            # Effect size (Cohen's d)
            effect_size = (mean_similarity - min_similarity) / np.std(similarities)
            
            # Confidence interval
            se = np.std(similarities) / np.sqrt(len(similarities))
            margin = self.critical_values[self.alpha] * se
            ci = (mean_similarity - margin, mean_similarity + margin)
            
            # Interpretation
            if p_value < self.alpha and mean_similarity >= min_similarity:
                interpretation = f"Semantic similarity ({mean_similarity:.3f}) significantly preserves content"
            elif p_value < self.alpha:
                interpretation = f"Semantic similarity ({mean_similarity:.3f}) significantly below threshold"
            else:
                interpretation = f"Semantic similarity ({mean_similarity:.3f}) not significantly different from threshold"
            
            return StatisticalTest(
                test_name="semantic_preservation_test",
                statistic=t_stat,
                p_value=p_value,
                degrees_of_freedom=len(similarities) - 1,
                critical_value=self.critical_values[self.alpha],
                effect_size=effect_size,
                confidence_interval=ci,
                interpretation=interpretation,
                significant=p_value < self.alpha
            )
            
        except Exception as e:
            logger.error(f"Semantic preservation test failed: {e}")
            return self._create_failed_test("semantic_preservation_test")
    
    def robustness_analysis(self,
                          clean_detections: List[bool],
                          attacked_detections: List[bool],
                          attack_type: str) -> StatisticalTest:
        """
        Analyze robustness against specific attacks.
        
        Args:
            clean_detections: Detection results on clean watermarked text
            attacked_detections: Detection results on attacked watermarked text
            attack_type: Type of attack applied
            
        Returns:
            Statistical test results for robustness
        """
        try:
            clean_rate = np.mean(clean_detections)
            attacked_rate = np.mean(attacked_detections)
            
            # McNemar's test for paired proportions
            # Create contingency table
            both_detected = sum(1 for c, a in zip(clean_detections, attacked_detections) if c and a)
            clean_only = sum(1 for c, a in zip(clean_detections, attacked_detections) if c and not a)
            attacked_only = sum(1 for c, a in zip(clean_detections, attacked_detections) if not c and a)
            neither_detected = sum(1 for c, a in zip(clean_detections, attacked_detections) if not c and not a)
            
            # McNemar statistic
            if clean_only + attacked_only > 0:
                mcnemar_stat = (abs(clean_only - attacked_only) - 1)**2 / (clean_only + attacked_only)
                p_value = 1 - mcnemar_stat / (mcnemar_stat + 1)  # Approximation
            else:
                mcnemar_stat = 0
                p_value = 1.0
            
            # Effect size (difference in proportions)
            effect_size = clean_rate - attacked_rate
            
            # Robustness percentage
            robustness = attacked_rate / clean_rate if clean_rate > 0 else 0
            
            # Interpretation
            if robustness >= 0.9:
                interpretation = f"Highly robust against {attack_type} (retention: {robustness:.1%})"
            elif robustness >= 0.7:
                interpretation = f"Moderately robust against {attack_type} (retention: {robustness:.1%})"
            else:
                interpretation = f"Low robustness against {attack_type} (retention: {robustness:.1%})"
            
            return StatisticalTest(
                test_name=f"robustness_{attack_type}",
                statistic=mcnemar_stat,
                p_value=p_value,
                degrees_of_freedom=1,
                critical_value=3.84,  # Chi-square critical value for alpha=0.05
                effect_size=effect_size,
                confidence_interval=None,
                interpretation=interpretation,
                significant=p_value < self.alpha
            )
            
        except Exception as e:
            logger.error(f"Robustness analysis failed for {attack_type}: {e}")
            return self._create_failed_test(f"robustness_{attack_type}")
    
    def power_analysis(self,
                     sample_sizes: List[int],
                     effect_size: float,
                     test_type: str = "t_test") -> Dict[str, Any]:
        """
        Conduct power analysis for different sample sizes.
        
        Args:
            sample_sizes: List of sample sizes to analyze
            effect_size: Expected effect size
            test_type: Type of statistical test
            
        Returns:
            Power analysis results
        """
        try:
            power_results = {}
            
            for n in sample_sizes:
                if test_type == "t_test":
                    # Approximate power calculation for t-test
                    ncp = effect_size * np.sqrt(n)  # Non-centrality parameter
                    
                    # Critical t-value
                    t_crit = self.critical_values[self.alpha]
                    
                    # Power approximation using normal distribution
                    power = 1 - 0.5 * (1 + np.sign(ncp - t_crit) * 
                                      np.sqrt((ncp - t_crit)**2 / ((ncp - t_crit)**2 + 1)))
                    power = max(0, min(1, power))
                    
                elif test_type == "proportion":
                    # Power for proportion tests
                    se = np.sqrt(0.5 * (1 - 0.5) / n)  # Assume p=0.5 for conservative estimate
                    z_crit = self.critical_values[self.alpha]
                    z_beta = (effect_size - z_crit * se) / se
                    power = 0.5 * (1 + np.sign(z_beta) * np.sqrt(z_beta**2 / (z_beta**2 + 1)))
                    power = max(0, min(1, power))
                    
                else:
                    # Generic power estimate
                    power = min(1, n * effect_size / 100)
                
                power_results[n] = {
                    "power": power,
                    "adequate": power >= self.power_threshold,
                    "effect_size": effect_size
                }
            
            # Find minimum sample size for adequate power
            min_n_adequate = min([n for n, result in power_results.items() 
                                if result["adequate"]], default=None)
            
            return {
                "power_by_sample_size": power_results,
                "min_sample_size_adequate": min_n_adequate,
                "power_threshold": self.power_threshold,
                "effect_size": effect_size,
                "test_type": test_type
            }
            
        except Exception as e:
            logger.error(f"Power analysis failed: {e}")
            return {"error": str(e)}
    
    def normality_test(self, data: List[float], variable_name: str = "data") -> StatisticalTest:
        """
        Test for normality of data distribution.
        
        Args:
            data: Data to test for normality
            variable_name: Name of the variable being tested
            
        Returns:
            Normality test results
        """
        try:
            data_array = np.array(data)
            
            # D'Agostino and Pearson's normality test
            stat, p_value = stats.normaltest(data_array)
            
            # Interpretation
            if p_value > self.alpha:
                interpretation = f"{variable_name} appears to be normally distributed"
                normal = True
            else:
                interpretation = f"{variable_name} significantly deviates from normal distribution"
                normal = False
            
            return StatisticalTest(
                test_name="normality_test",
                statistic=stat,
                p_value=p_value,
                degrees_of_freedom=2,
                critical_value=5.99,  # Chi-square critical value for alpha=0.05, df=2
                effect_size=None,
                confidence_interval=None,
                interpretation=interpretation,
                significant=p_value < self.alpha
            )
            
        except Exception as e:
            logger.error(f"Normality test failed: {e}")
            return self._create_failed_test("normality_test")
    
    def generate_performance_metrics(self,
                                   method_name: str,
                                   true_positives: int,
                                   false_positives: int,
                                   true_negatives: int,
                                   false_negatives: int,
                                   confidence_scores: List[float] = None,
                                   processing_times: List[float] = None,
                                   semantic_similarities: List[float] = None) -> PerformanceMetrics:
        """
        Generate comprehensive performance metrics.
        
        Args:
            method_name: Name of the watermarking method
            true_positives: Number of true positive detections
            false_positives: Number of false positive detections
            true_negatives: Number of true negative detections
            false_negatives: Number of false negative detections
            confidence_scores: Optional confidence scores
            processing_times: Optional processing times
            semantic_similarities: Optional semantic similarity scores
            
        Returns:
            Complete performance metrics
        """
        try:
            # Basic counts
            total = true_positives + false_positives + true_negatives + false_negatives
            
            # Rates
            detection_rate = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            fp_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
            fn_rate = false_negatives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            # Precision, Recall, F1
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = detection_rate  # Same as detection rate
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # ROC AUC (simplified approximation)
            sensitivity = recall
            specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
            roc_auc = (sensitivity + specificity) / 2  # Simplified estimate
            
            return PerformanceMetrics(
                method_name=method_name,
                sample_size=total,
                detection_rate=detection_rate,
                false_positive_rate=fp_rate,
                false_negative_rate=fn_rate,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                roc_auc=roc_auc,
                confidence_scores=confidence_scores or [],
                processing_times=processing_times or [],
                semantic_similarities=semantic_similarities or []
            )
            
        except Exception as e:
            logger.error(f"Performance metrics generation failed: {e}")
            return PerformanceMetrics(
                method_name=method_name,
                sample_size=0,
                detection_rate=0.0,
                false_positive_rate=1.0,
                false_negative_rate=1.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                roc_auc=0.5,
                confidence_scores=[],
                processing_times=[],
                semantic_similarities=[]
            )
    
    def _create_failed_test(self, test_name: str) -> StatisticalTest:
        """Create a failed test result."""
        return StatisticalTest(
            test_name=test_name,
            statistic=0.0,
            p_value=1.0,
            degrees_of_freedom=None,
            critical_value=None,
            effect_size=None,
            confidence_interval=None,
            interpretation=f"Test {test_name} failed to execute",
            significant=False
        )


class ResearchReportGenerator:
    """Generate statistical reports for research papers."""
    
    def __init__(self, analyzer: StatisticalAnalyzer = None):
        """Initialize report generator."""
        self.analyzer = analyzer or StatisticalAnalyzer()
    
    def generate_method_report(self,
                             method_name: str,
                             performance_data: Dict[str, Any],
                             output_dir: str = "reports") -> str:
        """
        Generate comprehensive statistical report for a single method.
        
        Args:
            method_name: Name of the method
            performance_data: Performance data dictionary
            output_dir: Output directory for reports
            
        Returns:
            Path to generated report
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate statistical tests
            tests = []
            
            # Detection rate test
            if "watermarked_detections" in performance_data:
                test = self.analyzer.detection_rate_test(
                    performance_data["watermarked_detections"],
                    performance_data.get("unwatermarked_detections", [])
                )
                tests.append(test)
            
            # False positive test
            if "unwatermarked_detections" in performance_data:
                test = self.analyzer.false_positive_test(
                    performance_data["unwatermarked_detections"]
                )
                tests.append(test)
            
            # Semantic preservation test
            if "semantic_similarities" in performance_data:
                test = self.analyzer.semantic_preservation_test(
                    performance_data["semantic_similarities"]
                )
                tests.append(test)
            
            # Generate report content
            report_content = self._format_method_report(method_name, tests, performance_data)
            
            # Save report
            report_file = output_path / f"{method_name}_statistical_report.md"
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Statistical report generated: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return ""
    
    def _format_method_report(self,
                            method_name: str,
                            tests: List[StatisticalTest],
                            performance_data: Dict[str, Any]) -> str:
        """Format statistical report as markdown."""
        
        report = f"""# Statistical Analysis Report: {method_name.upper()}

## Summary

This report presents comprehensive statistical analysis for the {method_name} watermarking method.

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}

## Statistical Tests

"""
        
        for test in tests:
            report += f"""### {test.test_name.replace('_', ' ').title()}

- **Test Statistic:** {test.statistic:.4f}
- **P-value:** {test.p_value:.4f}
- **Significant:** {'Yes' if test.significant else 'No'} (α = 0.05)
- **Effect Size:** {test.effect_size:.4f if test.effect_size else 'N/A'}
- **Interpretation:** {test.interpretation}

"""
            
            if test.confidence_interval:
                ci_lower, ci_upper = test.confidence_interval
                report += f"- **95% Confidence Interval:** [{ci_lower:.4f}, {ci_upper:.4f}]\\n\\n"
        
        # Performance summary
        report += """## Performance Summary

"""
        
        if "watermarked_detections" in performance_data:
            detection_rate = np.mean(performance_data["watermarked_detections"])
            report += f"- **Detection Rate:** {detection_rate:.3f}\\n"
        
        if "unwatermarked_detections" in performance_data:
            fp_rate = np.mean(performance_data["unwatermarked_detections"])
            report += f"- **False Positive Rate:** {fp_rate:.3f}\\n"
        
        if "semantic_similarities" in performance_data:
            mean_sim = np.mean(performance_data["semantic_similarities"])
            report += f"- **Mean Semantic Similarity:** {mean_sim:.3f}\\n"
        
        if "processing_times" in performance_data:
            mean_time = np.mean(performance_data["processing_times"])
            report += f"- **Mean Processing Time:** {mean_time:.4f}s\\n"
        
        report += """
## Research Conclusions

"""
        
        significant_tests = [t for t in tests if t.significant]
        if significant_tests:
            report += f"- {len(significant_tests)} out of {len(tests)} statistical tests showed significant results\\n"
            report += "- Method demonstrates statistically significant performance characteristics\\n"
        else:
            report += "- No statistically significant differences found in primary metrics\\n"
        
        return report


# Convenience functions
def quick_statistical_analysis(method_name: str,
                             watermarked_detections: List[bool],
                             unwatermarked_detections: List[bool],
                             semantic_similarities: List[float] = None) -> Dict[str, StatisticalTest]:
    """Quick statistical analysis for a method."""
    
    analyzer = StatisticalAnalyzer()
    results = {}
    
    # Detection rate test
    results["detection_rate"] = analyzer.detection_rate_test(
        watermarked_detections, unwatermarked_detections
    )
    
    # False positive test
    results["false_positive"] = analyzer.false_positive_test(unwatermarked_detections)
    
    # Semantic preservation test (if data provided)
    if semantic_similarities:
        results["semantic_preservation"] = analyzer.semantic_preservation_test(semantic_similarities)
    
    return results


def generate_power_analysis_report(effect_sizes: List[float] = None,
                                 sample_sizes: List[int] = None) -> Dict[str, Any]:
    """Generate power analysis report for experimental design."""
    
    effect_sizes = effect_sizes or [0.2, 0.5, 0.8]  # Small, medium, large
    sample_sizes = sample_sizes or [10, 20, 30, 50, 100, 200]
    
    analyzer = StatisticalAnalyzer()
    
    power_analyses = {}
    for effect_size in effect_sizes:
        power_analyses[f"effect_size_{effect_size}"] = analyzer.power_analysis(
            sample_sizes, effect_size, "t_test"
        )
    
    return {
        "power_analyses": power_analyses,
        "recommendations": {
            "minimum_sample_size": 30,
            "recommended_sample_size": 50,
            "power_threshold": 0.8
        }
    }


__all__ = [
    "StatisticalAnalyzer",
    "StatisticalTest", 
    "PerformanceMetrics",
    "ResearchReportGenerator",
    "quick_statistical_analysis",
    "generate_power_analysis_report"
]