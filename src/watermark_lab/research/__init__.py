"""Research framework for watermarking experiments and academic publication."""

from .experimental_framework import ExperimentalFramework, ExperimentConfig, run_watermark_experiment
from .comparative_study import ComparativeStudy, StatisticalAnalyzer as CompAnalyzer, quick_pairwise_comparison, benchmark_novel_algorithms
from .statistical_analysis import StatisticalAnalyzer, StatisticalTest, PerformanceMetrics, ResearchReportGenerator, quick_statistical_analysis, generate_power_analysis_report
from .reproducibility import ReproducibilityManager, ExperimentEnvironment, ReproducibilityResult, ensure_reproducible_environment, run_reproducibility_check
from .publication_prep import PublicationPrep, PublicationFigure, PublicationTable, ResearchContribution, prepare_publication_materials

__all__ = [
    # Core frameworks
    "ExperimentalFramework",
    "ExperimentConfig", 
    "ComparativeStudy",
    "StatisticalAnalyzer",
    "ReproducibilityManager",
    "PublicationPrep",
    
    # Data structures
    "StatisticalTest",
    "PerformanceMetrics",
    "ExperimentEnvironment", 
    "ReproducibilityResult",
    "PublicationFigure",
    "PublicationTable",
    "ResearchContribution",
    
    # Report generators
    "ResearchReportGenerator",
    
    # Convenience functions
    "run_watermark_experiment",
    "quick_pairwise_comparison", 
    "benchmark_novel_algorithms",
    "quick_statistical_analysis",
    "generate_power_analysis_report",
    "ensure_reproducible_environment",
    "run_reproducibility_check",
    "prepare_publication_materials"
]