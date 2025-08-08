"""Research framework for watermarking experiments and academic publication."""

from .experimental_framework import ExperimentalFramework, ExperimentConfig
from .comparative_study import ComparativeStudy 
from .statistical_analysis import StatisticalAnalyzer
from .reproducibility import ReproducibilityManager
from .publication_prep import PublicationPrep

__all__ = [
    "ExperimentalFramework",
    "ExperimentConfig",
    "ComparativeStudy",
    "StatisticalAnalyzer", 
    "ReproducibilityManager",
    "PublicationPrep"
]