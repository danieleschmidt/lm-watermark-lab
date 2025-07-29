"""LM Watermark Lab - Comprehensive toolkit for watermarking, detecting, and attacking LLM-generated text."""

__version__ = "1.0.0"
__author__ = "Terragon Labs"
__email__ = "contact@terragonlabs.com"
__license__ = "Apache-2.0"

from watermark_lab.core.factory import WatermarkFactory
from watermark_lab.core.detector import WatermarkDetector
from watermark_lab.core.benchmark import WatermarkBenchmark

__all__ = [
    "WatermarkFactory",
    "WatermarkDetector", 
    "WatermarkBenchmark",
    "__version__",
]