"""
lm-watermark-lab: Comparative toolkit for LLM text watermarking algorithms.

Pure Python + NumPy implementation. No LLM API calls required.
Works directly on token/word sequences.
"""

from .green_red import GreenRedWatermark
from .semantic import SemanticWatermark
from .statistical import StatisticalWatermark
from .detector import WatermarkDetector
from .robustness import RobustnessEvaluator

__version__ = "1.0.0"
__all__ = [
    "GreenRedWatermark",
    "SemanticWatermark",
    "StatisticalWatermark",
    "WatermarkDetector",
    "RobustnessEvaluator",
]
