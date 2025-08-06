"""LM Watermark Lab - Comprehensive toolkit for watermarking, detecting, and attacking LLM-generated text."""

__version__ = "1.0.0"
__author__ = "Terragon Labs"
__email__ = "contact@terragonlabs.com"
__license__ = "Apache-2.0"

# Avoid circular imports by importing at module level only when needed
__all__ = [
    "WatermarkFactory",
    "WatermarkDetector", 
    "WatermarkBenchmark",
    "QuantumTaskPlanner",
    "create_watermarking_workflow",
    "__version__",
]

def __getattr__(name: str):
    """Lazy import to avoid circular dependencies."""
    if name == "WatermarkFactory":
        from watermark_lab.core.factory import WatermarkFactory
        return WatermarkFactory
    elif name == "WatermarkDetector":
        from watermark_lab.core.detector import WatermarkDetector
        return WatermarkDetector
    elif name == "WatermarkBenchmark":
        from watermark_lab.core.benchmark import WatermarkBenchmark
        return WatermarkBenchmark
    elif name == "QuantumTaskPlanner":
        from watermark_lab.core.quantum_planner import QuantumTaskPlanner
        return QuantumTaskPlanner
    elif name == "create_watermarking_workflow":
        from watermark_lab.core.quantum_planner import create_watermarking_workflow
        return create_watermarking_workflow
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")