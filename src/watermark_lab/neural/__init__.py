"""Neural detection pipeline for watermark identification with training capabilities."""

from .detector import NeuralWatermarkDetector, DetectorConfig
from .trainer import NeuralTrainer, TrainingConfig
from .models import DetectionModel, TransformerDetector
from .dataset import WatermarkDataset, DatasetBuilder

__all__ = [
    "NeuralWatermarkDetector",
    "DetectorConfig",
    "NeuralTrainer", 
    "TrainingConfig",
    "DetectionModel",
    "TransformerDetector",
    "WatermarkDataset",
    "DatasetBuilder"
]