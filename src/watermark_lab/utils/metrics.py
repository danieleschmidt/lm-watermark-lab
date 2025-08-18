"""Metrics calculation and formatting utilities."""

import time
import statistics
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
try:
    import psutil
except ImportError:
    from .fallback_imports import psutil


@dataclass
class SystemMetrics:
    """System resource metrics."""
    
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    timestamp: datetime


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    
    operation: str
    duration: float
    throughput: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    cpu_usage: Optional[float] = None
    success: bool = True
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class MetricsCollector:
    """Collect and aggregate metrics."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.start_time = time.time()
    
    def record_operation(
        self,
        operation: str,
        duration: float,
        throughput: Optional[float] = None,
        memory_peak_mb: Optional[float] = None,
        success: bool = True,
        **kwargs  # Accept additional keyword arguments
    ) -> None:
        """Record operation metrics with flexible metadata."""
        metrics = PerformanceMetrics(
            operation=operation,
            duration=duration,
            throughput=throughput,
            memory_peak_mb=memory_peak_mb,
            success=success
        )
        # Store additional metadata if needed
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
        self.metrics.append(metrics)
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        op_metrics = [m for m in self.metrics if m.operation == operation]
        
        if not op_metrics:
            return {"operation": operation, "count": 0}
        
        durations = [m.duration for m in op_metrics]
        success_count = sum(1 for m in op_metrics if m.success)
        
        stats = {
            "operation": operation,
            "count": len(op_metrics),
            "success_rate": success_count / len(op_metrics),
            "duration_stats": {
                "mean": statistics.mean(durations),
                "median": statistics.median(durations),
                "min": min(durations),
                "max": max(durations),
                "std": statistics.stdev(durations) if len(durations) > 1 else 0.0
            }
        }
        
        # Add throughput stats if available  
        throughputs = [m.throughput for m in op_metrics if m.throughput is not None]
        if throughputs:
            stats["throughput_stats"] = {
                "mean": statistics.mean(throughputs),
                "median": statistics.median(throughputs),
                "min": min(throughputs),
                "max": max(throughputs)
            }
        
        return stats
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        if not self.metrics:
            return {"total_operations": 0, "uptime": time.time() - self.start_time}
        
        operations = set(m.operation for m in self.metrics)
        operation_stats = {op: self.get_operation_stats(op) for op in operations}
        
        total_duration = sum(m.duration for m in self.metrics)
        success_count = sum(1 for m in self.metrics if m.success)
        
        return {
            "total_operations": len(self.metrics),
            "unique_operations": len(operations),
            "success_rate": success_count / len(self.metrics),
            "total_duration": total_duration,
            "uptime": time.time() - self.start_time,
            "operations": operation_stats
        }


def get_system_metrics() -> SystemMetrics:
    """Get current system resource metrics."""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return SystemMetrics(
        cpu_percent=psutil.cpu_percent(interval=1),
        memory_percent=memory.percent,
        memory_used_mb=memory.used / 1024 / 1024,
        memory_available_mb=memory.available / 1024 / 1024,
        disk_usage_percent=disk.percent,
        timestamp=datetime.utcnow()
    )


def calculate_text_metrics(text: str) -> Dict[str, Any]:
    """Calculate basic text metrics."""
    if not text:
        return {
            "length": 0,
            "words": 0,
            "sentences": 0,
            "avg_word_length": 0.0,
            "avg_sentence_length": 0.0
        }
    
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    return {
        "length": len(text),
        "words": len(words),
        "sentences": max(sentences, 1),  # Avoid division by zero
        "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0.0,
        "avg_sentence_length": len(words) / max(sentences, 1)
    }


def calculate_similarity_metrics(text1: str, text2: str) -> Dict[str, float]:
    """Calculate similarity metrics between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    jaccard = intersection / union if union > 0 else 0.0
    
    # Cosine similarity (simplified)
    all_words = words1 | words2
    vec1 = [1 if word in words1 else 0 for word in all_words]
    vec2 = [1 if word in words2 else 0 for word in all_words]
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    
    cosine = dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 > 0 else 0.0
    
    # Overlap coefficient
    overlap = intersection / min(len(words1), len(words2)) if min(len(words1), len(words2)) > 0 else 0.0
    
    return {
        "jaccard": jaccard,
        "cosine": cosine,
        "overlap": overlap,
        "word_overlap_ratio": intersection / max(len(words1), len(words2)) if max(len(words1), len(words2)) > 0 else 0.0
    }


def calculate_detection_metrics(
    true_labels: List[bool],
    predicted_labels: List[bool],
    confidence_scores: Optional[List[float]] = None
) -> Dict[str, float]:
    """Calculate detection performance metrics."""
    if len(true_labels) != len(predicted_labels):
        raise ValueError("True labels and predicted labels must have the same length")
    
    # Confusion matrix
    tp = sum(1 for t, p in zip(true_labels, predicted_labels) if t and p)
    fp = sum(1 for t, p in zip(true_labels, predicted_labels) if not t and p)
    tn = sum(1 for t, p in zip(true_labels, predicted_labels) if not t and not p)
    fn = sum(1 for t, p in zip(true_labels, predicted_labels) if t and not p)
    
    # Basic metrics
    accuracy = (tp + tn) / len(true_labels) if len(true_labels) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "specificity": specificity,
        "true_positive_rate": recall,
        "false_positive_rate": 1 - specificity,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    }
    
    # Add confidence-based metrics if available
    if confidence_scores:
        avg_confidence = statistics.mean(confidence_scores)
        confidence_std = statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0.0
        
        metrics.update({
            "avg_confidence": avg_confidence,
            "confidence_std": confidence_std,
            "min_confidence": min(confidence_scores),
            "max_confidence": max(confidence_scores)
        })
    
    return metrics


def calculate_quality_metrics(
    original_texts: List[str],
    generated_texts: List[str]
) -> Dict[str, float]:
    """Calculate text quality metrics."""
    if len(original_texts) != len(generated_texts):
        raise ValueError("Original and generated texts must have the same length")
    
    if not original_texts:
        return {}
    
    # Length preservation
    length_ratios = []
    for orig, gen in zip(original_texts, generated_texts):
        if len(orig) > 0:
            length_ratios.append(len(gen) / len(orig))
    
    avg_length_ratio = statistics.mean(length_ratios) if length_ratios else 1.0
    
    # Similarity metrics
    similarities = []
    for orig, gen in zip(original_texts, generated_texts):
        sim_metrics = calculate_similarity_metrics(orig, gen)
        similarities.append(sim_metrics["jaccard"])
    
    avg_similarity = statistics.mean(similarities) if similarities else 0.0
    
    # Diversity (unique words ratio)
    all_generated = " ".join(generated_texts)
    generated_words = all_generated.split()
    unique_words = set(generated_words)
    diversity = len(unique_words) / len(generated_words) if generated_words else 0.0
    
    return {
        "avg_length_ratio": avg_length_ratio,
        "avg_similarity": avg_similarity,
        "diversity": diversity,
        "length_preservation": 1.0 - abs(1.0 - avg_length_ratio)
    }


def format_metrics(metrics: Dict[str, Any], precision: int = 3) -> str:
    """Format metrics for display."""
    lines = []
    
    for key, value in metrics.items():
        if isinstance(value, dict):
            lines.append(f"{key.title().replace('_', ' ')}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, float):
                    lines.append(f"  {sub_key.replace('_', ' ')}: {sub_value:.{precision}f}")
                else:
                    lines.append(f"  {sub_key.replace('_', ' ')}: {sub_value}")
        elif isinstance(value, float):
            lines.append(f"{key.title().replace('_', ' ')}: {value:.{precision}f}")
        else:
            lines.append(f"{key.title().replace('_', ' ')}: {value}")
    
    return "\n".join(lines)


def export_metrics_to_csv(metrics_list: List[Dict[str, Any]], filename: str) -> None:
    """Export metrics to CSV file."""
    import csv
    from pathlib import Path
    
    if not metrics_list:
        return
    
    # Get all unique keys
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    all_keys = sorted(all_keys)
    
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_keys)
        writer.writeheader()
        
        for metrics in metrics_list:
            # Flatten nested dictionaries
            flattened = {}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flattened[f"{key}_{sub_key}"] = sub_value
                else:
                    flattened[key] = value
            
            writer.writerow(flattened)


# Global metrics collector instance
_global_metrics = MetricsCollector()


def get_global_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    return _global_metrics


def record_operation_metric(
    operation: str,
    duration: float,
    success: bool = True,
    **kwargs
) -> None:
    """Record an operation metric globally."""
    _global_metrics.record_operation(operation, duration, success=success, **kwargs)