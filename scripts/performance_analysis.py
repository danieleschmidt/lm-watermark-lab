#!/usr/bin/env python3
"""
Advanced Performance Analysis Script for LM Watermark Lab.

This script provides comprehensive performance analysis including:
- CPU and memory profiling
- Benchmark execution and analysis
- Performance regression detection
- Optimization recommendations
"""

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cProfile
import pstats
import io
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    operation: str
    duration: float
    memory_peak: int
    memory_current: int
    cpu_percent: float
    timestamp: str
    
    
@dataclass
class BenchmarkResult:
    """Benchmark result data structure."""
    name: str
    metrics: List[PerformanceMetrics]
    summary: Dict[str, Any]
    

class PerformanceProfiler:
    """Advanced performance profiler with multiple profiling backends."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("performance/profiles")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.process = psutil.Process()
        
    @contextmanager
    def profile_context(self, name: str, enable_memory: bool = True):
        """Context manager for profiling code blocks."""
        # Start memory tracking
        if enable_memory:
            tracemalloc.start()
            
        # Start CPU profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Record start metrics
        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss
        start_cpu = self.process.cpu_percent()
        
        try:
            yield
        finally:
            # Stop profiling and collect metrics
            end_time = time.perf_counter()
            profiler.disable()
            
            end_memory = self.process.memory_info().rss
            end_cpu = self.process.cpu_percent()
            
            duration = end_time - start_time
            
            # Memory tracking
            memory_peak = end_memory
            if enable_memory and tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                memory_peak = max(memory_peak, peak)
                tracemalloc.stop()
            
            # Save CPU profile
            self._save_cpu_profile(profiler, name)
            
            # Create metrics
            metrics = PerformanceMetrics(
                operation=name,
                duration=duration,
                memory_peak=memory_peak,
                memory_current=end_memory,
                cpu_percent=(start_cpu + end_cpu) / 2,
                timestamp=datetime.now().isoformat()
            )
            
            self._save_metrics(metrics)
            
    def _save_cpu_profile(self, profiler: cProfile.Profile, name: str):
        """Save CPU profile data."""
        # Save binary profile
        profile_file = self.output_dir / f"{name}_{int(time.time())}.prof"
        profiler.dump_stats(str(profile_file))
        
        # Save text report
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(50)  # Top 50 functions
        
        report_file = self.output_dir / f"{name}_{int(time.time())}.txt"
        with open(report_file, 'w') as f:
            f.write(s.getvalue())
            
    def _save_metrics(self, metrics: PerformanceMetrics):
        """Save performance metrics."""
        metrics_file = self.output_dir / "metrics.jsonl"
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + '\n')
            

class WatermarkBenchmark:
    """Comprehensive benchmarking suite for watermark operations."""
    
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.results: List[BenchmarkResult] = []
        
    def run_generation_benchmark(self, iterations: int = 100) -> BenchmarkResult:
        """Benchmark text generation performance."""
        print(f"Running generation benchmark ({iterations} iterations)...")
        
        metrics = []
        for i in range(iterations):
            with self.profiler.profile_context(f"generation_{i}"):
                # Simulate watermark generation
                self._simulate_generation()
                
        # Collect metrics for this benchmark
        benchmark_metrics = self._load_recent_metrics("generation", iterations)
        
        # Calculate summary statistics
        durations = [m.duration for m in benchmark_metrics]
        memory_peaks = [m.memory_peak for m in benchmark_metrics]
        
        summary = {
            "total_iterations": iterations,
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "avg_memory_peak": sum(memory_peaks) / len(memory_peaks),
            "max_memory_peak": max(memory_peaks),
            "tokens_per_second": self._calculate_throughput(durations),
        }
        
        result = BenchmarkResult(
            name="generation_benchmark",
            metrics=benchmark_metrics,
            summary=summary
        )
        
        self.results.append(result)
        return result
        
    def run_detection_benchmark(self, iterations: int = 100) -> BenchmarkResult:
        """Benchmark watermark detection performance."""
        print(f"Running detection benchmark ({iterations} iterations)...")
        
        for i in range(iterations):
            with self.profiler.profile_context(f"detection_{i}"):
                # Simulate watermark detection
                self._simulate_detection()
                
        benchmark_metrics = self._load_recent_metrics("detection", iterations)
        
        durations = [m.duration for m in benchmark_metrics]
        memory_peaks = [m.memory_peak for m in benchmark_metrics]
        
        summary = {
            "total_iterations": iterations,
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "avg_memory_peak": sum(memory_peaks) / len(memory_peaks),
            "detections_per_second": len(durations) / sum(durations),
        }
        
        result = BenchmarkResult(
            name="detection_benchmark", 
            metrics=benchmark_metrics,
            summary=summary
        )
        
        self.results.append(result)
        return result
        
    def run_memory_stress_test(self, max_memory_mb: int = 1024) -> BenchmarkResult:
        """Run memory stress test to find memory limits."""
        print(f"Running memory stress test (max {max_memory_mb}MB)...")
        
        metrics = []
        memory_sizes = [64, 128, 256, 512, 1024, 2048]  # MB
        
        for size_mb in memory_sizes:
            if size_mb > max_memory_mb:
                break
                
            with self.profiler.profile_context(f"memory_stress_{size_mb}mb"):
                self._simulate_memory_usage(size_mb)
                
        benchmark_metrics = self._load_recent_metrics("memory_stress", len(memory_sizes))
        
        summary = {
            "tested_sizes_mb": memory_sizes[:len(benchmark_metrics)],
            "max_tested_mb": max(memory_sizes[:len(benchmark_metrics)]),
            "memory_efficiency": self._calculate_memory_efficiency(benchmark_metrics),
        }
        
        result = BenchmarkResult(
            name="memory_stress_test",
            metrics=benchmark_metrics,
            summary=summary
        )
        
        self.results.append(result)
        return result
        
    def _simulate_generation(self):
        """Simulate watermark generation workload."""
        # Simulate text generation with realistic computation
        import random
        text_length = random.randint(100, 1000)
        
        # Simulate model computation
        data = [random.random() for _ in range(text_length * 100)]
        
        # Simulate watermark application
        for i in range(len(data)):
            data[i] = data[i] * 1.1 + 0.1
            
        # Simulate text decoding
        result = "".join([chr(65 + int(x * 26) % 26) for x in data[:text_length]])
        return result
        
    def _simulate_detection(self):
        """Simulate watermark detection workload."""
        import random
        
        # Simulate text tokenization
        tokens = [random.randint(0, 50000) for _ in range(500)]
        
        # Simulate detection computation
        scores = []
        for token in tokens:
            # Simulate statistical computation
            score = sum([token * i * 0.001 for i in range(100)])
            scores.append(score)
            
        # Simulate decision making
        avg_score = sum(scores) / len(scores)
        return avg_score > 0.5
        
    def _simulate_memory_usage(self, size_mb: int):
        """Simulate memory usage of specified size."""
        # Allocate memory arrays
        size_bytes = size_mb * 1024 * 1024
        chunk_size = 1024 * 1024  # 1MB chunks
        
        data = []
        for i in range(0, size_bytes, chunk_size):
            chunk = bytearray(min(chunk_size, size_bytes - i))
            # Fill with data to ensure memory allocation
            for j in range(len(chunk)):
                chunk[j] = j % 256
            data.append(chunk)
            
        # Simulate processing
        total = 0
        for chunk in data:
            total += sum(chunk)
            
        return total
        
    def _load_recent_metrics(self, operation_prefix: str, count: int) -> List[PerformanceMetrics]:
        """Load recent metrics from the metrics file."""
        metrics_file = self.profiler.output_dir / "metrics.jsonl"
        if not metrics_file.exists():
            return []
            
        metrics = []
        with open(metrics_file, 'r') as f:
            for line in f:
                metric_data = json.loads(line.strip())
                metric = PerformanceMetrics(**metric_data)
                if metric.operation.startswith(operation_prefix):
                    metrics.append(metric)
                    
        # Return the most recent metrics
        return metrics[-count:] if len(metrics) >= count else metrics
        
    def _calculate_throughput(self, durations: List[float]) -> float:
        """Calculate throughput in operations per second."""
        if not durations:
            return 0.0
        total_time = sum(durations)
        return len(durations) / total_time if total_time > 0 else 0.0
        
    def _calculate_memory_efficiency(self, metrics: List[PerformanceMetrics]) -> float:
        """Calculate memory efficiency score."""
        if not metrics:
            return 0.0
            
        # Simple efficiency metric: operations per MB
        total_memory = sum(m.memory_peak for m in metrics)
        return len(metrics) / (total_memory / 1024 / 1024) if total_memory > 0 else 0.0


class PerformanceAnalyzer:
    """Analyze performance data and generate reports."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        
    def analyze_results(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze benchmark results and generate insights."""
        analysis = {
            "summary": self._generate_summary(results),
            "trends": self._analyze_trends(results),
            "bottlenecks": self._identify_bottlenecks(results),
            "recommendations": self._generate_recommendations(results),
        }
        
        return analysis
        
    def _generate_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate overall performance summary."""
        summary = {
            "total_benchmarks": len(results),
            "benchmarks": {},
        }
        
        for result in results:
            summary["benchmarks"][result.name] = result.summary
            
        return summary
        
    def _analyze_trends(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze performance trends."""
        trends = {}
        
        for result in results:
            if len(result.metrics) < 2:
                continue
                
            durations = [m.duration for m in result.metrics]
            memory_peaks = [m.memory_peak for m in result.metrics]
            
            # Simple trend analysis
            duration_trend = "stable"
            if len(durations) > 1:
                if durations[-1] > durations[0] * 1.1:
                    duration_trend = "increasing"
                elif durations[-1] < durations[0] * 0.9:
                    duration_trend = "decreasing"
                    
            trends[result.name] = {
                "duration_trend": duration_trend,
                "avg_duration": sum(durations) / len(durations),
                "duration_variance": self._calculate_variance(durations),
                "memory_trend": self._analyze_memory_trend(memory_peaks),
            }
            
        return trends
        
    def _identify_bottlenecks(self, results: List[BenchmarkResult]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        for result in results:
            if result.name == "generation_benchmark":
                if result.summary.get("avg_duration", 0) > 1.0:  # > 1 second
                    bottlenecks.append("Text generation is slow (>1s average)")
                    
            elif result.name == "detection_benchmark":
                if result.summary.get("avg_duration", 0) > 0.1:  # > 100ms
                    bottlenecks.append("Detection is slow (>100ms average)")
                    
            elif result.name == "memory_stress_test":
                max_memory = result.summary.get("max_tested_mb", 0)
                if max_memory < 512:  # Less than 512MB
                    bottlenecks.append(f"Low memory capacity ({max_memory}MB)")
                    
        return bottlenecks
        
    def _generate_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        for result in results:
            if result.name == "generation_benchmark":
                avg_duration = result.summary.get("avg_duration", 0)
                if avg_duration > 0.5:
                    recommendations.append("Consider model quantization to improve generation speed")
                    recommendations.append("Implement batching for multiple text generation")
                    
            elif result.name == "detection_benchmark":
                detections_per_sec = result.summary.get("detections_per_second", 0)
                if detections_per_sec < 100:
                    recommendations.append("Optimize detection algorithms for better throughput")
                    recommendations.append("Consider caching detection results")
                    
            elif result.name == "memory_stress_test":
                efficiency = result.summary.get("memory_efficiency", 0)
                if efficiency < 1.0:
                    recommendations.append("Implement memory pooling to reduce allocations")
                    recommendations.append("Add garbage collection optimization")
                    
        # General recommendations
        recommendations.extend([
            "Monitor performance metrics in production",
            "Set up automated performance regression testing",
            "Consider using GPU acceleration for compute-intensive operations",
            "Implement request queuing for high-load scenarios",
        ])
        
        return recommendations
        
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
        
    def _analyze_memory_trend(self, memory_values: List[int]) -> str:
        """Analyze memory usage trend."""
        if len(memory_values) < 2:
            return "insufficient_data"
            
        if memory_values[-1] > memory_values[0] * 1.2:
            return "increasing"
        elif memory_values[-1] < memory_values[0] * 0.8:
            return "decreasing"
        else:
            return "stable"
            
    def generate_report(self, analysis: Dict[str, Any], output_file: Path):
        """Generate a comprehensive performance report."""
        with open(output_file, 'w') as f:
            f.write("# Performance Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().isoformat()}\n\n")
            
            # Summary section
            f.write("## Executive Summary\n\n")
            summary = analysis["summary"]
            f.write(f"- Total benchmarks executed: {summary['total_benchmarks']}\n")
            
            for name, bench_summary in summary["benchmarks"].items():
                f.write(f"- {name}: {bench_summary}\n")
            f.write("\n")
            
            # Trends section
            f.write("## Performance Trends\n\n")
            for name, trend in analysis["trends"].items():
                f.write(f"### {name}\n")
                f.write(f"- Duration trend: {trend['duration_trend']}\n")
                f.write(f"- Average duration: {trend['avg_duration']:.4f}s\n")
                f.write(f"- Memory trend: {trend['memory_trend']}\n\n")
                
            # Bottlenecks section
            f.write("## Identified Bottlenecks\n\n")
            for bottleneck in analysis["bottlenecks"]:
                f.write(f"- {bottleneck}\n")
            f.write("\n")
            
            # Recommendations section
            f.write("## Optimization Recommendations\n\n")
            for i, recommendation in enumerate(analysis["recommendations"], 1):
                f.write(f"{i}. {recommendation}\n")


def main():
    """Main performance analysis entry point."""
    parser = argparse.ArgumentParser(description="Advanced Performance Analysis for LM Watermark Lab")
    parser.add_argument("--output-dir", type=Path, default="performance/analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--iterations", type=int, default=50,
                       help="Number of iterations for benchmarks")
    parser.add_argument("--benchmarks", nargs="+", 
                       choices=["generation", "detection", "memory", "all"],
                       default=["all"], help="Benchmarks to run")
    parser.add_argument("--profile", action="store_true",
                       help="Enable detailed profiling")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize profiler and benchmark suite
    profiler = PerformanceProfiler(args.output_dir / "profiles")
    benchmark = WatermarkBenchmark(profiler)
    
    print("Starting performance analysis...")
    print(f"Output directory: {args.output_dir}")
    print(f"Iterations per benchmark: {args.iterations}")
    
    # Run benchmarks
    results = []
    
    if "all" in args.benchmarks or "generation" in args.benchmarks:
        results.append(benchmark.run_generation_benchmark(args.iterations))
        
    if "all" in args.benchmarks or "detection" in args.benchmarks:
        results.append(benchmark.run_detection_benchmark(args.iterations))
        
    if "all" in args.benchmarks or "memory" in args.benchmarks:
        results.append(benchmark.run_memory_stress_test())
        
    # Analyze results
    analyzer = PerformanceAnalyzer(args.output_dir)
    analysis = analyzer.analyze_results(results)
    
    # Generate reports
    report_file = args.output_dir / "performance_report.md"
    analyzer.generate_report(analysis, report_file)
    
    # Save analysis data
    analysis_file = args.output_dir / "analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
        
    print(f"\nPerformance analysis complete!")
    print(f"Report: {report_file}")
    print(f"Analysis data: {analysis_file}")
    print(f"Profiles: {profiler.output_dir}")
    
    # Print summary
    print("\n=== Performance Summary ===")
    for result in results:
        print(f"{result.name}: {result.summary}")
        
    if analysis["bottlenecks"]:
        print(f"\n⚠️  Bottlenecks identified: {len(analysis['bottlenecks'])}")
        for bottleneck in analysis["bottlenecks"][:3]:  # Show top 3
            print(f"  - {bottleneck}")
            
    print(f"\n💡 Generated {len(analysis['recommendations'])} optimization recommendations")


if __name__ == "__main__":
    main()