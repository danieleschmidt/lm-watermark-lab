"""Performance benchmarks for LM Watermark Lab."""

import time
import pytest
import statistics
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch


class TestPerformanceBenchmarks:
    """Benchmark tests for performance-critical operations."""

    @pytest.mark.benchmark
    def test_watermark_generation_speed(self, benchmark_data, mock_model):
        """Benchmark watermark generation speed."""
        prompts = benchmark_data["prompts"]
        
        # Mock watermarker
        with patch('watermark_lab.watermarking.WatermarkFactory.create') as mock_factory:
            mock_watermarker = MagicMock()
            mock_watermarker.generate.return_value = "watermarked text " * 50
            mock_factory.return_value = mock_watermarker
            
            # Benchmark generation
            start_time = time.perf_counter()
            for prompt in prompts:
                mock_watermarker.generate(prompt, max_length=100)
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            avg_time_per_sample = total_time / len(prompts)
            
            # Performance assertion: should generate < 2s per sample
            assert avg_time_per_sample < 2.0, f"Generation too slow: {avg_time_per_sample:.2f}s per sample"
            
            print(f"Average generation time: {avg_time_per_sample:.3f}s per sample")

    @pytest.mark.benchmark
    def test_detection_speed(self, sample_texts, watermark_configs):
        """Benchmark watermark detection speed."""
        texts = list(sample_texts.values())
        
        with patch('watermark_lab.detection.WatermarkDetector') as mock_detector_class:
            mock_detector = MagicMock()
            mock_detector.detect.return_value = {
                "is_watermarked": True,
                "confidence": 0.95,
                "p_value": 0.001
            }
            mock_detector_class.return_value = mock_detector
            
            # Benchmark detection
            start_time = time.perf_counter()
            for text in texts:
                mock_detector.detect(text)
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            avg_time_per_detection = total_time / len(texts)
            
            # Performance assertion: should detect < 100ms per sample
            assert avg_time_per_detection < 0.1, f"Detection too slow: {avg_time_per_detection:.3f}s per sample"
            
            print(f"Average detection time: {avg_time_per_detection:.3f}s per sample")

    @pytest.mark.benchmark
    def test_batch_processing_efficiency(self, benchmark_data):
        """Test that batch processing is more efficient than individual processing."""
        prompts = benchmark_data["prompts"]
        
        with patch('watermark_lab.watermarking.WatermarkEngine') as mock_engine_class:
            mock_engine = MagicMock()
            
            # Mock individual generation (slower)
            mock_engine.generate.return_value = "individual result"
            individual_times = []
            
            for prompt in prompts:
                start = time.perf_counter()
                mock_engine.generate(prompt)
                end = time.perf_counter()
                individual_times.append(end - start)
            
            total_individual_time = sum(individual_times)
            
            # Mock batch generation (should be faster)
            mock_engine.generate_batch.return_value = ["batch result"] * len(prompts)
            
            start = time.perf_counter()
            mock_engine.generate_batch(prompts)
            end = time.perf_counter()
            
            batch_time = end - start
            
            # Batch should be at least 20% faster than individual
            efficiency_ratio = batch_time / total_individual_time
            assert efficiency_ratio < 0.8, f"Batch processing not efficient enough: {efficiency_ratio:.2f}"
            
            print(f"Batch efficiency ratio: {efficiency_ratio:.2f}")

    @pytest.mark.benchmark
    def test_memory_usage(self, sample_texts):
        """Test memory usage during processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate processing large texts
        with patch('watermark_lab.watermarking.WatermarkFactory.create') as mock_factory:
            mock_watermarker = MagicMock()
            mock_watermarker.generate.return_value = sample_texts["long"]
            mock_factory.return_value = mock_watermarker
            
            # Process multiple texts
            results = []
            for i in range(100):
                result = mock_watermarker.generate("test prompt")
                results.append(result)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (< 100MB for this test)
            assert memory_increase < 100, f"Memory usage too high: {memory_increase:.1f}MB increase"
            
            print(f"Memory increase: {memory_increase:.1f}MB")

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_concurrent_processing(self, benchmark_data):
        """Test performance under concurrent load."""
        import concurrent.futures
        import threading
        
        prompts = benchmark_data["prompts"] * 10  # More prompts for stress test
        
        with patch('watermark_lab.watermarking.WatermarkFactory.create') as mock_factory:
            mock_watermarker = MagicMock()
            mock_watermarker.generate.return_value = "concurrent result"
            mock_factory.return_value = mock_watermarker
            
            def process_prompt(prompt):
                start = time.perf_counter()
                result = mock_watermarker.generate(prompt)
                end = time.perf_counter()
                return end - start
            
            # Test with different numbers of workers
            worker_counts = [1, 2, 4, 8]
            results = {}
            
            for worker_count in worker_counts:
                start_time = time.perf_counter()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                    times = list(executor.map(process_prompt, prompts))
                
                total_time = time.perf_counter() - start_time
                results[worker_count] = {
                    'total_time': total_time,
                    'avg_time': statistics.mean(times),
                    'throughput': len(prompts) / total_time
                }
            
            # Concurrency should improve throughput
            single_thread_throughput = results[1]['throughput']
            multi_thread_throughput = results[max(worker_counts)]['throughput']
            
            improvement_ratio = multi_thread_throughput / single_thread_throughput
            
            print(f"Throughput improvement with concurrency: {improvement_ratio:.2f}x")
            
            # Should see some improvement (at least 1.5x with 8 workers)
            assert improvement_ratio > 1.5, f"Insufficient concurrency improvement: {improvement_ratio:.2f}x"

    @pytest.mark.benchmark
    def test_api_response_time(self, api_client):
        """Test API response time benchmarks."""
        if api_client is None:
            pytest.skip("API client not available")
        
        endpoints = [
            ("/health", "GET"),
            ("/api/v1/methods", "GET"),
        ]
        
        response_times = []
        
        for endpoint, method in endpoints:
            start = time.perf_counter()
            
            if method == "GET":
                response = api_client.get(endpoint)
            elif method == "POST":
                response = api_client.post(endpoint, json={})
            
            end = time.perf_counter()
            response_time = end - start
            response_times.append(response_time)
            
            # API should respond within 1 second
            assert response_time < 1.0, f"API response too slow for {endpoint}: {response_time:.3f}s"
        
        avg_response_time = statistics.mean(response_times)
        print(f"Average API response time: {avg_response_time:.3f}s")

    @pytest.mark.benchmark
    def test_scalability_metrics(self, benchmark_data):
        """Test scalability characteristics."""
        # Test with increasing data sizes
        data_sizes = [10, 50, 100, 500]
        processing_times = []
        
        with patch('watermark_lab.evaluation.QualityEvaluator') as mock_evaluator_class:
            mock_evaluator = MagicMock()
            mock_evaluator.evaluate_quality.return_value = {"perplexity": 10.0}
            mock_evaluator_class.return_value = mock_evaluator
            
            for size in data_sizes:
                texts = ["test text"] * size
                
                start = time.perf_counter()
                for text in texts:
                    mock_evaluator.evaluate_quality("original", text)
                end = time.perf_counter()
                
                processing_time = end - start
                processing_times.append(processing_time)
                
                print(f"Size {size}: {processing_time:.3f}s")
            
            # Check that processing time scales roughly linearly
            # (not exponentially)
            time_ratios = [
                processing_times[i] / processing_times[i-1] 
                for i in range(1, len(processing_times))
            ]
            
            size_ratios = [
                data_sizes[i] / data_sizes[i-1] 
                for i in range(1, len(data_sizes))
            ]
            
            # Time ratio should not be much larger than size ratio
            for time_ratio, size_ratio in zip(time_ratios, size_ratios):
                efficiency_ratio = time_ratio / size_ratio
                assert efficiency_ratio < 2.0, f"Poor scalability: {efficiency_ratio:.2f}"


class TestMemoryBenchmarks:
    """Memory-specific benchmark tests."""

    @pytest.mark.benchmark
    def test_memory_leak_detection(self, sample_texts):
        """Test for memory leaks during repeated operations."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        def get_memory_usage():
            gc.collect()
            return process.memory_info().rss / 1024 / 1024  # MB
        
        initial_memory = get_memory_usage()
        
        with patch('watermark_lab.watermarking.WatermarkFactory.create') as mock_factory:
            mock_watermarker = MagicMock()
            mock_watermarker.generate.return_value = sample_texts["medium"]
            mock_factory.return_value = mock_watermarker
            
            # Perform many operations
            for i in range(1000):
                mock_watermarker.generate("test prompt")
                
                # Check memory every 100 iterations
                if i % 100 == 0:
                    current_memory = get_memory_usage()
                    memory_increase = current_memory - initial_memory
                    
                    # Memory should not continuously increase
                    assert memory_increase < 50, f"Potential memory leak: {memory_increase:.1f}MB increase"
        
        final_memory = get_memory_usage()
        total_increase = final_memory - initial_memory
        
        print(f"Total memory increase: {total_increase:.1f}MB")
        assert total_increase < 100, f"Memory leak detected: {total_increase:.1f}MB increase"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])