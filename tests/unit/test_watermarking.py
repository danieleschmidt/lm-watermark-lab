"""Unit tests for watermarking functionality."""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from src.watermark_lab.core.factory import WatermarkFactory, BaseWatermark, KirchenbauerWatermark
from src.watermark_lab.core.detector import WatermarkDetector
from src.watermark_lab.core.benchmark import WatermarkBenchmark


class TestWatermarkFactory:
    """Test the watermark factory."""
    
    def test_create_watermark(self):
        """Test watermark creation."""
        watermark = WatermarkFactory.create("kirchenbauer", gamma=0.25, delta=2.0)
        assert isinstance(watermark, BaseWatermark)
        assert watermark.gamma == 0.25
        assert watermark.delta == 2.0
    
    def test_list_methods(self):
        """Test listing available methods."""
        methods = WatermarkFactory.list_methods()
        assert isinstance(methods, list)
        assert "kirchenbauer" in methods
        assert "markllm" in methods
        assert "aaronson" in methods
        assert "zhao" in methods
    
    def test_unknown_method(self):
        """Test creating unknown watermark method."""
        with pytest.raises(ValueError):
            WatermarkFactory.create("unknown_method")


class TestKirchenbauerWatermark:
    """Test Kirchenbauer watermarking algorithm."""
    
    def test_initialization(self):
        """Test Kirchenbauer watermark initialization."""
        watermark = WatermarkFactory.create("kirchenbauer", gamma=0.5, delta=3.0)
        assert watermark.gamma == 0.5
        assert watermark.delta == 3.0
    
    def test_default_parameters(self):
        """Test default parameter values."""
        watermark = WatermarkFactory.create("kirchenbauer")
        assert watermark.gamma == 0.25
        assert watermark.delta == 2.0
    
    def test_generate_text(self):
        """Test text generation."""
        watermark = WatermarkFactory.create("kirchenbauer")
        result = watermark.generate("Test prompt", max_length=50)
        assert isinstance(result, str)
        assert len(result) > len("Test prompt")
        assert "Test prompt" in result
    
    def test_get_config(self):
        """Test configuration retrieval."""
        watermark = WatermarkFactory.create("kirchenbauer", gamma=0.3, delta=2.5)
        config = watermark.get_config()
        assert config["method"] == "kirchenbauer"
        assert config["gamma"] == 0.3
        assert config["delta"] == 2.5


class TestWatermarkDetection:
    """Test watermark detection functionality."""
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        config = {"method": "kirchenbauer", "gamma": 0.25, "delta": 2.0}
        detector = WatermarkDetector(config)
        assert detector.method == "kirchenbauer"
        assert detector.config["gamma"] == 0.25
    
    def test_detect_watermarked_text(self):
        """Test detecting watermarked text."""
        # Generate watermarked text
        watermark = WatermarkFactory.create("kirchenbauer", gamma=0.25, delta=2.0, seed=42)
        watermarked_text = watermark.generate("This is a test prompt", max_length=50)
        
        # Detect watermark
        detector = WatermarkDetector(watermark.get_config())
        result = detector.detect(watermarked_text)
        
        assert hasattr(result, 'is_watermarked')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'p_value')
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
    
    def test_detect_clean_text(self):
        """Test detecting clean (non-watermarked) text."""
        clean_text = "This is clean text without any watermark."
        config = {"method": "kirchenbauer", "gamma": 0.25, "delta": 2.0}
        detector = WatermarkDetector(config)
        
        result = detector.detect(clean_text)
        assert hasattr(result, 'is_watermarked')
        assert isinstance(result.confidence, float)
    
    def test_empty_text_detection(self):
        """Test detection with empty text."""
        config = {"method": "kirchenbauer"}
        detector = WatermarkDetector(config)
        
        result = detector.detect("")
        assert result.is_watermarked == False
        assert result.confidence == 0.0


class TestBenchmarking:
    """Test benchmarking functionality."""
    
    def test_benchmark_initialization(self):
        """Test benchmark suite initialization."""
        benchmark = WatermarkBenchmark(num_samples=5)
        assert benchmark.num_samples == 5
        assert len(benchmark.test_prompts) >= 5
    
    def test_compare_methods(self):
        """Test comparing watermark methods."""
        benchmark = WatermarkBenchmark(num_samples=3)
        methods = ["kirchenbauer", "markllm"]
        prompts = benchmark.test_prompts[:3]
        
        results = benchmark.compare(methods, prompts, ["detectability", "quality"])
        
        assert isinstance(results, dict)
        assert "kirchenbauer" in results
        assert "markllm" in results
        
        for method_results in results.values():
            assert isinstance(method_results, dict)
            assert "detectability" in method_results or "quality" in method_results
    
    def test_single_method_benchmark(self):
        """Test benchmarking single method."""
        benchmark = WatermarkBenchmark(num_samples=2)
        
        config = {"gamma": 0.25, "delta": 2.0}
        result = benchmark.benchmark_method("kirchenbauer", config)
        
        assert hasattr(result, 'method')
        assert hasattr(result, 'quality_metrics')
        assert hasattr(result, 'detectability_metrics')
        assert result.method == "kirchenbauer"


@pytest.mark.parametrize("method", ["kirchenbauer", "markllm", "aaronson", "zhao"])
class TestAllMethods:
    """Test all watermarking methods."""
    
    def test_method_creation(self, method):
        """Test creating all supported methods."""
        watermark = WatermarkFactory.create(method)
        assert watermark is not None
        config = watermark.get_config()
        assert config["method"] == method
    
    def test_method_generation(self, method):
        """Test text generation for all methods."""
        watermark = WatermarkFactory.create(method)
        result = watermark.generate("Test prompt for " + method, max_length=30)
        assert isinstance(result, str)
        assert len(result) > len("Test prompt for " + method)
    
    def test_method_detection(self, method):
        """Test detection for all methods."""
        # Generate watermarked text
        watermark = WatermarkFactory.create(method, seed=42)
        text = watermark.generate("Detection test", max_length=40)
        
        # Test detection
        detector = WatermarkDetector(watermark.get_config())
        result = detector.detect(text)
        
        assert hasattr(result, 'is_watermarked')
        assert hasattr(result, 'confidence')
        assert result.method == method


@pytest.mark.integration
class TestEndToEndWorkflow:
    """End-to-end integration tests."""
    
    def test_complete_workflow(self):
        """Test complete watermarking workflow."""
        # 1. Create watermark
        watermark = WatermarkFactory.create("kirchenbauer", gamma=0.25, delta=2.0, seed=123)
        
        # 2. Generate watermarked text
        prompt = "This is a comprehensive test of the watermarking system"
        watermarked_text = watermark.generate(prompt, max_length=100)
        
        # 3. Detect watermark
        detector = WatermarkDetector(watermark.get_config())
        detection_result = detector.detect(watermarked_text)
        
        # 4. Verify results
        assert isinstance(watermarked_text, str)
        assert len(watermarked_text) > len(prompt)
        assert detection_result.method == "kirchenbauer"
        assert isinstance(detection_result.confidence, float)
    
    def test_benchmark_integration(self):
        """Test benchmarking integration."""
        benchmark = WatermarkBenchmark(num_samples=2)
        methods = ["kirchenbauer", "markllm"]
        
        results = benchmark.compare(methods, benchmark.test_prompts[:2], ["detectability"])
        
        assert len(results) == 2
        assert "kirchenbauer" in results
        assert "markllm" in results
    
    def test_attack_simulation_integration(self):
        """Test attack simulation integration."""
        from src.watermark_lab.core.attacks import AttackSimulator
        
        # Generate watermarked text
        watermark = WatermarkFactory.create("kirchenbauer")
        text = watermark.generate("Attack test text", max_length=50)
        
        # Run attack
        simulator = AttackSimulator()
        attack_result = simulator.run_attack(text, "paraphrase", strength="medium")
        
        assert attack_result.original_text == text
        assert isinstance(attack_result.attacked_text, str)
        assert attack_result.attack_type == "paraphrase"
        assert isinstance(attack_result.quality_score, float)
        assert isinstance(attack_result.similarity_score, float)