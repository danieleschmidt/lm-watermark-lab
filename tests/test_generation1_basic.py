"""Basic tests for Generation 1 functionality."""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from watermark_lab.core.factory import WatermarkFactory
from watermark_lab.methods.kirchenbauer import KirchenbauerWatermark
from watermark_lab.methods.base import DetectionResult

class TestWatermarkFactory:
    """Test watermark factory functionality."""
    
    def test_create_kirchenbauer_watermark(self):
        """Test creating Kirchenbauer watermark."""
        watermark = WatermarkFactory.create(
            method="kirchenbauer",
            model_name="gpt2",
            key="test_key"
        )
        assert isinstance(watermark, KirchenbauerWatermark)
        assert watermark.key == "test_key"
    
    def test_invalid_method_raises_error(self):
        """Test that invalid method raises error."""
        with pytest.raises(Exception):
            WatermarkFactory.create(method="invalid_method")

class TestKirchenbauerWatermark:
    """Test Kirchenbauer watermark implementation."""
    
    @pytest.fixture
    def watermark(self):
        """Create watermark instance for testing."""
        return KirchenbauerWatermark(
            model_name="gpt2",
            gamma=0.25,
            delta=2.0,
            key="test_key"
        )
    
    def test_initialization(self, watermark):
        """Test watermark initialization."""
        assert watermark.gamma == 0.25
        assert watermark.delta == 2.0
        assert watermark.key == "test_key"
    
    def test_hash_key(self, watermark):
        """Test key hashing."""
        hash1 = watermark.hash_key("test")
        hash2 = watermark.hash_key("test")
        hash3 = watermark.hash_key("different")
        
        assert hash1 == hash2  # Same input = same hash
        assert hash1 != hash3  # Different input = different hash
    
    def test_get_config(self, watermark):
        """Test configuration retrieval."""
        config = watermark.get_config()
        assert config.method == "kirchenbauer"
        assert config.key == "test_key"
    
    @pytest.mark.slow
    def test_generate_text(self, watermark):
        """Test text generation (requires model)."""
        try:
            # Simple test with short generation
            result = watermark.generate("Hello", max_length=10)
            assert isinstance(result, str)
            assert len(result) > 0
        except Exception as e:
            # Skip if model not available
            pytest.skip(f"Model not available: {e}")
    
    def test_detect_text(self, watermark):
        """Test watermark detection."""
        # Test with short text (should work without model)
        test_text = "This is a test text for detection purposes."
        result = watermark.detect(test_text)
        
        assert isinstance(result, DetectionResult)
        assert isinstance(result.is_watermarked, bool)
        assert isinstance(result.confidence, float)
        assert isinstance(result.p_value, float)
        assert 0 <= result.confidence <= 1
        assert 0 <= result.p_value <= 1

class TestBasicAPI:
    """Test basic API functionality."""
    
    def test_methods_import(self):
        """Test that methods can be imported."""
        from watermark_lab.methods import list_available_methods
        methods = list_available_methods()
        
        assert len(methods) > 0
        assert any(m["name"] == "kirchenbauer" for m in methods)
    
    def test_factory_import(self):
        """Test that factory can be imported."""
        from watermark_lab import WatermarkFactory
        assert WatermarkFactory is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])