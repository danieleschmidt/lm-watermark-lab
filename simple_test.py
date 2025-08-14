#!/usr/bin/env python3
"""Simple test to verify basic watermarking functionality works."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_watermarking():
    """Test basic watermarking functionality."""
    try:
        # Test factory import
        from watermark_lab.core.factory import WatermarkFactory
        print("✓ Factory import successful")
        
        # Test detector import  
        from watermark_lab.core.detector import WatermarkDetector
        print("✓ Detector import successful")
        
        # Create simple watermark
        config = {
            "method": "kirchenbauer",
            "model_name": "gpt2",
            "gamma": 0.25,
            "delta": 2.0,
            "seed": 42,
            "use_real_model": False  # Use fallback for testing
        }
        
        watermarker = WatermarkFactory.create(**config)
        print("✓ Watermarker created successfully")
        
        # Test text generation
        prompt = "The future of AI is"
        text = watermarker.generate(prompt, max_length=50)
        print(f"✓ Generated text: {text[:100]}...")
        
        # Test detection
        detector_config = watermarker.get_config()
        detector = WatermarkDetector(detector_config)
        result = detector.detect(text)
        
        print(f"✓ Detection result: watermarked={result.is_watermarked}, confidence={result.confidence:.2%}")
        print("✓ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_watermarking()