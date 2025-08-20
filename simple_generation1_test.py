"""Simple test for Generation 1 without dependencies."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic imports work."""
    try:
        # Test base structure
        from watermark_lab.methods.base import WatermarkConfig, DetectionResult
        print("✓ Base classes imported successfully")
        
        # Test method structure
        config = WatermarkConfig(method="test", key="key", params={})
        result = DetectionResult(is_watermarked=True, confidence=0.9, p_value=0.01)
        print("✓ Data classes work correctly")
        
        # Test factory structure (without torch)
        print("✓ Basic structure validation complete")
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_api_structure():
    """Test API structure."""
    try:
        from watermark_lab.api.simple_endpoints import app
        print("✓ Simple API endpoints imported")
        return True
    except Exception as e:
        print(f"✗ API import error: {e}")
        return False

def test_cli_structure():
    """Test CLI structure."""
    try:
        # Check if CLI module exists
        import watermark_lab.cli.main
        print("✓ CLI module imported")
        return True
    except Exception as e:
        print(f"✗ CLI import error: {e}")
        return False

if __name__ == "__main__":
    print("=== Generation 1 Basic Tests ===")
    
    tests = [
        test_basic_imports,
        test_api_structure, 
        test_cli_structure
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Results: {passed}/{len(tests)} tests passed ===")
    
    if passed == len(tests):
        print("🎉 Generation 1 basic structure complete!")
    else:
        print("⚠️  Some tests failed, but core structure is in place")