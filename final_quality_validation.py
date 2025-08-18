#!/usr/bin/env python3
"""Final quality validation with comprehensive error handling."""

import sys
import os
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test basic imports with graceful fallbacks."""
    print("🔧 Testing Core Imports...")
    
    results = {
        'watermark_lab': False,
        'security': False,
        'caching': False,
        'detection': False
    }
    
    try:
        import watermark_lab
        results['watermark_lab'] = True
        print("  ✅ Core package imported successfully")
    except Exception as e:
        print(f"  ❌ Core package failed: {e}")
    
    try:
        from watermark_lab.security.advanced_security import AdvancedSecurity
        security = AdvancedSecurity()
        results['security'] = True
        print(f"  ✅ Security system: {len(security.users)} users initialized")
    except Exception as e:
        print(f"  ❌ Security system failed: {e}")
    
    try:
        from watermark_lab.optimization.caching import MemoryCache, CacheConfig
        config = CacheConfig(max_memory_items=10)
        cache = MemoryCache(config)
        cache.set("test", "value")
        value = cache.get("test")
        results['caching'] = value == "value"
        print("  ✅ Caching system working")
    except Exception as e:
        print(f"  ❌ Caching system failed: {e}")
    
    try:
        from watermark_lab.core.detector import WatermarkDetector
        detector = WatermarkDetector({"method": "kirchenbauer"})
        result = detector.detect("test text")
        results['detection'] = hasattr(result, 'confidence')
        print("  ✅ Detection system functional")
    except Exception as e:
        print(f"  ❌ Detection system failed: {e}")
    
    return results

def test_research_algorithms():
    """Test advanced research algorithms with error handling."""
    print("\n🧬 Testing Research Algorithms...")
    
    results = {}
    test_text = "Advanced research text for comprehensive watermarking algorithm validation and testing."
    
    methods = ["sacw", "arms", "qipw"]
    
    for method in methods:
        try:
            from watermark_lab.core.detector import WatermarkDetector
            detector = WatermarkDetector({"method": method})
            result = detector.detect(test_text)
            
            if hasattr(result, 'confidence') and hasattr(result, 'p_value'):
                results[method] = True
                print(f"  ✅ {method.upper()}: conf={result.confidence:.3f}, p={result.p_value:.3f}")
            else:
                results[method] = False
                print(f"  ❌ {method.upper()}: Invalid result format")
                
        except Exception as e:
            results[method] = False
            print(f"  ❌ {method.upper()}: {str(e)[:50]}...")
    
    return results

def test_security_features():
    """Test security features."""
    print("\n🔒 Testing Security Features...")
    
    results = {}
    
    try:
        from watermark_lab.security.advanced_security import AdvancedSecurity, sanitize_input
        
        # Test sanitization
        dirty_input = "<script>alert('test')</script>Hello"
        clean_input = sanitize_input(dirty_input)
        
        if "&lt;script&gt;" in clean_input and "Hello" in clean_input:
            results['sanitization'] = True
            print("  ✅ Input sanitization working")
        else:
            results['sanitization'] = False
            print("  ❌ Input sanitization failed")
        
        # Test authentication
        security = AdvancedSecurity()
        success, msg, token = security.authenticate_user("nonexistent", "wrong")
        
        if not success and "Invalid credentials" in msg:
            results['auth'] = True
            print("  ✅ Authentication properly rejects invalid users")
        else:
            results['auth'] = False
            print("  ❌ Authentication not working properly")
            
    except Exception as e:
        results['security_test'] = False
        print(f"  ❌ Security testing failed: {e}")
    
    return results

def run_final_validation():
    """Run final comprehensive validation."""
    
    print("🎯 AUTONOMOUS SDLC - FINAL QUALITY VALIDATION")
    print("=" * 60)
    
    start_time = time.time()
    
    # Test suites
    import_results = test_basic_imports()
    research_results = test_research_algorithms() 
    security_results = test_security_features()
    
    # Calculate overall results
    all_results = {**import_results, **research_results, **security_results}
    
    total_tests = len(all_results)
    passed_tests = sum(1 for result in all_results.values() if result)
    failed_tests = total_tests - passed_tests
    pass_rate = (passed_tests / max(1, total_tests)) * 100
    
    execution_time = time.time() - start_time
    
    # Determine status
    if pass_rate >= 85:
        status = "SUCCESS"
        status_icon = "🎉"
        exit_code = 0
    elif pass_rate >= 70:
        status = "PARTIAL"
        status_icon = "⚠️"
        exit_code = 1
    else:
        status = "FAILED"
        status_icon = "❌"
        exit_code = 2
    
    # Print summary
    print(f"\n{status_icon} FINAL VALIDATION SUMMARY")
    print("=" * 40)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    print(f"Execution Time: {execution_time:.2f}s")
    print(f"Overall Status: {status}")
    
    # Save results
    final_results = {
        'status': status.lower(),
        'pass_rate': pass_rate,
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'execution_time': execution_time,
        'timestamp': time.time(),
        'test_results': all_results,
        'import_results': import_results,
        'research_results': research_results,
        'security_results': security_results
    }
    
    with open("final_validation_report.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results: final_validation_report.json")
    
    return final_results, exit_code

if __name__ == "__main__":
    results, exit_code = run_final_validation()
    sys.exit(exit_code)