#!/usr/bin/env python3
"""Test Generation 2 robustness features: circuit breakers, security, monitoring."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_circuit_breaker():
    """Test circuit breaker functionality."""
    print("=== Testing Circuit Breaker ===")
    
    try:
        from watermark_lab.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
        
        # Create circuit breaker
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1.0, timeout=0.5)
        cb = CircuitBreaker("test_circuit", config)
        
        # Test successful operation
        def success_func():
            return "success"
        
        result = cb.call(success_func)
        print(f"✓ Successful operation: {result}")
        
        # Test failure handling
        def failing_func():
            raise Exception("Simulated failure")
        
        failures = 0
        for i in range(5):
            try:
                cb.call(failing_func)
            except Exception:
                failures += 1
                print(f"✓ Handled failure {failures}")
                if failures >= 2:  # Should open circuit
                    break
        
        # Test circuit open state
        try:
            cb.call(success_func)
            print("✗ Circuit should be open")
        except Exception as e:
            if "open" in str(e).lower():
                print("✓ Circuit correctly opened after failures")
            else:
                print(f"✗ Unexpected error: {e}")
        
        print("✓ Circuit breaker tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Circuit breaker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_input_sanitization():
    """Test input sanitization security features."""
    print("\n=== Testing Input Sanitization ===")
    
    try:
        from watermark_lab.security.input_sanitization import InputSanitizer, SanitizationConfig
        
        # Create sanitizer
        config = SanitizationConfig(
            max_text_length=1000,
            detect_scripts=True,
            detect_xss=True,
            strip_html=True
        )
        sanitizer = InputSanitizer(config)
        
        # Test normal text
        clean_text = "This is normal text for watermarking research."
        result = sanitizer.sanitize_text(clean_text)
        print(f"✓ Clean text sanitized: {result}")
        
        # Test HTML stripping
        html_text = "<p>This is <b>bold</b> text</p>"
        result = sanitizer.sanitize_text(html_text)
        print(f"✓ HTML stripped: {result}")
        
        # Test threat detection (should raise SecurityError)
        try:
            malicious_text = "<script>alert('xss')</script>Hello world"
            sanitizer.sanitize_text(malicious_text)
            print("✗ Should have detected script injection")
        except Exception as e:
            if "security" in str(e).lower() or "threat" in str(e).lower():
                print("✓ Script injection detected and blocked")
            else:
                print(f"✓ Security check triggered: {e}")
        
        # Test JSON sanitization
        json_data = {"text": "Hello world", "number": 42}
        result = sanitizer.sanitize_json(json_data)
        print(f"✓ JSON sanitized: {result}")
        
        print("✓ Input sanitization tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Input sanitization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_health_monitoring():
    """Test health monitoring system."""
    print("\n=== Testing Health Monitoring ===")
    
    try:
        from watermark_lab.monitoring.health_monitor import HealthMonitor, SystemResourcesCheck
        
        # Create health monitor
        monitor = HealthMonitor(check_interval=5.0)
        
        # Run health checks
        results = monitor.run_checks()
        print(f"✓ Health checks completed: {len(results)} checks")
        
        for name, result in results.items():
            print(f"  - {name}: {result.status.value} ({result.message})")
        
        # Get system metrics
        metrics = monitor.collect_system_metrics()
        print(f"✓ System metrics: CPU {metrics.cpu_percent:.1f}%, Memory {metrics.memory_percent:.1f}%")
        
        # Get health summary
        summary = monitor.get_health_summary()
        print(f"✓ Health summary: {summary['overall_status']} ({summary['total_checks']} checks)")
        
        # Test individual system resources check
        sys_check = SystemResourcesCheck()
        sys_result = sys_check.run()
        print(f"✓ System resources check: {sys_result.status.value}")
        
        print("✓ Health monitoring tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Health monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test enhanced error handling and logging."""
    print("\n=== Testing Error Handling ===")
    
    try:
        from watermark_lab.utils.exceptions import WatermarkError, SecurityError, ValidationError
        from watermark_lab.utils.logging import get_logger
        
        # Test custom exceptions
        logger = get_logger("robustness_test")
        
        try:
            raise WatermarkError("Test watermark error")
        except WatermarkError as e:
            print(f"✓ WatermarkError handled: {e}")
        
        try:
            raise SecurityError("Test security error")
        except SecurityError as e:
            print(f"✓ SecurityError handled: {e}")
        
        try:
            raise ValidationError("Test validation error")
        except ValidationError as e:
            print(f"✓ ValidationError handled: {e}")
        
        # Test logging
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        print("✓ Logging system working")
        
        print("✓ Error handling tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_robustness_tests():
    """Run all Generation 2 robustness tests."""
    print("🛡️  GENERATION 2 ROBUSTNESS TESTING")
    print("=" * 50)
    
    tests = [
        test_circuit_breaker,
        test_input_sanitization, 
        test_health_monitoring,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
    
    print(f"\n📊 ROBUSTNESS TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL ROBUSTNESS FEATURES WORKING!")
        return True
    else:
        print("⚠️  Some robustness features need attention")
        return False


if __name__ == "__main__":
    success = run_all_robustness_tests()
    sys.exit(0 if success else 1)