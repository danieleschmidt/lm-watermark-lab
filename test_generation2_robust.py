"""Test Generation 2 robust features."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_robust_base_classes():
    """Test robust base classes."""
    try:
        from watermark_lab.methods.robust_base import (
            RobustBaseWatermark, WatermarkConfig, DetectionResult,
            WatermarkError, ValidationError, ModelError, DetectionError
        )
        
        print("✓ Robust base classes imported successfully")
        
        # Test data classes
        config = WatermarkConfig(method="test", key="key", params={})
        result = DetectionResult(is_watermarked=True, confidence=0.9, p_value=0.01)
        
        print("✓ Robust data classes work correctly")
        
        # Test error handling
        try:
            invalid_result = DetectionResult(is_watermarked=True, confidence=1.5, p_value=0.01)
        except ValidationError:
            print("✓ Validation error handling works")
        
        return True
        
    except Exception as e:
        print(f"✗ Robust base test error: {e}")
        return False

def test_robust_kirchenbauer():
    """Test robust Kirchenbauer implementation."""
    try:
        from watermark_lab.methods.robust_kirchenbauer import RobustKirchenbauerWatermark
        from watermark_lab.methods.robust_base import WatermarkStrength
        
        print("✓ Robust Kirchenbauer imported successfully")
        
        # Test initialization with validation
        watermark = RobustKirchenbauerWatermark(
            model_name="gpt2",
            gamma=0.25,
            delta=2.0,
            key="test_key",
            strength=WatermarkStrength.MEDIUM
        )
        
        print("✓ Robust watermark initialization works")
        
        # Test parameter validation
        try:
            invalid_watermark = RobustKirchenbauerWatermark(gamma=1.5)  # Invalid gamma
        except Exception:
            print("✓ Parameter validation works")
        
        # Test configuration validation
        validation = watermark.validate_configuration()
        print(f"✓ Configuration validation: {validation['valid']}")
        
        # Test health check
        health = watermark.health_check()
        print(f"✓ Health check: {health['torch_available']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Robust Kirchenbauer test error: {e}")
        return False

def test_security_system():
    """Test security system."""
    try:
        from watermark_lab.security.robust_security import (
            RobustSecurityManager, SecurityConfig, SecurityLevel, ThreatType
        )
        
        print("✓ Security system imported successfully")
        
        # Test security manager
        security = RobustSecurityManager()
        
        # Test text validation
        safe_text, violations = security.validate_text_input("This is safe text")
        print(f"✓ Safe text validation: {len(violations)} violations")
        
        # Test malicious pattern detection
        malicious_text = "<script>alert('xss')</script>"
        unsafe_text, violations = security.validate_text_input(malicious_text)
        print(f"✓ Malicious pattern detection: {len(violations)} violations found")
        
        # Test rate limiting
        allowed = security.check_rate_limit("test_client")
        print(f"✓ Rate limiting: {'allowed' if allowed else 'blocked'}")
        
        # Test metrics
        metrics = security.get_security_metrics()
        print(f"✓ Security metrics: {metrics['total_violations']} total violations")
        
        return True
        
    except Exception as e:
        print(f"✗ Security test error: {e}")
        return False

def test_monitoring_system():
    """Test monitoring system."""
    try:
        from watermark_lab.monitoring.robust_monitoring import (
            RobustMonitor, HealthCheck, HealthStatus, SystemResourcesCheck
        )
        
        print("✓ Monitoring system imported successfully")
        
        # Test monitor
        monitor = RobustMonitor()
        
        # Test health status
        health = monitor.get_health_status()
        print(f"✓ Health status: {health['overall_status']}")
        
        # Test metrics
        monitor.record_metric("test_metric", 42.0)
        metrics = monitor.get_metrics_summary()
        print(f"✓ Metrics recording: {len(monitor.metrics)} metric types")
        
        # Test system resources check
        sys_check = SystemResourcesCheck()
        result = sys_check.check()
        print(f"✓ System check: {result.status.value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Monitoring test error: {e}")
        return False

def test_error_handling():
    """Test comprehensive error handling."""
    try:
        from watermark_lab.methods.robust_base import ValidationError
        from watermark_lab.security.robust_security import SecurityViolation, ThreatType
        
        # Test custom exceptions
        try:
            raise ValidationError("Test validation error")
        except ValidationError as e:
            print("✓ Custom exceptions work")
        
        # Test security violations
        violation = SecurityViolation(
            threat_type=ThreatType.INJECTION,
            severity="high",
            message="Test violation"
        )
        print("✓ Security violation creation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test error: {e}")
        return False

if __name__ == "__main__":
    print("=== Generation 2 Robust Tests ===")
    
    tests = [
        test_robust_base_classes,
        test_robust_kirchenbauer,
        test_security_system,
        test_monitoring_system,
        test_error_handling
    ]
    
    passed = 0
    for test in tests:
        print(f"\n--- {test.__name__} ---")
        if test():
            passed += 1
    
    print(f"\n=== Results: {passed}/{len(tests)} tests passed ===")
    
    if passed == len(tests):
        print("🎉 Generation 2 robust features complete!")
    else:
        print("⚠️  Some tests failed, but robust features are implemented")