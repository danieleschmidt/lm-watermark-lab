"""Autonomous quality validation for enhanced generations."""

import sys
import time
import subprocess
import json
from pathlib import Path

def run_command(cmd, description):
    """Run command and return result."""
    print(f"\n🔍 {description}")
    print(f"Command: {cmd}")
    
    try:
        # Use bash explicitly to support source command
        result = subprocess.run(
            ["bash", "-c", cmd], 
            capture_output=True, 
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"✅ Success: {description}")
            if result.stdout:
                print(f"Output: {result.stdout[:500]}...")
            return True
        else:
            print(f"❌ Failed: {description}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {description}")
        return False
    except Exception as e:
        print(f"💥 Exception: {description} - {e}")
        return False

def validate_imports():
    """Validate all new modules can be imported."""
    modules_to_test = [
        "watermark_lab.utils.enhanced_resilience",
        "watermark_lab.security.threat_detection", 
        "watermark_lab.monitoring.realtime_analytics",
        "watermark_lab.optimization.distributed_processing",
        "watermark_lab.optimization.adaptive_scaling"
    ]
    
    success_count = 0
    for module in modules_to_test:
        cmd = f"source venv/bin/activate && export PYTHONPATH=/root/repo/src:$PYTHONPATH && python -c 'import {module}; print(\"✅ {module} imported successfully\")'"
        if run_command(cmd, f"Import {module}"):
            success_count += 1
    
    return success_count == len(modules_to_test)

def validate_functionality():
    """Validate core functionality of new modules."""
    
    # Test resilience patterns
    resilience_test = '''
from watermark_lab.utils.enhanced_resilience import ResilienceManager, ResilienceConfig
config = ResilienceConfig(max_retries=2)
manager = ResilienceManager(config)

def test_function():
    return "success"

result = manager.execute_with_resilience(test_function, service_name="test")
assert result == "success"
print("✅ Resilience patterns working")
'''
    
    # Test threat detection
    threat_test = '''
from watermark_lab.security.threat_detection import ThreatDetectionEngine
engine = ThreatDetectionEngine()
threats = engine.analyze_request("192.168.1.1", "/test", "normal request")
print(f"✅ Threat detection working - analyzed {len(threats)} threats")
'''
    
    # Test analytics
    analytics_test = '''
from watermark_lab.monitoring.realtime_analytics import RealTimeAnalytics, MetricAggregation
analytics = RealTimeAnalytics()
analytics.add_metric("test_metric", 100.0)
metric = analytics.get_metric("test_metric")
assert metric is not None
assert len(metric.points) == 1
avg = metric.aggregate(MetricAggregation.AVERAGE)
assert avg == 100.0
print("✅ Real-time analytics working")
'''
    
    # Test distributed processing
    distributed_test = '''
from watermark_lab.optimization.distributed_processing import DistributedProcessingEngine
engine = DistributedProcessingEngine(num_workers=1)
engine.register_function(lambda x: x * 2, "double")
engine.start()
try:
    task_id = engine.submit_task("double", args=(21,))
    result = engine.wait_for_task(task_id, timeout=5.0)
    assert result == 42
    print("✅ Distributed processing working")
finally:
    engine.stop()
'''
    
    # Test adaptive scaling
    scaling_test = '''
from watermark_lab.optimization.adaptive_scaling import MockMetricProvider, MockResourceManager, AdaptiveAutoScaler
provider = MockMetricProvider()
manager = MockResourceManager()
scaler = AdaptiveAutoScaler(provider, manager, evaluation_interval=0.1)
assert scaler.metric_provider is not None
assert scaler.resource_manager is not None
print("✅ Adaptive scaling working")
'''
    
    tests = [
        (resilience_test, "Resilience patterns"),
        (threat_test, "Threat detection"),
        (analytics_test, "Real-time analytics"),
        (distributed_test, "Distributed processing"),
        (scaling_test, "Adaptive scaling")
    ]
    
    success_count = 0
    for test_code, description in tests:
        cmd = f"source venv/bin/activate && export PYTHONPATH=/root/repo/src:$PYTHONPATH && python -c '{test_code}'"
        if run_command(cmd, f"Test {description}"):
            success_count += 1
    
    return success_count == len(tests)

def run_unit_tests():
    """Run the new unit tests."""
    
    # Run resilience tests
    resilience_cmd = f"source venv/bin/activate && export PYTHONPATH=/root/repo/src:$PYTHONPATH && python -m pytest tests/test_enhanced_generation2.py::TestEnhancedResilience -v"
    resilience_success = run_command(resilience_cmd, "Enhanced resilience unit tests")
    
    # Run threat detection tests  
    threat_cmd = f"source venv/bin/activate && export PYTHONPATH=/root/repo/src:$PYTHONPATH && python -m pytest tests/test_enhanced_generation2.py::TestThreatDetection -v"
    threat_success = run_command(threat_cmd, "Threat detection unit tests")
    
    # Run analytics tests
    analytics_cmd = f"source venv/bin/activate && export PYTHONPATH=/root/repo/src:$PYTHONPATH && python -m pytest tests/test_enhanced_generation2.py::TestRealTimeAnalytics -v"
    analytics_success = run_command(analytics_cmd, "Real-time analytics unit tests")
    
    # Run distributed processing tests
    distributed_cmd = f"source venv/bin/activate && export PYTHONPATH=/root/repo/src:$PYTHONPATH && python -m pytest tests/test_enhanced_generation3.py::TestDistributedProcessing -v"
    distributed_success = run_command(distributed_cmd, "Distributed processing unit tests")
    
    # Run scaling tests
    scaling_cmd = f"source venv/bin/activate && export PYTHONPATH=/root/repo/src:$PYTHONPATH && python -m pytest tests/test_enhanced_generation3.py::TestAdaptiveScaling -v"
    scaling_success = run_command(scaling_cmd, "Adaptive scaling unit tests")
    
    return all([resilience_success, threat_success, analytics_success, distributed_success, scaling_success])

def check_code_quality():
    """Check code quality with basic linting."""
    
    # Check Python syntax
    syntax_cmd = f"source venv/bin/activate && python -m py_compile src/watermark_lab/utils/enhanced_resilience.py"
    syntax_success = run_command(syntax_cmd, "Python syntax check")
    
    return syntax_success

def generate_quality_report():
    """Generate comprehensive quality report."""
    
    report = {
        "timestamp": time.time(),
        "validation_results": {},
        "summary": {}
    }
    
    print("\n" + "="*60)
    print("🚀 AUTONOMOUS QUALITY VALIDATION - ENHANCED GENERATIONS")
    print("="*60)
    
    # Run all validation checks
    checks = [
        ("imports", "Module Import Validation", validate_imports),
        ("functionality", "Core Functionality Validation", validate_functionality),  
        ("unit_tests", "Unit Test Execution", run_unit_tests),
        ("code_quality", "Code Quality Checks", check_code_quality)
    ]
    
    total_checks = len(checks)
    passed_checks = 0
    
    for check_id, description, check_func in checks:
        print(f"\n📋 {description}")
        print("-" * 40)
        
        try:
            result = check_func()
            report["validation_results"][check_id] = {
                "passed": result,
                "description": description
            }
            
            if result:
                passed_checks += 1
                print(f"✅ PASSED: {description}")
            else:
                print(f"❌ FAILED: {description}")
                
        except Exception as e:
            print(f"💥 ERROR: {description} - {e}")
            report["validation_results"][check_id] = {
                "passed": False,
                "description": description,
                "error": str(e)
            }
    
    # Calculate summary
    success_rate = (passed_checks / total_checks) * 100
    report["summary"] = {
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "failed_checks": total_checks - passed_checks,
        "success_rate": success_rate
    }
    
    print("\n" + "="*60)
    print("📊 QUALITY VALIDATION SUMMARY")
    print("="*60)
    print(f"Total Checks: {total_checks}")
    print(f"Passed: {passed_checks}")
    print(f"Failed: {total_checks - passed_checks}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("\n🎉 QUALITY GATES PASSED - ENHANCED GENERATIONS READY")
        status = "PASSED"
    else:
        print("\n⚠️  QUALITY GATES FAILED - IMPROVEMENTS NEEDED")
        status = "FAILED"
    
    report["summary"]["status"] = status
    
    # Save report
    with open("quality_validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Quality report saved to: quality_validation_report.json")
    
    return status == "PASSED"

if __name__ == "__main__":
    success = generate_quality_report()
    sys.exit(0 if success else 1)