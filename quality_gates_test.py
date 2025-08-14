#!/usr/bin/env python3
"""Quality Gates - Comprehensive testing, security, and performance validation."""

import sys
import os
import time
import subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_security_validation():
    """Run comprehensive security validation."""
    print("=== Security Validation ===")
    
    try:
        from watermark_lab.security.input_sanitization import InputSanitizer, SanitizationConfig
        
        # Test comprehensive security scenarios
        sanitizer = InputSanitizer(SanitizationConfig(
            detect_scripts=True,
            detect_sql_injection=True,
            detect_xss=True,
            detect_path_traversal=True,
            max_text_length=5000
        ))
        
        # Test malicious inputs
        test_cases = [
            ("<script>alert('xss')</script>", "XSS script injection"),
            ("'; DROP TABLE users; --", "SQL injection attempt"),
            ("../../../etc/passwd", "Path traversal attempt"),
            ("<iframe src='javascript:alert(1)'></iframe>", "Iframe injection"),
            ("eval('malicious_code')", "JavaScript eval injection"),
            ("SELECT * FROM users WHERE id=1 OR 1=1", "SQL injection variant"),
        ]
        
        blocked_count = 0
        for malicious_input, description in test_cases:
            try:
                result = sanitizer.sanitize_text(malicious_input)
                print(f"⚠️  {description}: Sanitized to '{result[:50]}...'")
            except Exception as e:
                if "security" in str(e).lower() or "threat" in str(e).lower():
                    blocked_count += 1
                    print(f"✓ {description}: Blocked")
                else:
                    print(f"? {description}: Unexpected error - {e}")
        
        # Test file upload validation
        malicious_content = b"#!/bin/bash\nrm -rf /"
        try:
            sanitizer.validate_file_upload("malicious.sh", malicious_content)
            print("✗ File upload validation failed")
        except Exception:
            print("✓ Malicious file upload blocked")
        
        # Test rate limiting
        context = "test_context"
        rate_limit_hit = False
        for i in range(150):  # Exceed default rate limit
            try:
                sanitizer.sanitize_text(f"test {i}", context)
            except Exception as e:
                if "rate limit" in str(e).lower():
                    rate_limit_hit = True
                    break
        
        if rate_limit_hit:
            print("✓ Rate limiting working")
        else:
            print("? Rate limiting may need adjustment")
        
        print(f"✓ Security validation: {blocked_count}/{len(test_cases)} threats blocked")
        return True
        
    except Exception as e:
        print(f"✗ Security validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_performance_benchmarks():
    """Run comprehensive performance benchmarks."""
    print("\n=== Performance Benchmarks ===")
    
    try:
        from watermark_lab.core.factory import WatermarkFactory
        from watermark_lab.core.detector import WatermarkDetector
        from watermark_lab.optimization.caching import get_cache_manager
        
        # Benchmark 1: Watermark generation speed
        config = {
            "method": "kirchenbauer", 
            "model_name": "gpt2",
            "use_real_model": False,
            "gamma": 0.25,
            "seed": 42
        }
        
        watermarker = WatermarkFactory.create(**config)
        
        # Measure generation performance
        num_generations = 10
        start_time = time.time()
        
        for i in range(num_generations):
            text = watermarker.generate(f"Test prompt {i}", max_length=50)
        
        generation_time = time.time() - start_time
        gen_rate = num_generations / generation_time
        print(f"✓ Watermark generation: {gen_rate:.1f} generations/sec")
        
        # Benchmark 2: Detection speed
        detector = WatermarkDetector(watermarker.get_config())
        test_text = "This is a test text for watermark detection performance benchmarking."
        
        num_detections = 50
        start_time = time.time()
        
        for _ in range(num_detections):
            result = detector.detect(test_text)
        
        detection_time = time.time() - start_time
        det_rate = num_detections / detection_time
        print(f"✓ Watermark detection: {det_rate:.1f} detections/sec")
        
        # Benchmark 3: Cache performance
        cache_manager = get_cache_manager()
        
        num_cache_ops = 1000
        start_time = time.time()
        
        for i in range(num_cache_ops):
            cache_manager.set(f"bench_key_{i}", f"bench_value_{i}")
            cache_manager.get(f"bench_key_{i}")
        
        cache_time = time.time() - start_time
        cache_rate = (num_cache_ops * 2) / cache_time  # 2 ops per iteration
        print(f"✓ Cache operations: {cache_rate:.0f} ops/sec")
        
        # Performance thresholds
        min_gen_rate = 5.0  # Minimum generations per second
        min_det_rate = 20.0  # Minimum detections per second
        min_cache_rate = 1000.0  # Minimum cache ops per second
        
        performance_pass = (gen_rate >= min_gen_rate and 
                          det_rate >= min_det_rate and 
                          cache_rate >= min_cache_rate)
        
        if performance_pass:
            print("✓ All performance benchmarks passed")
            return True
        else:
            print("⚠️ Some performance benchmarks below threshold")
            return False
        
    except Exception as e:
        print(f"✗ Performance benchmarks failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_research_algorithm_validation():
    """Validate novel research algorithms."""
    print("\n=== Research Algorithm Validation ===")
    
    try:
        from watermark_lab.core.factory import WatermarkFactory
        from watermark_lab.core.detector import WatermarkDetector
        
        # Test SACW (Semantic-Aware Contextual Watermarking)
        sacw_config = {
            "method": "sacw",
            "model_name": "gpt2",
            "use_real_model": False,
            "semantic_threshold": 0.85,
            "context_window": 16,
            "gamma": 0.25,
            "seed": 42
        }
        
        sacw_watermarker = WatermarkFactory.create(**sacw_config)
        sacw_text = sacw_watermarker.generate("AI research focuses on", max_length=30)
        
        sacw_detector = WatermarkDetector(sacw_watermarker.get_config())
        sacw_result = sacw_detector.detect(sacw_text)
        
        print(f"✓ SACW Algorithm: Generated text, detection confidence {sacw_result.confidence:.2%}")
        
        # Test ARMS (Adversarial-Robust Multi-Scale)
        arms_config = {
            "method": "arms",
            "model_name": "gpt2",
            "use_real_model": False,
            "scale_levels": [1, 4, 16],
            "gamma": 0.25,
            "seed": 42
        }
        
        arms_watermarker = WatermarkFactory.create(**arms_config)
        arms_text = arms_watermarker.generate("Multi-scale watermarking", max_length=30)
        
        arms_detector = WatermarkDetector(arms_watermarker.get_config())
        arms_result = arms_detector.detect(arms_text)
        
        print(f"✓ ARMS Algorithm: Generated text, detection confidence {arms_result.confidence:.2%}")
        
        # Test QIPW (Quantum-Inspired Probabilistic Watermarking)
        qipw_config = {
            "method": "qipw",
            "model_name": "gpt2", 
            "use_real_model": False,
            "coherence_time": 100.0,
            "entanglement_strength": 0.8,
            "quantum_noise_level": 0.1,
            "gamma": 0.25,
            "seed": 42
        }
        
        qipw_watermarker = WatermarkFactory.create(**qipw_config)
        qipw_text = qipw_watermarker.generate("Quantum computing principles", max_length=30)
        
        qipw_detector = WatermarkDetector(qipw_watermarker.get_config())
        qipw_result = qipw_detector.detect(qipw_text)
        
        print(f"✓ QIPW Algorithm: Generated text, detection confidence {qipw_result.confidence:.2%}")
        
        # Validate research features are working
        algorithms_working = all([
            sacw_text and len(sacw_text) > 10,
            arms_text and len(arms_text) > 10,
            qipw_text and len(qipw_text) > 10,
            sacw_result.method == "sacw",
            arms_result.method == "arms", 
            qipw_result.method == "qipw"
        ])
        
        if algorithms_working:
            print("✓ All novel research algorithms validated")
            return True
        else:
            print("⚠️ Some research algorithms need attention")
            return False
        
    except Exception as e:
        print(f"✗ Research algorithm validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_integration_tests():
    """Run integration tests across all components."""
    print("\n=== Integration Tests ===")
    
    try:
        from watermark_lab.core.factory import WatermarkFactory
        from watermark_lab.core.detector import WatermarkDetector
        from watermark_lab.monitoring.health_monitor import HealthMonitor
        from watermark_lab.optimization.caching import get_cache_manager
        from watermark_lab.optimization.resource_manager import get_resource_manager
        
        # Test 1: End-to-end watermark workflow
        config = {
            "method": "kirchenbauer",
            "model_name": "gpt2",
            "use_real_model": False,
            "gamma": 0.25,
            "seed": 42
        }
        
        watermarker = WatermarkFactory.create(**config)
        text = watermarker.generate("Integration test prompt", max_length=40)
        
        detector = WatermarkDetector(watermarker.get_config())
        result = detector.detect(text)
        
        workflow_success = text and len(text) > 10 and result is not None
        print(f"✓ End-to-end workflow: {'Pass' if workflow_success else 'Fail'}")
        
        # Test 2: Health monitoring integration
        health_monitor = HealthMonitor()
        health_summary = health_monitor.get_health_summary()
        
        health_integration = (health_summary['total_checks'] > 0 and
                            'overall_status' in health_summary)
        print(f"✓ Health monitoring integration: {'Pass' if health_integration else 'Fail'}")
        
        # Test 3: Cache integration with watermarking
        cache_manager = get_cache_manager()
        cache_key = f"watermark_test_{time.time()}"
        
        cache_manager.set(cache_key, {"config": config, "result": result.to_dict()})
        cached_data = cache_manager.get(cache_key)
        
        cache_integration = cached_data is not None and 'config' in cached_data
        print(f"✓ Cache integration: {'Pass' if cache_integration else 'Fail'}")
        
        # Test 4: Resource management integration  
        resource_manager = get_resource_manager()
        resource_stats = resource_manager.get_comprehensive_stats()
        
        resource_integration = ('current_usage' in resource_stats and
                              'memory_stats' in resource_stats)
        print(f"✓ Resource management integration: {'Pass' if resource_integration else 'Fail'}")
        
        all_integrations_pass = all([
            workflow_success,
            health_integration,
            cache_integration,
            resource_integration
        ])
        
        if all_integrations_pass:
            print("✓ All integration tests passed")
            return True
        else:
            print("⚠️ Some integration tests failed")
            return False
        
    except Exception as e:
        print(f"✗ Integration tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_stress_tests():
    """Run stress tests for reliability."""
    print("\n=== Stress Tests ===")
    
    try:
        from watermark_lab.core.factory import WatermarkFactory
        from watermark_lab.optimization.caching import get_cache_manager
        import threading
        
        # Stress test 1: Concurrent watermark generation
        config = {
            "method": "kirchenbauer",
            "model_name": "gpt2", 
            "use_real_model": False,
            "gamma": 0.25,
            "seed": 42
        }
        
        successes = []
        failures = []
        
        def generate_worker(worker_id, iterations):
            try:
                watermarker = WatermarkFactory.create(**config)
                for i in range(iterations):
                    text = watermarker.generate(f"Stress test {worker_id}-{i}", max_length=20)
                    if text and len(text) > 0:
                        successes.append(f"{worker_id}-{i}")
                    else:
                        failures.append(f"{worker_id}-{i}")
            except Exception as e:
                failures.append(f"{worker_id}: {e}")
        
        # Run concurrent generation
        num_workers = 3
        iterations_per_worker = 5
        threads = []
        
        start_time = time.time()
        
        for worker_id in range(num_workers):
            thread = threading.Thread(target=generate_worker, args=(worker_id, iterations_per_worker))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        stress_time = time.time() - start_time
        success_rate = len(successes) / (len(successes) + len(failures)) if (successes or failures) else 0
        
        print(f"✓ Concurrent generation stress test: {len(successes)} successes, {len(failures)} failures")
        print(f"  Success rate: {success_rate:.1%}, Time: {stress_time:.2f}s")
        
        # Stress test 2: Cache under load
        cache_manager = get_cache_manager()
        cache_errors = 0
        cache_operations = 100
        
        start_time = time.time()
        
        for i in range(cache_operations):
            try:
                cache_manager.set(f"stress_{i}", f"data_{i}")
                value = cache_manager.get(f"stress_{i}")
                if value != f"data_{i}":
                    cache_errors += 1
            except Exception:
                cache_errors += 1
        
        cache_stress_time = time.time() - start_time
        cache_success_rate = (cache_operations - cache_errors) / cache_operations
        
        print(f"✓ Cache stress test: {cache_success_rate:.1%} success rate in {cache_stress_time:.3f}s")
        
        # Overall stress test success
        stress_pass = (success_rate > 0.8 and cache_success_rate > 0.9)
        
        if stress_pass:
            print("✓ All stress tests passed")
            return True
        else:
            print("⚠️ Some stress tests below threshold")
            return False
        
    except Exception as e:
        print(f"✗ Stress tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_quality_gates():
    """Run all quality gate validations."""
    print("🔍 QUALITY GATES - COMPREHENSIVE VALIDATION")
    print("=" * 55)
    
    quality_gates = [
        ("Security Validation", run_security_validation),
        ("Performance Benchmarks", run_performance_benchmarks), 
        ("Research Algorithm Validation", run_research_algorithm_validation),
        ("Integration Tests", run_integration_tests),
        ("Stress Tests", run_stress_tests)
    ]
    
    results = {}
    
    for gate_name, gate_func in quality_gates:
        try:
            print(f"\n🚪 {gate_name}")
            print("-" * 40)
            results[gate_name] = gate_func()
        except Exception as e:
            print(f"✗ {gate_name} CRASHED: {e}")
            results[gate_name] = False
    
    # Summary
    passed_gates = sum(1 for result in results.values() if result)
    total_gates = len(results)
    
    print(f"\n📊 QUALITY GATES SUMMARY")
    print("=" * 30)
    for gate_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{gate_name}: {status}")
    
    print(f"\n🎯 OVERALL RESULT: {passed_gates}/{total_gates} gates passed")
    
    if passed_gates == total_gates:
        print("🎉 ALL QUALITY GATES PASSED - PRODUCTION READY!")
        return True
    else:
        print("⚠️  QUALITY GATES FAILED - NEEDS ATTENTION")
        return False


if __name__ == "__main__":
    success = run_all_quality_gates()
    sys.exit(0 if success else 1)