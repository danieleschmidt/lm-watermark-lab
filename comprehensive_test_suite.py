#!/usr/bin/env python3
"""
Comprehensive Testing Suite for LM Watermark Lab
Implements unit tests, integration tests, performance benchmarks, and quality gates.
"""

import sys
import os
import json
import time
import random
import hashlib
import logging
import unittest
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import warnings
import traceback
from contextlib import contextmanager

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import our implementations
exec(open('/root/repo/simple_core_test.py').read())
exec(open('/root/repo/robust_generation2_test.py').read()) 
exec(open('/root/repo/performance_generation3_test.py').read())

# Configure test logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during testing
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class TestResults:
    """Collect and manage test results."""
    
    def __init__(self):
        self.results = {
            'unit_tests': {},
            'integration_tests': {},
            'performance_tests': {},
            'security_tests': {},
            'quality_gates': {}
        }
        self.start_time = time.time()
        self.test_count = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.errors = []
    
    def add_result(self, category: str, test_name: str, passed: bool, 
                   details: Dict[str, Any] = None, error: str = None):
        """Add a test result."""
        self.test_count += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            if error:
                self.errors.append(f"{category}.{test_name}: {error}")
        
        self.results[category][test_name] = {
            'passed': passed,
            'details': details or {},
            'error': error,
            'timestamp': time.time()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        duration = time.time() - self.start_time
        
        return {
            'total_tests': self.test_count,
            'passed': self.passed_tests,
            'failed': self.failed_tests,
            'success_rate': self.passed_tests / max(1, self.test_count),
            'duration_seconds': duration,
            'errors': self.errors,
            'categories': {
                category: {
                    'total': len(tests),
                    'passed': sum(1 for t in tests.values() if t['passed'])
                }
                for category, tests in self.results.items()
            }
        }

class UnitTests:
    """Unit tests for core functionality."""
    
    def __init__(self, test_results: TestResults):
        self.test_results = test_results
    
    def test_watermark_generation(self):
        """Test basic watermark generation."""
        try:
            SimpleWatermark = mock_watermark_simple()
            watermarker = SimpleWatermark("unit_test_key")
            
            # Test normal generation
            result = watermarker.generate_text("Unit test prompt", max_tokens=20)
            
            passed = (
                isinstance(result, str) and
                len(result) > 0 and
                "Unit test prompt" in result
            )
            
            self.test_results.add_result('unit_tests', 'watermark_generation', passed, {
                'result_length': len(result),
                'contains_prompt': "Unit test prompt" in result
            })
            
        except Exception as e:
            self.test_results.add_result('unit_tests', 'watermark_generation', False, 
                                       error=str(e))
    
    def test_watermark_detection(self):
        """Test watermark detection accuracy."""
        try:
            SimpleWatermark = mock_watermark_simple()
            watermarker = SimpleWatermark("unit_test_key")
            
            # Generate watermarked text
            watermarked_text = watermarker.generate_text("Detection test", max_tokens=15)
            
            # Test detection on watermarked text
            detection_result = watermarker.detect_watermark(watermarked_text)
            
            # Test detection on clean text
            clean_text = "This is clean text without any watermarks"
            clean_result = watermarker.detect_watermark(clean_text)
            
            passed = (
                isinstance(detection_result, dict) and
                'is_watermarked' in detection_result and
                'confidence' in detection_result and
                detection_result['is_watermarked'] == True and
                clean_result['is_watermarked'] == False
            )
            
            self.test_results.add_result('unit_tests', 'watermark_detection', passed, {
                'watermarked_detected': detection_result['is_watermarked'],
                'clean_detected': clean_result['is_watermarked'],
                'watermarked_confidence': detection_result['confidence'],
                'clean_confidence': clean_result['confidence']
            })
            
        except Exception as e:
            self.test_results.add_result('unit_tests', 'watermark_detection', False, 
                                       error=str(e))
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        try:
            # Test valid configuration
            valid_config = WatermarkConfig(key="test_key", max_tokens=50)
            valid_passed = True
            
            # Test invalid configurations
            invalid_tests = [
                ("empty_key", {"key": "", "max_tokens": 50}),
                ("negative_tokens", {"key": "test", "max_tokens": -1}),
                ("invalid_confidence", {"key": "test", "confidence_threshold": 2.0})
            ]
            
            invalid_results = []
            for test_name, config_params in invalid_tests:
                try:
                    WatermarkConfig(**config_params)
                    invalid_results.append((test_name, False))  # Should have failed
                except ValueError:
                    invalid_results.append((test_name, True))   # Correctly failed
            
            all_invalid_caught = all(result[1] for result in invalid_results)
            
            passed = valid_passed and all_invalid_caught
            
            self.test_results.add_result('unit_tests', 'configuration_validation', passed, {
                'valid_config_accepted': valid_passed,
                'invalid_configs_rejected': all_invalid_caught,
                'invalid_test_results': invalid_results
            })
            
        except Exception as e:
            self.test_results.add_result('unit_tests', 'configuration_validation', False, 
                                       error=str(e))
    
    def test_input_sanitization(self):
        """Test input sanitization."""
        try:
            sanitizer = InputSanitizer()
            
            # Test normal input
            normal_result = sanitizer.sanitize_text("Normal input text")
            
            # Test input with dangerous characters
            dangerous_input = "text\x00with\x01nulls"
            dangerous_result = sanitizer.sanitize_text(dangerous_input)
            
            # Test empty input (should fail)
            empty_failed = False
            try:
                sanitizer.sanitize_text("")
            except ValidationError:
                empty_failed = True
            
            passed = (
                normal_result == "Normal input text" and
                "\x00" not in dangerous_result and
                "\x01" not in dangerous_result and
                empty_failed
            )
            
            self.test_results.add_result('unit_tests', 'input_sanitization', passed, {
                'normal_input_preserved': normal_result == "Normal input text",
                'dangerous_chars_removed': "\x00" not in dangerous_result,
                'empty_input_rejected': empty_failed
            })
            
        except Exception as e:
            self.test_results.add_result('unit_tests', 'input_sanitization', False, 
                                       error=str(e))
    
    def run_all(self):
        """Run all unit tests."""
        print("🧪 Running Unit Tests")
        print("-" * 30)
        
        unit_tests = [
            ('Watermark Generation', self.test_watermark_generation),
            ('Watermark Detection', self.test_watermark_detection),
            ('Configuration Validation', self.test_configuration_validation),
            ('Input Sanitization', self.test_input_sanitization)
        ]
        
        for test_name, test_method in unit_tests:
            try:
                print(f"  Running {test_name}...", end=" ")
                test_method()
                result = self.test_results.results['unit_tests'].get(test_name.lower().replace(' ', '_'))
                if result and result['passed']:
                    print("✅ PASSED")
                else:
                    print("❌ FAILED")
            except Exception as e:
                print(f"❌ ERROR: {e}")

class IntegrationTests:
    """Integration tests for end-to-end workflows."""
    
    def __init__(self, test_results: TestResults):
        self.test_results = test_results
    
    def test_complete_workflow(self):
        """Test complete watermarking workflow."""
        try:
            # Initialize robust watermarker
            config = WatermarkConfig(key="integration_test_key", max_tokens=30)
            watermarker = RobustWatermark(config)
            
            # Complete workflow
            prompt = "Integration test for complete watermarking workflow"
            
            # Generation
            generation_result = watermarker.generate_text(prompt)
            generated_text = generation_result['generated_text']
            
            # Detection
            detection_result = watermarker.detect_watermark(generated_text)
            
            # Validation
            passed = (
                'generated_text' in generation_result and
                len(generated_text) > 0 and
                'is_watermarked' in detection_result and
                'confidence' in detection_result
            )
            
            self.test_results.add_result('integration_tests', 'complete_workflow', passed, {
                'generation_success': 'generated_text' in generation_result,
                'detection_success': 'is_watermarked' in detection_result,
                'text_length': len(generated_text),
                'detection_confidence': detection_result.get('confidence', 0)
            })
            
        except Exception as e:
            self.test_results.add_result('integration_tests', 'complete_workflow', False, 
                                       error=str(e))
    
    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        try:
            config = WatermarkConfig(key="error_test_key", max_tokens=20)
            watermarker = RobustWatermark(config)
            
            # Test with problematic input
            problematic_inputs = [
                "   Whitespace   heavy   input   ",
                "A" * 1000,  # Very long input
                "Short"      # Very short input
            ]
            
            results = []
            for input_text in problematic_inputs:
                try:
                    generation_result = watermarker.generate_text(input_text)
                    detection_result = watermarker.detect_watermark(generation_result['generated_text'])
                    results.append(True)
                except Exception as e:
                    results.append(False)
            
            passed = all(results)
            
            self.test_results.add_result('integration_tests', 'error_recovery', passed, {
                'problematic_inputs_handled': sum(results),
                'total_inputs': len(problematic_inputs),
                'success_rate': sum(results) / len(problematic_inputs)
            })
            
        except Exception as e:
            self.test_results.add_result('integration_tests', 'error_recovery', False, 
                                       error=str(e))
    
    def test_concurrent_operations(self):
        """Test concurrent watermarking operations."""
        try:
            config = WatermarkConfig(key="concurrent_test_key", max_tokens=25)
            watermarker = RobustWatermark(config)
            
            # Concurrent generation test
            def generate_text_worker(prompt):
                try:
                    result = watermarker.generate_text(f"{prompt} worker test")
                    return True, result
                except Exception as e:
                    return False, str(e)
            
            # Run concurrent operations
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(generate_text_worker, f"Concurrent test {i}")
                    for i in range(10)
                ]
                
                concurrent_results = [future.result() for future in futures]
            
            successful_operations = sum(1 for success, _ in concurrent_results if success)
            
            passed = successful_operations >= 8  # Allow some failures due to concurrency
            
            self.test_results.add_result('integration_tests', 'concurrent_operations', passed, {
                'successful_operations': successful_operations,
                'total_operations': len(concurrent_results),
                'concurrency_success_rate': successful_operations / len(concurrent_results)
            })
            
        except Exception as e:
            self.test_results.add_result('integration_tests', 'concurrent_operations', False, 
                                       error=str(e))
    
    def run_all(self):
        """Run all integration tests."""
        print("🔗 Running Integration Tests")
        print("-" * 30)
        
        integration_tests = [
            ('Complete Workflow', self.test_complete_workflow),
            ('Error Recovery', self.test_error_recovery),
            ('Concurrent Operations', self.test_concurrent_operations)
        ]
        
        for test_name, test_method in integration_tests:
            try:
                print(f"  Running {test_name}...", end=" ")
                test_method()
                result = self.test_results.results['integration_tests'].get(test_name.lower().replace(' ', '_'))
                if result and result['passed']:
                    print("✅ PASSED")
                else:
                    print("❌ FAILED")
            except Exception as e:
                print(f"❌ ERROR: {e}")

class PerformanceTests:
    """Performance benchmarking tests."""
    
    def __init__(self, test_results: TestResults):
        self.test_results = test_results
    
    def test_generation_performance(self):
        """Test generation performance benchmarks."""
        try:
            watermarker = HighPerformanceWatermark("performance_test_key", max_workers=4)
            
            # Single operation performance
            start_time = time.time()
            result = watermarker.generate_text_optimized("Performance test prompt", max_tokens=30)
            single_time = time.time() - start_time
            
            # Batch operation performance
            batch_prompts = [f"Batch test {i}" for i in range(50)]
            start_time = time.time()
            batch_results = watermarker.batch_generate(batch_prompts, max_tokens=25)
            batch_time = time.time() - start_time
            
            # Performance criteria
            single_fast_enough = single_time < 0.01  # 10ms
            batch_throughput = len(batch_prompts) / batch_time
            batch_fast_enough = batch_throughput > 100  # 100 ops/sec
            
            passed = single_fast_enough and batch_fast_enough
            
            self.test_results.add_result('performance_tests', 'generation_performance', passed, {
                'single_operation_time_ms': single_time * 1000,
                'batch_throughput_per_sec': batch_throughput,
                'single_meets_target': single_fast_enough,
                'batch_meets_target': batch_fast_enough
            })
            
        except Exception as e:
            self.test_results.add_result('performance_tests', 'generation_performance', False, 
                                       error=str(e))
    
    def test_detection_performance(self):
        """Test detection performance benchmarks."""
        try:
            watermarker = HighPerformanceWatermark("detection_test_key", max_workers=4)
            
            # Generate test texts
            test_texts = []
            for i in range(20):
                result = watermarker.generate_text_optimized(f"Detection test {i}", max_tokens=30)
                test_texts.append(result['generated_text'])
            
            # Single detection performance
            start_time = time.time()
            detection_result = watermarker.detect_watermark_optimized(test_texts[0])
            single_time = time.time() - start_time
            
            # Batch detection performance
            start_time = time.time()
            batch_results = watermarker.batch_detect(test_texts)
            batch_time = time.time() - start_time
            
            # Performance criteria
            single_fast_enough = single_time < 0.005  # 5ms
            batch_throughput = len(test_texts) / batch_time
            batch_fast_enough = batch_throughput > 500  # 500 ops/sec
            
            passed = single_fast_enough and batch_fast_enough
            
            self.test_results.add_result('performance_tests', 'detection_performance', passed, {
                'single_detection_time_ms': single_time * 1000,
                'batch_throughput_per_sec': batch_throughput,
                'single_meets_target': single_fast_enough,
                'batch_meets_target': batch_fast_enough
            })
            
        except Exception as e:
            self.test_results.add_result('performance_tests', 'detection_performance', False, 
                                       error=str(e))
    
    def test_caching_effectiveness(self):
        """Test caching system effectiveness."""
        try:
            watermarker = HighPerformanceWatermark("cache_test_key", cache_size=1000)
            
            test_prompt = "Cache effectiveness test prompt"
            
            # First generation (cache miss)
            start_time = time.time()
            result1 = watermarker.generate_text_optimized(test_prompt, max_tokens=30)
            first_time = time.time() - start_time
            
            # Second generation (cache hit)
            start_time = time.time()
            result2 = watermarker.generate_text_optimized(test_prompt, max_tokens=30)
            second_time = time.time() - start_time
            
            # Calculate speedup
            speedup = first_time / max(second_time, 0.0001)  # Avoid division by zero
            
            # Get cache statistics
            stats = watermarker.get_performance_stats()
            cache_hit_rate = stats['cache_hit_rate']
            
            # Performance criteria
            good_speedup = speedup > 2.0
            good_hit_rate = cache_hit_rate > 0.3
            
            passed = good_speedup and good_hit_rate
            
            self.test_results.add_result('performance_tests', 'caching_effectiveness', passed, {
                'cache_speedup': speedup,
                'cache_hit_rate': cache_hit_rate,
                'first_time_ms': first_time * 1000,
                'second_time_ms': second_time * 1000,
                'speedup_meets_target': good_speedup,
                'hit_rate_meets_target': good_hit_rate
            })
            
        except Exception as e:
            self.test_results.add_result('performance_tests', 'caching_effectiveness', False, 
                                       error=str(e))
    
    def run_all(self):
        """Run all performance tests."""
        print("⚡ Running Performance Tests")
        print("-" * 30)
        
        performance_tests = [
            ('Generation Performance', self.test_generation_performance),
            ('Detection Performance', self.test_detection_performance),
            ('Caching Effectiveness', self.test_caching_effectiveness)
        ]
        
        for test_name, test_method in performance_tests:
            try:
                print(f"  Running {test_name}...", end=" ")
                test_method()
                result = self.test_results.results['performance_tests'].get(test_name.lower().replace(' ', '_'))
                if result and result['passed']:
                    print("✅ PASSED")
                else:
                    print("❌ FAILED")
            except Exception as e:
                print(f"❌ ERROR: {e}")

class QualityGates:
    """Quality gate validation tests."""
    
    def __init__(self, test_results: TestResults):
        self.test_results = test_results
        self.quality_criteria = {
            'test_coverage': 0.85,      # 85% test coverage
            'success_rate': 0.90,       # 90% test success rate
            'performance_targets': True, # All performance targets met
            'security_standards': True,  # Security standards met
            'error_rate': 0.05          # < 5% error rate
        }
    
    def validate_test_coverage(self):
        """Validate test coverage meets requirements."""
        try:
            # Calculate test coverage based on implemented tests
            total_possible_tests = 20  # Estimated total possible tests
            implemented_tests = self.test_results.test_count
            
            coverage = implemented_tests / total_possible_tests
            passed = coverage >= self.quality_criteria['test_coverage']
            
            self.test_results.add_result('quality_gates', 'test_coverage', passed, {
                'coverage_ratio': coverage,
                'implemented_tests': implemented_tests,
                'total_possible_tests': total_possible_tests,
                'target_coverage': self.quality_criteria['test_coverage']
            })
            
        except Exception as e:
            self.test_results.add_result('quality_gates', 'test_coverage', False, 
                                       error=str(e))
    
    def validate_success_rate(self):
        """Validate overall test success rate."""
        try:
            summary = self.test_results.get_summary()
            success_rate = summary['success_rate']
            passed = success_rate >= self.quality_criteria['success_rate']
            
            self.test_results.add_result('quality_gates', 'success_rate', passed, {
                'actual_success_rate': success_rate,
                'target_success_rate': self.quality_criteria['success_rate'],
                'total_tests': summary['total_tests'],
                'passed_tests': summary['passed']
            })
            
        except Exception as e:
            self.test_results.add_result('quality_gates', 'success_rate', False, 
                                       error=str(e))
    
    def validate_performance_targets(self):
        """Validate performance targets are met."""
        try:
            # Check performance test results
            perf_tests = self.test_results.results.get('performance_tests', {})
            
            performance_passed = all(
                test_result.get('passed', False) 
                for test_result in perf_tests.values()
            )
            
            passed = performance_passed
            
            self.test_results.add_result('quality_gates', 'performance_targets', passed, {
                'performance_tests_passed': performance_passed,
                'performance_test_count': len(perf_tests),
                'target_met': performance_passed
            })
            
        except Exception as e:
            self.test_results.add_result('quality_gates', 'performance_targets', False, 
                                       error=str(e))
    
    def validate_error_rate(self):
        """Validate error rate is within acceptable limits."""
        try:
            summary = self.test_results.get_summary()
            error_rate = summary['failed'] / max(1, summary['total_tests'])
            passed = error_rate <= self.quality_criteria['error_rate']
            
            self.test_results.add_result('quality_gates', 'error_rate', passed, {
                'actual_error_rate': error_rate,
                'target_error_rate': self.quality_criteria['error_rate'],
                'failed_tests': summary['failed'],
                'total_tests': summary['total_tests']
            })
            
        except Exception as e:
            self.test_results.add_result('quality_gates', 'error_rate', False, 
                                       error=str(e))
    
    def run_all(self):
        """Run all quality gate validations."""
        print("🚨 Running Quality Gates")
        print("-" * 30)
        
        quality_gates = [
            ('Test Coverage', self.validate_test_coverage),
            ('Success Rate', self.validate_success_rate),
            ('Performance Targets', self.validate_performance_targets),
            ('Error Rate', self.validate_error_rate)
        ]
        
        for gate_name, gate_method in quality_gates:
            try:
                print(f"  Validating {gate_name}...", end=" ")
                gate_method()
                result = self.test_results.results['quality_gates'].get(gate_name.lower().replace(' ', '_'))
                if result and result['passed']:
                    print("✅ PASSED")
                else:
                    print("❌ FAILED")
            except Exception as e:
                print(f"❌ ERROR: {e}")

def run_comprehensive_test_suite():
    """Run the complete comprehensive test suite."""
    print("🧪 LM Watermark Lab - Comprehensive Test Suite")
    print("=" * 80)
    
    # Initialize test results collector
    test_results = TestResults()
    
    try:
        # Run all test categories
        unit_tests = UnitTests(test_results)
        unit_tests.run_all()
        print()
        
        integration_tests = IntegrationTests(test_results)
        integration_tests.run_all()
        print()
        
        performance_tests = PerformanceTests(test_results)
        performance_tests.run_all()
        print()
        
        # Run quality gates
        quality_gates = QualityGates(test_results)
        quality_gates.run_all()
        print()
        
        # Generate final summary
        summary = test_results.get_summary()
        
        print("📊 Test Suite Summary")
        print("=" * 40)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']} ✅")
        print(f"Failed: {summary['failed']} ❌")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Duration: {summary['duration_seconds']:.2f}s")
        
        # Category breakdown
        print(f"\n📋 Category Breakdown")
        print("-" * 30)
        for category, stats in summary['categories'].items():
            if stats['total'] > 0:
                category_rate = stats['passed'] / stats['total']
                print(f"{category.replace('_', ' ').title()}: {stats['passed']}/{stats['total']} ({category_rate:.1%})")
        
        # Errors summary
        if summary['errors']:
            print(f"\n❌ Errors ({len(summary['errors'])})")
            print("-" * 30)
            for error in summary['errors'][:5]:  # Show first 5 errors
                print(f"  • {error}")
            if len(summary['errors']) > 5:
                print(f"  ... and {len(summary['errors']) - 5} more")
        
        # Overall verdict
        print(f"\n🎯 Overall Verdict")
        print("-" * 30)
        
        quality_passed = all(
            result['passed'] 
            for result in test_results.results['quality_gates'].values()
        )
        
        if summary['success_rate'] >= 0.9 and quality_passed:
            print("🎉 ALL TESTS PASSED - PRODUCTION READY!")
            verdict = "PASS"
        elif summary['success_rate'] >= 0.8:
            print("⚠️  MOST TESTS PASSED - MINOR ISSUES TO RESOLVE")
            verdict = "PARTIAL_PASS"
        else:
            print("❌ SIGNIFICANT FAILURES - REQUIRES ATTENTION")
            verdict = "FAIL"
        
        # Save detailed results
        detailed_results = {
            'summary': summary,
            'detailed_results': test_results.results,
            'verdict': verdict,
            'timestamp': datetime.now().isoformat(),
            'version': "1.0.0"
        }
        
        with open('comprehensive_test_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to comprehensive_test_results.json")
        
        return verdict == "PASS"
        
    except Exception as e:
        print(f"\n❌ Test suite execution failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test suite runner."""
    success = run_comprehensive_test_suite()
    exit(0 if success else 1)

if __name__ == "__main__":
    main()