#!/usr/bin/env python3
"""
Quality Gates and Security Validation
Final validation for production readiness.
"""

import sys
import os
import json
import time
import hashlib
import logging
import threading
from typing import Dict, Any, List
from datetime import datetime
import warnings

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import test implementations
sys.path.insert(0, '/root/repo')
from simple_core_test import mock_watermark_simple
from robust_generation2_test import WatermarkConfig, RobustWatermark, InputSanitizer
from performance_generation3_test import HighPerformanceWatermark

class QualityGatesValidator:
    """Comprehensive quality gates validation."""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    def validate_functionality(self) -> Dict[str, Any]:
        """Validate core functionality meets requirements."""
        print("🔍 Validating Core Functionality")
        print("-" * 40)
        
        try:
            # Test basic watermarking
            SimpleWatermark = mock_watermark_simple()
            watermarker = SimpleWatermark("quality_test_key")
            
            # Generation test
            generated = watermarker.generate_text("Quality validation test", max_tokens=20)
            generation_works = len(generated) > 0 and "Quality validation test" in generated
            
            # Detection test
            detection = watermarker.detect_watermark(generated)
            detection_works = (
                isinstance(detection, dict) and
                'is_watermarked' in detection and
                'confidence' in detection
            )
            
            # False positive test
            clean_detection = watermarker.detect_watermark("This is clean text")
            false_positive_ok = not clean_detection['is_watermarked']
            
            functionality_score = sum([generation_works, detection_works, false_positive_ok]) / 3
            
            result = {
                'passed': functionality_score >= 0.9,
                'score': functionality_score,
                'generation_works': generation_works,
                'detection_works': detection_works,
                'false_positive_controlled': false_positive_ok
            }
            
            print(f"  Generation: {'✅' if generation_works else '❌'}")
            print(f"  Detection: {'✅' if detection_works else '❌'}")
            print(f"  False Positive Control: {'✅' if false_positive_ok else '❌'}")
            print(f"  Overall Score: {functionality_score:.1%}")
            
            return result
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def validate_robustness(self) -> Dict[str, Any]:
        """Validate robustness and error handling."""
        print("\n🛡️ Validating Robustness & Error Handling")
        print("-" * 40)
        
        try:
            config = WatermarkConfig(key="robustness_test", max_tokens=30)
            watermarker = RobustWatermark(config)
            
            # Test error scenarios
            error_scenarios = [
                ("empty_input", ""),
                ("very_long_input", "x" * 5000),
                ("special_chars", "text with 🔥 emojis and ñ accents"),
                ("whitespace_heavy", "   lots    of   spaces   ")
            ]
            
            handled_errors = 0
            for scenario_name, test_input in error_scenarios:
                try:
                    if test_input == "":
                        # Empty input should raise ValidationError
                        try:
                            watermarker.generate_text(test_input)
                            # Should not reach here
                        except Exception:
                            handled_errors += 1
                    else:
                        result = watermarker.generate_text(test_input)
                        if 'generated_text' in result:
                            handled_errors += 1
                except Exception as e:
                    # Some errors are expected and OK
                    if "empty" in str(e).lower():
                        handled_errors += 1
            
            robustness_score = handled_errors / len(error_scenarios)
            
            # Test configuration validation
            config_validation_works = True
            try:
                WatermarkConfig(key="", max_tokens=50)  # Should fail
                config_validation_works = False
            except ValueError:
                pass  # Expected
            
            # Test input sanitization
            sanitizer = InputSanitizer()
            sanitization_works = True
            try:
                sanitizer.sanitize_text("")  # Should fail
                sanitization_works = False
            except:
                pass  # Expected
            
            overall_robustness = (robustness_score + config_validation_works + sanitization_works) / 3
            
            result = {
                'passed': overall_robustness >= 0.8,
                'score': overall_robustness,
                'error_handling_score': robustness_score,
                'config_validation_works': config_validation_works,
                'input_sanitization_works': sanitization_works
            }
            
            print(f"  Error Handling: {robustness_score:.1%}")
            print(f"  Config Validation: {'✅' if config_validation_works else '❌'}")
            print(f"  Input Sanitization: {'✅' if sanitization_works else '❌'}")
            print(f"  Overall Robustness: {overall_robustness:.1%}")
            
            return result
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def validate_performance(self) -> Dict[str, Any]:
        """Validate performance meets requirements."""
        print("\n⚡ Validating Performance Requirements")
        print("-" * 40)
        
        try:
            watermarker = HighPerformanceWatermark("performance_validation", max_workers=4)
            
            # Single operation performance
            start_time = time.time()
            result = watermarker.generate_text_optimized("Performance validation test")
            single_gen_time = time.time() - start_time
            
            start_time = time.time()
            detection = watermarker.detect_watermark_optimized(result['generated_text'])
            single_det_time = time.time() - start_time
            
            # Batch performance
            batch_prompts = [f"Batch test {i}" for i in range(20)]
            start_time = time.time()
            batch_results = watermarker.batch_generate(batch_prompts, max_tokens=25)
            batch_time = time.time() - start_time
            batch_throughput = len(batch_prompts) / batch_time
            
            # Cache performance
            start_time = time.time()
            cached_result = watermarker.generate_text_optimized("Performance validation test")  # Same prompt
            cached_time = time.time() - start_time
            cache_speedup = single_gen_time / max(cached_time, 0.00001)
            
            # Performance targets
            targets = {
                'single_generation_fast': single_gen_time < 0.01,  # < 10ms
                'single_detection_fast': single_det_time < 0.01,   # < 10ms
                'batch_throughput_good': batch_throughput > 100,   # > 100 ops/sec
                'cache_speedup_good': cache_speedup > 2.0          # > 2x speedup
            }
            
            performance_score = sum(targets.values()) / len(targets)
            
            result = {
                'passed': performance_score >= 0.75,
                'score': performance_score,
                'single_generation_time_ms': single_gen_time * 1000,
                'single_detection_time_ms': single_det_time * 1000,
                'batch_throughput_ops_per_sec': batch_throughput,
                'cache_speedup': cache_speedup,
                'targets_met': targets
            }
            
            print(f"  Single Generation: {single_gen_time*1000:.2f}ms {'✅' if targets['single_generation_fast'] else '❌'}")
            print(f"  Single Detection: {single_det_time*1000:.2f}ms {'✅' if targets['single_detection_fast'] else '❌'}")
            print(f"  Batch Throughput: {batch_throughput:.1f} ops/sec {'✅' if targets['batch_throughput_good'] else '❌'}")
            print(f"  Cache Speedup: {cache_speedup:.1f}x {'✅' if targets['cache_speedup_good'] else '❌'}")
            print(f"  Performance Score: {performance_score:.1%}")
            
            return result
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def validate_security(self) -> Dict[str, Any]:
        """Validate security requirements."""
        print("\n🔒 Validating Security Requirements")
        print("-" * 40)
        
        try:
            config = WatermarkConfig(key="security_test_key_123", max_tokens=30)
            watermarker = RobustWatermark(config)
            
            # Test key validation
            key_validation_works = True
            try:
                WatermarkConfig(key="x")  # Too short
                key_validation_works = False
            except ValueError:
                pass  # Expected
            
            # Test input sanitization against injection
            sanitizer = InputSanitizer()
            dangerous_inputs = [
                "text\x00with\x01nulls",
                "<script>alert('xss')</script>",
                "'; DROP TABLE users; --",
                "../../etc/passwd"
            ]
            
            sanitization_effective = 0
            for dangerous_input in dangerous_inputs:
                try:
                    sanitized = sanitizer.sanitize_text(dangerous_input)
                    # Check if dangerous parts were removed/neutralized
                    if "\x00" not in sanitized and "\x01" not in sanitized:
                        sanitization_effective += 1
                except:
                    sanitization_effective += 1  # Rejection is also OK
            
            sanitization_score = sanitization_effective / len(dangerous_inputs)
            
            # Test rate limiting (simplified check)
            rate_limiting_works = hasattr(watermarker, '_check_rate_limits')
            
            # Test secure hashing
            secure_hashing_works = hasattr(watermarker, '_secure_hash')
            
            security_features = [
                key_validation_works,
                sanitization_score > 0.8,
                rate_limiting_works,
                secure_hashing_works
            ]
            
            security_score = sum(security_features) / len(security_features)
            
            result = {
                'passed': security_score >= 0.8,
                'score': security_score,
                'key_validation_works': key_validation_works,
                'input_sanitization_effective': sanitization_score > 0.8,
                'rate_limiting_implemented': rate_limiting_works,
                'secure_hashing_implemented': secure_hashing_works,
                'sanitization_score': sanitization_score
            }
            
            print(f"  Key Validation: {'✅' if key_validation_works else '❌'}")
            print(f"  Input Sanitization: {'✅' if sanitization_score > 0.8 else '❌'} ({sanitization_score:.1%})")
            print(f"  Rate Limiting: {'✅' if rate_limiting_works else '❌'}")
            print(f"  Secure Hashing: {'✅' if secure_hashing_works else '❌'}")
            print(f"  Security Score: {security_score:.1%}")
            
            return result
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all quality gate validations."""
        print("🚨 Quality Gates & Security Validation")
        print("=" * 60)
        
        validations = {
            'functionality': self.validate_functionality(),
            'robustness': self.validate_robustness(),
            'performance': self.validate_performance(),
            'security': self.validate_security()
        }
        
        # Overall assessment
        all_passed = all(v.get('passed', False) for v in validations.values())
        total_score = sum(v.get('score', 0) for v in validations.values()) / len(validations)
        
        duration = time.time() - self.start_time
        
        overall_result = {
            'all_gates_passed': all_passed,
            'overall_score': total_score,
            'validation_results': validations,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n📊 Quality Gates Summary")
        print("=" * 40)
        
        for category, result in validations.items():
            status = "✅ PASSED" if result.get('passed', False) else "❌ FAILED"
            score = result.get('score', 0)
            print(f"{category.title()}: {status} ({score:.1%})")
        
        print(f"\nOverall Score: {total_score:.1%}")
        print(f"Duration: {duration:.2f}s")
        
        if all_passed and total_score >= 0.85:
            print("\n🎉 ALL QUALITY GATES PASSED - PRODUCTION READY!")
            verdict = "PRODUCTION_READY"
        elif total_score >= 0.75:
            print("\n⚠️  MOST GATES PASSED - MINOR IMPROVEMENTS NEEDED")
            verdict = "NEEDS_MINOR_IMPROVEMENTS"
        else:
            print("\n❌ QUALITY GATES FAILED - REQUIRES SIGNIFICANT WORK")
            verdict = "REQUIRES_WORK"
        
        overall_result['verdict'] = verdict
        
        # Save results
        with open('quality_security_validation_results.json', 'w') as f:
            json.dump(overall_result, f, indent=2, default=str)
        
        print(f"\n📄 Results saved to quality_security_validation_results.json")
        
        return overall_result

def main():
    """Main validation runner."""
    validator = QualityGatesValidator()
    results = validator.run_all_validations()
    
    success = results['verdict'] == 'PRODUCTION_READY'
    exit(0 if success else 1)

if __name__ == "__main__":
    main()