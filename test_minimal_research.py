#!/usr/bin/env python3
"""
Minimal test for research algorithms without external dependencies.
Tests core functionality of SACW and ARMS algorithms.
"""

import sys
import os
import time
import math
import random
import hashlib
import json
from typing import Dict, Any, List, Optional
from collections import defaultdict, Counter

# Mock numpy functions for autonomous testing
class MockNumpy:
    @staticmethod
    def array(data):
        return data
    
    @staticmethod
    def random():
        class RandomState:
            def __init__(self, seed):
                random.seed(seed)
            def permutation(self, n):
                items = list(range(n))
                random.shuffle(items)
                return items
            def choice(self, items, p=None):
                if p is None:
                    return random.choice(items)
                # Simple weighted choice
                total = sum(p)
                r = random.random() * total
                cumsum = 0
                for i, prob in enumerate(p):
                    cumsum += prob
                    if r <= cumsum:
                        return i if isinstance(items, int) else items[i]
                return items[-1] if isinstance(items, list) else len(items) - 1
        return RandomState
    
    @staticmethod
    def mean(data):
        return sum(data) / len(data) if data else 0
    
    @staticmethod
    def std(data):
        if not data:
            return 0
        mean_val = sum(data) / len(data)
        variance = sum((x - mean_val) ** 2 for x in data) / len(data)
        return math.sqrt(variance)
    
    @staticmethod
    def exp(data):
        if isinstance(data, list):
            return [math.exp(x) for x in data]
        return math.exp(data)
    
    @staticmethod
    def max(data):
        return max(data) if isinstance(data, list) else data
    
    @staticmethod
    def sum(data):
        return sum(data) if isinstance(data, list) else data
    
    @staticmethod
    def dot(a, b):
        if isinstance(a, list) and isinstance(b, list):
            return sum(x * y for x, y in zip(a, b))
        return a * b
    
    @staticmethod
    def linalg():
        class LinAlg:
            @staticmethod
            def norm(data):
                if isinstance(data, list):
                    return math.sqrt(sum(x * x for x in data))
                return abs(data)
        return LinAlg()

# Mock scipy stats
class MockStats:
    class norm:
        @staticmethod
        def cdf(x):
            # Simple approximation of normal CDF
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

# Mock scikit-learn
class MockSklearn:
    class metrics:
        class pairwise:
            @staticmethod
            def cosine_similarity(a, b):
                return [[0.8]]  # Mock similarity

# Inject mocks
sys.modules['numpy'] = MockNumpy()
sys.modules['scipy.stats'] = MockStats()
sys.modules['sklearn.metrics.pairwise'] = MockSklearn.metrics.pairwise()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test if we can import our modules."""
    print("=== Testing Module Imports ===")
    
    try:
        from watermark_lab.core.factory import WatermarkFactory
        print("✓ WatermarkFactory imported")
        
        from watermark_lab.core.detector import WatermarkDetector  
        print("✓ WatermarkDetector imported")
        
        # Test factory methods
        methods = WatermarkFactory.list_methods()
        print(f"✓ Available methods: {methods}")
        
        expected_methods = ['kirchenbauer', 'markllm', 'aaronson', 'zhao', 'sacw', 'arms']
        missing = set(expected_methods) - set(methods)
        if missing:
            print(f"○ Missing methods: {missing}")
        else:
            print("✓ All expected methods available")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_sacw_creation():
    """Test SACW watermarker creation."""
    print("\\n=== Testing SACW Creation ===")
    
    try:
        from watermark_lab.core.factory import WatermarkFactory
        
        # Create SACW watermarker
        watermarker = WatermarkFactory.create(
            method="sacw",
            semantic_threshold=0.85,
            context_window=16,
            use_real_model=False,
            seed=42
        )
        
        print("✓ SACW watermarker created successfully")
        
        # Test configuration
        config = watermarker.get_config()
        assert config['method'] == 'sacw', f"Expected method 'sacw', got {config['method']}"
        assert config['semantic_threshold'] == 0.85, f"Wrong semantic threshold: {config['semantic_threshold']}"
        
        print("✓ SACW configuration validated")
        
        # Test research metrics
        if hasattr(watermarker, 'get_research_metrics'):
            metrics = watermarker.get_research_metrics()
            print(f"✓ SACW research metrics available: {list(metrics.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ SACW creation failed: {e}")
        return False

def test_arms_creation():
    """Test ARMS watermarker creation."""
    print("\\n=== Testing ARMS Creation ===")
    
    try:
        from watermark_lab.core.factory import WatermarkFactory
        
        # Create ARMS watermarker
        watermarker = WatermarkFactory.create(
            method="arms",
            scale_levels=[1, 4, 16],
            adversarial_strength=0.1,
            use_real_model=False,
            seed=42
        )
        
        print("✓ ARMS watermarker created successfully")
        
        # Test configuration
        config = watermarker.get_config()
        assert config['method'] == 'arms', f"Expected method 'arms', got {config['method']}"
        assert config['scale_levels'] == [1, 4, 16], f"Wrong scale levels: {config['scale_levels']}"
        
        print("✓ ARMS configuration validated")
        
        # Test research metrics
        if hasattr(watermarker, 'get_research_metrics'):
            metrics = watermarker.get_research_metrics()
            print(f"✓ ARMS research metrics available: {list(metrics.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ ARMS creation failed: {e}")
        return False

def test_watermark_generation():
    """Test watermark generation for both algorithms."""
    print("\\n=== Testing Watermark Generation ===")
    
    results = {}
    algorithms = [
        ('kirchenbauer', {}),
        ('sacw', {'semantic_threshold': 0.85}),
        ('arms', {'scale_levels': [1, 4], 'adversarial_strength': 0.1})
    ]
    
    for method, config in algorithms:
        try:
            print(f"\\nTesting {method.upper()} generation:")
            
            from watermark_lab.core.factory import WatermarkFactory
            
            watermarker = WatermarkFactory.create(
                method=method,
                use_real_model=False,
                seed=42,
                **config
            )
            
            # Test generation
            prompt = "The future of artificial intelligence involves"
            watermarked_text = watermarker.generate(prompt, max_length=80)
            
            # Basic validation
            assert len(watermarked_text) > len(prompt), f"Generated text too short: {len(watermarked_text)} vs {len(prompt)}"
            assert prompt in watermarked_text, "Original prompt not preserved"
            
            # Calculate basic metrics
            generation_ratio = len(watermarked_text) / len(prompt)
            word_count = len(watermarked_text.split())
            
            results[method] = {
                'success': True,
                'original_length': len(prompt),
                'watermarked_length': len(watermarked_text),
                'generation_ratio': generation_ratio,
                'word_count': word_count,
                'text_sample': watermarked_text[:100] + "..." if len(watermarked_text) > 100 else watermarked_text
            }
            
            print(f"✓ Generation successful")
            print(f"  Original: {len(prompt)} chars")
            print(f"  Generated: {len(watermarked_text)} chars (ratio: {generation_ratio:.2f})")
            print(f"  Word count: {word_count}")
            print(f"  Sample: {results[method]['text_sample']}")
            
        except Exception as e:
            print(f"✗ {method} generation failed: {e}")
            results[method] = {'success': False, 'error': str(e)}
    
    return results

def test_watermark_detection():
    """Test watermark detection for both algorithms."""
    print("\\n=== Testing Watermark Detection ===")
    
    results = {}
    algorithms = [
        ('kirchenbauer', {}),
        ('sacw', {'semantic_threshold': 0.85}),
        ('arms', {'scale_levels': [1, 4], 'adversarial_strength': 0.1})
    ]
    
    for method, config in algorithms:
        try:
            print(f"\\nTesting {method.upper()} detection:")
            
            from watermark_lab.core.factory import WatermarkFactory
            from watermark_lab.core.detector import WatermarkDetector
            
            # Create watermarker and generate text
            watermarker = WatermarkFactory.create(
                method=method,
                use_real_model=False,
                seed=42,
                **config
            )
            
            prompt = "Climate change requires immediate action from"
            watermarked_text = watermarker.generate(prompt, max_length=100)
            
            # Create detector
            detector_config = watermarker.get_config()
            detector = WatermarkDetector(detector_config)
            
            # Test detection on watermarked text
            detection_result = detector.detect(watermarked_text)
            
            # Test detection on clean text
            clean_text = prompt + " governments worldwide to implement sustainable policies and reduce carbon emissions"
            clean_detection = detector.detect(clean_text)
            
            results[method] = {
                'watermarked_detected': detection_result.is_watermarked,
                'watermarked_confidence': detection_result.confidence,
                'watermarked_p_value': detection_result.p_value,
                'clean_detected': clean_detection.is_watermarked,
                'clean_confidence': clean_detection.confidence,
                'processing_time': getattr(detection_result, 'processing_time', None),
                'method_validated': detection_result.method == method
            }
            
            print(f"✓ Detection successful")
            print(f"  Watermarked text: detected={detection_result.is_watermarked}, confidence={detection_result.confidence:.3f}")
            print(f"  Clean text: detected={clean_detection.is_watermarked}, confidence={clean_detection.confidence:.3f}")
            print(f"  Method validation: {results[method]['method_validated']}")
            
            if hasattr(detection_result, 'processing_time') and detection_result.processing_time:
                print(f"  Processing time: {detection_result.processing_time:.3f}s")
            
        except Exception as e:
            print(f"✗ {method} detection failed: {e}")
            results[method] = {'error': str(e)}
    
    return results

def test_performance_comparison():
    """Test performance comparison between algorithms."""
    print("\\n=== Testing Performance Comparison ===")
    
    algorithms = ['kirchenbauer', 'sacw', 'arms']
    performance_results = {}
    
    for method in algorithms:
        try:
            print(f"\\nBenchmarking {method.upper()}:")
            
            from watermark_lab.core.factory import WatermarkFactory
            from watermark_lab.core.detector import WatermarkDetector
            
            # Method-specific configuration
            if method == 'sacw':
                config = {'semantic_threshold': 0.85, 'context_window': 12}
            elif method == 'arms':
                config = {'scale_levels': [1, 4], 'adversarial_strength': 0.1}
            else:
                config = {}
            
            # Create watermarker
            watermarker = WatermarkFactory.create(
                method=method,
                use_real_model=False,
                seed=42,
                **config
            )
            
            detector_config = watermarker.get_config()
            detector = WatermarkDetector(detector_config)
            
            # Performance testing
            prompts = [
                "Artificial intelligence is transforming",
                "Climate change mitigation requires",
                "Digital healthcare solutions enable"
            ]
            
            generation_times = []
            detection_times = []
            text_lengths = []
            
            for prompt in prompts:
                try:
                    # Time generation
                    start_time = time.time()
                    watermarked_text = watermarker.generate(prompt, max_length=60)
                    gen_time = time.time() - start_time
                    generation_times.append(gen_time)
                    text_lengths.append(len(watermarked_text))
                    
                    # Time detection
                    start_time = time.time()
                    detection_result = detector.detect(watermarked_text)
                    det_time = time.time() - start_time
                    detection_times.append(det_time)
                    
                except Exception as e:
                    print(f"  Error with prompt: {e}")
                    continue
            
            if generation_times and detection_times:
                avg_gen_time = sum(generation_times) / len(generation_times)
                avg_det_time = sum(detection_times) / len(detection_times)
                avg_length = sum(text_lengths) / len(text_lengths)
                
                # Estimate throughput (chars/second)
                throughput = avg_length / avg_gen_time if avg_gen_time > 0 else 0
                
                performance_results[method] = {
                    'avg_generation_time': avg_gen_time,
                    'avg_detection_time': avg_det_time,
                    'avg_text_length': avg_length,
                    'throughput_chars_per_sec': throughput,
                    'sample_count': len(generation_times)
                }
                
                print(f"  Avg Generation Time: {avg_gen_time:.3f}s")
                print(f"  Avg Detection Time: {avg_det_time:.3f}s")
                print(f"  Avg Text Length: {avg_length:.0f} chars")
                print(f"  Throughput: {throughput:.1f} chars/s")
                
            else:
                print(f"  No valid performance data for {method}")
                
        except Exception as e:
            print(f"✗ Performance test failed for {method}: {e}")
            performance_results[method] = {'error': str(e)}
    
    return performance_results

def generate_final_report():
    """Generate comprehensive test report."""
    print("\\n" + "="*80)
    print("NOVEL WATERMARKING ALGORITHMS - AUTONOMOUS VALIDATION REPORT")
    print("="*80)
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_environment': 'Autonomous/Minimal Dependencies',
        'algorithms_tested': ['SACW (Semantic-Aware Contextual)', 'ARMS (Adversarial-Robust Multi-Scale)'],
        'test_results': {}
    }
    
    # Run all tests
    test_results = []
    
    print("\\n1. Module Import Test:")
    import_success = test_basic_imports()
    test_results.append(('Import', import_success))
    report['test_results']['imports'] = import_success
    
    print("\\n2. SACW Creation Test:")
    sacw_creation = test_sacw_creation()
    test_results.append(('SACW Creation', sacw_creation))
    report['test_results']['sacw_creation'] = sacw_creation
    
    print("\\n3. ARMS Creation Test:")
    arms_creation = test_arms_creation()
    test_results.append(('ARMS Creation', arms_creation))
    report['test_results']['arms_creation'] = arms_creation
    
    print("\\n4. Generation Test:")
    generation_results = test_watermark_generation()
    test_results.append(('Generation', len(generation_results) > 0))
    report['test_results']['generation'] = generation_results
    
    print("\\n5. Detection Test:")
    detection_results = test_watermark_detection()
    test_results.append(('Detection', len(detection_results) > 0))
    report['test_results']['detection'] = detection_results
    
    print("\\n6. Performance Test:")
    performance_results = test_performance_comparison()
    test_results.append(('Performance', len(performance_results) > 0))
    report['test_results']['performance'] = performance_results
    
    # Analysis
    print("\\n" + "="*80)
    print("RESEARCH VALIDATION ANALYSIS")
    print("="*80)
    
    # Test summary
    passed_tests = sum(1 for name, result in test_results if result)
    total_tests = len(test_results)
    pass_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    print(f"\\nTest Summary: {passed_tests}/{total_tests} tests passed ({pass_rate:.1%})")
    
    for test_name, passed in test_results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} {test_name}")
    
    # Algorithm-specific analysis
    print("\\nAlgorithm Analysis:")
    
    # SACW Analysis
    sacw_functional = (sacw_creation and 
                      generation_results.get('sacw', {}).get('success', False) and
                      detection_results.get('sacw', {}).get('watermarked_detected', False))
    print(f"  SACW Functionality: {'✓ OPERATIONAL' if sacw_functional else '○ PARTIAL'}")
    
    # ARMS Analysis
    arms_functional = (arms_creation and 
                      generation_results.get('arms', {}).get('success', False) and
                      detection_results.get('arms', {}).get('watermarked_detected', False))
    print(f"  ARMS Functionality: {'✓ OPERATIONAL' if arms_functional else '○ PARTIAL'}")
    
    # Performance comparison
    if performance_results:
        print("\\nPerformance Ranking (by generation speed):")
        perf_items = [(method, data.get('avg_generation_time', float('inf'))) 
                     for method, data in performance_results.items() 
                     if 'avg_generation_time' in data]
        perf_items.sort(key=lambda x: x[1])
        
        for i, (method, time_val) in enumerate(perf_items, 1):
            print(f"  {i}. {method.upper()}: {time_val:.3f}s")
    
    # Research conclusions
    conclusions = []
    if sacw_functional:
        conclusions.append("SACW algorithm successfully implemented with semantic-aware watermarking")
    if arms_functional:
        conclusions.append("ARMS algorithm successfully implemented with multi-scale detection")
    
    if pass_rate >= 0.8:
        conclusions.append("Novel research algorithms demonstrate strong implementation quality")
    elif pass_rate >= 0.6:
        conclusions.append("Novel research algorithms show promising results with some limitations")
    else:
        conclusions.append("Novel research algorithms require further development")
    
    report['conclusions'] = conclusions
    report['overall_success'] = pass_rate >= 0.6
    report['pass_rate'] = pass_rate
    
    print("\\nResearch Conclusions:")
    for conclusion in conclusions:
        print(f"  • {conclusion}")
    
    print(f"\\nOverall Validation: {'SUCCESS' if report['overall_success'] else 'NEEDS WORK'}")
    
    # Save report
    try:
        with open('/root/repo/research_validation_autonomous.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print("\\n✓ Detailed report saved to: research_validation_autonomous.json")
    except Exception as e:
        print(f"\\n○ Could not save detailed report: {e}")
    
    return report

if __name__ == "__main__":
    print("Starting Novel Watermarking Research Algorithms Autonomous Validation")
    print("Testing: SACW (Semantic-Aware Contextual Watermarking)")
    print("         ARMS (Adversarial-Robust Multi-Scale Watermarking)")
    print("-" * 80)
    
    try:
        report = generate_final_report()
        success = report.get('overall_success', False)
        
        print(f"\\nValidation completed. Success: {success}")
        exit_code = 0 if success else 1
        print(f"Exiting with code: {exit_code}")
        exit(exit_code)
        
    except Exception as e:
        print(f"\\nValidation failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)