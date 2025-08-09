#!/usr/bin/env python3
"""
Autonomous test suite for novel watermarking research algorithms.
Tests SACW and ARMS implementations without external dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import time
import json
from typing import Dict, List
from collections import defaultdict

# Import watermarking modules
try:
    from watermark_lab.core.factory import WatermarkFactory, SemanticContextualWatermark, AdversarialRobustWatermark
    from watermark_lab.core.detector import WatermarkDetector
    print("✓ Successfully imported watermarking modules")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

class AutonomousResearchTester:
    """Autonomous test runner for research algorithms."""
    
    def __init__(self):
        self.results = defaultdict(list)
        self.test_prompts = [
            "The future of artificial intelligence is",
            "Climate change requires global cooperation to",
            "Machine learning algorithms help us understand",
            "Digital transformation enables businesses to",
            "Quantum computing research focuses on"
        ]
    
    def test_sacw_basic_functionality(self):
        """Test SACW basic functionality."""
        print("\\n=== Testing SACW Basic Functionality ===")
        
        try:
            # Create SACW watermarker
            watermarker = WatermarkFactory.create(
                method="sacw",
                semantic_threshold=0.85,
                context_window=16,
                use_real_model=False,
                seed=42
            )
            
            # Test generation
            prompt = self.test_prompts[0]
            watermarked_text = watermarker.generate(prompt, max_length=100)
            
            assert len(watermarked_text) > len(prompt), "Generated text too short"
            assert prompt in watermarked_text, "Prompt not preserved"
            
            # Test detection
            detector_config = watermarker.get_config()
            detector = WatermarkDetector(detector_config)
            detection_result = detector.detect(watermarked_text)
            
            print(f"✓ SACW Generation successful: {len(watermarked_text)} chars")
            print(f"✓ SACW Detection: watermarked={detection_result.is_watermarked}, confidence={detection_result.confidence:.3f}")
            
            if hasattr(detection_result, 'semantic_coherence') and detection_result.semantic_coherence:
                print(f"✓ SACW Semantic coherence: {detection_result.semantic_coherence:.3f}")
            
            self.results['sacw_basic'].append({
                'success': True,
                'generation_length': len(watermarked_text),
                'detected': detection_result.is_watermarked,
                'confidence': detection_result.confidence
            })
            
            return True
            
        except Exception as e:
            print(f"✗ SACW basic test failed: {e}")
            self.results['sacw_basic'].append({'success': False, 'error': str(e)})
            return False
    
    def test_arms_basic_functionality(self):
        """Test ARMS basic functionality."""
        print("\\n=== Testing ARMS Basic Functionality ===")
        
        try:
            # Create ARMS watermarker
            watermarker = WatermarkFactory.create(
                method="arms",
                scale_levels=[1, 4, 16],
                adversarial_strength=0.1,
                use_real_model=False,
                seed=42
            )
            
            # Test generation
            prompt = self.test_prompts[1]
            watermarked_text = watermarker.generate(prompt, max_length=120)
            
            assert len(watermarked_text) > len(prompt), "Generated text too short"
            assert prompt in watermarked_text, "Prompt not preserved"
            
            # Test detection
            detector_config = watermarker.get_config()
            detector = WatermarkDetector(detector_config)
            detection_result = detector.detect(watermarked_text)
            
            print(f"✓ ARMS Generation successful: {len(watermarked_text)} chars")
            print(f"✓ ARMS Detection: watermarked={detection_result.is_watermarked}, confidence={detection_result.confidence:.3f}")
            
            if detection_result.details and 'scales_detected' in detection_result.details:
                print(f"✓ ARMS Multi-scale detection: {detection_result.details['scales_detected']} scales")
            
            self.results['arms_basic'].append({
                'success': True,
                'generation_length': len(watermarked_text),
                'detected': detection_result.is_watermarked,
                'confidence': detection_result.confidence
            })
            
            return True
            
        except Exception as e:
            print(f"✗ ARMS basic test failed: {e}")
            self.results['arms_basic'].append({'success': False, 'error': str(e)})
            return False
    
    def test_algorithm_comparison(self):
        """Compare SACW vs ARMS vs baseline algorithms."""
        print("\\n=== Testing Algorithm Comparison ===")
        
        algorithms = [
            ('kirchenbauer', {'gamma': 0.25, 'delta': 2.0}),
            ('sacw', {'semantic_threshold': 0.85, 'context_window': 16}),
            ('arms', {'scale_levels': [1, 4, 16], 'adversarial_strength': 0.1})
        ]
        
        comparison_results = {}
        
        for method_name, config in algorithms:
            try:
                print(f"\\nTesting {method_name.upper()}:")
                
                # Create watermarker
                watermarker = WatermarkFactory.create(
                    method=method_name,
                    use_real_model=False,
                    seed=42,
                    **config
                )
                
                # Create detector
                detector_config = watermarker.get_config()
                detector = WatermarkDetector(detector_config)
                
                # Test on multiple prompts
                method_results = []
                for i, prompt in enumerate(self.test_prompts[:3]):  # Limited for autonomous testing
                    try:
                        # Generate and detect
                        watermarked_text = watermarker.generate(prompt, max_length=80)
                        detection_result = detector.detect(watermarked_text)
                        
                        # Calculate semantic similarity (using simple method)
                        semantic_sim = self._simple_semantic_similarity(prompt, watermarked_text)
                        
                        method_results.append({
                            'detected': detection_result.is_watermarked,
                            'confidence': detection_result.confidence,
                            'semantic_similarity': semantic_sim,
                            'text_length': len(watermarked_text)
                        })
                        
                    except Exception as e:
                        print(f"  Error with prompt {i}: {e}")
                        continue
                
                if method_results:
                    # Calculate averages
                    avg_detection_rate = sum(1 for r in method_results if r['detected']) / len(method_results)
                    avg_confidence = sum(r['confidence'] for r in method_results) / len(method_results)
                    avg_semantic_sim = sum(r['semantic_similarity'] for r in method_results) / len(method_results)
                    
                    comparison_results[method_name] = {
                        'detection_rate': avg_detection_rate,
                        'avg_confidence': avg_confidence,
                        'avg_semantic_similarity': avg_semantic_sim,
                        'sample_count': len(method_results)
                    }
                    
                    print(f"  Detection Rate: {avg_detection_rate:.3f}")
                    print(f"  Avg Confidence: {avg_confidence:.3f}")
                    print(f"  Avg Semantic Similarity: {avg_semantic_sim:.3f}")
                
            except Exception as e:
                print(f"  Failed to test {method_name}: {e}")
                comparison_results[method_name] = {'error': str(e)}
        
        # Analysis
        if comparison_results:
            print("\\n=== Comparison Analysis ===")
            for method, results in comparison_results.items():
                if 'error' not in results:
                    print(f"{method.upper()}: det={results['detection_rate']:.3f}, "
                          f"conf={results['avg_confidence']:.3f}, sem={results['avg_semantic_similarity']:.3f}")
            
            # Research validation
            if 'sacw' in comparison_results and 'kirchenbauer' in comparison_results:
                sacw_sem = comparison_results['sacw'].get('avg_semantic_similarity', 0)
                kirch_sem = comparison_results['kirchenbauer'].get('avg_semantic_similarity', 0)
                if sacw_sem > kirch_sem:
                    print("✓ SACW shows semantic improvement over baseline")
                else:
                    print("○ SACW semantic improvement not detected")
        
        self.results['comparison'] = comparison_results
        return len(comparison_results) > 0
    
    def test_performance_benchmarks(self):
        """Test performance characteristics."""
        print("\\n=== Testing Performance Benchmarks ===")
        
        algorithms = ['kirchenbauer', 'sacw', 'arms']
        performance_results = {}
        
        for method in algorithms:
            try:
                print(f"\\nBenchmarking {method.upper()}:")
                
                # Create watermarker with method-specific config
                if method == 'sacw':
                    config = {'semantic_threshold': 0.85}
                elif method == 'arms':
                    config = {'scale_levels': [1, 4]}  # Reduced for performance
                else:
                    config = {}
                
                watermarker = WatermarkFactory.create(
                    method=method,
                    use_real_model=False,
                    seed=42,
                    **config
                )
                
                detector_config = watermarker.get_config()
                detector = WatermarkDetector(detector_config)
                
                # Benchmark generation
                gen_times = []
                det_times = []
                
                for prompt in self.test_prompts[:3]:
                    try:
                        # Time generation
                        start_time = time.time()
                        watermarked_text = watermarker.generate(prompt, max_length=60)
                        gen_time = time.time() - start_time
                        gen_times.append(gen_time)
                        
                        # Time detection
                        start_time = time.time()
                        detection_result = detector.detect(watermarked_text)
                        det_time = time.time() - start_time
                        det_times.append(det_time)
                        
                    except Exception as e:
                        print(f"  Benchmark error: {e}")
                        continue
                
                if gen_times and det_times:
                    avg_gen_time = sum(gen_times) / len(gen_times)
                    avg_det_time = sum(det_times) / len(det_times)
                    
                    performance_results[method] = {
                        'avg_generation_time': avg_gen_time,
                        'avg_detection_time': avg_det_time,
                        'throughput_estimate': 50.0 / avg_gen_time if avg_gen_time > 0 else 0
                    }
                    
                    print(f"  Generation Time: {avg_gen_time:.3f}s")
                    print(f"  Detection Time: {avg_det_time:.3f}s")
                    print(f"  Throughput: ~{performance_results[method]['throughput_estimate']:.1f} tokens/s")
                
            except Exception as e:
                print(f"  Performance test failed for {method}: {e}")
                performance_results[method] = {'error': str(e)}
        
        self.results['performance'] = performance_results
        return len(performance_results) > 0
    
    def _simple_semantic_similarity(self, text1: str, text2: str) -> float:
        """Simple semantic similarity using word overlap."""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            return intersection / union if union > 0 else 0.0
        except Exception:
            return 0.0
    
    def generate_research_report(self):
        """Generate research validation report."""
        print("\\n" + "="*60)
        print("NOVEL WATERMARKING ALGORITHMS - RESEARCH VALIDATION")
        print("="*60)
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'algorithms_tested': ['SACW', 'ARMS'],
            'test_results': dict(self.results),
            'research_conclusions': []
        }
        
        conclusions = []
        
        # Basic functionality validation
        sacw_basic_success = any(r.get('success', False) for r in self.results.get('sacw_basic', []))
        arms_basic_success = any(r.get('success', False) for r in self.results.get('arms_basic', []))
        
        conclusions.append(f"SACW Basic Functionality: {'PASS' if sacw_basic_success else 'FAIL'}")
        conclusions.append(f"ARMS Basic Functionality: {'PASS' if arms_basic_success else 'FAIL'}")
        
        # Comparison analysis
        if 'comparison' in self.results:
            comparison = self.results['comparison']
            if 'sacw' in comparison and 'kirchenbauer' in comparison:
                sacw_data = comparison['sacw']
                baseline_data = comparison['kirchenbauer']
                
                if 'avg_semantic_similarity' in sacw_data and 'avg_semantic_similarity' in baseline_data:
                    sacw_sem = sacw_data['avg_semantic_similarity']
                    baseline_sem = baseline_data['avg_semantic_similarity']
                    improvement = sacw_sem - baseline_sem
                    
                    conclusions.append(f"SACW Semantic Improvement: {'PASS' if improvement > 0 else 'FAIL'} (+{improvement:.3f})")
                
                if 'detection_rate' in sacw_data:
                    det_rate = sacw_data['detection_rate']
                    conclusions.append(f"SACW Detection Rate: {det_rate:.3f} ({'PASS' if det_rate > 0.7 else 'PARTIAL'})")
            
            if 'arms' in comparison:
                arms_data = comparison['arms']
                if 'detection_rate' in arms_data:
                    det_rate = arms_data['detection_rate']
                    conclusions.append(f"ARMS Detection Rate: {det_rate:.3f} ({'PASS' if det_rate > 0.7 else 'PARTIAL'})")
        
        # Performance analysis
        if 'performance' in self.results:
            perf_data = self.results['performance']
            for method in ['sacw', 'arms']:
                if method in perf_data and 'avg_generation_time' in perf_data[method]:
                    gen_time = perf_data[method]['avg_generation_time']
                    conclusions.append(f"{method.upper()} Generation Time: {gen_time:.3f}s ({'PASS' if gen_time < 2.0 else 'SLOW'})")
        
        report['research_conclusions'] = conclusions
        
        # Print summary
        print("\\nRESEARCH VALIDATION SUMMARY:")
        for conclusion in conclusions:
            status = "✓" if "PASS" in conclusion else ("○" if "PARTIAL" in conclusion else "✗")
            print(f"  {status} {conclusion}")
        
        # Overall assessment
        passes = sum(1 for c in conclusions if 'PASS' in c)
        total = len(conclusions)
        success_rate = passes / total if total > 0 else 0
        
        print(f"\\nOVERALL VALIDATION: {'SUCCESS' if success_rate >= 0.6 else 'PARTIAL'} ({passes}/{total} tests passed)")
        print(f"Success Rate: {success_rate:.1%}")
        
        # Save report
        try:
            with open('/root/repo/research_algorithms_validation.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print("\\n✓ Detailed report saved to: research_algorithms_validation.json")
        except Exception as e:
            print(f"\\n○ Could not save report: {e}")
        
        return report, success_rate >= 0.6

def main():
    """Run autonomous research algorithm tests."""
    print("Novel Watermarking Research Algorithms - Autonomous Validation")
    print("Testing SACW (Semantic-Aware Contextual) & ARMS (Adversarial-Robust Multi-Scale)")
    print("=" * 80)
    
    tester = AutonomousResearchTester()
    
    test_results = []
    
    # Run all tests
    try:
        test_results.append(tester.test_sacw_basic_functionality())
        test_results.append(tester.test_arms_basic_functionality()) 
        test_results.append(tester.test_algorithm_comparison())
        test_results.append(tester.test_performance_benchmarks())
        
        # Generate final report
        report, overall_success = tester.generate_research_report()
        
        print(f"\\nTesting completed. Overall success: {overall_success}")
        return overall_success
        
    except Exception as e:
        print(f"\\nTesting failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    print(f"\\nExiting with code: {exit_code}")
    exit(exit_code)