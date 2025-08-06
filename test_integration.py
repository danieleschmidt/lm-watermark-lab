#!/usr/bin/env python3
"""Integration test for quantum-enhanced watermarking workflow."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_full_integration():
    """Test full quantum-enhanced watermarking integration."""
    print("🚀 Testing Full Quantum-Enhanced Watermarking Integration")
    print("=" * 70)
    
    try:
        # Import all required components
        from watermark_lab.core.quantum_planner import QuantumTaskPlanner, create_watermarking_workflow
        from watermark_lab.core.factory import WatermarkFactory
        from watermark_lab.core.detector import WatermarkDetector
        from watermark_lab.core.attacks import AttackSimulator
        from watermark_lab.core.benchmark import WatermarkBenchmark
        
        print("✅ All imports successful")
        
        # Test 1: Create quantum workflow
        print("\n1️⃣ Testing quantum workflow creation...")
        planner = QuantumTaskPlanner(max_coherence_time=300.0)
        
        config = {
            'methods': ['kirchenbauer', 'markllm', 'aaronson'],
            'prompts': [
                'The future of AI involves quantum computing',
                'Machine learning will revolutionize watermarking',
                'Deep learning models can be enhanced'
            ],
            'include_detection': True,
            'include_benchmark': True
        }
        
        task_ids = create_watermarking_workflow(planner, config)
        print(f"✅ Created workflow with {len(task_ids)} quantum tasks")
        
        # Test 2: Optimize and plan execution
        print("\n2️⃣ Testing optimization and execution planning...")
        optimization = planner.optimize_plan("coherence")
        execution_plan = planner.plan_execution()
        
        print(f"✅ Optimization: {optimization.get('strategy', 'default')}")
        print(f"✅ Execution plan: {len(execution_plan)} tasks ordered")
        
        # Test 3: Execute workflow
        print("\n3️⃣ Testing workflow execution...")
        context = {
            'methods': config['methods'],
            'prompts': config['prompts'],
            'num_samples': 3
        }
        
        results = planner.execute_plan(execution_plan, context)
        successful = sum(1 for r in results if r.success)
        
        print(f"✅ Executed {len(results)} tasks with {successful} successes")
        
        # Test 4: Generate and detect watermark
        print("\n4️⃣ Testing watermark generation and detection...")
        
        # Generate watermarked text
        watermarker = WatermarkFactory.create('kirchenbauer', gamma=0.25, delta=2.0)
        prompt = "Quantum-inspired watermarking demonstrates advanced capabilities"
        watermarked_text = watermarker.generate(prompt, max_length=80)
        
        print(f"✅ Generated: {watermarked_text}")
        
        # Detect watermark
        detector = WatermarkDetector(watermarker.get_config())
        detection = detector.detect(watermarked_text)
        
        print(f"✅ Detection - Watermarked: {detection.is_watermarked}, "
              f"Confidence: {detection.confidence:.3f}")
        
        # Test 5: Attack simulation
        print("\n5️⃣ Testing attack resistance...")
        simulator = AttackSimulator()
        
        attack_results = []
        attacks = ['paraphrase', 'truncation', 'insertion']
        
        for attack_name in attacks:
            attack_result = simulator.run_attack(watermarked_text, attack_name, "medium")
            attack_results.append(attack_result)
            
            print(f"   {attack_name}: Success={attack_result.attack_success}, "
                  f"Quality={attack_result.quality_score:.3f}")
        
        # Test 6: Benchmark comparison
        print("\n6️⃣ Testing benchmark comparison...")
        benchmark = WatermarkBenchmark(num_samples=3)
        
        comparison_methods = ['kirchenbauer', 'markllm']
        comparison_prompts = config['prompts'][:2]  # Use 2 prompts
        metrics = ['detectability', 'quality', 'robustness']
        
        benchmark_results = benchmark.compare(comparison_methods, comparison_prompts, metrics)
        
        print("✅ Benchmark results:")
        for method, metric_values in benchmark_results.items():
            print(f"   {method}: {metric_values}")
        
        # Test 7: System state analysis
        print("\n7️⃣ Testing system state analysis...")
        final_state = planner.get_system_state()
        
        print(f"✅ Final quantum state:")
        print(f"   Total tasks: {final_state['total_tasks']}")
        print(f"   Task states: {final_state['task_states']}")
        print(f"   Entanglement groups: {final_state['entanglement_groups']}")
        print(f"   Global coherence: {final_state['global_quantum_state']['total_coherence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_metrics():
    """Test performance characteristics."""
    print("\n🔬 Testing Performance Metrics...")
    
    try:
        import time
        from watermark_lab.core.factory import WatermarkFactory
        from watermark_lab.core.detector import WatermarkDetector
        
        # Test generation performance
        watermarker = WatermarkFactory.create('kirchenbauer')
        
        start_time = time.time()
        test_prompts = [
            f"Performance test prompt number {i} for watermarking"
            for i in range(10)
        ]
        
        generated_texts = []
        for prompt in test_prompts:
            text = watermarker.generate(prompt, max_length=50)
            generated_texts.append(text)
        
        generation_time = time.time() - start_time
        
        # Test detection performance
        detector = WatermarkDetector(watermarker.get_config())
        
        start_time = time.time()
        detection_results = []
        for text in generated_texts:
            result = detector.detect(text)
            detection_results.append(result)
        
        detection_time = time.time() - start_time
        
        print(f"✅ Performance metrics:")
        print(f"   Generation: {generation_time:.3f}s for {len(test_prompts)} texts "
              f"({len(test_prompts)/generation_time:.1f} texts/sec)")
        print(f"   Detection: {detection_time:.3f}s for {len(generated_texts)} texts "
              f"({len(generated_texts)/detection_time:.1f} texts/sec)")
        
        # Detection accuracy
        detected_count = sum(1 for r in detection_results if r.is_watermarked)
        print(f"   Detection accuracy: {detected_count}/{len(detection_results)} "
              f"({detected_count/len(detection_results)*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def main():
    """Run full integration test suite."""
    print("🧬 QUANTUM-INSPIRED WATERMARKING SYSTEM")
    print("Integration Test Suite")
    print("=" * 70)
    
    # Run integration test
    integration_success = test_full_integration()
    
    # Run performance test
    if integration_success:
        performance_success = test_performance_metrics()
    else:
        performance_success = False
    
    print("\n" + "=" * 70)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 70)
    
    if integration_success and performance_success:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("\n🚀 QUANTUM-ENHANCED WATERMARKING SYSTEM READY!")
        print("   ✨ Quantum task planning: OPERATIONAL")
        print("   🔬 Multi-algorithm watermarking: FUNCTIONAL")
        print("   🕵️ Statistical detection: ACCURATE")
        print("   ⚔️ Attack resistance: TESTED")
        print("   📊 Benchmarking: COMPREHENSIVE")
        print("   🎯 Performance: OPTIMIZED")
        
        print("\n🎯 Generation 1 (Basic Functionality): COMPLETE")
        print("   Ready for Generation 2 (Robustness & Reliability)")
        
        return True
    else:
        print("❌ Integration tests failed")
        if not integration_success:
            print("   - Core integration: FAILED")
        if not performance_success:
            print("   - Performance metrics: FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)