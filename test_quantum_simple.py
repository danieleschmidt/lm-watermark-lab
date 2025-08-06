#!/usr/bin/env python3
"""Simple test of quantum-inspired task planning without external dependencies."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_core_quantum_functionality():
    """Test core quantum functionality without numpy dependency."""
    print("🧪 Testing Core Quantum Functionality...")
    
    try:
        # Test quantum task and planner basic classes
        print("\n1️⃣ Testing quantum task creation...")
        
        from watermark_lab.core.quantum_planner import (
            QuantumTask, TaskPriority, TaskState, QuantumTaskPlanner
        )
        
        # Create a simple quantum task
        task = QuantumTask(
            id="test_1",
            name="Test Watermark Generation", 
            description="Generate watermarked text for testing",
            priority=TaskPriority.GROUND_STATE
        )
        
        print(f"✅ Created quantum task: {task.name}")
        print(f"   Task ID: {task.id}")
        print(f"   Priority: {task.priority}")
        print(f"   State: {task.state}")
        print(f"   Probability: {task.calculate_probability():.3f}")
        
        # Test planner initialization
        print("\n2️⃣ Testing quantum planner...")
        planner = QuantumTaskPlanner(max_coherence_time=200.0)
        print(f"✅ Initialized planner with coherence time: {planner.max_coherence_time}")
        
        # Add tasks to planner
        task_id_1 = planner.add_task(
            "Generate kirchenbauer watermark",
            "Generate text using Kirchenbauer method",
            TaskPriority.EXCITED_1
        )
        
        task_id_2 = planner.add_task(
            "Detect watermark",
            "Detect watermark in generated text", 
            TaskPriority.EXCITED_2,
            dependencies=[task_id_1]
        )
        
        print(f"✅ Added 2 tasks to planner")
        print(f"   Task 1: {task_id_1}")
        print(f"   Task 2: {task_id_2} (depends on Task 1)")
        
        # Test entanglement
        print("\n3️⃣ Testing quantum entanglement...")
        planner.entangle_tasks([task_id_1, task_id_2], "test_entanglement")
        
        state = planner.get_system_state()
        print(f"✅ System state after entanglement:")
        print(f"   Total tasks: {state['total_tasks']}")
        print(f"   Entanglement groups: {state['entanglement_groups']}")
        print(f"   Total coherence: {state['global_quantum_state']['total_coherence']:.3f}")
        
        # Test execution planning
        print("\n4️⃣ Testing execution planning...")
        execution_plan = planner.plan_execution()
        print(f"✅ Generated execution plan with {len(execution_plan)} tasks")
        
        # Test optimization
        print("\n5️⃣ Testing quantum optimization...")
        optimization = planner.optimize_plan("coherence")
        print(f"✅ Optimization completed: {optimization}")
        
        return True
        
    except Exception as e:
        print(f"❌ Core test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_watermark_factory():
    """Test basic watermark factory functionality."""
    print("\n🔬 Testing Watermark Factory...")
    
    try:
        from watermark_lab.core.factory import WatermarkFactory, KirchenbauerWatermark
        
        # Test factory
        print("1️⃣ Testing factory creation...")
        methods = WatermarkFactory.list_methods()
        print(f"✅ Available methods: {methods}")
        
        # Test creating watermark instances
        print("2️⃣ Testing watermark instance creation...")
        watermarker = WatermarkFactory.create('kirchenbauer', gamma=0.25, delta=2.0)
        print(f"✅ Created {watermarker.__class__.__name__}")
        
        config = watermarker.get_config()
        print(f"   Config: {config}")
        
        return True
        
    except Exception as e:
        print(f"❌ Watermark factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_detector():
    """Test detector functionality."""
    print("\n🕵️ Testing Detector...")
    
    try:
        from watermark_lab.core.detector import WatermarkDetector
        
        # Test detector creation
        config = {"method": "kirchenbauer", "gamma": 0.25, "delta": 2.0}
        detector = WatermarkDetector(config)
        
        # Test detection on sample text
        test_text = "The future of artificial intelligence involves many exciting developments"
        result = detector.detect(test_text)
        
        print(f"✅ Detection completed:")
        print(f"   Watermarked: {result.is_watermarked}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   P-value: {result.p_value:.6f}")
        print(f"   Method: {result.method}")
        
        return True
        
    except Exception as e:
        print(f"❌ Detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_attacks():
    """Test attack simulation."""
    print("\n⚔️ Testing Attack Simulation...")
    
    try:
        from watermark_lab.core.attacks import AttackSimulator, ParaphraseAttack
        
        # Test individual attack
        print("1️⃣ Testing paraphrase attack...")
        paraphrase_attack = ParaphraseAttack()
        
        test_text = "The algorithm demonstrates excellent performance in watermark detection tasks"
        attack_result = paraphrase_attack.attack(test_text, "medium")
        
        print(f"✅ Paraphrase attack completed:")
        print(f"   Original: {attack_result.original_text}")
        print(f"   Attacked: {attack_result.attacked_text}")
        print(f"   Success: {attack_result.attack_success}")
        print(f"   Quality: {attack_result.quality_score:.3f}")
        print(f"   Similarity: {attack_result.similarity_score:.3f}")
        
        # Test attack simulator
        print("2️⃣ Testing attack simulator...")
        simulator = AttackSimulator()
        available_attacks = list(simulator.attacks.keys())
        print(f"   Available attacks: {available_attacks}")
        
        # Run one attack
        result = simulator.run_attack(test_text, "paraphrase", "medium")
        print(f"✅ Simulator attack completed: {result.attack_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ Attack test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Quantum-Inspired Watermarking System Tests")
    print("=" * 60)
    
    tests = [
        test_core_quantum_functionality,
        test_watermark_factory, 
        test_detector,
        test_attacks
    ]
    
    results = []
    
    for test_func in tests:
        try:
            success = test_func()
            results.append(success)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_func, success) in enumerate(zip(tests, results)):
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{i+1}. {test_func.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("   ✨ Quantum-inspired task planning: OPERATIONAL")
        print("   🔬 Watermarking algorithms: FUNCTIONAL") 
        print("   🕵️ Detection system: WORKING")
        print("   ⚔️ Attack simulation: READY")
        print("\n🚀 Generation 1 (Basic Functionality) COMPLETE!")
    else:
        print(f"\n⚠️ {total - passed} tests failed - system needs debugging")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)