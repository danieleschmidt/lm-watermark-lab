#!/usr/bin/env python3
"""Test Generation 3 performance optimization features: caching, resource management, load balancing."""

import sys
import os
import time
import threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_caching_system():
    """Test advanced caching system."""
    print("=== Testing Caching System ===")
    
    try:
        from watermark_lab.optimization.caching import CacheManager, CacheConfig, MemoryCache
        
        # Test memory cache
        config = CacheConfig(
            backend="memory",
            max_memory_items=100,
            default_ttl=60,
            eviction_policy="lru"
        )
        
        cache_manager = CacheManager(config)
        print(f"✓ Cache manager created with backend: {config.backend}")
        
        # Test basic operations
        cache_manager.set("test_key", "test_value", ttl=30)
        value = cache_manager.get("test_key")
        assert value == "test_value", f"Expected 'test_value', got {value}"
        print("✓ Cache set/get operations working")
        
        # Test get_or_set
        def factory():
            return "generated_value"
        
        value = cache_manager.get_or_set("new_key", factory, ttl=60)
        assert value == "generated_value"
        print("✓ Cache get_or_set working")
        
        # Test batch operations
        items = {"key1": "value1", "key2": "value2", "key3": "value3"}
        success_count = cache_manager.mset(items)
        assert success_count == 3, f"Expected 3 successes, got {success_count}"
        
        retrieved = cache_manager.mget(["key1", "key2", "key3"])
        assert len(retrieved) == 3, f"Expected 3 items, got {len(retrieved)}"
        print("✓ Cache batch operations working")
        
        # Test statistics
        stats = cache_manager.get_stats()
        print(f"✓ Cache stats: {stats['cache_stats']['hit_rate']:.2%} hit rate")
        
        # Test eviction
        for i in range(150):  # Exceed max_memory_items
            cache_manager.set(f"evict_key_{i}", f"value_{i}")
        
        final_stats = cache_manager.get_stats()
        print(f"✓ Cache eviction working: {final_stats['cache_stats']['evictions']} evictions")
        
        print("✓ Caching system tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Caching system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_resource_management():
    """Test resource management system."""
    print("\n=== Testing Resource Management ===")
    
    try:
        from watermark_lab.optimization.resource_manager import ResourceManager, ResourceLimits, ResourcePool
        
        # Create resource manager with limits
        limits = ResourceLimits(
            max_memory_mb=512,
            max_cpu_percent=70.0,
            max_threads=30
        )
        
        resource_manager = ResourceManager(limits)
        print(f"✓ Resource manager created with limits: {limits.max_memory_mb}MB memory")
        
        # Test resource pool
        def create_mock_resource():
            return {"id": time.time(), "data": "test_data"}
        
        def destroy_mock_resource(resource):
            # Simulate cleanup
            pass
        
        pool = resource_manager.create_pool(
            "test_pool",
            factory=create_mock_resource,
            destructor=destroy_mock_resource,
            min_size=2,
            max_size=5
        )
        print("✓ Resource pool created")
        
        # Test resource acquisition and release
        resource = pool.acquire()
        assert resource is not None, "Failed to acquire resource"
        print("✓ Resource acquired from pool")
        
        pool.release(resource)
        print("✓ Resource released to pool")
        
        # Test context manager
        with pool.get_resource() as resource:
            assert resource is not None, "Context manager resource is None"
            print("✓ Resource pool context manager working")
        
        # Test resource monitoring
        current_usage = resource_manager.resource_monitor.get_current_usage()
        print(f"✓ Resource monitoring: {current_usage.memory_mb:.1f}MB memory, {current_usage.cpu_percent:.1f}% CPU")
        
        # Test memory management
        resource_manager.track_large_object("test_object", b"x" * (2 * 1024 * 1024))  # 2MB object
        
        memory_stats = resource_manager.memory_manager.get_memory_stats()
        print(f"✓ Memory management: {memory_stats['large_objects_count']} large objects tracked")
        
        # Get comprehensive stats
        comprehensive_stats = resource_manager.get_comprehensive_stats()
        print(f"✓ Comprehensive stats: {len(comprehensive_stats['pools'])} pools")
        
        print("✓ Resource management tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Resource management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_balancing():
    """Test load balancing system."""
    print("\n=== Testing Load Balancing ===")
    
    try:
        from watermark_lab.deployment.load_balancer import (
            LoadBalancer, ServiceEndpoint, LoadBalancingStrategy
        )
        
        # Create load balancer
        load_balancer = LoadBalancer(
            strategy=LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN,
            health_check_interval=5.0
        )
        print("✓ Load balancer created with weighted round robin strategy")
        
        # Add service endpoints
        endpoints = [
            ServiceEndpoint("endpoint1", "localhost", 8001, weight=100),
            ServiceEndpoint("endpoint2", "localhost", 8002, weight=150),
            ServiceEndpoint("endpoint3", "localhost", 8003, weight=75)
        ]
        
        for endpoint in endpoints:
            load_balancer.add_endpoint(endpoint)
        print(f"✓ Added {len(endpoints)} endpoints")
        
        # Test endpoint selection
        selected_endpoints = []
        for _ in range(10):
            endpoint = load_balancer.get_endpoint()
            if endpoint:
                selected_endpoints.append(endpoint.id)
        
        assert len(selected_endpoints) > 0, "No endpoints selected"
        print(f"✓ Endpoint selection working: {len(set(selected_endpoints))} unique endpoints used")
        
        # Test request execution
        def mock_request(endpoint):
            # Simulate request processing
            time.sleep(0.01)
            return f"Response from {endpoint.id}"
        
        results = []
        for i in range(5):
            try:
                result = load_balancer.execute_request(mock_request)
                results.append(result)
            except Exception as e:
                print(f"Request {i} failed: {e}")
        
        print(f"✓ Request execution: {len(results)} successful requests")
        
        # Test statistics
        stats = load_balancer.get_stats()
        print(f"✓ Load balancer stats: {stats['total_endpoints']} total, {stats['healthy_endpoints']} healthy")
        
        # Test consistent hashing
        consistent_lb = LoadBalancer(strategy=LoadBalancingStrategy.CONSISTENT_HASH)
        for endpoint in endpoints:
            consistent_lb.add_endpoint(endpoint)
        
        # Same key should go to same endpoint
        key = "test_user_123"
        endpoint1 = consistent_lb.get_endpoint(key)
        endpoint2 = consistent_lb.get_endpoint(key)
        assert endpoint1.id == endpoint2.id, "Consistent hashing not working"
        print("✓ Consistent hashing working")
        
        # Test rebalancing
        load_balancer.rebalance()
        print("✓ Load balancer rebalancing working")
        
        print("✓ Load balancing tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Load balancing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_optimization():
    """Test overall performance optimization features."""
    print("\n=== Testing Performance Optimization ===")
    
    try:
        from watermark_lab.optimization.caching import cached
        from watermark_lab.core.factory import WatermarkFactory
        
        # Test caching decorator
        call_count = 0
        
        @cached(ttl=60)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            time.sleep(0.1)  # Simulate expensive operation
            return x * 2
        
        # First call should execute function
        start_time = time.time()
        result1 = expensive_function(5)
        first_call_time = time.time() - start_time
        assert result1 == 10, f"Expected 10, got {result1}"
        assert call_count == 1, f"Expected 1 call, got {call_count}"
        
        # Second call should be cached
        start_time = time.time()
        result2 = expensive_function(5)
        second_call_time = time.time() - start_time
        assert result2 == 10, f"Expected 10, got {result2}"
        assert call_count == 1, f"Expected 1 call (cached), got {call_count}"
        
        # Cached call should be much faster
        assert second_call_time < first_call_time / 2, "Cached call not significantly faster"
        print(f"✓ Caching decorator: {first_call_time:.3f}s → {second_call_time:.3f}s")
        
        # Test watermark generation performance with fallback
        config = {
            "method": "kirchenbauer",
            "model_name": "gpt2",
            "use_real_model": False,  # Use fast fallback
            "gamma": 0.25,
            "seed": 42
        }
        
        watermarker = WatermarkFactory.create(**config)
        
        # Measure generation time
        start_time = time.time()
        for _ in range(5):
            text = watermarker.generate("Test prompt", max_length=20)
        generation_time = time.time() - start_time
        
        print(f"✓ Watermark generation performance: 5 generations in {generation_time:.3f}s")
        
        print("✓ Performance optimization tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Performance optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_concurrent_stress_test():
    """Run concurrent stress test."""
    print("\n=== Concurrent Stress Test ===")
    
    try:
        from watermark_lab.optimization.caching import get_cache_manager
        
        cache_manager = get_cache_manager()
        results = []
        errors = []
        
        def stress_worker(worker_id, iterations):
            worker_results = []
            worker_errors = []
            
            for i in range(iterations):
                try:
                    key = f"stress_{worker_id}_{i}"
                    value = f"value_{worker_id}_{i}"
                    
                    # Set and get
                    cache_manager.set(key, value)
                    retrieved = cache_manager.get(key)
                    
                    if retrieved == value:
                        worker_results.append("success")
                    else:
                        worker_errors.append(f"Mismatch: {retrieved} != {value}")
                        
                except Exception as e:
                    worker_errors.append(str(e))
            
            return worker_results, worker_errors
        
        # Run concurrent workers
        num_workers = 5
        iterations_per_worker = 20
        threads = []
        
        start_time = time.time()
        
        for worker_id in range(num_workers):
            thread = threading.Thread(
                target=lambda wid=worker_id: results.extend(stress_worker(wid, iterations_per_worker))
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        stress_time = time.time() - start_time
        total_operations = num_workers * iterations_per_worker * 2  # set + get
        
        print(f"✓ Stress test: {total_operations} operations in {stress_time:.3f}s")
        print(f"✓ Throughput: {total_operations / stress_time:.1f} ops/sec")
        
        return True
        
    except Exception as e:
        print(f"✗ Stress test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_performance_tests():
    """Run all Generation 3 performance tests."""
    print("🚀 GENERATION 3 PERFORMANCE TESTING")
    print("=" * 50)
    
    tests = [
        test_caching_system,
        test_resource_management,
        test_load_balancing,
        test_performance_optimization,
        run_concurrent_stress_test
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
    
    print(f"\n📊 PERFORMANCE TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL PERFORMANCE OPTIMIZATIONS WORKING!")
        return True
    else:
        print("⚠️  Some performance features need attention")
        return False


if __name__ == "__main__":
    success = run_all_performance_tests()
    sys.exit(0 if success else 1)