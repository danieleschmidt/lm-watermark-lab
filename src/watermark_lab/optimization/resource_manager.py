"""Advanced resource management for optimal performance and scalability."""

import os
import gc
import psutil
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from contextlib import contextmanager
from collections import defaultdict, deque

from ..utils.logging import get_logger
from ..utils.exceptions import ResourceError, WatermarkLabError


@dataclass
class ResourceLimits:
    """Resource limits configuration."""
    
    max_memory_mb: int = 1024
    max_cpu_percent: float = 80.0
    max_disk_io_mb_per_sec: float = 100.0
    max_network_io_mb_per_sec: float = 50.0
    max_open_files: int = 1000
    max_threads: int = 50
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'max_memory_mb': self.max_memory_mb,
            'max_cpu_percent': self.max_cpu_percent,
            'max_disk_io_mb_per_sec': self.max_disk_io_mb_per_sec,
            'max_network_io_mb_per_sec': self.max_network_io_mb_per_sec,
            'max_open_files': self.max_open_files,
            'max_threads': self.max_threads
        }


@dataclass
class ResourceUsage:
    """Current resource usage statistics."""
    
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    disk_read_mb_per_sec: float = 0.0
    disk_write_mb_per_sec: float = 0.0
    network_sent_mb_per_sec: float = 0.0
    network_recv_mb_per_sec: float = 0.0
    open_files: int = 0
    threads: int = 0
    
    def exceeds_limits(self, limits: ResourceLimits) -> List[str]:
        """Check which limits are exceeded."""
        violations = []
        
        if self.memory_mb > limits.max_memory_mb:
            violations.append(f"Memory: {self.memory_mb:.1f}MB > {limits.max_memory_mb}MB")
        
        if self.cpu_percent > limits.max_cpu_percent:
            violations.append(f"CPU: {self.cpu_percent:.1f}% > {limits.max_cpu_percent}%")
        
        if self.disk_read_mb_per_sec + self.disk_write_mb_per_sec > limits.max_disk_io_mb_per_sec:
            total_io = self.disk_read_mb_per_sec + self.disk_write_mb_per_sec
            violations.append(f"Disk I/O: {total_io:.1f}MB/s > {limits.max_disk_io_mb_per_sec}MB/s")
        
        if self.network_sent_mb_per_sec + self.network_recv_mb_per_sec > limits.max_network_io_mb_per_sec:
            total_net = self.network_sent_mb_per_sec + self.network_recv_mb_per_sec
            violations.append(f"Network I/O: {total_net:.1f}MB/s > {limits.max_network_io_mb_per_sec}MB/s")
        
        if self.open_files > limits.max_open_files:
            violations.append(f"Open files: {self.open_files} > {limits.max_open_files}")
        
        if self.threads > limits.max_threads:
            violations.append(f"Threads: {self.threads} > {limits.max_threads}")
        
        return violations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'memory_mb': self.memory_mb,
            'memory_percent': self.memory_percent,
            'cpu_percent': self.cpu_percent,
            'disk_read_mb_per_sec': self.disk_read_mb_per_sec,
            'disk_write_mb_per_sec': self.disk_write_mb_per_sec,
            'network_sent_mb_per_sec': self.network_sent_mb_per_sec,
            'network_recv_mb_per_sec': self.network_recv_mb_per_sec,
            'open_files': self.open_files,
            'threads': self.threads
        }


class ResourcePool:
    """Generic resource pool with limits and lifecycle management."""
    
    def __init__(
        self,
        name: str,
        factory: Callable[[], Any],
        destructor: Optional[Callable[[Any], None]] = None,
        min_size: int = 0,
        max_size: int = 10,
        idle_timeout: float = 300.0
    ):
        self.name = name
        self.factory = factory
        self.destructor = destructor
        self.min_size = min_size
        self.max_size = max_size
        self.idle_timeout = idle_timeout
        
        self._pool = deque()
        self._active_resources = {}
        self._resource_times = {}
        self._lock = threading.RLock()
        self._created_count = 0
        self._destroyed_count = 0
        
        self.logger = get_logger(f"resource_pool.{name}")
        
        # Pre-populate minimum resources
        self._populate_minimum()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_running = True
        self._cleanup_thread.start()
    
    def acquire(self, timeout: Optional[float] = None) -> Any:
        """Acquire a resource from the pool."""
        with self._lock:
            # Try to get from pool first
            if self._pool:
                resource = self._pool.popleft()
                self._active_resources[id(resource)] = resource
                self.logger.debug(f"Acquired existing resource from {self.name}")
                return resource
            
            # Create new resource if under limit
            if len(self._active_resources) < self.max_size:
                resource = self._create_resource()
                if resource:
                    self._active_resources[id(resource)] = resource
                    self.logger.debug(f"Created new resource for {self.name}")
                    return resource
            
            # Pool exhausted
            raise ResourceError(f"Resource pool {self.name} exhausted (max: {self.max_size})")
    
    def release(self, resource: Any):
        """Release a resource back to the pool."""
        with self._lock:
            resource_id = id(resource)
            
            if resource_id not in self._active_resources:
                self.logger.warning(f"Attempting to release unknown resource to {self.name}")
                return
            
            del self._active_resources[resource_id]
            
            # Return to pool if under minimum or pool is not full
            if len(self._pool) < self.min_size or len(self._pool) < (self.max_size - len(self._active_resources)):
                self._pool.append(resource)
                self._resource_times[id(resource)] = time.time()
                self.logger.debug(f"Released resource to {self.name} pool")
            else:
                self._destroy_resource(resource)
                self.logger.debug(f"Destroyed excess resource from {self.name}")
    
    def _create_resource(self) -> Optional[Any]:
        """Create a new resource."""
        try:
            resource = self.factory()
            self._created_count += 1
            return resource
        except Exception as e:
            self.logger.error(f"Failed to create resource for {self.name}: {e}")
            return None
    
    def _destroy_resource(self, resource: Any):
        """Destroy a resource."""
        try:
            if self.destructor:
                self.destructor(resource)
            self._destroyed_count += 1
        except Exception as e:
            self.logger.error(f"Failed to destroy resource from {self.name}: {e}")
    
    def _populate_minimum(self):
        """Populate pool with minimum resources."""
        with self._lock:
            for _ in range(self.min_size):
                resource = self._create_resource()
                if resource:
                    self._pool.append(resource)
                    self._resource_times[id(resource)] = time.time()
    
    def _cleanup_loop(self):
        """Cleanup idle resources."""
        while self._cleanup_running:
            try:
                current_time = time.time()
                
                with self._lock:
                    # Remove idle resources
                    expired_resources = []
                    
                    for resource in list(self._pool):
                        resource_id = id(resource)
                        if resource_id in self._resource_times:
                            idle_time = current_time - self._resource_times[resource_id]
                            if idle_time > self.idle_timeout and len(self._pool) > self.min_size:
                                expired_resources.append(resource)
                    
                    for resource in expired_resources:
                        self._pool.remove(resource)
                        resource_id = id(resource)
                        if resource_id in self._resource_times:
                            del self._resource_times[resource_id]
                        self._destroy_resource(resource)
                        self.logger.debug(f"Cleaned up idle resource from {self.name}")
                
            except Exception as e:
                self.logger.error(f"Cleanup error in {self.name}: {e}")
            
            time.sleep(60)  # Cleanup every minute
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                'name': self.name,
                'pool_size': len(self._pool),
                'active_resources': len(self._active_resources),
                'min_size': self.min_size,
                'max_size': self.max_size,
                'created_count': self._created_count,
                'destroyed_count': self._destroyed_count,
                'idle_timeout': self.idle_timeout
            }
    
    def shutdown(self):
        """Shutdown the resource pool."""
        self._cleanup_running = False
        
        with self._lock:
            # Destroy all pooled resources
            for resource in list(self._pool):
                self._destroy_resource(resource)
            self._pool.clear()
            
            # Destroy active resources (should ideally be empty)
            for resource in list(self._active_resources.values()):
                self._destroy_resource(resource)
            self._active_resources.clear()
        
        self.logger.info(f"Shut down resource pool {self.name}")
    
    @contextmanager
    def get_resource(self):
        """Context manager for resource acquisition and release."""
        resource = self.acquire()
        try:
            yield resource
        finally:
            self.release(resource)


class MemoryManager:
    """Advanced memory management with monitoring and optimization."""
    
    def __init__(self, gc_threshold: float = 0.8, emergency_threshold: float = 0.95):
        self.gc_threshold = gc_threshold
        self.emergency_threshold = emergency_threshold
        self.logger = get_logger("memory_manager")
        
        # Memory tracking
        self.memory_history = deque(maxlen=100)
        self.large_objects = {}
        self.allocation_tracking = defaultdict(int)
        
        # Configure garbage collection
        self._configure_gc()
        
        # Start monitoring
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._memory_monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def _configure_gc(self):
        """Configure garbage collection for optimal performance."""
        # Tune garbage collection thresholds
        gc.set_threshold(700, 10, 10)  # More aggressive than default
        
        # Enable garbage collection debugging in development
        if os.getenv('WATERMARK_DEBUG'):
            gc.set_debug(gc.DEBUG_STATS)
    
    def _memory_monitor_loop(self):
        """Monitor memory usage and trigger cleanup when needed."""
        while self._monitoring_active:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()
                
                # Record memory usage
                self.memory_history.append({
                    'timestamp': time.time(),
                    'rss': memory_info.rss,
                    'vms': memory_info.vms,
                    'percent': memory_percent
                })
                
                # Check thresholds and take action
                if memory_percent > self.emergency_threshold:
                    self.logger.warning(f"Emergency memory usage: {memory_percent:.1f}%")
                    self._emergency_cleanup()
                elif memory_percent > self.gc_threshold:
                    self.logger.info(f"High memory usage: {memory_percent:.1f}%, triggering GC")
                    self._trigger_gc()
                
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
            
            time.sleep(30)  # Monitor every 30 seconds
    
    def _trigger_gc(self):
        """Trigger garbage collection."""
        collected = gc.collect()
        self.logger.debug(f"Garbage collection freed {collected} objects")
    
    def _emergency_cleanup(self):
        """Emergency memory cleanup procedures."""
        # Force garbage collection
        for generation in range(3):
            collected = gc.collect(generation)
            self.logger.debug(f"Emergency GC generation {generation}: {collected} objects")
        
        # Clear caches if available
        try:
            from ..optimization.caching import get_cache_manager
            cache_manager = get_cache_manager()
            cache_manager.clear()
            self.logger.info("Cleared application caches")
        except Exception:
            pass
        
        # Clear large objects if tracked
        if self.large_objects:
            self.large_objects.clear()
            self.logger.info("Cleared large object references")
    
    def track_large_object(self, name: str, obj: Any):
        """Track large objects for memory management."""
        import sys
        size = sys.getsizeof(obj)
        
        if size > 1024 * 1024:  # Track objects > 1MB
            self.large_objects[name] = {
                'object': obj,
                'size': size,
                'timestamp': time.time()
            }
            self.logger.debug(f"Tracking large object {name}: {size} bytes")
    
    def untrack_object(self, name: str):
        """Remove object from tracking."""
        if name in self.large_objects:
            del self.large_objects[name]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get garbage collection stats
        gc_stats = gc.get_stats()
        
        # Calculate memory growth rate
        growth_rate = 0.0
        if len(self.memory_history) >= 2:
            recent = self.memory_history[-1]
            older = self.memory_history[-min(10, len(self.memory_history))]
            time_diff = recent['timestamp'] - older['timestamp']
            if time_diff > 0:
                memory_diff = recent['rss'] - older['rss']
                growth_rate = memory_diff / time_diff
        
        return {
            'current_memory_mb': memory_info.rss / (1024 * 1024),
            'current_memory_percent': process.memory_percent(),
            'virtual_memory_mb': memory_info.vms / (1024 * 1024),
            'memory_growth_rate_mb_per_sec': growth_rate / (1024 * 1024),
            'gc_stats': gc_stats,
            'large_objects_count': len(self.large_objects),
            'large_objects_total_size_mb': sum(
                obj['size'] for obj in self.large_objects.values()
            ) / (1024 * 1024),
            'gc_threshold': self.gc_threshold,
            'emergency_threshold': self.emergency_threshold
        }


class ResourceMonitor:
    """System resource monitoring and alerting."""
    
    def __init__(
        self,
        limits: ResourceLimits,
        check_interval: float = 10.0,
        alert_callbacks: Optional[List[Callable[[str, ResourceUsage], None]]] = None
    ):
        self.limits = limits
        self.check_interval = check_interval
        self.alert_callbacks = alert_callbacks or []
        
        self.logger = get_logger("resource_monitor")
        self._monitoring_active = False
        self._monitor_thread = None
        
        # Resource usage history
        self.usage_history = deque(maxlen=360)  # Keep 1 hour at 10s intervals
        
        # Baseline measurements for calculating rates
        self._last_disk_io = None
        self._last_network_io = None
        self._last_measurement_time = None
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info(f"Started resource monitoring with {self.check_interval}s interval")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        self.logger.info("Stopped resource monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                usage = self._collect_resource_usage()
                
                # Store in history
                self.usage_history.append({
                    'timestamp': time.time(),
                    'usage': usage
                })
                
                # Check limits and send alerts
                violations = usage.exceeds_limits(self.limits)
                if violations:
                    alert_message = f"Resource limits exceeded: {'; '.join(violations)}"
                    self.logger.warning(alert_message)
                    
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert_message, usage)
                        except Exception as e:
                            self.logger.error(f"Alert callback failed: {e}")
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
            
            time.sleep(self.check_interval)
    
    def _collect_resource_usage(self) -> ResourceUsage:
        """Collect current resource usage."""
        try:
            process = psutil.Process()
            
            # Memory usage
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            memory_percent = process.memory_percent()
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            
            # Disk I/O
            disk_io = process.io_counters()
            disk_read_rate, disk_write_rate = self._calculate_disk_io_rates(disk_io)
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_sent_rate, network_recv_rate = self._calculate_network_io_rates(network_io)
            
            # Open files and threads
            open_files = len(process.open_files())
            threads = process.num_threads()
            
            return ResourceUsage(
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                cpu_percent=cpu_percent,
                disk_read_mb_per_sec=disk_read_rate,
                disk_write_mb_per_sec=disk_write_rate,
                network_sent_mb_per_sec=network_sent_rate,
                network_recv_mb_per_sec=network_recv_rate,
                open_files=open_files,
                threads=threads
            )
            
        except Exception as e:
            self.logger.error(f"Resource collection failed: {e}")
            return ResourceUsage()
    
    def _calculate_disk_io_rates(self, current_io) -> tuple:
        """Calculate disk I/O rates."""
        current_time = time.time()
        
        if self._last_disk_io and self._last_measurement_time:
            time_diff = current_time - self._last_measurement_time
            
            if time_diff > 0:
                read_diff = current_io.read_bytes - self._last_disk_io.read_bytes
                write_diff = current_io.write_bytes - self._last_disk_io.write_bytes
                
                read_rate = (read_diff / time_diff) / (1024 * 1024)  # MB/s
                write_rate = (write_diff / time_diff) / (1024 * 1024)  # MB/s
            else:
                read_rate = write_rate = 0.0
        else:
            read_rate = write_rate = 0.0
        
        self._last_disk_io = current_io
        self._last_measurement_time = current_time
        
        return read_rate, write_rate
    
    def _calculate_network_io_rates(self, current_io) -> tuple:
        """Calculate network I/O rates."""
        current_time = time.time()
        
        if self._last_network_io and self._last_measurement_time:
            time_diff = current_time - self._last_measurement_time
            
            if time_diff > 0:
                sent_diff = current_io.bytes_sent - self._last_network_io.bytes_sent
                recv_diff = current_io.bytes_recv - self._last_network_io.bytes_recv
                
                sent_rate = (sent_diff / time_diff) / (1024 * 1024)  # MB/s
                recv_rate = (recv_diff / time_diff) / (1024 * 1024)  # MB/s
            else:
                sent_rate = recv_rate = 0.0
        else:
            sent_rate = recv_rate = 0.0
        
        self._last_network_io = current_io
        
        return sent_rate, recv_rate
    
    def get_current_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        return self._collect_resource_usage()
    
    def get_usage_history(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get resource usage history."""
        cutoff_time = time.time() - (minutes * 60)
        return [
            entry for entry in self.usage_history
            if entry['timestamp'] > cutoff_time
        ]
    
    def add_alert_callback(self, callback: Callable[[str, ResourceUsage], None]):
        """Add alert callback."""
        self.alert_callbacks.append(callback)


class ResourceManager:
    """Unified resource management system."""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.logger = get_logger("resource_manager")
        
        # Components
        self.memory_manager = MemoryManager()
        self.resource_monitor = ResourceMonitor(self.limits)
        self.resource_pools: Dict[str, ResourcePool] = {}
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
        
        self.logger.info("Resource manager initialized")
    
    def create_pool(
        self,
        name: str,
        factory: Callable[[], Any],
        destructor: Optional[Callable[[Any], None]] = None,
        min_size: int = 0,
        max_size: int = 10,
        idle_timeout: float = 300.0
    ) -> ResourcePool:
        """Create a new resource pool."""
        if name in self.resource_pools:
            raise ValueError(f"Resource pool {name} already exists")
        
        pool = ResourcePool(name, factory, destructor, min_size, max_size, idle_timeout)
        self.resource_pools[name] = pool
        
        self.logger.info(f"Created resource pool: {name}")
        return pool
    
    def get_pool(self, name: str) -> Optional[ResourcePool]:
        """Get existing resource pool."""
        return self.resource_pools.get(name)
    
    def track_large_object(self, name: str, obj: Any):
        """Track large object for memory management."""
        self.memory_manager.track_large_object(name, obj)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive resource statistics."""
        stats = {
            'limits': self.limits.to_dict(),
            'current_usage': self.resource_monitor.get_current_usage().to_dict(),
            'memory_stats': self.memory_manager.get_memory_stats(),
            'pools': {}
        }
        
        # Add pool statistics
        for name, pool in self.resource_pools.items():
            stats['pools'][name] = pool.get_stats()
        
        return stats
    
    def shutdown(self):
        """Shutdown resource manager."""
        self.logger.info("Shutting down resource manager...")
        
        # Stop monitoring
        self.resource_monitor.stop_monitoring()
        self.memory_manager._monitoring_active = False
        
        # Shutdown all pools
        for pool in self.resource_pools.values():
            pool.shutdown()
        
        self.logger.info("Resource manager shutdown complete")


# Global resource manager instance
_global_resource_manager = None


def get_resource_manager(limits: Optional[ResourceLimits] = None) -> ResourceManager:
    """Get global resource manager instance."""
    global _global_resource_manager
    
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager(limits)
    
    return _global_resource_manager


__all__ = [
    "ResourceManager",
    "ResourceLimits",
    "ResourceUsage",
    "ResourcePool",
    "MemoryManager",
    "ResourceMonitor",
    "get_resource_manager"
]