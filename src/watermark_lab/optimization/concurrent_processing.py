"""Concurrent processing and resource pooling for watermark operations."""

import asyncio
import concurrent.futures
import threading
import time
import logging
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic, Union
from dataclasses import dataclass, field
from enum import Enum
import queue
import weakref
from contextlib import contextmanager
import resource

T = TypeVar('T')
R = TypeVar('R')

class PoolType(Enum):
    """Types of worker pools."""
    THREAD = "thread"
    PROCESS = "process"
    ASYNC = "async"

class Priority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    """Task representation for concurrent processing."""
    id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: Priority = Priority.NORMAL
    timeout: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None

@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    worker_id: Optional[str] = None

class ResourcePool(Generic[T]):
    """Generic resource pool with lifecycle management."""
    
    def __init__(self, 
                 create_func: Callable[[], T],
                 cleanup_func: Optional[Callable[[T], None]] = None,
                 max_size: int = 10,
                 min_size: int = 1,
                 max_idle_time: float = 300.0,
                 health_check_func: Optional[Callable[[T], bool]] = None):
        """
        Initialize resource pool.
        
        Args:
            create_func: Function to create new resources
            cleanup_func: Function to cleanup resources
            max_size: Maximum pool size
            min_size: Minimum pool size
            max_idle_time: Maximum idle time before cleanup
            health_check_func: Function to check resource health
        """
        self.create_func = create_func
        self.cleanup_func = cleanup_func
        self.max_size = max_size
        self.min_size = min_size
        self.max_idle_time = max_idle_time
        self.health_check_func = health_check_func
        
        self._pool: queue.Queue = queue.Queue(maxsize=max_size)
        self._lock = threading.RLock()
        self._created_count = 0
        self._active_count = 0
        self._resource_timestamps: Dict[int, float] = {}
        
        self.logger = logging.getLogger(f"pool.{id(self)}")
        
        # Pre-populate with minimum resources
        self._ensure_minimum_size()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._stop_cleanup = threading.Event()
        self._cleanup_thread.start()
    
    def _ensure_minimum_size(self) -> None:
        """Ensure pool has minimum number of resources."""
        with self._lock:
            current_size = self._pool.qsize()
            for _ in range(max(0, self.min_size - current_size)):
                try:
                    resource = self.create_func()
                    self._pool.put(resource, block=False)
                    self._created_count += 1
                    self._resource_timestamps[id(resource)] = time.time()
                except Exception as e:
                    self.logger.error(f"Failed to create resource: {e}")
                    break
    
    def _cleanup_loop(self) -> None:
        """Background cleanup of idle resources."""
        while not self._stop_cleanup.wait(60):  # Check every minute
            try:
                self._cleanup_idle_resources()
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
    
    def _cleanup_idle_resources(self) -> None:
        """Remove idle resources exceeding max_idle_time."""
        with self._lock:
            current_time = time.time()
            temp_resources = []
            
            # Check all resources in pool
            while not self._pool.empty():
                try:
                    resource = self._pool.get_nowait()
                    resource_id = id(resource)
                    
                    # Check if resource is too old or unhealthy
                    is_old = (current_time - self._resource_timestamps.get(resource_id, current_time)) > self.max_idle_time
                    is_unhealthy = self.health_check_func and not self.health_check_func(resource)
                    
                    if (is_old or is_unhealthy) and len(temp_resources) + self._pool.qsize() >= self.min_size:
                        # Cleanup old/unhealthy resource
                        if self.cleanup_func:
                            try:
                                self.cleanup_func(resource)
                            except Exception as e:
                                self.logger.warning(f"Resource cleanup failed: {e}")
                        
                        self._resource_timestamps.pop(resource_id, None)
                        self.logger.debug(f"Cleaned up {'old' if is_old else 'unhealthy'} resource")
                    else:
                        temp_resources.append(resource)
                        
                except queue.Empty:
                    break
            
            # Put remaining resources back
            for resource in temp_resources:
                try:
                    self._pool.put_nowait(resource)
                except queue.Full:
                    # Pool is full, cleanup extra resource
                    if self.cleanup_func:
                        self.cleanup_func(resource)
    
    @contextmanager
    def get_resource(self, timeout: Optional[float] = None):
        """Context manager to get and return resource."""
        resource = None
        try:
            # Try to get existing resource
            try:
                resource = self._pool.get(timeout=timeout or 5.0)
                with self._lock:
                    self._active_count += 1
            except queue.Empty:
                # Create new resource if under limit
                with self._lock:
                    if self._created_count < self.max_size:
                        resource = self.create_func()
                        self._created_count += 1
                        self._active_count += 1
                        self._resource_timestamps[id(resource)] = time.time()
                    else:
                        raise RuntimeError("Resource pool exhausted")
            
            # Health check
            if self.health_check_func and not self.health_check_func(resource):
                raise RuntimeError("Resource failed health check")
            
            yield resource
            
        finally:
            if resource is not None:
                with self._lock:
                    self._active_count -= 1
                    
                # Return resource to pool
                try:
                    self._resource_timestamps[id(resource)] = time.time()
                    self._pool.put_nowait(resource)
                except queue.Full:
                    # Pool is full, cleanup resource
                    if self.cleanup_func:
                        self.cleanup_func(resource)
                    self._created_count -= 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "pool_size": self._pool.qsize(),
                "active_count": self._active_count,
                "created_count": self._created_count,
                "max_size": self.max_size,
                "min_size": self.min_size,
                "utilization": self._active_count / self.max_size if self.max_size > 0 else 0
            }
    
    def shutdown(self) -> None:
        """Shutdown pool and cleanup all resources."""
        self._stop_cleanup.set()
        
        # Cleanup all resources in pool
        while not self._pool.empty():
            try:
                resource = self._pool.get_nowait()
                if self.cleanup_func:
                    self.cleanup_func(resource)
            except queue.Empty:
                break

class ConcurrentExecutor:
    """High-performance concurrent executor with resource management."""
    
    def __init__(self,
                 pool_type: PoolType = PoolType.THREAD,
                 max_workers: int = 4,
                 queue_size: int = 1000,
                 enable_priorities: bool = True):
        """
        Initialize concurrent executor.
        
        Args:
            pool_type: Type of worker pool
            max_workers: Maximum number of workers
            queue_size: Maximum queue size
            enable_priorities: Enable priority-based scheduling
        """
        self.pool_type = pool_type
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.enable_priorities = enable_priorities
        
        # Task queue with priority support
        if enable_priorities:
            self._task_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=queue_size)
        else:
            self._task_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        
        self._results: Dict[str, TaskResult] = {}
        self._executor: Optional[concurrent.futures.Executor] = None
        self._workers: List[threading.Thread] = []
        self._running = False
        
        # Statistics
        self._submitted_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0
        
        self.logger = logging.getLogger(f"executor.{pool_type.value}")
        
        # Start executor
        self._start()
    
    def _start(self) -> None:
        """Start the executor."""
        self._running = True
        
        if self.pool_type == PoolType.THREAD:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="watermark_worker"
            )
        elif self.pool_type == PoolType.PROCESS:
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            )
        
        # Start worker threads for task dispatch
        for i in range(min(self.max_workers, 2)):  # Limited dispatcher threads
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"dispatcher_{i}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
        
        self.logger.info(f"Started {self.pool_type.value} executor with {self.max_workers} workers")
    
    def _worker_loop(self) -> None:
        """Main worker loop for task processing."""
        while self._running:
            try:
                # Get task from queue
                if self.enable_priorities:
                    priority_item = self._task_queue.get(timeout=1.0)
                    _, task = priority_item  # (priority, task)
                else:
                    task = self._task_queue.get(timeout=1.0)
                
                # Execute task
                self._execute_task(task)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
    
    def _execute_task(self, task: Task) -> None:
        """Execute a single task."""
        start_time = time.time()
        worker_id = threading.current_thread().name
        
        try:
            # Execute with timeout
            if self.pool_type == PoolType.ASYNC:
                # Handle async tasks
                result = asyncio.run(task.func(*task.args, **task.kwargs))
            else:
                # Use executor for thread/process pools
                future = self._executor.submit(task.func, *task.args, **task.kwargs)
                result = future.result(timeout=task.timeout)
            
            # Create success result
            task_result = TaskResult(
                task_id=task.id,
                success=True,
                result=result,
                execution_time=time.time() - start_time,
                worker_id=worker_id
            )
            
            self._completed_tasks += 1
            
            # Call success callback
            if task.callback:
                try:
                    task.callback(task_result)
                except Exception as e:
                    self.logger.warning(f"Success callback failed: {e}")
            
        except Exception as e:
            # Create error result
            task_result = TaskResult(
                task_id=task.id,
                success=False,
                error=e,
                execution_time=time.time() - start_time,
                worker_id=worker_id
            )
            
            self._failed_tasks += 1
            
            # Call error callback
            if task.error_callback:
                try:
                    task.error_callback(task_result)
                except Exception as callback_error:
                    self.logger.warning(f"Error callback failed: {callback_error}")
        
        # Store result
        self._results[task.id] = task_result
    
    def submit(self, 
               func: Callable,
               *args,
               task_id: Optional[str] = None,
               priority: Priority = Priority.NORMAL,
               timeout: Optional[float] = None,
               callback: Optional[Callable] = None,
               error_callback: Optional[Callable] = None,
               **kwargs) -> str:
        """
        Submit task for execution.
        
        Returns:
            Task ID for result retrieval
        """
        if not self._running:
            raise RuntimeError("Executor is not running")
        
        # Generate task ID
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000000)}"
        
        # Create task
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            callback=callback,
            error_callback=error_callback
        )
        
        # Submit to queue
        try:
            if self.enable_priorities:
                # Higher priority value = higher priority (reverse for queue)
                priority_value = -priority.value
                self._task_queue.put((priority_value, task), timeout=1.0)
            else:
                self._task_queue.put(task, timeout=1.0)
            
            self._submitted_tasks += 1
            return task_id
            
        except queue.Full:
            raise RuntimeError("Task queue is full")
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """Get task result (blocking)."""
        deadline = time.time() + (timeout or float('inf'))
        
        while time.time() < deadline:
            if task_id in self._results:
                return self._results[task_id]
            time.sleep(0.01)  # 10ms polling
        
        raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
    
    def get_result_nowait(self, task_id: str) -> Optional[TaskResult]:
        """Get task result (non-blocking)."""
        return self._results.get(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            "pool_type": self.pool_type.value,
            "max_workers": self.max_workers,
            "queue_size": self._task_queue.qsize(),
            "max_queue_size": self.queue_size,
            "submitted_tasks": self._submitted_tasks,
            "completed_tasks": self._completed_tasks,
            "failed_tasks": self._failed_tasks,
            "success_rate": self._completed_tasks / max(self._submitted_tasks, 1),
            "active_workers": len([w for w in self._workers if w.is_alive()]),
            "results_cached": len(self._results)
        }
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown executor."""
        self._running = False
        
        if self._executor:
            self._executor.shutdown(wait=wait)
        
        if wait:
            for worker in self._workers:
                worker.join(timeout=5.0)
        
        self.logger.info("Executor shutdown complete")

class ResourceManager:
    """Global resource manager for pools and executors."""
    
    def __init__(self):
        self._pools: Dict[str, ResourcePool] = {}
        self._executors: Dict[str, ConcurrentExecutor] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger("resource.manager")
    
    def get_model_pool(self) -> ResourcePool:
        """Get model resource pool."""
        return self._get_or_create_pool(
            "models",
            create_func=self._create_model,
            cleanup_func=self._cleanup_model,
            max_size=5,  # Limited model pool
            min_size=1
        )
    
    def get_executor(self, 
                    name: str,
                    pool_type: PoolType = PoolType.THREAD,
                    max_workers: Optional[int] = None) -> ConcurrentExecutor:
        """Get or create named executor."""
        with self._lock:
            if name not in self._executors:
                if max_workers is None:
                    # Auto-detect based on system
                    import os
                    cpu_count = os.cpu_count() or 4
                    max_workers = min(cpu_count, 8)  # Reasonable limit
                
                self._executors[name] = ConcurrentExecutor(
                    pool_type=pool_type,
                    max_workers=max_workers
                )
                self.logger.info(f"Created executor: {name}")
            
            return self._executors[name]
    
    def _get_or_create_pool(self, name: str, **kwargs) -> ResourcePool:
        """Get or create named resource pool."""
        with self._lock:
            if name not in self._pools:
                self._pools[name] = ResourcePool(**kwargs)
                self.logger.info(f"Created resource pool: {name}")
            return self._pools[name]
    
    def _create_model(self):
        """Create model instance (placeholder)."""
        # This would create actual model instances
        return {"type": "model", "created_at": time.time()}
    
    def _cleanup_model(self, model):
        """Cleanup model instance (placeholder)."""
        # This would cleanup actual model instances
        pass
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all resources."""
        with self._lock:
            return {
                "pools": {name: pool.get_stats() for name, pool in self._pools.items()},
                "executors": {name: executor.get_stats() for name, executor in self._executors.items()},
                "system": {
                    "cpu_count": resource.cpu_count(),
                    "memory_usage": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                }
            }
    
    def shutdown_all(self) -> None:
        """Shutdown all resources."""
        with self._lock:
            # Shutdown executors
            for executor in self._executors.values():
                executor.shutdown(wait=False)
            
            # Shutdown pools
            for pool in self._pools.values():
                pool.shutdown()
            
            self.logger.info("All resources shut down")

# Global resource manager
_resource_manager = None

def get_resource_manager() -> ResourceManager:
    """Get global resource manager."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager