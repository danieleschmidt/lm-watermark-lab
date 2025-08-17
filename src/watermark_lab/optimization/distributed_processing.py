"""Distributed processing system for large-scale watermarking operations."""

import time
import asyncio
import threading
import multiprocessing as mp
import pickle
import json
import hashlib
from typing import Dict, Any, List, Optional, Callable, Union, Iterator, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from enum import Enum
import logging
import queue
import socket
from pathlib import Path

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Distributed task definition."""
    
    task_id: str
    function_name: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    dependencies: List[str] = field(default_factory=list)
    
    # Execution state
    status: TaskStatus = field(default=TaskStatus.PENDING, init=False)
    started_at: Optional[float] = field(default=None, init=False)
    completed_at: Optional[float] = field(default=None, init=False)
    result: Any = field(default=None, init=False)
    error: Optional[str] = field(default=None, init=False)
    worker_id: Optional[str] = field(default=None, init=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'task_id': self.task_id,
            'function_name': self.function_name,
            'args': self.args,
            'kwargs': self.kwargs,
            'priority': self.priority.value,
            'timeout': self.timeout,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'created_at': self.created_at,
            'dependencies': self.dependencies,
            'status': self.status.value,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'result': self.result,
            'error': self.error,
            'worker_id': self.worker_id
        }


@dataclass
class WorkerStats:
    """Worker performance statistics."""
    
    worker_id: str
    started_at: float = field(default_factory=time.time)
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    current_task: Optional[str] = None
    last_heartbeat: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    
    def get_success_rate(self) -> float:
        """Calculate task success rate."""
        total = self.tasks_completed + self.tasks_failed
        return self.tasks_completed / total if total > 0 else 1.0
    
    def get_avg_execution_time(self) -> float:
        """Calculate average execution time."""
        return self.total_execution_time / self.tasks_completed if self.tasks_completed > 0 else 0.0


class TaskFunction:
    """Wrapper for distributed task functions."""
    
    def __init__(self, func: Callable, name: str = None):
        self.func = func
        self.name = name or func.__name__
        
    def __call__(self, *args, **kwargs):
        """Execute function with error handling."""
        try:
            return self.func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Task function {self.name} failed: {e}")
            raise


class TaskQueue:
    """Distributed task queue with priority handling."""
    
    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self._queues = {
            TaskPriority.CRITICAL: queue.PriorityQueue(),
            TaskPriority.HIGH: queue.PriorityQueue(),
            TaskPriority.NORMAL: queue.PriorityQueue(),
            TaskPriority.LOW: queue.PriorityQueue()
        }
        self._task_map: Dict[str, Task] = {}
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        
    def put(self, task: Task) -> bool:
        """Add task to queue."""
        with self._condition:
            if len(self._task_map) >= self.maxsize:
                return False
            
            # Check if dependencies are satisfied
            if not self._are_dependencies_satisfied(task):
                task.status = TaskStatus.PENDING
                self._task_map[task.task_id] = task
                return True
            
            # Add to appropriate priority queue
            priority_value = (-task.priority.value, task.created_at)  # Negative for highest priority first
            self._queues[task.priority].put((priority_value, task))
            self._task_map[task.task_id] = task
            
            # Update dependency graph
            for dep_id in task.dependencies:
                self._dependency_graph[dep_id].add(task.task_id)
            
            self._condition.notify_all()
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[Task]:
        """Get next task from queue."""
        with self._condition:
            deadline = time.time() + timeout if timeout else None
            
            while True:
                # Try to get task from priority queues
                for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, 
                               TaskPriority.NORMAL, TaskPriority.LOW]:
                    try:
                        _, task = self._queues[priority].get_nowait()
                        task.status = TaskStatus.RUNNING
                        task.started_at = time.time()
                        return task
                    except queue.Empty:
                        continue
                
                # Check for newly available tasks due to dependency resolution
                self._check_dependencies()
                
                # Wait for new tasks or timeout
                if deadline and time.time() >= deadline:
                    return None
                
                wait_time = deadline - time.time() if deadline else None
                self._condition.wait(timeout=wait_time)
    
    def complete_task(self, task_id: str, result: Any = None, error: str = None):
        """Mark task as completed and check dependencies."""
        with self._condition:
            if task_id in self._task_map:
                task = self._task_map[task_id]
                task.completed_at = time.time()
                
                if error:
                    task.status = TaskStatus.FAILED
                    task.error = error
                else:
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                
                # Check if any pending tasks can now be scheduled
                self._check_dependencies()
                self._condition.notify_all()
    
    def _are_dependencies_satisfied(self, task: Task) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_id in task.dependencies:
            if dep_id not in self._task_map:
                return False
            dep_task = self._task_map[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
    
    def _check_dependencies(self):
        """Check pending tasks for satisfied dependencies."""
        pending_tasks = [
            task for task in self._task_map.values()
            if task.status == TaskStatus.PENDING and self._are_dependencies_satisfied(task)
        ]
        
        for task in pending_tasks:
            priority_value = (-task.priority.value, task.created_at)
            self._queues[task.priority].put((priority_value, task))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            status_counts = defaultdict(int)
            for task in self._task_map.values():
                status_counts[task.status.value] += 1
            
            queue_sizes = {
                priority.name: self._queues[priority].qsize()
                for priority in TaskPriority
            }
            
            return {
                'total_tasks': len(self._task_map),
                'status_counts': dict(status_counts),
                'queue_sizes': queue_sizes,
                'pending_dependencies': len([
                    t for t in self._task_map.values() 
                    if t.status == TaskStatus.PENDING
                ])
            }


class DistributedWorker:
    """Distributed task worker."""
    
    def __init__(self, worker_id: str, task_queue: TaskQueue, task_registry: Dict[str, TaskFunction]):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.task_registry = task_registry
        self.stats = WorkerStats(worker_id)
        self.running = False
        self._thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start worker thread."""
        if self.running:
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        logger.info(f"Worker {self.worker_id} started")
    
    def stop(self):
        """Stop worker thread."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info(f"Worker {self.worker_id} stopped")
    
    def _worker_loop(self):
        """Main worker loop."""
        while self.running:
            try:
                task = self.task_queue.get(timeout=1.0)
                if task is None:
                    continue
                
                self._execute_task(task)
                
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                time.sleep(0.1)
    
    def _execute_task(self, task: Task):
        """Execute a single task."""
        self.stats.current_task = task.task_id
        start_time = time.time()
        
        try:
            # Get task function
            if task.function_name not in self.task_registry:
                raise ValueError(f"Unknown function: {task.function_name}")
            
            func = self.task_registry[task.function_name]
            
            # Execute with timeout
            if task.timeout:
                # For simplicity, we don't implement timeout here
                # In production, you'd use signal-based timeouts or subprocess
                result = func(*task.args, **task.kwargs)
            else:
                result = func(*task.args, **task.kwargs)
            
            # Task completed successfully
            execution_time = time.time() - start_time
            self.stats.tasks_completed += 1
            self.stats.total_execution_time += execution_time
            self.task_queue.complete_task(task.task_id, result=result)
            
            logger.debug(f"Worker {self.worker_id} completed task {task.task_id} in {execution_time:.2f}s")
            
        except Exception as e:
            # Task failed
            execution_time = time.time() - start_time
            self.stats.tasks_failed += 1
            self.stats.total_execution_time += execution_time
            self.task_queue.complete_task(task.task_id, error=str(e))
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                task.started_at = None
                task.error = None
                self.task_queue.put(task)
                logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries}): {e}")
            else:
                logger.error(f"Task {task.task_id} failed permanently: {e}")
        
        finally:
            self.stats.current_task = None
            self.stats.last_heartbeat = time.time()


class DistributedProcessingEngine:
    """Main distributed processing engine."""
    
    def __init__(self, num_workers: int = None, max_queue_size: int = 10000):
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        self.num_workers = num_workers
        self.task_queue = TaskQueue(max_queue_size)
        self.task_registry: Dict[str, TaskFunction] = {}
        self.workers: List[DistributedWorker] = []
        self.worker_stats: Dict[str, WorkerStats] = {}
        self.running = False
        
        # Task results storage
        self.results: Dict[str, Any] = {}
        self.errors: Dict[str, str] = {}
        
        # Performance metrics
        self.start_time: Optional[float] = None
        self.total_tasks_processed = 0
        
    def register_function(self, func: Callable, name: str = None) -> str:
        """Register function for distributed execution."""
        task_func = TaskFunction(func, name)
        self.task_registry[task_func.name] = task_func
        logger.info(f"Registered function: {task_func.name}")
        return task_func.name
    
    def start(self):
        """Start distributed processing engine."""
        if self.running:
            return
        
        self.running = True
        self.start_time = time.time()
        
        # Create and start workers
        for i in range(self.num_workers):
            worker_id = f"worker-{i}"
            worker = DistributedWorker(worker_id, self.task_queue, self.task_registry)
            worker.start()
            self.workers.append(worker)
            self.worker_stats[worker_id] = worker.stats
        
        logger.info(f"Distributed processing engine started with {self.num_workers} workers")
    
    def stop(self):
        """Stop distributed processing engine."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop all workers
        for worker in self.workers:
            worker.stop()
        
        logger.info("Distributed processing engine stopped")
    
    def submit_task(
        self, 
        function_name: str,
        args: tuple = (),
        kwargs: dict = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        dependencies: List[str] = None,
        task_id: str = None
    ) -> str:
        """Submit task for distributed execution."""
        if kwargs is None:
            kwargs = {}
        if dependencies is None:
            dependencies = []
        if task_id is None:
            task_id = self._generate_task_id(function_name, args, kwargs)
        
        task = Task(
            task_id=task_id,
            function_name=function_name,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries,
            dependencies=dependencies
        )
        
        if not self.task_queue.put(task):
            raise RuntimeError("Task queue is full")
        
        logger.debug(f"Submitted task: {task_id}")
        return task_id
    
    def submit_batch(
        self,
        function_name: str,
        arg_batches: List[tuple],
        kwarg_batches: List[dict] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        max_retries: int = 3
    ) -> List[str]:
        """Submit batch of tasks."""
        if kwarg_batches is None:
            kwarg_batches = [{}] * len(arg_batches)
        
        task_ids = []
        for i, args in enumerate(arg_batches):
            kwargs = kwarg_batches[i] if i < len(kwarg_batches) else {}
            task_id = self.submit_task(
                function_name, args, kwargs, priority, timeout, max_retries
            )
            task_ids.append(task_id)
        
        logger.info(f"Submitted batch of {len(task_ids)} tasks")
        return task_ids
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for task completion and return result."""
        deadline = time.time() + timeout if timeout else None
        
        while True:
            if task_id in self.task_queue._task_map:
                task = self.task_queue._task_map[task_id]
                
                if task.status == TaskStatus.COMPLETED:
                    return task.result
                elif task.status == TaskStatus.FAILED:
                    raise RuntimeError(f"Task {task_id} failed: {task.error}")
                elif task.status == TaskStatus.CANCELLED:
                    raise RuntimeError(f"Task {task_id} was cancelled")
            
            if deadline and time.time() >= deadline:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
            
            time.sleep(0.1)
    
    def wait_for_tasks(self, task_ids: List[str], timeout: Optional[float] = None) -> List[Any]:
        """Wait for multiple tasks to complete."""
        results = []
        for task_id in task_ids:
            result = self.wait_for_task(task_id, timeout)
            results.append(result)
        return results
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status."""
        if task_id in self.task_queue._task_map:
            return self.task_queue._task_map[task_id].status
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel pending task."""
        with self.task_queue._lock:
            if task_id in self.task_queue._task_map:
                task = self.task_queue._task_map[task_id]
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.CANCELLED
                    return True
        return False
    
    def _generate_task_id(self, function_name: str, args: tuple, kwargs: dict) -> str:
        """Generate unique task ID."""
        data = f"{function_name}:{str(args)}:{str(kwargs)}:{time.time()}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        queue_stats = self.task_queue.get_stats()
        
        # Worker statistics
        worker_stats = {}
        total_completed = 0
        total_failed = 0
        total_execution_time = 0.0
        
        for worker_id, stats in self.worker_stats.items():
            worker_stats[worker_id] = {
                'tasks_completed': stats.tasks_completed,
                'tasks_failed': stats.tasks_failed,
                'success_rate': stats.get_success_rate(),
                'avg_execution_time': stats.get_avg_execution_time(),
                'current_task': stats.current_task,
                'last_heartbeat': stats.last_heartbeat
            }
            
            total_completed += stats.tasks_completed
            total_failed += stats.tasks_failed
            total_execution_time += stats.total_execution_time
        
        # Engine statistics
        uptime = time.time() - self.start_time if self.start_time else 0
        throughput = total_completed / uptime if uptime > 0 else 0
        
        return {
            'uptime': uptime,
            'total_tasks_completed': total_completed,
            'total_tasks_failed': total_failed,
            'total_execution_time': total_execution_time,
            'throughput_tasks_per_second': throughput,
            'success_rate': total_completed / (total_completed + total_failed) if (total_completed + total_failed) > 0 else 1.0,
            'avg_execution_time': total_execution_time / total_completed if total_completed > 0 else 0,
            'num_workers': self.num_workers,
            'queue_stats': queue_stats,
            'worker_stats': worker_stats
        }
    
    def scale_workers(self, target_workers: int):
        """Dynamically scale number of workers."""
        current_workers = len(self.workers)
        
        if target_workers > current_workers:
            # Add workers
            for i in range(current_workers, target_workers):
                worker_id = f"worker-{i}"
                worker = DistributedWorker(worker_id, self.task_queue, self.task_registry)
                worker.start()
                self.workers.append(worker)
                self.worker_stats[worker_id] = worker.stats
            
            logger.info(f"Scaled up from {current_workers} to {target_workers} workers")
            
        elif target_workers < current_workers:
            # Remove workers
            workers_to_remove = self.workers[target_workers:]
            self.workers = self.workers[:target_workers]
            
            for worker in workers_to_remove:
                worker.stop()
                del self.worker_stats[worker.worker_id]
            
            logger.info(f"Scaled down from {current_workers} to {target_workers} workers")
        
        self.num_workers = target_workers


# Global distributed processing engine
_global_engine: Optional[DistributedProcessingEngine] = None

def get_global_engine() -> DistributedProcessingEngine:
    """Get or create global distributed processing engine."""
    global _global_engine
    if _global_engine is None:
        _global_engine = DistributedProcessingEngine()
        _global_engine.start()
    return _global_engine


def distributed_task(
    priority: TaskPriority = TaskPriority.NORMAL,
    timeout: Optional[float] = None,
    max_retries: int = 3
):
    """Decorator for creating distributed tasks."""
    def decorator(func):
        engine = get_global_engine()
        function_name = engine.register_function(func)
        
        def wrapper(*args, **kwargs):
            # If called directly, execute locally
            return func(*args, **kwargs)
        
        def submit(*args, **kwargs):
            # Submit for distributed execution
            return engine.submit_task(
                function_name, args, kwargs, priority, timeout, max_retries
            )
        
        def submit_batch(arg_batches, kwarg_batches=None):
            # Submit batch for distributed execution
            return engine.submit_batch(
                function_name, arg_batches, kwarg_batches, priority, timeout, max_retries
            )
        
        wrapper.submit = submit
        wrapper.submit_batch = submit_batch
        wrapper.function_name = function_name
        
        return wrapper
    
    return decorator