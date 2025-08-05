"""Parallel processing utilities for scalability."""

import asyncio
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Callable, Any, Optional, Dict, Iterator, Union
from dataclasses import dataclass
import time
import queue

from .logging import get_logger
from .exceptions import WatermarkLabError, TimeoutError
from .metrics import record_operation_metric


@dataclass
class BatchResult:
    """Result of batch processing."""
    
    results: List[Any]
    successes: int
    failures: int
    total_time: float
    errors: List[Exception]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.0


class ParallelProcessor:
    """Parallel processing manager with automatic scaling."""
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = False,
        timeout: Optional[float] = None
    ):
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.use_processes = use_processes
        self.timeout = timeout
        self.logger = get_logger("parallel_processor")
    
    def map(
        self,
        func: Callable,
        items: List[Any],
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult:
        """Execute function on items in parallel."""
        start_time = time.time()
        
        if not items:
            return BatchResult([], 0, 0, 0.0, [])
        
        # Determine optimal batch size
        if batch_size is None:
            batch_size = max(1, len(items) // self.max_workers)
        
        # Create batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        results = []
        errors = []
        successes = 0
        failures = 0
        
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self._process_batch, func, batch): batch
                for batch in batches
            }
            
            # Collect results
            for future in as_completed(future_to_batch, timeout=self.timeout):
                batch = future_to_batch[future]
                
                try:
                    batch_results, batch_errors = future.result()
                    results.extend(batch_results)
                    errors.extend(batch_errors)
                    
                    # Count successes and failures
                    batch_successes = len([r for r in batch_results if r is not None])
                    batch_failures = len(batch) - batch_successes
                    
                    successes += batch_successes
                    failures += batch_failures
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(len(results), len(items))
                
                except Exception as e:
                    self.logger.error(f"Batch processing failed: {e}")
                    errors.append(e)
                    failures += len(batch)
        
        total_time = time.time() - start_time
        
        # Record metrics
        record_operation_metric(
            "parallel_processing",
            total_time,
            success=failures == 0,
            throughput=len(items) / total_time if total_time > 0 else 0
        )
        
        return BatchResult(results, successes, failures, total_time, errors)
    
    def _process_batch(self, func: Callable, batch: List[Any]) -> tuple:
        """Process a batch of items."""
        results = []
        errors = []
        
        for item in batch:
            try:
                result = func(item)
                results.append(result)
            except Exception as e:
                results.append(None)
                errors.append(e)
        
        return results, errors


class AsyncBatchProcessor:
    """Asynchronous batch processor for I/O bound operations."""
    
    def __init__(self, max_concurrent: int = 50, timeout: float = 30.0):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.logger = get_logger("async_batch_processor")
    
    async def process_batch(
        self,
        async_func: Callable,
        items: List[Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult:
        """Process batch of items asynchronously."""
        start_time = time.time()
        
        if not items:
            return BatchResult([], 0, 0, 0.0, [])
        
        results = []
        errors = []
        completed = 0
        
        async def process_item(item):
            async with self.semaphore:
                try:
                    result = await asyncio.wait_for(async_func(item), timeout=self.timeout)
                    return result, None
                except Exception as e:
                    return None, e
        
        # Create tasks
        tasks = [process_item(item) for item in items]
        
        # Process tasks with progress tracking
        for coro in asyncio.as_completed(tasks):
            try:
                result, error = await coro
                results.append(result)
                if error:
                    errors.append(error)
                
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, len(items))
                    
            except Exception as e:
                self.logger.error(f"Task processing failed: {e}")
                results.append(None)
                errors.append(e)
                completed += 1
        
        successes = len([r for r in results if r is not None])
        failures = len(results) - successes
        total_time = time.time() - start_time
        
        return BatchResult(results, successes, failures, total_time, errors)


class WorkerPool:
    """Custom worker pool with job queue."""
    
    def __init__(self, num_workers: int = 4, queue_size: int = 1000):
        self.num_workers = num_workers
        self.job_queue = queue.Queue(maxsize=queue_size)
        self.result_queue = queue.Queue()
        self.workers = []
        self.running = False
        self.logger = get_logger("worker_pool")
    
    def start(self):
        """Start worker threads."""
        if self.running:
            return
        
        self.running = True
        
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        self.logger.info(f"Started {self.num_workers} workers")
    
    def stop(self, timeout: float = 5.0):
        """Stop worker threads."""
        if not self.running:
            return
        
        self.running = False
        
        # Send stop signals
        for _ in range(self.num_workers):
            self.job_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)
        
        self.workers.clear()
        self.logger.info("Stopped all workers")
    
    def submit(self, func: Callable, *args, **kwargs) -> str:
        """Submit job to worker pool."""
        if not self.running:
            raise WatermarkLabError("Worker pool not started")
        
        job_id = f"job_{time.time()}_{id(func)}"
        job = {
            "id": job_id,
            "func": func,
            "args": args,
            "kwargs": kwargs
        }
        
        try:
            self.job_queue.put(job, timeout=1.0)
            return job_id
        except queue.Full:
            raise WatermarkLabError("Job queue is full")
    
    def get_result(self, job_id: str, timeout: Optional[float] = None) -> Any:
        """Get result for submitted job."""
        deadline = time.time() + timeout if timeout else None
        
        while True:
            try:
                queue_timeout = None
                if deadline:
                    queue_timeout = max(0.1, deadline - time.time())
                
                result = self.result_queue.get(timeout=queue_timeout)
                
                if result["id"] == job_id:
                    if result["error"]:
                        raise result["error"]
                    return result["result"]
                else:
                    # Put back result for other job
                    self.result_queue.put(result)
                    
            except queue.Empty:
                if deadline and time.time() >= deadline:
                    raise TimeoutError(f"Timeout waiting for job {job_id}")
                continue
    
    def _worker_loop(self, worker_id: int):
        """Main worker loop."""
        self.logger.debug(f"Worker {worker_id} started")
        
        while self.running:
            try:
                job = self.job_queue.get(timeout=1.0)
                
                if job is None:  # Stop signal
                    break
                
                # Execute job
                try:
                    result = job["func"](*job["args"], **job["kwargs"])
                    self.result_queue.put({
                        "id": job["id"],
                        "result": result,
                        "error": None
                    })
                except Exception as e:
                    self.result_queue.put({
                        "id": job["id"],
                        "result": None,
                        "error": e
                    })
                
                self.job_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
        
        self.logger.debug(f"Worker {worker_id} stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class AdaptiveProcessor:
    """Processor that adapts parallelism based on workload."""
    
    def __init__(self, min_workers: int = 1, max_workers: int = 16):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        self.performance_history = []
        self.logger = get_logger("adaptive_processor")
    
    def process(self, func: Callable, items: List[Any]) -> BatchResult:
        """Process items with adaptive parallelism."""
        if not items:
            return BatchResult([], 0, 0, 0.0, [])
        
        # Determine optimal worker count based on workload
        optimal_workers = self._calculate_optimal_workers(len(items))
        
        processor = ParallelProcessor(max_workers=optimal_workers)
        result = processor.map(func, items)
        
        # Update performance history
        throughput = len(items) / result.total_time if result.total_time > 0 else 0
        self.performance_history.append({
            "workers": optimal_workers,
            "items": len(items),
            "throughput": throughput,
            "success_rate": result.success_rate
        })
        
        # Keep limited history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]
        
        return result
    
    def _calculate_optimal_workers(self, num_items: int) -> int:
        """Calculate optimal number of workers based on workload and history."""
        # Base calculation on item count
        base_workers = min(self.max_workers, max(self.min_workers, num_items // 10))
        
        # Adjust based on performance history
        if len(self.performance_history) >= 3:
            recent_performance = self.performance_history[-3:]
            avg_throughput = sum(p["throughput"] for p in recent_performance) / len(recent_performance)
            
            # If performance is declining, reduce workers
            if len(self.performance_history) >= 6:
                older_performance = self.performance_history[-6:-3]
                old_avg_throughput = sum(p["throughput"] for p in older_performance) / len(older_performance)
                
                if avg_throughput < old_avg_throughput * 0.9:  # 10% decrease
                    base_workers = max(self.min_workers, base_workers - 1)
                elif avg_throughput > old_avg_throughput * 1.1:  # 10% increase
                    base_workers = min(self.max_workers, base_workers + 1)
        
        self.current_workers = base_workers
        return base_workers


# Utility functions for common parallel operations
def parallel_map(
    func: Callable,
    items: List[Any],
    max_workers: Optional[int] = None,
    use_processes: bool = False,
    timeout: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Any]:
    """Simple parallel map function."""
    processor = ParallelProcessor(max_workers, use_processes, timeout)
    result = processor.map(func, items, progress_callback=progress_callback)
    
    if result.failures > 0:
        raise WatermarkLabError(
            f"Parallel processing failed: {result.failures} failures out of {len(items)} items"
        )
    
    return result.results


async def async_map(
    async_func: Callable,
    items: List[Any],
    max_concurrent: int = 50,
    timeout: float = 30.0,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Any]:
    """Simple async map function."""
    processor = AsyncBatchProcessor(max_concurrent, timeout)
    result = await processor.process_batch(async_func, items, progress_callback)
    
    if result.failures > 0:
        raise WatermarkLabError(
            f"Async processing failed: {result.failures} failures out of {len(items)} items"
        )
    
    return result.results


def batch_generator(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
    """Generate batches from list of items."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# Global adaptive processor instance
_global_processor = AdaptiveProcessor()


def get_adaptive_processor() -> AdaptiveProcessor:
    """Get global adaptive processor instance."""
    return _global_processor