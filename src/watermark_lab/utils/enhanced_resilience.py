"""Enhanced resilience patterns for robust operation under adverse conditions."""

import time
import asyncio
import threading
import functools
import random
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class BackoffStrategy(Enum):
    """Backoff strategies for retry operations."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"
    JITTERED = "jittered"


@dataclass
class ResilienceConfig:
    """Configuration for resilience patterns."""
    
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    jitter_factor: float = 0.1
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 30.0
    bulkhead_max_concurrent: int = 10
    timeout_seconds: float = 30.0


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open" 
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics tracking."""
    
    total_requests: int = 0
    failed_requests: int = 0
    success_requests: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changes: int = 0


class EnhancedCircuitBreaker:
    """Advanced circuit breaker with metrics and adaptive thresholds."""
    
    def __init__(self, config: ResilienceConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self._lock = threading.Lock()
        
    def should_allow_request(self) -> bool:
        """Check if request should be allowed through."""
        with self._lock:
            if self.state == CircuitBreakerState.CLOSED:
                return True
            elif self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.metrics.state_changes += 1
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                    return True
                return False
            else:  # HALF_OPEN
                return True
    
    def record_success(self):
        """Record successful operation."""
        with self._lock:
            self.metrics.success_requests += 1
            self.metrics.total_requests += 1
            self.metrics.last_success_time = time.time()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.metrics.state_changes += 1
                logger.info("Circuit breaker reset to CLOSED")
    
    def record_failure(self):
        """Record failed operation."""
        with self._lock:
            self.metrics.failed_requests += 1
            self.metrics.total_requests += 1
            self.metrics.last_failure_time = time.time()
            
            failure_rate = (self.metrics.failed_requests / 
                          max(1, self.metrics.total_requests))
            
            if (self.state != CircuitBreakerState.OPEN and 
                failure_rate >= 0.5 and 
                self.metrics.failed_requests >= self.config.circuit_breaker_threshold):
                self.state = CircuitBreakerState.OPEN
                self.metrics.state_changes += 1
                logger.warning(f"Circuit breaker opened due to failure rate: {failure_rate:.2%}")
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.metrics.last_failure_time is None:
            return True
            
        return (time.time() - self.metrics.last_failure_time) >= self.config.circuit_breaker_timeout
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self._lock:
            return {
                'state': self.state.value,
                'total_requests': self.metrics.total_requests,
                'failed_requests': self.metrics.failed_requests,
                'success_requests': self.metrics.success_requests,
                'failure_rate': (self.metrics.failed_requests / 
                               max(1, self.metrics.total_requests)),
                'state_changes': self.metrics.state_changes,
                'last_failure_time': self.metrics.last_failure_time,
                'last_success_time': self.metrics.last_success_time
            }


class AdaptiveRetry:
    """Adaptive retry mechanism with intelligent backoff."""
    
    def __init__(self, config: ResilienceConfig):
        self.config = config
        self._success_history = deque(maxlen=100)
        
    def calculate_delay(self, attempt: int, base_delay: float = None) -> float:
        """Calculate delay for retry attempt."""
        if base_delay is None:
            base_delay = self.config.initial_delay
            
        # Adjust based on recent success rate
        success_rate = self._get_recent_success_rate()
        adjustment = 1.0 + (1.0 - success_rate) * 0.5  # Slower when success rate is low
        
        if self.config.backoff_strategy == BackoffStrategy.LINEAR:
            delay = base_delay * attempt * adjustment
        elif self.config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = base_delay * (2 ** (attempt - 1)) * adjustment
        elif self.config.backoff_strategy == BackoffStrategy.FIBONACCI:
            delay = base_delay * self._fibonacci(attempt) * adjustment
        else:  # JITTERED
            exponential_delay = base_delay * (2 ** (attempt - 1)) * adjustment
            jitter = random.uniform(-self.config.jitter_factor, 
                                  self.config.jitter_factor)
            delay = exponential_delay * (1 + jitter)
            
        return min(delay, self.config.max_delay)
    
    def record_outcome(self, success: bool):
        """Record operation outcome for adaptive behavior."""
        self._success_history.append(success)
    
    def _get_recent_success_rate(self) -> float:
        """Get recent success rate."""
        if not self._success_history:
            return 0.5  # Neutral assumption
        return sum(self._success_history) / len(self._success_history)
    
    def _fibonacci(self, n: int) -> int:
        """Calculate fibonacci number."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b


class BulkheadIsolation:
    """Bulkhead pattern for resource isolation."""
    
    def __init__(self, config: ResilienceConfig):
        self.config = config
        self._semaphores: Dict[str, threading.Semaphore] = {}
        self._metrics: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {'active': 0, 'queued': 0, 'total': 0, 'rejected': 0}
        )
        self._lock = threading.Lock()
    
    @contextmanager
    def acquire_slot(self, pool_name: str = "default", timeout: float = None):
        """Acquire slot in resource pool."""
        if timeout is None:
            timeout = self.config.timeout_seconds
            
        # Get or create semaphore for pool
        if pool_name not in self._semaphores:
            with self._lock:
                if pool_name not in self._semaphores:
                    self._semaphores[pool_name] = threading.Semaphore(
                        self.config.bulkhead_max_concurrent
                    )
        
        semaphore = self._semaphores[pool_name]
        metrics = self._metrics[pool_name]
        
        # Try to acquire with timeout
        metrics['queued'] += 1
        metrics['total'] += 1
        
        try:
            acquired = semaphore.acquire(timeout=timeout)
            if not acquired:
                metrics['rejected'] += 1
                raise TimeoutError(f"Failed to acquire slot in pool '{pool_name}' within {timeout}s")
            
            metrics['queued'] -= 1
            metrics['active'] += 1
            
            try:
                yield
            finally:
                metrics['active'] -= 1
                semaphore.release()
                
        except Exception:
            metrics['queued'] -= 1
            raise
    
    def get_pool_metrics(self, pool_name: str = "default") -> Dict[str, int]:
        """Get metrics for specific pool."""
        return dict(self._metrics[pool_name])
    
    def get_all_metrics(self) -> Dict[str, Dict[str, int]]:
        """Get metrics for all pools."""
        return {pool: dict(metrics) for pool, metrics in self._metrics.items()}


class ResilienceManager:
    """Centralized resilience management."""
    
    def __init__(self, config: ResilienceConfig = None):
        self.config = config or ResilienceConfig()
        self.circuit_breakers: Dict[str, EnhancedCircuitBreaker] = {}
        self.retry_handlers: Dict[str, AdaptiveRetry] = {}
        self.bulkheads: Dict[str, BulkheadIsolation] = {}
        self._lock = threading.Lock()
    
    def get_circuit_breaker(self, name: str) -> EnhancedCircuitBreaker:
        """Get or create circuit breaker."""
        if name not in self.circuit_breakers:
            with self._lock:
                if name not in self.circuit_breakers:
                    self.circuit_breakers[name] = EnhancedCircuitBreaker(self.config)
        return self.circuit_breakers[name]
    
    def get_retry_handler(self, name: str) -> AdaptiveRetry:
        """Get or create retry handler."""
        if name not in self.retry_handlers:
            with self._lock:
                if name not in self.retry_handlers:
                    self.retry_handlers[name] = AdaptiveRetry(self.config)
        return self.retry_handlers[name]
    
    def get_bulkhead(self, name: str) -> BulkheadIsolation:
        """Get or create bulkhead."""
        if name not in self.bulkheads:
            with self._lock:
                if name not in self.bulkheads:
                    self.bulkheads[name] = BulkheadIsolation(self.config)
        return self.bulkheads[name]
    
    def execute_with_resilience(
        self,
        func: Callable,
        *args,
        service_name: str = "default",
        pool_name: str = "default",
        **kwargs
    ) -> Any:
        """Execute function with full resilience patterns."""
        circuit_breaker = self.get_circuit_breaker(service_name)
        retry_handler = self.get_retry_handler(service_name)
        bulkhead = self.get_bulkhead(pool_name)
        
        last_exception = None
        
        for attempt in range(1, self.config.max_retries + 1):
            if not circuit_breaker.should_allow_request():
                raise Exception(f"Circuit breaker open for service '{service_name}'")
            
            try:
                with bulkhead.acquire_slot(pool_name):
                    result = func(*args, **kwargs)
                    circuit_breaker.record_success()
                    retry_handler.record_outcome(True)
                    return result
                    
            except Exception as e:
                last_exception = e
                circuit_breaker.record_failure()
                retry_handler.record_outcome(False)
                
                if attempt < self.config.max_retries:
                    delay = retry_handler.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt} failed for service '{service_name}'. "
                        f"Retrying in {delay:.2f}s. Error: {str(e)}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {self.config.max_retries} attempts failed for service '{service_name}'"
                    )
        
        raise last_exception
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get metrics from all resilience components."""
        return {
            'circuit_breakers': {
                name: cb.get_metrics() 
                for name, cb in self.circuit_breakers.items()
            },
            'bulkheads': {
                name: bh.get_all_metrics() 
                for name, bh in self.bulkheads.items()
            },
            'config': {
                'max_retries': self.config.max_retries,
                'initial_delay': self.config.initial_delay,
                'max_delay': self.config.max_delay,
                'backoff_strategy': self.config.backoff_strategy.value,
                'circuit_breaker_threshold': self.config.circuit_breaker_threshold,
                'bulkhead_max_concurrent': self.config.bulkhead_max_concurrent
            }
        }


def resilient(
    service_name: str = "default",
    pool_name: str = "default",
    config: ResilienceConfig = None
):
    """Decorator for adding resilience patterns to functions."""
    
    if config is None:
        config = ResilienceConfig()
    
    manager = ResilienceManager(config)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return manager.execute_with_resilience(
                func, *args, 
                service_name=service_name,
                pool_name=pool_name,
                **kwargs
            )
        return wrapper
    return decorator


async def resilient_async(
    func: Callable,
    *args,
    service_name: str = "default", 
    pool_name: str = "default",
    config: ResilienceConfig = None,
    **kwargs
) -> Any:
    """Async version of resilient execution."""
    
    if config is None:
        config = ResilienceConfig()
    
    manager = ResilienceManager(config)
    circuit_breaker = manager.get_circuit_breaker(service_name)
    retry_handler = manager.get_retry_handler(service_name)
    
    last_exception = None
    
    for attempt in range(1, config.max_retries + 1):
        if not circuit_breaker.should_allow_request():
            raise Exception(f"Circuit breaker open for service '{service_name}'")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                
            circuit_breaker.record_success()
            retry_handler.record_outcome(True)
            return result
            
        except Exception as e:
            last_exception = e
            circuit_breaker.record_failure()
            retry_handler.record_outcome(False)
            
            if attempt < config.max_retries:
                delay = retry_handler.calculate_delay(attempt)
                logger.warning(
                    f"Async attempt {attempt} failed for service '{service_name}'. "
                    f"Retrying in {delay:.2f}s. Error: {str(e)}"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"All {config.max_retries} async attempts failed for service '{service_name}'"
                )
    
    raise last_exception


# Global resilience manager instance
_global_manager = ResilienceManager()

def get_global_resilience_manager() -> ResilienceManager:
    """Get global resilience manager instance."""
    return _global_manager