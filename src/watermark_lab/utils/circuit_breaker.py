"""Circuit breaker implementation for fault tolerance and reliability."""

import time
import threading
from typing import Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
from functools import wraps

from .logging import get_logger
from .exceptions import WatermarkLabError, TimeoutError, ResourceError


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, blocking requests
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5       # Number of failures before opening
    recovery_timeout: float = 60.0   # Seconds before trying to recover
    timeout: float = 30.0            # Request timeout in seconds
    max_requests: int = 3            # Max requests in half-open state
    success_threshold: int = 2       # Successes needed to close circuit
    sliding_window: int = 100        # Size of metrics sliding window


class CircuitBreakerMetrics:
    """Metrics tracking for circuit breaker."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.successes = []
        self.failures = []
        self.timeouts = []
        self.requests = []
        self.lock = threading.RLock()
    
    def record_success(self):
        """Record a successful operation."""
        with self.lock:
            current_time = time.time()
            self.successes.append(current_time)
            self.requests.append(current_time)
            self._trim_window()
    
    def record_failure(self):
        """Record a failed operation."""
        with self.lock:
            current_time = time.time()
            self.failures.append(current_time)
            self.requests.append(current_time)
            self._trim_window()
    
    def record_timeout(self):
        """Record a timeout operation."""
        with self.lock:
            current_time = time.time()
            self.timeouts.append(current_time)
            self.requests.append(current_time)
            self._trim_window()
    
    def get_failure_rate(self) -> float:
        """Get current failure rate."""
        with self.lock:
            total_requests = len(self.requests)
            if total_requests == 0:
                return 0.0
            
            total_failures = len(self.failures) + len(self.timeouts)
            return total_failures / total_requests
    
    def get_recent_failures(self, time_window: float = 60.0) -> int:
        """Get number of failures in recent time window."""
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - time_window
            
            recent_failures = sum(1 for t in self.failures if t > cutoff_time)
            recent_timeouts = sum(1 for t in self.timeouts if t > cutoff_time)
            
            return recent_failures + recent_timeouts
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        with self.lock:
            return {
                'total_requests': len(self.requests),
                'total_successes': len(self.successes),
                'total_failures': len(self.failures),
                'total_timeouts': len(self.timeouts),
                'failure_rate': self.get_failure_rate(),
                'recent_failures_1min': self.get_recent_failures(60),
                'recent_failures_5min': self.get_recent_failures(300),
            }
    
    def _trim_window(self):
        """Trim metrics to sliding window size."""
        if len(self.requests) > self.window_size:
            # Remove oldest entries
            excess = len(self.requests) - self.window_size
            oldest_time = sorted(self.requests)[excess]
            
            self.requests = [t for t in self.requests if t > oldest_time]
            self.successes = [t for t in self.successes if t > oldest_time]
            self.failures = [t for t in self.failures if t > oldest_time]
            self.timeouts = [t for t in self.timeouts if t > oldest_time]


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0.0
        self.half_open_requests = 0
        self.half_open_successes = 0
        self.metrics = CircuitBreakerMetrics(self.config.sliding_window)
        self.lock = threading.RLock()
        self.logger = get_logger(f"circuit_breaker.{name}")
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        
        with self.lock:
            # Check if circuit should be opened
            self._update_state()
            
            if self.state == CircuitState.OPEN:
                self.logger.warning(f"Circuit breaker {self.name} is OPEN - rejecting request")
                raise ResourceError(f"Circuit breaker {self.name} is open")
            
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_requests >= self.config.max_requests:
                    self.logger.warning(f"Circuit breaker {self.name} HALF_OPEN - max requests reached")
                    raise ResourceError(f"Circuit breaker {self.name} is half-open with max requests")
                
                self.half_open_requests += 1
        
        # Execute the function with timeout
        start_time = time.time()
        try:
            if hasattr(func, '__name__'):
                self.logger.debug(f"Executing {func.__name__} via circuit breaker {self.name}")
            
            result = self._execute_with_timeout(func, *args, **kwargs)
            
            execution_time = time.time() - start_time
            self._record_success(execution_time)
            
            return result
            
        except TimeoutError as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Circuit breaker {self.name} - timeout after {execution_time:.2f}s")
            self._record_timeout()
            raise
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Circuit breaker {self.name} - failure after {execution_time:.2f}s: {e}")
            self._record_failure()
            raise
    
    def _execute_with_timeout(self, func: Callable, *args, **kwargs):
        """Execute function with timeout protection."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {self.config.timeout}s")
        
        # Set timeout (Unix-like systems only)
        try:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.config.timeout))
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
        except AttributeError:
            # Windows or signal not available - use threading fallback
            return self._execute_with_thread_timeout(func, *args, **kwargs)
    
    def _execute_with_thread_timeout(self, func: Callable, *args, **kwargs):
        """Fallback timeout implementation using threads."""
        import threading
        import queue
        
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def worker():
            try:
                result = func(*args, **kwargs)
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.config.timeout)
        
        if thread.is_alive():
            # Thread is still running - timeout occurred
            raise TimeoutError(f"Operation timed out after {self.config.timeout}s")
        
        # Check for exceptions
        if not exception_queue.empty():
            raise exception_queue.get()
        
        # Get result
        if not result_queue.empty():
            return result_queue.get()
        else:
            raise RuntimeError("Function completed but no result available")
    
    def _record_success(self, execution_time: float):
        """Record successful execution."""
        with self.lock:
            self.metrics.record_success()
            
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_successes += 1
                self.logger.debug(f"Circuit breaker {self.name} - half-open success {self.half_open_successes}/{self.config.success_threshold}")
                
                if self.half_open_successes >= self.config.success_threshold:
                    self._close_circuit()
            
            self.logger.debug(f"Circuit breaker {self.name} - success in {execution_time:.3f}s")
    
    def _record_failure(self):
        """Record failed execution."""
        with self.lock:
            self.metrics.record_failure()
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self._open_circuit()
            elif self.state == CircuitState.CLOSED:
                recent_failures = self.metrics.get_recent_failures()
                if recent_failures >= self.config.failure_threshold:
                    self._open_circuit()
    
    def _record_timeout(self):
        """Record timeout execution."""
        with self.lock:
            self.metrics.record_timeout()
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self._open_circuit()
            elif self.state == CircuitState.CLOSED:
                recent_failures = self.metrics.get_recent_failures()
                if recent_failures >= self.config.failure_threshold:
                    self._open_circuit()
    
    def _update_state(self):
        """Update circuit breaker state based on conditions."""
        current_time = time.time()
        
        if (self.state == CircuitState.OPEN and 
            current_time - self.last_failure_time >= self.config.recovery_timeout):
            self._half_open_circuit()
    
    def _open_circuit(self):
        """Open the circuit breaker."""
        if self.state != CircuitState.OPEN:
            self.logger.warning(f"Circuit breaker {self.name} - OPENING circuit")
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
    
    def _close_circuit(self):
        """Close the circuit breaker."""
        if self.state != CircuitState.CLOSED:
            self.logger.info(f"Circuit breaker {self.name} - CLOSING circuit")
            self.state = CircuitState.CLOSED
            self.half_open_requests = 0
            self.half_open_successes = 0
    
    def _half_open_circuit(self):
        """Put circuit breaker in half-open state."""
        if self.state != CircuitState.HALF_OPEN:
            self.logger.info(f"Circuit breaker {self.name} - entering HALF_OPEN state")
            self.state = CircuitState.HALF_OPEN
            self.half_open_requests = 0
            self.half_open_successes = 0
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        with self.lock:
            self._update_state()
            return self.state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self.lock:
            stats = self.metrics.get_stats()
            stats.update({
                'name': self.name,
                'state': self.state.value,
                'last_failure_time': self.last_failure_time,
                'half_open_requests': self.half_open_requests,
                'half_open_successes': self.half_open_successes,
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'recovery_timeout': self.config.recovery_timeout,
                    'timeout': self.config.timeout,
                }
            })
            return stats
    
    def reset(self):
        """Reset circuit breaker to closed state."""
        with self.lock:
            self.logger.info(f"Circuit breaker {self.name} - RESET to closed state")
            self.state = CircuitState.CLOSED
            self.last_failure_time = 0.0
            self.half_open_requests = 0
            self.half_open_successes = 0
            self.metrics = CircuitBreakerMetrics(self.config.sliding_window)
    
    def force_open(self):
        """Force circuit breaker to open state."""
        with self.lock:
            self.logger.warning(f"Circuit breaker {self.name} - FORCED to open state")
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
    
    def force_close(self):
        """Force circuit breaker to closed state."""
        with self.lock:
            self.logger.info(f"Circuit breaker {self.name} - FORCED to closed state")
            self.state = CircuitState.CLOSED
            self.half_open_requests = 0
            self.half_open_successes = 0


class CircuitBreakerRegistry:
    """Global registry for circuit breakers."""
    
    def __init__(self):
        self.circuit_breakers = {}
        self.lock = threading.RLock()
        self.logger = get_logger("circuit_breaker.registry")
    
    def get_or_create(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        with self.lock:
            if name not in self.circuit_breakers:
                self.circuit_breakers[name] = CircuitBreaker(name, config)
                self.logger.debug(f"Created circuit breaker: {name}")
            
            return self.circuit_breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get existing circuit breaker."""
        with self.lock:
            return self.circuit_breakers.get(name)
    
    def remove(self, name: str) -> bool:
        """Remove circuit breaker from registry."""
        with self.lock:
            if name in self.circuit_breakers:
                del self.circuit_breakers[name]
                self.logger.debug(f"Removed circuit breaker: {name}")
                return True
            return False
    
    def list_circuit_breakers(self) -> list:
        """List all registered circuit breaker names."""
        with self.lock:
            return list(self.circuit_breakers.keys())
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all circuit breakers."""
        with self.lock:
            return {
                name: cb.get_metrics()
                for name, cb in self.circuit_breakers.items()
            }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self.lock:
            for cb in self.circuit_breakers.values():
                cb.reset()
            self.logger.info("Reset all circuit breakers")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all circuit breakers."""
        with self.lock:
            summary = {
                'total_circuits': len(self.circuit_breakers),
                'states': {'closed': 0, 'open': 0, 'half_open': 0},
                'total_requests': 0,
                'total_failures': 0,
            }
            
            for cb in self.circuit_breakers.values():
                metrics = cb.get_metrics()
                summary['states'][metrics['state']] += 1
                summary['total_requests'] += metrics['total_requests']
                summary['total_failures'] += metrics['total_failures'] + metrics['total_timeouts']
            
            return summary


# Global registry instance
_registry = CircuitBreakerRegistry()


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create circuit breaker from global registry."""
    return _registry.get_or_create(name, config)


def circuit_breaker(
    name: Optional[str] = None,
    config: Optional[CircuitBreakerConfig] = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    timeout: float = 30.0
):
    """Decorator to wrap functions with circuit breaker protection."""
    
    def decorator(func: Callable) -> Callable:
        circuit_name = name or f"{func.__module__}.{func.__name__}"
        
        # Create config if not provided
        if config is None:
            cb_config = CircuitBreakerConfig(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                timeout=timeout
            )
        else:
            cb_config = config
        
        circuit = get_circuit_breaker(circuit_name, cb_config)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return circuit.call(func, *args, **kwargs)
        
        # Add circuit breaker methods to wrapped function
        wrapper.circuit_breaker = circuit
        wrapper.get_circuit_metrics = circuit.get_metrics
        wrapper.reset_circuit = circuit.reset
        wrapper.force_open = circuit.force_open
        wrapper.force_close = circuit.force_close
        
        return wrapper
    
    return decorator


# Specialized circuit breakers for watermarking components
def watermark_circuit_breaker(method: str, **kwargs):
    """Circuit breaker specifically for watermarking operations."""
    return circuit_breaker(
        name=f"watermark.{method}",
        failure_threshold=kwargs.get('failure_threshold', 3),
        recovery_timeout=kwargs.get('recovery_timeout', 30.0),
        timeout=kwargs.get('timeout', 15.0)
    )


def detection_circuit_breaker(method: str, **kwargs):
    """Circuit breaker specifically for detection operations.""" 
    return circuit_breaker(
        name=f"detection.{method}",
        failure_threshold=kwargs.get('failure_threshold', 5),
        recovery_timeout=kwargs.get('recovery_timeout', 60.0),
        timeout=kwargs.get('timeout', 10.0)
    )


def model_circuit_breaker(model_name: str, **kwargs):
    """Circuit breaker specifically for model operations."""
    return circuit_breaker(
        name=f"model.{model_name}",
        failure_threshold=kwargs.get('failure_threshold', 2),
        recovery_timeout=kwargs.get('recovery_timeout', 120.0),
        timeout=kwargs.get('timeout', 60.0)
    )


# Health check integration
def get_circuit_breaker_health() -> Dict[str, Any]:
    """Get health status of all circuit breakers."""
    summary = _registry.get_summary()
    
    # Determine overall health
    if summary['states']['open'] > 0:
        health_status = "unhealthy"
    elif summary['states']['half_open'] > 0:
        health_status = "degraded"
    else:
        health_status = "healthy"
    
    return {
        'status': health_status,
        'details': summary,
        'individual_circuits': _registry.get_all_metrics()
    }


__all__ = [
    'CircuitBreaker',
    'CircuitBreakerConfig', 
    'CircuitState',
    'CircuitBreakerRegistry',
    'get_circuit_breaker',
    'circuit_breaker',
    'watermark_circuit_breaker',
    'detection_circuit_breaker', 
    'model_circuit_breaker',
    'get_circuit_breaker_health'
]