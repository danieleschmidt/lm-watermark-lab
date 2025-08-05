"""Retry and resilience utilities."""

import time
import random
import functools
from typing import Callable, Any, Type, Tuple, Optional, Union
from dataclasses import dataclass

from .exceptions import WatermarkLabError, TimeoutError
from .logging import get_logger


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    jitter: bool = True
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    stop_exceptions: Tuple[Type[Exception], ...] = ()


class CircuitBreakerError(WatermarkLabError):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        
        self.logger = get_logger("circuit_breaker")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
                self.logger.info("Circuit breaker moving to half-open state")
            else:
                raise CircuitBreakerError("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        
        if self.state == "half-open":
            self.state = "closed"
            self.logger.info("Circuit breaker reset to closed state")
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )


def retry(
    config: Optional[RetryConfig] = None,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_backoff: bool = True,
    jitter: bool = True,
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    stop_exceptions: Tuple[Type[Exception], ...] = ()
) -> Callable:
    """Retry decorator with configurable behavior."""
    
    if config is None:
        config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            exponential_backoff=exponential_backoff,
            jitter=jitter,
            retry_exceptions=retry_exceptions,
            stop_exceptions=stop_exceptions
        )
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return retry_call(func, config, *args, **kwargs)
        return wrapper
    
    return decorator


def retry_call(
    func: Callable,
    config: RetryConfig,
    *args,
    **kwargs
) -> Any:
    """Execute function with retry logic."""
    logger = get_logger(f"retry.{func.__name__}")
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            logger.debug(f"Attempt {attempt + 1}/{config.max_attempts}")
            return func(*args, **kwargs)
            
        except config.stop_exceptions as e:
            logger.info(f"Stopping retry due to stop exception: {e}")
            raise
            
        except config.retry_exceptions as e:
            last_exception = e
            
            if attempt == config.max_attempts - 1:
                logger.error(f"All {config.max_attempts} attempts failed")
                break
            
            delay = calculate_delay(
                attempt,
                config.base_delay,
                config.max_delay,
                config.exponential_backoff,
                config.jitter
            )
            
            logger.warning(
                f"Attempt {attempt + 1} failed: {e}. "
                f"Retrying in {delay:.2f}s"
            )
            
            time.sleep(delay)
    
    # All attempts failed
    if last_exception:
        raise last_exception
    else:
        raise WatermarkLabError("All retry attempts failed")


def calculate_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    exponential_backoff: bool,
    jitter: bool
) -> float:
    """Calculate delay for retry attempt."""
    if exponential_backoff:
        delay = base_delay * (2 ** attempt)
    else:
        delay = base_delay
    
    # Apply maximum delay limit
    delay = min(delay, max_delay)
    
    # Add jitter to avoid thundering herd
    if jitter:
        jitter_amount = delay * 0.1 * random.random()
        delay += jitter_amount
    
    return delay


class Timeout:
    """Timeout context manager and decorator."""
    
    def __init__(self, seconds: float, error_message: str = "Operation timed out"):
        self.seconds = seconds
        self.error_message = error_message
        self.logger = get_logger("timeout")
    
    def __enter__(self):
        """Start timeout."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(self.error_message)
        
        # Set up signal handler (Unix only)
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.seconds))
        except (AttributeError, ValueError):
            # Windows or invalid timeout - use threading approach
            import threading
            
            def timeout_thread():
                time.sleep(self.seconds)
                # This is a simplified approach - in production,
                # use more sophisticated timeout mechanisms
            
            self.timeout_thread = threading.Thread(target=timeout_thread)
            self.timeout_thread.daemon = True
            self.timeout_thread.start()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cancel timeout."""
        import signal
        
        try:
            signal.alarm(0)  # Cancel alarm
        except AttributeError:
            # Windows - threading approach cleanup
            pass
    
    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


class RateLimiter:
    """Rate limiting utility."""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.logger = get_logger("rate_limiter")
    
    def acquire(self, block: bool = True) -> bool:
        """Acquire permission to make a call."""
        now = time.time()
        
        # Remove old calls outside the window
        self.calls = [call_time for call_time in self.calls 
                     if now - call_time < self.time_window]
        
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        
        if not block:
            return False
        
        # Calculate wait time
        oldest_call = min(self.calls)
        wait_time = self.time_window - (now - oldest_call)
        
        if wait_time > 0:
            self.logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
            time.sleep(wait_time)
            return self.acquire(block=False)
        
        return True
    
    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.acquire()
            return func(*args, **kwargs)
        return wrapper


def with_fallback(fallback_func: Callable, exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """Decorator to provide fallback function on failure."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger = get_logger(f"fallback.{func.__name__}")
                logger.warning(f"Primary function failed: {e}. Using fallback.")
                return fallback_func(*args, **kwargs)
        return wrapper
    return decorator


def graceful_degradation(
    degraded_func: Callable,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    log_degradation: bool = True
):
    """Decorator for graceful degradation when primary function fails."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if log_degradation:
                    logger = get_logger(f"degradation.{func.__name__}")
                    logger.warning(
                        f"Primary function degraded due to: {e}. "
                        f"Using degraded functionality."
                    )
                return degraded_func(*args, **kwargs)
        return wrapper
    return decorator


# Predefined configurations
WATERMARK_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=10.0,
    exponential_backoff=True,
    jitter=True,
    retry_exceptions=(WatermarkLabError, ConnectionError, TimeoutError),
    stop_exceptions=(ValueError, TypeError)
)

DETECTION_RETRY_CONFIG = RetryConfig(
    max_attempts=2,
    base_delay=0.5,
    max_delay=5.0,
    exponential_backoff=False,
    jitter=True,
    retry_exceptions=(WatermarkLabError, ConnectionError),
    stop_exceptions=(ValueError, TypeError)
)

BENCHMARK_RETRY_CONFIG = RetryConfig(
    max_attempts=2,
    base_delay=2.0,
    max_delay=30.0,
    exponential_backoff=True,
    jitter=True,
    retry_exceptions=(WatermarkLabError, ConnectionError, TimeoutError),
    stop_exceptions=(ValueError, TypeError, KeyboardInterrupt)
)