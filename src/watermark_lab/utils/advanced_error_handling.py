"""Advanced error handling system with recovery, reporting, and monitoring."""

import time
import traceback
import threading
import functools
from typing import Dict, Any, List, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
from enum import Enum
import logging
import json
import hashlib

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    COMPUTATION = "computation"
    NETWORK = "network"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for errors."""
    
    timestamp: float = field(default_factory=time.time)
    thread_id: int = field(default_factory=threading.get_ident)
    function_name: str = ""
    module_name: str = ""
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'thread_id': self.thread_id,
            'function_name': self.function_name,
            'module_name': self.module_name,
            'user_id': self.user_id,
            'request_id': self.request_id,
            'additional_data': self.additional_data
        }


@dataclass
class ErrorRecord:
    """Comprehensive error record."""
    
    error_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])
    error_type: str = ""
    error_message: str = ""
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.UNKNOWN
    context: ErrorContext = field(default_factory=ErrorContext)
    stack_trace: str = ""
    resolution_attempted: bool = False
    resolution_successful: bool = False
    resolution_method: Optional[str] = None
    occurrence_count: int = 1
    first_occurrence: float = field(default_factory=time.time)
    last_occurrence: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'error_id': self.error_id,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'severity': self.severity.value,
            'category': self.category.value,
            'context': self.context.to_dict(),
            'stack_trace': self.stack_trace,
            'resolution_attempted': self.resolution_attempted,
            'resolution_successful': self.resolution_successful,
            'resolution_method': self.resolution_method,
            'occurrence_count': self.occurrence_count,
            'first_occurrence': self.first_occurrence,
            'last_occurrence': self.last_occurrence
        }


class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def __init__(self, name: str, max_attempts: int = 3):
        self.name = name
        self.max_attempts = max_attempts
        self.attempt_count = 0
    
    def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        """Check if this strategy can recover from the error."""
        return self.attempt_count < self.max_attempts
    
    def attempt_recovery(self, error: Exception, context: ErrorContext) -> bool:
        """Attempt recovery from the error."""
        self.attempt_count += 1
        return self._perform_recovery(error, context)
    
    def _perform_recovery(self, error: Exception, context: ErrorContext) -> bool:
        """Override this method to implement specific recovery logic."""
        return False
    
    def reset(self):
        """Reset the recovery strategy."""
        self.attempt_count = 0


class RetryStrategy(ErrorRecoveryStrategy):
    """Retry strategy with exponential backoff."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        super().__init__("retry", max_attempts)
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        """Check if retry is appropriate for this error."""
        if not super().can_recover(error, context):
            return False
        
        # Don't retry validation errors
        if isinstance(error, (ValueError, TypeError)):
            return False
        
        return True
    
    def _perform_recovery(self, error: Exception, context: ErrorContext) -> bool:
        """Perform retry with exponential backoff."""
        if self.attempt_count > 1:
            delay = min(self.base_delay * (2 ** (self.attempt_count - 1)), self.max_delay)
            time.sleep(delay)
        
        # Return True to indicate retry should be attempted
        return True


class FallbackStrategy(ErrorRecoveryStrategy):
    """Fallback strategy using alternative implementations."""
    
    def __init__(self, fallback_function: Callable):
        super().__init__("fallback", 1)
        self.fallback_function = fallback_function
    
    def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        """Check if fallback is available."""
        return super().can_recover(error, context) and self.fallback_function is not None
    
    def _perform_recovery(self, error: Exception, context: ErrorContext) -> bool:
        """Attempt to use fallback function."""
        try:
            # Note: In real implementation, would need to handle function call
            return True
        except Exception:
            return False


class ResourceCleanupStrategy(ErrorRecoveryStrategy):
    """Strategy for cleaning up resources."""
    
    def __init__(self):
        super().__init__("resource_cleanup", 1)
    
    def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        """Check if this is a resource-related error."""
        error_message = str(error).lower()
        resource_indicators = ['memory', 'disk', 'connection', 'file', 'timeout']
        
        has_resource_error = any(indicator in error_message for indicator in resource_indicators)
        return super().can_recover(error, context) and has_resource_error
    
    def _perform_recovery(self, error: Exception, context: ErrorContext) -> bool:
        """Attempt resource cleanup."""
        try:
            # Simulate garbage collection
            import gc
            gc.collect()
            
            # Close any open resources if available
            # In real implementation, would have resource tracking
            
            return True
        except Exception:
            return False


class AdvancedErrorHandler:
    """Advanced error handling system with recovery and monitoring."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # Error storage and tracking
        self.error_records: Dict[str, ErrorRecord] = {}
        self.error_patterns = defaultdict(list)
        self.recent_errors = deque(maxlen=1000)
        
        # Recovery strategies
        self.recovery_strategies: List[ErrorRecoveryStrategy] = [
            RetryStrategy(max_attempts=3),
            FallbackStrategy(None),  # Would be configured per use case
            ResourceCleanupStrategy()
        ]
        
        # Monitoring and alerting
        self.error_counts = defaultdict(int)
        self.alert_thresholds = {
            ErrorSeverity.LOW: 50,
            ErrorSeverity.MEDIUM: 20,
            ErrorSeverity.HIGH: 10,
            ErrorSeverity.CRITICAL: 1
        }
        self.alert_callbacks: List[Callable[[ErrorRecord], None]] = []
        
        # Configuration
        self.auto_recovery_enabled = True
        self.error_reporting_enabled = True
        self.monitoring_enabled = True
        
        # Thread safety
        self._lock = threading.RLock()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup error handling logging."""
        logger = logging.getLogger("advanced_error_handler")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def handle_error(
        self, 
        error: Exception, 
        context: Optional[ErrorContext] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        attempt_recovery: bool = True
    ) -> Optional[Any]:
        """Comprehensive error handling with recovery and reporting."""
        
        with self._lock:
            # Create context if not provided
            if context is None:
                context = ErrorContext(
                    function_name=self._get_calling_function(),
                    module_name=self._get_calling_module()
                )
            
            # Create or update error record
            error_record = self._create_error_record(error, context, severity, category)
            
            # Log the error
            self._log_error(error_record)
            
            # Attempt recovery if enabled and appropriate
            recovery_result = None
            if attempt_recovery and self.auto_recovery_enabled:
                recovery_result = self._attempt_recovery(error, error_record, context)
            
            # Monitor and alert if necessary
            if self.monitoring_enabled:
                self._monitor_error(error_record)
            
            # Report error if enabled
            if self.error_reporting_enabled:
                self._report_error(error_record)
            
            return recovery_result
    
    def _create_error_record(
        self, 
        error: Exception, 
        context: ErrorContext,
        severity: ErrorSeverity,
        category: ErrorCategory
    ) -> ErrorRecord:
        """Create or update error record."""
        
        error_signature = self._compute_error_signature(error)
        
        if error_signature in self.error_records:
            # Update existing record
            record = self.error_records[error_signature]
            record.occurrence_count += 1
            record.last_occurrence = time.time()
        else:
            # Create new record
            record = ErrorRecord(
                error_type=type(error).__name__,
                error_message=str(error),
                severity=severity,
                category=category,
                context=context,
                stack_trace=traceback.format_exc()
            )
            self.error_records[error_signature] = record
        
        # Add to recent errors
        self.recent_errors.append(record)
        
        # Update error patterns
        self.error_patterns[category].append(record)
        
        return record
    
    def _compute_error_signature(self, error: Exception) -> str:
        """Compute unique signature for error deduplication."""
        error_type = type(error).__name__
        error_message = str(error)[:100]  # Truncate long messages
        
        signature_data = f"{error_type}:{error_message}"
        return hashlib.md5(signature_data.encode()).hexdigest()[:16]
    
    def _get_calling_function(self) -> str:
        """Get the name of the calling function."""
        try:
            frame = traceback.extract_stack()[-4]  # Go back to the actual caller
            return frame.name
        except (IndexError, AttributeError):
            return "unknown"
    
    def _get_calling_module(self) -> str:
        """Get the name of the calling module."""
        try:
            frame = traceback.extract_stack()[-4]
            return frame.filename.split('/')[-1]
        except (IndexError, AttributeError):
            return "unknown"
    
    def _log_error(self, record: ErrorRecord):
        """Log error with appropriate level."""
        
        log_message = (
            f"[{record.error_id}] {record.error_type}: {record.error_message} "
            f"(severity: {record.severity.value}, count: {record.occurrence_count})"
        )
        
        if record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif record.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _attempt_recovery(
        self, 
        error: Exception, 
        record: ErrorRecord, 
        context: ErrorContext
    ) -> Optional[Any]:
        """Attempt error recovery using available strategies."""
        
        for strategy in self.recovery_strategies:
            if strategy.can_recover(error, context):
                self.logger.info(f"Attempting recovery using {strategy.name} strategy")
                
                try:
                    if strategy.attempt_recovery(error, context):
                        record.resolution_attempted = True
                        record.resolution_successful = True
                        record.resolution_method = strategy.name
                        
                        self.logger.info(f"Recovery successful using {strategy.name}")
                        return True
                    
                except Exception as recovery_error:
                    self.logger.warning(f"Recovery strategy {strategy.name} failed: {recovery_error}")
                    continue
        
        record.resolution_attempted = True
        record.resolution_successful = False
        self.logger.warning(f"All recovery strategies failed for error {record.error_id}")
        return None
    
    def _monitor_error(self, record: ErrorRecord):
        """Monitor error for patterns and thresholds."""
        
        # Update error counts
        self.error_counts[record.severity] += 1
        
        # Check alert thresholds
        threshold = self.alert_thresholds.get(record.severity, float('inf'))
        
        if self.error_counts[record.severity] >= threshold:
            self._trigger_alert(record)
            # Reset counter after alert
            self.error_counts[record.severity] = 0
    
    def _trigger_alert(self, record: ErrorRecord):
        """Trigger alert for critical errors."""
        
        self.logger.critical(f"ALERT: Error threshold exceeded for {record.severity.value}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(record)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def _report_error(self, record: ErrorRecord):
        """Report error to external systems."""
        
        # In a real implementation, would send to error tracking service
        # (Sentry, Rollbar, etc.)
        
        self.logger.debug(f"Error reported: {record.error_id}")
    
    def add_recovery_strategy(self, strategy: ErrorRecoveryStrategy):
        """Add custom recovery strategy."""
        with self._lock:
            self.recovery_strategies.append(strategy)
    
    def add_alert_callback(self, callback: Callable[[ErrorRecord], None]):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def set_alert_threshold(self, severity: ErrorSeverity, threshold: int):
        """Set alert threshold for severity level."""
        self.alert_thresholds[severity] = threshold
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        
        with self._lock:
            total_errors = len(self.error_records)
            
            severity_counts = defaultdict(int)
            category_counts = defaultdict(int)
            resolution_success_rate = 0
            
            for record in self.error_records.values():
                severity_counts[record.severity.value] += record.occurrence_count
                category_counts[record.category.value] += record.occurrence_count
                
                if record.resolution_attempted:
                    if record.resolution_successful:
                        resolution_success_rate += 1
            
            resolution_success_rate = (
                resolution_success_rate / max(1, sum(1 for r in self.error_records.values() if r.resolution_attempted))
            )
            
            return {
                'total_unique_errors': total_errors,
                'total_error_occurrences': sum(r.occurrence_count for r in self.error_records.values()),
                'severity_breakdown': dict(severity_counts),
                'category_breakdown': dict(category_counts),
                'resolution_success_rate': resolution_success_rate,
                'recent_errors_count': len(self.recent_errors),
                'active_recovery_strategies': len(self.recovery_strategies),
                'alert_callbacks': len(self.alert_callbacks)
            }
    
    def get_error_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze error patterns."""
        
        patterns = {}
        
        with self._lock:
            for category, records in self.error_patterns.items():
                if records:
                    patterns[category.value] = [
                        {
                            'error_type': record.error_type,
                            'frequency': record.occurrence_count,
                            'first_seen': record.first_occurrence,
                            'last_seen': record.last_occurrence
                        }
                        for record in records[-10:]  # Last 10 for each category
                    ]
        
        return patterns
    
    def clear_error_history(self):
        """Clear error history (use with caution)."""
        with self._lock:
            self.error_records.clear()
            self.error_patterns.clear()
            self.recent_errors.clear()
            self.error_counts.clear()
        
        self.logger.info("Error history cleared")


# Decorators for automatic error handling
def handle_errors(
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    attempt_recovery: bool = True,
    fallback_result: Any = None
):
    """Decorator for automatic error handling."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    function_name=func.__name__,
                    module_name=func.__module__,
                    additional_data={'args': str(args)[:100], 'kwargs': str(kwargs)[:100]}
                )
                
                recovery_result = error_handler.handle_error(
                    e, context, severity, category, attempt_recovery
                )
                
                if recovery_result is not None:
                    # Recovery successful, retry the function
                    return func(*args, **kwargs)
                else:
                    # Recovery failed, return fallback or re-raise
                    if fallback_result is not None:
                        return fallback_result
                    raise
        
        return wrapper
    return decorator


@contextmanager
def error_context(
    operation_name: str,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    **additional_data
):
    """Context manager for error handling."""
    
    context = ErrorContext(
        function_name=operation_name,
        user_id=user_id,
        request_id=request_id,
        additional_data=additional_data
    )
    
    try:
        yield context
    except Exception as e:
        error_handler.handle_error(e, context)
        raise


# Global error handler instance
error_handler = AdvancedErrorHandler()


# Convenience functions
def handle_critical_error(error: Exception, context: Optional[ErrorContext] = None):
    """Handle critical error."""
    return error_handler.handle_error(error, context, ErrorSeverity.CRITICAL)


def handle_validation_error(error: Exception, context: Optional[ErrorContext] = None):
    """Handle validation error."""
    return error_handler.handle_error(error, context, ErrorSeverity.MEDIUM, ErrorCategory.VALIDATION)


def handle_network_error(error: Exception, context: Optional[ErrorContext] = None):
    """Handle network error."""
    return error_handler.handle_error(error, context, ErrorSeverity.HIGH, ErrorCategory.NETWORK)


def get_error_summary() -> Dict[str, Any]:
    """Get error summary."""
    return error_handler.get_error_statistics()


__all__ = [
    'AdvancedErrorHandler',
    'ErrorContext',
    'ErrorRecord',
    'ErrorSeverity',
    'ErrorCategory',
    'ErrorRecoveryStrategy',
    'RetryStrategy',
    'FallbackStrategy',
    'ResourceCleanupStrategy',
    'handle_errors',
    'error_context',
    'error_handler',
    'handle_critical_error',
    'handle_validation_error',
    'handle_network_error',
    'get_error_summary'
]