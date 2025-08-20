"""Robust monitoring and health check implementation."""

import time
import logging
import threading
import json
import psutil
import os
from typing import Dict, List, Any, Optional, Callable, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics

class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    response_time: Optional[float] = None

@dataclass
class Metric:
    """Metric data structure."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)

class HealthCheck:
    """Base class for health checks."""
    
    def __init__(self, name: str, timeout: float = 10.0):
        self.name = name
        self.timeout = timeout
    
    def check(self) -> HealthCheckResult:
        """Perform health check."""
        start_time = time.time()
        
        try:
            result = self._check_implementation()
            result.response_time = time.time() - start_time
            return result
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                response_time=time.time() - start_time
            )
    
    def _check_implementation(self) -> HealthCheckResult:
        """Override this method in subclasses."""
        return HealthCheckResult(
            name=self.name,
            status=HealthStatus.HEALTHY,
            message="OK"
        )

class SystemResourcesCheck(HealthCheck):
    """System resources health check."""
    
    def __init__(self, cpu_threshold: float = 80.0, memory_threshold: float = 85.0):
        super().__init__("system_resources")
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
    
    def _check_implementation(self) -> HealthCheckResult:
        """Check system resources."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        details = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "disk_percent": disk_percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_free_gb": disk.free / (1024**3)
        }
        
        # Determine status
        if cpu_percent > self.cpu_threshold or memory_percent > self.memory_threshold:
            status = HealthStatus.CRITICAL
            message = f"High resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
        elif cpu_percent > self.cpu_threshold * 0.8 or memory_percent > self.memory_threshold * 0.8:
            status = HealthStatus.WARNING
            message = f"Elevated resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Resources OK: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
        
        return HealthCheckResult(
            name=self.name,
            status=status,
            message=message,
            details=details
        )

class DependencyCheck(HealthCheck):
    """Check availability of dependencies."""
    
    def __init__(self, dependencies: List[str]):
        super().__init__("dependencies")
        self.dependencies = dependencies
    
    def _check_implementation(self) -> HealthCheckResult:
        """Check if dependencies are available."""
        available = {}
        missing = []
        
        for dep in self.dependencies:
            try:
                __import__(dep)
                available[dep] = True
            except ImportError:
                available[dep] = False
                missing.append(dep)
        
        if missing:
            status = HealthStatus.WARNING
            message = f"Missing dependencies: {', '.join(missing)}"
        else:
            status = HealthStatus.HEALTHY
            message = "All dependencies available"
        
        return HealthCheckResult(
            name=self.name,
            status=status,
            message=message,
            details={
                "available": available,
                "missing": missing
            }
        )

class WatermarkModelCheck(HealthCheck):
    """Check watermark model availability."""
    
    def __init__(self, model_name: str = "gpt2"):
        super().__init__("watermark_model")
        self.model_name = model_name
    
    def _check_implementation(self) -> HealthCheckResult:
        """Check if watermark model can be loaded."""
        try:
            # Try to import and create a basic watermark
            from ..core.factory import WatermarkFactory
            
            # This is a lightweight check - just verify factory works
            factory_available = WatermarkFactory is not None
            
            if factory_available:
                status = HealthStatus.HEALTHY
                message = f"Watermark factory available for {self.model_name}"
            else:
                status = HealthStatus.CRITICAL
                message = "Watermark factory not available"
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details={
                    "model_name": self.model_name,
                    "factory_available": factory_available
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Model check failed: {str(e)}",
                details={"error": str(e)}
            )

class RobustMonitor:
    """Comprehensive monitoring system."""
    
    def __init__(self, check_interval: float = 30.0):
        """Initialize monitoring system."""
        self.check_interval = check_interval
        self.health_checks: List[HealthCheck] = []
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger("monitor")
        
        # Performance tracking
        self._request_times: deque = deque(maxlen=1000)
        self._error_counts: Dict[str, int] = defaultdict(int)
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self) -> None:
        """Register default health checks."""
        # System resources
        self.add_health_check(SystemResourcesCheck())
        
        # Dependencies
        self.add_health_check(DependencyCheck([
            "torch", "transformers", "numpy", "scipy"
        ]))
        
        # Watermark model
        self.add_health_check(WatermarkModelCheck())
    
    def add_health_check(self, check: HealthCheck) -> None:
        """Add a health check."""
        self.health_checks.append(check)
        self.logger.info(f"Added health check: {check.name}")
    
    def remove_health_check(self, name: str) -> bool:
        """Remove a health check by name."""
        for i, check in enumerate(self.health_checks):
            if check.name == name:
                del self.health_checks[i]
                self.logger.info(f"Removed health check: {name}")
                return True
        return False
    
    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE, 
                     tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags or {}
        )
        
        self.metrics[name].append(metric)
    
    def record_request_time(self, duration: float) -> None:
        """Record request processing time."""
        self._request_times.append(duration)
        self.record_metric("request_duration", duration, MetricType.TIMER)
    
    def record_error(self, error_type: str) -> None:
        """Record an error occurrence."""
        self._error_counts[error_type] += 1
        self.record_metric(f"error_{error_type}", 1, MetricType.COUNTER)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        results = []
        overall_status = HealthStatus.HEALTHY
        
        for check in self.health_checks:
            try:
                result = check.check()
                results.append({
                    "name": result.name,
                    "status": result.status.value,
                    "message": result.message,
                    "timestamp": result.timestamp,
                    "response_time": result.response_time,
                    "details": result.details
                })
                
                # Determine overall status
                if result.status == HealthStatus.CRITICAL:
                    overall_status = HealthStatus.CRITICAL
                elif result.status == HealthStatus.WARNING and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.WARNING
                    
            except Exception as e:
                self.logger.error(f"Health check {check.name} failed: {e}")
                results.append({
                    "name": check.name,
                    "status": HealthStatus.CRITICAL.value,
                    "message": f"Check failed: {str(e)}",
                    "timestamp": time.time(),
                    "response_time": None,
                    "details": {"error": str(e)}
                })
                overall_status = HealthStatus.CRITICAL
        
        return {
            "overall_status": overall_status.value,
            "timestamp": time.time(),
            "checks": results,
            "summary": {
                "total_checks": len(results),
                "healthy": sum(1 for r in results if r["status"] == "healthy"),
                "warnings": sum(1 for r in results if r["status"] == "warning"),
                "critical": sum(1 for r in results if r["status"] == "critical")
            }
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        summary = {}
        
        # Request performance
        if self._request_times:
            summary["request_performance"] = {
                "avg_duration": statistics.mean(self._request_times),
                "median_duration": statistics.median(self._request_times),
                "min_duration": min(self._request_times),
                "max_duration": max(self._request_times),
                "total_requests": len(self._request_times)
            }
        
        # Error counts
        summary["errors"] = dict(self._error_counts)
        
        # Metric counts
        summary["metrics"] = {
            name: len(values) for name, values in self.metrics.items()
        }
        
        # System info
        try:
            summary["system"] = {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "boot_time": psutil.boot_time(),
                "uptime_hours": (time.time() - psutil.boot_time()) / 3600
            }
        except Exception as e:
            self.logger.warning(f"Could not get system info: {e}")
        
        return summary
    
    def start_monitoring(self) -> None:
        """Start the monitoring thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        self.logger.info("Started monitoring thread")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        self.logger.info("Stopped monitoring thread")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                # Perform health checks
                status = self.get_health_status()
                
                # Log critical issues
                if status["overall_status"] == "critical":
                    critical_checks = [c for c in status["checks"] if c["status"] == "critical"]
                    self.logger.critical(f"Critical health issues detected: {len(critical_checks)} checks failed")
                
                # Record system metrics
                try:
                    cpu_percent = psutil.cpu_percent()
                    memory_percent = psutil.virtual_memory().percent
                    
                    self.record_metric("system_cpu_percent", cpu_percent)
                    self.record_metric("system_memory_percent", memory_percent)
                except Exception as e:
                    self.logger.warning(f"Could not record system metrics: {e}")
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(min(self.check_interval, 60))  # Don't spam errors
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        data = {
            "timestamp": time.time(),
            "health_status": self.get_health_status(),
            "metrics_summary": self.get_metrics_summary(),
            "raw_metrics": {
                name: [
                    {
                        "value": m.value,
                        "timestamp": m.timestamp,
                        "type": m.metric_type.value,
                        "tags": m.tags
                    }
                    for m in values
                ]
                for name, values in self.metrics.items()
            }
        }
        
        if format.lower() == "json":
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

# Global monitor instance
_monitor = None

def get_monitor() -> RobustMonitor:
    """Get global monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = RobustMonitor()
    return _monitor

def health_check_decorator(func: Callable) -> Callable:
    """Decorator to monitor function execution."""
    def wrapper(*args, **kwargs):
        monitor = get_monitor()
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            monitor.record_request_time(duration)
            return result
            
        except Exception as e:
            monitor.record_error(type(e).__name__)
            raise
    
    return wrapper