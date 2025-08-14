"""Comprehensive monitoring system with metrics, health checks, and alerting."""

import time
import threading
import psutil
import json
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from contextlib import contextmanager
from enum import Enum
import logging
import hashlib
from pathlib import Path

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"  
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricValue:
    """Single metric value with metadata."""
    
    value: Union[int, float]
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'value': self.value,
            'timestamp': self.timestamp,
            'labels': self.labels
        }


@dataclass
class Metric:
    """Metric definition with history."""
    
    name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    labels: List[str] = field(default_factory=list)
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_value(self, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """Add a new value to the metric."""
        metric_value = MetricValue(value=value, labels=labels or {})
        self.values.append(metric_value)
    
    def get_current_value(self) -> Optional[MetricValue]:
        """Get the most recent value."""
        return self.values[-1] if self.values else None
    
    def get_values_in_range(self, start_time: float, end_time: float) -> List[MetricValue]:
        """Get values within time range."""
        return [
            v for v in self.values 
            if start_time <= v.timestamp <= end_time
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'type': self.metric_type.value,
            'description': self.description,
            'unit': self.unit,
            'labels': self.labels,
            'current_value': self.get_current_value().to_dict() if self.get_current_value() else None,
            'value_count': len(self.values)
        }


@dataclass
class HealthCheck:
    """Health check definition."""
    
    name: str
    check_function: Callable[[], bool]
    description: str = ""
    timeout: float = 5.0
    critical: bool = False
    enabled: bool = True
    last_result: Optional[bool] = None
    last_check_time: Optional[float] = None
    failure_count: int = 0
    
    def execute(self) -> bool:
        """Execute the health check."""
        if not self.enabled:
            return True
        
        start_time = time.time()
        
        try:
            # Execute with timeout (simplified - real implementation would use threading)
            result = self.check_function()
            self.last_result = result
            self.last_check_time = time.time()
            
            if result:
                self.failure_count = 0
            else:
                self.failure_count += 1
            
            return result
            
        except Exception as e:
            self.last_result = False
            self.last_check_time = time.time()
            self.failure_count += 1
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'critical': self.critical,
            'enabled': self.enabled,
            'last_result': self.last_result,
            'last_check_time': self.last_check_time,
            'failure_count': self.failure_count,
            'timeout': self.timeout
        }


@dataclass
class Alert:
    """Alert definition and state."""
    
    id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])
    name: str = ""
    condition: str = ""
    severity: AlertSeverity = AlertSeverity.WARNING
    message: str = ""
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_resolved(self) -> bool:
        """Check if alert is resolved."""
        return self.resolved_at is not None
    
    def resolve(self):
        """Mark alert as resolved."""
        self.resolved_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self):
        self.logger = logging.getLogger("system_monitor")
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=1)
        except Exception:
            return 0.0
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total': memory.total / (1024**3),  # GB
                'available': memory.available / (1024**3),  # GB
                'used': memory.used / (1024**3),  # GB
                'percentage': memory.percent
            }
        except Exception:
            return {'total': 0, 'available': 0, 'used': 0, 'percentage': 0}
    
    def get_disk_usage(self, path: str = '/') -> Dict[str, float]:
        """Get disk usage information."""
        try:
            disk = psutil.disk_usage(path)
            return {
                'total': disk.total / (1024**3),  # GB
                'used': disk.used / (1024**3),  # GB
                'free': disk.free / (1024**3),  # GB
                'percentage': (disk.used / disk.total) * 100
            }
        except Exception:
            return {'total': 0, 'used': 0, 'free': 0, 'percentage': 0}
    
    def get_network_stats(self) -> Dict[str, int]:
        """Get network statistics."""
        try:
            net_io = psutil.net_io_counters()
            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_received': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_received': net_io.packets_recv
            }
        except Exception:
            return {'bytes_sent': 0, 'bytes_received': 0, 'packets_sent': 0, 'packets_received': 0}
    
    def get_process_info(self) -> Dict[str, Any]:
        """Get current process information."""
        try:
            process = psutil.Process()
            return {
                'pid': process.pid,
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'memory_info': process.memory_info()._asdict(),
                'num_threads': process.num_threads(),
                'create_time': process.create_time()
            }
        except Exception:
            return {}


class ComprehensiveMonitor:
    """Comprehensive monitoring system."""
    
    def __init__(self, collection_interval: float = 10.0):
        self.collection_interval = collection_interval
        self.logger = self._setup_logging()
        
        # Metrics storage
        self.metrics: Dict[str, Metric] = {}
        
        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Alerts
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Alert rules and callbacks
        self.alert_rules: List[Dict[str, Any]] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # System monitoring
        self.system_monitor = SystemMonitor()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize default metrics and health checks
        self._setup_default_monitoring()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup monitoring logging."""
        logger = logging.getLogger("comprehensive_monitor")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def _setup_default_monitoring(self):
        """Setup default metrics and health checks."""
        
        # Default metrics
        self.register_metric("system_cpu_usage", MetricType.GAUGE, "System CPU usage percentage", "%")
        self.register_metric("system_memory_usage", MetricType.GAUGE, "System memory usage percentage", "%")
        self.register_metric("system_disk_usage", MetricType.GAUGE, "System disk usage percentage", "%")
        self.register_metric("process_memory_usage", MetricType.GAUGE, "Process memory usage", "MB")
        self.register_metric("request_count", MetricType.COUNTER, "Total number of requests")
        self.register_metric("error_count", MetricType.COUNTER, "Total number of errors")
        self.register_metric("response_time", MetricType.HISTOGRAM, "Response time distribution", "ms")
        
        # Default health checks
        self.register_health_check("system_health", self._check_system_health, "Overall system health")
        self.register_health_check("memory_health", self._check_memory_health, "Memory usage health")
        self.register_health_check("disk_health", self._check_disk_health, "Disk usage health")
        
        # Default alert rules
        self.add_alert_rule("high_cpu_usage", "system_cpu_usage > 90", AlertSeverity.WARNING, "High CPU usage detected")
        self.add_alert_rule("high_memory_usage", "system_memory_usage > 90", AlertSeverity.WARNING, "High memory usage detected")
        self.add_alert_rule("high_disk_usage", "system_disk_usage > 85", AlertSeverity.ERROR, "High disk usage detected")
        self.add_alert_rule("critical_memory", "system_memory_usage > 95", AlertSeverity.CRITICAL, "Critical memory usage")
    
    def register_metric(
        self, 
        name: str, 
        metric_type: MetricType, 
        description: str = "", 
        unit: str = "",
        labels: Optional[List[str]] = None
    ):
        """Register a new metric."""
        
        with self._lock:
            if name in self.metrics:
                self.logger.warning(f"Metric {name} already exists, updating")
            
            self.metrics[name] = Metric(
                name=name,
                metric_type=metric_type,
                description=description,
                unit=unit,
                labels=labels or []
            )
            
            self.logger.debug(f"Registered metric: {name}")
    
    def record_metric(
        self, 
        name: str, 
        value: Union[int, float], 
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a metric value."""
        
        with self._lock:
            if name not in self.metrics:
                # Auto-register as gauge metric
                self.register_metric(name, MetricType.GAUGE, f"Auto-registered metric: {name}")
            
            self.metrics[name].add_value(value, labels)
    
    def register_health_check(
        self,
        name: str,
        check_function: Callable[[], bool],
        description: str = "",
        timeout: float = 5.0,
        critical: bool = False
    ):
        """Register a health check."""
        
        with self._lock:
            self.health_checks[name] = HealthCheck(
                name=name,
                check_function=check_function,
                description=description,
                timeout=timeout,
                critical=critical
            )
            
            self.logger.debug(f"Registered health check: {name}")
    
    def add_alert_rule(
        self,
        name: str,
        condition: str,
        severity: AlertSeverity,
        message: str
    ):
        """Add an alert rule."""
        
        rule = {
            'name': name,
            'condition': condition,
            'severity': severity,
            'message': message,
            'enabled': True
        }
        
        self.alert_rules.append(rule)
        self.logger.debug(f"Added alert rule: {name}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start the monitoring system."""
        
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info(f"Started monitoring with {self.collection_interval}s interval")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("Stopped monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Run health checks
                self._run_health_checks()
                
                # Evaluate alert rules
                self._evaluate_alert_rules()
                
                # Sleep until next collection
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system metrics."""
        
        try:
            # CPU usage
            cpu_usage = self.system_monitor.get_cpu_usage()
            self.record_metric("system_cpu_usage", cpu_usage)
            
            # Memory usage
            memory_info = self.system_monitor.get_memory_usage()
            self.record_metric("system_memory_usage", memory_info['percentage'])
            
            # Disk usage
            disk_info = self.system_monitor.get_disk_usage()
            self.record_metric("system_disk_usage", disk_info['percentage'])
            
            # Process memory
            process_info = self.system_monitor.get_process_info()
            if 'memory_info' in process_info:
                memory_mb = process_info['memory_info'].get('rss', 0) / (1024 * 1024)
                self.record_metric("process_memory_usage", memory_mb)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _run_health_checks(self):
        """Run all health checks."""
        
        with self._lock:
            for name, health_check in self.health_checks.items():
                try:
                    result = health_check.execute()
                    
                    if not result and health_check.critical:
                        # Create critical alert for failed critical health check
                        self._create_alert(
                            name=f"critical_health_check_{name}",
                            condition=f"health_check_{name} == false",
                            severity=AlertSeverity.CRITICAL,
                            message=f"Critical health check failed: {name}",
                            metadata={'health_check': name, 'failure_count': health_check.failure_count}
                        )
                    
                except Exception as e:
                    self.logger.error(f"Health check {name} failed: {e}")
    
    def _evaluate_alert_rules(self):
        """Evaluate all alert rules."""
        
        for rule in self.alert_rules:
            if not rule.get('enabled', True):
                continue
            
            try:
                condition_met = self._evaluate_condition(rule['condition'])
                
                if condition_met:
                    alert_id = f"{rule['name']}_{int(time.time())}"
                    
                    # Check if similar alert already active
                    similar_active = any(
                        alert.name == rule['name'] and not alert.is_resolved
                        for alert in self.active_alerts.values()
                    )
                    
                    if not similar_active:
                        self._create_alert(
                            name=rule['name'],
                            condition=rule['condition'],
                            severity=rule['severity'],
                            message=rule['message']
                        )
                
            except Exception as e:
                self.logger.error(f"Error evaluating alert rule {rule['name']}: {e}")
    
    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate alert condition (simplified implementation)."""
        
        try:
            # Simple condition evaluation for metric thresholds
            # Format: "metric_name operator value"
            parts = condition.split()
            
            if len(parts) >= 3:
                metric_name = parts[0]
                operator = parts[1]
                threshold = float(parts[2])
                
                if metric_name in self.metrics:
                    current_value = self.metrics[metric_name].get_current_value()
                    if current_value:
                        value = current_value.value
                        
                        if operator == '>':
                            return value > threshold
                        elif operator == '<':
                            return value < threshold
                        elif operator == '>=':
                            return value >= threshold
                        elif operator == '<=':
                            return value <= threshold
                        elif operator == '==':
                            return value == threshold
                        elif operator == '!=':
                            return value != threshold
            
            return False
            
        except Exception:
            return False
    
    def _create_alert(
        self,
        name: str,
        condition: str,
        severity: AlertSeverity,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Create a new alert."""
        
        alert = Alert(
            name=name,
            condition=condition,
            severity=severity,
            message=message,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.active_alerts[alert.id] = alert
            self.alert_history.append(alert)
        
        # Trigger alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
        
        self.logger.warning(f"ALERT [{severity.value}]: {message}")
        return alert
    
    def resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolve()
                del self.active_alerts[alert_id]
                self.logger.info(f"Resolved alert: {alert.name}")
                return True
            
            return False
    
    # Default health check implementations
    def _check_system_health(self) -> bool:
        """Check overall system health."""
        try:
            cpu_usage = self.system_monitor.get_cpu_usage()
            memory_info = self.system_monitor.get_memory_usage()
            
            return cpu_usage < 95 and memory_info['percentage'] < 95
        except Exception:
            return False
    
    def _check_memory_health(self) -> bool:
        """Check memory health."""
        try:
            memory_info = self.system_monitor.get_memory_usage()
            return memory_info['percentage'] < 90
        except Exception:
            return False
    
    def _check_disk_health(self) -> bool:
        """Check disk health."""
        try:
            disk_info = self.system_monitor.get_disk_usage()
            return disk_info['percentage'] < 90
        except Exception:
            return False
    
    # Reporting and export methods
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        
        with self._lock:
            summary = {}
            
            for name, metric in self.metrics.items():
                summary[name] = metric.to_dict()
            
            return summary
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health check status."""
        
        with self._lock:
            status = {
                'overall_healthy': True,
                'checks': {},
                'last_check_time': time.time()
            }
            
            for name, health_check in self.health_checks.items():
                check_status = health_check.to_dict()
                status['checks'][name] = check_status
                
                # Update overall health
                if health_check.critical and health_check.last_result is False:
                    status['overall_healthy'] = False
            
            return status
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        
        with self._lock:
            active_alerts_data = [alert.to_dict() for alert in self.active_alerts.values()]
            recent_alerts = list(self.alert_history)[-20:]  # Last 20 alerts
            
            severity_counts = defaultdict(int)
            for alert in self.active_alerts.values():
                severity_counts[alert.severity.value] += 1
            
            return {
                'active_alerts': active_alerts_data,
                'active_count': len(self.active_alerts),
                'recent_alerts': [alert.to_dict() for alert in recent_alerts],
                'severity_breakdown': dict(severity_counts),
                'total_alert_rules': len(self.alert_rules)
            }
    
    def export_metrics(self, format_type: str = "json", time_range: Optional[int] = None) -> str:
        """Export metrics in specified format."""
        
        end_time = time.time()
        start_time = end_time - (time_range or 3600)  # Default 1 hour
        
        export_data = {
            'timestamp': end_time,
            'time_range_seconds': time_range or 3600,
            'metrics': {}
        }
        
        with self._lock:
            for name, metric in self.metrics.items():
                values_in_range = metric.get_values_in_range(start_time, end_time)
                
                export_data['metrics'][name] = {
                    'type': metric.metric_type.value,
                    'description': metric.description,
                    'unit': metric.unit,
                    'values': [v.to_dict() for v in values_in_range]
                }
        
        if format_type.lower() == "json":
            return json.dumps(export_data, indent=2)
        else:
            # Could add other formats like Prometheus, CSV, etc.
            return json.dumps(export_data, indent=2)
    
    @contextmanager
    def measure_time(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for measuring execution time."""
        
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.record_metric(metric_name, duration, labels)


# Global monitor instance
comprehensive_monitor = ComprehensiveMonitor()


# Convenience functions
def start_monitoring(interval: float = 10.0):
    """Start comprehensive monitoring."""
    comprehensive_monitor.collection_interval = interval
    comprehensive_monitor.start_monitoring()


def stop_monitoring():
    """Stop comprehensive monitoring."""
    comprehensive_monitor.stop_monitoring()


def record_metric(name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
    """Record a metric value."""
    comprehensive_monitor.record_metric(name, value, labels)


def get_health_status() -> Dict[str, Any]:
    """Get system health status."""
    return comprehensive_monitor.get_health_status()


def get_metrics_summary() -> Dict[str, Any]:
    """Get metrics summary."""
    return comprehensive_monitor.get_metric_summary()


def get_alerts_summary() -> Dict[str, Any]:
    """Get alerts summary."""
    return comprehensive_monitor.get_alert_summary()


__all__ = [
    'ComprehensiveMonitor',
    'SystemMonitor',
    'Metric',
    'MetricType',
    'HealthCheck',
    'Alert',
    'AlertSeverity',
    'comprehensive_monitor',
    'start_monitoring',
    'stop_monitoring', 
    'record_metric',
    'get_health_status',
    'get_metrics_summary',
    'get_alerts_summary'
]