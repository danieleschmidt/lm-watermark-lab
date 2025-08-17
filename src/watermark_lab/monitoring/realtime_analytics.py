"""Real-time analytics and performance monitoring system."""

import time
import threading
import asyncio
import json
import statistics
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from contextlib import contextmanager
from enum import Enum
import logging
from pathlib import Path
import psutil

logger = logging.getLogger(__name__)


class MetricAggregation(Enum):
    """Metric aggregation types."""
    SUM = "sum"
    AVERAGE = "average"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    PERCENTILE_50 = "p50"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"


class AlertCondition(Enum):
    """Alert condition types."""
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    THRESHOLD_BREACH = "threshold_breach"
    TREND_ANOMALY = "trend_anomaly"


@dataclass
class TimeSeriesPoint:
    """Single time series data point."""
    
    timestamp: float
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'value': self.value,
            'labels': self.labels
        }


@dataclass
class TimeSeries:
    """Time series data container."""
    
    name: str
    points: deque = field(default_factory=lambda: deque(maxlen=10000))
    labels: Dict[str, str] = field(default_factory=dict)
    retention_seconds: float = 3600  # 1 hour default
    
    def add_point(self, value: Union[int, float], timestamp: float = None, labels: Dict[str, str] = None):
        """Add data point to time series."""
        if timestamp is None:
            timestamp = time.time()
        
        if labels is None:
            labels = {}
        
        point = TimeSeriesPoint(timestamp, value, {**self.labels, **labels})
        self.points.append(point)
        
        # Clean old points
        cutoff_time = timestamp - self.retention_seconds
        while self.points and self.points[0].timestamp < cutoff_time:
            self.points.popleft()
    
    def get_points(self, start_time: float = None, end_time: float = None) -> List[TimeSeriesPoint]:
        """Get points within time range."""
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = time.time()
        
        return [
            point for point in self.points
            if start_time <= point.timestamp <= end_time
        ]
    
    def aggregate(
        self, 
        aggregation: MetricAggregation, 
        start_time: float = None, 
        end_time: float = None
    ) -> Optional[float]:
        """Aggregate values over time range."""
        points = self.get_points(start_time, end_time)
        if not points:
            return None
        
        values = [point.value for point in points]
        
        if aggregation == MetricAggregation.SUM:
            return sum(values)
        elif aggregation == MetricAggregation.AVERAGE:
            return statistics.mean(values)
        elif aggregation == MetricAggregation.COUNT:
            return len(values)
        elif aggregation == MetricAggregation.MIN:
            return min(values)
        elif aggregation == MetricAggregation.MAX:
            return max(values)
        elif aggregation == MetricAggregation.PERCENTILE_50:
            return statistics.median(values)
        elif aggregation == MetricAggregation.PERCENTILE_95:
            return statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values)
        elif aggregation == MetricAggregation.PERCENTILE_99:
            return statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
        else:
            return None


@dataclass
class AlertRule:
    """Alert rule configuration."""
    
    name: str
    metric_name: str
    condition: AlertCondition
    threshold: float
    aggregation: MetricAggregation = MetricAggregation.AVERAGE
    time_window: float = 300  # 5 minutes
    cooldown: float = 600  # 10 minutes
    enabled: bool = True
    labels: Dict[str, str] = field(default_factory=dict)
    
    # Internal state
    last_triggered: Optional[float] = field(default=None, init=False)
    trigger_count: int = field(default=0, init=False)


@dataclass
class Alert:
    """Alert instance."""
    
    rule_name: str
    metric_name: str
    condition: AlertCondition
    threshold: float
    actual_value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'rule_name': self.rule_name,
            'metric_name': self.metric_name,
            'condition': self.condition.value,
            'threshold': self.threshold,
            'actual_value': self.actual_value,
            'timestamp': self.timestamp,
            'labels': self.labels,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at
        }


class PerformanceProfiler:
    """Real-time performance profiling."""
    
    def __init__(self):
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        self.completed_operations: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    @contextmanager
    def profile_operation(self, operation_name: str, labels: Dict[str, str] = None):
        """Profile operation execution time."""
        if labels is None:
            labels = {}
        
        operation_id = f"{operation_name}:{time.time()}:{threading.get_ident()}"
        start_time = time.time()
        
        with self._lock:
            self.active_operations[operation_id] = {
                'name': operation_name,
                'start_time': start_time,
                'labels': labels,
                'thread_id': threading.get_ident()
            }
        
        try:
            yield operation_id
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            with self._lock:
                if operation_id in self.active_operations:
                    operation_data = self.active_operations.pop(operation_id)
                    operation_data.update({
                        'end_time': end_time,
                        'duration': duration,
                        'success': True
                    })
                    self.completed_operations.append(operation_data)
    
    def get_operation_stats(self, operation_name: str = None, time_window: float = 300) -> Dict[str, Any]:
        """Get operation statistics."""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        with self._lock:
            operations = [
                op for op in self.completed_operations
                if (op.get('end_time', 0) >= cutoff_time and 
                    (operation_name is None or op['name'] == operation_name))
            ]
        
        if not operations:
            return {'count': 0}
        
        durations = [op['duration'] for op in operations]
        
        return {
            'count': len(operations),
            'avg_duration': statistics.mean(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'p50_duration': statistics.median(durations),
            'p95_duration': statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else max(durations),
            'operations_per_second': len(operations) / time_window,
            'active_operations': len([op for op in self.active_operations.values() if op['name'] == operation_name]) if operation_name else len(self.active_operations)
        }


class SystemMetricsCollector:
    """System-level metrics collection."""
    
    def __init__(self, collection_interval: float = 10.0):
        self.collection_interval = collection_interval
        self.metrics: Dict[str, TimeSeries] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def start(self):
        """Start metrics collection."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
        logger.info("System metrics collection started")
    
    def stop(self):
        """Stop metrics collection."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("System metrics collection stopped")
    
    def _collect_loop(self):
        """Main collection loop."""
        while self._running:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system metrics."""
        current_time = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        self._add_metric('system_cpu_percent', cpu_percent, current_time)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self._add_metric('system_memory_percent', memory.percent, current_time)
        self._add_metric('system_memory_used_gb', memory.used / (1024**3), current_time)
        self._add_metric('system_memory_available_gb', memory.available / (1024**3), current_time)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self._add_metric('system_disk_percent', (disk.used / disk.total) * 100, current_time)
        self._add_metric('system_disk_used_gb', disk.used / (1024**3), current_time)
        self._add_metric('system_disk_free_gb', disk.free / (1024**3), current_time)
        
        # Network metrics
        network = psutil.net_io_counters()
        self._add_metric('system_network_bytes_sent', network.bytes_sent, current_time)
        self._add_metric('system_network_bytes_recv', network.bytes_recv, current_time)
        
        # Process metrics
        process = psutil.Process()
        self._add_metric('process_cpu_percent', process.cpu_percent(), current_time)
        self._add_metric('process_memory_mb', process.memory_info().rss / (1024**2), current_time)
        self._add_metric('process_num_threads', process.num_threads(), current_time)
        self._add_metric('process_num_fds', process.num_fds() if hasattr(process, 'num_fds') else 0, current_time)
    
    def _add_metric(self, name: str, value: float, timestamp: float):
        """Add metric value."""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = TimeSeries(name)
            self.metrics[name].add_point(value, timestamp)
    
    def get_metric(self, name: str) -> Optional[TimeSeries]:
        """Get metric time series."""
        with self._lock:
            return self.metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, TimeSeries]:
        """Get all metrics."""
        with self._lock:
            return self.metrics.copy()


class RealTimeAnalytics:
    """Main real-time analytics engine."""
    
    def __init__(self):
        self.time_series: Dict[str, TimeSeries] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.profiler = PerformanceProfiler()
        self.system_collector = SystemMetricsCollector()
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()
        
        # Start system metrics collection
        self.system_collector.start()
    
    def add_metric(
        self, 
        name: str, 
        value: Union[int, float], 
        timestamp: float = None,
        labels: Dict[str, str] = None
    ):
        """Add metric data point."""
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            if name not in self.time_series:
                self.time_series[name] = TimeSeries(name)
            
            self.time_series[name].add_point(value, timestamp, labels)
        
        # Check alert rules
        self._check_alerts(name)
    
    def get_metric(self, name: str) -> Optional[TimeSeries]:
        """Get metric time series."""
        with self._lock:
            return self.time_series.get(name)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add alert rule."""
        with self._lock:
            self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove alert rule."""
        with self._lock:
            if rule_name in self.alert_rules:
                del self.alert_rules[rule_name]
        logger.info(f"Removed alert rule: {rule_name}")
    
    def _check_alerts(self, metric_name: str):
        """Check alert rules for metric."""
        current_time = time.time()
        
        with self._lock:
            relevant_rules = [
                rule for rule in self.alert_rules.values()
                if rule.metric_name == metric_name and rule.enabled
            ]
        
        for rule in relevant_rules:
            if (rule.last_triggered and 
                current_time - rule.last_triggered < rule.cooldown):
                continue  # Still in cooldown
            
            metric = self.time_series.get(metric_name)
            if not metric:
                continue
            
            start_time = current_time - rule.time_window
            aggregated_value = metric.aggregate(rule.aggregation, start_time, current_time)
            
            if aggregated_value is None:
                continue
            
            should_trigger = False
            
            if rule.condition == AlertCondition.GREATER_THAN:
                should_trigger = aggregated_value > rule.threshold
            elif rule.condition == AlertCondition.LESS_THAN:
                should_trigger = aggregated_value < rule.threshold
            elif rule.condition == AlertCondition.EQUALS:
                should_trigger = abs(aggregated_value - rule.threshold) < 0.001
            elif rule.condition == AlertCondition.NOT_EQUALS:
                should_trigger = abs(aggregated_value - rule.threshold) >= 0.001
            elif rule.condition == AlertCondition.THRESHOLD_BREACH:
                # Check if value crossed threshold in either direction
                recent_points = metric.get_points(start_time, current_time)
                if len(recent_points) >= 2:
                    prev_value = recent_points[-2].value
                    curr_value = recent_points[-1].value
                    should_trigger = ((prev_value <= rule.threshold < curr_value) or 
                                    (prev_value >= rule.threshold > curr_value))
            
            if should_trigger:
                self._trigger_alert(rule, aggregated_value, current_time)
    
    def _trigger_alert(self, rule: AlertRule, value: float, timestamp: float):
        """Trigger alert."""
        alert = Alert(
            rule_name=rule.name,
            metric_name=rule.metric_name,
            condition=rule.condition,
            threshold=rule.threshold,
            actual_value=value,
            timestamp=timestamp,
            labels=rule.labels.copy()
        )
        
        with self._lock:
            alert_id = f"{rule.name}:{timestamp}"
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Update rule state
            rule.last_triggered = timestamp
            rule.trigger_count += 1
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(
            f"Alert triggered: {rule.name} - {rule.metric_name} "
            f"{rule.condition.value} {rule.threshold} (actual: {value:.2f})"
        )
    
    def resolve_alert(self, alert_id: str):
        """Resolve active alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts.pop(alert_id)
                alert.resolved = True
                alert.resolved_at = time.time()
                logger.info(f"Alert resolved: {alert.rule_name}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add alert callback."""
        self.alert_callbacks.append(callback)
    
    def get_dashboard_data(self, time_window: float = 300) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        current_time = time.time()
        start_time = current_time - time_window
        
        # Get metric summaries
        metric_summaries = {}
        with self._lock:
            for name, ts in self.time_series.items():
                metric_summaries[name] = {
                    'current': ts.points[-1].value if ts.points else 0,
                    'avg': ts.aggregate(MetricAggregation.AVERAGE, start_time, current_time),
                    'min': ts.aggregate(MetricAggregation.MIN, start_time, current_time),
                    'max': ts.aggregate(MetricAggregation.MAX, start_time, current_time),
                    'count': len(ts.get_points(start_time, current_time))
                }
        
        # Get system metrics
        system_metrics = {}
        for name, ts in self.system_collector.get_all_metrics().items():
            system_metrics[name] = {
                'current': ts.points[-1].value if ts.points else 0,
                'avg': ts.aggregate(MetricAggregation.AVERAGE, start_time, current_time)
            }
        
        # Get performance stats
        performance_stats = self.profiler.get_operation_stats(time_window=time_window)
        
        # Get alert summaries
        with self._lock:
            active_alerts = [alert.to_dict() for alert in self.active_alerts.values()]
            recent_alerts = [
                alert.to_dict() for alert in self.alert_history
                if current_time - alert.timestamp <= time_window
            ]
        
        return {
            'timestamp': current_time,
            'time_window': time_window,
            'metrics': metric_summaries,
            'system_metrics': system_metrics,
            'performance': performance_stats,
            'alerts': {
                'active': active_alerts,
                'recent': recent_alerts,
                'total_active': len(active_alerts),
                'total_recent': len(recent_alerts)
            }
        }
    
    def export_metrics(self, filename: str, time_window: float = 3600):
        """Export metrics to file."""
        current_time = time.time()
        start_time = current_time - time_window
        
        export_data = {
            'export_time': current_time,
            'time_window': time_window,
            'metrics': {},
            'alerts': []
        }
        
        with self._lock:
            for name, ts in self.time_series.items():
                points = ts.get_points(start_time, current_time)
                export_data['metrics'][name] = [point.to_dict() for point in points]
            
            export_data['alerts'] = [alert.to_dict() for alert in self.alert_history]
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to {filename}")
    
    def shutdown(self):
        """Shutdown analytics system."""
        self.system_collector.stop()
        logger.info("Real-time analytics system shutdown")


# Global analytics instance
_global_analytics = RealTimeAnalytics()

def get_global_analytics() -> RealTimeAnalytics:
    """Get global analytics instance."""
    return _global_analytics


def track_metric(name: str, value: Union[int, float], labels: Dict[str, str] = None):
    """Track metric using global analytics."""
    _global_analytics.add_metric(name, value, labels=labels)


@contextmanager
def profile_operation(operation_name: str, labels: Dict[str, str] = None):
    """Profile operation using global analytics."""
    with _global_analytics.profiler.profile_operation(operation_name, labels) as operation_id:
        yield operation_id