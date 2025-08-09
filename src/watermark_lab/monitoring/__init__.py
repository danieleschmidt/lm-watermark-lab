"""Advanced monitoring and observability stack for watermarking systems."""

from .health_monitor import HealthMonitor, HealthStatus

# Import optional modules with fallbacks
try:
    from .metrics_collector import AdvancedMetricsCollector, MetricConfig
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

try:
    from .performance_monitor import PerformanceMonitor, PerformanceConfig
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False

try:
    from .alerting import AlertManager, AlertConfig
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from .dashboard import MonitoringDashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

# Build __all__ dynamically based on available modules
__all__ = ["HealthMonitor", "HealthStatus"]

if METRICS_AVAILABLE:
    __all__.extend(["AdvancedMetricsCollector", "MetricConfig"])

if PERFORMANCE_AVAILABLE:
    __all__.extend(["PerformanceMonitor", "PerformanceConfig"])

if ALERTING_AVAILABLE:
    __all__.extend(["AlertManager", "AlertConfig"])

if DASHBOARD_AVAILABLE:
    __all__.append("MonitoringDashboard")