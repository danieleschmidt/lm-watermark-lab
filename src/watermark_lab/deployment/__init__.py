"""Production deployment system with auto-scaling and orchestration."""

from .load_balancer import LoadBalancer, ServiceEndpoint, LoadBalancingStrategy, AutoScaler

# Import optional modules with fallbacks
try:
    from .auto_scaler import ScalingConfig
    AUTO_SCALER_AVAILABLE = True
except ImportError:
    AUTO_SCALER_AVAILABLE = False

try:
    from .load_balancer import LoadBalancerConfig  
    LB_CONFIG_AVAILABLE = True
except ImportError:
    LB_CONFIG_AVAILABLE = False

try:
    from .health_checker import HealthChecker, HealthConfig
    HEALTH_CHECKER_AVAILABLE = True
except ImportError:
    HEALTH_CHECKER_AVAILABLE = False

try:
    from .deployment_manager import DeploymentManager, DeploymentConfig
    DEPLOYMENT_MANAGER_AVAILABLE = True
except ImportError:
    DEPLOYMENT_MANAGER_AVAILABLE = False

# Build __all__ dynamically based on available modules
__all__ = [
    "LoadBalancer",
    "ServiceEndpoint", 
    "LoadBalancingStrategy",
    "AutoScaler"
]

if AUTO_SCALER_AVAILABLE:
    __all__.append("ScalingConfig")

if LB_CONFIG_AVAILABLE:
    __all__.append("LoadBalancerConfig")

if HEALTH_CHECKER_AVAILABLE:
    __all__.extend(["HealthChecker", "HealthConfig"])

if DEPLOYMENT_MANAGER_AVAILABLE:
    __all__.extend(["DeploymentManager", "DeploymentConfig"])