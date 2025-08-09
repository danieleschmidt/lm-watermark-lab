"""Advanced load balancing for scalable watermarking services."""

import random
import time
import threading
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import heapq

from ..utils.logging import get_logger
from ..utils.exceptions import WatermarkLabError, ResourceError
from ..utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    CONSISTENT_HASH = "consistent_hash"
    RANDOM = "random"
    WEIGHTED_RANDOM = "weighted_random"


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration."""
    
    id: str
    host: str
    port: int
    weight: int = 100
    max_connections: int = 1000
    timeout: float = 30.0
    
    # Runtime state
    active_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_request_time: float = 0.0
    healthy: bool = True
    
    # Circuit breaker
    circuit_breaker: Optional[CircuitBreaker] = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize circuit breaker."""
        self.circuit_breaker = CircuitBreaker(
            f"endpoint_{self.id}",
            CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=30.0,
                timeout=self.timeout
            )
        )
    
    @property
    def url(self) -> str:
        """Get endpoint URL."""
        return f"http://{self.host}:{self.port}"
    
    @property
    def load_factor(self) -> float:
        """Calculate current load factor (0.0 to 1.0)."""
        if self.max_connections == 0:
            return 0.0
        return min(1.0, self.active_connections / self.max_connections)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.successful_requests + self.failed_requests
        if total == 0:
            return 1.0
        return self.successful_requests / total
    
    @property
    def score(self) -> float:
        """Calculate endpoint score for ranking."""
        if not self.healthy:
            return 0.0
        
        # Combine multiple factors
        load_score = 1.0 - self.load_factor
        success_score = self.success_rate
        response_score = 1.0 / (1.0 + self.avg_response_time)
        weight_score = self.weight / 100.0
        
        return (load_score * 0.3 + success_score * 0.3 + 
                response_score * 0.2 + weight_score * 0.2)
    
    def record_request(self, response_time: float, success: bool):
        """Record request metrics."""
        self.total_requests += 1
        self.last_request_time = time.time()
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Update average response time
        if self.total_requests == 1:
            self.avg_response_time = response_time
        else:
            alpha = 0.1  # Exponential smoothing factor
            self.avg_response_time = (
                alpha * response_time + (1 - alpha) * self.avg_response_time
            )
    
    def increment_connections(self):
        """Increment active connections."""
        self.active_connections += 1
    
    def decrement_connections(self):
        """Decrement active connections."""
        self.active_connections = max(0, self.active_connections - 1)
    
    def can_accept_request(self) -> bool:
        """Check if endpoint can accept new requests."""
        return (self.healthy and 
                self.active_connections < self.max_connections and
                self.circuit_breaker.get_state().value != "open")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'host': self.host,
            'port': self.port,
            'weight': self.weight,
            'active_connections': self.active_connections,
            'max_connections': self.max_connections,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'avg_response_time': self.avg_response_time,
            'success_rate': self.success_rate,
            'load_factor': self.load_factor,
            'score': self.score,
            'healthy': self.healthy,
            'url': self.url
        }


class ConsistentHashRing:
    """Consistent hash ring for consistent hashing load balancing."""
    
    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, ServiceEndpoint] = {}
        self.sorted_keys: List[int] = []
        self._lock = threading.RLock()
    
    def add_endpoint(self, endpoint: ServiceEndpoint):
        """Add endpoint to hash ring."""
        with self._lock:
            for i in range(self.virtual_nodes):
                virtual_key = self._hash(f"{endpoint.id}:{i}")
                self.ring[virtual_key] = endpoint
            
            self.sorted_keys = sorted(self.ring.keys())
    
    def remove_endpoint(self, endpoint: ServiceEndpoint):
        """Remove endpoint from hash ring."""
        with self._lock:
            keys_to_remove = []
            for key, ep in self.ring.items():
                if ep.id == endpoint.id:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.ring[key]
            
            self.sorted_keys = sorted(self.ring.keys())
    
    def get_endpoint(self, key: str) -> Optional[ServiceEndpoint]:
        """Get endpoint for given key using consistent hashing."""
        if not self.sorted_keys:
            return None
        
        hash_key = self._hash(key)
        
        with self._lock:
            # Find first endpoint with key >= hash_key
            for ring_key in self.sorted_keys:
                if ring_key >= hash_key:
                    return self.ring[ring_key]
            
            # Wrap around to first endpoint
            return self.ring[self.sorted_keys[0]]
    
    def _hash(self, key: str) -> int:
        """Hash function for consistent hashing."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)


class LoadBalancer:
    """Advanced load balancer with multiple strategies and health checking."""
    
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN,
        health_check_interval: float = 30.0,
        health_check_timeout: float = 5.0
    ):
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout
        
        self.endpoints: List[ServiceEndpoint] = []
        self.current_index = 0
        self.consistent_hash_ring = ConsistentHashRing()
        
        # Health checking
        self.health_check_running = False
        self.health_check_thread = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger = get_logger("load_balancer")
    
    def add_endpoint(self, endpoint: ServiceEndpoint):
        """Add service endpoint."""
        with self._lock:
            self.endpoints.append(endpoint)
            
            if self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                self.consistent_hash_ring.add_endpoint(endpoint)
        
        self.logger.info(f"Added endpoint: {endpoint.url}")
    
    def remove_endpoint(self, endpoint_id: str) -> bool:
        """Remove service endpoint."""
        with self._lock:
            for i, endpoint in enumerate(self.endpoints):
                if endpoint.id == endpoint_id:
                    removed_endpoint = self.endpoints.pop(i)
                    
                    if self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                        self.consistent_hash_ring.remove_endpoint(removed_endpoint)
                    
                    self.logger.info(f"Removed endpoint: {removed_endpoint.url}")
                    return True
            
            return False
    
    def get_endpoint(self, request_key: Optional[str] = None) -> Optional[ServiceEndpoint]:
        """Get next endpoint based on load balancing strategy."""
        with self._lock:
            if not self.endpoints:
                return None
            
            healthy_endpoints = [ep for ep in self.endpoints if ep.can_accept_request()]
            if not healthy_endpoints:
                # Fallback to any healthy endpoint
                healthy_endpoints = [ep for ep in self.endpoints if ep.healthy]
                if not healthy_endpoints:
                    return None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin(healthy_endpoints)
            
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin(healthy_endpoints)
            
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections(healthy_endpoints)
            
            elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                return self._least_response_time(healthy_endpoints)
            
            elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                if request_key:
                    endpoint = self.consistent_hash_ring.get_endpoint(request_key)
                    if endpoint and endpoint.can_accept_request():
                        return endpoint
                # Fallback to round robin if consistent hash fails
                return self._round_robin(healthy_endpoints)
            
            elif self.strategy == LoadBalancingStrategy.RANDOM:
                return random.choice(healthy_endpoints)
            
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_RANDOM:
                return self._weighted_random(healthy_endpoints)
            
            else:
                return self._round_robin(healthy_endpoints)
    
    def _round_robin(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Round robin load balancing."""
        endpoint = endpoints[self.current_index % len(endpoints)]
        self.current_index = (self.current_index + 1) % len(endpoints)
        return endpoint
    
    def _weighted_round_robin(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Weighted round robin load balancing."""
        total_weight = sum(ep.weight for ep in endpoints)
        if total_weight == 0:
            return self._round_robin(endpoints)
        
        # Create weighted list
        weighted_endpoints = []
        for endpoint in endpoints:
            weight_ratio = endpoint.weight / total_weight
            count = max(1, int(weight_ratio * 100))  # Scale to reasonable numbers
            weighted_endpoints.extend([endpoint] * count)
        
        endpoint = weighted_endpoints[self.current_index % len(weighted_endpoints)]
        self.current_index = (self.current_index + 1) % len(weighted_endpoints)
        return endpoint
    
    def _least_connections(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Least connections load balancing."""
        return min(endpoints, key=lambda ep: ep.active_connections)
    
    def _least_response_time(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Least response time load balancing."""
        return min(endpoints, key=lambda ep: ep.avg_response_time)
    
    def _weighted_random(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Weighted random load balancing."""
        weights = [ep.weight for ep in endpoints]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(endpoints)
        
        rand = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for endpoint, weight in zip(endpoints, weights):
            cumulative_weight += weight
            if rand <= cumulative_weight:
                return endpoint
        
        return endpoints[-1]  # Fallback
    
    def execute_request(
        self,
        request_func: Callable[[ServiceEndpoint], Any],
        request_key: Optional[str] = None,
        max_retries: int = 3
    ) -> Any:
        """Execute request with load balancing and retries."""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            endpoint = self.get_endpoint(request_key)
            if not endpoint:
                raise ResourceError("No healthy endpoints available")
            
            start_time = time.time()
            endpoint.increment_connections()
            
            try:
                # Execute request through circuit breaker
                result = endpoint.circuit_breaker.call(request_func, endpoint)
                
                # Record success
                response_time = time.time() - start_time
                endpoint.record_request(response_time, success=True)
                
                return result
                
            except Exception as e:
                # Record failure
                response_time = time.time() - start_time
                endpoint.record_request(response_time, success=False)
                
                last_exception = e
                self.logger.warning(
                    f"Request failed on {endpoint.url} (attempt {attempt + 1}): {e}"
                )
                
                # Don't retry on certain errors
                if isinstance(e, (ValueError, TypeError)):
                    break
                
            finally:
                endpoint.decrement_connections()
        
        raise WatermarkLabError(f"Request failed after {max_retries + 1} attempts") from last_exception
    
    def start_health_checks(self, health_check_func: Callable[[ServiceEndpoint], bool]):
        """Start health checking for endpoints."""
        if self.health_check_running:
            return
        
        self.health_check_running = True
        self.health_check_func = health_check_func
        
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self.health_check_thread.start()
        
        self.logger.info(f"Started health checks with {self.health_check_interval}s interval")
    
    def stop_health_checks(self):
        """Stop health checking."""
        self.health_check_running = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5.0)
        
        self.logger.info("Stopped health checks")
    
    def _health_check_loop(self):
        """Health check loop."""
        while self.health_check_running:
            try:
                for endpoint in self.endpoints[:]:  # Copy to avoid modification during iteration
                    try:
                        is_healthy = self.health_check_func(endpoint)
                        
                        if endpoint.healthy != is_healthy:
                            endpoint.healthy = is_healthy
                            status = "healthy" if is_healthy else "unhealthy"
                            self.logger.info(f"Endpoint {endpoint.url} is now {status}")
                        
                    except Exception as e:
                        self.logger.error(f"Health check failed for {endpoint.url}: {e}")
                        endpoint.healthy = False
                
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
            
            time.sleep(self.health_check_interval)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self._lock:
            endpoint_stats = [ep.to_dict() for ep in self.endpoints]
            
            total_requests = sum(ep.total_requests for ep in self.endpoints)
            healthy_count = sum(1 for ep in self.endpoints if ep.healthy)
            
            return {
                'strategy': self.strategy.value,
                'total_endpoints': len(self.endpoints),
                'healthy_endpoints': healthy_count,
                'total_requests': total_requests,
                'endpoints': endpoint_stats
            }
    
    def rebalance(self):
        """Trigger rebalancing of endpoints."""
        with self._lock:
            # Sort endpoints by score for better distribution
            self.endpoints.sort(key=lambda ep: ep.score, reverse=True)
            
            # Reset round robin counter
            self.current_index = 0
            
            self.logger.info("Rebalanced endpoints")


class AutoScaler:
    """Auto-scaling controller based on load metrics."""
    
    def __init__(
        self,
        load_balancer: LoadBalancer,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        min_instances: int = 2,
        max_instances: int = 10
    ):
        self.load_balancer = load_balancer
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.min_instances = min_instances
        self.max_instances = max_instances
        
        self.scaling_history: List[Dict[str, Any]] = []
        self.last_scaling_action = 0.0
        self.scaling_cooldown = 300.0  # 5 minutes
        
        self.logger = get_logger("auto_scaler")
    
    def check_scaling(self, create_instance_func: Optional[Callable] = None,
                     remove_instance_func: Optional[Callable] = None):
        """Check if scaling action is needed."""
        current_time = time.time()
        
        # Respect cooldown period
        if current_time - self.last_scaling_action < self.scaling_cooldown:
            return
        
        stats = self.load_balancer.get_stats()
        healthy_endpoints = stats['healthy_endpoints']
        
        if healthy_endpoints == 0:
            return
        
        # Calculate average load
        endpoint_stats = stats['endpoints']
        healthy_endpoint_stats = [ep for ep in endpoint_stats if ep['healthy']]
        
        if not healthy_endpoint_stats:
            return
        
        avg_load = sum(ep['load_factor'] for ep in healthy_endpoint_stats) / len(healthy_endpoint_stats)
        avg_response_time = sum(ep['avg_response_time'] for ep in healthy_endpoint_stats) / len(healthy_endpoint_stats)
        
        # Scale up conditions
        if (avg_load > self.scale_up_threshold or avg_response_time > 2.0) and healthy_endpoints < self.max_instances:
            if create_instance_func:
                self._scale_up(create_instance_func, avg_load, avg_response_time)
        
        # Scale down conditions
        elif avg_load < self.scale_down_threshold and avg_response_time < 0.5 and healthy_endpoints > self.min_instances:
            if remove_instance_func:
                self._scale_down(remove_instance_func, avg_load, avg_response_time)
    
    def _scale_up(self, create_func: Callable, avg_load: float, avg_response_time: float):
        """Scale up instances."""
        try:
            new_endpoint = create_func()
            if new_endpoint:
                self.load_balancer.add_endpoint(new_endpoint)
                
                self.scaling_history.append({
                    'action': 'scale_up',
                    'timestamp': time.time(),
                    'avg_load': avg_load,
                    'avg_response_time': avg_response_time,
                    'endpoint_count': len(self.load_balancer.endpoints)
                })
                
                self.last_scaling_action = time.time()
                self.logger.info(f"Scaled up: added endpoint {new_endpoint.id}")
        
        except Exception as e:
            self.logger.error(f"Scale up failed: {e}")
    
    def _scale_down(self, remove_func: Callable, avg_load: float, avg_response_time: float):
        """Scale down instances."""
        try:
            # Find endpoint with lowest load
            endpoint_to_remove = min(
                self.load_balancer.endpoints,
                key=lambda ep: ep.active_connections
            )
            
            if remove_func(endpoint_to_remove):
                self.load_balancer.remove_endpoint(endpoint_to_remove.id)
                
                self.scaling_history.append({
                    'action': 'scale_down',
                    'timestamp': time.time(),
                    'avg_load': avg_load,
                    'avg_response_time': avg_response_time,
                    'endpoint_count': len(self.load_balancer.endpoints)
                })
                
                self.last_scaling_action = time.time()
                self.logger.info(f"Scaled down: removed endpoint {endpoint_to_remove.id}")
        
        except Exception as e:
            self.logger.error(f"Scale down failed: {e}")
    
    def get_scaling_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get scaling history."""
        cutoff_time = time.time() - (hours * 3600)
        return [event for event in self.scaling_history if event['timestamp'] > cutoff_time]


__all__ = [
    "LoadBalancer",
    "LoadBalancingStrategy",
    "ServiceEndpoint",
    "AutoScaler",
    "ConsistentHashRing"
]