"""Auto-scaling and load balancing system for watermark operations."""

import time
import threading
import logging
import statistics
from typing import Dict, List, Any, Optional, Callable, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json

class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"

class LoadMetric(Enum):
    """Load metrics for scaling decisions."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    QUEUE_SIZE = "queue_size"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"

@dataclass
class ScalingRule:
    """Auto-scaling rule definition."""
    metric: LoadMetric
    threshold_up: float
    threshold_down: float
    min_duration: float = 60.0  # Minimum duration before scaling
    cooldown: float = 300.0     # Cooldown period after scaling
    scale_factor: float = 1.5   # How much to scale by
    enabled: bool = True

@dataclass
class LoadSample:
    """Load measurement sample."""
    timestamp: float
    metric: LoadMetric
    value: float
    worker_id: Optional[str] = None

@dataclass
class ScalingEvent:
    """Auto-scaling event."""
    timestamp: float
    direction: ScalingDirection
    reason: str
    old_capacity: int
    new_capacity: int
    triggered_by: LoadMetric

class LoadBalancer:
    """Intelligent load balancer with health-aware routing."""
    
    def __init__(self, 
                 health_check_interval: float = 30.0,
                 unhealthy_threshold: int = 3,
                 recovery_threshold: int = 2):
        """
        Initialize load balancer.
        
        Args:
            health_check_interval: How often to check worker health
            unhealthy_threshold: Failed checks before marking unhealthy
            recovery_threshold: Successful checks before marking healthy
        """
        self.health_check_interval = health_check_interval
        self.unhealthy_threshold = unhealthy_threshold
        self.recovery_threshold = recovery_threshold
        
        # Worker tracking
        self._workers: Dict[str, Dict[str, Any]] = {}
        self._worker_loads: Dict[str, deque] = {}
        self._worker_health: Dict[str, Dict[str, Any]] = {}
        
        # Round-robin state
        self._current_worker_index = 0
        self._lock = threading.RLock()
        
        # Health checking
        self._health_thread: Optional[threading.Thread] = None
        self._stop_health_check = threading.Event()
        
        self.logger = logging.getLogger("load_balancer")
        self._start_health_monitoring()
    
    def register_worker(self, 
                       worker_id: str,
                       endpoint: str,
                       capacity: int = 100,
                       metadata: Optional[Dict] = None) -> None:
        """Register a worker with the load balancer."""
        with self._lock:
            self._workers[worker_id] = {
                "endpoint": endpoint,
                "capacity": capacity,
                "current_load": 0,
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "avg_response_time": 0.0,
                "last_request": 0.0,
                "metadata": metadata or {}
            }
            
            self._worker_loads[worker_id] = deque(maxlen=100)  # Last 100 load samples
            self._worker_health[worker_id] = {
                "status": "healthy",
                "consecutive_failures": 0,
                "consecutive_successes": 0,
                "last_check": 0.0,
                "last_error": None
            }
        
        self.logger.info(f"Registered worker: {worker_id} at {endpoint}")
    
    def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker."""
        with self._lock:
            self._workers.pop(worker_id, None)
            self._worker_loads.pop(worker_id, None)
            self._worker_health.pop(worker_id, None)
        
        self.logger.info(f"Unregistered worker: {worker_id}")
    
    def get_best_worker(self, 
                       exclude_unhealthy: bool = True,
                       strategy: str = "least_loaded") -> Optional[str]:
        """
        Get the best worker for the next request.
        
        Args:
            exclude_unhealthy: Whether to exclude unhealthy workers
            strategy: Load balancing strategy ('round_robin', 'least_loaded', 'weighted')
        
        Returns:
            Worker ID or None if no workers available
        """
        with self._lock:
            available_workers = []
            
            for worker_id, worker_info in self._workers.items():
                # Check health
                if exclude_unhealthy:
                    health = self._worker_health.get(worker_id, {})
                    if health.get("status") != "healthy":
                        continue
                
                # Check capacity
                if worker_info["current_load"] >= worker_info["capacity"]:
                    continue
                
                available_workers.append(worker_id)
            
            if not available_workers:
                return None
            
            # Apply strategy
            if strategy == "round_robin":
                worker_id = available_workers[self._current_worker_index % len(available_workers)]
                self._current_worker_index += 1
                return worker_id
            
            elif strategy == "least_loaded":
                # Find worker with lowest current load percentage
                best_worker = min(available_workers, 
                                key=lambda w: self._workers[w]["current_load"] / self._workers[w]["capacity"])
                return best_worker
            
            elif strategy == "weighted":
                # Weighted selection based on capacity and performance
                def worker_score(worker_id: str) -> float:
                    worker = self._workers[worker_id]
                    load_ratio = worker["current_load"] / worker["capacity"]
                    
                    # Consider response time (lower is better)
                    response_time_factor = 1.0 / max(worker["avg_response_time"], 0.001)
                    
                    # Consider success rate
                    total_requests = worker["total_requests"]
                    if total_requests > 0:
                        success_rate = worker["successful_requests"] / total_requests
                    else:
                        success_rate = 1.0
                    
                    # Combined score (higher is better)
                    return (1.0 - load_ratio) * response_time_factor * success_rate
                
                best_worker = max(available_workers, key=worker_score)
                return best_worker
            
            else:
                # Default to least loaded
                return available_workers[0]
    
    def record_request_start(self, worker_id: str) -> None:
        """Record that a request started on a worker."""
        with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id]["current_load"] += 1
                self._workers[worker_id]["total_requests"] += 1
                self._workers[worker_id]["last_request"] = time.time()
    
    def record_request_end(self, 
                          worker_id: str,
                          success: bool,
                          response_time: float) -> None:
        """Record that a request completed on a worker."""
        with self._lock:
            if worker_id in self._workers:
                worker = self._workers[worker_id]
                worker["current_load"] = max(0, worker["current_load"] - 1)
                
                if success:
                    worker["successful_requests"] += 1
                else:
                    worker["failed_requests"] += 1
                
                # Update average response time (exponential moving average)
                if worker["avg_response_time"] == 0.0:
                    worker["avg_response_time"] = response_time
                else:
                    alpha = 0.1  # Smoothing factor
                    worker["avg_response_time"] = (
                        alpha * response_time + 
                        (1 - alpha) * worker["avg_response_time"]
                    )
                
                # Record load sample
                load_percentage = worker["current_load"] / worker["capacity"]
                self._worker_loads[worker_id].append(LoadSample(
                    timestamp=time.time(),
                    metric=LoadMetric.CPU_USAGE,  # Using as proxy for load
                    value=load_percentage,
                    worker_id=worker_id
                ))
    
    def _start_health_monitoring(self) -> None:
        """Start health monitoring thread."""
        self._health_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._health_thread.start()
    
    def _health_check_loop(self) -> None:
        """Health check monitoring loop."""
        while not self._stop_health_check.wait(self.health_check_interval):
            try:
                self._check_all_workers_health()
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
    
    def _check_all_workers_health(self) -> None:
        """Check health of all workers."""
        with self._lock:
            current_time = time.time()
            
            for worker_id in list(self._workers.keys()):
                try:
                    # Perform health check
                    is_healthy = self._check_worker_health(worker_id)
                    health = self._worker_health[worker_id]
                    
                    if is_healthy:
                        health["consecutive_failures"] = 0
                        health["consecutive_successes"] += 1
                        
                        # Mark as healthy if enough consecutive successes
                        if (health["status"] != "healthy" and 
                            health["consecutive_successes"] >= self.recovery_threshold):
                            health["status"] = "healthy"
                            self.logger.info(f"Worker {worker_id} recovered")
                    else:
                        health["consecutive_successes"] = 0
                        health["consecutive_failures"] += 1
                        
                        # Mark as unhealthy if enough consecutive failures
                        if (health["status"] == "healthy" and 
                            health["consecutive_failures"] >= self.unhealthy_threshold):
                            health["status"] = "unhealthy"
                            self.logger.warning(f"Worker {worker_id} marked unhealthy")
                    
                    health["last_check"] = current_time
                    
                except Exception as e:
                    self.logger.error(f"Health check failed for worker {worker_id}: {e}")
                    health = self._worker_health[worker_id]
                    health["consecutive_failures"] += 1
                    health["last_error"] = str(e)
    
    def _check_worker_health(self, worker_id: str) -> bool:
        """Check health of a specific worker."""
        worker = self._workers.get(worker_id)
        if not worker:
            return False
        
        # Simple health checks
        current_time = time.time()
        
        # Check if worker has been inactive too long
        if worker["last_request"] > 0:
            inactive_time = current_time - worker["last_request"]
            if inactive_time > 3600:  # 1 hour of inactivity
                return False
        
        # Check error rate
        total_requests = worker["total_requests"]
        if total_requests > 10:  # Only check after some requests
            error_rate = worker["failed_requests"] / total_requests
            if error_rate > 0.5:  # More than 50% errors
                return False
        
        # Check if overloaded
        if worker["current_load"] > worker["capacity"] * 1.5:
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self._lock:
            worker_stats = {}
            
            for worker_id, worker in self._workers.items():
                health = self._worker_health[worker_id]
                recent_loads = list(self._worker_loads[worker_id])
                
                worker_stats[worker_id] = {
                    **worker,
                    "health_status": health["status"],
                    "load_percentage": worker["current_load"] / worker["capacity"] * 100,
                    "success_rate": (
                        worker["successful_requests"] / max(worker["total_requests"], 1) * 100
                    ),
                    "recent_avg_load": (
                        statistics.mean([s.value for s in recent_loads]) * 100
                        if recent_loads else 0
                    )
                }
            
            return {
                "total_workers": len(self._workers),
                "healthy_workers": sum(
                    1 for h in self._worker_health.values() 
                    if h["status"] == "healthy"
                ),
                "total_capacity": sum(w["capacity"] for w in self._workers.values()),
                "current_load": sum(w["current_load"] for w in self._workers.values()),
                "workers": worker_stats
            }

class AutoScaler:
    """Intelligent auto-scaling system."""
    
    def __init__(self,
                 min_workers: int = 1,
                 max_workers: int = 10,
                 target_utilization: float = 0.7,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.5,
                 evaluation_window: float = 300.0,
                 cooldown_period: float = 600.0):
        """
        Initialize auto-scaler.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            target_utilization: Target utilization percentage
            scale_up_threshold: Threshold to scale up
            scale_down_threshold: Threshold to scale down
            evaluation_window: Window for evaluation (seconds)
            cooldown_period: Cooldown between scaling actions
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.evaluation_window = evaluation_window
        self.cooldown_period = cooldown_period
        
        # State tracking
        self._current_workers = min_workers
        self._last_scale_time = 0.0
        self._scaling_history: deque = deque(maxlen=100)
        self._load_samples: deque = deque(maxlen=1000)
        
        # Callbacks
        self._scale_up_callback: Optional[Callable[[int], None]] = None
        self._scale_down_callback: Optional[Callable[[int], None]] = None
        
        # Monitoring
        self._monitoring = True
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()
        
        self.logger = logging.getLogger("auto_scaler")
        self._start_monitoring()
    
    def set_scale_callbacks(self,
                           scale_up_callback: Callable[[int], None],
                           scale_down_callback: Callable[[int], None]) -> None:
        """Set callbacks for scaling actions."""
        self._scale_up_callback = scale_up_callback
        self._scale_down_callback = scale_down_callback
    
    def record_load_sample(self, 
                          metric: LoadMetric,
                          value: float,
                          worker_id: Optional[str] = None) -> None:
        """Record a load sample for scaling decisions."""
        sample = LoadSample(
            timestamp=time.time(),
            metric=metric,
            value=value,
            worker_id=worker_id
        )
        
        self._load_samples.append(sample)
    
    def _start_monitoring(self) -> None:
        """Start monitoring thread."""
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_monitor.wait(30):  # Check every 30 seconds
            try:
                self._evaluate_scaling()
            except Exception as e:
                self.logger.error(f"Scaling evaluation error: {e}")
    
    def _evaluate_scaling(self) -> None:
        """Evaluate whether scaling is needed."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self._last_scale_time < self.cooldown_period:
            return
        
        # Get recent load samples
        cutoff_time = current_time - self.evaluation_window
        recent_samples = [
            s for s in self._load_samples
            if s.timestamp >= cutoff_time
        ]
        
        if len(recent_samples) < 10:  # Need minimum samples
            return
        
        # Calculate metrics
        cpu_samples = [s.value for s in recent_samples if s.metric == LoadMetric.CPU_USAGE]
        queue_samples = [s.value for s in recent_samples if s.metric == LoadMetric.QUEUE_SIZE]
        response_samples = [s.value for s in recent_samples if s.metric == LoadMetric.RESPONSE_TIME]
        
        # Determine scaling action
        scaling_decision = self._make_scaling_decision(
            cpu_samples, queue_samples, response_samples
        )
        
        if scaling_decision != ScalingDirection.STABLE:
            self._execute_scaling(scaling_decision)
    
    def _make_scaling_decision(self,
                              cpu_samples: List[float],
                              queue_samples: List[float],
                              response_samples: List[float]) -> ScalingDirection:
        """Make scaling decision based on metrics."""
        scale_up_votes = 0
        scale_down_votes = 0
        
        # CPU utilization
        if cpu_samples:
            avg_cpu = statistics.mean(cpu_samples)
            if avg_cpu > self.scale_up_threshold:
                scale_up_votes += 2
            elif avg_cpu < self.scale_down_threshold:
                scale_down_votes += 1
        
        # Queue size
        if queue_samples:
            avg_queue = statistics.mean(queue_samples)
            max_queue = max(queue_samples)
            
            if max_queue > 50 or avg_queue > 20:  # High queue
                scale_up_votes += 3
            elif avg_queue < 5:  # Low queue
                scale_down_votes += 1
        
        # Response time
        if response_samples:
            avg_response = statistics.mean(response_samples)
            p95_response = sorted(response_samples)[int(len(response_samples) * 0.95)]
            
            if p95_response > 5.0 or avg_response > 2.0:  # Slow responses
                scale_up_votes += 2
            elif avg_response < 0.5:  # Fast responses
                scale_down_votes += 1
        
        # Make decision
        if scale_up_votes >= 3 and self._current_workers < self.max_workers:
            return ScalingDirection.UP
        elif scale_down_votes >= 2 and self._current_workers > self.min_workers:
            return ScalingDirection.DOWN
        else:
            return ScalingDirection.STABLE
    
    def _execute_scaling(self, direction: ScalingDirection) -> None:
        """Execute scaling action."""
        old_workers = self._current_workers
        
        if direction == ScalingDirection.UP:
            new_workers = min(self.max_workers, 
                            int(self._current_workers * 1.5))  # 50% increase
            if self._scale_up_callback:
                self._scale_up_callback(new_workers - old_workers)
        
        elif direction == ScalingDirection.DOWN:
            new_workers = max(self.min_workers,
                            int(self._current_workers * 0.7))  # 30% decrease
            if self._scale_down_callback:
                self._scale_down_callback(old_workers - new_workers)
        
        else:
            return
        
        # Record scaling event
        event = ScalingEvent(
            timestamp=time.time(),
            direction=direction,
            reason="Auto-scaling triggered",
            old_capacity=old_workers,
            new_capacity=new_workers,
            triggered_by=LoadMetric.CPU_USAGE  # Simplified
        )
        
        self._scaling_history.append(event)
        self._current_workers = new_workers
        self._last_scale_time = time.time()
        
        self.logger.info(f"Scaled {direction.value}: {old_workers} -> {new_workers} workers")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get auto-scaler statistics."""
        recent_events = [
            {
                "timestamp": e.timestamp,
                "direction": e.direction.value,
                "reason": e.reason,
                "old_capacity": e.old_capacity,
                "new_capacity": e.new_capacity
            }
            for e in list(self._scaling_history)[-10:]  # Last 10 events
        ]
        
        return {
            "current_workers": self._current_workers,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "target_utilization": self.target_utilization,
            "last_scale_time": self._last_scale_time,
            "recent_events": recent_events,
            "load_samples_count": len(self._load_samples),
            "scaling_history_count": len(self._scaling_history)
        }
    
    def stop(self) -> None:
        """Stop auto-scaling."""
        self._monitoring = False
        self._stop_monitor.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)