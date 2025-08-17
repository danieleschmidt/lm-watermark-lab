"""Adaptive auto-scaling system for dynamic resource management."""

import time
import threading
import asyncio
import math
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


class ScalingTrigger(Enum):
    """Scaling trigger types."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    CUSTOM_METRIC = "custom_metric"


@dataclass
class ScalingRule:
    """Auto-scaling rule configuration."""
    
    name: str
    trigger: ScalingTrigger
    scale_up_threshold: float
    scale_down_threshold: float
    scale_up_adjustment: int  # Number of instances to add
    scale_down_adjustment: int  # Number of instances to remove
    cooldown_period: float = 300.0  # 5 minutes
    evaluation_period: float = 60.0  # 1 minute
    min_instances: int = 1
    max_instances: int = 100
    enabled: bool = True
    
    # Internal state
    last_scaling_time: float = field(default=0.0, init=False)
    consecutive_breaches: int = field(default=0, init=False)
    required_breaches: int = field(default=2, init=False)


@dataclass
class ScalingEvent:
    """Scaling event record."""
    
    timestamp: float
    rule_name: str
    direction: ScalingDirection
    trigger_value: float
    threshold: float
    old_instances: int
    new_instances: int
    reason: str
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'rule_name': self.rule_name,
            'direction': self.direction.value,
            'trigger_value': self.trigger_value,
            'threshold': self.threshold,
            'old_instances': self.old_instances,
            'new_instances': self.new_instances,
            'reason': self.reason,
            'success': self.success,
            'error': self.error
        }


class MetricProvider:
    """Base class for metric providers."""
    
    def get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current metric value."""
        raise NotImplementedError
    
    def get_metric_history(self, metric_name: str, duration: float) -> List[Tuple[float, float]]:
        """Get metric history as (timestamp, value) pairs."""
        raise NotImplementedError


class ResourceManager:
    """Base class for resource managers."""
    
    def get_current_instances(self) -> int:
        """Get current number of instances."""
        raise NotImplementedError
    
    def scale_instances(self, target_instances: int) -> bool:
        """Scale to target number of instances."""
        raise NotImplementedError
    
    def get_instance_health(self) -> Dict[str, Any]:
        """Get instance health information."""
        raise NotImplementedError


class PredictiveScaler:
    """Predictive scaling based on historical patterns."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.scaling_history: deque = deque(maxlen=window_size)
        
    def record_metric(self, metric_name: str, value: float, timestamp: float = None):
        """Record metric value for prediction."""
        if timestamp is None:
            timestamp = time.time()
        
        self.metric_history[metric_name].append((timestamp, value))
    
    def record_scaling_event(self, event: ScalingEvent):
        """Record scaling event for learning."""
        self.scaling_history.append(event)
    
    def predict_metric_trend(self, metric_name: str, lookahead_seconds: float = 300) -> Optional[float]:
        """Predict metric value trend."""
        history = list(self.metric_history[metric_name])
        if len(history) < 10:
            return None
        
        # Simple linear regression for trend prediction
        timestamps = [point[0] for point in history[-20:]]  # Last 20 points
        values = [point[1] for point in history[-20:]]
        
        if len(set(timestamps)) < 2:  # Need at least 2 different timestamps
            return None
        
        # Calculate slope
        n = len(timestamps)
        sum_x = sum(timestamps)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(timestamps, values))
        sum_x2 = sum(x * x for x in timestamps)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Predict future value
        future_timestamp = timestamps[-1] + lookahead_seconds
        current_value = values[-1]
        predicted_change = slope * lookahead_seconds
        
        return current_value + predicted_change
    
    def should_proactive_scale(
        self, 
        metric_name: str, 
        current_value: float, 
        threshold: float,
        direction: ScalingDirection
    ) -> bool:
        """Determine if proactive scaling is recommended."""
        predicted_value = self.predict_metric_trend(metric_name)
        if predicted_value is None:
            return False
        
        if direction == ScalingDirection.UP:
            return predicted_value > threshold and current_value < threshold
        elif direction == ScalingDirection.DOWN:
            return predicted_value < threshold and current_value > threshold
        
        return False
    
    def get_optimal_instances(self, metric_name: str, target_value: float) -> Optional[int]:
        """Calculate optimal number of instances for target metric value."""
        # Look for historical correlation between instance count and metric
        scaling_events = list(self.scaling_history)
        if len(scaling_events) < 5:
            return None
        
        # Find events where scaling successfully achieved target
        successful_scaling = []
        for event in scaling_events:
            if event.success and abs(event.trigger_value - target_value) < target_value * 0.1:
                successful_scaling.append((event.new_instances, event.trigger_value))
        
        if not successful_scaling:
            return None
        
        # Use weighted average of successful scaling instances
        weights = []
        instances = []
        for instance_count, metric_value in successful_scaling:
            weight = 1.0 / (abs(metric_value - target_value) + 0.1)  # Higher weight for closer values
            weights.append(weight)
            instances.append(instance_count)
        
        weighted_avg = sum(w * i for w, i in zip(weights, instances)) / sum(weights)
        return max(1, round(weighted_avg))


class AdaptiveAutoScaler:
    """Adaptive auto-scaling engine with predictive capabilities."""
    
    def __init__(
        self,
        metric_provider: MetricProvider,
        resource_manager: ResourceManager,
        evaluation_interval: float = 30.0
    ):
        self.metric_provider = metric_provider
        self.resource_manager = resource_manager
        self.evaluation_interval = evaluation_interval
        
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.scaling_events: deque = deque(maxlen=1000)
        self.predictor = PredictiveScaler()
        
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Adaptive parameters
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        self.enable_predictive_scaling = True
        
        # Performance tracking
        self.evaluation_count = 0
        self.scaling_count = 0
        self.successful_scaling_count = 0
        
    def add_scaling_rule(self, rule: ScalingRule):
        """Add scaling rule."""
        with self._lock:
            self.scaling_rules[rule.name] = rule
        logger.info(f"Added scaling rule: {rule.name}")
    
    def remove_scaling_rule(self, rule_name: str):
        """Remove scaling rule."""
        with self._lock:
            if rule_name in self.scaling_rules:
                del self.scaling_rules[rule_name]
        logger.info(f"Removed scaling rule: {rule_name}")
    
    def start(self):
        """Start auto-scaling engine."""
        if self.running:
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self._thread.start()
        logger.info("Adaptive auto-scaler started")
    
    def stop(self):
        """Stop auto-scaling engine."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Adaptive auto-scaler stopped")
    
    def _scaling_loop(self):
        """Main scaling evaluation loop."""
        while self.running:
            try:
                self._evaluate_scaling_rules()
                self.evaluation_count += 1
                time.sleep(self.evaluation_interval)
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                time.sleep(self.evaluation_interval)
    
    def _evaluate_scaling_rules(self):
        """Evaluate all scaling rules."""
        current_instances = self.resource_manager.get_current_instances()
        current_time = time.time()
        
        with self._lock:
            rules_to_evaluate = [rule for rule in self.scaling_rules.values() if rule.enabled]
        
        for rule in rules_to_evaluate:
            try:
                self._evaluate_single_rule(rule, current_instances, current_time)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
    
    def _evaluate_single_rule(self, rule: ScalingRule, current_instances: int, current_time: float):
        """Evaluate single scaling rule."""
        # Check cooldown period
        if current_time - rule.last_scaling_time < rule.cooldown_period:
            return
        
        # Get current metric value
        metric_value = self._get_rule_metric_value(rule)
        if metric_value is None:
            return
        
        # Record metric for prediction
        self.predictor.record_metric(rule.trigger.value, metric_value, current_time)
        
        # Determine scaling direction
        scaling_direction = self._determine_scaling_direction(rule, metric_value, current_instances)
        
        if scaling_direction == ScalingDirection.NONE:
            rule.consecutive_breaches = 0
            return
        
        # Check if threshold has been breached for required duration
        rule.consecutive_breaches += 1
        if rule.consecutive_breaches < rule.required_breaches:
            logger.debug(f"Rule {rule.name}: Threshold breach {rule.consecutive_breaches}/{rule.required_breaches}")
            return
        
        # Calculate target instances
        target_instances = self._calculate_target_instances(
            rule, scaling_direction, current_instances, metric_value
        )
        
        # Perform scaling
        if target_instances != current_instances:
            self._perform_scaling(rule, scaling_direction, current_instances, target_instances, metric_value)
            rule.consecutive_breaches = 0
            rule.last_scaling_time = current_time
    
    def _get_rule_metric_value(self, rule: ScalingRule) -> Optional[float]:
        """Get metric value for rule evaluation."""
        if rule.trigger in [ScalingTrigger.CPU_USAGE, ScalingTrigger.MEMORY_USAGE, 
                           ScalingTrigger.RESPONSE_TIME, ScalingTrigger.THROUGHPUT]:
            return self.metric_provider.get_metric_value(rule.trigger.value)
        elif rule.trigger == ScalingTrigger.QUEUE_LENGTH:
            return self.metric_provider.get_metric_value("queue_length")
        elif rule.trigger == ScalingTrigger.CUSTOM_METRIC:
            return self.metric_provider.get_metric_value(rule.name)
        else:
            return None
    
    def _determine_scaling_direction(
        self, 
        rule: ScalingRule, 
        metric_value: float, 
        current_instances: int
    ) -> ScalingDirection:
        """Determine if scaling is needed."""
        
        # Check scale up conditions
        if (metric_value > rule.scale_up_threshold and 
            current_instances < rule.max_instances):
            
            # Predictive scaling check
            if self.enable_predictive_scaling:
                if self.predictor.should_proactive_scale(
                    rule.trigger.value, metric_value, rule.scale_up_threshold, ScalingDirection.UP
                ):
                    logger.info(f"Predictive scale up triggered for rule {rule.name}")
                    return ScalingDirection.UP
            
            return ScalingDirection.UP
        
        # Check scale down conditions
        elif (metric_value < rule.scale_down_threshold and 
              current_instances > rule.min_instances):
            
            # Predictive scaling check
            if self.enable_predictive_scaling:
                if self.predictor.should_proactive_scale(
                    rule.trigger.value, metric_value, rule.scale_down_threshold, ScalingDirection.DOWN
                ):
                    logger.info(f"Predictive scale down triggered for rule {rule.name}")
                    return ScalingDirection.DOWN
            
            return ScalingDirection.DOWN
        
        return ScalingDirection.NONE
    
    def _calculate_target_instances(
        self, 
        rule: ScalingRule, 
        direction: ScalingDirection, 
        current_instances: int,
        metric_value: float
    ) -> int:
        """Calculate target number of instances."""
        
        # Try predictive optimal instance calculation
        if self.enable_predictive_scaling:
            if direction == ScalingDirection.UP:
                optimal = self.predictor.get_optimal_instances(
                    rule.trigger.value, rule.scale_up_threshold * 0.9
                )
                if optimal and optimal > current_instances:
                    return min(optimal, rule.max_instances)
            elif direction == ScalingDirection.DOWN:
                optimal = self.predictor.get_optimal_instances(
                    rule.trigger.value, rule.scale_down_threshold * 1.1
                )
                if optimal and optimal < current_instances:
                    return max(optimal, rule.min_instances)
        
        # Fallback to rule-based adjustment
        if direction == ScalingDirection.UP:
            # Adaptive scaling: scale more aggressively if metric is very high
            multiplier = 1.0
            if metric_value > rule.scale_up_threshold * 1.5:
                multiplier = 2.0
            elif metric_value > rule.scale_up_threshold * 1.2:
                multiplier = 1.5
            
            adjustment = max(1, int(rule.scale_up_adjustment * multiplier))
            return min(current_instances + adjustment, rule.max_instances)
            
        elif direction == ScalingDirection.DOWN:
            # Conservative scaling down to avoid oscillation
            adjustment = max(1, rule.scale_down_adjustment)
            return max(current_instances - adjustment, rule.min_instances)
        
        return current_instances
    
    def _perform_scaling(
        self, 
        rule: ScalingRule, 
        direction: ScalingDirection,
        current_instances: int, 
        target_instances: int, 
        metric_value: float
    ):
        """Perform actual scaling operation."""
        
        threshold = (rule.scale_up_threshold if direction == ScalingDirection.UP 
                    else rule.scale_down_threshold)
        
        scaling_event = ScalingEvent(
            timestamp=time.time(),
            rule_name=rule.name,
            direction=direction,
            trigger_value=metric_value,
            threshold=threshold,
            old_instances=current_instances,
            new_instances=target_instances,
            reason=f"{rule.trigger.value} {direction.value} scaling"
        )
        
        try:
            # Perform scaling
            success = self.resource_manager.scale_instances(target_instances)
            scaling_event.success = success
            
            if success:
                self.successful_scaling_count += 1
                logger.info(
                    f"Scaled {direction.value} from {current_instances} to {target_instances} instances "
                    f"(rule: {rule.name}, metric: {metric_value:.2f}, threshold: {threshold:.2f})"
                )
            else:
                scaling_event.error = "Resource manager failed to scale"
                logger.error(f"Failed to scale {direction.value} for rule {rule.name}")
            
        except Exception as e:
            scaling_event.success = False
            scaling_event.error = str(e)
            logger.error(f"Error during scaling for rule {rule.name}: {e}")
        
        # Record scaling event
        self.scaling_events.append(scaling_event)
        self.predictor.record_scaling_event(scaling_event)
        self.scaling_count += 1
        
        # Adaptive learning: adjust rule parameters based on outcomes
        self._adaptive_learning(rule, scaling_event, metric_value)
    
    def _adaptive_learning(self, rule: ScalingRule, event: ScalingEvent, metric_value: float):
        """Adapt rule parameters based on scaling outcomes."""
        if not event.success:
            return
        
        # Check if scaling was effective (wait for next evaluation)
        # This is a simplified learning approach
        
        # Adjust thresholds based on metric behavior
        if event.direction == ScalingDirection.UP:
            # If we're still above threshold after scaling up, be more aggressive
            if metric_value > rule.scale_up_threshold * 1.2:
                adjustment = rule.scale_up_threshold * self.learning_rate * 0.1
                rule.scale_up_threshold = max(0.1, rule.scale_up_threshold - adjustment)
                logger.debug(f"Lowered scale-up threshold for rule {rule.name} to {rule.scale_up_threshold:.2f}")
        
        elif event.direction == ScalingDirection.DOWN:
            # If we scale down too frequently, raise the threshold
            recent_scale_downs = [
                e for e in list(self.scaling_events)[-10:]
                if (e.rule_name == rule.name and 
                    e.direction == ScalingDirection.DOWN and
                    time.time() - e.timestamp < 3600)  # Last hour
            ]
            
            if len(recent_scale_downs) > 3:
                adjustment = rule.scale_down_threshold * self.learning_rate * 0.1
                rule.scale_down_threshold = min(rule.scale_up_threshold * 0.8, 
                                              rule.scale_down_threshold + adjustment)
                logger.debug(f"Raised scale-down threshold for rule {rule.name} to {rule.scale_down_threshold:.2f}")
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics."""
        current_time = time.time()
        
        # Recent scaling events (last hour)
        recent_events = [
            event for event in self.scaling_events
            if current_time - event.timestamp <= 3600
        ]
        
        # Rule statistics
        rule_stats = {}
        with self._lock:
            for rule_name, rule in self.scaling_rules.items():
                rule_events = [e for e in recent_events if e.rule_name == rule_name]
                rule_stats[rule_name] = {
                    'enabled': rule.enabled,
                    'scale_up_threshold': rule.scale_up_threshold,
                    'scale_down_threshold': rule.scale_down_threshold,
                    'min_instances': rule.min_instances,
                    'max_instances': rule.max_instances,
                    'last_scaling_time': rule.last_scaling_time,
                    'consecutive_breaches': rule.consecutive_breaches,
                    'recent_scaling_events': len(rule_events),
                    'recent_scale_ups': len([e for e in rule_events if e.direction == ScalingDirection.UP]),
                    'recent_scale_downs': len([e for e in rule_events if e.direction == ScalingDirection.DOWN])
                }
        
        # Overall statistics
        success_rate = (self.successful_scaling_count / max(1, self.scaling_count))
        
        return {
            'evaluation_count': self.evaluation_count,
            'scaling_count': self.scaling_count,
            'successful_scaling_count': self.successful_scaling_count,
            'success_rate': success_rate,
            'recent_events_count': len(recent_events),
            'current_instances': self.resource_manager.get_current_instances(),
            'enable_predictive_scaling': self.enable_predictive_scaling,
            'learning_rate': self.learning_rate,
            'rule_statistics': rule_stats,
            'recent_events': [event.to_dict() for event in recent_events[-10:]]  # Last 10 events
        }
    
    def export_scaling_history(self, filename: str):
        """Export scaling history to file."""
        export_data = {
            'export_time': time.time(),
            'statistics': self.get_scaling_statistics(),
            'scaling_events': [event.to_dict() for event in self.scaling_events]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Scaling history exported to {filename}")


# Example metric provider and resource manager implementations
class MockMetricProvider(MetricProvider):
    """Mock metric provider for testing."""
    
    def __init__(self):
        self.metrics = {
            'cpu_usage': 50.0,
            'memory_usage': 60.0,
            'queue_length': 10.0,
            'response_time': 100.0,
            'throughput': 1000.0
        }
    
    def set_metric(self, name: str, value: float):
        """Set metric value for testing."""
        self.metrics[name] = value
    
    def get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current metric value."""
        return self.metrics.get(metric_name)
    
    def get_metric_history(self, metric_name: str, duration: float) -> List[Tuple[float, float]]:
        """Get metric history."""
        # Mock implementation
        current_time = time.time()
        current_value = self.metrics.get(metric_name, 0.0)
        return [(current_time - i * 60, current_value + (i % 3 - 1) * 5) for i in range(int(duration / 60))]


class MockResourceManager(ResourceManager):
    """Mock resource manager for testing."""
    
    def __init__(self, initial_instances: int = 2):
        self.current_instances = initial_instances
        self.instance_health = {}
    
    def get_current_instances(self) -> int:
        """Get current number of instances."""
        return self.current_instances
    
    def scale_instances(self, target_instances: int) -> bool:
        """Scale to target number of instances."""
        if target_instances < 1 or target_instances > 100:
            return False
        
        self.current_instances = target_instances
        logger.info(f"Scaled to {target_instances} instances")
        return True
    
    def get_instance_health(self) -> Dict[str, Any]:
        """Get instance health information."""
        return {
            'healthy_instances': self.current_instances,
            'unhealthy_instances': 0,
            'total_instances': self.current_instances
        }


# Global auto-scaler instance
_global_scaler: Optional[AdaptiveAutoScaler] = None

def get_global_scaler(
    metric_provider: MetricProvider = None, 
    resource_manager: ResourceManager = None
) -> AdaptiveAutoScaler:
    """Get or create global adaptive auto-scaler."""
    global _global_scaler
    if _global_scaler is None:
        if metric_provider is None:
            metric_provider = MockMetricProvider()
        if resource_manager is None:
            resource_manager = MockResourceManager()
        
        _global_scaler = AdaptiveAutoScaler(metric_provider, resource_manager)
    return _global_scaler