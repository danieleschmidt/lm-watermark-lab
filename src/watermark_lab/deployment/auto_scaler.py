"""Intelligent auto-scaling system for watermarking services with predictive capabilities."""

import time
import threading
import math
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import statistics

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from ..utils.logging import get_logger
from ..utils.exceptions import DeploymentError, ScalingError
from ..utils.metrics import MetricsCollector
from ..monitoring.health_monitor import HealthMonitor, HealthStatus

logger = get_logger("deployment.auto_scaler")


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


class ScalingTrigger(Enum):
    """Scaling trigger types."""
    CPU_THRESHOLD = "cpu_threshold"
    MEMORY_THRESHOLD = "memory_threshold"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_LENGTH = "queue_length"
    PREDICTIVE = "predictive"


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling system."""
    
    # Instance limits
    min_instances: int = 1
    max_instances: int = 10
    target_instances: int = 2
    
    # Scaling triggers
    cpu_threshold_up: float = 70.0  # Scale up when CPU > 70%
    cpu_threshold_down: float = 30.0  # Scale down when CPU < 30%
    memory_threshold_up: float = 80.0
    memory_threshold_down: float = 40.0
    response_time_threshold: float = 2.0  # seconds
    queue_length_threshold: int = 100
    
    # Scaling behavior
    scale_up_factor: float = 2.0  # Double instances when scaling up
    scale_down_factor: float = 0.5  # Half instances when scaling down
    cooldown_period: int = 300  # 5 minutes between scaling actions
    
    # Metrics collection
    metrics_window: int = 300  # 5 minutes of metrics
    evaluation_interval: int = 60  # Evaluate every minute
    
    # Predictive scaling
    enable_predictive: bool = True
    prediction_window: int = 900  # 15 minutes ahead
    prediction_accuracy_threshold: float = 0.8
    
    # Safety limits
    max_scale_up_per_action: int = 5
    max_scale_down_per_action: int = 3
    emergency_scale_up_threshold: float = 95.0  # CPU/Memory
    
    def __post_init__(self):
        """Validate configuration."""
        if self.min_instances > self.max_instances:
            raise ValueError("min_instances cannot be greater than max_instances")
        
        if self.target_instances < self.min_instances or self.target_instances > self.max_instances:
            self.target_instances = max(self.min_instances, min(self.max_instances, self.target_instances))


@dataclass
class MetricPoint:
    """Single metric data point."""
    
    timestamp: float
    cpu_percent: float
    memory_percent: float
    request_rate: float
    response_time: float
    queue_length: int
    active_instances: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ScalingDecision:
    """Decision made by the auto-scaler."""
    
    action: ScalingAction
    current_instances: int
    target_instances: int
    trigger: ScalingTrigger
    confidence: float
    reasoning: str
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'action': self.action.value,
            'current_instances': self.current_instances,
            'target_instances': self.target_instances,
            'trigger': self.trigger.value,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp
        }


class MetricsPredictor:
    """Predictive analytics for scaling decisions."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.logger = get_logger("predictor")
        
        # Historical metrics
        self.cpu_history = deque(maxlen=window_size)
        self.memory_history = deque(maxlen=window_size)
        self.request_history = deque(maxlen=window_size)
        
    def add_metrics(self, metric_point: MetricPoint):
        """Add new metric point to history."""
        
        self.cpu_history.append((metric_point.timestamp, metric_point.cpu_percent))
        self.memory_history.append((metric_point.timestamp, metric_point.memory_percent))
        self.request_history.append((metric_point.timestamp, metric_point.request_rate))
    
    def predict_cpu(self, future_seconds: int) -> Tuple[float, float]:
        """Predict CPU usage and confidence."""
        
        if len(self.cpu_history) < 10:
            return 50.0, 0.0  # Default prediction with low confidence
        
        # Simple linear trend prediction
        timestamps = [point[0] for point in self.cpu_history]
        values = [point[1] for point in self.cpu_history]
        
        return self._predict_trend(timestamps, values, future_seconds)
    
    def predict_memory(self, future_seconds: int) -> Tuple[float, float]:
        """Predict memory usage and confidence."""
        
        if len(self.memory_history) < 10:
            return 50.0, 0.0
        
        timestamps = [point[0] for point in self.memory_history]
        values = [point[1] for point in self.memory_history]
        
        return self._predict_trend(timestamps, values, future_seconds)
    
    def predict_request_rate(self, future_seconds: int) -> Tuple[float, float]:
        """Predict request rate and confidence."""
        
        if len(self.request_history) < 10:
            return 10.0, 0.0
        
        timestamps = [point[0] for point in self.request_history]
        values = [point[1] for point in self.request_history]
        
        return self._predict_trend(timestamps, values, future_seconds)
    
    def _predict_trend(self, timestamps: List[float], values: List[float], future_seconds: int) -> Tuple[float, float]:
        """Predict future value based on linear trend."""
        
        if NUMPY_AVAILABLE:
            return self._numpy_prediction(timestamps, values, future_seconds)
        else:
            return self._simple_prediction(timestamps, values, future_seconds)
    
    def _numpy_prediction(self, timestamps: List[float], values: List[float], future_seconds: int) -> Tuple[float, float]:
        """NumPy-based prediction with better accuracy."""
        
        try:
            timestamps_array = np.array(timestamps)
            values_array = np.array(values)
            
            # Linear regression
            coeffs = np.polyfit(timestamps_array, values_array, 1)
            
            # Predict future value
            future_timestamp = timestamps[-1] + future_seconds
            predicted_value = np.polyval(coeffs, future_timestamp)
            
            # Calculate confidence based on R-squared
            y_pred = np.polyval(coeffs, timestamps_array)
            ss_res = np.sum((values_array - y_pred) ** 2)
            ss_tot = np.sum((values_array - np.mean(values_array)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            confidence = max(0.0, min(1.0, r_squared))
            predicted_value = max(0.0, predicted_value)  # Ensure non-negative
            
            return float(predicted_value), float(confidence)
            
        except Exception as e:
            self.logger.warning(f"NumPy prediction failed: {e}")
            return self._simple_prediction(timestamps, values, future_seconds)
    
    def _simple_prediction(self, timestamps: List[float], values: List[float], future_seconds: int) -> Tuple[float, float]:
        """Simple prediction without NumPy."""
        
        try:
            # Calculate simple moving average and trend
            recent_values = values[-10:]  # Last 10 points
            avg_value = statistics.mean(recent_values)
            
            # Simple trend calculation
            if len(values) >= 2:
                trend = (values[-1] - values[-2]) / max(1, timestamps[-1] - timestamps[-2])
                predicted_value = values[-1] + trend * future_seconds
            else:
                predicted_value = avg_value
            
            # Calculate confidence based on variance
            if len(recent_values) > 1:
                variance = statistics.variance(recent_values)
                confidence = max(0.0, min(1.0, 1.0 - (variance / 100.0)))
            else:
                confidence = 0.5
            
            predicted_value = max(0.0, predicted_value)  # Ensure non-negative
            
            return float(predicted_value), float(confidence)
            
        except Exception as e:
            self.logger.warning(f"Simple prediction failed: {e}")
            return statistics.mean(values) if values else 50.0, 0.1


class AutoScaler:
    """Intelligent auto-scaling system with predictive capabilities."""
    
    def __init__(self, config: ScalingConfig, health_monitor: Optional[HealthMonitor] = None):
        self.config = config
        self.health_monitor = health_monitor
        self.logger = get_logger("auto_scaler")
        
        # Current state
        self.current_instances = config.target_instances
        self.last_scaling_time = 0.0
        self.metrics_history = deque(maxlen=config.metrics_window // config.evaluation_interval)
        
        # Predictive components
        self.predictor = MetricsPredictor() if config.enable_predictive else None
        
        # Scaling callbacks
        self.scale_up_callback: Optional[Callable[[int], bool]] = None
        self.scale_down_callback: Optional[Callable[[int], bool]] = None
        
        # Monitoring
        self.running = False
        self.monitor_thread = None
        
        # Statistics
        self.scaling_history = deque(maxlen=100)
        self.stats = {
            'total_scale_ups': 0,
            'total_scale_downs': 0,
            'avg_response_time': 0.0,
            'prediction_accuracy': 0.0
        }
    
    def set_scaling_callbacks(
        self,
        scale_up: Callable[[int], bool],
        scale_down: Callable[[int], bool]
    ):
        """Set callbacks for actual scaling operations."""
        self.scale_up_callback = scale_up
        self.scale_down_callback = scale_down
    
    def start_monitoring(self):
        """Start automatic scaling monitoring."""
        
        if self.running:
            self.logger.warning("Auto-scaler already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info(f"Started auto-scaler monitoring with {self.config.evaluation_interval}s interval")
    
    def stop_monitoring(self):
        """Stop automatic scaling monitoring."""
        
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("Stopped auto-scaler monitoring")
    
    def add_metrics(
        self,
        cpu_percent: float,
        memory_percent: float,
        request_rate: float,
        response_time: float,
        queue_length: int = 0
    ):
        """Add current metrics for scaling decisions."""
        
        metric_point = MetricPoint(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            request_rate=request_rate,
            response_time=response_time,
            queue_length=queue_length,
            active_instances=self.current_instances
        )
        
        self.metrics_history.append(metric_point)
        
        if self.predictor:
            self.predictor.add_metrics(metric_point)
    
    def evaluate_scaling(self) -> ScalingDecision:
        """Evaluate current metrics and make scaling decision."""
        
        if not self.metrics_history:
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                current_instances=self.current_instances,
                target_instances=self.current_instances,
                trigger=ScalingTrigger.CPU_THRESHOLD,
                confidence=0.0,
                reasoning="No metrics available",
                timestamp=time.time()
            )
        
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scaling_time < self.config.cooldown_period:
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                current_instances=self.current_instances,
                target_instances=self.current_instances,
                trigger=ScalingTrigger.CPU_THRESHOLD,
                confidence=1.0,
                reasoning="In cooldown period",
                timestamp=current_time
            )
        
        # Get recent metrics
        recent_metrics = list(self.metrics_history)[-min(5, len(self.metrics_history)):]
        
        if not recent_metrics:
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                current_instances=self.current_instances,
                target_instances=self.current_instances,
                trigger=ScalingTrigger.CPU_THRESHOLD,
                confidence=0.0,
                reasoning="No recent metrics",
                timestamp=current_time
            )
        
        # Calculate average metrics
        avg_cpu = statistics.mean([m.cpu_percent for m in recent_metrics])
        avg_memory = statistics.mean([m.memory_percent for m in recent_metrics])
        avg_response_time = statistics.mean([m.response_time for m in recent_metrics])
        avg_queue_length = statistics.mean([m.queue_length for m in recent_metrics])
        
        # Check for emergency scaling
        emergency_decision = self._check_emergency_scaling(avg_cpu, avg_memory)
        if emergency_decision.action != ScalingAction.NO_ACTION:
            return emergency_decision
        
        # Evaluate different scaling triggers
        decisions = []
        
        # CPU-based scaling
        cpu_decision = self._evaluate_cpu_scaling(avg_cpu)
        if cpu_decision.action != ScalingAction.NO_ACTION:
            decisions.append(cpu_decision)
        
        # Memory-based scaling
        memory_decision = self._evaluate_memory_scaling(avg_memory)
        if memory_decision.action != ScalingAction.NO_ACTION:
            decisions.append(memory_decision)
        
        # Response time scaling
        response_decision = self._evaluate_response_time_scaling(avg_response_time)
        if response_decision.action != ScalingAction.NO_ACTION:
            decisions.append(response_decision)
        
        # Queue length scaling
        queue_decision = self._evaluate_queue_scaling(avg_queue_length)
        if queue_decision.action != ScalingAction.NO_ACTION:
            decisions.append(queue_decision)
        
        # Predictive scaling
        if self.config.enable_predictive and self.predictor:
            predictive_decision = self._evaluate_predictive_scaling()
            if predictive_decision.action != ScalingAction.NO_ACTION:
                decisions.append(predictive_decision)
        
        # Choose the best decision
        if not decisions:
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                current_instances=self.current_instances,
                target_instances=self.current_instances,
                trigger=ScalingTrigger.CPU_THRESHOLD,
                confidence=1.0,
                reasoning="All metrics within normal ranges",
                timestamp=current_time
            )
        
        # Prioritize scale-up decisions and choose highest confidence
        scale_up_decisions = [d for d in decisions if d.action == ScalingAction.SCALE_UP]
        if scale_up_decisions:
            return max(scale_up_decisions, key=lambda d: d.confidence)
        
        scale_down_decisions = [d for d in decisions if d.action == ScalingAction.SCALE_DOWN]
        if scale_down_decisions:
            return max(scale_down_decisions, key=lambda d: d.confidence)
        
        return decisions[0]  # Fallback
    
    def execute_scaling(self, decision: ScalingDecision) -> bool:
        """Execute a scaling decision."""
        
        if decision.action == ScalingAction.NO_ACTION:
            return True
        
        # Check instance limits
        if decision.target_instances < self.config.min_instances:
            decision.target_instances = self.config.min_instances
        elif decision.target_instances > self.config.max_instances:
            decision.target_instances = self.config.max_instances
        
        # Check safety limits
        instances_change = abs(decision.target_instances - self.current_instances)
        
        if decision.action == ScalingAction.SCALE_UP:
            if instances_change > self.config.max_scale_up_per_action:
                decision.target_instances = self.current_instances + self.config.max_scale_up_per_action
        else:  # SCALE_DOWN
            if instances_change > self.config.max_scale_down_per_action:
                decision.target_instances = self.current_instances - self.config.max_scale_down_per_action
        
        # Execute scaling
        success = False
        
        if decision.action == ScalingAction.SCALE_UP and self.scale_up_callback:
            success = self.scale_up_callback(decision.target_instances)
            if success:
                self.stats['total_scale_ups'] += 1
                
        elif decision.action == ScalingAction.SCALE_DOWN and self.scale_down_callback:
            success = self.scale_down_callback(decision.target_instances)
            if success:
                self.stats['total_scale_downs'] += 1
        
        if success:
            self.current_instances = decision.target_instances
            self.last_scaling_time = time.time()
            self.scaling_history.append(decision)
            
            self.logger.info(
                f"Scaling executed: {decision.action.value} "
                f"to {decision.target_instances} instances "
                f"(trigger: {decision.trigger.value}, confidence: {decision.confidence:.2f})"
            )
        else:
            self.logger.error(f"Scaling failed: {decision.action.value}")
        
        return success
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        
        while self.running:
            try:
                # Collect metrics if health monitor is available
                if self.health_monitor:
                    self._collect_metrics_from_monitor()
                
                # Evaluate scaling decision
                decision = self.evaluate_scaling()
                
                # Execute if needed
                if decision.action != ScalingAction.NO_ACTION:
                    self.execute_scaling(decision)
                
                # Sleep until next evaluation
                time.sleep(self.config.evaluation_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.config.evaluation_interval)
    
    def _collect_metrics_from_monitor(self):
        """Collect metrics from health monitor."""
        
        try:
            summary = self.health_monitor.get_health_summary()
            system_metrics = summary.get('system_metrics', {})
            
            self.add_metrics(
                cpu_percent=system_metrics.get('cpu_percent', 0.0),
                memory_percent=system_metrics.get('memory_percent', 0.0),
                request_rate=10.0,  # Would come from API metrics
                response_time=0.1,  # Would come from API metrics
                queue_length=0      # Would come from processing queue
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics from monitor: {e}")
    
    def _check_emergency_scaling(self, cpu_percent: float, memory_percent: float) -> ScalingDecision:
        """Check for emergency scaling conditions."""
        
        emergency_threshold = self.config.emergency_scale_up_threshold
        
        if cpu_percent > emergency_threshold or memory_percent > emergency_threshold:
            target_instances = min(
                self.config.max_instances,
                self.current_instances + self.config.max_scale_up_per_action
            )
            
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                current_instances=self.current_instances,
                target_instances=target_instances,
                trigger=ScalingTrigger.CPU_THRESHOLD if cpu_percent > emergency_threshold else ScalingTrigger.MEMORY_THRESHOLD,
                confidence=1.0,
                reasoning=f"Emergency scaling: {'CPU' if cpu_percent > emergency_threshold else 'Memory'} at {cpu_percent if cpu_percent > emergency_threshold else memory_percent:.1f}%",
                timestamp=time.time()
            )
        
        return ScalingDecision(
            action=ScalingAction.NO_ACTION,
            current_instances=self.current_instances,
            target_instances=self.current_instances,
            trigger=ScalingTrigger.CPU_THRESHOLD,
            confidence=0.0,
            reasoning="No emergency conditions",
            timestamp=time.time()
        )
    
    def _evaluate_cpu_scaling(self, avg_cpu: float) -> ScalingDecision:
        """Evaluate CPU-based scaling."""
        
        current_time = time.time()
        
        if avg_cpu > self.config.cpu_threshold_up:
            target_instances = min(
                self.config.max_instances,
                int(self.current_instances * self.config.scale_up_factor)
            )
            
            confidence = min(1.0, (avg_cpu - self.config.cpu_threshold_up) / 20.0)
            
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                current_instances=self.current_instances,
                target_instances=target_instances,
                trigger=ScalingTrigger.CPU_THRESHOLD,
                confidence=confidence,
                reasoning=f"High CPU usage: {avg_cpu:.1f}%",
                timestamp=current_time
            )
        
        elif avg_cpu < self.config.cpu_threshold_down and self.current_instances > self.config.min_instances:
            target_instances = max(
                self.config.min_instances,
                int(self.current_instances * self.config.scale_down_factor)
            )
            
            confidence = min(1.0, (self.config.cpu_threshold_down - avg_cpu) / 20.0)
            
            return ScalingDecision(
                action=ScalingAction.SCALE_DOWN,
                current_instances=self.current_instances,
                target_instances=target_instances,
                trigger=ScalingTrigger.CPU_THRESHOLD,
                confidence=confidence,
                reasoning=f"Low CPU usage: {avg_cpu:.1f}%",
                timestamp=current_time
            )
        
        return ScalingDecision(
            action=ScalingAction.NO_ACTION,
            current_instances=self.current_instances,
            target_instances=self.current_instances,
            trigger=ScalingTrigger.CPU_THRESHOLD,
            confidence=0.0,
            reasoning="CPU within normal range",
            timestamp=current_time
        )
    
    def _evaluate_memory_scaling(self, avg_memory: float) -> ScalingDecision:
        """Evaluate memory-based scaling."""
        
        current_time = time.time()
        
        if avg_memory > self.config.memory_threshold_up:
            target_instances = min(
                self.config.max_instances,
                int(self.current_instances * self.config.scale_up_factor)
            )
            
            confidence = min(1.0, (avg_memory - self.config.memory_threshold_up) / 15.0)
            
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                current_instances=self.current_instances,
                target_instances=target_instances,
                trigger=ScalingTrigger.MEMORY_THRESHOLD,
                confidence=confidence,
                reasoning=f"High memory usage: {avg_memory:.1f}%",
                timestamp=current_time
            )
        
        elif avg_memory < self.config.memory_threshold_down and self.current_instances > self.config.min_instances:
            target_instances = max(
                self.config.min_instances,
                int(self.current_instances * self.config.scale_down_factor)
            )
            
            confidence = min(1.0, (self.config.memory_threshold_down - avg_memory) / 20.0)
            
            return ScalingDecision(
                action=ScalingAction.SCALE_DOWN,
                current_instances=self.current_instances,
                target_instances=target_instances,
                trigger=ScalingTrigger.MEMORY_THRESHOLD,
                confidence=confidence,
                reasoning=f"Low memory usage: {avg_memory:.1f}%",
                timestamp=current_time
            )
        
        return ScalingDecision(
            action=ScalingAction.NO_ACTION,
            current_instances=self.current_instances,
            target_instances=self.current_instances,
            trigger=ScalingTrigger.MEMORY_THRESHOLD,
            confidence=0.0,
            reasoning="Memory within normal range",
            timestamp=current_time
        )
    
    def _evaluate_response_time_scaling(self, avg_response_time: float) -> ScalingDecision:
        """Evaluate response time-based scaling."""
        
        current_time = time.time()
        
        if avg_response_time > self.config.response_time_threshold:
            target_instances = min(
                self.config.max_instances,
                self.current_instances + 1
            )
            
            confidence = min(1.0, (avg_response_time - self.config.response_time_threshold) / 2.0)
            
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                current_instances=self.current_instances,
                target_instances=target_instances,
                trigger=ScalingTrigger.RESPONSE_TIME,
                confidence=confidence,
                reasoning=f"High response time: {avg_response_time:.3f}s",
                timestamp=current_time
            )
        
        return ScalingDecision(
            action=ScalingAction.NO_ACTION,
            current_instances=self.current_instances,
            target_instances=self.current_instances,
            trigger=ScalingTrigger.RESPONSE_TIME,
            confidence=0.0,
            reasoning="Response time within acceptable range",
            timestamp=current_time
        )
    
    def _evaluate_queue_scaling(self, avg_queue_length: float) -> ScalingDecision:
        """Evaluate queue length-based scaling."""
        
        current_time = time.time()
        
        if avg_queue_length > self.config.queue_length_threshold:
            target_instances = min(
                self.config.max_instances,
                self.current_instances + int(avg_queue_length / self.config.queue_length_threshold)
            )
            
            confidence = min(1.0, avg_queue_length / (2 * self.config.queue_length_threshold))
            
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                current_instances=self.current_instances,
                target_instances=target_instances,
                trigger=ScalingTrigger.QUEUE_LENGTH,
                confidence=confidence,
                reasoning=f"High queue length: {avg_queue_length:.0f}",
                timestamp=current_time
            )
        
        return ScalingDecision(
            action=ScalingAction.NO_ACTION,
            current_instances=self.current_instances,
            target_instances=self.current_instances,
            trigger=ScalingTrigger.QUEUE_LENGTH,
            confidence=0.0,
            reasoning="Queue length within acceptable range",
            timestamp=current_time
        )
    
    def _evaluate_predictive_scaling(self) -> ScalingDecision:
        """Evaluate predictive scaling based on trends."""
        
        current_time = time.time()
        
        if not self.predictor:
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                current_instances=self.current_instances,
                target_instances=self.current_instances,
                trigger=ScalingTrigger.PREDICTIVE,
                confidence=0.0,
                reasoning="Predictor not available",
                timestamp=current_time
            )
        
        # Predict metrics for the prediction window
        pred_cpu, cpu_confidence = self.predictor.predict_cpu(self.config.prediction_window)
        pred_memory, memory_confidence = self.predictor.predict_memory(self.config.prediction_window)
        
        # Only act on predictions with high confidence
        min_confidence = self.config.prediction_accuracy_threshold
        
        if cpu_confidence > min_confidence and pred_cpu > self.config.cpu_threshold_up:
            target_instances = min(
                self.config.max_instances,
                int(self.current_instances * 1.5)  # Conservative predictive scaling
            )
            
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                current_instances=self.current_instances,
                target_instances=target_instances,
                trigger=ScalingTrigger.PREDICTIVE,
                confidence=cpu_confidence,
                reasoning=f"Predicted high CPU: {pred_cpu:.1f}% in {self.config.prediction_window}s",
                timestamp=current_time
            )
        
        elif memory_confidence > min_confidence and pred_memory > self.config.memory_threshold_up:
            target_instances = min(
                self.config.max_instances,
                int(self.current_instances * 1.5)
            )
            
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                current_instances=self.current_instances,
                target_instances=target_instances,
                trigger=ScalingTrigger.PREDICTIVE,
                confidence=memory_confidence,
                reasoning=f"Predicted high memory: {pred_memory:.1f}% in {self.config.prediction_window}s",
                timestamp=current_time
            )
        
        return ScalingDecision(
            action=ScalingAction.NO_ACTION,
            current_instances=self.current_instances,
            target_instances=self.current_instances,
            trigger=ScalingTrigger.PREDICTIVE,
            confidence=max(cpu_confidence, memory_confidence),
            reasoning="No predictive scaling triggers",
            timestamp=current_time
        )
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics and history."""
        
        return {
            'current_instances': self.current_instances,
            'target_instances': self.config.target_instances,
            'min_instances': self.config.min_instances,
            'max_instances': self.config.max_instances,
            'last_scaling_time': self.last_scaling_time,
            'cooldown_remaining': max(0, self.config.cooldown_period - (time.time() - self.last_scaling_time)),
            'total_scale_ups': self.stats['total_scale_ups'],
            'total_scale_downs': self.stats['total_scale_downs'],
            'recent_decisions': [d.to_dict() for d in list(self.scaling_history)[-10:]],
            'metrics_history_size': len(self.metrics_history),
            'running': self.running
        }


# Export main classes
__all__ = [
    "AutoScaler",
    "ScalingConfig",
    "ScalingDecision",
    "ScalingAction",
    "ScalingTrigger",
    "MetricPoint",
    "MetricsPredictor"
]