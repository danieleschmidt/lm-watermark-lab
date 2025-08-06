"""Quantum-inspired task planning for watermarking operations."""

import math
import random
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import time
import json
from collections import defaultdict, deque

from ..utils.exceptions import PlanningError, ValidationError
from ..utils.validation import validate_text, validate_positive_integer
from ..utils.logging import get_logger
from ..utils.metrics import record_operation_metric


class TaskPriority(Enum):
    """Task priority levels inspired by quantum energy states."""
    GROUND_STATE = 0      # Highest priority (ground state)
    EXCITED_1 = 1         # High priority
    EXCITED_2 = 2         # Medium priority  
    EXCITED_3 = 3         # Low priority
    SUPERPOSITION = 4     # Adaptive priority


class TaskState(Enum):
    """Task execution states with quantum properties."""
    SUPERPOSITION = "superposition"    # Task in multiple potential states
    ENTANGLED = "entangled"           # Task dependent on others
    COLLAPSED = "collapsed"           # Task state determined
    DECOHERENT = "decoherent"        # Task failed/corrupted
    COMPLETED = "completed"          # Task successfully finished


@dataclass
class QuantumTask:
    """A task with quantum-inspired properties."""
    id: str
    name: str
    description: str
    priority: TaskPriority
    state: TaskState = TaskState.SUPERPOSITION
    dependencies: List[str] = field(default_factory=list)
    entangled_tasks: List[str] = field(default_factory=list)
    probability_amplitude: complex = complex(1.0, 0.0)
    coherence_time: float = 100.0  # Time before decoherence
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Initialize quantum properties."""
        if not self.id:
            self.id = self._generate_task_id()
    
    def _generate_task_id(self) -> str:
        """Generate unique task ID using quantum-inspired hashing."""
        data = f"{self.name}_{self.description}_{time.time()}_{random.random()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def collapse_state(self, target_state: TaskState) -> None:
        """Collapse task from superposition to definite state."""
        self.state = target_state
        self.probability_amplitude = complex(1.0, 0.0) if target_state != TaskState.DECOHERENT else complex(0.0, 0.0)
    
    def calculate_probability(self) -> float:
        """Calculate task success probability from amplitude."""
        return abs(self.probability_amplitude) ** 2
    
    def evolve_amplitude(self, time_delta: float) -> None:
        """Evolve probability amplitude over time."""
        # Quantum evolution with phase rotation
        phase = -time_delta / self.coherence_time
        rotation = complex(math.cos(phase), math.sin(phase))
        self.probability_amplitude *= rotation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority.value,
            "state": self.state.value,
            "dependencies": self.dependencies,
            "entangled_tasks": self.entangled_tasks,
            "probability": self.calculate_probability(),
            "coherence_time": self.coherence_time,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "created_at": self.created_at
        }


@dataclass
class ExecutionResult:
    """Result of task execution."""
    task_id: str
    success: bool
    duration: float
    output: Any = None
    error: Optional[str] = None
    quantum_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class QuantumExecutor(ABC):
    """Abstract base class for task executors."""
    
    @abstractmethod
    def execute(self, task: QuantumTask, context: Dict[str, Any]) -> ExecutionResult:
        """Execute a quantum task."""
        pass
    
    @abstractmethod
    def can_execute(self, task: QuantumTask) -> bool:
        """Check if executor can handle this task."""
        pass


class WatermarkExecutor(QuantumExecutor):
    """Executor for watermarking tasks."""
    
    def __init__(self):
        self.logger = get_logger("quantum.watermark_executor")
    
    def can_execute(self, task: QuantumTask) -> bool:
        """Check if task is watermarking-related."""
        watermark_keywords = ["watermark", "generate", "embed", "detection", "kirchenbauer", "markllm"]
        task_text = f"{task.name} {task.description}".lower()
        return any(keyword in task_text for keyword in watermark_keywords)
    
    def execute(self, task: QuantumTask, context: Dict[str, Any]) -> ExecutionResult:
        """Execute watermarking task."""
        start_time = time.time()
        
        try:
            # Simulate watermark generation based on task
            if "generate" in task.name.lower():
                result = self._generate_watermark(task, context)
            elif "detect" in task.name.lower():
                result = self._detect_watermark(task, context)
            elif "benchmark" in task.name.lower():
                result = self._run_benchmark(task, context)
            else:
                result = self._generic_watermark_task(task, context)
            
            duration = time.time() - start_time
            task.execution_time = duration
            
            # Calculate quantum metrics
            coherence_loss = duration / task.coherence_time
            quantum_efficiency = max(0.0, 1.0 - coherence_loss)
            
            return ExecutionResult(
                task_id=task.id,
                success=True,
                duration=duration,
                output=result,
                quantum_metrics={
                    "coherence_loss": coherence_loss,
                    "quantum_efficiency": quantum_efficiency,
                    "probability": task.calculate_probability()
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Watermark task execution failed: {e}")
            
            return ExecutionResult(
                task_id=task.id,
                success=False,
                duration=duration,
                error=str(e),
                quantum_metrics={"coherence_loss": 1.0, "quantum_efficiency": 0.0}
            )
    
    def _generate_watermark(self, task: QuantumTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate watermarked text."""
        prompt = context.get("prompt", "The future of AI involves")
        method = context.get("method", "kirchenbauer")
        
        # Simulate watermark generation
        watermarked_text = f"{prompt} quantum-enhanced watermarked content using {method} method"
        
        return {
            "type": "watermark_generation",
            "method": method,
            "prompt": prompt,
            "watermarked_text": watermarked_text,
            "quantum_properties": {
                "coherence": task.calculate_probability(),
                "entanglement_strength": len(task.entangled_tasks)
            }
        }
    
    def _detect_watermark(self, task: QuantumTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect watermark in text."""
        text = context.get("text", "")
        method = context.get("method", "kirchenbauer")
        
        # Simulate detection with quantum enhancement
        detection_probability = task.calculate_probability() * 0.95  # Quantum-enhanced accuracy
        is_detected = detection_probability > 0.5
        
        return {
            "type": "watermark_detection",
            "method": method,
            "text_length": len(text),
            "is_watermarked": is_detected,
            "confidence": detection_probability,
            "quantum_enhancement": task.calculate_probability()
        }
    
    def _run_benchmark(self, task: QuantumTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run watermarking benchmark."""
        methods = context.get("methods", ["kirchenbauer", "markllm"])
        num_samples = context.get("num_samples", 10)
        
        # Simulate quantum-enhanced benchmarking
        results = {}
        for method in methods:
            quantum_boost = task.calculate_probability()
            base_accuracy = 0.85 + random.uniform(-0.1, 0.1)
            enhanced_accuracy = min(0.99, base_accuracy + quantum_boost * 0.1)
            
            results[method] = {
                "accuracy": enhanced_accuracy,
                "quantum_enhancement": quantum_boost,
                "samples_processed": num_samples
            }
        
        return {
            "type": "quantum_benchmark",
            "methods": methods,
            "results": results,
            "quantum_properties": {
                "superposition_utilized": len(methods) > 1,
                "entanglement_factor": len(task.entangled_tasks)
            }
        }
    
    def _generic_watermark_task(self, task: QuantumTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic watermarking task."""
        return {
            "type": "generic_watermark",
            "task_name": task.name,
            "quantum_state": task.state.value,
            "execution_probability": task.calculate_probability(),
            "context_keys": list(context.keys())
        }


class QuantumTaskPlanner:
    """Quantum-inspired task planner for watermarking operations."""
    
    def __init__(self, max_coherence_time: float = 1000.0):
        """Initialize quantum task planner."""
        self.tasks: Dict[str, QuantumTask] = {}
        self.execution_queue: deque = deque()
        self.dependency_graph: Dict[str, List[str]] = defaultdict(list)
        self.entanglement_groups: Dict[str, List[str]] = defaultdict(list)
        self.executors: List[QuantumExecutor] = []
        self.max_coherence_time = max_coherence_time
        self.logger = get_logger("quantum.planner")
        
        # Register default executors
        self.register_executor(WatermarkExecutor())
        
        # Quantum state tracking
        self.global_quantum_state = {
            "total_coherence": 1.0,
            "entanglement_strength": 0.0,
            "system_energy": 0.0
        }
    
    def register_executor(self, executor: QuantumExecutor) -> None:
        """Register a quantum executor."""
        self.executors.append(executor)
        self.logger.info(f"Registered executor: {executor.__class__.__name__}")
    
    def add_task(self, 
                 name: str,
                 description: str,
                 priority: TaskPriority = TaskPriority.EXCITED_2,
                 dependencies: Optional[List[str]] = None,
                 coherence_time: Optional[float] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a quantum task to the planner."""
        
        task = QuantumTask(
            id="",  # Will be auto-generated
            name=name,
            description=description,
            priority=priority,
            dependencies=dependencies or [],
            coherence_time=coherence_time or self.max_coherence_time,
            metadata=metadata or {}
        )
        
        self.tasks[task.id] = task
        
        # Update dependency graph
        for dep_id in task.dependencies:
            self.dependency_graph[dep_id].append(task.id)
        
        self.logger.info(f"Added quantum task: {name} (ID: {task.id})")
        return task.id
    
    def entangle_tasks(self, task_ids: List[str], group_name: str = None) -> None:
        """Create quantum entanglement between tasks."""
        if len(task_ids) < 2:
            raise PlanningError("At least 2 tasks required for entanglement")
        
        group_id = group_name or f"entanglement_{time.time()}"
        self.entanglement_groups[group_id] = task_ids
        
        # Update task entanglement information
        for task_id in task_ids:
            if task_id in self.tasks:
                self.tasks[task_id].entangled_tasks = [tid for tid in task_ids if tid != task_id]
                self.tasks[task_id].state = TaskState.ENTANGLED
        
        # Update global quantum state
        self.global_quantum_state["entanglement_strength"] += len(task_ids) * 0.1
        
        self.logger.info(f"Entangled {len(task_ids)} tasks in group: {group_id}")
    
    def plan_execution(self) -> List[str]:
        """Plan optimal execution order using quantum-inspired algorithms."""
        
        # Update quantum states over time
        self._evolve_quantum_system()
        
        # Build execution plan using quantum superposition
        execution_plan = []
        available_tasks = self._get_available_tasks()
        
        while available_tasks:
            # Use quantum-inspired selection
            selected_task = self._quantum_task_selection(available_tasks)
            
            if selected_task:
                execution_plan.append(selected_task.id)
                selected_task.collapse_state(TaskState.COLLAPSED)
                
                # Update available tasks
                available_tasks = self._get_available_tasks()
            else:
                break
        
        self.logger.info(f"Generated quantum execution plan with {len(execution_plan)} tasks")
        return execution_plan
    
    def execute_plan(self, execution_plan: List[str], context: Dict[str, Any] = None) -> List[ExecutionResult]:
        """Execute the planned tasks using quantum executors."""
        context = context or {}
        results = []
        
        for task_id in execution_plan:
            if task_id not in self.tasks:
                self.logger.error(f"Task {task_id} not found")
                continue
            
            task = self.tasks[task_id]
            
            # Find suitable executor
            executor = self._find_executor(task)
            if not executor:
                self.logger.error(f"No executor found for task {task_id}")
                continue
            
            # Execute task
            self.logger.info(f"Executing quantum task: {task.name}")
            result = executor.execute(task, context)
            results.append(result)
            
            # Update task state based on result
            if result.success:
                task.collapse_state(TaskState.COMPLETED)
            else:
                task.collapse_state(TaskState.DECOHERENT)
            
            # Record metrics
            record_operation_metric(
                f"quantum_task_execution",
                result.duration,
                success=result.success,
                task_type=task.name
            )
        
        return results
    
    def _evolve_quantum_system(self) -> None:
        """Evolve the quantum system over time."""
        current_time = time.time()
        
        for task in self.tasks.values():
            if task.state == TaskState.SUPERPOSITION:
                time_delta = current_time - task.created_at
                task.evolve_amplitude(time_delta)
                
                # Check for decoherence
                if time_delta > task.coherence_time:
                    task.collapse_state(TaskState.DECOHERENT)
        
        # Update global coherence
        active_tasks = [t for t in self.tasks.values() if t.state in [TaskState.SUPERPOSITION, TaskState.ENTANGLED]]
        if active_tasks:
            avg_coherence = sum(t.calculate_probability() for t in active_tasks) / len(active_tasks)
            self.global_quantum_state["total_coherence"] = avg_coherence
    
    def _get_available_tasks(self) -> List[QuantumTask]:
        """Get tasks that can be executed (dependencies satisfied)."""
        available = []
        
        for task in self.tasks.values():
            if task.state in [TaskState.SUPERPOSITION, TaskState.ENTANGLED]:
                # Check if all dependencies are completed
                deps_satisfied = all(
                    self.tasks.get(dep_id, QuantumTask("", "", "", TaskPriority.GROUND_STATE)).state == TaskState.COMPLETED
                    for dep_id in task.dependencies
                )
                
                if deps_satisfied:
                    available.append(task)
        
        return available
    
    def _quantum_task_selection(self, available_tasks: List[QuantumTask]) -> Optional[QuantumTask]:
        """Select next task using quantum-inspired algorithm."""
        if not available_tasks:
            return None
        
        # Calculate quantum weights
        weights = []
        for task in available_tasks:
            # Combine priority, probability amplitude, and quantum properties
            priority_weight = (4 - task.priority.value) / 4.0  # Higher priority = higher weight
            probability_weight = task.calculate_probability()
            entanglement_weight = len(task.entangled_tasks) * 0.1
            
            total_weight = priority_weight * probability_weight + entanglement_weight
            weights.append(total_weight)
        
        # Weighted random selection (quantum measurement)
        if sum(weights) == 0:
            return random.choice(available_tasks)
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Quantum measurement - select based on probability amplitudes
        rand = random.random()
        cumulative = 0.0
        
        for i, weight in enumerate(normalized_weights):
            cumulative += weight
            if rand <= cumulative:
                return available_tasks[i]
        
        return available_tasks[-1]  # Fallback
    
    def _find_executor(self, task: QuantumTask) -> Optional[QuantumExecutor]:
        """Find suitable executor for task."""
        for executor in self.executors:
            if executor.can_execute(task):
                return executor
        return None
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get current quantum system state."""
        task_states = defaultdict(int)
        for task in self.tasks.values():
            task_states[task.state.value] += 1
        
        return {
            "total_tasks": len(self.tasks),
            "task_states": dict(task_states),
            "entanglement_groups": len(self.entanglement_groups),
            "global_quantum_state": self.global_quantum_state.copy(),
            "active_coherence": sum(t.calculate_probability() for t in self.tasks.values() 
                                  if t.state == TaskState.SUPERPOSITION)
        }
    
    def optimize_plan(self, criteria: str = "coherence") -> Dict[str, Any]:
        """Optimize task execution using quantum algorithms."""
        
        if criteria == "coherence":
            return self._optimize_for_coherence()
        elif criteria == "entanglement":
            return self._optimize_for_entanglement()
        elif criteria == "energy":
            return self._optimize_for_energy()
        else:
            return {"error": f"Unknown optimization criteria: {criteria}"}
    
    def _optimize_for_coherence(self) -> Dict[str, Any]:
        """Optimize plan to maximize quantum coherence."""
        optimization_results = {}
        
        # Find tasks with highest coherence potential
        high_coherence_tasks = [
            task for task in self.tasks.values() 
            if task.calculate_probability() > 0.7 and task.state == TaskState.SUPERPOSITION
        ]
        
        if high_coherence_tasks:
            # Prioritize high-coherence tasks
            for task in high_coherence_tasks:
                task.priority = TaskPriority.GROUND_STATE
            
            optimization_results["optimized_tasks"] = len(high_coherence_tasks)
            optimization_results["coherence_boost"] = 0.15
        
        optimization_results["strategy"] = "coherence_maximization"
        return optimization_results
    
    def _optimize_for_entanglement(self) -> Dict[str, Any]:
        """Optimize plan to maximize quantum entanglement benefits."""
        optimization_results = {}
        
        # Group related tasks for entanglement
        watermark_tasks = [t for t in self.tasks.values() if "watermark" in t.name.lower()]
        detection_tasks = [t for t in self.tasks.values() if "detect" in t.name.lower()]
        
        if len(watermark_tasks) >= 2:
            self.entangle_tasks([t.id for t in watermark_tasks[:3]], "watermark_group")
            optimization_results["watermark_entanglement"] = True
        
        if len(detection_tasks) >= 2:
            self.entangle_tasks([t.id for t in detection_tasks[:3]], "detection_group")
            optimization_results["detection_entanglement"] = True
        
        optimization_results["strategy"] = "entanglement_maximization"
        return optimization_results
    
    def _optimize_for_energy(self) -> Dict[str, Any]:
        """Optimize plan to minimize quantum energy (execution time)."""
        optimization_results = {}
        
        # Prioritize shorter tasks and parallelize when possible
        for task in self.tasks.values():
            if task.coherence_time < 50.0:  # Short tasks
                if task.priority.value > TaskPriority.EXCITED_1.value:
                    task.priority = TaskPriority.EXCITED_1
        
        optimization_results["strategy"] = "energy_minimization"
        optimization_results["priority_adjustments"] = len([
            t for t in self.tasks.values() if t.priority == TaskPriority.EXCITED_1
        ])
        
        return optimization_results
    
    def export_plan(self, filename: str) -> None:
        """Export quantum execution plan to file."""
        plan_data = {
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "entanglement_groups": dict(self.entanglement_groups),
            "system_state": self.get_system_state(),
            "export_timestamp": time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(plan_data, f, indent=2)
        
        self.logger.info(f"Quantum plan exported to {filename}")
    
    def import_plan(self, filename: str) -> None:
        """Import quantum execution plan from file."""
        with open(filename, 'r') as f:
            plan_data = json.load(f)
        
        # Reconstruct tasks
        for task_id, task_dict in plan_data.get("tasks", {}).items():
            task = QuantumTask(
                id=task_id,
                name=task_dict["name"],
                description=task_dict["description"],
                priority=TaskPriority(task_dict["priority"]),
                state=TaskState(task_dict["state"]),
                dependencies=task_dict["dependencies"],
                entangled_tasks=task_dict["entangled_tasks"],
                coherence_time=task_dict["coherence_time"],
                execution_time=task_dict["execution_time"],
                metadata=task_dict["metadata"],
                created_at=task_dict["created_at"]
            )
            self.tasks[task_id] = task
        
        # Reconstruct entanglement groups
        self.entanglement_groups.update(plan_data.get("entanglement_groups", {}))
        
        self.logger.info(f"Quantum plan imported from {filename}")


def create_watermarking_workflow(planner: QuantumTaskPlanner, 
                                config: Dict[str, Any]) -> List[str]:
    """Create a quantum-enhanced watermarking workflow."""
    
    # Extract configuration
    methods = config.get("methods", ["kirchenbauer", "markllm"])
    prompts = config.get("prompts", ["The future of AI involves"])
    include_detection = config.get("include_detection", True)
    include_benchmark = config.get("include_benchmark", True)
    
    task_ids = []
    
    # Add generation tasks for each method
    for method in methods:
        task_id = planner.add_task(
            name=f"Generate {method} watermark",
            description=f"Generate watermarked text using {method} method",
            priority=TaskPriority.EXCITED_1,
            coherence_time=200.0,
            metadata={"method": method, "type": "generation"}
        )
        task_ids.append(task_id)
    
    # Add detection tasks if requested
    if include_detection:
        detection_deps = task_ids.copy()  # Detection depends on generation
        for method in methods:
            task_id = planner.add_task(
                name=f"Detect {method} watermark",
                description=f"Detect watermarks using {method} method",
                priority=TaskPriority.EXCITED_2,
                dependencies=detection_deps,
                coherence_time=150.0,
                metadata={"method": method, "type": "detection"}
            )
            task_ids.append(task_id)
    
    # Add benchmark task if requested
    if include_benchmark:
        benchmark_deps = task_ids.copy()  # Benchmark depends on all previous tasks
        task_id = planner.add_task(
            name="Quantum benchmark",
            description="Compare watermarking methods with quantum enhancement",
            priority=TaskPriority.GROUND_STATE,  # Highest priority
            dependencies=benchmark_deps,
            coherence_time=300.0,
            metadata={"methods": methods, "type": "benchmark"}
        )
        task_ids.append(task_id)
    
    # Create quantum entanglement between related tasks
    generation_tasks = [tid for tid in task_ids if "Generate" in planner.tasks[tid].name]
    if len(generation_tasks) >= 2:
        planner.entangle_tasks(generation_tasks, "generation_entanglement")
    
    detection_tasks = [tid for tid in task_ids if "Detect" in planner.tasks[tid].name]
    if len(detection_tasks) >= 2:
        planner.entangle_tasks(detection_tasks, "detection_entanglement")
    
    return task_ids