"""FastAPI main application."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time
import uuid
from datetime import datetime

from ..core.factory import WatermarkFactory
from ..core.detector import WatermarkDetector
from ..core.benchmark import WatermarkBenchmark
from ..core.attacks import AttackSimulator

# Initialize FastAPI app
app = FastAPI(
    title="LM Watermark Lab API",
    description="Comprehensive toolkit for watermarking, detecting, and attacking LLM-generated text",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for async tasks (in production, use Redis/database)
task_storage = {}


# Pydantic models
class WatermarkRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for text generation")
    method: str = Field(default="kirchenbauer", description="Watermarking method")
    max_length: int = Field(default=100, description="Maximum generated text length")
    config: Dict[str, Any] = Field(default_factory=dict, description="Method-specific configuration")


class WatermarkResponse(BaseModel):
    watermarked_text: str
    method: str
    config: Dict[str, Any]
    generation_time: float
    timestamp: datetime


class DetectionRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for watermarks")
    method: str = Field(default="kirchenbauer", description="Detection method")
    config: Dict[str, Any] = Field(default_factory=dict, description="Detection configuration")


class DetectionResponse(BaseModel):
    is_watermarked: bool
    confidence: float
    p_value: float
    test_statistic: float
    method: str
    details: Dict[str, Any]
    analysis_time: float
    timestamp: datetime


class BenchmarkRequest(BaseModel):
    methods: List[str] = Field(default=["kirchenbauer", "markllm"], description="Methods to compare")
    num_samples: int = Field(default=10, description="Number of test samples")
    metrics: List[str] = Field(default=["detectability", "quality", "robustness"], description="Metrics to evaluate")
    custom_prompts: Optional[List[str]] = Field(default=None, description="Custom test prompts")


class AttackRequest(BaseModel):
    text: str = Field(..., description="Text to attack")
    attack_type: str = Field(default="paraphrase", description="Type of attack")
    strength: str = Field(default="medium", description="Attack strength")
    config: Dict[str, Any] = Field(default_factory=dict, description="Attack-specific configuration")


class TaskStatus(BaseModel):
    task_id: str
    status: str  # "running", "completed", "failed"
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LM Watermark Lab API",
        "version": "1.0.0",
        "description": "Comprehensive watermarking toolkit for LLM-generated text",
        "endpoints": {
            "watermark": "/watermark - Generate watermarked text",
            "detect": "/detect - Detect watermarks in text", 
            "benchmark": "/benchmark - Compare watermarking methods",
            "attack": "/attack - Test attack resistance",
            "methods": "/methods - List available methods",
            "health": "/health - API health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }


@app.get("/methods")
async def list_methods():
    """List available watermarking methods."""
    try:
        methods = WatermarkFactory.list_methods()
        method_info = {
            "kirchenbauer": {
                "name": "Kirchenbauer et al.",
                "description": "Statistical watermarking with green/red lists",
                "parameters": ["gamma", "delta", "seed"]
            },
            "markllm": {
                "name": "MarkLLM",
                "description": "Key-based watermarking toolkit",
                "parameters": ["algorithm", "watermark_strength", "key"]
            },
            "aaronson": {
                "name": "Aaronson",
                "description": "Cryptographic pseudorandom watermarking",
                "parameters": ["secret_key", "threshold"]
            },
            "zhao": {
                "name": "Zhao et al.",
                "description": "Robust multi-bit watermarking",
                "parameters": ["message_bits", "redundancy"]
            }
        }
        
        return {
            "methods": methods,
            "details": {method: method_info.get(method, {}) for method in methods}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing methods: {str(e)}")


@app.post("/watermark", response_model=WatermarkResponse)
async def generate_watermark(request: WatermarkRequest):
    """Generate watermarked text."""
    try:
        start_time = time.time()
        
        # Create watermark instance
        watermarker = WatermarkFactory.create(request.method, **request.config)
        
        # Generate watermarked text
        watermarked_text = watermarker.generate(
            request.prompt,
            max_length=request.max_length
        )
        
        generation_time = time.time() - start_time
        
        return WatermarkResponse(
            watermarked_text=watermarked_text,
            method=request.method,
            config=watermarker.get_config(),
            generation_time=generation_time,
            timestamp=datetime.now()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Watermarking failed: {str(e)}")


@app.post("/detect", response_model=DetectionResponse)
async def detect_watermark(request: DetectionRequest):
    """Detect watermark in text."""
    try:
        start_time = time.time()
        
        # Create detector
        detector_config = {"method": request.method, **request.config}
        detector = WatermarkDetector(detector_config)
        
        # Perform detection
        result = detector.detect(request.text)
        
        analysis_time = time.time() - start_time
        
        return DetectionResponse(
            is_watermarked=result.is_watermarked,
            confidence=result.confidence,
            p_value=result.p_value,
            test_statistic=result.test_statistic,
            method=result.method,
            details=result.to_dict().get("details", {}),
            analysis_time=analysis_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/attack")
async def test_attack(request: AttackRequest):
    """Test attack on watermarked text."""
    try:
        start_time = time.time()
        
        # Create attack simulator
        simulator = AttackSimulator()
        
        # Run attack
        result = simulator.run_attack(
            request.text,
            request.attack_type,
            strength=request.strength,
            **request.config
        )
        
        attack_time = time.time() - start_time
        
        return {
            "original_text": result.original_text,
            "attacked_text": result.attacked_text,
            "attack_type": result.attack_type,
            "success": result.success,
            "quality_score": result.quality_score,
            "similarity_score": result.similarity_score,
            "metadata": result.metadata,
            "attack_time": attack_time,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attack failed: {str(e)}")


@app.post("/benchmark/start")
async def start_benchmark(request: BenchmarkRequest, background_tasks: BackgroundTasks):
    """Start asynchronous benchmark comparison."""
    try:
        task_id = str(uuid.uuid4())
        
        # Initialize task status
        task_storage[task_id] = TaskStatus(
            task_id=task_id,
            status="running",
            progress=0.0,
            created_at=datetime.now()
        )
        
        # Start benchmark in background
        background_tasks.add_task(
            run_benchmark_task,
            task_id,
            request.methods,
            request.num_samples,
            request.metrics,
            request.custom_prompts
        )
        
        return {"task_id": task_id, "status": "started"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start benchmark: {str(e)}")


@app.get("/benchmark/status/{task_id}")
async def get_benchmark_status(task_id: str):
    """Get benchmark task status."""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task_storage[task_id]


async def run_benchmark_task(task_id: str, methods: List[str], num_samples: int, 
                           metrics: List[str], custom_prompts: Optional[List[str]]):
    """Run benchmark task in background."""
    try:
        # Update progress
        task_storage[task_id].progress = 0.1
        
        # Create benchmark suite
        benchmark = WatermarkBenchmark(num_samples=num_samples)
        
        # Update progress
        task_storage[task_id].progress = 0.2
        
        # Prepare prompts
        test_prompts = custom_prompts or benchmark.test_prompts[:num_samples]
        
        # Update progress
        task_storage[task_id].progress = 0.3
        
        # Run comparison
        results = benchmark.compare(methods, test_prompts, metrics)
        
        # Update task as completed
        task_storage[task_id].status = "completed"
        task_storage[task_id].progress = 1.0
        task_storage[task_id].result = results
        task_storage[task_id].completed_at = datetime.now()
        
    except Exception as e:
        # Update task as failed
        task_storage[task_id].status = "failed"
        task_storage[task_id].error = str(e)
        task_storage[task_id].completed_at = datetime.now()


@app.get("/attacks")
async def list_attacks():
    """List available attack types."""
    try:
        simulator = AttackSimulator()
        attacks = simulator.list_attacks()
        
        attack_info = {
            "paraphrase": {
                "description": "Synonym replacement and sentence reordering",
                "strengths": ["light", "medium", "heavy"]
            },
            "truncation": {
                "description": "Remove parts of text",
                "strengths": ["light", "medium", "heavy"]
            },
            "insertion": {
                "description": "Add noise words",
                "strengths": ["light", "medium", "heavy"]
            },
            "substitution": {
                "description": "Character-level substitutions",
                "strengths": ["light", "medium", "heavy"]
            },
            "translation": {
                "description": "Back-translation simulation",
                "strengths": ["light", "medium", "heavy"]
            },
            "combined": {
                "description": "Multiple attack strategies",
                "strengths": ["light", "medium", "heavy"]
            }
        }
        
        return {
            "attacks": attacks,
            "details": {attack: attack_info.get(attack, {}) for attack in attacks}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing attacks: {str(e)}")


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)