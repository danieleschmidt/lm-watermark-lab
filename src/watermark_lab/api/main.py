"""FastAPI main application with production-grade reliability and error handling."""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
from typing import List, Dict, Any, Optional, Union
import time
import uuid
import asyncio
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import traceback

from ..core.factory import WatermarkFactory
from ..core.detector import WatermarkDetector
from ..core.benchmark import WatermarkBenchmark
from ..core.attacks import AttackSimulator
from ..utils.exceptions import (
    WatermarkLabError, ValidationError, SecurityError, 
    TimeoutError, ResourceError, RateLimitError, format_error_response
)
from ..utils.metrics import record_operation_metric
from ..utils.validation import (
    validate_text, validate_config, WATERMARK_CONFIG_SCHEMA,
    DETECTION_CONFIG_SCHEMA, ATTACK_CONFIG_SCHEMA, validate_batch_size
)
from ..utils.circuit_breaker import (
    get_circuit_breaker, CircuitBreakerConfig,
    watermark_circuit_breaker, detection_circuit_breaker
)
from ..utils.retry import retry, WATERMARK_RETRY_CONFIG, DETECTION_RETRY_CONFIG
from ..utils.logging import get_logger, StructuredLogger, LoggingContext
from ..security.input_sanitization import InputSanitizer, SanitizationConfig
from ..security.authentication import (
    get_auth_system, User, UserRole, Permission, 
    AuthenticationError, AuthorizationError
)
from ..monitoring.health_monitor import HealthMonitor
from ..config.settings import get_settings

# Global components
settings = get_settings()
logger = get_logger("api.main")
structured_logger = StructuredLogger("api")
input_sanitizer = InputSanitizer()
health_monitor = HealthMonitor()
auth_system = get_auth_system()

# Request tracking and rate limiting
request_counts = {}
active_requests = set()
MAX_ACTIVE_REQUESTS = settings.max_concurrent_requests

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting LM Watermark Lab API")
    health_monitor.start_monitoring()
    
    # Register health checks
    from ..utils.model_loader import ModelManager
    try:
        model_manager = ModelManager()
        from ..monitoring.health_monitor import ModelHealthCheck
        health_monitor.register_check(ModelHealthCheck(model_manager))
    except Exception as e:
        logger.warning(f"Could not initialize model health check: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LM Watermark Lab API")
    health_monitor.stop_monitoring()

# Initialize FastAPI app
app = FastAPI(
    title="LM Watermark Lab API",
    description="Production-ready comprehensive toolkit for watermarking, detecting, and attacking LLM-generated text",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.debug else ["localhost", "127.0.0.1"]
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Restrict to necessary methods
    allow_headers=["Content-Type", "Authorization", "X-Trace-ID"],
)

# Request tracking middleware
@app.middleware("http")
async def request_tracking_middleware(request: Request, call_next):
    """Track requests and implement basic rate limiting."""
    request_id = str(uuid.uuid4())
    
    # Add to active requests
    if len(active_requests) >= MAX_ACTIVE_REQUESTS:
        return JSONResponse(
            status_code=503,
            content={"error": "Service temporarily unavailable - too many requests"}
        )
    
    active_requests.add(request_id)
    start_time = time.time()
    
    try:
        # Set trace ID in context
        import contextvars
        trace_context = contextvars.copy_context()
        trace_context["trace_id"] = request_id
        
        # Process request
        response = await call_next(request)
        
        # Log request
        duration = time.time() - start_time
        structured_logger.log_api_request(
            endpoint=str(request.url.path),
            method=request.method,
            status_code=response.status_code,
            response_time=duration,
            request_size=int(request.headers.get("content-length", 0))
        )
        
        # Add trace ID to response headers
        response.headers["X-Trace-ID"] = request_id
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Request failed: {e}")
        structured_logger.log_api_request(
            endpoint=str(request.url.path),
            method=request.method,
            status_code=500,
            response_time=duration,
            error=str(e)
        )
        
        return JSONResponse(
            status_code=500,
            content=format_error_response(e, include_traceback=settings.debug)
        )
    finally:
        active_requests.discard(request_id)

# Error handling middleware
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    """Global error handling middleware."""
    try:
        return await call_next(request)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=408,
            content={"error": "Request timeout"}
        )
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=format_error_response(e, include_traceback=settings.debug)
        )

# Global storage for async tasks (in production, use Redis/database)
task_storage = {}

# Task cleanup - remove old completed tasks
async def cleanup_old_tasks():
    """Clean up old completed tasks to prevent memory leaks."""
    cutoff_time = datetime.now() - timedelta(hours=24)
    
    tasks_to_remove = [
        task_id for task_id, task in task_storage.items()
        if task.completed_at and task.completed_at < cutoff_time
    ]
    
    for task_id in tasks_to_remove:
        del task_storage[task_id]
        logger.debug(f"Cleaned up old task: {task_id}")
    
    if tasks_to_remove:
        logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")

# Schedule periodic cleanup
@app.on_event("startup")
async def schedule_cleanup():
    """Schedule periodic task cleanup."""
    import asyncio
    
    async def periodic_cleanup():
        while True:
            try:
                await cleanup_old_tasks()
            except Exception as e:
                logger.error(f"Task cleanup failed: {e}")
            await asyncio.sleep(3600)  # Clean up every hour
    
    asyncio.create_task(periodic_cleanup())

# Dependencies
async def validate_request_size(request: Request):
    """Validate request size to prevent DoS attacks."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=413, detail="Request too large")

async def get_current_user():
    """Get current user (placeholder for authentication)."""
    # TODO: Implement proper authentication
    return {"user_id": "anonymous"}

def get_input_sanitizer() -> InputSanitizer:
    """Get input sanitizer dependency."""
    return input_sanitizer


# Pydantic models
class WatermarkRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for text generation", min_length=1, max_length=10000)
    method: str = Field(default="kirchenbauer", description="Watermarking method")
    max_length: int = Field(default=100, description="Maximum generated text length", ge=1, le=4096)
    config: Dict[str, Any] = Field(default_factory=dict, description="Method-specific configuration")
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Write a short story about artificial intelligence.",
                "method": "kirchenbauer",
                "max_length": 200,
                "config": {"gamma": 0.25, "delta": 2.0}
            }
        }


class WatermarkResponse(BaseModel):
    watermarked_text: str
    method: str
    config: Dict[str, Any]
    generation_time: float
    timestamp: datetime


class DetectionRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for watermarks", min_length=1, max_length=50000)
    method: str = Field(default="kirchenbauer", description="Detection method")
    config: Dict[str, Any] = Field(default_factory=dict, description="Detection configuration")
    batch_id: Optional[str] = Field(default=None, description="Batch processing ID")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "This is a sample text to analyze for watermarks.",
                "method": "kirchenbauer",
                "config": {"threshold": 0.05}
            }
        }


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
    methods: List[str] = Field(default=["kirchenbauer", "markllm"], description="Methods to compare", min_items=1, max_items=10)
    num_samples: int = Field(default=10, description="Number of test samples", ge=1, le=100)
    metrics: List[str] = Field(default=["detectability", "quality", "robustness"], description="Metrics to evaluate")
    custom_prompts: Optional[List[str]] = Field(default=None, description="Custom test prompts", max_items=50)
    timeout: Optional[int] = Field(default=300, description="Benchmark timeout in seconds", ge=1, le=3600)
    
    class Config:
        schema_extra = {
            "example": {
                "methods": ["kirchenbauer", "markllm"],
                "num_samples": 20,
                "metrics": ["detectability", "quality"]
            }
        }


class AttackRequest(BaseModel):
    text: str = Field(..., description="Text to attack", min_length=10, max_length=20000)
    attack_type: str = Field(default="paraphrase", description="Type of attack")
    strength: str = Field(default="medium", description="Attack strength")
    config: Dict[str, Any] = Field(default_factory=dict, description="Attack-specific configuration")
    preserve_semantics: bool = Field(default=True, description="Preserve semantic meaning")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "This is watermarked text that will be attacked.",
                "attack_type": "paraphrase",
                "strength": "medium",
                "preserve_semantics": True
            }
        }


class TaskStatus(BaseModel):
    task_id: str
    status: str  # "running", "completed", "failed", "cancelled"
    progress: float = Field(ge=0.0, le=1.0)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    estimated_time_remaining: Optional[int] = None
    retry_count: int = 0
    
class BatchRequest(BaseModel):
    items: List[Union[WatermarkRequest, DetectionRequest]] = Field(..., min_items=1, max_items=100)
    batch_name: Optional[str] = Field(default=None, description="Optional batch name")
    
class HealthCheckResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    checks: Dict[str, Any]
    uptime_seconds: float


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information and system status."""
    uptime = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    
    return {
        "name": settings.app_name,
        "version": settings.version,
        "description": "Production-ready comprehensive watermarking toolkit for LLM-generated text",
        "status": "operational",
        "uptime_seconds": uptime,
        "endpoints": {
            "watermark": "/watermark - Generate watermarked text",
            "detect": "/detect - Detect watermarks in text", 
            "benchmark": "/benchmark - Compare watermarking methods",
            "attack": "/attack - Test attack resistance",
            "batch": "/batch - Process multiple requests",
            "methods": "/methods - List available methods",
            "health": "/health - Comprehensive health check",
            "metrics": "/metrics - System metrics"
        },
        "limits": {
            "max_text_length": 50000,
            "max_batch_size": 100,
            "request_timeout": settings.request_timeout,
            "concurrent_requests": MAX_ACTIVE_REQUESTS
        }
    }

@app.on_event("startup")
async def startup_event():
    """Initialize application state on startup."""
    app.state.start_time = time.time()
    logger.info(f"API started at {datetime.now()}")


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        # Get comprehensive health summary
        health_summary = health_monitor.get_health_summary()
        uptime = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
        
        return HealthCheckResponse(
            status=health_summary['overall_status'],
            timestamp=datetime.now(),
            version=settings.version,
            checks=health_summary['checks'],
            uptime_seconds=uptime
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            version=settings.version,
            checks={"error": str(e)},
            uptime_seconds=0
        )


@app.get("/methods")
@retry(WATERMARK_RETRY_CONFIG)
async def list_methods():
    """List available watermarking methods with enhanced error handling."""
    with LoggingContext(structured_logger, "list_methods") as ctx:
        try:
            # Use circuit breaker for external calls
            circuit = get_circuit_breaker("watermark_methods")
            
            methods = await asyncio.wait_for(
                asyncio.to_thread(circuit.call, WatermarkFactory.list_methods),
                timeout=30.0
            )
            
            method_info = {
                "kirchenbauer": {
                    "name": "Kirchenbauer et al.",
                    "description": "Statistical watermarking with green/red lists",
                    "parameters": ["gamma", "delta", "seed"],
                    "supports_batch": True,
                    "min_text_length": 10
                },
                "markllm": {
                    "name": "MarkLLM",
                    "description": "Key-based watermarking toolkit",
                    "parameters": ["algorithm", "watermark_strength", "key"],
                    "supports_batch": True,
                    "min_text_length": 20
                },
                "aaronson": {
                    "name": "Aaronson",
                    "description": "Cryptographic pseudorandom watermarking",
                    "parameters": ["secret_key", "threshold"],
                    "supports_batch": True,
                    "min_text_length": 15
                },
                "zhao": {
                    "name": "Zhao et al.",
                    "description": "Robust multi-bit watermarking",
                    "parameters": ["message_bits", "redundancy"],
                    "supports_batch": True,
                    "min_text_length": 25
                },
                "sacw": {
                    "name": "Semantic-Aware Contextual Watermarking",
                    "description": "Novel semantic-aware adaptive watermarking",
                    "parameters": ["semantic_threshold", "context_window", "gamma"],
                    "supports_batch": True,
                    "min_text_length": 50
                },
                "arms": {
                    "name": "Adversarial-Robust Multi-Scale Watermarking",
                    "description": "Multi-scale adversarial-resistant watermarking",
                    "parameters": ["scale_levels", "gamma", "robustness_factor"],
                    "supports_batch": True,
                    "min_text_length": 100
                },
                "qipw": {
                    "name": "Quantum-Inspired Probabilistic Watermarking",
                    "description": "Quantum-inspired probabilistic watermarking",
                    "parameters": ["coherence_time", "entanglement_strength", "quantum_noise_level"],
                    "supports_batch": True,
                    "min_text_length": 75
                }
            }
            
            return {
                "methods": methods,
                "total_methods": len(methods),
                "details": {method: method_info.get(method, {"name": method, "description": "Unknown method"}) for method in methods},
                "timestamp": datetime.now()
            }
            
        except asyncio.TimeoutError:
            logger.error("Method listing timed out")
            raise HTTPException(status_code=408, detail="Request timed out while listing methods")
        except ResourceError as e:
            logger.error(f"Resource error listing methods: {e}")
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")
        except Exception as e:
            logger.error(f"Error listing methods: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e) if settings.debug else 'Unable to list methods'}")


@app.post("/watermark", response_model=WatermarkResponse)
@watermark_circuit_breaker("generation")
@retry(WATERMARK_RETRY_CONFIG)
async def generate_watermark(
    request: WatermarkRequest, 
    sanitizer: InputSanitizer = Depends(get_input_sanitizer),
    user = Depends(get_current_user),
    _: None = Depends(validate_request_size)
):
    """Generate watermarked text with comprehensive error handling and validation."""
    with LoggingContext(structured_logger, "watermark_generation", method=request.method, user_id=user["user_id"]) as ctx:
        start_time = time.time()
        
        try:
            # Record metrics
            record_operation_metric("api_request", 1, tags={"endpoint": "watermark", "method": request.method})
            
            # Input validation and sanitization
            sanitized_prompt = sanitizer.sanitize_text(request.prompt, context="watermark_prompt")
            
            # Validate configuration
            validated_config = validate_config(request.config, WATERMARK_CONFIG_SCHEMA)
            validated_config["method"] = request.method
            
            # Validate text length requirements
            validate_text(
                sanitized_prompt,
                min_length=10,
                max_length=settings.default_max_length * 10
            )
            
            # Create watermark instance with timeout
            watermarker = await asyncio.wait_for(
                asyncio.to_thread(WatermarkFactory.create, request.method, **validated_config),
                timeout=60.0
            )
            
            # Generate watermarked text with timeout
            watermarked_text = await asyncio.wait_for(
                asyncio.to_thread(
                    watermarker.generate,
                    sanitized_prompt,
                    max_length=request.max_length
                ),
                timeout=settings.request_timeout
            )
            
            generation_time = time.time() - start_time
            
            # Validate output
            if not watermarked_text or len(watermarked_text.strip()) == 0:
                raise WatermarkLabError("Watermarking produced empty output")
            
            # Log successful generation
            structured_logger.log_watermark_generation(
                method=request.method,
                prompt_length=len(sanitized_prompt),
                output_length=len(watermarked_text),
                generation_time=generation_time,
                success=True
            )
            
            return WatermarkResponse(
                watermarked_text=watermarked_text,
                method=request.method,
                config=watermarker.get_config(),
                generation_time=generation_time,
                timestamp=datetime.now()
            )
            
        except asyncio.TimeoutError:
            generation_time = time.time() - start_time
            error_msg = f"Watermarking timed out after {generation_time:.2f}s"
            logger.error(error_msg)
            structured_logger.log_watermark_generation(
                method=request.method,
                prompt_length=len(request.prompt),
                output_length=0,
                generation_time=generation_time,
                success=False,
                error=error_msg
            )
            raise HTTPException(status_code=408, detail="Watermarking request timed out")
            
        except (ValidationError, PydanticValidationError, SecurityError) as e:
            generation_time = time.time() - start_time
            logger.warning(f"Validation error in watermarking: {e}")
            structured_logger.log_watermark_generation(
                method=request.method,
                prompt_length=len(request.prompt),
                output_length=0,
                generation_time=generation_time,
                success=False,
                error=str(e)
            )
            raise HTTPException(status_code=400, detail=str(e))
            
        except ResourceError as e:
            generation_time = time.time() - start_time
            logger.error(f"Resource error in watermarking: {e}")
            structured_logger.log_watermark_generation(
                method=request.method,
                prompt_length=len(request.prompt),
                output_length=0,
                generation_time=generation_time,
                success=False,
                error=str(e)
            )
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")
            
        except WatermarkLabError as e:
            generation_time = time.time() - start_time
            logger.error(f"Watermarking error: {e}")
            structured_logger.log_watermark_generation(
                method=request.method,
                prompt_length=len(request.prompt),
                output_length=0,
                generation_time=generation_time,
                success=False,
                error=str(e)
            )
            raise HTTPException(status_code=422, detail=str(e))
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"Unexpected error in watermarking: {e}", exc_info=True)
            structured_logger.log_watermark_generation(
                method=request.method,
                prompt_length=len(request.prompt),
                output_length=0,
                generation_time=generation_time,
                success=False,
                error=str(e)
            )
            
            error_detail = str(e) if settings.debug else "Watermarking failed due to internal error"
            raise HTTPException(status_code=500, detail=error_detail)


@app.post("/detect", response_model=DetectionResponse)
@detection_circuit_breaker("analysis")
@retry(DETECTION_RETRY_CONFIG)
async def detect_watermark(
    request: DetectionRequest,
    sanitizer: InputSanitizer = Depends(get_input_sanitizer),
    user = Depends(get_current_user),
    _: None = Depends(validate_request_size)
):
    """Detect watermark in text with comprehensive validation and error handling."""
    with LoggingContext(structured_logger, "watermark_detection", method=request.method, user_id=user["user_id"]) as ctx:
        start_time = time.time()
        
        try:
            # Input validation and sanitization
            sanitized_text = sanitizer.sanitize_text(request.text, context="detection_text")
            
            # Validate configuration
            validated_config = validate_config(request.config, DETECTION_CONFIG_SCHEMA)
            validated_config["method"] = request.method
            
            # Text length validation
            validate_text(
                sanitized_text,
                min_length=10,
                max_length=50000
            )
            
            # Create detector with timeout
            detector = await asyncio.wait_for(
                asyncio.to_thread(WatermarkDetector, validated_config),
                timeout=30.0
            )
            
            # Perform detection with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(detector.detect, sanitized_text),
                timeout=settings.request_timeout
            )
            
            analysis_time = time.time() - start_time
            
            # Validate detection result
            if result is None:
                raise WatermarkLabError("Detection returned no result")
            
            # Log successful detection
            structured_logger.log_detection(
                method=request.method,
                text_length=len(sanitized_text),
                is_watermarked=result.is_watermarked,
                confidence=result.confidence,
                detection_time=analysis_time,
                success=True
            )
            
            response_details = result.to_dict().get("details", {})
            # Limit details size for API response
            if len(str(response_details)) > 1000:
                response_details = {"summary": "Details truncated for response size"}
            
            return DetectionResponse(
                is_watermarked=result.is_watermarked,
                confidence=min(max(result.confidence, 0.0), 1.0),  # Ensure valid range
                p_value=min(max(result.p_value, 0.0), 1.0),  # Ensure valid range
                test_statistic=result.test_statistic or 0.0,
                method=result.method,
                details=response_details,
                analysis_time=analysis_time,
                timestamp=datetime.now()
            )
            
        except asyncio.TimeoutError:
            analysis_time = time.time() - start_time
            error_msg = f"Detection timed out after {analysis_time:.2f}s"
            logger.error(error_msg)
            structured_logger.log_detection(
                method=request.method,
                text_length=len(request.text),
                is_watermarked=False,
                confidence=0.0,
                detection_time=analysis_time,
                success=False,
                error=error_msg
            )
            raise HTTPException(status_code=408, detail="Detection request timed out")
            
        except (ValidationError, PydanticValidationError, SecurityError) as e:
            analysis_time = time.time() - start_time
            logger.warning(f"Validation error in detection: {e}")
            structured_logger.log_detection(
                method=request.method,
                text_length=len(request.text),
                is_watermarked=False,
                confidence=0.0,
                detection_time=analysis_time,
                success=False,
                error=str(e)
            )
            raise HTTPException(status_code=400, detail=str(e))
            
        except ResourceError as e:
            analysis_time = time.time() - start_time
            logger.error(f"Resource error in detection: {e}")
            structured_logger.log_detection(
                method=request.method,
                text_length=len(request.text),
                is_watermarked=False,
                confidence=0.0,
                detection_time=analysis_time,
                success=False,
                error=str(e)
            )
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")
            
        except WatermarkLabError as e:
            analysis_time = time.time() - start_time
            logger.error(f"Detection error: {e}")
            structured_logger.log_detection(
                method=request.method,
                text_length=len(request.text),
                is_watermarked=False,
                confidence=0.0,
                detection_time=analysis_time,
                success=False,
                error=str(e)
            )
            raise HTTPException(status_code=422, detail=str(e))
            
        except Exception as e:
            analysis_time = time.time() - start_time
            logger.error(f"Unexpected error in detection: {e}", exc_info=True)
            structured_logger.log_detection(
                method=request.method,
                text_length=len(request.text),
                is_watermarked=False,
                confidence=0.0,
                detection_time=analysis_time,
                success=False,
                error=str(e)
            )
            
            error_detail = str(e) if settings.debug else "Detection failed due to internal error"
            raise HTTPException(status_code=500, detail=error_detail)


@app.post("/attack")
@retry(WATERMARK_RETRY_CONFIG)
async def test_attack(
    request: AttackRequest,
    sanitizer: InputSanitizer = Depends(get_input_sanitizer),
    user = Depends(get_current_user),
    _: None = Depends(validate_request_size)
):
    """Test attack on watermarked text with comprehensive validation and error handling."""
    with LoggingContext(structured_logger, "attack_simulation", attack_type=request.attack_type, user_id=user["user_id"]) as ctx:
        start_time = time.time()
        
        try:
            # Input validation and sanitization
            sanitized_text = sanitizer.sanitize_text(request.text, context="attack_text")
            
            # Validate configuration
            validated_config = validate_config(request.config, ATTACK_CONFIG_SCHEMA)
            validated_config["attack_type"] = request.attack_type
            validated_config["strength"] = request.strength
            
            # Text validation
            validate_text(
                sanitized_text,
                min_length=10,
                max_length=20000
            )
            
            # Create attack simulator with timeout
            simulator = await asyncio.wait_for(
                asyncio.to_thread(AttackSimulator),
                timeout=30.0
            )
            
            # Run attack with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    simulator.run_attack,
                    sanitized_text,
                    request.attack_type,
                    strength=request.strength,
                    preserve_semantics=request.preserve_semantics,
                    **validated_config
                ),
                timeout=min(settings.request_timeout, 600)  # Max 10 minutes for attacks
            )
            
            attack_time = time.time() - start_time
            
            # Validate result
            if result is None:
                raise WatermarkLabError("Attack simulation returned no result")
            
            # Calculate quality metrics with bounds checking
            quality_score = max(0.0, min(1.0, result.quality_score or 0.0))
            similarity_score = max(0.0, min(1.0, result.similarity_score or 0.0))
            
            # Log successful attack
            structured_logger.log_attack(
                attack_type=request.attack_type,
                original_length=len(sanitized_text),
                attacked_length=len(result.attacked_text or ""),
                quality_score=quality_score,
                similarity_score=similarity_score,
                attack_time=attack_time,
                success=True
            )
            
            return {
                "original_text": sanitized_text,
                "attacked_text": result.attacked_text or "",
                "attack_type": result.attack_type,
                "success": bool(result.success),
                "quality_score": quality_score,
                "similarity_score": similarity_score,
                "preserve_semantics": request.preserve_semantics,
                "metadata": result.metadata or {},
                "attack_time": attack_time,
                "timestamp": datetime.now()
            }
            
        except asyncio.TimeoutError:
            attack_time = time.time() - start_time
            error_msg = f"Attack simulation timed out after {attack_time:.2f}s"
            logger.error(error_msg)
            structured_logger.log_attack(
                attack_type=request.attack_type,
                original_length=len(request.text),
                attacked_length=0,
                quality_score=0.0,
                similarity_score=0.0,
                attack_time=attack_time,
                success=False,
                error=error_msg
            )
            raise HTTPException(status_code=408, detail="Attack simulation timed out")
            
        except (ValidationError, SecurityError) as e:
            attack_time = time.time() - start_time
            logger.warning(f"Validation error in attack: {e}")
            structured_logger.log_attack(
                attack_type=request.attack_type,
                original_length=len(request.text),
                attacked_length=0,
                quality_score=0.0,
                similarity_score=0.0,
                attack_time=attack_time,
                success=False,
                error=str(e)
            )
            raise HTTPException(status_code=400, detail=str(e))
            
        except WatermarkLabError as e:
            attack_time = time.time() - start_time
            logger.error(f"Attack simulation error: {e}")
            structured_logger.log_attack(
                attack_type=request.attack_type,
                original_length=len(request.text),
                attacked_length=0,
                quality_score=0.0,
                similarity_score=0.0,
                attack_time=attack_time,
                success=False,
                error=str(e)
            )
            raise HTTPException(status_code=422, detail=str(e))
            
        except Exception as e:
            attack_time = time.time() - start_time
            logger.error(f"Unexpected error in attack: {e}", exc_info=True)
            structured_logger.log_attack(
                attack_type=request.attack_type,
                original_length=len(request.text),
                attacked_length=0,
                quality_score=0.0,
                similarity_score=0.0,
                attack_time=attack_time,
                success=False,
                error=str(e)
            )
            
            error_detail = str(e) if settings.debug else "Attack simulation failed due to internal error"
            raise HTTPException(status_code=500, detail=error_detail)


@app.post("/benchmark/start")
async def start_benchmark(
    request: BenchmarkRequest, 
    background_tasks: BackgroundTasks,
    user = Depends(get_current_user),
    _: None = Depends(validate_request_size)
):
    """Start asynchronous benchmark comparison with enhanced validation."""
    with LoggingContext(structured_logger, "benchmark_start", user_id=user["user_id"]) as ctx:
        try:
            # Validate batch size
            validate_batch_size(request.num_samples)
            
            # Validate methods
            available_methods = WatermarkFactory.list_methods()
            invalid_methods = set(request.methods) - set(available_methods)
            if invalid_methods:
                raise ValidationError(f"Invalid methods: {invalid_methods}")
            
            # Validate custom prompts if provided
            if request.custom_prompts:
                for i, prompt in enumerate(request.custom_prompts):
                    validate_text(prompt, min_length=10, max_length=1000)
            
            task_id = str(uuid.uuid4())
            
            # Initialize task status with enhanced fields
            task_storage[task_id] = TaskStatus(
                task_id=task_id,
                status="running",
                progress=0.0,
                created_at=datetime.now(),
                estimated_time_remaining=request.num_samples * len(request.methods) * 30  # Rough estimate
            )
            
            # Start benchmark in background with timeout
            background_tasks.add_task(
                run_benchmark_task_enhanced,
                task_id,
                request.methods,
                request.num_samples,
                request.metrics,
                request.custom_prompts,
                request.timeout or 300,
                user["user_id"]
            )
            
            logger.info(f"Started benchmark task {task_id} for user {user['user_id']}")
            
            return {
                "task_id": task_id,
                "status": "started",
                "methods": request.methods,
                "num_samples": request.num_samples,
                "estimated_completion": datetime.now() + timedelta(seconds=request.num_samples * len(request.methods) * 30)
            }
            
        except ValidationError as e:
            logger.warning(f"Benchmark validation error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to start benchmark: {e}", exc_info=True)
            error_detail = str(e) if settings.debug else "Failed to start benchmark"
            raise HTTPException(status_code=500, detail=error_detail)


@app.get("/benchmark/status/{task_id}")
async def get_benchmark_status(task_id: str, user = Depends(get_current_user)):
    """Get benchmark task status with validation."""
    try:
        # Validate task ID format
        try:
            uuid.UUID(task_id)
        except ValueError:
            raise ValidationError("Invalid task ID format")
        
        if task_id not in task_storage:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task_status = task_storage[task_id]
        
        # Calculate updated progress if still running
        if task_status.status == "running":
            elapsed_time = (datetime.now() - task_status.created_at).total_seconds()
            if task_status.estimated_time_remaining:
                progress = min(0.95, elapsed_time / task_status.estimated_time_remaining)
                task_status.progress = progress
        
        return task_status
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting benchmark status: {e}")
        raise HTTPException(status_code=500, detail="Internal error retrieving task status")


async def run_benchmark_task_enhanced(
    task_id: str, 
    methods: List[str], 
    num_samples: int,
    metrics: List[str], 
    custom_prompts: Optional[List[str]],
    timeout: int,
    user_id: str
):
    """Run benchmark task in background with enhanced error handling and monitoring."""
    start_time = time.time()
    
    try:
        logger.info(f"Starting benchmark task {task_id} for user {user_id}")
        
        # Update progress with validation
        def update_progress(progress: float, status: str = "running"):
            if task_id in task_storage:
                task_storage[task_id].progress = min(max(progress, 0.0), 1.0)
                task_storage[task_id].status = status
                remaining_time = (1 - progress) * (time.time() - start_time) / max(progress, 0.01)
                task_storage[task_id].estimated_time_remaining = int(remaining_time)
        
        update_progress(0.1)
        
        # Create benchmark suite with timeout
        benchmark = await asyncio.wait_for(
            asyncio.to_thread(WatermarkBenchmark, num_samples=num_samples),
            timeout=60.0
        )
        
        update_progress(0.2)
        
        # Prepare and validate prompts
        if custom_prompts:
            # Validate custom prompts
            validated_prompts = []
            for prompt in custom_prompts[:num_samples]:
                try:
                    validated_prompt = input_sanitizer.sanitize_text(prompt, context="benchmark_prompt")
                    validate_text(validated_prompt, min_length=10, max_length=1000)
                    validated_prompts.append(validated_prompt)
                except Exception as e:
                    logger.warning(f"Invalid prompt in benchmark, skipping: {e}")
            
            test_prompts = validated_prompts[:num_samples]
        else:
            test_prompts = benchmark.test_prompts[:num_samples]
        
        if not test_prompts:
            raise WatermarkLabError("No valid prompts available for benchmarking")
        
        update_progress(0.3)
        
        # Run comparison with timeout and progress tracking
        total_operations = len(methods) * len(test_prompts) * len(metrics)
        completed_operations = 0
        
        results = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: benchmark.compare_with_progress(
                    methods, test_prompts, metrics,
                    progress_callback=lambda: update_progress(0.3 + 0.6 * (completed_operations / total_operations))
                )
            ),
            timeout=timeout
        )
        
        update_progress(0.9)
        
        # Validate and sanitize results
        if not results or not isinstance(results, dict):
            raise WatermarkLabError("Benchmark produced invalid results")
        
        # Log successful completion
        benchmark_time = time.time() - start_time
        structured_logger.log_benchmark(
            methods=methods,
            num_samples=len(test_prompts),
            metrics=metrics,
            total_time=benchmark_time,
            success=True
        )
        
        # Update task as completed
        task_storage[task_id].status = "completed"
        task_storage[task_id].progress = 1.0
        task_storage[task_id].result = results
        task_storage[task_id].completed_at = datetime.now()
        task_storage[task_id].estimated_time_remaining = 0
        
        logger.info(f"Benchmark task {task_id} completed successfully in {benchmark_time:.2f}s")
        
    except asyncio.TimeoutError:
        benchmark_time = time.time() - start_time
        error_msg = f"Benchmark timed out after {benchmark_time:.2f}s"
        logger.error(f"Task {task_id}: {error_msg}")
        
        structured_logger.log_benchmark(
            methods=methods,
            num_samples=num_samples,
            metrics=metrics,
            total_time=benchmark_time,
            success=False,
            error=error_msg
        )
        
        task_storage[task_id].status = "failed"
        task_storage[task_id].error = error_msg
        task_storage[task_id].completed_at = datetime.now()
        
    except Exception as e:
        benchmark_time = time.time() - start_time
        error_msg = str(e)
        logger.error(f"Task {task_id} failed: {error_msg}", exc_info=True)
        
        structured_logger.log_benchmark(
            methods=methods,
            num_samples=num_samples,
            metrics=metrics,
            total_time=benchmark_time,
            success=False,
            error=error_msg
        )
        
        task_storage[task_id].status = "failed"
        task_storage[task_id].error = error_msg
        task_storage[task_id].completed_at = datetime.now()
        task_storage[task_id].retry_count = getattr(task_storage[task_id], 'retry_count', 0) + 1


@app.get("/attacks")
@retry(WATERMARK_RETRY_CONFIG)
async def list_attacks():
    """List available attack types with enhanced error handling."""
    with LoggingContext(structured_logger, "list_attacks") as ctx:
        try:
            # Use circuit breaker for external calls
            circuit = get_circuit_breaker("attack_methods")
            
            simulator = await asyncio.wait_for(
                asyncio.to_thread(circuit.call, AttackSimulator),
                timeout=30.0
            )
            
            attacks = await asyncio.wait_for(
                asyncio.to_thread(simulator.list_attacks),
                timeout=30.0
            )
            
            attack_info = {
                "paraphrase": {
                    "description": "Synonym replacement and sentence reordering",
                    "strengths": ["light", "medium", "heavy"],
                    "effectiveness": "high",
                    "speed": "medium",
                    "semantic_preservation": "high"
                },
                "truncation": {
                    "description": "Remove parts of text",
                    "strengths": ["light", "medium", "heavy"],
                    "effectiveness": "medium",
                    "speed": "fast",
                    "semantic_preservation": "medium"
                },
                "insertion": {
                    "description": "Add noise words and phrases",
                    "strengths": ["light", "medium", "heavy"],
                    "effectiveness": "medium",
                    "speed": "fast",
                    "semantic_preservation": "low"
                },
                "substitution": {
                    "description": "Character and word-level substitutions",
                    "strengths": ["light", "medium", "heavy"],
                    "effectiveness": "high",
                    "speed": "medium",
                    "semantic_preservation": "medium"
                },
                "translation": {
                    "description": "Back-translation attack simulation",
                    "strengths": ["light", "medium", "heavy"],
                    "effectiveness": "very_high",
                    "speed": "slow",
                    "semantic_preservation": "high"
                },
                "combined": {
                    "description": "Multiple attack strategies combined",
                    "strengths": ["light", "medium", "heavy"],
                    "effectiveness": "very_high",
                    "speed": "slow",
                    "semantic_preservation": "variable"
                }
            }
            
            return {
                "attacks": attacks,
                "total_attacks": len(attacks),
                "details": {attack: attack_info.get(attack, {"name": attack, "description": "Unknown attack"}) for attack in attacks},
                "default_strength": "medium",
                "timestamp": datetime.now()
            }
            
        except asyncio.TimeoutError:
            logger.error("Attack listing timed out")
            raise HTTPException(status_code=408, detail="Request timed out while listing attacks")
        except ResourceError as e:
            logger.error(f"Resource error listing attacks: {e}")
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")
        except Exception as e:
            logger.error(f"Error listing attacks: {e}", exc_info=True)
            error_detail = str(e) if settings.debug else "Unable to list attacks"
            raise HTTPException(status_code=500, detail=error_detail)


# New enhanced endpoints
@app.post("/batch")
async def process_batch(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    user = Depends(get_current_user),
    _: None = Depends(validate_request_size)
):
    """Process batch of watermarking or detection requests."""
    with LoggingContext(structured_logger, "batch_processing", user_id=user["user_id"]) as ctx:
        try:
            # Validate batch size
            validate_batch_size(len(request.items))
            
            task_id = str(uuid.uuid4())
            
            # Initialize task status
            task_storage[task_id] = TaskStatus(
                task_id=task_id,
                status="running",
                progress=0.0,
                created_at=datetime.now(),
                estimated_time_remaining=len(request.items) * 10  # Rough estimate
            )
            
            # Start batch processing in background
            background_tasks.add_task(
                process_batch_task,
                task_id,
                request.items,
                request.batch_name or f"batch_{task_id[:8]}",
                user["user_id"]
            )
            
            logger.info(f"Started batch task {task_id} with {len(request.items)} items")
            
            return {
                "task_id": task_id,
                "status": "started",
                "batch_size": len(request.items),
                "estimated_completion": datetime.now() + timedelta(seconds=len(request.items) * 10)
            }
            
        except ValidationError as e:
            logger.warning(f"Batch validation error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to start batch: {e}", exc_info=True)
            error_detail = str(e) if settings.debug else "Failed to start batch processing"
            raise HTTPException(status_code=500, detail=error_detail)

@app.get("/metrics")
async def get_metrics():
    """Get system metrics and health information."""
    try:
        # Get system metrics
        metrics = health_monitor.get_health_summary()
        
        # Get circuit breaker metrics
        from ..utils.circuit_breaker import get_circuit_breaker_health
        circuit_health = get_circuit_breaker_health()
        
        # Get active requests count
        metrics['api_metrics'] = {
            'active_requests': len(active_requests),
            'max_concurrent_requests': MAX_ACTIVE_REQUESTS,
            'total_tasks_created': len(task_storage),
            'running_tasks': sum(1 for task in task_storage.values() if task.status == 'running')
        }
        
        metrics['circuit_breakers'] = circuit_health
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Unable to retrieve metrics")

@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str, user = Depends(get_current_user)):
    """Cancel a running task."""
    try:
        # Validate task ID format
        try:
            uuid.UUID(task_id)
        except ValueError:
            raise ValidationError("Invalid task ID format")
        
        if task_id not in task_storage:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task_status = task_storage[task_id]
        
        if task_status.status != "running":
            raise HTTPException(status_code=400, detail="Task is not running")
        
        # Cancel the task
        task_status.status = "cancelled"
        task_status.completed_at = datetime.now()
        task_status.error = "Task cancelled by user"
        
        logger.info(f"Task {task_id} cancelled by user {user['user_id']}")
        
        return {"message": "Task cancelled successfully", "task_id": task_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling task: {e}")
        raise HTTPException(status_code=500, detail="Unable to cancel task")

async def process_batch_task(task_id: str, items: List, batch_name: str, user_id: str):
    """Process batch of requests in background."""
    start_time = time.time()
    results = []
    
    try:
        logger.info(f"Processing batch {batch_name} with {len(items)} items for user {user_id}")
        
        for i, item in enumerate(items):
            try:
                # Update progress
                progress = (i + 1) / len(items)
                task_storage[task_id].progress = progress
                
                # Process individual item based on type
                if isinstance(item, WatermarkRequest):
                    # Simulate watermark generation (simplified)
                    sanitized_prompt = input_sanitizer.sanitize_text(item.prompt, context="batch_watermark")
                    watermarker = WatermarkFactory.create(item.method, **item.config)
                    result = watermarker.generate(sanitized_prompt, max_length=item.max_length)
                    
                    results.append({
                        "type": "watermark",
                        "original_prompt": sanitized_prompt,
                        "watermarked_text": result,
                        "method": item.method,
                        "success": True
                    })
                    
                elif isinstance(item, DetectionRequest):
                    # Simulate detection (simplified)
                    sanitized_text = input_sanitizer.sanitize_text(item.text, context="batch_detection")
                    detector_config = {"method": item.method, **item.config}
                    detector = WatermarkDetector(detector_config)
                    result = detector.detect(sanitized_text)
                    
                    results.append({
                        "type": "detection",
                        "text": sanitized_text,
                        "is_watermarked": result.is_watermarked,
                        "confidence": result.confidence,
                        "method": item.method,
                        "success": True
                    })
                
            except Exception as e:
                logger.error(f"Batch item {i} failed: {e}")
                results.append({
                    "type": "error",
                    "error": str(e),
                    "success": False
                })
        
        # Mark as completed
        batch_time = time.time() - start_time
        task_storage[task_id].status = "completed"
        task_storage[task_id].progress = 1.0
        task_storage[task_id].result = {
            "batch_name": batch_name,
            "total_items": len(items),
            "successful_items": sum(1 for r in results if r.get("success", False)),
            "failed_items": sum(1 for r in results if not r.get("success", True)),
            "processing_time": batch_time,
            "results": results
        }
        task_storage[task_id].completed_at = datetime.now()
        
        logger.info(f"Batch {batch_name} completed in {batch_time:.2f}s")
        
    except Exception as e:
        batch_time = time.time() - start_time
        logger.error(f"Batch {batch_name} failed: {e}", exc_info=True)
        
        task_storage[task_id].status = "failed"
        task_storage[task_id].error = str(e)
        task_storage[task_id].completed_at = datetime.now()

# Enhanced error handlers
@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Handle validation errors."""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content=format_error_response(exc)
    )

@app.exception_handler(SecurityError)
async def security_error_handler(request: Request, exc: SecurityError):
    """Handle security errors."""
    logger.error(f"Security error: {exc}")
    return JSONResponse(
        status_code=403,
        content=format_error_response(exc)
    )

@app.exception_handler(TimeoutError)
async def timeout_error_handler(request: Request, exc: TimeoutError):
    """Handle timeout errors."""
    logger.error(f"Timeout error: {exc}")
    return JSONResponse(
        status_code=408,
        content=format_error_response(exc)
    )

@app.exception_handler(ResourceError)
async def resource_error_handler(request: Request, exc: ResourceError):
    """Handle resource errors."""
    logger.error(f"Resource error: {exc}")
    return JSONResponse(
        status_code=503,
        content=format_error_response(exc)
    )

@app.exception_handler(RateLimitError)
async def rate_limit_error_handler(request: Request, exc: RateLimitError):
    """Handle rate limit errors."""
    logger.warning(f"Rate limit exceeded: {exc}")
    response = JSONResponse(
        status_code=429,
        content=format_error_response(exc)
    )
    if exc.retry_after:
        response.headers["Retry-After"] = str(exc.retry_after)
    return response

@app.exception_handler(WatermarkLabError)
async def watermark_lab_error_handler(request: Request, exc: WatermarkLabError):
    """Handle application-specific errors."""
    logger.error(f"Application error: {exc}")
    return JSONResponse(
        status_code=422,
        content=format_error_response(exc)
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle value errors."""
    logger.warning(f"Value error: {exc}")
    return JSONResponse(
        status_code=400,
        content=format_error_response(exc)
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    
    # Don't expose internal details in production
    error_detail = str(exc) if settings.debug else "Internal server error"
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": error_detail,
            "timestamp": datetime.now().isoformat()
        }
    )


# Health check for Kubernetes/Docker
@app.get("/health/ready")
async def readiness_check():
    """Readiness check for container orchestration."""
    try:
        # Check if all systems are ready
        health_summary = health_monitor.get_health_summary()
        
        if health_summary['overall_status'] == 'critical':
            return JSONResponse(status_code=503, content={"status": "not ready"})
        
        return {"status": "ready", "timestamp": datetime.now()}
        
    except Exception:
        return JSONResponse(status_code=503, content={"status": "not ready"})

@app.get("/health/live")
async def liveness_check():
    """Liveness check for container orchestration."""
    return {"status": "alive", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    from ..utils.logging import setup_logging
    setup_logging(
        level=settings.log_level,
        log_file=f"{settings.logs_dir}/api.log" if not settings.debug else None,
        json_format=not settings.debug
    )
    
    logger.info(f"Starting {settings.app_name} v{settings.version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Max concurrent requests: {MAX_ACTIVE_REQUESTS}")
    
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload and settings.debug,
        workers=settings.api_workers if not settings.debug else 1,
        access_log=settings.debug,
        log_level=settings.log_level.lower()
    )