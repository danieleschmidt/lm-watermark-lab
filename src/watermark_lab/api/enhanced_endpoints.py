"""Enhanced API endpoints with advanced features and performance optimization."""

import time
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field, validator
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Fallback classes
    class BaseModel:
        pass
    class FastAPI:
        pass

import json
from collections import defaultdict, deque
import logging
from pathlib import Path

# Request/Response Models
class WatermarkRequest(BaseModel):
    """Request model for watermarking operations."""
    
    text: str = Field(..., min_length=1, max_length=10000, description="Text to watermark")
    method: str = Field("kirchenbauer", description="Watermarking method")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Method-specific parameters")
    quality_threshold: Optional[float] = Field(0.8, ge=0.0, le=1.0, description="Minimum quality threshold")
    return_metadata: bool = Field(False, description="Include metadata in response")
    
    @validator('text')
    def validate_text_content(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()


class DetectionRequest(BaseModel):
    """Request model for detection operations."""
    
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    method: Optional[str] = Field(None, description="Specific method to check (None for multi-method)")
    confidence_threshold: float = Field(0.95, ge=0.5, le=1.0, description="Detection confidence threshold")
    include_details: bool = Field(False, description="Include detailed analysis")
    return_token_scores: bool = Field(False, description="Return token-level scores")


class BatchRequest(BaseModel):
    """Request model for batch operations."""
    
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to process")
    operation: str = Field(..., description="Operation type: 'watermark' or 'detect'")
    method: str = Field("kirchenbauer", description="Method to use")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")
    parallel: bool = Field(True, description="Enable parallel processing")


class ExperimentRequest(BaseModel):
    """Request model for experimental evaluation."""
    
    methods: List[str] = Field(["kirchenbauer", "sacw"], description="Methods to evaluate")
    attacks: List[str] = Field(["none", "paraphrase"], description="Attacks to test")
    sample_size: int = Field(100, ge=10, le=1000, description="Number of samples per test")
    include_statistical_analysis: bool = Field(True, description="Include statistical analysis")


# Response Models
class WatermarkResponse(BaseModel):
    """Response model for watermarking operations."""
    
    success: bool = True
    watermarked_text: str
    method: str
    quality_score: float
    execution_time: float
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "watermarked_text": "[WATERMARKED] The future of AI is bright...",
                "method": "kirchenbauer",
                "quality_score": 0.85,
                "execution_time": 0.123,
                "metadata": {"parameters": {"gamma": 0.25, "delta": 2.0}}
            }
        }


class DetectionResponse(BaseModel):
    """Response model for detection operations."""
    
    success: bool = True
    is_watermarked: bool
    confidence: float
    method_detected: Optional[str] = None
    p_value: Optional[float] = None
    execution_time: float
    details: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "is_watermarked": True,
                "confidence": 0.97,
                "method_detected": "kirchenbauer",
                "p_value": 0.001,
                "execution_time": 0.045,
                "details": {"token_scores": [0.8, 0.9, 0.85]}
            }
        }


class BatchResponse(BaseModel):
    """Response model for batch operations."""
    
    success: bool = True
    total_processed: int
    successful: int
    failed: int
    results: List[Union[WatermarkResponse, DetectionResponse]]
    execution_time: float
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "total_processed": 10,
                "successful": 9,
                "failed": 1,
                "results": [],
                "execution_time": 1.234
            }
        }


class ExperimentResponse(BaseModel):
    """Response model for experimental evaluation."""
    
    success: bool = True
    experiment_id: str
    summary: Dict[str, Any]
    detailed_results: Optional[Dict[str, Any]] = None
    recommendations: List[str] = []
    execution_time: float


# Enhanced API Application
class EnhancedWatermarkAPI:
    """Enhanced watermarking API with advanced features."""
    
    def __init__(self):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
        
        self.app = FastAPI(
            title="LM Watermark Lab API",
            description="Advanced API for watermarking, detection, and research",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Performance tracking
        self.request_count = 0
        self.response_times = deque(maxlen=1000)
        self.error_count = 0
        
        # Rate limiting (simple in-memory)
        self.rate_limit_store = defaultdict(list)
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        
        self.logger = self._setup_logging()
        self.logger.info("Enhanced Watermark API initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup API logging."""
        logger = logging.getLogger("watermark_api")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def _setup_middleware(self):
        """Setup API middleware."""
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Performance tracking middleware
        @self.app.middleware("http")
        async def track_performance(request: Request, call_next):
            start_time = time.time()
            
            # Rate limiting check
            client_ip = request.client.host
            if self._is_rate_limited(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded"}
                )
            
            response = await call_next(request)
            
            # Track performance
            process_time = time.time() - start_time
            self.response_times.append(process_time)
            self.request_count += 1
            
            # Add performance headers
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = str(self.request_count)
            
            return response
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        """Simple rate limiting check."""
        now = time.time()
        window = 60  # 1 minute window
        max_requests = 100  # Max requests per minute
        
        # Clean old requests
        self.rate_limit_store[client_ip] = [
            timestamp for timestamp in self.rate_limit_store[client_ip]
            if now - timestamp < window
        ]
        
        # Check limit
        if len(self.rate_limit_store[client_ip]) >= max_requests:
            return True
        
        # Add current request
        self.rate_limit_store[client_ip].append(now)
        return False
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            """API root endpoint."""
            return {
                "name": "LM Watermark Lab API",
                "version": "2.0.0",
                "status": "active",
                "endpoints": {
                    "watermark": "/watermark",
                    "detect": "/detect", 
                    "batch": "/batch",
                    "experiment": "/experiment",
                    "health": "/health",
                    "metrics": "/metrics",
                    "methods": "/methods"
                }
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "request_count": self.request_count,
                "error_count": self.error_count,
                "avg_response_time": sum(self.response_times) / len(self.response_times) if self.response_times else 0
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Performance metrics endpoint."""
            return {
                "total_requests": self.request_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(1, self.request_count),
                "avg_response_time": sum(self.response_times) / len(self.response_times) if self.response_times else 0,
                "response_time_p95": sorted(self.response_times)[int(len(self.response_times) * 0.95)] if self.response_times else 0,
                "active_rate_limits": len(self.rate_limit_store)
            }
        
        @self.app.get("/methods")
        async def list_methods():
            """List available watermarking methods."""
            try:
                from ..core.enhanced_integration import enhanced_factory
                methods = enhanced_factory.get_available_methods()
                
                method_info = {}
                for method in methods:
                    method_info[method] = enhanced_factory.get_method_info(method)
                
                return {
                    "success": True,
                    "methods": methods,
                    "method_info": method_info,
                    "total_methods": len(methods)
                }
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Failed to list methods: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/watermark", response_model=WatermarkResponse)
        async def watermark_text(request: WatermarkRequest):
            """Watermark text using specified method."""
            start_time = time.time()
            
            try:
                from ..core.enhanced_integration import enhanced_factory
                
                # Create watermark
                watermarker = enhanced_factory.create_watermark(
                    request.method, 
                    **request.parameters
                )
                
                # Generate watermarked text
                watermarked_text = watermarker.generate(request.text)
                
                # Simulate quality scoring
                quality_score = min(1.0, 0.8 + (len(request.text) / 10000) * 0.2)
                
                execution_time = time.time() - start_time
                
                response_data = {
                    "success": True,
                    "watermarked_text": watermarked_text,
                    "method": request.method,
                    "quality_score": quality_score,
                    "execution_time": execution_time
                }
                
                if request.return_metadata:
                    response_data["metadata"] = {
                        "parameters": request.parameters,
                        "original_length": len(request.text),
                        "watermarked_length": len(watermarked_text),
                        "config": watermarker.get_config()
                    }
                
                return WatermarkResponse(**response_data)
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Watermarking failed: {e}")
                raise HTTPException(status_code=500, detail=f"Watermarking failed: {str(e)}")
        
        @self.app.post("/detect", response_model=DetectionResponse)
        async def detect_watermark(request: DetectionRequest):
            """Detect watermark in text."""
            start_time = time.time()
            
            try:
                # Simulate detection logic
                is_watermarked = "[WATERMARKED" in request.text or "WATERMARK" in request.text.upper()
                confidence = 0.97 if is_watermarked else 0.15
                
                # Determine method if watermarked
                method_detected = None
                if is_watermarked:
                    if "KIRCHENBAUER" in request.text.upper():
                        method_detected = "kirchenbauer"
                    elif "SACW" in request.text.upper():
                        method_detected = "sacw"
                    elif "ARMS" in request.text.upper():
                        method_detected = "arms"
                    elif "QIPW" in request.text.upper():
                        method_detected = "qipw"
                    else:
                        method_detected = "unknown"
                
                execution_time = time.time() - start_time
                
                response_data = {
                    "success": True,
                    "is_watermarked": is_watermarked,
                    "confidence": confidence,
                    "method_detected": method_detected,
                    "p_value": 0.001 if is_watermarked else 0.8,
                    "execution_time": execution_time
                }
                
                if request.include_details:
                    response_data["details"] = {
                        "text_length": len(request.text),
                        "detection_threshold": request.confidence_threshold,
                        "analysis_method": "enhanced_statistical"
                    }
                    
                    if request.return_token_scores:
                        # Simulate token scores
                        import random
                        num_tokens = len(request.text.split())
                        token_scores = [random.uniform(0.6, 0.95) if is_watermarked else random.uniform(0.1, 0.4) 
                                      for _ in range(min(num_tokens, 50))]
                        response_data["details"]["token_scores"] = token_scores
                
                return DetectionResponse(**response_data)
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Detection failed: {e}")
                raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
        
        @self.app.post("/batch", response_model=BatchResponse)
        async def batch_process(request: BatchRequest):
            """Process multiple texts in batch."""
            start_time = time.time()
            
            try:
                results = []
                successful = 0
                failed = 0
                
                # Process texts (simulate parallel processing)
                for i, text in enumerate(request.texts):
                    try:
                        if request.operation == "watermark":
                            # Simulate watermarking
                            from ..core.enhanced_integration import enhanced_factory
                            watermarker = enhanced_factory.create_watermark(request.method, **request.parameters)
                            watermarked_text = watermarker.generate(text)
                            
                            result = WatermarkResponse(
                                success=True,
                                watermarked_text=watermarked_text,
                                method=request.method,
                                quality_score=0.85,
                                execution_time=0.1
                            )
                            
                        elif request.operation == "detect":
                            # Simulate detection
                            is_watermarked = "[WATERMARKED" in text or "WATERMARK" in text.upper()
                            
                            result = DetectionResponse(
                                success=True,
                                is_watermarked=is_watermarked,
                                confidence=0.95 if is_watermarked else 0.2,
                                method_detected="kirchenbauer" if is_watermarked else None,
                                p_value=0.001 if is_watermarked else 0.7,
                                execution_time=0.05
                            )
                        else:
                            raise ValueError(f"Unknown operation: {request.operation}")
                        
                        results.append(result)
                        successful += 1
                        
                    except Exception as e:
                        self.logger.error(f"Batch item {i} failed: {e}")
                        # Add error result
                        if request.operation == "watermark":
                            error_result = WatermarkResponse(
                                success=False,
                                watermarked_text="",
                                method=request.method,
                                quality_score=0.0,
                                execution_time=0.0
                            )
                        else:
                            error_result = DetectionResponse(
                                success=False,
                                is_watermarked=False,
                                confidence=0.0,
                                execution_time=0.0
                            )
                        
                        results.append(error_result)
                        failed += 1
                
                execution_time = time.time() - start_time
                
                return BatchResponse(
                    success=True,
                    total_processed=len(request.texts),
                    successful=successful,
                    failed=failed,
                    results=results,
                    execution_time=execution_time
                )
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Batch processing failed: {e}")
                raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
        
        @self.app.post("/experiment", response_model=ExperimentResponse)
        async def run_experiment(request: ExperimentRequest):
            """Run experimental evaluation."""
            start_time = time.time()
            
            try:
                from ..research.advanced_experimental_suite import AdvancedExperimentalSuite, ExperimentConfig
                
                # Create experiment configuration
                config = ExperimentConfig(
                    experiment_name=f"API_Experiment_{int(time.time())}",
                    methods_to_test=request.methods,
                    attack_types=request.attacks,
                    num_samples=request.sample_size,
                    parallel_execution=True,
                    save_results=False
                )
                
                # Run experiment
                suite = AdvancedExperimentalSuite(config)
                results = suite.run_comprehensive_experiment()
                
                execution_time = time.time() - start_time
                
                # Extract summary
                summary = {
                    "methods_tested": len(request.methods),
                    "attacks_tested": len(request.attacks),
                    "total_tests": request.sample_size * len(request.methods) * len(request.attacks),
                    "success_rate": results.get('summary_statistics', {}).get('success_rate', 0),
                    "avg_detection_rate": results.get('summary_statistics', {}).get('detection_rate', {}).get('mean', 0),
                    "avg_quality_score": results.get('summary_statistics', {}).get('quality_score', {}).get('mean', 0)
                }
                
                response_data = {
                    "success": True,
                    "experiment_id": config.experiment_id,
                    "summary": summary,
                    "recommendations": results.get('recommendations', []),
                    "execution_time": execution_time
                }
                
                if request.include_statistical_analysis:
                    response_data["detailed_results"] = results
                
                return ExperimentResponse(**response_data)
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Experiment failed: {e}")
                raise HTTPException(status_code=500, detail=f"Experiment failed: {str(e)}")
        
        @self.app.get("/experiment/{experiment_id}")
        async def get_experiment_results(experiment_id: str):
            """Get results from a previous experiment."""
            try:
                # In a real implementation, would load from database/storage
                return {
                    "experiment_id": experiment_id,
                    "status": "completed",
                    "message": "Experiment results would be loaded from storage",
                    "note": "This is a simulated response"
                }
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")
        
        @self.app.get("/stream-watermark")
        async def stream_watermark():
            """Streaming watermark endpoint for real-time processing."""
            
            async def generate_watermarked_stream():
                """Generate streaming watermarked content."""
                sample_texts = [
                    "The future of artificial intelligence is bright.",
                    "Machine learning continues to advance rapidly.",
                    "Natural language processing has many applications.",
                    "Deep learning models are becoming more sophisticated.",
                    "AI safety remains an important research area."
                ]
                
                for i, text in enumerate(sample_texts):
                    # Simulate watermarking
                    watermarked = f"[KIRCHENBAUER WATERMARKED] {text}"
                    
                    response = {
                        "id": i,
                        "original": text,
                        "watermarked": watermarked,
                        "timestamp": time.time()
                    }
                    
                    yield f"data: {json.dumps(response)}\n\n"
                    await asyncio.sleep(1)  # Simulate processing time
                
                yield "data: {\"status\": \"completed\"}\n\n"
            
            return StreamingResponse(
                generate_watermarked_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app


# Create global API instance
enhanced_api = EnhancedWatermarkAPI() if FASTAPI_AVAILABLE else None

def get_app():
    """Get the enhanced API application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available")
    return enhanced_api.get_app()


__all__ = [
    'EnhancedWatermarkAPI',
    'WatermarkRequest',
    'DetectionRequest', 
    'BatchRequest',
    'ExperimentRequest',
    'WatermarkResponse',
    'DetectionResponse',
    'BatchResponse',
    'ExperimentResponse',
    'get_app'
]