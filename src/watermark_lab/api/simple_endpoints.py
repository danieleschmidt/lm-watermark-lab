"""Simple API endpoints for watermarking operations."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import traceback

from ..core.factory import WatermarkFactory
from ..methods.base import DetectionResult

# Simple request/response models
class WatermarkRequest(BaseModel):
    prompt: str
    method: str = "kirchenbauer"
    max_length: int = 100
    temperature: float = 0.7
    key: str = "default"

class WatermarkResponse(BaseModel):
    watermarked_text: str
    method: str
    success: bool

class DetectionRequest(BaseModel):
    text: str
    method: str = "kirchenbauer"
    key: str = "default"

class DetectionResponse(BaseModel):
    is_watermarked: bool
    confidence: float
    p_value: float
    method: str
    success: bool

# Create simple app
app = FastAPI(
    title="LM Watermark Lab - Simple API",
    description="Simple API for watermarking and detection",
    version="1.0.0"
)

@app.post("/watermark", response_model=WatermarkResponse)
async def generate_watermark(request: WatermarkRequest):
    """Generate watermarked text."""
    try:
        # Create watermarker
        watermarker = WatermarkFactory.create(
            method=request.method,
            model_name="gpt2",  # Default to GPT2 for simplicity
            key=request.key
        )
        
        # Generate watermarked text
        watermarked_text = watermarker.generate(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        return WatermarkResponse(
            watermarked_text=watermarked_text,
            method=request.method,
            success=True
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating watermark: {str(e)}"
        )

@app.post("/detect", response_model=DetectionResponse)
async def detect_watermark(request: DetectionRequest):
    """Detect watermark in text."""
    try:
        # Create watermarker for detection
        watermarker = WatermarkFactory.create(
            method=request.method,
            model_name="gpt2",  # Default to GPT2 for simplicity
            key=request.key
        )
        
        # Detect watermark
        result = watermarker.detect(request.text)
        
        return DetectionResponse(
            is_watermarked=result.is_watermarked,
            confidence=result.confidence,
            p_value=result.p_value,
            method=request.method,
            success=True
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error detecting watermark: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "watermark-lab"}

@app.get("/methods")
async def list_methods():
    """List available watermarking methods."""
    from ..methods import list_available_methods
    return {"methods": list_available_methods()}