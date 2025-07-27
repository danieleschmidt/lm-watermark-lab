# LM Watermark Lab API Documentation

This document provides comprehensive documentation for the LM Watermark Lab REST API.

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Base URL and Versioning](#base-url-and-versioning)
- [Request/Response Format](#requestresponse-format)
- [Rate Limiting](#rate-limiting)
- [Error Handling](#error-handling)
- [Endpoints](#endpoints)
- [WebSocket API](#websocket-api)
- [SDKs and Libraries](#sdks-and-libraries)
- [Examples](#examples)

## Overview

The LM Watermark Lab API provides programmatic access to watermarking, detection, and evaluation capabilities. The API follows RESTful principles and supports both synchronous and asynchronous operations.

### Key Features

- **Watermark Generation**: Generate watermarked text using various algorithms
- **Watermark Detection**: Detect and analyze watermarks in text
- **Batch Processing**: Handle multiple texts efficiently
- **Attack Simulation**: Test watermark robustness
- **Quality Evaluation**: Assess text quality metrics
- **Real-time Streaming**: WebSocket support for real-time operations

## Authentication

### API Key Authentication

```bash
# Include API key in header
curl -H "X-API-Key: your-api-key" https://api.watermark-lab.com/api/v1/methods
```

### JWT Token Authentication

```bash
# Login to get JWT token
curl -X POST https://api.watermark-lab.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'

# Use JWT token
curl -H "Authorization: Bearer your-jwt-token" \
  https://api.watermark-lab.com/api/v1/generate
```

### OAuth 2.0 (Enterprise)

```bash
# Get access token
curl -X POST https://api.watermark-lab.com/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&client_id=your-client-id&client_secret=your-secret"
```

## Base URL and Versioning

- **Base URL**: `https://api.watermark-lab.com`
- **Current Version**: `v1`
- **API Endpoint**: `https://api.watermark-lab.com/api/v1`

### API Versioning

```bash
# Current version (recommended)
curl https://api.watermark-lab.com/api/v1/methods

# Specific version
curl -H "Accept: application/vnd.watermark-lab.v1+json" \
  https://api.watermark-lab.com/api/methods
```

## Request/Response Format

### Content Types

- **Request**: `application/json`
- **Response**: `application/json`
- **File Upload**: `multipart/form-data`

### Standard Response Format

```json
{
  "success": true,
  "data": {
    "result": "response data"
  },
  "meta": {
    "timestamp": "2024-01-27T10:00:00Z",
    "request_id": "req_123456789",
    "version": "1.0.0"
  }
}
```

### Error Response Format

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "prompt",
      "reason": "Required field is missing"
    }
  },
  "meta": {
    "timestamp": "2024-01-27T10:00:00Z",
    "request_id": "req_123456789"
  }
}
```

## Rate Limiting

### Default Limits

- **Free Tier**: 100 requests/hour
- **Pro Tier**: 1,000 requests/hour  
- **Enterprise**: 10,000 requests/hour

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1643273400
X-RateLimit-Retry-After: 3600
```

### Rate Limit Response

```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 3600 seconds."
  }
}
```

## Error Handling

### HTTP Status Codes

| Status Code | Description |
|-------------|-------------|
| `200` | Success |
| `201` | Created |
| `400` | Bad Request |
| `401` | Unauthorized |
| `403` | Forbidden |
| `404` | Not Found |
| `422` | Validation Error |
| `429` | Rate Limited |
| `500` | Internal Server Error |
| `503` | Service Unavailable |

### Error Codes

| Error Code | Description |
|------------|-------------|
| `VALIDATION_ERROR` | Invalid input parameters |
| `AUTH_ERROR` | Authentication failed |
| `PERMISSION_DENIED` | Insufficient permissions |
| `NOT_FOUND` | Resource not found |
| `RATE_LIMIT_EXCEEDED` | Rate limit exceeded |
| `MODEL_NOT_AVAILABLE` | Requested model unavailable |
| `PROCESSING_ERROR` | Internal processing error |

## Endpoints

### Health and Status

#### GET /health

Basic health check endpoint.

```bash
curl https://api.watermark-lab.com/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-27T10:00:00Z",
  "version": "1.0.0"
}
```

#### GET /api/v1/status

Detailed system status.

```bash
curl https://api.watermark-lab.com/api/v1/status
```

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "operational",
    "services": {
      "api": "healthy",
      "database": "healthy",
      "cache": "healthy",
      "models": "healthy"
    },
    "metrics": {
      "uptime_seconds": 86400,
      "active_connections": 45,
      "queue_length": 0
    }
  }
}
```

### Watermarking Methods

#### GET /api/v1/methods

List available watermarking methods.

```bash
curl -H "X-API-Key: your-api-key" \
  https://api.watermark-lab.com/api/v1/methods
```

**Response:**
```json
{
  "success": true,
  "data": {
    "methods": [
      {
        "id": "kirchenbauer",
        "name": "Kirchenbauer et al.",
        "description": "Green-red list watermarking",
        "parameters": {
          "gamma": {"type": "float", "default": 0.25, "range": [0.1, 0.9]},
          "delta": {"type": "float", "default": 2.0, "range": [0.5, 5.0]},
          "seed": {"type": "integer", "default": 42}
        },
        "models_supported": ["gpt2", "opt", "llama"],
        "paper_url": "https://arxiv.org/abs/2301.10226"
      }
    ]
  }
}
```

### Text Generation

#### POST /api/v1/generate

Generate watermarked text.

```bash
curl -X POST https://api.watermark-lab.com/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "method": "kirchenbauer",
    "model": "gpt2-medium",
    "prompt": "The future of AI is",
    "max_length": 200,
    "temperature": 0.7,
    "watermark_config": {
      "gamma": 0.25,
      "delta": 2.0,
      "seed": 42
    }
  }'
```

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `method` | string | Yes | Watermarking method ID |
| `model` | string | Yes | Model identifier |
| `prompt` | string | Yes | Input prompt |
| `max_length` | integer | No | Maximum generated length (default: 100) |
| `temperature` | float | No | Generation temperature (default: 0.7) |
| `watermark_config` | object | No | Method-specific parameters |

**Response:**
```json
{
  "success": true,
  "data": {
    "text": "The future of AI is promising with advances in machine learning...",
    "metadata": {
      "method": "kirchenbauer",
      "model": "gpt2-medium",
      "generation_time_ms": 1250,
      "token_count": 45,
      "watermark_strength": 2.3
    },
    "watermark_config": {
      "gamma": 0.25,
      "delta": 2.0,
      "seed": 42
    }
  }
}
```

#### POST /api/v1/generate/batch

Generate multiple watermarked texts.

```bash
curl -X POST https://api.watermark-lab.com/api/v1/generate/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "method": "kirchenbauer",
    "model": "gpt2-medium",
    "prompts": [
      "The future of AI is",
      "Machine learning enables",
      "Natural language processing"
    ],
    "watermark_config": {
      "gamma": 0.25,
      "delta": 2.0
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "prompt": "The future of AI is",
        "text": "The future of AI is promising...",
        "metadata": {...}
      }
    ],
    "batch_metadata": {
      "total_prompts": 3,
      "successful": 3,
      "failed": 0,
      "total_time_ms": 3500
    }
  }
}
```

### Watermark Detection

#### POST /api/v1/detect

Detect watermarks in text.

```bash
curl -X POST https://api.watermark-lab.com/api/v1/detect \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "text": "The text to analyze for watermarks...",
    "watermark_config": {
      "method": "kirchenbauer",
      "gamma": 0.25,
      "delta": 2.0,
      "seed": 42
    },
    "return_details": true
  }'
```

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to analyze |
| `watermark_config` | object | Yes | Watermark configuration |
| `return_details` | boolean | No | Include detailed analysis |

**Response:**
```json
{
  "success": true,
  "data": {
    "is_watermarked": true,
    "confidence": 0.95,
    "p_value": 0.001,
    "score": 4.2,
    "threshold": 2.0,
    "details": {
      "test_statistic": 4.2,
      "num_tokens": 45,
      "green_tokens": 28,
      "red_tokens": 17,
      "token_scores": [0.8, 0.3, 0.9, ...]
    },
    "metadata": {
      "detection_time_ms": 150,
      "method": "kirchenbauer"
    }
  }
}
```

#### POST /api/v1/detect/batch

Detect watermarks in multiple texts.

```bash
curl -X POST https://api.watermark-lab.com/api/v1/detect/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "texts": ["Text 1...", "Text 2...", "Text 3..."],
    "watermark_config": {
      "method": "kirchenbauer",
      "gamma": 0.25,
      "delta": 2.0
    }
  }'
```

### Quality Evaluation

#### POST /api/v1/evaluate/quality

Evaluate text quality metrics.

```bash
curl -X POST https://api.watermark-lab.com/api/v1/evaluate/quality \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "original_text": "Original reference text...",
    "watermarked_text": "Watermarked version of text...",
    "metrics": ["perplexity", "bleu", "bertscore"]
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "metrics": {
      "perplexity": {
        "original": 15.3,
        "watermarked": 18.7,
        "degradation": 0.22
      },
      "bleu": {
        "score": 0.85,
        "precision": [0.92, 0.86, 0.81, 0.78]
      },
      "bertscore": {
        "precision": 0.89,
        "recall": 0.87,
        "f1": 0.88
      }
    },
    "overall_score": 0.82,
    "quality_assessment": "good"
  }
}
```

### Attack Simulation

#### POST /api/v1/attack

Simulate attacks on watermarked text.

```bash
curl -X POST https://api.watermark-lab.com/api/v1/attack \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "text": "Watermarked text to attack...",
    "attack_type": "paraphrase",
    "attack_config": {
      "strength": "medium",
      "model": "t5-base"
    },
    "original_watermark_config": {
      "method": "kirchenbauer",
      "gamma": 0.25,
      "delta": 2.0
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "attacked_text": "Paraphrased version of the text...",
    "attack_success": true,
    "watermark_preserved": false,
    "similarity_score": 0.78,
    "attack_metadata": {
      "attack_type": "paraphrase",
      "strength": "medium",
      "processing_time_ms": 2300
    },
    "detection_results": {
      "before_attack": {
        "is_watermarked": true,
        "confidence": 0.95
      },
      "after_attack": {
        "is_watermarked": false,
        "confidence": 0.12
      }
    }
  }
}
```

### File Operations

#### POST /api/v1/upload

Upload files for batch processing.

```bash
curl -X POST https://api.watermark-lab.com/api/v1/upload \
  -H "X-API-Key: your-api-key" \
  -F "file=@texts.jsonl" \
  -F "operation=generate" \
  -F "config={\"method\":\"kirchenbauer\"}"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "upload_id": "upload_123456",
    "status": "processing",
    "file_info": {
      "filename": "texts.jsonl",
      "size_bytes": 1024,
      "num_records": 100
    },
    "estimated_completion": "2024-01-27T10:05:00Z"
  }
}
```

#### GET /api/v1/upload/{upload_id}/status

Check upload processing status.

```bash
curl -H "X-API-Key: your-api-key" \
  https://api.watermark-lab.com/api/v1/upload/upload_123456/status
```

#### GET /api/v1/upload/{upload_id}/download

Download processed results.

```bash
curl -H "X-API-Key: your-api-key" \
  https://api.watermark-lab.com/api/v1/upload/upload_123456/download \
  -o results.jsonl
```

## WebSocket API

### Connection

```javascript
const ws = new WebSocket('wss://api.watermark-lab.com/ws/v1/stream');
```

### Authentication

```javascript
// Send authentication message
ws.send(JSON.stringify({
  type: 'auth',
  token: 'your-jwt-token'
}));
```

### Real-time Generation

```javascript
// Request streaming generation
ws.send(JSON.stringify({
  type: 'generate',
  data: {
    method: 'kirchenbauer',
    model: 'gpt2-medium',
    prompt: 'The future of AI is',
    stream: true
  }
}));

// Receive streaming tokens
ws.onmessage = function(event) {
  const message = JSON.parse(event.data);
  if (message.type === 'token') {
    console.log('New token:', message.data.token);
  } else if (message.type === 'complete') {
    console.log('Generation complete:', message.data.text);
  }
};
```

## SDKs and Libraries

### Python SDK

```python
from watermark_lab import WatermarkLabClient

# Initialize client
client = WatermarkLabClient(api_key="your-api-key")

# Generate watermarked text
result = client.generate(
    method="kirchenbauer",
    model="gpt2-medium",
    prompt="The future of AI is",
    max_length=200
)

# Detect watermark
detection = client.detect(
    text=result.text,
    watermark_config=result.watermark_config
)

print(f"Watermarked: {detection.is_watermarked}")
print(f"Confidence: {detection.confidence}")
```

### JavaScript SDK

```javascript
import { WatermarkLabClient } from 'watermark-lab-js';

const client = new WatermarkLabClient({
  apiKey: 'your-api-key'
});

// Generate watermarked text
const result = await client.generate({
  method: 'kirchenbauer',
  model: 'gpt2-medium',
  prompt: 'The future of AI is',
  maxLength: 200
});

// Detect watermark
const detection = await client.detect({
  text: result.text,
  watermarkConfig: result.watermarkConfig
});

console.log(`Watermarked: ${detection.isWatermarked}`);
```

## Examples

### Complete Workflow Example

```python
import requests

API_BASE = "https://api.watermark-lab.com/api/v1"
API_KEY = "your-api-key"
headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# 1. List available methods
methods_response = requests.get(f"{API_BASE}/methods", headers=headers)
methods = methods_response.json()["data"]["methods"]
print(f"Available methods: {[m['id'] for m in methods]}")

# 2. Generate watermarked text
generate_data = {
    "method": "kirchenbauer",
    "model": "gpt2-medium",
    "prompt": "The impact of artificial intelligence on society",
    "max_length": 150,
    "watermark_config": {
        "gamma": 0.25,
        "delta": 2.0,
        "seed": 42
    }
}

generate_response = requests.post(
    f"{API_BASE}/generate", 
    json=generate_data, 
    headers=headers
)
generated = generate_response.json()["data"]
print(f"Generated text: {generated['text']}")

# 3. Detect watermark
detect_data = {
    "text": generated["text"],
    "watermark_config": generated["watermark_config"],
    "return_details": True
}

detect_response = requests.post(
    f"{API_BASE}/detect", 
    json=detect_data, 
    headers=headers
)
detection = detect_response.json()["data"]
print(f"Watermark detected: {detection['is_watermarked']}")
print(f"Confidence: {detection['confidence']}")

# 4. Evaluate quality
quality_data = {
    "original_text": "The impact of artificial intelligence on society is profound...",
    "watermarked_text": generated["text"],
    "metrics": ["perplexity", "bleu"]
}

quality_response = requests.post(
    f"{API_BASE}/evaluate/quality", 
    json=quality_data, 
    headers=headers
)
quality = quality_response.json()["data"]
print(f"Quality score: {quality['overall_score']}")

# 5. Simulate attack
attack_data = {
    "text": generated["text"],
    "attack_type": "paraphrase",
    "attack_config": {"strength": "light"},
    "original_watermark_config": generated["watermark_config"]
}

attack_response = requests.post(
    f"{API_BASE}/attack", 
    json=attack_data, 
    headers=headers
)
attack_result = attack_response.json()["data"]
print(f"Attack success: {attack_result['attack_success']}")
print(f"Watermark preserved: {attack_result['watermark_preserved']}")
```

This API documentation provides comprehensive coverage of all available endpoints, authentication methods, and usage examples for the LM Watermark Lab API.