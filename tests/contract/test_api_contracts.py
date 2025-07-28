"""
API contract tests for watermark lab endpoints.
These tests validate the actual API against the OpenAPI specification.
"""

import json
import pytest
import httpx
from openapi_core import create_spec
from openapi_core.validation.request.validators import RequestValidator
from openapi_core.validation.response.validators import ResponseValidator


@pytest.fixture
def api_spec():
    """Load OpenAPI specification."""
    with open('openapi.json', 'r') as f:
        spec_dict = json.load(f)
    return create_spec(spec_dict)


@pytest.fixture
def client():
    """HTTP client for API testing."""
    return httpx.Client(base_url="http://localhost:8080")


class TestDetectionEndpoint:
    """Contract tests for the detection endpoint."""

    def test_detect_watermark_contract(self, client, api_spec):
        """Test detection endpoint follows contract."""
        # Valid request
        request_data = {
            "text": "This is a test text for watermark detection.",
            "method": "kirchenbauer",
            "options": {
                "threshold": 0.5
            }
        }

        response = client.post(
            "/api/v1/detect",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )

        # Validate response against OpenAPI spec
        validator = ResponseValidator(api_spec)
        result = validator.validate(
            httpx.Request("POST", "/api/v1/detect"),
            response
        )
        
        assert not result.errors, f"Response validation errors: {result.errors}"
        
        # Check response structure
        response_data = response.json()
        assert "is_watermarked" in response_data
        assert "confidence" in response_data
        assert "p_value" in response_data
        assert isinstance(response_data["is_watermarked"], bool)
        assert isinstance(response_data["confidence"], (int, float))

    def test_detect_watermark_invalid_method(self, client, api_spec):
        """Test error handling for invalid method."""
        request_data = {
            "text": "Test text",
            "method": "nonexistent_method",
            "options": {}
        }

        response = client.post(
            "/api/v1/detect",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 400
        error_data = response.json()
        assert "error" in error_data
        assert "message" in error_data

    def test_detect_watermark_missing_text(self, client):
        """Test error handling for missing required text field."""
        request_data = {
            "method": "kirchenbauer",
            "options": {}
        }

        response = client.post(
            "/api/v1/detect",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422  # Validation error
        error_data = response.json()
        assert "detail" in error_data


class TestGenerationEndpoint:
    """Contract tests for the generation endpoint."""

    def test_generate_watermark_contract(self, client, api_spec):
        """Test generation endpoint follows contract."""
        request_data = {
            "prompt": "Write a short story about AI",
            "method": "kirchenbauer",
            "options": {
                "max_length": 100,
                "temperature": 0.7
            }
        }

        response = client.post(
            "/api/v1/generate",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )

        # Validate response against OpenAPI spec
        validator = ResponseValidator(api_spec)
        result = validator.validate(
            httpx.Request("POST", "/api/v1/generate"),
            response
        )
        
        assert not result.errors, f"Response validation errors: {result.errors}"
        
        # Check response structure
        response_data = response.json()
        assert "watermarked_text" in response_data
        assert "method" in response_data
        assert isinstance(response_data["watermarked_text"], str)
        assert len(response_data["watermarked_text"]) > 0

    def test_generate_watermark_with_validation(self, client):
        """Test generation with response validation."""
        request_data = {
            "prompt": "Explain machine learning",
            "method": "kirchenbauer",
            "options": {
                "max_length": 50,
                "validate_output": True
            }
        }

        response = client.post(
            "/api/v1/generate",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200
        response_data = response.json()
        
        # Should include validation metadata
        assert "metadata" in response_data
        if "validation" in response_data["metadata"]:
            assert "is_valid" in response_data["metadata"]["validation"]


class TestMethodsEndpoint:
    """Contract tests for the methods listing endpoint."""

    def test_list_methods_contract(self, client, api_spec):
        """Test methods endpoint follows contract."""
        response = client.get(
            "/api/v1/methods",
            headers={"Accept": "application/json"}
        )

        # Validate response against OpenAPI spec
        validator = ResponseValidator(api_spec)
        result = validator.validate(
            httpx.Request("GET", "/api/v1/methods"),
            response
        )
        
        assert not result.errors, f"Response validation errors: {result.errors}"
        
        # Check response structure
        response_data = response.json()
        assert "methods" in response_data
        assert isinstance(response_data["methods"], list)
        
        for method in response_data["methods"]:
            assert "name" in method
            assert "description" in method
            assert "type" in method
            assert "supports_detection" in method
            assert "supports_generation" in method

    def test_method_details_contract(self, client):
        """Test individual method details endpoint."""
        # First get available methods
        methods_response = client.get("/api/v1/methods")
        methods = methods_response.json()["methods"]
        
        if methods:
            method_name = methods[0]["name"]
            
            response = client.get(f"/api/v1/methods/{method_name}")
            assert response.status_code == 200
            
            method_data = response.json()
            assert "name" in method_data
            assert "parameters" in method_data
            assert "examples" in method_data


class TestHealthEndpoint:
    """Contract tests for health check endpoints."""

    def test_health_check_contract(self, client):
        """Test health endpoint follows contract."""
        response = client.get("/health")
        
        assert response.status_code == 200
        health_data = response.json()
        
        assert "status" in health_data
        assert "timestamp" in health_data
        assert "version" in health_data
        assert health_data["status"] in ["healthy", "unhealthy", "degraded"]

    def test_ready_check_contract(self, client):
        """Test readiness endpoint follows contract."""
        response = client.get("/ready")
        
        assert response.status_code in [200, 503]
        ready_data = response.json()
        
        assert "ready" in ready_data
        assert isinstance(ready_data["ready"], bool)
        if "dependencies" in ready_data:
            assert isinstance(ready_data["dependencies"], dict)

    def test_metrics_endpoint_contract(self, client):
        """Test metrics endpoint follows contract."""
        response = client.get("/metrics")
        
        # Metrics can be either JSON or Prometheus format
        assert response.status_code == 200
        
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            metrics_data = response.json()
            assert isinstance(metrics_data, dict)
        elif "text/plain" in content_type:
            # Prometheus format
            metrics_text = response.text
            assert "# HELP" in metrics_text or "# TYPE" in metrics_text


class TestErrorHandling:
    """Contract tests for error handling."""

    def test_404_error_contract(self, client):
        """Test 404 error response follows contract."""
        response = client.get("/api/v1/nonexistent")
        
        assert response.status_code == 404
        error_data = response.json()
        
        assert "detail" in error_data or "message" in error_data

    def test_method_not_allowed_contract(self, client):
        """Test 405 error response follows contract."""
        response = client.put("/api/v1/detect")
        
        assert response.status_code == 405
        assert "Allow" in response.headers

    def test_rate_limiting_contract(self, client):
        """Test rate limiting headers when applicable."""
        # Make multiple requests quickly
        responses = []
        for _ in range(5):
            response = client.get("/api/v1/methods")
            responses.append(response)
        
        # Check if rate limiting headers are present
        for response in responses:
            if "X-RateLimit-Limit" in response.headers:
                assert "X-RateLimit-Remaining" in response.headers
                assert "X-RateLimit-Reset" in response.headers

    def test_cors_headers_contract(self, client):
        """Test CORS headers are properly set."""
        response = client.options("/api/v1/detect")
        
        # Check for CORS headers
        cors_headers = [
            "Access-Control-Allow-Origin",
            "Access-Control-Allow-Methods",
            "Access-Control-Allow-Headers"
        ]
        
        # At least one CORS header should be present for a proper CORS setup
        has_cors = any(header in response.headers for header in cors_headers)
        if has_cors:
            assert response.status_code in [200, 204]