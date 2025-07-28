"""
Consumer contract tests for watermark detection API.
These tests define the expected behavior from the consumer's perspective.
"""

import json
import pytest
from pact import Consumer, Provider
from watermark_lab.client import WatermarkClient


@pytest.fixture
def pact():
    """Set up Pact consumer-provider contract."""
    return Consumer('watermark-client').has_pact_with(
        Provider('watermark-api'),
        host_name='localhost',
        port=1234
    )


class TestWatermarkDetectionPact:
    """Consumer contract tests for watermark detection."""

    def test_detect_watermark_success(self, pact):
        """Test successful watermark detection."""
        expected_response = {
            "is_watermarked": True,
            "confidence": 0.95,
            "p_value": 0.001,
            "method": "kirchenbauer",
            "metadata": {
                "token_count": 150,
                "watermark_strength": 2.0
            }
        }

        (pact
         .given('a text with valid watermark')
         .upon_receiving('a detection request')
         .with_request(
             method='POST',
             path='/api/v1/detect',
             headers={'Content-Type': 'application/json'},
             body={
                 "text": "This is a watermarked text sample...",
                 "method": "kirchenbauer",
                 "options": {}
             }
         )
         .will_respond_with(
             status=200,
             headers={'Content-Type': 'application/json'},
             body=expected_response
         ))

        with pact:
            client = WatermarkClient(base_url='http://localhost:1234')
            result = client.detect(
                text="This is a watermarked text sample...",
                method="kirchenbauer"
            )
            
            assert result.is_watermarked is True
            assert result.confidence == 0.95
            assert result.method == "kirchenbauer"

    def test_detect_watermark_not_found(self, pact):
        """Test detection when no watermark is present."""
        expected_response = {
            "is_watermarked": False,
            "confidence": 0.12,
            "p_value": 0.87,
            "method": "kirchenbauer",
            "metadata": {
                "token_count": 100,
                "watermark_strength": 0.0
            }
        }

        (pact
         .given('a text without watermark')
         .upon_receiving('a detection request')
         .with_request(
             method='POST',
             path='/api/v1/detect',
             headers={'Content-Type': 'application/json'},
             body={
                 "text": "This is clean text without any watermark.",
                 "method": "kirchenbauer",
                 "options": {}
             }
         )
         .will_respond_with(
             status=200,
             headers={'Content-Type': 'application/json'},
             body=expected_response
         ))

        with pact:
            client = WatermarkClient(base_url='http://localhost:1234')
            result = client.detect(
                text="This is clean text without any watermark.",
                method="kirchenbauer"
            )
            
            assert result.is_watermarked is False
            assert result.confidence == 0.12

    def test_generate_watermark_success(self, pact):
        """Test successful watermark generation."""
        expected_response = {
            "watermarked_text": "This is a sample watermarked text output...",
            "method": "kirchenbauer",
            "metadata": {
                "original_length": 25,
                "watermarked_length": 45,
                "gamma": 0.25,
                "delta": 2.0
            }
        }

        (pact
         .given('valid generation parameters')
         .upon_receiving('a generation request')
         .with_request(
             method='POST',
             path='/api/v1/generate',
             headers={'Content-Type': 'application/json'},
             body={
                 "prompt": "Write a story about AI",
                 "method": "kirchenbauer",
                 "options": {
                     "max_length": 100,
                     "temperature": 0.7
                 }
             }
         )
         .will_respond_with(
             status=200,
             headers={'Content-Type': 'application/json'},
             body=expected_response
         ))

        with pact:
            client = WatermarkClient(base_url='http://localhost:1234')
            result = client.generate(
                prompt="Write a story about AI",
                method="kirchenbauer",
                max_length=100,
                temperature=0.7
            )
            
            assert "watermarked" in result.watermarked_text.lower()
            assert result.method == "kirchenbauer"

    def test_list_methods(self, pact):
        """Test listing available watermark methods."""
        expected_response = {
            "methods": [
                {
                    "name": "kirchenbauer",
                    "description": "Kirchenbauer et al. watermarking",
                    "type": "statistical",
                    "supports_detection": True,
                    "supports_generation": True
                },
                {
                    "name": "markllm",
                    "description": "MarkLLM watermarking toolkit",
                    "type": "neural",
                    "supports_detection": True,
                    "supports_generation": True
                }
            ]
        }

        (pact
         .given('service is running')
         .upon_receiving('a request for available methods')
         .with_request(
             method='GET',
             path='/api/v1/methods',
             headers={'Accept': 'application/json'}
         )
         .will_respond_with(
             status=200,
             headers={'Content-Type': 'application/json'},
             body=expected_response
         ))

        with pact:
            client = WatermarkClient(base_url='http://localhost:1234')
            methods = client.list_methods()
            
            assert len(methods) >= 2
            assert any(m['name'] == 'kirchenbauer' for m in methods)
            assert any(m['name'] == 'markllm' for m in methods)

    def test_error_handling_invalid_method(self, pact):
        """Test error handling for invalid watermark method."""
        expected_error = {
            "error": "InvalidMethod",
            "message": "Watermark method 'invalid_method' not supported",
            "supported_methods": ["kirchenbauer", "markllm", "aaronson"]
        }

        (pact
         .given('invalid method name')
         .upon_receiving('a detection request with invalid method')
         .with_request(
             method='POST',
             path='/api/v1/detect',
             headers={'Content-Type': 'application/json'},
             body={
                 "text": "Sample text",
                 "method": "invalid_method",
                 "options": {}
             }
         )
         .will_respond_with(
             status=400,
             headers={'Content-Type': 'application/json'},
             body=expected_error
         ))

        with pact:
            client = WatermarkClient(base_url='http://localhost:1234')
            with pytest.raises(Exception) as exc_info:
                client.detect(
                    text="Sample text",
                    method="invalid_method"
                )
            assert "not supported" in str(exc_info.value)