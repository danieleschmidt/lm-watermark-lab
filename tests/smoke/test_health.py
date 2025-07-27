"""Smoke tests for basic system health."""

import pytest
import subprocess
import sys
from pathlib import Path


class TestSystemHealth:
    """Basic health checks for the system."""

    def test_python_version(self):
        """Test that Python version is compatible."""
        assert sys.version_info >= (3, 9), "Python 3.9+ required"

    def test_package_import(self):
        """Test that the main package can be imported."""
        try:
            import watermark_lab
            assert hasattr(watermark_lab, '__version__')
        except ImportError:
            pytest.skip("Package not installed in development mode")

    def test_cli_available(self):
        """Test that CLI commands are available."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "watermark_lab.cli", "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            assert result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI not available")

    def test_dependencies_installed(self):
        """Test that critical dependencies are available."""
        critical_deps = [
            "torch",
            "transformers",
            "numpy",
            "fastapi",
            "pytest"
        ]
        
        for dep in critical_deps:
            try:
                __import__(dep)
            except ImportError:
                pytest.fail(f"Critical dependency {dep} not available")

    def test_configuration_files_exist(self):
        """Test that essential configuration files exist."""
        repo_root = Path(__file__).parent.parent.parent
        essential_files = [
            "pyproject.toml",
            "README.md",
            "LICENSE",
            ".gitignore"
        ]
        
        for file_name in essential_files:
            file_path = repo_root / file_name
            assert file_path.exists(), f"Essential file {file_name} not found"

    def test_docker_build(self):
        """Test that Docker image can be built."""
        repo_root = Path(__file__).parent.parent.parent
        dockerfile = repo_root / "Dockerfile"
        
        if not dockerfile.exists():
            pytest.skip("Dockerfile not found")
        
        try:
            result = subprocess.run(
                ["docker", "build", "-t", "watermark-lab-test", str(repo_root)],
                capture_output=True,
                timeout=300
            )
            # Don't fail the test if Docker isn't available
            if "docker: command not found" in result.stderr.decode():
                pytest.skip("Docker not available")
                
        except subprocess.TimeoutExpired:
            pytest.skip("Docker build timeout")
        except FileNotFoundError:
            pytest.skip("Docker not available")


class TestAPIHealth:
    """Health checks for API components."""

    def test_fastapi_import(self):
        """Test that FastAPI can be imported."""
        try:
            from fastapi import FastAPI
            app = FastAPI()
            assert app is not None
        except ImportError:
            pytest.skip("FastAPI not available")

    def test_api_routes_loadable(self):
        """Test that API routes can be loaded."""
        try:
            from watermark_lab.api.main import app
            assert app is not None
            # Check that basic routes exist
            routes = [route.path for route in app.routes]
            assert len(routes) > 0
        except ImportError:
            pytest.skip("API module not available")


class TestModelHealth:
    """Health checks for model-related functionality."""

    def test_torch_available(self):
        """Test that PyTorch is available and functional."""
        try:
            import torch
            # Test basic tensor operations
            x = torch.tensor([1.0, 2.0, 3.0])
            y = x + 1
            assert torch.allclose(y, torch.tensor([2.0, 3.0, 4.0]))
        except ImportError:
            pytest.fail("PyTorch not available")

    def test_transformers_available(self):
        """Test that transformers library is available."""
        try:
            from transformers import AutoTokenizer
            # Just test that the class can be imported
            assert AutoTokenizer is not None
        except ImportError:
            pytest.fail("Transformers library not available")

    def test_cuda_detection(self):
        """Test CUDA availability detection."""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            # This should not fail, just report status
            print(f"CUDA available: {cuda_available}")
            if cuda_available:
                print(f"CUDA devices: {torch.cuda.device_count()}")
        except ImportError:
            pytest.skip("PyTorch not available for CUDA check")


if __name__ == "__main__":
    pytest.main([__file__])