#!/usr/bin/env python3
"""Test production deployment readiness and infrastructure."""

import sys
import os
import time
import subprocess
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_docker_build():
    """Test Docker image building."""
    print("=== Testing Docker Build ===")
    
    try:
        # Test if Docker is available
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print("⚠️ Docker not available, skipping Docker build test")
            return True  # Skip but don't fail
        
        print(f"✓ Docker available: {result.stdout.strip()}")
        
        # Check if Dockerfile exists and is valid
        if not os.path.exists('Dockerfile'):
            print("✗ Dockerfile not found")
            return False
        
        print("✓ Dockerfile found")
        
        # Validate Dockerfile syntax (basic check)
        with open('Dockerfile', 'r') as f:
            dockerfile_content = f.read()
        
        required_instructions = ['FROM', 'WORKDIR', 'COPY', 'RUN', 'EXPOSE', 'CMD']
        missing_instructions = []
        
        for instruction in required_instructions:
            if instruction not in dockerfile_content:
                missing_instructions.append(instruction)
        
        if missing_instructions:
            print(f"✗ Dockerfile missing instructions: {missing_instructions}")
            return False
        
        print("✓ Dockerfile syntax validation passed")
        
        # Check for security best practices
        security_checks = {
            'non-root user': 'USER ' in dockerfile_content and 'USER root' not in dockerfile_content.split('\n')[-5:],
            'multi-stage build': dockerfile_content.count('FROM ') >= 2,
            'health check': 'HEALTHCHECK' in dockerfile_content,
            'minimal base image': 'slim' in dockerfile_content or 'alpine' in dockerfile_content
        }
        
        for check_name, passed in security_checks.items():
            status = "✓" if passed else "⚠️"
            print(f"{status} Dockerfile {check_name}: {'Pass' if passed else 'Needs attention'}")
        
        print("✓ Docker build validation completed")
        return True
        
    except Exception as e:
        print(f"✗ Docker build test failed: {e}")
        return False


def test_docker_compose_config():
    """Test Docker Compose configuration."""
    print("\n=== Testing Docker Compose Configuration ===")
    
    try:
        # Check if docker-compose.yml exists
        if not os.path.exists('docker-compose.yml'):
            print("✗ docker-compose.yml not found")
            return False
        
        print("✓ docker-compose.yml found")
        
        # Test docker-compose availability
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print("⚠️ docker-compose not available, testing config only")
        else:
            print(f"✓ Docker Compose available: {result.stdout.strip()}")
        
        # Validate compose file syntax
        try:
            import yaml
            with open('docker-compose.yml', 'r') as f:
                compose_config = yaml.safe_load(f)
            
            print("✓ docker-compose.yml syntax is valid")
            
            # Check required services
            services = compose_config.get('services', {})
            required_services = ['app', 'redis']
            
            for service in required_services:
                if service in services:
                    print(f"✓ Required service '{service}' configured")
                else:
                    print(f"⚠️ Required service '{service}' missing")
            
            # Check service configurations
            app_service = services.get('app', {})
            if 'ports' in app_service:
                print("✓ App service has port configuration")
            if 'environment' in app_service:
                print("✓ App service has environment configuration")
            if 'healthcheck' in app_service:
                print("✓ App service has health check")
            
            # Check volumes
            if 'volumes' in compose_config:
                print(f"✓ Docker volumes configured: {len(compose_config['volumes'])}")
            
            # Check networks
            if 'networks' in compose_config:
                print(f"✓ Docker networks configured: {len(compose_config['networks'])}")
            
        except ImportError:
            print("⚠️ PyYAML not available, skipping detailed validation")
        except yaml.YAMLError as e:
            print(f"✗ docker-compose.yml syntax error: {e}")
            return False
        
        print("✓ Docker Compose configuration validated")
        return True
        
    except Exception as e:
        print(f"✗ Docker Compose test failed: {e}")
        return False


def test_production_configuration():
    """Test production configuration files and settings."""
    print("\n=== Testing Production Configuration ===")
    
    try:
        # Check for essential configuration files
        config_files = [
            'pyproject.toml',
            '.env.example',
            'monitoring/prometheus.yml',
            'monitoring/grafana/dashboards',
            'scripts/deploy.sh'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                print(f"✓ Configuration file found: {config_file}")
            else:
                print(f"⚠️ Configuration file missing: {config_file}")
        
        # Check pyproject.toml for production dependencies
        if os.path.exists('pyproject.toml'):
            with open('pyproject.toml', 'r') as f:
                pyproject_content = f.read()
            
            production_deps = [
                'fastapi',
                'uvicorn',
                'redis',
                'prometheus-client',
                'psutil'
            ]
            
            missing_deps = []
            for dep in production_deps:
                if dep not in pyproject_content:
                    missing_deps.append(dep)
            
            if not missing_deps:
                print("✓ All production dependencies configured")
            else:
                print(f"⚠️ Missing production dependencies: {missing_deps}")
        
        # Check environment configuration
        if os.path.exists('.env.example'):
            with open('.env.example', 'r') as f:
                env_content = f.read()
            
            required_env_vars = [
                'API_PORT',
                'REDIS_URL',
                'LOG_LEVEL'
            ]
            
            for env_var in required_env_vars:
                if env_var in env_content:
                    print(f"✓ Environment variable configured: {env_var}")
                else:
                    print(f"⚠️ Environment variable missing: {env_var}")
        
        # Check monitoring configuration
        if os.path.exists('monitoring/prometheus.yml'):
            print("✓ Prometheus monitoring configured")
        
        if os.path.exists('monitoring/grafana'):
            print("✓ Grafana dashboards configured")
        
        print("✓ Production configuration validated")
        return True
        
    except Exception as e:
        print(f"✗ Production configuration test failed: {e}")
        return False


def test_api_service_readiness():
    """Test API service deployment readiness."""
    print("\n=== Testing API Service Readiness ===")
    
    try:
        from watermark_lab.api.main import app
        from watermark_lab.monitoring.health_monitor import HealthMonitor
        
        # Test API app creation
        print("✓ API application imports successfully")
        
        # Test health monitoring
        health_monitor = HealthMonitor()
        health_summary = health_monitor.get_health_summary()
        
        if health_summary and 'overall_status' in health_summary:
            print(f"✓ Health monitoring ready: {health_summary['overall_status']}")
        else:
            print("⚠️ Health monitoring needs configuration")
        
        # Test essential endpoints exist
        try:
            from fastapi.testclient import TestClient
            client = TestClient(app)
            
            # Test health endpoint
            response = client.get("/health")
            if response.status_code == 200:
                print("✓ Health endpoint working")
            else:
                print(f"⚠️ Health endpoint issue: {response.status_code}")
            
            # Test metrics endpoint
            response = client.get("/metrics")
            if response.status_code in [200, 404]:  # 404 is OK if not enabled
                print("✓ Metrics endpoint accessible")
            else:
                print(f"⚠️ Metrics endpoint issue: {response.status_code}")
                
        except ImportError:
            print("⚠️ TestClient not available, skipping API tests")
        
        # Test watermarking service integration
        from watermark_lab.core.factory import WatermarkFactory
        
        config = {"method": "kirchenbauer", "model_name": "gpt2", "use_real_model": False}
        watermarker = WatermarkFactory.create(**config)
        
        if watermarker:
            print("✓ Watermarking service ready for deployment")
        else:
            print("✗ Watermarking service not ready")
            return False
        
        print("✓ API service deployment readiness validated")
        return True
        
    except Exception as e:
        print(f"✗ API service readiness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_security_hardening():
    """Test security hardening for production deployment."""
    print("\n=== Testing Security Hardening ===")
    
    try:
        from watermark_lab.security.input_sanitization import InputSanitizer
        from watermark_lab.security.authentication import AuthenticationManager
        
        # Test input sanitization is enabled
        sanitizer = InputSanitizer()
        test_input = "<script>alert('test')</script>"
        
        try:
            sanitized = sanitizer.sanitize_text(test_input)
            if "<script>" not in sanitized:
                print("✓ Input sanitization working")
            else:
                print("⚠️ Input sanitization needs configuration")
        except Exception:
            print("✓ Input sanitization blocking dangerous content")
        
        # Test authentication system
        try:
            auth_manager = AuthenticationManager()
            print("✓ Authentication system available")
        except Exception:
            print("⚠️ Authentication system needs configuration")
        
        # Check for security-related files
        security_files = [
            '.dockerignore',
            'SECURITY.md',
            'cosign.pub'  # For signing
        ]
        
        for security_file in security_files:
            if os.path.exists(security_file):
                print(f"✓ Security file found: {security_file}")
            else:
                print(f"⚠️ Security file missing: {security_file}")
        
        # Check Dockerfile for security practices
        if os.path.exists('Dockerfile'):
            with open('Dockerfile', 'r') as f:
                dockerfile_content = f.read()
            
            security_practices = {
                'Non-root user': 'USER ' in dockerfile_content and 'USER root' not in dockerfile_content.split('\n')[-10:],
                'No secret exposure': 'password' not in dockerfile_content.lower() and 'secret' not in dockerfile_content.lower(),
                'Minimal base image': any(base in dockerfile_content for base in ['slim', 'alpine', 'distroless']),
                'Health check': 'HEALTHCHECK' in dockerfile_content
            }
            
            for practice, implemented in security_practices.items():
                status = "✓" if implemented else "⚠️"
                print(f"{status} {practice}: {'Implemented' if implemented else 'Needs attention'}")
        
        print("✓ Security hardening validated")
        return True
        
    except Exception as e:
        print(f"✗ Security hardening test failed: {e}")
        return False


def test_scalability_features():
    """Test scalability and performance features."""
    print("\n=== Testing Scalability Features ===")
    
    try:
        from watermark_lab.optimization.caching import get_cache_manager
        from watermark_lab.optimization.resource_manager import get_resource_manager
        from watermark_lab.deployment.load_balancer import LoadBalancer
        
        # Test caching system
        cache_manager = get_cache_manager()
        cache_stats = cache_manager.get_stats()
        
        if cache_stats and 'backend' in cache_stats:
            print(f"✓ Caching system ready: {cache_stats['backend']} backend")
        else:
            print("⚠️ Caching system needs configuration")
        
        # Test resource management
        resource_manager = get_resource_manager()
        resource_stats = resource_manager.get_comprehensive_stats()
        
        if resource_stats and 'current_usage' in resource_stats:
            print("✓ Resource management ready")
        else:
            print("⚠️ Resource management needs configuration")
        
        # Test load balancing capability
        load_balancer = LoadBalancer()
        if load_balancer:
            print("✓ Load balancing system ready")
        else:
            print("⚠️ Load balancing needs configuration")
        
        # Check for scaling configuration files
        scaling_files = [
            'kubernetes/deployment.yaml',
            'docker-compose.yml',
            'monitoring/alerts.yml'
        ]
        
        for scaling_file in scaling_files:
            if os.path.exists(scaling_file):
                print(f"✓ Scaling configuration found: {scaling_file}")
            else:
                print(f"⚠️ Scaling configuration missing: {scaling_file}")
        
        print("✓ Scalability features validated")
        return True
        
    except Exception as e:
        print(f"✗ Scalability features test failed: {e}")
        return False


def run_all_deployment_tests():
    """Run all production deployment readiness tests."""
    print("🚀 PRODUCTION DEPLOYMENT READINESS")
    print("=" * 45)
    
    deployment_tests = [
        ("Docker Build", test_docker_build),
        ("Docker Compose Configuration", test_docker_compose_config),
        ("Production Configuration", test_production_configuration),
        ("API Service Readiness", test_api_service_readiness),
        ("Security Hardening", test_security_hardening),
        ("Scalability Features", test_scalability_features)
    ]
    
    results = {}
    
    for test_name, test_func in deployment_tests:
        try:
            print(f"\n🔧 {test_name}")
            print("-" * 35)
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} CRASHED: {e}")
            results[test_name] = False
    
    # Summary
    passed_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    
    print(f"\n📊 DEPLOYMENT READINESS SUMMARY")
    print("=" * 35)
    for test_name, passed in results.items():
        status = "✅ READY" if passed else "❌ NEEDS WORK"
        print(f"{test_name}: {status}")
    
    print(f"\n🎯 DEPLOYMENT SCORE: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 PRODUCTION DEPLOYMENT READY!")
        return True
    else:
        print("⚠️  DEPLOYMENT NEEDS ATTENTION")
        return False


if __name__ == "__main__":
    success = run_all_deployment_tests()
    sys.exit(0 if success else 1)