#!/usr/bin/env python3
"""Minimal deployment readiness test focusing on core functionality."""

import os
import sys
import json
import subprocess
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test that core modules can be imported."""
    results = {"status": "TESTING", "tests": []}
    
    # Test basic package import
    try:
        import watermark_lab
        results["tests"].append({"name": "Core package import", "status": "PASS"})
    except Exception as e:
        results["tests"].append({"name": "Core package import", "status": "FAIL", "error": str(e)})
    
    # Test detector import (should work without torch)
    try:
        from watermark_lab.core.detector import WatermarkDetector
        results["tests"].append({"name": "Detector import", "status": "PASS"})
    except Exception as e:
        results["tests"].append({"name": "Detector import", "status": "FAIL", "error": str(e)})
    
    # Test security system
    try:
        from watermark_lab.security.advanced_security import AdvancedSecuritySystem
        security = AdvancedSecuritySystem()
        results["tests"].append({"name": "Security system", "status": "PASS"})
    except Exception as e:
        results["tests"].append({"name": "Security system", "status": "FAIL", "error": str(e)})
    
    return results

def test_docker_availability():
    """Test Docker and Docker Compose availability."""
    results = {"status": "TESTING", "tests": []}
    
    # Test Docker
    try:
        docker_version = subprocess.check_output(["docker", "--version"], text=True).strip()
        results["tests"].append({"name": "Docker available", "status": "PASS", "version": docker_version})
    except Exception as e:
        results["tests"].append({"name": "Docker available", "status": "FAIL", "error": str(e)})
    
    # Test Docker Compose
    try:
        compose_version = subprocess.check_output(["docker-compose", "--version"], text=True).strip()
        results["tests"].append({"name": "Docker Compose available", "status": "PASS", "version": compose_version})
    except Exception as e:
        results["tests"].append({"name": "Docker Compose available", "status": "FAIL", "error": str(e)})
    
    return results

def test_configuration_files():
    """Test that essential configuration files exist."""
    results = {"status": "TESTING", "tests": []}
    
    essential_files = [
        "pyproject.toml",
        "Dockerfile", 
        "docker-compose.yml",
        "src/watermark_lab/__init__.py",
        "scripts/deploy.sh"
    ]
    
    for file_path in essential_files:
        if os.path.exists(file_path):
            results["tests"].append({"name": f"Config file {file_path}", "status": "PASS"})
        else:
            results["tests"].append({"name": f"Config file {file_path}", "status": "FAIL", "error": "File not found"})
    
    return results

def main():
    """Run minimal deployment tests."""
    print("🚀 MINIMAL DEPLOYMENT READINESS TEST")
    print("=" * 50)
    
    all_results = {
        "basic_imports": test_basic_imports(),
        "docker_availability": test_docker_availability(), 
        "configuration_files": test_configuration_files()
    }
    
    # Calculate overall stats
    total_tests = 0
    passed_tests = 0
    
    for category, results in all_results.items():
        print(f"\n🔧 {category.replace('_', ' ').title()}")
        print("-" * 30)
        
        for test in results["tests"]:
            status_icon = "✅" if test["status"] == "PASS" else "❌"
            print(f"{status_icon} {test['name']}")
            if test["status"] == "FAIL":
                print(f"   Error: {test.get('error', 'Unknown error')}")
            if "version" in test:
                print(f"   Version: {test['version']}")
            
            total_tests += 1
            if test["status"] == "PASS":
                passed_tests += 1
    
    # Summary
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f"\n📊 DEPLOYMENT READINESS SUMMARY")
    print("=" * 40)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    
    # Overall status
    if pass_rate >= 80:
        print("🎉 DEPLOYMENT STATUS: READY")
        status = "READY"
    elif pass_rate >= 60:
        print("⚠️  DEPLOYMENT STATUS: PARTIAL")
        status = "PARTIAL"
    else:
        print("❌ DEPLOYMENT STATUS: NOT READY")
        status = "NOT_READY"
    
    # Save results
    summary = {
        "status": status,
        "pass_rate": pass_rate,
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": total_tests - passed_tests,
        "detailed_results": all_results
    }
    
    with open("minimal_deployment_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Report saved to: minimal_deployment_report.json")
    return 0 if status in ["READY", "PARTIAL"] else 1

if __name__ == "__main__":
    exit(main())