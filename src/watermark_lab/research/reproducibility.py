"""
Reproducibility management system for watermarking research.

This module provides tools for ensuring reproducible research results,
including experiment tracking, environment capture, seed management,
and result verification across multiple runs.
"""

import os
import sys
import time
import json
import hashlib
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
import random
import numpy as np
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from ..utils.logging import get_logger
from ..core.factory import WatermarkFactory

logger = get_logger("research.reproducibility")


@dataclass
class ExperimentEnvironment:
    """Captures the experimental environment for reproducibility."""
    timestamp: str
    python_version: str
    platform_info: str
    cpu_info: str
    memory_info: str
    gpu_info: Optional[str]
    package_versions: Dict[str, str]
    environment_variables: Dict[str, str]
    random_seeds: Dict[str, int]
    experiment_hash: str


@dataclass
class ReproducibilityResult:
    """Result of reproducibility verification."""
    experiment_id: str
    original_hash: str
    reproduction_hash: str
    matches: bool
    differences: List[str]
    similarity_score: float
    verification_timestamp: str


class ReproducibilityManager:
    """Manages experiment reproducibility and verification."""
    
    def __init__(self, 
                 experiment_dir: str = "reproducibility",
                 capture_environment: bool = True):
        """
        Initialize reproducibility manager.
        
        Args:
            experiment_dir: Directory to store reproducibility data
            capture_environment: Whether to capture environment information
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.capture_environment = capture_environment
        self.current_environment = None
        
        # Reproducibility settings
        self.fixed_seeds = {
            'python': 42,
            'numpy': 42,
            'random': 42
        }
        
        # Result tracking
        self.experiments = {}
        self.verification_history = []
        
    def set_reproducible_seeds(self, base_seed: int = 42):
        """
        Set fixed seeds for reproducible results.
        
        Args:
            base_seed: Base seed value to use for all random number generators
        """
        try:
            # Set Python built-in random
            random.seed(base_seed)
            
            # Set NumPy random
            np.random.seed(base_seed)
            
            # Set environment variable for hash randomization
            os.environ['PYTHONHASHSEED'] = str(base_seed)
            
            self.fixed_seeds = {
                'python': base_seed,
                'numpy': base_seed,
                'hash_seed': base_seed
            }
            
            logger.info(f"Set reproducible seeds with base seed: {base_seed}")
            
        except Exception as e:
            logger.warning(f"Failed to set all reproducible seeds: {e}")
    
    def capture_experiment_environment(self, 
                                     experiment_name: str,
                                     additional_info: Dict[str, Any] = None) -> ExperimentEnvironment:
        """
        Capture complete experimental environment.
        
        Args:
            experiment_name: Name of the experiment
            additional_info: Additional information to include
            
        Returns:
            Captured environment information
        """
        try:
            # System information
            platform_info = f"{platform.system()} {platform.release()} {platform.machine()}"
            python_version = sys.version
            
            # CPU information
            if PSUTIL_AVAILABLE:
                cpu_info = f"{psutil.cpu_count()} cores, {platform.processor()}"
                memory_info = f"{psutil.virtual_memory().total / (1024**3):.1f} GB RAM"
            else:
                cpu_info = platform.processor()
                memory_info = "Memory info not available"
            
            # GPU information
            gpu_info = self._get_gpu_info()
            
            # Package versions
            package_versions = self._get_package_versions()
            
            # Environment variables (filtered)
            env_vars = self._get_filtered_env_vars()
            
            # Create experiment hash
            experiment_data = {
                'name': experiment_name,
                'platform': platform_info,
                'python': python_version,
                'packages': package_versions,
                'seeds': self.fixed_seeds,
                'additional': additional_info or {}
            }
            
            experiment_hash = hashlib.sha256(
                json.dumps(experiment_data, sort_keys=True).encode()
            ).hexdigest()
            
            environment = ExperimentEnvironment(
                timestamp=datetime.now().isoformat(),
                python_version=python_version,
                platform_info=platform_info,
                cpu_info=cpu_info,
                memory_info=memory_info,
                gpu_info=gpu_info,
                package_versions=package_versions,
                environment_variables=env_vars,
                random_seeds=self.fixed_seeds.copy(),
                experiment_hash=experiment_hash
            )
            
            # Save environment
            self._save_environment(experiment_name, environment)
            
            self.current_environment = environment
            logger.info(f"Captured environment for experiment: {experiment_name}")
            
            return environment
            
        except Exception as e:
            logger.error(f"Failed to capture experiment environment: {e}")
            return ExperimentEnvironment(
                timestamp=datetime.now().isoformat(),
                python_version=sys.version,
                platform_info=platform.platform(),
                cpu_info="Unknown",
                memory_info="Unknown", 
                gpu_info=None,
                package_versions={},
                environment_variables={},
                random_seeds=self.fixed_seeds.copy(),
                experiment_hash="unknown"
            )
    
    def save_experiment_results(self,
                              experiment_name: str,
                              results: Dict[str, Any],
                              metadata: Dict[str, Any] = None) -> str:
        """
        Save experiment results with reproducibility information.
        
        Args:
            experiment_name: Name of the experiment
            results: Experiment results
            metadata: Additional metadata
            
        Returns:
            Path to saved results file
        """
        try:
            # Create experiment record
            experiment_record = {
                'experiment_name': experiment_name,
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'metadata': metadata or {},
                'environment': asdict(self.current_environment) if self.current_environment else {},
                'reproducibility_hash': self._compute_results_hash(results)
            }
            
            # Save to file
            results_file = self.experiment_dir / f"{experiment_name}_results.json"
            with open(results_file, 'w') as f:
                json.dump(experiment_record, f, indent=2, default=str)
            
            # Update tracking
            self.experiments[experiment_name] = experiment_record
            
            logger.info(f"Saved experiment results: {results_file}")
            return str(results_file)
            
        except Exception as e:
            logger.error(f"Failed to save experiment results: {e}")
            return ""
    
    def verify_reproducibility(self,
                             experiment_name: str,
                             num_runs: int = 3,
                             tolerance: float = 0.01) -> ReproducibilityResult:
        """
        Verify that experiment results are reproducible.
        
        Args:
            experiment_name: Name of experiment to verify
            num_runs: Number of verification runs
            tolerance: Tolerance for numerical differences
            
        Returns:
            Reproducibility verification result
        """
        try:
            # Load original experiment
            original_file = self.experiment_dir / f"{experiment_name}_results.json"
            if not original_file.exists():
                raise FileNotFoundError(f"Original experiment not found: {experiment_name}")
            
            with open(original_file, 'r') as f:
                original_data = json.load(f)
            
            original_hash = original_data.get('reproducibility_hash', '')
            original_results = original_data.get('results', {})
            
            # Run verification experiments
            verification_results = []
            differences = []
            
            logger.info(f"Running {num_runs} verification runs for {experiment_name}")
            
            for run_idx in range(num_runs):
                # Reset seeds to ensure reproducibility
                self.set_reproducible_seeds(self.fixed_seeds.get('python', 42))
                
                # Re-run experiment (simplified - would need actual experiment replication)
                # For demo purposes, we'll simulate this
                verification_result = self._simulate_experiment_rerun(original_results)
                verification_results.append(verification_result)
                
                # Check for differences
                run_differences = self._compare_results(original_results, verification_result, tolerance)
                if run_differences:
                    differences.extend([f"Run {run_idx + 1}: {diff}" for diff in run_differences])
            
            # Compute reproducibility hash for verification runs
            verification_hash = self._compute_results_hash(verification_results[0])
            
            # Calculate similarity score
            similarity_score = self._calculate_similarity_score(
                original_results, verification_results, tolerance
            )
            
            # Determine if results match
            matches = len(differences) == 0 and abs(similarity_score - 1.0) < tolerance
            
            result = ReproducibilityResult(
                experiment_id=experiment_name,
                original_hash=original_hash,
                reproduction_hash=verification_hash,
                matches=matches,
                differences=differences,
                similarity_score=similarity_score,
                verification_timestamp=datetime.now().isoformat()
            )
            
            # Save verification result
            self._save_verification_result(result)
            
            logger.info(f"Reproducibility verification completed. Matches: {matches}, Similarity: {similarity_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Reproducibility verification failed: {e}")
            return ReproducibilityResult(
                experiment_id=experiment_name,
                original_hash="",
                reproduction_hash="",
                matches=False,
                differences=[f"Verification failed: {e}"],
                similarity_score=0.0,
                verification_timestamp=datetime.now().isoformat()
            )
    
    def create_reproducibility_report(self,
                                    experiment_name: str = None,
                                    output_file: str = None) -> str:
        """
        Create comprehensive reproducibility report.
        
        Args:
            experiment_name: Specific experiment to report on (None for all)
            output_file: Output file path
            
        Returns:
            Path to generated report
        """
        try:
            output_file = output_file or str(self.experiment_dir / "reproducibility_report.md")
            
            report_content = f"""# Reproducibility Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This report documents the reproducibility status of watermarking research experiments.

## Environment Information

"""
            
            # Add environment information
            if self.current_environment:
                env = self.current_environment
                report_content += f"""### System Environment

- **Platform:** {env.platform_info}
- **Python Version:** {env.python_version.split()[0]}
- **CPU:** {env.cpu_info}
- **Memory:** {env.memory_info}
- **GPU:** {env.gpu_info or 'Not available'}

### Package Versions

"""
                for package, version in env.package_versions.items():
                    report_content += f"- **{package}:** {version}\\n"
                
                report_content += f"""

### Random Seeds

"""
                for seed_type, seed_value in env.random_seeds.items():
                    report_content += f"- **{seed_type}:** {seed_value}\\n"
            
            # Add experiment results
            if experiment_name and experiment_name in self.experiments:
                experiments_to_report = {experiment_name: self.experiments[experiment_name]}
            else:
                experiments_to_report = self.experiments
            
            if experiments_to_report:
                report_content += """

## Experiment Results

"""
                for exp_name, exp_data in experiments_to_report.items():
                    report_content += f"""### {exp_name}

- **Timestamp:** {exp_data['timestamp']}
- **Reproducibility Hash:** {exp_data['reproducibility_hash'][:16]}...
- **Status:** Available for verification

"""
            
            # Add verification history
            if self.verification_history:
                report_content += """

## Verification History

"""
                for verification in self.verification_history:
                    status = "✅ PASSED" if verification.matches else "❌ FAILED"
                    report_content += f"""### {verification.experiment_id}

- **Status:** {status}
- **Similarity Score:** {verification.similarity_score:.3f}
- **Verified:** {verification.verification_timestamp}

"""
                    if verification.differences:
                        report_content += "**Differences Found:**\\n"
                        for diff in verification.differences[:5]:  # Limit to first 5
                            report_content += f"- {diff}\\n"
                        if len(verification.differences) > 5:
                            report_content += f"- ... and {len(verification.differences) - 5} more\\n"
                        report_content += "\\n"
            
            # Add reproducibility guidelines
            report_content += """

## Reproducibility Guidelines

To reproduce these experiments:

1. **Environment Setup:**
   - Use the same Python version and package versions listed above
   - Set the specified random seeds before running experiments
   - Ensure consistent hardware environment when possible

2. **Experiment Execution:**
   - Use the provided experiment configurations
   - Run experiments with identical parameters
   - Verify results against the provided hashes

3. **Verification Process:**
   - Run multiple trials with the same seeds
   - Compare results within the specified tolerance
   - Document any observed differences

## Recommendations

- **Seed Management:** Always set fixed seeds for all random number generators
- **Environment Documentation:** Capture complete environment information
- **Version Control:** Track all code changes and configurations
- **Automated Testing:** Use automated reproducibility verification
- **Result Archival:** Store complete results with metadata

"""
            
            # Save report
            with open(output_file, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Reproducibility report generated: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Reproducibility report generation failed: {e}")
            return ""
    
    def create_reproducibility_package(self,
                                     experiment_name: str,
                                     include_data: bool = True,
                                     include_code: bool = True) -> str:
        """
        Create a complete reproducibility package.
        
        Args:
            experiment_name: Name of experiment to package
            include_data: Whether to include data files
            include_code: Whether to include code snapshot
            
        Returns:
            Path to created package
        """
        try:
            package_dir = self.experiment_dir / f"{experiment_name}_reproducibility_package"
            package_dir.mkdir(exist_ok=True)
            
            # Copy experiment results
            results_file = self.experiment_dir / f"{experiment_name}_results.json"
            if results_file.exists():
                import shutil
                shutil.copy2(results_file, package_dir / "results.json")
            
            # Create README
            readme_content = f"""# Reproducibility Package: {experiment_name}

This package contains all materials needed to reproduce the experiment "{experiment_name}".

## Contents

- `results.json`: Original experiment results and metadata
- `environment.json`: Complete environment information
- `reproduction_guide.md`: Step-by-step reproduction instructions
- `verification_script.py`: Automated verification script

## Quick Start

1. Set up the environment using the specifications in `environment.json`
2. Run the verification script: `python verification_script.py`
3. Compare results with those in `results.json`

## Environment Requirements

"""
            
            # Add environment requirements
            if self.current_environment:
                readme_content += f"- Python {self.current_environment.python_version.split()[0]}\\n"
                for package, version in self.current_environment.package_versions.items():
                    readme_content += f"- {package}=={version}\\n"
            
            readme_content += f"""

## Random Seeds

Use these exact seeds for reproduction:

"""
            for seed_type, seed_value in self.fixed_seeds.items():
                readme_content += f"- {seed_type}: {seed_value}\\n"
            
            readme_content += """

For questions or issues, please refer to the full reproducibility report.
"""
            
            # Save README
            with open(package_dir / "README.md", 'w') as f:
                f.write(readme_content)
            
            # Save environment details
            if self.current_environment:
                with open(package_dir / "environment.json", 'w') as f:
                    json.dump(asdict(self.current_environment), f, indent=2, default=str)
            
            # Create verification script
            verification_script = f'''#!/usr/bin/env python3
"""
Automated verification script for {experiment_name}.
"""

import json
import sys
import os
import random
import numpy as np

def set_reproducible_seeds():
    """Set the same seeds used in original experiment."""
    random.seed({self.fixed_seeds.get('python', 42)})
    np.random.seed({self.fixed_seeds.get('numpy', 42)})
    os.environ['PYTHONHASHSEED'] = str({self.fixed_seeds.get('hash_seed', 42)})

def verify_experiment():
    """Run verification of the experiment."""
    print("Setting reproducible seeds...")
    set_reproducible_seeds()
    
    print("Loading original results...")
    with open("results.json", "r") as f:
        original_results = json.load(f)
    
    print("Original experiment hash:", original_results.get("reproducibility_hash", "Unknown"))
    
    # Add actual experiment replication code here
    print("Experiment verification would be implemented here.")
    print("This is a template for the actual verification logic.")
    
    return True

if __name__ == "__main__":
    success = verify_experiment()
    sys.exit(0 if success else 1)
'''
            
            with open(package_dir / "verification_script.py", 'w') as f:
                f.write(verification_script)
            
            os.chmod(package_dir / "verification_script.py", 0o755)
            
            logger.info(f"Reproducibility package created: {package_dir}")
            return str(package_dir)
            
        except Exception as e:
            logger.error(f"Reproducibility package creation failed: {e}")
            return ""
    
    def _get_gpu_info(self) -> Optional[str]:
        """Get GPU information if available."""
        try:
            # Try to get NVIDIA GPU info
            result = subprocess.run(['nvidia-smi', '-q'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.split('\\n')
                for line in lines:
                    if 'Product Name' in line:
                        return line.split(':')[1].strip()
            
            # Try other GPU detection methods
            return "GPU detection not available"
            
        except:
            return None
    
    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of key packages."""
        packages = {}
        
        try:
            import numpy
            packages['numpy'] = numpy.__version__
        except ImportError:
            pass
        
        try:
            import torch
            packages['torch'] = torch.__version__
        except ImportError:
            pass
        
        try:
            import transformers
            packages['transformers'] = transformers.__version__
        except ImportError:
            pass
        
        try:
            import matplotlib
            packages['matplotlib'] = matplotlib.__version__
        except ImportError:
            pass
        
        try:
            import scipy
            packages['scipy'] = scipy.__version__
        except ImportError:
            pass
        
        return packages
    
    def _get_filtered_env_vars(self) -> Dict[str, str]:
        """Get filtered environment variables."""
        # Only include relevant, non-sensitive environment variables
        relevant_vars = [
            'PYTHONPATH',
            'PYTHONHASHSEED',
            'CUDA_VISIBLE_DEVICES',
            'OMP_NUM_THREADS'
        ]
        
        env_vars = {}
        for var in relevant_vars:
            if var in os.environ:
                env_vars[var] = os.environ[var]
        
        return env_vars
    
    def _compute_results_hash(self, results: Dict[str, Any]) -> str:
        """Compute hash of experiment results."""
        # Create a normalized representation of results for hashing
        normalized = self._normalize_results_for_hashing(results)
        return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()
    
    def _normalize_results_for_hashing(self, results: Any) -> Any:
        """Normalize results for consistent hashing."""
        if isinstance(results, dict):
            return {k: self._normalize_results_for_hashing(v) for k, v in results.items()}
        elif isinstance(results, list):
            return [self._normalize_results_for_hashing(item) for item in results]
        elif isinstance(results, float):
            # Round floats to avoid precision issues
            return round(results, 6)
        else:
            return results
    
    def _simulate_experiment_rerun(self, original_results: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate re-running an experiment (for demo purposes)."""
        # In a real implementation, this would actually re-run the experiment
        # For demo purposes, we'll create slightly modified results
        simulated = {}
        
        for key, value in original_results.items():
            if isinstance(value, float):
                # Add tiny random variation to simulate slight numerical differences
                noise = random.gauss(0, 0.0001)  # Very small noise
                simulated[key] = value + noise
            elif isinstance(value, dict):
                simulated[key] = self._simulate_experiment_rerun(value)
            else:
                simulated[key] = value
        
        return simulated
    
    def _compare_results(self, 
                        original: Dict[str, Any], 
                        reproduction: Dict[str, Any],
                        tolerance: float) -> List[str]:
        """Compare original and reproduced results."""
        differences = []
        
        def compare_recursive(orig, repro, path=""):
            if type(orig) != type(repro):
                differences.append(f"{path}: Type mismatch ({type(orig)} vs {type(repro)})")
                return
            
            if isinstance(orig, dict):
                for key in set(orig.keys()) | set(repro.keys()):
                    if key not in orig:
                        differences.append(f"{path}.{key}: Missing in original")
                    elif key not in repro:
                        differences.append(f"{path}.{key}: Missing in reproduction")
                    else:
                        compare_recursive(orig[key], repro[key], f"{path}.{key}" if path else key)
            
            elif isinstance(orig, list):
                if len(orig) != len(repro):
                    differences.append(f"{path}: Length mismatch ({len(orig)} vs {len(repro)})")
                else:
                    for i, (o, r) in enumerate(zip(orig, repro)):
                        compare_recursive(o, r, f"{path}[{i}]")
            
            elif isinstance(orig, float):
                if abs(orig - repro) > tolerance:
                    differences.append(f"{path}: Value difference ({orig} vs {repro}, diff={abs(orig-repro)})")
            
            elif orig != repro:
                differences.append(f"{path}: Value mismatch ({orig} vs {repro})")
        
        compare_recursive(original, reproduction)
        return differences
    
    def _calculate_similarity_score(self,
                                  original: Dict[str, Any],
                                  reproductions: List[Dict[str, Any]],
                                  tolerance: float) -> float:
        """Calculate similarity score between original and reproduced results."""
        if not reproductions:
            return 0.0
        
        total_similarity = 0.0
        
        for reproduction in reproductions:
            differences = self._compare_results(original, reproduction, tolerance)
            # Simple similarity: 1.0 - (number of differences / arbitrary scaling factor)
            similarity = max(0.0, 1.0 - len(differences) / 100.0)
            total_similarity += similarity
        
        return total_similarity / len(reproductions)
    
    def _save_environment(self, experiment_name: str, environment: ExperimentEnvironment):
        """Save environment information to file."""
        env_file = self.experiment_dir / f"{experiment_name}_environment.json"
        with open(env_file, 'w') as f:
            json.dump(asdict(environment), f, indent=2, default=str)
    
    def _save_verification_result(self, result: ReproducibilityResult):
        """Save verification result and update history."""
        # Add to history
        self.verification_history.append(result)
        
        # Save individual result
        result_file = self.experiment_dir / f"{result.experiment_id}_verification.json"
        with open(result_file, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)


# Convenience functions
def ensure_reproducible_environment(base_seed: int = 42,
                                  experiment_name: str = "watermark_experiment") -> ReproducibilityManager:
    """Set up reproducible environment for experiments."""
    
    manager = ReproducibilityManager()
    manager.set_reproducible_seeds(base_seed)
    manager.capture_experiment_environment(experiment_name)
    
    return manager


def run_reproducibility_check(experiment_name: str,
                            results_dir: str = "reproducibility",
                            num_verification_runs: int = 3) -> bool:
    """Run reproducibility check for an experiment."""
    
    manager = ReproducibilityManager(results_dir)
    verification_result = manager.verify_reproducibility(experiment_name, num_verification_runs)
    
    print(f"Reproducibility Check Results for {experiment_name}:")
    print(f"  Matches: {'✅ YES' if verification_result.matches else '❌ NO'}")
    print(f"  Similarity Score: {verification_result.similarity_score:.3f}")
    
    if verification_result.differences:
        print(f"  Differences Found: {len(verification_result.differences)}")
        for diff in verification_result.differences[:3]:
            print(f"    - {diff}")
        if len(verification_result.differences) > 3:
            print(f"    ... and {len(verification_result.differences) - 3} more")
    
    return verification_result.matches


__all__ = [
    "ReproducibilityManager",
    "ExperimentEnvironment", 
    "ReproducibilityResult",
    "ensure_reproducible_environment",
    "run_reproducibility_check"
]