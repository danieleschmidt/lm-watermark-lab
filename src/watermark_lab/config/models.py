"""Configuration data models."""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class WatermarkMethod(str, Enum):
    """Supported watermarking methods."""
    KIRCHENBAUER = "kirchenbauer"
    MARKLLM = "markllm"
    AARONSON = "aaronson"
    ZHAO = "zhao"


class AttackType(str, Enum):
    """Supported attack types."""
    PARAPHRASE = "paraphrase"
    TRUNCATION = "truncation"
    INSERTION = "insertion"
    SUBSTITUTION = "substitution"
    TRANSLATION = "translation"
    COMBINED = "combined"


class AttackStrength(str, Enum):
    """Attack strength levels."""
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


class WatermarkConfig(BaseModel):
    """Watermark configuration model."""
    
    method: WatermarkMethod = Field(..., description="Watermarking method")
    max_length: int = Field(default=100, ge=1, le=4096, description="Maximum generation length")
    seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")
    
    # Kirchenbauer-specific parameters
    gamma: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Green list ratio")
    delta: Optional[float] = Field(default=None, ge=0.0, description="Bias strength")
    
    # MarkLLM-specific parameters
    algorithm: Optional[str] = Field(default=None, description="MarkLLM algorithm (KGW, SWEET)")
    watermark_strength: Optional[float] = Field(default=None, ge=0.0, description="Watermark strength")
    key: Optional[str] = Field(default=None, description="Secret key for key-based methods")
    
    # Aaronson-specific parameters
    secret_key: Optional[str] = Field(default=None, description="Cryptographic secret key")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Detection threshold")
    
    # Zhao-specific parameters
    message_bits: Optional[str] = Field(default=None, description="Message bits to embed")
    redundancy: Optional[int] = Field(default=None, ge=1, description="Redundancy factor")
    
    # Additional parameters
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="Additional method-specific parameters")
    
    @validator("gamma")
    def validate_gamma(cls, v, values):
        """Validate gamma parameter for Kirchenbauer method."""
        if values.get("method") == WatermarkMethod.KIRCHENBAUER and v is None:
            return 0.25  # Default value
        return v
    
    @validator("delta")
    def validate_delta(cls, v, values):
        """Validate delta parameter for Kirchenbauer method."""
        if values.get("method") == WatermarkMethod.KIRCHENBAUER and v is None:
            return 2.0  # Default value
        return v
    
    @validator("algorithm")
    def validate_algorithm(cls, v, values):
        """Validate algorithm parameter for MarkLLM method."""
        if values.get("method") == WatermarkMethod.MARKLLM:
            valid_algorithms = ["KGW", "SWEET", "XSIR"]
            if v and v not in valid_algorithms:
                raise ValueError(f"Algorithm must be one of: {valid_algorithms}")
            return v or "KGW"  # Default value
        return v
    
    @validator("message_bits")
    def validate_message_bits(cls, v, values):
        """Validate message bits for Zhao method."""
        if values.get("method") == WatermarkMethod.ZHAO:
            if v is None:
                return "101010"  # Default value
            if not all(bit in "01" for bit in v):
                raise ValueError("Message bits must be a binary string")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.dict().items() if v is not None}


class DetectionConfig(BaseModel):
    """Detection configuration model."""
    
    method: WatermarkMethod = Field(..., description="Detection method (should match watermark method)")
    threshold: float = Field(default=0.05, ge=0.0, le=1.0, description="Statistical significance threshold")
    confidence_level: float = Field(default=0.95, ge=0.0, le=1.0, description="Confidence level")
    min_tokens: int = Field(default=10, ge=1, description="Minimum tokens required for detection")
    
    # Method-specific parameters (inherited from watermark config)
    gamma: Optional[float] = Field(default=None, description="Green list ratio (Kirchenbauer)")
    delta: Optional[float] = Field(default=None, description="Bias strength (Kirchenbauer)")
    secret_key: Optional[str] = Field(default=None, description="Secret key")
    key: Optional[str] = Field(default=None, description="Key for key-based methods")
    message_bits: Optional[str] = Field(default=None, description="Expected message bits (Zhao)")
    redundancy: Optional[int] = Field(default=None, description="Redundancy factor (Zhao)")
    
    # Detection-specific parameters
    test_type: str = Field(default="z_test", description="Statistical test type")
    batch_size: int = Field(default=32, ge=1, description="Batch size for batch detection")
    
    @classmethod
    def from_watermark_config(cls, watermark_config: WatermarkConfig, **overrides) -> "DetectionConfig":
        """Create detection config from watermark config."""
        config_dict = {
            "method": watermark_config.method,
            "gamma": watermark_config.gamma,
            "delta": watermark_config.delta,
            "secret_key": watermark_config.secret_key,
            "key": watermark_config.key,
            "message_bits": watermark_config.message_bits,
            "redundancy": watermark_config.redundancy,
        }
        
        # Add overrides
        config_dict.update(overrides)
        
        # Remove None values
        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        
        return cls(**config_dict)


class AttackConfig(BaseModel):
    """Attack configuration model."""
    
    attack_type: AttackType = Field(..., description="Type of attack")
    strength: AttackStrength = Field(default=AttackStrength.MEDIUM, description="Attack strength")
    
    # Paraphrase attack parameters
    paraphrase_method: Optional[str] = Field(default="synonym", description="Paraphrasing method")
    replacement_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Word replacement probability")
    
    # Truncation attack parameters
    truncation_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Fraction of text to remove")
    truncation_position: Optional[str] = Field(default="random", description="Where to truncate (beginning/end/middle/random)")
    
    # Insertion attack parameters
    insertion_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Noise word insertion probability")
    noise_words: Optional[List[str]] = Field(default=None, description="Custom noise words")
    
    # Substitution attack parameters
    substitution_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Character substitution probability")
    
    # Combined attack parameters
    attack_sequence: Optional[List[AttackType]] = Field(default=None, description="Sequence of attacks for combined attack")
    
    @validator("replacement_probability")
    def set_replacement_probability(cls, v, values):
        """Set default replacement probability based on strength."""
        if v is None and values.get("attack_type") == AttackType.PARAPHRASE:
            strength_map = {
                AttackStrength.LIGHT: 0.1,
                AttackStrength.MEDIUM: 0.3,
                AttackStrength.HEAVY: 0.5
            }
            return strength_map.get(values.get("strength"), 0.3)
        return v
    
    @validator("truncation_ratio")
    def set_truncation_ratio(cls, v, values):
        """Set default truncation ratio based on strength."""
        if v is None and values.get("attack_type") == AttackType.TRUNCATION:
            strength_map = {
                AttackStrength.LIGHT: 0.1,
                AttackStrength.MEDIUM: 0.3,
                AttackStrength.HEAVY: 0.5
            }
            return strength_map.get(values.get("strength"), 0.3)
        return v


class BenchmarkConfig(BaseModel):
    """Benchmark configuration model."""
    
    methods: List[WatermarkMethod] = Field(..., min_items=1, description="Methods to benchmark")
    num_samples: int = Field(default=100, ge=1, le=10000, description="Number of test samples")
    max_workers: int = Field(default=4, ge=1, le=16, description="Maximum parallel workers")
    
    # Test configuration
    test_prompts: Optional[List[str]] = Field(default=None, description="Custom test prompts")
    prompt_source: str = Field(default="builtin", description="Source of test prompts")
    max_prompt_length: int = Field(default=50, ge=1, description="Maximum prompt length")
    
    # Evaluation metrics
    quality_metrics: List[str] = Field(
        default=["perplexity", "bleu", "semantic_similarity"], 
        description="Quality metrics to evaluate"
    )
    detectability_metrics: List[str] = Field(
        default=["precision", "recall", "f1_score"], 
        description="Detectability metrics to evaluate"
    )
    robustness_attacks: List[AttackType] = Field(
        default=[AttackType.PARAPHRASE, AttackType.TRUNCATION], 
        description="Attacks to test robustness against"
    )
    
    # Output configuration
    save_results: bool = Field(default=True, description="Save benchmark results")
    output_format: str = Field(default="json", description="Output format (json/yaml/csv)")
    output_file: Optional[str] = Field(default=None, description="Output file path")
    generate_plots: bool = Field(default=True, description="Generate visualization plots")
    
    # Performance configuration
    timeout_per_sample: int = Field(default=300, ge=1, description="Timeout per sample in seconds")
    memory_limit_mb: Optional[int] = Field(default=None, ge=100, description="Memory limit in MB")
    
    @validator("methods", each_item=True)
    def validate_methods(cls, v):
        """Validate watermark methods."""
        if isinstance(v, str):
            return WatermarkMethod(v)
        return v
    
    @validator("quality_metrics", each_item=True)
    def validate_quality_metrics(cls, v):
        """Validate quality metrics."""
        valid_metrics = [
            "perplexity", "bleu", "rouge", "bertscore", 
            "semantic_similarity", "diversity", "fluency", "coherence"
        ]
        if v not in valid_metrics:
            raise ValueError(f"Quality metric must be one of: {valid_metrics}")
        return v
    
    @validator("detectability_metrics", each_item=True)
    def validate_detectability_metrics(cls, v):
        """Validate detectability metrics."""
        valid_metrics = [
            "precision", "recall", "f1_score", "accuracy", 
            "tpr", "fpr", "auc_roc", "auc_pr"
        ]
        if v not in valid_metrics:
            raise ValueError(f"Detectability metric must be one of: {valid_metrics}")
        return v


class ExperimentConfig(BaseModel):
    """Experiment configuration for research and evaluation."""
    
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(default=None, description="Experiment description")
    
    # Watermark configurations to test
    watermark_configs: List[WatermarkConfig] = Field(..., min_items=1, description="Watermark configurations")
    
    # Detection configurations
    detection_configs: Optional[List[DetectionConfig]] = Field(default=None, description="Detection configurations")
    
    # Attack configurations for robustness testing
    attack_configs: Optional[List[AttackConfig]] = Field(default=None, description="Attack configurations")
    
    # Benchmark configuration
    benchmark_config: BenchmarkConfig = Field(..., description="Benchmark configuration")
    
    # Experimental parameters
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    repeat_runs: int = Field(default=1, ge=1, le=10, description="Number of repeated runs")
    
    # Output and logging
    output_dir: str = Field(default="./experiments", description="Output directory")
    log_level: str = Field(default="INFO", description="Logging level")
    save_intermediate: bool = Field(default=False, description="Save intermediate results")
    
    # Resource limits
    max_runtime_hours: Optional[float] = Field(default=None, ge=0.1, description="Maximum runtime in hours")
    max_memory_gb: Optional[float] = Field(default=None, ge=0.1, description="Maximum memory usage in GB")
    
    def get_experiment_id(self) -> str:
        """Get unique experiment ID."""
        import hashlib
        import json
        
        # Create hash from configuration
        config_str = json.dumps(self.dict(), sort_keys=True)
        hash_obj = hashlib.md5(config_str.encode())
        return f"{self.name}_{hash_obj.hexdigest()[:8]}"