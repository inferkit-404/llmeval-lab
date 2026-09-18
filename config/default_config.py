"""Default configuration for LLMEval-Lab."""

from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "gpt-4"
    backend: str = "api"  # api, hf, vllm
    device: str = "cuda"
    tensor_parallel_size: int = 1
    max_model_len: int = 4096
    batch_size: int = 8
    num_gpus: int = 1


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    name: str
    task_type: str = "lm"  # lm, multimodal
    dataset_path: str = ""
    prompt_template: str = ""
    num_fewshot: int = 0
    metrics: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class WorkflowConfig:
    """Workflow automation configuration."""
    enabled: bool = True
    schedule: str = ""  # cron expression
    max_retries: int = 3
    retry_delay: int = 60
    timeout: int = 3600
    notification_webhook: str = ""


@dataclass
class CheckpointTrackerConfig:
    """Checkpoint tracking configuration."""
    enabled: bool = True
    storage_path: str = "./checkpoints_eval"
    save_interval: int = 1
    track_metrics: List[str] = field(default_factory=lambda: ["accuracy", "loss"])
    enable_alerts: bool = True
    alert_threshold: float = 0.05  # 5% degradation triggers alert


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    enabled: bool = True
    output_dir: str = "./eval_results"
    format: str = "html"  # html, json, csv
    include_charts: bool = True
    include_radar: bool = True


@dataclass
class LLMEvalConfig:
    """Main configuration for LLMEval-Lab."""
    model: ModelConfig = field(default_factory=ModelConfig)
    benchmarks: List[BenchmarkConfig] = field(default_factory=list)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    checkpoint_tracker: CheckpointTrackerConfig = field(default_factory=CheckpointTrackerConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # Paths
    data_dir: str = "./data"
    cache_dir: str = "./cache"
    output_dir: str = "./results"

    # Logging
    log_level: str = "INFO"
    log_file: str = "./llmeval.log"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LLMEvalConfig":
        """Create config from dictionary."""
        def dict_to_dataclass(dc_class, data):
            if data is None:
                return dc_class()
            field_values = {}
            for field_name, field_def in dc_class.__dataclass_fields__.items():
                if field_name in data:
                    field_values[field_name] = data[field_name]
                else:
                    field_values[field_name] = field_def.default
            return dc_class(**field_values)

        return cls(
            model=dict_to_dataclass(ModelConfig, config_dict.get("model", {})),
            benchmarks=[dict_to_dataclass(BenchmarkConfig, b) for b in config_dict.get("benchmarks", [])],
            workflow=dict_to_dataclass(WorkflowConfig, config_dict.get("workflow", {})),
            checkpoint_tracker=dict_to_dataclass(CheckpointTrackerConfig, config_dict.get("checkpoint_tracker", {})),
            visualization=dict_to_dataclass(VisualizationConfig, config_dict.get("visualization", {})),
            data_dir=config_dict.get("data_dir", "./data"),
            cache_dir=config_dict.get("cache_dir", "./cache"),
            output_dir=config_dict.get("output_dir", "./results"),
            log_level=config_dict.get("log_level", "INFO"),
            log_file=config_dict.get("log_file", "./llmeval.log"),
        )


# Default benchmarks registry
DEFAULT_BENCHMARKS: Dict[str, BenchmarkConfig] = {
    "mmlu": BenchmarkConfig(
        name="mmlu",
        task_type="lm",
        description="Massive Multitask Language Understanding",
        metrics=["accuracy"],
    ),
    "hellaswag": BenchmarkConfig(
        name="hellaswag",
        task_type="lm",
        description="Commonsense inference benchmark",
        metrics=["accuracy"],
    ),
    "truthfulqa": BenchmarkConfig(
        name="truthfulqa",
        task_type="lm",
        description="TruthfulQA benchmark",
        metrics=["accuracy", "truthful_score"],
    ),
    "gsm8k": BenchmarkConfig(
        name="gsm8k",
        task_type="lm",
        description="Grade School Math 8K",
        metrics=["accuracy"],
    ),
    "mmmu": BenchmarkConfig(
        name="mmmu",
        task_type="multimodal",
        description="Massive Multimodal Understanding",
        metrics=["accuracy"],
    ),
}
