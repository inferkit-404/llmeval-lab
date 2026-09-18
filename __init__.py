"""LLMEval-Lab - Lightweight Multimodal Evaluation Laboratory.

A comprehensive evaluation framework for language models and multimodal models.
"""

__version__ = "0.1.0"
__author__ = "LLMEval Team"

# Core components
from .core import (
    LLMEvaluator,
    EvalResult,
    CheckpointTracker,
    CheckpointRecord,
    ResultAnalyzer,
    AnalysisReport,
)

# Configuration
from .config import (
    LLMEvalConfig,
    ModelConfig,
    BenchmarkConfig,
    WorkflowConfig,
    CheckpointTrackerConfig,
    VisualizationConfig,
)

# Adapters
from .adapters import (
    BaseAdapter,
    HuggingFaceAdapter,
    VLLMAdapter,
    APIAdapter,
    get_adapter,
)

# Benchmarks
from .benchmarks import (
    BenchmarkRegistry,
    BaseBenchmark,
    BenchmarkMetadata,
)

# Data processing
from .data import (
    DataProcessor,
    DataCleaner,
    DataFormatter,
)

# Workflow
from .workflow import (
    TaskScheduler,
    ExecutionMonitor,
    ReportGenerator,
)

# Visualization
from .visualization import (
    Dashboard,
    ChartGenerator,
)


__all__ = [
    # Version
    "__version__",
    # Core
    "LLMEvaluator",
    "EvalResult",
    "CheckpointTracker",
    "CheckpointRecord",
    "ResultAnalyzer",
    "AnalysisReport",
    # Config
    "LLMEvalConfig",
    "ModelConfig",
    "BenchmarkConfig",
    "WorkflowConfig",
    "CheckpointTrackerConfig",
    "VisualizationConfig",
    # Adapters
    "BaseAdapter",
    "HuggingFaceAdapter",
    "VLLMAdapter",
    "APIAdapter",
    "get_adapter",
    # Benchmarks
    "BenchmarkRegistry",
    "BaseBenchmark",
    "BenchmarkMetadata",
    # Data
    "DataProcessor",
    "DataCleaner",
    "DataFormatter",
    # Workflow
    "TaskScheduler",
    "ExecutionMonitor",
    "ReportGenerator",
    # Visualization
    "Dashboard",
    "ChartGenerator",
]
