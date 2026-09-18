"""Benchmarks module for LLMEval-Lab."""

from typing import Dict, Optional, List

from .registry import BenchmarkRegistry
from .base import BaseBenchmark, BenchmarkMetadata


__all__ = [
    "BenchmarkRegistry",
    "BaseBenchmark",
    "BenchmarkMetadata",
]

# Import built-in benchmarks
from .lm_benchmarks import MMLU, HellaSwag, TruthfulQA, GSM8K
from .multimodal_benchmarks import MMMU

# Auto-register built-in benchmarks
BUILTIN_BENCHMARKS = {
    "mmlu": MMLU,
    "hellaswag": HellaSwag,
    "truthfulqa": TruthfulQA,
    "gsm8k": GSM8K,
    "mmmu": MMMU,
}

__all__.extend(["MMLU", "HellaSwag", "TruthfulQA", "GSM8K", "MMMU"])
