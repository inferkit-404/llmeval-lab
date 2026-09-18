"""Benchmark registry for managing available benchmarks."""

from typing import Dict, Optional, Type, List
from dataclasses import dataclass

from .base import BaseBenchmark, BenchmarkMetadata


@dataclass
class BenchmarkInfo:
    """Information about a registered benchmark."""
    name: str
    benchmark_class: Type[BaseBenchmark]
    task_type: str
    description: str


class BenchmarkRegistry:
    """Registry for managing benchmark implementations."""

    def __init__(self):
        """Initialize benchmark registry."""
        self._benchmarks: Dict[str, BenchmarkInfo] = {}
        self._register_builtin_benchmarks()

    def _register_builtin_benchmarks(self):
        """Register built-in benchmarks."""
        from .lm_benchmarks import MMLU, HellaSwag, TruthfulQA, GSM8K
        from .multimodal_benchmarks import MMMU

        benchmarks = [
            ("mmlu", MMLU, "lm", "Massive Multitask Language Understanding"),
            ("hellaswag", HellaSwag, "lm", "Commonsense inference benchmark"),
            ("truthfulqa", TruthfulQA, "lm", "TruthfulQA benchmark"),
            ("gsm8k", GSM8K, "lm", "Grade School Math 8K"),
            ("mmmu", MMMU, "multimodal", "Massive Multimodal Understanding"),
        ]

        for name, bench_class, task_type, description in benchmarks:
            self.register(name, bench_class, task_type, description)

    def register(
        self,
        name: str,
        benchmark_class: Type[BaseBenchmark],
        task_type: str = "lm",
        description: str = "",
    ):
        """Register a benchmark.

        Args:
            name: Benchmark name
            benchmark_class: Benchmark class
            task_type: Type of task ("lm" or "multimodal")
            description: Brief description
        """
        self._benchmarks[name] = BenchmarkInfo(
            name=name,
            benchmark_class=benchmark_class,
            task_type=task_type,
            description=description,
        )

    def get_benchmark(self, name: str) -> Optional[BaseBenchmark]:
        """Get a benchmark instance by name.

        Args:
            name: Benchmark name

        Returns:
            Benchmark instance or None if not found
        """
        if name not in self._benchmarks:
            return None

        info = self._benchmarks[name]
        return info.benchmark_class()

    def list_benchmarks(self, task_type: Optional[str] = None) -> List[str]:
        """List available benchmark names.

        Args:
            task_type: Optional filter by task type

        Returns:
            List of benchmark names
        """
        if task_type is None:
            return list(self._benchmarks.keys())

        return [
            name for name, info in self._benchmarks.items()
            if info.task_type == task_type
        ]

    def get_info(self, name: str) -> Optional[BenchmarkInfo]:
        """Get benchmark information.

        Args:
            name: Benchmark name

        Returns:
            BenchmarkInfo or None if not found
        """
        return self._benchmarks.get(name)

    def get_all_info(self) -> List[BenchmarkInfo]:
        """Get all registered benchmark information."""
        return list(self._benchmarks.values())
