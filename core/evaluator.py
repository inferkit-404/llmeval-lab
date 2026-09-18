"""Core evaluation engine for LLMEval-Lab."""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime

from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from adapters import get_adapter
from benchmarks import BenchmarkRegistry
from config import LLMEvalConfig, BenchmarkConfig


console = Console()


@dataclass
class EvalResult:
    """Single evaluation result."""
    benchmark_name: str
    metric: str
    value: float
    std: Optional[float] = None
    num_samples: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "metric": self.metric,
            "value": self.value,
            "std": self.std,
            "num_samples": self.num_samples,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class EvalRunResult:
    """Complete evaluation run result."""
    model_name: str
    checkpoint_path: Optional[str]
    results: List[EvalResult]
    total_time: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "checkpoint_path": self.checkpoint_path,
            "results": [r.to_dict() for r in self.results],
            "total_time": self.total_time,
            "timestamp": self.timestamp,
            "errors": self.errors,
        }

    def summary(self) -> Dict[str, float]:
        """Get summary of results."""
        summary = {}
        for r in self.results:
            key = f"{r.benchmark_name}_{r.metric}"
            summary[key] = r.value
        return summary


class LLMEvaluator:
    """Main evaluation engine."""

    def __init__(
        self,
        model: Union[str, Dict[str, Any]],
        backend: str = "hf",
        config: Optional[LLMEvalConfig] = None,
        **kwargs,
    ):
        """Initialize evaluator.

        Args:
            model: Model name or configuration dict
            backend: Backend type (hf, vllm, api)
            config: Optional configuration object
            **kwargs: Additional model arguments
        """
        self.config = config or LLMEvalConfig()
        self.model_name = model if isinstance(model, str) else model.get("name", "unknown")
        self.backend = backend

        # Initialize adapter
        model_config = self.config.model
        model_config.name = self.model_name
        model_config.backend = backend

        self.adapter = get_adapter(backend, model, **kwargs)
        console.print(f"[green]Initialized {backend} adapter for {self.model_name}[/green]")

        # Initialize benchmark registry
        self.benchmark_registry = BenchmarkRegistry()

    def add_benchmark(self, benchmark_name: str, **kwargs) -> "LLMEvaluator":
        """Add a benchmark to evaluate.

        Args:
            benchmark_name: Name of the benchmark
            **kwargs: Additional benchmark configuration
        """
        if benchmark_name not in self.benchmark_registry.list_benchmarks():
            console.print(f"[yellow]Warning: Unknown benchmark '{benchmark_name}'[/yellow]")
        console.print(f"[blue]Added benchmark: {benchmark_name}[/blue]")
        return self

    def run(
        self,
        benchmarks: Optional[List[str]] = None,
        checkpoint_path: Optional[str] = None,
        batch_size: Optional[int] = None,
        num_samples: Optional[int] = None,
        **kwargs,
    ) -> EvalRunResult:
        """Run evaluation on specified benchmarks.

        Args:
            benchmarks: List of benchmark names to evaluate
            checkpoint_path: Optional checkpoint path for tracking
            batch_size: Batch size for evaluation
            num_samples: Number of samples to evaluate (None for full)
            **kwargs: Additional run arguments

        Returns:
            EvalRunResult containing all results
        """
        start_time = time.time()
        errors = []

        if benchmarks is None:
            benchmarks = ["mmlu", "hellaswag"]

        results = []

        for benchmark_name in benchmarks:
            try:
                console.print(f"\n[bold cyan]Evaluating: {benchmark_name}[/bold cyan]")
                benchmark_result = self._evaluate_single(
                    benchmark_name,
                    batch_size=batch_size,
                    num_samples=num_samples,
                    **kwargs,
                )
                results.extend(benchmark_result)
            except Exception as e:
                error_msg = f"Error evaluating {benchmark_name}: {str(e)}"
                console.print(f"[red]{error_msg}[/red]")
                errors.append(error_msg)

        total_time = time.time() - start_time

        return EvalRunResult(
            model_name=self.model_name,
            checkpoint_path=checkpoint_path,
            results=results,
            total_time=total_time,
            errors=errors,
        )

    def _evaluate_single(
        self,
        benchmark_name: str,
        batch_size: Optional[int] = None,
        num_samples: Optional[int] = None,
        **kwargs,
    ) -> List[EvalResult]:
        """Evaluate on a single benchmark."""
        benchmark = self.benchmark_registry.get_benchmark(benchmark_name)
        if benchmark is None:
            raise ValueError(f"Benchmark '{benchmark_name}' not found")

        # Get dataset
        dataset = benchmark.load_dataset()

        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        # Get prompt template
        prompt_template = benchmark.get_prompt_template()

        # Run inference
        predictions = []
        references = []

        batch_size = batch_size or self.config.model.batch_size

        for i in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {benchmark_name}"):
            batch = dataset[i:i + batch_size]
            batch_preds, batch_refs = self._process_batch(batch, prompt_template)
            predictions.extend(batch_preds)
            references.extend(batch_refs)

        # Compute metrics
        metrics = benchmark.compute_metrics(predictions, references)

        # Build results
        eval_results = []
        for metric_name, metric_value in metrics.items():
            eval_results.append(EvalResult(
                benchmark_name=benchmark_name,
                metric=metric_name,
                value=metric_value["value"],
                std=metric_value.get("std"),
                num_samples=len(dataset),
                metadata=benchmark.get_metadata(),
            ))

        return eval_results

    def _process_batch(
        self,
        batch: Any,
        prompt_template: str,
    ) -> tuple[List[str], List[str]]:
        """Process a batch of samples."""
        predictions = []
        references = []

        for sample in batch:
            prompt = prompt_template.format(**sample)
            pred = self.adapter.generate(prompt)
            predictions.append(pred)
            references.append(sample.get("answer", sample.get("label", "")))

        return predictions, references

    def save_results(self, result: EvalRunResult, output_path: str) -> None:
        """Save evaluation results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        console.print(f"[green]Results saved to {output_path}[/green]")

    def print_summary(self, result: EvalRunResult) -> None:
        """Print summary table of results."""
        table = Table(title=f"Evaluation Summary: {result.model_name}")
        table.add_column("Benchmark", style="cyan")
        table.add_column("Metric", style="magenta")
        table.add_column("Value", style="green", justify="right")
        table.add_column("Samples", style="yellow", justify="right")

        for r in result.results:
            value_str = f"{r.value:.4f}" + (f" ± {r.std:.4f}" if r.std else "")
            table.add_row(r.benchmark_name, r.metric, value_str, str(r.num_samples))

        console.print(table)
        console.print(f"\n[bold]Total time:[/bold] {result.total_time:.2f}s")
