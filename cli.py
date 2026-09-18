#!/usr/bin/env python3
"""Command-line interface for LLMEval-Lab."""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from llmeval_lab import LLMEvaluator, CheckpointTracker, ResultAnalyzer
from llmeval_lab.config import LLMEvalConfig
from llmeval_lab.workflow import TaskScheduler
from llmeval_lab.utils import setup_logger


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLMEval-Lab: Lightweight Multimodal Evaluation Laboratory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--model", "-m", required=True, help="Model name or path")
    eval_parser.add_argument("--backend", "-b", default="hf", choices=["hf", "vllm", "api"],
                            help="Model backend")
    eval_parser.add_argument("--benchmarks", "-t", nargs="+", default=["mmlu", "hellaswag"],
                            help="Benchmarks to evaluate")
    eval_parser.add_argument("--output", "-o", default="./results/eval_results.json",
                            help="Output file path")
    eval_parser.add_argument("--num-samples", "-n", type=int, default=None,
                            help="Number of samples to evaluate")
    eval_parser.add_argument("--batch-size", "-s", type=int, default=8,
                            help="Batch size")

    # Track command
    track_parser = subparsers.add_parser("track", help="Track checkpoint")
    track_parser.add_argument("--model", "-m", required=True, help="Model name")
    track_parser.add_argument("--checkpoint", "-c", required=True, help="Checkpoint path")
    track_parser.add_argument("--metrics", "-t", nargs="+", required=True,
                             help="Metrics to record")
    track_parser.add_argument("--step", type=int, help="Training step")

    # List command
    list_parser = subparsers.add_parser("list", help="List available benchmarks")
    list_parser.add_argument("--type", choices=["lm", "multimodal", "all"],
                           default="all", help="Filter by type")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze results")
    analyze_parser.add_argument("--input", "-i", required=True, help="Input results file")
    analyze_parser.add_argument("--model", "-m", required=True, help="Model name")
    analyze_parser.add_argument("--output", "-o", default="./results/report.html",
                               help="Output report path")
    analyze_parser.add_argument("--format", "-f", choices=["html", "json"],
                               default="html", help="Report format")

    # Serve command (for dashboard)
    serve_parser = subparsers.add_parser("serve", help="Start dashboard server")
    serve_parser.add_argument("--port", "-p", type=int, default=8050, help="Port number")
    serve_parser.add_argument("--host", default="localhost", help="Host address")

    args = parser.parse_args()

    if args.command == "eval":
        run_evaluation(args)
    elif args.command == "track":
        track_checkpoint(args)
    elif args.command == "list":
        list_benchmarks(args)
    elif args.command == "analyze":
        analyze_results(args)
    elif args.command == "serve":
        serve_dashboard(args)
    else:
        parser.print_help()


def run_evaluation(args):
    """Run evaluation."""
    logger = setup_logger()
    logger.info(f"Evaluating {args.model} on {args.benchmarks}")

    evaluator = LLMEvaluator(
        model=args.model,
        backend=args.backend,
    )

    for benchmark in args.benchmarks:
        evaluator.add_benchmark(benchmark)

    results = evaluator.run(
        benchmarks=args.benchmarks,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )

    evaluator.save_results(results, args.output)
    evaluator.print_summary(results)

    logger.info(f"Results saved to {args.output}")


def track_checkpoint(args):
    """Track checkpoint evaluation."""
    logger = setup_logger()

    tracker = CheckpointTracker()

    # Parse metrics from command line
    metrics = {}
    for metric_str in args.metrics:
        if "=" in metric_str:
            key, value = metric_str.split("=", 1)
            metrics[key] = float(value)

    tracker.record(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        metrics=metrics,
        step=args.step,
    )

    tracker.print_capability_report(args.model)


def list_benchmarks(args):
    """List available benchmarks."""
    from llmeval_lab.benchmarks import BenchmarkRegistry

    registry = BenchmarkRegistry()
    benchmarks = registry.list_benchmarks()

    print("Available benchmarks:")
    print("-" * 40)

    for name in benchmarks:
        info = registry.get_info(name)
        if info:
            task_type = info.task_type.upper()
            print(f"  {name:15} [{task_type}] - {info.description}")

    print("-" * 40)
    print(f"Total: {len(benchmarks)} benchmarks")


def analyze_results(args):
    """Analyze evaluation results."""
    from llmeval_lab.utils import load_json

    logger = setup_logger()
    logger.info(f"Analyzing results from {args.input}")

    # Load results
    data = load_json(args.input)
    results = data.get("results", [])

    analyzer = ResultAnalyzer(output_dir=str(Path(args.output).parent))
    report = analyzer.analyze(results, args.model)

    if args.format == "html":
        output_path = analyzer.save_report(report, format="html")
    else:
        output_path = analyzer.save_report(report, format="json")

    logger.info(f"Report saved to {output_path}")


def serve_dashboard(args):
    """Start dashboard server."""
    print(f"Starting dashboard on {args.host}:{args.port}")
    print("Note: Run 'llmeval analyze' first to generate results, then open the HTML report directly.")


if __name__ == "__main__":
    main()
