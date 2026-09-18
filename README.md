# LLMEval-Lab

**Lightweight Multimodal Evaluation Laboratory**

A comprehensive, production-ready evaluation framework for language models and multimodal models. LLMEval-Lab provides unified interfaces for benchmarking, checkpoint tracking, and automated evaluation workflows.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)

## Features

- **Unified Evaluation Interface** - Evaluate language and multimodal models with a single API
- **Multiple Backend Support** - HuggingFace Transformers, vLLM, OpenAI API, Anthropic Claude
- **Benchmark Library** - Pre-built support for MMLU, HellaSwag, TruthfulQA, GSM8K, MMMU, and more
- **Checkpoint Tracking** - Monitor model capability evolution across training checkpoints
- **Automated Workflows** - Schedule and monitor evaluation tasks with retry logic
- **Interactive Dashboards** - Visualize results with radar charts, capability comparisons, and trend analysis
- **Extensible Design** - Easy to add new benchmarks and model adapters

## Installation

```bash
# Basic installation
pip install llmeval-lab

# With all dependencies
pip install llmeval-lab[all]

# With vLLM support
pip install llmeval-lab[vllm]

# With API support
pip install llmeval-lab[api]
```

## Quick Start

```python
from llmeval_lab import LLMEvaluator

# Initialize evaluator
evaluator = LLMEvaluator(
    model="gpt-4",
    backend="api"
)

# Add benchmarks
evaluator.add_benchmark("mmlu")
evaluator.add_benchmark("hellaswag")

# Run evaluation
results = evaluator.run(
    benchmarks=["mmlu", "hellaswag"],
    num_samples=10
)

# Print summary
evaluator.print_summary(results)
```

## Command Line Interface

```bash
# Run evaluation
llmeval eval --model gpt-4 --backend api --benchmarks mmlu hellaswag

# Track checkpoints
llmeval track --model llama-2 --checkpoint ./checkpoints/step1000 --metrics mmlu=0.65

# List benchmarks
llmeval list

# Analyze results
llmeval analyze --input results.json --model gpt-4 --output report.html
```

## Supported Models

| Backend | Models | Installation |
|---------|--------|--------------|
| HuggingFace | GPT-2, LLaMA, Mistral, Qwen, etc. | `pip install transformers torch` |
| vLLM | All HuggingFace models with vLLM optimization | `pip install vllm` |
| OpenAI | GPT-4, GPT-3.5-turbo | `pip install openai` |
| Anthropic | Claude 3, Claude 2 | `pip install anthropic` |

## Supported Benchmarks

### Language Model Benchmarks
- **MMLU** - Massive Multitask Language Understanding
- **HellaSwag** - Commonsense inference
- **TruthfulQA** - Truthfulness evaluation
- **GSM8K** - Mathematical reasoning

### Multimodal Benchmarks
- **MMMU** - Massive Multimodal Understanding
- **SEED-Bench** - Multimodal comprehension

## Architecture

```
LLMEval-Lab/
├── core/              # Core evaluation engine
├── adapters/         # Model backends
├── benchmarks/       # Benchmark implementations
├── data/            # Data processing
├── workflow/        # Task scheduling
├── visualization/   # Charts and dashboards
└── cli.py           # Command-line interface
```

## Checkpoint Tracking

Monitor your model's capabilities throughout training:

```python
from llmeval_lab import CheckpointTracker

tracker = CheckpointTracker()

# Record checkpoint evaluation
tracker.record(
    model_name="llama-2-7b",
    checkpoint_path="./checkpoints/step_5000",
    metrics={"mmlu": 0.72, "hellaswag": 0.78},
    step=5000
)

# Generate capability report
tracker.print_capability_report("llama-2-7b")
```

## Automated Workflows

Schedule evaluation tasks with cron-like expressions:

```python
from llmeval_lab import TaskScheduler

scheduler = TaskScheduler(max_concurrent=4)

scheduler.add_task(
    task_id="nightly_eval",
    name="Nightly Evaluation",
    func=run_evaluation,
)

scheduler.schedule_task("nightly_eval", "0 9 * * *")  # Daily at 9 AM
```

## Visualization

Generate interactive HTML reports:

```python
from llmeval_lab import Dashboard

dashboard = Dashboard()
dashboard.create_model_comparison_dashboard(
    results={"model_a": {...}, "model_b": {...}},
    output_path="comparison.html"
)
```

## Development

```bash
# Clone repository
git clone https://github.com/ai-never/LLMEval-Lab.git
cd LLMEval-Lab

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black .
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{llmeval_lab,
  title = {LLMEval-Lab: Lightweight Multimodal Evaluation Laboratory},
  author = {LLMEval Team},
  year = {2024},
  url = {https://github.com/ai-never/LLMEval-Lab}
}
```
