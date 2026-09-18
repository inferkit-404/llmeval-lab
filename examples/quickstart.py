"""Quickstart example for LLMEval-Lab.

This example demonstrates the basic usage of LLMEval-Lab for evaluating
language models on standard benchmarks.
"""

from llmeval_lab import LLMEvaluator


def quickstart_example():
    """Basic quickstart example."""
    print("=" * 60)
    print("LLMEval-Lab Quickstart Example")
    print("=" * 60)

    # Initialize evaluator with a HuggingFace model
    evaluator = LLMEvaluator(
        model="gpt2",  # Using GPT-2 as an example
        backend="hf",
    )

    # Add benchmarks to evaluate
    evaluator.add_benchmark("mmlu")
    evaluator.add_benchmark("hellaswag")

    # Run evaluation
    print("\nRunning evaluation on mmlu and hellaswag...")
    results = evaluator.run(
        benchmarks=["mmlu", "hellaswag"],
        num_samples=5,  # Limit samples for quick demo
    )

    # Print summary
    evaluator.print_summary(results)

    # Save results
    evaluator.save_results(results, "./results/quickstart_results.json")
    print("\nResults saved to ./results/quickstart_results.json")


def api_example():
    """Example using API-based models."""
    print("\n" + "=" * 60)
    print("API Model Example")
    print("=" * 60)

    # Initialize evaluator with an API model
    evaluator = LLMEvaluator(
        model="gpt-4",
        backend="api",
    )

    # Add benchmark
    evaluator.add_benchmark("truthfulqa")

    # Run evaluation
    print("\nRunning evaluation on truthfulqa...")
    results = evaluator.run(
        benchmarks=["truthfulqa"],
        num_samples=3,
    )

    evaluator.print_summary(results)


if __name__ == "__main__":
    print("This example demonstrates LLMEval-Lab basic usage.\n")

    # Note: This requires installing transformers and having gpt2 model available
    # For a real test, run: quickstart_example()
    # api_example()  # Requires OpenAI API key

    print("To run this example:")
    print("1. pip install llmeval-lab")
    print("2. python examples/quickstart.py")
