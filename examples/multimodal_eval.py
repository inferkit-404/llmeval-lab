"""Multimodal evaluation example.

This example demonstrates how to evaluate multimodal models on benchmarks
like MMMU that require understanding of both images and text.
"""

from llmeval_lab import LLMEvaluator


def multimodal_evaluation_example():
    """Example of evaluating multimodal models."""
    print("=" * 60)
    print("Multimodal Evaluation Example")
    print("=" * 60)

    # Initialize evaluator with a multimodal model
    evaluator = LLMEvaluator(
        model="llava-1.5-7b",  # Example multimodal model
        backend="hf",
    )

    # Add multimodal benchmarks
    evaluator.add_benchmark("mmmu")

    # Run evaluation
    print("\nRunning multimodal evaluation on MMMU...")
    results = evaluator.run(
        benchmarks=["mmmu"],
        num_samples=10,
    )

    # Print summary
    evaluator.print_summary(results)

    # Save results
    evaluator.save_results(results, "./results/multimodal_results.json")
    print("\nResults saved to ./results/multimodal_results.json")


def mixed_evaluation_example():
    """Example evaluating both language and multimodal benchmarks."""
    print("\n" + "=" * 60)
    print("Mixed Language + Multimodal Evaluation")
    print("=" * 60)

    # Initialize evaluator
    evaluator = LLMEvaluator(
        model="gpt-4-vision",
        backend="api",
    )

    # Add both language and multimodal benchmarks
    evaluator.add_benchmark("mmlu")      # Language
    evaluator.add_benchmark("hellaswag") # Language
    evaluator.add_benchmark("mmmu")       # Multimodal

    # Run evaluation
    print("\nRunning mixed evaluation...")
    results = evaluator.run(
        benchmarks=["mmlu", "hellaswag", "mmmu"],
        num_samples=5,
    )

    evaluator.print_summary(results)


if __name__ == "__main__":
    print("This example demonstrates multimodal evaluation.\n")

    # Note: Running requires appropriate multimodal model and data
    print("To run this example:")
    print("1. Install multimodal dependencies: pip install llmeval-lab[multimodal]")
    print("2. Set up vision model (e.g., LLaVA, IDEFICS)")
    print("3. python examples/multimodal_eval.py")
