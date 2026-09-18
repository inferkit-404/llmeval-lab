"""Checkpoint tracking example.

This example demonstrates how to track model capabilities across
different training checkpoints to monitor capability changes and detect degradation.
"""

from llmeval_lab import CheckpointTracker


def checkpoint_tracking_example():
    """Example of tracking model capabilities over checkpoints."""
    print("=" * 60)
    print("Checkpoint Tracking Example")
    print("=" * 60)

    # Initialize tracker
    tracker = CheckpointTracker()

    # Simulate tracking checkpoints during training
    checkpoints = [
        {
            "path": "./checkpoints/step_1000",
            "step": 1000,
            "metrics": {"mmlu": 0.65, "hellaswag": 0.72, "truthfulqa": 0.45},
        },
        {
            "path": "./checkpoints/step_2000",
            "step": 2000,
            "metrics": {"mmlu": 0.68, "hellaswag": 0.75, "truthfulqa": 0.48},
        },
        {
            "path": "./checkpoints/step_3000",
            "step": 3000,
            "metrics": {"mmlu": 0.70, "hellaswag": 0.77, "truthfulqa": 0.50},
        },
        {
            "path": "./checkpoints/step_4000",
            "step": 4000,
            "metrics": {"mmlu": 0.72, "hellaswag": 0.76, "truthfulqa": 0.52},
        },
        {
            "path": "./checkpoints/step_5000",
            "step": 5000,
            "metrics": {"mmlu": 0.71, "hellaswag": 0.74, "truthfulqa": 0.44},  # Degradation!
        },
    ]

    model_name = "Llama-2-7b"

    print(f"\nTracking {len(checkpoints)} checkpoints for {model_name}...")

    # Record each checkpoint
    for ckpt in checkpoints:
        tracker.record(
            model_name=model_name,
            checkpoint_path=ckpt["path"],
            metrics=ckpt["metrics"],
            step=ckpt["step"],
        )

    # Print capability report
    print("\n" + "=" * 60)
    print(f"Capability Report: {model_name}")
    print("=" * 60)
    tracker.print_capability_report(model_name)

    # Analyze specific metric
    print("\n" + "=" * 60)
    print("Detailed Analysis: MMLU")
    print("=" * 60)

    analysis = tracker.analyze_capability_change(model_name, "mmlu")
    print(f"Trend: {analysis['trend']}")
    print(f"Change: {analysis['change_pct']:+.1f}%")
    print(f"First: {analysis['first_value']:.4f}")
    print(f"Last: {analysis['last_value']:.4f}")

    if analysis['alerts']:
        print("\n⚠️ Alerts:")
        for alert in analysis['alerts']:
            print(f"  - Step {alert['step']}: {alert['previous_value']:.4f} → {alert['current_value']:.4f} "
                  f"(-{alert['drop_pct']:.1f}%)")


if __name__ == "__main__":
    checkpoint_tracking_example()
