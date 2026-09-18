"""MMMU (Massive Multimodal Understanding) benchmark."""

from typing import Dict, List, Any
import numpy as np

from ..base import BaseBenchmark, BenchmarkMetadata


class MMMU(BaseBenchmark):
    """MMMU benchmark for multimodal understanding."""

    def _load_metadata(self) -> BenchmarkMetadata:
        """Load MMMU metadata."""
        return BenchmarkMetadata(
            name="mmmu",
            description="Massive Multimodal Understanding - A large-scale multimodal benchmark",
            task_type="multimodal",
            citation="@article{yue2023mmmu, ...}",
            homepage="https://mmmu-benchmark.github.io/",
        )

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load MMMU dataset.

        Returns:
            List of multimodal question dictionaries
        """
        # Sample MMMU questions (image + text)
        return [
            {
                "question": "What is shown in this image?",
                "image_description": "a red apple on a wooden table",
                "choices": ["A red apple", "A green pear", "A yellow lemon", "An orange"],
                "answer": "A red apple",
                "subject": "general",
            },
            {
                "question": "Based on the diagram, what is the output?",
                "image_description": "a simple flowchart with input A, process B, output C",
                "choices": ["A leads to B leads to C", "A leads to C directly", "B leads to A", "None of the above"],
                "answer": "A leads to B leads to C",
                "subject": "science",
            },
            {
                "question": "What does this chart show?",
                "image_description": "a bar chart showing sales over 4 quarters",
                "choices": ["Quarterly sales trend", "Annual revenue", "Monthly expenses", "Weekly growth"],
                "answer": "Quarterly sales trend",
                "subject": "business",
            },
        ]

    def get_prompt_template(self) -> str:
        """Get MMMU prompt template."""
        return (
            "Given an image (described as: '{image_description}'), "
            "answer the following multiple choice question.\n\n"
            "Question: {question}\n"
            "Options: {choices}\n\n"
            "The correct answer is:"
        )

    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute accuracy metrics.

        Args:
            predictions: Model predictions
            references: Reference answers

        Returns:
            Dictionary of metrics
        """
        correct = 0
        total = len(predictions)

        for pred, ref in zip(predictions, references):
            pred_clean = pred.strip().lower()
            ref_clean = ref.strip().lower()

            # Check if prediction contains or matches reference
            if ref_clean in pred_clean or pred_clean in ref_clean:
                correct += 1
            # Check first few words match
            elif (pred_clean.split()[:2] == ref_clean.split()[:2] if pred_clean.split() and ref_clean.split() else False):
                correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": {
                "value": accuracy,
                "std": np.std([1 if ref.strip().lower() in pred.strip().lower() or
                              pred.strip().lower() in ref.strip().lower() else 0
                              for pred, ref in zip(predictions, references)]) if total > 0 else 0.0,
            }
        }
