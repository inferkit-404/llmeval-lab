"""HellaSwag benchmark for commonsense inference."""

from typing import Dict, List, Any
import numpy as np

from ..base import BaseBenchmark, BenchmarkMetadata


class HellaSwag(BaseBenchmark):
    """HellaSwag benchmark implementation."""

    def _load_metadata(self) -> BenchmarkMetadata:
        """Load HellaSwag metadata."""
        return BenchmarkMetadata(
            name="hellaswag",
            description="Commonsense inference benchmark",
            task_type="lm",
            citation="@article{zellers2019hellaswag, ...}",
            homepage="https://github.com Rowan/hellaswag",
        )

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load HellaSwag dataset.

        Returns:
            List of context and ending pairs
        """
        # Sample HellaSwag questions
        return [
            {
                "context": "A woman sits at a piano. She places her fingers on the keys.",
                "endings": [
                    "She begins to play a melody.",
                    "She closes the piano lid.",
                    "She takes a photograph.",
                    "She leaves the room.",
                ],
                "answer": 0,  # Correct ending index
            },
            {
                "context": "A man is outdoor. He is gardening.",
                "endings": [
                    "He plants flowers in a pot.",
                    "He goes swimming.",
                    "He reads a book.",
                    "He drives a car.",
                ],
                "answer": 0,
            },
            {
                "context": "A chef is preparing a meal. He adds ingredients to a pot.",
                "endings": [
                    "He stirs the contents.",
                    "He cleans the floor.",
                    "He opens a window.",
                    "He turns off the TV.",
                ],
                "answer": 0,
            },
        ]

    def get_prompt_template(self) -> str:
        """Get HellaSwag prompt template."""
        return (
            "Given a context, select the most appropriate ending.\n\n"
            "Context: {context}\n"
            "Endings:\n"
            "0: {ending_0}\n"
            "1: {ending_1}\n"
            "2: {ending_2}\n"
            "3: {ending_3}\n\n"
            "The most appropriate ending is number"
        )

    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute accuracy metrics.

        Args:
            predictions: Model predictions (ending indices)
            references: Reference ending indices

        Returns:
            Dictionary of metrics
        """
        correct = 0
        total = len(predictions)

        for pred, ref in zip(predictions, references):
            # Extract number from prediction
            pred_str = str(pred).strip().lower()
            ref_str = str(ref).strip()

            # Check if prediction contains the correct number
            if ref_str in pred_str:
                correct += 1
            # Try to extract ending index
            elif pred_str.startswith(tuple(str(i) for i in range(10))):
                first_char = pred_str[0]
                if first_char == ref_str:
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": {
                "value": accuracy,
                "std": np.std([1 if str(ref) in str(pred) else 0
                              for pred, ref in zip(predictions, references)]) if total > 0 else 0.0,
            }
        }
