"""MMLU (Massive Multitask Language Understanding) benchmark."""

from typing import Dict, List, Any
import numpy as np

from ..base import BaseBenchmark, BenchmarkMetadata


class MMLU(BaseBenchmark):
    """MMLU benchmark implementation."""

    def _load_metadata(self) -> BenchmarkMetadata:
        """Load MMLU metadata."""
        return BenchmarkMetadata(
            name="mmlu",
            description="Massive Multitask Language Understanding",
            task_type="lm",
            citation="@article{huang2019mnli, ...}",
            homepage="https://github.com/hendrycks/test",
        )

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load MMLU dataset.

        Returns:
            List of question dictionaries
        """
        # Sample MMLU questions for demonstration
        # In production, this would load from actual dataset
        return [
            {
                "question": "What is the capital of France?",
                "choices": ["London", "Paris", "Berlin", "Madrid"],
                "answer": "Paris",
                "subject": "geography",
            },
            {
                "question": "Which of the following is a noble gas?",
                "choices": ["Nitrogen", "Oxygen", "Helium", "Carbon"],
                "answer": "Helium",
                "subject": "chemistry",
            },
            {
                "question": "What is the derivative of x^2?",
                "choices": ["x", "2x", "x^2", "2"],
                "answer": "2x",
                "subject": "mathematics",
            },
            {
                "question": "Who wrote 'Romeo and Juliet'?",
                "choices": ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"],
                "answer": "William Shakespeare",
                "subject": "literature",
            },
            {
                "question": "What is the largest planet in our solar system?",
                "choices": ["Earth", "Mars", "Jupiter", "Saturn"],
                "answer": "Jupiter",
                "subject": "astronomy",
            },
        ]

    def get_prompt_template(self) -> str:
        """Get MMLU prompt template."""
        return (
            "The following is a multiple choice question. "
            "Answer the question by selecting one of the options.\n\n"
            "Question: {question}\n"
            "Options: {choices}\n"
            "Answer: Let me think carefully about this. The correct answer is"
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
            # Simple matching - in production would use more sophisticated matching
            pred_clean = pred.strip().lower()
            ref_clean = ref.strip().lower()

            if ref_clean in pred_clean or pred_clean in ref_clean:
                correct += 1
            # Check first word match
            elif pred_clean.split()[0] if pred_clean else "" == ref_clean.split()[0] if ref_clean else "":
                correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": {
                "value": accuracy,
                "std": np.std([1 if p.strip().lower() == r.strip().lower() else 0
                              for p, r in zip(predictions, references)]) if total > 0 else 0.0,
            }
        }
