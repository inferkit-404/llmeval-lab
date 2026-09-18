"""GSM8K (Grade School Math 8K) benchmark."""

from typing import Dict, List, Any
import re
import numpy as np

from ..base import BaseBenchmark, BenchmarkMetadata


class GSM8K(BaseBenchmark):
    """GSM8K benchmark for mathematical reasoning."""

    def _load_metadata(self) -> BenchmarkMetadata:
        """Load GSM8K metadata."""
        return BenchmarkMetadata(
            name="gsm8k",
            description="Grade School Math 8K - Mathematical reasoning benchmark",
            task_type="lm",
            citation="@article{cobbe2021gsm8k, ...}",
            homepage="https://github.com/openai/grade-school-math",
        )

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load GSM8K dataset.

        Returns:
            List of math problems with solutions
        """
        # Sample GSM8K problems
        return [
            {
                "question": "There are 5 apples. I eat 2 apples. How many apples are left?",
                "answer": "3",
                "solution": "I started with 5 apples and ate 2, so 5 - 2 = 3 apples remain.",
            },
            {
                "question": "Tom has 12 cookies. He gives 4 cookies to his friend. How many cookies does Tom have now?",
                "answer": "8",
                "solution": "Tom had 12 cookies and gave away 4, so 12 - 4 = 8 cookies remain.",
            },
            {
                "question": "A store has 15 shirts. They sell 7 shirts. How many shirts are left?",
                "answer": "8",
                "solution": "The store had 15 shirts and sold 7, so 15 - 7 = 8 shirts remain.",
            },
            {
                "question": "There are 3 bags with 4 oranges in each bag. How many oranges are there in total?",
                "answer": "12",
                "solution": "3 bags × 4 oranges per bag = 12 oranges total.",
            },
        ]

    def get_prompt_template(self) -> str:
        """Get GSM8K prompt template."""
        return (
            "Solve the following math problem step by step.\n\n"
            "Problem: {question}\n\n"
            "Provide your final answer (just the number):"
        )

    def _extract_number(self, text: str) -> str:
        """Extract the final number answer from text."""
        # Look for patterns like "the answer is X" or just a number at the end
        text = text.strip()

        # Try to find number after "answer is" or "="
        patterns = [
            r"(?:the )?answer is:?\s*(\d+)",
            r"=\s*(\d+)",
            r"(\d+)\s*$",
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1)

        # Last resort: get last number in text
        numbers = re.findall(r"\d+", text)
        return numbers[-1] if numbers else ""

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
            pred_num = self._extract_number(pred)
            ref_num = self._extract_number(ref)

            if pred_num and ref_num and pred_num == ref_num:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": {
                "value": accuracy,
                "std": np.std([1 if self._extract_number(p) == self._extract_number(r)
                              else 0 for p, r in zip(predictions, references)]) if total > 0 else 0.0,
            }
        }
