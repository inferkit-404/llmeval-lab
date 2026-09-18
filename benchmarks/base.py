"""Base benchmark interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class BenchmarkMetadata:
    """Metadata for a benchmark."""
    name: str
    description: str
    task_type: str  # "lm" or "multimodal"
    num_samples: int = 0
    citation: Optional[str] = None
    homepage: Optional[str] = None
    version: str = "1.0"


class BaseBenchmark(ABC):
    """Abstract base class for benchmarks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize benchmark.

        Args:
            config: Optional benchmark configuration
        """
        self.config = config or {}
        self._metadata = self._load_metadata()
        self._dataset = None

    @abstractmethod
    def _load_metadata(self) -> BenchmarkMetadata:
        """Load benchmark metadata."""
        pass

    @abstractmethod
    def load_dataset(self) -> Any:
        """Load the benchmark dataset.

        Returns:
            Dataset object
        """
        pass

    @abstractmethod
    def get_prompt_template(self) -> str:
        """Get the prompt template for this benchmark.

        Returns:
            Prompt template string
        """
        pass

    @abstractmethod
    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute evaluation metrics.

        Args:
            predictions: List of model predictions
            references: List of reference answers

        Returns:
            Dictionary mapping metric names to {value, std} dictionaries
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Get benchmark metadata as dictionary."""
        return {
            "name": self._metadata.name,
            "description": self._metadata.description,
            "task_type": self._metadata.task_type,
            "num_samples": self._metadata.num_samples,
            "citation": self._metadata.citation,
            "homepage": self._metadata.homepage,
            "version": self._metadata.version,
        }

    def validate_dataset(self, dataset: Any) -> bool:
        """Validate dataset format.

        Args:
            dataset: Dataset to validate

        Returns:
            True if valid
        """
        return True
