"""Data processor for benchmark datasets."""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json


class DataProcessor:
    """Process and transform benchmark datasets."""

    def __init__(self, cache_dir: str = "./cache"):
        """Initialize data processor.

        Args:
            cache_dir: Directory for caching processed data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_json(self, path: str) -> List[Dict[str, Any]]:
        """Load dataset from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            List of data samples
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_json(self, data: List[Dict[str, Any]], path: str) -> None:
        """Save dataset to JSON file.

        Args:
            data: Dataset to save
            path: Output path
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def filter_samples(
        self,
        data: List[Dict[str, Any]],
        filter_fn,
    ) -> List[Dict[str, Any]]:
        """Filter samples based on a function.

        Args:
            data: Input dataset
            filter_fn: Function that takes a sample and returns True/False

        Returns:
            Filtered dataset
        """
        return [sample for sample in data if filter_fn(sample)]

    def transform_samples(
        self,
        data: List[Dict[str, Any]],
        transform_fn,
    ) -> List[Dict[str, Any]]:
        """Transform samples using a function.

        Args:
            data: Input dataset
            transform_fn: Function that transforms a sample

        Returns:
            Transformed dataset
        """
        return [transform_fn(sample) for sample in data]

    def split_dataset(
        self,
        data: List[Dict[str, Any]],
        ratios: tuple = (0.8, 0.1, 0.1),
    ) -> tuple:
        """Split dataset into train/val/test.

        Args:
            data: Input dataset
            ratios: Split ratios (must sum to 1.0)

        Returns:
            Tuple of (train, val, test) datasets
        """
        if len(ratios) != 3:
            raise ValueError("Split ratios must be a tuple of 3 values")

        if not abs(sum(ratios) - 1.0) < 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        total = len(data)
        train_size = int(total * ratios[0])
        val_size = int(total * ratios[1])

        train = data[:train_size]
        val = data[train_size:train_size + val_size]
        test = data[train_size + val_size:]

        return train, val, test

    def cache_key(self, benchmark: str, config: Dict[str, Any]) -> str:
        """Generate cache key for a benchmark configuration.

        Args:
            benchmark: Benchmark name
            config: Configuration dict

        Returns:
            Cache key string
        """
        import hashlib
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"{benchmark}_{config_hash}"
