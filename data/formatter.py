"""Data formatter for converting between formats."""

from typing import Dict, List, Any, Optional
import json
import csv
from pathlib import Path


class DataFormatter:
    """Format datasets for different benchmark requirements."""

    # Standard field mappings for common benchmarks
    STANDARD_FIELDS = {
        "question": ["question", "prompt", "input", "text"],
        "answer": ["answer", "label", "target", "output"],
        "choices": ["choices", "options", "alternatives"],
    }

    def __init__(self):
        """Initialize data formatter."""
        pass

    def to_standard_format(
        self,
        data: List[Dict[str, Any]],
        field_mapping: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Convert dataset to standard format.

        Args:
            data: Input dataset
            field_mapping: Optional custom field mapping

        Returns:
            Standardized dataset
        """
        if field_mapping is None:
            field_mapping = self._detect_field_mapping(data)

        standardized = []
        for sample in data:
            new_sample = {}
            for standard_field, source_fields in self.STANDARD_FIELDS.items():
                # Try mapped field first
                if standard_field in field_mapping:
                    source = field_mapping[standard_field]
                    if source in sample:
                        new_sample[standard_field] = sample[source]
                else:
                    # Try standard field names
                    for source in source_fields:
                        if source in sample:
                            new_sample[standard_field] = sample[source]
                            break

            # Copy remaining fields
            for key, value in sample.items():
                if key not in field_mapping.values() and key not in self.STANDARD_FIELDS:
                    new_sample[key] = value

            standardized.append(new_sample)

        return standardized

    def _detect_field_mapping(self, data: List[Dict[str, Any]]) -> Dict[str, str]:
        """Auto-detect field mapping from sample data.

        Args:
            data: Input dataset

        Returns:
            Detected field mapping
        """
        if not data:
            return {}

        sample = data[0]
        mapping = {}

        for standard_field, source_fields in self.STANDARD_FIELDS.items():
            for source in source_fields:
                if source in sample:
                    mapping[standard_field] = source
                    break

        return mapping

    def to_jsonl(self, data: List[Dict[str, Any]], output_path: str) -> None:
        """Save dataset as JSONL format.

        Args:
            data: Input dataset
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def from_jsonl(self, path: str) -> List[Dict[str, Any]]:
        """Load dataset from JSONL format.

        Args:
            path: Input file path

        Returns:
            Loaded dataset
        """
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))

        return data

    def to_csv(self, data: List[Dict[str, Any]], output_path: str) -> None:
        """Save dataset as CSV format.

        Args:
            data: Input dataset
            output_path: Output file path
        """
        if not data:
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get all unique fields
        fieldnames = list(data[0].keys())

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    def from_csv(self, path: str) -> List[Dict[str, Any]]:
        """Load dataset from CSV format.

        Args:
            path: Input file path

        Returns:
            Loaded dataset
        """
        data = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))

        return data

    def format_for_benchmark(
        self,
        data: List[Dict[str, Any]],
        benchmark_name: str,
    ) -> List[Dict[str, Any]]:
        """Format dataset for a specific benchmark.

        Args:
            data: Input dataset
            benchmark_name: Target benchmark name

        Returns:
            Formatted dataset
        """
        if benchmark_name == "mmlu":
            return self._format_mmlu(data)
        elif benchmark_name == "hellaswag":
            return self._format_hellaswag(data)
        elif benchmark_name == "mmmu":
            return self._format_mmmu(data)
        else:
            return data  # Return as-is for unknown benchmarks

    def _format_mmlu(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format data for MMLU benchmark."""
        formatted = []
        for sample in data:
            new_sample = {
                "question": sample.get("question", ""),
                "choices": sample.get("choices", []),
                "answer": sample.get("answer", ""),
                "subject": sample.get("subject", "general"),
            }
            formatted.append(new_sample)
        return formatted

    def _format_hellaswag(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format data for HellaSwag benchmark."""
        formatted = []
        for sample in data:
            new_sample = {
                "context": sample.get("context", ""),
                "endings": sample.get("endings", []),
                "answer": sample.get("answer", 0),
            }
            formatted.append(new_sample)
        return formatted

    def _format_mmmu(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format data for MMMU benchmark."""
        formatted = []
        for sample in data:
            new_sample = {
                "question": sample.get("question", ""),
                "image_description": sample.get("image_description", ""),
                "choices": sample.get("choices", []),
                "answer": sample.get("answer", ""),
                "subject": sample.get("subject", "general"),
            }
            formatted.append(new_sample)
        return formatted
