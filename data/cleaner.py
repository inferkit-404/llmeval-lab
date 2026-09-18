"""Data cleaner for quality control."""

from typing import Dict, List, Any, Callable, Optional
import re


class DataCleaner:
    """Clean and validate benchmark datasets."""

    def __init__(self):
        """Initialize data cleaner."""
        self.validation_errors: List[str] = []

    def clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and normalizing.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return str(text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def remove_duplicates(self, data: List[Dict[str, Any]], key: str = "question") -> List[Dict[str, Any]]:
        """Remove duplicate entries based on a key field.

        Args:
            data: Input dataset
            key: Field to check for duplicates

        Returns:
            Dataset with duplicates removed
        """
        seen = set()
        result = []

        for sample in data:
            value = sample.get(key, "")
            if value not in seen:
                seen.add(value)
                result.append(sample)

        return result

    def validate_required_fields(
        self,
        data: List[Dict[str, Any]],
        required_fields: List[str],
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """Validate that all samples have required fields.

        Args:
            data: Input dataset
            required_fields: List of required field names

        Returns:
            Tuple of (valid_samples, error_messages)
        """
        valid = []
        errors = []

        for i, sample in enumerate(data):
            missing = [f for f in required_fields if f not in sample]
            if missing:
                errors.append(f"Sample {i}: Missing fields {missing}")
            else:
                valid.append(sample)

        return valid, errors

    def validate_choices(
        self,
        data: List[Dict[str, Any]],
        min_choices: int = 2,
        max_choices: int = 10,
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """Validate that choice fields have valid number of options.

        Args:
            data: Input dataset
            min_choices: Minimum number of choices
            max_choices: Maximum number of choices

        Returns:
            Tuple of (valid_samples, error_messages)
        """
        valid = []
        errors = []

        for i, sample in enumerate(data):
            if "choices" in sample:
                choices = sample["choices"]
                if not isinstance(choices, (list, tuple)):
                    errors.append(f"Sample {i}: 'choices' is not a list")
                elif not (min_choices <= len(choices) <= max_choices):
                    errors.append(f"Sample {i}: Invalid number of choices ({len(choices)})")
                else:
                    valid.append(sample)
            else:
                valid.append(sample)  # No choices field is OK

        return valid, errors

    def apply_custom_validators(
        self,
        data: List[Dict[str, Any]],
        validators: List[Callable[[Dict[str, Any]], bool]],
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """Apply custom validation functions.

        Args:
            data: Input dataset
            validators: List of validator functions

        Returns:
            Tuple of (valid_samples, error_messages)
        """
        valid = []
        errors = []

        for i, sample in enumerate(data):
            is_valid = all(validator(sample) for validator in validators)
            if is_valid:
                valid.append(sample)
            else:
                errors.append(f"Sample {i}: Failed custom validation")

        return valid, errors

    def get_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the dataset.

        Args:
            data: Input dataset

        Returns:
            Statistics dictionary
        """
        return {
            "total_samples": len(data),
            "fields_present": self._get_all_fields(data),
            "avg_sample_length": sum(len(str(s)) for s in data) / len(data) if data else 0,
        }

    def _get_all_fields(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get count of all fields across samples."""
        field_counts: Dict[str, int] = {}
        for sample in data:
            for field in sample.keys():
                field_counts[field] = field_counts.get(field, 0) + 1
        return field_counts
