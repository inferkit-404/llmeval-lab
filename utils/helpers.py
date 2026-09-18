"""Helper utilities."""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def ensure_dir(path: str) -> Path:
    """Ensure directory exists.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file.

    Args:
        path: Config file path

    Returns:
        Configuration dictionary
    """
    path = Path(path)

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f) or {}
        elif path.suffix == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def save_json(data: Dict[str, Any], path: str, indent: int = 2) -> None:
    """Save data to JSON file.

    Args:
        data: Data to save
        path: Output path
        indent: JSON indentation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    """Load data from JSON file.

    Args:
        path: JSON file path

    Returns:
        Loaded data
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def truncate_string(s: str, max_length: int = 100) -> str:
    """Truncate string with ellipsis.

    Args:
        s: Input string
        max_length: Maximum length

    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - 3] + "..."
