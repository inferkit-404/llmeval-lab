"""Model adapters module."""

from typing import Dict, Any, Union

from .base import BaseAdapter
from .hf_adapter import HuggingFaceAdapter
from .vllm_adapter import VLLMAdapter
from .api_adapter import APIAdapter


# Registry of available adapters
ADAPTERS = {
    "hf": HuggingFaceAdapter,
    "huggingface": HuggingFaceAdapter,
    "vllm": VLLMAdapter,
    "api": APIAdapter,
    "openai": APIAdapter,
}


def get_adapter(backend: str, model: Union[str, Dict[str, Any]], **kwargs) -> BaseAdapter:
    """Get appropriate adapter for backend.

    Args:
        backend: Backend type (hf, vllm, api)
        model: Model name or configuration
        **kwargs: Additional arguments for adapter

    Returns:
        Adapter instance

    Raises:
        ValueError: If backend is not supported
    """
    backend_lower = backend.lower()

    if backend_lower not in ADAPTERS:
        raise ValueError(
            f"Unsupported backend '{backend}'. "
            f"Available: {list(ADAPTERS.keys())}"
        )

    adapter_class = ADAPTERS[backend_lower]
    return adapter_class(model, **kwargs)


__all__ = [
    "BaseAdapter",
    "HuggingFaceAdapter",
    "VLLMAdapter",
    "APIAdapter",
    "get_adapter",
]
