"""Base adapter interface for model backends."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseAdapter(ABC):
    """Abstract base class for model adapters."""

    def __init__(self, model_name: str, **kwargs):
        """Initialize adapter.

        Args:
            model_name: Name of the model to load
            **kwargs: Additional configuration
        """
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """Generate response for a single prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            **kwargs: Additional generation arguments

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> List[str]:
        """Generate responses for a batch of prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            **kwargs: Additional generation arguments

        Returns:
            List of generated texts
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration.

        Returns:
            Configuration dictionary
        """
        pass

    def supports_batch(self) -> bool:
        """Check if adapter supports batch generation.

        Returns:
            True if batch generation is supported
        """
        return True
