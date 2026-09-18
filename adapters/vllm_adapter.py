"""vLLM adapter for high-throughput inference."""

from typing import List, Dict, Any, Optional
import os

from .base import BaseAdapter


class VLLMAdapter(BaseAdapter):
    """Adapter for vLLM inference engine."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        **kwargs,
    ):
        """Initialize vLLM adapter.

        Args:
            model_name: Model name or path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum model context length
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)

        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.model = None

        # Lazy import to make vLLM optional
        try:
            from vllm import LLM
            self.llm_class = LLM
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Install with: pip install vllm"
            )

    def _ensure_model_loaded(self):
        """Lazy load the model."""
        if self.model is None:
            from vllm import LLM
            self.model = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
            )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """Generate response for a single prompt."""
        self._ensure_model_loaded()

        outputs = self.model.generate(
            [prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

        return outputs[0].outputs[0].text

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> List[str]:
        """Generate responses for a batch of prompts."""
        self._ensure_model_loaded()

        outputs = self.model.generate(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

        return [output.outputs[0].text for output in outputs]

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "model_name": self.model_name,
            "backend": "vllm",
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
        }
