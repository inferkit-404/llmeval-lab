"""API-based model adapter (OpenAI, Anthropic, etc.)."""

import os
from typing import List, Dict, Any, Optional

from .base import BaseAdapter


class APIAdapter(BaseAdapter):
    """Adapter for API-based models."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ):
        """Initialize API adapter.

        Args:
            model_name: Model name (e.g., "gpt-4", "claude-3-sonnet")
            api_key: API key (defaults to env var)
            api_base: API base URL for custom endpoints
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)

        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.api_base = api_base

        # Determine provider
        if "gpt" in model_name.lower() or "o1" in model_name.lower():
            self.provider = "openai"
        elif "claude" in model_name.lower():
            self.provider = "anthropic"
        else:
            self.provider = "openai"  # default

    def _call_openai(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Call OpenAI API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")

        client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

        return response.choices[0].message.content

    def _call_anthropic(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Call Anthropic API."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")

        client = Anthropic(api_key=self.api_key)

        response = client.messages.create(
            model=self.model_name,
            max_tokens=max_new_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

        return response.content[0].text

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """Generate response for a single prompt."""
        if self.provider == "openai":
            return self._call_openai(prompt, max_new_tokens, temperature, **kwargs)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt, max_new_tokens, temperature, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        batch_size: int = 10,
        **kwargs,
    ) -> List[str]:
        """Generate responses for a batch of prompts (with rate limiting)."""
        import time

        responses = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_responses = []

            for prompt in batch_prompts:
                try:
                    response = self.generate(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        **kwargs,
                    )
                    batch_responses.append(response)
                except Exception as e:
                    print(f"Error generating response: {e}")
                    batch_responses.append("")

                # Rate limiting
                time.sleep(0.1)

            responses.extend(batch_responses)

        return responses

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "api_base": self.api_base,
        }
