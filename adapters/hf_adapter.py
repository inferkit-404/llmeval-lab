"""HuggingFace Transformers adapter."""

from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

from .base import BaseAdapter


class HuggingFaceAdapter(BaseAdapter):
    """Adapter for HuggingFace transformers models."""

    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        """Initialize HuggingFace adapter.

        Args:
            model_name: HuggingFace model name or path
            device: Device to use (cuda or cpu)
            **kwargs: Additional model configuration
        """
        super().__init__(model_name, **kwargs)

        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Determine model type and load accordingly
        self.model_type = kwargs.get("model_type", "causal")

        if self.model_type == "causal":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                **kwargs,
            )
        else:
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                **kwargs,
            )

        self.model.to(self.device)
        self.model.eval()

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """Generate response for a single prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        return response

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        batch_size: int = 8,
        **kwargs,
    ) -> List[str]:
        """Generate responses for a batch of prompts."""
        responses = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs,
                )

            # Decode each response
            input_length = inputs["input_ids"].shape[1]
            for j, output in enumerate(outputs):
                response = self.tokenizer.decode(output[input_length:], skip_special_tokens=True)
                responses.append(response)

        return responses

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": self.device,
            "vocab_size": self.tokenizer.vocab_size,
        }
