"""Pre-trained core wrapper (HuggingFace transformers)."""

from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class PretrainedCore:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device)
        self.model.eval()
        self.hidden_dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """Return pooled embedding (mean of last hidden states)."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # (1, T, d)
        # mean pooling
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled.squeeze(0).cpu()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.0) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if temperature == 0:
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        else:
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=True, temperature=temperature,
                                     pad_token_id=self.tokenizer.eos_token_id)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @torch.no_grad()
    def hidden_state(self, text: str, layer: int = -1) -> torch.Tensor:
        """Return pooled hidden state for text (mean over tokens)."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model(**inputs, output_hidden_states=True)
        h = out.hidden_states[layer]
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled.squeeze(0).cpu()

    @torch.no_grad()
    def hidden_states_seq(self, text: str, layer: int = -1) -> torch.Tensor:
        """Return full sequence hidden states (T, d)."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model(**inputs, output_hidden_states=True)
        return out.hidden_states[layer].squeeze(0).cpu()
