"""Embedding model wrapper (dedicated retrieval encoder)."""

from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModel


class EmbeddingModel:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, text: str, is_query: bool = False) -> torch.Tensor:
        # bge convention: queries get a prefix for better retrieval
        if is_query and "bge" in self.model_name.lower():
            text = "Represent this sentence for searching relevant passages: " + text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=512).to(self.device)
        out = self.model(**inputs)
        emb = out.last_hidden_state[:, 0]  # CLS
        return emb.squeeze(0).cpu()

    @torch.no_grad()
    def encode_batch(self, texts: list[str], is_query: bool = False) -> torch.Tensor:
        if is_query and "bge" in self.model_name.lower():
            texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True,
                               max_length=512, padding=True).to(self.device)
        out = self.model(**inputs)
        return out.last_hidden_state[:, 0].cpu()
