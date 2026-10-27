"""DC-GEM configuration (pre-trained core, no training from scratch)."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DCGemConfig:
    # Pre-trained core model
    model_name: str = "Qwen/Qwen3-0.6B"
    device: str = "cuda"
    dtype: str = "bfloat16"

    # Seeds
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])

    # Memory (GEM)
    memory_size: int = 4096
    K: int = 16  # dense candidates
    tau: float = 0.1

    # Retrieval
    retrieval_weights: dict = field(default_factory=lambda: {"cos": 0.5, "lex": 0.5})
    rerank_top_k: int = 16

    # Injection
    injection_channel: str = "text"  # "text" or "hidden"

    # Generation
    max_new_tokens: int = 64
    temperature: float = 0.0
    gen_timeout_s: float = 60.0

    # Embedding
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_pooling: str = "cls"

    # Sleep / consolidation
    lora_rank: int = 16
    lora_alpha: int = 32
    sleep_steps: int = 500
    sleep_lr: float = 1e-4
    replay_ratios: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.3])

    # VRAM guard
    vram_limit_gb: float = 14.0

    # Eval
    n_facts_default: int = 20
