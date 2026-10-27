"""Fact-recall demo: ingest facts into GEM, then generate."""

from __future__ import annotations
import torch
from dc_gem.config import DCGemConfig
from dc_gem.core import PretrainedCore
from dc_gem.memory import EpisodicMemory

def main():
    cfg = DCGemConfig(model_name="Qwen/Qwen3-0.6B", device="cuda")
    core = PretrainedCore(cfg.model_name, cfg.device)
    mem = EpisodicMemory(size=256, d_k=core.hidden_dim, d_v=core.hidden_dim, K=4, tau=0.1)

    facts = [
        "The secret password of castle ZORG is 8492.",
        "The retry cap for module FOOBAR is 7.",
    ]

    # Ingest
    for fact in facts:
        emb = core.encode(fact)
        slot = mem.n_valid
        mem.write_slot(slot, emb, emb, text=fact)
    print(f"Ingested {mem.n_valid} facts")

    # Query
    query = "What is the secret password of castle ZORG?"
    q_emb = core.encode(query)
    hits = mem.search_top_k(q_emb, 4)
    print(f"Top-4 hits:")
    for idx, sc in hits:
        print(f"  [{sc:.3f}] {mem.get_text(idx)}")

    # Generate with retrieved context
    context = "\n".join(mem.get_text(idx) for idx, _ in hits)
    prompt = f"{context}\n\nQuestion: {query}\nAnswer:"
    answer = core.generate(prompt, max_new_tokens=24, temperature=0.0)
    print(f"Generated: {answer}")

if __name__ == "__main__":
    main()