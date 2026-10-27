#!/usr/bin/env python3
"""Semantic ablation: core mean-pool vs separate sentence-embedding model."""

from __future__ import annotations
import json, re
import torch
from dc_gem.core import PretrainedCore
from dc_gem.embeddings import EmbeddingModel
from dc_gem.memory import EpisodicMemory

def main():
    gen = PretrainedCore("Qwen/Qwen3-0.6B", "cuda")
    bge = EmbeddingModel("BAAI/bge-small-en-v1.5", "cuda")
    N = 50
    facts = [f"The secret password of castle Z{i:02d} is {1000+(i*37)%9000}." for i in range(N)]
    queries = [f"What is the password for castle Z{i:02d}?" for i in range(N)]

    # bge
    mem_bge = EpisodicMemory(size=N+10, d_k=bge.dim, d_v=bge.dim, K=4, tau=0.1)
    for i, f in enumerate(facts):
        mem_bge.write_slot(i, bge.encode(f), bge.encode(f), text=f)
    bge_top1 = 0
    for i, q in enumerate(queries):
        hits = mem_bge.search_top_k(bge.encode(q, is_query=True), 1)
        ret = mem_bge.get_text(hits[0][0]) if hits else ""
        m = re.search(r"castle (Z\d+)", ret)
        if m and m.group(1) == f"Z{i:02d}":
            bge_top1 += 1

    # core mean-pool (Qwen hidden state, mean pooling)
    mem_core = EpisodicMemory(size=N+10, d_k=gen.hidden_dim, d_v=gen.hidden_dim, K=4, tau=0.1)
    for i, f in enumerate(facts):
        mem_core.write_slot(i, gen.encode(f), gen.encode(f), text=f)
    core_top1 = 0
    for i, q in enumerate(queries):
        hits = mem_core.search_top_k(gen.encode(q), 1)
        ret = mem_core.get_text(hits[0][0]) if hits else ""
        m = re.search(r"castle (Z\d+)", ret)
        if m and m.group(1) == f"Z{i:02d}":
            core_top1 += 1

    result = {
        "N": N,
        "bge_top1": bge_top1 / N,
        "core_meanpool_top1": core_top1 / N,
        "chance": 1.0 / N,
    }
    print(json.dumps(result, indent=2))
    Path("runs/evidence/py").mkdir(parents=True, exist_ok=True)
    with open("runs/evidence/py/semantic_ablation.json", "w") as fh:
        json.dump(result, fh, indent=2)

if __name__ == "__main__":
    from pathlib import Path
    main()
