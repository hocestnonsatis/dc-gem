#!/usr/bin/env python3
"""Agent demo on obscure (novel) functions: GEM enables recall core can't do."""

from __future__ import annotations
import ast
import json
import os
from pathlib import Path

import torch

from dc_gem.core import PretrainedCore
from dc_gem.embeddings import EmbeddingModel
from dc_gem.memory import EpisodicMemory
from dc_gem.retrieval import hybrid_retrieve


def extract_functions(root):
    funcs = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            src = open(path).read()
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    seg = ast.get_source_segment(src, node) or ""
                    doc = ast.get_docstring(node) or ""
                    funcs.append({"name": node.name, "doc": doc, "code": seg.strip(),
                                 "text": f"Function {node.name}: {doc} Implementation: {seg.strip()}"})
    return funcs


def main():
    gen = PretrainedCore("Qwen/Qwen3-0.6B", "cuda")
    embed = EmbeddingModel("BAAI/bge-small-en-v1.5", "cuda")

    funcs = extract_functions("/tmp/obscure_codebase")
    mem = EpisodicMemory(size=256, d_k=embed.dim, d_v=embed.dim, K=4, tau=0.1)
    for i, f in enumerate(funcs):
        mem.write_slot(i, embed.encode(f["text"]), embed.encode(f["text"]), text=f["text"])

    # simpler: ask to reproduce implementation
    questions = [
        ("Write the Python implementation of the function named zorp.", "zorp"),
        ("Write the Python implementation of the function named flimble.", "flimble"),
        ("Write the Python implementation of the function named quuxify.", "quuxify"),
        ("Write the Python implementation of the function named bazzle.", "bazzle"),
        ("Write the Python implementation of the function named snarfle.", "snarfle"),
    ]

    results = []
    for q, expected in questions:
        q_emb = embed.encode(q, is_query=True)
        ret = hybrid_retrieve(mem, q_emb, q, k_dense=4, top_n=1)
        ctx = ret[0][2] if ret else ""
        prompt_on = f"Code context:\n{ctx}\n\n{q}\nAnswer:"
        out_on = gen.generate(prompt_on, max_new_tokens=80, temperature=0.0)
        prompt_off = f"{q}\nAnswer:"
        out_off = gen.generate(prompt_off, max_new_tokens=80, temperature=0.0)
        hit = expected in (ctx or "")
        results.append({"question": q, "expected": expected, "retrieval_hit": bool(hit),
                       "memON": out_on.strip()[:300], "memOFF": out_off.strip()[:300]})
        print(f"\nQ: {q}")
        print(f"  retrieval hit: {hit}")
        print(f"  memON: {out_on.strip()[:200]}")
        print(f"  memOFF: {out_off.strip()[:200]}")

    hr = sum(1 for r in results if r["retrieval_hit"]) / len(results)
    Path("runs").mkdir(exist_ok=True)
    with open("runs/obscure_agent.json", "w") as fh:
        json.dump({"retrieval_hit_rate": hr, "results": results}, fh, indent=2)
    print(f"\nRetrieval hit rate: {hr:.3f}")


if __name__ == "__main__":
    main()
