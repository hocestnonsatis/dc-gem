#!/usr/bin/env python3
"""Codebase Q&A agent: GEM over a small codebase.

Indexes functions (name + docstring + body) into GEM, then answers questions
using retrieved context. Compares memON (retrieve) vs memOFF (no retrieval).
"""

from __future__ import annotations
import argparse
import ast
import json
import os
import re
from pathlib import Path

import torch

from dc_gem.core import PretrainedCore
from dc_gem.embeddings import EmbeddingModel
from dc_gem.memory import EpisodicMemory
from dc_gem.retrieval import hybrid_retrieve, extract_code


def extract_functions(root: str) -> list[dict]:
    funcs = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                src = open(path).read()
            except Exception:
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # get source segment
                    seg = ast.get_source_segment(src, node) or ""
                    doc = ast.get_docstring(node) or ""
                    funcs.append({
                        "name": node.name,
                        "file": os.path.relpath(path, root),
                        "doc": doc,
                        "code": seg.strip(),
                        "text": f"Function {node.name}: {doc} Code: {seg.strip()}",
                    })
    return funcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebase", default="/tmp/sample_codebase")
    ap.add_argument("--gen-model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="runs/codebase_agent.json")
    args = ap.parse_args()

    print("Loading models...")
    gen = PretrainedCore(args.gen_model, args.device)
    embed = EmbeddingModel(args.embed_model, args.device)

    funcs = extract_functions(args.codebase)
    print(f"Extracted {len(funcs)} functions")

    mem = EpisodicMemory(size=max(len(funcs) + 100, 256),
                        d_k=embed.dim, d_v=embed.dim, K=4, tau=0.1)
    for i, f in enumerate(funcs):
        emb = embed.encode(f["text"])
        mem.write_slot(i, emb, emb, text=f["text"])
    print(f"Ingested {mem.n_valid} functions")

    questions = [
        ("What does the factorial function do?", "factorial"),
        ("Which function checks if a number is prime?", "is_prime"),
        ("How do I reverse a string?", "reverse_string"),
        ("Which function computes Fibonacci numbers?", "fibonacci"),
        ("What function counts vowels in a string?", "count_vowels"),
    ]

    results = []
    for q, expected in questions:
        # memON
        q_emb = embed.encode(q, is_query=True)
        ret = hybrid_retrieve(mem, q_emb, q, k_dense=4, top_n=1)
        ctx = ret[0][2] if ret else ""
        prompt_on = (f"Code context:\n{ctx}\n\n"
                     f"Question: {q}\nAnswer in one sentence:")
        out_on = gen.generate(prompt_on, max_new_tokens=40, temperature=0.0)
        # memOFF
        prompt_off = f"Question: {q}\nAnswer in one sentence:"
        out_off = gen.generate(prompt_off, max_new_tokens=40, temperature=0.0)

        hit = expected in (ctx or "")
        results.append({
            "question": q, "expected_fn": expected,
            "retrieved": ctx[:200], "retrieval_hit": bool(hit),
            "memON_answer": out_on.strip(),
            "memOFF_answer": out_off.strip(),
        })
        print(f"\nQ: {q}")
        print(f"  retrieval hit: {hit} (expected {expected})")
        print(f"  memON: {out_on.strip()[:120]}")
        print(f"  memOFF: {out_off.strip()[:120]}")

    hit_rate = sum(1 for r in results if r["retrieval_hit"]) / len(results)
    summary = {"n_functions": len(funcs), "n_questions": len(questions),
               "retrieval_hit_rate": hit_rate, "results": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nRetrieval hit rate: {hit_rate:.3f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
